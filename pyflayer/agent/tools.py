"""Tools exposed to the LLM (OpenAI function-calling format). Each returns a short text result."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Callable

from ..bot.actions import ActionError
from ..bot.pathfinder import GoalNear, GoalXZ
from ..bot.scanner import Scanner


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    fn: Callable
    required: tuple = ()
    long_running: bool = False

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.params, "required": list(self.required)}}}


def S(desc, default=None):
    d = {"type": "string", "description": desc}
    return d


def I(desc):
    return {"type": "integer", "description": desc}


def N(desc):
    return {"type": "number", "description": desc}


def B(desc):
    return {"type": "boolean", "description": desc}


class ToolBox:
    def __init__(self, agent):
        self.agent = agent
        self.tools: dict[str, Tool] = {}
        self._register()

    @property
    def bot(self):
        return self.agent.bot

    @property
    def gd(self):
        return self.agent.bot.gd

    def add(self, name, description, params=None, required=(), long_running=False):
        def deco(fn):
            self.tools[name] = Tool(name, description, params or {}, fn, tuple(required), long_running)
            return fn
        return deco

    def schemas(self, names=None) -> list[dict]:
        return [t.schema() for n, t in self.tools.items() if names is None or n in names]

    def call(self, name: str, args: dict) -> str:
        t = self.tools.get(name)
        if not t:
            close = [n for n in self.tools if name.lower() in n or n in name.lower()]
            return f"ERROR: unknown tool {name!r}." + (f" Did you mean {close[:3]}?" if close else "")
        try:
            clean = {k: v for k, v in (args or {}).items() if k in t.params}
            res = t.fn(**clean)
            return str(res) if res is not None else "ok"
        except ActionError as e:
            return f"ERROR: {e}"
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}. Expected {list(t.params)} (required {list(t.required)})"
        except Exception as e:  # noqa: BLE001
            self.agent.log.exception("tool %s crashed", name)
            return f"ERROR: {type(e).__name__}: {e}"

    # ================================================================== registration
    def _register(self):
        a = self.agent
        add = self.add

        def norm(name: str) -> str:
            name = str(name).strip().lower().replace(" ", "_")
            return name.split(":", 1)[1] if name.startswith("minecraft:") else name

        # ------------------------------------------------------------ perception
        @add("get_status", "Your health, hunger, position, dimension, biome, time, inventory, armor and nearby mobs/players.")
        def get_status():
            return a.status_text()

        @add("scan_surroundings", "Scan up to 100 blocks around you: ores (with nearest coordinates), trees, water/lava, "
             "useful blocks, biomes, structures (villages, strongholds, temples, mineshafts, fortresses...) and mobs.",
             {"radius": I("scan radius in blocks, max 112 (default 100)")})
        def scan_surroundings(radius=100):
            res = a.scanner.scan(min(112, max(8, int(radius))))
            a.after_scan(res)
            return Scanner.summarize(res)

        @add("find_blocks", "Find the nearest blocks of a type (block id like 'iron_ore', or a tag like '#logs').",
             {"block": S("block id or #tag"), "radius": I("search radius (default 100)"), "count": I("max results")},
             required=("block",))
        def find_blocks(block, radius=100, count=5):
            block = norm(block)
            pts = self.bot.find_blocks(block, min(128, radius), max(1, min(20, count)))
            if not pts:
                return f"no {block} within {radius} blocks"
            p = self.bot.position
            return "; ".join(f"{x},{y},{z} ({math.dist((x, y, z), p):.0f}m)" for x, y, z in pts)

        @add("detect_structures", "Detect structures within a radius (villages, dungeons, mineshafts, strongholds, "
             "temples, monuments, ruined portals, ancient cities, trial chambers, fortresses, bastions...).",
             {"radius": I("radius (default 100)")})
        def detect_structures(radius=100):
            s = a.scanner.detect_structures(min(112, radius))
            for st in s:
                a.brain.set_waypoint(f"{st['structure']}@{st['pos'][0]},{st['pos'][2]}", self.bot.dimension,
                                     st["pos"], "structure")
            return json.dumps(s) if s else "no structures detected nearby"

        @add("locate_structure", "Use the server /locate command to find a structure far away (needs cheats allowed).",
             {"structure": S("structure id e.g. village_plains, stronghold, fortress, ancient_city")},
             required=("structure",))
        def locate_structure(structure):
            if not a.cfg["agent"]["allow_cheats"]:
                return "ERROR: cheats are disabled (agent.allow_cheats=false). Explore or follow eyes of ender instead."
            return a.scanner.locate_structure(norm(structure))

        @add("block_at", "What block is at x,y,z.", {"x": I("x"), "y": I("y"), "z": I("z")}, ("x", "y", "z"))
        def block_at(x, y, z):
            n = self.bot.world.name_at(int(x), int(y), int(z))
            return n or "unloaded chunk"

        @add("nearby_entities", "List mobs, animals, players and dropped items near you.",
             {"radius": I("radius (default 32)")})
        def nearby_entities(radius=32):
            ents = self.bot.entities.within(self.bot.position, min(128, radius),
                                            lambda e: e.type not in ("experience_orb",))
            if not ents:
                return "nothing nearby"
            return "; ".join(e.describe(self.bot.position) + f" [id {e.id}]" for e in ents[:25])

        # ------------------------------------------------------------ movement
        @add("goto", "Walk/path to coordinates (digs, bridges and pillars when needed).",
             {"x": N("x"), "y": N("y (optional; omit to only reach x,z)"), "z": N("z"), "range": N("stop distance")},
             ("x", "z"), long_running=True)
        def goto(x, z, y=None, range=1.5):
            goal = GoalXZ(x, z, range) if y is None else GoalNear(x, y, z, range)
            self.bot.goto(goal, timeout=300)
            return f"arrived at {tuple(round(v, 1) for v in self.bot.position)}"

        @add("goto_block", "Walk next to the nearest block of a type.", {"block": S("block id or #tag")},
             ("block",), long_running=True)
        def goto_block(block):
            pts = self.bot.find_blocks(norm(block), 128, 1)
            if not pts:
                return f"ERROR: no {block} nearby; explore first"
            from ..bot.pathfinder import GoalLookAt
            self.bot.goto(GoalLookAt(*pts[0], reach=3.5), timeout=240)
            return f"next to {block} at {pts[0]}"

        @add("goto_player", "Walk to a player.", {"name": S("player name")}, ("name",), long_running=True)
        def goto_player(name):
            e = self.bot.entities.nearest(self.bot.position, lambda e: e.name and e.name.lower() == name.lower(), 512)
            if not e:
                return f"ERROR: player {name} not visible"
            self.bot.goto(GoalNear(e.x, e.y, e.z, 2.5), timeout=240)
            return f"reached {name}"

        @add("follow_player", "Follow a player for some seconds.",
             {"name": S("player"), "seconds": I("duration (default 60)")}, ("name",), long_running=True)
        def follow_player(name, seconds=60):
            e = self.bot.entities.nearest(self.bot.position, lambda e: e.name and e.name.lower() == name.lower(), 512)
            if not e:
                return f"ERROR: player {name} not visible"
            return self.bot.follow(e, 3, min(600, seconds))

        @add("explore", "Walk a distance in a direction to discover new terrain (direction: north/south/east/west/"
             "random or an angle in degrees).", {"direction": S("direction"), "distance": I("blocks (default 100)")},
             long_running=True)
        def explore(direction="random", distance=100):
            dirs = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
            if str(direction).lower() in dirs:
                dx, dz = dirs[str(direction).lower()]
            else:
                try:
                    ang = math.radians(float(direction))
                except (TypeError, ValueError):
                    import random
                    ang = random.uniform(0, 2 * math.pi)
                dx, dz = math.cos(ang), math.sin(ang)
            x, _, z = self.bot.position
            self.bot.goto(GoalXZ(x + dx * distance, z + dz * distance, 4), timeout=max(60, distance * 2))
            self.bot.wait_ticks(20)
            res = a.scanner.scan(100)
            a.after_scan(res)
            return "explored; now at " + str(tuple(round(v) for v in self.bot.position)) + "\n" + Scanner.summarize(res, 900)

        @add("look_at", "Turn to look at a position.", {"x": N("x"), "y": N("y"), "z": N("z")}, ("x", "y", "z"))
        def look_at(x, y, z):
            self.bot.look_at(x, y, z, wait=True)
            return "looking"

        # ------------------------------------------------------------ gathering / building
        @add("mine_block", "Mine `count` blocks of a type nearby and pick up the drops (equips the best tool). "
             "Use ids like oak_log, stone, iron_ore, or tags like #logs.",
             {"block": S("block id or #tag"), "count": I("how many (default 1)")}, ("block",), long_running=True)
        def mine_block(block, count=1):
            return self.bot.collect_block(norm(block), max(1, min(64, int(count))), timeout=600)

        @add("dig_at", "Break the block at x,y,z (must be within reach ~4.5 blocks).",
             {"x": I("x"), "y": I("y"), "z": I("z")}, ("x", "y", "z"))
        def dig_at(x, y, z):
            from ..bot.pathfinder import GoalLookAt
            pos = (int(x), int(y), int(z))
            if self.bot.distance_to_block(pos) > 4.3:
                self.bot.goto(GoalLookAt(*pos, reach=3.8), timeout=60)
            name = self.bot.world.name_at(*pos)
            self.bot.dig(pos)
            self.bot.collect_items(5, 5)
            return f"broke {name} at {pos}"

        @add("place_block", "Place a block from your inventory at x,y,z.",
             {"item": S("block item"), "x": I("x"), "y": I("y"), "z": I("z")}, ("item", "x", "y", "z"))
        def place_block(item, x, y, z):
            self.bot.place_at((int(x), int(y), int(z)), norm(item))
            return f"placed {item} at {x},{y},{z}"

        @add("collect_items", "Pick up dropped items lying around.", {"radius": I("radius (default 10)")},
             long_running=True)
        def collect_items(radius=10):
            return f"picked up {self.bot.collect_items(min(32, radius), 20)} item stacks"

        @add("build_shelter", "Emergency: wall yourself in with blocks (for the night / when hurt).", long_running=True)
        def build_shelter():
            return self.bot.shelter()

        @add("hide_for_the_night", "Dig a sealed 2-deep hole and wait there safely until morning.", long_running=True)
        def hide_for_the_night():
            return self.bot.hide_underground()

        @add("build_nether_portal", "Build and light a nether portal next to you (needs 10 obsidian, 4 extra blocks, "
             "flint_and_steel).", long_running=True)
        def build_nether_portal():
            from ..bot.advanced import build_nether_portal as bnp
            return bnp(self.bot)

        # ------------------------------------------------------------ crafting
        @add("craft", "Craft an item (uses a nearby crafting table or places one from inventory if the recipe needs 3x3).",
             {"item": S("item id"), "count": I("how many items (default 1)")}, ("item",))
        def craft(item, count=1):
            return self.bot.craft(norm(item), max(1, int(count)))

        @add("smelt", "Smelt/cook items into `item` using a furnace (placed from inventory if none nearby) and fuel.",
             {"item": S("output item, e.g. iron_ingot, cooked_beef, glass, charcoal"), "count": I("how many")},
             ("item",), long_running=True)
        def smelt(item, count=1):
            return self.bot.smelt(norm(item), max(1, int(count)))

        @add("fill_bucket", "Fill an empty bucket with water or lava from the nearest source block.",
             {"fluid": S("water or lava")}, ("fluid",), long_running=True)
        def fill_bucket(fluid):
            from ..bot.liquids import fill_bucket as fb
            return fb(self.bot, "lava" if "lava" in str(fluid).lower() else "water")

        @add("pour_bucket", "Pour a water or lava bucket on top of the block at x,y,z (put out fire, make a "
             "waterfall, block a path, make obsidian...).", {"fluid": S("water or lava"), "x": I("x"), "y": I("y"),
                                                             "z": I("z")}, ("fluid", "x", "y", "z"))
        def pour_bucket(fluid, x, y, z):
            from ..bot.liquids import pour
            return f"poured {fluid} at {pour(self.bot, 'lava' if 'lava' in str(fluid).lower() else 'water', (int(x), int(y), int(z)))}"

        @add("burn_with_lava", "Pour lava under a mob (by type or entity id) from a safe distance, then take the "
             "lava back. Not for fire-immune mobs.", {"target": S("mob type or entity id")}, ("target",),
             long_running=True)
        def burn_with_lava(target):
            from ..bot.liquids import burn_enemy
            t = str(target).lower()
            e = self.bot.entities.get(int(t)) if t.isdigit() else self.bot.entities.nearest(
                self.bot.position, lambda x: x.type == norm(t), 16)
            if not e:
                return f"ERROR: no {target} nearby"
            return burn_enemy(self.bot, e)

        @add("make_obsidian", "Make obsidian by pouring water over a lava pool and mine it (needs a diamond "
             "pickaxe and a water bucket).", {"count": I("how many (default 10)")}, long_running=True)
        def make_obsidian(count=10):
            from ..bot.liquids import cast_obsidian
            return cast_obsidian(self.bot, int(count))

        @add("achieve_goal", "Work through the whole beat-the-game plan up to a big goal, step by step: one of "
             "iron_armor, diamonds, nether_portal, enter_nether, blaze_rods, ender_pearls, eyes_of_ender, stronghold, "
             "end_portal, enter_end, kill_dragon (= beat the game). Runs in the background.",
             {"goal": S("goal step name")}, ("goal",))
        def achieve_goal(goal):
            a.start_goal(str(goal).strip().lower(), "AI")
            return f"started working towards {goal}; progress is reported in chat"

        @add("obtain_item", "AUTOMATIC: get `count` of any item. Works out the full plan from game data (craft, smelt, "
             "mine with the right tool tier, hunt mobs) and executes it, including getting tools first. Prefer this "
             "for anything that needs several steps.", {"item": S("item id"), "count": I("how many")},
             ("item",), long_running=True)
        def obtain_item(item, count=1):
            r = self.bot.obtain(norm(item), max(1, int(count)), timeout=1500)
            plan = self.bot.planner.log[-12:]
            return r + ("\nplan log:\n" + "\n".join(plan) if plan else "")

        @add("recipe", "Show how an item is made in this exact game version (recipes, smelting, drops, uses).",
             {"item": S("item id")}, ("item",))
        def recipe(item):
            return self.gd.describe_item(norm(item))

        @add("block_info", "Authoritative info about a block in this version: hardness, best tool, tier needed, drops.",
             {"block": S("block id")}, ("block",))
        def block_info(block):
            return self.gd.describe_block(norm(block))

        @add("missing_for", "Which ingredients you are missing to craft an item.",
             {"item": S("item id"), "count": I("count")}, ("item",))
        def missing_for(item, count=1):
            m = self.bot.missing_for(norm(item), max(1, int(count)))
            return json.dumps(m) if m else "you have everything needed"

        # ------------------------------------------------------------ inventory
        @add("inventory", "List your inventory.")
        def inventory():
            inv = self.bot.inventory
            armor = {k: (v.name if v else None) for k, v in inv.armor().items()}
            return f"{inv.summary()} | armor {armor} | free slots {inv.empty_slots()}"

        @add("equip", "Hold an item in your hand (or 'offhand'/'armor').",
             {"item": S("item id"), "slot": S("hand | offhand | armor")}, ("item",))
        def equip(item, slot="hand"):
            ok = self.bot.equip(norm(item), slot if slot in ("hand", "offhand", "armor") else "hand")
            return f"equipped {item}" if ok else f"ERROR: no {item} in inventory"

        @add("equip_best_armor", "Wear the best armor you carry.")
        def equip_best_armor():
            w = self.bot.equip_armor()
            return f"now wearing {w}" if w else "no better armor to wear"

        @add("drop_item", "Drop/throw items from your inventory.", {"item": S("item id"), "count": I("count")},
             ("item",))
        def drop_item(item, count=None):
            return f"dropped {self.bot.toss(norm(item), count)} {item}"

        @add("eat", "Eat the best food in your inventory (or a specific one).", {"food": S("optional food item id")})
        def eat(food=None):
            return self.bot.eat(norm(food) if food else None)

        # ------------------------------------------------------------ containers / interaction
        @add("use_block", "Right-click a block: open doors/gates, press buttons, pull levers, use beds, "
             "open chests/furnaces/crafting tables.", {"x": I("x"), "y": I("y"), "z": I("z")}, ("x", "y", "z"))
        def use_block(x, y, z):
            self.bot.activate_block((int(x), int(y), int(z)))
            time.sleep(0.4)
            w = self.bot.inventory.window
            if w:
                items = ", ".join(f"{it.name}x{it.count}" for _, it in w.container_items())
                self.bot.close_window()
                return f"opened {w.type}: {items or 'empty'}"
            return "used block"

        @add("chest_contents", "Look inside a chest/barrel/container at x,y,z.",
             {"x": I("x"), "y": I("y"), "z": I("z")}, ("x", "y", "z"))
        def chest_contents(x, y, z):
            c = self.bot.container_contents((int(x), int(y), int(z)))
            a.brain.set_waypoint(f"chest@{x},{y},{z}", self.bot.dimension, (x, y, z), "container",
                                 ", ".join(f"{k}x{v}" for k, v in list(c.items())[:10]))
            return json.dumps(c) if c else "empty"

        @add("deposit_items", "Put items into a container at x,y,z (all non-tool/food items, or a list).",
             {"x": I("x"), "y": I("y"), "z": I("z"), "items": {"type": "array", "items": {"type": "string"}}},
             ("x", "y", "z"))
        def deposit_items(x, y, z, items=None):
            return self.bot.deposit((int(x), int(y), int(z)), set(norm(i) for i in items) if items else None)

        @add("withdraw_items", "Take items out of a container at x,y,z.",
             {"x": I("x"), "y": I("y"), "z": I("z"), "item": S("item id"), "count": I("count")},
             ("x", "y", "z", "item"))
        def withdraw_items(x, y, z, item, count=64):
            return self.bot.withdraw((int(x), int(y), int(z)), norm(item), int(count))

        @add("sleep", "Sleep in a nearby bed (skips the night).", long_running=True)
        def sleep():
            return self.bot.sleep_in_bed()

        @add("interact_entity", "Right-click an entity by id (trade with villager, ride, feed/breed animals, shear).",
             {"entity_id": I("entity id from nearby_entities")}, ("entity_id",))
        def interact_entity(entity_id):
            e = self.bot.entities.get(int(entity_id))
            if not e:
                return "ERROR: entity not found"
            if self.bot.entity_distance(e) > 3:
                self.bot.goto(GoalNear(e.x, e.y, e.z, 2), timeout=30)
            return "interacted" if self.bot.interact_entity(e) else "ERROR: out of reach"

        # ------------------------------------------------------------ combat
        @add("attack", "Fight a mob until it dies: a type like 'zombie'/'cow', 'hostile' for the nearest monster, "
             "or an entity id.", {"target": S("mob type, 'hostile', or entity id")}, ("target",), long_running=True)
        def attack(target):
            bot = self.bot
            t = str(target).strip().lower()
            if t.isdigit():
                e = bot.entities.get(int(t))
            elif t in ("hostile", "monster", "nearest"):
                e = bot.entities.nearest(bot.position, lambda e: e.hostile, 32)
            else:
                e = bot.entities.nearest(bot.position, lambda e: e.type == norm(t) or (e.name or "").lower() == t, 64)
            if not e:
                return f"ERROR: no {target} nearby"
            r = bot.fight(e, timeout=60)
            bot.collect_items(6, 6)
            return r

        @add("flee", "Run away from danger (nearest hostile mob by default).", {"distance": I("blocks (default 20)")},
             long_running=True)
        def flee(distance=20):
            e = self.bot.entities.nearest(self.bot.position, lambda e: e.hostile, 32)
            src = e.pos if e else self.bot.position
            return self.bot.flee(src, distance)

        # ------------------------------------------------------------ communication
        @add("chat", "Say something in the game chat.", {"message": S("text")}, ("message",))
        def chat(message):
            self.bot.chat(str(message)[:250])
            return "said it"

        # ------------------------------------------------------------ knowledge
        @add("wiki", "Look something up on the Minecraft Wiki (minecraft.wiki). Cached offline after the first lookup.",
             {"query": S("topic, e.g. 'Nether portal', 'Blaze', 'Stronghold'")}, ("query",))
        def wiki(query):
            return a.wiki.lookup(str(query), 3000)

        @add("recall", "Search your long-term memory (lessons, past events, knowledge).", {"query": S("what to recall")},
             ("query",))
        def recall(query):
            return a.brain.recall_text(str(query), 8) or "nothing relevant remembered"

        @add("remember", "Store an important lesson or fact in long-term memory.",
             {"fact": S("the lesson/fact"), "topic": S("short topic")}, ("fact",))
        def remember(fact, topic="general"):
            a.brain.remember(str(fact), str(topic), 0.8, "agent")
            return "remembered"

        @add("team_notes", "Read the latest notes from your watcher teammate (threats, opportunities, advice), "
             "optionally posting a note for it.", {"post": S("optional note to post")})
        def team_notes(post=None):
            if post:
                a.board.post("worker", str(post)[:200])
            return "\n".join(a.board.recent(12)) or "no notes yet"

        @add("set_waypoint", "Remember the current position (or x,y,z) under a name, e.g. 'base', 'iron_cave'.",
             {"name": S("name"), "x": I("x"), "y": I("y"), "z": I("z"), "note": S("note")}, ("name",))
        def set_waypoint(name, x=None, y=None, z=None, note=""):
            pos = (x, y, z) if None not in (x, y, z) else self.bot.block_pos
            a.brain.set_waypoint(str(name), self.bot.dimension, pos, "user", note)
            return f"waypoint {name} = {pos}"

        @add("list_waypoints", "List remembered places (nearest first).", {"kind": S("optional kind filter")})
        def list_waypoints(kind=None):
            w = a.brain.waypoints(self.bot.dimension, self.bot.position, kind, 25)
            return "; ".join(f"{x['name']} {x['pos']} ({x['kind']})" for x in w) or "no waypoints"

        @add("goto_waypoint", "Walk to a remembered waypoint.", {"name": S("waypoint name")}, ("name",),
             long_running=True)
        def goto_waypoint(name):
            w = a.brain.waypoint(str(name))
            if not w:
                return f"ERROR: unknown waypoint {name}"
            if w["dim"] != self.bot.dimension:
                return f"ERROR: waypoint is in {w['dim']}, you are in {self.bot.dimension}"
            self.bot.goto(GoalNear(*w["pos"], 2), timeout=600)
            return f"at waypoint {name}"

        @add("gather_nearest_wood",
             "Gather any nearby wood, choosing actually loaded species (oak over distant birch).",
             {"count": I("number of logs, 1-32")})
        def gather_nearest_wood(count=8):
            return self.bot.planner.gather_wood(max(1, min(32, int(count))))

        @add("saved_finds",
             "List the valuable ores (diamonds, ancient debris) I have seen and saved but not mined yet.", {})
        def saved_finds():
            return a.finds.summary()

        @add("mine_saved_finds",
             "Go back to saved diamonds / ancient debris and mine them now (needs the right pickaxe). "
             "Do this before other work when it says they are ready.", {})
        def mine_saved_finds():
            if not a.finds.pending():
                return "nothing ready: " + a.finds.summary()
            a.finds.run_pending()
            return "done; still saved: " + a.finds.summary()

        @add("practice_skill",
             "Practise ONE dry-ground movement/placement/resource skill, record real success or failure. "
             "Never start a fall/lava stunt to train.",
             {"name": S("optional: resource_choice, safe_navigation, sprint_jump, controlled_placement, chat_teamwork")})
        def practice_skill(name=None):
            with a._actuation_lock:
                return a.skill_coach.practice_one(name)

        @add("skill_progress",
             "Show measured skill training progress and locked advanced maneuvers.")
        def skill_progress():
            skills = a.skill_coach.mastery()
            out = [f"{name}: {p['successes']}/{p['attempts']} verified; "
                   f"{'CERTIFIED' if p['certified'] else 'training'}"
                   for name, p in skills.items()]
            out.append("advanced lava-gap jump: " +
                       ("unlocked" if a.skill_coach.lava_jump_ready()
                        else "LOCKED (needs certified dry jumps and placements, explicit opt-in)"))
            return "\n".join(out)

        # ------------------------------------------------------------ skills
        @add("list_skills", "List learned skills (reusable action programs), optionally matching a query.",
             {"query": S("optional search")})
        def list_skills(query=None):
            sk = a.brain.skills(query, 15)
            return "\n".join(f"- {s['name']}({', '.join(s['params'])}): {s['description']} "
                             f"[ok {s['successes']}, fail {s['failures']}]" for s in sk) or "no skills yet"

        @add("run_skill", "Run a learned skill by name with parameters.",
             {"name": S("skill name"), "params": {"type": "object", "description": "parameter values"}},
             ("name",), long_running=True)
        def run_skill(name, params=None):
            return a.run_skill(str(name), params or {})

        @add("save_skill", "Save a reusable skill: a named list of tool calls. Use {param} placeholders in args.",
             {"name": S("snake_case name"), "description": S("what it does / when to use"),
              "params": {"type": "array", "items": {"type": "string"}},
              "steps": {"type": "array", "items": {"type": "object"},
                        "description": "[{\"tool\": \"mine_block\", \"args\": {\"block\": \"{log}\", \"count\": 4}}, ...]"}},
             ("name", "description", "steps"))
        def save_skill(name, description, steps, params=None):
            bad = [s for s in steps if not isinstance(s, dict) or s.get("tool") not in self.tools]
            if bad:
                return f"ERROR: invalid steps {bad[:2]}"
            n = a.brain.save_skill(name, description, steps, params or [])
            return f"saved skill {n}"

        # ------------------------------------------------------------ flow control
        @add("wait", "Wait a number of seconds (e.g. for the furnace or for day).", {"seconds": N("seconds (max 60)")})
        def wait(seconds=5):
            end = time.time() + min(60.0, max(0.0, float(seconds)))
            while time.time() < end:
                self.bot._check()
                time.sleep(0.25)
            return "waited"

        @add("task_complete", "Call when the current task is finished (or impossible). Summarise what happened.",
             {"summary": S("what you achieved / why you stopped"), "success": B("true if the goal was achieved")},
             ("summary",))
        def task_complete(summary, success=True):
            a.finish_task(str(summary), bool(success))
            return "task closed"
