"""Goal planner: obtain(item, count) figures out how to get any item from the game's own data.

For an item it compares (with a recursive cost estimate):
  * crafting  (recipes from the server jar, tags resolved),
  * smelting  (needs furnace + fuel, both obtained recursively),
  * mining    (block loot tables; obtains the right tool tier first),
  * hunting   (entity loot tables),
then executes the cheapest plan, recursively, re-planning after every step.
"""
from __future__ import annotations

import logging
import math
import random
import time

from .actions import ActionError, _fuel_value
from .pathfinder import GoalXZ

log = logging.getLogger("pyflayer.planner")

TIERS = ["wooden", "stone", "iron", "diamond", "netherite"]
INF = 1e9
# items that are never worth crafting backwards from storage blocks / nuggets
NO_UNCRAFT = {"iron_ingot", "gold_ingot", "copper_ingot", "diamond", "emerald", "coal", "redstone", "lapis_lazuli",
              "netherite_ingot", "raw_iron", "raw_gold", "raw_copper", "wheat", "slime_ball", "dried_kelp",
              "bone_meal", "honey_bottle", "string", "iron_nugget", "gold_nugget"}
MOB_DANGER = {"enderman": 60, "blaze": 80, "spider": 25, "skeleton": 30, "zombie": 15, "creeper": 40,
              "witch": 60, "ghast": 90, "wither_skeleton": 120}


class Planner:
    def __init__(self, bot):
        self.bot = bot
        self.gd = bot.gd
        self.log: list[str] = []
        self._memo: dict = {}
        self._near_cache: dict = {}

    # ------------------------------------------------------------------ helpers
    def _say(self, msg):
        log.info(msg)
        self.log.append(msg)
        self.bot.events.emit("plan", msg)

    def have(self, item: str) -> int:
        return self.bot.inventory.count(item)

    def _nearby_block(self, block: str, radius=96):
        # A persistent empty cache made the bot walk past real trees after chunks
        # loaded. Tie the cache to its current chunk and recheck absent blocks.
        x, _, z = self.bot.position
        key = (block, radius, math.floor(x / 16), math.floor(z / 16))
        cached = self._near_cache.get(key)
        if cached is not None and time.time() - cached[1] < (8 if cached[0] else 1):
            return cached[0]
        found = self.bot.find_blocks(block, radius, 1)
        pos = found[0] if found else None
        self._near_cache[key] = (pos, time.time())
        if len(self._near_cache) > 200:
            self._near_cache.clear()
        return pos

    def nearby_wood(self, radius=96) -> list[tuple[str, float]]:
        """Wood varieties actually loaded in this area, nearest first."""
        names = ("oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log",
                 "dark_oak_log", "mangrove_log", "cherry_log", "pale_oak_log",
                 "crimson_stem", "warped_stem")
        found = []
        for name in names:
            if name not in self.gd.block_states:
                continue
            pos = self._nearby_block(name, radius)
            if pos is not None:
                found.append((name, math.dist(pos, self.bot.position)))
        return sorted(found, key=lambda row: row[1])

    ALL_LOGS = ("oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log",
                "mangrove_log", "cherry_log", "pale_oak_log")

    def gather_wood(self, count: int = 8, radius: float = 96) -> str:
        """Harvest whichever wood is closest and reachable. Unreachable trees are skipped, and if none are left
        the bot widens its search and walks somewhere new instead of giving up."""
        if count < 1:
            return "no wood requested"
        logs = list(self.ALL_LOGS) + (["crimson_stem", "warped_stem"] if getattr(self.bot, "dimension", "") == "the_nether" else [])
        logs = [n for n in logs if n in self.gd.block_states]
        start = sum(self.bot.inventory.count(n) for n in logs)
        got = lambda: sum(self.bot.inventory.count(n) for n in logs) - start   # noqa: E731
        r = max(radius, 64)
        for rnd in range(6):
            if hasattr(self.bot, "_check"):
                self.bot._check()
            if got() >= count:
                break
            self._near_cache.clear()
            if not self.bot.find_blocks(logs, r, 1):
                self._say("no trees in reach; scouting for any tree type")
                self.explore_for(logs, max_legs=4, leg=60)
            try:
                self.bot.collect_block(logs, count - got(), radius=r, timeout=240)
            except ActionError as e:
                if any(k in str(e) for k in ("stopped", "died", "disconnected")):
                    raise
                self._say(f"  wood: {e}; trying other trees")
                if rnd >= 1:
                    self.explore_for(logs, max_legs=2, leg=60)   # these trees are unreachable: move on
                r = min(r + 48, 160)
        n = got()
        if n <= 0:
            raise ActionError("no reachable trees found after scouting")
        return f"gathered {n} logs"

    def required_tool(self, block: str) -> tuple[str | None, str | None]:
        """(tool kind, minimum tier) needed to harvest a block's drops."""
        gd = self.gd
        if not gd.requires_tool(block):
            return None, None
        kind = next((k for k in ("pickaxe", "shovel", "axe", "hoe") if block in gd.block_tag("mineable/" + k)), None)
        if block in ("cobweb",):
            return "sword", "wooden"
        tier = "wooden"
        for t, tag in (("diamond", "needs_diamond_tool"), ("iron", "needs_iron_tool"), ("stone", "needs_stone_tool")):
            if block in gd.block_tag(tag):
                tier = t
                break
        return kind or "pickaxe", tier

    def has_tool_for(self, block: str) -> bool:
        names = [it.name for it in self.bot.inventory.items()]
        return self.gd.best_tool_for(block, names)[2]

    def tool_item_for(self, block: str) -> str | None:
        kind, tier = self.required_tool(block)
        if not kind:
            return None
        if self.has_tool_for(block):
            return None
        return f"{tier}_{kind}"

    # ------------------------------------------------------------------ cost model
    def cost(self, item: str, count: int = 1, depth: int = 0, stack=()) -> float:
        if self.have(item) >= count:
            return 0.0
        if depth > 7 or item in stack:
            return INF
        key = (item, count, depth)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = INF   # cycle guard
        best = INF
        stack = stack + (item,)
        need = count - self.have(item)
        # mining
        for b in self.gd.item_from_blocks.get(item, []):
            c = self._mine_cost(b, need, depth, stack)
            best = min(best, c)
        # crafting
        if item not in NO_UNCRAFT or not self.gd.item_from_blocks.get(item):
            for r in self.gd.recipes_for(item):
                crafts = math.ceil(need / r.count)
                c = 1.0 + (0 if not r.needs_table else self._station_cost("crafting_table", depth, stack))
                for opts, n in r.ingredient_counts():
                    c += min((self.cost(o, n * crafts, depth + 1, stack) for o in self._rank_options(opts)[:4]),
                             default=INF)
                    if c >= best:
                        break
                best = min(best, c)
        # smelting
        for r in self.gd.smelt_recipes_for(item):
            c = 6.0 + need * 0.5 + self._station_cost("furnace", depth, stack) + self._fuel_cost(need, depth, stack)
            c += min((self.cost(o, need, depth + 1, stack) for o in self._rank_options(r.ingredients[0])[:3]),
                     default=INF)
            best = min(best, c)
        # hunting
        for ent in self.gd.item_from_entities.get(item, [])[:6]:
            near = self.bot.entities.nearest(self.bot.position, lambda e, t=ent: e.type == t, 64)
            c = (12 if near else 150) + MOB_DANGER.get(ent, 20) + need * 4
            best = min(best, c)
        self._memo[key] = best
        return best

    def _mine_cost(self, block, need, depth, stack):
        if self.gd.hardness(block) < 0:
            return INF
        pos = self._nearby_block(block)
        c = (4.0 + need * 2.0) if pos else 120.0 + need * 3
        if pos:
            c += min(40.0, math.dist(pos, self.bot.position) / 4)
        tool = self.tool_item_for(block)
        if tool:
            c += self.cost(tool, 1, depth + 1, stack) + 2
        return c

    def _station_cost(self, station, depth, stack):
        if self.have(station) or self._nearby_block(station, 32):
            return 0.0
        return self.cost(station, 1, depth + 1, stack)

    def _fuel_cost(self, items, depth, stack):
        counts = self.bot.inventory.counts()
        have = sum(_fuel_value(self.gd, n) * c for n, c in counts.items())
        if have >= items:
            return 0.0
        return min(self.cost("coal", math.ceil(items / 8), depth + 1, stack),
                   self.cost("oak_planks", math.ceil(items / 1.5), depth + 1, stack) + 1)

    def _wood_rank(self, item: str) -> float:
        """For wood items (logs/planks/...): how close is a tree of that wood? Lower is better."""
        for suf in ("_log", "_planks", "_wood", "_stem", "_hyphae"):
            if item.endswith(suf):
                wood = item[: -len(suf)]
                break
        else:
            return 0.0
        for log in (f"{wood}_log", f"{wood}_stem"):
            if self.have(log):
                return -1.0
        for log in (f"{wood}_log", f"{wood}_stem"):
            if log in self.gd.block_states:
                p = self._nearby_block(log, 128)
                if p:
                    return math.dist(self.bot.position, p)
        return 1e6                                     # no such trees around: don't go hunting for them

    def _rank_options(self, opts):
        counts = self.bot.inventory.counts()
        return sorted(opts, key=lambda o: (-counts.get(o, 0), self._wood_rank(o),
                                           0 if o.startswith(("oak", "cobblestone")) else 1))

    def _best_option(self, opts, n, depth=0, stack=()):
        counts = self.bot.inventory.counts()
        for o in opts:
            if counts.get(o, 0) >= n:
                return o
        ranked = self._rank_options(opts)[:8]
        if sum(1 for o in ranked if self._wood_rank(o) != 0.0) > 1:
            return ranked[0]                           # any wood works: use the trees closest to us
        return min(ranked, key=lambda o: self.cost(o, n, depth + 1, stack))

    # ------------------------------------------------------------------ execution
    def obtain(self, item: str, count: int = 1, depth: int = 0, deadline: float | None = None, stack=()) -> str:
        bot = self.bot
        deadline = deadline or time.time() + 900
        if depth == 0:
            self.log = []
        if depth > 8 or item in stack:
            raise ActionError(f"cannot plan {item}: recursion")
        stack = stack + (item,)
        attempts = 0
        if item in self.ALL_LOGS and depth > 0 and self.have(item) < count:
            # a recipe needs "some log": any reachable tree works, the caller re-picks the wood type it got
            self._say(f"{'  ' * depth}obtain {count}x wood: nearest reachable trees")
            return self.gather_wood(count - self.have(item))
        from . import liquids
        if item in ("water_bucket", "lava_bucket") and self.have(item) < count:
            while self.have(item) < count:
                if not self.have("bucket"):
                    self.obtain("bucket", 1, depth + 1, deadline, stack)
                self._say(f"{'  ' * depth}fill a bucket with {item.split('_')[0]}")
                liquids.fill_bucket(bot, item.split("_")[0])
            return f"have {self.have(item)}x {item}"
        if item == "obsidian" and self.have(item) < count and not self._nearby_block("obsidian", 128) \
                and liquids.find_sources(bot, "lava", 96, 1) and bot.dimension != "the_nether":
            # no obsidian around but there is lava: make it like a player (water over lava), then mine it
            self._say(f"{'  ' * depth}obtain {count}x obsidian: pour water over lava, then mine it")
            if not self.have("diamond_pickaxe") and not self.have("netherite_pickaxe"):
                self.obtain("diamond_pickaxe", 1, depth + 1, deadline, stack)
            if not self.have("water_bucket"):
                self.obtain("water_bucket", 1, depth + 1, deadline, stack)
            try:
                liquids.cast_obsidian(bot, count - self.have(item))
            except ActionError as e:
                self._say(f"{'  ' * depth}  casting failed: {e}")
        while self.have(item) < count:
            bot._check()
            attempts += 1
            if time.time() > deadline:
                raise ActionError(f"obtain {item} timed out")
            if attempts > 6:
                raise ActionError(f"gave up obtaining {item} after {attempts - 1} attempts")
            self._memo.clear()
            self._near_cache.clear()
            need = count - self.have(item)
            options = []
            for b in self.gd.item_from_blocks.get(item, []):
                options.append((self._mine_cost(b, need, depth, stack), "mine", b))
            if item not in NO_UNCRAFT or not self.gd.item_from_blocks.get(item):
                for r in self.gd.recipes_for(item):
                    crafts = math.ceil(need / r.count)
                    c = 1.0 + (0 if not r.needs_table else self._station_cost("crafting_table", depth, stack))
                    for opts, n in r.ingredient_counts():
                        c += min((self.cost(o, n * crafts, depth + 1, stack) for o in self._rank_options(opts)[:4]),
                                 default=INF)
                    options.append((c, "craft", r))
            for r in self.gd.smelt_recipes_for(item):
                c = 6.0 + need * 0.5 + self._station_cost("furnace", depth, stack) + self._fuel_cost(need, depth, stack)
                c += min((self.cost(o, need, depth + 1, stack) for o in self._rank_options(r.ingredients[0])[:3]),
                         default=INF)
                options.append((c, "smelt", r))
            for ent in self.gd.item_from_entities.get(item, [])[:6]:
                near = bot.entities.nearest(bot.position, lambda e, t=ent: e.type == t, 64)
                options.append(((12 if near else 150) + MOB_DANGER.get(ent, 20) + need * 4, "hunt", ent))
            options = [o for o in options if o[0] < INF]
            if not options:
                raise ActionError(f"no known way to obtain {item}")
            options.sort(key=lambda o: o[0])
            cost, how, what = options[0]
            self._say(f"{'  ' * depth}obtain {need}x {item}: {how} {getattr(what, 'rid', what)} (cost {cost:.0f})")
            try:
                if how == "mine":
                    self._do_mine(what, item, need, depth, deadline, stack)
                elif how == "craft":
                    self._do_craft(what, item, need, depth, deadline, stack)
                elif how == "smelt":
                    self._do_smelt(what, item, need, depth, deadline, stack)
                elif how == "hunt":
                    self._do_hunt(what, item, need)
            except ActionError as e:
                if any(k in str(e) for k in ("stopped", "died", "disconnected")):
                    raise
                self._say(f"{'  ' * depth}  step failed: {e}")
                if depth == 0 and attempts >= 6:
                    raise
        return f"have {self.have(item)}x {item}"

    def _do_mine(self, block, item, need, depth, deadline, stack):
        tool = self.tool_item_for(block)
        if tool:
            self._say(f"{'  ' * depth}  need {tool} for {block}")
            self.obtain(tool, 1, depth + 1, deadline, stack)
        if block in self.ALL_LOGS:
            self.gather_wood(max(1, need))             # any reachable tree will do for the planner's next pass
            return
        if not self._nearby_block(block, 128):
            self.explore_for([block])
        try:
            self.bot.collect_block(block, max(1, need), radius=128, timeout=max(60, min(600, deadline - time.time())))
        except ActionError as e:
            if "could not find/reach" not in str(e):
                raise
            self._say(f"  {block} here is unreachable; looking somewhere else")
            self.explore_for([block])
            self.bot.collect_block(block, max(1, need), radius=128, timeout=max(60, min(600, deadline - time.time())))

    def _do_craft(self, r, item, need, depth, deadline, stack):
        crafts = math.ceil(need / r.count)
        bot = self.bot
        # the table first: crafting it later could eat ingredients we already gathered
        if r.needs_table and not self.have("crafting_table") and not bot.find_blocks("crafting_table", 24, 1):
            self.obtain("crafting_table", 1, depth + 1, deadline, stack)
        for _ in range(4):
            missing = [(opts, n * crafts) for opts, n in r.ingredient_counts()
                       if max(bot.inventory.count(o) for o in opts) < n * crafts]
            if not missing:
                break
            picks = [(self._best_option(opts, cnt, depth, stack), cnt) for opts, cnt in missing]
            # crafted ingredients first (they may consume raw ones), raw materials last
            picks.sort(key=lambda p: 0 if self.gd.recipes_for(p[0]) else 1)
            for o, cnt in picks:
                self.obtain(o, cnt, depth + 1, deadline, stack)
        bot.craft(item, need)

    def _do_smelt(self, r, item, need, depth, deadline, stack):
        opts = r.ingredients[0]
        if max(self.bot.inventory.count(o) for o in opts) < need:
            o = self._best_option(opts, need, depth, stack)
            self.obtain(o, need, depth + 1, deadline, stack)
        counts = self.bot.inventory.counts()
        fuel = sum(_fuel_value(self.gd, n) * c for n, c in counts.items() if n not in opts)
        if fuel < need:
            if self.cost("coal", 1) < 40:
                self.obtain("coal", math.ceil((need - fuel) / 8), depth + 1, deadline, stack)
            else:
                logs = [n for n in counts if n.endswith("_log")]
                plank = (logs[0].replace("_log", "_planks") if logs else "oak_planks")
                self.obtain(plank, math.ceil((need - fuel) / 1.5) + 1, depth + 1, deadline, stack)
        if not self.have("furnace") and not self.bot.find_blocks("furnace", 24, 1):
            self.obtain("furnace", 1, depth + 1, deadline, stack)
        self.bot.smelt(item, need)

    def _do_hunt(self, ent, item, need):
        bot = self.bot
        end = time.time() + 240
        while time.time() < end and self.have(item) < need + 0:
            target = bot.entities.nearest(bot.position, lambda e: e.type == ent, 96)
            if not target:
                self.explore_for([], entity=ent)
                target = bot.entities.nearest(bot.position, lambda e: e.type == ent, 96)
                if not target:
                    raise ActionError(f"no {ent} found to hunt")
            bot.fight(target, timeout=40)
            bot.collect_items(radius=8, timeout=8)
            if bot.health < 8:
                raise ActionError("too hurt to keep hunting")

    def explore_for(self, blocks, entity=None, max_legs: int = 6, leg: int = 80):
        """Walk outward in a random direction until the target appears in loaded chunks."""
        bot = self.bot
        states = self.gd.states_of(*blocks) if blocks else None
        ang = random.uniform(0, 2 * math.pi)
        for i in range(max_legs):
            bot._check()
            x, y, z = bot.position
            tx, tz = x + math.cos(ang) * leg, z + math.sin(ang) * leg
            self._say(f"  exploring towards {tx:.0f},{tz:.0f} for {blocks or entity}")
            try:
                bot.goto(GoalXZ(tx, tz, 6), timeout=90)
            except ActionError as e:
                self._say(f"  explore leg failed: {e}")
                ang += math.pi / 2
            bot.wait_ticks(20)   # let chunks arrive
            if states is not None and len(states) and bot.world.find_blocks(bot.eye, states, 128, 1):
                return True
            if entity and bot.entities.nearest(bot.position, lambda e: e.type == entity, 96):
                return True
            ang += random.uniform(-0.7, 0.7)
        return False
