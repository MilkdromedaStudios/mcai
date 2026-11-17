"""The AI agent: an LLM (any OpenAI-compatible endpoint) driving the bot through tools, with
long-term memory, learned skills, the wiki, and fast non-LLM survival reflexes.

Without an LLM configured, the agent still works through chat commands (!obtain, !goto, !mine ...)
and the deterministic planner.
"""
from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
import time
import traceback

from ..bot.actions import ActionError
from ..bot.pathfinder import GoalNear
from ..bot.scanner import Scanner
from .llm import LLM, LLMNotConfigured
from .memory import Brain
from .tools import ToolBox
from .wiki import Wiki

SYSTEM_PROMPT = """You are {name}, an autonomous Minecraft player (Java Edition {version}) controlled through tools.
You see the world only through tool results. Be efficient: prefer high-level tools.

How to act:
- Use obtain_item for anything that needs a chain of steps (tools, ingots, armor, food...). It plans from the
  real game data (recipes, smelting, drops, tool tiers) and executes everything.
- Use scan_surroundings to see ores, trees, structures and biomes within 100 blocks; find_blocks for one type.
- Use recipe/block_info for exact game facts of this version; use wiki for strategy and anything else.
- Check recall/list_skills before hard tasks: reuse what you learned in earlier sessions.
- After learning something non-obvious, call remember. When a multi-step routine worked, save_skill it.
- Keep yourself alive: eat when food < 14, flee or build_shelter when health is low, avoid lava.
- When the task is done or impossible, call task_complete with a short summary.
Progression to beat the game: wood -> crafting table -> wooden/stone tools -> furnace + coal -> iron tools,
armor, bucket, shield -> food supply + bed -> diamonds -> obsidian -> nether portal -> nether fortress (blaze rods)
-> ender pearls (endermen) -> eyes of ender -> stronghold (throw eyes) -> end portal -> destroy end crystals ->
kill the ender dragon.
"""


class Agent:
    def __init__(self, bot, cfg, brain: Brain, wiki: Wiki, llm: LLM | None = None, server=None):
        self.bot = bot
        self.cfg = cfg
        self.brain = brain
        self.wiki = wiki
        self.llm = llm or LLM(cfg["llm"])
        self.server = server
        self.scanner = Scanner(bot)
        self.tools = ToolBox(self)
        self.log = logging.getLogger("pyflayer.agent")
        self.tasks: queue.Queue = queue.Queue()
        self.current_task: str | None = None
        self.task_done = threading.Event()
        self.task_result = ("", False)
        self.trace: list[dict] = []        # tool calls of the current task (for skill learning)
        self.history: list[str] = []       # short log lines for the GUI
        self.running = False
        self.paused_reflexes = False
        self._reflex_busy = False
        self._last_scan = 0.0
        self.interrupted_at = 0.0
        self.last_death: tuple | None = None
        self._thread: threading.Thread | None = None
        self._reflex_thread: threading.Thread | None = None
        bot.learned_hardness.update(brain.learned_hardness())
        bot.events.on("learn", self._on_learn)
        bot.events.on("death", self._on_death)
        bot.events.on("chat", self._on_chat)
        bot.events.on("crafted", lambda item, n: brain.bump(f"crafted:{item}", n))
        bot.events.on("dug", lambda pos, name: brain.bump("blocks_mined"))
        bot.events.on("plan", lambda msg: self._note(msg))

    # ================================================================== lifecycle
    def start(self):
        if self.running:
            return
        self.running = True
        self.brain.start_session("agent")
        self._thread = threading.Thread(target=self._loop, name="agent", daemon=True)
        self._thread.start()
        self._reflex_thread = threading.Thread(target=self._reflex_loop, name="reflexes", daemon=True)
        self._reflex_thread.start()

    def stop(self):
        self.running = False
        self.bot.stop()
        self.brain.end_session("stopped")

    def submit(self, task: str, source: str = "user"):
        self.tasks.put((task, source))
        self._note(f"queued task from {source}: {task}")

    def _note(self, msg: str):
        line = time.strftime("%H:%M:%S ") + msg
        self.history.append(line)
        del self.history[:-400]
        self.log.info(msg)

    # ================================================================== event hooks
    def _on_learn(self, kind, data):
        if kind == "hardness":
            self.brain.save_hardness(data["block"], data["hardness"])
            self.brain.remember(f"{data['block']} takes longer to break than expected (effective hardness "
                                f"{data['hardness']})", "mining", 0.6, "measurement")

    def _on_death(self, msg):
        pos = self.bot.block_pos
        self.brain.log("death", f"died at {pos} in {self.bot.dimension}: {msg}", 3.0)
        self.brain.set_waypoint(f"death@{int(time.time())}", self.bot.dimension, pos, "death", msg)
        self.brain.bump("deaths")
        self.last_death = (time.time(), self.bot.dimension, pos)
        self.interrupted_at = time.time()
        self.brain.remember(f"I died: {msg} (at y={pos[1]}, {self.bot.dimension}, while doing: "
                            f"{self.current_task or 'nothing'})", "danger", 0.9, "death")
        self._note(f"DIED: {msg}")

    def _on_chat(self, sender, text):
        if not self.cfg["agent"]["listen_to_chat"]:
            return
        owner = self.cfg["agent"].get("owner")
        if owner and sender.lower() != owner.lower():
            return
        name = self.bot.username.lower()
        t = text.strip()
        self.brain.log("chat", f"<{sender}> {t}")
        if t.startswith("!"):
            threading.Thread(target=self._chat_command, args=(sender, t), daemon=True).start()
            return
        if name in t.lower() or t.lower().startswith(("bot", "hey")):
            if self.llm.configured:
                self.submit(f"Player {sender} says to you in chat: \"{t}\". Respond in chat and do what they ask "
                            f"if reasonable.", source=sender)
            else:
                self.bot.chat(f"{sender}: no AI endpoint configured yet. Try !help")

    # ================================================================== chat commands (no LLM needed)
    def _chat_command(self, sender, text):
        bot = self.bot
        parts = text[1:].split()
        if not parts:
            return
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd == "help":
                bot.chat("!obtain <item> [n] | !mine <block> [n] | !craft <item> [n] | !smelt <item> [n] | !goto x y z | "
                         "!come | !follow | !stop | !scan | !status | !eat | !fight | !task <text> | !train")
            elif cmd == "stop":
                bot.stop()
                with self.tasks.mutex:
                    self.tasks.queue.clear()
                bot.chat("stopped")
            elif cmd == "status":
                bot.chat(self.status_text(short=True)[:250])
            elif cmd == "scan":
                res = self.scanner.scan(100)
                self.after_scan(res)
                for line in Scanner.summarize(res, 600).split("\n")[:4]:
                    bot.chat(line[:250])
            elif cmd == "come":
                e = bot.entities.nearest(bot.position, lambda e: e.name == sender, 512)
                if not e:
                    bot.chat("I can't see you")
                else:
                    bot.goto(GoalNear(e.x, e.y, e.z, 2), timeout=180)
                    bot.chat("here")
            elif cmd == "follow":
                e = bot.entities.nearest(bot.position, lambda e: e.name == sender, 512)
                if e:
                    bot.chat(bot.follow(e, 3, 300))
            elif cmd == "goto" and len(args) >= 3:
                bot.goto(float(args[0]), float(args[1]), float(args[2]), timeout=600)
                bot.chat(f"arrived {bot.block_pos}")
            else:
                mapping = {"obtain": "obtain_item", "get": "obtain_item", "mine": "mine_block", "craft": "craft",
                           "smelt": "smelt", "eat": "eat", "fight": "attack", "attack": "attack"}
                if cmd in mapping:
                    tool = mapping[cmd]
                    a = {}
                    if tool in ("obtain_item", "craft", "smelt"):
                        a = {"item": args[0], "count": int(args[1]) if len(args) > 1 else 1}
                    elif tool == "mine_block":
                        a = {"block": args[0], "count": int(args[1]) if len(args) > 1 else 1}
                    elif tool == "attack":
                        a = {"target": args[0] if args else "hostile"}
                    res = self.tools.call(tool, a)
                    bot.chat(res.split("\n")[0][:250])
                    self.brain.log("command", f"{sender}: {text} -> {res[:300]}")
                elif cmd == "task":
                    self.submit(" ".join(args), sender)
                    bot.chat("ok, on it" if self.llm.configured else "no AI endpoint configured")
                else:
                    bot.chat(f"unknown command {cmd}; try !help")
        except (ActionError, ValueError, IndexError) as e:
            bot.chat(f"failed: {e}"[:250])

    # ================================================================== perception text
    def status_text(self, short: bool = False) -> str:
        b = self.bot
        x, y, z = b.position
        biome = b.world.biome_at(*b.block_pos)
        s = (f"pos {x:.1f},{y:.1f},{z:.1f} {b.dimension} biome {biome} | health {b.health:.0f}/20 food {b.food}/20 "
             f"| {b.time_phase()} (t={b.time_of_day}) | holding {b.inventory.held_item!r}")
        if short:
            return s
        armor = {k: v.name for k, v in b.inventory.armor().items() if v}
        ents = b.entities.within(b.position, 24, lambda e: e.type not in ("item", "experience_orb"))
        items = b.entities.within(b.position, 12, lambda e: e.type == "item")
        s += f"\ninventory: {b.inventory.summary()}\narmor: {armor or 'none'}"
        if ents:
            s += "\nnearby: " + "; ".join(e.describe(b.position) for e in ents[:10])
        if items:
            s += f"\ndropped items nearby: {len(items)}"
        wps = self.brain.waypoints(b.dimension, b.position, None, 5)
        if wps:
            s += "\nknown places: " + "; ".join(f"{w['name']} {w['pos']}" for w in wps)
        return s

    def after_scan(self, res: dict):
        """Turn a scan into long-term spatial memory + 'seen' knowledge."""
        dim = self.bot.dimension
        self.brain.mark_seen("block", list(res["census"].keys()))
        self.brain.mark_seen("biome", list(res["biomes"].keys()))
        for st in res["structures"]:
            self.brain.set_waypoint(f"{st['structure']}@{st['pos'][0]},{st['pos'][2]}", dim, st["pos"], "structure")
            self.brain.mark_seen("structure", [st["structure"]])
        for ore, info in res["ores"].items():
            if ore in ("diamond_ore", "deepslate_diamond_ore", "ancient_debris", "emerald_ore", "gold_ore",
                       "deepslate_gold_ore", "iron_ore", "deepslate_iron_ore"):
                p = info["nearest"]
                self.brain.set_waypoint(f"{ore}@{p[0]},{p[1]},{p[2]}", dim, p, "ore")
        for name, info in res["biomes"].items():
            self.brain.set_waypoint(f"biome:{name}@{info['nearest'][0]},{info['nearest'][2]}", dim, info["nearest"], "biome")
        for u in ("crafting_table", "furnace", "chest", "bed", "nether_portal", "end_portal"):
            if u in res["useful_blocks"]:
                p = res["useful_blocks"][u]["nearest"]
                self.brain.set_waypoint(f"{u}@{p[0]},{p[1]},{p[2]}", dim, p, "station")
        self._last_scan = time.time()

    # ================================================================== reflexes (no LLM, fast)
    def _reflex_loop(self):
        last_check = 0.0
        while self.running:
            time.sleep(0.5)
            if self.paused_reflexes or not self.bot.spawned.is_set() or self.bot.dead:
                continue
            try:
                if time.time() - last_check >= 3.0:      # look at the surroundings every 3 seconds
                    last_check = time.time()
                    self._situation_check()
                self._reflexes()
            except ActionError:
                pass
            except Exception:  # noqa: BLE001
                self.log.debug("reflex error\n%s", traceback.format_exc())

    def situation(self) -> dict:
        """What is around the bot right now (cheap: a few block lookups)."""
        b = self.bot
        gd, w = b.gd, b.world
        fx, fy, fz = b.block_pos
        hx, hy, hz = (int(math.floor(v)) for v in b.eye)
        feet, head = w.get(fx, fy, fz), w.get(hx, hy, hz)
        below = w.get(fx, fy - 1, fz)
        name = lambda s: gd.block_name(s) if s is not None else "unloaded"  # noqa: E731
        return {
            "in_water": bool(feet is not None and gd.is_water[feet]) or b.state.in_water,
            "head_under_water": bool(head is not None and gd.is_water[head]),
            "in_lava": bool(feet is not None and gd.is_lava[feet]) or b.state.in_lava,
            "suffocating": bool(head is not None and gd.shape_kind[head] == 1 and not gd.is_liquid[head]),
            "on_fire_block": name(feet) in ("fire", "soul_fire") or name(below) == "magma_block",
            "feet": name(feet), "head": name(head), "below": name(below),
        }

    def _situation_check(self):
        """Every 3 s: re-read the surroundings and fix bad situations (water, lava, stuck in a block...)."""
        b = self.bot
        sit = self.situation()
        pos = b.position
        prev = getattr(self, "_last_pos", None)
        self._last_pos = (pos, time.time())
        moved = prev is None or math.dist(prev[0], pos) > 0.5
        action = b.current_action
        problem = None
        if sit["in_lava"] or sit["on_fire_block"]:
            problem = "lava"
        elif sit["suffocating"]:
            problem = "suffocating"
        elif sit["in_water"]:
            # swimming as part of a path is fine; idling or going nowhere in water is not
            self._water_since = getattr(self, "_water_since", 0) or time.time()
            if action == "idle" or (not moved and time.time() - self._water_since > 6):
                problem = "water"
        else:
            self._water_since = 0
        if not problem or self._reflex_busy:
            return
        self._reflex_busy = True
        try:
            self._note(f"awareness: {problem} (feet={sit['feet']}, head={sit['head']}) - fixing it")
            if not self._take_over():
                return
            if problem in ("lava", "water"):
                b.controls.jump = True
                self._note(b.escape_liquid())
            elif problem == "suffocating":
                hx, hy, hz = (int(math.floor(v)) for v in b.eye)
                b.dig((hx, hy, hz))
            self.brain.log("awareness", f"fixed {problem} at {b.block_pos}")
        except ActionError as e:
            self._note(f"awareness: could not fix {problem}: {e}")
        finally:
            b.controls.jump = False
            self._reflex_busy = False

    def wait_reflexes(self, timeout: float = 180):
        """Block while a reflex (fight/escape/eat) is controlling the bot."""
        end = time.time() + timeout
        while self._reflex_busy and time.time() < end:
            time.sleep(0.2)

    def _take_over(self) -> bool:
        """Interrupt the running action and wait until the bot is free."""
        b = self.bot
        if b.current_action != "idle" or b._action_depth:
            b.stop()
            self.interrupted_at = time.time()
            end = time.time() + 4
            while (b._action_depth or b.current_action != "idle") and time.time() < end:
                time.sleep(0.05)
        return b._action_depth == 0

    def _recover(self, threat):
        """Low health: get out of melee reach, eat, and wait to regenerate."""
        b = self.bot
        scaffold = b._scaffold_item()
        if scaffold and b.inventory.count(scaffold) >= 3 and threat.type not in ("spider", "cave_spider", "creeper"):
            self._note(f"reflex: low health ({b.health:.0f}) - pillaring up away from {threat.type}")
            b.escape_up(3)
        else:
            self._note(f"reflex: low health ({b.health:.0f}) - fleeing from {threat.type}")
            b.flee(threat.pos, 24, timeout=20)
        end = time.time() + 150
        while time.time() < end and self.running and not b.dead:
            if b.food < 20 and any(b.gd.food(i.name) for i in b.inventory.items()):
                try:
                    b.eat()
                except ActionError:
                    pass
            near = b.entities.nearest(b.position, lambda e: e.hostile, 8)
            if b.health >= 16 or (b.health >= 10 and not near):
                break
            if near and b.entity_distance(near) <= 3.0 and near.type != "creeper":
                b.attack(near)
            time.sleep(0.5)

    def _reflexes(self):
        b = self.bot
        if self._reflex_busy:
            return
        idle = b.current_action == "idle"
        hostile = b.entities.nearest(b.position, lambda e: e.hostile and e.type not in ("ender_dragon",), 12)
        dist = b.entity_distance(hostile) if hostile else 99
        action = None
        if hostile and b.health <= 8 and dist < 8:
            action = ("recover", hostile)
        elif hostile and hostile.type == "creeper" and dist < 5:
            action = ("creeper", hostile)
        elif hostile and dist < 3.5 and hostile.type not in ("creeper", "enderman") and b.health > 8:
            armed = max([b.gd.attack_damage(i.name) for i in b.inventory.items()] + [1.0]) >= 4
            weak = hostile.type in ("silverfish", "endermite", "slime", "magma_cube", "zombie", "husk")
            action = ("fight", hostile) if (armed or (weak and b.health > 14)) else ("evade", hostile)
        elif idle and b.food <= 14 and any(b.gd.food(i.name) for i in b.inventory.items()):
            action = ("eat", None)
        if action:
            self._reflex_busy = True
            try:
                if not self._take_over():
                    return
                kind, target = action
                if kind == "recover":
                    self._recover(target)
                elif kind == "creeper":
                    self._note("reflex: creeper close - backing off")
                    b.flee(target.pos, 10, timeout=6)
                elif kind == "evade":
                    self._note(f"reflex: unarmed vs {target.type} - taking cover")
                    try:
                        if b.time_phase() in ("night", "sunset"):
                            b.hide_underground(max_wait=420)
                        elif b._scaffold_item():
                            b.escape_up(3)
                        else:
                            b.flee(target.pos, 20, timeout=15)
                    except ActionError:
                        b.flee(target.pos, 20, timeout=15)
                elif kind == "fight":
                    self._note(f"reflex: defending against {target.type}")
                    b.fight(target, timeout=25)
                    b.collect_items(5, 4)
                elif kind == "eat":
                    b.eat()
            finally:
                self._reflex_busy = False
            return
        # drowning: hold jump while the head is under water
        head = b.world.get(*map(math.floor, b.eye))
        if head is not None and b.gd.is_water[head] and idle:
            b.controls.jump = True
        elif idle and b.controls.jump and not b.state.in_water:
            b.controls.jump = False
        # idle housekeeping: wear better armor, periodic scan into memory
        if idle and time.time() - self._last_scan > 90:
            res = self.scanner.scan(100)
            self.after_scan(res)
            b.equip_armor()

    # ================================================================== task loop
    def _loop(self):
        while self.running:
            try:
                task, source = self.tasks.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self.bot.spawned.is_set():
                self.bot.spawned.wait(60)
            try:
                self.run_task(task)
            except LLMNotConfigured as e:
                self._note(str(e))
                self.bot.chat("No AI endpoint configured (llm.base_url). Chat commands still work: !help")
            except Exception as e:  # noqa: BLE001
                self._note(f"task crashed: {e}")
                self.log.exception("task crashed")

    def finish_task(self, summary: str, success: bool):
        self.task_result = (summary, success)
        self.task_done.set()

    def _context_messages(self, task: str) -> list[dict]:
        b = self.bot
        mem = self.brain.recall_text(task, 6, 2200)
        skills = self.brain.skills(task, 6)
        skill_txt = "\n".join(f"- {s['name']}({', '.join(s['params'])}): {s['description']} "
                              f"[ok {s['successes']}/fail {s['failures']}]" for s in skills)
        sys = SYSTEM_PROMPT.format(name=b.username, version=b.gd.version)
        if mem:
            sys += "\n# Relevant memories\n" + mem
        if skill_txt:
            sys += "\n# Learned skills you can run_skill\n" + skill_txt
        ms = self.brain.milestones()
        done = [k for k, v in ms.items() if v["achieved"]]
        if done:
            sys += "\n# Milestones achieved in past sessions: " + ", ".join(done[:30])
        return [{"role": "system", "content": sys},
                {"role": "user", "content": f"TASK: {task}\n\nCurrent status:\n{self.status_text()}"}]

    def run_task(self, task: str, max_steps: int | None = None, time_limit: float = 1800) -> tuple[str, bool]:
        """Run one task with the LLM in a tool-calling loop. Returns (summary, success)."""
        if not self.llm.configured:
            raise LLMNotConfigured("LLM endpoint not configured (set llm.base_url / llm.api_key in config.json)")
        max_steps = max_steps or self.cfg["llm"].get("max_steps_per_task", 40)
        self.current_task = task
        self.task_done.clear()
        self.task_result = ("", False)
        self.trace = []
        t0 = time.time()
        self._note(f"TASK: {task}")
        self.brain.log("task_start", task, 1.5)
        messages = self._context_messages(task)
        schemas = self.tools.schemas()
        step = 0
        consecutive_errors = 0
        try:
            while step < max_steps and not self.task_done.is_set() and self.running:
                if time.time() - t0 > time_limit:
                    self.finish_task("time limit reached", False)
                    break
                step += 1
                try:
                    reply = self.llm.chat(self._trim(messages), schemas)
                except LLMNotConfigured:
                    raise
                except Exception as e:  # noqa: BLE001
                    consecutive_errors += 1
                    self._note(f"LLM error: {e}")
                    if consecutive_errors >= 3:
                        self.finish_task(f"LLM unavailable: {e}", False)
                        break
                    time.sleep(5 * consecutive_errors)
                    continue
                consecutive_errors = 0
                if not reply.tool_calls:
                    text = (reply.content or "").strip()
                    messages.append({"role": "assistant", "content": text})
                    if text:
                        self._note(f"LLM: {text[:200]}")
                    messages.append({"role": "user", "content": "Use a tool to make progress, or call task_complete."})
                    continue
                tc_list = reply.tool_calls[:3]
                messages.append({"role": "assistant", "content": reply.content or "",
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                                                for tc in tc_list]})
                for tc in tc_list:
                    self.wait_reflexes()
                    self._note(f"-> {tc.name}({json.dumps(tc.args)[:160]})")
                    t1 = time.time()
                    result = self.tools.call(tc.name, tc.args)
                    ok = not result.startswith("ERROR")
                    self.trace.append({"tool": tc.name, "args": tc.args, "ok": ok, "secs": round(time.time() - t1, 1)})
                    self._note(f"   {result[:220]}")
                    self.brain.log("action", f"{tc.name}({json.dumps(tc.args)[:200]}) -> {result[:400]}",
                                   1.0 if ok else 1.5)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                     "content": result[:3500]})
                    if self.task_done.is_set():
                        break
                # refresh status every few steps so the model does not reason on stale data
                if step % 4 == 0 and not self.task_done.is_set():
                    messages.append({"role": "user", "content": "Status update:\n" + self.status_text()})
            if not self.task_done.is_set():
                self.finish_task("step limit reached", False)
        finally:
            summary, success = self.task_result
            secs = time.time() - t0
            self.brain.log("task_end", f"{task} -> {'SUCCESS' if success else 'FAILED'}: {summary} ({secs:.0f}s)", 2.0)
            self.brain.bump("tasks_succeeded" if success else "tasks_failed")
            self._note(f"TASK {'DONE' if success else 'FAILED'}: {summary}")
            try:
                self._learn_from_task(task, summary, success, secs)
            except Exception as e:  # noqa: BLE001
                self.log.debug("reflection failed: %s", e)
            self.current_task = None
        return self.task_result

    @staticmethod
    def _trim(messages: list[dict], keep_tail: int = 14) -> list[dict]:
        """Keep the system + task messages and the most recent exchanges (tool results shortened)."""
        if len(messages) <= keep_tail + 2:
            return messages
        head = messages[:2]
        tail = messages[-keep_tail:]
        # never start the tail with orphan tool results
        while tail and tail[0]["role"] == "tool":
            tail = tail[1:]
        dropped = len(messages) - len(head) - len(tail)
        return head + [{"role": "user", "content": f"({dropped} earlier messages omitted)"}] + tail

    # ================================================================== learning
    def _learn_from_task(self, task, summary, success, secs):
        good = [t for t in self.trace if t["ok"] and t["tool"] not in (
            "task_complete", "get_status", "inventory", "recall", "remember", "list_skills", "wiki", "recipe",
            "block_info", "list_waypoints", "scan_surroundings", "nearby_entities", "find_blocks", "missing_for",
            "save_skill", "chat", "wait", "detect_structures")]
        # 1) automatic skill from a successful action sequence
        if success and 1 <= len(good) <= 14:
            name = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_")
            steps = [{"tool": t["tool"], "args": t["args"]} for t in good]
            n = self.brain.save_skill(name, f"learned: {task} ({summary[:120]})", steps, [])
            self.brain.skill_result(n, True, secs)
        # 2) reflection with the LLM -> lessons
        if not self.llm.configured:
            return
        trace_txt = "\n".join(f"{'OK ' if t['ok'] else 'ERR'} {t['tool']} {json.dumps(t['args'])[:120]}"
                              for t in self.trace[-30:])
        prompt = (f"You just {'completed' if success else 'FAILED'} the Minecraft task: {task}\n"
                  f"Outcome: {summary}\nActions:\n{trace_txt}\n\n"
                  "Write 1-3 short, specific, reusable lessons for next time (what worked, what to avoid, "
                  "exact numbers/blocks/coordinates if useful). Reply ONLY with JSON: "
                  "{\"lessons\": [{\"topic\": \"...\", \"text\": \"...\"}]}")
        try:
            r = self.llm.chat([{"role": "user", "content": prompt}], None, temperature=0.2, max_tokens=400)
            m = re.search(r"\{.*\}", r.content or "", re.S)
            data = json.loads(m.group(0)) if m else {}
            for lesson in data.get("lessons", [])[:3]:
                if isinstance(lesson, dict) and lesson.get("text"):
                    self.brain.remember(lesson["text"], lesson.get("topic", "lesson"), 0.75, "reflection")
        except Exception as e:  # noqa: BLE001
            self.log.debug("reflection parse failed: %s", e)

    def run_skill(self, name: str, params: dict) -> str:
        sk = self.brain.skill(name)
        if not sk:
            close = self.brain.skills(name, 3)
            return f"ERROR: no skill {name!r}" + (f"; similar: {[s['name'] for s in close]}" if close else "")
        t0 = time.time()
        out = []

        def fill(v):
            if isinstance(v, str):
                return re.sub(r"\{(\w+)\}", lambda m: str(params.get(m.group(1), m.group(0))), v)
            if isinstance(v, dict):
                return {k: fill(x) for k, x in v.items()}
            if isinstance(v, list):
                return [fill(x) for x in v]
            return v
        for i, st in enumerate(sk["steps"]):
            res = self.tools.call(st["tool"], fill(st.get("args", {})))
            out.append(f"{i + 1}. {st['tool']}: {res[:160]}")
            if res.startswith("ERROR"):
                self.brain.skill_result(sk["name"], False, time.time() - t0)
                return f"skill {name} failed at step {i + 1}:\n" + "\n".join(out)
        self.brain.skill_result(sk["name"], True, time.time() - t0)
        return f"skill {name} succeeded:\n" + "\n".join(out)
