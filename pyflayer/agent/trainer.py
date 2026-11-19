"""Overnight training (default 7 hours).

Three things happen in parallel:
 1. Knowledge: every block, item, mob, biome and structure of this exact version is written into the
    brain from the server's own data (recipes, drops, tools, tags, food...), then enriched with
    minecraft.wiki summaries (rate-limited, cached offline).
 2. Practice: a curriculum of milestones that leads to beating the game (wood -> ... -> ender dragon).
    With an LLM configured, each milestone is a task for the agent (which learns lessons + skills);
    without one, the deterministic planner practises it. Results, timings and failures are saved.
 3. Survey: periodic exploration legs that scan 100 blocks around the bot and record every block type,
    biome and structure actually seen (waypoints go into spatial memory).
A Markdown + JSON progress report is written continuously to data/training/.
"""
from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from ..bot.actions import ActionError

from ..bot import advanced
from ..bot.actions import ActionError
from ..bot.pathfinder import GoalXZ
from ..config import data_dir
from .wiki import title_from_id

log = logging.getLogger("pyflayer.trainer")

LOGS = ["oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log", "mangrove_log",
        "cherry_log", "pale_oak_log"]
MEATS = ["cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken", "cooked_rabbit", "bread"]


@dataclass
class Milestone:
    name: str
    goal: str                                   # natural-language goal for the LLM agent
    check: Callable                             # (trainer) -> bool
    run: Callable                               # (trainer) -> str   deterministic attempt
    minutes: float = 25.0


def _has(item, n=1):
    return lambda t: t.bot.inventory.count(item) >= n


def _has_any(items, n=1):
    return lambda t: sum(t.bot.inventory.count(i) for i in items) >= n


def _cheapest(t, items, n):
    t.bot.planner._memo.clear()
    t.bot.planner._near_cache.clear()
    return min(items, key=lambda i: t.bot.planner.cost(i, n))


def _obtain(item, n=1):
    return lambda t: t.bot.obtain(item, n, timeout=1500)


def _obtain_any(items, n):
    def run(t):
        return t.bot.obtain(_cheapest(t, items, n), n, timeout=1500)
    return run


def _any_logs(n):
    """Chop whatever trees are closest and reachable (any wood type) instead of hunting for one kind."""
    def run(t):
        return t.bot.planner.gather_wood(n - sum(t.bot.inventory.count(i) for i in LOGS) or n)
    return run


def _flag(name):
    return lambda t: bool(t.brain.milestones().get(name, {}).get("achieved"))


def _armor_set(t):
    out = []
    for piece in ("iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots"):
        if not t.bot.inventory.count(piece) and not any(
                a and a.name == piece for a in t.bot.inventory.armor().values()):
            out.append(t.bot.obtain(piece, 1, timeout=900))
    t.bot.equip_armor()
    return "; ".join(out) or "already have iron armor"


def _wearing_iron(t):
    return sum(1 for a in t.bot.inventory.armor().values() if a and a.name.startswith(("iron_", "diamond_"))) >= 4


def _enter(dim, portal):
    def run(t):
        if t.bot.dimension == dim:
            return f"already in {dim}"
        return advanced.enter_portal(t.bot, portal)
    return run


def _eyes(t):
    need = 12 - t.bot.inventory.count("ender_eye")
    if need <= 0:
        return "have 12 eyes"
    return t.bot.obtain("ender_eye", 12, timeout=3000)


def build_curriculum() -> list[Milestone]:
    M = Milestone
    return [
        M("wood", "Collect at least 8 logs of any wood type.", _has_any(LOGS, 8), _any_logs(8), 15),
        M("crafting_table", "Craft a crafting table and keep it.", _has("crafting_table"), _obtain("crafting_table"), 10),
        M("wooden_pickaxe", "Make a wooden pickaxe.", _has("wooden_pickaxe"), _obtain("wooden_pickaxe"), 10),
        M("stone_pickaxe", "Make a stone pickaxe.", _has("stone_pickaxe"), _obtain("stone_pickaxe"), 15),
        M("stone_sword", "Make a stone sword for defence.", _has("stone_sword"), _obtain("stone_sword"), 10),
        M("furnace", "Craft a furnace.", _has("furnace"), _obtain("furnace"), 15),
        M("fuel", "Get 8 coal or charcoal.", _has_any(["coal", "charcoal"], 8), _obtain_any(["coal", "charcoal"], 8), 20),
        M("torches", "Craft 16 torches.", _has("torch", 16), _obtain("torch", 16), 15),
        M("food", "Get at least 6 cooked meat (or bread) so you never starve.", _has_any(MEATS, 6),
          _obtain_any(MEATS, 6), 25),
        M("iron_pickaxe", "Make an iron pickaxe (mine iron ore, smelt raw iron).", _has("iron_pickaxe"),
          _obtain("iron_pickaxe"), 30),
        M("iron_sword", "Make an iron sword.", _has("iron_sword"), _obtain("iron_sword"), 20),
        M("bucket", "Make a bucket.", _has("bucket"), _obtain("bucket"), 20),
        M("shield", "Make a shield.", _has("shield"), _obtain("shield"), 20),
        M("iron_armor", "Make and wear a full set of iron armor.", _wearing_iron, _armor_set, 45),
        M("bed", "Make a bed (wool from sheep + planks).", lambda t: any(
            i.name.endswith("_bed") for i in t.bot.inventory.items()), _obtain("white_bed"), 30),
        M("diamonds", "Find and mine at least 3 diamonds (deep underground, y below 16).", _has("diamond", 3),
          _obtain("diamond", 3), 45),
        M("diamond_pickaxe", "Craft a diamond pickaxe.", _has("diamond_pickaxe"), _obtain("diamond_pickaxe"), 30),
        M("obsidian", "Get 10 obsidian (mine it with a diamond pickaxe).", _has("obsidian", 10), _obtain("obsidian", 10), 45),
        M("flint_and_steel", "Make flint and steel.", _has("flint_and_steel"), _obtain("flint_and_steel"), 20),
        M("nether_portal", "Build and light a nether portal.", _flag("nether_portal"),
          lambda t: advanced.build_nether_portal(t.bot), 25),
        M("enter_nether", "Go through the nether portal into the Nether.", lambda t: t.bot.dimension == "the_nether"
          or _flag("enter_nether")(t), _enter("the_nether", "nether_portal"), 15),
        M("blaze_rods", "Find a nether fortress and collect 6 blaze rods from blazes.", _has("blaze_rod", 6),
          _obtain("blaze_rod", 6), 60),
        M("ender_pearls", "Collect 12 ender pearls (hunt endermen, best at night or in warped forests).",
          _has("ender_pearl", 12), _obtain("ender_pearl", 12), 60),
        M("eyes_of_ender", "Craft 12 eyes of ender.", _has("ender_eye", 12), _eyes, 20),
        M("stronghold", "Find the stronghold portal room by throwing eyes of ender.", _flag("stronghold"),
          lambda t: advanced.find_stronghold(t.bot), 60),
        M("end_portal", "Fill the end portal frame with eyes of ender to activate it.",
          lambda t: bool(t.bot.find_blocks("end_portal", 32, 1)), lambda t: advanced.fill_end_portal(t.bot), 15),
        M("enter_end", "Jump into the end portal.", lambda t: t.bot.dimension == "the_end",
          _enter("the_end", "end_portal"), 15),
        M("kill_dragon", "Destroy the end crystals and kill the ender dragon.", _flag("kill_dragon"),
          lambda t: advanced.fight_dragon(t.bot), 60),
    ]



def checkpoint_path(bot: str):
    return data_dir("training") / f"{bot}-checkpoint.json"


def _atomic_json(path, data):
    import os
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def interrupted_runs(max_age_h: float = 72) -> list[dict]:
    """Training runs that were still 'running' when the program died (power cut, crash, unplugged)."""
    out = []
    for p in data_dir("training").glob("*-checkpoint.json"):
        try:
            ck = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if ck.get("state") == "running" and ck.get("remaining_s", 0) > 120 and \
                time.time() - ck.get("saved_at", 0) < max_age_h * 3600:
            out.append(ck)
    return out

class Trainer:
    def __init__(self, agent, hours: float = 7.0):
        self.agent = agent
        self.bot = agent.bot
        self.brain = agent.brain
        self.wiki = agent.wiki
        self.cfg = agent.cfg
        self.hours = hours
        self.curriculum = build_curriculum()
        self.start_time = 0.0
        self.end_time = 0.0
        self.running = False
        self.status = "idle"
        self.current = ""
        self.knowledge_progress = {"gamedata": 0, "wiki": 0, "wiki_total": 0}
        self.events: list[str] = []
        self.report_dir = data_dir("training")
        self._threads: list[threading.Thread] = []
        self.done_this_run: set[str] = set()
        self.last_error = ""
        self.completed_attempts = 0
        self.successful_attempts = 0
        self.recovery_attempts: dict[str, int] = {}
        self._finish_lock = threading.Lock()
        self._finished = False
        self._autosave_stop = threading.Event()
        self._model_exports = 0
        self.ai = "default"                 # which AI the run uses (saved in the checkpoint)
        self.resume_from: dict | None = None

    # ================================================================== checkpoint (power loss)
    @property
    def checkpoint_path(self):
        return checkpoint_path(self.bot.username)

    def checkpoint(self, state: str = "running"):
        """Write the run's progress to disk (atomically, fsynced) so an unplugged Pi can carry on later."""
        if not getattr(self.bot, "username", None):
            return
        data = {"bot": self.bot.username, "state": state, "ai": self.ai, "hours": self.hours,
                "remaining_s": round(self.time_left), "current": self.current,
                "done": sorted(self.done_this_run), "recovery_attempts": self.recovery_attempts,
                "completed_attempts": self.completed_attempts, "successful_attempts": self.successful_attempts,
                "dimension": getattr(self.bot, "dimension", ""), "saved_at": time.time()}
        _atomic_json(self.checkpoint_path, data)

    def note(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        self.events.append(line)
        del self.events[:-300]
        log.info(msg)
        self.agent._note("[train] " + msg)

    # ================================================================== lifecycle
    def start(self, background: bool = True):
        if self.running:
            raise RuntimeError("Overnight training is already running for this bot")
        self.last_error = ""
        self.status = "starting"
        self.current = ""
        self.completed_attempts = 0
        self.successful_attempts = 0
        self.done_this_run.clear()
        self.recovery_attempts.clear()
        ck = self.resume_from
        if ck:
            self.done_this_run.update(ck.get("done", []))
            self.recovery_attempts.update(ck.get("recovery_attempts", {}))
            self.completed_attempts = int(ck.get("completed_attempts", 0))
            self.successful_attempts = int(ck.get("successful_attempts", 0))
        self._finished = False
        self._model_exports = 0
        self._autosave_stop.clear()
        self.running = True
        self.start_time = time.time()
        self.end_time = self.start_time + self.hours * 3600
        self.brain.start_session("training")
        if ck:
            self.note(f"RESUMED after an interruption: {self.hours * 60:.0f} min left, "
                      f"{len(self.done_this_run)} steps already done, was on '{ck.get('current') or '-'}'")
        else:
            self.note(f"overnight training started for {self.hours:.1f} h")
        self.checkpoint()
        self._save_models()  # create both files before any practice; crash-safe snapshot
        autosave = threading.Thread(target=self._autosave_loop,
                                    name="train-model-autosave", daemon=True)
        autosave.start()
        self._threads.append(autosave)
        self.note("training plan: pursue wood -> tools -> food -> iron -> diamonds -> Nether -> End; "
                  "explore only when resources cannot be found")
        if not self.agent.llm.configured:
            self.note("No AI model configured: using the deterministic recipe/resource planner for practice")
        k = threading.Thread(target=self._knowledge_worker, name="train-knowledge", daemon=True)
        k.start()
        self._threads.append(k)
        if background:
            m = threading.Thread(target=self._practice_loop, name="train-practice", daemon=True)
            m.start()
            self._threads.append(m)
        else:
            self._practice_loop()

    def stop(self):
        if self.running:
            self.status = "stopping"
            self.note("training stop requested")
        self.running = False
        self._autosave_stop.set()
        self.bot.stop()

    def _save_models(self):
        """Update both full backup and reusable cross-seed model atomically."""
        full = self.brain.export_latest()
        universal = self.brain.export_universal()
        self._model_exports += 1
        self.note(f"MODEL SAVED: {universal} | full backup: {full}")
        return universal

    def _autosave_loop(self):
        # Separate from the action worker: save even during a long mining task.
        n = 0
        while not self._autosave_stop.wait(60):
            if not self.running:
                break
            n += 1
            try:
                self.brain.commit()
                self.checkpoint()
                if n % 3 == 0:                      # every 3 minutes
                    self._save_models()
                    srv = getattr(self.agent, "server", None)
                    if srv and getattr(srv, "running", False):
                        srv.command("save-all")     # write the Minecraft world to disk too
            except Exception as e:  # noqa: BLE001
                log.exception("model autosave failed")
                self.note(f"MODEL AUTOSAVE ERROR: {e}")

    @property
    def time_left(self) -> float:
        return max(0.0, self.end_time - time.time())

    # ================================================================== 1) knowledge
    def _knowledge_worker(self):
        kb = KnowledgeBuilder(self.bot.gd, self.brain, self.wiki, self.note,
                              lambda: self.running and time.time() < self.end_time, self.knowledge_progress)
        try:
            kb.build_gamedata()
            if self.cfg["training"]["wiki_crawl"] and self.wiki.online:
                kb.crawl_wiki()
        except Exception as e:  # noqa: BLE001
            self.note(f"knowledge worker error: {e}")

    # ================================================================== 2/3) practice + survey
    def _ensure_connected(self) -> bool:
        b = self.bot
        if b.dead:
            end = time.time() + 30
            while b.dead and b.connected and time.time() < end:
                time.sleep(0.5)
            if b.dead and b.connected:
                b.respawn()
                time.sleep(2)
        if b.connected and b.spawned.is_set() and not b.dead:
            return True
        srv = self.agent.server
        for attempt in range(20):
            if not self.running:
                return False
            try:
                if srv and not srv.running:
                    self.note("restarting the Minecraft server")
                    srv.start()
                if not b.connected:
                    self.note("reconnecting bot")
                    b.connect()
                if b.wait_spawn(60):
                    b.wait_chunks()
                    return True
            except Exception as e:  # noqa: BLE001
                self.note(f"reconnect failed: {e}")
            time.sleep(min(60, 5 * (attempt + 1)))
        return False

    def _survival_upkeep(self):
        b = self.bot
        self.agent.wait_reflexes()
        ld = self.agent.last_death
        if ld and time.time() - ld[0] < 240 and ld[1] == b.dimension and not b.dead:
            self.agent.last_death = None
            self.note(f"recovering items dropped at death point {ld[2]}")
            try:
                if math.dist(ld[2], b.position) < 400:
                    b.goto(GoalXZ(ld[2][0], ld[2][2], 2), timeout=180)
                    b.collect_items(10, 25)
            except ActionError as e:
                self.note(f"item recovery failed: {e}")
        try:
            n = b.drop_junk()
            if n:
                self.note(f"dropped {n} junk items to free inventory space")
            if b.food <= 14 and any(b.gd.food(i.name) for i in b.inventory.items()):
                b.eat()
            b.equip_armor()
            has_bed = any(i.name.endswith("_bed") for i in b.inventory.items())
            if b.time_phase() == "night" and b.dimension == "overworld" and not has_bed:
                self.status = "sheltering for the night"
                self.note("night: hiding underground until morning")
                try:
                    self.note(b.hide_underground())
                except ActionError as e:
                    self.note(f"could not hide ({e}); pillaring up instead")
                    if b._scaffold_item():
                        b.escape_up(3)
            if b.time_phase() == "night" and b.dimension == "overworld" and has_bed:
                pos = b.find_or_place(next(i.name for i in b.inventory.items() if i.name.endswith("_bed")))
                if pos:
                    b.activate_block(pos)
                    b.wait_ticks(60)
                    b.dig(pos)     # take the bed with us
                    b.collect_items(5, 5)
        except ActionError as e:
            self.note(f"upkeep: {e}")

    def survey(self, distance: int = 140):
        """Exploration leg: walk somewhere new and record everything seen within 100 blocks."""
        b = self.bot
        ang = random.uniform(0, 2 * math.pi)
        x, _, z = b.position
        self.current = "survey"
        try:
            b.goto(GoalXZ(x + math.cos(ang) * distance, z + math.sin(ang) * distance, 6), timeout=240)
        except ActionError as e:
            self.note(f"survey leg ended early: {e}")
        b.wait_ticks(30)
        res = self.agent.scanner.scan(100)
        self.agent.after_scan(res)
        seen = len(self.brain.seen("block"))
        self.note(f"survey: {res['block_types_seen']} block types here, {seen} block types seen in total; "
                  f"structures: {[s['structure'] for s in res['structures']] or 'none'}")

    def _practice_loop(self):
        """Work on an actual goal immediately, not a random 140-block walk.

        Only explore after the current goal repeatedly fails, and expose a
        terminal blocked/error state rather than silently looping surveys.
        """
        last_report = 0.0
        fails: dict[str, int] = {}
        try:
            while self.running and time.time() < self.end_time:
                if not self._ensure_connected():
                    if self.running:
                        self.status = "error"
                        self.last_error = "Lost connection to the Minecraft server after reconnect attempts"
                        self.note(self.last_error)
                    break
                if time.time() - last_report > 120:
                    self.write_report()
                    last_report = time.time()

                # The agent's queued goals/chat commands also use this lock.
                # Only one controller can move/dig/place at a time.
                with self.agent._actuation_lock:
                    if not self.running:
                        break
                    self._survival_upkeep()
                    if self._priority_finds():
                        continue
                    ms = self._next_milestone(fails)
                    if ms is None:
                        self.status = "completed"
                        self.current = "all milestones complete"
                        self.note("every curriculum milestone achieved or verified; training complete")
                        break
                    attempts = fails.get(ms.name, 0)
                    if attempts >= 3:
                        recoveries = self.recovery_attempts.get(ms.name, 0)
                        if recoveries >= 3:
                            self.status = "blocked"
                            self.last_error = (
                                f"Could not complete '{ms.name}' after {attempts + 3 * recoveries} "
                                "failed attempts across multiple locations. "
                                "See the training report for the specific errors."
                            )
                            self.note("TRAINING BLOCKED: " + self.last_error)
                            break
                        self.recovery_attempts[ms.name] = recoveries + 1
                        self.status = "recovering"
                        self.note(f"'{ms.name}' failed 3 times; looking for a different nearby "
                                  f"resource area (recovery {recoveries + 1}/3)")
                        self.survey(distance=55)
                        fails[ms.name] = 0
                        continue
                    self.status = f"practising {ms.name}"
                    self.current = ms.name
                    self._attempt(ms, fails)
                    # One measured dry-ground skill drill after every two
                    # curriculum attempts. World actions use the same lock.
                    if (self.running and self.completed_attempts and self.completed_attempts % 2 == 0
                            and getattr(self.agent, 'skill_coach', None)):
                        self.status = "skill practice"
                        result = self.agent.skill_coach.practice_one()
                        self.note(result)
        except Exception as e:  # noqa: BLE001
            log.exception("training loop crashed")
            self.status = "error"
            self.last_error = f"{type(e).__name__}: {e}"
            self.note("TRAINING ERROR: " + self.last_error)
        finally:
            self.running = False
            if self.status not in ("completed", "blocked", "error"):
                self.status = "finished" if time.time() >= self.end_time else "stopped"
            self.note(f"training ended: {self.status}"
                      + (f" - {self.last_error}" if self.last_error else ""))
            try:
                self.finish()
            except Exception as e:  # noqa: BLE001
                log.exception("could not save training report")
                self.status = "error"
                self.last_error = f"Could not save training results: {e}"
                self.note(self.last_error)

    def run_until(self, target: str, max_hours: float = 12.0) -> str:
        """Work through the beat-the-game plan, step by step, until `target` is reached (used for spoken goals
        like 'get to the end'). Each step is retried; steps already done are skipped."""
        names = [m.name for m in self.curriculum]
        if target not in names:
            raise ValueError(f"unknown goal {target}")
        self.running = True
        self.start_time = time.time()
        self.end_time = self.start_time + max_hours * 3600
        self.status = f"goal: {target}"
        fails: dict[str, int] = {}
        try:
            for ms in self.curriculum[:names.index(target) + 1]:
                while True:
                    if not self.running:
                        return "stopped"
                    if time.time() > self.end_time:
                        return f"ran out of time at '{ms.name}'"
                    if not self._ensure_connected():
                        return "lost the connection to the server"
                    try:
                        done = ms.check(self)
                    except Exception:  # noqa: BLE001
                        done = False
                    if done or ms.name in self.done_this_run:
                        self.done_this_run.add(ms.name)
                        break
                    self._survival_upkeep()
                    if self._priority_finds():
                        continue
                    self.current = ms.name
                    self.bot.chat(f"Next step: {ms.goal}"[:250])
                    self._attempt(ms, fails)
                    if fails.get(ms.name, 0) >= 4:
                        return f"I'm stuck at '{ms.goal}' after 4 tries - check the log"
            return f"done! ({target.replace('_', ' ')})"
        finally:
            self.running = False
            self.status = "idle"
            self.write_report()

    def _priority_finds(self) -> bool:
        """Saved diamonds (etc.) we can now mine come first - don't walk past them to do the next step."""
        finds = getattr(self.agent, "finds", None)
        if not finds or not finds.pending():
            return False
        prev = self.current
        self.current = "mining saved finds"
        try:
            return finds.run_pending()
        except ActionError as e:
            if any(k in str(e) for k in ("stopped", "died", "disconnected")):
                raise
            self.note(f"priority finds: {e}")
            return False
        finally:
            self.current = prev

    def _next_milestone(self, fails):
        """Follow prerequisites in order; never skip wood/tools to wander aimlessly."""
        for ms in self.curriculum:
            if ms.name in self.done_this_run:
                continue
            try:
                if ms.check(self):
                    self.done_this_run.add(ms.name)
                    rec = self.brain.milestones().get(ms.name)
                    if not rec or not rec["achieved"]:
                        self.brain.milestone(ms.name, True, 0)
                        self.note(f"milestone already satisfied: {ms.name}")
                    continue
            except Exception as e:  # noqa: BLE001
                self.note(f"milestone check '{ms.name}' failed: {e}")
            return ms
        return None

    def _verified(self, ms: Milestone) -> bool:
        """A claimed AI success is not proof of a completed Minecraft action."""
        if ms.check(self):
            return True
        if ms.name == "nether_portal":
            return bool(self.bot.find_blocks("nether_portal", 32, 1))
        if ms.name == "stronghold":
            return bool(self.bot.find_blocks("end_portal_frame", 112, 1))
        if ms.name == "kill_dragon" and self.bot.dimension == "the_end":
            # An actual End gateway is produced after a successful dragon fight.
            return bool(self.bot.find_blocks("end_gateway", 200, 1))
        return False

    def _attempt(self, ms: Milestone, fails):
        t0 = time.time()
        self.note(f"milestone '{ms.name}': {ms.goal}")
        err = ""
        ok = False
        try:
            if self.agent.llm.configured and ms.name != "wood":
                goal = ms.goal + " Tip: the obtain_item tool can do most item goals automatically."
                summary, _ = self.agent.run_task(goal, time_limit=ms.minutes * 60)
                ok = self._verified(ms)
                if not ok:
                    err = summary or "AI did not complete the milestone"
                    if self.running:
                        self.note(f"AI did not verify '{ms.name}'; trying the built-in resource planner")
                        try:
                            res = ms.run(self)
                            self.note(f"  planner -> {str(res)[:200]}")
                            ok = self._verified(ms)
                            if not ok:
                                err += "; planner: " + str(res)[:150]
                        except ActionError as e:
                            err += "; planner: " + str(e)
            else:
                res = ms.run(self)
                self.note(f"  planner -> {str(res)[:200]}")
                ok = self._verified(ms)
                if not ok:
                    err = f"goal still not verified after planner attempt: {str(res)[:150]}"
        except ActionError as e:
            err = str(e)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            log.exception("milestone crashed")
        secs = time.time() - t0
        self.completed_attempts += 1
        try:
            self.checkpoint()                       # after every step, not only once a minute
        except OSError:
            pass
        if not ok and (err == "stopped" or "bot died" in err) and time.time() - self.agent.interrupted_at < secs + 5:
            self.note(f"milestone '{ms.name}' interrupted ({err}); will retry")
            self.agent.wait_reflexes()
            return
        self.brain.milestone(ms.name, ok, secs, err)
        if ok:
            fails.pop(ms.name, None)
            self.successful_attempts += 1
            self.recovery_attempts.pop(ms.name, None)
            self.done_this_run.add(ms.name)
            self.note(f"milestone ACHIEVED: {ms.name} in {secs / 60:.1f} min")
            self.brain.remember(f"Achieved '{ms.name}' in {secs / 60:.1f} minutes. Approach: {ms.goal}",
                                "milestone", 0.8, "training")
        else:
            fails[ms.name] = fails.get(ms.name, 0) + 1
            self.last_error = f"{ms.name}: {err[:230] or 'could not verify progress'}"
            self.note(f"milestone failed ({fails[ms.name]}/3 before recovery): {self.last_error}")
            if err:
                self.brain.remember(f"Attempt at '{ms.name}' failed: {err[:250]}", "failure", 0.6, "training")

    # ================================================================== reporting
    def progress(self) -> dict:
        gd = self.bot.gd
        ms = self.brain.milestones()
        seen = self.brain.seen("block")
        total_blocks = len([b for b in gd.block_names if b not in ("air", "cave_air", "void_air")])
        return {
            "elapsed_h": round((time.time() - self.start_time) / 3600, 2) if self.start_time else 0,
            "hours": self.hours,
            "status": self.status,
            "current": self.current,
            "last_error": self.last_error,
            "model_exports": self._model_exports,
            "universal_model_path": str(self.brain.path.with_name(self.brain.path.stem + "-universal-model.json.gz")),
            "full_model_path": str(self.brain.path.with_name(self.brain.path.stem + "-export.json.gz")),
            "skills": self.agent.skill_coach.mastery() if getattr(self.agent, "skill_coach", None) else {},
            "completed_attempts": self.completed_attempts,
            "successful_attempts": self.successful_attempts,
            "recent_events": self.events[-20:],
            "milestones": [{"name": m.name, "achieved": bool(ms.get(m.name, {}).get("achieved")),
                            "attempts": ms.get(m.name, {}).get("attempts", 0),
                            "minutes": round((ms.get(m.name, {}).get("seconds") or 0) / 60, 1),
                            "last_error": ms.get(m.name, {}).get("last_error")} for m in self.curriculum],
            "knowledge": {
                "blocks_known_gamedata": self.brain.knowledge_count("block"),
                "blocks_total": total_blocks,
                "entries_with_wiki": self.brain.knowledge_count(source_like="wiki"),
                "wiki_titles_total": self.knowledge_progress.get("wiki_total", 0),
                "block_types_seen_in_world": len(seen),
                "biomes_seen": len(self.brain.seen("biome")),
                "structures_seen": sorted(self.brain.seen("structure")),
            },
            "brain": self.brain.overview(),
            "stats": self.brain.stats(),
        }

    def write_report(self) -> Path:
        p = self.progress()
        stamp = self.bot.username + "-" + time.strftime("%Y%m%d", time.localtime(self.start_time or time.time()))
        (self.report_dir / f"training-{stamp}.json").write_text(json.dumps(p, indent=2))
        k = p["knowledge"]
        lines = [f"# PyFlayer overnight training report ({stamp})", "",
                 f"Elapsed: {p['elapsed_h']} h of {p['hours']} h - status: {p['status']}", "",
                 f"Current goal: {p['current']} | Attempted: {p['completed_attempts']} | "
                 f"Succeeded: {p['successful_attempts']}", "",
                 f"Last error: {p['last_error'] or 'none'}", "",
                 "## Milestones (beat-the-game curriculum)", "", "| milestone | done | attempts | minutes | last error |",
                 "|---|---|---|---|---|"]
        for m in p["milestones"]:
            lines.append(f"| {m['name']} | {'yes' if m['achieved'] else 'no'} | {m['attempts']} | {m['minutes']} | "
                         f"{(m['last_error'] or '')[:80]} |")
        lines += ["", "## Measured skill practice", ""]
        for name, stats in p["skills"].items():
            lines.append(f"- {name}: {stats['successes']}/{stats['attempts']} verified, "
                         f"{'certified' if stats['certified'] else 'training'}; "
                         f"last: {stats['last_result'][:100]}")
        lines += ["", "## Knowledge", "",
                  f"- Blocks with full game-data knowledge: {k['blocks_known_gamedata']} / {k['blocks_total']}",
                  f"- Knowledge entries enriched from minecraft.wiki: {k['entries_with_wiki']} "
                  f"(of {k['wiki_titles_total']} titles)",
                  f"- Block types actually seen in the world: {k['block_types_seen_in_world']}",
                  f"- Biomes seen: {k['biomes_seen']}; structures seen: {', '.join(k['structures_seen']) or 'none'}",
                  "", "## Brain", "", "```", json.dumps(p["brain"], indent=1), "```", "",
                  "## Recent events", "", *[f"- {e}" for e in self.events[-40:]]]
        path = self.report_dir / f"training-{stamp}.md"
        path.write_text("\n".join(lines))
        return path

    def finish(self):
        # Runtime.shutdown and the training worker may call finish together.
        with self._finish_lock:
            if self._finished:
                return
            self._autosave_stop.set()
            path = self.write_report()
            done = sum(1 for m in self.progress()["milestones"] if m["achieved"])
            self.brain.end_session(f"training: {done}/{len(self.curriculum)} milestones")
            universal = self._save_models()
            self._finished = True
            try:
                left = self.time_left
                self.checkpoint("finished" if left < 60 or self.status == "completed" else "stopped")
            except OSError:
                pass
            self.note(f"training finished: {done}/{len(self.curriculum)} milestones; "
                      f"report {path}; universal model {universal}")


class KnowledgeBuilder:
    """Fills the brain's knowledge base: game data for everything, then wiki summaries + strategy guides."""

    GUIDES = ("Tutorial:Beating the game", "Tutorial:Your first day", "Tutorial:Mining", "Nether portal",
              "Nether fortress", "Stronghold", "Eye of Ender", "Ender Dragon", "End crystal",
              "Tutorial:Defeating the Ender Dragon", "Tutorial:Surviving your first night", "Blaze",
              "Enderman", "Tutorial:Strip mining", "Diamond", "Bastion Remnant")

    def __init__(self, gd, brain, wiki, note=print, keep_going=lambda: True, progress: dict | None = None):
        self.gd, self.brain, self.wiki, self.note, self.keep_going = gd, brain, wiki, note, keep_going
        self.progress = progress if progress is not None else {"gamedata": 0, "wiki": 0, "wiki_total": 0}

    def build_gamedata(self) -> int:
        gd = self.gd
        n = 0
        for b in gd.block_names:
            tags = [t for t in ("mineable/pickaxe", "mineable/axe", "mineable/shovel", "mineable/hoe", "logs",
                                "leaves", "flowers", "beds", "doors", "climbable", "infiniburn_overworld")
                    if b in gd.block_tag(t)]
            text = gd.describe_block(b) + (f"; tags: {', '.join(tags)}" if tags else "")
            if b in gd.item_ids:
                text += "\n" + gd.describe_item(b)
            self.brain.set_knowledge(b, "block", text, "gamedata")
            n += 1
        for i in gd.item_ids:
            if i in gd.block_index:
                continue
            self.brain.set_knowledge(i, "item", gd.describe_item(i), "gamedata")
            n += 1
        from ..bot.entities import DIMENSIONS, HOSTILE, NEUTRAL_HOSTILE_IF_PROVOKED
        for e in gd.entity_type_ids:
            drops = sorted(gd.entity_drops.get(e, []))
            kind = "hostile" if e in HOSTILE else ("neutral" if e in NEUTRAL_HOSTILE_IF_PROVOKED else "passive/other")
            w, h = DIMENSIONS.get(e, (0.6, 1.8))
            self.brain.set_knowledge(e, "mob", f"{e}: {kind} entity, size {w}x{h}; drops: {', '.join(drops) or 'nothing'}",
                                     "gamedata")
            n += 1
        root = gd.dir / "data" / "worldgen"
        for p in sorted((root / "biome").glob("*.json")) if (root / "biome").exists() else []:
            try:
                d = json.loads(p.read_text())
                self.brain.set_knowledge(p.stem, "biome", f"{p.stem}: biome; temperature {d.get('temperature')}, "
                                         f"downfall {d.get('downfall')}, precipitation {d.get('has_precipitation')}",
                                         "gamedata")
                n += 1
            except ValueError:
                pass
        for p in sorted((root / "structure").glob("*.json")) if (root / "structure").exists() else []:
            try:
                d = json.loads(p.read_text())
                biomes = d.get("biomes")
                self.brain.set_knowledge(p.stem, "structure", f"{p.stem}: structure (type {d.get('type')}), "
                                         f"generates in {biomes if isinstance(biomes, str) else 'several biomes'}",
                                         "gamedata")
                n += 1
            except ValueError:
                pass
        self.brain.commit()
        self.progress["gamedata"] = n
        self.note(f"knowledge base: {n} entries from game data (every block, item, mob, biome, structure)")
        return n

    def crawl_wiki(self) -> int:
        gd = self.gd
        names = list(gd.block_names) + [i for i in gd.item_ids if i not in gd.block_index] + list(gd.entity_type_ids)
        names = [n for n in names if n not in ("air", "cave_air", "void_air")]
        titles: dict[str, list[str]] = {}
        for n in names:
            titles.setdefault(title_from_id(n), []).append(n)
        todo = list(titles)
        self.progress["wiki_total"] = len(todo)
        for i in range(0, len(todo), 20):
            if not self.keep_going():
                break
            res = self.wiki.summaries(todo[i:i + 20])
            for t, summary in res.items():
                for n in titles.get(t, []):
                    kind = "block" if n in gd.block_index else ("mob" if n in gd.entity_type_ids else "item")
                    self.brain.set_knowledge(n, kind, summary[:1500], "wiki")
            self.brain.commit()
            self.progress["wiki"] = self.brain.knowledge_count(source_like="wiki")
            if i % 400 == 0:
                self.note(f"wiki crawl: {i + 20}/{len(todo)} titles")
        for page in self.GUIDES:
            if not self.keep_going():
                break
            text = self.wiki.page(page, 12000)
            self.brain.set_knowledge("guide:" + page, "guide", text[:4000], "wiki")
        self.brain.commit()
        n = self.brain.knowledge_count(source_like="wiki")
        self.note(f"wiki knowledge: {n} entries cached for offline use")
        return n
