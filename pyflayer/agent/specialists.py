"""Coordinated specialists for ONE Minecraft bot.

Parallel roles may read world state and reason, but ONLY the Agent worker may issue
world-changing tasks. Reflexes retain their emergency override. This avoids two
specialists fighting over the same movement controls.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time


class SpecialistTeam:
    ROLES = ("interpreter", "chatter", "planner", "builder", "scout", "escaper")

    def __init__(self, agent):
        self.agent = agent
        self.requests = queue.Queue(maxsize=64)
        self.conversations = queue.Queue(maxsize=64)
        self.stop_event = threading.Event()
        self.threads = []
        self.running = False
        self._slots = threading.BoundedSemaphore(2)
        self._last_ores = {}
        self._last_alert = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        targets = {
            "interpreter": self._interpret_loop,
            "chatter": self._chatter_loop,
            "planner": self._planner_loop,
            "builder": self._builder_loop,
            "scout": self._scout_loop,
            "escaper": self._escape_loop,
        }
        self.threads = [
            threading.Thread(target=fn, name="specialist-" + name, daemon=True)
            for name, fn in targets.items()
        ]
        for thread in self.threads:
            thread.start()
        self.agent._note("Specialists online: " + ", ".join(self.ROLES))

    def stop(self):
        self.running = False
        self.stop_event.set()

    def receive(self, text, sender="app"):
        """Accept messages immediately without waiting for an active movement task."""
        try:
            self.requests.put_nowait((str(text)[:1200], sender))
        except queue.Full:
            self.agent._note("Interpreter queue full; try again in a moment.")

    def _think(self, role, prompt, max_tokens=180):
        if not self.agent.llm.configured or self.stop_event.is_set():
            return ""
        # A Raspberry Pi or local Ollama model should not receive six simultaneous
        # completions. The main action worker remains independent.
        if not self._slots.acquire(timeout=15):
            return ""
        try:
            model_role = "chat" if role in ("interpreter", "chatter") else "world"
            response = self.agent.llm.chat([
                {"role": "system", "content": "You are the Minecraft bot's " + role +
                 " specialist. Cooperate with other roles and never pretend you executed actions."},
                {"role": "user", "content": prompt},
            ], None, temperature=0.2, max_tokens=max_tokens, model_role=model_role)
            return (response.content or "").strip()
        finally:
            self._slots.release()

    @staticmethod
    def parse_route(text):
        """Accept one small JSON decision; fail closed to worker task on bad JSON."""
        found = re.search(r"\{[^{}]*\}", text, re.S)
        if not found:
            return {"kind": "task", "text": ""}
        try:
            value = json.loads(found.group())
        except (ValueError, TypeError):
            return {"kind": "task", "text": ""}
        if not isinstance(value, dict):
            return {"kind": "task", "text": ""}
        kind = str(value.get("kind", "task")).lower()
        if kind not in ("chat", "task", "command", "goal"):
            kind = "task"
        return {"kind": kind, "text": str(value.get("text", ""))[:1200]}

    def _interpret_loop(self):
        while not self.stop_event.is_set():
            try:
                text, sender = self.requests.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text.strip():
                continue
            try:
                prompt = (
                    "Route this PLAYER message for a Minecraft bot.\n"
                    "Return ONLY JSON: {\\\"kind\\\":\\\"chat|task|command|goal\\\",\\\"text\\\":\\\"...\\\"}.\n"
                    "chat = conversation/question that can be answered while bot is moving.\n"
                    "task = action such as mining, building, collecting, searching ores, water-clutch "
                    "practice, or following a player.\n"
                    "command = an exact supported ! command, such as !scan, !mine or !goto.\n"
                    "goal = long game progression such as get to the nether or beat the game.\n"
                    "Never invent commands. For a task, keep the FULL user's request as text. "
                    "Do not claim success.\n"
                    "PLAYER (" + sender + "): " + text
                )
                route = self.parse_route(self._think("interpreter", prompt, 150))
                kind, instruction = route["kind"], route["text"] or text
                self.agent._note("[interpreter] " + kind + ": " + instruction[:130])
                if kind == "chat":
                    self.conversations.put((text, sender))
                elif kind == "command":
                    if instruction.startswith("!"):
                        self.agent.submit_command(instruction, sender)
                    else:
                        self.agent.submit(instruction, source=sender)
                elif kind == "goal":
                    self.agent.start_goal(instruction, sender)
                else:
                    self.agent.submit(
                        sender + " asks: " + instruction +
                        ". Coordinate with planner/building/scout notes, tell the player what you are doing, "
                        "then use tools. If a special move is not implemented, say so rather than claiming it.",
                        source=sender,
                    )
            except Exception as exc:
                self.agent._note("[interpreter] " + str(exc)[:160])
                self.agent.submit(sender + " asks: " + text, source=sender)

    def _chatter_loop(self):
        while not self.stop_event.is_set():
            try:
                text, sender = self.conversations.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                status = self.agent.status_text(short=True)
                reply = self._think(
                    "chatter", "A player (" + sender + ") says: " + text +
                    "\nLive Minecraft status: " + status +
                    "\nThe bot may be mining or exploring at the same time. "
                    "Reply in one friendly, helpful sentence (under 180 characters). "
                    "Do not promise to perform a task.", 110
                ).strip().strip('"')
                if reply:
                    self.agent.bot.chat(reply[:240])
                    self.agent._note("[chatter] " + reply[:220])
            except Exception as exc:
                self.agent._note("[chatter] " + str(exc)[:160])

    def _planner_loop(self):
        while not self.stop_event.wait(35):
            a = self.agent
            if not a.current_task or not a.llm.configured:
                continue
            try:
                answer = self._think(
                    "planner", "Worker objective: " + a.current_task[:350] +
                    "\nStatus: " + a.status_text(short=True) +
                    "\nRecent teammate notes: " + "; ".join(a.board.recent(5)) +
                    "\nGive exactly one concise, practical next-step suggestion (under 28 words). "
                    "Reply OK if nothing useful.", 85
                )
                if answer and answer.upper().strip(".") != "OK":
                    a.board.post("planner", answer[:180])
            except Exception as exc:
                a._note("[planner] " + str(exc)[:130])

    def _builder_loop(self):
        while not self.stop_event.wait(50):
            a = self.agent
            task = a.current_task or ""
            if not a.llm.configured or not re.search(
                    r"build|bridge|clutch|place|shelter|tower|house|wall|roof|construct|scaffold", task, re.I):
                continue
            try:
                answer = self._think(
                    "builder", "Task: " + task[:350] +
                    "\nInventory: " + a.bot.inventory.summary()[:450] +
                    "\nStatus: " + a.status_text(short=True) +
                    "\nGive one executable Minecraft building/safety suggestion using existing tools. "
                    "Water-bucket clutches require precise timing and are NOT implemented as a reliable "
                    "special move. Do not suggest a blind jump or promise a clutch.", 110
                )
                if answer:
                    a.board.post("builder", answer[:220])
            except Exception as exc:
                a._note("[builder] " + str(exc)[:130])

    def _scout_loop(self):
        """Read nearby ore locations during other work; never move the shared bot."""
        ores = ("diamond_ore", "deepslate_diamond_ore", "iron_ore",
                "deepslate_iron_ore", "gold_ore", "ancient_debris")
        while not self.stop_event.wait(24):
            a, b = self.agent, self.agent.bot
            if not b.spawned.is_set() or b.dead:
                continue
            for ore in ores:
                if self.stop_event.is_set():
                    break
                try:
                    pts = b.find_blocks(ore, 64, 1)
                    if not pts:
                        continue
                    pos = tuple(pts[0])
                    key = (b.dimension, ore, pos)
                    if key in self._last_ores and time.time() - self._last_ores[key] < 180:
                        continue
                    self._last_ores[key] = time.time()
                    a.board.post("scout", ore + " spotted at " + str(pos))
                except Exception as exc:
                    a.log.debug("scout %s: %s", ore, exc)
            if len(self._last_ores) > 200:
                cutoff = time.time() - 200
                self._last_ores = {k: v for k, v in self._last_ores.items() if v > cutoff}

    def _escape_loop(self):
        """Only raise alerts: the existing fast survival reflex controls emergency motion."""
        while not self.stop_event.wait(1.0):
            a, b = self.agent, self.agent.bot
            if not b.spawned.is_set() or b.dead:
                continue
            if b.health <= 6 and time.time() - self._last_alert > 15:
                self._last_alert = time.time()
                a.board.post("escaper", "Critical health: escape/cover first; survival reflex has priority", urgent=True)
