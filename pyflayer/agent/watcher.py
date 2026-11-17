"""Second AI ("watcher") that monitors the live world while the worker AI performs tasks.

They collaborate through a shared blackboard:
  * every 3 s the watcher looks at the world (threats, health, hunger, inventory, nearby resources,
    structures, players, stalled progress) and posts short notes;
  * urgent notes (danger) interrupt the worker immediately through the reflex system;
  * every ~20 s, if an AI endpoint is set, the watcher asks the model for one line of advice about
    the worker's current task, given what it sees;
  * the worker reads all new notes before each of its decisions (they are added to its prompt).
"""
from __future__ import annotations

import collections
import json
import threading
import time


class Blackboard:
    def __init__(self, size: int = 50):
        self.notes: collections.deque = collections.deque(maxlen=size)
        self.lock = threading.Lock()
        self._seen_by_worker = 0
        self._counter = 0
        self.on_post = None

    def post(self, source: str, text: str, urgent: bool = False):
        with self.lock:
            # don't repeat the same note within 30 s
            for t, s, txt, _ in list(self.notes)[-10:]:
                if txt == text and time.time() - t < 30:
                    return
            self._counter += 1
            self.notes.append((time.time(), source, text, urgent))
        if self.on_post:
            self.on_post(f"[{source}{' URGENT' if urgent else ''}] {text}")

    def unread(self) -> list[str]:
        """Notes the worker has not seen yet (marks them read)."""
        with self.lock:
            n = self._counter - self._seen_by_worker
            new = list(self.notes)[-n:] if n > 0 else []
            self._seen_by_worker = self._counter
        return [f"[{s}{' URGENT' if u else ''}] {txt}" for _, s, txt, u in new]

    def recent(self, n: int = 12) -> list[str]:
        with self.lock:
            return [time.strftime("%H:%M:%S ", time.localtime(t)) + f"{s}: {txt}" for t, s, txt, _ in list(self.notes)[-n:]]


class Watcher:
    def __init__(self, agent):
        self.agent = agent
        self.bot = agent.bot
        self.board = agent.board
        self.running = False
        self._last_llm = 0.0
        self._last = {}
        self._thread: threading.Thread | None = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="watcher", daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running and self.agent.running:
            time.sleep(3)
            b = self.bot
            if not b.spawned.is_set() or b.dead:
                continue
            try:
                self.observe()
                if self.agent.llm.configured and self.agent.current_task and time.time() - self._last_llm > 20:
                    self._last_llm = time.time()
                    self.advise()
            except Exception as e:  # noqa: BLE001
                self.agent.log.debug("watcher error: %s", e)

    # ------------------------------------------------------------------ rule-based monitoring (fast)
    def observe(self):
        b, post = self.bot, self.board.post
        pos = b.position
        hostile = b.entities.within(pos, 16, lambda e: e.hostile)
        if hostile:
            kinds = collections.Counter(e.type for e in hostile)
            near = min(hostile, key=lambda e: e.distance_to(*pos))
            post("watcher", f"{sum(kinds.values())} hostile mob(s) within 16 blocks ({dict(kinds)}); nearest "
                            f"{near.type} at {near.distance_to(*pos):.0f}m", urgent=near.distance_to(*pos) < 5)
        if b.health <= 8:
            post("watcher", f"health low ({b.health:.0f}/20): avoid fights, eat, take cover", urgent=b.health <= 5)
        if b.food <= 8:
            post("watcher", f"hungry ({b.food}/20): eat or get food soon")
        if b.inventory.empty_slots() <= 2:
            post("watcher", "inventory almost full: drop junk or store items")
        if b.time_phase() == "sunset" and b.dimension == "overworld":
            post("watcher", "night is coming: finish up, get a bed or prepare to hide")
        players = b.entities.within(pos, 48, lambda e: e.type == "player" and e.name != b.username)
        names = sorted(e.name or "?" for e in players)
        if names and names != self._last.get("players"):
            post("watcher", f"players nearby: {', '.join(names)}")
        self._last["players"] = names
        # progress: has the worker been in the same spot for long while busy?
        moved_key = (round(pos[0]), round(pos[2]))
        if b.current_action != "idle":
            if self._last.get("spot") == moved_key:
                self._last["still"] = self._last.get("still", 0) + 1
                if self._last["still"] == 10:
                    post("watcher", f"no movement for 30 s while doing '{b.current_action}': maybe stuck or "
                                    f"the plan is wrong")
            else:
                self._last["still"] = 0
        self._last["spot"] = moved_key
        # opportunities from the most recent scan (refreshed by the reflex loop every ~90 s)
        scan = self.agent.scanner.last
        if scan and time.time() - self.agent.scanner.last_time < 120:
            for ore in ("diamond_ore", "deepslate_diamond_ore", "ancient_debris", "emerald_ore"):
                if ore in scan["ores"]:
                    o = scan["ores"][ore]
                    post("watcher", f"{ore} spotted at {o['nearest']} ({o['distance']}m)")
            for s in scan["structures"][:3]:
                post("watcher", f"structure nearby: {s['structure']} at {s['pos']} ({s['distance']}m)")

    # ------------------------------------------------------------------ AI advice (slow, optional)
    def advise(self):
        a = self.agent
        prompt = (f"You are the WATCHER teammate of a Minecraft bot. The worker is doing: {a.current_task}\n"
                  f"Live status:\n{a.status_text()}\nRecent notes:\n" + "\n".join(self.board.recent(8)) +
                  "\nGive ONE short, concrete tip (max 25 words) that helps the worker right now, or reply OK if "
                  "nothing needs saying.")
        r = a.llm.chat([{"role": "user", "content": prompt}], None, temperature=0.2, max_tokens=60)
        tip = (r.content or "").strip().strip('"')
        if tip and tip.upper() not in ("OK", "OK."):
            self.board.post("watcher-ai", tip[:200])
