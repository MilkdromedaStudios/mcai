"""Fast PvP / PvE engine. Runs inside the 20 TPS tick loop (reacts every 50 ms, no LLM in the loop).

Per tick: pick/keep a target, aim at it, sprint toward it, hit exactly when the weapon cooldown is
ready, strafe while in range, jump over obstacles, eat or back off when low. Pathfinding is used only
when the target is far away or out of straight reach.
"""
from __future__ import annotations

import math
import threading
import time

from ..protocol.buffer import Writer


class Combat:
    def __init__(self, bot):
        self.bot = bot
        self.target_id: int | None = None
        self.target_name: str | None = None      # player name to keep hunting across respawns
        self.active = False
        self.hits = 0
        self.kills = 0
        self._strafe = 1
        self._strafe_until = 0.0
        self._wtap = 0
        self._path_thread: threading.Thread | None = None
        self._last_seen = None
        self.last_path = None
        self.last_error = ""
        bot.tick_hooks.append(self._tick)

    # ------------------------------------------------------------------ control
    def engage(self, entity=None, player: str | None = None):
        b = self.bot
        b.equip_best_weapon()
        self.target_name = player
        self.target_id = entity.id if entity else None
        self.active = True

    def disengage(self):
        self.active = False
        self.target_id = None
        self.target_name = None
        self.bot.controls.clear()

    def target(self):
        b = self.bot
        e = b.entities.get(self.target_id) if self.target_id is not None else None
        if e is None and self.target_name:
            e = b.entities.nearest(b.position, lambda x: x.name == self.target_name, 512)
            self.target_id = e.id if e else None
        return e

    # ------------------------------------------------------------------ tick (50 ms)
    def _tick(self):
        if not self.active or self.bot.dead:
            return
        b, c = self.bot, self.bot.controls
        e = self.target()
        if e is None:
            if not self.target_name:
                self.active = False
                c.clear()
            else:
                self._chase_far()
            return
        self._last_seen = (e.x, e.y, e.z, time.time())
        dist = b.entity_distance(e)
        horiz = math.hypot(e.x - b.state.x, e.z - b.state.z)
        walking_path = self._path_thread is not None and self._path_thread.is_alive()
        if horiz > 24 or abs(e.y - b.state.y) > 3 or (walking_path and horiz > 6):
            self._chase_far(e)        # the path follower steers; do not fight it for the controls
            return
        # aim (instant - a player flicks; the server only sees the rotation we send this tick)
        ex, ey, ez = b.eye
        tx, ty, tz = e.x, e.y + e.height * 0.85, e.z
        dx, dy, dz = tx - ex, ty - ey, tz - ez
        b.state.yaw = math.degrees(math.atan2(-dx, dz))
        b.state.pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
        # close range fighting
        c.forward = dist > 2.2 or self._wtap <= 0
        c.sprint = c.forward and b.food > 6
        c.jump = b.state.on_ground and b.state.collided_h
        if dist > 4 and b.state.collided_h and b.state.on_ground and abs(b.state.vx) + abs(b.state.vz) < 0.02:
            self._chase_far(e)        # a wall/hill in the way: let the pathfinder handle it
        now = time.time()
        if dist <= 3.0:
            if now > self._strafe_until:
                self._strafe = -self._strafe
                self._strafe_until = now + 0.6 + (hash(int(now * 10)) % 5) / 10
            c.left, c.right = self._strafe > 0, self._strafe < 0
            ready = now - b._last_attack >= b._attack_cooldown() * 0.95
            if ready:
                b.send("attack", Writer().varint(e.id))
                b.send("punch")
                b._last_attack = now
                self.hits += 1
                self._wtap = 2                       # release forward 2 ticks after a hit (w-tap knockback)
        else:
            c.left = c.right = False
        self._wtap -= 1

    def _chase_far(self, e=None):
        """Target out of direct reach: path toward it on a helper thread (non-blocking)."""
        b = self.bot
        if self._path_thread and self._path_thread.is_alive():
            return
        dest = (e.x, e.y, e.z) if e else (self._last_seen[:3] if self._last_seen else None)
        if dest is None:
            return
        from .pathfinder import GoalNear

        def run():
            try:
                b.pathfinder.can_dig = True
                res = b.pathfinder.find(b._start_node(), GoalNear(*dest, 2.0), max_nodes=6000, timeout=0.4)
                self.last_path = (res.status, len(res.steps))
                if res.steps:
                    b._follow(res.steps[:6], time.time() + 4)
            except Exception as ex:  # noqa: BLE001
                self.last_error = repr(ex)
        self._path_thread = threading.Thread(target=run, daemon=True)
        self._path_thread.start()

    def fight_until(self, entity=None, player=None, timeout: float = 60) -> str:
        """Blocking helper: fight until the target dies/disappears, we die, or timeout."""
        b = self.bot
        self.engage(entity, player)
        start_hits, end = self.hits, time.time() + timeout
        tid = entity.id if entity else None
        try:
            while time.time() < end and self.active and not b.dead:
                if tid is not None and tid not in b.entities.by_id:
                    self.kills += 1
                    return f"target gone after {self.hits - start_hits} hits"
                time.sleep(0.1)
            return "we died" if b.dead else f"stopped after {self.hits - start_hits} hits"
        finally:
            self.disengage()
