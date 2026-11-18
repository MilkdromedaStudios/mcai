"""Priority finds: remember where valuable ores were seen, and go back for them as soon as the bot can mine them.

While the bot walks, digs or explores, a light background scan notices diamonds (and ancient debris) in the
loaded world and saves them as waypoints in the bot's brain - they survive restarts. Before every training /
goal step the trainer asks `run_pending()`: if a saved find is ready (we have the right pickaxe) and still
wanted, the bot goes there and mines it first, instead of walking past it to do the next step.
"""
from __future__ import annotations

import logging
import math
import threading
import time

from ..bot.actions import ActionError
from ..bot.pathfinder import GoalNear

log = logging.getLogger("pyflayer.priority")

# kind -> (ore blocks, pickaxes that can mine it, inventory item it gives, how many we want, dimension)
FINDS = {
    "diamonds": (("diamond_ore", "deepslate_diamond_ore"),
                 ("iron_pickaxe", "diamond_pickaxe", "netherite_pickaxe"), "diamond", 24, "overworld"),
    "ancient_debris": (("ancient_debris",), ("diamond_pickaxe", "netherite_pickaxe"), "ancient_debris", 8,
                       "the_nether"),
}
CLUSTER = 6          # finds closer than this are one vein
MAX_TRIES = 3


class Finds:
    def __init__(self, agent, scan_every: float = 4.0, radius: float = 64):
        self.agent = agent
        self.bot = agent.bot
        self.brain = agent.brain
        self.scan_every = scan_every
        self.radius = radius
        self.running = False
        self._thread = None

    # ------------------------------------------------------------------ spotting
    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="finds-scan", daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            time.sleep(self.scan_every)
            try:
                if self.bot.spawned.is_set() and not self.bot.dead:
                    self.scan()
            except Exception as e:  # noqa: BLE001 - never kill the scanner
                log.debug("finds scan: %s", e)

    def saved(self, kind: str | None = None, dim: str | None = None) -> list[dict]:
        out = []
        for k in ([kind] if kind else FINDS):
            out += self.brain.waypoints(dim=dim, kind="find:" + k, near=self.bot.position, k=200)
        return out

    def scan(self) -> list[tuple]:
        """Look for valuable ores in the loaded world and save new ones. Returns the new finds."""
        b = self.bot
        new = []
        for kind, (blocks, _picks, _item, _want, dim) in FINDS.items():
            if b.dimension != dim:
                continue
            pts = b.find_blocks(list(blocks), self.radius, 24)
            if not pts:
                continue
            known = [w["pos"] for w in self.saved(kind, dim)]
            for p in pts:
                p = tuple(int(v) for v in p)
                if any(math.dist(p, k) < CLUSTER for k in known):
                    continue
                vein = sum(1 for q in pts if math.dist(p, q) < CLUSTER)
                name = f"{kind}@{p[0]},{p[1]},{p[2]}"
                self.brain.set_waypoint(name, dim, p, "find:" + kind, f"{vein} block(s) seen; tries=0")
                self.brain.remember(f"{kind.replace('_', ' ')} seen at {p[0]} {p[1]} {p[2]} ({dim})", "finds", 0.9,
                                    "scan")
                known.append(p)
                new.append((kind, p, vein))
                ready = self.ready(kind)
                self.agent._note(f"[finds] saved {kind.replace('_', ' ')} at {p[0]} {p[1]} {p[2]}"
                                 + ("" if ready else f" - will mine it once I have a {FINDS[kind][1][0].replace('_', ' ')}"))
                try:
                    b.chat(f"Found {kind.replace('_', ' ')} at {p[0]} {p[1]} {p[2]}! "
                           + ("Mining it next." if ready else "Saved for later."))
                except Exception:  # noqa: BLE001
                    pass
        return new

    # ------------------------------------------------------------------ going back for them
    def ready(self, kind: str) -> bool:
        _blocks, picks, _item, _want, _dim = FINDS[kind]
        return any(self.bot.inventory.count(p) for p in picks)

    def wanted(self, kind: str) -> bool:
        _blocks, _picks, item, want, _dim = FINDS[kind]
        return self.bot.inventory.count(item) < want

    def pending(self) -> list[dict]:
        """Saved finds we can mine right now, nearest first."""
        b = self.bot
        out = []
        for kind in FINDS:
            if FINDS[kind][4] != b.dimension or not self.ready(kind) or not self.wanted(kind):
                continue
            out += self.saved(kind, b.dimension)
        out.sort(key=lambda w: math.dist(w["pos"], b.position))
        return out

    def _tries(self, w) -> int:
        try:
            return int(w["note"].rsplit("tries=", 1)[1])
        except (IndexError, ValueError):
            return 0

    def run_pending(self, max_finds: int = 3) -> bool:
        """Mine saved finds that are ready. Returns True if it did anything (so the caller re-checks goals)."""
        did = False
        for w in self.pending()[:max_finds]:
            kind = w["kind"].split(":", 1)[1]
            blocks, _picks, item, _want, _dim = FINDS[kind]
            x, y, z = w["pos"]
            before = self.bot.inventory.count(item)
            self.agent._note(f"[finds] priority: going back for {kind.replace('_', ' ')} at {x} {y} {z}")
            self.bot.chat(f"Going to mine the {kind.replace('_', ' ')} I saved at {x} {y} {z}.")
            try:
                if math.dist(self.bot.position, (x, y, z)) > 5:
                    self.bot.goto(GoalNear(x, y, z, 3), timeout=300)
                left = self.bot.find_blocks(list(blocks), 10, 16)
                if left:
                    self.bot.collect_block(list(blocks), len(left), radius=12, timeout=240)
            except ActionError as e:
                if any(k in str(e) for k in ("stopped", "died", "disconnected")):
                    raise
                self.agent._note(f"[finds] couldn't reach {kind} at {x} {y} {z}: {e}")
            did = True
            got = self.bot.inventory.count(item) - before
            near = (self.bot.find_blocks(list(blocks), 8, 1)
                    if math.dist(self.bot.position, (x, y, z)) < 12 else ["?"])
            tries = self._tries(w) + 1
            if not near or tries >= MAX_TRIES:
                self.brain.delete_waypoint(w["name"])
                msg = (f"mined the {kind.replace('_', ' ')} at {x} {y} {z} (+{got} {item})" if got or not near
                       else f"gave up on {kind} at {x} {y} {z} after {tries} tries")
            else:
                self.brain.set_waypoint(w["name"], w["dim"], w["pos"], w["kind"],
                                        w["note"].rsplit("tries=", 1)[0] + f"tries={tries}")
                msg = f"{kind} at {x} {y} {z} still there; will retry"
            self.agent._note("[finds] " + msg)
            if got:
                self.brain.remember(msg, "finds", 0.8, "mined")
        return did

    def summary(self) -> str:
        rows = self.saved()
        if not rows:
            return "no saved finds yet"
        return "; ".join(f"{w['kind'].split(':')[1]} at {w['pos'][0]} {w['pos'][1]} {w['pos'][2]} ({w['dim']})"
                         for w in rows[:8])
