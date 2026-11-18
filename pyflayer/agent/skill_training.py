"""Measured, progressive Minecraft skill practice.

These are real world-state/inventory checks, not AI self-assessment. The coach
runs in the main actor lane. Dangerous clutch maneuvers are observed from
normal gameplay, NEVER triggered by deliberately jumping off a cliff.
"""
from __future__ import annotations

import math
import time

from ..bot.actions import ActionError
from ..bot.pathfinder import GoalXZ

DRILLS = ("resource_choice", "safe_navigation", "sprint_jump", "controlled_placement")
# No high-stakes actions on the default setting. The bot needs verified
# land-based practice before any optional lava-gap jump is even considered.


class SkillCoach:
    def __init__(self, agent):
        self.agent = agent
        self.bot = agent.bot
        self.brain = agent.brain
        agent.bot.events.on("clutch_result", self._clutch_result)

    def _clutch_result(self, success: bool, detail: str = ""):
        result = self.brain.record_drill("water_clutch", bool(success), str(detail))
        self.agent._note(
            f"[skills] water clutch {'verified' if success else 'failed'} "
            f"({result['successes']}/{result['attempts']})")

    def mastery(self) -> dict:
        return {n: self.brain.drill(n) for n in (*DRILLS, "water_clutch")}

    def lava_jump_ready(self) -> bool:
        """Only an explicit opt-in AND successful dry drills unlock a risky jump."""
        return (
            bool(self.agent.cfg["training"].get("allow_advanced_moves", False))
            and self.brain.drill("sprint_jump")["certified"]
            and self.brain.drill("controlled_placement")["certified"]
            and self.bot.health >= 16
            and self.bot.food >= 12
            and not self.bot.dead
        )

    def _dry_pad(self, x: int, y: int, z: int) -> bool:
        """Require known solid ground, headroom and no nearby lava."""
        pf = self.bot.pathfinder
        if not pf.can_stand(x, y, z) or pf.water(x, y, z):
            return False
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for dy in (-1, 0):
                    pos = self.bot.world.get(x + dx, y + dy, z + dz)
                    if pos is None or self.bot.gd.is_lava[pos]:
                        return False
        return True

    def _path_pads(self, dx: int, dz: int, length: int = 6) -> bool:
        x, y, z = self.bot.block_pos
        return all(self._dry_pad(x + dx * i, y, z + dz * i) for i in range(length + 1))

    def _resource_choice(self) -> tuple[bool, str] | None:
        kinds = self.bot.planner.nearby_wood(radius=48)
        if not kinds:
            return None
        if not self.bot.inventory.empty_slots():
            return None
        wanted = kinds[0][0]
        names = [name for name, dist in kinds]
        before = sum(self.bot.inventory.count(n) for n in names)
        self.bot.planner.gather_wood(count=2, radius=48)
        after = sum(self.bot.inventory.count(n) for n in names)
        return after > before, f"chose loaded {wanted} at {kinds[0][1]:.1f} blocks; gained {after - before}"

    def _safe_navigation(self) -> tuple[bool, str] | None:
        x, y, z = self.bot.block_pos
        for dx, dz in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            if not self._path_pads(dx, dz, 5):
                continue
            target = (x + dx * 4, z + dz * 4)
            start = self.bot.position
            self.bot.goto(GoalXZ(*target, 1.5), timeout=25, dig=False, place=False, max_replans=5)
            pos = self.bot.position
            ok = math.hypot(pos[0] - target[0], pos[2] - target[1]) <= 2.0
            return ok, f"walk {start} -> {pos}, target {target}"
        return None

    def _sprint_jump(self) -> tuple[bool, str] | None:
        """Jump while moving across fully solid DRY ground. No cliff training."""
        b = self.bot
        if not b.state.on_ground or b.state.in_water:
            return None
        start = b.position
        start_health = b.health
        for dx, dz in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            if not self._path_pads(dx, dz, 8):
                continue
            b.controls.clear()
            b.state.yaw = math.degrees(math.atan2(-dx, dz))
            max_y = start[1]
            try:
                b.controls.sprint = True
                b.controls.forward = True
                b.controls.jump = True
                for _ in range(4):
                    b._wait(1)
                    max_y = max(max_y, b.state.y)
                b.controls.jump = False
                for _ in range(10):
                    b._wait(1)
                    max_y = max(max_y, b.state.y)
            finally:
                b.controls.clear()
            forward = (b.position[0] - start[0]) * dx + (b.position[2] - start[2]) * dz
            ok = (max_y > start[1] + 0.3 and forward > 0.7
                  and b.health >= start_health and b.state.on_ground)
            return ok, f"dry jump advanced {forward:.2f} blocks; apex {max_y-start[1]:.2f}"
        return None

    def _controlled_placement(self) -> tuple[bool, str] | None:
        b = self.bot
        item = b._scaffold_item()
        if not item or not b.inventory.count(item):
            return None
        x, y, z = b.block_pos
        for dx, dz in ((2, 0), (0, 2), (-2, 0), (0, -2)):
            if not self._path_pads(1 if dx > 0 else -1 if dx < 0 else 0,
                                   1 if dz > 0 else -1 if dz < 0 else 0, 3):
                continue
            target = (x + dx, y, z + dz)
            state = b.world.get(*target)
            if state is None or not b.gd.is_air[state]:
                continue
            health = b.health
            b.place_at(target, item)
            for _ in range(12):
                b._wait(1)
                actual = b.world.get(*target)
                if actual is not None and b.gd.has_collision[actual]:
                    break
            else:
                return False, f"placement did not appear at {target}"
            ok = b.health >= health
            # Do not turn the surrounding area into an obstacle course.
            # Reclaim the practice block after the placement was verified.
            try:
                b.dig(target)
                b.collect_items(radius=4, timeout=5)
            except ActionError:
                pass
            return ok, f"verified place + reclaim {item} at {target}"
        return None

    def practice_one(self, name: str | None = None) -> str:
        """Choose the least-practiced available dry drill and measure the result."""
        if self.bot.dead or not self.bot.spawned.is_set() or self.bot.health < 12:
            return "Skill practice skipped: not safe/connected"
        if name is not None and name not in DRILLS:
            return f"Unknown or locked skill: {name}. Safe drills: {', '.join(DRILLS)}"
        sequence = [name] if name else sorted(
            DRILLS, key=lambda n: (self.brain.drill(n)["attempts"],
                                   self.brain.drill(n)["successes"]))
        for skill in sequence:
            try:
                result = getattr(self, "_" + skill)()
            except (ActionError, ValueError, RuntimeError) as exc:
                result = (False, str(exc))
            if result is None:
                continue
            ok, detail = result
            stats = self.brain.record_drill(skill, bool(ok), detail)
            message = (f"{skill}: {'PASS' if ok else 'FAIL'} — {detail}. "
                       f"Mastery {stats['successes']}/{stats['attempts']}"
                       + (" (certified)" if stats["certified"] else ""))
            self.agent._note("[skills] " + message)
            return message
        return "Skill practice skipped: no safe available drill; need flat dry terrain or blocks"
