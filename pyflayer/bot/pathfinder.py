"""A* pathfinding over the block grid (mineflayer-pathfinder style).

Moves: walk (4 + diagonal), jump up 1, drop down (up to 3, deeper into water), swim, plus optional
digging through obstacles, digging down, pillaring up and bridging over gaps with scaffold blocks.

Performance: per-state flag lookups are numpy arrays; the node budget and a wall-clock budget keep
searches bounded on a Raspberry Pi. When the goal is out of reach (unloaded or too far) the best
partial path is returned and the executor re-plans as it walks.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


# ---------------------------------------------------------------------------- goals
class Goal:
    def is_end(self, x, y, z) -> bool:
        raise NotImplementedError

    def heuristic(self, x, y, z) -> float:
        raise NotImplementedError


class GoalNear(Goal):
    def __init__(self, x, y, z, r=1.0):
        self.x, self.y, self.z, self.r = int(math.floor(x)), int(math.floor(y)), int(math.floor(z)), r
        self.r2 = r * r

    def is_end(self, x, y, z):
        return (x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2 <= self.r2

    def heuristic(self, x, y, z):
        dx, dy, dz = abs(x - self.x), abs(y - self.y), abs(z - self.z)
        return math.sqrt(dx * dx + dz * dz) + dy * 0.9


class GoalBlock(GoalNear):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, 0.0)


class GoalXZ(Goal):
    def __init__(self, x, z, r=1.5):
        self.x, self.z, self.r2 = int(math.floor(x)), int(math.floor(z)), r * r

    def is_end(self, x, y, z):
        return (x - self.x) ** 2 + (z - self.z) ** 2 <= self.r2

    def heuristic(self, x, y, z):
        return math.hypot(x - self.x, z - self.z)


class GoalLookAt(Goal):
    """Stand somewhere from which block (bx,by,bz) is within reach (for digging/placing/using)."""

    def __init__(self, bx, by, bz, reach=4.2):
        self.b = (bx, by, bz)
        self.reach2 = reach * reach

    def is_end(self, x, y, z):
        bx, by, bz = self.b
        ex, ey, ez = x + 0.5, y + 1.62, z + 0.5
        d2 = (bx + 0.5 - ex) ** 2 + (by + 0.5 - ey) ** 2 + (bz + 0.5 - ez) ** 2
        return d2 <= self.reach2 and not (x == bx and z == bz and by in (y, y + 1))

    def heuristic(self, x, y, z):
        bx, by, bz = self.b
        return max(0.0, math.sqrt((x - bx) ** 2 + (y + 1 - by) ** 2 + (z - bz) ** 2) - 3.5)


class GoalInvert(Goal):
    """Get at least `dist` away from a point (fleeing)."""

    def __init__(self, x, y, z, dist):
        self.x, self.y, self.z, self.d = x, y, z, dist

    def is_end(self, x, y, z):
        return math.hypot(x - self.x, z - self.z) >= self.d

    def heuristic(self, x, y, z):
        return max(0.0, self.d - math.hypot(x - self.x, z - self.z))


# ---------------------------------------------------------------------------- path data
@dataclass
class Step:
    x: int
    y: int
    z: int
    kind: str = "walk"                     # walk | jump | drop | swim | pillar | bridge | dig_down
    dig: list = field(default_factory=list)  # blocks to break before moving
    place: tuple | None = None            # block to place before moving (bridge) / under us (pillar)

    @property
    def pos(self):
        return (self.x, self.y, self.z)


@dataclass
class PathResult:
    steps: list
    status: str            # success | partial | noPath | timeout
    visited: int
    time_ms: float
    cost: float


class Pathfinder:
    def __init__(self, bot):
        self.bot = bot
        self.gd = bot.gd
        self.world = bot.world
        self.can_dig = True
        self.can_place = True
        self.max_drop = 3
        self.dig_cost_factor = 1.0
        self.avoid: set[tuple[int, int, int]] = set()
        self._tool_cache: dict = {}   # blocks never to break (e.g. containers)
        self.scaffold_items = ("cobblestone", "dirt", "cobbled_deepslate", "netherrack", "stone", "andesite",
                               "diorite", "granite", "tuff", "blackstone", "sandstone", "oak_planks",
                               "spruce_planks", "birch_planks", "end_stone", "deepslate", "basalt")

    # ------------------------------------------------------------------ block predicates
    def _s(self, x, y, z):
        return self.world.get(x, y, z)

    def passable(self, x, y, z) -> bool:
        s = self.world.get(x, y, z)
        return s is not None and bool(self.gd.passable[s])

    def standable(self, x, y, z) -> bool:
        s = self.world.get(x, y, z)
        return s is not None and bool(self.gd.standable[s])

    def water(self, x, y, z) -> bool:
        s = self.world.get(x, y, z)
        return s is not None and bool(self.gd.is_water[s])

    def lava(self, x, y, z) -> bool:
        s = self.world.get(x, y, z)
        return s is not None and bool(self.gd.is_lava[s])

    def climbable(self, x, y, z) -> bool:
        s = self.world.get(x, y, z)
        return s is not None and bool(self.gd.is_climbable[s])

    def can_stand(self, x, y, z) -> bool:
        if not (self.passable(x, y, z) and self.passable(x, y + 1, z)):
            return False
        return self.standable(x, y - 1, z) or self.water(x, y, z) or self.climbable(x, y, z)

    def dig_cost(self, x, y, z) -> float | None:
        """Cost (in 'walk units') to break a block, or None if we must not / cannot break it."""
        if not self.can_dig:
            return None
        s = self.world.get(x, y, z)
        if s is None:
            return None
        if self.gd.passable[s]:
            return 0.0
        if self.gd.is_liquid[s] or (x, y, z) in self.avoid:
            return None
        h = float(self.gd.hardness_arr[s])
        if h < 0 or h > 30:
            return None
        # never open a hole into liquid
        for dx, dy, dz in ((0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)):
            n = self.world.get(x + dx, y + dy, z + dz)
            if n is not None and self.gd.is_liquid[n]:
                return None
        name = self.gd.block_names[self.gd.state_block[s]]
        if name in ("chest", "trapped_chest", "barrel", "furnace", "crafting_table", "spawner",
                    "ender_chest", "bed") or name.endswith("_bed") or name.endswith("shulker_box"):
            return None
        tool_cache = self._tool_cache
        key = name
        if key not in tool_cache:
            items = [it.name for it in self.bot.inventory.items()]
            tool, _, _ = self.gd.best_tool_for(name, items)
            ticks = self.gd.dig_ticks(name, tool, learned=self.bot.learned_hardness)
            tool_cache[key] = ticks
        ticks = tool_cache[key]
        if ticks == math.inf:
            return None
        cost = 1.0 + ticks / 20.0 * 4.0 * self.dig_cost_factor
        if self.gd.is_gravity[s]:
            cost += 1.0
        return cost

    def _clear_cost(self, positions) -> tuple[float, list] | None:
        total = 0.0
        digs = []
        for p in positions:
            if self.passable(*p):
                continue
            c = self.dig_cost(*p)
            if c is None:
                return None
            total += c
            digs.append(p)
        return total, digs

    # ------------------------------------------------------------------ search
    def find(self, start, goal: Goal, max_nodes=30000, timeout=2.5, scaffold=None) -> PathResult:
        t0 = time.time()
        self._tool_cache: dict = {}
        # Read the proficiency gate ONCE per path search; not once per node,
        # because model/brain reads can otherwise overwhelm A* on a Pi.
        gate = getattr(self.bot, "allow_lava_jumps", None)
        self._gap_jumps_enabled = bool(callable(gate) and gate())
        if scaffold is None:
            scaffold = self.bot.inventory.count(set(self.scaffold_items)) if self.can_place else 0
        sx, sy, sz = start
        start_n = (sx, sy, sz)
        open_heap = [(goal.heuristic(*start_n), 0.0, start_n)]
        came: dict[tuple, tuple] = {}
        g = {start_n: 0.0}
        step_of: dict[tuple, Step] = {}
        used_scaffold = {start_n: 0}
        best = start_n
        best_h = goal.heuristic(*start_n)
        visited = 0
        status = "noPath"
        end = None
        while open_heap:
            f, cost, node = heapq.heappop(open_heap)
            if cost > g.get(node, math.inf) + 1e-9:
                continue
            visited += 1
            if goal.is_end(*node):
                end = node
                status = "success"
                break
            h = f - cost
            if h < best_h:
                best_h, best = h, node
            if visited >= max_nodes or (visited & 255 == 0 and time.time() - t0 > timeout):
                status = "timeout"
                break
            sc_left = scaffold - used_scaffold.get(node, 0)
            for nb, c, st in self._neighbors(node, sc_left):
                ng = cost + c
                if ng < g.get(nb, math.inf) - 1e-9:
                    g[nb] = ng
                    came[nb] = node
                    step_of[nb] = st
                    used_scaffold[nb] = used_scaffold.get(node, 0) + (1 if st.place else 0)
                    heapq.heappush(open_heap, (ng + goal.heuristic(*nb) * 1.05, ng, nb))
        if end is None:
            end = best
            if status != "timeout":
                status = "partial" if end != start_n else "noPath"
            elif end == start_n:
                status = "noPath"
            else:
                status = "partial"
        steps = []
        n = end
        while n in came:
            steps.append(step_of[n])
            n = came[n]
        steps.reverse()
        return PathResult(steps, status, visited, (time.time() - t0) * 1000, g.get(end, 0.0))

    def _neighbors(self, node, scaffold_left):
        x, y, z = node
        out = []
        in_water = self.water(x, y, z)
        # vertical moves in water / on ladders
        if in_water or self.climbable(x, y, z):
            # rise only through water / along a ladder; leaving the water is done by the jump move
            if self.passable(x, y + 1, z) and self.passable(x, y + 2, z) and (
                    self.water(x, y + 1, z) or self.climbable(x, y + 1, z)):
                out.append(((x, y + 1, z), 1.5, Step(x, y + 1, z, "swim")))
            if self.passable(x, y - 1, z) and not self.lava(x, y - 1, z) and (
                    self.water(x, y - 1, z) or self.climbable(x, y - 1, z) or self.can_stand(x, y - 1, z)):
                out.append(((x, y - 1, z), 1.5, Step(x, y - 1, z, "swim")))
        for dx, dz in CARDINALS:
            nx, nz = x + dx, z + dz
            # Experimental one-block lava gap jump. Locked by default: allowed
            # only after VERIFIED dry sprint jumps/placements and explicit opt-in.
            # Landing must already be solid; no blind jumps into unknown terrain.
            if (getattr(self, "_gap_jumps_enabled", False) and not in_water
                    and self.standable(x, y - 1, z)
                    and self.lava(nx, y - 1, nz)
                    and self.passable(nx, y, nz)
                    and self.passable(nx, y + 1, nz)
                    and self.can_stand(nx + dx, y, nz + dz)
                    and self.passable(nx + dx, y + 1, nz + dz)
                    and not self.water(nx + dx, y, nz + dz)):
                out.append(((nx + dx, y, nz + dz), 2.7,
                            Step(nx + dx, y, nz + dz, "gap_jump")))
            # 1) walk
            if self.can_stand(nx, y, nz):
                cost = 1.0 + (1.5 if self.water(nx, y, nz) else 0.0)
                out.append(((nx, y, nz), cost, Step(nx, y, nz, "walk")))
                continue
            # 2) jump up one
            if self.passable(x, y + 2, z) and self.can_stand(nx, y + 1, nz) and not in_water:
                out.append(((nx, y + 1, nz), 2.0, Step(nx, y + 1, nz, "jump")))
            elif self.passable(x, y + 2, z) and self.water(x, y, z) and self.can_stand(nx, y + 1, nz):
                out.append(((nx, y + 1, nz), 2.5, Step(nx, y + 1, nz, "jump")))
            # 3) drop down
            if self.passable(nx, y, nz) and self.passable(nx, y + 1, nz) and not self.standable(nx, y - 1, nz):
                for k in range(1, 12):
                    ty = y - k
                    if not self.passable(nx, ty, nz):
                        break
                    if self.lava(nx, ty, nz):
                        break
                    if self.water(nx, ty, nz):
                        out.append(((nx, ty, nz), 1.0 + k * 0.5, Step(nx, ty, nz, "drop")))
                        break
                    if self.standable(nx, ty - 1, nz):
                        if k <= self.max_drop:
                            out.append(((nx, ty, nz), 1.0 + k * 0.6 + (2.0 if k == 3 else 0.0),
                                        Step(nx, ty, nz, "drop")))
                        break
                # 7) bridge over a gap - or over lava: a block placed on lava becomes the floor
                lava_below = self.lava(nx, y - 1, nz)
                # the current floor may itself be a block we plan to place (chained bridging over lava/gaps)
                floor_ok = self.standable(x, y - 1, z) or self.lava(x, y - 1, z) or \
                    (self.passable(x, y - 1, z) and not self.water(x, y, z))
                if scaffold_left > 0 and self.can_place and floor_ok and \
                        (self.passable(nx, y - 1, nz) or lava_below) and not self.lava(nx, y, nz):
                    out.append(((nx, y, nz), 3.0 if lava_below else 3.5,
                                Step(nx, y, nz, "bridge", place=(nx, y - 1, nz))))
                continue
            # 5) dig through (feet+head) and walk
            if self.can_dig and self.standable(nx, y - 1, nz):
                r = self._clear_cost([(nx, y + 1, nz), (nx, y, nz)])
                if r is not None and r[1]:
                    out.append(((nx, y, nz), 1.0 + r[0], Step(nx, y, nz, "walk", dig=r[1])))
            # dig up-and-over: clear head room above us and space at target+1
            if self.can_dig and self.standable(nx, y, nz) and not in_water:
                r = self._clear_cost([(x, y + 2, z), (nx, y + 2, nz), (nx, y + 1, nz)])
                if r is not None and r[1]:
                    out.append(((nx, y + 1, nz), 2.0 + r[0], Step(nx, y + 1, nz, "jump", dig=r[1])))
        # diagonals (no digging, need both side cells free)
        for dx, dz in DIAGONALS:
            nx, nz = x + dx, z + dz
            if not self.can_stand(nx, y, nz):
                continue
            if all(self.passable(a, b, c) for a, b, c in
                   ((x + dx, y, z), (x + dx, y + 1, z), (x, y, z + dz), (x, y + 1, z + dz))):
                out.append(((nx, y, nz), 1.4142 + (1.5 if self.water(nx, y, nz) else 0.0), Step(nx, y, nz, "walk")))
        # 4) dig straight down
        if self.can_dig and not in_water:
            c = self.dig_cost(x, y - 1, z)
            if c is not None and c > 0 and self.standable(x, y - 2, z) and not self.lava(x, y - 2, z):
                out.append(((x, y - 1, z), c + 0.5, Step(x, y - 1, z, "dig_down", dig=[(x, y - 1, z)])))
        # 6) pillar up
        if scaffold_left > 0 and self.can_place and not in_water and self.standable(x, y - 1, z):
            r = self._clear_cost([(x, y + 2, z)]) if self.can_dig else (
                (0.0, []) if self.passable(x, y + 2, z) else None)
            if r is not None:
                out.append(((x, y + 1, z), 3.0 + r[0], Step(x, y + 1, z, "pillar", dig=r[1], place=(x, y, z))))
        return out
