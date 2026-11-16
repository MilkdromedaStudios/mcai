"""Player physics (gravity, friction, collisions, step-up, water, ladders) close to vanilla.

The server re-simulates every move, so these numbers follow vanilla to avoid "moved wrongly"
corrections. Unknown/partial block shapes are approximated *conservatively* (as bigger), which can
only make the bot stop early, never walk through something the server considers solid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

GRAVITY = 0.08
AIR_DRAG = 0.98
JUMP_VEL = 0.42
STEP_HEIGHT = 0.6
PLAYER_W = 0.6
PLAYER_H = 1.8
SNEAK_H = 1.5
WALK_SPEED = 0.1
SPRINT_MULT = 1.3
SNEAK_MULT = 0.3
SLIPPERINESS = {"ice": 0.98, "packed_ice": 0.98, "frosted_ice": 0.98, "blue_ice": 0.989, "slime_block": 0.8}
EPS = 1e-7


@dataclass
class Controls:
    forward: bool = False
    back: bool = False
    left: bool = False
    right: bool = False
    jump: bool = False
    sprint: bool = False
    sneak: bool = False

    def clear(self):
        self.forward = self.back = self.left = self.right = self.jump = self.sprint = self.sneak = False


@dataclass
class PlayerState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    on_ground: bool = False
    collided_h: bool = False
    in_water: bool = False
    in_lava: bool = False
    on_climbable: bool = False
    jump_ticks: int = 0
    fall_distance: float = 0.0


def _clip(axis, boxes, ent, d):
    """Clip movement `d` of AABB `ent` along axis (0=x,1=y,2=z) against boxes."""
    if d == 0:
        return 0.0
    a1, a2 = (axis + 1) % 3, (axis + 2) % 3
    for b in boxes:
        # overlap on the two other axes?
        if b[a1] >= ent[a1 + 3] - EPS or b[a1 + 3] <= ent[a1] + EPS:
            continue
        if b[a2] >= ent[a2 + 3] - EPS or b[a2 + 3] <= ent[a2] + EPS:
            continue
        if d > 0 and b[axis] >= ent[axis + 3] - EPS:
            gap = b[axis] - ent[axis + 3]
            if gap < d:
                d = max(0.0, gap)
        elif d < 0 and b[axis + 3] <= ent[axis] + EPS:
            gap = b[axis + 3] - ent[axis]
            if gap > d:
                d = min(0.0, gap)
    return d


def _offset(box, dx, dy, dz):
    return (box[0] + dx, box[1] + dy, box[2] + dz, box[3] + dx, box[4] + dy, box[5] + dz)


class Physics:
    def __init__(self, world, gd):
        self.world = world
        self.gd = gd

    # ------------------------------------------------------------------ world queries
    def block_boxes(self, minx, miny, minz, maxx, maxy, maxz) -> list[tuple]:
        out = []
        world = self.world
        shape_boxes = self.gd.shape_boxes
        kinds = self.gd.shape_kind
        for bx in range(math.floor(minx), math.floor(maxx) + 1):
            for bz in range(math.floor(minz), math.floor(maxz) + 1):
                for by in range(math.floor(miny) - 1, math.floor(maxy) + 1):
                    s = world.get(bx, by, bz)
                    if s is None:
                        out.append((bx, by, bz, bx + 1, by + 1, bz + 1))   # unloaded = solid
                        continue
                    if kinds[s] == 0:
                        continue
                    for b in shape_boxes(s):
                        out.append((bx + b[0], by + b[1], bz + b[2], bx + b[3], by + b[4], bz + b[5]))
        return out

    def player_box(self, st: PlayerState, height=PLAYER_H):
        h = PLAYER_W / 2
        return (st.x - h, st.y, st.z - h, st.x + h, st.y + height, st.z + h)

    def _fluid_flags(self, st: PlayerState):
        bx, bz = math.floor(st.x), math.floor(st.z)
        water = lava = climb = False
        for by in (math.floor(st.y + 0.1), math.floor(st.y + 0.9)):
            s = self.world.get(bx, by, bz)
            if s is None:
                continue
            water |= bool(self.gd.is_water[s])
            lava |= bool(self.gd.is_lava[s])
        s = self.world.get(bx, math.floor(st.y), bz)
        if s is not None:
            climb = bool(self.gd.is_climbable[s])
        return water, lava, climb

    # ------------------------------------------------------------------ movement
    def move(self, st: PlayerState, dx, dy, dz, height=PLAYER_H):
        box = self.player_box(st, height)
        sweep = (min(box[0], box[0] + dx), min(box[1], box[1] + dy), min(box[2], box[2] + dz),
                 max(box[3], box[3] + dx), max(box[4], box[4] + dy), max(box[5], box[5] + dz))
        boxes = self.block_boxes(sweep[0] - 0.01, sweep[1] - STEP_HEIGHT, sweep[2] - 0.01,
                                 sweep[3] + 0.01, sweep[4] + STEP_HEIGHT, sweep[5] + 0.01)
        odx, ody, odz = dx, dy, dz
        rdx, rdy, rdz, nb = self._collide(boxes, box, dx, dy, dz)
        on_ground_now = ody < 0 and rdy != ody
        # step up
        if (st.on_ground or on_ground_now) and (rdx != odx or rdz != odz):
            sdx, sdy, sdz, sb = self._collide(boxes, box, odx, STEP_HEIGHT, odz)
            # come back down
            down = _clip(1, boxes, sb, -STEP_HEIGHT - 0.0)
            sb2 = _offset(sb, 0, down, 0)
            if sdx * sdx + sdz * sdz > rdx * rdx + rdz * rdz + 1e-9:
                rdx, rdy, rdz, nb = sdx, sdy + down, sdz, sb2
        st.x = (nb[0] + nb[3]) / 2
        st.y = nb[1]
        st.z = (nb[2] + nb[5]) / 2
        st.collided_h = (abs(rdx - odx) > 1e-6) or (abs(rdz - odz) > 1e-6)
        collided_v = abs(rdy - ody) > 1e-6
        st.on_ground = collided_v and ody < 0
        if rdx != odx:
            st.vx = 0.0
        if rdz != odz:
            st.vz = 0.0
        if collided_v:
            st.vy = 0.0
        if st.on_ground:
            st.fall_distance = 0.0
        elif rdy < 0:
            st.fall_distance -= rdy

    @staticmethod
    def _collide(boxes, box, dx, dy, dz):
        dy = _clip(1, boxes, box, dy)
        box = _offset(box, 0, dy, 0)
        if abs(dx) < abs(dz):
            dz = _clip(2, boxes, box, dz)
            box = _offset(box, 0, 0, dz)
            dx = _clip(0, boxes, box, dx)
            box = _offset(box, dx, 0, 0)
        else:
            dx = _clip(0, boxes, box, dx)
            box = _offset(box, dx, 0, 0)
            dz = _clip(2, boxes, box, dz)
            box = _offset(box, 0, 0, dz)
        return dx, dy, dz, box

    def _input_vector(self, c: Controls):
        fwd = (1.0 if c.forward else 0.0) - (1.0 if c.back else 0.0)
        strafe = (1.0 if c.left else 0.0) - (1.0 if c.right else 0.0)
        if c.sneak:
            fwd *= SNEAK_MULT
            strafe *= SNEAK_MULT
        return fwd * 0.98, strafe * 0.98

    @staticmethod
    def _accelerate(st, fwd, strafe, speed):
        ln = fwd * fwd + strafe * strafe
        if ln < 1e-4:
            return
        ln = math.sqrt(ln)
        if ln < 1.0:
            ln = 1.0
        fwd, strafe = fwd / ln * speed, strafe / ln * speed
        y = math.radians(st.yaw)
        s, c = math.sin(y), math.cos(y)
        st.vx += strafe * c - fwd * s
        st.vz += fwd * c + strafe * s

    def step(self, st: PlayerState, c: Controls, sneak_edge_guard: bool = True):
        """Advance one game tick (50 ms)."""
        if not self.world.is_loaded(math.floor(st.x), math.floor(st.z)):
            return  # vanilla client freezes in unloaded chunks
        st.in_water, st.in_lava, st.on_climbable = self._fluid_flags(st)
        fwd, strafe = self._input_vector(c)
        sprint = c.sprint and fwd > 0 and not c.sneak
        height = SNEAK_H if c.sneak else PLAYER_H
        if st.jump_ticks > 0:
            st.jump_ticks -= 1
        # tiny velocities are zeroed like vanilla
        if abs(st.vx) < 0.003:
            st.vx = 0.0
        if abs(st.vz) < 0.003:
            st.vz = 0.0
        if abs(st.vy) < 0.003:
            st.vy = 0.0

        if st.in_water or st.in_lava:
            if c.jump:
                st.vy += 0.04
            self._accelerate(st, fwd, strafe, 0.02)
            self.move(st, st.vx, st.vy, st.vz, height)
            drag = 0.8 if st.in_water else 0.5
            st.vx *= drag
            st.vz *= drag
            st.vy = st.vy * drag - 0.02
            if st.collided_h and self._free_above(st, height):
                st.vy = 0.3   # climb out of the water onto a ledge
            return

        if c.jump and st.on_ground and st.jump_ticks == 0:
            st.vy = JUMP_VEL
            if sprint:
                y = math.radians(st.yaw)
                st.vx -= math.sin(y) * 0.2
                st.vz += math.cos(y) * 0.2
            st.jump_ticks = 10

        if st.on_ground:
            below = self.world.name_at(math.floor(st.x), math.floor(st.y - 0.5), math.floor(st.z)) or ""
            slip = SLIPPERINESS.get(below, 0.6)
            f = slip * 0.91
            speed = WALK_SPEED * (SPRINT_MULT if sprint else 1.0) * (0.21600002 / (f * f * f))
        else:
            f = 0.91
            speed = 0.026 if sprint else 0.02
        self._accelerate(st, fwd, strafe, speed)

        if st.on_climbable:
            st.vx = max(-0.15, min(0.15, st.vx))
            st.vz = max(-0.15, min(0.15, st.vz))
            st.vy = max(st.vy, -0.15)
            if c.sneak and st.vy < 0:
                st.vy = 0.0

        # sneaking never walks off an edge (vanilla backOffFromEdge)
        if c.sneak and st.on_ground and sneak_edge_guard:
            self._edge_guard(st, height)

        self.move(st, st.vx, st.vy, st.vz, height)
        if st.on_climbable and (st.collided_h or c.jump):
            st.vy = 0.2
        st.vy = (st.vy - GRAVITY) * AIR_DRAG
        st.vx *= f
        st.vz *= f

    def _free_above(self, st, height) -> bool:
        box = _offset(self.player_box(st, height), st.vx, 0.6, st.vz)
        return not self._any_collision(box)

    def _any_collision(self, box) -> bool:
        for b in self.block_boxes(*box):
            if b[0] < box[3] - EPS and b[3] > box[0] + EPS and b[1] < box[4] - EPS and \
                    b[4] > box[1] + EPS and b[2] < box[5] - EPS and b[5] > box[2] + EPS:
                return True
        return False

    def _edge_guard(self, st, height):
        step = 0.05
        box = self.player_box(st, height)

        def has_floor(dx, dz):
            b = _offset(box, dx, -0.6, dz)
            return self._any_collision((b[0], b[1], b[2], b[3], box[1] - 0.01, b[5]))
        dx, dz = st.vx, st.vz
        while dx != 0 and not has_floor(dx, 0):
            dx = 0 if abs(dx) < step else dx - math.copysign(step, dx)
        while dz != 0 and not has_floor(0, dz):
            dz = 0 if abs(dz) < step else dz - math.copysign(step, dz)
        st.vx, st.vz = dx, dz

    def is_standing_space_free(self, x, y, z) -> bool:
        h = PLAYER_W / 2
        return not self._any_collision((x - h, y, z - h, x + h, y + PLAYER_H, z + h))
