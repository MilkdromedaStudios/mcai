"""Buckets: scoop/pour water and lava, water-bucket clutch when falling, burn enemies with lava, and make
obsidian by pouring water over lava (for the nether portal).

Bucket use goes through the "use item" packet: the server ray-casts from the head rotation we send with it,
so we always aim first at the exact block face / fluid surface we want.
"""
from __future__ import annotations

import logging
import math
import time

from .actions import ActionError
from .pathfinder import GoalLookAt

log = logging.getLogger("pyflayer.liquids")
FIRE_IMMUNE = {"blaze", "ghast", "magma_cube", "strider", "wither_skeleton", "zombified_piglin", "piglin",
               "piglin_brute", "hoglin", "zoglin", "wither", "ender_dragon", "warden", "breeze"}


# ---------------------------------------------------------------------------- helpers
def source_states(bot, fluid: str):
    gd = bot.gd
    return [s for s in gd.block_states.get(fluid, []) if gd.state_props[s].get("level") == "0"]


def find_sources(bot, fluid: str, radius: float = 64, count: int = 30, open_top: bool = True):
    import numpy as np
    states = np.array(source_states(bot, fluid), dtype=np.uint16)
    pts = bot.world.find_blocks(bot.eye, states, radius, count * 3)
    if open_top:
        pts = [p for p in pts if bot.world.name_at(p[0], p[1] + 1, p[2]) in ("air", "cave_air")]
    return pts[:count]


def _hotbar_slot(bot, name: str):
    for i in range(9):
        it = bot.inventory.slots[36 + i]
        if it and it.name == name:
            return i
    return None


def keep_in_hotbar(bot, name: str, slot: int = 8):
    """Make sure an item (e.g. the water bucket) sits in the hotbar so reflexes can use it instantly."""
    if _hotbar_slot(bot, name) is not None or bot.inventory.window:
        return
    from .inventory import SWAP
    src = bot.inventory.find(lambda it: it.name == name)
    if src is not None and not 36 <= src <= 44:
        bot.click(src, slot, SWAP)


def _use_at(bot, x: float, y: float, z: float):
    bot.look_at(x, y, z, wait=True)
    bot.use_item()
    bot.swing()


def _wait_for(bot, cond, ticks=20) -> bool:
    for _ in range(ticks):
        bot.wait_ticks(1)
        if cond():
            return True
    return False


# ---------------------------------------------------------------------------- scoop / pour
def fill_bucket(bot, fluid: str = "water", radius: float = 96) -> str:
    """Walk to the nearest source block of water/lava and fill an empty bucket."""
    if not bot.inventory.count("bucket"):
        raise ActionError("no empty bucket (craft one from 3 iron ingots)")
    if fluid == "water" and bot.dimension == "the_nether":
        raise ActionError("there is no water in the Nether")
    full = f"{fluid}_bucket"
    before = bot.inventory.count(full)
    srcs = find_sources(bot, fluid, radius, 20) or find_sources(bot, fluid, radius, 20, open_top=False)
    if not srcs:
        raise ActionError(f"no {fluid} source within {radius:.0f} blocks")
    for src in srcs[:6]:
        try:
            if bot.distance_to_block(src) > 4.0:
                bot.goto(GoalLookAt(*src, reach=3.6), timeout=90)
            bot.equip("bucket")
            _use_at(bot, src[0] + 0.5, src[1] + 0.85, src[2] + 0.5)
            if _wait_for(bot, lambda: bot.inventory.count(full) > before, 15):
                return f"filled a {full.replace('_', ' ')} at {src}"
        except ActionError as e:
            log.debug("fill at %s failed: %s", src, e)
    raise ActionError(f"could not fill a bucket with {fluid}")


def pour(bot, fluid: str, onto, face_up: bool = True) -> tuple:
    """Pour a water/lava bucket onto the top face of block `onto` (liquid appears in the cell above it)."""
    full = f"{fluid}_bucket"
    if not bot.equip(full):
        raise ActionError(f"no {full.replace('_', ' ')}")
    x, y, z = onto
    target = (x, y + 1, z)
    _use_at(bot, x + 0.5, y + 1.0, z + 0.5)
    if not _wait_for(bot, lambda: bot.world.name_at(*target) == fluid, 15):
        raise ActionError(f"pouring {fluid} at {target} failed")
    return target


def scoop(bot, pos) -> bool:
    """Pick a liquid source back up with an empty bucket."""
    if not bot.equip("bucket"):
        return False
    name = bot.world.name_at(*pos)
    _use_at(bot, pos[0] + 0.5, pos[1] + 0.85, pos[2] + 0.5)
    return _wait_for(bot, lambda: bot.world.name_at(*pos) != name, 15)


# ---------------------------------------------------------------------------- burn enemies with lava
def burn_enemy(bot, e) -> str:
    """Pour lava under a mob, back off while it burns, then take the lava back."""
    if e.type in FIRE_IMMUNE:
        raise ActionError(f"{e.type} is immune to fire")
    if not bot.inventory.count("lava_bucket"):
        raise ActionError("no lava bucket")
    floor = (math.floor(e.x), math.floor(e.y) - 1, math.floor(e.z))
    s = bot.world.get(*floor)
    if s is None or not bot.gd.standable[s]:
        raise ActionError("no solid ground under the target to pour lava on")
    me = bot.block_pos
    if math.hypot(floor[0] - me[0], floor[2] - me[2]) < 2:
        raise ActionError("too close - the lava would hit us")
    if bot.distance_to_block(floor) > 4.5:
        raise ActionError("target too far for the bucket")
    lava = pour(bot, "lava", floor)
    bot.chat(f"burning the {e.type}!")
    bot.flee(lava, 6, timeout=5)
    end = time.time() + 6
    while time.time() < end and e.id in bot.entities.by_id:
        time.sleep(0.3)
    # take the lava back so it doesn't spread fire or block our path
    if bot.world.name_at(*lava) == "lava" and bot.inventory.count("bucket"):
        try:
            if bot.distance_to_block(lava) > 4.2:
                bot.goto(GoalLookAt(*lava, reach=3.8), timeout=20, place=False)
            scoop(bot, lava)
        except ActionError:
            pass
    return f"{e.type} {'burned' if e.id not in bot.entities.by_id else 'escaped the lava'}"


# ---------------------------------------------------------------------------- obsidian for the portal
def cast_obsidian(bot, count: int = 10, max_tries: int = 6) -> str:
    """Pour water next to surface lava so lava sources turn into obsidian, take the water back, mine it."""
    if not bot.inventory.count("diamond_pickaxe") and not bot.inventory.count("netherite_pickaxe"):
        raise ActionError("need a diamond pickaxe to mine obsidian")
    if not bot.inventory.count("water_bucket"):
        raise ActionError("need a water bucket")
    start = bot.inventory.count("obsidian")
    tries = 0
    while bot.inventory.count("obsidian") - start < count and tries < max_tries:
        tries += 1
        lava = find_sources(bot, "lava", 96, 40)
        rim = None
        for lx, ly, lz in lava:
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r = (lx + dx, ly, lz + dz)
                s = bot.world.get(*r)
                above = bot.world.name_at(r[0], r[1] + 1, r[2])
                if s is not None and bot.gd.standable[s] and above in ("air", "cave_air"):
                    rim = r
                    break
            if rim:
                break
        if not rim:
            if bot.find_blocks("obsidian", 96, 1):
                break
            raise ActionError("no lava pool with a solid edge nearby")
        bot.goto(GoalLookAt(*rim, reach=3.5), timeout=90)
        if not bot.inventory.count("water_bucket"):
            fill_bucket(bot, "water")
            bot.goto(GoalLookAt(*rim, reach=3.5), timeout=90)
        water = pour(bot, "water", rim)
        bot.wait_ticks(60)                                   # let the water flow over the lava
        scoop(bot, water)
        if bot.find_blocks("obsidian", 24, 1):
            bot.collect_block("obsidian", count - (bot.inventory.count("obsidian") - start), radius=32, timeout=300)
    got = bot.inventory.count("obsidian") - start
    if got <= 0:
        raise ActionError("could not make obsidian")
    return f"made and mined {got} obsidian"


# ---------------------------------------------------------------------------- water-bucket clutch (every tick)
class Clutch:
    """While falling far, aim down and place water just before landing; pick it up again afterwards."""

    def __init__(self, bot):
        self.bot = bot
        self.placed = None
        self.pickup_tick = 0
        self.saves = 0
        self._pending = None     # verified result must come from server inventory
        bot.tick_hooks.append(self._tick)

    def _ground_below(self):
        b = self.bot
        x, z = math.floor(b.state.x), math.floor(b.state.z)
        y0 = math.floor(b.state.y)
        for dy in range(0, 40):
            s = b.world.get(x, y0 - dy, z)
            if s is None:
                return None
            if b.gd.has_collision[s] or b.gd.is_water[s]:
                return y0 - dy, b.gd.is_water[s] or b.gd.is_lava[s]
        return None

    def _tick(self):
        b = self.bot
        st = b.state
        if b.dead or not b.spawned.is_set():
            return
        # pick the water back up once we're down (retry for ~2 s until the source is gone)
        if self.placed:
            if self._pending and not self._pending["reported"]:
                here_now = b.world.get(*self.placed)
                # The local world may predict a water block, so additionally
                # require the SERVER to exchange water_bucket -> bucket.
                bucket_exchanged = b.inventory.count("bucket") > self._pending["bucket_before"]
                landed_safely = (st.on_ground or st.in_water) and b.health >= self._pending["health"]
                if (b.tick_count >= self.pickup_tick and bucket_exchanged and landed_safely
                        and here_now is not None and b.gd.is_water[here_now]):
                    self._pending["reported"] = True
                    self.saves += 1
                    b.events.emit("clutch_result", True, f"server bucket changed; landed at {self.placed}")
                elif b.tick_count > self.pickup_tick + 40:
                    self._pending["reported"] = True
                    b.events.emit("clutch_result", False, "landing/water placement unverified after 40 ticks")
            if b.tick_count < self.pickup_tick or not (st.on_ground or st.in_water):
                return
            here = b.world.get(*self.placed)
            if here is None or not b.gd.is_water[here] or b.tick_count > self.pickup_tick + 40:
                self.placed = None
                self._pending = None
                return
            if (b.tick_count - self.pickup_tick) % 5 == 0:
                slot = _hotbar_slot(b, "bucket")
                if slot is None:
                    self.placed = None
                    return
                b.select_hotbar(slot)
                px, py, pz = self.placed
                ex, ey, ez = b.eye
                yaw = math.degrees(math.atan2(-(px + 0.5 - ex), pz + 0.5 - ez))
                pitch = -math.degrees(math.atan2(py + 0.8 - ey, math.hypot(px + 0.5 - ex, pz + 0.5 - ez)))
                st.yaw, st.pitch = yaw, pitch
                b.send_use_item_now(yaw, pitch)
            return
        if st.on_ground or st.in_water or st.vy > -0.5 or st.fall_distance < 3 or b.dimension == "the_nether":
            return
        slot = _hotbar_slot(b, "water_bucket")
        if slot is None or self.placed:
            return
        g = self._ground_below()
        if not g:
            return
        gy, soft = g
        height = st.y - (gy + 1)
        if soft or st.fall_distance + height < 4:
            return                                           # landing in liquid / not a damaging fall
        b.select_hotbar(slot)
        st.pitch = 90.0
        if height <= 2.8:                                    # within reach of the ground's top face
            b.send_use_item_now(st.yaw, 90.0)
            self.placed = (math.floor(st.x), gy + 1, math.floor(st.z))
            self._pending = {
                "bucket_before": b.inventory.count("bucket"),
                "health": b.health,
                "reported": False,
            }
            # predict the water like the real client does: our physics must enter the water instead of
            # reporting "on ground" (which would make the server apply fall damage before the water counts)
            water = b.gd.block_default.get("water")
            if water is not None:
                b.world.set(*self.placed, water)
            self.pickup_tick = b.tick_count + 8
            b.events.emit("clutch_attempt", self.placed)
            # Do not count a save as successful until the authoritative
            # inventory shows the water bucket was used and we landed safely.
