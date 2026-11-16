"""Late-game actions: nether portals, eyes of ender / strongholds, end portal, bow, ender dragon."""
from __future__ import annotations

import logging
import math
import time

from ..protocol.buffer import Writer
from .actions import RELEASE_USE, ActionError
from .pathfinder import GoalBlock, GoalNear, GoalXZ

log = logging.getLogger("pyflayer.advanced")
FILLERS = ("cobblestone", "dirt", "cobbled_deepslate", "netherrack", "stone", "andesite", "diorite", "granite",
           "oak_planks", "spruce_planks", "birch_planks", "jungle_planks", "deepslate", "tuff", "blackstone")


def _filler(bot) -> str | None:
    return next((f for f in FILLERS if bot.inventory.count(f) > 0), None)


def _clear(bot, p) -> bool:
    s = bot.world.get(*p)
    return s is not None and (bot.gd.is_air[s] or bot.gd.is_replaceable[s]) and not bot.gd.is_liquid[s]


def _find_portal_site(bot, radius=10):
    bx, by, bz = bot.block_pos
    pf = bot.pathfinder
    best = None
    for dy in (0, -1, 1, -2, 2):
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                x0, y0, z0 = bx + dx, by + dy, bz + dz
                for axis in ("x", "z"):
                    cells = [(x0 + i, z0) if axis == "x" else (x0, z0 + i) for i in range(4)]
                    if not all(pf.standable(cx, y0 - 1, cz) for cx, cz in cells):
                        continue
                    if not all(_clear(bot, (cx, y0 + h, cz)) for cx, cz in cells for h in range(5)):
                        continue
                    # standing room in front
                    front = [(x0 + 1, z0 + 2), (x0 + 2, z0 + 2)] if axis == "x" else [(x0 + 2, z0 + 1), (x0 + 2, z0 + 2)]
                    if not all(pf.can_stand(fx, y0, fz) for fx, fz in front):
                        continue
                    d = abs(dx) + abs(dz) + abs(dy) * 2
                    if best is None or d < best[0]:
                        best = (d, (x0, y0, z0), axis, front[0])
        if best:
            return best[1:]
    return None


def build_nether_portal(bot) -> str:
    if bot.inventory.count("obsidian") < 10:
        raise ActionError(f"need 10 obsidian (have {bot.inventory.count('obsidian')})")
    if not bot.inventory.count("flint_and_steel") and not bot.inventory.count("fire_charge"):
        raise ActionError("need flint_and_steel (iron ingot + flint) or a fire charge")
    if sum(bot.inventory.count(f) for f in FILLERS) < 4:
        raise ActionError("need 4 filler blocks (cobblestone/dirt) for the frame corners")
    site = _find_portal_site(bot)
    if not site:
        raise ActionError("no flat 4x5 space nearby; move to open flat ground")
    (x0, y0, z0), axis, stand = site

    def P(i, h):
        return (x0 + i, y0 + h, z0) if axis == "x" else (x0, y0 + h, z0 + i)
    bot.goto(GoalBlock(stand[0], y0, stand[1]), timeout=60)
    order = [(P(0, 0), "f"), (P(3, 0), "f"), (P(1, 0), "o"), (P(2, 0), "o")]
    order += [(P(0, h), "o") for h in (1, 2, 3)] + [(P(3, h), "o") for h in (1, 2, 3)]
    order += [(P(0, 4), "f"), (P(3, 4), "f"), (P(1, 4), "o"), (P(2, 4), "o")]
    for pos, kind in order:
        s = bot.world.get(*pos)
        if s is not None and not bot.gd.is_air[s] and not bot.gd.is_replaceable[s]:
            continue
        item = "obsidian" if kind == "o" else _filler(bot)
        if not item:
            raise ActionError("ran out of blocks while building the portal")
        bot.place_at(pos, item)
    lighter = "flint_and_steel" if bot.inventory.count("flint_and_steel") else "fire_charge"
    bot.equip(lighter)
    inside = P(1, 0)
    bot.look_at(inside[0] + 0.5, inside[1] + 1.0, inside[2] + 0.5, wait=True)
    bot.send("use_item_on", Writer().varint(0).position(*inside).varint(1).float(0.5).float(1.0).float(0.5)
             .bool(False).bool(False).varint(bot._next_seq()))
    bot.swing()
    for _ in range(30):
        bot.wait_ticks(1)
        if bot.world.name_at(*P(1, 1)) == "nether_portal":
            bot.events.emit("portal_built", P(1, 1))
            return f"nether portal built and lit at {P(1, 1)}"
    raise ActionError("portal frame built but did not light (check the frame shape)")


def enter_portal(bot, kind: str = "nether_portal", timeout: float = 25) -> str:
    pts = bot.find_blocks(kind, 96, 1)
    if not pts:
        raise ActionError(f"no {kind} nearby")
    dim = bot.dimension
    x, y, z = pts[0]
    bot.goto(GoalBlock(x, y, z), timeout=90, dig=False)
    end = time.time() + timeout
    while time.time() < end:
        if bot.dimension != dim:
            bot.wait_spawn(30)
            bot.wait_ticks(40)
            return f"travelled {dim} -> {bot.dimension}, now at {tuple(round(v) for v in bot.position)}"
        bot.wait_ticks(5)
        # keep standing inside the portal
        if bot.world.name_at(*bot.block_pos) != kind:
            try:
                bot.goto(GoalBlock(x, y, z), timeout=10, dig=False)
            except ActionError:
                pass
    raise ActionError("portal did not teleport us")


def throw_ender_eye(bot, watch: float = 3.0):
    """Throw an eye of ender and return (direction angle radians, horizontal flight vector) or None."""
    if not bot.equip("ender_eye"):
        raise ActionError("no ender_eye")
    start = bot.position
    seen: dict[int, list] = {}

    def on_spawn(e):
        if e.type == "eye_of_ender":
            seen[e.id] = [(e.x, e.y, e.z)]
    bot.events.on("entity_spawn", on_spawn)
    try:
        bot.look(bot.state.yaw, -10)
        bot.wait_ticks(1)
        bot.use_item()
        end = time.time() + watch
        while time.time() < end:
            time.sleep(0.1)
            for eid in list(seen):
                e = bot.entities.get(eid)
                if e:
                    seen[eid].append((e.x, e.y, e.z))
    finally:
        bot.events.off("entity_spawn", on_spawn)
    if not seen:
        return None
    track = max(seen.values(), key=len)
    if len(track) < 2:
        return None
    (x1, y1, z1), (x2, y2, z2) = track[0], track[-1]
    dx, dz = x2 - x1, z2 - z1
    if math.hypot(dx, dz) < 0.5:
        return ("here", start)        # eye went straight up/down: stronghold is right here
    return (math.atan2(dz, dx), start)


def _intersect(p1, a1, p2, a2):
    d1 = (math.cos(a1), math.sin(a1))
    d2 = (math.cos(a2), math.sin(a2))
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 0.05:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[2] - p1[2]) * d2[0]) / den
    return (p1[0] + d1[0] * t, p1[2] + d1[1] * t)


def find_stronghold(bot, max_throws: int = 6) -> str:
    """Triangulate the stronghold with eyes of ender, travel there and dig down to the portal room."""
    readings = []
    for i in range(max_throws):
        if not bot.inventory.count("ender_eye"):
            raise ActionError("out of eyes of ender")
        r = throw_ender_eye(bot)
        bot.collect_items(10, 6, names={"ender_eye"})
        if r is None:
            continue
        ang, pos = r
        if ang == "here":
            break
        readings.append((pos, ang))
        if len(readings) >= 2:
            est = _intersect(readings[-2][0], readings[-2][1], readings[-1][0], readings[-1][1])
            if est and math.hypot(est[0] - pos[0], est[1] - pos[2]) < 3000:
                bot.goto(GoalXZ(est[0], est[1], 8), timeout=1200)
                break
        # walk sideways-ish so the second line gives a good intersection
        side = ang + (math.pi / 2.5 if i % 2 == 0 else -math.pi / 2.5)
        bot.goto(GoalXZ(pos[0] + math.cos(side) * 150, pos[2] + math.sin(side) * 150, 8), timeout=600)
    # look for the portal room
    for depth in range(6):
        frames = bot.find_blocks("end_portal_frame", 112, 12)
        if frames:
            fx, fy, fz = frames[0]
            bot.goto(GoalNear(fx, fy + 1, fz, 3), timeout=600)
            return f"found stronghold portal room near {frames[0]}"
        x, y, z = bot.position
        bot.goto(GoalNear(x, max(bot.world.min_y + 10, y - 20), z, 3), timeout=300)
    raise ActionError("stronghold not found yet; keep following the eyes")


def fill_end_portal(bot) -> str:
    frames = bot.find_blocks("end_portal_frame", 32, 12)
    if not frames:
        raise ActionError("no end portal frame nearby")
    placed = 0
    for f in frames:
        s = bot.world.get(*f)
        if bot.gd.state_props[s].get("eye") == "true":
            continue
        if not bot.equip("ender_eye"):
            raise ActionError(f"out of eyes of ender ({placed} placed)")
        if bot.distance_to_block(f) > 4:
            bot.goto(GoalNear(f[0], f[1] + 1, f[2], 3), timeout=60)
        bot.look_at(f[0] + 0.5, f[1] + 0.8, f[2] + 0.5, wait=True)
        bot.send("use_item_on", Writer().varint(0).position(*f).varint(1).float(0.5).float(0.8125).float(0.5)
                 .bool(False).bool(False).varint(bot._next_seq()))
        bot.swing()
        bot.wait_ticks(4)
        placed += 1
    if bot.find_blocks("end_portal", 32, 1):
        return f"end portal activated ({placed} eyes placed)"
    return f"placed {placed} eyes; portal not active yet (all 12 frames need eyes)"


def bow_pitch(dx: float, dy: float, v: float = 3.0, g: float = 0.05, drag: float = 0.99) -> float | None:
    """Pitch (degrees, MC convention: negative = up) to hit a target dx away horizontally and dy up."""
    best = None
    for p10 in range(-800, 801, 5):
        pitch = p10 / 10.0
        r = math.radians(-pitch)
        vx, vy = v * math.cos(r), v * math.sin(r)
        x = y = 0.0
        for _ in range(200):
            x += vx
            y += vy
            vx *= drag
            vy = vy * drag - g
            if x >= dx:
                err = abs(y - dy)
                if best is None or err < best[0]:
                    best = (err, pitch)
                break
            if y < dy - 40:
                break
    return best[1] if best and best[0] < 1.0 else None


def shoot_bow(bot, target) -> bool:
    if not bot.inventory.count("bow") or not bot.inventory.count("arrow"):
        return False
    bot.equip("bow")
    ex, ey, ez = bot.eye
    tx, ty, tz = target.x, target.y + target.height * 0.5, target.z
    dx = math.hypot(tx - ex, tz - ez)
    pitch = bow_pitch(dx, ty - ey)
    if pitch is None:
        return False
    yaw = math.degrees(math.atan2(-(tx - ex), tz - ez))
    bot.look(yaw, pitch)
    bot.use_item()
    bot.wait_ticks(22)      # full draw
    bot.look(yaw, pitch)
    bot.wait_ticks(1)
    bot.send("player_action", Writer().varint(RELEASE_USE).position(0, 0, 0).byte(0).varint(0))
    return True


def fight_dragon(bot, timeout: float = 1200) -> str:
    """Basic dragon strategy: destroy end crystals (bow or melee from a pillar), then hit the dragon when it
    perches on the fountain. Eats and retreats when hurt."""
    if bot.dimension != "the_end":
        raise ActionError("not in the End")
    end = time.time() + timeout
    while time.time() < end:
        bot._check()
        if bot.health < 10 and bot.food > 0:
            try:
                bot.eat()
            except ActionError:
                pass
        crystals = bot.entities.within(bot.position, 150, lambda e: e.type == "end_crystal")
        dragon = bot.entities.nearest(bot.position, lambda e: e.type == "ender_dragon", 300)
        if crystals:
            c = crystals[0]
            if shoot_bow(bot, c):
                bot.wait_ticks(10)
                continue
            # climb the pillar with blocks and punch the crystal
            try:
                bot.goto(GoalNear(c.x, c.y - 1, c.z, 2.5), timeout=180)
                bot.attack(c, wait_cooldown=False)
            except ActionError as e:
                log.info("crystal approach failed: %s", e)
            continue
        if not dragon:
            if bot.find_blocks("end_gateway", 200, 1) or not bot.entities.nearest(
                    bot.position, lambda e: e.type == "ender_dragon", 600):
                return "the ender dragon is dead (or not loaded) - the game is beaten!"
            bot.wait_ticks(20)
            continue
        # wait at the fountain for the perch
        if math.hypot(dragon.x, dragon.z) < 12 and dragon.y < 75:
            try:
                bot.goto(GoalNear(dragon.x, dragon.y, dragon.z, 3), timeout=10, dig=False, place=False)
            except ActionError:
                pass
            for _ in range(10):
                if bot.entity_distance(dragon) <= 3.0:
                    bot.attack(dragon)
                bot.wait_ticks(2)
        else:
            if math.hypot(bot.state.x, bot.state.z) > 8:
                try:
                    bot.goto(GoalXZ(0, 4, 5), timeout=30, dig=False)
                except ActionError:
                    pass
            if not shoot_bow(bot, dragon):
                bot.wait_ticks(10)
    return "dragon fight timed out"
