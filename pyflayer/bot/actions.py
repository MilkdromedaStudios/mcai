"""High level actions (mixed into Bot): movement, digging, placing, combat, inventory, crafting..."""
from __future__ import annotations

import functools
import logging
import math
import threading
import time

from ..protocol.buffer import Writer
from ..protocol.items import ItemCodec
from .inventory import PICKUP, QUICK_MOVE, SWAP, THROW
from .pathfinder import Goal, GoalBlock, GoalInvert, GoalLookAt, GoalNear, GoalXZ, Pathfinder

log = logging.getLogger("pyflayer.actions")

FACES = {0: (0, -1, 0), 1: (0, 1, 0), 2: (0, 0, -1), 3: (0, 0, 1), 4: (-1, 0, 0), 5: (1, 0, 0)}
FACE_OF = {v: k for k, v in FACES.items()}
REACH = 4.5
# ServerboundPlayerActionPacket.Action ordinals in 26.x (CHANGE_DESTROY_DIRECTION was inserted at 1)
DIG_START, DIG_CHANGE_DIRECTION, DIG_ABORT, DIG_STOP, DROP_ALL, DROP_ONE, RELEASE_USE, SWAP_HANDS, STAB = range(9)
ATTACK_REACH = 3.0
FUEL_VALUE = {"coal": 8, "charcoal": 8, "coal_block": 80, "lava_bucket": 100, "blaze_rod": 12,
              "dried_kelp_block": 20, "stick": 0.5, "bamboo": 0.25}
ARMOR_RANK = {"leather": 1, "golden": 2, "chainmail": 3, "copper": 3, "iron": 4, "diamond": 5, "netherite": 6}
TOOL_RANK = {"wooden": 1, "golden": 1, "stone": 2, "copper": 2, "iron": 3, "diamond": 4, "netherite": 5}


class ActionError(Exception):
    pass


def action(fn):
    """Marks a top-level action: tracks the current action name and resets the stop flag."""
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        outer = self._action_depth == 0
        if outer:
            self._stop_flag.clear()
            self.current_action = fn.__name__
        self._action_depth += 1
        try:
            return fn(self, *a, **kw)
        finally:
            self._action_depth -= 1
            if outer:
                self.current_action = "idle"
                self.controls.clear()
    return wrapper


def _fuel_value(gd, name: str) -> float:
    if name in FUEL_VALUE:
        return FUEL_VALUE[name]
    if name.endswith(("_planks", "_log", "_wood", "_stem", "_hyphae")) or name in gd.item_tag("logs"):
        return 1.5
    if name.endswith("_slab") and "wood" not in name and name.split("_")[0] in ("oak", "spruce", "birch"):
        return 0.75
    return 0.0


class ActionsMixin:
    def _init_actions(self):
        self.learned_hardness: dict[str, float] = {}
        self.pathfinder = Pathfinder(self)
        self.current_action = "idle"
        self._stop_flag = threading.Event()
        self._action_depth = 0
        self._last_attack = 0.0
        self.unreachable: dict[tuple, float] = {}   # block -> time we gave up on it

    # ================================================================== control
    def stop(self):
        """Interrupt whatever action is running."""
        self._stop_flag.set()
        self.controls.clear()

    def _check(self):
        if self._stop_flag.is_set():
            raise ActionError("stopped")
        if not self.connected:
            raise ActionError("disconnected")
        if self.dead:
            raise ActionError("bot died")

    def _wait(self, ticks=1):
        self._check()
        self.wait_ticks(ticks)

    # ================================================================== hand / swing helpers
    def swing(self):
        self.send("punch")

    def select_hotbar(self, i: int):
        if self.inventory.held_slot != i:
            self.send("set_carried_item", Writer().short(i))
            self.inventory.held_slot = i

    # ================================================================== inventory clicks
    def click(self, slot: int, button: int = 0, mode: int = PICKUP, settle: float = 0.35):
        inv = self.inventory
        w = (Writer().varint(inv.active_id()).varint(inv.active_state()).short(slot).byte(button)
             .varint(mode).varint(0))
        ItemCodec.write_hashed(w, inv.cursor)
        v = inv.version
        self.send("container_click", w)
        if inv.wait_change(v, settle):
            # updates for one click arrive together; give stragglers a moment
            while inv.wait_change(inv.version, 0.04):
                pass

    def close_window(self):
        inv = self.inventory
        if inv.window:
            self.send("container_close", Writer().varint(inv.window.id))
            inv.on_close()
        elif inv.cursor:
            # put back anything held on the cursor in the player inventory
            self.send("container_close", Writer().varint(0))

    def _active_index(self, ws: int) -> int:
        return self.inventory.player_slot_to_active(ws)

    def equip(self, name_or_pred, dest: str = "hand") -> bool:
        inv = self.inventory
        pred = name_or_pred if callable(name_or_pred) else (lambda it: it.name == name_or_pred)
        held = inv.held_item
        if dest == "hand" and held and pred(held):
            return True
        slot = inv.find(pred)
        if slot is None:
            return False
        if dest == "hand":
            if 36 <= slot <= 44:
                self.select_hotbar(slot - 36)
                return True
            hb = next((i for i in range(9) if inv.slots[36 + i] is None), inv.held_slot)
            self.click(self._active_index(slot), hb, SWAP)
            self.select_hotbar(hb)
            return bool(inv.held_item and pred(inv.held_item))
        if dest == "offhand":
            self.click(self._active_index(slot), 40, SWAP)
            return True
        # armor: shift click equips into the right slot
        if self.inventory.window:
            self.close_window()
        self.click(slot, 0, QUICK_MOVE)
        return True

    def equip_best_tool(self, block_name: str):
        names = [it.name for it in self.inventory.items()]
        tool, speed, _ = self.gd.best_tool_for(block_name, names)
        if tool and speed > 1.0:
            self.equip(tool)
        elif self.inventory.held_item and self.gd.tool_rules(self.inventory.held_item.name):
            # don't waste durability: switch to an empty/non-tool hotbar slot
            for i in range(9):
                it = self.inventory.slots[36 + i]
                if it is None or not self.gd.tool_rules(it.name):
                    self.select_hotbar(i)
                    break

    def equip_best_weapon(self):
        best, dmg = None, 1.0
        for it in self.inventory.items():
            d = self.gd.attack_damage(it.name)
            if d > dmg:
                best, dmg = it.name, d
        if best:
            self.equip(best)

    def equip_armor(self) -> list[str]:
        """Put on the best armor pieces we carry."""
        worn = []
        slots = {"helmet": 5, "chestplate": 6, "leggings": 7, "boots": 8}
        for piece, ws in slots.items():
            def rank(n):
                return ARMOR_RANK.get(n.split("_")[0], 0)
            cur = self.inventory.slots[ws]
            cands = [it for it in self.inventory.slots[9:45] if it and it.name.endswith("_" + piece)]
            if not cands:
                continue
            best = max(cands, key=lambda it: rank(it.name))
            if cur and rank(cur.name) >= rank(best.name):
                continue
            if cur:  # take off the worse piece first
                self.click(ws, 0, QUICK_MOVE)
            self.equip(lambda it, n=best.name: it.name == n, "armor")
            worn.append(best.name)
        return worn

    def toss(self, name: str, count: int | None = None) -> int:
        """Drop items on the ground. Returns how many were dropped."""
        dropped = 0
        while count is None or dropped < count:
            slot = self.inventory.find(lambda it: it.name == name)
            if slot is None:
                break
            it = self.inventory.slots[slot]
            idx = self._active_index(slot)
            if count is None or it.count <= count - dropped:
                self.click(idx, 1, THROW)        # ctrl+Q: whole stack
                dropped += it.count
            else:
                for _ in range(count - dropped):
                    self.click(idx, 0, THROW)
                    dropped += 1
        return dropped

    JUNK = {"dirt": 32, "gravel": 8, "andesite": 0, "diorite": 0, "granite": 0, "tuff": 0, "cobbled_deepslate": 64,
            "netherrack": 64, "cobblestone": 128, "rotten_flesh": 0, "wheat_seeds": 0, "poisonous_potato": 0,
            "spider_eye": 4, "flint": 8, "sand": 32, "calcite": 0, "dripstone_block": 0, "pointed_dripstone": 0,
            "moss_block": 0, "clay_ball": 16, "bone": 8, "arrow": 64, "stick": 32, "kelp": 0, "sunflower": 0}

    def drop_junk(self, min_free: int = 4) -> int:
        """Throw away low-value items (keeping a reserve of building blocks) when space runs low."""
        if self.inventory.empty_slots() >= min_free:
            return 0
        if self.inventory.window:
            self.close_window()
        dropped = 0
        counts = self.inventory.counts()
        for name, keep in self.JUNK.items():
            extra = counts.get(name, 0) - keep
            if extra > 0:
                dropped += self.toss(name, extra)
        for name in [n for n in counts if n.endswith("_sapling") or n in ("short_grass", "fern")]:
            dropped += self.toss(name)
        return dropped

    # ================================================================== looking / raycast
    def _block_center(self, pos):
        return (pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5)

    def _face_point(self, pos, face):
        dx, dy, dz = FACES[face]
        return (pos[0] + 0.5 + dx * 0.5, pos[1] + 0.5 + dy * 0.5, pos[2] + 0.5 + dz * 0.5)

    def _best_face(self, pos) -> int:
        ex, ey, ez = self.eye
        best, bd = None, 1e9
        fallback, fd = 1, 1e9
        for f, (dx, dy, dz) in FACES.items():
            cx, cy, cz = self._face_point(pos, f)
            dot = (ex - cx) * dx + (ey - cy) * dy + (ez - cz) * dz
            d = (ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2
            if dot > 0 and d < fd:
                fallback, fd = f, d
            n = self.world.get(pos[0] + dx, pos[1] + dy, pos[2] + dz)
            exposed = n is not None and (not self.gd.has_collision[n] or self.gd.is_liquid[n])
            if dot > 0 and exposed and d < bd:
                best, bd = f, d
        return best if best is not None else fallback

    def can_see_block(self, pos) -> bool:
        """Voxel raycast from the eye to the block center; True if nothing solid is in between."""
        ex, ey, ez = self.eye
        tx, ty, tz = self._block_center(pos)
        dx, dy, dz = tx - ex, ty - ey, tz - ez
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-6:
            return True
        steps = int(dist * 4)
        for i in range(1, steps):
            t = i / steps
            bx, by, bz = math.floor(ex + dx * t), math.floor(ey + dy * t), math.floor(ez + dz * t)
            if (bx, by, bz) == tuple(pos):
                return True
            s = self.world.get(bx, by, bz)
            if s is None or (self.gd.shape_kind[s] == 1 and not self.gd.is_liquid[s]):
                return False
        return True

    def distance_to_block(self, pos) -> float:
        ex, ey, ez = self.eye
        cx, cy, cz = self._block_center(pos)
        return math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2)

    def look_at_entity(self, e, wait=True):
        self.look_at(e.x, e.y + e.height * 0.8, e.z, wait=wait)

    # ================================================================== digging
    @action
    def dig(self, pos, force_look=True) -> bool:
        x, y, z = (int(v) for v in pos)
        s = self.world.get(x, y, z)
        if s is None:
            raise ActionError("block is in an unloaded chunk")
        if self.gd.is_air[s] or (self.gd.is_liquid[s] and not self.gd.has_collision[s] and
                                 self.gd.block_name(s) in ("water", "lava")):
            return True
        name = self.gd.block_name(s)
        if self.gd.hardness(name) < 0:
            raise ActionError(f"{name} is unbreakable")
        if self.distance_to_block((x, y, z)) > REACH + 0.5:
            raise ActionError(f"{name} at {x},{y},{z} is out of reach")
        if self.gamemode != 1:
            self.equip_best_tool(name)
        face = self._best_face((x, y, z))
        if force_look:
            self.look_at(*self._face_point((x, y, z), face), wait=True)
        held = self.inventory.held_item
        ticks = 0 if self.gamemode == 1 else self.gd.dig_ticks(
            name, held.name if held else None, self.state.in_water, self.state.on_ground or self.state.in_water,
            self.learned_hardness)
        if ticks == math.inf:
            raise ActionError(f"cannot break {name}")
        self.send("player_action", Writer().varint(DIG_START).position(x, y, z).byte(face).varint(self._next_seq()))
        self.swing()
        t0 = time.time()
        try:
            if ticks > 0:
                end = t0 + ticks * 0.05
                n = 0
                while time.time() < end:
                    self._wait(1)
                    n += 1
                    if n % 4 == 0:
                        self.swing()
                    if self.world.get(x, y, z) != s:
                        break
                if self.world.get(x, y, z) == s:
                    self.send("player_action", Writer().varint(DIG_STOP).position(x, y, z).byte(face)
                              .varint(self._next_seq()))
            # wait for the server to confirm the break (delayed destroy covers small mis-estimates)
            limit = time.time() + max(1.5, ticks * 0.05 * 1.5 + 1.0)
            while self.world.get(x, y, z) == s and time.time() < limit:
                self._wait(1)
                if int((time.time() - t0) * 20) % 4 == 0:
                    self.swing()
        except ActionError:
            self.send("player_action", Writer().varint(DIG_ABORT).position(x, y, z).byte(0).varint(self._next_seq()))
            raise
        if self.world.get(x, y, z) == s:
            self.send("player_action", Writer().varint(DIG_ABORT).position(x, y, z).byte(0).varint(self._next_seq()))
            raise ActionError(f"failed to break {name} at {x},{y},{z}")
        actual = time.time() - t0
        self._learn_dig(name, held.name if held else None, actual, ticks)
        self.events.emit("dug", (x, y, z), name)
        return True

    def _learn_dig(self, block, tool, seconds, predicted_ticks):
        """Learn real hardness when the server took noticeably longer than predicted."""
        actual_ticks = seconds * 20
        if predicted_ticks <= 0 or actual_ticks < predicted_ticks * 1.6 + 6:
            return
        speed, correct = self.gd.tool_speed(tool, block)
        if not self.state.on_ground:
            return
        h = speed * actual_ticks / (30 if correct else 100)
        old = self.learned_hardness.get(block)
        self.learned_hardness[block] = round(h if old is None else (old + h) / 2, 3)
        self.events.emit("learn", "hardness", {"block": block, "hardness": self.learned_hardness[block]})

    # ================================================================== placing / using
    def _find_reference(self, target):
        """A solid neighbour we can click on to place a block at target. Returns (ref, face)."""
        x, y, z = target
        order = [(0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0)]
        ex, ey, ez = self.eye
        cands = []
        for dx, dy, dz in order:
            rx, ry, rz = x + dx, y + dy, z + dz
            s = self.world.get(rx, ry, rz)
            if s is None or not self.gd.has_collision[s] or self.gd.is_replaceable[s]:
                continue
            face = FACE_OF[(-dx, -dy, -dz)]
            cx, cy, cz = self._face_point((rx, ry, rz), face)
            cands.append(((ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2, (rx, ry, rz), face))
        if not cands:
            return None
        cands.sort()
        return cands[0][1], cands[0][2]

    @action
    def place_block(self, ref, face: int, item: str | None = None, sneak: bool = True) -> bool:
        if item and not self.equip(item):
            raise ActionError(f"no {item} in inventory")
        held = self.inventory.held_item
        if not held:
            raise ActionError("nothing in hand to place")
        dx, dy, dz = FACES[face]
        target = (ref[0] + dx, ref[1] + dy, ref[2] + dz)
        before = self.world.get(*target)
        cur = (0.5 + dx * 0.5, 0.5 + dy * 0.5, 0.5 + dz * 0.5)
        self.look_at(ref[0] + cur[0], ref[1] + cur[1], ref[2] + cur[2], wait=True)
        prev_sneak = self.controls.sneak
        if sneak:   # sneaking avoids opening containers / activating the clicked block
            self.controls.sneak = True
            self._wait(1)
        self.send("use_item_on", Writer().varint(0).position(*ref).varint(face).float(cur[0]).float(cur[1])
                  .float(cur[2]).bool(False).bool(False).varint(self._next_seq()))
        self.swing()
        ok = False
        for _ in range(20):
            self._wait(1)
            s = self.world.get(*target)
            if s is not None and s != before and not self.gd.is_air[s]:
                ok = True
                break
        self.controls.sneak = prev_sneak
        if not ok:
            raise ActionError(f"placing {held.name} at {target} failed")
        self.events.emit("placed", target, held.name)
        return True

    @action
    def place_at(self, target, item: str) -> bool:
        target = tuple(int(v) for v in target)
        s = self.world.get(*target)
        if s is not None and not self.gd.is_replaceable[s] and not self.gd.is_air[s]:
            raise ActionError(f"{target} is occupied by {self.gd.block_name(s)}")
        ref = self._find_reference(target)
        if not ref:
            raise ActionError(f"nothing to place against at {target}")
        if self.distance_to_block(target) > REACH:
            self.goto(GoalLookAt(*target, reach=3.8), timeout=30)
        # don't place into our own body
        bx, by, bz = self.block_pos
        if (target[0], target[2]) == (bx, bz) and target[1] in (by, by + 1):
            self.goto(GoalInvert(target[0] + 0.5, target[1], target[2] + 0.5, 1.6), timeout=10)
        return self.place_block(ref[0], ref[1], item)

    @action
    def activate_block(self, pos) -> bool:
        """Right-click a block (doors, buttons, levers, containers, beds...)."""
        pos = tuple(int(v) for v in pos)
        if self.distance_to_block(pos) > REACH:
            self.goto(GoalLookAt(*pos, reach=3.8), timeout=40)
        face = self._best_face(pos)
        cx, cy, cz = self._face_point(pos, face)
        self.look_at(cx, cy, cz, wait=True)
        prev = self.controls.sneak
        self.controls.sneak = False
        self.send("use_item_on", Writer().varint(0).position(*pos).varint(face)
                  .float(cx - pos[0]).float(cy - pos[1]).float(cz - pos[2]).bool(False).bool(False)
                  .varint(self._next_seq()))
        self.swing()
        self.controls.sneak = prev
        return True

    @action
    def open_container(self, pos, timeout=4.0):
        if self.inventory.window:
            self.close_window()
        self.activate_block(pos)
        end = time.time() + timeout
        while time.time() < end:
            if self.inventory.window:
                # wait for its contents
                self.inventory.wait_change(self.inventory.version, 0.3)
                return self.inventory.window
            self._wait(1)
        raise ActionError(f"container at {pos} did not open")

    def send_use_item_now(self, yaw: float, pitch: float):
        """Right-click with the held item immediately (safe to call from the tick loop)."""
        self.send("use_item", Writer().varint(0).varint(self._next_seq()).float(yaw).float(pitch))

    def use_item(self, hold_ticks: int = 0):
        self.send("use_item", Writer().varint(0).varint(self._next_seq()).float(self.state.yaw)
                  .float(self.state.pitch))
        if hold_ticks:
            self._wait(hold_ticks)
            self.send("player_action", Writer().varint(RELEASE_USE).position(0, 0, 0).byte(0).varint(0))

    @action
    def eat(self, food: str | None = None) -> str:
        if self.food >= 20 and food is None:
            return "not hungry"
        cands = [it for it in self.inventory.items() if self.gd.food(it.name)]
        bad = {"rotten_flesh", "spider_eye", "poisonous_potato", "pufferfish", "chicken"}
        if food:
            cands = [it for it in cands if it.name == food]
        else:
            good = [it for it in cands if it.name not in bad]
            cands = good or cands
        if not cands:
            raise ActionError("no food in inventory")
        best = max(cands, key=lambda it: self.gd.food(it.name).get("nutrition", 0))
        self.equip(best.name)
        before_food, before_cnt = self.food, self.inventory.count(best.name)
        self.use_item()
        end = time.time() + 3.0
        while time.time() < end:
            self._wait(2)
            if self.inventory.count(best.name) < before_cnt or self.food > before_food:
                break
        return f"ate {best.name} (food {before_food} -> {self.food})"

    # ================================================================== combat
    def _attack_cooldown(self) -> float:
        held = self.inventory.held_item
        speed = 4.0
        if held:
            for m in self.gd.item_components.get(held.name, {}).get("minecraft:attribute_modifiers", []) or []:
                if isinstance(m, dict) and m.get("type", "").endswith("attack_speed"):
                    speed += float(m.get("amount", 0))
        return 1.0 / max(0.5, speed)

    def entity_distance(self, e) -> float:
        """Distance from our eye to the closest point of the entity's hitbox."""
        ex, ey, ez = self.eye
        hw = e.width / 2
        cx = min(max(ex, e.x - hw), e.x + hw)
        cy = min(max(ey, e.y), e.y + e.height)
        cz = min(max(ez, e.z - hw), e.z + hw)
        return math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2)

    def attack(self, e, wait_cooldown=True) -> bool:
        if e is None or e.id not in self.entities.by_id:
            return False
        if self.entity_distance(e) > ATTACK_REACH:
            return False
        self.look_at_entity(e, wait=True)
        if wait_cooldown:
            left = self._last_attack + self._attack_cooldown() - time.time()
            if left > 0:
                time.sleep(left)
        self.send("attack", Writer().varint(e.id))
        self.swing()
        self._last_attack = time.time()
        return True

    def interact_entity(self, e) -> bool:
        if self.entity_distance(e) > ATTACK_REACH:
            return False
        self.look_at_entity(e, wait=True)
        self.send("interact", Writer().varint(e.id).varint(0).lpvec3(0.0, e.height / 2, 0.0).bool(False))
        self.swing()
        return True

    @action
    def fight(self, target=None, timeout: float = 45, flee_health: float = 5) -> str:
        """Fight one entity (or the nearest hostile) with the 20 TPS combat engine."""
        if target is None:
            target = self.entities.nearest(self.position, lambda e: e.hostile, 24)
            if target is None:
                return "no hostile mobs nearby"
        # a player trick: pour lava under the mob from a safe distance (then take the lava back)
        if self.inventory.count("lava_bucket") and self.dimension != "the_nether" and target.type != "player" \
                and 2.5 < self.entity_distance(target) < 4.5:
            from .liquids import FIRE_IMMUNE, burn_enemy
            if target.type not in FIRE_IMMUNE:
                try:
                    res = burn_enemy(self, target)
                    if target.id not in self.entities.by_id:
                        return f"{target.type}: {res}"
                except ActionError:
                    pass
        res = self.combat.fight_until(target, None, timeout)
        return f"{target.name or target.type}: {res}"

    @action
    def flee(self, from_pos, distance: float = 16, timeout: float = 20) -> str:
        try:
            self.goto(GoalInvert(from_pos[0], from_pos[1], from_pos[2], distance), timeout=timeout,
                      dig=False, place=True)
        except ActionError as e:
            return f"flee incomplete: {e}"
        return "fled"

    # ================================================================== movement
    def _start_node(self):
        x, y, z = self.state.x, self.state.y, self.state.z
        bx, by, bz = math.floor(x), math.floor(y + 0.01), math.floor(z)
        pf = self.pathfinder
        if pf.can_stand(bx, by, bz):
            return (bx, by, bz)
        # standing on a partial block / edge: pick the nearest standable neighbour cell
        best, bd = (bx, by, bz), 9.0
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for dy in (0, 1, -1):
                    nx, ny, nz = bx + dx, by + dy, bz + dz
                    if pf.can_stand(nx, ny, nz):
                        d = (nx + 0.5 - x) ** 2 + (nz + 0.5 - z) ** 2 + (ny - y) ** 2
                        if d < bd:
                            best, bd = (nx, ny, nz), d
        return best

    @action
    def goto(self, goal, y=None, z=None, range_: float = 1.0, timeout: float = 120,
             dig: bool = True, place: bool = True, max_replans: int = 60) -> bool:
        """Walk to a Goal (or x,y,z). Digs / pillars / bridges when allowed."""
        if not isinstance(goal, Goal):
            goal = GoalNear(goal, y, z, range_) if z is not None else GoalNear(*goal, range_)
        pf = self.pathfinder
        pf.can_dig = dig
        pf.can_place = place
        end = time.time() + timeout
        replans = 0
        last_progress = (time.time(), self.position)
        try:
            while True:
                self._check()
                start = self._start_node()
                if goal.is_end(*start):
                    return True
                if time.time() > end:
                    raise ActionError("goto timed out")
                res = pf.find(start, goal, timeout=min(2.0, max(0.3, end - time.time())))
                if not res.steps:
                    raise ActionError(f"no path ({res.status}, {res.visited} nodes)")
                log.debug("path %s: %d steps, %d nodes, %.0fms", res.status, len(res.steps), res.visited, res.time_ms)
                self._follow(res.steps[:48], end)
                replans += 1
                p = self.position
                if math.dist(p, last_progress[1]) > 2:
                    last_progress = (time.time(), p)
                elif time.time() - last_progress[0] > 20:
                    raise ActionError("stuck: no progress for 20s")
                if replans > max_replans:
                    raise ActionError("too many replans")
        finally:
            self.controls.clear()

    def _follow(self, steps, deadline) -> bool:
        for i, st in enumerate(steps):
            self._check()
            if time.time() > deadline:
                return False
            if st.dig:
                for b in st.dig:
                    s = self.world.get(*b)
                    if s is not None and not self.gd.passable[s]:
                        self.controls.clear()
                        try:
                            self.dig(b)
                        except ActionError as e:
                            log.debug("dig during path failed: %s", e)
                            self.pathfinder.avoid.add(tuple(b))
                            return False
            if st.kind == "pillar":
                if not self._pillar(st):
                    return False
                continue
            if st.kind == "bridge":
                if not self._bridge(st):
                    return False
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            if not self._move_to(st, nxt, steps[i + 1:i + 4]):
                return False
        return True

    def _move_to(self, st, nxt, lookahead) -> bool:
        tx, ty, tz = st.x + 0.5, st.y, st.z + 0.5
        c = self.controls
        start = time.time()
        best = 1e9
        stall = 0
        limit = 6.0 if st.kind in ("walk", "jump", "swim", "bridge") else 10.0
        straight = all(s.kind == "walk" and not s.dig for s in lookahead) and len(lookahead) >= 2
        jumped = False
        while True:
            self._wait(1)
            s = self.state
            if st.kind == "gap_jump" and s.in_lava:
                c.clear()
                return False
            dx, dz = tx - s.x, tz - s.z
            hd = math.hypot(dx, dz)
            dy = ty - s.y
            tol = 0.35 if nxt is not None else 0.25
            if hd < tol and (abs(dy) < 0.6 or (st.kind == "drop" and dy > -0.6 and (s.on_ground or s.in_water))):
                if st.kind in ("drop", "dig_down") and not (s.on_ground or s.in_water):
                    c.forward = False
                    continue
                if nxt is None:
                    c.forward = c.sprint = c.jump = False
                return True
            if hd < best - 0.05:
                best = hd
                stall = 0
            else:
                stall += 1
            if stall > 40 or time.time() - start > limit:
                c.clear()
                return False
            s.yaw = math.degrees(math.atan2(-dx, dz))
            s.pitch = 0.0
            c.forward = hd > 0.15
            c.sprint = (straight or st.kind == "gap_jump") and hd > 1.0 and not s.in_water and self.food > 6
            if st.kind == "gap_jump":
                # Sprint-jump is only allowed for a validated, previously
                # mapped, single-cell lava gap with a solid landing pad.
                c.jump = bool(s.on_ground and not jumped and hd > 1.0)
                if c.jump:
                    jumped = True
            elif s.in_water:
                c.jump = dy > -0.4 or s.collided_h
            elif st.kind in ("jump",) and dy > 0.4:
                c.jump = s.on_ground and hd < 1.4
            elif st.kind == "swim":
                c.jump = dy > 0
            else:
                c.jump = s.on_ground and s.collided_h and dy > -0.5
            if s.on_climbable and dy > 0.5:
                c.jump = True

    def _pillar(self, st) -> bool:
        item = self._scaffold_item()
        if not item:
            return False
        bx, by, bz = st.place
        # centre on the block
        self.controls.clear()
        self._center_on(bx, bz)
        self.equip(item)
        self.look(self.state.yaw, 90.0)
        self._wait(1)
        self.controls.jump = True
        ok = False
        for _ in range(14):
            self._wait(1)
            if self.state.y >= by + 1.0:
                self.controls.jump = False
                self.send("use_item_on", Writer().varint(0).position(bx, by - 1, bz).varint(1).float(0.5)
                          .float(1.0).float(0.5).bool(False).bool(False).varint(self._next_seq()))
                self.swing()
                ok = True
                break
        self.controls.jump = False
        for _ in range(20):
            self._wait(1)
            if self.state.on_ground:
                break
        s = self.world.get(bx, by, bz)
        return ok and s is not None and bool(self.gd.has_collision[s])

    def _bridge(self, st) -> bool:
        item = self._scaffold_item()
        if not item:
            return False
        tx, ty, tz = st.place
        bx, by, bz = self.block_pos
        ref = (bx, by - 1, bz)
        d = (tx - ref[0], ty - ref[1], tz - ref[2])
        face = FACE_OF.get(d)
        if face is None:
            r = self._find_reference((tx, ty, tz))
            if not r:
                return False
            ref, face = r
        self.controls.clear()
        self.controls.sneak = True
        # shuffle to the edge so the side face is in reach & view
        self.state.yaw = math.degrees(math.atan2(-(tx + 0.5 - self.state.x), (tz + 0.5 - self.state.z)))
        self.controls.forward = True
        for _ in range(10):
            self._wait(1)
        self.controls.forward = False
        try:
            self.place_block(ref, face, item, sneak=True)
        except ActionError:
            self.controls.sneak = False
            return False
        # Never step onto a claimed bridge until a solid block is visible
        # there. This is crucial when bridging over lava, where a rejected
        # placement would otherwise cause a fatal fall.
        placed = False
        for _ in range(14):
            self._wait(1)
            state = self.world.get(tx, ty, tz)
            if state is not None and self.gd.has_collision[state]:
                placed = True
                break
        self.controls.sneak = False
        return placed

    def _center_on(self, bx, bz, timeout_ticks=30):
        for _ in range(timeout_ticks):
            dx, dz = bx + 0.5 - self.state.x, bz + 0.5 - self.state.z
            if math.hypot(dx, dz) < 0.15:
                break
            self.state.yaw = math.degrees(math.atan2(-dx, dz))
            self.controls.forward = True
            self.controls.sneak = True
            self._wait(1)
        self.controls.forward = False
        self.controls.sneak = False
        self._wait(2)

    def _scaffold_item(self) -> str | None:
        for name in self.pathfinder.scaffold_items:
            if self.inventory.count(name) > 0:
                return name
        return None

    @action
    def escape_up(self, height: int = 3) -> str:
        """Pillar straight up (zombies/skeleton melee can't reach you 3 blocks up)."""
        from .pathfinder import Step
        if not self._scaffold_item():
            raise ActionError("no blocks to pillar with")
        up = 0
        for _ in range(height):
            x, y, z = self.block_pos
            if not self.pathfinder.passable(x, y + 2, z):
                try:
                    self.dig((x, y + 2, z))
                except ActionError:
                    break
            if not self._scaffold_item() or not self._pillar(Step(x, y + 1, z, "pillar", place=(x, y, z))):
                break
            up += 1
        return f"pillared up {up} blocks"

    @action
    def hide_underground(self, until_day: bool = True, max_wait: float = 600) -> str:
        """Night-time safety: dig a 3-deep hole, seal it at ground level, wait (for morning), climb back out."""
        x, y, z = self.block_pos
        hole = [(x, y - 1, z), (x, y - 2, z), (x, y - 3, z)]
        for b in hole:
            if self.pathfinder.dig_cost(*b) is None:
                raise ActionError("unsafe ground for a hole here (liquid/unbreakable)")
        if not self.pathfinder.standable(x, y - 4, z):
            raise ActionError("no solid floor under the hole")
        # the seal goes at y-1; it needs a solid side neighbour to be placed against
        if not any(self.pathfinder.standable(x + dx, y - 1, z + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            raise ActionError("hole would have no wall to seal against")
        hand_diggable = all(not self.gd.requires_tool(self.world.name_at(*b) or "stone") for b in hole[:1])
        if not self._scaffold_item() and not hand_diggable:
            raise ActionError("no block to seal the hole")
        self.controls.clear()
        self._center_on(x, z)
        for b in hole:
            self.dig(b)
        self._wait(6)       # pick up the dug blocks (dirt etc. becomes the seal)
        item = self._scaffold_item()
        if not item:
            raise ActionError("no block to seal the hole")
        for _ in range(40):
            self._wait(1)
            if self.state.on_ground and self.state.y < y - 2.5:
                break
        self.place_at((x, y - 1, z), item)
        end = time.time() + max_wait
        while time.time() < end:
            self._wait(20)
            if not until_day or (self.time_phase() == "day" and self.time_of_day > 200):
                break
        out = self.escape_up(3)
        return f"hid underground until {self.time_phase()}; {out}"

    def nearest_dry_land(self, radius: int = 16):
        """Closest cell (feet position) where we can stand out of water/lava, or None."""
        bx, by, bz = self.block_pos
        pf = self.pathfinder
        best, bd = None, 1e9
        for dy in (1, 0, 2, -1, 3, -2):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    d = dx * dx + dz * dz + dy * dy * 4
                    if d >= bd:
                        continue
                    x, y, z = bx + dx, by + dy, bz + dz
                    if pf.can_stand(x, y, z) and not pf.water(x, y, z) and not pf.lava(x, y, z) \
                            and not pf.water(x, y - 1, z):
                        best, bd = (x, y, z), d
        return best

    @action
    def escape_liquid(self) -> str:
        """Get out of water or lava onto the nearest dry land (swims; digs/bridges if needed)."""
        land = self.nearest_dry_land(16) or self.nearest_dry_land(32)
        if land is None:
            # nothing nearby: swim up and head in any direction until land appears
            self.controls.jump = True
            self._wait(20)
            self.controls.jump = False
            raise ActionError("no dry land within 32 blocks")
        self.goto(GoalBlock(*land), timeout=60)
        return f"got out of the liquid at {land}"

    @action
    def follow(self, e, distance: float = 3.0, duration: float = 60) -> str:
        end = time.time() + duration
        while time.time() < end:
            self._check()
            if e.id not in self.entities.by_id:
                return "target lost"
            if e.distance_to(*self.position) > distance + 1:
                try:
                    self.goto(GoalNear(e.x, e.y, e.z, distance), timeout=10, max_replans=3)
                except ActionError:
                    pass
            else:
                self.look_at_entity(e, wait=False)
                self._wait(5)
        return "done following"

    # ================================================================== gathering
    def nearby_items(self, radius=8.0, names=None):
        def ok(e):
            if e.type != "item":
                return False
            if names and e.item and e.item.name not in names:
                return False
            return time.time() - e.spawned_at > 0.25
        return self.entities.within(self.position, radius, ok)

    @action
    def collect_items(self, radius: float = 8.0, timeout: float = 15.0, names=None) -> int:
        end = time.time() + timeout
        picked = 0
        tried: set[int] = set()
        while time.time() < end:
            items = [e for e in self.nearby_items(radius, names) if e.id not in tried]
            if not items:
                break
            e = items[0]
            tried.add(e.id)
            try:
                self.goto(GoalNear(e.x, e.y, e.z, 0.8), timeout=min(10, end - time.time()), max_replans=4)
            except ActionError:
                continue
            for _ in range(8):
                if e.id not in self.entities.by_id:
                    picked += 1
                    break
                self._wait(1)
        return picked

    def _reachable(self, pos) -> bool:
        """Fast check (<~0.3 s) that a block can be mined: not touching lava/water, and a path exists."""
        x, y, z = pos
        for dx, dy, dz in ((0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0)):
            s = self.world.get(x + dx, y + dy, z + dz)
            if s is not None and self.gd.is_lava[s]:
                return False
        if self.distance_to_block(pos) <= REACH - 0.3:
            return True
        res = self.pathfinder.find(self._start_node(), GoalLookAt(x, y, z, reach=4.0), max_nodes=4000, timeout=0.3)
        return res.status == "success" or (res.status == "timeout" and len(res.steps) > 3)

    def find_blocks(self, names, radius: float = 64, count: int = 16):
        if isinstance(names, str):
            names = [names]
        states = self.gd.states_of(*names)
        return self.world.find_blocks(self.eye, states, radius, count)

    @action
    def collect_block(self, names, count: int = 1, radius: float = 96, timeout: float = 300) -> str:
        """Mine `count` of a block type (e.g. "oak_log", "#logs", "iron_ore") and pick up the drops."""
        if isinstance(names, str):
            names = [names]
        states = self.gd.states_of(*names)
        if len(states) == 0:
            raise ActionError(f"unknown block {names}")
        block_names = {self.gd.block_name(int(s)) for s in states}
        drops = set()
        for b in block_names:
            drops |= self.gd.block_drops.get(b, {b})
        drops = drops or block_names
        start_count = self.inventory.count(drops)
        mined = 0
        failures = 0
        end = time.time() + timeout
        while time.time() < end and self.inventory.count(drops) - start_count < count and mined < count * 2 + 6:
            self._check()
            if self.inventory.empty_slots() == 0:
                return f"inventory full after mining {mined}"
            cands = self.world.find_blocks(self.eye, states, radius, 40)
            now = time.time()
            cands = [p for p in cands if now - self.unreachable.get(p, 0) > 120]
            if not cands:
                break
            # prefer blocks we can reach without much climbing
            ex, ey, ez = self.eye
            cands.sort(key=lambda p: math.dist((ex, ey, ez), (p[0] + 0.5, p[1] + 0.5, p[2] + 0.5))
                       + max(0, p[1] - ey) * 1.5)
            target = None
            for p in cands[:8]:                      # instant reachability check, nearest first
                if self._reachable(p):
                    target = p
                    break
                self.unreachable[p] = time.time()
            if target is None:
                continue
            bname = self.gd.block_name(self.world.get(*target))
            if self.gd.requires_tool(bname):
                names_inv = [it.name for it in self.inventory.items()]
                _, _, correct = self.gd.best_tool_for(bname, names_inv)
                if not correct:
                    tier = next((t.replace("needs_", "").replace("_tool", "") for t in
                                 ("needs_diamond_tool", "needs_iron_tool", "needs_stone_tool")
                                 if bname in self.gd.block_tag(t)), "wooden")
                    raise ActionError(f"need a {tier} (or better) pickaxe/tool to harvest {bname}")
            try:
                if self.distance_to_block(target) > REACH - 0.3:
                    self.goto(GoalLookAt(*target, reach=4.0), timeout=60, max_replans=12)
                self.dig(target)
                mined += 1
                failures = 0
                self._wait(2)
                self.collect_items(radius=7, timeout=8)
            except ActionError as e:
                if "stopped" in str(e) or "died" in str(e) or "disconnected" in str(e):
                    raise
                self.unreachable[target] = time.time()
                failures += 1
                log.debug("collect_block: %s", e)
                if failures > 8:
                    break
        got = self.inventory.count(drops) - start_count
        if mined == 0:
            raise ActionError(f"could not find/reach any {', '.join(sorted(block_names)[:5])} within {radius:.0f} blocks")
        return f"mined {mined} block(s); inventory gained {got} of {', '.join(sorted(drops)[:5])}"

    # ================================================================== crafting
    def _plan_recipe(self, recipe, crafts: int):
        """Pick concrete items for each ingredient. Returns {options_tuple: item} or None."""
        counts = self.inventory.counts()
        choice = {}
        used: dict[str, int] = {}
        for opts, n in recipe.ingredient_counts():
            need = n * crafts
            pick = None
            for o in sorted(opts, key=lambda o: -counts.get(o, 0)):
                if counts.get(o, 0) - used.get(o, 0) >= need:
                    pick = o
                    break
            if pick is None:
                return None
            used[pick] = used.get(pick, 0) + need
            choice[opts] = pick
        return choice

    def can_craft(self, item: str, count: int = 1):
        best = None
        for r in self.gd.recipes_for(item):
            crafts = math.ceil(count / r.count)
            plan = self._plan_recipe(r, crafts)
            if plan is not None:
                if best is None or (best[0].needs_table and not r.needs_table):
                    best = (r, plan, crafts)
        return best

    def missing_for(self, item: str, count: int = 1) -> dict[str, int]:
        """Cheapest missing ingredients for crafting item (by recipe), as {item_or_tag: count}."""
        best = None
        counts = self.inventory.counts()
        for r in self.gd.recipes_for(item):
            crafts = math.ceil(count / r.count)
            miss = {}
            for opts, n in r.ingredient_counts():
                have = max(counts.get(o, 0) for o in opts)
                if have < n * crafts:
                    key = opts[0] if len(opts) == 1 else "/".join(opts[:4]) + ("..." if len(opts) > 4 else "")
                    miss[key] = n * crafts - have
            if best is None or sum(miss.values()) < sum(best.values()):
                best = miss
        return best or {}

    def find_or_place(self, block: str, radius: float = 24):
        """Nearest block of a type (crafting_table/furnace); places one from inventory if needed."""
        found = self.find_blocks(block, radius, 1)
        if found:
            return found[0]
        if self.inventory.count(block) == 0:
            return None
        bx, by, bz = self.block_pos
        spots = []
        for dx in (-2, -1, 0, 1, 2):
            for dz in (-2, -1, 0, 1, 2):
                if abs(dx) + abs(dz) < 2:
                    continue
                for dy in (0, 1, -1):
                    p = (bx + dx, by + dy, bz + dz)
                    s = self.world.get(*p)
                    below = self.world.get(p[0], p[1] - 1, p[2])
                    if s is not None and (self.gd.is_air[s] or self.gd.is_replaceable[s]) and \
                            below is not None and self.gd.standable[below]:
                        spots.append(p)
        spots.sort(key=lambda p: math.dist(p, (bx, by, bz)))
        for p in spots[:6]:
            try:
                self.place_at(p, block)
                return p
            except ActionError:
                continue
        return None

    @action
    def craft(self, item: str, count: int = 1) -> str:
        plan = self.can_craft(item, count)
        if plan is None:
            if not self.gd.recipes_for(item):
                raise ActionError(f"{item} has no crafting recipe")
            raise ActionError(f"missing ingredients for {item}: {self.missing_for(item, count)}")
        recipe, choice, crafts = plan
        before = self.inventory.count(item)
        if recipe.needs_table:
            table = self.find_or_place("crafting_table")
            if table is None:
                raise ActionError("need a crafting table (craft one from 4 planks)")
            win = self.open_container(table)
            if win.type != "crafting":
                self.close_window()
                raise ActionError(f"opened {win.type}, expected crafting table")
            gw = 3
        else:
            if self.inventory.window:
                self.close_window()
            gw = 2
        try:
            remaining = crafts
            stack_limit = min([self.gd.max_stack(i) for i in choice.values()] + [64])
            while remaining > 0:
                self._check()
                k = min(remaining, stack_limit, max(1, self.gd.max_stack(item) // max(1, recipe.count)))
                cells = self._recipe_cells(recipe, choice, gw)
                for slot, name in cells:
                    self._put_into(slot, name, k)
                v = self.inventory.version
                self.click(0, 0, QUICK_MOVE)
                self.inventory.wait_change(v, 0.5)
                self._clear_grid(gw)
                remaining -= k
        finally:
            if gw == 3:
                self.close_window()
        got = self.inventory.count(item) - before
        if got <= 0:
            raise ActionError(f"crafting {item} failed (grid did not produce a result)")
        self.events.emit("crafted", item, got)
        return f"crafted {got} {item}"

    def _recipe_cells(self, recipe, choice, gw):
        cells = []
        if recipe.kind == "shaped":
            for r, row in enumerate(recipe.grid):
                for c, opts in enumerate(row):
                    if opts:
                        cells.append((1 + r * gw + c, choice[tuple(opts)]))
        else:
            for i, opts in enumerate(recipe.ingredients):
                cells.append((1 + i, choice[tuple(opts)]))
        return cells

    def _put_into(self, target_slot: int, name: str, k: int):
        """Move exactly k items called `name` from the player inventory into a window slot."""
        inv = self.inventory
        placed = 0
        guard = 0
        while placed < k and guard < 200:
            guard += 1
            if not inv.cursor or inv.cursor.name != name:
                if inv.cursor:
                    self._drop_cursor_home()
                src = inv.find(lambda it: it.name == name)
                if src is None:
                    raise ActionError(f"ran out of {name}")
                src_i = self._active_index(src)
                self.click(src_i, 0, PICKUP)
                if not inv.cursor:
                    raise ActionError(f"could not pick up {name}")
            if inv.cursor.count <= k - placed:
                n = inv.cursor.count
                self.click(target_slot, 0, PICKUP)
                placed += n
            else:
                self.click(target_slot, 1, PICKUP)
                placed += 1
        if inv.cursor:
            self._drop_cursor_home()

    def _drop_cursor_home(self):
        """Put the cursor stack back into the player inventory."""
        inv = self.inventory
        if not inv.cursor:
            return
        name = inv.cursor.name
        start = inv.window.player_start if inv.window else 9
        slots = inv.active_slots()
        n_player = 36
        # merge into existing stack, else first empty slot
        for i in range(start, start + n_player):
            it = slots[i]
            if it and it.name == name and it.count < self.gd.max_stack(name):
                self.click(i, 0, PICKUP)
                if not inv.cursor:
                    return
        for i in range(start, start + n_player):
            if slots[i] is None:
                self.click(i, 0, PICKUP)
                return

    def _clear_grid(self, gw):
        slots = self.inventory.active_slots()
        for s in range(1, 1 + gw * gw):
            if s < len(slots) and slots[s]:
                self.click(s, 0, QUICK_MOVE)

    # ================================================================== furnace
    @action
    def smelt(self, output: str, count: int = 1, timeout: float | None = None) -> str:
        recipes = self.gd.smelt_recipes_for(output)
        if not recipes:
            raise ActionError(f"{output} cannot be smelted")
        counts = self.inventory.counts()
        inp = None
        for r in recipes:
            for o in r.ingredients[0]:
                if counts.get(o, 0) > 0:
                    inp = o
                    break
            if inp:
                break
        if not inp:
            raise ActionError(f"nothing to smelt into {output} (needs {recipes[0].ingredients[0][:4]})")
        count = min(count, counts[inp])
        fuels = sorted(((n, _fuel_value(self.gd, n)) for n in counts if _fuel_value(self.gd, n) > 0 and n != inp),
                       key=lambda t: -t[1])
        if not fuels:
            raise ActionError("no fuel (coal, charcoal, planks, logs...)")
        furnace = self.find_or_place("furnace")
        if furnace is None:
            raise ActionError("need a furnace (craft one from 8 cobblestone)")
        before = self.inventory.count(output)
        win = self.open_container(furnace)
        try:
            self._put_into(0, inp, count)
            need_fuel = count
            fuel_slot_item = win.slots[1]
            have = (fuel_slot_item.count * _fuel_value(self.gd, fuel_slot_item.name)) if fuel_slot_item else 0
            for fname, val in fuels:
                if have >= need_fuel:
                    break
                n = min(self.inventory.count(fname), math.ceil((need_fuel - have) / val))
                cur = win.slots[1]
                if cur and cur.name != fname:
                    continue
                self._put_into(1, fname, n)
                have += n * val
            took = 0
            deadline = time.time() + (timeout or count * 10.5 + 15)
            stalled_since = None
            while took < count and time.time() < deadline:
                self._check()
                out = win.slots[2]
                if out:
                    self.click(2, 0, QUICK_MOVE)
                    took = self.inventory.count(output) - before
                elif win.slots[0] is None:
                    break
                # fuel check: property 0 = burn time left; empty fuel slot + not burning = stalled
                burning = win.properties.get(0, 0) > 0
                if not burning and win.slots[1] is None and win.slots[0] is not None:
                    fuel = next((n for n, v in sorted(((n, _fuel_value(self.gd, n)) for n in self.inventory.counts()),
                                                      key=lambda t: -t[1]) if v > 0 and n != inp), None)
                    if fuel:
                        left = win.slots[0].count if win.slots[0] else 0
                        n = min(self.inventory.count(fuel), math.ceil(left / _fuel_value(self.gd, fuel)))
                        self._put_into(1, fuel, max(1, n))
                        stalled_since = None
                    else:
                        stalled_since = stalled_since or time.time()
                        if time.time() - stalled_since > 3:
                            raise ActionError(f"furnace ran out of fuel with {win.slots[0].count} {inp} left; "
                                              f"get coal/charcoal/planks and smelt again")
                time.sleep(0.5)
        finally:
            self.close_window()
        got = self.inventory.count(output) - before
        if got <= 0:
            raise ActionError(f"smelting produced nothing (input {inp})")
        return f"smelted {got} {output} from {inp}"

    # ================================================================== containers
    @action
    def deposit(self, pos, names=None, keep_tools: bool = True) -> str:
        win = self.open_container(pos)
        moved = 0
        try:
            for i in range(win.player_start, win.player_start + 36):
                it = win.slots[i]
                if not it:
                    continue
                if names and it.name not in names:
                    continue
                if keep_tools and (self.gd.tool_rules(it.name) or self.gd.food(it.name)) and not names:
                    continue
                self.click(i, 0, QUICK_MOVE)
                moved += it.count
        finally:
            self.close_window()
        return f"deposited {moved} items"

    @action
    def withdraw(self, pos, name: str, count: int = 64) -> str:
        win = self.open_container(pos)
        got = 0
        try:
            for i, it in win.container_items():
                if it.name == name and got < count:
                    self.click(i, 0, QUICK_MOVE)
                    got += it.count
        finally:
            self.close_window()
        return f"took {got} {name}"

    def container_contents(self, pos) -> dict[str, int]:
        win = self.open_container(pos)
        out: dict[str, int] = {}
        for _, it in win.container_items():
            out[it.name] = out.get(it.name, 0) + it.count
        self.close_window()
        return out

    # ================================================================== misc
    @action
    def sleep_in_bed(self) -> str:
        beds = self.find_blocks("#beds", 32, 1)
        if not beds:
            raise ActionError("no bed nearby")
        self.activate_block(beds[0])
        self._wait(40)
        return "tried to sleep (works only at night / thunderstorm and with no monsters nearby)"

    @action
    def shelter(self) -> str:
        """Emergency shelter: wall in the 2-block body space and put a roof on (needs ~10 blocks)."""
        item = self._scaffold_item()
        if not item or self.inventory.count(item) < 6:
            raise ActionError("need at least 6 scaffold blocks (dirt/cobblestone)")
        bx, by, bz = self.block_pos
        placed = 0
        spots = [(bx + dx, by + dy, bz + dz) for dy in (0, 1) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        spots.append((bx, by + 2, bz))
        for p in spots:
            s = self.world.get(*p)
            if s is not None and (self.gd.is_air[s] or self.gd.is_replaceable[s]):
                item = self._scaffold_item()
                if not item:
                    break
                try:
                    ref = self._find_reference(p)
                    if ref:
                        self.place_block(ref[0], ref[1], item)
                        placed += 1
                except ActionError:
                    continue
        return f"shelter built ({placed} blocks placed)"


def _planner(self):
    from .planner import Planner
    if not hasattr(self, "_planner_obj"):
        self._planner_obj = Planner(self)
    return self._planner_obj


@action
def _obtain(self, item: str, count: int = 1, timeout: float = 900) -> str:
    """Get `count` of any item: crafts, smelts, mines (getting the right tools first) or hunts."""
    return self.planner.obtain(item, count, deadline=time.time() + timeout)


ActionsMixin.planner = property(_planner)
ActionsMixin.obtain = _obtain
