"""Entity tracking."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

from ..protocol.buffer import ProtocolError, Reader

# (width, height) for common entity types; everything else defaults to (0.6, 1.8)
DIMENSIONS = {
    "player": (0.6, 1.8), "zombie": (0.6, 1.95), "husk": (0.6, 1.95), "drowned": (0.6, 1.95),
    "zombie_villager": (0.6, 1.95), "skeleton": (0.6, 1.99), "stray": (0.6, 1.99), "bogged": (0.6, 1.99),
    "wither_skeleton": (0.7, 2.4), "creeper": (0.6, 1.7), "spider": (1.4, 0.9), "cave_spider": (0.7, 0.5),
    "enderman": (0.6, 2.9), "witch": (0.6, 1.95), "slime": (0.52, 0.52), "magma_cube": (0.52, 0.52),
    "blaze": (0.6, 1.8), "ghast": (4.0, 4.0), "piglin": (0.6, 1.95), "zombified_piglin": (0.6, 1.95),
    "piglin_brute": (0.6, 1.95), "hoglin": (1.4, 1.4), "zoglin": (1.4, 1.4), "pillager": (0.6, 1.95),
    "vindicator": (0.6, 1.95), "evoker": (0.6, 1.95), "ravager": (1.95, 2.2), "phantom": (0.9, 0.5),
    "silverfish": (0.4, 0.3), "endermite": (0.4, 0.3), "guardian": (0.85, 0.85), "elder_guardian": (2.0, 2.0),
    "shulker": (1.0, 1.0), "warden": (0.9, 2.9), "breeze": (0.6, 1.77), "ender_dragon": (16.0, 8.0),
    "cow": (0.9, 1.4), "mooshroom": (0.9, 1.4), "pig": (0.9, 0.9), "sheep": (0.9, 1.3), "chicken": (0.4, 0.7),
    "rabbit": (0.4, 0.5), "horse": (1.3965, 1.6), "donkey": (1.3965, 1.5), "mule": (1.3965, 1.6),
    "llama": (0.9, 1.87), "goat": (0.9, 1.3), "wolf": (0.6, 0.85), "cat": (0.6, 0.7), "fox": (0.6, 0.7),
    "villager": (0.6, 1.95), "wandering_trader": (0.6, 1.95), "iron_golem": (1.4, 2.7),
    "snow_golem": (0.7, 1.9), "bee": (0.7, 0.6), "squid": (0.8, 0.8), "glow_squid": (0.8, 0.8),
    "cod": (0.5, 0.3), "salmon": (0.7, 0.4), "turtle": (1.2, 0.4), "polar_bear": (1.4, 1.4),
    "panda": (1.3, 1.25), "item": (0.25, 0.25), "experience_orb": (0.5, 0.5), "arrow": (0.5, 0.5),
    "boat": (1.375, 0.5625), "minecart": (0.98, 0.7), "armor_stand": (0.5, 1.975), "bat": (0.5, 0.9),
    "parrot": (0.5, 0.9), "axolotl": (0.75, 0.42), "frog": (0.5, 0.5), "camel": (1.7, 2.375),
    "sniffer": (1.9, 1.75), "armadillo": (0.7, 0.65), "strider": (0.9, 1.7), "allay": (0.35, 0.6),
}

HOSTILE = {"zombie", "husk", "drowned", "zombie_villager", "skeleton", "stray", "bogged", "wither_skeleton",
           "creeper", "spider", "cave_spider", "witch", "slime", "magma_cube", "blaze", "ghast", "piglin_brute",
           "hoglin", "zoglin", "pillager", "vindicator", "evoker", "ravager", "phantom", "silverfish",
           "endermite", "guardian", "elder_guardian", "shulker", "warden", "breeze", "vex", "ender_dragon",
           "wither", "creaking"}
NEUTRAL_HOSTILE_IF_PROVOKED = {"enderman", "zombified_piglin", "piglin", "wolf", "iron_golem", "bee",
                               "polar_bear", "llama", "panda", "goat", "dolphin"}
FOOD_ANIMALS = {"cow", "pig", "sheep", "chicken", "rabbit", "mooshroom", "cod", "salmon", "goat"}


@dataclass
class Entity:
    id: int
    uuid: str
    type: str
    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0
    head_yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    on_ground: bool = False
    data: int = 0
    name: str | None = None          # players
    item: object = None              # dropped items: Item
    health: float | None = None
    metadata: dict = field(default_factory=dict)
    spawned_at: float = field(default_factory=time.time)
    # protocol base for relative moves (vanilla tracks this separately from the visual position)
    bx: float = 0.0
    by: float = 0.0
    bz: float = 0.0

    @property
    def width(self) -> float:
        return DIMENSIONS.get(self.type, (0.6, 1.8))[0]

    @property
    def height(self) -> float:
        return DIMENSIONS.get(self.type, (0.6, 1.8))[1]

    @property
    def pos(self):
        return (self.x, self.y, self.z)

    def distance_to(self, x, y, z) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z) ** 2)

    @property
    def hostile(self) -> bool:
        return self.type in HOSTILE

    def describe(self, origin=None) -> str:
        s = self.name or (f"item:{self.item!r}" if self.type == "item" and self.item else self.type)
        if origin:
            s += f" @ {self.x:.0f},{self.y:.0f},{self.z:.0f} ({self.distance_to(*origin):.1f}m)"
        if self.health is not None:
            s += f" hp={self.health:.0f}"
        return s


class Entities:
    def __init__(self, gd, item_codec):
        self.gd = gd
        self.codec = item_codec
        self.by_id: dict[int, Entity] = {}
        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.by_id.clear()

    def add(self, r: Reader, players: dict) -> Entity:
        eid = r.varint()
        uid = str(r.uuid())
        etype = self.gd.entity_types.get(r.varint(), "unknown")
        x, y, z = r.double(), r.double(), r.double()
        vx, vy, vz = r.lpvec3()
        pitch, yaw, head = r.angle(), r.angle(), r.angle()
        data = r.varint()
        e = Entity(eid, uid, etype, x, y, z, yaw, pitch, head, vx, vy, vz, False, data)
        e.bx, e.by, e.bz = x, y, z
        if etype == "player":
            e.name = players.get(uid)
        with self.lock:
            self.by_id[eid] = e
        return e

    def get(self, eid: int) -> Entity | None:
        return self.by_id.get(eid)

    def remove(self, ids):
        with self.lock:
            for i in ids:
                self.by_id.pop(i, None)

    @staticmethod
    def _read_deltas(r: Reader):
        props = r.varint()
        on_ground = bool(props & 1)
        steps = props >> 1
        if steps == 0:
            return [(r.short() / 4096.0, r.short() / 4096.0, r.short() / 4096.0)], on_ground
        out = []
        for _ in range(steps):
            r.varint()
            out.append((r.short() / 4096.0, r.short() / 4096.0, r.short() / 4096.0))
        return out, on_ground

    def move(self, r: Reader, rotation: bool):
        eid = r.varint()
        deltas, on_ground = self._read_deltas(r)
        e = self.by_id.get(eid)
        yaw = pitch = None
        if rotation:
            yaw, pitch = r.angle(), r.angle()
        if not e:
            return None
        for dx, dy, dz in deltas:
            e.bx += dx
            e.by += dy
            e.bz += dz
        e.x, e.y, e.z = e.bx, e.by, e.bz
        e.on_ground = on_ground
        if rotation:
            e.yaw, e.pitch = yaw, pitch
        return e

    def rotate(self, r: Reader):
        e = self.by_id.get(r.varint())
        og = r.bool()
        yaw, pitch = r.angle(), r.angle()
        if e:
            e.on_ground, e.yaw, e.pitch = og, yaw, pitch

    def position_sync(self, r: Reader):
        """entity_position_sync (wiki: Teleport Entity)."""
        e = self.by_id.get(r.varint())
        path_type = r.varint()
        if path_type == 0:
            x, y, z = r.double(), r.double(), r.double()
        else:
            x = y = z = None
            for _ in range(r.varint()):
                x, y, z = r.double(), r.double(), r.double()
                r.varint()
        yaw, pitch = r.float(), r.float()
        og = r.bool()
        if e and x is not None:
            e.x, e.y, e.z = x, y, z
            e.bx, e.by, e.bz = x, y, z
            e.yaw, e.pitch, e.on_ground = yaw, pitch, og

    def teleport(self, r: Reader):
        """teleport_entity (wiki: Synchronize Vehicle Position)."""
        e = self.by_id.get(r.varint())
        x, y, z = r.double(), r.double(), r.double()
        r.double(), r.double(), r.double()
        yaw, pitch = r.float(), r.float()
        flags = r.int()
        if e:
            e.x = e.x + x if flags & 1 else x
            e.y = e.y + y if flags & 2 else y
            e.z = e.z + z if flags & 4 else z
            e.bx, e.by, e.bz = e.x, e.y, e.z
            e.yaw = e.yaw + yaw if flags & 8 else yaw
            e.pitch = e.pitch + pitch if flags & 16 else pitch

    def velocity(self, r: Reader):
        eid = r.varint()
        v = r.lpvec3()
        e = self.by_id.get(eid)
        if e:
            e.vx, e.vy, e.vz = v
        return eid, v

    def metadata(self, r: Reader):
        e = self.by_id.get(r.varint())
        if not e:
            return
        try:
            while True:
                idx = r.ubyte()
                if idx == 0xFF:
                    break
                t = r.varint()
                if t == 0:
                    v = r.byte()
                elif t in (1, 12, 14, 15, 19, 20):
                    v = r.varint()
                elif t == 2:
                    v = r.varlong()
                elif t == 3:
                    v = r.float()
                elif t == 4:
                    v = r.string()
                elif t == 5:
                    v = r.nbt()
                elif t == 6:
                    v = r.optional(r.nbt)
                elif t == 7:
                    v = self.codec.read_slot(r)
                elif t == 8:
                    v = r.bool()
                elif t == 9:
                    v = (r.float(), r.float(), r.float())
                elif t == 10:
                    v = r.position()
                elif t == 11:
                    v = r.optional(r.position)
                elif t == 13:
                    v = r.optional(r.uuid)
                elif t == 18:
                    v = (r.varint(), r.varint(), r.varint())
                else:
                    break  # unknown type: stop (remaining entries are not needed)
                e.metadata[idx] = v
                if e.type == "item" and idx == 8:
                    e.item = v
                elif idx == 9 and t == 3 and e.type not in ("item", "experience_orb"):
                    e.health = v
        except ProtocolError:
            pass

    # ------------------------------------------------------------------ queries
    def nearest(self, origin, predicate=None, max_dist: float = 64) -> Entity | None:
        best, bd = None, max_dist
        with self.lock:
            ents = list(self.by_id.values())
        for e in ents:
            if predicate and not predicate(e):
                continue
            d = e.distance_to(*origin)
            if d < bd:
                best, bd = e, d
        return best

    def within(self, origin, radius: float, predicate=None) -> list[Entity]:
        with self.lock:
            ents = list(self.by_id.values())
        out = [e for e in ents if e.distance_to(*origin) <= radius and (not predicate or predicate(e))]
        out.sort(key=lambda e: e.distance_to(*origin))
        return out
