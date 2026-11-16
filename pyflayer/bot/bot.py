"""The Bot: a headless Minecraft 26.x client with a mineflayer-like API.

    gd = GameData(ensure_gamedata("26.3"))
    bot = Bot(gd, "PyBot", "127.0.0.1", 25565)
    bot.connect(); bot.wait_spawn()
    bot.chat("hello")
    bot.goto(100, 64, 100)          # pathfinding (actions mixin)
    bot.collect_block("oak_log", 4)
"""
from __future__ import annotations

import collections
import hashlib
import logging
import math
import random
import threading
import time
import uuid as _uuid

from ..data.gamedata import GameData
from ..protocol.buffer import ProtocolError, Reader, Writer
from ..protocol.connection import Connection, PacketTable
from ..protocol.items import ItemCodec, UnknownComponent
from ..protocol.text import to_plain
from ..world.world import World
from .actions import ActionsMixin
from .entities import Entities
from .inventory import Inventory
from .physics import Controls, Physics, PlayerState

log = logging.getLogger("pyflayer.bot")

EYE_HEIGHT = 1.62


def offline_uuid(name: str) -> _uuid.UUID:
    h = bytearray(hashlib.md5(("OfflinePlayer:" + name).encode()).digest())
    h[6] = (h[6] & 0x0F) | 0x30
    h[8] = (h[8] & 0x3F) | 0x80
    return _uuid.UUID(bytes=bytes(h))


class EventBus:
    def __init__(self):
        self._handlers: dict[str, list] = collections.defaultdict(list)
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._last: dict[str, tuple] = {}
        self._counter: dict[str, int] = collections.defaultdict(int)

    def on(self, name, fn):
        with self._lock:
            self._handlers[name].append(fn)
        return fn

    def off(self, name, fn):
        with self._lock:
            if fn in self._handlers[name]:
                self._handlers[name].remove(fn)

    def emit(self, name, *args):
        with self._lock:
            hs = list(self._handlers.get(name, ()))
        for h in hs:
            try:
                h(*args)
            except Exception:  # noqa: BLE001
                log.exception("event handler for %s failed", name)
        with self._cond:
            self._last[name] = args
            self._counter[name] += 1
            self._cond.notify_all()

    def wait_for(self, name, timeout=10.0, predicate=None):
        """Wait for the *next* emission of `name` (optionally matching predicate). Returns args or None."""
        end = time.time() + timeout
        with self._cond:
            seen = self._counter[name]
            while True:
                if self._counter[name] != seen:
                    seen = self._counter[name]
                    args = self._last[name]
                    if predicate is None or predicate(*args):
                        return args
                left = end - time.time()
                if left <= 0:
                    return None
                self._cond.wait(left)


class Bot(ActionsMixin):
    def __init__(self, gd: GameData, username: str = "PyBot", host: str = "127.0.0.1", port: int = 25565,
                 view_distance: int = 7, auto_respawn: bool = True):
        self.gd = gd
        self.username = username
        self.host = host
        self.port = port
        self.view_distance = view_distance
        self.auto_respawn = auto_respawn
        self.uuid = offline_uuid(username)
        self.table = PacketTable(gd.packets)
        self.codec = ItemCodec(gd.items, gd.component_types)
        self.events = EventBus()
        self.world = World(gd)
        self.entities = Entities(gd, self.codec)
        self.inventory = Inventory(self)
        self.physics = Physics(self.world, gd)
        self.state = PlayerState()
        self.controls = Controls()
        self.conn: Connection | None = None
        self.players: dict[str, str] = {}   # uuid -> name
        self.registries: dict[str, list[str]] = {}
        self.dimension_types: dict[str, tuple[int, int]] = {}
        self.entity_id = -1
        self.health = 20.0
        self.food = 20
        self.saturation = 5.0
        self.xp_level = 0
        self.gamemode = 0
        self.dimension = "overworld"
        self.spawn_point = None
        self.time_of_day = 0
        self.world_age = 0
        self.dead = False
        self.spawned = threading.Event()
        self.disconnected = threading.Event()
        self.disconnect_reason = ""
        self.seq = 0
        self.chat_log: collections.deque = collections.deque(maxlen=200)
        self.physics_enabled = True
        self.tick_hooks: list = []
        self.tick_count = 0
        self._tick_thread: threading.Thread | None = None
        self._sent = None                 # last sent movement (x,y,z,yaw,pitch,flags)
        self._ticks_since_pos = 0
        self._sprinting = False
        self._input_flags = 0
        self._loaded_sent = False
        self._first_teleport = threading.Event()
        self._force_move = False
        self._last_tick_time = 0.0
        self.tps_measured = 20.0
        self.abilities_flags = 0
        self.lock = threading.RLock()     # serialises high-level actions
        self._init_actions()
        from .combat import Combat
        self.combat = Combat(self)   # 20 TPS reflex fighting

    # ================================================================== connection
    def connect(self, timeout: float = 30):
        self.disconnected.clear()
        self.spawned.clear()
        self._first_teleport.clear()
        self._loaded_sent = False
        self.conn = Connection(self.table, self._on_packet, self._on_close)
        self.conn.connect(self.host, self.port)
        self.conn.state = "handshake"
        self.conn.send("intention", Writer().varint(self.gd.protocol).string(self.host)
                       .ushort(self.port).varint(2))
        self.conn.state = "login"
        self.conn.send("hello", Writer().string(self.username).uuid(self.uuid))
        self.conn.start_reader()
        if not self._tick_thread or not self._tick_thread.is_alive():
            self._tick_thread = threading.Thread(target=self._tick_loop, name="mc-tick", daemon=True)
            self._tick_thread.start()
        log.info("Connecting to %s:%s as %s (protocol %s)", self.host, self.port, self.username, self.gd.protocol)

    def wait_spawn(self, timeout: float = 60) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.spawned.wait(0.5):
                return True
            if self.disconnected.is_set():
                raise ConnectionError(f"disconnected: {self.disconnect_reason}")
        return False

    def wait_chunks(self, timeout: float = 20.0, settle: float = 1.5) -> int:
        """Wait until the initial burst of chunks around the player has arrived."""
        end = time.time() + timeout
        last_n, last_change = -1, time.time()
        while time.time() < end:
            n = len(self.world.columns)
            if n != last_n:
                last_n, last_change = n, time.time()
            elif n > 0 and time.time() - last_change >= settle:
                break
            time.sleep(0.25)
        return len(self.world.columns)

    def disconnect(self, reason="bye"):
        if self.conn:
            self.conn.close(reason)

    def _on_close(self, reason):
        self.disconnect_reason = reason
        self.spawned.clear()
        self.disconnected.set()
        log.warning("Disconnected: %s", reason)
        self.events.emit("end", reason)

    @property
    def connected(self) -> bool:
        return bool(self.conn and not self.conn.closed.is_set())

    def send(self, name: str, w: Writer | None = None):
        if self.conn:
            self.conn.send(name, w)

    # ================================================================== packet dispatch
    def _on_packet(self, state: str, name: str, r: Reader):
        h = getattr(self, f"_{state}_{name.replace('/', '_')}", None)
        if h is None:
            h = getattr(self, f"_any_{name.replace('/', '_')}", None)
        if h is not None:
            h(r)

    # ------------------------------------------------------------------ login
    def _login_login_compression(self, r):
        self.conn.compression = r.varint()

    def _login_login_finished(self, r):
        uid = r.uuid()
        name = r.string()
        log.info("Logged in as %s (%s)", name, uid)
        self.conn.send("login_acknowledged")
        self.conn.state = "configuration"
        self.conn.send("client_information", self._client_info())
        self.conn.send("custom_payload", Writer().string("minecraft:brand").string("vanilla"))

    def _login_login_disconnect(self, r):
        reason = to_plain(r.string())
        self.conn.close("login refused: " + reason)

    def _login_custom_query(self, r):
        mid = r.varint()
        self.conn.send("custom_query_answer", Writer().varint(mid).bool(False))

    def _login_cookie_request(self, r):
        self.conn.send("cookie_response", Writer().string(r.string()).bool(False))

    def _login_hello(self, r):
        self.conn.close("server is in online mode (encryption requested); this bot needs an offline/cracked server")

    def _client_info(self) -> Writer:
        return (Writer().string("en_us").byte(self.view_distance).varint(0).bool(True)
                .ubyte(0x7F).varint(1).bool(False).bool(True).varint(2))

    # ------------------------------------------------------------------ configuration
    def _configuration_select_known_packs(self, r):
        # Claim no packs -> server sends full registry data (we need dimension heights etc.)
        self.conn.send("select_known_packs", Writer().varint(0))

    def _configuration_registry_data(self, r):
        reg = r.string().split(":")[-1]
        names = []
        n = r.varint()
        for _ in range(n):
            ename = r.string()
            data = r.optional(r.nbt)
            names.append(ename)
            if reg == "dimension_type" and isinstance(data, dict):
                self.dimension_types[ename.split(":")[-1]] = (int(data.get("min_y", -64)), int(data.get("height", 384)))
        self.registries[reg] = names
        if reg == "worldgen/biome":
            self.world.set_biomes(names)

    def _configuration_finish_configuration(self, r):
        self.conn.send("finish_configuration")
        self.conn.state = "play"

    def _configuration_keep_alive(self, r):
        self.conn.send("keep_alive", Writer().long(r.long()))

    def _configuration_ping(self, r):
        self.conn.send("pong", Writer().int(r.int()))

    def _configuration_disconnect(self, r):
        self.conn.close("kicked: " + to_plain(r.nbt()))

    def _configuration_code_of_conduct(self, r):
        self.conn.send("accept_code_of_conduct")

    def _configuration_resource_pack_push(self, r):
        self._resource_pack(r)

    def _configuration_cookie_request(self, r):
        self.conn.send("cookie_response", Writer().string(r.string()).bool(False))

    def _resource_pack(self, r):
        pid = r.uuid()
        self.conn.send("resource_pack", Writer().uuid(pid).varint(3))
        self.conn.send("resource_pack", Writer().uuid(pid).varint(0))

    # ------------------------------------------------------------------ play: core
    def _play_login(self, r):
        self.entity_id = r.int()
        r.bool()                                 # hardcore
        r.array(r.string)                        # dimension names
        r.varint()                               # max players
        r.varint()                               # view distance
        r.varint()                               # simulation distance
        r.bool(); r.bool(); r.bool()             # reduced debug, respawn screen, limited crafting
        self._read_spawn_info(r)
        log.info("Joined game: entity %s, dimension %s, gamemode %s", self.entity_id, self.dimension, self.gamemode)
        self.events.emit("login")

    def _read_spawn_info(self, r):
        dim_type_id = r.varint()
        dim_name = r.string()
        r.long()                                 # hashed seed
        self.gamemode = r.varint()
        r.varint()                               # previous gamemode
        r.bool(); r.bool()                       # debug, flat
        if r.bool():
            r.string(); r.position()
        r.varint()                               # portal cooldown
        r.varint()                               # sea level
        types = self.registries.get("dimension_type", [])
        tname = types[dim_type_id].split(":")[-1] if dim_type_id < len(types) else dim_name.split(":")[-1]
        min_y, height = self.dimension_types.get(tname, (-64, 384) if "overworld" in tname else (0, 256))
        self.dimension = dim_name.split(":")[-1]
        self.world.set_dimension(dim_name, min_y, height)
        self.entities.clear()

    def _play_respawn(self, r):
        self._read_spawn_info(r)
        self.dead = False
        self._first_teleport.clear()
        self._loaded_sent = False
        self.events.emit("respawn")

    def _play_start_configuration(self, r):
        self.spawned.clear()
        self.conn.send("configuration_acknowledged")
        self.conn.state = "configuration"

    def _play_keep_alive(self, r):
        self.conn.send("keep_alive", Writer().long(r.long()))

    def _play_ping(self, r):
        self.conn.send("pong", Writer().int(r.int()))

    def _play_disconnect(self, r):
        self.conn.close("kicked: " + to_plain(r.nbt()))

    def _play_resource_pack_push(self, r):
        self._resource_pack(r)

    def _play_player_position(self, r):
        tid = r.varint()
        x, y, z = r.double(), r.double(), r.double()
        vx, vy, vz = r.double(), r.double(), r.double()
        yaw, pitch = r.float(), r.float()
        flags = r.int()
        st = self.state
        st.x = st.x + x if flags & 0x01 else x
        st.y = st.y + y if flags & 0x02 else y
        st.z = st.z + z if flags & 0x04 else z
        st.yaw = st.yaw + yaw if flags & 0x08 else yaw
        st.pitch = st.pitch + pitch if flags & 0x10 else pitch
        st.vx = st.vx + vx if flags & 0x20 else vx
        st.vy = st.vy + vy if flags & 0x40 else vy
        st.vz = st.vz + vz if flags & 0x80 else vz
        st.fall_distance = 0
        self.conn.send("accept_teleportation", Writer().varint(tid).double(st.x).double(st.y).double(st.z)
                       .float(st.yaw).float(st.pitch))
        # the confirming position is sent by the tick loop: 26.x allows one movement packet per client tick
        self._force_move = True
        if not self._first_teleport.is_set():
            self._first_teleport.set()
            if not self._loaded_sent:
                self._loaded_sent = True
                self.conn.send("player_loaded")
            self.spawned.set()
            self.events.emit("spawn")
        self.events.emit("forced_move", st.x, st.y, st.z)

    def _play_player_rotation(self, r):
        yaw, ry, pitch, rp = r.float(), r.bool(), r.float(), r.bool()
        self.state.yaw = self.state.yaw + yaw if ry else yaw
        self.state.pitch = self.state.pitch + pitch if rp else pitch

    def _play_player_abilities(self, r):
        self.abilities_flags = r.byte()

    def _play_set_health(self, r):
        self.health = r.float()
        self.food = r.varint()
        self.saturation = r.float()
        self.events.emit("health", self.health, self.food)
        if self.health <= 0 and not self.dead:
            self._on_death("health reached 0")

    def _play_player_combat_kill(self, r):
        pid = r.varint()
        msg = to_plain(r.nbt())
        if pid == self.entity_id:
            self._on_death(msg)

    def _on_death(self, msg):
        self.dead = True
        self.controls.clear()
        log.warning("Died: %s", msg)
        self.events.emit("death", msg)
        if self.auto_respawn:
            def _respawn():
                time.sleep(1.0)
                if self.dead and self.connected:
                    self.respawn()
            threading.Thread(target=_respawn, daemon=True).start()

    def respawn(self):
        self.send("client_command", Writer().varint(0))

    def _play_set_experience(self, r):
        r.float()
        self.xp_level = r.varint()

    def _play_game_event(self, r):
        ev = r.ubyte()
        val = r.float()
        if ev == 3:
            self.gamemode = int(val)
        self.events.emit("game_event", ev, val)

    def _play_set_time(self, r):
        self.world_age = r.long()
        n = r.varint()
        for i in range(n):
            r.varint()
            t = r.varlong()
            r.float(); r.float()
            if i == 0:
                self.time_of_day = t % 24000

    def _play_set_default_spawn_position(self, r):
        try:
            r.string()
            self.spawn_point = r.position()
        except ProtocolError:
            pass

    # ------------------------------------------------------------------ chunks / blocks
    def _play_level_chunk_with_light(self, r):
        cx, cz = self.world.load_chunk(r)
        self.events.emit("chunk", cx, cz)

    def _play_forget_level_chunk(self, r):
        cz = r.int()
        cx = r.int()
        self.world.unload_chunk(cx, cz)

    def _play_chunk_batch_finished(self, r):
        r.varint()
        self.conn.send("chunk_batch_received", Writer().float(25.0))

    def _play_block_update(self, r):
        x, y, z = r.position()
        st = r.varint()
        old = self.world.get(x, y, z)
        self.world.set(x, y, z, st)
        self.events.emit("block_update", (x, y, z), old, st)

    def _play_section_blocks_update(self, r):
        self.world.multi_update(r)
        self.events.emit("block_update", None, None, None)

    def _play_chunks_biomes(self, r):
        self.world.update_biomes(r)

    def _play_block_changed_ack(self, r):
        self.events.emit("block_ack", r.varint())

    # ------------------------------------------------------------------ inventory
    def _read_slot(self, r):
        try:
            return self.codec.read_slot(r)
        except UnknownComponent as e:
            log.debug("slot decode: %s", e)
            raise

    def _play_container_set_content(self, r):
        wid = r.varint()
        sid = r.varint()
        n = r.varint()
        items = []
        try:
            for _ in range(n):
                items.append(self._read_slot(r))
            carried = self._read_slot(r)
        except UnknownComponent:
            # keep what we could decode; the rest of the slots stay as they were
            items += [None] * 0
            carried = self.inventory.cursor
            cur = self.inventory.active_slots() if wid != 0 else self.inventory.slots
            items = items + list(cur[len(items):n])
        self.inventory.on_set_content(wid, sid, items, carried)

    def _play_container_set_slot(self, r):
        wid = r.varint()
        sid = r.varint()
        slot = r.short()
        try:
            item = self._read_slot(r)
        except UnknownComponent:
            return
        self.inventory.on_set_slot(wid, sid, slot, item)

    def _play_set_player_inventory(self, r):
        idx = r.varint()
        try:
            self.inventory.on_set_player_slot(idx, self._read_slot(r))
        except UnknownComponent:
            pass

    def _play_set_cursor_item(self, r):
        try:
            self.inventory.cursor = self._read_slot(r)
            self.inventory._notify()
        except UnknownComponent:
            pass

    def _play_set_held_slot(self, r):
        self.inventory.held_slot = r.varint()

    def _play_open_screen(self, r):
        wid = r.varint()
        wtype = self.gd.menus.get(r.varint(), "unknown")
        title = to_plain(r.nbt())
        self.inventory.on_open(wid, wtype, title)
        self.events.emit("window_open", wid, wtype, title)

    def _play_container_close(self, r):
        self.inventory.on_close(r.varint())
        self.events.emit("window_close")

    def _play_container_set_data(self, r):
        self.inventory.on_property(r.varint(), r.short(), r.short())

    # ------------------------------------------------------------------ entities
    def _play_add_entity(self, r):
        e = self.entities.add(r, self.players)
        self.events.emit("entity_spawn", e)

    def _play_remove_entities(self, r):
        ids = r.array(r.varint)
        self.entities.remove(ids)
        self.events.emit("entity_gone", ids)

    def _play_move_entity_pos(self, r):
        self.entities.move(r, False)

    def _play_move_entity_pos_rot(self, r):
        self.entities.move(r, True)

    def _play_move_entity_rot(self, r):
        self.entities.rotate(r)

    def _play_entity_position_sync(self, r):
        self.entities.position_sync(r)

    def _play_teleport_entity(self, r):
        self.entities.teleport(r)

    def _play_set_entity_motion(self, r):
        eid, v = self.entities.velocity(r)
        if eid == self.entity_id:
            self.state.vx, self.state.vy, self.state.vz = v
            self.events.emit("knockback", v)

    def _play_set_entity_data(self, r):
        self.entities.metadata(r)

    def _play_take_item_entity(self, r):
        collected = r.varint()
        collector = r.varint()
        r.varint()
        if collector == self.entity_id:
            self.events.emit("collect", collected)

    def _play_damage_event(self, r):
        eid = r.varint()
        if eid == self.entity_id:
            self.events.emit("hurt")

    # ------------------------------------------------------------------ players / chat
    def _play_player_info_update(self, r):
        actions = r.ubyte()
        for _ in range(r.varint()):
            uid = str(r.uuid())
            if actions & 0x01:
                name = r.string()
                r.array(lambda: (r.string(), r.string(), r.optional(r.string)))
                self.players[uid] = name
                for e in list(self.entities.by_id.values()):
                    if e.uuid == uid:
                        e.name = name
            if actions & 0x02:
                if r.bool():
                    r.uuid(); r.long(); r.byte_array(); r.byte_array()
            if actions & 0x04:
                r.varint()
            if actions & 0x08:
                r.bool()
            if actions & 0x10:
                r.varint()
            if actions & 0x20:
                r.optional(r.nbt)
            if actions & 0x40:
                r.varint()
            if actions & 0x80:
                r.bool()

    def _play_player_info_remove(self, r):
        for uid in r.array(r.uuid):
            self.players.pop(str(uid), None)

    def _play_system_chat(self, r):
        text = to_plain(r.nbt())
        overlay = r.bool()
        if not overlay:
            self._add_chat(None, text, system=True)

    def _play_player_chat(self, r):
        r.varint()
        sender = str(r.uuid())
        r.varint()
        if r.bool():
            r.read(256)
        msg = r.string()
        name = self.players.get(sender, sender[:8])
        self._add_chat(name, msg)

    def _play_disguised_chat(self, r):
        msg = to_plain(r.nbt())
        cid = r.varint()
        if cid == 0:
            return
        name = to_plain(r.nbt())
        self._add_chat(name, msg)

    def _add_chat(self, sender, text, system=False):
        self.chat_log.append((time.time(), sender, text))
        log.info("[chat] %s%s", f"<{sender}> " if sender else "", text)
        if sender and sender != self.username:
            self.events.emit("chat", sender, text)
        self.events.emit("message", sender, text, system)

    # ================================================================== outgoing helpers
    def chat(self, message: str):
        """Send a chat message (or a /command)."""
        message = message.strip()
        if not message:
            return
        if message.startswith("/"):
            self.send("chat_command", Writer().string(message[1:][:32000]))
            return
        for i in range(0, len(message), 250):
            part = message[i:i + 250]
            w = (Writer().string(part).long(int(time.time() * 1000)).long(random.getrandbits(63))
                 .bool(False).varint(0).raw(b"\x00\x00\x00").byte(1))
            self.send("chat", w)

    def whisper(self, player: str, message: str):
        self.chat(f"/msg {player} {message}")

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    # ================================================================== tick loop
    def _tick_loop(self):
        next_t = time.monotonic()
        while True:
            if self.disconnected.is_set() and self.conn and self.conn.closed.is_set():
                if self.spawned.is_set():
                    self.spawned.clear()
                time.sleep(0.2)
                if self.conn.closed.is_set():
                    next_t = time.monotonic()
                    # loop stays alive so a reconnect reuses it
                    continue
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += 0.05
            if time.monotonic() - next_t > 1.0:   # we fell far behind (system hiccup): resync
                next_t = time.monotonic() + 0.05
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                log.exception("tick failed")

    def _tick(self):
        if not self.connected or self.conn.state != "play" or not self.spawned.is_set():
            return
        self.tick_count += 1
        t = time.monotonic()
        if self._last_tick_time:
            dt = t - self._last_tick_time
            self.tps_measured = self.tps_measured * 0.95 + (1.0 / max(dt, 1e-3)) * 0.05
        self._last_tick_time = t
        for hook in list(self.tick_hooks):
            try:
                hook()
            except Exception:  # noqa: BLE001
                log.exception("tick hook failed")
        if not self.dead and self.physics_enabled and not self._force_move:
            self.physics.step(self.state, self.controls)
        self._send_inputs()
        if not self.dead:
            force, self._force_move = self._force_move, False
            self._send_move(force=force)
        self.conn.send("client_tick_end")
        self.events.emit("tick", self.tick_count)

    def _send_inputs(self):
        c = self.controls
        flags = ((1 if c.forward else 0) | (2 if c.back else 0) | (4 if c.left else 0) | (8 if c.right else 0)
                 | (16 if c.jump else 0) | (32 if c.sneak else 0) | (64 if c.sprint else 0))
        if flags != self._input_flags:
            self._input_flags = flags
            self.conn.send("player_input", Writer().ubyte(flags))
        sprint = c.sprint and c.forward and not c.sneak and self.food > 6
        if sprint != self._sprinting:
            self._sprinting = sprint
            self.conn.send("player_command", Writer().varint(self.entity_id).varint(1 if sprint else 2).varint(0))

    def _send_move(self, force=False):
        st = self.state
        flags = (1 if st.on_ground else 0) | (2 if st.collided_h else 0)
        cur = (st.x, st.y, st.z, st.yaw, st.pitch, flags)
        last = self._sent
        self._ticks_since_pos += 1
        if last is None or force:
            pos_changed = rot_changed = True
        else:
            pos_changed = ((cur[0] - last[0]) ** 2 + (cur[1] - last[1]) ** 2 + (cur[2] - last[2]) ** 2 > 4e-8
                           or self._ticks_since_pos >= 20)
            rot_changed = cur[3] != last[3] or cur[4] != last[4]
        if pos_changed and rot_changed:
            self.conn.send("move_player_pos_rot", Writer().double(st.x).double(st.y).double(st.z)
                           .float(st.yaw).float(st.pitch).byte(flags))
        elif pos_changed:
            self.conn.send("move_player_pos", Writer().double(st.x).double(st.y).double(st.z).byte(flags))
        elif rot_changed:
            self.conn.send("move_player_rot", Writer().float(st.yaw).float(st.pitch).byte(flags))
        elif last is None or flags != last[5]:
            self.conn.send("move_player_status_only", Writer().byte(flags))
        if pos_changed:
            self._ticks_since_pos = 0
        self._sent = cur

    def wait_ticks(self, n: int = 1):
        target = self.tick_count + n
        end = time.time() + n * 0.05 * 4 + 1
        while self.tick_count < target and time.time() < end and self.connected:
            self.events.wait_for("tick", 0.2)

    # ================================================================== position helpers
    @property
    def position(self) -> tuple[float, float, float]:
        return (self.state.x, self.state.y, self.state.z)

    @property
    def block_pos(self) -> tuple[int, int, int]:
        return (math.floor(self.state.x), math.floor(self.state.y + 1e-6), math.floor(self.state.z))

    @property
    def eye(self) -> tuple[float, float, float]:
        return (self.state.x, self.state.y + (1.27 if self.controls.sneak else EYE_HEIGHT), self.state.z)

    def look(self, yaw: float, pitch: float):
        self.state.yaw = yaw
        self.state.pitch = max(-90.0, min(90.0, pitch))

    def look_at(self, x: float, y: float, z: float, wait: bool = False):
        ex, ey, ez = self.eye
        dx, dy, dz = x - ex, y - ey, z - ez
        yaw = math.degrees(math.atan2(-dx, dz))
        pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
        self.look(yaw, pitch)
        if wait:
            self.wait_ticks(1)

    def time_phase(self) -> str:
        t = self.time_of_day
        if t < 12000:
            return "day"
        if t < 13800:
            return "sunset"
        if t < 22200:
            return "night"
        return "sunrise"
