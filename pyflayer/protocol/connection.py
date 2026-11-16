"""TCP connection with Minecraft framing + zlib compression. Packets are dispatched by name."""
from __future__ import annotations

import logging
import socket
import threading
import zlib
from typing import Callable

from .buffer import ProtocolError, Reader, Writer

log = logging.getLogger("pyflayer.net")


class PacketTable:
    """Maps packet names <-> ids for each (state, direction) using the server's packets.json report."""

    def __init__(self, packets_report: dict):
        self.by_id: dict[tuple[str, str], dict[int, str]] = {}
        self.by_name: dict[tuple[str, str], dict[str, int]] = {}
        for state, dirs in packets_report.items():
            for direction, pk in dirs.items():
                ids = {}
                names = {}
                for name, info in pk.items():
                    short = name.split(":", 1)[1] if ":" in name else name
                    ids[info["protocol_id"]] = short
                    names[short] = info["protocol_id"]
                self.by_id[(state, direction)] = ids
                self.by_name[(state, direction)] = names

    def id_of(self, state: str, name: str) -> int:
        try:
            return self.by_name[(state, "serverbound")][name]
        except KeyError:
            raise ProtocolError(f"no serverbound packet {name!r} in state {state}") from None

    def has(self, state: str, name: str, direction: str = "serverbound") -> bool:
        return name in self.by_name.get((state, direction), {})

    def name_of(self, state: str, pid: int) -> str | None:
        return self.by_id.get((state, "clientbound"), {}).get(pid)


class Connection:
    def __init__(self, table: PacketTable, on_packet: Callable[[str, str, Reader], None],
                 on_close: Callable[[str], None] | None = None):
        self.table = table
        self.on_packet = on_packet
        self.on_close = on_close
        self.sock: socket.socket | None = None
        self.state = "handshake"
        self.compression = -1
        self._send_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.closed = threading.Event()
        self.close_reason = ""
        self.bytes_in = 0
        self.bytes_out = 0

    def connect(self, host: str, port: int, timeout: float = 15.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        self._thread = threading.Thread(target=self._read_loop, name="mc-net-reader", daemon=True)

    def start_reader(self):
        if self._thread and not self._thread.is_alive():
            self._thread.start()

    # ------------------------------------------------------------------ sending
    def send(self, name: str, payload: Writer | bytes | None = None):
        pid = self.table.id_of(self.state, name)
        self.send_raw(pid, payload.bytes() if isinstance(payload, Writer) else (payload or b""))

    def send_raw(self, pid: int, body: bytes):
        if self.closed.is_set() or self.sock is None:
            return
        inner = Writer().varint(pid).raw(body).bytes()
        if self.compression >= 0:
            if len(inner) >= self.compression:
                comp = zlib.compress(inner, 1)
                inner = Writer().varint(len(inner)).raw(comp).bytes()
            else:
                inner = b"\x00" + inner
        frame = Writer().varint(len(inner)).raw(inner).bytes()
        with self._send_lock:
            try:
                self.sock.sendall(frame)
                self.bytes_out += len(frame)
            except OSError as e:
                self.close(f"send failed: {e}")

    # ------------------------------------------------------------------ receiving
    def _read_loop(self):
        buf = bytearray()
        sock = self.sock
        try:
            while not self.closed.is_set():
                chunk = sock.recv(262144)
                if not chunk:
                    self.close("connection closed by server")
                    return
                self.bytes_in += len(chunk)
                buf += chunk
                pos = 0
                n = len(buf)
                while True:
                    # decode frame length varint
                    length = 0
                    shift = 0
                    p = pos
                    ok = False
                    while p < n and shift < 35:
                        b = buf[p]
                        p += 1
                        length |= (b & 0x7F) << shift
                        if not b & 0x80:
                            ok = True
                            break
                        shift += 7
                    if not ok or p + length > n:
                        break
                    frame = bytes(buf[p:p + length])
                    pos = p + length
                    self._handle_frame(frame)
                if pos:
                    del buf[:pos]
        except (OSError, ProtocolError) as e:
            self.close(f"read error: {e}")
        except Exception as e:  # noqa: BLE001 - never kill silently
            log.exception("reader crashed")
            self.close(f"reader crashed: {e}")

    def _handle_frame(self, frame: bytes):
        if self.compression >= 0:
            r = Reader(frame)
            dlen = r.varint()
            data = zlib.decompress(r.rest()) if dlen else frame[r.pos:]
        else:
            data = frame
        r = Reader(data)
        pid = r.varint()
        state = self.state
        name = self.table.name_of(state, pid) or f"unknown_{pid}"
        try:
            self.on_packet(state, name, r)
        except Exception:  # noqa: BLE001 - a bad packet must not kill the connection
            log.exception("error handling %s/%s", state, name)

    def close(self, reason: str = "closed"):
        if self.closed.is_set():
            return
        self.close_reason = reason
        self.closed.set()
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        if self.on_close:
            try:
                self.on_close(reason)
            except Exception:  # noqa: BLE001
                log.exception("on_close failed")
