"""Binary reader/writer for the Minecraft Java protocol (VarInt, strings, NBT, positions, ...)."""
from __future__ import annotations

import math
import struct
import uuid as _uuid

_S_BYTE = struct.Struct(">b")
_S_UBYTE = struct.Struct(">B")
_S_SHORT = struct.Struct(">h")
_S_USHORT = struct.Struct(">H")
_S_INT = struct.Struct(">i")
_S_UINT = struct.Struct(">I")
_S_LONG = struct.Struct(">q")
_S_ULONG = struct.Struct(">Q")
_S_FLOAT = struct.Struct(">f")
_S_DOUBLE = struct.Struct(">d")


class ProtocolError(Exception):
    pass


# --------------------------------------------------------------------------- NBT tag wrappers
# Python values are used directly for most tags. Typed wrappers are only needed when writing.
class TagList(list):
    def __init__(self, items=(), tag_type: int = 0):
        super().__init__(items)
        self.tag_type = tag_type


class Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes | bytearray | memoryview, pos: int = 0):
        self.data = bytes(data) if not isinstance(data, bytes) else data
        self.pos = pos

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise ProtocolError(f"read past end ({n} bytes at {self.pos}/{len(self.data)})")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def rest(self) -> bytes:
        b = self.data[self.pos:]
        self.pos = len(self.data)
        return b

    def _unpack(self, s: struct.Struct):
        if self.pos + s.size > len(self.data):
            raise ProtocolError("read past end")
        v = s.unpack_from(self.data, self.pos)[0]
        self.pos += s.size
        return v

    def bool(self) -> bool:
        return self._unpack(_S_UBYTE) != 0

    def byte(self) -> int:
        return self._unpack(_S_BYTE)

    def ubyte(self) -> int:
        return self._unpack(_S_UBYTE)

    def short(self) -> int:
        return self._unpack(_S_SHORT)

    def ushort(self) -> int:
        return self._unpack(_S_USHORT)

    def int(self) -> int:
        return self._unpack(_S_INT)

    def uint(self) -> int:
        return self._unpack(_S_UINT)

    def long(self) -> int:
        return self._unpack(_S_LONG)

    def float(self) -> float:
        return self._unpack(_S_FLOAT)

    def double(self) -> float:
        return self._unpack(_S_DOUBLE)

    def varint(self) -> int:
        result = 0
        shift = 0
        data = self.data
        while True:
            if self.pos >= len(data):
                raise ProtocolError("varint past end")
            b = data[self.pos]
            self.pos += 1
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
            if shift >= 35:
                raise ProtocolError("varint too big")
        if result & (1 << 31):
            result -= 1 << 32
        return result

    def varlong(self) -> int:
        result = 0
        shift = 0
        while True:
            b = self.ubyte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
            if shift >= 70:
                raise ProtocolError("varlong too big")
        if result & (1 << 63):
            result -= 1 << 64
        return result

    def string(self) -> str:
        n = self.varint()
        return self.read(n).decode("utf-8", errors="replace")

    def byte_array(self) -> bytes:
        return self.read(self.varint())

    def uuid(self) -> _uuid.UUID:
        return _uuid.UUID(bytes=self.read(16))

    def position(self) -> tuple[int, int, int]:
        v = self._unpack(_S_ULONG)
        x = v >> 38
        z = (v >> 12) & 0x3FFFFFF
        y = v & 0xFFF
        if x >= 1 << 25:
            x -= 1 << 26
        if z >= 1 << 25:
            z -= 1 << 26
        if y >= 1 << 11:
            y -= 1 << 12
        return x, y, z

    def angle(self) -> float:
        return self.ubyte() * 360.0 / 256.0

    def optional(self, fn):
        return fn() if self.bool() else None

    def array(self, fn) -> list:
        return [fn() for _ in range(self.varint())]

    def bitset(self) -> list[int]:
        return [self.long() for _ in range(self.varint())]

    def lpvec3(self) -> tuple[float, float, float]:
        b1 = self.ubyte()
        if b1 == 0:
            return 0.0, 0.0, 0.0
        b2 = self.ubyte()
        b36 = self.uint()
        packed = (b36 << 16) | (b2 << 8) | b1
        scale = b1 & 3
        if b1 & 4:
            scale |= (self.varint() & 0xFFFFFFFF) << 2

        def unpack(v):
            return min(float(v & 32767), 32766.0) * 2.0 / 32766.0 - 1.0

        return (unpack(packed >> 3) * scale, unpack(packed >> 18) * scale, unpack(packed >> 33) * scale)

    # ------------------------------------------------------------------ NBT
    def nbt(self, named_root: bool = False):
        """Network NBT (1.20.2+): root tag type + payload, no root name."""
        t = self.ubyte()
        if t == 0:
            return None
        if named_root:
            self.read(self.ushort())
        return self._nbt_payload(t)

    def _nbt_payload(self, t: int):
        if t == 1:
            return self.byte()
        if t == 2:
            return self.short()
        if t == 3:
            return self.int()
        if t == 4:
            return self.long()
        if t == 5:
            return self.float()
        if t == 6:
            return self.double()
        if t == 7:
            n = self.int()
            return self.read(n)
        if t == 8:
            n = self.ushort()
            return self.read(n).decode("utf-8", errors="replace")
        if t == 9:
            et = self.ubyte()
            n = self.int()
            return TagList([self._nbt_payload(et) for _ in range(max(0, n))], et)
        if t == 10:
            out = {}
            while True:
                ct = self.ubyte()
                if ct == 0:
                    return out
                name = self.read(self.ushort()).decode("utf-8", errors="replace")
                out[name] = self._nbt_payload(ct)
        if t == 11:
            n = self.int()
            return list(struct.unpack_from(f">{n}i", self.read(4 * n)))
        if t == 12:
            n = self.int()
            return list(struct.unpack_from(f">{n}q", self.read(8 * n)))
        raise ProtocolError(f"bad nbt tag {t}")


class Writer:
    __slots__ = ("buf",)

    def __init__(self):
        self.buf = bytearray()

    def bytes(self) -> bytes:
        return bytes(self.buf)

    def raw(self, b: bytes) -> "Writer":
        self.buf += b
        return self

    def bool(self, v: bool) -> "Writer":
        self.buf.append(1 if v else 0)
        return self

    def byte(self, v: int) -> "Writer":
        self.buf += _S_BYTE.pack(v)
        return self

    def ubyte(self, v: int) -> "Writer":
        self.buf.append(v & 0xFF)
        return self

    def short(self, v: int) -> "Writer":
        self.buf += _S_SHORT.pack(v)
        return self

    def ushort(self, v: int) -> "Writer":
        self.buf += _S_USHORT.pack(v)
        return self

    def int(self, v: int) -> "Writer":
        self.buf += _S_INT.pack(v)
        return self

    def long(self, v: int) -> "Writer":
        self.buf += _S_LONG.pack(v)
        return self

    def float(self, v: float) -> "Writer":
        self.buf += _S_FLOAT.pack(v)
        return self

    def double(self, v: float) -> "Writer":
        self.buf += _S_DOUBLE.pack(v)
        return self

    def varint(self, v: int) -> "Writer":
        v &= 0xFFFFFFFF
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                self.buf.append(b | 0x80)
            else:
                self.buf.append(b)
                return self

    def varlong(self, v: int) -> "Writer":
        v &= 0xFFFFFFFFFFFFFFFF
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                self.buf.append(b | 0x80)
            else:
                self.buf.append(b)
                return self

    def string(self, s: str) -> "Writer":
        b = s.encode("utf-8")
        self.varint(len(b))
        self.buf += b
        return self

    def byte_array(self, b: bytes) -> "Writer":
        self.varint(len(b))
        self.buf += b
        return self

    def uuid(self, u: _uuid.UUID) -> "Writer":
        self.buf += u.bytes
        return self

    def position(self, x: int, y: int, z: int) -> "Writer":
        v = ((x & 0x3FFFFFF) << 38) | ((z & 0x3FFFFFF) << 12) | (y & 0xFFF)
        self.buf += _S_ULONG.pack(v)
        return self

    def lpvec3(self, x: float, y: float, z: float) -> "Writer":
        m = max(abs(x), abs(y), abs(z))
        if math.isnan(m) or m < 1.0 / 32766.0:
            self.buf.append(0)
            return self
        scale = int(math.ceil(m))
        cont = (scale & 3) != scale
        packed_scale = (scale & 3) | 4 if cont else scale

        def pack(v):
            return int(round((v * 0.5 + 0.5) * 32766.0))

        packed = (pack(z / scale) << 33) | (pack(y / scale) << 18) | (pack(x / scale) << 3) | packed_scale
        self.buf.append(packed & 0xFF)
        self.buf.append((packed >> 8) & 0xFF)
        self.buf += _S_UINT.pack((packed >> 16) & 0xFFFFFFFF)
        if cont:
            self.varint(scale >> 2)
        return self

    def nbt_compound(self, d: dict) -> "Writer":
        self.ubyte(10)
        self._nbt_payload(10, d)
        return self

    def _nbt_type(self, v) -> int:
        if isinstance(v, bool):
            return 1
        if isinstance(v, int):
            return 3 if -2**31 <= v < 2**31 else 4
        if isinstance(v, float):
            return 6
        if isinstance(v, (bytes, bytearray)):
            return 7
        if isinstance(v, str):
            return 8
        if isinstance(v, dict):
            return 10
        if isinstance(v, list):
            return 9
        raise ProtocolError(f"cannot encode {type(v)} as nbt")

    def _nbt_payload(self, t: int, v):
        if t == 1:
            self.byte(int(v))
        elif t == 2:
            self.short(v)
        elif t == 3:
            self.int(v)
        elif t == 4:
            self.long(v)
        elif t == 5:
            self.float(v)
        elif t == 6:
            self.double(v)
        elif t == 7:
            self.int(len(v))
            self.buf += v
        elif t == 8:
            b = v.encode("utf-8")
            self.ushort(len(b))
            self.buf += b
        elif t == 9:
            et = getattr(v, "tag_type", 0) or (self._nbt_type(v[0]) if v else 0)
            self.ubyte(et)
            self.int(len(v))
            for item in v:
                self._nbt_payload(et, item)
        elif t == 10:
            for k, item in v.items():
                ct = self._nbt_type(item)
                self.ubyte(ct)
                b = k.encode("utf-8")
                self.ushort(len(b))
                self.buf += b
                self._nbt_payload(ct, item)
            self.ubyte(0)
