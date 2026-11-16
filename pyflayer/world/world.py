"""Client-side world: chunk storage (numpy), block lookups, block/biome search.

Optimisations for small machines:
  * every chunk column is one (sections x 4096) uint16 numpy array -> O(1) block lookups and
    vectorised searches;
  * every section keeps a set of the block states it contains, so a search for e.g. diamond ore
    skips >99% of sections without touching their data;
  * palette unpacking is fully vectorised.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from ..protocol.buffer import Reader


def _unpack_longs(data: bytes, bpe: int, count: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=">u8").astype(np.uint64)
    per = 64 // bpe
    shifts = (np.arange(per, dtype=np.uint64) * np.uint64(bpe))
    mask = np.uint64((1 << bpe) - 1)
    vals = (arr[:, None] >> shifts[None, :]) & mask
    return vals.reshape(-1)[:count]


def read_paletted(r: Reader, entries: int, direct_threshold: int, global_bits: int):
    """Returns (values ndarray[int], palette set)."""
    bpe = r.ubyte()
    if bpe == 0:
        v = r.varint()
        return np.full(entries, v, dtype=np.uint16), {v}
    if bpe <= direct_threshold:
        palette = np.array([r.varint() for _ in range(r.varint())], dtype=np.uint16)
        per = 64 // bpe
        n_longs = (entries + per - 1) // per
        idx = _unpack_longs(r.read(n_longs * 8), bpe, entries)
        idx = np.minimum(idx, len(palette) - 1) if len(palette) else idx
        vals = palette[idx.astype(np.intp)]
        return vals, set(palette.tolist())
    per = 64 // bpe
    n_longs = (entries + per - 1) // per
    vals = _unpack_longs(r.read(n_longs * 8), bpe, entries).astype(np.uint16)
    return vals, set(np.unique(vals).tolist())


class Column:
    __slots__ = ("cx", "cz", "blocks", "biomes", "palettes")

    def __init__(self, cx, cz, n_sections):
        self.cx = cx
        self.cz = cz
        self.blocks = np.zeros((n_sections, 4096), dtype=np.uint16)
        self.biomes = np.zeros((n_sections, 64), dtype=np.uint16)
        self.palettes: list[set] = [set() for _ in range(n_sections)]


class World:
    def __init__(self, gamedata):
        self.gd = gamedata
        self.columns: dict[tuple[int, int], Column] = {}
        self.lock = threading.RLock()
        self.min_y = -64
        self.height = 384
        self.dimension = "overworld"
        self.biome_names: list[str] = []
        self.biome_bits = 7
        self.version = 0          # bumps on every change (cheap cache invalidation)

    # ------------------------------------------------------------------ dimension
    def set_dimension(self, name: str, min_y: int, height: int):
        with self.lock:
            self.dimension = name.split(":")[-1]
            self.min_y = min_y
            self.height = height
            self.columns.clear()
            self.version += 1

    def set_biomes(self, names: list[str]):
        self.biome_names = [n.split(":")[-1] for n in names]
        self.biome_bits = max(1, math.ceil(math.log2(max(2, len(names)))))

    @property
    def n_sections(self) -> int:
        return self.height // 16

    # ------------------------------------------------------------------ chunk packets
    def load_chunk(self, r: Reader):
        cx = r.int()
        cz = r.int()
        for _ in range(r.varint()):          # heightmaps
            r.varint()
            n = r.varint()
            r.read(8 * n)
        data = r.read(r.varint())
        dr = Reader(data)
        col = Column(cx, cz, self.n_sections)
        gbits = self.gd.global_bits_blocks
        for s in range(self.n_sections):
            if dr.remaining() <= 0:
                break
            dr.short()   # block count
            dr.short()   # fluid count
            vals, pal = read_paletted(dr, 4096, 8, gbits)
            if not (len(pal) == 1 and 0 in pal):
                col.blocks[s] = vals          # all-air sections stay untouched zero pages (saves RAM on a Pi)
            col.palettes[s] = pal
            bvals, _ = read_paletted(dr, 64, 3, self.biome_bits)
            col.biomes[s] = bvals
        with self.lock:
            self.columns[(cx, cz)] = col
            self.version += 1
        return cx, cz

    def unload_chunk(self, cx, cz):
        with self.lock:
            self.columns.pop((cx, cz), None)
            self.version += 1

    def update_biomes(self, r: Reader):
        for _ in range(r.varint()):
            z = r.int()
            x = r.int()
            data = Reader(r.read(r.varint()))
            col = self.columns.get((x, z))
            if not col:
                continue
            for s in range(self.n_sections):
                if data.remaining() <= 0:
                    break
                bvals, _ = read_paletted(data, 64, 3, self.biome_bits)
                col.biomes[s] = bvals

    # ------------------------------------------------------------------ access
    def is_loaded(self, x: int, z: int) -> bool:
        return (x >> 4, z >> 4) in self.columns

    def get(self, x: int, y: int, z: int) -> int | None:
        """Block state id at a position, or None if unloaded / out of the world."""
        col = self.columns.get((x >> 4, z >> 4))
        if col is None:
            return None
        sy = (y - self.min_y) >> 4
        if sy < 0 or sy >= len(col.blocks):
            return 0
        return int(col.blocks[sy, ((y & 15) << 8) | ((z & 15) << 4) | (x & 15)])

    def name_at(self, x, y, z) -> str | None:
        s = self.get(x, y, z)
        return None if s is None else self.gd.block_name(s)

    def set(self, x: int, y: int, z: int, state: int):
        col = self.columns.get((x >> 4, z >> 4))
        if col is None:
            return
        sy = (y - self.min_y) >> 4
        if 0 <= sy < len(col.blocks):
            col.blocks[sy, ((y & 15) << 8) | ((z & 15) << 4) | (x & 15)] = state
            col.palettes[sy].add(state)
            self.version += 1

    def multi_update(self, r: Reader):
        v = r.long()                                   # x:22 bits | z:22 bits | y:20 bits (all signed)
        sx = v >> 42                                   # arithmetic shift keeps the sign
        sz = (((v >> 20) & 0x3FFFFF) ^ 0x200000) - 0x200000
        sy = ((v & 0xFFFFF) ^ 0x80000) - 0x80000
        for _ in range(r.varint()):
            e = r.varlong()
            state = e >> 12
            lx = (e >> 8) & 15
            lz = (e >> 4) & 15
            ly = e & 15
            self.set((sx << 4) + lx, (sy << 4) + ly, (sz << 4) + lz, state)

    def biome_at(self, x: int, y: int, z: int) -> str | None:
        col = self.columns.get((x >> 4, z >> 4))
        if col is None:
            return None
        sy = (y - self.min_y) >> 4
        sy = min(max(sy, 0), len(col.biomes) - 1)
        idx = (((y & 15) >> 2) << 4) | (((z & 15) >> 2) << 2) | ((x & 15) >> 2)
        b = int(col.biomes[sy, idx])
        return self.biome_names[b] if b < len(self.biome_names) else f"biome#{b}"

    # ------------------------------------------------------------------ search
    def _columns_near(self, x, z, radius):
        r = int(radius) // 16 + 1
        cx0, cz0 = int(x) >> 4, int(z) >> 4
        with self.lock:
            cols = [c for (cx, cz), c in self.columns.items() if abs(cx - cx0) <= r and abs(cz - cz0) <= r]
        return cols

    def find_blocks(self, origin, states: np.ndarray, radius: float = 64, max_count: int = 64,
                    y_range: tuple[int, int] | None = None) -> list[tuple[int, int, int]]:
        """All positions of the given block states within radius, nearest first."""
        if len(states) == 0:
            return []
        ox, oy, oz = origin
        target = set(int(s) for s in states)
        r2 = radius * radius
        found_xyz = []
        for col in self._columns_near(ox, oz, radius):
            # closest-possible distance from origin to this column (horizontal)
            dx = max(col.cx * 16 - ox, 0, ox - (col.cx * 16 + 15))
            dz = max(col.cz * 16 - oz, 0, oz - (col.cz * 16 + 15))
            if dx * dx + dz * dz > r2:
                continue
            for s, pal in enumerate(col.palettes):
                if target.isdisjoint(pal):
                    continue
                base_y = self.min_y + s * 16
                if y_range and (base_y + 15 < y_range[0] or base_y > y_range[1]):
                    continue
                if abs(base_y + 8 - oy) > radius + 8:
                    continue
                idx = np.nonzero(np.isin(col.blocks[s], states))[0]
                if idx.size == 0:
                    continue
                xs = (idx & 15) + col.cx * 16
                zs = ((idx >> 4) & 15) + col.cz * 16
                ys = (idx >> 8) + base_y
                found_xyz.append(np.stack([xs, ys, zs], axis=1))
        if not found_xyz:
            return []
        allp = np.concatenate(found_xyz).astype(np.int64)
        d2 = (allp[:, 0] + 0.5 - ox) ** 2 + (allp[:, 1] + 0.5 - oy) ** 2 + (allp[:, 2] + 0.5 - oz) ** 2
        keep = d2 <= r2
        if y_range:
            keep &= (allp[:, 1] >= y_range[0]) & (allp[:, 1] <= y_range[1])
        allp = allp[keep]
        d2 = d2[keep]
        order = np.argsort(d2, kind="stable")[:max_count]
        return [tuple(int(v) for v in allp[i]) for i in order]

    def block_census(self, origin, radius: float = 100) -> dict[str, int]:
        """Counts of every block type within a horizontal radius (fast, vectorised)."""
        ox, oy, oz = origin
        counts = np.zeros(self.gd.num_states, dtype=np.int64)
        for col in self._columns_near(ox, oz, radius):
            dx = max(col.cx * 16 - ox, 0, ox - (col.cx * 16 + 15))
            dz = max(col.cz * 16 - oz, 0, oz - (col.cz * 16 + 15))
            if dx * dx + dz * dz > radius * radius:
                continue
            counts += np.bincount(col.blocks.ravel(), minlength=self.gd.num_states)[:self.gd.num_states]
        by_block = np.bincount(self.gd.state_block, weights=counts, minlength=len(self.gd.block_names))
        out = {}
        for i in np.nonzero(by_block)[0]:
            out[self.gd.block_names[i]] = int(by_block[i])
        return out

    def biome_census(self, origin, radius: float = 100) -> dict[str, tuple[int, tuple[int, int, int]]]:
        """biome -> (cells, nearest cell position) within radius."""
        ox, oy, oz = origin
        res: dict[str, list] = {}
        for col in self._columns_near(ox, oz, radius):
            for s in range(len(col.biomes)):
                base_y = self.min_y + s * 16
                if abs(base_y + 8 - oy) > 48:      # biomes near our altitude only
                    continue
                vals = col.biomes[s]
                for b in np.unique(vals):
                    idx = np.nonzero(vals == b)[0]
                    xs = (idx & 3) * 4 + col.cx * 16 + 2
                    zs = ((idx >> 2) & 3) * 4 + col.cz * 16 + 2
                    ys = (idx >> 4) * 4 + base_y + 2
                    d2 = (xs - ox) ** 2 + (zs - oz) ** 2
                    ok = d2 <= radius * radius
                    if not ok.any():
                        continue
                    k = int(np.argmin(np.where(ok, d2, 1 << 60)))
                    name = self.biome_names[b] if b < len(self.biome_names) else f"biome#{b}"
                    pos = (int(xs[k]), int(ys[k]), int(zs[k]))
                    dist = float(d2[k])
                    cur = res.get(name)
                    if cur is None:
                        res[name] = [int(ok.sum()), pos, dist]
                    else:
                        cur[0] += int(ok.sum())
                        if dist < cur[2]:
                            cur[1], cur[2] = pos, dist
        return {k: (v[0], v[1]) for k, v in sorted(res.items(), key=lambda kv: kv[1][2])}

    def surface_y(self, x: int, z: int) -> int | None:
        """Highest non-air block y in a column (used by the minimap & exploration)."""
        col = self.columns.get((x >> 4, z >> 4))
        if col is None:
            return None
        idx_xz = ((z & 15) << 4) | (x & 15)
        air = self.gd.is_air
        for s in range(len(col.blocks) - 1, -1, -1):
            if len(col.palettes[s]) == 1 and air[next(iter(col.palettes[s]))]:
                continue
            column = col.blocks[s, idx_xz::256]   # 16 values, y ascending
            solid = np.nonzero(~air[column])[0]
            if solid.size:
                return self.min_y + s * 16 + int(solid[-1])
        return None
