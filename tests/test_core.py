"""Offline tests: python -m pytest tests   (or: python tests/test_core.py)

Game-data dependent tests run only after `python -m pyflayer setup` generated data/gamedata/<version>.
"""
import math
import os
import sys
import struct
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from pyflayer.protocol.buffer import Reader, Writer  # noqa: E402
from pyflayer.protocol.text import to_plain  # noqa: E402
from pyflayer.world.world import read_paletted  # noqa: E402


class TestBuffer(unittest.TestCase):
    def test_varint_roundtrip(self):
        for v in (0, 1, 127, 128, 255, 25565, 2097151, 2147483647, -1, -2147483648):
            self.assertEqual(Reader(Writer().varint(v).bytes()).varint(), v)

    def test_varlong_roundtrip(self):
        for v in (0, 1, 2 ** 40, -1, -(2 ** 63)):
            self.assertEqual(Reader(Writer().varlong(v).bytes()).varlong(), v)

    def test_position(self):
        for p in ((0, 0, 0), (-1, -64, -1), (18357644, 831, -20882616), (30000000 - 1, 2047, -30000000)):
            self.assertEqual(Reader(Writer().position(*p).bytes()).position(), p)

    def test_lpvec3_matches_wiki_sample(self):
        self.assertEqual(Writer().lpvec3(1.0, 0.0, -1.0).bytes(), bytes([0xF1, 0xFF, 0x00, 0x00, 0xFF, 0xFF]))
        self.assertEqual(Writer().lpvec3(0, 0, 0).bytes(), b"\x00")
        x, y, z = Reader(Writer().lpvec3(10.0, 0.2, -5.0).bytes()).lpvec3()
        self.assertAlmostEqual(x, 10.0, 2)
        self.assertAlmostEqual(z, -5.0, 2)

    def test_nbt_roundtrip(self):
        d = {"name": "test", "n": 5, "big": 2 ** 40, "f": 1.5, "list": [1, 2, 3], "c": {"x": "y"}}
        out = Reader(Writer().nbt_compound(d).bytes()).nbt()
        self.assertEqual(out["name"], "test")
        self.assertEqual(out["big"], 2 ** 40)
        self.assertEqual(list(out["list"]), [1, 2, 3])
        self.assertEqual(out["c"], {"x": "y"})

    def test_text(self):
        self.assertEqual(to_plain({"text": "a", "extra": [{"text": "b"}]}), "ab")
        self.assertEqual(to_plain({"translate": "chat.type.text", "with": ["Steve", "hi"]}), "<Steve> hi")


class TestPalette(unittest.TestCase):
    def _container(self, bpe, palette, values, entries):
        w = Writer().ubyte(bpe)
        if palette is not None:
            w.varint(len(palette))
            for p in palette:
                w.varint(p)
        per = 64 // bpe
        longs = []
        for i in range(0, entries, per):
            v = 0
            for j, val in enumerate(values[i:i + per]):
                v |= val << (j * bpe)
            longs.append(v)
        for v in longs:
            w.raw(struct.pack(">Q", v))
        return w.bytes()

    def test_single_value(self):
        vals, pal = read_paletted(Reader(Writer().ubyte(0).varint(9).bytes()), 4096, 8, 16)
        self.assertTrue((vals == 9).all())
        self.assertEqual(pal, {9})

    def test_indirect(self):
        rng = np.random.default_rng(1)
        idx = rng.integers(0, 5, 4096).tolist()
        data = self._container(4, [0, 1, 10, 200, 3000], idx, 4096)
        vals, pal = read_paletted(Reader(data), 4096, 8, 16)
        expect = np.array([0, 1, 10, 200, 3000])[idx]
        self.assertTrue((vals == expect).all())

    def test_direct(self):
        rng = np.random.default_rng(2)
        idx = rng.integers(0, 35000, 4096).tolist()
        vals, _ = read_paletted(Reader(self._container(16, None, idx, 4096)), 4096, 8, 16)
        self.assertTrue((vals == np.array(idx)).all())


def _gamedata():
    from pyflayer.server.manager import gamedata_dir
    from pyflayer.data.gamedata import GameData
    d = gamedata_dir("26.3")
    if not (d / "complete.json").exists():
        return None
    return GameData(d)


@unittest.skipIf(_gamedata() is None, "game data not generated yet (run: python -m pyflayer setup)")
class TestWithGameData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gd = _gamedata()

    def _flat_world(self):
        from pyflayer.world.world import Column, World
        w = World(self.gd)
        w.set_dimension("overworld", -64, 384)
        stone = int(self.gd.block_default["stone"])
        for cx in range(-2, 3):
            for cz in range(-2, 3):
                col = Column(cx, cz, w.n_sections)
                # ground: y in [-64, 63] stone, surface at y=63
                for s in range(8):
                    col.blocks[s] = stone
                    col.palettes[s] = {stone}
                w.columns[(cx, cz)] = col
        return w, stone

    def test_recipes_and_drops(self):
        gd = self.gd
        self.assertTrue(gd.recipes_for("crafting_table"))
        self.assertIn("cobblestone", gd.block_drops["stone"])
        self.assertEqual(gd.dig_ticks("stone", "wooden_pickaxe"), 23)
        self.assertTrue(gd.requires_tool("iron_ore"))
        self.assertIn("blaze", gd.item_from_entities.get("blaze_rod", []))

    def test_physics_lands_and_walks(self):
        from pyflayer.bot.physics import Controls, Physics, PlayerState
        w, _ = self._flat_world()
        ph = Physics(w, self.gd)
        st = PlayerState(x=0.5, y=70.0, z=0.5)
        c = Controls()
        for _ in range(60):
            ph.step(st, c)
        self.assertTrue(st.on_ground)
        self.assertAlmostEqual(st.y, 64.0, 5)
        c.forward = True
        st.yaw = 0   # +z
        for _ in range(20):
            ph.step(st, c)
        self.assertGreater(st.z, 3.5)       # ~4.3 blocks/s walking

    def test_pathfinder_flat_and_wall(self):
        from pyflayer.bot.pathfinder import GoalBlock, Pathfinder

        class FakeInv:
            def items(self):
                return []

            def count(self, _):
                return 0

        class FakeBot:
            pass
        w, stone = self._flat_world()
        bot = FakeBot()
        bot.gd, bot.world, bot.inventory, bot.learned_hardness = self.gd, w, FakeInv(), {}
        pf = Pathfinder(bot)
        res = pf.find((0, 64, 0), GoalBlock(10, 64, 0))
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.steps), 10)
        # 3-high wall across x=5 -> must dig through (no scaffold, digging allowed)
        for z in range(-32, 48):          # spans the whole synthetic world
            for y in (64, 65, 66):
                w.set(5, y, z, stone)
        res = pf.find((0, 64, 0), GoalBlock(10, 64, 0))
        self.assertEqual(res.status, "success")
        self.assertTrue(any(s.dig for s in res.steps))
        pf.can_dig = False
        res = pf.find((0, 64, 0), GoalBlock(10, 64, 0), max_nodes=3000, timeout=1)
        self.assertNotEqual(res.status, "success")


if __name__ == "__main__":
    unittest.main(verbosity=2)
