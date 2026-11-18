"""Saved diamonds are mined first once the bot has an iron pickaxe."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pyflayer.agent.memory import Brain
from pyflayer.agent.priority import Finds


class FakeBot:
    def __init__(self):
        self.dimension = "overworld"
        self.position = (0.5, 64, 0.5)
        self.inv = {}
        self.ores = {(30, 12, 4), (31, 12, 4)}
        self.chats = []
        self.inventory = SimpleNamespace(count=lambda n: self.inv.get(n, 0))
        self.spawned = SimpleNamespace(is_set=lambda: True)
        self.dead = False

    def find_blocks(self, names, radius, count):
        return [p for p in sorted(self.ores) if math.dist(p, self.position) <= radius][:count]

    def chat(self, m):
        self.chats.append(m)

    def goto(self, goal, timeout=0):
        self.position = (goal.x, goal.y, goal.z)

    def collect_block(self, names, n, radius, timeout):
        got = [p for p in self.ores if math.dist(p, self.position) <= radius]
        self.ores -= set(got)
        self.inv["diamond"] = self.inv.get("diamond", 0) + len(got)


class TestFinds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.brain = Brain(Path(self.tmp.name) / "b.db")
        self.bot = FakeBot()
        self.notes = []
        agent = SimpleNamespace(bot=self.bot, brain=self.brain, _note=self.notes.append)
        self.finds = Finds(agent)

    def tearDown(self):
        self.brain.close() if hasattr(self.brain, "close") else None
        self.tmp.cleanup()

    def test_saves_once_per_vein_and_waits_for_pickaxe(self):
        self.assertEqual(len(self.finds.scan()), 1)          # two touching ores = one vein
        self.assertEqual(self.finds.scan(), [])               # not saved twice
        self.assertEqual(self.finds.pending(), [])            # no iron pickaxe yet: keep it for later
        self.bot.inv["iron_pickaxe"] = 1
        self.assertEqual(len(self.finds.pending()), 1)
        self.assertTrue(self.finds.run_pending())
        self.assertEqual(self.bot.inv["diamond"], 2)
        self.assertEqual(self.finds.saved(), [])              # mined -> waypoint removed

    def test_survives_restart(self):
        self.finds.scan()
        agent2 = SimpleNamespace(bot=self.bot, brain=self.brain, _note=lambda m: None)
        self.assertEqual(len(Finds(agent2).saved("diamonds")), 1)


if __name__ == "__main__":
    unittest.main()
