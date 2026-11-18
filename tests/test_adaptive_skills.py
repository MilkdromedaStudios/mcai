"""Offline regression tests for adaptive resource selection, skill drills and lava safety."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pyflayer.agent.memory import Brain
from pyflayer.agent.skill_training import SkillCoach
from pyflayer.bot.combat import Combat
from pyflayer.bot.pathfinder import Pathfinder
from pyflayer.bot.planner import Planner


class FakeEvents:
    def __init__(self):
        self.listeners = {}
        self.hits = []

    def on(self, name, cb):
        self.listeners.setdefault(name, []).append(cb)

    def emit(self, name, *args):
        self.hits.append((name, args))
        for cb in self.listeners.get(name, ()):
            cb(*args)


class TestNearbyWood(unittest.TestCase):
    def test_oak_nearby_does_not_seek_unavailable_birch(self):
        hits = []
        inv = {}

        def find(names, radius, count):
            names = [names] if isinstance(names, str) else names
            return [(3, 64, 0)] if "oak_log" in names else []

        def collect(names, count, radius, timeout):
            hits.append((sorted(names), count))
            inv["oak_log"] = inv.get("oak_log", 0) + count     # the nearest reachable tree is oak
            return "chopped"

        bot = SimpleNamespace(
            position=(0, 64, 0),
            gd=SimpleNamespace(block_states={"oak_log": [], "birch_log": []}),
            find_blocks=find, inventory=SimpleNamespace(count=lambda n: inv.get(n, 0)),
            collect_block=collect, events=FakeEvents())
        planner = Planner(bot)
        self.assertEqual(planner.nearby_wood(), [("oak_log", 3.0)])
        self.assertEqual(planner.gather_wood(5), "gathered 5 logs")
        self.assertEqual(hits, [(["birch_log", "oak_log"], 5)])   # any wood, one trip, no birch hunt

    def test_unreachable_trees_do_not_stop_gathering(self):
        inv = {}
        tries = []

        def collect(names, count, radius, timeout):
            tries.append(radius)
            if len(tries) == 1:
                from pyflayer.bot.actions import ActionError
                raise ActionError("could not find/reach any birch_log within 48 blocks")
            inv["oak_log"] = count
            return "chopped"

        bot = SimpleNamespace(
            position=(0, 64, 0), gd=SimpleNamespace(block_states={"oak_log": [], "birch_log": []}),
            find_blocks=lambda names, radius, count: [(3, 64, 0)],
            inventory=SimpleNamespace(count=lambda n: inv.get(n, 0)), collect_block=collect, events=FakeEvents())
        planner = Planner(bot)
        planner.explore_for = lambda *a, **k: True
        self.assertEqual(planner.gather_wood(2, radius=48), "gathered 2 logs")
        self.assertGreater(tries[1], tries[0])                     # widened the search after the failure

    def test_tree_cache_updates_when_player_enters_new_chunk(self):
        pos = [0, 64, 0]
        calls = []
        bot = SimpleNamespace(
            position=pos, gd=SimpleNamespace(block_states={"oak_log": []}),
            inventory=SimpleNamespace(count=lambda n: 0),
            find_blocks=lambda name, radius, count: (
                calls.append((tuple(pos), name)), [(21, 64, 0)] if pos[0] >= 16 else []
            )[1], events=FakeEvents())
        planner = Planner(bot)
        self.assertIsNone(planner._nearby_block("oak_log"))
        pos[0] = 20
        self.assertEqual(planner._nearby_block("oak_log"), (21, 64, 0))
        self.assertEqual(len(calls), 2)


class TestSkillCoach(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.brain = Brain(Path(tmp.name) / "brain.db")
        self.bot = SimpleNamespace(
            events=FakeEvents(), dead=False, health=20, food=20,
            spawned=SimpleNamespace(is_set=lambda: True))
        self.cfg = {"training": {"allow_advanced_moves": False}}
        self.agent = SimpleNamespace(bot=self.bot, brain=self.brain, cfg=self.cfg, _note=Mock())
        self.coach = SkillCoach(self.agent)

    def test_real_success_counts_and_mastery_persists(self):
        for name in ("sprint_jump", "controlled_placement"):
            for i in range(8):
                self.brain.record_drill(name, True, "verified physics/world observation")
        self.assertTrue(self.brain.drill("sprint_jump")["certified"])
        self.assertFalse(self.coach.lava_jump_ready())
        self.cfg["training"]["allow_advanced_moves"] = True
        self.assertTrue(self.coach.lava_jump_ready())
        self.bot.health = 8
        self.assertFalse(self.coach.lava_jump_ready())

    def test_bounded_practice_only_credits_success_after_real_result(self):
        self.coach._resource_choice = lambda: (True, "oak +2 logs")
        text = self.coach.practice_one("resource_choice")
        self.assertIn("PASS", text)
        self.assertEqual(self.brain.drill("resource_choice")["successes"], 1)
        text = self.coach.practice_one("water_clutch")
        self.assertIn("Unknown or locked", text)

    def test_clutch_result_is_recorded_only_when_verified(self):
        self.bot.events.emit("clutch_result", False, "no server inventory update")
        self.assertEqual(self.brain.drill("water_clutch")["successes"], 0)
        self.bot.events.emit("clutch_result", True, "real bucket exchange and landing")
        self.assertEqual(self.brain.drill("water_clutch")["successes"], 1)

    def test_brain_export_contains_drill_scores(self):
        self.brain.record_drill("safe_navigation", True, "verified arrival")
        with tempfile.TemporaryDirectory() as folder:
            backup = Path(folder) / "skills.json.gz"
            self.brain.export(backup)
            loaded = Brain(Path(folder) / "restored.db")
            loaded.import_(backup)
            self.assertEqual(loaded.drill("safe_navigation")["successes"], 1)


class TestLavaHazards(unittest.TestCase):
    def test_fast_combat_will_not_sprint_into_lava(self):
        combat = Combat.__new__(Combat)
        bot = SimpleNamespace(
            state=SimpleNamespace(x=0.5, y=64, z=0.5, in_water=False, on_climbable=False),
            pathfinder=SimpleNamespace(
                lava=lambda x, y, z: x == 1 and y == 63,
                can_stand=lambda x, y, z: True, water=lambda x, y, z: False))
        combat.bot = bot
        self.assertTrue(combat._direct_chase_unsafe(SimpleNamespace(x=3, z=0.5)))
        bot.pathfinder.lava = lambda x, y, z: False
        self.assertFalse(combat._direct_chase_unsafe(SimpleNamespace(x=3, z=0.5)))

    def test_lava_gap_jump_requires_unlocked_training(self):
        pf = Pathfinder.__new__(Pathfinder)
        pf.can_dig = False
        pf.can_place = False
        pf.water = lambda x, y, z: False
        pf.climbable = lambda x, y, z: False
        pf.lava = lambda x, y, z: (x, y, z) == (1, 63, 0)
        pf.standable = lambda x, y, z: (x, y, z) in ((0, 63, 0), (2, 63, 0))
        pf.can_stand = lambda x, y, z: (x, y, z) in ((0, 64, 0), (2, 64, 0))
        pf.passable = lambda x, y, z: 0 <= x <= 2 and y in (64, 65, 66) and z == 0
        pf.bot = SimpleNamespace(allow_lava_jumps=lambda: True)
        pf._gap_jumps_enabled = False
        before = pf._neighbors((0, 64, 0), 0)
        self.assertFalse(any(st.kind == "gap_jump" for dest, cost, st in before))
        pf._gap_jumps_enabled = True
        after = pf._neighbors((0, 64, 0), 0)
        self.assertTrue(any(st.kind == "gap_jump" and dest == (2, 64, 0)
                            for dest, cost, st in after))


if __name__ == "__main__":
    unittest.main()
