"""Verify portable world-independent training models and crash-safe autosaves."""
import gzip
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pyflayer.agent.memory import Brain
from pyflayer.agent.trainer import Trainer


class TestPortableUniversalModel(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.brain = Brain(self.root / "FirstBot.db")
        self.brain.remember("Our base is at 100 64 200", "location")
        self.brain.log("navigation", "returned to 100 64 200")
        self.brain.set_waypoint("home", "overworld", (100, 64, 200))
        self.brain.milestone("wood", True)
        self.brain.set_knowledge("oak_log", "block", "Any oak log", "gamedata")
        self.brain.set_knowledge("my_base", "place", "100 64 200", "agent")
        self.brain.save_skill("gather_any_wood", "find local trees", [
            {"tool": "gather_nearest_wood", "args": {"count": 3}}
        ])
        self.brain.save_skill("go_back_home", "fixed old-world waypoint", [
            {"tool": "goto", "args": {"x": 100, "y": 64, "z": 200}}
        ])
        for _ in range(8):
            self.brain.record_drill("sprint_jump", True, "real verified dry-ground jump")

    def test_export_import_learns_without_source_world_location(self):
        path = self.brain.export_universal()
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "FirstBot-universal-model.json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["format"], "pyflayer-universal-1")
        self.assertEqual(payload["scope"], "seed-independent")
        for key in ("waypoints", "milestones", "episodes", "facts", "seen", "stats"):
            self.assertNotIn(key, payload)
        self.assertEqual([x["name"] for x in payload["skills"]], ["gather_any_wood"])
        self.assertTrue(all(row[0] != "my_base" for row in payload["knowledge"]))

        other = Brain(self.root / "OtherSeedBot.db")
        other.import_(path)
        self.assertIsNotNone(other.skill("gather_any_wood"))
        self.assertIsNone(other.skill("go_back_home"))
        self.assertEqual(other.waypoints(), [])
        self.assertEqual(other.milestones(), {})
        self.assertEqual(other.drill("sprint_jump")["prior_successes"], 8)
        self.assertEqual(other.drill("sprint_jump")["attempts"], 0)
        self.assertFalse(other.drill("sprint_jump")["certified"])
        self.assertIsNotNone(other.knowledge("oak_log"))
        self.assertIsNone(other.knowledge("my_base"))
        other.import_(path)
        self.assertEqual(other.drill("sprint_jump")["prior_successes"], 8)

    def test_full_backup_keeps_the_old_world_and_actual_scores(self):
        full = self.brain.export_latest()
        with gzip.open(full, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["format"], "pyflayer-brain-2")
        self.assertTrue(payload["waypoints"])
        self.assertTrue(payload["milestones"])
        self.assertTrue(payload["episodes"])
        self.assertTrue(payload["skill_drills"])
        other = Brain(self.root / "RestoreSameWorld.db")
        other.import_(full)
        self.assertTrue(other.milestones()["wood"]["achieved"])
        self.assertEqual(other.drill("sprint_jump")["successes"], 8)

    def test_failed_export_does_not_destroy_previous_model(self):
        path = self.brain.export_universal()
        original = path.read_bytes()
        with patch("pyflayer.agent.memory.gzip.open", side_effect=OSError("storage full")):
            with self.assertRaisesRegex(OSError, "storage full"):
                self.brain.export_universal()
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_training_snapshot_writes_both_files(self):
        t = Trainer.__new__(Trainer)
        t.brain = self.brain
        t._model_exports = 0
        t.note = Mock()
        t._save_models()
        self.assertEqual(t._model_exports, 1)
        self.assertTrue((self.root / "FirstBot-export.json.gz").exists())
        self.assertTrue((self.root / "FirstBot-universal-model.json.gz").exists())


if __name__ == "__main__":
    unittest.main()
