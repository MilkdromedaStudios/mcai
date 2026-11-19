"""Power-loss checkpoint: a run that was 'running' is found again and resumes where it was."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pyflayer.agent import trainer as T


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = mock.patch.object(T, "data_dir", lambda *a: Path(self.tmp.name).joinpath(*a))
        self.p.start()
        Path(self.tmp.name, "training").mkdir()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def _trainer(self):
        t = T.Trainer.__new__(T.Trainer)
        t.bot = SimpleNamespace(username="PyBot", dimension="overworld")
        t.ai, t.hours, t.current = "local", 7.0, "iron_pickaxe"
        t.done_this_run = {"wood", "crafting_table"}
        t.recovery_attempts = {"iron_pickaxe": 1}
        t.completed_attempts, t.successful_attempts = 5, 3
        t.end_time = time.time() + 3 * 3600
        return t

    def test_power_cut_leaves_a_resumable_run(self):
        self._trainer().checkpoint()                    # ... and then the power goes out
        runs = T.interrupted_runs()
        self.assertEqual(len(runs), 1)
        ck = runs[0]
        self.assertEqual((ck["bot"], ck["ai"], ck["current"]), ("PyBot", "local", "iron_pickaxe"))
        self.assertEqual(set(ck["done"]), {"wood", "crafting_table"})
        self.assertAlmostEqual(ck["remaining_s"], 3 * 3600, delta=5)

    def test_finished_or_stopped_runs_are_not_resumed(self):
        t = self._trainer()
        t.checkpoint("stopped")
        self.assertEqual(T.interrupted_runs(), [])
        t.checkpoint("finished")
        self.assertEqual(T.interrupted_runs(), [])

    def test_write_is_atomic(self):
        path = Path(self.tmp.name, "training", "x.json")
        T._atomic_json(path, {"a": 1})
        self.assertEqual(path.read_text().strip()[0], "{")
        self.assertFalse(os.path.exists(str(path) + ".tmp"))


if __name__ == "__main__":
    unittest.main()
