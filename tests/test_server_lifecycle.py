"""No-live-server regression tests for single-owner Minecraft lifecycle."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyflayer.server.manager import MinecraftServer


class FakeConfig:
    def __getitem__(self, key):
        return {"minecraft_version": "26.3"}[key]


@unittest.skipUnless(sys.platform.startswith("linux"), "flock and /proc are Linux-specific")
class TestServerOwnership(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch("pyflayer.server.manager.server_dir", return_value=self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_second_instance_cannot_take_world_lock(self):
        first, second = MinecraftServer(FakeConfig()), MinecraftServer(FakeConfig())
        first._acquire_owner()
        self.addCleanup(first._release_owner)
        with self.assertRaisesRegex(RuntimeError, "Another PyFlayer instance"):
            second._acquire_owner()
        first._release_owner()
        second._acquire_owner()
        second._release_owner()

    def test_stale_pid_of_unrelated_process_is_ignored(self):
        (self.root / "server.pid").write_text(str(os.getpid()))
        MinecraftServer(FakeConfig())._check_existing_server()

    def test_older_running_server_is_never_killed(self):
        (self.root / "server.pid").write_text(str(os.getpid()))
        # Simulate a Java process actually running from this world's directory.
        with patch("pathlib.Path.read_bytes", return_value=b"/usr/bin/java\\0-jar\\0server.jar\\0"), \\
             patch("pathlib.Path.resolve", return_value=self.root), \\
             patch("os.kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                MinecraftServer(FakeConfig())._check_existing_server()
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
