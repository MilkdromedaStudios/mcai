"""Offline regression tests for Minecraft's 'FAILED TO BIND TO PORT' startup error."""
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyflayer.config import Config
from pyflayer.server.manager import MinecraftServer


class TestPortFallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.patch = patch("pyflayer.server.manager.server_dir", return_value=self.folder)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.cfg = Config(self.folder / "config.json")

    def _occupied_socket(self):
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        return blocker.getsockname()[1]

    def test_busy_port_moves_to_free_port_and_saves_it(self):
        busy = self._occupied_socket()
        self.cfg.set("server.port", busy)
        self.cfg.set("server.extra_properties.server-port", "12345")
        server = MinecraftServer(self.cfg)
        new_port = server._resolve_port()
        self.assertNotEqual(new_port, busy)
        self.assertTrue(MinecraftServer._port_available(new_port))
        self.assertEqual(new_port, self.cfg.get("server.port"))
        self.assertEqual(new_port, Config(self.cfg.path).get("server.port"))
        self.assertIn(f"server-port={new_port}\n", (self.folder / "server.properties").read_text())
        self.assertNotIn("server-port=12345\n", (self.folder / "server.properties").read_text())
        self.assertIn(str(busy), server.port_notice)
        self.assertIn(str(new_port), server.port_notice)

    def test_disabled_fallback_reports_port_owner_command(self):
        busy = self._occupied_socket()
        self.cfg.set("server.port", busy)
        self.cfg.set("server.auto_port_fallback", False)
        server = MinecraftServer(self.cfg)
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            server._resolve_port()
        self.assertEqual(self.cfg.get("server.port"), busy)

    def test_available_port_is_not_changed(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.cfg.set("server.port", port)
        self.assertEqual(MinecraftServer(self.cfg)._resolve_port(), port)


if __name__ == "__main__":
    unittest.main()
