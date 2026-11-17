"""Offline tests for coordinated specialists and safe startup diagnostics."""
import unittest

from pyflayer.agent.specialists import SpecialistTeam
from pyflayer.server.manager import MinecraftServer
from pyflayer.config import DEFAULTS


class TestSpecialistRouting(unittest.TestCase):
    def test_interpret_json(self):
        self.assertEqual(
            SpecialistTeam.parse_route('Sure: {"kind":"chat","text":"hello"}'),
            {"kind": "chat", "text": "hello"},
        )
        self.assertEqual(
            SpecialistTeam.parse_route('{"kind":"command","text":"!mine iron_ore 2"}')["kind"],
            "command",
        )

    def test_invalid_or_unsupported_routes_fall_back_to_task(self):
        for text in ("", "not json", '{"kind":"teleport","text":"xyz"}'):
            self.assertEqual(SpecialistTeam.parse_route(text)["kind"], "task")

    def test_specialists_have_clear_roles(self):
        self.assertEqual(set(SpecialistTeam.ROLES), {
            "interpreter", "chatter", "planner", "builder", "scout", "escaper",
        })


class TestServerJavaArgs(unittest.TestCase):
    def test_small_heap_uses_safe_defaults(self):
        class FakeConfig:
            def __getitem__(self, key):
                return {"minecraft_version": "26.3", "server": {"memory_mb": 768}}[key]

        s = MinecraftServer(FakeConfig())
        s.java = None
        args = s.jvm_args()
        self.assertIn("-Xmx768M", args)
        self.assertIn("-Xms512M", args)
        self.assertNotIn("-XX:G1HeapRegionSize=8M", args)


if __name__ == "__main__":
    unittest.main()
