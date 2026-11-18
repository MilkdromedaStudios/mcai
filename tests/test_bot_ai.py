"""Each provider keeps its own key; each bot can have its own AI."""
import json
import tempfile
import unittest
from pathlib import Path

from pyflayer.agent.llm import bot_llm_config, resolve
from pyflayer.config import Config


class TestKeys(unittest.TestCase):
    def test_old_single_key_is_kept_for_its_provider(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps({"llm": {"provider": "groq", "api_key": "gsk_x"}}))
            cfg = Config(p)
            self.assertEqual(cfg.get("llm.keys"), {"groq": "gsk_x"})

    def test_switching_provider_uses_that_providers_key(self):
        llm = {"provider": "groq", "api_key": "gsk_x", "keys": {"groq": "gsk_x", "gemini": "AIza_y"}}
        self.assertEqual(resolve(llm)["api_key"], "gsk_x")
        self.assertEqual(resolve(dict(llm, provider="gemini", api_key=""))["api_key"], "AIza_y")

    def test_bot_override_never_borrows_another_providers_key(self):
        llm = {"provider": "groq", "api_key": "gsk_x", "keys": {"groq": "gsk_x"}, "model": "m"}
        cfg = bot_llm_config(llm, {"provider": "openrouter", "model": "x:free"})
        self.assertEqual(cfg["provider"], "openrouter")
        self.assertEqual(cfg["model"], "x:free")
        self.assertEqual(resolve(dict(cfg, keys={}))["api_key"], "")
        self.assertEqual(bot_llm_config(llm, None), llm)
        self.assertEqual(bot_llm_config(llm, {"provider": ""}), llm)


if __name__ == "__main__":
    unittest.main()
