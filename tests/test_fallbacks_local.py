"""Emergency fallback AIs, local-AI pacing, and the LiteLLM config."""
import time
import unittest
from types import SimpleNamespace

from pyflayer.agent import llm as L
from pyflayer.agent.litellm_cfg import build


class Limited(Exception):
    status_code = 429
    response = SimpleNamespace(status_code=429, headers={})


def fake_client(fn):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fn)))


def ok(model, **kw):
    return SimpleNamespace(usage=SimpleNamespace(total_tokens=1), choices=[], model=model)


class TestEmergency(unittest.TestCase):
    def test_rate_limited_main_ai_uses_fallback_at_once(self):
        cfg = {"provider": "groq", "api_key": "g", "keys": {"groq": "g", "openrouter": "o"}, "model": "a",
               "emergency_fallbacks": ["openrouter"]}
        main = L.LLM(cfg)
        main.client = fake_client(lambda model, **kw: (_ for _ in ()).throw(Limited("rate limit")))
        main._chain = lambda first: ["a"] if not main._cool else []
        fb = L.LLM(L.bot_llm_config(cfg, {"provider": "openrouter"}))
        fb.client = fake_client(ok)
        fb._chain = lambda first: [first]
        main._fb = {"openrouter:": fb}
        t = time.time()
        r = main._create(messages=[])
        self.assertLess(time.time() - t, 2)            # no waiting
        self.assertIn("emergency", main.last_model)
        self.assertEqual(r.usage.total_tokens, 1)


class TestLocalPacing(unittest.TestCase):
    def test_one_request_at_a_time_with_gap(self):
        cfg = {"provider": "ollama", "model": "m", "local_limits": {"min_interval_s": 0.3, "max_tokens": 50}}
        llm = L.LLM(cfg)
        seen = []
        llm.client = fake_client(lambda model, **kw: (seen.append((time.time(), kw["max_tokens"])), ok(model))[1])
        llm._chain = lambda first: [first]
        llm._create(messages=[], max_tokens=400)
        llm._create(messages=[], max_tokens=400)
        self.assertEqual([m for _, m in seen], [50, 50])     # answers capped for speed
        self.assertGreaterEqual(seen[1][0] - seen[0][0], 0.29)


class TestLiteLLM(unittest.TestCase):
    def test_config_lists_providers_and_fallbacks(self):
        cfg = SimpleNamespace(data={})
        conf = {"llm": {"provider": "groq", "keys": {"groq": "gk", "gemini": "gm"}, "model": ""},
                "training.local": {"model": "qwen2.5:3b"}}
        cfg = type("C", (), {"__getitem__": lambda s, k: conf[k], "get": lambda s, k, d=None: conf.get(k, d)})()
        y = build(cfg)
        self.assertIn('model: "groq/llama-3.3-70b-versatile"', y)
        self.assertIn('model: "gemini/gemini-2.5-flash"', y)
        self.assertIn('ollama_chat/qwen2.5:3b', y)
        self.assertIn("fallbacks", y)

    def test_local_only_and_master_key(self):
        conf = {"llm": {"provider": "litellm", "keys": {"litellm": "sk-pyflayer-x"}}, "training.local": {}}
        cfg = type("C", (), {"__getitem__": lambda s, k: conf[k], "get": lambda s, k, d=None: conf.get(k, d)})()
        y = build(cfg)
        self.assertIn("model_name: pyflayer\n", y)            # local Ollama becomes the main model
        self.assertIn('master_key: "sk-pyflayer-x"', y)
        self.assertNotIn("litellm/", y)


if __name__ == "__main__":
    unittest.main()
