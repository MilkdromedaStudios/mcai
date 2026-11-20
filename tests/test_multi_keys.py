"""Many API keys for one provider: a rate-limited key hands over to the next one at once."""
import time
import unittest
from types import SimpleNamespace

from pyflayer.agent import llm as L


class Limited(Exception):
    status_code = 429
    response = SimpleNamespace(status_code=429, headers={})


class TestMultiKeys(unittest.TestCase):
    def test_split(self):
        self.assertEqual(L.split_keys("a, b\nc;a"), ["a", "b", "c"])
        self.assertEqual(L.split_keys(["x", "y"]), ["x", "y"])
        self.assertEqual(L.keys_for({"provider": "groq", "keys": {"groq": "k1,k2"}}), ["k1", "k2"])

    def _llm(self, cfg, behaviour):
        llm = L.LLM(cfg)
        used = []

        def maybe_client():
            slot = llm.slots[llm._slot_i]

            def create(model, **kw):
                used.append((slot["provider"], slot["key"]))
                err = behaviour.get(slot["key"])
                if err:
                    raise err
                return SimpleNamespace(usage=SimpleNamespace(total_tokens=1), choices=[])
            llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm._maybe_client = maybe_client
        maybe_client()
        llm._chain = lambda first: [m for m in [first or "m"] if llm._cool.get(m, 0) <= time.time()]
        return llm, used

    def test_cycles_to_next_key_without_waiting(self):
        cfg = {"provider": "groq", "keys": {"groq": "k1, k2, k3"}, "model": "m"}
        llm, used = self._llm(cfg, {"k1": Limited("rate limit"), "k2": Limited("rate limit")})
        t = time.time()
        llm._create(messages=[])
        self.assertLess(time.time() - t, 2)
        self.assertEqual([k for _, k in used], ["k1", "k2", "k3"])

    def test_cycles_across_providers_in_tab_order(self):
        class BadKey(Exception):
            status_code = 401
            response = SimpleNamespace(status_code=401, headers={})
        cfg = {"pool": [{"provider": "groq", "key": "g1", "model": "llama"},
                        {"provider": "gemini", "key": "m1", "model": "gemini-2.5-flash"},
                        {"provider": "openrouter", "key": "o1", "model": "x:free"}]}
        llm, used = self._llm(cfg, {"g1": Limited("rate limit"), "m1": BadKey("invalid api key")})
        llm._create(messages=[])
        self.assertEqual(used, [("groq", "g1"), ("gemini", "m1"), ("openrouter", "o1")])
        self.assertEqual(llm.cfg["provider"], "openrouter")
        self.assertIn("API 3/3", llm.slot_label)

    def test_bot_override_ignores_pool(self):
        cfg = {"pool": [{"provider": "groq", "key": "g"}], "keys": {"gemini": "m"}}
        b = L.LLM(L.bot_llm_config(cfg, {"provider": "gemini"}))
        self.assertEqual([s["provider"] for s in b.slots], ["gemini"])

    def test_litellm_gets_one_deployment_per_key(self):
        from pyflayer.agent.litellm_cfg import build
        conf = {"llm": {"provider": "groq", "keys": {"groq": "g1, g2"}}, "training.local": {"model": "x"}}
        cfg = type("C", (), {"__getitem__": lambda s, k: conf[k], "get": lambda s, k, d=None: conf.get(k, d)})()
        y = build(cfg)
        self.assertIn('api_key: "g1"', y)
        self.assertIn('api_key: "g2"', y)


if __name__ == "__main__":
    unittest.main()
