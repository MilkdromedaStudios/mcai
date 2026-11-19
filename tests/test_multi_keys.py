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

    def test_cycles_to_next_key_without_waiting(self):
        cfg = {"provider": "groq", "keys": {"groq": "k1, k2, k3"}, "model": "m"}
        llm = L.LLM(cfg)
        used = []

        def make_client():
            key = llm._keys[llm._key_i]

            def create(model, **kw):
                used.append(key)
                if key in ("k1", "k2"):
                    raise Limited("rate limit reached")
                return SimpleNamespace(usage=SimpleNamespace(total_tokens=1), choices=[])
            llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        llm._keys, llm._key_i = ["k1", "k2", "k3"], 0
        llm._make_client = make_client
        make_client()
        llm._chain = lambda first: [m for m in ["m"] if llm._cool.get(m, 0) <= time.time()]
        t = time.time()
        llm._create(messages=[])
        self.assertLess(time.time() - t, 2)
        self.assertEqual(used, ["k1", "k2", "k3"])
        self.assertEqual(llm._key_i, 2)

    def test_litellm_gets_one_deployment_per_key(self):
        from pyflayer.agent.litellm_cfg import build
        conf = {"llm": {"provider": "groq", "keys": {"groq": "g1, g2"}}, "training.local": {"model": "x"}}
        cfg = type("C", (), {"__getitem__": lambda s, k: conf[k], "get": lambda s, k, d=None: conf.get(k, d)})()
        y = build(cfg)
        self.assertIn('api_key: "g1"', y)
        self.assertIn('api_key: "g2"', y)


if __name__ == "__main__":
    unittest.main()
