"""All models rate-limited -> wait and retry instead of failing."""
import time
import unittest
from types import SimpleNamespace

from pyflayer.agent import llm as llm_mod


class Limited(Exception):
    status_code = 429
    response = SimpleNamespace(status_code=429, headers={"retry-after": "1"})


class TestRateLimitWait(unittest.TestCase):
    def test_waits_then_succeeds(self):
        calls = []

        def create(model, **kw):
            calls.append(model)
            if len(calls) <= 2:
                raise Limited("rate limit reached, try again in 1s")
            return SimpleNamespace(usage=SimpleNamespace(total_tokens=1), choices=[])

        l = llm_mod.LLM({"provider": "custom", "base_url": "http://x/v1", "api_key": "k", "model": "a"})
        l.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        l._chain = lambda first: [m for m in ("a", "b") if l._cool.get(m, 0) <= time.time()]
        waits = []
        l.on_wait = waits.append
        t = time.time()
        l._create(messages=[])
        self.assertTrue(waits)
        self.assertEqual(len(calls), 3)
        self.assertLess(time.time() - t, 15)

    def test_can_be_cancelled(self):
        l = llm_mod.LLM({"provider": "custom", "base_url": "http://x/v1", "api_key": "k", "model": "a"})
        l._chain = lambda first: []
        l._cool = {"a": time.time() + 999}
        l.cancelled = True
        with self.assertRaises(RuntimeError):
            l._create(messages=[])


if __name__ == "__main__":
    unittest.main()
