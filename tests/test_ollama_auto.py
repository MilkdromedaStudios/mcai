"""Local AI auto-setup: model choice, and download of a missing model through the Ollama API."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from pyflayer.agent import ollama_auto as O


class TestChoose(unittest.TestCase):
    def test_fits_the_machine(self):
        self.assertEqual(O.choose_model(8, 0, pi=True), "qwen2.5:1.5b")   # Pi: speed first
        self.assertEqual(O.choose_model(4, 0, pi=False), "qwen2.5:1.5b")
        self.assertEqual(O.choose_model(8, 0, pi=False), "qwen2.5:3b")
        self.assertEqual(O.choose_model(16, 12, pi=False), "qwen2.5:7b")  # PC with a GPU
        self.assertTrue(O.is_auto("auto") and O.is_auto("") and not O.is_auto("qwen2.5:3b"))


class FakeOllama(BaseHTTPRequestHandler):
    models = set()

    def _send(self, body):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send(b'{"version":"x"}')
        self._send(json.dumps({"models": [{"name": m} for m in self.models]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.models.add(body["model"])
        self._send(b'{"status":"pulling","total":100,"completed":50}\n{"status":"success"}\n')

    def log_message(self, *a):
        pass


class TestEnsure(unittest.TestCase):
    def test_pulls_missing_model_on_running_server(self):
        srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_port}/v1"
        said = []
        O.choose_model, real = (lambda *a, **k: "qwen2.5:3b"), O.choose_model
        try:
            self.assertEqual(O.ensure(url, "auto", said.append), "qwen2.5:3b")
        finally:
            O.choose_model = real
            srv.shutdown()
        self.assertIn("qwen2.5:3b", FakeOllama.models)
        self.assertTrue(any("ready" in m for m in said))


if __name__ == "__main__":
    unittest.main()
