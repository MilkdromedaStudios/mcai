"""Runs the LiteLLM proxy for you: writes its config from your saved keys, starts it in the background when an AI
setting uses it (provider "litellm", or "litellm" in the emergency fallbacks), restarts it when you change keys,
and stops it when PyFlayer quits. LiteLLM lives in its own virtual environment (.venv-litellm) made by install.sh.
"""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from ..config import ROOT, data_dir

log = logging.getLogger("pyflayer.litellm")


def binary() -> str | None:
    venv = ROOT / ".venv-litellm" / "bin" / "litellm"
    if venv.exists():
        return str(venv)
    return shutil.which("litellm")


def wanted(cfg) -> bool:
    llm = cfg["llm"]
    uses = (llm.get("provider") == "litellm"
            or any((f if isinstance(f, str) else (f or {}).get("provider")) == "litellm"
                   for f in llm.get("emergency_fallbacks") or [])
            or any((o or {}).get("provider") == "litellm" for o in (cfg.get("bot_ai") or {}).values()))
    return uses and (llm.get("litellm") or {}).get("auto_start", True)


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


class LiteLLMProxy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.status = "off"
        self._lock = threading.Lock()
        self._config_text = ""

    @property
    def port(self) -> int:
        return int((self.cfg["llm"].get("litellm") or {}).get("port", 4000))

    def ensure(self, wait: float = 120) -> str:
        """Start (or restart after a key change) if some AI setting uses LiteLLM. Returns a status line."""
        from .litellm_cfg import build
        with self._lock:
            if not wanted(self.cfg):
                return "not needed"
            exe = binary()
            if not exe:
                self.status = "not installed"
                return "LiteLLM is not installed - run ./install.sh --no-setup"
            keys = dict(self.cfg.get("llm.keys") or {})
            if not str(keys.get("litellm", "")).startswith("sk-"):
                # the proxy needs a password; PyFlayer makes one and uses it itself (kept in config.json)
                import secrets
                keys["litellm"] = "sk-pyflayer-" + secrets.token_urlsafe(24)
                self.cfg.set("llm.keys", keys)
                try:
                    self.cfg.save()
                except Exception:  # noqa: BLE001 - tests / read-only config
                    pass
            try:
                text = build(self.cfg)
            except RuntimeError as e:
                self.status = f"no config: {e}"
                return self.status
            path = data_dir() / "litellm_config.yaml"
            running = self.proc is not None and self.proc.poll() is None
            if running and text == self._config_text:
                return "running"
            if running:
                self._stop()                              # keys changed: restart with the new config
            elif _port_open(self.port):
                self.status = "running (started outside PyFlayer)"
                return self.status
            path.write_text(text)
            path.chmod(0o600)                             # contains your API keys
            self._config_text = text
            logf = open(Path(data_dir("logs")) / "litellm.log", "a", encoding="utf-8")  # noqa: SIM115
            self.proc = subprocess.Popen([exe, "--config", str(path), "--host", "127.0.0.1", "--port", str(self.port)],
                                         stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
            self.status = "starting"
            log.info("starting LiteLLM proxy on port %s", self.port)
        end = time.time() + wait
        while time.time() < end:
            if self.proc.poll() is not None:
                self.status = f"crashed (exit {self.proc.returncode}) - see data/logs/litellm.log"
                return self.status
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health/liveliness", timeout=2) as r:
                    if r.status == 200:
                        self.status = "running"
                        log.info("LiteLLM proxy ready")
                        return "running"
            except OSError:
                pass
            time.sleep(1.5)
        self.status = "slow to start (still trying)"
        return self.status

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.status = "off"

    def stop(self):
        with self._lock:
            self._stop()
