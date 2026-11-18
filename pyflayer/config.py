"""Configuration (JSON file) with sane Raspberry Pi defaults."""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

ROOT = Path(os.environ.get("PYFLAYER_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config.json"

DEFAULTS: dict = {
    "minecraft_version": "26.3",
    "server": {
        "auto_download": True,
        "auto_start": True,          # start the local server when the bot/agent starts
        "host": "127.0.0.1",
        "port": 25565,
        "auto_port_fallback": True,  # if port is busy, choose a free local port instead of crashing
        "memory_mb": 0,              # 0 = auto from system RAM
        "view_distance": 7,          # 7 chunks ~= 112 blocks -> enough for the 100 block scanner
        "simulation_distance": 4,
        "difficulty": "easy",        # easy mode
        "gamemode": "survival",
        "level_seed": "",
        "level_name": "world",
        "motd": "PyFlayer AI training server",
        "max_players": 8,
        "op_bot": True,
        "public_join": True,         # look up and show the public join address
        "upnp": False,               # opt-in: ask your home router to forward the port automatically (UPnP)              # op the bot on the local server (needed for /locate when allow_cheats)
        "extra_properties": {},
    },
    "profiles": ["PyBot"],          # up to 5 bot names that can join the same server
    "bot": {
        "username": "PyBot",
        "view_distance": 7,
        "auto_respawn": True,
        "auto_reconnect": True,
        "allow_dig_in_paths": True,
        "allow_place_in_paths": True,
        "chat_prefix": "",
    },
    "llm": {
        "provider": "custom",        # custom | openai | openrouter | anthropic | gemini | groq | deepseek |
                                     # mistral | together | xai | ollama | lmstudio
        "base_url": "",             # only needed for "custom" (or to override a preset), e.g. http://pc:8000/v1
        "api_key": "",              # or set the provider's env var (OPENAI_API_KEY, OPENROUTER_API_KEY, ...)
        "model": "",
        "models": {
            "chat": "llama-3.1-8b-instant",
            "world": "meta-llama/llama-4-scout-17b-16e-instruct",
            "logic": "openai/gpt-oss-120b",
            "fallback": "qwen/qwen3.8-27b",
        },
        "tool_mode": "auto",         # auto | native | json
        "temperature": 0.3,
        "max_tokens": 400,           # short answers = fast decisions
        "timeout": 120,
        "max_steps_per_task": 0,         # 0 = no step limit (the task time limit still applies)
    },
    "agent": {
        "allow_cheats": False,       # allow /locate etc. on the local server
        "wiki_online": True,
        "listen_to_chat": True,
        "owner": "",                # only obey chat commands from this player (empty = anyone)
    },
    "training": {
        "hours": 7.0,
        "keep_inventory": True,      # gamerule keep_inventory on the local server while training
        "wiki_crawl": True,
        "wiki_rate_per_sec": 1.0,
        "reflect_every_min": 20,
    },
    "update": {
        "repo": "MilkdromedaStudios/mcai",   # GitHub repo whose releases the updater installs
        "check_on_start": True,
    },
    "gui": {
        "compact": "auto",           # auto (small screens) | true | false
        "refresh_ms": 500,
        "minimap": True,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    _lock = threading.Lock()

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self.data = copy.deepcopy(DEFAULTS)
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.data = _merge(DEFAULTS, json.loads(self.path.read_text()))
            except (OSError, ValueError) as e:
                print(f"[config] could not read {self.path}: {e}; using defaults")
        # environment overrides (handy for systemd / quick tests)
        env = os.environ
        if env.get("PYFLAYER_LLM_BASE_URL"):
            self.data["llm"]["base_url"] = env["PYFLAYER_LLM_BASE_URL"]
        if env.get("PYFLAYER_LLM_API_KEY"):
            self.data["llm"]["api_key"] = env["PYFLAYER_LLM_API_KEY"]
        if env.get("PYFLAYER_LLM_PROVIDER"):
            self.data["llm"]["provider"] = env["PYFLAYER_LLM_PROVIDER"]
        if env.get("PYFLAYER_LLM_MODEL"):
            self.data["llm"]["model"] = env["PYFLAYER_LLM_MODEL"]

    def save(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2))
            tmp.replace(self.path)

    def __getitem__(self, k):
        return self.data[k]

    def get(self, dotted: str, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value):
        cur = self.data
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value


def data_dir(*parts) -> Path:
    p = DATA_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
