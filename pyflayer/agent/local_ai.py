"""Local AI for overnight training (Ollama / LM Studio / any OpenAI-compatible server on your network).

Tuned for long unattended runs: small fast models, one request at a time, short answers, a pause between
requests, and (for Ollama) a capped copy of the model that uses only part of the CPU threads / GPU layers,
so the machine stays cool and usable all night.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

from .llm import PROVIDERS

log = logging.getLogger("pyflayer.local_ai")

DEFAULTS = {
    "provider": "ollama",          # ollama | lmstudio | custom
    "base_url": "",                # blank = provider default (http://localhost:11434/v1 for Ollama)
    "model": "auto",               # auto = chosen for this machine's RAM/GPU and downloaded automatically
    "cpu_threads": 0,              # 0 = all cores but one (one stays free for Minecraft / the desktop)
    "gpu_layers": -1,              # -1 = let Ollama decide, 0 = CPU only, N = only N layers on the GPU
    "context": 2048,               # smaller context = less memory and much faster answers on a Pi
    "max_tokens": 300,
    "min_interval_s": 1.5,         # pause between requests so the CPU/GPU never runs flat out
}


def settings(cfg) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg.get("training.local") or {}).items() if v not in (None, "")})
    return out


def _ollama_root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def prepare_ollama(base_url: str, model: str, threads: int, gpu_layers: int, ctx: int) -> str:
    """Make a resource-capped copy of the model ("pyflayer-<model>") so every request uses the limits.
    Returns the model name to use (the original one if Ollama refuses)."""
    name = "pyflayer-" + model.replace(":", "-").replace("/", "-")
    params = {"num_thread": threads, "num_ctx": ctx}
    if gpu_layers >= 0:
        params["num_gpu"] = gpu_layers
    root = _ollama_root(base_url)
    bodies = [{"model": name, "from": model, "parameters": params, "stream": False},          # Ollama >= 0.5
              {"name": name, "stream": False,                                                 # older Ollama
               "modelfile": f"FROM {model}\n" + "".join(f"PARAMETER {k} {v}\n" for k, v in params.items())}]
    for body in bodies:
        try:
            req = urllib.request.Request(root + "/api/create", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
            return name
        except Exception as e:  # noqa: BLE001
            log.debug("ollama create (%s): %s", list(body)[0], e)
    log.warning("couldn't make a capped copy of %s in Ollama; using it as-is (limits: pacing + 1 request at a time)",
                model)
    return model


def llm_config(cfg, progress=log.info) -> dict:
    """The llm config dict for overnight training on a local AI."""
    s = settings(cfg)
    prov = s["provider"]
    base = s["base_url"] or PROVIDERS.get(prov, PROVIDERS["custom"])["base_url"]
    threads = int(s["cpu_threads"]) or max(1, (os.cpu_count() or 2) - 1)
    model = s["model"]
    if prov == "ollama":
        from . import ollama_auto
        model = ollama_auto.ensure(base, model, progress=progress)     # install/start/choose/download as needed
        model = prepare_ollama(base, model, threads, int(s["gpu_layers"]), int(s["context"]))
    llm = dict(cfg["llm"])
    llm.update({"provider": prov, "base_url": base, "model": model, "api_key": "",
                "backup_models": [], "pool": [], "max_tokens": int(s["max_tokens"]), "timeout": 300,
                "local_limits": {"min_interval_s": float(s["min_interval_s"]), "max_tokens": int(s["max_tokens"])}})
    return llm
