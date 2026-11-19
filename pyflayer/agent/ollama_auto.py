"""Zero-setup local AI: pick "ollama" and PyFlayer does the rest.

- No Ollama on this machine? Download the official build into ~/mcai/.ollama (no sudo, no system changes).
- Not running? Start it in the background (one model loaded, one request at a time) and stop it on quit.
- Model "auto" (or blank)? Choose a model that fits this machine's memory / GPU.
- Model missing? Download it, reporting progress.
For Ollama on another PC (a non-local URL) it can't install anything there, but it still pulls the model.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import ROOT, data_dir

log = logging.getLogger("pyflayer.ollama")
HOME = ROOT / ".ollama"
RELEASE = "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-{arch}.tar.zst"
_lock = threading.Lock()
_proc: subprocess.Popen | None = None


# ---------------------------------------------------------------------- choosing a model
def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 4.0


def _vram_gb() -> float:
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(float(x) for x in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def is_raspberry_pi() -> bool:
    try:
        return "raspberry pi" in Path("/proc/device-tree/model").read_text(errors="ignore").lower()
    except OSError:
        return False


def choose_model(ram_gb: float | None = None, vram_gb: float | None = None, pi: bool | None = None) -> str:
    """A small, fast model that can use tools and still leaves room for Minecraft."""
    ram = _ram_gb() if ram_gb is None else ram_gb
    vram = _vram_gb() if vram_gb is None else vram_gb
    if is_raspberry_pi() if pi is None else pi:
        # a Pi CPU manages ~2-4 words/s with a 3B model but ~6-10 with 1.5B - speed matters more here
        return "qwen2.5:1.5b"
    if vram >= 10 or (vram >= 6 and ram >= 16):
        return "qwen2.5:7b"
    if ram >= 24:
        return "qwen2.5:7b"
    if ram >= 6:
        return "qwen2.5:3b"         # e.g. Pi 5 8 GB
    return "qwen2.5:1.5b"           # Pi 4/5 with 4 GB


def is_auto(model: str | None) -> bool:
    return not model or model.strip().lower() in ("auto", "llama3.1")   # llama3.1 = the old preset default


# ---------------------------------------------------------------------- the Ollama program
def _root(base_url: str) -> str:
    return (base_url or "http://127.0.0.1:11434").rstrip("/").removesuffix("/v1")


def is_local(base_url: str) -> bool:
    host = urllib.parse.urlparse(_root(base_url)).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1", "")


def running(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(_root(base_url) + "/api/version", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def binary() -> str | None:
    own = HOME / "bin" / "ollama"
    if own.exists():
        return str(own)
    return shutil.which("ollama")


def install(progress=log.info) -> str:
    """Download the official Ollama build into .ollama (no sudo)."""
    arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(platform.machine().lower())
    if not arch:
        raise RuntimeError(f"Ollama has no build for {platform.machine()} (needs 64-bit)")
    if not shutil.which("zstd") and not shutil.which("unzstd"):
        raise RuntimeError("need the 'zstd' tool to unpack Ollama: sudo apt install zstd  (install.sh does this)")
    dl = data_dir("downloads") / f"ollama-linux-{arch}.tar.zst"
    url = RELEASE.format(arch=arch)
    progress("downloading Ollama (local AI engine, ~1-2 GB with its libraries) ...")
    req = urllib.request.Request(url, headers={"User-Agent": "pyflayer"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dl, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and time.time() - last > 3:
                last = time.time()
                progress(f"downloading Ollama {done * 100 // total}% ({done >> 20}/{total >> 20} MB)")
    HOME.mkdir(parents=True, exist_ok=True)
    progress("unpacking Ollama ...")
    res = subprocess.run(["tar", "--zstd", "-xf", str(dl), "-C", str(HOME)], capture_output=True, text=True)
    dl.unlink(missing_ok=True)
    if res.returncode != 0 or not (HOME / "bin" / "ollama").exists():
        raise RuntimeError("unpacking Ollama failed: " + (res.stderr or res.stdout)[-300:])
    progress("Ollama installed in .ollama")
    return str(HOME / "bin" / "ollama")


def start(base_url: str, progress=log.info, wait: float = 60) -> None:
    """Start our own Ollama server in the background (light settings), unless one is already running."""
    global _proc
    if running(base_url):
        return
    exe = binary() or install(progress)
    host = urllib.parse.urlparse(_root(base_url))
    env = dict(os.environ,
               OLLAMA_HOST=f"127.0.0.1:{host.port or 11434}",
               OLLAMA_NUM_PARALLEL="1",          # one request at a time
               OLLAMA_MAX_LOADED_MODELS="1",     # never keep two models in memory
               OLLAMA_KEEP_ALIVE="15m")
    if exe.startswith(str(HOME)):
        env["OLLAMA_MODELS"] = str(data_dir("ollama-models"))
    logf = open(Path(data_dir("logs")) / "ollama.log", "a", encoding="utf-8")  # noqa: SIM115
    progress("starting the local AI engine (Ollama) ...")
    _proc = subprocess.Popen([exe, "serve"], env=env, stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    end = time.time() + wait
    while time.time() < end:
        if _proc.poll() is not None:
            raise RuntimeError(f"Ollama stopped (exit {_proc.returncode}) - see data/logs/ollama.log")
        if running(base_url):
            return
        time.sleep(1)
    raise RuntimeError("Ollama didn't start in time - see data/logs/ollama.log")


def stop():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


# ---------------------------------------------------------------------- models
def has_model(base_url: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(_root(base_url) + "/api/tags", timeout=10) as r:
            names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
    except OSError:
        return False
    want = model if ":" in model else model + ":latest"
    return want in names or model in names


def pull(base_url: str, model: str, progress=log.info) -> None:
    progress(f"downloading AI model {model} (one time) ...")
    req = urllib.request.Request(_root(base_url) + "/api/pull", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = 0.0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if ev.get("error"):
                raise RuntimeError(f"Ollama couldn't download {model}: {ev['error']}")
            if ev.get("total") and time.time() - last > 3:
                last = time.time()
                progress(f"downloading {model}: {ev.get('completed', 0) * 100 // ev['total']}% "
                         f"of {ev['total'] >> 20} MB")
    progress(f"AI model {model} ready")


def ensure(base_url: str, model: str | None, progress=log.info) -> str:
    """Make the local AI usable: install + start Ollama if needed, choose a model if 'auto', download it.
    Returns the model name to use."""
    with _lock:
        if is_auto(model):
            model = choose_model()
            progress(f"local AI: chose {model} for this machine ({_ram_gb():.0f} GB RAM)")
        if not running(base_url):
            if is_local(base_url):
                start(base_url, progress)
            else:
                raise RuntimeError(f"Ollama at {_root(base_url)} isn't reachable - start Ollama on that PC")
        if not has_model(base_url, model):
            pull(base_url, model, progress)
        return model
