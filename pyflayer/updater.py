"""Self-updater: installs the newest GitHub release (or `git pull` for a source checkout).

Your data/ folder (world, bot brains, wiki cache), config.json and .venv are never touched.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .config import ROOT, data_dir

DEFAULT_REPO = "MilkdromedaStudios/mcai"
APP_FILES = ("pyflayer", "assets", "tests", "install.sh", "run.sh", "requirements.txt", "README.md", "LICENSE",
             "VERSION", ".github")
UA = {"User-Agent": "pyflayer-updater", "Accept": "application/vnd.github+json"}


def current_version() -> str:
    v = ROOT / "VERSION"
    if v.exists():
        return v.read_text().strip()
    if (ROOT / ".git").exists():
        try:
            out = subprocess.run(["git", "-C", str(ROOT), "describe", "--tags", "--always"], capture_output=True,
                                 text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return "dev"


def _vtuple(v: str):
    nums = re.findall(r"\d+", v.split("-")[0])
    return tuple(int(n) for n in nums[:3]) if nums else (0,)


def _get(url: str, timeout: float = 30) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def latest_release(repo: str = DEFAULT_REPO) -> dict | None:
    try:
        d = json.loads(_get(f"https://api.github.com/repos/{repo}/releases/latest"))
    except Exception:  # noqa: BLE001 - no release yet / offline
        return None
    assets = {a["name"]: a["browser_download_url"] for a in d.get("assets", [])}
    return {"tag": d.get("tag_name", ""), "url": d.get("html_url", ""), "assets": assets,
            "notes": (d.get("body") or "")[:2000]}


def _git(*args, timeout: float = 120):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=timeout)


def git_state() -> dict:
    """Fetch origin/main and report how far behind this checkout is."""
    fetch = _git("fetch", "--quiet", "--tags", "origin", "main", timeout=300)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    behind = _git("rev-list", "--count", "HEAD..origin/main").stdout.strip() if fetch.returncode == 0 else "0"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    return {"fetched": fetch.returncode == 0, "error": fetch.stderr.strip()[-300:], "branch": branch,
            "behind": int(behind or 0), "dirty": dirty}


def check(repo: str = DEFAULT_REPO) -> dict:
    """{'current', 'latest', 'available': bool, 'message'}."""
    if (ROOT / ".git").exists():
        # source install: compare with the newest code on GitHub's main branch (works with or without releases)
        st = git_state()
        cur = current_version()
        if not st["fetched"]:
            return {"current": cur, "latest": None, "available": False,
                    "message": f"couldn't reach GitHub to check for updates ({st['error'] or 'offline?'})"}
        if st["branch"] != "main":
            return {"current": cur, "latest": "main", "available": True,
                    "message": f"you're on branch '{st['branch']}': update switches to the latest main"}
        if st["behind"]:
            return {"current": cur, "latest": "main", "available": True,
                    "message": f"update available: {st['behind']} new change(s) on main"}
        return {"current": cur, "latest": "main", "available": False, "message": f"up to date ({cur})"}
    cur = current_version()
    rel = latest_release(repo)
    if rel is None:
        return {"current": cur, "latest": None, "available": False,
                "message": f"could not find a release of {repo} (offline, or none published yet)"}
    git = (ROOT / ".git").exists()
    avail = (_vtuple(rel["tag"]) > _vtuple(cur)) or (cur == "dev" and not git)
    msg = (f"update available: {cur} -> {rel['tag']}" if avail else f"up to date ({cur}; latest {rel['tag']})")
    return {"current": cur, "latest": rel["tag"], "available": avail, "message": msg, "release": rel}


def update(repo: str = DEFAULT_REPO, progress=print) -> str:
    """Install the latest release in place. Returns a status message (restart the app afterwards)."""
    if (ROOT / ".git").exists():
        progress("downloading the latest code from GitHub ...")
        st = git_state()
        if not st["fetched"]:
            raise RuntimeError(f"couldn't reach GitHub: {st['error'] or 'offline?'}")
        if st["dirty"]:
            # keep any edits the user made to app files, but out of the way of the update
            _git("stash", "push", "-m", "pyflayer-update: local changes")
            progress("your local file changes were saved with 'git stash'")
        if st["branch"] != "main":
            has_main = _git("rev-parse", "--verify", "--quiet", "main").returncode == 0
            out = _git("checkout", "main") if has_main else _git("checkout", "-b", "main", "origin/main")
            if out.returncode != 0:
                raise RuntimeError("couldn't switch to main: " + (out.stderr or out.stdout)[-400:])
        out = _git("merge", "--ff-only", "origin/main")
        if out.returncode != 0:
            raise RuntimeError("your main branch has its own commits; update it yourself with 'git pull' "
                               "(" + (out.stderr or out.stdout)[-300:] + ")")
        for sh in ("install.sh", "run.sh"):
            if (ROOT / sh).exists():
                (ROOT / sh).chmod(0o755)
        _pip_install(progress)
        return f"updated to the latest main ({current_version()}) - restart PyFlayer"
    info = check(repo)
    if not info["available"]:
        return info["message"]
    rel = info["release"]
    tag = rel["tag"]
    name = f"pyflayer-{tag}.tar.gz"
    if name not in rel["assets"]:
        raise RuntimeError(f"release {tag} has no {name}")
    progress(f"downloading {name} ...")
    blob = _get(rel["assets"][name], timeout=300)
    sums_url = rel["assets"].get("SHA256SUMS.txt")
    if sums_url:
        sums = _get(sums_url).decode()
        want = next((line.split()[0] for line in sums.splitlines() if line.strip().endswith(name)), None)
        if want and hashlib.sha256(blob).hexdigest() != want:
            raise RuntimeError("download checksum mismatch - update aborted")
    progress("installing ...")
    with tempfile.TemporaryDirectory(dir=data_dir("updates")) as tmp:
        tar_path = Path(tmp) / name
        tar_path.write_bytes(blob)
        with tarfile.open(tar_path) as tf:
            try:
                tf.extractall(tmp, filter="data")
            except TypeError:
                tf.extractall(tmp)
        src = Path(tmp) / f"pyflayer-{tag}"
        if not (src / "pyflayer").is_dir():
            raise RuntimeError("unexpected package layout")
        backup = data_dir("updates") / f"backup-{current_version()}"
        if backup.exists():
            shutil.rmtree(backup)
        backup.mkdir(parents=True)
        for item in APP_FILES:
            s, d = src / item, ROOT / item
            if not s.exists():
                continue
            if d.exists():
                shutil.move(str(d), str(backup / item))
            if s.is_dir():
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        for sh in ("install.sh", "run.sh"):
            if (ROOT / sh).exists():
                (ROOT / sh).chmod(0o755)
    _pip_install(progress)
    return f"updated to {tag} (previous version saved in data/updates) - restart PyFlayer"


def restart_app(args=("gui",)):
    """Re-launch PyFlayer in place (after an update)."""
    import os
    os.execv(sys.executable, [sys.executable, "-m", "pyflayer", *args])


def _pip_install(progress):
    req = ROOT / "requirements.txt"
    if req.exists():
        progress("updating Python packages ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], timeout=900)
