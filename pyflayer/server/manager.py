"""Download, configure and run a vanilla Minecraft server (offline / "cracked" mode).

Also provisions the *game data* the bot needs (packet ids, block states, registries, recipes,
tags, loot tables, item components) by running the official data generator from the server jar.
Nothing is hardcoded per version: point it at another release id and it regenerates everything.
"""
from __future__ import annotations

import collections
import errno
import hashlib
import io
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from ..config import Config, data_dir

log = logging.getLogger("pyflayer.server")

MANIFEST_URL = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"
ADOPTIUM_URL = "https://api.adoptium.net/v3/binary/latest/{major}/ga/linux/{arch}/jre/hotspot/normal/eclipse"
USER_AGENT = "pyflayer/1.0 (+https://github.com)"


# --------------------------------------------------------------------------- helpers
def _http_get(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download(url: str, dest: Path, sha1: str | None = None, progress=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    h = hashlib.sha1()
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress and time.time() - last > 0.5:
                last = time.time()
                progress(done, total)
    if sha1 and h.hexdigest() != sha1:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}")
    tmp.replace(dest)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def total_ram_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 4096


def auto_server_memory_mb() -> int:
    ram = total_ram_mb()
    if ram <= 2200:
        return 1024
    if ram <= 4400:
        return 1792
    if ram <= 8800:
        return 3072
    return 4096


# --------------------------------------------------------------------------- version metadata
def resolve_version(version: str) -> dict:
    manifest = json.loads(_http_get(MANIFEST_URL))
    if version in ("latest", "release"):
        version = manifest["latest"]["release"]
    for v in manifest["versions"]:
        if v["id"] == version:
            return json.loads(_http_get(v["url"]))
    raise RuntimeError(f"Minecraft version {version!r} not found in Mojang's manifest")


def server_dir(version: str) -> Path:
    return data_dir("server", version)


def ensure_server_jar(version: str, progress=None) -> tuple[Path, int]:
    """Returns (jar path, required java major)."""
    sdir = server_dir(version)
    jar = sdir / "server.jar"
    meta_path = sdir / "version_meta.json"
    if jar.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if _sha1(jar) == meta["sha1"]:
            return jar, meta["java"]
    log.info("Resolving Minecraft %s from Mojang...", version)
    info = resolve_version(version)
    srv = info["downloads"]["server"]
    java_major = int(info.get("javaVersion", {}).get("majorVersion", 21))
    log.info("Downloading server.jar (%.1f MB)...", srv["size"] / 1e6)
    _download(srv["url"], jar, srv["sha1"], progress)
    meta_path.write_text(json.dumps({"sha1": srv["sha1"], "java": java_major, "id": info["id"]}))
    return jar, java_major


# --------------------------------------------------------------------------- java runtime
def _java_major(java: str) -> int:
    try:
        env = dict(os.environ)
        env.pop("JAVA_TOOL_OPTIONS", None)
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=30, env=env)
        m = re.search(r'version "(\d+)(?:\.(\d+))?', out.stderr + out.stdout)
        if m:
            major = int(m.group(1))
            return int(m.group(2)) if major == 1 and m.group(2) else major
    except (OSError, subprocess.SubprocessError):
        pass
    return 0


def find_java(major: int, auto_download: bool = True, progress=None) -> str:
    candidates = []
    rt = data_dir("runtime")
    for p in sorted(rt.glob(f"java-{major}*/**/bin/java")):
        candidates.append(str(p))
    if os.environ.get("JAVA_HOME"):
        candidates.append(os.path.join(os.environ["JAVA_HOME"], "bin", "java"))
    if shutil.which("java"):
        candidates.append(shutil.which("java"))
    candidates += [str(p) for p in Path("/usr/lib/jvm").glob("*/bin/java")]
    for c in candidates:
        if os.path.exists(c) and _java_major(c) >= major:
            return c
    if not auto_download:
        raise RuntimeError(f"Java {major}+ is required but was not found")
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "aarch64", "arm64": "aarch64",
            "armv7l": "arm", "armv6l": "arm", "ppc64le": "ppc64le", "s390x": "s390x"}.get(machine, machine)
    if arch == "arm":
        raise RuntimeError(
            f"Java {major} is not published for 32-bit ARM. Install the 64-bit Raspberry Pi OS "
            "(aarch64) to run the local server, or point the bot at a server on another machine.")
    url = ADOPTIUM_URL.format(major=major, arch=arch)
    log.info("Downloading Temurin Java %s JRE for %s ...", major, arch)
    archive = rt / f"java-{major}.tar.gz"
    _download(url, archive, None, progress)
    dest = rt / f"java-{major}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(dest)
    archive.unlink(missing_ok=True)
    for p in dest.glob("**/bin/java"):
        p.chmod(0o755)
        return str(p)
    raise RuntimeError("downloaded Java archive did not contain bin/java")


def _java_env() -> dict:
    env = dict(os.environ)
    env.pop("JAVA_TOOL_OPTIONS", None)  # avoid noisy/foreign options leaking into the server
    return env


# --------------------------------------------------------------------------- game data
DATA_SUBDIRS = ("recipe", "tags/block", "tags/item", "tags/entity_type", "loot_table/blocks",
                "loot_table/entities", "dimension_type", "worldgen/structure", "worldgen/biome")


def gamedata_dir(version: str) -> Path:
    return data_dir("gamedata", version)


def ensure_gamedata(version: str, java: str | None = None, progress=None) -> Path:
    """Generate reports + extract datapack files from the server jar (cached per version)."""
    gd = gamedata_dir(version)
    marker = gd / "complete.json"
    if marker.exists():
        return gd
    jar, java_major = ensure_server_jar(version, progress)
    java = java or find_java(java_major, progress=progress)
    work = gd / "gen"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    log.info("Running the data generator (one time, ~1 min on a Pi)...")
    cmd = [java, "-Xmx1G", "-DbundlerMainClass=net.minecraft.data.Main", "-jar", str(jar),
           "--reports", "--output", str(work / "out")]
    res = subprocess.run(cmd, cwd=work, capture_output=True, text=True, env=_java_env(), timeout=1800)
    reports = work / "out" / "reports"
    if res.returncode != 0 or not (reports / "packets.json").exists():
        raise RuntimeError("data generator failed:\n" + (res.stdout + res.stderr)[-3000:])
    for name in ("packets.json", "blocks.json", "registries.json"):
        shutil.copy(reports / name, gd / name)
    # item default components (tools, food, stack sizes ...) -> one compact json
    comps = {}
    for p in (reports / "minecraft" / "components" / "item").glob("*.json"):
        try:
            comps["minecraft:" + p.stem] = json.loads(p.read_text()).get("components", {})
        except ValueError:
            pass
    (gd / "item_components.json").write_text(json.dumps(comps))
    # datapack content lives in the inner versioned jar
    with zipfile.ZipFile(jar) as outer:
        inner_name = next((n for n in outer.namelist()
                           if n.startswith("META-INF/versions/") and n.endswith(".jar")), None)
        inner_bytes = outer.read(inner_name) if inner_name else None
        version_json = outer.read("version.json") if "version.json" in outer.namelist() else None
    src = zipfile.ZipFile(io.BytesIO(inner_bytes)) if inner_bytes else zipfile.ZipFile(jar)
    with src:
        if version_json is None:
            version_json = src.read("version.json")
        for n in src.namelist():
            if not n.startswith("data/minecraft/") or n.endswith("/"):
                continue
            rel = n[len("data/minecraft/"):]
            if any(rel.startswith(s + "/") for s in DATA_SUBDIRS):
                dest = gd / "data" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read(n))
    (gd / "version.json").write_bytes(version_json)
    shutil.rmtree(work, ignore_errors=True)
    marker.write_text(json.dumps({"generated": time.time()}))
    log.info("Game data ready in %s", gd)
    return gd


# --------------------------------------------------------------------------- server process
class MinecraftServer:
    """Runs the vanilla server as a child process with a console we can write to."""

    def __init__(self, config: Config):
        self.cfg = config
        self.version = config["minecraft_version"]
        self.proc: subprocess.Popen | None = None
        self.ready = threading.Event()
        self.lines: collections.deque[str] = collections.deque(maxlen=1500)
        self.listeners: list = []
        self._reader: threading.Thread | None = None
        self.java = None
        self.status = "stopped"
        self._want_running = False
        self._ever_ready = False  # don't restart processes that never completed startup
        self._watchdog: threading.Thread | None = None
        self._start_lock = threading.RLock()
        self._owner_handle = None
        self.port_notice = ""

    @property
    def dir(self) -> Path:
        return server_dir(self.version)

    def prepare(self, progress=None):
        self.status = "downloading"
        jar, major = ensure_server_jar(self.version, progress)
        self.java = find_java(major, progress=progress)
        ensure_gamedata(self.version, self.java, progress)
        self._write_files()
        self.status = "prepared"
        return jar

    def _write_files(self):
        s = self.cfg["server"]
        (self.dir / "eula.txt").write_text(
            "# Accepted by the pyflayer installer on behalf of the user (https://aka.ms/MinecraftEULA)\neula=true\n")
        props_path = self.dir / "server.properties"
        props: dict[str, str] = {}
        if props_path.exists():
            for line in props_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    props[k.strip()] = v.strip()
        props.update({
            "online-mode": "false",                 # cracked/offline server
            "white-list": "false",                  # 26.x defaults to a whitelist; bots must be able to join
            "enforce-whitelist": "false",
            "enforce-secure-profile": "false",      # unsigned chat from offline bots
            "server-port": str(s["port"]),
            "server-ip": "",
            "view-distance": str(s["view_distance"]),
            "simulation-distance": str(s["simulation_distance"]),
            "difficulty": s["difficulty"],
            "gamemode": s["gamemode"],
            "level-name": s["level_name"],
            "level-seed": str(s["level_seed"]),
            "motd": s["motd"],
            "max-players": str(s["max_players"]),
            "spawn-protection": "0",
            "allow-flight": "true",                 # avoids false "flying" kicks during pillaring
            "sync-chunk-writes": "false",           # big win on SD cards
            "network-compression-threshold": "-1" if s["host"] in ("127.0.0.1", "localhost") else "256",
            "entity-broadcast-range-percentage": "75",
            "max-tick-time": "-1",                   # do not kill the server on a slow Pi tick
            "pause-when-empty-seconds": "60",
            "enable-command-block": "false",
            "log-ips": "false",
        })
        props.update({k: str(v) for k, v in s.get("extra_properties", {}).items()})
        # This port must match the bot connection and the app\'s Join address,
        # even if an older extra_properties value specified a different port.
        props["server-port"] = str(s["port"])
        props_path.write_text("#Minecraft server properties (managed by pyflayer)\n" +
                              "".join(f"{k}={v}\n" for k, v in sorted(props.items())))

    def jvm_args(self) -> list[str]:
        # Java's default G1 tuning is safer than forcing 8 MB regions and fixed
        # young-generation percentages on a memory-constrained Raspberry Pi.
        mem = int(self.cfg["server"]["memory_mb"] or auto_server_memory_mb())
        mem = max(512, mem)
        args = [f"-Xms{min(mem, 512)}M", f"-Xmx{mem}M",
                "-XX:+UseG1GC", "-XX:+UseStringDeduplication", "-Dfile.encoding=UTF-8"]
        if self.java and _java_major(self.java) >= 24:
            # Java 25 JNA emits a noisy *non-fatal* native-access warning.
            args.append("--enable-native-access=ALL-UNNAMED")
        return args

    def _acquire_owner(self):
        """Keep other PyFlayer processes from launching or killing our server."""
        import fcntl
        handle = (self.dir / ".pyflayer-server.lock").open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                "Another PyFlayer instance is already managing this Minecraft server. "
                "Close the other app before launching another server (worlds are preserved)."
            ) from exc
        self._owner_handle = handle

    def _release_owner(self):
        handle = self._owner_handle
        self._owner_handle = None
        if handle is not None:
            handle.close()  # closing the FD releases its flock on Linux

    @staticmethod
    def _port_available(port: int) -> bool:
        """Probe the wildcard bind used by the vanilla server, not localhost only."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("0.0.0.0", int(port)))
            return True
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return False
            raise RuntimeError(f"Cannot check Minecraft port {port}: {exc}") from exc

    def _resolve_port(self) -> int:
        """Move to a free local port instead of launching a server that cannot bind.

        This never kills or attaches to the process occupying the original port.
        The chosen port is saved to config.json and reflected in server.properties,
        so the bot and Minecraft UI always use the same address.
        """
        requested = int(self.cfg["server"]["port"])
        if not 1 <= requested <= 65535:
            raise ValueError(f"Minecraft server port must be 1..65535, got {requested}")
        if self._port_available(requested):
            return requested
        if not self.cfg["server"].get("auto_port_fallback", True):
            raise RuntimeError(
                f"Minecraft port {requested} is already in use. "
                f"Find the owner with: sudo ss -ltnp '( sport = :{requested} )' "
                "or turn on server.auto_port_fallback."
            )
        # Prefer the nearest higher port, then the conventional local range.
        alternatives = list(range(requested + 1, min(65536, requested + 51)))
        alternatives += list(range(25565, 25616))
        for candidate in dict.fromkeys(alternatives):
            if candidate == requested:
                continue
            if self._port_available(candidate):
                self.cfg.set("server.port", candidate)
                self.cfg.save()
                self._write_files()
                self.port_notice = (
                    f"Port {requested} was occupied. Started Minecraft on {candidate} "
                    f"instead. Join localhost:{candidate} (or use the address shown in the app)."
                )
                log.warning(self.port_notice)
                return candidate
        raise RuntimeError(
            f"Minecraft port {requested} is occupied and no fallback port is available. "
            f"Run: sudo ss -ltnp '( sport = :{requested} )' "
            "to identify the other service. PyFlayer will not kill it."
        )

    def _check_existing_server(self):
        """Never SIGTERM another server just because an old server.pid exists."""
        pid_file = self.dir / "server.pid"
        try:
            pid = int(pid_file.read_text())
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            cwd = Path(f"/proc/{pid}/cwd").resolve()
        except (OSError, ValueError):
            return
        if not argv:
            return
        java = Path(os.fsdecode(argv[0])).name.lower()
        is_minecraft = "java" in java and any(arg.endswith(b"server.jar") for arg in argv)
        if is_minecraft and cwd == self.dir.resolve():
            raise RuntimeError(
                f"Minecraft server already running as PID {pid} in {self.dir}. "
                "PyFlayer will not terminate it automatically. Close the previous "
                "app/server gracefully before restarting. Do not delete your world."
            )

    def start(self, progress=None, wait: bool = True, timeout: float = 600) -> bool:
        with self._start_lock:
            if self.proc and self.proc.poll() is None:
                return True
            if self._reader and self._reader.is_alive():
                # A previous exited process may still be releasing its file lock.
                self._reader.join(timeout=2)
            # Ownership is acquired before checking a stale PID to prevent two
            # apps from racing to launch servers into the same world.
            self._acquire_owner()
            try:
                jar = self.prepare(progress)
                self._check_existing_server()
                self._resolve_port()
                cmd = [self.java, *self.jvm_args(), "-jar", str(jar), "--nogui"]
                log.info("Starting Minecraft server %s ...", self.version)
                self.ready.clear()
                self.lines.clear()
                self._ever_ready = False
                self.status = "starting"
                self.proc = subprocess.Popen(cmd, cwd=self.dir, stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, bufsize=1, env=_java_env())
                (self.dir / "server.pid").write_text(str(self.proc.pid))
                self._want_running = True
                self._reader = threading.Thread(target=self._read_output,
                                                name="mc-server-out", daemon=True)
                self._reader.start()
                if not self._watchdog or not self._watchdog.is_alive():
                    self._watchdog = threading.Thread(target=self._watch,
                                                      name="mc-server-watchdog", daemon=True)
                    self._watchdog.start()
            except BaseException:
                self._release_owner()
                raise
            # Do not keep the startup lock during the potentially long world load.
        if wait:
            return self.wait_ready(timeout)
        return True

    def wait_ready(self, timeout: float = 600) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.ready.wait(1.0):
                return True
            if self.proc and self.proc.poll() is not None:
                # Give the output reader a moment to capture the FIRST error.
                if self._reader:
                    self._reader.join(timeout=2)
                self._want_running = False
                self.status = "crashed"
                tail = list(self.lines)
                important = [ln for ln in tail if re.search(
                    r"error|exception|failed|fatal|cannot|unable|outofmemory|killed|port.*bind|eula",
                    ln, re.I) and "restricted method" not in ln.lower()]
                detail = "\n".join((important[:16] + ["--- last console lines ---"] + tail[-65:]))
                raise RuntimeError(
                    f"Minecraft server exited during startup (exit code {self.proc.returncode}).\n"
                    f"Full output: {self.dir / 'server-console.log'}\n"
                    f"Vanilla server log: {self.dir / 'logs' / 'latest.log'}\n"
                    "The Java native-access warning alone is not a startup failure.\n" + detail)
        return False

    def _read_output(self):
        proc = self.proc
        # Persistent log includes the BEGINNING of startup, not just the final
        # "Stopping server" lines. Never overwrite an earlier crash diagnosis.
        log_path = self.dir / "server-console.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as output:
            output.write("\n=== Minecraft server launch " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
            output.flush()
            for line in proc.stdout:
                line = line.rstrip()
                self.lines.append(line)
                self.line_total = getattr(self, "line_total", 0) + 1
                output.write(line + "\n")
                output.flush()
                if ("Done (" in line and "For help" in line) or re.search(
                        r"Done \([^)]*\)!?\s*(?:For help|Type /help)", line):
                    self.ready.set()
                    self._ever_ready = True
                    self.status = "running"
                    log.info("Server ready")
                for fn in list(self.listeners):
                    try:
                        fn(line)
                    except Exception:  # noqa: BLE001
                        pass
            output.write("=== Process exit: " + str(proc.wait()) + " ===\n")
        if self.proc is proc:
            if not self._ever_ready and self._want_running:
                self.status = "crashed"
            else:
                self.status = "stopped"
            self.ready.clear()
            self._release_owner()

    def _watch(self):
        """Restart the server if it crashes while we still want it running."""
        crashes = 0
        while True:
            time.sleep(5)
            if self._want_running and self.proc and self.proc.poll() is not None:
                if not self._ever_ready:
                    # Startup error: do not repeatedly launch/kill the same world.
                    self._want_running = False
                    continue
                crashes += 1
                log.warning("Minecraft server exited (code %s); restarting (%d)", self.proc.returncode, crashes)
                if crashes > 5:
                    log.error("Server keeps crashing; giving up")
                    self._want_running = False
                    continue
                try:
                    self.start(wait=False)
                except Exception:  # noqa: BLE001
                    log.exception("restart failed")

    def command(self, cmd: str):
        if self.proc and self.proc.poll() is None and self.proc.stdin:
            try:
                self.proc.stdin.write(cmd.lstrip("/") + "\n")
                self.proc.stdin.flush()
            except OSError:
                pass

    def stop(self, timeout: float = 60):
        self._want_running = False
        if not self.proc or self.proc.poll() is not None:
            return
        log.info("Stopping server...")
        self.command("stop")
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.status = "stopped"

    @property
    def running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)
