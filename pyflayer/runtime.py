"""Wires config + server + game data + bots (up to 5 profiles) + brains + wiki + LLM + agents + trainers."""
from __future__ import annotations

import logging
import logging.handlers
import re
import socket
import threading
import time

from .config import Config, data_dir
from .server.network import lan_ip, public_ip, upnp_forward

log = logging.getLogger("pyflayer")
MAX_BOTS = 5


def setup_logging(level=logging.INFO, console: bool = True):
    root = logging.getLogger()
    if getattr(root, "_pyflayer", False):
        return
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    fh = logging.handlers.RotatingFileHandler(data_dir("logs") / "pyflayer.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    root.addHandler(fh)
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(level)
        root.addHandler(ch)
    root._pyflayer = True


class Session:
    """One bot profile: its own connection, brain (memory/skills), agent and trainer."""

    def __init__(self, name: str):
        self.name = name
        self.bot = None
        self.brain = None
        self.agent = None
        self.trainer = None


class Runtime:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.gd = None
        self.server = None
        self.wiki = None
        self.llm = None
        self.sessions: dict[str, Session] = {}
        self.current = self.profiles()[0]
        self.status = "idle"
        self.progress_text = ""
        self.public_ip: str | None = None
        self.upnp_status = ""
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ profiles
    def profiles(self) -> list[str]:
        names = [p for p in self.cfg.get("profiles", []) if p] or [self.cfg["bot"]["username"]]
        return names[:MAX_BOTS]

    def add_profile(self, name: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_]", "", name)[:16]
        if len(name) < 3:
            raise ValueError("bot names need 3-16 letters, digits or _")
        names = self.profiles()
        if name not in names:
            if len(names) >= MAX_BOTS:
                raise ValueError(f"maximum {MAX_BOTS} bots per server")
            names.append(name)
            self.cfg.set("profiles", names)
            self.cfg.save()
        return name

    def session(self, name: str | None = None) -> Session:
        name = name or self.current
        if name not in self.sessions:
            self.sessions[name] = Session(name)
        return self.sessions[name]

    # the "current" profile, for the GUI / CLI
    bot = property(lambda self: self.session().bot)
    agent = property(lambda self: self.session().agent)
    trainer = property(lambda self: self.session().trainer)
    brain = property(lambda self: self.open_brain())

    # ------------------------------------------------------------------ preparation
    def _progress(self, done, total):
        self.progress_text = f"downloading {done / 1e6:.1f}/{total / 1e6:.1f} MB" if total else f"{done / 1e6:.1f} MB"

    @property
    def local_server(self) -> bool:
        return self.cfg["server"]["host"] in ("127.0.0.1", "localhost", "::1") and self.cfg["server"]["auto_start"]

    def prepare(self):
        from .data.gamedata import GameData
        from .server.manager import MinecraftServer, ensure_gamedata, ensure_server_jar, find_java
        if self.gd is not None:
            return self.gd
        version = self.cfg["minecraft_version"]
        self.status = "preparing game data"
        if self.local_server:
            self.server = self.server or MinecraftServer(self.cfg)
            self.server.prepare(self._progress)
            gd_dir = ensure_gamedata(version, self.server.java)
        else:
            _, major = ensure_server_jar(version, self._progress)
            gd_dir = ensure_gamedata(version, find_java(major, progress=self._progress))
        self.gd = GameData(gd_dir)
        self.status = "prepared"
        return self.gd

    def open_brain(self, name: str | None = None):
        from .agent.memory import Brain
        from .agent.wiki import Wiki
        s = self.session(name)
        if s.brain is None:
            bdir = data_dir("brain")
            legacy = bdir / "model.db"
            path = bdir / f"{s.name}.db"
            if not path.exists() and legacy.exists() and s.name == self.profiles()[0]:
                legacy.rename(path)          # first profile inherits the old single-bot brain
            s.brain = Brain(path)
        if self.wiki is None:
            self.wiki = Wiki(data_dir("wiki") / "wiki.db", online=self.cfg["agent"]["wiki_online"],
                             rate_per_sec=self.cfg["training"]["wiki_rate_per_sec"])
        return s.brain

    # ------------------------------------------------------------------ server
    def start_server(self):
        if not self.local_server:
            return None
        with self._lock:
            if self.server is None or self.gd is None:
                self.prepare()
            if not self.server.running:
                self.status = "starting server"
                self.server.start(self._progress)
                self.server.command(f"difficulty {self.cfg['server']['difficulty']}")
                self.server.command("gamerule spawn_monsters true")      # zombies & co stay on
                if self.cfg["training"].get("keep_inventory", True):
                    self.server.command("gamerule keep_inventory true")
                if self.cfg["server"].get("public_join", True):
                    self.setup_public_join()
                for line in self.join_info():
                    log.info(line)
        return self.server

    def setup_public_join(self):
        """Look up the public IP; optionally (server.upnp=true) ask the home router to forward the port."""
        port = self.cfg["server"]["port"]
        self.public_ip = public_ip()
        if self.cfg["server"].get("upnp", False):
            self.upnp_status = upnp_forward(port)
            log.info(self.upnp_status)

    def join_info(self) -> list[str]:
        s = self.cfg["server"]
        port = s["port"]
        if not self.local_server:
            return [f"Remote server {s['host']}:{port} (Minecraft Java {self.cfg['minecraft_version']})"]
        running = bool(self.server and self.server.running)
        pub = f"{self.public_ip}:{port}" if self.public_ip else "(shown once the server has started)"
        lines = [
            f"Minecraft Java {self.cfg['minecraft_version']} server is {'RUNNING' if running else 'not running'}",
            f"  PUBLIC address (other devices, anywhere):  {pub}",
            f"  Same Wi-Fi / home network:                  {lan_ip()}:{port}",
            f"  On this computer:                           localhost:{port}",
        ]
        if self.upnp_status:
            lines.append("  " + self.upnp_status)
        if "now forwards" not in self.upnp_status:
            lines.append(f"  Public address needs TCP port {port} forwarded on your router to {lan_ip()} "
                         f"(or enable automatic UPnP in Settings).")
        lines += [
            "  Minecraft: Multiplayer -> Add Server -> paste an address above.",
            "  Offline/cracked mode: any username works, no Microsoft login needed on the server side.",
            f"  Bots: {', '.join(self.profiles())}. Talk to them in chat: !help, !come, !follow, !obtain iron_pickaxe",
            "  Make yourself operator from the Play tab or the server console: op <your name>",
            "  Offline mode: anyone who has the public address can join. Keep it private.",
        ]
        return lines

    # ------------------------------------------------------------------ bots
    def start_bot(self, name: str | None = None):
        from .bot.bot import Bot
        s = self.session(name)
        self.prepare()
        self.start_server()
        b = self.cfg["bot"]
        with self._lock:
            if s.bot is None:
                s.bot = Bot(self.gd, s.name, self.cfg["server"]["host"], self.cfg["server"]["port"],
                            b["view_distance"], b["auto_respawn"])
                s.bot.events.on("end", lambda reason, s=s: self._on_bot_end(s, reason))
        if not s.bot.connected:
            # The local server may have moved to a different port due to a
            # collision. Always reconnect to its actual configured endpoint.
            s.bot.host = self.cfg["server"]["host"]
            s.bot.port = self.cfg["server"]["port"]
            self.status = f"connecting {s.name}"
            s.bot.connect()
            if not s.bot.wait_spawn(90):
                raise RuntimeError(f"{s.name} did not spawn")
            s.bot.wait_chunks()
        if self.server and self.cfg["server"]["op_bot"]:
            self.server.command(f"op {s.name}")
        self.status = f"{s.name} online"
        return s.bot

    def _on_bot_end(self, s: Session, reason):
        if not self.cfg["bot"]["auto_reconnect"] or "bye" in reason or self.status == "shutting down":
            return
        if s.trainer and s.trainer.running:
            return    # the trainer handles reconnects itself

        def reconnect():
            for attempt in range(30):
                time.sleep(min(60, 5 * (attempt + 1)))
                if self.status == "shutting down":
                    return
                try:
                    if self.server and not self.server.running:
                        self.start_server()
                    s.bot.host = self.cfg["server"]["host"]
                    s.bot.port = self.cfg["server"]["port"]
                    s.bot.connect()
                    if s.bot.wait_spawn(60):
                        log.info("%s reconnected", s.name)
                        return
                except Exception as e:  # noqa: BLE001
                    log.warning("reconnect failed: %s", e)
        threading.Thread(target=reconnect, daemon=True).start()

    def start_agent(self, name: str | None = None):
        from .agent.agent import Agent
        from .agent.llm import LLM
        s = self.session(name)
        self.start_bot(s.name)
        self.open_brain(s.name)
        if self.llm is None:
            self.llm = LLM(self.cfg["llm"])
        if s.agent is None:
            s.agent = Agent(s.bot, self.cfg, s.brain, self.wiki, self.llm_for(s.name), self.server)
        s.agent.start()
        self.status = f"{s.name}: agent running"
        return s.agent

    def start_training(self, hours: float | None = None, background: bool = True, name: str | None = None):
        from .agent.trainer import Trainer
        with self._lock:
            s = self.session(name)
            if s.trainer and s.trainer.running:
                # Double-clicking Train must not start two controllers competing
                # over the same movement controls or duplicate wiki workers.
                return s.trainer
            self.start_agent(s.name)
            if self.server and self.cfg["training"].get("keep_inventory", True):
                self.server.command("gamerule keep_inventory true")
            if self.server:
                self.server.command(f"difficulty {self.cfg['server']['difficulty']}")
                self.server.command("gamerule spawn_monsters true")
            s.trainer = Trainer(s.agent, hours if hours is not None else self.cfg["training"]["hours"])
            self.status = f"{s.name}: training"
            s.trainer.start(background=background)
            return s.trainer

    def start_all(self, training: bool = False, hours: float | None = None):
        """Start every profile (staggered so a Pi is not hammered by 5 logins at once)."""
        for n in self.profiles():
            if training:
                self.start_training(hours, True, n)
            else:
                self.start_agent(n)
            time.sleep(3)

    def llm_for(self, name: str):
        """Bots without their own AI share the default one; a bot with its own provider gets its own LLM."""
        from .agent.llm import LLM, bot_llm_config
        override = (self.cfg.get("bot_ai") or {}).get(name)
        if not override or not override.get("provider"):
            return self.llm
        self._bot_llms = getattr(self, "_bot_llms", {})
        cfg = bot_llm_config(self.cfg["llm"], override)
        if name in self._bot_llms:
            self._bot_llms[name].reconfigure(cfg)
        else:
            self._bot_llms[name] = LLM(cfg)
        return self._bot_llms[name]

    def reload_llm(self):
        from .agent.llm import bot_llm_config
        if self.llm:
            self.llm.reconfigure(self.cfg["llm"])   # agents share this object, so they switch immediately
        for name, s in self.sessions.items():
            if not s.agent:
                continue
            want = self.llm_for(name)
            if s.agent.llm is not want:            # bot switched between default and its own AI
                s.agent.llm = want
            elif want is not self.llm:
                want.reconfigure(bot_llm_config(self.cfg["llm"], self.cfg["bot_ai"].get(name)))

    def shutdown(self):
        self.status = "shutting down"
        try:
            for s in list(self.sessions.values()):
                try:
                    if s.trainer and s.trainer.running:
                        s.trainer.stop()
                        s.trainer.finish()
                    if s.agent:
                        s.agent.stop()
                    if s.bot:
                        s.bot.disconnect("bye")
                    if s.brain:
                        s.brain.commit()
                        # Persist transferable skills even if the user only
                        # chatted, played, or trained one skill this session.
                        s.brain.export_latest()
                        s.brain.export_universal()
                except Exception:  # noqa: BLE001
                    log.exception("shutdown of %s failed", s.name)
            if self.server:
                self.server.stop()
        finally:
            self.status = "stopped"
