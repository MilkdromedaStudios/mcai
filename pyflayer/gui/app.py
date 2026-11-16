"""Desktop app (Tkinter - ships with Raspberry Pi OS). Light on CPU: refreshes twice a second,
the minimap every few seconds, and all heavy work runs in background threads."""
from __future__ import annotations

import json
import math
import threading
import time
import tkinter as tk
import traceback
from tkinter import messagebox, scrolledtext, ttk

import numpy as np

from ..config import data_dir

BG, PANEL, FIELD, FG, MUTED, ACCENT = "#1e1f22", "#2b2d31", "#26272b", "#e6e6e6", "#9aa0a6", "#3d7eff"

PALETTE = [  # (substring, colour) first match wins
    ("water", "#3a66d8"), ("lava", "#ff6a00"), ("ice", "#9fc3ff"), ("snow", "#f4f8ff"), ("powder_snow", "#f4f8ff"),
    ("grass_block", "#5f9e35"), ("leaves", "#2e6b1f"), ("sand", "#dccf8f"), ("sandstone", "#d4c47c"),
    ("red_sand", "#bf6a2f"), ("terracotta", "#a0583a"), ("gravel", "#8a8580"), ("dirt", "#7a5434"),
    ("mud", "#4d3f35"), ("podzol", "#6b4a26"), ("mycelium", "#7d6a7a"), ("log", "#6b4f2c"), ("planks", "#a8844f"),
    ("stone", "#7f7f7f"), ("deepslate", "#4f4f55"), ("andesite", "#8a8a8a"), ("granite", "#9a6b5a"),
    ("diorite", "#c8c8c8"), ("tuff", "#6c6d66"), ("calcite", "#dfe0dc"), ("netherrack", "#6b2a2a"),
    ("nether", "#5a1c1c"), ("soul", "#4d3a2e"), ("basalt", "#4a4a50"), ("blackstone", "#2f2a30"),
    ("end_stone", "#dcdba0"), ("obsidian", "#1d1426"), ("moss", "#59782f"), ("clay", "#9aa3b3"),
    ("ore", "#b07a5a"), ("wool", "#e8e8e8"), ("farmland", "#5a3a20"), ("path", "#9c844f"),
    ("crimson", "#8a2433"), ("warped", "#167e86"), ("bamboo", "#6b8f2a"), ("cactus", "#2f7a2f"),
]


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


class App:
    def __init__(self, rt):
        self.rt = rt
        self.cfg = rt.cfg
        self.root = tk.Tk()
        self.root.title("PyFlayer AI - Minecraft " + self.cfg["minecraft_version"])
        self.root.geometry("1180x740")
        self.root.minsize(900, 600)
        self.busy = False
        self._colors = None
        self._map_photo = None
        self._last_map = 0.0
        self._chat_seen = 0
        self._srv_seen = 0
        self._hist_seen = 0
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(500, self.refresh)

    # ================================================================== layout
    def _theme(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg, panel, field, fg, mut = BG, PANEL, FIELD, FG, MUTED
        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg, fieldbackground=field, bordercolor="#3a3c41",
                        lightcolor=panel, darkcolor=panel, troughcolor=panel, focuscolor=ACCENT,
                        selectbackground=ACCENT, selectforeground="#ffffff", insertcolor=fg)
        style.configure("TButton", background=panel, foreground=fg, padding=(8, 4))
        style.map("TButton", background=[("active", "#404249"), ("pressed", ACCENT)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#5a92ff")])
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background=panel, foreground=mut, padding=(12, 5))
        style.map("TNotebook.Tab", background=[("selected", "#404249")], foreground=[("selected", fg)])
        style.configure("TLabelframe", background=bg, bordercolor="#3a3c41")
        style.configure("TLabelframe.Label", background=bg, foreground=ACCENT)
        style.configure("TEntry", fieldbackground=field, foreground=fg)
        style.configure("TCombobox", fieldbackground=field, foreground=fg, background=panel, arrowcolor=fg)
        style.map("TCombobox", fieldbackground=[("readonly", field)], foreground=[("readonly", fg)])
        style.configure("Health.Horizontal.TProgressbar", background="#e5484d", troughcolor=panel)
        style.configure("Food.Horizontal.TProgressbar", background="#f5a524", troughcolor=panel)
        for opt, val in (("*Text.Background", field), ("*Text.Foreground", fg), ("*Text.insertBackground", fg),
                         ("*Text.selectBackground", ACCENT), ("*Listbox.Background", field),
                         ("*Listbox.Foreground", fg), ("*Listbox.selectBackground", ACCENT),
                         ("*TCombobox*Listbox.background", field), ("*TCombobox*Listbox.foreground", fg),
                         ("*Toplevel.Background", bg)):
            self.root.option_add(opt, val)

    def _build(self):
        self._theme()
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Bot:").pack(side="left")
        self.profile_var = tk.StringVar(value=self.rt.current)
        self.profile_box = ttk.Combobox(top, textvariable=self.profile_var, values=self.rt.profiles(),
                                        width=12, state="readonly")
        self.profile_box.pack(side="left", padx=(2, 2))
        self.profile_box.bind("<<ComboboxSelected>>", lambda e: self.switch_profile(self.profile_var.get()))
        ttk.Button(top, text="+ Bot", command=self.add_bot).pack(side="left", padx=(0, 8))
        self.btns = {}
        for key, text, cmd in (("server", "Start server", self.do_server), ("bot", "Connect bot", self.do_bot),
                               ("agent", "Start AI", self.do_agent),
                               ("train", "Train overnight", self.do_train),
                               ("stop", "Stop", self.do_stop), ("settings", "Settings", self.do_settings)):
            b = ttk.Button(top, text=text, command=cmd)
            b.pack(side="left", padx=3)
            self.btns[key] = b
        self.status_var = tk.StringVar(value="idle")
        ttk.Button(top, text="Start all", command=self.do_all).pack(side="left", padx=3)
        ttk.Label(top, textvariable=self.status_var, foreground=MUTED).pack(side="right", padx=6)

        body = tk.PanedWindow(self.root, orient="horizontal", sashwidth=5, bd=0, bg=BG)
        body.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, minsize=380, width=380)
        body.add(right, minsize=420)

        # ---- status panel
        box = ttk.LabelFrame(left, text="Bot", padding=6)
        box.pack(fill="x")
        self.info = {}
        for k in ("server", "connection", "position", "dimension", "biome", "health", "food", "time", "action",
                  "task", "tps"):
            row = ttk.Frame(box)
            row.pack(fill="x")
            ttk.Label(row, text=k + ":", width=10).pack(side="left")
            v = tk.StringVar(value="-")
            ttk.Label(row, textvariable=v).pack(side="left", fill="x")
            self.info[k] = v
        self.health_bar = ttk.Progressbar(box, maximum=20, style="Health.Horizontal.TProgressbar")
        self.health_bar.pack(fill="x", pady=(4, 0))
        self.food_bar = ttk.Progressbar(box, maximum=20, style="Food.Horizontal.TProgressbar")
        self.food_bar.pack(fill="x", pady=2)

        inv = ttk.LabelFrame(left, text="Inventory", padding=4)
        inv.pack(fill="both", expand=True, pady=4)
        self.inv_list = tk.Listbox(inv, height=10, font="TkFixedFont")
        self.inv_list.pack(fill="both", expand=True)

        tr = ttk.LabelFrame(left, text="Training", padding=4)
        tr.pack(fill="both", expand=True)
        self.train_var = tk.StringVar(value="not running")
        ttk.Label(tr, textvariable=self.train_var, wraplength=360).pack(fill="x")
        self.ms_list = tk.Listbox(tr, height=10, font="TkFixedFont")
        self.ms_list.pack(fill="both", expand=True)

        # ---- tabs
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.activity = self._text_tab(nb, "Activity")
        chat_tab = ttk.Frame(nb)
        nb.add(chat_tab, text="Chat / Tasks")
        self.chat = scrolledtext.ScrolledText(chat_tab, height=10, state="disabled", wrap="word")
        self.chat.pack(fill="both", expand=True)
        row = ttk.Frame(chat_tab)
        row.pack(fill="x", pady=4)
        self.chat_entry = ttk.Entry(row)
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=2)
        self.chat_entry.bind("<Return>", lambda e: self.send_task())
        ttk.Button(row, text="Give AI task", command=self.send_task).pack(side="left", padx=2)
        ttk.Button(row, text="Say in chat", command=self.send_chat).pack(side="left", padx=2)
        ttk.Label(chat_tab, text="Tasks: plain English (needs AI endpoint). Commands: !obtain iron_pickaxe, "
                                 "!mine oak_log 10, !goto x y z, !scan, !stop, !help", foreground=MUTED).pack(fill="x")

        srv_tab = ttk.Frame(nb)
        nb.add(srv_tab, text="Server console")
        self.srv_text = scrolledtext.ScrolledText(srv_tab, state="disabled", wrap="none", font="TkFixedFont")
        self.srv_text.pack(fill="both", expand=True)
        row = ttk.Frame(srv_tab)
        row.pack(fill="x", pady=4)
        self.srv_entry = ttk.Entry(row)
        self.srv_entry.pack(side="left", fill="x", expand=True)
        self.srv_entry.bind("<Return>", lambda e: self.send_server_cmd())
        ttk.Button(row, text="Run command", command=self.send_server_cmd).pack(side="left", padx=2)

        play = ttk.Frame(nb, padding=10)
        nb.add(play, text="Play / Test")
        self.join_text = tk.Text(play, height=12, wrap="word", relief="flat", padx=8, pady=8)
        self.join_text.pack(fill="x")
        row = ttk.Frame(play)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="Your Minecraft name:").pack(side="left")
        self.player_var = tk.StringVar(value=self.cfg.get("agent.owner") or "")
        ttk.Entry(row, textvariable=self.player_var, width=18).pack(side="left", padx=4)
        row = ttk.Frame(play)
        row.pack(fill="x", pady=(0, 8))
        for text, fn in (("Make me operator", lambda p: f"op {p}"),
                         ("Bring bot to me", lambda p: f"tp {self.rt.current} {p}"),
                         ("Teleport me to bot", lambda p: f"tp {p} {self.rt.current}"),
                         ("Creative mode for me", lambda p: f"gamemode creative {p}"),
                         ("Survival mode for me", lambda p: f"gamemode survival {p}")):
            ttk.Button(row, text=text, command=lambda f=fn: self.player_cmd(f)).pack(side="left", padx=2)
        row2 = ttk.Frame(play)
        row2.pack(fill="x")
        row3 = ttk.Frame(play)
        row3.pack(fill="x")
        for i, (text, cmd) in enumerate((("Set day", "time set day"), ("Set night", "time set night"),
                                         ("Clear weather", "weather clear"), ("Easy mode", "difficulty easy"),
                                         ("Normal mode", "difficulty normal"),
                                         ("Keep inventory ON", "gamerule keep_inventory true"),
                                         ("Keep inventory OFF", "gamerule keep_inventory false"))):
            ttk.Button(row2 if i < 4 else row3, text=text,
                       command=lambda c=cmd: self.server_cmd(c)).pack(side="left", padx=2, pady=2)
        ttk.Label(play, text="Join with Minecraft Java 26.3 → Multiplayer → Add Server → the address above. "
                             "In game, talk to the bots: '!come', '!follow', '!obtain iron_pickaxe', '!help', "
                             "or say the bot's name followed by a request (AI endpoint needed).",
                  foreground=MUTED, wraplength=700).pack(fill="x", pady=10)

        map_tab = ttk.Frame(nb)
        nb.add(map_tab, text="Map / Scan")
        self.map_canvas = tk.Canvas(map_tab, width=384, height=384, bg="#111", highlightthickness=0)
        self.map_canvas.pack(side="left", padx=6, pady=6)
        side = ttk.Frame(map_tab)
        side.pack(side="left", fill="both", expand=True)
        ttk.Button(side, text="Scan 100 blocks", command=self.do_scan).pack(anchor="w", pady=2)
        self.scan_text = scrolledtext.ScrolledText(side, wrap="word", height=20)
        self.scan_text.pack(fill="both", expand=True)

        brain_tab = ttk.Frame(nb)
        nb.add(brain_tab, text="Brain")
        row = ttk.Frame(brain_tab)
        row.pack(fill="x", pady=4)
        self.recall_entry = ttk.Entry(row)
        self.recall_entry.pack(side="left", fill="x", expand=True)
        self.recall_entry.bind("<Return>", lambda e: self.do_recall())
        ttk.Button(row, text="Recall", command=self.do_recall).pack(side="left", padx=2)
        ttk.Button(row, text="Export model file", command=self.do_export).pack(side="left", padx=2)
        self.brain_text = scrolledtext.ScrolledText(brain_tab, wrap="word")
        self.brain_text.pack(fill="both", expand=True)

    def _text_tab(self, nb, title):
        f = ttk.Frame(nb)
        nb.add(f, text=title)
        t = scrolledtext.ScrolledText(f, state="disabled", wrap="word", font="TkFixedFont")
        t.pack(fill="both", expand=True)
        return t

    @staticmethod
    def _append(widget, text, max_lines=1500):
        widget.configure(state="normal")
        widget.insert("end", text)
        lines = int(widget.index("end-1c").split(".")[0])
        if lines > max_lines:
            widget.delete("1.0", f"{lines - max_lines}.0")
        widget.see("end")
        widget.configure(state="disabled")

    # ================================================================== actions
    def _bg(self, label, fn):
        if self.busy:
            messagebox.showinfo("Busy", "Please wait for the current operation to finish.")
            return

        def run():
            self.busy = True
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                self.root.after(0, lambda: messagebox.showerror(label, err))
                traceback.print_exc()
            finally:
                self.busy = False
        threading.Thread(target=run, daemon=True).start()

    def do_server(self):
        self._bg("Server", self.rt.start_server)

    def do_bot(self):
        self._bg("Bot", self.rt.start_bot)

    def do_agent(self):
        def go():
            self.rt.start_agent()
            if not self.rt.llm.configured:
                self.root.after(0, lambda: messagebox.showinfo(
                    "AI endpoint", "The agent runs (reflexes, chat commands, planner), but no AI endpoint is set.\n"
                                   "Open Settings and enter your OpenAI-compatible base URL, key and model."))
        self._bg("Agent", go)

    def do_train(self):
        hours = self.cfg["training"]["hours"]
        if not messagebox.askyesno("Overnight training", f"Start {hours} hours of training now?\n\n"
                                   "The bot will practise the beat-the-game curriculum, build its knowledge base "
                                   "and explore. You can close this window; use 'Stop' to end early."):
            return
        self._bg("Training", lambda: self.rt.start_training(hours))

    def do_stop(self):
        if self.rt.bot:
            self.rt.bot.stop()
        if self.rt.agent:
            with self.rt.agent.tasks.mutex:
                self.rt.agent.tasks.queue.clear()

    def send_task(self):
        text = self.chat_entry.get().strip()
        if not text:
            return
        self.chat_entry.delete(0, "end")
        if not self.rt.agent:
            messagebox.showinfo("Agent", "Start the AI agent first.")
            return
        if text.startswith("!"):
            threading.Thread(target=self.rt.agent._chat_command, args=("gui", text), daemon=True).start()
        elif self.rt.llm and self.rt.llm.configured:
            self.rt.agent.submit(text, "gui")
        else:
            messagebox.showinfo("AI endpoint", "No AI endpoint configured. Use !commands or open Settings.")

    def send_chat(self):
        text = self.chat_entry.get().strip()
        if text and self.rt.bot and self.rt.bot.connected:
            self.rt.bot.chat(text)
            self.chat_entry.delete(0, "end")

    def send_server_cmd(self):
        cmd = self.srv_entry.get().strip()
        if cmd and self.rt.server:
            self.rt.server.command(cmd)
            self.srv_entry.delete(0, "end")

    def do_scan(self):
        def go():
            from ..bot.scanner import Scanner
            if not self.rt.agent:
                self.rt.start_agent()
            res = self.rt.agent.scanner.scan(100)
            self.rt.agent.after_scan(res)
            txt = Scanner.summarize(res, 6000)
            self.root.after(0, lambda: (self.scan_text.delete("1.0", "end"), self.scan_text.insert("end", txt)))
        self._bg("Scan", go)

    def do_recall(self):
        q = self.recall_entry.get().strip()
        self.rt.open_brain()
        txt = json.dumps(self.rt.brain.overview(), indent=1) + "\n\n"
        if q:
            txt += self.rt.brain.recall_text(q, 10, 6000) or "nothing found"
        else:
            sk = self.rt.brain.skills(None, 20)
            txt += "Skills:\n" + "\n".join(f"- {s['name']}: {s['description']} (ok {s['successes']})" for s in sk)
            ms = self.rt.brain.milestones()
            txt += "\n\nMilestones:\n" + "\n".join(f"- {k}: {'done' if v['achieved'] else 'not yet'} "
                                                  f"({v['attempts']} attempts)" for k, v in ms.items())
        self.brain_text.delete("1.0", "end")
        self.brain_text.insert("end", txt)

    def do_export(self):
        self.rt.open_brain()
        p = self.rt.brain.export(data_dir("brain") / f"model-{time.strftime('%Y%m%d-%H%M')}.json.gz")
        messagebox.showinfo("Exported", f"Brain exported to\n{p}")

    def switch_profile(self, name):
        self.rt.current = name
        self._chat_seen = 0
        self._hist_seen = 0
        for w in (self.chat, self.activity):
            w.configure(state="normal")
            w.delete("1.0", "end")
            w.configure(state="disabled")
        self.inv_list.delete(0, "end")
        self.ms_list.delete(0, "end")

    def add_bot(self):
        win = tk.Toplevel(self.root)
        win.title("Add bot")
        win.configure(bg=BG)
        frm = ttk.Frame(win, padding=12)
        frm.pack()
        ttk.Label(frm, text=f"New bot name (max {5} bots per server):").pack(anchor="w")
        var = tk.StringVar()
        e = ttk.Entry(frm, textvariable=var, width=24)
        e.pack(pady=6)
        e.focus_set()

        def ok():
            try:
                name = self.rt.add_profile(var.get().strip())
            except ValueError as err:
                messagebox.showerror("Add bot", str(err), parent=win)
                return
            self.profile_box["values"] = self.rt.profiles()
            self.profile_var.set(name)
            self.switch_profile(name)
            win.destroy()
        ttk.Button(frm, text="Add", style="Accent.TButton", command=ok).pack(anchor="e")
        e.bind("<Return>", lambda ev: ok())

    def do_all(self):
        self._bg("Start all", self.rt.start_all)

    def server_cmd(self, cmd):
        if not self.rt.server or not self.rt.server.running:
            messagebox.showinfo("Server", "Start the server first.")
            return
        self.rt.server.command(cmd)

    def player_cmd(self, fn):
        p = self.player_var.get().strip()
        if not p:
            messagebox.showinfo("Player", "Enter your Minecraft name first.")
            return
        self.server_cmd(fn(p))

    def do_settings(self):
        SettingsDialog(self)

    # ================================================================== refresh (cheap)
    def refresh(self):
        try:
            self._refresh()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self.root.after(int(self.cfg["gui"]["refresh_ms"]), self.refresh)

    def _refresh(self):
        rt = self.rt
        self.status_var.set(rt.status + (f" - {rt.progress_text}" if rt.progress_text and "prep" in rt.status else ""))
        srv = rt.server
        info = "\n".join(rt.join_info())
        if self.join_text.get("1.0", "end-1c") != info:
            self.join_text.configure(state="normal")
            self.join_text.delete("1.0", "end")
            self.join_text.insert("end", info)
            self.join_text.configure(state="disabled")
        self.info["server"].set(srv.status if srv else ("remote" if not rt.local_server else "not started"))
        b = rt.bot
        if b:
            self.info["connection"].set("online" if b.spawned.is_set() else ("connecting" if b.connected else
                                                                             f"offline {b.disconnect_reason[:40]}"))
            x, y, z = b.position
            self.info["position"].set(f"{x:.1f}, {y:.1f}, {z:.1f}")
            self.info["dimension"].set(b.dimension)
            self.info["biome"].set(b.world.biome_at(*b.block_pos) or "-")
            self.info["health"].set(f"{b.health:.1f} / 20")
            self.info["food"].set(f"{b.food} / 20")
            self.info["time"].set(f"{b.time_phase()} ({b.time_of_day})")
            self.info["action"].set(b.current_action)
            self.info["tps"].set(f"{b.tps_measured:.1f}")
            self.health_bar["value"] = b.health
            self.food_bar["value"] = b.food
            items = sorted(b.inventory.counts().items(), key=lambda kv: -kv[1])
            cur = self.inv_list.get(0, "end")
            new = [f"{n:<28} {c:>4}" for n, c in items]
            if list(cur) != new:
                self.inv_list.delete(0, "end")
                for line in new:
                    self.inv_list.insert("end", line)
            if len(b.chat_log) != self._chat_seen:
                for ts, sender, text in list(b.chat_log)[self._chat_seen - len(b.chat_log):] if self._chat_seen else b.chat_log:
                    self._append(self.chat, time.strftime("%H:%M ", time.localtime(ts)) +
                                 (f"<{sender}> " if sender else "") + text + "\n")
                self._chat_seen = len(b.chat_log)
            if time.time() - self._last_map > 3 and self.cfg["gui"]["minimap"]:
                self._last_map = time.time()
                self.draw_map()
        if rt.agent:
            self.info["task"].set((rt.agent.current_task or "-")[:60])
            h = rt.agent.history
            if len(h) != self._hist_seen:
                new = h[self._hist_seen:] if self._hist_seen <= len(h) else h[-50:]
                self._append(self.activity, "\n".join(new) + "\n")
                self._hist_seen = len(h)
        if srv:
            lines = list(srv.lines)
            if lines and len(lines) != self._srv_seen:
                self._append(self.srv_text, "\n".join(lines[-min(len(lines), max(1, len(lines) - self._srv_seen)):]) + "\n")
                self._srv_seen = len(lines)
        tr = rt.trainer
        if tr:
            p = tr.progress()
            done = sum(m["achieved"] for m in p["milestones"])
            k = p["knowledge"]
            left = tr.time_left / 3600
            self.train_var.set(f"{p['status']} | {p['elapsed_h']:.2f}h elapsed, {left:.2f}h left\n"
                               f"milestones {done}/{len(p['milestones'])} | knowledge: {k['blocks_known_gamedata']} "
                               f"blocks (game data), {k['entries_with_wiki']} wiki, {k['block_types_seen_in_world']} "
                               f"seen in world")
            lines = [f"{'✓' if m['achieved'] else '·'} {m['name']:<16} {m['attempts']}x" for m in p["milestones"]]
            if list(self.ms_list.get(0, "end")) != lines:
                self.ms_list.delete(0, "end")
                for line in lines:
                    self.ms_list.insert("end", line)

    # ================================================================== minimap (numpy, ~10ms)
    def _state_colors(self):
        if self._colors is None:
            gd = self.rt.gd
            cols = np.zeros((gd.num_states, 3), dtype=np.uint8)
            for name, ids in gd.block_states.items():
                c = next((_hex(col) for key, col in PALETTE if key in name), (110, 110, 110))
                cols[ids] = c
            self._colors = cols
        return self._colors

    def draw_map(self, size=128, scale=3):
        b = self.rt.bot
        if not b or not b.spawned.is_set():
            return
        gd = b.gd
        colors = self._state_colors()
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (17, 17, 17)
        bx, by, bz = b.block_pos
        x0, z0 = bx - size // 2, bz - size // 2
        air = gd.is_air
        world = b.world
        with world.lock:
            cols = dict(world.columns)
        # nether / caves: start the search a bit above the bot instead of from the sky
        ceiling = (by + 4 - world.min_y) if b.dimension == "the_nether" else None
        for (cx, cz), col in cols.items():
            px, pz = cx * 16 - x0, cz * 16 - z0
            if px <= -16 or pz <= -16 or px >= size or pz >= size:
                continue
            blocks = col.blocks.reshape(-1, 16, 16)        # (y, z, x)
            if ceiling is not None:
                blocks = blocks[:max(1, min(len(blocks), ceiling))]
            solid = ~air[blocks]
            top = blocks.shape[0] - 1 - np.argmax(solid[::-1], axis=0)   # (z, x)
            has = solid.any(axis=0)
            zi, xi = np.indices((16, 16))
            states = blocks[top, zi, xi]
            rgb = colors[states].astype(np.int16)
            shade = ((top - (by - world.min_y)) * 2).clip(-40, 40)[..., None]
            rgb = (rgb + shade).clip(0, 255).astype(np.uint8)
            rgb[~has] = (17, 17, 17)
            sx0, sz0 = max(0, -px), max(0, -pz)
            sx1, sz1 = min(16, size - px), min(16, size - pz)
            img[pz + sz0:pz + sz1, px + sx0:px + sx1] = rgb[sz0:sz1, sx0:sx1]
        # entities
        for e in b.entities.within(b.position, size / 2):
            ex, ez = int(e.x) - x0, int(e.z) - z0
            if 0 <= ex < size and 0 <= ez < size:
                img[ez, ex] = (255, 40, 40) if e.hostile else ((255, 255, 0) if e.type == "player" else (240, 240, 240))
        img[size // 2 - 1:size // 2 + 2, size // 2 - 1:size // 2 + 2] = (0, 255, 255)
        big = img.repeat(scale, axis=0).repeat(scale, axis=1)
        header = f"P6 {big.shape[1]} {big.shape[0]} 255 ".encode()
        self._map_photo = tk.PhotoImage(data=header + big.tobytes(), format="PPM")
        self.map_canvas.delete("all")
        self.map_canvas.create_image(0, 0, image=self._map_photo, anchor="nw")
        yaw = math.radians(b.state.yaw)
        c = size * scale / 2
        self.map_canvas.create_line(c, c, c - math.sin(yaw) * 14, c + math.cos(yaw) * 14, fill="cyan", width=2)

    def on_close(self):
        if self.rt.trainer and self.rt.trainer.running:
            if not messagebox.askyesno("Quit", "Training is running. Stop it and quit?"):
                return
        self.root.destroy()
        threading.Thread(target=self.rt.shutdown, daemon=True).start()
        time.sleep(0.5)

    def run(self):
        self.root.mainloop()


class SettingsDialog:
    FIELDS = [("llm.api_key", "AI API key"), ("llm.model", "AI model (blank = provider default)"),
              ("llm.base_url", "Base URL (custom/local, or override)"), ("llm.tool_mode", "Tool mode (auto/native/json)"),
              ("bot.username", "Bot username"), ("server.host", "Server host"), ("server.port", "Server port"),
              ("server.auto_start", "Auto-start local server (true/false)"),
              ("server.view_distance", "Server view distance"), ("server.difficulty", "Difficulty (easy/normal/...)"),
              ("training.keep_inventory", "Keep inventory while training (true/false)"), ("server.memory_mb", "Server RAM MB (0=auto)"),
              ("training.hours", "Training hours"), ("agent.allow_cheats", "Allow /locate (true/false)"),
              ("agent.owner", "Only obey this player (blank = anyone)"),
              ("server.upnp", "Auto port-forward on router / UPnP (true/false)"), ("agent.wiki_online", "Wiki online (true/false)")]

    def __init__(self, app: App):
        self.app = app
        cfg = app.cfg
        self.win = tk.Toplevel(app.root)
        self.win.title("Settings")
        self.win.configure(bg=BG)
        self.win.transient(app.root)
        frm = ttk.Frame(self.win, padding=10)
        frm.pack(fill="both", expand=True)
        from ..agent.llm import PROVIDERS
        self.vars = {}
        ttk.Label(frm, text="AI provider").grid(row=0, column=0, sticky="w", pady=2)
        self.provider = tk.StringVar(value=cfg.get("llm.provider") or "custom")
        box = ttk.Combobox(frm, textvariable=self.provider, values=list(PROVIDERS), state="readonly", width=46)
        box.grid(row=0, column=1, pady=2, sticky="w")
        self.hint = tk.StringVar()
        ttk.Label(frm, textvariable=self.hint, foreground=MUTED).grid(row=1, column=0, columnspan=2, sticky="w")
        box.bind("<<ComboboxSelected>>", lambda e: self._provider_changed(reset=True))
        for i, (key, label) in enumerate(self.FIELDS, start=2):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=str(cfg.get(key, "")))
            ttk.Entry(frm, textvariable=v, width=48, show="*" if "api_key" in key else "").grid(row=i, column=1, pady=2)
            self.vars[key] = v
        self._provider_changed(reset=False)
        n = len(self.FIELDS) + 2
        self.test_var = tk.StringVar()
        ttk.Label(frm, textvariable=self.test_var, wraplength=520).grid(row=n, column=0, columnspan=2, sticky="w")
        btns = ttk.Frame(frm)
        btns.grid(row=n + 1, column=1, sticky="e", pady=8)
        ttk.Button(btns, text="Test AI connection", command=self.test_ai).pack(side="left", padx=4)
        ttk.Button(btns, text="Save", style="Accent.TButton", command=self.save).pack(side="left")

    def _provider_changed(self, reset: bool):
        from ..agent.llm import PROVIDERS
        p = PROVIDERS.get(self.provider.get(), PROVIDERS["custom"])
        if reset:
            self.vars["llm.base_url"].set("")
            self.vars["llm.model"].set("")
        key = f"key from {p['key_env']} or the field above" if p.get("key_env") else "no key needed"
        self.hint.set(f"{p['base_url'] or 'enter your OpenAI-compatible base URL below'} | default model "
                      f"{p['model'] or '-'} | {key}")

    def test_ai(self):
        from ..agent.llm import LLM
        cfg = dict(self.app.cfg["llm"])
        cfg["provider"] = self.provider.get()
        for k in ("llm.api_key", "llm.model", "llm.base_url", "llm.tool_mode"):
            cfg[k.split(".")[1]] = self.vars[k].get().strip()
        self.test_var.set("testing...")
        threading.Thread(target=lambda: self.test_var.set(LLM(cfg).test()), daemon=True).start()

    def save(self):
        cfg = self.app.cfg
        cfg.set("llm.provider", self.provider.get())
        for key, v in self.vars.items():
            old = cfg.get(key)
            val = v.get().strip()
            if isinstance(old, bool):
                val = val.lower() in ("1", "true", "yes", "on")
            elif isinstance(old, int):
                try:
                    val = int(float(val))
                except ValueError:
                    val = old
            elif isinstance(old, float):
                try:
                    val = float(val)
                except ValueError:
                    val = old
            cfg.set(key, val)
        cfg.save()
        self.app.rt.reload_llm()
        self.win.destroy()
        messagebox.showinfo("Settings", "Saved. Server/bot changes apply on the next start.")


def run_gui(rt):
    App(rt).run()
