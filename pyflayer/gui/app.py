"""PyFlayer desktop app - one simple screen.

  [bot hero cards]                       click a card to select a bot, "+" to add one
  [Start bot] [Stop] [Server] [Join server] [Run from model file] [Train overnight] [Settings]
  status line of the selected bot
  activity log of the selected bot
  join address (when the server runs)
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, simpledialog, ttk

from ..config import DATA_DIR

BG, PANEL, FIELD, FG, MUTED, ACCENT = "#1e1f22", "#2b2d31", "#26272b", "#e6e6e6", "#9aa0a6", "#3d7eff"
CARD_COLORS = ["#3d7eff", "#e5484d", "#30a46c", "#f5a524", "#8e4ec6"]


class ScrollFrame(ttk.Frame):
    """A frame whose content scrolls (vertical or horizontal) with the mouse wheel / touch-pad / scrollbar."""

    def __init__(self, master, padding=0, orient="vertical", height=None):
        super().__init__(master)
        self.orient = orient
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0, height=height or 0)
        if orient == "vertical":
            self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.bar.set)
        else:
            self.bar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
            self.canvas.configure(xscrollcommand=self.bar.set)
        self.inner = ttk.Frame(self.canvas, padding=padding)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        if orient == "vertical":
            self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
            self.canvas.pack(side="left", fill="both", expand=True)
            self.bar.pack(side="right", fill="y")
        else:
            self.canvas.pack(side="top", fill="x", expand=True)
            # the horizontal bar only appears when the content is wider than the window
            self.inner.bind("<Configure>", self._fit_h, add="+")
            self.canvas.bind("<Configure>", self._fit_h, add="+")
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", self._bind_wheel)
            w.bind("<Leave>", self._unbind_wheel)

    def _fit_h(self, _e=None):
        need = self.inner.winfo_reqwidth() > self.canvas.winfo_width() > 1
        if need and not self.bar.winfo_ismapped():
            self.bar.pack(side="bottom", fill="x")
        elif not need and self.bar.winfo_ismapped():
            self.bar.pack_forget()

    def _scroll(self, n):
        (self.canvas.yview_scroll if self.orient == "vertical" else self.canvas.xview_scroll)(n, "units")

    def _bind_wheel(self, _e=None):
        self.canvas.bind_all("<MouseWheel>", lambda e: self._scroll(-1 if e.delta > 0 else 1))
        self.canvas.bind_all("<Button-4>", lambda e: self._scroll(-2))
        self.canvas.bind_all("<Button-5>", lambda e: self._scroll(2))

    def _unbind_wheel(self, _e=None):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(seq)


class App:
    def __init__(self, rt):
        self.rt = rt
        self.cfg = rt.cfg
        self.root = tk.Tk()
        self.root.title("PyFlayer AI")
        mode = str(self.cfg.get("gui.compact", "auto")).lower()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.compact = mode in ("1", "true", "yes", "on") or (mode == "auto" and (sw < 1000 or sh < 700))
        if self.compact:
            self.root.geometry(f"{min(sw, 800)}x{min(sh - 40, 460)}")
        else:
            self.root.geometry("980x660")
        self.root.minsize(420, 300)
        self.busy = False
        self.update_notice = ""
        self._hist_seen = 0
        self._chat_seen = 0
        self._cards: dict[str, dict] = {}
        self._theme()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(400, self.refresh)
        if self.cfg.get("update.check_on_start", True):
            threading.Thread(target=self._update_check, daemon=True).start()

    # ================================================================== look
    def _theme(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg=BG)
        style.configure(".", background=BG, foreground=FG, fieldbackground=FIELD, bordercolor="#3a3c41",
                        lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL, focuscolor=ACCENT,
                        selectbackground=ACCENT, selectforeground="#ffffff", insertcolor=FG)
        pad = (8, 6) if self.compact else (12, 8)
        style.configure("TButton", background=PANEL, foreground=FG, padding=pad)
        style.map("TButton", background=[("active", "#404249"), ("pressed", ACCENT)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", padding=pad)
        style.map("Accent.TButton", background=[("active", "#5a92ff")])
        style.configure("TLabelframe", background=BG, bordercolor="#3a3c41")
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT)
        style.configure("TEntry", fieldbackground=FIELD, foreground=FG)
        style.configure("TCombobox", fieldbackground=FIELD, foreground=FG, background=PANEL, arrowcolor=FG)
        style.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", FG)])
        style.configure("Card.TFrame", background=PANEL)
        style.configure("Sel.TFrame", background="#35373d")
        for opt, val in (("*Text.Background", FIELD), ("*Text.Foreground", FG), ("*Text.selectBackground", ACCENT),
                         ("*Listbox.Background", FIELD), ("*Listbox.Foreground", FG),
                         ("*TCombobox*Listbox.background", FIELD), ("*TCombobox*Listbox.foreground", FG),
                         ("*Toplevel.Background", BG)):
            self.root.option_add(opt, val)

    # ================================================================== layout
    def _build(self):
        # ---- hero cards
        self.cards_scroll = ScrollFrame(self.root, padding=(6, 6, 6, 2), orient="horizontal",
                                        height=76 if self.compact else 84)
        self.cards_scroll.pack(fill="x")
        self.cards_row = self.cards_scroll.inner
        self._rebuild_cards()

        # ---- the only buttons
        bar = ttk.Frame(self.root, padding=(6, 2))
        bar.pack(fill="x")
        buttons = [("Start bot", self.do_start, "Accent.TButton"), ("Stop", self.do_stop, "TButton"),
                   ("Server", self.do_server, "TButton"), ("Join server", self.do_join, "TButton"),
                   ("Import brain backup", self.do_model, "TButton"), ("Train overnight", self.do_train, "TButton"),
                   ("Train one skill", self.do_train_skill, "TButton"),
                   ("Settings", self.do_settings, "TButton")]
        per_row = 4 if self.compact else 7
        for i, (text, cmd, style) in enumerate(buttons):
            ttk.Button(bar, text=text, command=cmd, style=style).grid(row=i // per_row, column=i % per_row,
                                                                       padx=2, pady=2, sticky="ew")
        for c in range(per_row):
            bar.columnconfigure(c, weight=1)

        # ---- direct chat with the selected bot (also supports action requests)
        ask = ttk.Frame(self.root, padding=(8, 2, 8, 0))
        ask.pack(fill="x")
        self.ask_var = tk.StringVar()
        entry = ttk.Entry(ask, textvariable=self.ask_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.insert(0, "")
        entry.bind("<Return>", lambda e: self.do_tell())
        ttk.Button(ask, text="Send", command=self.do_tell).pack(side="left", padx=(4, 0))
        self._ask_hint = ttk.Label(self.root, text="Message your bot during mining or exploring: hi · find iron · build a shelter · "
                                   "kill that zombie · what does a blaze drop? · stop", foreground=MUTED, padding=(10, 0))
        self._ask_hint.pack(fill="x")

        # ---- two-way in-app conversation, separate from low-level activity log
        chatf = ttk.LabelFrame(self.root, text="Bot conversation", padding=(6, 2))
        chatf.pack(fill="x", padx=6, pady=(2, 4))
        self.chat = tk.Text(chatf, height=4 if self.compact else 6, wrap="word",
                            state="disabled", relief="flat", font="TkDefaultFont", padx=8, pady=4)
        chatsb = ttk.Scrollbar(chatf, orient="vertical", command=self.chat.yview)
        self.chat.configure(yscrollcommand=chatsb.set)
        self.chat.pack(side="left", fill="both", expand=True)
        chatsb.pack(side="right", fill="y")
        self.model_info = tk.StringVar()
        ttk.Label(self.root, textvariable=self.model_info, foreground=MUTED,
                  padding=(8, 2), wraplength=760 if self.compact else 940).pack(fill="x")

        # ---- status + log + join info
        self.status_var = tk.StringVar(value="Select a bot and press Start bot.")
        ttk.Label(self.root, textvariable=self.status_var, padding=(8, 4), foreground=FG,
                  wraplength=760 if self.compact else 940).pack(fill="x")
        logf = ttk.Frame(self.root, padding=(6, 0, 6, 0))
        logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, wrap="word", state="disabled", relief="flat", font="TkFixedFont", padx=6, pady=4)
        sb = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.join_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.join_var, padding=(8, 4), foreground=MUTED,
                  font="TkFixedFont").pack(fill="x")

    # ================================================================== hero cards
    def _rebuild_cards(self):
        for w in self.cards_row.winfo_children():
            w.destroy()
        self._cards.clear()
        big = not self.compact
        for i, name in enumerate(self.rt.profiles()):
            card = ttk.Frame(self.cards_row, style="Sel.TFrame" if name == self.rt.current else "Card.TFrame",
                             padding=8)
            card.pack(side="left", padx=4, fill="y")
            av = tk.Canvas(card, width=48 if big else 36, height=48 if big else 36, highlightthickness=0,
                           bg=CARD_COLORS[i % len(CARD_COLORS)])
            av.create_text((24 if big else 18), (24 if big else 18), text=name[:1].upper(), fill="white",
                           font=("TkDefaultFont", 18 if big else 14, "bold"))
            av.pack(side="left", padx=(0, 8))
            info = ttk.Frame(card, style=card.cget("style"))
            info.pack(side="left")
            ttk.Label(info, text=name, font=("TkDefaultFont", 11, "bold"), background=card_bg(card)).pack(anchor="w")
            st = tk.StringVar(value="offline")
            ttk.Label(info, textvariable=st, foreground=MUTED, background=card_bg(card)).pack(anchor="w")
            hp = tk.Canvas(info, width=110, height=6, highlightthickness=0, bg="#3a3c41")
            hp.pack(anchor="w", pady=(4, 0))
            bar = hp.create_rectangle(0, 0, 0, 6, fill="#e5484d", width=0)
            self._cards[name] = {"frame": card, "status": st, "hp": hp, "bar": bar}
            for w in (card, av, info, *info.winfo_children()):
                w.bind("<Button-1>", lambda e, n=name: self.select(n))
        if len(self.rt.profiles()) < 5:
            add = ttk.Frame(self.cards_row, style="Card.TFrame", padding=8)
            add.pack(side="left", padx=4, fill="y")
            lbl = ttk.Label(add, text="+\nNew bot", justify="center", foreground=ACCENT, background=PANEL,
                            font=("TkDefaultFont", 11, "bold"))
            lbl.pack(expand=True, padx=14)
            for w in (add, lbl):
                w.bind("<Button-1>", lambda e: self.add_bot())

    def select(self, name):
        self.rt.current = name
        self._hist_seen = 0
        self._chat_seen = 0
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._rebuild_cards()

    def add_bot(self):
        name = simpledialog.askstring("New bot", "Name for the new bot (3-16 letters/digits):", parent=self.root)
        if not name:
            return
        try:
            name = self.rt.add_profile(name.strip())
        except ValueError as e:
            messagebox.showerror("New bot", str(e))
            return
        self.select(name)

    # ================================================================== actions
    def _bg(self, label, fn):
        # steps may overlap (e.g. Start bot while the server is still starting): the runtime serialises them

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

    def do_start(self):
        def go():
            self.rt.start_agent(self.rt.current)
            if not self.rt.llm.configured and not getattr(self, "_ai_hint_shown", False):
                self._ai_hint_shown = True
                self.root.after(0, lambda: messagebox.showinfo(
                    "AI", "The bot is playing on its own (planner, survival, chat commands).\n"
                          "For the AI brain, open Settings and choose an AI provider + key."))
        self._bg("Start bot", go)

    def do_tell(self):
        text = self.ask_var.get().strip()
        if not text:
            return
        self.ask_var.set("")

        def go():
            agent = self.rt.session().agent or self.rt.start_agent(self.rt.current)
            agent.instruct(text, sender="app")
        self._bg("Message bot", go)

    def do_stop(self):
        s = self.rt.sessions.get(self.rt.current)
        if not s:
            return
        if s.trainer and s.trainer.running:
            if not messagebox.askyesno("Stop", "Stop overnight training for this bot?"):
                return
            s.trainer.stop()
        if s.agent:
            s.agent.stop_goal()
            with s.agent.tasks.mutex:
                s.agent.tasks.queue.clear()
        if s.bot:
            s.bot.stop()
            s.bot.disconnect("bye")

    def do_server(self):
        srv = self.rt.server
        if srv and srv.running:
            if messagebox.askyesno("Server", "Stop the Minecraft server? Bots on it will disconnect."):
                self._bg("Server", srv.stop)
            return
        self.cfg.set("server.host", "127.0.0.1")
        self.cfg.set("server.auto_start", True)
        self.cfg.save()
        self._bg("Server", self.rt.start_server)

    def do_join(self):
        cur = f"{self.cfg['server']['host']}:{self.cfg['server']['port']}"
        addr = simpledialog.askstring("Join server", "Server address to join (host:port):\n"
                                      "(another PyFlayer server, or any offline-mode 26.3 server)",
                                      initialvalue=cur, parent=self.root)
        if not addr:
            return
        host, _, port = addr.strip().partition(":")
        try:
            port_n = int(port or 25565)
        except ValueError:
            messagebox.showerror("Join server", "Port must be a number")
            return
        local = host in ("127.0.0.1", "localhost")
        self.cfg.set("server.host", host)
        self.cfg.set("server.port", port_n)
        self.cfg.set("server.auto_start", local)
        self.cfg.save()
        s = self.rt.session()
        if s.bot:                      # reconnect the selected bot to the new address
            s.bot.disconnect("bye")
            s.bot.host, s.bot.port = host, port_n
        self._bg("Join server", lambda: self.rt.start_agent(self.rt.current))

    def do_model(self):
        path = filedialog.askopenfilename(title="Import a PyFlayer brain backup (.json.gz)", parent=self.root,
                                          filetypes=[("PyFlayer model", "*.json.gz *.gz"), ("All files", "*")])
        if not path:
            return

        def go():
            brain = self.rt.open_brain(self.rt.current)
            brain.import_(path)
            self.rt.session().bot and self.rt.session().bot.learned_hardness.update(brain.learned_hardness())
            self.rt.start_agent(self.rt.current)
            self.root.after(0, lambda: messagebox.showinfo(
                "Brain backup imported", f"Imported memory and skills from:\n{path}\n\n"
                f"Saved into: {DATA_DIR / 'brain' / (self.rt.current + '.db')}"))
        self._bg("Run from model file", go)

    def do_train(self):
        hours = self.cfg["training"]["hours"]
        if not messagebox.askyesno("Train overnight", f"Train {self.rt.current} for {hours} hours?\n\n"
                                   "It will practise beating the game, build its knowledge and save everything to "
                                   "its model file. Training now prioritizes actual goals, verifies each result, "
                                   "and reports when blocked. Press Stop to end early."):
            return
        self._bg("Train overnight", lambda: self.rt.start_training(hours, True, self.rt.current))

    def do_train_skill(self):
        def go():
            agent = self.rt.start_agent(self.rt.current)
            with agent._actuation_lock:
                result = agent.skill_coach.practice_one()
            self.root.after(0, lambda: messagebox.showinfo("Skill practice", result))
        self._bg("Train one skill", go)

    def do_settings(self):
        SettingsDialog(self)

    def _update_check(self):
        from .. import updater
        try:
            info = updater.check(self.cfg.get("update.repo") or updater.DEFAULT_REPO)
        except Exception:  # noqa: BLE001
            return
        if info.get("available"):
            self.update_notice = f"{info['message']} (Settings > Check for updates)"

    # ================================================================== refresh (twice a second, cheap)
    def refresh(self):
        try:
            self._refresh()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self.root.after(int(self.cfg["gui"]["refresh_ms"]), self.refresh)

    def _refresh(self):
        rt = self.rt
        for name, c in self._cards.items():
            s = rt.sessions.get(name)
            b = s.bot if s else None
            goal = getattr(s.agent, "goal_trainer", None) if s and s.agent else None
            if s and s.trainer and s.trainer.running:
                text = f"training: {s.trainer.current or 'starting'}"
            elif s and s.trainer and s.trainer.status in ("blocked", "error"):
                text = "training " + s.trainer.status
            elif goal and goal.running:
                text = f"goal: {goal.current.replace('_', ' ')}"
            elif b and b.spawned.is_set():
                text = f"{b.current_action if b.current_action != 'idle' else 'online'} · {b.dimension}"
            elif b and b.connected:
                text = "connecting"
            else:
                text = "offline"
            c["status"].set(text)
            hp = (b.health / 20) if b and b.spawned.is_set() else 0
            c["hp"].coords(c["bar"], 0, 0, int(110 * max(0, min(1, hp))), 6)
        s = rt.sessions.get(rt.current)
        b = s.bot if s else None
        parts = [f"{rt.current}"]
        if b and b.spawned.is_set():
            x, y, z = b.position
            parts.append(f"{b.dimension} {x:.0f},{y:.0f},{z:.0f} · ♥ {b.health:.0f}/20 · food {b.food}/20 · "
                         f"{b.time_phase()}")
            if s.agent and s.agent.current_task:
                parts.append(f"task: {s.agent.current_task[:60]}")
            if s.trainer:
                p = s.trainer.progress()
                done = sum(m["achieved"] for m in p["milestones"])
                if s.trainer.running or p["status"] in ("blocked", "error"):
                    parts.append(f"training {p['status']} · goal {p['current'] or '-'} · "
                                 f"{p['completed_attempts']} attempts / {p['successful_attempts']} successes · "
                                 f"milestones {done}/{len(p['milestones'])}")
                    if p["status"] in ("blocked", "error") and p["last_error"]:
                        parts.append("training issue: " + p["last_error"][:180])
        else:
            parts.append(rt.status if rt.status not in ("idle",) else "offline")
        if self.update_notice:
            parts.append(self.update_notice)
        if rt.server and rt.server.port_notice:
            parts.append(rt.server.port_notice)
        self.status_var.set("   |   ".join(parts))
        if s and s.agent:
            h = s.agent.history
            if len(h) != self._hist_seen:
                new = h[self._hist_seen:] if self._hist_seen <= len(h) else h[-80:]
                self.log.configure(state="normal")
                self.log.insert("end", "\n".join(new) + "\n")
                lines = int(self.log.index("end-1c").split(".")[0])
                if lines > 1500:
                    self.log.delete("1.0", f"{lines - 1500}.0")
                self.log.see("end")
                self.log.configure(state="disabled")
                self._hist_seen = len(h)
            messages = s.agent.messages
            if len(messages) != self._chat_seen:
                recent = messages[self._chat_seen:] if self._chat_seen <= len(messages) else messages[-80:]
                self.chat.configure(state="normal")
                for stamp, sender, msg in recent:
                    self.chat.insert("end", time.strftime("%H:%M", time.localtime(stamp)) +
                                     "  " + str(sender) + ": " + str(msg) + "\n")
                line_count = int(self.chat.index("end-1c").split(".")[0])
                if line_count > 450:
                    self.chat.delete("1.0", f"{line_count - 450}.0")
                self.chat.see("end")
                self.chat.configure(state="disabled")
                self._chat_seen = len(messages)
        self.model_info.set("Brain file: " + str(DATA_DIR / "brain" / (rt.current + ".db")) +
                            "  |  Training backup: " +
                            str(DATA_DIR / "brain" / (rt.current + "-export.json.gz")))
        srv = rt.server
        if srv and srv.running:
            info = rt.join_info()
            self.join_var.set("Join: " + "   ·   ".join(l.split(":", 1)[1].strip() for l in info[1:4]))
        elif not rt.local_server:
            self.join_var.set(f"Joined server: {self.cfg['server']['host']}:{self.cfg['server']['port']}")
        else:
            self.join_var.set("Server: stopped")

    def on_close(self):
        if any(s.trainer and s.trainer.running for s in self.rt.sessions.values()):
            if not messagebox.askyesno("Quit", "Training is running. Stop it and quit?"):
                return
        self.root.destroy()
        # A daemon shutdown thread gets killed when the GUI's main thread exits.
        # Keep it non-daemon so the vanilla server can save worlds and stop cleanly.
        threading.Thread(target=self.rt.shutdown, name="pyflayer-shutdown",
                         daemon=False).start()

    def run(self):
        self.root.mainloop()


def card_bg(frame) -> str:
    return "#35373d" if frame.cget("style") == "Sel.TFrame" else PANEL


class SettingsDialog:
    FIELDS = [("llm.api_key", "AI API key"), ("llm.model", "AI model (blank = provider default)"),
              ("llm.base_url", "Base URL (only for custom / local AI)"),
              ("llm.models.chat", "Groq fast chat / routing model"),
              ("llm.models.world", "Groq world / inventory model"),
              ("llm.models.logic", "Groq precise logic / tools model"),
              ("llm.models.fallback", "Groq logic fallback model"),
              ("server.port", "Server port"),
              ("server.auto_port_fallback", "Automatically use another port if busy (true / false)"),
              ("server.difficulty", "Difficulty (easy / normal / hard)"),
              ("training.keep_inventory", "Keep inventory while training (true / false)"),
              ("training.allow_advanced_moves", "Experimental trained lava jumps (default false)"),
              ("training.hours", "Overnight training hours"),
              ("agent.owner", "Bots only obey this player (blank = anyone)"),
              ("gui.compact", "Small-screen layout (auto / true / false)")]

    def __init__(self, app: App):
        self.app = app
        cfg = app.cfg
        self.win = tk.Toplevel(app.root)
        self.win.title("Settings")
        self.win.configure(bg=BG)
        self.win.transient(app.root)
        sw, sh = app.root.winfo_screenwidth(), app.root.winfo_screenheight()
        self.win.geometry(f"{min(760, sw - 20)}x{min(640, sh - 80)}")
        scroll = ScrollFrame(self.win, padding=10)
        scroll.pack(fill="both", expand=True)
        frm = scroll.inner
        from ..agent.llm import PROVIDERS
        self.vars = {}
        ttk.Label(frm, text="AI provider").grid(row=0, column=0, sticky="w", pady=2)
        self.provider = tk.StringVar(value=cfg.get("llm.provider") or "custom")
        box = ttk.Combobox(frm, textvariable=self.provider, values=list(PROVIDERS), state="readonly",
                           width=24 if app.compact else 46)
        box.grid(row=0, column=1, pady=2, sticky="w")
        self.hint = tk.StringVar()
        ttk.Label(frm, textvariable=self.hint, foreground=MUTED, wraplength=520 if app.compact else 700).grid(
            row=1, column=0, columnspan=2, sticky="w")
        box.bind("<<ComboboxSelected>>", lambda e: self._provider_changed(reset=True))
        for i, (key, label) in enumerate(self.FIELDS, start=2):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=str(cfg.get(key, "")))
            ttk.Entry(frm, textvariable=v, width=26 if app.compact else 48,
                      show="*" if "api_key" in key else "").grid(row=i, column=1, pady=2, sticky="w")
            self.vars[key] = v
        self._provider_changed(reset=False)
        n = len(self.FIELDS) + 2
        self.test_var = tk.StringVar()
        ttk.Label(frm, textvariable=self.test_var, wraplength=520).grid(row=n, column=0, columnspan=2, sticky="w")
        btns = ttk.Frame(frm)
        btns.grid(row=n + 1, column=1, sticky="e", pady=8)
        ttk.Button(btns, text="Check for updates", command=self.check_updates).pack(side="left", padx=4)
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

    def check_updates(self):
        from .. import updater
        repo = self.app.cfg.get("update.repo") or updater.DEFAULT_REPO
        self.test_var.set("checking for updates...")

        ui = self.app.root.after          # Tk may only be touched from the main thread

        def say(text):
            ui(0, lambda: self.test_var.set(text))

        def installed(msg, ok):
            self.test_var.set(msg)
            if ok and messagebox.askyesno("Update", f"{msg}\n\nRestart PyFlayer now?", parent=self.win):
                self.app.rt.shutdown()
                updater.restart_app()
            elif not ok:
                messagebox.showerror("Update", msg, parent=self.win)

        def install():
            try:
                msg, ok = updater.update(repo, progress=say), True
            except Exception as e:  # noqa: BLE001
                msg, ok = f"update failed: {e}", False
            ui(0, lambda: installed(msg, ok))

        def ask(info):
            self.test_var.set(info["message"])
            if info["available"] and messagebox.askyesno(
                    "Update", f"{info['message']}\n\nInstall it now? Your world, bots' memory and settings are kept.",
                    parent=self.win):
                self.test_var.set("updating...")
                threading.Thread(target=install, daemon=True).start()

        def check():
            try:
                info = updater.check(repo)
            except Exception as e:  # noqa: BLE001
                info = {"available": False, "message": f"update check failed: {e}"}
            ui(0, lambda: ask(info))
        threading.Thread(target=check, daemon=True).start()

    def test_ai(self):
        from ..agent.llm import LLM
        cfg = dict(self.app.cfg["llm"])
        cfg["provider"] = self.provider.get()
        for k in ("llm.api_key", "llm.model", "llm.base_url"):
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
