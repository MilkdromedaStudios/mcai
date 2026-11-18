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
from .theme import (ACCENT, BG, CARD_COLORS, DANGER, FG, FONT_BOLD, MUTED, SELECTED, SUCCESS, SURFACE,
                    TextPanel, Theme)


class ScrollFrame(ttk.Frame):
    """A frame whose content scrolls with the scrollbar, mouse wheel or touch-pad.

    The scrollbar only shows when the content doesn't fit, and the content stretches to fill the space
    when it does fit (so text boxes inside can still grow with the window)."""

    def __init__(self, master, padding=0, orient="vertical", height=None, fill=True):
        super().__init__(master)
        self.orient = orient
        self.fill = fill
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0, height=height or 0)
        self.canvas.scrollable = self
        vertical = orient == "vertical"
        self.bar = ttk.Scrollbar(self, orient=orient,
                                 command=self.canvas.yview if vertical else self.canvas.xview)
        self.canvas.configure(**{("yscrollcommand" if vertical else "xscrollcommand"): self._set})
        self.inner = ttk.Frame(self.canvas, padding=padding)
        self.inner.scrollable = self
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        # grid keeps a fixed slot for the bar, so showing/hiding it can't make the layout jump or loop
        if vertical:
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.bar.grid(row=0, column=1, sticky="ns", padx=(2, 2), pady=2)
            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)
        else:
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.bar.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 2))
            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)
        self.inner.bind("<Configure>", self._layout)
        self.canvas.bind("<Configure>", self._layout)
        # the content's *requested* size can change without any <Configure> event (text wraps, rows added),
        # so re-check it now and then; otherwise the scroll area goes stale and the scrollbar breaks
        self._last_req = None
        self.after(300, self._watch)

    def _watch(self):
        try:
            req = (self.inner.winfo_reqwidth(), self.inner.winfo_reqheight(),
                   self.canvas.winfo_width(), self.canvas.winfo_height())
        except tk.TclError:
            return
        if req != self._last_req:
            self._last_req = req
            self._layout()
        self.after(300, self._watch)

    def _layout(self, _e=None):
        c, inner = self.canvas, self.inner
        cw, ch = c.winfo_width(), c.winfo_height()
        if self.orient == "vertical":
            h = max(inner.winfo_reqheight(), ch if self.fill else 0)
            c.itemconfigure(self._win, width=cw, height=h)
            c.configure(scrollregion=(0, 0, cw, h))
        else:
            w = max(inner.winfo_reqwidth(), cw)
            c.itemconfigure(self._win, width=w, height=ch)
            c.configure(scrollregion=(0, 0, w, ch))

    def _set(self, lo, hi):
        self.bar.set(lo, hi)
        fits = float(lo) <= 0.0 and float(hi) >= 1.0
        # keep the slot but hide the thumb when everything fits
        self.bar.state(["disabled"] if fits else ["!disabled"])
        if fits:
            self.bar.grid_remove()
        else:
            self.bar.grid()

    def wheel(self, units):
        lo, hi = (self.canvas.yview() if self.orient == "vertical" else self.canvas.xview())
        if lo <= 0.0 and hi >= 1.0:
            return False
        (self.canvas.yview_scroll if self.orient == "vertical" else self.canvas.xview_scroll)(units, "units")
        return True


def _wheel_target(widget):
    """Nearest scrollable thing under the mouse: a Text box, or a ScrollFrame."""
    w = widget
    while w is not None:
        if isinstance(w, tk.Text):
            return w
        sf = getattr(w, "scrollable", None)
        if sf is not None:
            return sf
        w = getattr(w, "master", None)
    return None


def install_wheel(root):
    """One global mouse-wheel handler: scroll whatever is under the pointer (Text first, then the page).
    Works on Windows/macOS (<MouseWheel>) and Linux / Raspberry Pi (<Button-4/5>)."""
    def on_wheel(e, units):
        try:
            w = root.winfo_containing(e.x_root, e.y_root)
        except (KeyError, tk.TclError):
            return
        t = _wheel_target(w)
        while t is not None:
            if isinstance(t, tk.Text):
                lo, hi = t.yview()
                if not (lo <= 0.0 and hi >= 1.0) and not (units < 0 and lo <= 0.0) and not (units > 0 and hi >= 1.0):
                    t.yview_scroll(units, "units")
                    return "break"
            elif t.wheel(units if t.orient == "vertical" or not (e.state & 1) else units):
                return "break"
            t = _wheel_target(getattr(t, "master", None))
        return "break"

    root.bind_all("<MouseWheel>", lambda e: on_wheel(e, -1 if e.delta > 0 else 1), add="")
    root.bind_all("<Shift-MouseWheel>", lambda e: on_wheel(e, -1 if e.delta > 0 else 1), add="")
    root.bind_all("<Button-4>", lambda e: on_wheel(e, -2), add="")
    root.bind_all("<Button-5>", lambda e: on_wheel(e, 2), add="")


def _rounded(canvas, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2, x1, y2,
           x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


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
            self.root.geometry(f"{min(sw, 720)}x{min(sh - 40, 480)}")
        else:
            self.root.geometry("1000x700")
        self.root.minsize(360, 280)
        self.busy = False
        self.update_notice = ""
        self._hist_seen = 0
        self._chat_seen = 0
        self._cards: dict[str, dict] = {}
        self._theme()
        install_wheel(self.root)
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(400, self.refresh)
        if self.cfg.get("update.check_on_start", True):
            threading.Thread(target=self._update_check, daemon=True).start()

    # ================================================================== look
    def _theme(self):
        self.theme = Theme(self.root, self.compact)

    # ================================================================== layout
    def _build(self):
        c = self.compact
        gap = 6 if c else 10
        # everything lives in one page that scrolls when the window is too small
        page = ScrollFrame(self.root, padding=(gap, gap, gap, gap))
        page.pack(fill="both", expand=True)
        root = self.page = page.inner

        # ---- header
        head = ttk.Frame(root)
        head.pack(fill="x", pady=(0, gap // 2))
        ttk.Label(head, text="PyFlayer", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="  AI Minecraft bots", style="Muted.TLabel").pack(side="left", pady=(3, 0))
        ttk.Button(head, text="Settings", command=self.do_settings).pack(side="right")

        # ---- hero cards (scroll sideways when there are many bots)
        self.cards_scroll = ScrollFrame(root, padding=(0, 2, 0, 4), orient="horizontal",
                                        height=72 if c else 84)
        self.cards_scroll.pack(fill="x")
        self.cards_row = self.cards_scroll.inner
        self._rebuild_cards()

        # ---- actions
        ttk.Label(root, text="BOT", style="Section.TLabel").pack(anchor="w", pady=(gap // 2, 2))
        bar = ttk.Frame(root)
        bar.pack(fill="x")
        buttons = [("Start bot", self.do_start, "Accent.TButton"), ("Stop", self.do_stop, "Danger.TButton"),
                   ("Server", self.do_server, "TButton"), ("Join server", self.do_join, "TButton"),
                   ("Train overnight", self.do_train, "TButton"), ("Train one skill", self.do_train_skill, "TButton"),
                   ("Import model", self.do_model, "TButton"),
                   ("Export model", self.do_export_model, "TButton")]
        self._btns = [ttk.Button(bar, text=t, command=cmd, style=st) for t, cmd, st in buttons]
        self._btn_cols = 0

        def regrid(e=None):
            width = e.width if e else (bar.winfo_width() or self.root.winfo_width())
            longest = max(len(t) for t, _c, _s in buttons)
            fit = width // max(110, longest * 7 + 24)
            cols = 4 if fit >= 4 else 2 if fit >= 2 else 1
            if cols == self._btn_cols:
                return
            self._btn_cols = cols
            for i, b in enumerate(self._btns):
                b.grid(row=i // cols, column=i % cols, padx=3, pady=3, sticky="ew")
            for col in range(4):
                bar.columnconfigure(col, weight=1 if col < cols else 0, uniform="b" if col < cols else "")
        bar.bind("<Configure>", regrid)
        regrid()

        # ---- status pill
        st = ttk.Frame(root, style="Card.TFrame", padding=(12, 6) if c else (14, 8))
        st.pack(fill="x", pady=(gap, 0))
        self.status_dot = tk.Canvas(st, width=10, height=10, bg=SURFACE, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 8))
        self._dot = self.status_dot.create_oval(1, 1, 9, 9, fill=MUTED, width=0)
        self.status_var = tk.StringVar(value="Select a bot and press Start bot.")
        self.status_label = ttk.Label(st, textvariable=self.status_var, style="Panel.TLabel", wraplength=600)
        self.status_label.pack(side="left", fill="x", expand=True)
        st.bind("<Configure>", lambda e: self.status_label.configure(wraplength=max(200, e.width - 50)))

        # ---- message box
        ask = ttk.Frame(root)
        ask.pack(fill="x", pady=(gap, 0))
        self.ask_var = tk.StringVar()
        entry = ttk.Entry(ask, textvariable=self.ask_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.do_tell())
        ttk.Button(ask, text="Send", style="Accent.TButton", command=self.do_tell).pack(side="left", padx=(6, 0))
        self._placeholder(entry, "Tell your bot anything: find iron, build a shelter, get to the End ...")

        # ---- conversation + activity, side by side on wide windows, stacked on small ones
        panes = ttk.Frame(root)
        panes.pack(fill="both", expand=True, pady=(gap, 0))
        convo = TextPanel(panes, "Conversation", height=5 if c else 6, compact=c)
        activity = TextPanel(panes, "Activity", height=6 if c else 6, mono=True, compact=c)
        self.chat, self.log = convo.text, activity.text
        self.chat.tag_configure("you", foreground=ACCENT, font=FONT_BOLD)
        self.chat.tag_configure("bot", foreground=SUCCESS, font=FONT_BOLD)
        self.chat.tag_configure("time", foreground=MUTED)
        self.log.configure(fg="#c3c7d0")
        self._panes = (panes, convo, activity)
        self._stacked = None
        panes.bind("<Configure>", self._arrange_panes)
        self._arrange_panes()

        # ---- footer: join address + model files
        self.join_var = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.join_var, style="Muted.TLabel", font="TkFixedFont").pack(
            fill="x", pady=(gap // 2, 0))
        self.model_info = tk.StringVar()
        self.model_label = ttk.Label(root, textvariable=self.model_info, style="Muted.TLabel",
                                     font=("DejaVu Sans", 8), wraplength=900)
        self.model_label.pack(fill="x")
        root.bind("<Configure>", lambda e: self.model_label.configure(wraplength=max(200, e.width - 20)), add="+")

    def _arrange_panes(self, _e=None):
        panes, convo, activity = self._panes
        width = panes.winfo_width() or self.root.winfo_width()
        stacked = width < 760
        if stacked == self._stacked:
            return
        self._stacked = stacked
        convo.grid_forget()
        activity.grid_forget()
        for i in range(2):
            panes.columnconfigure(i, weight=0, uniform="")
            panes.rowconfigure(i, weight=0)
        if stacked:
            convo.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
            activity.grid(row=1, column=0, sticky="nsew")
            panes.columnconfigure(0, weight=1)
            panes.rowconfigure(0, weight=1)
            panes.rowconfigure(1, weight=1)
        else:
            convo.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
            activity.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
            panes.columnconfigure(0, weight=1, uniform="p")
            panes.columnconfigure(1, weight=1, uniform="p")
            panes.rowconfigure(0, weight=1)

    def _placeholder(self, entry, text):
        var = entry.cget("textvariable")
        state = {"on": False}

        def show(_e=None):
            if not entry.get():
                entry.insert(0, text)
                entry.configure(foreground=MUTED)
                state["on"] = True

        def hide(_e=None):
            if state["on"]:
                entry.delete(0, "end")
                entry.configure(foreground=FG)
                state["on"] = False
        entry.bind("<FocusIn>", hide, add="+")
        entry.bind("<FocusOut>", show, add="+")
        self._ph = state
        show()
        del var

    # ================================================================== hero cards
    def _rebuild_cards(self):
        for w in self.cards_row.winfo_children():
            w.destroy()
        self._cards.clear()
        big = not self.compact
        av_size = 40 if big else 32
        for i, name in enumerate(self.rt.profiles()):
            sel = name == self.rt.current
            bg = SELECTED if sel else SURFACE
            card = ttk.Frame(self.cards_row, style="Sel.TFrame" if sel else "Card.TFrame",
                             padding=(10, 8) if big else (8, 6), cursor="hand2")
            card.pack(side="left", padx=(0, 8), fill="y")
            av = tk.Canvas(card, width=av_size, height=av_size, highlightthickness=0, bg=bg, cursor="hand2")
            col = CARD_COLORS[i % len(CARD_COLORS)]
            av.create_oval(1, 1, av_size - 1, av_size - 1, fill=col, outline="")
            av.create_text(av_size // 2, av_size // 2, text=name[:1].upper(), fill="white",
                           font=("DejaVu Sans", 15 if big else 12, "bold"))
            av.pack(side="left", padx=(0, 10))
            info = tk.Frame(card, bg=bg, cursor="hand2")
            info.pack(side="left")
            title = tk.Label(info, text=name, bg=bg, fg=FG, font=FONT_BOLD, cursor="hand2")
            title.pack(anchor="w")
            st = tk.StringVar(value="offline")
            sub = tk.Label(info, textvariable=st, bg=bg, fg=MUTED, font=("DejaVu Sans", 8), cursor="hand2")
            sub.pack(anchor="w")
            hp = tk.Canvas(info, width=110, height=6, highlightthickness=0, bg=bg)
            hp.pack(anchor="w", pady=(4, 0))
            _rounded(hp, 0, 0, 110, 6, 3, fill="#2c2f37", outline="")
            bar = _rounded(hp, 0, 0, 0, 6, 3, fill=DANGER, outline="")
            self._cards[name] = {"frame": card, "status": st, "hp": hp, "bar": bar, "hpw": -1}
            for w in (card, av, info, title, sub, hp):
                w.bind("<Button-1>", lambda e, n=name: self.select(n))
        if len(self.rt.profiles()) < 5:
            add = ttk.Frame(self.cards_row, style="Card.TFrame", padding=(14, 8), cursor="hand2")
            add.pack(side="left", fill="y")
            lbl = tk.Label(add, text="＋  New bot", fg=ACCENT, bg=SURFACE, font=FONT_BOLD, cursor="hand2")
            lbl.pack(expand=True)
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
        if getattr(self, "_ph", {}).get("on"):
            return
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
        path = filedialog.askopenfilename(
            title="Import universal model (any world) or full brain backup (same world)",
            parent=self.root, filetypes=[("PyFlayer model", "*.json.gz *.gz"), ("All files", "*")])
        if not path:
            return

        def go():
            brain = self.rt.open_brain(self.rt.current)
            brain.import_(path)
            self.rt.session().bot and self.rt.session().bot.learned_hardness.update(brain.learned_hardness())
            self.rt.start_agent(self.rt.current)
            self.root.after(0, lambda: messagebox.showinfo(
                "Model imported", f"Imported compatible skills and knowledge from:\n{path}\n\n"
                "A universal model does not import the old world's coordinates or completed quests.\n"
                f"Saved into: {DATA_DIR / 'brain' / (self.rt.current + '.db')}"))
        self._bg("Import model", go)

    def do_export_model(self):
        def go():
            brain = self.rt.open_brain(self.rt.current)
            output = brain.export_universal()
            full = brain.export_latest()
            self.root.after(0, lambda: messagebox.showinfo(
                "Universal model exported",
                f"Reusable on other seeds and worlds:\n{output}\n\n"
                f"Complete backup for this world:\n{full}\n\n"
                "Use Import model on a new bot profile to reuse the learned skills."))
        self._bg("Export universal model", go)

    def do_train(self):
        hours = self.cfg["training"]["hours"]
        if not messagebox.askyesno("Train overnight", f"Train {self.rt.current} for {hours} hours?\n\n"
                                   "It will practise beating the game, build its knowledge and save everything to "
                                   "its model file. Training now prioritizes actual goals, verifies each result, "
                                   "and reports when blocked. Press Stop to end early."):
            return
        self._bg("Train overnight", lambda: self.rt.start_training(hours, True, self.rt.current))

    def do_train_skill(self):
        """Pick what to train: one skill drill, or train until a goal is reached."""
        from ..agent.skill_training import DRILL_LABELS
        win = tk.Toplevel(self.root)
        win.title("Train")
        win.transient(self.root)
        win.configure(bg=BG)
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=f"What should {self.rt.current} train?").pack(anchor="w", pady=(0, 8))

        def drill(name):
            win.destroy()

            def go():
                agent = self.rt.start_agent(self.rt.current)
                with agent._actuation_lock:
                    result = agent.skill_coach.practice_one(name)
                    universal = agent.brain.export_universal()
                    agent.brain.export_latest()
                self.root.after(0, lambda: messagebox.showinfo(
                    "Skill practice", result + f"\n\nUniversal model saved: {universal}"))
            self._bg("Train " + DRILL_LABELS.get(name, name or "skill"), go)

        def goal():
            text = simpledialog.askstring(
                "Train to goal", "Goal to train towards (e.g. 'get to the end', 'iron armor', 'blaze rods',\n"
                "'beat the dragon', or anything in plain English):", parent=win)
            if not text:
                return
            win.destroy()

            def go():
                agent = self.rt.start_agent(self.rt.current)
                agent.instruct(text, "app")
            self._bg("Train to goal", go)

        ttk.Button(frm, text="Least-practised skill (auto)", command=lambda: drill(None)).pack(fill="x", pady=2)
        for name, label in DRILL_LABELS.items():
            ttk.Button(frm, text=label, command=lambda n=name: drill(n)).pack(fill="x", pady=2)
        ttk.Separator(frm).pack(fill="x", pady=6)
        ttk.Button(frm, text="Train to goal...", command=goal).pack(fill="x", pady=2)
        ttk.Button(frm, text="Cancel", command=win.destroy).pack(fill="x", pady=(8, 0))

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
            w = int(110 * max(0, min(1, hp)))
            if w != c["hpw"]:
                if c["bar"]:
                    c["hp"].delete(c["bar"])
                col = SUCCESS if hp > .6 else "#f5a524" if hp > .3 else DANGER
                c["bar"] = _rounded(c["hp"], 0, 0, max(w, 6), 6, 3, fill=col, outline="") if w else None
                c["hpw"] = w
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
        self.status_var.set("   ·   ".join(parts))
        online = bool(b and b.spawned.is_set())
        self.status_dot.itemconfigure(self._dot, fill=SUCCESS if online else "#f5a524" if (b and b.connected)
                                      else MUTED)
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
                    self.chat.insert("end", time.strftime("%H:%M ", time.localtime(stamp)), "time")
                    self.chat.insert("end", str(sender), "you" if sender == "You" else "bot")
                    self.chat.insert("end", "  " + str(msg) + "\n")
                line_count = int(self.chat.index("end-1c").split(".")[0])
                if line_count > 450:
                    self.chat.delete("1.0", f"{line_count - 450}.0")
                self.chat.see("end")
                self.chat.configure(state="disabled")
                self._chat_seen = len(messages)
        self.model_info.set("Live brain: " + str(DATA_DIR / "brain" / (rt.current + ".db")) +
                            "  |  Universal model: " +
                            str(DATA_DIR / "brain" / (rt.current + "-universal-model.json.gz")) +
                            "  |  Full world backup: " +
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
                           width=18)
        box.grid(row=0, column=1, pady=3, sticky="ew")
        frm.columnconfigure(1, weight=1)
        self.hint = tk.StringVar()
        ttk.Label(frm, textvariable=self.hint, foreground=MUTED, wraplength=520 if app.compact else 700).grid(
            row=1, column=0, columnspan=2, sticky="w")
        box.bind("<<ComboboxSelected>>", lambda e: self._provider_changed(reset=True))
        for i, (key, label) in enumerate(self.FIELDS, start=2):
            ttk.Label(frm, text=label, wraplength=220 if app.compact else 300).grid(
                row=i, column=0, sticky="w", pady=3, padx=(0, 10))
            v = tk.StringVar(value=str(cfg.get(key, "")))
            ttk.Entry(frm, textvariable=v, width=18,
                      show="*" if "api_key" in key else "").grid(row=i, column=1, pady=3, sticky="ew")
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
