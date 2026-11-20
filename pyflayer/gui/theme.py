"""Modern dark theme for Tk/ttk: rounded buttons, cards, inputs and slim pill scrollbars.

Plain Tk has no rounded corners, so we draw small anti-aliased rounded-rectangle images in Python
(no Pillow needed) and use them as 9-slice ttk image elements that stretch to any widget size.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# palette
BG = "#16171b"          # window
SURFACE = "#1f2026"     # panels / cards
SURFACE2 = "#282a31"    # buttons, raised
HOVER = "#32353d"
FIELD = "#121317"       # text boxes
BORDER = "#2f323a"
FG = "#eceef2"
MUTED = "#8b909a"
ACCENT = "#5b8cff"
ACCENT_HOVER = "#7aa2ff"
ACCENT_PRESS = "#4673e6"
DANGER = "#ef5a5f"
SUCCESS = "#3ecf8e"
SELECTED = "#262c3d"    # selected card
CARD_COLORS = ["#5b8cff", "#ef5a5f", "#3ecf8e", "#f5a524", "#a275f0"]

FONT = ("DejaVu Sans", 10)
FONT_BOLD = ("DejaVu Sans", 10, "bold")
FONT_TITLE = ("DejaVu Sans", 13, "bold")
FONT_MONO = ("DejaVu Sans Mono", 9)


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _mix(a, b, t):
    return "#%02x%02x%02x" % tuple(round(x * t + y * (1 - t)) for x, y in zip(a, b))


def rounded_image(master, w, h, r, fill, outside, border=None, bw=1):
    """Anti-aliased rounded rectangle as a PhotoImage (4x4 supersampling)."""
    img = tk.PhotoImage(master=master, width=w, height=h)
    f, o = _hex(fill), _hex(outside)
    bc = _hex(border) if border else None
    ss = 4
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            cov = inner = 0
            for sy in range(ss):
                for sx in range(ss):
                    px, py = x + (sx + .5) / ss, y + (sy + .5) / ss
                    # distance outside the rounded rect
                    dx = max(r - px, 0, px - (w - r))
                    dy = max(r - py, 0, py - (h - r))
                    d = (dx * dx + dy * dy) ** .5
                    if d <= r:
                        cov += 1
                        if bc is None or d <= r - bw and bw <= px <= w - bw and bw <= py <= h - bw:
                            inner += 1
            n = ss * ss
            if bc is None:
                row.append(_mix(f, o, cov / n))
            else:
                edge = (cov - inner) / n
                body = inner / n
                col = tuple(fv * body + bv * edge + ov * (1 - body - edge) for fv, bv, ov in zip(f, bc, o))
                row.append("#%02x%02x%02x" % tuple(round(v) for v in col))
        rows.append("{" + " ".join(row) + "}")
    img.put(" ".join(rows))
    return img


class Theme:
    def __init__(self, root: tk.Misc, compact: bool = False):
        self.root = root
        self.compact = compact
        self.images = []        # keep references alive
        self.style = ttk.Style(root)
        self._build()

    def img(self, *a, **k):
        im = rounded_image(self.root, *a, **k)
        self.images.append(im)
        return im

    def _el(self, name, normal, states=(), border=10, padding=None):
        spec = [normal]
        for st, im in states:
            spec.append((st, im))
        kw = {"border": border, "sticky": "nsew"}
        if padding is not None:
            kw["padding"] = padding
        self.style.element_create(name, "image", *spec, **kw)

    def _build(self):
        s = self.style
        if "pyflayer" not in s.theme_names():
            s.theme_create("pyflayer", parent="clam")
        s.theme_use("pyflayer")
        R = 9 if self.compact else 10
        S = 2 * R + 4
        pad = (8, 4) if self.compact else (14, 8)

        s.configure(".", background=BG, foreground=FG, fieldbackground=FIELD, bordercolor=BORDER,
                    lightcolor=BG, darkcolor=BG, troughcolor=BG, focuscolor=ACCENT, font=FONT,
                    selectbackground=ACCENT, selectforeground="#ffffff", insertcolor=FG)

        # ---- buttons (plain, accent, danger) on window background and on card surfaces
        def button(style, base, hover, press, on=BG):
            n = style.replace(".", "_")
            self._el(f"{n}.rbg", self.img(S, S, R, base, on),
                     [("disabled", self.img(S, S, R, _mix(_hex(base), _hex(on), .5), on)),
                      ("pressed", self.img(S, S, R, press, on)),
                      ("active", self.img(S, S, R, hover, on))], border=R, padding=2)
            s.layout(style, [(f"{n}.rbg", {"sticky": "nsew", "children": [
                ("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})]})])
            s.configure(style, padding=pad, anchor="center", foreground=FG, font=FONT_BOLD if "Accent" in style
                        else FONT, background=base)
            s.map(style, foreground=[("disabled", MUTED)], background=[("active", hover)])

        button("TButton", SURFACE2, HOVER, ACCENT_PRESS)
        button("Accent.TButton", ACCENT, ACCENT_HOVER, ACCENT_PRESS)
        s.configure("Accent.TButton", foreground="#ffffff")
        button("Danger.TButton", "#3a2427", "#4a2a2e", DANGER)
        s.configure("Danger.TButton", foreground="#ffb4b6")
        button("Card.TButton", SURFACE2, HOVER, ACCENT_PRESS, on=SURFACE)

        # ---- cards / panels
        CR = 12 if not self.compact else 10
        CS = 2 * CR + 4
        for style, fill, border in (("Card.TFrame", SURFACE, BORDER), ("Sel.TFrame", SELECTED, ACCENT),
                                    ("Panel.TFrame", SURFACE, BORDER)):
            n = style.replace(".", "_")
            self._el(f"{n}.rbg", self.img(CS, CS, CR, fill, BG, border=border, bw=1 if border != ACCENT else 2),
                     border=CR)
            s.layout(style, [(f"{n}.rbg", {"sticky": "nsew"})])
            s.configure(style, background=fill)
        s.configure("TFrame", background=BG)
        s.configure("Panel.TLabel", background=SURFACE, foreground=FG)
        s.configure("PanelMuted.TLabel", background=SURFACE, foreground=MUTED)
        s.configure("PanelTitle.TLabel", background=SURFACE, foreground=FG, font=FONT_BOLD)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Muted.TLabel", background=BG, foreground=MUTED)
        s.configure("Title.TLabel", background=BG, foreground=FG, font=FONT_TITLE)
        s.configure("Section.TLabel", background=BG, foreground=MUTED, font=("DejaVu Sans", 8, "bold"))
        s.configure("TLabelframe", background=BG, bordercolor=BORDER)
        s.configure("TLabelframe.Label", background=BG, foreground=MUTED)
        s.configure("TSeparator", background=BORDER)

        # ---- inputs: rounded field with an accent ring on focus
        self._el("RField.rbg", self.img(S, S, R, FIELD, BG, border=BORDER),
                 [("focus", self.img(S, S, R, FIELD, BG, border=ACCENT, bw=2)),
                  ("disabled", self.img(S, S, R, SURFACE, BG, border=BORDER))], border=R, padding=3)
        s.layout("TEntry", [("RField.rbg", {"sticky": "nsew", "children": [
            ("Entry.padding", {"sticky": "nsew", "children": [("Entry.textarea", {"sticky": "nsew"})]})]})])
        s.configure("TEntry", padding=(8, 5) if not self.compact else (6, 3), foreground=FG, insertcolor=FG)
        s.layout("TCombobox", [("RField.rbg", {"sticky": "nsew", "children": [
            ("Combobox.downarrow", {"side": "right", "sticky": "ns"}),
            ("Combobox.padding", {"sticky": "nsew", "children": [("Combobox.textarea", {"sticky": "nsew"})]})]})])
        s.configure("TCombobox", padding=(8, 4), foreground=FG, arrowcolor=MUTED, background=FIELD, bordercolor=FIELD,
                    lightcolor=FIELD, darkcolor=FIELD,
                    fieldbackground=FIELD, arrowsize=14)
        s.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", FG)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", FG)],
              background=[("active", FIELD), ("pressed", FIELD)])

        # ---- slim rounded scrollbars (no arrows) that light up on hover/drag
        T = 8 if self.compact else 10
        TR = T // 2
        for orient in ("Vertical", "Horizontal"):
            w, h = (T, 2 * T + 4) if orient == "Vertical" else (2 * T + 4, T)
            thumb = self.img(w, h, TR, "#3a3d46", BG)
            hot = self.img(w, h, TR, "#555a66", BG)
            press = self.img(w, h, TR, ACCENT, BG)
            trough = self.img(w, h, TR, "#1c1d22", BG)
            self._el(f"{orient}.Scrollbar.thumb", thumb, [("pressed", press), ("active", hot)], border=TR)
            self._el(f"{orient}.Scrollbar.trough", trough, border=TR)
            s.layout(f"{orient}.TScrollbar", [(f"{orient}.Scrollbar.trough", {"sticky": "nsew", "children": [
                (f"{orient}.Scrollbar.thumb", {"expand": "1", "sticky": "nsew"})]})])
            s.configure(f"{orient}.TScrollbar", width=T, arrowsize=T, gripcount=0)

        # ---- tabs
        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        s.configure("TNotebook.Tab", background=SURFACE2, foreground=MUTED, padding=(12, 6), borderwidth=0,
                    lightcolor=SURFACE2, bordercolor=BORDER)
        s.map("TNotebook.Tab", background=[("selected", SURFACE), ("active", HOVER)],
              foreground=[("selected", FG)], lightcolor=[("selected", ACCENT)])

        # plain Tk widgets (Text, Listbox, menus, dialogs)
        r = self.root
        for opt, val in (("*Text.Background", FIELD), ("*Text.Foreground", FG), ("*Text.selectBackground", ACCENT),
                         ("*Text.insertBackground", FG), ("*Text.relief", "flat"), ("*Text.highlightThickness", 0),
                         ("*Text.borderWidth", 0),
                         ("*Listbox.Background", SURFACE), ("*Listbox.Foreground", FG),
                         ("*Listbox.selectBackground", ACCENT), ("*Listbox.relief", "flat"),
                         ("*TCombobox*Listbox.background", SURFACE), ("*TCombobox*Listbox.foreground", FG),
                         ("*TCombobox*Listbox.selectBackground", ACCENT), ("*Toplevel.Background", BG),
                         ("*Menu.background", SURFACE), ("*Menu.foreground", FG), ("*Text.Font", FONT), ("*Label.Font", FONT),
                         ("*Dialog.msg.font", FONT), ("*Message.Background", BG), ("*Message.Foreground", FG),
                         ("*Label.Background", BG), ("*Label.Foreground", FG), ("*Button.Background", SURFACE2),
                         ("*Button.Foreground", FG), ("*Entry.Background", FIELD), ("*Entry.Foreground", FG),
                         ("*Frame.Background", BG)):
            r.option_add(opt, val)
        r.configure(bg=BG)


class TextPanel(ttk.Frame):
    """A rounded panel with a title and a scrolling read-only Text inside."""

    def __init__(self, master, title, height=6, mono=False, compact=False):
        super().__init__(master, style="Panel.TFrame", padding=(10, 6, 6, 8) if compact else (12, 8, 8, 10))
        if title:
            self.title = ttk.Label(self, text=title.upper(), style="PanelMuted.TLabel",
                                   font=("DejaVu Sans", 8, "bold"))
            self.title.pack(anchor="w", pady=(0, 4))
        ttk.Style(self).configure("Flat.TFrame", background=SURFACE)
        body = ttk.Frame(self, style="Flat.TFrame")
        body.pack(fill="both", expand=True)
        self.text = tk.Text(body, height=height, wrap="word", state="disabled", bg=SURFACE, fg=FG,
                            font=FONT_MONO if mono else FONT, padx=4, pady=2, relief="flat", bd=0,
                            highlightthickness=0, selectbackground=ACCENT, cursor="arrow")
        self.bar = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self._set)
        self.text.grid(row=0, column=0, sticky="nsew")
        self.bar.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

    def _set(self, lo, hi):
        self.bar.set(lo, hi)
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.bar.grid_remove()          # nothing to scroll: hide the bar
        else:
            self.bar.grid()
