"""Text component (chat) helpers: convert NBT/JSON text components into plain strings."""
from __future__ import annotations

import json
import re

# A small set of translation keys we render nicely; anything else falls back to joining args.
_TRANSLATIONS = {
    "chat.type.text": "<%s> %s",
    "chat.type.announcement": "[%s] %s",
    "chat.type.emote": "* %s %s",
    "chat.type.admin": "[%s: %s]",
    "multiplayer.player.joined": "%s joined the game",
    "multiplayer.player.left": "%s left the game",
    "commands.message.display.incoming": "%s whispers to you: %s",
    "commands.message.display.outgoing": "You whisper to %s: %s",
    "death.attack.generic": "%s died",
}


def to_plain(comp) -> str:
    """Flatten a text component (str / dict / list / JSON string) into plain text."""
    if comp is None:
        return ""
    if isinstance(comp, (bytes, bytearray)):
        comp = comp.decode("utf-8", "replace")
    if isinstance(comp, str):
        s = comp.strip()
        if s[:1] in "{[" and s[-1:] in "}]":
            try:
                return to_plain(json.loads(s))
            except ValueError:
                pass
        return comp
    if isinstance(comp, list):
        return "".join(to_plain(c) for c in comp)
    if not isinstance(comp, dict):
        return str(comp)
    out = ""
    if "text" in comp:
        out += str(comp["text"])
    elif "" in comp:  # NBT lists of mixed types are wrapped as {"": value}
        out += to_plain(comp[""])
    elif "translate" in comp:
        key = comp["translate"]
        args = [to_plain(a) for a in comp.get("with", [])]
        fmt = _TRANSLATIONS.get(key)
        if fmt is None:
            fmt = comp.get("fallback")
        if fmt:
            try:
                n = fmt.count("%s")
                out += fmt % tuple((args + [""] * n)[:n])
            except (TypeError, ValueError):
                out += fmt + " " + " ".join(args)
        else:
            # death messages etc: show the key's last part plus args
            nice = key.split(".")[-1].replace("_", " ")
            out += (" ".join(args) + " " + nice).strip() if args else key
    elif "keybind" in comp:
        out += comp["keybind"]
    elif "score" in comp:
        out += str(comp["score"].get("value", ""))
    elif "selector" in comp:
        out += comp["selector"]
    for extra in comp.get("extra", []) or []:
        out += to_plain(extra)
    return out


_COLOR_CODES = re.compile("§.")


def strip_codes(s: str) -> str:
    return _COLOR_CODES.sub("", s)
