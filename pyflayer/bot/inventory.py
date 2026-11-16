"""Player inventory + open container windows.

Window-0 (player inventory) slot layout:
  0 crafting output, 1-4 crafting grid, 5-8 armor (head, chest, legs, feet),
  9-35 main inventory, 36-44 hotbar, 45 offhand.
Other windows: their own slots first, then 27 main + 9 hotbar.

Clicks never predict results: we tell the server nothing changed and it sends back the authoritative
slot diffs (that is how vanilla resynchronises), so the local model can never drift.
"""
from __future__ import annotations

import threading
import time

from ..protocol.items import Item

CONTAINER_SIZES = {
    "generic_9x1": 9, "generic_9x2": 18, "generic_9x3": 27, "generic_9x4": 36, "generic_9x5": 45,
    "generic_9x6": 54, "generic_3x3": 9, "crafter_3x3": 10, "crafting": 10, "furnace": 3, "blast_furnace": 3,
    "smoker": 3, "anvil": 3, "beacon": 1, "brewing_stand": 5, "enchantment": 2, "grindstone": 3, "hopper": 5,
    "lectern": 1, "loom": 4, "merchant": 3, "shulker_box": 27, "smithing": 4, "cartography_table": 3,
    "stonecutter": 2,
}

# click modes
PICKUP, QUICK_MOVE, SWAP, CLONE, THROW, QUICK_CRAFT, PICKUP_ALL = range(7)


class Window:
    def __init__(self, wid: int, wtype: str, title: str, size: int):
        self.id = wid
        self.type = wtype
        self.title = title
        self.container_size = size
        self.slots: list[Item | None] = [None] * (size + 36)
        self.state_id = 0
        self.properties: dict[int, int] = {}
        self.opened_at = time.time()

    @property
    def player_start(self) -> int:
        return self.container_size

    def container_items(self) -> list[tuple[int, Item]]:
        return [(i, it) for i, it in enumerate(self.slots[:self.container_size]) if it]


class Inventory:
    def __init__(self, bot):
        self.bot = bot
        self.slots: list[Item | None] = [None] * 46
        self.state_id = 0
        self.cursor: Item | None = None
        self.held_slot = 0                 # 0..8
        self.window: Window | None = None  # open container (None = only player inventory)
        self.changed = threading.Condition()
        self.version = 0

    # ------------------------------------------------------------------ packet handlers
    def _notify(self):
        with self.changed:
            self.version += 1
            self.changed.notify_all()

    def on_set_content(self, wid, state_id, items, carried):
        if wid == 0:
            for i in range(min(len(items), 46)):
                self.slots[i] = items[i]
            self.state_id = state_id
        elif self.window and self.window.id == wid:
            w = self.window
            for i in range(min(len(items), len(w.slots))):
                w.slots[i] = items[i]
            w.state_id = state_id
            self._sync_player_from_window()
        self.cursor = carried
        self._notify()

    def on_set_slot(self, wid, state_id, slot, item):
        if wid == -1 and slot == -1:
            self.cursor = item
        elif wid == 0 or (wid == -2):
            if 0 <= slot < 46:
                self.slots[slot] = item
            self.state_id = state_id if wid == 0 else self.state_id
            if self.window and slot >= 9:
                ws = self.window.player_start + (slot - 9) if slot < 45 else None
                if ws is not None and ws < len(self.window.slots):
                    self.window.slots[ws] = item
        elif self.window and self.window.id == wid:
            if 0 <= slot < len(self.window.slots):
                self.window.slots[slot] = item
            self.window.state_id = state_id
            self._sync_player_from_window()
        self._notify()

    def on_set_player_slot(self, inv_index, item):
        """set_player_inventory uses Inventory indices (0-8 hotbar, 9-35 main, 36-39 armor, 40 offhand)."""
        if 0 <= inv_index <= 8:
            ws = 36 + inv_index
        elif 9 <= inv_index <= 35:
            ws = inv_index
        elif 36 <= inv_index <= 39:
            ws = 8 - (inv_index - 36)
        elif inv_index == 40:
            ws = 45
        else:
            return
        self.slots[ws] = item
        if self.window and 9 <= ws <= 44:
            self.window.slots[self.window.player_start + ws - 9] = item
        self._notify()

    def _sync_player_from_window(self):
        w = self.window
        if not w:
            return
        for i in range(36):
            self.slots[9 + i] = w.slots[w.player_start + i]

    def on_open(self, wid, wtype, title):
        size = CONTAINER_SIZES.get(wtype, 27)
        self.window = Window(wid, wtype, title, size)
        for i in range(36):
            self.window.slots[size + i] = self.slots[9 + i]
        self._notify()

    def on_close(self, wid=None):
        if self.window and (wid is None or self.window.id == wid):
            self.window = None
        self._notify()

    def on_property(self, wid, prop, value):
        if self.window and self.window.id == wid:
            self.window.properties[prop] = value
            self._notify()

    # ------------------------------------------------------------------ queries
    @property
    def held_item(self) -> Item | None:
        return self.slots[36 + self.held_slot]

    def items(self) -> list[Item]:
        """All items in main inventory + hotbar + offhand (+ armor)."""
        return [it for it in self.slots[5:46] if it]

    def count(self, name: str | set | list) -> int:
        names = {name} if isinstance(name, str) else set(name)
        return sum(it.count for it in self.slots[9:46] if it and it.name in names)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self.slots[9:46]:
            if it:
                out[it.name] = out.get(it.name, 0) + it.count
        return out

    def find(self, pred) -> int | None:
        """Window-0 slot index of the first matching item (hotbar first)."""
        for i in list(range(36, 45)) + list(range(9, 36)) + [45]:
            it = self.slots[i]
            if it and pred(it):
                return i
        return None

    def empty_slots(self) -> int:
        return sum(1 for it in self.slots[9:45] if it is None)

    def armor(self) -> dict[str, Item | None]:
        return {"head": self.slots[5], "chest": self.slots[6], "legs": self.slots[7], "feet": self.slots[8]}

    def summary(self) -> str:
        c = self.counts()
        if not c:
            return "empty"
        held = self.held_item
        s = ", ".join(f"{n} x{k}" for n, k in sorted(c.items(), key=lambda kv: -kv[1]))
        return f"{s} | holding: {held.name if held else 'nothing'}"

    # ------------------------------------------------------------------ window-relative helpers
    def active_slots(self) -> list[Item | None]:
        return self.window.slots if self.window else self.slots

    def active_id(self) -> int:
        return self.window.id if self.window else 0

    def active_state(self) -> int:
        return self.window.state_id if self.window else self.state_id

    def player_slot_to_active(self, ws: int) -> int:
        """Convert a window-0 slot (9..44) to the open window's index."""
        if not self.window:
            return ws
        return self.window.player_start + (ws - 9)

    def wait_change(self, since_version: int, timeout: float = 0.6) -> bool:
        end = time.time() + timeout
        with self.changed:
            while self.version == since_version:
                left = end - time.time()
                if left <= 0:
                    return False
                self.changed.wait(left)
        return True
