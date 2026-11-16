"""Item stacks (Slot data) with structured data components, plus the Hashed Slot format."""
from __future__ import annotations

from dataclasses import dataclass, field

from .buffer import ProtocolError, Reader, Writer
from .text import to_plain


class UnknownComponent(ProtocolError):
    pass


@dataclass
class Item:
    name: str
    count: int
    id: int
    components: dict = field(default_factory=dict)   # name -> decoded value
    removed: list = field(default_factory=list)
    incomplete: bool = False                          # True if some component could not be decoded

    @property
    def damage(self) -> int:
        return int(self.components.get("damage", 0) or 0)

    @property
    def display_name(self) -> str:
        cn = self.components.get("custom_name")
        return to_plain(cn) if cn else self.name

    def __repr__(self):
        return f"{self.name}x{self.count}"


# --------------------------------------------------------------------------- component decoders
def _skip_text(r: Reader):
    return r.nbt()


def _id_set(r: Reader):
    t = r.varint()
    if t == 0:
        return {"tag": r.string()}
    return {"ids": [r.varint() for _ in range(t - 1)]}


def _sound_event(r: Reader):
    name = r.string()
    rng = r.optional(r.float)
    return {"sound": name, "range": rng}


def _id_or(r: Reader, inline):
    i = r.varint()
    if i == 0:
        return inline(r)
    return i - 1


def _effect_detail(r: Reader):
    d = {"amplifier": r.varint(), "duration": r.varint(), "ambient": r.bool(),
         "particles": r.bool(), "icon": r.bool()}
    if r.bool():
        d["hidden"] = _effect_detail(r)
    return d


def _potion_effect(r: Reader):
    return {"id": r.varint(), **_effect_detail(r)}


def _potion_contents(r: Reader):
    pot = r.optional(r.varint)
    color = r.optional(r.int)
    effects = r.array(lambda: _potion_effect(r))
    name = r.optional(r.string)
    return {"potion": pot, "color": color, "effects": effects, "custom_name": name}


def _enchants(r: Reader):
    return [(r.varint(), r.varint()) for _ in range(r.varint())]


def _attr_mods(r: Reader):
    out = []
    for _ in range(r.varint()):
        attr = r.varint()
        mid = r.string()
        val = r.double()
        op = r.varint()
        slot = r.varint()
        out.append((attr, mid, val, op, slot))
    return out


def _custom_model_data(r: Reader):
    return {"floats": r.array(r.float), "flags": r.array(r.bool), "strings": r.array(r.string),
            "colors": r.array(r.int)}


def _tooltip_display(r: Reader):
    return {"hide": r.bool(), "hidden": r.array(r.varint)}


def _food(r: Reader):
    return {"nutrition": r.varint(), "saturation": r.float(), "can_always_eat": r.bool()}


def _lodestone(r: Reader):
    has = r.bool()
    d = {}
    if has:
        d["dimension"] = r.string()
        d["pos"] = r.position()
    d["tracked"] = r.bool()
    return d


def _firework_explosion(r: Reader):
    return {"shape": r.varint(), "colors": r.array(r.int), "fade": r.array(r.int),
            "trail": r.bool(), "twinkle": r.bool()}


def _profile(r: Reader):
    # Resolvable profile: bool (complete profile?) then either a full game profile or partial fields
    if r.bool():
        uid = r.uuid()
        name = r.string()
        props = r.array(lambda: (r.string(), r.string(), r.optional(r.string)))
        d = {"uuid": str(uid), "name": name, "properties": len(props)}
    else:
        d = {"name": r.optional(r.string), "uuid": r.optional(r.uuid)}
        r.array(lambda: (r.string(), r.string(), r.optional(r.string)))
    # skin patch: body, cape, elytra (optional identifiers), model (optional enum)
    r.optional(r.string)
    r.optional(r.string)
    r.optional(r.string)
    r.optional(r.varint)
    return d


def _writable_book(r: Reader):
    return [(r.string(), r.optional(r.string)) for _ in range(r.varint())]


def _written_book(r: Reader):
    d = {"title": r.string()}
    r.optional(r.string)
    d["author"] = r.string()
    d["generation"] = r.varint()
    d["pages"] = [(r.nbt(), r.optional(r.nbt)) for _ in range(r.varint())]
    d["resolved"] = r.bool()
    return d


def _bees(r: Reader):
    out = []
    for _ in range(r.varint()):
        out.append((r.varint(), r.nbt(), r.varint(), r.varint()))
    return out


def _block_state(r: Reader):
    return {r.string(): r.string() for _ in range(r.varint())}


def _stew(r: Reader):
    return [(r.varint(), r.varint()) for _ in range(r.varint())]


def _use_cooldown(r: Reader):
    return {"seconds": r.float(), "group": r.optional(r.string)}


def _empty(r: Reader):
    return True


def _nbt(r: Reader):
    return r.nbt()


def _entity_data(r: Reader):
    return {"type": r.varint(), "data": r.nbt()}


def _equippable(r: Reader):
    d = {"slot": r.varint()}
    d["sound"] = _id_or(r, _sound_event)
    d["model"] = r.optional(r.string)
    d["overlay"] = r.optional(r.string)
    d["allowed"] = r.optional(lambda: _id_set(r))
    d["dispensable"] = r.bool()
    d["swappable"] = r.bool()
    d["damage_on_hurt"] = r.bool()
    d["equip_on_interact"] = r.bool()
    d["can_be_sheared"] = r.bool()
    d["shearing_sound"] = _id_or(r, _sound_event)
    return d


def _make_decoders(slot_reader):
    """Component decoders. Unknown/complex components raise UnknownComponent."""
    return {
        "custom_data": _nbt, "max_stack_size": lambda r: r.varint(), "max_damage": lambda r: r.varint(),
        "damage": lambda r: r.varint(), "unbreakable": _empty, "custom_name": _skip_text,
        "item_name": _skip_text, "item_model": lambda r: r.string(),
        "lore": lambda r: r.array(r.nbt), "rarity": lambda r: r.varint(), "enchantments": _enchants,
        "stored_enchantments": _enchants, "attribute_modifiers": _attr_mods,
        "custom_model_data": _custom_model_data, "tooltip_display": _tooltip_display,
        "repair_cost": lambda r: r.varint(), "creative_slot_lock": _empty,
        "enchantment_glint_override": lambda r: r.bool(), "intangible_projectile": _nbt, "food": _food,
        "use_remainder": slot_reader, "use_cooldown": _use_cooldown,
        "damage_resistant": _id_set, "enchantable": lambda r: r.varint(), "repairable": _id_set,
        "glider": _empty, "tooltip_style": lambda r: r.string(), "dyed_color": lambda r: r.int(),
        "map_color": lambda r: r.int(), "map_id": lambda r: r.varint(), "map_decorations": _nbt,
        "map_post_processing": lambda r: r.varint(),
        "charged_projectiles": lambda r: r.array(lambda: slot_reader(r)),
        "bundle_contents": lambda r: r.array(lambda: slot_reader(r)),
        "container": lambda r: r.array(lambda: slot_reader(r)),
        "potion_contents": _potion_contents, "potion_duration_scale": lambda r: r.float(),
        "suspicious_stew_effects": _stew, "writable_book_content": _writable_book,
        "written_book_content": _written_book, "debug_stick_state": _nbt, "entity_data": _entity_data,
        "bucket_entity_data": _nbt, "block_entity_data": _entity_data,
        "ominous_bottle_amplifier": lambda r: r.varint(), "recipes": _nbt, "lodestone_tracker": _lodestone,
        "firework_explosion": _firework_explosion,
        "fireworks": lambda r: (r.varint(), r.array(lambda: _firework_explosion(r))),
        "profile": _profile, "note_block_sound": lambda r: r.string(),
        "base_color": lambda r: r.varint(), "pot_decorations": lambda r: r.array(r.varint),
        "block_state": _block_state, "bees": _bees, "lock": _nbt, "container_loot": _nbt,
        "sulfur_cube_content": slot_reader, "additional_trade_cost": lambda r: r.varint(),
        "minimum_attack_charge": lambda r: r.float(), "damage_type": lambda r: r.varint(),
        "weapon": lambda r: (r.varint(), r.float()), "equippable": _equippable,
        "break_sound": lambda r: _id_or(r, _sound_event), "dye": lambda r: r.varint(),
        "instrument": _instrument,
        "provides_banner_patterns": _id_set, "jukebox_playable": _jukebox,
        **{k: (lambda r: r.varint()) for k in (
            "villager/variant", "wolf/variant", "wolf/sound_variant", "wolf/collar", "fox/variant",
            "salmon/size", "parrot/variant", "tropical_fish/pattern", "tropical_fish/base_color",
            "tropical_fish/pattern_color", "mooshroom/variant", "rabbit/variant", "pig/variant",
            "pig/sound_variant", "cow/variant", "cow/sound_variant", "chicken/variant", "chicken/sound_variant",
            "frog/variant", "horse/variant", "llama/variant", "axolotl/variant", "zombie_nautilus/variant",
            "cat/variant", "cat/sound_variant", "cat/collar", "sheep/color", "shulker/color", "cushion/color")},
    }


def _no_inline(r: Reader):
    raise UnknownComponent("inline value not supported")


def _instrument(r: Reader):
    i = r.varint()  # ID or Instrument
    if i == 0:
        raise UnknownComponent("inline instrument")
    return i - 1


def _jukebox(r: Reader):
    if r.bool():
        i = r.varint()
        if i == 0:
            raise UnknownComponent("inline jukebox song")
        return i - 1
    return r.string()


class ItemCodec:
    def __init__(self, items: dict[int, str], component_types: dict[int, str]):
        self.items = items
        self.item_ids = {v: k for k, v in items.items()}
        self.component_types = component_types
        self.decoders = _make_decoders(self.read_slot)

    def read_slot(self, r: Reader) -> Item | None:
        count = r.varint()
        if count <= 0:
            return None
        iid = r.varint()
        n_add = r.varint()
        n_rem = r.varint()
        item = Item(self.items.get(iid, f"item#{iid}"), count, iid)
        for _ in range(n_add):
            ct = r.varint()
            cname = self.component_types.get(ct, f"#{ct}")
            dec = self.decoders.get(cname)
            if dec is None:
                raise UnknownComponent(f"cannot decode component {cname} on {item.name}")
            item.components[cname] = dec(r)
        for _ in range(n_rem):
            item.removed.append(self.component_types.get(r.varint(), "?"))
        return item

    @staticmethod
    def write_hashed(w: Writer, item: Item | None):
        """Hashed slot. We send component-free hashes; the server is authoritative and resyncs."""
        if item is None:
            w.bool(False)
            return
        w.bool(True)
        w.varint(item.id)
        w.varint(item.count)
        w.varint(0)   # added components (hash list) - unknown hashes are fine, server resyncs
        w.varint(0)   # removed components
