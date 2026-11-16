"""Version-specific game knowledge, loaded from data generated out of the official server jar.

Everything here (packet ids, block states, items, recipes, tags, loot, tool rules, food) comes from
the server itself, so it is always exactly right for the Minecraft version in use.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .hardness import hardness_of

log = logging.getLogger("pyflayer.data")

# collision shape kinds (indices into SHAPES)
SHAPES: list[tuple[tuple[float, ...], ...]] = [
    (),                                                   # 0 empty
    ((0, 0, 0, 1, 1, 1),),                                # 1 full
    ((0, 0, 0, 1, 0.5, 1),),                              # 2 bottom slab
    ((0, 0.5, 0, 1, 1, 1),),                              # 3 top slab
    ((0, 0, 0, 1, 1.5, 1),),                              # 4 fence / wall (tall)
    ((0, 0, 0, 1, 0.0625, 1),),                           # 5 carpet-like
    ((0, 0, 0, 1, 0.9375, 1),),                           # 6 farmland / path / cactus / honey
    ((0, 0, 0, 1, 0.875, 1),),                            # 7 chest / soul sand / mud
    ((0, 0, 0, 1, 0.5625, 1),),                           # 8 bed / stonecutter
    ((0, 0, 0, 1, 0.75, 1),),                             # 9 enchanting table
    ((0, 0, 0, 1, 0.1875, 1),),                           # 10 trapdoor bottom (closed)
    ((0, 0.8125, 0, 1, 1, 1),),                           # 11 trapdoor top (closed)
    ((0, 0, 0, 1, 0.375, 1),),                            # 12 daylight detector / low stuff
    ((0, 0, 0, 1, 0.4375, 1),),                           # 13 campfire
    ((0, 0, 0, 1, 0.8125, 1),),                           # 14 end portal frame
    ((0, 0, 0, 1, 0.25, 1),),                             # 15 snow layer 3 / misc
    ((0, 0, 0, 1, 0.125, 1),),                            # 16 snow layer 2
    ((0, 0.5, 0, 1, 1, 1), (0, 0, 0, 1, 0.5, 1)),         # 17 (unused placeholder for stairs below)
]
# stairs: bottom half + upper quarter on the facing side; generated dynamically
_STAIR_UPPER = {"north": (0, 0.5, 0, 1, 1, 0.5), "south": (0, 0.5, 0.5, 1, 1, 1),
                "west": (0, 0.5, 0, 0.5, 1, 1), "east": (0.5, 0.5, 0, 1, 1, 1)}
_STAIR_LOWER = {"north": (0, 0, 0, 1, 0.5, 0.5), "south": (0, 0, 0.5, 1, 0.5, 1),
                "west": (0, 0, 0, 0.5, 0.5, 1), "east": (0.5, 0, 0, 1, 0.5, 1)}

_NO_COLLISION_PAT = re.compile(
    r"(^air$|cave_air|void_air|^water$|^lava$|bubble_column|_sapling|torch|_sign$|_banner$|^rail$|_rail$|"
    r"_button$|pressure_plate|redstone_wire|tripwire|^lever$|^vine$|_vines?$|cave_vines|^ladder$|cobweb|"
    r"sugar_cane|^kelp|seagrass|_coral$|_coral_fan$|_coral_wall_fan$|^fire$|soul_fire|^wheat$|^carrots$|"
    r"^potatoes$|^beetroots$|_stem$|nether_wart$|sweet_berry_bush|bamboo_sapling|tall_grass|short_grass|"
    r"^fern$|large_fern|dead_bush|glow_lichen|sculk_vein|^light$|structure_void|nether_portal|end_portal$|"
    r"end_gateway|pink_petals|leaf_litter|wildflowers|spore_blossom|hanging_roots|small_dripleaf|"
    r"big_dripleaf_stem|^poppy$|dandelion|orchid|allium|azure_bluet|_tulip$|oxeye_daisy|cornflower|"
    r"lily_of_the_valley|wither_rose|torchflower$|pitcher_plant|pitcher_crop|^sunflower$|^lilac$|rose_bush|"
    r"peony|crimson_roots|warped_roots|nether_sprouts|^brown_mushroom$|^red_mushroom$|crimson_fungus|"
    r"warped_fungus|powder_snow$|frogspawn|tripwire_hook|_wall_torch|^bush$|firefly_bush|short_dry_grass|"
    r"tall_dry_grass|cactus_flower|eyeblossom|resin_clump|^snow$|moving_piston|^scaffolding$)")
_DANGEROUS_PAT = re.compile(r"(^lava$|^fire$|soul_fire|magma_block|^cactus$|sweet_berry_bush|"
                            r"powder_snow$|wither_rose|campfire|pointed_dripstone|^cobweb$)")
_CLIMBABLE_PAT = re.compile(r"(^ladder$|^vine$|scaffolding|twisting_vines|weeping_vines|cave_vines)")


def strip_ns(name: str) -> str:
    return name.split(":", 1)[1] if name.startswith("minecraft:") else name


def ns(name: str) -> str:
    return name if ":" in name else "minecraft:" + name


@dataclass
class Recipe:
    kind: str                      # shaped | shapeless | smelting | blasting | smoking | campfire | stonecutting
    result: str
    count: int
    # shaped: grid (rows of ingredient-option-lists or None); shapeless: list of option-lists
    grid: list = field(default_factory=list)
    width: int = 0
    height: int = 0
    ingredients: list = field(default_factory=list)
    cook_time: int = 200
    group: str = ""
    rid: str = ""

    @property
    def needs_table(self) -> bool:
        if self.kind == "shaped":
            return self.width > 2 or self.height > 2
        if self.kind == "shapeless":
            return len(self.ingredients) > 4
        return False

    def ingredient_counts(self) -> list[tuple[tuple[str, ...], int]]:
        """Group identical option-sets with their total count."""
        groups: dict[tuple[str, ...], int] = {}
        items = [c for row in self.grid for c in row if c] if self.kind == "shaped" else self.ingredients
        for opts in items:
            key = tuple(opts)
            groups[key] = groups.get(key, 0) + 1
        return list(groups.items())


class GameData:
    def __init__(self, gd_dir: Path):
        self.dir = Path(gd_dir)
        self.version_info = json.loads((self.dir / "version.json").read_text())
        self.version = self.version_info["id"]
        self.protocol = int(self.version_info["protocol_version"])
        self.packets = json.loads((self.dir / "packets.json").read_text())
        self._load_registries()
        self._load_blocks()
        self._load_tags()
        self._load_item_components()
        self._load_recipes()
        self._load_loot()
        self._build_block_flags()
        log.info("Game data %s: %d blocks, %d states, %d items, %d recipes", self.version,
                 len(self.block_names), self.num_states, len(self.items), len(self.recipes_by_result))

    # ------------------------------------------------------------------ registries
    def _load_registries(self):
        reg = json.loads((self.dir / "registries.json").read_text())
        self.registries = {}
        for rname, r in reg.items():
            by_id = {}
            for ename, e in r["entries"].items():
                by_id[e["protocol_id"]] = strip_ns(ename)
            self.registries[strip_ns(rname)] = by_id
        self.items = self.registries["item"]                       # id -> name
        self.item_ids = {v: k for k, v in self.items.items()}      # name -> id
        self.entity_types = self.registries["entity_type"]
        self.entity_type_ids = {v: k for k, v in self.entity_types.items()}
        self.component_types = self.registries["data_component_type"]
        self.menus = self.registries["menu"]
        self.mob_effects = self.registries.get("mob_effect", {})

    # ------------------------------------------------------------------ blocks
    def _load_blocks(self):
        blocks = json.loads((self.dir / "blocks.json").read_text())
        n = 1 + max(s["id"] for b in blocks.values() for s in b["states"])
        self.num_states = n
        self.state_block = np.zeros(n, dtype=np.uint16)      # state id -> block index
        self.state_props: list[dict] = [{}] * n
        self.block_names: list[str] = []
        self.block_index: dict[str, int] = {}
        self.block_states: dict[str, list[int]] = {}
        self.block_default: dict[str, int] = {}
        for i, (bname, b) in enumerate(blocks.items()):
            name = strip_ns(bname)
            self.block_names.append(name)
            self.block_index[name] = i
            ids = []
            for s in b["states"]:
                sid = s["id"]
                ids.append(sid)
                self.state_block[sid] = i
                self.state_props[sid] = s.get("properties", {})
                if s.get("default"):
                    self.block_default[name] = sid
            self.block_states[name] = ids
            self.block_default.setdefault(name, ids[0])
        self.global_bits_blocks = max(1, math.ceil(math.log2(n)))

    def block_name(self, state: int) -> str:
        if 0 <= state < self.num_states:
            return self.block_names[self.state_block[state]]
        return "unknown"

    def states_of(self, *names: str) -> np.ndarray:
        out = []
        for nm in names:
            nm = strip_ns(nm)
            if nm.startswith("#"):
                for b in self.block_tag(nm[1:]):
                    out += self.block_states.get(b, [])
            else:
                out += self.block_states.get(nm, [])
        return np.array(sorted(set(out)), dtype=np.uint16)

    # ------------------------------------------------------------------ tags
    def _load_tags(self):
        self._raw_tags: dict[str, dict[str, list[str]]] = {}
        base = self.dir / "data" / "tags"
        for kind in ("block", "item", "entity_type"):
            d = {}
            root = base / kind
            if root.exists():
                for p in root.rglob("*.json"):
                    tag = p.relative_to(root).with_suffix("").as_posix()
                    try:
                        d[tag] = json.loads(p.read_text()).get("values", [])
                    except ValueError:
                        continue
            self._raw_tags[kind] = d
        self._tag_cache: dict[tuple[str, str], frozenset[str]] = {}

    def _resolve_tag(self, kind: str, tag: str, depth: int = 0) -> frozenset[str]:
        key = (kind, strip_ns(tag))
        if key in self._tag_cache:
            return self._tag_cache[key]
        out: set[str] = set()
        for v in self._raw_tags.get(kind, {}).get(key[1], []):
            if isinstance(v, dict):
                v = v.get("id", "")
            if v.startswith("#"):
                if depth < 16:
                    out |= self._resolve_tag(kind, v[1:], depth + 1)
            else:
                out.add(strip_ns(v))
        fs = frozenset(out)
        self._tag_cache[key] = fs
        return fs

    def block_tag(self, tag: str) -> frozenset[str]:
        return self._resolve_tag("block", tag)

    def item_tag(self, tag: str) -> frozenset[str]:
        return self._resolve_tag("item", tag)

    def block_tags_of(self, block: str) -> list[str]:
        return sorted(t for t in self._raw_tags["block"] if block in self.block_tag(t))

    # ------------------------------------------------------------------ items
    def _load_item_components(self):
        p = self.dir / "item_components.json"
        raw = json.loads(p.read_text()) if p.exists() else {}
        self.item_components = {strip_ns(k): v for k, v in raw.items()}

    def max_stack(self, item: str) -> int:
        return int(self.item_components.get(strip_ns(item), {}).get("minecraft:max_stack_size", 64))

    def food(self, item: str) -> dict | None:
        return self.item_components.get(strip_ns(item), {}).get("minecraft:food")

    def attack_damage(self, item: str) -> float:
        for m in self.item_components.get(strip_ns(item), {}).get("minecraft:attribute_modifiers", []) or []:
            if isinstance(m, dict) and strip_ns(m.get("type", "")) == "attack_damage":
                return 1.0 + float(m.get("amount", 0))
        return 1.0

    def max_damage(self, item: str) -> int:
        return int(self.item_components.get(strip_ns(item), {}).get("minecraft:max_damage", 0))

    def tool_rules(self, item: str) -> dict | None:
        return self.item_components.get(strip_ns(item), {}).get("minecraft:tool")

    def _blocks_match(self, spec, block: str) -> bool:
        if isinstance(spec, list):
            return any(self._blocks_match(s, block) for s in spec)
        spec = str(spec)
        if spec.startswith("#"):
            return block in self.block_tag(spec[1:])
        return strip_ns(spec) == block

    def tool_speed(self, item: str | None, block: str) -> tuple[float, bool]:
        """(mining speed multiplier, correct_for_drops) of an item against a block."""
        rules = self.tool_rules(item) if item else None
        if not rules:
            return 1.0, not self.requires_tool(block)
        speed = float(rules.get("default_mining_speed", 1.0))
        correct = None
        for rule in rules.get("rules", []):
            if self._blocks_match(rule.get("blocks"), block):
                if "speed" in rule:
                    speed = float(rule["speed"])
                if "correct_for_drops" in rule:
                    correct = bool(rule["correct_for_drops"])
                break
        if correct is None:
            correct = not self.requires_tool(block)
        return speed, correct

    def requires_tool(self, block: str) -> bool:
        if block in self._requires_tool_cache:
            return self._requires_tool_cache[block]
        r = False
        if block in self.block_tag("mineable/pickaxe"):
            r = not any(k in block for k in ("ice", "_glass")) and block not in (
                "ladder", "scaffolding", "bell", "piston", "sticky_piston", "piston_head")
        elif block in ("snow", "snow_block", "cobweb"):
            r = True
        self._requires_tool_cache[block] = r
        return r

    def hardness(self, block: str) -> float:
        return hardness_of(block)

    def dig_ticks(self, block: str, item: str | None, in_water=False, on_ground=True,
                  learned: dict | None = None) -> float:
        """Vanilla break-time formula. Returns inf for unbreakable blocks."""
        h = learned.get(block) if learned and block in learned else self.hardness(block)
        if h < 0:
            return math.inf
        if h == 0:
            return 0
        speed, correct = self.tool_speed(item, block)
        if in_water:
            speed /= 5
        if not on_ground:
            speed /= 5
        dmg = speed / h / (30 if correct else 100)
        if dmg >= 1:
            return 0
        return math.ceil(1 / dmg)

    def best_tool_for(self, block: str, item_names: list[str]) -> tuple[str | None, float, bool]:
        best = (None, 1.0, not self.requires_tool(block))
        for it in item_names:
            sp, corr = self.tool_speed(it, block)
            if (corr and not best[2]) or (corr == best[2] and sp > best[1]):
                best = (it, sp, corr)
        return best

    # ------------------------------------------------------------------ recipes
    def _ingredient(self, spec) -> list[str]:
        if isinstance(spec, list):
            out = []
            for s in spec:
                out += self._ingredient(s)
            return sorted(set(out))
        if isinstance(spec, dict):
            spec = spec.get("item") or ("#" + spec["tag"] if "tag" in spec else "")
        spec = str(spec)
        if spec.startswith("#"):
            return sorted(self.item_tag(spec[1:]))
        return [strip_ns(spec)]

    def _load_recipes(self):
        self.recipes_by_result: dict[str, list[Recipe]] = {}
        self.recipes_by_input: dict[str, list[Recipe]] = {}
        root = self.dir / "data" / "recipe"
        for p in sorted(root.glob("*.json")) if root.exists() else []:
            try:
                d = json.loads(p.read_text())
            except ValueError:
                continue
            t = strip_ns(d.get("type", ""))
            res = d.get("result")
            if not isinstance(res, dict):
                continue
            result = strip_ns(res.get("id", res.get("item", "")))
            count = int(res.get("count", 1))
            r = None
            try:
                if t == "crafting_shaped":
                    key = {k: self._ingredient(v) for k, v in d["key"].items()}
                    pattern = d["pattern"]
                    grid = [[(key.get(ch) if ch != " " else None) for ch in row] for row in pattern]
                    r = Recipe("shaped", result, count, grid=grid, width=max(len(x) for x in pattern),
                               height=len(pattern), group=d.get("group", ""), rid=p.stem)
                elif t == "crafting_shapeless":
                    ings = [self._ingredient(x) for x in d["ingredients"]]
                    r = Recipe("shapeless", result, count, ingredients=ings, group=d.get("group", ""), rid=p.stem)
                elif t in ("smelting", "blasting", "smoking", "campfire_cooking"):
                    r = Recipe(t if t != "campfire_cooking" else "campfire", result, count,
                               ingredients=[self._ingredient(d["ingredient"])],
                               cook_time=int(d.get("cookingtime", 200)), rid=p.stem)
                elif t == "stonecutting":
                    r = Recipe("stonecutting", result, count, ingredients=[self._ingredient(d["ingredient"])], rid=p.stem)
            except (KeyError, TypeError, ValueError):
                continue
            if r is None or not all(r.ingredients or [True]) or (r.kind == "shaped" and not r.grid):
                continue
            if any(len(o) == 0 for o in (r.ingredients if r.kind != "shaped" else
                                         [c for row in r.grid for c in row if c is not None])):
                continue
            self.recipes_by_result.setdefault(result, []).append(r)
            for opts, _ in r.ingredient_counts():
                for o in opts:
                    self.recipes_by_input.setdefault(o, []).append(r)

    def recipes_for(self, item: str, kinds=("shaped", "shapeless")) -> list[Recipe]:
        return [r for r in self.recipes_by_result.get(strip_ns(item), []) if r.kind in kinds]

    def smelt_recipes_for(self, item: str) -> list[Recipe]:
        return [r for r in self.recipes_by_result.get(strip_ns(item), []) if r.kind == "smelting"]

    # ------------------------------------------------------------------ loot
    @staticmethod
    def _cond_special(cond) -> set[str]:
        """Which special harvest tools a loot condition requires (positive sense)."""
        if cond is None:
            return set()
        if isinstance(cond, str):
            c = strip_ns(cond)
            return {"silk_touch"} if "silk_touch" in c else ({"shears"} if "shear" in c else set())
        if isinstance(cond, list):
            out = set()
            for c in cond:
                out |= GameData._cond_special(c)
            return out
        t = strip_ns(cond.get("type", ""))
        if t == "inverted":
            return set()
        if t in ("any_of", "all_of", "alternative"):
            return GameData._cond_special(cond.get("terms", []))
        if t == "match_tool":
            s = json.dumps(cond)
            if "silk_touch" in s:
                return {"silk_touch"}
            if "shears" in s:
                return {"shears"}
        return set()

    def _walk_entries(self, entries, special: set, out_normal: set, out_special: dict, depth=0):
        for e in entries or []:
            if not isinstance(e, dict) or depth > 12:
                continue
            sp = special | self._cond_special(e.get("condition")) | self._cond_special(e.get("conditions"))
            t = strip_ns(e.get("type", ""))
            if t == "item":
                name = strip_ns(e.get("name", ""))
                if sp:
                    out_special.setdefault(name, set()).update(sp)
                else:
                    out_normal.add(name)
            elif t == "tag":
                for name in self.item_tag(e.get("name", "")):
                    (out_special.setdefault(name, set()).update(sp) if sp else out_normal.add(name))
            elif t in ("alternatives", "group", "sequence"):
                self._walk_entries(e.get("children"), sp, out_normal, out_special, depth + 1)

    def _parse_loot(self, d: dict) -> tuple[set, dict]:
        normal: set[str] = set()
        special: dict[str, set] = {}
        for pool in d.get("pools", []):
            sp = self._cond_special(pool.get("conditions")) | self._cond_special(pool.get("condition"))
            self._walk_entries(pool.get("entries"), sp, normal, special)
        return normal, special

    def _load_loot(self):
        self.block_drops: dict[str, set[str]] = {}
        self.block_special_drops: dict[str, dict[str, set]] = {}
        self.item_from_blocks: dict[str, list[str]] = {}
        self.entity_drops: dict[str, set[str]] = {}
        self.item_from_entities: dict[str, list[str]] = {}
        root = self.dir / "data" / "loot_table"
        for p in (root / "blocks").glob("*.json") if (root / "blocks").exists() else []:
            try:
                normal, special = self._parse_loot(json.loads(p.read_text()))
            except ValueError:
                continue
            self.block_drops[p.stem] = normal
            self.block_special_drops[p.stem] = special
            for it in normal:
                self.item_from_blocks.setdefault(it, []).append(p.stem)
        for p in (root / "entities").rglob("*.json") if (root / "entities").exists() else []:
            try:
                normal, _ = self._parse_loot(json.loads(p.read_text()))
            except ValueError:
                continue
            ent = p.relative_to(root / "entities").with_suffix("").as_posix().split("/")[0]
            self.entity_drops.setdefault(ent, set()).update(normal)
            for it in normal:
                lst = self.item_from_entities.setdefault(it, [])
                if ent not in lst:
                    lst.append(ent)
        self._requires_tool_cache: dict[str, bool] = {}

    # ------------------------------------------------------------------ per-state flags (numpy, fast)
    def _build_block_flags(self):
        n = self.num_states
        self.shape_kind = np.ones(n, dtype=np.uint8)          # index into SHAPES (255 = custom)
        self.custom_shapes: dict[int, tuple] = {}
        self.is_air = np.zeros(n, dtype=bool)
        self.is_liquid = np.zeros(n, dtype=bool)
        self.is_water = np.zeros(n, dtype=bool)
        self.is_lava = np.zeros(n, dtype=bool)
        self.is_dangerous = np.zeros(n, dtype=bool)
        self.is_climbable = np.zeros(n, dtype=bool)
        self.is_door = np.zeros(n, dtype=bool)
        self.is_gravity = np.zeros(n, dtype=bool)
        self.is_replaceable = np.zeros(n, dtype=bool)
        self.is_leaves = np.zeros(n, dtype=bool)
        replaceable = self.block_tag("replaceable")
        leaves = self.block_tag("leaves")
        gravity = {"sand", "red_sand", "gravel", "suspicious_sand", "suspicious_gravel", "anvil",
                   "chipped_anvil", "damaged_anvil", "pointed_dripstone"} | {
            b for b in self.block_names if b.endswith("concrete_powder")}
        for name, ids in self.block_states.items():
            no_col = bool(_NO_COLLISION_PAT.search(name)) or name in replaceable
            danger = bool(_DANGEROUS_PAT.search(name))
            climb = bool(_CLIMBABLE_PAT.search(name))
            for sid in ids:
                props = self.state_props[sid]
                kind = 1
                if no_col:
                    kind = 0
                elif name.endswith("_slab"):
                    kind = {"bottom": 2, "top": 3}.get(props.get("type"), 1)
                elif name.endswith("_stairs"):
                    facing = props.get("facing", "north")
                    if props.get("half") == "top":
                        boxes = ((0, 0.5, 0, 1, 1, 1), _STAIR_LOWER[facing])
                    else:
                        boxes = ((0, 0, 0, 1, 0.5, 1), _STAIR_UPPER[facing])
                    kind = 255
                    self.custom_shapes[sid] = boxes
                elif name.endswith(("_fence", "_wall")) or name in ("cobblestone_wall",):
                    kind = 4
                elif name.endswith("_fence_gate"):
                    kind = 0 if props.get("open") == "true" else 4
                elif name.endswith("_door"):
                    kind = 0 if props.get("open") == "true" else 1
                elif name.endswith("_trapdoor"):
                    kind = 0 if props.get("open") == "true" else (11 if props.get("half") == "top" else 10)
                elif name.endswith("_carpet") or name in ("moss_carpet", "lily_pad", "pale_moss_carpet"):
                    kind = 5
                elif name in ("farmland", "dirt_path", "cactus", "honey_block"):
                    kind = 6
                elif name in ("chest", "trapped_chest", "ender_chest", "soul_sand", "mud", "lectern") or \
                        name.endswith("copper_chest") or name.endswith("_shulker_box_never"):
                    kind = 7
                elif name.endswith("_bed") or name == "stonecutter":
                    kind = 8
                elif name == "enchanting_table":
                    kind = 9
                elif name in ("daylight_detector", "flower_pot") or name.startswith("potted_"):
                    kind = 12
                elif name.endswith("campfire"):
                    kind = 13
                elif name == "end_portal_frame":
                    kind = 14
                elif name == "snow":
                    layers = int(props.get("layers", 1))
                    kind = 0 if layers <= 1 else 255
                    if layers > 1:
                        self.custom_shapes[sid] = ((0, 0, 0, 1, (layers - 1) / 8.0, 1),)
                self.shape_kind[sid] = kind
                self.is_dangerous[sid] = danger
                self.is_climbable[sid] = climb
                self.is_door[sid] = name.endswith("_door") or name.endswith("_fence_gate")
                self.is_gravity[sid] = name in gravity
                self.is_replaceable[sid] = name in replaceable
                self.is_leaves[sid] = name in leaves
                if name in ("air", "cave_air", "void_air"):
                    self.is_air[sid] = True
                if name in ("water", "bubble_column") or props.get("waterlogged") == "true" or \
                        name in ("kelp", "kelp_plant", "seagrass", "tall_seagrass"):
                    self.is_water[sid] = True
                if name == "lava":
                    self.is_lava[sid] = True
        self.is_liquid = self.is_water | self.is_lava
        self.has_collision = self.shape_kind != 0
        # a block you can stand on top of (collision reaching the top half of the cell)
        tops = np.zeros(n, dtype=np.float32)
        for k, boxes in enumerate(SHAPES):
            if boxes:
                tops[self.shape_kind == k] = max(b[4] for b in boxes)
        for sid, boxes in self.custom_shapes.items():
            tops[sid] = max(b[4] for b in boxes)
        self.top_height = tops
        # "passable" for pathing: no collision or very low collision; never dangerous
        self.passable = (~self.has_collision | (tops <= 0.0626)) & ~self.is_dangerous & ~self.is_lava
        self.standable = self.has_collision & (tops >= 0.4) & ~self.is_dangerous
        self.hardness_arr = np.array([hardness_of(self.block_names[self.state_block[s]]) for s in range(n)],
                                     dtype=np.float32)

    def shape_boxes(self, state: int) -> tuple:
        k = self.shape_kind[state]
        if k == 255:
            return self.custom_shapes.get(state, ((0, 0, 0, 1, 1, 1),))
        return SHAPES[k]

    # ------------------------------------------------------------------ describing things (for the AI)
    def describe_item(self, item: str) -> str:
        item = strip_ns(item)
        lines = [f"{item}:"]
        comps = self.item_components.get(item, {})
        if comps:
            lines.append(f"  stack size {self.max_stack(item)}")
            if self.food(item):
                f = self.food(item)
                lines.append(f"  food: nutrition {f.get('nutrition')} saturation {f.get('saturation')}")
            if self.tool_rules(item):
                lines.append(f"  tool; attack damage {self.attack_damage(item):.1f}; durability {self.max_damage(item)}")
        for r in self.recipes_by_result.get(item, [])[:4]:
            ings = ", ".join(f"{n}x {'/'.join(o[:3])}{'...' if len(o) > 3 else ''}" for o, n in r.ingredient_counts())
            where = {"shaped": "crafting" + (" table" if r.needs_table else " (2x2 ok)"),
                     "shapeless": "crafting" + (" table" if r.needs_table else " (2x2 ok)"),
                     "smelting": "furnace"}.get(r.kind, r.kind)
            lines.append(f"  recipe [{where}] -> {r.count}: {ings}")
        src = self.item_from_blocks.get(item, [])
        if src:
            lines.append(f"  dropped by blocks: {', '.join(src[:8])}")
        ent = self.item_from_entities.get(item, [])
        if ent:
            lines.append(f"  dropped by mobs: {', '.join(ent[:8])}")
        used = sorted({r.result for r in self.recipes_by_input.get(item, [])})
        if used:
            lines.append(f"  used to make: {', '.join(used[:12])}{' ...' if len(used) > 12 else ''}")
        return "\n".join(lines)

    def describe_block(self, block: str) -> str:
        block = strip_ns(block)
        if block not in self.block_index:
            return f"unknown block {block}"
        tools = [t.split("/")[1] for t in ("mineable/pickaxe", "mineable/axe", "mineable/shovel", "mineable/hoe")
                 if block in self.block_tag(t)]
        tier = next((t for t in ("needs_diamond_tool", "needs_iron_tool", "needs_stone_tool")
                     if block in self.block_tag(t)), None)
        drops = sorted(self.block_drops.get(block, []))
        special = {k: sorted(v) for k, v in self.block_special_drops.get(block, {}).items()}
        s = f"{block}: hardness {self.hardness(block)}"
        s += f"; best tool {'/'.join(tools) or 'hand'}"
        if tier:
            s += f" (requires {tier.replace('needs_', '').replace('_tool', '')} tier or better)"
        if self.requires_tool(block):
            s += "; drops nothing without the right tool"
        s += f"; drops: {', '.join(drops) or 'nothing'}"
        if special:
            s += f"; with special tool: {special}"
        return s
