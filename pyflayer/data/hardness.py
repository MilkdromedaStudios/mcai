"""Block hardness values (not exported by the data generator, so they live here).

Exact values for common blocks, pattern rules for families; the bot also *learns* real dig times
at runtime and stores them in its brain, so anything missing here self-corrects.
"""
from __future__ import annotations

import re

EXACT = {
    "air": 0, "cave_air": 0, "void_air": 0, "water": -1, "lava": -1, "bedrock": -1, "barrier": -1,
    "end_portal": -1, "end_portal_frame": -1, "end_gateway": -1, "nether_portal": -1, "command_block": -1,
    "chain_command_block": -1, "repeating_command_block": -1, "structure_block": -1, "jigsaw": -1,
    "light": -1, "moving_piston": -1, "reinforced_deepslate": 55, "obsidian": 50, "crying_obsidian": 50,
    "respawn_anchor": 50, "ancient_debris": 30, "netherite_block": 50, "ender_chest": 22.5,
    "enchanting_table": 5, "anvil": 5, "chipped_anvil": 5, "damaged_anvil": 5, "iron_block": 5,
    "diamond_block": 5, "emerald_block": 5, "gold_block": 3, "copper_block": 3, "redstone_block": 5,
    "coal_block": 5, "lapis_block": 3, "iron_door": 5, "iron_trapdoor": 5, "iron_bars": 5, "hopper": 3,
    "spawner": 5, "trial_spawner": 50, "vault": 50, "bell": 5, "cobweb": 4, "dispenser": 3.5, "dropper": 3.5,
    "furnace": 3.5, "blast_furnace": 3.5, "smoker": 3.5, "observer": 3, "conduit": 3, "beacon": 3,
    "stone": 1.5, "cobblestone": 2, "mossy_cobblestone": 2, "smooth_stone": 2, "stone_bricks": 1.5,
    "deepslate": 3, "cobbled_deepslate": 3.5, "tuff": 1.5, "calcite": 0.75, "dripstone_block": 1.5,
    "granite": 1.5, "diorite": 1.5, "andesite": 1.5, "basalt": 1.25, "blackstone": 1.5, "end_stone": 3,
    "netherrack": 0.4, "nether_bricks": 2, "magma_block": 0.5, "glowstone": 0.3, "sea_lantern": 0.3,
    "dirt": 0.5, "coarse_dirt": 0.5, "rooted_dirt": 0.5, "grass_block": 0.6, "podzol": 0.5, "mycelium": 0.6,
    "farmland": 0.6, "dirt_path": 0.65, "mud": 0.5, "clay": 0.6, "sand": 0.5, "red_sand": 0.5, "gravel": 0.6,
    "soul_sand": 0.5, "soul_soil": 0.5, "snow": 0.1, "snow_block": 0.2, "powder_snow": 0.25, "ice": 0.5,
    "packed_ice": 0.5, "blue_ice": 2.8, "sandstone": 0.8, "red_sandstone": 0.8, "terracotta": 1.25,
    "glass": 0.3, "glass_pane": 0.3, "tinted_glass": 0.3, "crafting_table": 2.5, "bookshelf": 1.5,
    "chest": 2.5, "trapped_chest": 2.5, "barrel": 2.5, "lectern": 2.5, "note_block": 0.8, "jukebox": 2,
    "pumpkin": 1, "carved_pumpkin": 1, "jack_o_lantern": 1, "melon": 1, "hay_block": 0.5, "sponge": 0.6,
    "cactus": 0.4, "sugar_cane": 0, "bamboo": 1, "ladder": 0.4, "scaffolding": 0, "torch": 0,
    "redstone_torch": 0, "lever": 0.5, "tnt": 0, "slime_block": 0, "honey_block": 0, "bed": 0.2,
    "amethyst_block": 1.5, "budding_amethyst": 1.5, "prismarine": 1.5, "purpur_block": 1.5, "quartz_block": 0.8,
    "bone_block": 2, "brick": 2, "bricks": 2, "mud_bricks": 1.5, "packed_mud": 1, "moss_block": 0.1,
    "sculk": 0.2, "sculk_sensor": 1.5, "sculk_shrieker": 3, "sculk_catalyst": 3, "shroomlight": 1,
    "nether_wart_block": 1, "warped_wart_block": 1, "dried_kelp_block": 0.5, "target": 0.5, "lodestone": 3.5,
    "stonecutter": 3.5, "grindstone": 2, "smithing_table": 2.5, "cartography_table": 2.5,
    "fletching_table": 2.5, "loom": 2.5, "composter": 0.6, "cauldron": 2, "brewing_stand": 0.5,
    "campfire": 2, "soul_campfire": 2, "lantern": 3.5, "soul_lantern": 3.5, "chain": 5, "iron_chain": 5,
    "end_rod": 0, "chorus_plant": 0.4, "chorus_flower": 0.4, "dragon_egg": 3, "lightning_rod": 3,
    "pointed_dripstone": 1.5, "heavy_core": 10, "crafter": 1.5, "decorated_pot": 0, "suspicious_sand": 0.25,
    "suspicious_gravel": 0.25, "piston": 1.5, "sticky_piston": 1.5, "daylight_detector": 0.2,
}

PATTERNS = [
    (r"^deepslate_.*_ore$", 4.5), (r"_ore$", 3.0), (r"^raw_.*_block$", 5.0), (r"nether_gold_ore", 3.0),
    (r"_log$|_wood$|_stem$|_hyphae$|bamboo_block", 2.0), (r"_planks$", 2.0), (r"_leaves$", 0.2),
    (r"_wool$", 0.8), (r"_carpet$", 0.1), (r"_concrete$", 1.8), (r"_concrete_powder$", 0.5),
    (r"glazed_terracotta$", 1.4), (r"_terracotta$", 1.25), (r"_stained_glass(_pane)?$", 0.3),
    (r"_shulker_box$|^shulker_box$", 2.0), (r"_bed$", 0.2), (r"_banner$", 1.0), (r"_sign$", 1.0),
    (r"_door$", 3.0), (r"_trapdoor$", 3.0), (r"_fence_gate$", 2.0), (r"_fence$", 2.0),
    (r"_button$", 0.5), (r"_pressure_plate$", 0.5), (r"_candle$|^candle$", 0.1),
    (r"copper", 3.0), (r"deepslate", 3.5), (r"blackstone", 1.5), (r"_bricks?$", 2.0),
    (r"sandstone", 0.8), (r"(_slab|_stairs|_wall)$", 2.0), (r"_coral_block$", 1.5), (r"_coral", 0.0),
    (r"_mushroom_block$|mushroom_stem", 0.2), (r"froglight", 0.3), (r"_head$|_skull$", 1.0),
    (r"tuff", 1.5), (r"quartz", 0.8), (r"prismarine", 1.5), (r"purpur", 1.5), (r"_ice$", 0.5),
    (r"amethyst_bud|amethyst_cluster", 1.5), (r"infested_", 0.75), (r"resin", 0.0),
    (r"(sapling|flower|tulip|grass|fern|bush|roots|sprouts|vine|lichen|petals|wheat|carrots|potatoes|"
     r"beetroots|_stem|wart|kelp|seagrass|torch|rail|redstone_wire|tripwire|mushroom|fungus|lily|dandelion|"
     r"poppy|orchid|allium|bluet|daisy|cornflower|rose|peony|lilac|sunflower|dripleaf|spore|azalea$|"
     r"pickle|frogspawn|eyeblossom|cactus_flower|leaf_litter)", 0.0),
]
_COMPILED = [(re.compile(p), v) for p, v in PATTERNS]
_cache: dict[str, float] = {}


def hardness_of(block: str) -> float:
    if block in _cache:
        return _cache[block]
    if block in EXACT:
        v = EXACT[block]
    else:
        v = 1.5
        for rx, val in _COMPILED:
            if rx.search(block):
                v = val
                break
        # polished/chiseled/cut/smooth/mossy/cracked variants share the base block value
        if v == 1.5:
            base = re.sub(r"^(polished_|chiseled_|cut_|smooth_|mossy_|cracked_|infested_)", "", block)
            if base != block and base in EXACT:
                v = EXACT[base]
    _cache[block] = v
    return v
