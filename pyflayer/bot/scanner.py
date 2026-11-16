"""Surroundings scanner: blocks, ores, biomes and structures within a radius (default 100 blocks).

Structures are not sent to clients, so they are recognised from signature blocks
(e.g. a bell => village, end_portal_frame => stronghold), the same way a player would.
"""
from __future__ import annotations

import math
import re
import time

import numpy as np

# structure -> (signature blocks, minimum total count of those blocks, extra condition)
STRUCTURES = {
    "village": (["bell"], 1, None),
    "village (houses/paths)": (["dirt_path"], 25, ["composter", "lectern", "barrel", "smoker", "blast_furnace",
                                                  "cartography_table", "fletching_table", "grindstone", "loom",
                                                  "smithing_table", "stonecutter", "hay_block"]),
    "stronghold": (["end_portal_frame", "end_portal"], 1, None),
    "stronghold (corridors)": (["infested_stone_bricks", "infested_mossy_stone_bricks", "infested_cracked_stone_bricks"], 3, None),
    "dungeon (mob spawner room)": (["spawner"], 1, ["mossy_cobblestone"]),
    "mob spawner": (["spawner"], 1, None),
    "mineshaft": (["cobweb"], 6, ["rail", "oak_fence", "dark_oak_fence"]),
    "desert pyramid": (["orange_terracotta"], 6, ["chiseled_sandstone"]),
    "jungle temple": (["chiseled_stone_bricks"], 2, ["mossy_cobblestone", "tripwire_hook", "dispenser"]),
    "ocean monument": (["prismarine_bricks", "dark_prismarine", "sea_lantern"], 20, None),
    "ruined portal": (["crying_obsidian"], 1, ["obsidian", "netherrack"]),
    "pillager outpost": (["dark_oak_log"], 20, ["birch_planks"]),
    "woodland mansion": (["dark_oak_planks"], 400, ["cobblestone", "birch_planks"]),
    "ancient city": (["reinforced_deepslate", "sculk_shrieker"], 2, None),
    "trial chambers": (["trial_spawner", "vault"], 1, None),
    "trail ruins": (["suspicious_gravel"], 2, None),
    "desert well / ruins": (["suspicious_sand"], 1, None),
    "amethyst geode": (["budding_amethyst", "amethyst_block"], 4, None),
    "swamp hut": (["potted_red_mushroom"], 1, ["cauldron"]),
    "igloo": (["white_carpet"], 1, ["snow_block"]),
    "shipwreck": (["chest"], 1, ["water"]),
    "nether fortress": (["nether_bricks", "nether_brick_fence"], 30, ["nether_brick_fence"]),
    "bastion remnant": (["gilded_blackstone", "polished_blackstone_bricks"], 20, None),
    "end city": (["purpur_block", "purpur_pillar"], 20, ["end_rod"]),
}

ORES = ["coal_ore", "deepslate_coal_ore", "iron_ore", "deepslate_iron_ore", "copper_ore", "deepslate_copper_ore",
        "gold_ore", "deepslate_gold_ore", "redstone_ore", "deepslate_redstone_ore", "lapis_ore", "deepslate_lapis_ore",
        "diamond_ore", "deepslate_diamond_ore", "emerald_ore", "deepslate_emerald_ore", "nether_quartz_ore",
        "nether_gold_ore", "ancient_debris", "obsidian"]
USEFUL = ["crafting_table", "furnace", "chest", "barrel", "bed", "water", "lava", "sugar_cane", "wheat", "carrots",
          "potatoes", "melon", "pumpkin", "sweet_berry_bush", "hay_block", "cobweb", "nether_portal", "end_portal"]


class Scanner:
    def __init__(self, bot):
        self.bot = bot
        self.gd = bot.gd
        self.last: dict | None = None
        self.last_time = 0.0

    def _nearest(self, names, radius, count=1):
        return self.bot.world.find_blocks(self.bot.eye, self.gd.states_of(*names), radius, count)

    def detect_structures(self, radius: float = 100, census: dict | None = None) -> list[dict]:
        census = census or self.bot.world.block_census(self.bot.position, radius)
        found = []
        seen_pos = []
        for sname, (sig, min_count, extra) in STRUCTURES.items():
            total = sum(census.get(b, 0) for b in sig)
            if total < min_count:
                continue
            if extra and not any(census.get(b, 0) > 0 for b in extra):
                continue
            pts = self._nearest(sig, radius, 64)
            if not pts:
                continue
            if sname == "shipwreck":
                # chest with water next to it
                pts = [p for p in pts if any(self.bot.world.name_at(p[0] + dx, p[1], p[2] + dz) == "water"
                                             for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
                if not pts:
                    continue
            arr = np.array(pts[:64], dtype=float)
            first = arr[0]
            cluster = arr[np.linalg.norm(arr - first, axis=1) < 40]
            c = cluster.mean(axis=0)
            pos = (int(c[0]), int(c[1]), int(c[2]))
            if any(math.dist(pos, p) < 12 and sname.split(" ")[0] == n.split(" ")[0] for n, p in seen_pos):
                continue
            seen_pos.append((sname, pos))
            found.append({"structure": sname, "pos": pos,
                          "distance": round(math.dist(pos, self.bot.position), 1),
                          "evidence": {b: census.get(b, 0) for b in sig + (extra or []) if census.get(b, 0)}})
        found.sort(key=lambda d: d["distance"])
        return found

    def scan(self, radius: float = 100) -> dict:
        bot = self.bot
        t0 = time.time()
        census = bot.world.block_census(bot.position, radius)
        ores = {}
        for o in ORES:
            if census.get(o):
                p = self._nearest([o], radius, 1)
                if p:
                    ores[o] = {"count": census[o], "nearest": p[0], "distance": round(math.dist(p[0], bot.position), 1)}
        useful = {}
        for u in USEFUL:
            names = [u] if u != "bed" else [b for b in census if b.endswith("_bed")]
            cnt = sum(census.get(n, 0) for n in names)
            if cnt and names:
                p = self._nearest(names, radius, 1)
                if p:
                    useful[u] = {"count": cnt, "nearest": p[0], "distance": round(math.dist(p[0], bot.position), 1)}
        logs = {b: c for b, c in census.items() if b.endswith(("_log", "_stem")) and not b.startswith("stripped")}
        tree = None
        if logs:
            p = self._nearest(list(logs), radius, 1)
            tree = {"types": logs, "nearest": p[0] if p else None}
        biomes = bot.world.biome_census(bot.position, radius)
        structures = self.detect_structures(radius, census)
        mobs = {}
        for e in bot.entities.within(bot.position, radius):
            if e.type in ("item", "experience_orb", "arrow"):
                continue
            mobs.setdefault(e.type, []).append(round(e.distance_to(*bot.position), 1))
        res = {
            "position": tuple(round(v, 1) for v in bot.position),
            "dimension": bot.dimension,
            "biome_here": bot.world.biome_at(*bot.block_pos),
            "radius": radius,
            "ores": ores,
            "trees": tree,
            "useful_blocks": useful,
            "biomes": {k: {"cells": v[0], "nearest": v[1]} for k, v in list(biomes.items())[:12]},
            "structures": structures,
            "mobs": {k: {"count": len(v), "nearest": min(v)} for k, v in mobs.items()},
            "block_types_seen": len(census),
            "census": census,
            "scan_ms": round((time.time() - t0) * 1000),
        }
        self.last = res
        self.last_time = time.time()
        bot.events.emit("scan", res)
        return res

    @staticmethod
    def summarize(res: dict, max_len: int = 1800) -> str:
        lines = [f"Scan r={res['radius']} at {res['position']} ({res['dimension']}, biome {res['biome_here']}), "
                 f"{res['block_types_seen']} block types, {res['scan_ms']}ms"]
        if res["structures"]:
            lines.append("Structures: " + "; ".join(f"{s['structure']} ~{s['pos']} ({s['distance']}m)"
                                                    for s in res["structures"][:8]))
        if res["ores"]:
            lines.append("Ores: " + "; ".join(f"{k} x{v['count']} nearest {v['nearest']} ({v['distance']}m)"
                                              for k, v in sorted(res["ores"].items(), key=lambda kv: kv[1]["distance"])))
        if res["trees"]:
            t = res["trees"]
            lines.append(f"Trees: {', '.join(f'{k} x{v}' for k, v in sorted(t['types'].items(), key=lambda kv: -kv[1])[:5])}"
                         f" nearest {t['nearest']}")
        if res["useful_blocks"]:
            lines.append("Useful: " + "; ".join(f"{k} {v['nearest']} ({v['distance']}m)" for k, v in res["useful_blocks"].items()))
        if res["biomes"]:
            lines.append("Biomes: " + ", ".join(f"{k} {v['nearest']}" for k, v in res["biomes"].items()))
        if res["mobs"]:
            lines.append("Mobs: " + ", ".join(f"{k} x{v['count']} ({v['nearest']}m)" for k, v in res["mobs"].items()))
        s = "\n".join(lines)
        return s[:max_len]

    def locate_structure(self, structure: str, timeout: float = 8.0) -> str:
        """Ask the server via /locate (only when the bot is op and cheats are allowed)."""
        bot = self.bot
        structure = structure if ":" in structure or structure.startswith("#") else "minecraft:" + structure
        result = {}

        def on_msg(sender, text, system):
            if system and ("nearest" in text.lower() or "could not find" in text.lower()
                           or "unknown" in text.lower() or "permission" in text.lower()):
                result["text"] = text
        bot.events.on("message", on_msg)
        try:
            bot.chat(f"/locate structure {structure}")
            end = time.time() + timeout
            while time.time() < end and "text" not in result:
                time.sleep(0.1)
        finally:
            bot.events.off("message", on_msg)
        text = result.get("text", "no answer (is the bot op?)")
        m = re.search(r"\[(-?\d+), ~?(-?\d+|~), (-?\d+)\]", text)
        return text if not m else f"{structure} at x={m.group(1)} z={m.group(3)} ({text})"
