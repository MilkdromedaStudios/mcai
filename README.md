# PyFlayer AI: a Minecraft 26.3 AI bot for Raspberry Pi (Python)

PyFlayer is a Python Minecraft client library similar to [mineflayer](https://github.com/PrismarineJS/mineflayer),
written from scratch for **Minecraft Java 26.3 (protocol 777)**. On top of it sits an AI agent driven by any
**OpenAI-compatible API** (you set your own base URL later), with a persistent "brain" that learns every session
and an **overnight training mode**. It downloads and runs its own **offline-mode ("cracked") server** and joins it.
A Tkinter desktop app makes it a normal app on Raspberry Pi OS.

Everything below was tested against a real 26.3 server: login, chunks, digging, placing, crafting tables,
furnaces, pathfinding with digging, combat, scanning, the LLM tool loop and the trainer.

## What it can do

| Area | Details |
|---|---|
| Protocol | Handshake, offline login, configuration phase (registries, known packs, code of conduct, resource packs), play state, compression, keep-alive, chunk batches, 26.x one-movement-per-tick rule, LpVec3 velocities, stepped entity movement, item components |
| World | Numpy chunk storage, block and biome palettes, block updates, **block search across about 100 blocks in ~10 ms**, block and biome census |
| Physics | Vanilla-accurate gravity, friction, sprint jumping, step-up, slabs and stairs, water, lava, ladders, and sneak edge-guard, so the server never rejects a move |
| Pathfinding | A* with walking, diagonals, jumping, dropping, swimming, **digging through**, digging down, **pillaring up**, **bridging**, partial paths and re-planning |
| Interaction | Aim at blocks (picks the visible face) and entities (hitbox-aware), dig with best-tool selection and learned dig times, place blocks, use doors, levers, beds and buttons, open chests, furnaces and crafting tables |
| Inventory | Window model, clicks, swap and shift-click, equip (hand, offhand, armor), toss, **crafting in 2x2 and 3x3 grids**, **smelting**, deposit and withdraw |
| Combat and survival | Attack with cooldown, chase, flee, bow ballistics, creeper avoidance; reflexes fight back, pillar out of reach when hurt, eat; nights are spent in a sealed hole dug underground (or in a bed); items are recovered after a death |
| Planner | `bot.obtain("iron_pickaxe")` works out the whole tree from the server's own recipes, tags and loot tables (mine wood → planks → table → pickaxe → stone → furnace → iron ore → smelt → craft), including tool tiers |
| Scanner | 100-block scan: ores with nearest coordinates, trees, water and lava, stations, **biomes**, **structures** (villages, strongholds, mineshafts, dungeons, temples, monuments, ruined portals, ancient cities, trial chambers, geodes, fortresses, bastions, end cities…) and mobs |
| Late game | Build and light a nether portal, travel through portals, triangulate strongholds with eyes of ender, fill the end portal, and a basic dragon routine (crystals, then perching) |
| AI agent | 50+ tools for the LLM, memory-augmented prompts, reflection into lessons, **automatic skill learning**, survival reflexes, and chat commands that work without an LLM |
| Brain ("model file") | SQLite `data/brain/model.db` holding episodes, lessons, skills, waypoints, learned hardness, milestones, seen blocks and a knowledge base of every block, item, mob, biome and structure. Export or import as `model-*.json.gz` |
| Wiki | minecraft.wiki client with an offline full-text cache; the trainer pre-fetches a summary for every block, item and mob |
| Server | Auto-downloads the server jar (SHA-1 verified) and Java 25 (Temurin, aarch64 or x64), accepts the EULA, writes a tuned `server.properties`, restarts the server if it crashes, and provides a console |

## Install on Raspberry Pi OS

**You need:** 64-bit Raspberry Pi OS (Bookworm or Trixie), a Raspberry Pi 4 or 5 with 4–8 GB RAM, about 1 GB of free
disk, and an internet connection for the first start. Minecraft 26.x needs Java 25, which only exists for 64-bit ARM;
check with `uname -m`, which must print `aarch64`.

### Option A: download the latest release (recommended)

1. Open the [Releases page](https://github.com/MilkdromedaStudios/mcai/releases/latest) and note the version (e.g. `v1.0.0`).
2. In a terminal on the Pi, download and unpack it (replace `v1.0.0` with that version):

   ```bash
   curl -L -o pyflayer.tar.gz https://github.com/MilkdromedaStudios/mcai/releases/download/v1.0.0/pyflayer-v1.0.0.tar.gz
   tar xzf pyflayer.tar.gz && cd pyflayer-v1.0.0
   ```

3. Run the installer (it asks for your password once to install system packages):

   ```bash
   ./install.sh
   ```

### Option B: install from source

```bash
git clone https://github.com/MilkdromedaStudios/mcai pyflayer && cd pyflayer
./install.sh
```

### What the installer does

- installs `python3-tk python3-venv python3-numpy` with apt;
- creates a private Python environment in `.venv` with the dependencies;
- adds **PyFlayer AI** to the Games menu and the desktop, plus a `pyflayer` command in `~/.local/bin`;
- downloads the Minecraft 26.3 server, Java 25 and the game data (about 150 MB, one time).

Options: `./install.sh --nightly 23:00` also trains automatically every night at 23:00; `./install.sh --no-setup`
skips the downloads (they then happen on first start).

### After installing

1. Start **PyFlayer AI** from the Games menu (or run `~/.local/bin/pyflayer gui`).
2. Click **Settings**, pick an AI provider (OpenAI, OpenRouter, Anthropic, Gemini, Groq, Ollama, …), paste your
   API key, click **Test AI connection**, then **Save**.
3. Click **Server**, pick a bot card, then **Start bot** (or **Train overnight**).
4. To join in Minecraft yourself, use one of the addresses on the **Join:** line at the bottom of the window.

**Updating:** open **Settings → Check for updates** (or run `~/.local/bin/pyflayer update`). It downloads the newest
release, checks its checksum, installs it and keeps your `data/` folder (world, bots' memory) and `config.json`;
the previous version is saved in `data/updates/`. The app also tells you at start-up when an update is available.
For a source install (Option B) the updater runs `git pull`.

**Uninstalling:** delete the folder, `~/.local/bin/pyflayer`, `~/.local/share/applications/pyflayer.desktop` and,
if you enabled nightly training, run `systemctl --user disable --now pyflayer-train.timer`.

### Releases (for maintainers)

GitHub Actions runs the tests on every push and pull request. To publish a release, push a version tag:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

The workflow then builds `pyflayer-v1.0.0.tar.gz` / `.zip` (with SHA-256 checksums) and attaches them to a new
GitHub Release. You can also run it by hand from the Actions tab (**Test and release → Run workflow**).

You can also set the AI endpoint from the command line (any OpenAI-compatible server):

```bash
~/.local/bin/pyflayer config llm.base_url=http://192.168.1.50:8000/v1 llm.api_key=sk-... llm.model=my-model
```

You can also do this from **Settings** in the app. With no endpoint set, everything except free-text AI tasks still works.

## Using it

**Desktop app:** `pyflayer gui` (or the menu entry). One simple screen:

- **Bot cards** at the top: click a card to select that bot (it shows online/offline, what it is doing and its
  health); **+ New bot** adds one (up to 5).
- **Start bot**, **Stop**, **Server** (start/stop your own server), **Join server** (connect the selected bot to
  another server, e.g. a friend's PyFlayer server), **Run from model file** (load a saved `.json.gz` brain into the
  selected bot and start it), **Train overnight**, **Settings**.
- Below: the selected bot's status, its live activity log, and the **Join:** addresses of your server.

On small screens (like the 7" Raspberry Pi touchscreen) the app switches to a compact layout automatically
(force it with *Small-screen layout* in Settings); lists and Settings scroll with the mouse wheel or touch.

**AI providers:** `pyflayer ai providers` lists presets (OpenAI/ChatGPT, OpenRouter, Anthropic, Gemini, Groq, DeepSeek, Mistral, Together, xAI, Ollama, LM Studio, or any custom OpenAI-compatible URL). Connect with `pyflayer ai use openrouter --key sk-or-...` (optionally `--model ...`), or choose one under Settings, where *Test AI connection* checks it.

**Play with it yourself:** click **Server**, then join with Minecraft Java 26.3 using an address from the **Join:**
line (the **public address** for other devices anywhere, your network address on the same Wi-Fi, or `localhost`). For the public address, forward TCP 25565 on your router to the Pi, or set `server.upnp=true` to have PyFlayer ask the router automatically. The server is offline mode, so any username works. Server commands (e.g. `op YourName`) can be typed in
`pyflayer server`. The server prints the same join info.

**Multiple bots (up to 5 on one server):** add bots with **+ New bot** and start each from its card, or `pyflayer bots add Builder`, then `pyflayer run --all` or `pyflayer train --all`. Each bot has its own brain
file (`data/brain/<name>.db`), agent and training report. Use `--bot NAME` to run one profile from the CLI.

**Game rules:** the server runs in **Easy** difficulty with monsters on (zombies, skeletons, creepers spawn), and
**keep inventory** is switched on for training, so deaths cost time but not gear. Change it in Settings
(`server.difficulty`, `training.keep_inventory`).

The app uses a dark theme.

**Console:** `pyflayer run` starts the server, bot and agent. Type a task in plain English, or `!commands`.

**Overnight training:** `pyflayer train --hours 7` (the default comes from `training.hours`).

**In-game chat** (from any player; set `agent.owner` to restrict it to one player):

```
!obtain iron_pickaxe     !mine oak_log 10     !craft torch 16     !smelt iron_ingot 5
!goto 100 64 -20         !come    !follow     !fight    !eat     !scan    !status   !stop
PyBot, build me a shelter          (free text: handled by the LLM)
```

Other commands: `pyflayer setup`, `pyflayer server`, `pyflayer wiki-sync`,
`pyflayer brain --export model.json.gz`, `pyflayer selftest`.

## How the learning works (honest version)

This doesn't train a neural network on a Pi; that isn't feasible in 7 hours. The bot learns the way the
research agents Voyager and MineDojo do, by building up **memory, skills and knowledge** that the LLM and the
planner reuse:

1. **Knowledge base, which takes seconds.** Every block, item, mob, biome and structure in 26.3 gets an entry built
   from the server's own data generator: recipes, smelting, drops, tool tier, hardness, food and tags.
   That is 100% of the blocks, exact for this version.
2. **Wiki enrichment.** Summaries from minecraft.wiki for about 1,900 titles, plus strategy guides (beating
   the game, nether portal, strongholds, ender dragon…), cached for offline use. This took about 2 minutes
   in testing at 20 titles per request.
3. **Practice curriculum (28 milestones):** wood → crafting table → wooden/stone tools → furnace → fuel →
   torches → food → iron pickaxe/sword → bucket → shield → iron armor → bed → diamonds → diamond pickaxe →
   obsidian → flint and steel → nether portal → enter the nether → blaze rods → ender pearls → eyes of ender
   → stronghold → end portal → enter the end → kill the dragon. With an LLM, each milestone becomes an agent
   task. Without one, the deterministic planner practises it. Timings, attempts and failures go into the brain.
4. **After every task:** the successful action sequence is saved as a reusable **skill**, and the LLM writes
   1–3 **lessons** into memory. Deaths are remembered with their cause, place and task. Real dig times update
   the **learned hardness** table. Scans store **waypoints** for ores, structures, biomes and stations.
5. **Next session:** relevant lessons, skills, milestones and knowledge are retrieved (SQLite FTS5 with
   stemming) and put into the LLM prompt.

A report is written continuously to `data/training/training-YYYYMMDD.md` and `.json`.

Realistic expectations: the early and mid game (tools, iron, armor, food, diamonds) is well within reach in
one night. The nether, stronghold and dragon stages are implemented but are hard for any bot. Expect several
nights of attempts, and they depend heavily on the world seed and on the LLM's quality.

## Performance and Raspberry Pi optimisations

- **Server:** RAM picked automatically (1.8 GB on a 4 GB Pi, 3 GB on 8 GB), G1GC flags, `sync-chunk-writes=false`
  (much faster on SD cards), network compression off for the local bot, `max-tick-time=-1` (no watchdog
  kills on slow ticks), and pausing when empty.
- **Bot:** numpy chunk storage; each section keeps the set of block states it contains, so searches skip
  about 99% of the world; vectorized palette unpacking; a strict 20 TPS tick loop that resyncs after hiccups;
  A* bounded by node and time budgets with partial paths; and LLM calls that never block physics or networking.
- **GUI:** two refreshes per second, and nothing heavy
  runs on the UI thread.
- Measured here: 265 chunks decoded during login, a 100-block block search in about 10 ms, a full 100-block
  scan in about 0.1–0.4 s, and the bot holding a steady 20.0 TPS.

## Library usage (mineflayer-style)

```python
from pyflayer.data.gamedata import GameData
from pyflayer.server.manager import ensure_gamedata
from pyflayer.bot.bot import Bot

gd = GameData(ensure_gamedata("26.3"))
bot = Bot(gd, "PyBot", "127.0.0.1", 25565)
bot.connect(); bot.wait_spawn()
bot.events.on("chat", lambda who, msg: bot.chat(f"hi {who}") if "hi" in msg else None)
bot.collect_block("#logs", 4)
bot.craft("crafting_table")
bot.obtain("stone_pickaxe")              # full plan from game data
bot.goto(100, 70, -30)                   # digs / pillars / bridges if needed
zombie = bot.entities.nearest(bot.position, lambda e: e.type == "zombie")
bot.fight(zombie)
print(bot.inventory.summary(), bot.world.biome_at(*bot.block_pos))
```

## Configuration (`config.json`)

| key | default | meaning |
|---|---|---|
| `llm.base_url`, `llm.api_key`, `llm.model` | – | your OpenAI-compatible endpoint |
| `llm.tool_mode` | `auto` | `native` tool calls, `json` fallback, or auto-detect |
| `server.host` / `server.port` | 127.0.0.1:25565 | point at another cracked server with `server.auto_start=false` |
| `server.view_distance` | 7 | 7 chunks is about 112 blocks, enough for the 100-block scanner |
| `server.memory_mb` | 0 (auto) | server heap |
| `bot.username` | PyBot | offline-mode name |
| `agent.allow_cheats` | false | lets the agent use `/locate` (the bot is opped on the local server) |
| `agent.owner` | "" | only obey chat from this player |
| `training.hours` | 7 | overnight duration |

Environment overrides: `PYFLAYER_LLM_BASE_URL`, `PYFLAYER_LLM_API_KEY` (or `OPENAI_API_KEY`), `PYFLAYER_LLM_MODEL`, `PYFLAYER_HOME`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Covers the protocol codecs (VarInt, positions, LpVec3 against the wiki's sample bytes, NBT), palette decoding, and, once game data has been generated, recipes and drops, physics and pathfinding on a synthetic world.

## Project layout

```
pyflayer/protocol   buffer/NBT, connection, item components, text
pyflayer/data       game data loader (from the server's data generator), hardness table
pyflayer/world      chunk storage + search
pyflayer/bot        bot core, physics, pathfinder, actions, planner, scanner, advanced (portals/end)
pyflayer/agent      LLM client, tools, agent, brain (memory), wiki, trainer
pyflayer/server     download / Java / run the vanilla server
pyflayer/gui        Tkinter desktop app
```

## Notes

- Packet ids, block states, items, recipes, tags and loot tables are **generated from the server jar** on
  first run (`data/gamedata/26.3`), so nothing is hardcoded per version. Field layouts follow the
  minecraft.wiki protocol page for 26.3. One correction found during testing: in 26.x the Player Action
  enum has `CHANGE_DESTROY_DIRECTION` inserted at index 1, so *finish digging* is 3, not 2.
- The local server runs in offline mode with the whitelist off (26.3 enables it by default). Only expose
  it to networks you trust.
- By running the installer or `setup` you accept the [Minecraft EULA](https://aka.ms/MinecraftEULA).
