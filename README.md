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
| Brain ("model file") | Each bot's persistent SQLite brain is at `data/brain/<bot-name>.db`; overnight training exports `data/brain/<bot-name>-export.json.gz`. Import a `.json.gz` backup with the app button. This is memory/skills, **not** an LLM weight file. |
| Wiki | minecraft.wiki client with an offline full-text cache; the trainer pre-fetches a summary for every block, item and mob |
| Server | Auto-downloads the server jar (SHA-1 verified) and Java 25 (Temurin, aarch64 or x64), accepts the EULA, writes a tuned `server.properties`, restarts the server if it crashes, and provides a console |

## Coordinated specialist AI and desktop chat

One Minecraft character can **listen and reason about several things at once**, but it
has only one set of movement and interaction controls. There are six cooperating roles:

- **Interpreter** routes chat messages to conversation, a direct command, a game goal or an AI action.
- **Chatter** can reply to messages in the app/Minecraft chat while the worker mines or walks.
- **Planner** suggests the worker's next useful step using live world state.
- **Builder** offers construction/bridging and fall-safety strategies when a build task is underway.
- **Scout** independently reads loaded blocks for ores while the worker performs other actions.
- **Escaper** raises critical-health alerts; the existing fast survival reflexes handle escape movement.

Roles share the blackboard with the worker and **do not independently drive the
bot**. Movement, digging, crafting and building requests are serialized through
the worker. Model-based specialists require a configured API/local model. The
scout and emergency warning logic work without one. Automated water-bucket
clutches are **not yet a tested movement skill**: don't depend on the bot to
survive a planned fall.

In the app, type in the **message box** and press **Send**; the conversation
pane displays your message, the bot's reply, and players' in-game chat. The
activity log below remains available for tool details and errors.

### Universal Minecraft skill model — any seed, new worlds

**Training now maintains two separate model snapshots** in addition to the
live SQLite brain, so you don't lose progress if you stop an overnight run:

- **World-independent reusable model:**
  `~/mcai/data/brain/PyBot-universal-model.json.gz`
- **Complete backup of this bot/world:**
  `~/mcai/data/brain/PyBot-export.json.gz`
- **Live brain database:** `~/mcai/data/brain/PyBot.db`

The **Train overnight** button creates both snapshots immediately, refreshes
them every three minutes while training (including long tasks), and saves
them at the end, on error, or when stopped. **Train one skill** and app
shutdown also save both. The snapshots are written atomically to avoid
replacing a good model with an incomplete file.

A **universal model** includes reusable parameterised action programs, game
knowledge, learned block timings, and skill success history. It deliberately
**excludes** old-seed waypoints, coordinates, episodes, base locations,
completed missions and per-world achievements. This lets the bot start on an
oak forest seed and use oak rather than trying to revisit a birch-tree
coordinate from another seed.

For a second seed or world, create/select a **fresh bot profile** (new local
brain), then click **Import model** and choose the universal
`.json.gz` file from the original bot. The imported scores are historical
experience, *not* automatic certification: risky manoeuvres must be verified
again in the new world. Click **Export universal model** to save a new copy.

The bot replans against currently loaded terrain, items, inventory and nearby
hazards. It can reuse techniques across different vanilla seeds and worlds on
supported Minecraft versions, but **no bot can guarantee every skill works in
every seed** (e.g. missing trees, unfamiliar modded blocks, unreachable
structures). There is no LLM fine-tuning here: these are transferable
**skills + memory + verification metrics**, not new neural-network weights
(`.gguf` / `.safetensors`). Ollama / LM Studio manage the external
language model separately.

### If the Minecraft server exits during startup

**Port conflict / "FAILED TO BIND TO PORT" / "Address already in use":**
PyFlayer checks its TCP port *before* starting Minecraft. By default, if
`server.port` (normally `25565`) is occupied, PyFlayer finds the next free
port, saves it to `config.json`, rewrites `server.properties`, and connects
the bot to that new port automatically. The desktop status and join address
show the selected port (for example `localhost:25566`). It never stops or
kills the process already using the old port. This can be disabled in Settings
with `server.auto_port_fallback=false` if you need an exact port.

If another **PyFlayer instance using the same world** is running, the app
still refuses to start a duplicate server to avoid corrupting the world.
Close the other instance first. To see what is listening on port 25565:

```bash
sudo ss -ltnp '( sport = :25565 )'
```

Check `data/server/26.3/server-console.log` (full captured output, including
early errors) and `data/server/26.3/logs/latest.log` (vanilla Minecraft log).
The app now includes these paths and an error excerpt in the startup exception.
A `java.lang.System::load` native-access warning alone is not the cause of a
crash. Check earlier `ERROR`, exception, port-binding, EULA, memory or
world-lock lines before changing Java settings. Failed startup no longer triggers
an unlimited overlapping restart loop.

**Exit code 143** commonly indicates a termination signal (SIGTERM), rather
than a Java exception. Only run one PyFlayer app / training service against a
given world. The app now refuses to automatically terminate another Java server
or launch two servers for the same world, and closing the GUI waits for a clean
save/shutdown. On a Raspberry Pi, inspect possible competing instances and
memory pressure with:

```bash
ps -eo pid,ppid,stat,cmd | grep -E '[j]ava|[p]yflayer'
free -h
journalctl -b --no-pager | grep -Ei 'out of memory|oom|sigterm|java|pyflayer' | tail -n 50
```

These checks diagnose the likely external signal; they do not guarantee
that every source of SIGTERM is fixed.

## Install on Raspberry Pi OS

**You need:** 64-bit Raspberry Pi OS (Bookworm or Trixie), a Raspberry Pi 4 or 5 with 4–8 GB RAM, about 1 GB of free
disk, and an internet connection for the first start. Minecraft 26.x needs Java 25, which only exists for 64-bit ARM;
check with `uname -m`, which must print `aarch64`.

### Option A: download the latest release (recommended)

1. Open the [Releases page](https://github.com/MilkdromedaStudios/mcai/releases/latest) and note the version (e.g. `v2.4.0`).
2. In a terminal on the Pi, download and unpack it (replace `v2.4.0` with that version):

   ```bash
   curl -L -o pyflayer.tar.gz https://github.com/MilkdromedaStudios/mcai/releases/download/v2.4.0/pyflayer-v2.4.0.tar.gz
   tar xzf pyflayer.tar.gz && cd pyflayer-v2.4.0
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
git tag v2.4.0 && git push origin v2.4.0
```

The workflow then builds `pyflayer-v2.4.0.tar.gz` / `.zip` (with SHA-256 checksums) and attaches them to a new
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

### Groq multi-model routing

Choose **Groq** in Settings and enter your Groq API key. McAI routes requests by role:

- Chat, interpreter and quick routing: `llama-3.1-8b-instant`
- Planner/builder world and inventory interpretation: `meta-llama/llama-4-scout-17b-16e-instruct`
- Main tool-using agent and task reflection: `openai/gpt-oss-120b`
- Logic-model rate-limit fallback: `qwen/qwen3.8-27b`

If GPT-OSS returns HTTP 429, McAI switches logic requests to the Qwen fallback until the next UTC midnight. The four model IDs are configurable in Settings and in `config.json` under `llm.models`. Other providers continue using the single configured model.

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

The bot now **begins with a concrete milestone** (collect wood) instead of
randomly walking 140 blocks first. It attempts a real action, verifies the
inventory or world result, and records success/failure. When an AI model claims
to have completed a task without meeting the goal, the trainer tries its built-in
resource planner. If a required resource cannot be found, the bot tries three
short relocation surveys, then reports **TRAINING BLOCKED** with the specific
last error rather than walking forever or silently going idle. A disconnect or
unexpected exception shows **TRAINING ERROR**. The desktop status includes the
current goal, attempts, successes and failures.

Reports and resumable memory are saved under `data/training/` and
`data/brain/`, even when the bot stops early. This trains *action skills and
memory*, not the weights of an external language model. Keep the app open for
an overnight session; the one-bot training button will not launch duplicate
training threads. If it reports blocked/error, check the training report and
`data/logs/pyflayer.log` for the reason.

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

## Adaptive skills, resource choices, and hazard movement

**Nearest useful resource wins.** For generic wood tasks, the agent uses
`gather_nearest_wood` / the deterministic planner. It checks loaded trees and
gathers the nearest reachable species instead of demanding birch in an oak
forest. Tree locations are refreshed when moving between chunks.

**Practice is measured, not claimed.** Train overnight runs a dry-ground skill
drill after every two curriculum attempts. The app also has a **Train one
skill** button. Skills include nearest-resource selection, safe navigation,
sprint-jumping on solid ground, and placing/reclaiming blocks. Every attempt is
checked against real inventory, movement or world-state changes, and stored in
the selected bot's brain. Use `skill_progress` through the AI to see its
scores. Skills are certified only after **at least 7 actual successes in 8
attempts and at least 85% accuracy**. These measured results also persist in
brain backup files.

**Emergency survival:**
- The existing water-bucket clutch reflex reacts to an *accidental* real fall.
  It no longer counts a successful clutch just because a click packet was sent;
  it requires a server-side bucket inventory change and a safe landing.
- Close-range combat no longer blindly sprints into known lava/unsafe gaps.
  It gives such terrain to the pathfinder, which must confirm that any bridge
  block exists before stepping onto it.
- An experimental sprint-jump across exactly **one known lava gap with an
  existing solid landing** is implemented but **disabled by default**. It
  requires certified dry-ground jump and placement drills, good health/food,
  and explicit `training.allow_advanced_moves=true` in Settings. It does
  **not** yet attempt blind midair block clutching over lava or deliberately
  create a fatal training fall. Those techniques still need controlled live
  tests before they can be relied upon.

The automated tests validate these controls and skill-score logic; they do
not constitute a live proof of water-bucket timing or lava-gap safety on your
Raspberry Pi. Use a disposable test world for experimental maneuvers.

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

## Local AI, emergency fallbacks and LiteLLM

**Overnight training on a local AI (free, no rate limits, zero setup).** Press **Train overnight** and choose
**Local AI** (or pick provider **ollama** in Settings). The first time, PyFlayer does everything itself:
- downloads Ollama into `~/mcai/.ollama` if it isn't installed (no sudo; ~1.5 GB download) and starts it in the
  background (one model loaded, one request at a time) - and stops it when you quit;
- with model **auto** picks a model that fits the machine: `qwen2.5:1.5b` on a Raspberry Pi (a Pi CPU is slow, so
  speed comes first), `qwen2.5:3b` on a PC with 6 GB+, `qwen2.5:7b` with 24 GB RAM or a 10 GB+ GPU, and downloads it
  (shown in the activity log). For a much faster AI, run Ollama on a PC and put its address in the URL field;
- for Ollama on another PC (URL like `http://192.168.1.20:11434/v1`) it downloads the model there.

To keep the machine cool all night PyFlayer:
- sends one request at a time with a pause between them (`min_interval_s`, default 1.5 s) and short answers;
- creates a capped copy of the model in Ollama (`pyflayer-<model>`) using all CPU cores but one (one stays free
  for Minecraft), a 2048 context (faster) and your GPU-layer limit (`0` = CPU only).

**Power cuts.** During overnight training PyFlayer saves a checkpoint every minute and after every step
(`data/training/<bot>-checkpoint.json`, written safely so a power cut can't corrupt it), commits the bot's brain,
exports its model files and runs `save-all` on the Minecraft world every 3 minutes. While training runs it also
adds a desktop autostart entry, so when the Pi boots again PyFlayer opens by itself and resumes the run after a
30-second countdown (press *Don't resume* to cancel) with the remaining time and the steps already done. Without
the desktop: `python -m pyflayer resume`. The autostart entry is removed when training finishes or you stop it.

From the command line: `python -m pyflayer train --local`.

**Many API keys.** Put several keys for one provider in the API key box, separated by commas
(`gsk_aaa, gsk_bbb, gsk_ccc`). When one key hits its rate limit the bot switches to the next free key straight
away, and comes back to the first one once its limit has passed. With LiteLLM every key becomes its own deployment,
so the load is spread over all of them.

**Emergency fallback.** If every model of your main AI is rate-limited (or the provider is down), the bot answers
straight away with the next AI in *Settings > Emergency fallback AIs*, e.g. `openrouter, gemini, ollama`
(each uses the key saved for that provider), instead of pausing - handy while streaming. If all of them are
limited too, it waits for the soonest one.

**LiteLLM router (installed and managed automatically).** `install.sh` installs
[LiteLLM](https://github.com/BerriAI/litellm) in its own environment (`.venv-litellm`, so it can't clash with
PyFlayer's packages or Raspberry Pi OS; skip it with `--no-litellm`). Choose provider **litellm** in Settings (or
put `litellm` in the emergency fallbacks) and PyFlayer does the rest: it writes `data/litellm_config.yaml` from your
saved keys (free models first, a backup group, your local Ollama last), makes a private master key, starts the
router on `127.0.0.1:4000`, restarts it when you change keys and stops it when you quit. Its log is
`data/logs/litellm.log`. `python -m pyflayer litellm-config` just writes the config file.

## Buckets (water and lava)

- **Water clutch:** when falling far with a water bucket in the hotbar, the bot aims down, pours water just
  before landing (no fall damage) and scoops it back up. It keeps a water bucket in hotbar slot 9 for this.
- **Burning enemies:** in a fight with a lava bucket, it pours lava under the mob from a safe distance, backs
  off, then takes the lava back. It skips fire-immune mobs (blazes, piglins, ...).
- **Obsidian for the nether portal:** with no obsidian nearby but a lava pool, it pours water over the lava,
  scoops the water back and mines the obsidian with a diamond pickaxe. The goal planner does this on its own.
- **Filling buckets:** "get me a water bucket" or "lava bucket" works too.

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
