"""The AI agent: an LLM (any OpenAI-compatible endpoint) driving the bot through tools, with
long-term memory, learned skills, the wiki, and fast non-LLM survival reflexes.

Without an LLM configured, the agent still works through chat commands (!obtain, !goto, !mine ...)
and the deterministic planner.
"""
from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
import time
import traceback

from ..bot.actions import ActionError
from ..bot.pathfinder import GoalNear
from ..bot.scanner import Scanner
from .llm import LLM, LLMNotConfigured
from .memory import Brain
from .tools import ToolBox
from .wiki import Wiki

SYSTEM_PROMPT = """You are {name}, an autonomous Minecraft player (Java Edition {version}) controlled through tools.
You see the world only through tool results. Be efficient: prefer high-level tools.

How to act:
- Use obtain_item for anything that needs a chain of steps (tools, ingots, armor, food...). It plans from the
  real game data (recipes, smelting, drops, tool tiers) and executes everything.
- Use scan_surroundings to see ores, trees, structures and biomes within 100 blocks; find_blocks for one type.
- Use recipe/block_info for exact game facts of this version; use wiki for strategy and anything else.
- Check recall/list_skills before hard tasks: reuse what you learned in earlier sessions.
- After learning something non-obvious, call remember. When a multi-step routine worked, save_skill it.
- Keep yourself alive: eat when food < 14, flee or build_shelter when health is low, avoid lava.
- When the task is done or impossible, call task_complete with a short summary.
Progression to beat the game: wood -> crafting table -> wooden/stone tools -> furnace + coal -> iron tools,
armor, bucket, shield -> food supply + bed -> diamonds -> obsidian -> nether portal -> nether fortress (blaze rods)
-> ender pearls (endermen) -> eyes of ender -> stronghold (throw eyes) -> end portal -> destroy end crystals ->
kill the ender dragon.
"""


_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "some": 4, "a few": 3, "a stack of": 64, "stack of": 64}
_BLOCK_WORDS = {"wood": "#logs", "logs": "#logs", "log": "#logs", "trees": "#logs", "tree": "#logs", "stone": "stone",
                "cobble": "stone", "cobblestone": "stone", "dirt": "dirt", "sand": "sand", "coal": "coal_ore",
                "iron": "iron_ore", "diamonds": "diamond_ore", "diamond": "diamond_ore", "gravel": "gravel"}


def _resolve_item(words: str, gd) -> str | None:
    w = re.sub(r"[^a-z0-9 ]", "", words.lower()).strip()
    w = re.sub(r"^(the|some|me|my|us|of)\s+", "", w)
    cands = [w.replace(" ", "_")]
    if w.endswith("es"):
        cands.append(w[:-2].replace(" ", "_"))
    if w.endswith("s"):
        cands.append(w[:-1].replace(" ", "_"))
    for c in cands:
        if c in gd.item_ids:
            return c
    return None


def phrase_to_command(text: str, gd) -> str | None:
    """Turn a plain-English request into a ! command (no AI needed). None = not understood."""
    t = text.lower().strip(" .!?")
    if not t:
        return None
    if re.fullmatch(r"(stop|halt|wait|cancel|stand still|freeze)( it| that| now)?", t):
        return "!stop"
    if re.search(r"\b(come|get) (here|over here|to me)\b|^come$", t):
        return "!come"
    if re.search(r"\bfollow( me)?\b", t):
        return "!follow"
    if re.search(r"\b(status|how are you|where are you|your health)\b", t):
        return "!status"
    if re.search(r"\b(scan|look around|what('s| is) around)\b", t):
        return "!scan"
    if re.fullmatch(r"(eat|eat something|you('re| are) hungry)", t):
        return "!eat"
    m = re.search(r"\b(?:go|walk|move) to (-?\d+)[ ,]+(-?\d+)[ ,]+(-?\d+)", t)
    if m:
        return f"!goto {m.group(1)} {m.group(2)} {m.group(3)}"
    m = re.search(r"\b(?:fight|kill|attack)\s*(?:the|that|a|an)?\s*([a-z_ ]*)$", t)
    if m:
        target = m.group(1).strip().replace(" ", "_").rstrip("s") or "hostile"
        return f"!fight {target if target not in ('it', 'them', 'mob', 'monster') else 'hostile'}"

    def count_and_rest(s):
        s = s.strip()
        m2 = re.match(r"(\d+)\s+(.*)", s)
        if m2:
            return int(m2.group(1)), m2.group(2)
        for word, n in sorted(_NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
            if s.startswith(word + " "):
                return n, s[len(word) + 1:]
        return 1, s
    m = re.search(r"\b(?:mine|dig|chop|cut|collect|gather)\s+(?:me\s+|us\s+|some\s+)?(.+)", t)
    if m:
        n, rest = count_and_rest(m.group(1))
        rest = re.sub(r"\b(down|up|for me|please)\b", "", rest).strip()
        block = _BLOCK_WORDS.get(rest) or _resolve_item(rest, gd)
        if block:
            return f"!mine {block} {n}"
    m = re.search(r"\bsmelt\s+(?:me\s+)?(.+)", t)
    if m:
        n, rest = count_and_rest(m.group(1))
        item = _resolve_item(rest, gd)
        if item:
            return f"!smelt {item} {n}"
    m = re.search(r"\b(?:get|obtain|make|craft|build|bring|give)\s+(?:me\s+|us\s+)?(.+)", t)
    if m:
        n, rest = count_and_rest(m.group(1))
        rest = re.sub(r"\b(for me|please|now)\b", "", rest).strip()
        if rest in _BLOCK_WORDS and rest not in ("diamonds", "diamond", "coal", "iron"):
            word = {"wood": "oak_log", "logs": "oak_log", "log": "oak_log", "trees": "oak_log"}.get(rest)
            if word:
                return f"!mine #logs {n}"
        item = _resolve_item(rest, gd) or {"diamonds": "diamond", "coal": "coal", "iron": "iron_ingot",
                                            "wood": "oak_log"}.get(rest)
        if item:
            return f"!obtain {item} {n}"
    return None


_VERBS = [  # (words, action) - first match in the sentence wins
    (("stop", "halt", "cancel", "freeze", "stay still", "quit it"), "stop"),
    (("come", "over here"), "come"),
    (("follow", "stay with", "stick with"), "follow"),
    (("kill", "fight", "attack", "hunt", "slay", "defeat", "beat up"), "fight"),
    (("smelt", "cook", "roast"), "smelt"),
    (("eat", "hungry", "starving"), "eat"),
    (("mine", "dig", "chop", "cut down", "collect", "gather", "break", "harvest"), "mine"),
    (("find", "look for", "search", "locate", "where is", "where's", "go to the", "take me to"), "find"),
    (("get", "make", "craft", "build", "bring", "need", "want", "give", "obtain", "fetch", "grab", "create"), "obtain"),
    (("scan", "look around", "explore around", "what's around", "what is around", "surroundings"), "scan"),
    (("status", "how are you", "where are you", "health"), "status"),
]


def _ngrams(words, n):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def find_thing(text: str, gd):
    """Best item / block / mob named anywhere in a sentence (fuzzy, handles plurals and typos).
    Returns (kind, id) with kind in item|block|mob, or (None, None)."""
    import difflib
    text = re.sub(r"\b(\w+) (?:trees?|wood)\b", r"\1 log", text.lower())      # "birch trees" -> "birch log"
    words = re.findall(r"[a-z]+", text)
    mobs = {e.replace("_", " "): e for e in gd.entity_type_ids}
    items = {i.replace("_", " "): i for i in gd.item_ids}
    blocks = {b.replace("_", " "): b for b in gd.block_names}
    for n in (4, 3, 2, 1):                       # longest phrase first ("iron pickaxe" before "iron")
        for g in _ngrams(words, n):
            for cand in (g, g[:-1] if g.endswith("s") else g, g[:-2] if g.endswith("es") else g):
                if cand in items:            # items first: "ender pearl" is an item, not the thrown entity
                    return "item", items[cand]
                if cand in blocks:
                    return "block", blocks[cand]
                if cand in mobs:
                    return "mob", mobs[cand]
    for n in (3, 2, 1):                          # typos: "diamnd pickaxe", "furnce"
        for g in _ngrams(words, n):
            if len(g) < (6 if n == 1 else 4):
                continue
            for table, kind in ((items, "item"), (mobs, "mob")):
                m = difflib.get_close_matches(g, table.keys(), n=1, cutoff=0.88 if n == 1 else 0.84)
                if m:
                    return kind, table[m[0]]
    return None, None


# big goals -> step of the beat-the-game plan to work up to
GOALS = [
    (r"\b(beat|finish|win|complete|end)\b.*\b(game|minecraft)\b|\b(kill|defeat|slay|beat|fight)\b.*\bdragon\b",
     "kill_dragon"),
    (r"\b(go|get|reach|enter|travel|make it|take me|bring me|teleport|jump)\b.*\bthe end\b|\bend (dimension|island)\b",
     "enter_end"),
    (r"\bend portal\b", "end_portal"),
    (r"\bstronghold\b", "stronghold"),
    (r"\beyes? of ender\b|\bender eyes?\b", "eyes_of_ender"),
    (r"\bender ?pearls?\b", "ender_pearls"),
    (r"\bblaze rods?\b", "blaze_rods"),
    (r"\b(go|get|reach|enter|travel|take me|bring me)\b.*\bnether\b(?! portal)", "enter_nether"),
    (r"\bnether portal\b", "nether_portal"),
    (r"\b(full )?iron armou?r\b|\bfull iron\b", "iron_armor"),
    (r"\bgear up\b|\bget (geared|ready)\b|\bprepare\b", "iron_armor"),
]
QUESTION = re.compile(r"^\s*(what|how|why|which|who|when|does|do|can|could|is|are|tell me)\b|\?\s*$", re.I)


def match_goal(text: str) -> str | None:
    low = text.lower()
    for rx, goal in GOALS:
        if re.search(rx, low):
            return goal
    return None


def interpret(text: str, gd) -> tuple[str | None, str]:
    """Work out what any sentence means without an AI. Returns (! command or None, explanation)."""
    cmd = phrase_to_command(text, gd)
    if cmd:
        return cmd, f"understood: {cmd[1:]}"
    low = " " + text.lower() + " "
    action = next((a for words, a in _VERBS if any(f" {w}" in low for w in words)), None)
    m = re.search(r"\b(\d+)\b", low)
    count = int(m.group(1)) if m else next((n for w, n in _NUM_WORDS.items() if f" {w} " in low), 1)
    kind, thing = find_thing(text, gd)
    coords = re.findall(r"-?\d+", text)
    if action in ("stop", "come", "follow", "eat", "scan", "status"):
        return f"!{ {'scan': 'scan', 'status': 'status'}.get(action, action) }", f"understood: {action}"
    if action == "find" and len(coords) >= 3:
        return f"!goto {coords[0]} {coords[1]} {coords[2]}", "understood: go to those coordinates"
    if kind == "mob" and action in (None, "fight", "find", "mine"):
        return (f"!fight {thing}", f"understood: fight {thing}") if action != "find" else \
            (f"!find {thing}", f"understood: go to the nearest {thing}")
    if action == "fight":
        return "!fight hostile", "understood: fight the nearest monster"
    if thing is None:
        return None, "I couldn't tell which item, block or mob you mean"
    if action == "find":
        return f"!find {thing}", f"understood: go to the nearest {thing}"
    if action == "smelt":
        return f"!smelt {thing} {count}", f"understood: smelt {count} {thing}"
    if action == "mine" and thing in gd.block_index:
        return f"!mine {thing} {count}", f"understood: mine {count} {thing}"
    if thing in gd.item_ids:
        return f"!obtain {thing} {count}", f"understood: get {count} {thing}"
    return f"!mine {thing} {count}", f"understood: mine {count} {thing}"


class Agent:
    def __init__(self, bot, cfg, brain: Brain, wiki: Wiki, llm: LLM | None = None, server=None):
        self.bot = bot
        self.cfg = cfg
        self.brain = brain
        self.wiki = wiki
        self.llm = llm or LLM(cfg["llm"])
        self.server = server
        self.scanner = Scanner(bot)
        from .watcher import Blackboard, Watcher
        self.board = Blackboard()                  # shared notes between the worker AI and the watcher AI
        self.watcher = Watcher(self)
        from .specialists import SpecialistTeam
        self.specialists = SpecialistTeam(self)
        self.board.on_post = lambda line: self._note(line)
        self.messages: list[tuple] = []  # (timestamp, sender, message) for in-app chat
        self._actuation_lock = threading.RLock()
        self.tools = ToolBox(self)
        self.log = logging.getLogger("pyflayer.agent")
        self.tasks: queue.Queue = queue.Queue()
        self.current_task: str | None = None
        self.task_done = threading.Event()
        self.task_result = ("", False)
        self.trace: list[dict] = []        # tool calls of the current task (for skill learning)
        self.history: list[str] = []       # short log lines for the GUI
        self.running = False
        self.paused_reflexes = False
        self._reflex_busy = False
        self._last_scan = 0.0
        self.interrupted_at = 0.0
        self.last_death: tuple | None = None
        self._thread: threading.Thread | None = None
        self._reflex_thread: threading.Thread | None = None
        bot.learned_hardness.update(brain.learned_hardness())
        bot.events.on("learn", self._on_learn)
        bot.events.on("death", self._on_death)
        bot.events.on("chat", self._on_chat)
        bot.events.on("message", self._on_message)
        bot.events.on("crafted", lambda item, n: brain.bump(f"crafted:{item}", n))
        bot.events.on("dug", lambda pos, name: brain.bump("blocks_mined"))
        bot.events.on("plan", lambda msg: self._note(msg))

    # ================================================================== lifecycle
    def start(self):
        if self.running:
            return
        self.running = True
        self.brain.start_session("agent")
        self._thread = threading.Thread(target=self._loop, name="agent", daemon=True)
        self._thread.start()
        self._reflex_thread = threading.Thread(target=self._reflex_loop, name="reflexes", daemon=True)
        self._reflex_thread.start()
        self.watcher.start()
        self.specialists.start()

    def stop(self):
        self.running = False
        self.specialists.stop()
        self.watcher.stop()
        self.bot.stop()
        self.brain.end_session("stopped")

    def submit(self, task: str, source: str = "user", kind: str = "task"):
        self.tasks.put((task, source, kind))
        self._note(f"queued {kind} from {source}: {task}")

    def submit_command(self, command: str, source: str = "user"):
        # Commands that move/dig/build use the same actor queue as AI actions.
        # Safety stop is immediate even if another action is running.
        if command.split(maxsplit=1)[0].lower() == "!stop":
            self.bot.stop()
            with self.tasks.mutex:
                self.tasks.queue.clear()
            self._note("Stopped on request from " + source)
            return
        self.submit(command, source, kind="command")

    def _on_message(self, sender, text, system=False):
        if system or not text:
            return
        self.messages.append((time.time(), sender or "server", text))
        del self.messages[:-300]

    def _note(self, msg: str):
        line = time.strftime("%H:%M:%S ") + msg
        self.history.append(line)
        del self.history[:-400]
        self.log.info(msg)

    # ================================================================== event hooks
    def _on_learn(self, kind, data):
        if kind == "hardness":
            self.brain.save_hardness(data["block"], data["hardness"])
            self.brain.remember(f"{data['block']} takes longer to break than expected (effective hardness "
                                f"{data['hardness']})", "mining", 0.6, "measurement")

    def _on_death(self, msg):
        pos = self.bot.block_pos
        self.brain.log("death", f"died at {pos} in {self.bot.dimension}: {msg}", 3.0)
        self.brain.set_waypoint(f"death@{int(time.time())}", self.bot.dimension, pos, "death", msg)
        self.brain.bump("deaths")
        self.last_death = (time.time(), self.bot.dimension, pos)
        self.interrupted_at = time.time()
        self.brain.remember(f"I died: {msg} (at y={pos[1]}, {self.bot.dimension}, while doing: "
                            f"{self.current_task or 'nothing'})", "danger", 0.9, "death")
        self._note(f"DIED: {msg}")

    def _on_chat(self, sender, text):
        if not self.cfg["agent"]["listen_to_chat"]:
            return
        owner = self.cfg["agent"].get("owner")
        if owner and sender.lower() != owner.lower():
            return
        t = text.strip()
        self.brain.log("chat", f"<{sender}> {t}")
        low = t.lower()
        name = self.bot.username.lower()
        if t.startswith("!"):
            self.submit_command(t, sender)
            return
        # talk to a bot by name ("PyBot get me 5 oak logs"), or to all bots ("bots come here")
        if name in low or low.startswith(("bots", "everyone", "all bots", "hey bot", "bot ")):
            self.instruct(t, sender)

    def instruct(self, text: str, sender: str = "app"):
        """Do what a player (chat) or the app asks. Simple phrases work without an AI; anything else goes to the AI."""
        t = re.sub(re.escape(self.bot.username), "", text, flags=re.I)
        t = re.sub(r"^\W*(hey|ok|okay|yo|please|bots?|everyone|all bots)\b", "", t.strip(), flags=re.I).strip(" ,:!.?")
        self._note(f"instruction from {sender}: {t}")
        if sender == "app":
            self.messages.append((time.time(), "You", text))
            del self.messages[:-300]
        bot = self.bot
        low = t.lower()
        if re.fullmatch(r"(stop|halt|cancel|freeze|stay|stay still|wait)( it| that| now| please)?", low.strip(" !.")):
            self.stop_goal()
            bot.stop()
            bot.chat("Stopped.")
            return
        # Interpreter and chatter run separately from physical tasks.
        # Their replies can arrive while the main worker is gathering resources.
        if self.llm.configured and self.specialists.running:
            self.specialists.receive(t, sender)
            return
        # 1) big goals ("get to the end", "beat the game", "go to the nether", "get diamonds"...)
        goal = match_goal(t)
        if goal:
            self.start_goal(goal, sender)
            return
        # 2) questions are answered from the game's own data (and the wiki / AI)
        if QUESTION.search(t) and not re.search(r"\b(can|could) you\b(?!.*\?$)", low) and \
                not re.search(r"\bwhere\b.*\bfind\b", low):
            threading.Thread(target=self.answer, args=(t, sender), daemon=True).start()
            return
        # 3) with an AI: it works out anything else and plans it with all the bot's tools
        if self.llm.configured:
            quick, _ = interpret(t, bot.gd)
            hint = f" (a quick guess at the meaning is: {quick[1:]})" if quick else ""
            self._note("the AI is working out what that means...")
            self.submit(f"{sender} says: \"{t}\".{hint} Work out exactly what they want (anything a Minecraft player "
                        f"can do; for big goals use achieve_goal), say in one short chat line what you'll do, then do "
                        f"it with your tools. If it's unclear, ask one short question in chat.", source=sender)
            return
        # 4) without an AI: understand the sentence from the game data
        if re.search(r"\bwhere\b.*\bfind\b|\bfind\b", low):
            kind, thing = find_thing(t, bot.gd)
            if thing:
                bot.chat(f"Looking for {thing.replace('_', ' ')}...")
                self.submit_command(f"!find {thing}", sender)
                return
        cmd, why = interpret(t, bot.gd)
        self._note(why + (f" -> {cmd}" if cmd else ""))
        if cmd:
            bot.chat("On it: " + why.replace("understood: ", "").replace("_", " "))
            self.submit_command(cmd, sender)
        else:
            bot.chat(f"Sorry {sender}, I don't know how to do that yet. Try naming what you want, like "
                     f"'get 10 torches', 'kill the zombie', 'go to the nether' or 'get to the end'.")

    # ------------------------------------------------------------------ goals & questions
    def start_goal(self, goal: str, sender: str = "app"):
        """Long goals share the same actuation lane as mining and building."""
        from .trainer import Trainer
        if goal not in [m.name for m in Trainer(self, 12).curriculum]:
            self.submit(goal, source=sender)
            return
        self._note(f"goal requested: {goal}")
        self.submit(goal, source=sender, kind="goal")

    def stop_goal(self):
        tr = getattr(self, "goal_trainer", None)
        if tr and tr.running:
            tr.stop()

    def answer(self, question: str, sender: str = "app"):
        gd, bot = self.bot.gd, self.bot
        low = question.lower()
        kind, thing = find_thing(question, gd)
        text = None
        if thing:
            if kind == "mob":
                drops = sorted(gd.entity_drops.get(thing, []))
                text = f"{thing.replace('_', ' ')} drops: {', '.join(drops) or 'nothing'}"
            elif re.search(r"\b(make|craft|get|smelt|recipe|obtain)\b", low) and thing in gd.item_ids:
                text = gd.describe_item(thing)
            elif thing in gd.block_index and re.search(r"\b(mine|break|tool|pickaxe|drop|hard)\b", low):
                text = gd.describe_block(thing)
            else:
                text = gd.describe_item(thing) if thing in gd.item_ids else gd.describe_block(thing)
        if not text and self.llm.configured:
            self.submit(f"{sender} asks: \"{question}\". Answer briefly in chat (use recipe/block_info/wiki).", sender)
            return
        if not text:
            try:
                text = self.wiki.lookup(question, 400)
            except Exception:  # noqa: BLE001
                text = "I don't know that one."
        lines = [ln.strip() for ln in text.replace("\n", " | ").split(" | ") if ln.strip()]
        msg = ""
        for ln in lines:
            if len(msg) + len(ln) > 230:
                bot.chat(msg)
                msg = ""
            msg += (" | " if msg else "") + ln
        if msg:
            bot.chat(msg)
        self._note(f"answered: {text[:200]}")

    # ================================================================== chat commands (no LLM needed)
    def _chat_command(self, sender, text):
        bot = self.bot
        parts = text[1:].split()
        if not parts:
            return
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd == "help":
                bot.chat("!obtain <item> [n] | !mine <block> [n] | !craft <item> [n] | !smelt <item> [n] | !goto x y z | "
                         "!come | !follow | !stop | !scan | !status | !eat | !fight | !task <text> | !train")
            elif cmd == "stop":
                bot.stop()
                with self.tasks.mutex:
                    self.tasks.queue.clear()
                bot.chat("stopped")
            elif cmd == "status":
                bot.chat(self.status_text(short=True)[:250])
            elif cmd == "scan":
                res = self.scanner.scan(100)
                self.after_scan(res)
                for line in Scanner.summarize(res, 600).split("\n")[:4]:
                    bot.chat(line[:250])
            elif cmd == "come":
                e = bot.entities.nearest(bot.position, lambda e: e.name == sender, 512)
                if not e:
                    bot.chat("I can't see you")
                else:
                    bot.goto(GoalNear(e.x, e.y, e.z, 2), timeout=180)
                    bot.chat("here")
            elif cmd == "follow":
                e = bot.entities.nearest(bot.position, lambda e: e.name == sender, 512)
                if e:
                    bot.chat(bot.follow(e, 3, 300))
            elif cmd == "find" and args:
                what = args[0]
                if what in bot.gd.entity_type_ids:
                    e = bot.entities.nearest(bot.position, lambda x: x.type == what, 128)
                    if not e:
                        bot.chat(f"no {what} nearby")
                    else:
                        bot.goto(GoalNear(e.x, e.y, e.z, 3), timeout=180)
                        bot.chat(f"found a {what}")
                else:
                    pts = bot.find_blocks(what, 128, 1)
                    if not pts:
                        bot.chat(f"no {what} within 128 blocks")
                    else:
                        from ..bot.pathfinder import GoalLookAt
                        bot.goto(GoalLookAt(*pts[0], reach=3.5), timeout=240)
                        bot.chat(f"found {what} at {pts[0][0]} {pts[0][1]} {pts[0][2]}")
            elif cmd == "goto" and len(args) >= 3:
                bot.goto(float(args[0]), float(args[1]), float(args[2]), timeout=600)
                bot.chat(f"arrived {bot.block_pos}")
            else:
                mapping = {"obtain": "obtain_item", "get": "obtain_item", "mine": "mine_block", "craft": "craft",
                           "smelt": "smelt", "eat": "eat", "fight": "attack", "attack": "attack"}
                if cmd in mapping:
                    tool = mapping[cmd]
                    a = {}
                    if tool in ("obtain_item", "craft", "smelt"):
                        a = {"item": args[0], "count": int(args[1]) if len(args) > 1 else 1}
                    elif tool == "mine_block":
                        a = {"block": args[0], "count": int(args[1]) if len(args) > 1 else 1}
                    elif tool == "attack":
                        a = {"target": args[0] if args else "hostile"}
                    res = self.tools.call(tool, a)
                    bot.chat(res.split("\n")[0][:250])
                    self.brain.log("command", f"{sender}: {text} -> {res[:300]}")
                elif cmd == "task":
                    self.submit(" ".join(args), sender)
                    bot.chat("ok, on it" if self.llm.configured else "no AI endpoint configured")
                else:
                    bot.chat(f"unknown command {cmd}; try !help")
        except (ActionError, ValueError, IndexError) as e:
            bot.chat(f"failed: {e}"[:250])

    # ================================================================== perception text
    def status_text(self, short: bool = False) -> str:
        b = self.bot
        x, y, z = b.position
        biome = b.world.biome_at(*b.block_pos)
        s = (f"pos {x:.1f},{y:.1f},{z:.1f} {b.dimension} biome {biome} | health {b.health:.0f}/20 food {b.food}/20 "
             f"| {b.time_phase()} (t={b.time_of_day}) | holding {b.inventory.held_item!r}")
        if short:
            return s
        armor = {k: v.name for k, v in b.inventory.armor().items() if v}
        ents = b.entities.within(b.position, 24, lambda e: e.type not in ("item", "experience_orb"))
        items = b.entities.within(b.position, 12, lambda e: e.type == "item")
        s += f"\ninventory: {b.inventory.summary()}\narmor: {armor or 'none'}"
        if ents:
            s += "\nnearby: " + "; ".join(e.describe(b.position) for e in ents[:10])
        if items:
            s += f"\ndropped items nearby: {len(items)}"
        wps = self.brain.waypoints(b.dimension, b.position, None, 5)
        if wps:
            s += "\nknown places: " + "; ".join(f"{w['name']} {w['pos']}" for w in wps)
        return s

    def after_scan(self, res: dict):
        """Turn a scan into long-term spatial memory + 'seen' knowledge."""
        dim = self.bot.dimension
        self.brain.mark_seen("block", list(res["census"].keys()))
        self.brain.mark_seen("biome", list(res["biomes"].keys()))
        for st in res["structures"]:
            self.brain.set_waypoint(f"{st['structure']}@{st['pos'][0]},{st['pos'][2]}", dim, st["pos"], "structure")
            self.brain.mark_seen("structure", [st["structure"]])
        for ore, info in res["ores"].items():
            if ore in ("diamond_ore", "deepslate_diamond_ore", "ancient_debris", "emerald_ore", "gold_ore",
                       "deepslate_gold_ore", "iron_ore", "deepslate_iron_ore"):
                p = info["nearest"]
                self.brain.set_waypoint(f"{ore}@{p[0]},{p[1]},{p[2]}", dim, p, "ore")
        for name, info in res["biomes"].items():
            self.brain.set_waypoint(f"biome:{name}@{info['nearest'][0]},{info['nearest'][2]}", dim, info["nearest"], "biome")
        for u in ("crafting_table", "furnace", "chest", "bed", "nether_portal", "end_portal"):
            if u in res["useful_blocks"]:
                p = res["useful_blocks"][u]["nearest"]
                self.brain.set_waypoint(f"{u}@{p[0]},{p[1]},{p[2]}", dim, p, "station")
        self._last_scan = time.time()

    # ================================================================== reflexes (no LLM, fast)
    def _reflex_loop(self):
        last_check = 0.0
        while self.running:
            time.sleep(0.5)
            if self.paused_reflexes or not self.bot.spawned.is_set() or self.bot.dead:
                continue
            try:
                if time.time() - last_check >= 3.0:      # look at the surroundings every 3 seconds
                    last_check = time.time()
                    self._situation_check()
                self._reflexes()
            except ActionError:
                pass
            except Exception:  # noqa: BLE001
                self.log.debug("reflex error\n%s", traceback.format_exc())

    def situation(self) -> dict:
        """What is around the bot right now (cheap: a few block lookups)."""
        b = self.bot
        gd, w = b.gd, b.world
        fx, fy, fz = b.block_pos
        hx, hy, hz = (int(math.floor(v)) for v in b.eye)
        feet, head = w.get(fx, fy, fz), w.get(hx, hy, hz)
        below = w.get(fx, fy - 1, fz)
        name = lambda s: gd.block_name(s) if s is not None else "unloaded"  # noqa: E731
        return {
            "in_water": bool(feet is not None and gd.is_water[feet]) or b.state.in_water,
            "head_under_water": bool(head is not None and gd.is_water[head]),
            "in_lava": bool(feet is not None and gd.is_lava[feet]) or b.state.in_lava,
            "suffocating": bool(head is not None and gd.shape_kind[head] == 1 and not gd.is_liquid[head]),
            "on_fire_block": name(feet) in ("fire", "soul_fire") or name(below) == "magma_block",
            "feet": name(feet), "head": name(head), "below": name(below),
        }

    def _situation_check(self):
        """Every 3 s: re-read the surroundings and fix bad situations (water, lava, stuck in a block...)."""
        b = self.bot
        sit = self.situation()
        pos = b.position
        prev = getattr(self, "_last_pos", None)
        self._last_pos = (pos, time.time())
        moved = prev is None or math.dist(prev[0], pos) > 0.5
        action = b.current_action
        problem = None
        if sit["in_lava"] or sit["on_fire_block"]:
            problem = "lava"
        elif sit["suffocating"]:
            problem = "suffocating"
        elif sit["in_water"]:
            # swimming as part of a path is fine; idling or going nowhere in water is not
            self._water_since = getattr(self, "_water_since", 0) or time.time()
            if action == "idle" or (not moved and time.time() - self._water_since > 6):
                problem = "water"
        else:
            self._water_since = 0
        # trying to walk but not moving for 6 s = stuck on a wall/corner
        walking = b.controls.forward or b.controls.back or b.controls.left or b.controls.right
        if walking and not moved and action != "idle" and not problem:
            self._stuck_checks = getattr(self, "_stuck_checks", 0) + 1
            if self._stuck_checks >= 2:
                self._stuck_checks = 0
                self._unstick()
                return
        else:
            self._stuck_checks = 0
        if not problem or self._reflex_busy:
            return
        self._reflex_busy = True
        try:
            self._note(f"awareness: {problem} (feet={sit['feet']}, head={sit['head']}) - fixing it")
            if not self._take_over():
                return
            if problem in ("lava", "water"):
                b.controls.jump = True
                self._note(b.escape_liquid())
            elif problem == "suffocating":
                hx, hy, hz = (int(math.floor(v)) for v in b.eye)
                b.dig((hx, hy, hz))
            self.brain.log("awareness", f"fixed {problem} at {b.block_pos}")
        except ActionError as e:
            self._note(f"awareness: could not fix {problem}: {e}")
        finally:
            b.controls.jump = False
            self._reflex_busy = False

    def _unstick(self):
        """Wiggle free like a player: back up, sidestep, jump; then break what is in front of us."""
        import random
        b = self.bot
        c = b.controls
        start = b.position
        self._note(f"awareness: stuck at {b.block_pos} - wiggling free")
        self._reflex_busy = True
        try:
            self._take_over()          # stop the stuck action; the planner/trainer re-plans afterwards
            for keys in (("back",), ("left", "back") if random.random() < 0.5 else ("right", "back")):
                c.clear()
                for k in keys:
                    setattr(c, k, True)
                c.jump = True
                b.wait_ticks(6)
                c.clear()
                if math.dist(start, b.position) > 0.8:
                    return
            # still stuck: break the blocks in front of us (feet and head), unless they are unsafe
            yaw = math.radians(b.state.yaw)
            fx, fz = round(-math.sin(yaw)), round(math.cos(yaw))
            x, y, z = b.block_pos
            for p in ((x + fx, y + 1, z + fz), (x + fx, y, z + fz)):
                if b.pathfinder.dig_cost(*p):
                    try:
                        b.dig(p)
                    except ActionError:
                        pass
            self.brain.log("awareness", f"got stuck at {b.block_pos}")
        finally:
            self._reflex_busy = False

    def wait_reflexes(self, timeout: float = 180):
        """Block while a reflex (fight/escape/eat) is controlling the bot."""
        end = time.time() + timeout
        while self._reflex_busy and time.time() < end:
            time.sleep(0.2)

    def _take_over(self) -> bool:
        """Interrupt the running action and wait until the bot is free."""
        b = self.bot
        if b.current_action != "idle" or b._action_depth:
            b.stop()
            self.interrupted_at = time.time()
            end = time.time() + 4
            while (b._action_depth or b.current_action != "idle") and time.time() < end:
                time.sleep(0.05)
        return b._action_depth == 0

    def _recover(self, threat):
        """Low health: get out of melee reach, eat, and wait to regenerate."""
        b = self.bot
        scaffold = b._scaffold_item()
        if scaffold and b.inventory.count(scaffold) >= 3 and threat.type not in ("spider", "cave_spider", "creeper"):
            self._note(f"reflex: low health ({b.health:.0f}) - pillaring up away from {threat.type}")
            b.escape_up(3)
        else:
            self._note(f"reflex: low health ({b.health:.0f}) - fleeing from {threat.type}")
            b.flee(threat.pos, 24, timeout=20)
        end = time.time() + 150
        while time.time() < end and self.running and not b.dead:
            if b.food < 20 and any(b.gd.food(i.name) for i in b.inventory.items()):
                try:
                    b.eat()
                except ActionError:
                    pass
            near = b.entities.nearest(b.position, lambda e: e.hostile, 8)
            if b.health >= 16 or (b.health >= 10 and not near):
                break
            if near and b.entity_distance(near) <= 3.0 and near.type != "creeper":
                b.attack(near)
            time.sleep(0.5)

    def _reflexes(self):
        b = self.bot
        if self._reflex_busy:
            return
        idle = b.current_action == "idle"
        hostile = b.entities.nearest(b.position, lambda e: e.hostile and e.type not in ("ender_dragon",), 12)
        t_hurt, attacker_id, _ = b.last_hurt
        attacker = b.entities.get(attacker_id) if time.time() - t_hurt < 3 else None
        if attacker and attacker.type not in ("ender_dragon",) and attacker.distance_to(*b.position) < 8:
            hostile = attacker       # provoked neutral mobs (endermen, piglins, wolves...) count too
        dist = b.entity_distance(hostile) if hostile else 99
        action = None
        if hostile and b.health <= 8 and dist < 8:
            action = ("recover", hostile)
        elif hostile and hostile.type == "creeper" and dist < 5:
            action = ("creeper", hostile)
        elif hostile and dist < 3.5 and hostile.type != "creeper" and (hostile.type != "enderman" or hostile is attacker) \
                and b.health > 8:
            armed = max([b.gd.attack_damage(i.name) for i in b.inventory.items()] + [1.0]) >= 4
            weak = hostile.type in ("silverfish", "endermite", "slime", "magma_cube", "zombie", "husk")
            action = ("fight", hostile) if (armed or (weak and b.health > 14)) else ("evade", hostile)
        elif idle and b.food <= 14 and any(b.gd.food(i.name) for i in b.inventory.items()):
            action = ("eat", None)
        if action:
            self._reflex_busy = True
            try:
                if not self._take_over():
                    return
                kind, target = action
                if kind == "recover":
                    self._recover(target)
                elif kind == "creeper":
                    self._note("reflex: creeper close - backing off")
                    b.flee(target.pos, 10, timeout=6)
                elif kind == "evade":
                    self._note(f"reflex: unarmed vs {target.type} - taking cover")
                    try:
                        if b.time_phase() in ("night", "sunset"):
                            b.hide_underground(max_wait=420)
                        elif b._scaffold_item():
                            b.escape_up(3)
                        else:
                            b.flee(target.pos, 20, timeout=15)
                    except ActionError:
                        b.flee(target.pos, 20, timeout=15)
                elif kind == "fight":
                    self._note(f"reflex: defending against {target.type}")
                    b.fight(target, timeout=25)
                    b.collect_items(5, 4)
                elif kind == "eat":
                    b.eat()
            finally:
                self._reflex_busy = False
            return
        # drowning: hold jump while the head is under water
        head = b.world.get(*map(math.floor, b.eye))
        if head is not None and b.gd.is_water[head] and idle:
            b.controls.jump = True
        elif idle and b.controls.jump and not b.state.in_water:
            b.controls.jump = False
        # idle housekeeping: wear better armor, periodic scan into memory
        if idle and time.time() - self._last_scan > 90:
            res = self.scanner.scan(100)
            self.after_scan(res)
            b.equip_armor()

    # ================================================================== task loop
    def _loop(self):
        while self.running:
            try:
                request = self.tasks.get(timeout=0.5)
            except queue.Empty:
                continue
            task, source = request[:2]
            kind = request[2] if len(request) > 2 else "task"
            if not self.bot.spawned.is_set() and not self.bot.spawned.wait(60):
                self._note("bot offline: could not run " + task[:100])
                continue
            try:
                with self._actuation_lock:
                    if kind == "command":
                        self._chat_command(source, task)
                    elif kind == "goal":
                        from .trainer import Trainer
                        trainer = Trainer(self, 12)
                        self.goal_trainer = trainer
                        self._note("Starting goal: " + task)
                        result = trainer.run_until(task)
                        self._note("Goal result: " + str(result)[:250])
                        self.bot.chat(str(result)[:250])
                    else:
                        self.run_task(task)
            except LLMNotConfigured as e:
                self._note(str(e))
                self.bot.chat("No AI endpoint configured. Chat commands still work: !help")
            except Exception as e:  # noqa: BLE001
                self._note(f"task crashed: {e}")
                self.log.exception("task crashed")
            finally:
                self.tasks.task_done()

    def finish_task(self, summary: str, success: bool):
        self.task_result = (summary, success)
        self.task_done.set()

    def _context_messages(self, task: str) -> list[dict]:
        b = self.bot
        mem = self.brain.recall_text(task, 6, 2200)
        skills = self.brain.skills(task, 6)
        skill_txt = "\n".join(f"- {s['name']}({', '.join(s['params'])}): {s['description']} "
                              f"[ok {s['successes']}/fail {s['failures']}]" for s in skills)
        sys = SYSTEM_PROMPT.format(name=b.username, version=b.gd.version)
        sys += ("\n# Parallel specialist team\n"
                "Interpreter routes player requests; chatter answers during physical tasks; "
                "planner and builder offer suggestions; scout locates loaded ores; escaper "
                "alerts about danger. They only post notes or queue requests. YOU are the "
                "sole physical action worker; never assume a teammate has moved, dug or "
                "placed blocks. Treat urgent escape notes as highest priority, and use "
                "live observations before deciding on risky movements.\n")
        if mem:
            sys += "\n# Relevant memories\n" + mem
        if skill_txt:
            sys += "\n# Learned skills you can run_skill\n" + skill_txt
        ms = self.brain.milestones()
        done = [k for k, v in ms.items() if v["achieved"]]
        if done:
            sys += "\n# Milestones achieved in past sessions: " + ", ".join(done[:30])
        return [{"role": "system", "content": sys},
                {"role": "user", "content": f"TASK: {task}\n\nCurrent status:\n{self.status_text()}"}]

    def run_task(self, task: str, max_steps: int | None = None, time_limit: float = 1800) -> tuple[str, bool]:
        """All LLM-triggered world mutations are serialized, even during training."""
        with self._actuation_lock:
            return self._run_task_impl(task, max_steps, time_limit)

    def _run_task_impl(self, task: str, max_steps: int | None = None,
                       time_limit: float = 1800) -> tuple[str, bool]:
        """Run one task with the LLM in a tool-calling loop. Returns (summary, success)."""
        if not self.llm.configured:
            raise LLMNotConfigured("LLM endpoint not configured (set llm.base_url / llm.api_key in config.json)")
        max_steps = max_steps or self.cfg["llm"].get("max_steps_per_task", 0) or 10 ** 9   # 0 = unlimited
        self.current_task = task
        self.task_done.clear()
        self.task_result = ("", False)
        self.trace = []
        t0 = time.time()
        self._note(f"TASK: {task}")
        self.board.post("worker", f"started task: {task}")
        self.brain.log("task_start", task, 1.5)
        messages = self._context_messages(task)
        schemas = self.tools.schemas()
        step = 0
        consecutive_errors = 0
        try:
            while step < max_steps and not self.task_done.is_set() and self.running:
                if time.time() - t0 > time_limit:
                    self.finish_task("time limit reached", False)
                    break
                step += 1
                notes = self.board.unread()
                if notes:   # what the watcher AI saw since the last decision
                    messages.append({"role": "user", "content": "Teammates report:\n" + "\n".join(notes[-8:])})
                try:
                    reply = self.llm.chat(self._trim(messages), schemas, model_role="logic")
                except LLMNotConfigured:
                    raise
                except Exception as e:  # noqa: BLE001
                    consecutive_errors += 1
                    self._note(f"LLM error: {e}")
                    if consecutive_errors >= 3:
                        self.finish_task(f"LLM unavailable: {e}", False)
                        break
                    time.sleep(5 * consecutive_errors)
                    continue
                consecutive_errors = 0
                if not reply.tool_calls:
                    text = (reply.content or "").strip()
                    messages.append({"role": "assistant", "content": text})
                    if text:
                        self._note(f"LLM: {text[:200]}")
                    messages.append({"role": "user", "content": "Use a tool to make progress, or call task_complete."})
                    continue
                tc_list = reply.tool_calls[:3]
                messages.append({"role": "assistant", "content": reply.content or "",
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                                                for tc in tc_list]})
                for tc in tc_list:
                    self.wait_reflexes()
                    self._note(f"-> {tc.name}({json.dumps(tc.args)[:160]})")
                    t1 = time.time()
                    result = self.tools.call(tc.name, tc.args)
                    ok = not result.startswith("ERROR")
                    self.trace.append({"tool": tc.name, "args": tc.args, "ok": ok, "secs": round(time.time() - t1, 1)})
                    self._note(f"   {result[:220]}")
                    self.brain.log("action", f"{tc.name}({json.dumps(tc.args)[:200]}) -> {result[:400]}",
                                   1.0 if ok else 1.5)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                     "content": result[:3500]})
                    if self.task_done.is_set():
                        break
                # refresh status every few steps so the model does not reason on stale data
                if step % 4 == 0 and not self.task_done.is_set():
                    messages.append({"role": "user", "content": "Status update:\n" + self.status_text()})
            if not self.task_done.is_set():
                self.finish_task("step limit reached", False)
        finally:
            summary, success = self.task_result
            secs = time.time() - t0
            self.brain.log("task_end", f"{task} -> {'SUCCESS' if success else 'FAILED'}: {summary} ({secs:.0f}s)", 2.0)
            self.brain.bump("tasks_succeeded" if success else "tasks_failed")
            self._note(f"TASK {'DONE' if success else 'FAILED'}: {summary}")
            self.board.post("worker", f"{'finished' if success else 'failed'} task: {task} - {summary[:120]}")
            try:
                self._learn_from_task(task, summary, success, secs)
            except Exception as e:  # noqa: BLE001
                self.log.debug("reflection failed: %s", e)
            self.current_task = None
        return self.task_result

    @staticmethod
    def _trim(messages: list[dict], keep_tail: int = 14) -> list[dict]:
        """Keep the system + task messages and the most recent exchanges (tool results shortened)."""
        if len(messages) <= keep_tail + 2:
            return messages
        head = messages[:2]
        tail = messages[-keep_tail:]
        # never start the tail with orphan tool results
        while tail and tail[0]["role"] == "tool":
            tail = tail[1:]
        dropped = len(messages) - len(head) - len(tail)
        return head + [{"role": "user", "content": f"({dropped} earlier messages omitted)"}] + tail

    # ================================================================== learning
    def _learn_from_task(self, task, summary, success, secs):
        good = [t for t in self.trace if t["ok"] and t["tool"] not in (
            "task_complete", "get_status", "inventory", "recall", "remember", "list_skills", "wiki", "recipe",
            "block_info", "list_waypoints", "scan_surroundings", "nearby_entities", "find_blocks", "missing_for",
            "save_skill", "chat", "wait", "detect_structures")]
        # 1) automatic skill from a successful action sequence
        if success and 1 <= len(good) <= 14:
            name = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_")
            steps = [{"tool": t["tool"], "args": t["args"]} for t in good]
            n = self.brain.save_skill(name, f"learned: {task} ({summary[:120]})", steps, [])
            self.brain.skill_result(n, True, secs)
        # 2) reflection with the LLM -> lessons
        if not self.llm.configured:
            return
        trace_txt = "\n".join(f"{'OK ' if t['ok'] else 'ERR'} {t['tool']} {json.dumps(t['args'])[:120]}"
                              for t in self.trace[-30:])
        prompt = (f"You just {'completed' if success else 'FAILED'} the Minecraft task: {task}\n"
                  f"Outcome: {summary}\nActions:\n{trace_txt}\n\n"
                  "Write 1-3 short, specific, reusable lessons for next time (what worked, what to avoid, "
                  "exact numbers/blocks/coordinates if useful). Reply ONLY with JSON: "
                  "{\"lessons\": [{\"topic\": \"...\", \"text\": \"...\"}]}")
        try:
            r = self.llm.chat([{"role": "user", "content": prompt}], None, temperature=0.2, max_tokens=400, model_role="logic")
            m = re.search(r"\{.*\}", r.content or "", re.S)
            data = json.loads(m.group(0)) if m else {}
            for lesson in data.get("lessons", [])[:3]:
                if isinstance(lesson, dict) and lesson.get("text"):
                    self.brain.remember(lesson["text"], lesson.get("topic", "lesson"), 0.75, "reflection")
        except Exception as e:  # noqa: BLE001
            self.log.debug("reflection parse failed: %s", e)

    def run_skill(self, name: str, params: dict) -> str:
        sk = self.brain.skill(name)
        if not sk:
            close = self.brain.skills(name, 3)
            return f"ERROR: no skill {name!r}" + (f"; similar: {[s['name'] for s in close]}" if close else "")
        t0 = time.time()
        out = []

        def fill(v):
            if isinstance(v, str):
                return re.sub(r"\{(\w+)\}", lambda m: str(params.get(m.group(1), m.group(0))), v)
            if isinstance(v, dict):
                return {k: fill(x) for k, x in v.items()}
            if isinstance(v, list):
                return [fill(x) for x in v]
            return v
        for i, st in enumerate(sk["steps"]):
            res = self.tools.call(st["tool"], fill(st.get("args", {})))
            out.append(f"{i + 1}. {st['tool']}: {res[:160]}")
            if res.startswith("ERROR"):
                self.brain.skill_result(sk["name"], False, time.time() - t0)
                return f"skill {name} failed at step {i + 1}:\n" + "\n".join(out)
        self.brain.skill_result(sk["name"], True, time.time() - t0)
        return f"skill {name} succeeded:\n" + "\n".join(out)
