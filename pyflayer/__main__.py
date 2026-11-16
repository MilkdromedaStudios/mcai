"""Command line interface.

  python -m pyflayer setup                 download server + Java + game data, write config.json
  python -m pyflayer gui                   desktop app (Tkinter)
  python -m pyflayer run                   server + bot + AI agent with an interactive console
  python -m pyflayer train [--hours 7]     overnight training (headless)
  python -m pyflayer server                only run the Minecraft server
  python -m pyflayer config KEY=VALUE ...  e.g. llm.base_url=http://host:8000/v1 llm.model=my-model
  python -m pyflayer wiki-sync             pre-download wiki summaries for every block/item/mob
  python -m pyflayer brain [--export F | --import F]   brain statistics / portable model file
  python -m pyflayer selftest              connect, scan, and craft a wooden pickaxe
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time

from .config import Config, data_dir


def _parse_value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def cmd_setup(rt, args):
    rt.prepare()
    rt.cfg.save()
    print(f"Ready: Minecraft {rt.gd.version} (protocol {rt.gd.protocol}); config at {rt.cfg.path}")
    from .agent.llm import LLM
    if not LLM(rt.cfg["llm"]).configured:
        print("Tip: connect an AI:  pyflayer ai use openrouter --key sk-or-...   (see: pyflayer ai providers)")


def cmd_config(rt, args):
    for kv in args.pairs:
        if "=" not in kv:
            print(f"{kv} = {json.dumps(rt.cfg.get(kv))}")
            continue
        k, v = kv.split("=", 1)
        rt.cfg.set(k, _parse_value(v))
        print(f"set {k} = {json.dumps(rt.cfg.get(k))}")
    rt.cfg.save()
    if not args.pairs:
        print(json.dumps(rt.cfg.data, indent=2))


def cmd_server(rt, args):
    srv = rt.start_server()
    if not srv:
        print("server.auto_start is false or host is remote; nothing to run")
        return
    print("\n".join(rt.join_info()))
    srv.listeners.append(lambda line: print(line))
    print("Server running. Type server commands (e.g. 'op Steve'), 'stop' to quit.")
    try:
        for line in sys.stdin:
            if line.strip() == "stop":
                break
            srv.command(line.strip())
    except KeyboardInterrupt:
        pass
    srv.stop()


def _console(rt):
    agent = rt.agent
    bot = rt.bot
    print("\nPyFlayer console. Type a task for the AI (e.g. 'get an iron pickaxe'), '!cmd' for direct commands "
          "(!help), 'status', 'scan', 'brain', or 'quit'.")
    while True:
        try:
            line = input("pyflayer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        if line == "status":
            print(agent.status_text())
        elif line == "scan":
            from .bot.scanner import Scanner
            res = agent.scanner.scan(100)
            agent.after_scan(res)
            print(Scanner.summarize(res, 4000))
        elif line == "brain":
            print(json.dumps(rt.brain.overview(), indent=1))
        elif line.startswith("!"):
            agent._chat_command("console", line)
        elif line.startswith("/"):
            bot.chat(line)
        elif rt.llm.configured:
            agent.submit(line, "console")
        else:
            print("No AI configured; use !commands (e.g. !obtain iron_pickaxe) or: pyflayer ai use <provider> --key KEY")


def cmd_run(rt, args):
    if args.all:
        rt.start_all()
    else:
        rt.start_agent()
    print("\n".join(rt.join_info()))
    try:
        _console(rt)
    finally:
        rt.shutdown()


def cmd_train(rt, args):
    if args.all:
        rt.start_all(training=True, hours=args.hours)
    tr = rt.trainer or rt.start_training(args.hours, background=True)
    print(f"Training for {tr.hours} h. Report: {tr.report_dir}. Ctrl+C to stop early.")
    try:
        while tr.running:
            time.sleep(30)
            p = tr.progress()
            done = sum(m["achieved"] for m in p["milestones"])
            print(f"[{p['elapsed_h']:.2f}h] {p['status']} | milestones {done}/{len(p['milestones'])} | "
                  f"knowledge wiki {p['knowledge']['entries_with_wiki']} | seen blocks "
                  f"{p['knowledge']['block_types_seen_in_world']}", flush=True)
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        rt.shutdown()


def cmd_wiki(rt, args):
    from .agent.trainer import KnowledgeBuilder
    rt.prepare()
    rt.open_brain()
    kb = KnowledgeBuilder(rt.gd, rt.brain, rt.wiki, print)
    kb.build_gamedata()
    kb.crawl_wiki()
    print(f"wiki pages cached: {rt.wiki.count()}; knowledge entries: {rt.brain.knowledge_count()}")


def cmd_brain(rt, args):
    rt.open_brain()
    if args.export:
        print("exported to", rt.brain.export(args.export))
    elif args.import_:
        rt.brain.import_(args.import_)
        print("imported", args.import_)
    print(json.dumps(rt.brain.overview(), indent=1))
    ms = rt.brain.milestones()
    if ms:
        print("milestones:", ", ".join(f"{k}{'✓' if v['achieved'] else '✗'}" for k, v in ms.items()))


def cmd_selftest(rt, args):
    rt.start_agent()
    b = rt.bot
    try:
        print("status:", rt.agent.status_text(short=True))
        from .bot.scanner import Scanner
        res = rt.agent.scanner.scan(100)
        print(Scanner.summarize(res, 3000))
        t = time.time()
        print(b.obtain("wooden_pickaxe", 1, timeout=600), f"({time.time() - t:.0f}s)")
        print("inventory:", b.inventory.summary())
    finally:
        rt.shutdown()


def cmd_ai(rt, args):
    from .agent.llm import LLM, PROVIDERS, resolve
    if args.action == "providers":
        for name, p in PROVIDERS.items():
            print(f"  {name:<11} {p['base_url'] or '(your URL)':<58} default model: {p['model'] or '-'}"
                  f"{'   key env: ' + p['key_env'] if p.get('key_env') else ''}")
        return
    if args.action == "use":
        if not args.provider or args.provider not in PROVIDERS:
            print("choose one of:", ", ".join(PROVIDERS))
            return
        rt.cfg.set("llm.provider", args.provider)
        rt.cfg.set("llm.base_url", args.url or "")
        rt.cfg.set("llm.model", args.model or "")
        if args.key is not None:
            rt.cfg.set("llm.api_key", args.key)
        rt.cfg.save()
    r = resolve(rt.cfg["llm"])
    print(f"provider {rt.cfg['llm'].get('provider')} | url {r['base_url'] or '-'} | model {r['model'] or '-'} | "
          f"key {'set' if r['api_key'] else 'missing'}")
    print("test:", LLM(rt.cfg["llm"]).test())


def cmd_bots(rt, args):
    if args.action == "add" and args.name:
        print("added", rt.add_profile(args.name))
    elif args.action == "remove" and args.name:
        rt.cfg.set("profiles", [p for p in rt.profiles() if p != args.name] or ["PyBot"])
        rt.cfg.save()
    print("bot profiles (max 5):", ", ".join(rt.profiles()))


def cmd_gui(rt, args):
    from .gui.app import run_gui
    run_gui(rt)


def main(argv=None):
    from .runtime import Runtime, setup_logging
    p = argparse.ArgumentParser(prog="pyflayer", description="Minecraft 26.x AI bot")
    p.add_argument("--config", default=None, help="path to config.json")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--bot", default=None, help="bot profile name to use (see: pyflayer bots)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("setup")
    sub.add_parser("gui")
    r = sub.add_parser("run")
    r.add_argument("--all", action="store_true", help="start every bot profile")
    ai = sub.add_parser("ai", help="connect an AI provider: pyflayer ai use openrouter --key KEY [--model M]")
    ai.add_argument("action", nargs="?", choices=["providers", "use", "test"], default="test")
    ai.add_argument("provider", nargs="?")
    ai.add_argument("--key", default=None)
    ai.add_argument("--model", default=None)
    ai.add_argument("--url", default=None, help="base URL (custom provider or override)")
    bp = sub.add_parser("bots")
    bp.add_argument("action", nargs="?", choices=["list", "add", "remove"], default="list")
    bp.add_argument("name", nargs="?")
    sub.add_parser("server")
    t = sub.add_parser("train")
    t.add_argument("--hours", type=float, default=None)
    t.add_argument("--all", action="store_true", help="train every bot profile at once")
    c = sub.add_parser("config")
    c.add_argument("pairs", nargs="*")
    sub.add_parser("wiki-sync")
    br = sub.add_parser("brain")
    br.add_argument("--export", default=None)
    br.add_argument("--import", dest="import_", default=None)
    sub.add_parser("selftest")
    args = p.parse_args(argv)
    cfg = Config(args.config) if args.config else Config()
    setup_logging(logging.DEBUG if args.verbose else (logging.WARNING if args.cmd in ("run",) else logging.INFO),
                  console=args.cmd not in ("gui",))
    rt = Runtime(cfg)
    if args.bot:
        rt.current = rt.add_profile(args.bot)
    handlers = {"setup": cmd_setup, "config": cmd_config, "server": cmd_server, "run": cmd_run,
                "train": cmd_train, "wiki-sync": cmd_wiki, "brain": cmd_brain, "selftest": cmd_selftest,
                "gui": cmd_gui, "bots": cmd_bots, "ai": cmd_ai, None: cmd_gui}
    handlers[args.cmd](rt, args)


if __name__ == "__main__":
    main()
