"""LLM client (openai library) for any OpenAI-compatible endpoint (custom base_url set in config).

Supports native tool calling, and falls back to a JSON protocol for servers/models without tools.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

log = logging.getLogger("pyflayer.llm")


_LOCAL_LOCKS: dict = {}
_LOCAL_LAST: dict = {}


class _AllRateLimited(Exception):
    """Every usable model answered 429."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object = None


class LLMNotConfigured(RuntimeError):
    pass


# Provider presets. Every one of these speaks the OpenAI chat-completions API, so the openai library
# works with all of them; "custom" = any other OpenAI-compatible URL (vLLM, llama.cpp, LocalAI, a LAN box...).
# Models are only defaults - set llm.model to anything the provider offers.
PROVIDERS = {
    "custom":     {"base_url": "", "model": "", "key_env": "PYFLAYER_LLM_API_KEY"},
    "openai":     {"base_url": "https://api.openai.com/v1", "model": "gpt-4.1-mini", "key_env": "OPENAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4.1-mini",
                   "key_env": "OPENROUTER_API_KEY",
                   "headers": {"HTTP-Referer": "https://github.com/pyflayer", "X-Title": "PyFlayer Minecraft bot"}},
    "anthropic":  {"base_url": "https://api.anthropic.com/v1/", "model": "claude-sonnet-5-5", "key_env": "ANTHROPIC_API_KEY"},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.5-flash",
                   "key_env": "GEMINI_API_KEY"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.1-8b-instant", "key_env": "GROQ_API_KEY"},
    "deepseek":   {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat", "key_env": "DEEPSEEK_API_KEY"},
    "mistral":    {"base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest", "key_env": "MISTRAL_API_KEY"},
    "together":   {"base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                   "key_env": "TOGETHER_API_KEY"},
    "xai":        {"base_url": "https://api.x.ai/v1", "model": "grok-3-mini", "key_env": "XAI_API_KEY"},
    "litellm":    {"base_url": "http://localhost:4000/v1", "model": "pyflayer", "key_env": "LITELLM_API_KEY",
                   "local": True},     # PyFlayer runs it on this machine; your real keys live in its config
    "ollama":     {"base_url": "http://localhost:11434/v1", "model": "auto", "key_env": "", "local": True},
    "lmstudio":   {"base_url": "http://localhost:1234/v1", "model": "local-model", "key_env": "", "local": True},
}


def split_keys(value) -> list[str]:
    """One provider can have many API keys: a list, or one string with keys separated by commas/spaces."""
    if not value:
        return []
    items = value if isinstance(value, (list, tuple)) else re.split(r"[\s,;]+", str(value))
    out = []
    for k in items:
        k = str(k).strip()
        if k and k not in out:
            out.append(k)
    return out


def keys_for(cfg: dict) -> list[str]:
    import os
    name = (cfg.get("provider") or "custom").lower()
    prov = PROVIDERS.get(name, PROVIDERS["custom"])
    keys = split_keys((cfg.get("keys") or {}).get(name)) or split_keys(cfg.get("api_key"))
    if not keys and prov.get("key_env"):
        keys = split_keys(os.environ.get(prov["key_env"], ""))
    return keys


def resolve(cfg: dict) -> dict:
    """Effective base_url / api_key / model / headers for the configured provider."""
    import os
    prov = PROVIDERS.get((cfg.get("provider") or "custom").lower(), PROVIDERS["custom"])
    base = cfg.get("base_url") or prov["base_url"]
    name = (cfg.get("provider") or "custom").lower()
    # each provider keeps its own key (llm.keys), so switching provider never loses another provider's key
    keys = keys_for(cfg)
    key = keys[0] if keys else ""
    model = cfg.get("model") or prov["model"]
    return {"base_url": base, "api_key": key, "model": model, "headers": prov.get("headers", {}),
            "local": prov.get("local", False)}


JSON_TOOL_INSTRUCTIONS = """
You can call exactly ONE tool per reply. Reply ONLY with a JSON object, no prose, in this form:
{"thought": "<short reasoning>", "tool": "<tool name>", "args": {<arguments>}}
Available tools (name: description | JSON-schema of args):
"""


# free-tier-friendly backups per provider (only used if the primary model fails; filtered by what the key can
# actually use, via the provider's /models list)
BACKUP_MODELS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
             "qwen/qwen3-32b", "gemma2-9b-it"],
    "openrouter": ["meta-llama/llama-3.3-70b-instruct:free", "deepseek/deepseek-chat-v3-0324:free",
                   "google/gemini-2.0-flash-exp:free", "mistralai/mistral-small-3.1-24b-instruct:free"],
    "gemini": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "openai": ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1-nano"],
    "mistral": ["mistral-small-latest", "open-mistral-nemo"],
    "deepseek": ["deepseek-chat"],
    "together": ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"],
}


class LLM:
    def __init__(self, cfg: dict):
        self._set_base(cfg)
        self.client = None
        self.mode = cfg.get("tool_mode", "auto")
        self.calls = 0
        self.tokens = 0
        self.last_error = ""
        self._logic_fallback_until = 0.0
        self.on_wait = None                    # callback(seconds) while waiting out a rate limit
        self.on_status = None                  # callback(text) for setup progress (local AI download ...)
        self.waiting_until = 0.0
        self.cancelled = False
        self.wait_when_rate_limited = cfg.get("wait_when_rate_limited", True)
        self._reset_backups()
        self._maybe_client()

    def _reset_backups(self):
        self._dead: set[str] = set()          # models this key can't use (404/403)
        self._cool: dict[str, float] = {}     # rate-limited models -> usable again at
        self._available: set[str] | None = None
        self._available_at = 0.0
        self.active_backup: dict[str, str] = {}
        self.last_model = ""

    def _maybe_client(self):
        r = resolve(self.cfg)
        if r["base_url"] and (r["api_key"] or r["local"] or (self.cfg.get("provider") or "custom") == "custom"):
            self._make_client()

    @property
    def configured(self) -> bool:
        return self.client is not None

    def _make_client(self):
        from openai import OpenAI
        r = resolve(self.cfg)
        self.model = r["model"] or "gpt-4.1-mini"
        kw = {"api_key": r["api_key"] or "not-needed", "timeout": float(self.cfg.get("timeout", 120)),
              "max_retries": 1, "base_url": r["base_url"]}   # switch to a backup model fast
        if r["headers"]:
            kw["default_headers"] = r["headers"]
        self.client = OpenAI(**kw)

    # ------------------------------------------------------------------ API slots (keys / providers that take turns)
    def _set_base(self, cfg: dict):
        """cfg = the settings as saved. The LLM works on one 'slot' of it at a time: one provider + one key.
        Slots come from llm.pool (Settings > AI APIs tabs: API 1, API 2 ...), or else from the provider's keys."""
        self.base_cfg = cfg
        slots = []
        for e in cfg.get("pool") or []:
            if not e or not e.get("provider"):
                continue
            ks = split_keys(e.get("key")) or [""]
            for k in ks:
                slots.append({"provider": e["provider"].lower(), "key": k, "model": e.get("model", ""),
                              "base_url": e.get("base_url", "")})
        if not slots:
            ks = keys_for(cfg) or [""]
            slots = [{"provider": (cfg.get("provider") or "custom").lower(), "key": k, "model": cfg.get("model", ""),
                      "base_url": cfg.get("base_url", "")} for k in ks]
        self.slots = slots
        self._slot_i = 0
        self._slot_cool: dict[int, float] = {}
        self._slot_cools: dict[int, dict] = {}
        self.cfg = self._slot_cfg(0)

    def _slot_cfg(self, i: int) -> dict:
        s = self.slots[i]
        c = dict(self.base_cfg)
        c.update(provider=s["provider"], model=s["model"], base_url=s["base_url"], api_key=s["key"])
        if s["key"]:
            c["keys"] = dict(self.base_cfg.get("keys") or {}, **{s["provider"]: s["key"]})
        return c

    @property
    def slot_label(self) -> str:
        s = self.slots[self._slot_i]
        return f"API {self._slot_i + 1}/{len(self.slots)} ({s['provider']})"

    def reconfigure(self, cfg: dict):
        self._set_base(cfg)
        self._local_ready = False
        self.mode = cfg.get("tool_mode", "auto")
        self.client = None
        self._logic_fallback_until = 0.0
        self.on_wait = None                    # callback(seconds) while waiting out a rate limit
        self.waiting_until = 0.0
        self.cancelled = False
        self.wait_when_rate_limited = cfg.get("wait_when_rate_limited", True)
        self._reset_backups()
        self._maybe_client()

    def test(self) -> str:
        """Small round trip to check the endpoint, key and model. Returns a human readable result."""
        if not self.client:
            return "not configured: choose a provider and enter an API key (or a base URL for custom/local)"
        try:
            t = time.time()
            r = self.chat([{"role": "user", "content": "Reply with the single word: ready"}], None, 0.0, 10, model_role="chat")
            used = self.last_model or self.model
            backups = f"; backups in use: {self.active_backup}" if self.active_backup else ""
            return f"OK ({time.time() - t:.1f}s, model {used}): {r.content.strip()[:60]!r}{backups}"
        except Exception as e:  # noqa: BLE001
            return f"FAILED ({self.model} @ {resolve(self.cfg)['base_url']}): {e}"

    # ------------------------------------------------------------------ chat
    def model_for(self, role: str = "logic") -> str:
        """Return the configured Groq specialist model, or the normal provider model."""
        if (self.cfg.get("provider") or "").lower() != "groq":
            return self.model
        defaults = {"chat": "llama-3.1-8b-instant",
                    "world": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "logic": "openai/gpt-oss-120b", "fallback": "qwen/qwen3.8-27b"}
        models = self.cfg.get("models") or {}
        role = role if role in defaults else "logic"
        if role == "logic" and time.time() < self._logic_fallback_until:
            role = "fallback"
        return str(models.get(role) or defaults[role])

    def _next_utc_midnight(self) -> float:
        now = datetime.now(timezone.utc)
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float | None = None,
             max_tokens: int | None = None, model_role: str = "logic") -> LLMReply:
        if not self.client:
            raise LLMNotConfigured("AI not configured: set llm.provider (openai, openrouter, ...) and llm.api_key, "
                                   "or llm.base_url for a custom endpoint")
        temperature = self.cfg.get("temperature", 0.3) if temperature is None else temperature
        max_tokens = max_tokens or self.cfg.get("max_tokens", 900)
        model = self.model_for(model_role)
        if tools and self.mode in ("auto", "native"):
            try:
                return self._native(messages, tools, temperature, max_tokens, model)
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if self.mode == "auto" and any(k in msg for k in ("tool", "function", "not supported", "400",
                                                                    "unsupported", "invalid")):
                    log.warning("endpoint rejected native tool calls (%s); switching to JSON tool mode", e)
                    self.mode = "json"
                else:
                    self.last_error = str(e)
                    raise
        if tools:
            return self._json_mode(messages, tools, temperature, max_tokens, model)
        return self._plain(messages, temperature, max_tokens, model)

    # ------------------------------------------------------------------ model backups
    def available_models(self) -> set[str] | None:
        """Model ids this key may use (from the provider's /models list), cached for an hour. None = unknown."""
        if self._available is not None and time.time() - self._available_at < 3600:
            return self._available
        try:
            ids = {m.id for m in self.client.models.list()}
            skip = ("whisper", "tts", "guard", "embed", "playai", "orpheus", "distil", "moderation", "image", "audio")
            self._available = {i for i in ids if not any(k in i.lower() for k in skip)} or None
        except Exception as e:  # noqa: BLE001
            log.debug("could not list models: %s", e)
            self._available = None
        self._available_at = time.time()
        return self._available

    def _chain(self, first: str) -> list[str]:
        """The model to try first, then backups: the user's list, the other configured models, the provider's
        free-tier-friendly defaults, and finally anything the key is allowed to use."""
        prov = (self.cfg.get("provider") or "custom").lower()
        order = [first, *(self.cfg.get("backup_models") or []), *BACKUP_MODELS.get(prov, []),
                 *((self.cfg.get("models") or {}).values() if prov == "groq" else ()), self.model]     # strongest free backups first
        now = time.time()
        chain = []
        for m in order:
            if m and m not in chain and m not in self._dead and self._cool.get(m, 0) <= now:
                chain.append(m)
        avail = self.available_models()
        if avail:
            known = [m for m in chain if m in avail]
            extra = sorted((m for m in avail if m not in known),
                           key=lambda m: (0 if any(k in m for k in ("llama", "gpt-oss", "qwen", "gemma", "mistral",
                                                                    "deepseek", "gpt-4", "claude", "gemini")) else 1, m))
            chain = known + [m for m in extra if m not in self._dead and self._cool.get(m, 0) <= now][:6]
        return chain or [first]

    def _ensure_local(self):
        """Ollama: install / start it, choose and download the model the first time it's used (no setup)."""
        if getattr(self, "_local_ready", False) or (self.cfg.get("provider") or "").lower() != "ollama":
            return
        from . import ollama_auto
        say = self.on_status or (lambda m: log.info("%s", m))
        self.model = ollama_auto.ensure(resolve(self.cfg)["base_url"], self.model, progress=say)
        self._local_ready = True

    def _create(self, model=None, **kw):
        """Call the first working model; if every model is rate-limited, wait (like a player would) and retry."""
        from . import ollama_auto
        self._ensure_local()
        first = model or self.model
        if (self.cfg.get("provider") or "").lower() == "ollama" and ollama_auto.is_auto(first):
            first = self.model
        while True:
            try:
                return self._create_once(first, **kw)
            except _AllRateLimited:
                if self._rotate_key():
                    continue                       # next API (key / provider): no waiting
                if getattr(self, "_all_bad", False) and all(
                        t - time.time() > 600 for i, t in self._slot_cool.items() if i != self._slot_i):
                    raise RuntimeError("every API key was rejected (401) - check the keys in Settings") from None
                r = self._emergency(kw, "every model is rate-limited")
                if r is not None:
                    return r
                if not getattr(self, "wait_when_rate_limited", True) or getattr(self, "cancelled", False):
                    raise RuntimeError("all AI models are rate-limited right now") from None
                now = time.time()
                soonest = min((v for v in self._cool.values() if v > now), default=now + 30)
                wait = max(5.0, min(soonest - now, 300.0))
                self.waiting_until = now + wait
                log.warning("AI rate-limited on every model; waiting %ds before trying again", int(wait))
                if self.on_wait:
                    try:
                        self.on_wait(int(wait))
                    except Exception:  # noqa: BLE001
                        pass
                end = now + wait
                while time.time() < end and not getattr(self, "cancelled", False):
                    time.sleep(1)
                self.waiting_until = 0
                # short limits are over; daily limits get another chance (some providers reset early)
                for m, v in list(self._cool.items()):
                    if v <= time.time() + 1 or wait >= 300:
                        self._cool.pop(m, None)

    def _create_once(self, first, **kw):
        t = time.time()
        last_exc = None
        limited = False
        chain = self._chain(first)
        if not chain and self._cool:
            raise _AllRateLimited()
        for selected in chain:
            try:
                r = self._call(selected, kw)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                text = str(exc).lower()
                if status in (404, 403) or (status == 400 and any(k in text for k in (
                        "model", "decommissioned", "does not exist", "not found", "no access"))):
                    self._dead.add(selected)                         # not on this plan / removed: never retry
                    log.warning("AI model %s is not available on this key (%s); using a backup model", selected,
                                status)
                    continue
                if status == 401:
                    # this key is wrong / revoked: use the next API instead of stopping
                    self._bad_slot = True
                    if len(self.slots) > 1:
                        raise _AllRateLimited() from exc
                if status == 429:
                    daily = any(k in text for k in ("per day", "daily", "tokens per day", "requests per day", "tpd", "rpd"))
                    wait = self._retry_after(exc)
                    self._cool[selected] = self._next_utc_midnight() if daily else time.time() + (wait or 60)
                    log.warning("AI model %s is rate-limited (%s); using a backup model", selected,
                                "until tomorrow" if daily else f"for {int(wait or 60)}s")
                    limited = True
                    continue
                if status is None or status >= 500:
                    # provider down / unreachable: try the emergency fallback AIs before giving up
                    r = self._emergency(kw, f"{selected} failed: {str(exc)[:120]}")
                    if r is not None:
                        return r
                self.last_error = str(exc)
                raise
            if selected != first:
                self.active_backup[first] = selected
            self.last_model = selected
            self.calls += 1
            try:
                self.tokens += r.usage.total_tokens
            except AttributeError:
                pass
            log.debug("llm call %s %.1fs", selected, time.time() - t)
            return r
        if limited or (not chain and self._cool):
            raise _AllRateLimited()
        self.last_error = f"all models failed: {last_exc}"
        raise RuntimeError(f"no AI model on this key worked (last error: {last_exc}). Check your plan/key, or set "
                           f"llm.backup_models in config.json")

    def _rotate_key(self) -> bool:
        """Several APIs (keys and/or providers): when one is rate-limited or rejects the key, carry on with the
        next one that's free - immediately, no waiting."""
        bad = getattr(self, "_bad_slot", False)
        self._bad_slot = False
        self._all_bad = bad
        if len(self.slots) < 2:
            return False
        now = time.time()
        cur = self._slot_i
        self._slot_cool[cur] = now + 3600 if bad else \
            min((v for v in self._cool.values() if v > now), default=now + 60)
        self._slot_cools[cur] = dict(self._cool)          # remember this API's own model limits
        for step in range(1, len(self.slots)):
            nxt = (cur + step) % len(self.slots)
            if self._slot_cool.get(nxt, 0) > now:
                continue
            old = self.slot_label
            self._slot_i = nxt
            self.cfg = self._slot_cfg(nxt)
            self._cool = {m: t for m, t in self._slot_cools.get(nxt, {}).items() if t > now}
            self._available = None                       # other provider, other model list
            self._local_ready = False
            self.client = None
            self._maybe_client()
            if not self.client:
                self._slot_cool[nxt] = now + 3600        # not usable (no key / no URL)
                continue
            why = "rejected the key" if bad else "is rate-limited"
            log.warning("%s %s; switching to %s", old, why, self.slot_label)
            if self.on_status:
                try:
                    self.on_status(f"{old} {why} - switched to {self.slot_label}")
                except Exception:  # noqa: BLE001
                    pass
            self._all_bad = False
            return True
        return False

    def _call(self, model, kw):
        lim = self.cfg.get("local_limits")
        if not lim:
            return self.client.chat.completions.create(model=model, **kw)
        # local AI: never run two requests at once on the same machine, and leave gaps so the CPU/GPU can cool
        lock = _LOCAL_LOCKS.setdefault(resolve(self.cfg)["base_url"], threading.Lock())
        with lock:
            gap = float(lim.get("min_interval_s", 1.0)) - (time.time() - _LOCAL_LAST.get(id(lock), 0))
            if gap > 0:
                time.sleep(gap)
            kw = dict(kw)
            kw["max_tokens"] = min(int(kw.get("max_tokens") or 400), int(lim.get("max_tokens", 300)))
            try:
                return self.client.chat.completions.create(model=model, **kw)
            finally:
                _LOCAL_LAST[id(lock)] = time.time()

    def _emergency(self, kw, why):
        """Emergency fallback: other AIs (another provider with a saved key, a LiteLLM proxy or a local model)
        answer while the main one is rate-limited or down - so a live stream never stalls on an API limit."""
        fbs = [f for f in (self.cfg.get("emergency_fallbacks") or []) if f]
        if not fbs:
            return None
        self._fb = getattr(self, "_fb", {})
        for fb in fbs:
            fb = {"provider": fb} if isinstance(fb, str) else dict(fb)
            name = f"{fb.get('provider')}:{fb.get('model', '')}"
            if fb.get("provider") == (self.cfg.get("provider") or "") and not fb.get("model"):
                continue
            llm = self._fb.get(name)
            if llm is None:
                cfg = bot_llm_config(self.cfg, fb)
                cfg["emergency_fallbacks"] = []          # no fallback loops
                cfg["wait_when_rate_limited"] = False
                llm = self._fb[name] = LLM(cfg)
            if not llm.configured:
                continue
            try:
                kw2 = {k: v for k, v in kw.items() if k != "model"}
                r = llm._create_once(llm.model, **kw2)
            except Exception as e:  # noqa: BLE001
                log.warning("emergency fallback %s failed: %s", name, str(e)[:160])
                continue
            self.last_model = f"{name} (emergency)"
            self.active_backup["emergency"] = name
            log.warning("AI: %s - answered by emergency fallback %s", why, name)
            if self.on_wait:
                try:
                    self.on_wait(0)
                except Exception:  # noqa: BLE001
                    pass
            return r
        return None

    @staticmethod
    def _retry_after(exc) -> float | None:
        try:
            v = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
            return float(v) if v else None
        except (TypeError, ValueError):
            return None

    def _plain(self, messages, temperature, max_tokens, model) -> LLMReply:
        r = self._create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return LLMReply(content=r.choices[0].message.content or "", raw=r)

    def _native(self, messages, tools, temperature, max_tokens, model) -> LLMReply:
        r = self._create(model=model, messages=messages, tools=tools, tool_choice="auto", temperature=temperature,
                         max_tokens=max_tokens)
        m = r.choices[0].message
        calls = []
        for tc in m.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {}
            calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {}))
        if not calls and m.content:
            # some models still answer with JSON in the content
            parsed = self._parse_json_call(m.content)
            if parsed:
                calls.append(parsed)
        return LLMReply(content=m.content or "", tool_calls=calls, raw=r)

    def _json_mode(self, messages, tools, temperature, max_tokens, model) -> LLMReply:
        desc = "\n".join(f"- {t['function']['name']}: {t['function']['description']} | "
                         f"{json.dumps(t['function']['parameters'].get('properties', {}))}" for t in tools)
        sys_extra = JSON_TOOL_INSTRUCTIONS + desc
        msgs = []
        for m in messages:
            if m["role"] == "tool":
                msgs.append({"role": "user", "content": f"Tool result ({m.get('name', 'tool')}):\n{m['content']}"})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                tc = m["tool_calls"][0]["function"]
                msgs.append({"role": "assistant", "content": json.dumps(
                    {"tool": tc["name"], "args": json.loads(tc["arguments"] or "{}")})})
            else:
                msgs.append({"role": m["role"], "content": m.get("content") or ""})
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n" + sys_extra}
        else:
            msgs.insert(0, {"role": "system", "content": sys_extra})
        r = self._create(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        content = r.choices[0].message.content or ""
        call = self._parse_json_call(content)
        return LLMReply(content=content, tool_calls=[call] if call else [], raw=r)

    @staticmethod
    def _parse_json_call(text: str) -> ToolCall | None:
        text = text.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except ValueError:
                        return None
                    name = obj.get("tool") or obj.get("name") or obj.get("function")
                    args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    if name:
                        return ToolCall(f"json_{int(time.time() * 1000)}", str(name), args if isinstance(args, dict) else {})
                    return None
        return None


def bot_llm_config(llm_cfg: dict, override: dict | None) -> dict:
    """The AI settings one bot uses: the default ones, or its own provider / model / URL if it has them."""
    cfg = dict(llm_cfg)
    if not override or not override.get("provider"):
        return cfg
    prov = override["provider"].lower()
    cfg["provider"] = prov
    cfg["model"] = override.get("model", "")
    cfg["base_url"] = override.get("base_url", "")
    cfg["pool"] = []                                      # a bot's own AI doesn't take turns with the main APIs
    cfg["api_key"] = ""                                   # never send one provider's key to another:
    # keys_for() picks this provider's own saved key(s) from llm.keys
    return cfg
