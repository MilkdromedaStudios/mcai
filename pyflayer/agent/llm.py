"""LLM client (openai library) for any OpenAI-compatible endpoint (custom base_url set in config).

Supports native tool calling, and falls back to a JSON protocol for servers/models without tools.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

log = logging.getLogger("pyflayer.llm")


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
    "ollama":     {"base_url": "http://localhost:11434/v1", "model": "llama3.1", "key_env": "", "local": True},
    "lmstudio":   {"base_url": "http://localhost:1234/v1", "model": "local-model", "key_env": "", "local": True},
}


def resolve(cfg: dict) -> dict:
    """Effective base_url / api_key / model / headers for the configured provider."""
    import os
    prov = PROVIDERS.get((cfg.get("provider") or "custom").lower(), PROVIDERS["custom"])
    base = cfg.get("base_url") or prov["base_url"]
    key = cfg.get("api_key") or (os.environ.get(prov["key_env"], "") if prov.get("key_env") else "")
    model = cfg.get("model") or prov["model"]
    return {"base_url": base, "api_key": key, "model": model, "headers": prov.get("headers", {}),
            "local": prov.get("local", False)}


JSON_TOOL_INSTRUCTIONS = """
You can call exactly ONE tool per reply. Reply ONLY with a JSON object, no prose, in this form:
{"thought": "<short reasoning>", "tool": "<tool name>", "args": {<arguments>}}
Available tools (name: description | JSON-schema of args):
"""


class LLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = None
        self.mode = cfg.get("tool_mode", "auto")
        self.calls = 0
        self.tokens = 0
        self.last_error = ""
        self._logic_fallback_until = 0.0
        self._maybe_client()

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
              "max_retries": 2, "base_url": r["base_url"]}
        if r["headers"]:
            kw["default_headers"] = r["headers"]
        self.client = OpenAI(**kw)

    def reconfigure(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg.get("tool_mode", "auto")
        self.client = None
        self._maybe_client()

    def test(self) -> str:
        """Small round trip to check the endpoint, key and model. Returns a human readable result."""
        if not self.client:
            return "not configured: choose a provider and enter an API key (or a base URL for custom/local)"
        try:
            t = time.time()
            r = self.chat([{"role": "user", "content": "Reply with the single word: ready"}], None, 0.0, 10, model_role="chat")
            return f"OK ({time.time() - t:.1f}s, model {self.model}): {r.content.strip()[:60]!r}"
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

    def _create(self, model=None, **kw):
        t = time.time()
        selected = model or self.model
        try:
            r = self.client.chat.completions.create(model=selected, **kw)
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            logic_model = str((self.cfg.get("models") or {}).get("logic") or "openai/gpt-oss-120b")
            if (self.cfg.get("provider") or "").lower() == "groq" and selected == logic_model and status == 429:
                self._logic_fallback_until = self._next_utc_midnight()
                fallback = str((self.cfg.get("models") or {}).get("fallback") or "qwen/qwen3.8-27b")
                log.warning("Groq logic model rate-limited; switching to %s until UTC midnight", fallback)
                r = self.client.chat.completions.create(model=fallback, **kw)
            else:
                self.last_error = str(exc)
                raise
        self.calls += 1
        try:
            self.tokens += r.usage.total_tokens
        except AttributeError:
            pass
        log.debug("llm call %.1fs", time.time() - t)
        return r

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
