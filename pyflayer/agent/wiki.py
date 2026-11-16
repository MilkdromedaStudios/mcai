"""Minecraft Wiki (minecraft.wiki) client with an offline full-text cache.

Pages are fetched through the MediaWiki API (TextExtracts, plain text), cached in SQLite and
searchable offline. The overnight trainer pre-fetches summaries of every block/item/mob/biome.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("pyflayer.wiki")
API = "https://minecraft.wiki/api.php"
USER_AGENT = "pyflayer-bot/1.0 (Minecraft AI bot; low-rate cached fetches)"


def title_from_id(name: str) -> str:
    """minecraft id -> likely wiki title ('deepslate_iron_ore' -> 'Deepslate Iron Ore')."""
    name = name.split(":")[-1]
    return " ".join(w.capitalize() if w not in ("of", "and", "the", "on", "a") else w for w in name.split("_"))


class Wiki:
    def __init__(self, path: Path, online: bool = True, rate_per_sec: float = 1.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()
        self.online = online
        self.min_interval = 1.0 / max(0.1, rate_per_sec)
        self._last = 0.0
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS pages(title TEXT PRIMARY KEY, summary TEXT, full TEXT, fetched REAL);
            CREATE TABLE IF NOT EXISTS aliases(alias TEXT PRIMARY KEY, title TEXT);
        """)
        try:
            self.db.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(title, body, tokenize='porter unicode61');
            """)
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        self.db.commit()

    # ------------------------------------------------------------------ http
    def _api(self, params: dict, timeout: float = 30) -> dict:
        if not self.online:
            raise RuntimeError("wiki is offline")
        wait = self._last + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        params = {**params, "format": "json", "formatversion": "2"}
        url = API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(3):
            try:
                self._last = time.time()
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                log.debug("wiki request failed (%s), retrying", e)
                time.sleep(2 * (attempt + 1))
        return {}

    # ------------------------------------------------------------------ cache
    def _store(self, title: str, summary: str | None = None, full: str | None = None):
        with self.lock:
            row = self.db.execute("SELECT summary, full FROM pages WHERE title=?", (title,)).fetchone()
            s = summary if summary is not None else (row[0] if row else "")
            f = full if full is not None else (row[1] if row else None)
            self.db.execute("INSERT OR REPLACE INTO pages VALUES (?,?,?,?)", (title, s, f, time.time()))
            if self.fts:
                self.db.execute("DELETE FROM pages_fts WHERE title=?", (title,))
                self.db.execute("INSERT INTO pages_fts(title, body) VALUES (?,?)", (title, (f or s or "")[:20000]))
            self.db.commit()

    def _alias(self, alias: str, title: str):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO aliases VALUES (?,?)", (alias.lower(), title))

    def cached(self, title: str) -> tuple[str, str | None] | None:
        with self.lock:
            t = self.db.execute("SELECT title FROM aliases WHERE alias=?", (title.lower(),)).fetchone()
            t = t[0] if t else title
            row = self.db.execute("SELECT summary, full FROM pages WHERE title=? COLLATE NOCASE", (t,)).fetchone()
        return row

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

    # ------------------------------------------------------------------ public api
    def summaries(self, titles: list[str]) -> dict[str, str]:
        """Fetch intro summaries for up to 20 titles at once (follows redirects). alias -> summary."""
        out = {}
        todo = []
        for t in titles:
            c = self.cached(t)
            if c and c[0]:
                out[t] = c[0]
            else:
                todo.append(t)
        for i in range(0, len(todo), 20):
            batch = todo[i:i + 20]
            try:
                d = self._api({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                               "exlimit": 20, "redirects": 1, "titles": "|".join(batch)})
            except Exception as e:  # noqa: BLE001
                log.warning("wiki fetch failed: %s", e)
                continue
            q = d.get("query", {})
            redirect = {r["from"]: r["to"] for r in q.get("redirects", [])}
            norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
            pages = {p["title"]: p.get("extract", "") for p in q.get("pages", []) if not p.get("missing")}
            for t in batch:
                tt = norm.get(t, t)
                tt = redirect.get(tt, tt)
                if tt in pages and pages[tt]:
                    self._store(tt, summary=pages[tt].strip())
                    self._alias(t, tt)
                    out[t] = pages[tt].strip()
        return out

    def page(self, title: str, max_chars: int = 6000) -> str:
        c = self.cached(title)
        if c and c[1]:
            return c[1][:max_chars]
        try:
            d = self._api({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title})
            pages = d.get("query", {}).get("pages", [])
            if not pages or pages[0].get("missing"):
                hits = self.search(title, 1)
                if hits and hits[0][0].lower() != title.lower():
                    return self.page(hits[0][0], max_chars)
                return f"No wiki page for {title!r}."
            p = pages[0]
            text = re.sub(r"\n{3,}", "\n\n", p.get("extract", ""))
            self._store(p["title"], full=text, summary=text.split("\n\n")[0][:1500])
            self._alias(title, p["title"])
            return f"[{p['title']}]\n{text[:max_chars]}"
        except Exception as e:  # noqa: BLE001
            if c and c[0]:
                return c[0][:max_chars]
            return f"wiki unavailable: {e}"

    def search(self, query: str, k: int = 5) -> list[tuple[str, str]]:
        """(title, snippet). Offline FTS first, then the online search API."""
        res: list[tuple[str, str]] = []
        if self.fts:
            words = [w for w in re.findall(r"[a-zA-Z0-9]{3,}", query)][:8]
            if words:
                with self.lock:
                    try:
                        rows = self.db.execute("SELECT title FROM pages_fts WHERE pages_fts MATCH ? "
                                               "ORDER BY bm25(pages_fts) LIMIT ?",
                                               (" OR ".join(f'"{w}"*' for w in words), k)).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
                seen = set()
                for (t,) in rows:
                    c = self.cached(t)
                    if t not in seen:
                        seen.add(t)
                        res.append((t, (c[0] if c else "")[:300]))
        if len(res) < k and self.online:
            try:
                d = self._api({"action": "query", "list": "search", "srsearch": query, "srlimit": k,
                               "srnamespace": 0})
                for h in d.get("query", {}).get("search", []):
                    snippet = re.sub(r"<[^>]+>", "", h.get("snippet", ""))
                    if all(h["title"] != r[0] for r in res):
                        res.append((h["title"], snippet))
            except Exception as e:  # noqa: BLE001
                log.debug("online search failed: %s", e)
        return res[:k]

    def lookup(self, query: str, max_chars: int = 3000) -> str:
        """Best-effort answer text for an LLM tool call."""
        c = self.cached(query) or self.cached(title_from_id(query))
        if c and (c[1] or c[0]):
            return (c[1] or c[0])[:max_chars]
        hits = self.search(query, 3)
        if not hits:
            return f"Nothing found on the wiki for {query!r}."
        text = self.page(hits[0][0], max_chars)
        others = ", ".join(h[0] for h in hits[1:])
        return text + (f"\n(other pages: {others})" if others else "")
