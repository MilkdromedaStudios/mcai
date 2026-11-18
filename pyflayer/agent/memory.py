"""The bot's brain: a persistent SQLite "model file" that grows every session.

Holds
  * episodic memory   - everything that happened (actions, results, deaths, chats) with full-text recall
  * semantic facts    - lessons/insights learned (reflections, failures, tips)
  * knowledge base    - one entry per block/item/mob/biome/structure (game data + wiki)
  * skills            - reusable, parameterised action programs learned from successful tasks
  * waypoints         - spatial memory (base, crafting table, ores, structures, biomes, deaths)
  * learned numbers   - real block hardness measured while digging, stats, milestones
It can be exported to / imported from a portable JSON(.gz) model file.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, start REAL, end REAL, kind TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, session INTEGER, kind TEXT,
    text TEXT, importance REAL DEFAULT 1.0);
CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, topic TEXT, text TEXT UNIQUE,
    confidence REAL DEFAULT 0.7, source TEXT, uses INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS knowledge(name TEXT PRIMARY KEY, kind TEXT, text TEXT, source TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS skills(name TEXT PRIMARY KEY, description TEXT, params TEXT, steps TEXT,
    successes INTEGER DEFAULT 0, failures INTEGER DEFAULT 0, created REAL, last_used REAL, avg_seconds REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS waypoints(name TEXT PRIMARY KEY, dim TEXT, x INTEGER, y INTEGER, z INTEGER, kind TEXT,
    note TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS hardness(block TEXT PRIMARY KEY, value REAL, samples INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS stats(key TEXT PRIMARY KEY, value REAL);
CREATE TABLE IF NOT EXISTS seen(kind TEXT, name TEXT, first REAL, count INTEGER DEFAULT 1, PRIMARY KEY(kind, name));
CREATE TABLE IF NOT EXISTS milestones(name TEXT PRIMARY KEY, achieved REAL, seconds REAL, attempts INTEGER DEFAULT 0,
    last_error TEXT);
CREATE TABLE IF NOT EXISTS skill_drills(name TEXT PRIMARY KEY, attempts INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0, last_result TEXT DEFAULT '', last_practiced REAL DEFAULT 0);
"""
FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(text, content='episodes', content_rowid='id', tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(topic, text, content='facts', content_rowid='id', tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(name, text, content='knowledge', content_rowid='rowid', tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(name, description, content='skills', content_rowid='rowid', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS ep_ai AFTER INSERT ON episodes BEGIN
  INSERT INTO episodes_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS fa_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, topic, text) VALUES (new.id, new.topic, new.text); END;
CREATE TRIGGER IF NOT EXISTS fa_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, topic, text) VALUES('delete', old.id, old.topic, old.text); END;
CREATE TRIGGER IF NOT EXISTS kn_ai AFTER INSERT ON knowledge BEGIN
  INSERT INTO knowledge_fts(rowid, name, text) VALUES (new.rowid, new.name, new.text); END;
CREATE TRIGGER IF NOT EXISTS kn_ad AFTER DELETE ON knowledge BEGIN
  INSERT INTO knowledge_fts(knowledge_fts, rowid, name, text) VALUES('delete', old.rowid, old.name, old.text); END;
CREATE TRIGGER IF NOT EXISTS kn_au AFTER UPDATE ON knowledge BEGIN
  INSERT INTO knowledge_fts(knowledge_fts, rowid, name, text) VALUES('delete', old.rowid, old.name, old.text);
  INSERT INTO knowledge_fts(rowid, name, text) VALUES (new.rowid, new.name, new.text); END;
CREATE TRIGGER IF NOT EXISTS sk_ai AFTER INSERT ON skills BEGIN
  INSERT INTO skills_fts(rowid, name, description) VALUES (new.rowid, new.name, new.description); END;
CREATE TRIGGER IF NOT EXISTS sk_ad AFTER DELETE ON skills BEGIN
  INSERT INTO skills_fts(skills_fts, rowid, name, description) VALUES('delete', old.rowid, old.name, old.description); END;
CREATE TRIGGER IF NOT EXISTS sk_au AFTER UPDATE ON skills BEGIN
  INSERT INTO skills_fts(skills_fts, rowid, name, description) VALUES('delete', old.rowid, old.name, old.description);
  INSERT INTO skills_fts(rowid, name, description) VALUES (new.rowid, new.name, new.description); END;
"""
_WORD = re.compile(r"[a-z0-9_]{2,}")
_STOP = {"the", "and", "for", "with", "you", "how", "what", "get", "make", "can", "into", "from", "that", "this",
         "use", "are", "have", "has", "need", "some", "any", "minecraft"}


def _fts_query(text: str) -> str:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP][:12]
    parts = []
    for w in words:
        parts.append(f'"{w}"*')
        if "_" in w:
            parts += [f'"{p}"*' for p in w.split("_") if len(p) > 2 and p not in _STOP]
    return " OR ".join(dict.fromkeys(parts))


class Brain:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.lock = threading.RLock()
        self.export_lock = threading.Lock()
        self.db.executescript(SCHEMA)
        try:
            self.db.executescript(FTS)
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        self.db.commit()
        self.session_id = None

    # ------------------------------------------------------------------ sessions
    def start_session(self, kind: str = "play") -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO sessions(start, kind) VALUES (?,?)", (time.time(), kind))
            self.db.commit()
            self.session_id = cur.lastrowid
            self.bump("sessions")
            return self.session_id

    def end_session(self, summary: str = ""):
        if self.session_id is None:
            return
        with self.lock:
            self.db.execute("UPDATE sessions SET end=?, summary=? WHERE id=?", (time.time(), summary, self.session_id))
            self.db.commit()

    # ------------------------------------------------------------------ episodic
    def log(self, kind: str, text: str, importance: float = 1.0):
        with self.lock:
            self.db.execute("INSERT INTO episodes(ts, session, kind, text, importance) VALUES (?,?,?,?,?)",
                            (time.time(), self.session_id, kind, text[:2000], importance))
            self.db.commit()

    def recent(self, n: int = 20, kinds: tuple | None = None) -> list[tuple]:
        with self.lock:
            if kinds:
                q = f"SELECT ts, kind, text FROM episodes WHERE kind IN ({','.join('?' * len(kinds))}) ORDER BY id DESC LIMIT ?"
                rows = self.db.execute(q, (*kinds, n)).fetchall()
            else:
                rows = self.db.execute("SELECT ts, kind, text FROM episodes ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return rows[::-1]

    # ------------------------------------------------------------------ facts
    def remember(self, text: str, topic: str = "general", confidence: float = 0.7, source: str = "agent") -> bool:
        text = text.strip()
        if len(text) < 4:
            return False
        with self.lock:
            try:
                self.db.execute("INSERT INTO facts(ts, topic, text, confidence, source) VALUES (?,?,?,?,?)",
                                (time.time(), topic[:60], text[:600], confidence, source))
                self.db.commit()
                return True
            except sqlite3.IntegrityError:
                self.db.execute("UPDATE facts SET confidence=MIN(1.0, confidence+0.1), uses=uses+1 WHERE text=?", (text[:600],))
                self.db.commit()
                return False

    def _search(self, table: str, cols: str, query: str, k: int, extra_where: str = "") -> list:
        q = _fts_query(query)
        if not q:
            return []
        with self.lock:
            if self.fts:
                try:
                    return self.db.execute(
                        f"SELECT {cols} FROM {table}_fts JOIN {table} ON {table}.rowid = {table}_fts.rowid "
                        f"WHERE {table}_fts MATCH ? {extra_where} ORDER BY bm25({table}_fts) LIMIT ?", (q, k)).fetchall()
                except sqlite3.OperationalError:
                    pass
            words = [w.strip('"*') for w in q.split(" OR ")][:4]
            like = " OR ".join(["text LIKE ?"] * len(words))
            return self.db.execute(f"SELECT {cols} FROM {table} WHERE ({like}) {extra_where} LIMIT ?",
                                   (*[f"%{w}%" for w in words], k)).fetchall()

    def recall(self, query: str, k: int = 8) -> dict:
        facts = self._search("facts", "facts.topic, facts.text, facts.confidence", query, k)
        eps = self._search("episodes", "episodes.kind, episodes.text", query, k)
        know = self._search("knowledge", "knowledge.name, knowledge.text", query, max(3, k // 2))
        return {"facts": facts, "episodes": eps, "knowledge": know}

    def recall_text(self, query: str, k: int = 6, max_chars: int = 2500) -> str:
        r = self.recall(query, k)
        out = []
        if r["facts"]:
            out.append("Lessons/facts I learned:\n" + "\n".join(f"- [{t}] {x}" for t, x, _ in r["facts"]))
        if r["knowledge"]:
            out.append("Knowledge base:\n" + "\n".join(f"- {n}: {x[:300]}" for n, x in r["knowledge"]))
        if r["episodes"]:
            out.append("Related past events:\n" + "\n".join(f"- ({k_}) {x[:200]}" for k_, x in r["episodes"][:k]))
        s = "\n".join(out)
        return s[:max_chars]

    # ------------------------------------------------------------------ knowledge base
    def set_knowledge(self, name: str, kind: str, text: str, source: str = "gamedata"):
        with self.lock:
            row = self.db.execute("SELECT text, source FROM knowledge WHERE name=?", (name,)).fetchone()
            if row and source == "wiki" and "wiki" not in row[1]:
                text = row[0] + "\nWiki: " + text
                source = row[1] + "+wiki"
            elif row and source == "gamedata" and "wiki" in row[1]:
                return
            self.db.execute("INSERT OR REPLACE INTO knowledge(name, kind, text, source, ts) VALUES (?,?,?,?,?)",
                            (name, kind, text[:4000], source, time.time()))

    def commit(self):
        with self.lock:
            self.db.commit()

    def knowledge(self, name: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT text FROM knowledge WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    def knowledge_count(self, kind: str | None = None, source_like: str | None = None) -> int:
        q = "SELECT COUNT(*) FROM knowledge WHERE 1=1"
        args = []
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if source_like:
            q += " AND source LIKE ?"
            args.append(f"%{source_like}%")
        with self.lock:
            return self.db.execute(q, args).fetchone()[0]

    # ------------------------------------------------------------------ skills
    def save_skill(self, name: str, description: str, steps: list, params: list | None = None) -> str:
        name = re.sub(r"[^a-z0-9_]", "_", name.lower())[:48].strip("_") or "skill"
        with self.lock:
            row = self.db.execute("SELECT successes FROM skills WHERE name=?", (name,)).fetchone()
            if row:
                self.db.execute("UPDATE skills SET description=?, steps=?, params=? WHERE name=?",
                                (description, json.dumps(steps), json.dumps(params or []), name))
            else:
                self.db.execute("INSERT INTO skills(name, description, params, steps, created) VALUES (?,?,?,?,?)",
                                (name, description, json.dumps(params or []), json.dumps(steps), time.time()))
            self.db.commit()
        return name

    def skill(self, name: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT name, description, params, steps, successes, failures, avg_seconds "
                                  "FROM skills WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        return {"name": row[0], "description": row[1], "params": json.loads(row[2] or "[]"),
                "steps": json.loads(row[3] or "[]"), "successes": row[4], "failures": row[5], "avg_seconds": row[6]}

    def skill_result(self, name: str, ok: bool, seconds: float):
        with self.lock:
            if ok:
                self.db.execute("UPDATE skills SET successes=successes+1, last_used=?, "
                                "avg_seconds=(avg_seconds*successes+?)/(successes+1) WHERE name=?",
                                (time.time(), seconds, name))
            else:
                self.db.execute("UPDATE skills SET failures=failures+1, last_used=? WHERE name=?", (time.time(), name))
            self.db.commit()

    def skills(self, query: str | None = None, k: int = 12) -> list[dict]:
        if query:
            rows = self._search("skills", "skills.name", query, k)
            names = [r[0] for r in rows]
        else:
            with self.lock:
                names = [r[0] for r in self.db.execute(
                    "SELECT name FROM skills ORDER BY successes DESC, last_used DESC LIMIT ?", (k,)).fetchall()]
        return [s for s in (self.skill(n) for n in names) if s]

    # ------------------------------------------------------------------ waypoints
    def set_waypoint(self, name: str, dim: str, pos, kind: str = "place", note: str = ""):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO waypoints VALUES (?,?,?,?,?,?,?,?)",
                            (name[:64], dim, int(pos[0]), int(pos[1]), int(pos[2]), kind, note[:200], time.time()))
            self.db.commit()

    def waypoints(self, dim: str | None = None, near=None, kind: str | None = None, k: int = 30) -> list[dict]:
        q = "SELECT name, dim, x, y, z, kind, note FROM waypoints WHERE 1=1"
        args = []
        if dim:
            q += " AND dim=?"
            args.append(dim)
        if kind:
            q += " AND kind=?"
            args.append(kind)
        with self.lock:
            rows = self.db.execute(q, args).fetchall()
        out = [{"name": r[0], "dim": r[1], "pos": (r[2], r[3], r[4]), "kind": r[5], "note": r[6]} for r in rows]
        if near:
            out.sort(key=lambda w: math.dist(w["pos"], near))
        return out[:k]

    def waypoint(self, name: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT name, dim, x, y, z, kind, note FROM waypoints WHERE name=?", (name,)).fetchone()
        return {"name": r[0], "dim": r[1], "pos": (r[2], r[3], r[4]), "kind": r[5], "note": r[6]} if r else None

    # ------------------------------------------------------------------ numbers
    def save_hardness(self, block: str, value: float):
        with self.lock:
            self.db.execute("INSERT INTO hardness(block, value) VALUES (?,?) ON CONFLICT(block) DO UPDATE SET "
                            "value=(value*samples+excluded.value)/(samples+1), samples=samples+1", (block, value))
            self.db.commit()

    def learned_hardness(self) -> dict[str, float]:
        with self.lock:
            return dict(self.db.execute("SELECT block, value FROM hardness").fetchall())

    def bump(self, key: str, by: float = 1.0):
        with self.lock:
            self.db.execute("INSERT INTO stats(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=value+?",
                            (key, by, by))
            self.db.commit()

    def stat(self, key: str, default: float = 0.0) -> float:
        with self.lock:
            r = self.db.execute("SELECT value FROM stats WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def stats(self) -> dict:
        with self.lock:
            return dict(self.db.execute("SELECT key, value FROM stats").fetchall())

    def mark_seen(self, kind: str, names):
        now = time.time()
        with self.lock:
            for n in names:
                self.db.execute("INSERT INTO seen(kind, name, first) VALUES (?,?,?) ON CONFLICT(kind, name) "
                                "DO UPDATE SET count=count+1", (kind, n, now))
            self.db.commit()

    def seen(self, kind: str) -> set[str]:
        with self.lock:
            return {r[0] for r in self.db.execute("SELECT name FROM seen WHERE kind=?", (kind,)).fetchall()}

    def milestone(self, name: str, achieved: bool, seconds: float = 0, error: str = ""):
        with self.lock:
            row = self.db.execute("SELECT achieved, attempts FROM milestones WHERE name=?", (name,)).fetchone()
            if row is None:
                self.db.execute("INSERT INTO milestones VALUES (?,?,?,?,?)",
                                (name, time.time() if achieved else None, seconds, 1, error[:300]))
            else:
                ach = row[0] or (time.time() if achieved else None)
                self.db.execute("UPDATE milestones SET achieved=?, seconds=CASE WHEN ? THEN ? ELSE seconds END, "
                                "attempts=attempts+1, last_error=? WHERE name=?",
                                (ach, achieved and not row[0], seconds, error[:300], name))
            self.db.commit()

    def milestones(self) -> dict[str, dict]:
        with self.lock:
            rows = self.db.execute("SELECT name, achieved, seconds, attempts, last_error FROM milestones").fetchall()
        return {r[0]: {"achieved": r[1], "seconds": r[2], "attempts": r[3], "last_error": r[4]} for r in rows}

    # ------------------------------------------------------------------ measured movement / survival drills
    def record_drill(self, name: str, success: bool, detail: str = "") -> dict:
        """Persist verified action outcomes. LLM claims alone cannot mark success."""
        with self.lock:
            self.db.execute(
                "INSERT INTO skill_drills(name, attempts, successes, last_result, last_practiced) "
                "VALUES (?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                "attempts=attempts+1, successes=successes+excluded.successes, "
                "last_result=excluded.last_result, last_practiced=excluded.last_practiced",
                (name, 1, int(bool(success)), detail[:300], time.time()))
            self.db.commit()
        return self.drill(name)

    def drill(self, name: str) -> dict:
        with self.lock:
            row = self.db.execute(
                "SELECT attempts, successes, last_result, last_practiced FROM skill_drills WHERE name=?",
                (name,)).fetchone()
        attempts, successes, detail, when = row or (0, 0, "", 0.0)
        # Only actual successful controlled trials unlock high-risk skills.
        accuracy = successes / attempts if attempts else 0.0
        return {"name": name, "attempts": attempts, "successes": successes,
                "failures": attempts - successes, "accuracy": round(accuracy, 3),
                "certified": attempts >= 8 and successes >= 7 and accuracy >= 0.85,
                "last_result": detail, "last_practiced": when}

    def drills(self) -> dict[str, dict]:
        with self.lock:
            names = [r[0] for r in self.db.execute("SELECT name FROM skill_drills").fetchall()]
        return {name: self.drill(name) for name in names}

    # ------------------------------------------------------------------ portable model file
    def export_latest(self) -> Path:
        """Export a reusable, current model beside this bot's live brain database."""
        return self.export(self.path.with_name(self.path.stem + "-export.json.gz"))

    def export(self, path: Path) -> Path:
        """Atomically save learned memory, skill code, practice scores and game facts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.export_lock:
            with self.lock:
                data = {
                    "format": "pyflayer-brain-2", "exported": time.time(),
                    "facts": self.db.execute("SELECT topic,text,confidence,source FROM facts").fetchall(),
                    "episodes": list(reversed(self.db.execute(
                        "SELECT ts,kind,text,importance FROM episodes ORDER BY id DESC LIMIT 10000").fetchall())),
                    "skills": [self.skill(row[0]) for row in self.db.execute(
                        "SELECT name FROM skills").fetchall()],
                    "waypoints": self.db.execute("SELECT * FROM waypoints").fetchall(),
                    "hardness": self.db.execute("SELECT block,value,samples FROM hardness").fetchall(),
                    "stats": self.db.execute("SELECT key,value FROM stats").fetchall(),
                    "milestones": self.db.execute("SELECT * FROM milestones").fetchall(),
                    "skill_drills": self.db.execute("SELECT * FROM skill_drills").fetchall(),
                    "seen": self.db.execute("SELECT kind,name,first,count FROM seen").fetchall(),
                    "knowledge": self.db.execute("SELECT name,kind,text,source FROM knowledge").fetchall(),
                }
            tmp = path.with_name(path.name + ".part")
            try:
                with gzip.open(tmp, "wt", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        return path

    def import_(self, path: Path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") not in ("pyflayer-brain-1", "pyflayer-brain-2"):
            raise ValueError("Unsupported PyFlayer model backup")
        for topic, content, confidence, source in data.get("facts", []):
            self.remember(content, topic, confidence, source)
        for sk in data.get("skills", []):
            if not sk:
                continue
            name = self.save_skill(sk["name"], sk["description"], sk["steps"], sk.get("params"))
            with self.lock:
                self.db.execute(
                    "UPDATE skills SET successes=MAX(successes,?), failures=MAX(failures,?), "
                    "last_used=MAX(COALESCE(last_used,0),?), "
                    "avg_seconds=CASE WHEN successes <= ? THEN ? ELSE avg_seconds END WHERE name=?",
                    (sk.get("successes", 0), sk.get("failures", 0),
                     sk.get("last_used") or 0.0, sk.get("successes", 0),
                     sk.get("avg_seconds") or 0.0, name))
                self.db.commit()
        with self.lock:
            known = set(self.db.execute("SELECT ts,kind,text FROM episodes").fetchall())
            for ts, kind, content, importance in data.get("episodes", []):
                key = (ts, kind, content)
                if key not in known:
                    self.db.execute(
                        "INSERT INTO episodes(ts,session,kind,text,importance) VALUES (?,?,?,?,?)",
                        (ts, None, kind, content, importance))
                    known.add(key)
            for row in data.get("waypoints", []):
                self.db.execute("INSERT OR REPLACE INTO waypoints VALUES (?,?,?,?,?,?,?,?)", row)
            for block, value, samples in data.get("hardness", []):
                self.db.execute(
                    "INSERT OR REPLACE INTO hardness VALUES (?,?,?)", (block, value, samples))
            for key, value in data.get("stats", []):
                self.db.execute(
                    "INSERT INTO stats(key,value) VALUES (?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=MAX(stats.value, excluded.value)",
                    (key, value))
            for row in data.get("milestones", []):
                self.db.execute("INSERT OR IGNORE INTO milestones VALUES (?,?,?,?,?)", row)
            for row in data.get("skill_drills", []):
                self.db.execute(
                    "INSERT INTO skill_drills(name,attempts,successes,last_result,last_practiced) "
                    "VALUES (?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                    "attempts=MAX(skill_drills.attempts,excluded.attempts), "
                    "successes=MAX(skill_drills.successes,excluded.successes), "
                    "last_result=CASE WHEN excluded.last_practiced >= skill_drills.last_practiced "
                    "THEN excluded.last_result ELSE skill_drills.last_result END, "
                    "last_practiced=MAX(skill_drills.last_practiced,excluded.last_practiced)", row)
            for row in data.get("seen", []):
                self.db.execute("INSERT OR IGNORE INTO seen VALUES (?,?,?,?)", row)
            self.db.commit()
        for name, kind, content, source in data.get("knowledge", []):
            self.set_knowledge(name, kind, content, source)
        self.commit()

    def overview(self) -> dict:
        with self.lock:
            c = lambda t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: E731
            return {"episodes": c("episodes"), "facts": c("facts"), "skills": c("skills"),
                    "waypoints": c("waypoints"), "knowledge": c("knowledge"), "sessions": c("sessions"),
                    "milestones_done": self.db.execute("SELECT COUNT(*) FROM milestones WHERE achieved IS NOT NULL").fetchone()[0],
                    "skill_drills": c("skill_drills"),
                    "learned_hardness": c("hardness"), "seen_blocks": len(self.seen("block"))}
