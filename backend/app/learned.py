"""Learned knowledge: facts found by web research, stored separately from the
verified Crop Master workbook until an agronomist reviews them."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime

from .config import settings

_lock = threading.Lock()


def _key(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


class LearnedStore:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS learned (
            id TEXT PRIMARY KEY, kind TEXT, name TEXT, crop TEXT, aliases TEXT,
            data TEXT, sources TEXT, status TEXT DEFAULT 'unverified',
            created_at TEXT, reviewed_at TEXT, hits INTEGER DEFAULT 0)""")
        self.db.commit()

    # ---------- reading ----------
    def _row(self, r) -> dict:
        d = dict(r)
        d["data"] = json.loads(d["data"] or "{}")
        d["sources"] = json.loads(d["sources"] or "[]")
        d["aliases"] = json.loads(d["aliases"] or "[]")
        return d

    def get(self, rid: str) -> dict | None:
        r = self.db.execute("SELECT * FROM learned WHERE id=?", (rid.upper(),)).fetchone()
        return self._row(r) if r else None

    def _usable(self):
        return [self._row(r) for r in self.db.execute("SELECT * FROM learned WHERE status != 'rejected'")]

    def find(self, name: str, crop: str = "") -> dict | None:
        """Exact-ish match on name/aliases, preferring the same crop."""
        k = _key(name)
        if len(k) < 4:
            return None
        best = None
        for r in self._usable():
            keys = [_key(r["name"])] + [_key(a) for a in r["aliases"]]
            if any(x and (x == k or (len(x) >= 6 and (x in k or k in x))) for x in keys):
                if not crop or _key(crop) in _key(r["crop"]) or not best:
                    best = r
        return best

    def match_text(self, text: str, limit: int = 3) -> list[dict]:
        """Learned rows whose name or alias appears in a question."""
        t = _key(text)
        out = []
        for r in self._usable():
            keys = [_key(r["name"])] + [_key(a) for a in r["aliases"]]
            if any(len(x) >= 6 and x in t for x in keys):
                out.append(r)
        return out[:limit]

    def list(self, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM learned", ()
        if status:
            q, args = q + " WHERE status=?", (status,)
        return [self._row(r) for r in self.db.execute(q + " ORDER BY created_at DESC", args)]

    # ---------- writing ----------
    def add(self, kind: str, name: str, crop: str, aliases: list[str], data: dict, sources: list[dict]) -> dict:
        with _lock:
            n = self.db.execute("SELECT COUNT(*) FROM learned").fetchone()[0] + 1
            rid = f"LRN{n:03d}"
            self.db.execute(
                "INSERT INTO learned (id, kind, name, crop, aliases, data, sources, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (rid, kind, name, crop, json.dumps(aliases, ensure_ascii=False), json.dumps(data, ensure_ascii=False),
                 json.dumps(sources, ensure_ascii=False), datetime.now().isoformat(timespec="seconds")))
            self.db.commit()
        return self.get(rid)

    def hit(self, rid: str):
        with _lock:
            self.db.execute("UPDATE learned SET hits = hits + 1 WHERE id=?", (rid,))
            self.db.commit()

    def set_status(self, rid: str, status: str) -> dict | None:
        with _lock:
            self.db.execute("UPDATE learned SET status=?, reviewed_at=? WHERE id=?",
                            (status, datetime.now().isoformat(timespec="seconds"), rid.upper()))
            self.db.commit()
        return self.get(rid)

    # ---------- helpers for the agent and UI ----------
    @staticmethod
    def flag(r: dict) -> str:
        return "Verified" if r["status"] == "verified" else "Unverified (web)"

    def ref(self, r: dict) -> dict:
        return {"id": r["id"], "title": f'{r["name"]} ({r["crop"]})' if r["crop"] else r["name"], "flag": self.flag(r)}

    def context_line(self, r: dict) -> str:
        status = ("verified by agronomist" if r["status"] == "verified"
                  else "from web research, NOT yet checked by an agronomist")
        srcs = ", ".join(s.get("url", "") for s in r["sources"][:4])
        return (f'{r["id"]} [{status}] kind={r["kind"]}; name={r["name"]}; crop={r["crop"]}; '
                f'data={json.dumps(r["data"], ensure_ascii=False)}; sources={srcs}')


store = LearnedStore(settings.learned_db)