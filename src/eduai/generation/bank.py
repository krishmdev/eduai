"""SQLite-backed item bank with online (Elo) difficulty estimates."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from eduai.data.difficulty import LABEL_B
from eduai.kt.elo import DEFAULT_CALIBRATION, ItemCalibration

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    lo_id TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    source TEXT NOT NULL,
    grounded INTEGER NOT NULL,
    aligned INTEGER NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_lo ON items(lo_id);
CREATE INDEX IF NOT EXISTS items_subject ON items(subject);
CREATE TABLE IF NOT EXISTS item_stats (
    item_id TEXT PRIMARY KEY REFERENCES items(id),
    b REAL NOT NULL,
    b_label REAL NOT NULL,
    n INTEGER NOT NULL DEFAULT 0,
    n_correct INTEGER NOT NULL DEFAULT 0
);
"""


class ItemBank:
    def __init__(self, path: Path | str, cal: ItemCalibration = DEFAULT_CALIBRATION):
        self.cal = cal
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        """Per-thread connection. isolation_level=None: transactions are explicit (BEGIN IMMEDIATE)."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    def add(self, rows: Iterable[dict], replace: bool = False) -> int:
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        n = 0
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            for r in rows:
                cur = self.conn.execute(
                    f"{verb} INTO items (id, subject, unit_id, lo_id, difficulty, source, grounded, aligned, body)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        r["id"],
                        r["subject"],
                        r["unit_id"],
                        r["lo_id"],
                        r["difficulty"],
                        r.get("source", "sciq"),
                        int(r.get("grounded", True)),
                        int(r.get("aligned", True)),
                        json.dumps(r),
                    ),
                )
                n += cur.rowcount
                b0 = float(LABEL_B.get(r["difficulty"], 0.0))
                self.conn.execute(
                    "INSERT OR IGNORE INTO item_stats (item_id, b, b_label) VALUES (?, ?, ?)",
                    (r["id"], b0, b0),
                )
            self.conn.execute("COMMIT")
        return n

    def load_jsonl(self, path: Path) -> int:
        with open(path) as fh:
            return self.add(json.loads(line) for line in fh if line.strip())

    def count(self, subject: str | None = None) -> int:
        q, args = "SELECT COUNT(*) FROM items", ()
        if subject:
            q, args = q + " WHERE subject = ?", (subject,)
        return self.conn.execute(q, args).fetchone()[0]

    def subjects(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT subject, COUNT(*) AS n FROM items WHERE aligned = 1 GROUP BY subject"
        )
        return {r["subject"]: r["n"] for r in rows}

    def los_with_items(self, subject: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT lo_id FROM items WHERE subject = ? AND aligned = 1", (subject,)
        )
        return {r["lo_id"] for r in rows}

    def candidates(self, lo_id: str, exclude: Iterable[str] = ()) -> list[dict]:
        ex = set(exclude)
        rows = self.conn.execute(
            "SELECT i.body, s.b, s.n, s.b_label FROM items i JOIN item_stats s ON s.item_id = i.id"
            " WHERE i.lo_id = ? AND i.aligned = 1",
            (lo_id,),
        )
        out = []
        for r in rows:
            body = json.loads(r["body"])
            if body["id"] in ex:
                continue
            body["b"] = self.cal.effective_b(r["b"], r["n"], r["b_label"])
            body["b_hat"] = r["b"]
            body["n_responses"] = r["n"]
            out.append(body)
        return out

    def get(self, item_id: str) -> dict | None:
        r = self.conn.execute(
            "SELECT i.body, s.b, s.n, s.b_label FROM items i JOIN item_stats s ON s.item_id = i.id WHERE i.id = ?",
            (item_id,),
        ).fetchone()
        if r is None:
            return None
        body = json.loads(r["body"])
        body["b"] = self.cal.effective_b(r["b"], r["n"], r["b_label"])
        body["b_hat"] = r["b"]
        body["n_responses"] = r["n"]
        return body

    def stems(self, subject: str | None = None) -> list[str]:
        rows = self.conn.execute(
            "SELECT body FROM items" + (" WHERE subject = ?" if subject else ""),
            (subject,) if subject else (),
        )
        return [json.loads(r["body"])["stem"] for r in rows]

    def item_b(self, item_id: str) -> tuple[float, int, float]:
        r = self.conn.execute("SELECT b, n, b_label FROM item_stats WHERE item_id = ?", (item_id,)).fetchone()
        if r is None:
            raise KeyError(item_id)
        return r["b"], r["n"], r["b_label"]

    def calibrate(
        self, item_id: str, theta_pre: float, correct: bool, cal: ItemCalibration | None = None
    ) -> float:
        """One Elo step on b. Read and write happen inside one write transaction so two
        concurrent responses to the same item can't both start from the same b."""
        cal = cal or self.cal
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                r = self.conn.execute("SELECT b, n FROM item_stats WHERE item_id = ?", (item_id,)).fetchone()
                b_new = cal.step(r["b"], r["n"], theta_pre, correct)
                self.conn.execute(
                    "UPDATE item_stats SET b = ?, n = n + 1, n_correct = n_correct + ? WHERE item_id = ?",
                    (b_new, int(correct), item_id),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        return b_new
