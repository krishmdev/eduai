"""Session service shared by the HTML and JSON routes.

Sessions, responses and students live in SQLite next to the item bank. The in-memory session
engine (kt/state.py) is rebuilt by replaying a session's responses on each request, so the web
process holds no per-session state and restarts are harmless.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eduai.curriculum.neighbors import load_graph
from eduai.curriculum.taxonomy import Taxonomy
from eduai.kt import irt, scoring
from eduai.kt.policy import Candidate
from eduai.kt.state import SessionState, new_session

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    student TEXT NOT NULL,
    subject TEXT NOT NULL,
    mode TEXT NOT NULL,
    max_items INTEGER NOT NULL,
    sd_stop REAL NOT NULL,
    current_item TEXT,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS responses (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    seq INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    lo_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    b REAL NOT NULL,
    choice TEXT NOT NULL,
    correct INTEGER NOT NULL,
    created REAL NOT NULL,
    PRIMARY KEY (session_id, seq)
);
"""

# Web defaults. SD < 0.3 needs about 70 well-targeted items under this model (see
# reports/sim_report.md), so the app stops at SD < 0.5 or 30 items.
ASSESSMENT_SD_STOP = 0.5
ASSESSMENT_MAX = 30
PRACTICE_LENGTHS = (10, 20, 30)


class SessionNotFound(KeyError):
    pass


class SessionFinished(RuntimeError):
    pass


class AnswerConflict(RuntimeError):
    """The pending question was already answered by a concurrent request."""


@dataclass
class Question:
    item: dict
    number: int
    lo_text: str
    unit_name: str


class Service:
    def __init__(self, db_path: Path | str, bank, taxonomy: Taxonomy, generator_pipeline=None):
        self.bank = bank
        self.tax = taxonomy
        self.pipeline = generator_pipeline
        self.graph = load_graph()
        self.db_path = str(db_path)
        self._local = threading.local()
        self.lock = threading.Lock()
        with self.lock, self.conn:
            self.conn.executescript(SCHEMA)
        self.unit_names = {u.id: u.name for s in taxonomy.subjects.values() for u in s.units}

    @property
    def conn(self) -> sqlite3.Connection:
        # One connection per thread: sqlite3 connections must not be used from two threads at once.
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=10)
            c.row_factory = sqlite3.Row
            self._local.conn = c
        return c

    # -- sessions ---------------------------------------------------------------------------------
    def subjects(self) -> list[dict]:
        counts = self.bank.subjects()
        return [
            {"id": s.id, "name": s.name, "items": counts.get(s.id, 0)}
            for s in self.tax.subjects.values()
            if counts.get(s.id, 0)
        ]

    def create(self, subject: str, mode: str, length: int | None = None, student: str = "guest") -> str:
        if subject not in self.tax.subjects:
            raise ValueError(f"unknown subject {subject}")
        if mode not in ("practice", "assessment"):
            raise ValueError("mode must be practice or assessment")
        if mode == "assessment":
            max_items, sd = min(length or ASSESSMENT_MAX, 60), ASSESSMENT_SD_STOP
        else:
            max_items, sd = int(length or PRACTICE_LENGTHS[0]), -1.0
            if not 1 <= max_items <= 60:
                raise ValueError("length must be 1-60")
        sid = uuid.uuid4().hex[:12]
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO sessions (id, student, subject, mode, max_items, sd_stop, created) VALUES (?,?,?,?,?,?,?)",
                (sid, student, subject, mode, max_items, sd, time.time()),
            )
        return sid

    def _row(self, sid: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if row is None:
            raise SessionNotFound(sid)
        return row

    def state(self, sid: str) -> tuple[sqlite3.Row, SessionState]:
        row = self._row(sid)
        subj = self.tax.subjects[row["subject"]]
        available = self.bank.los_with_items(row["subject"])
        los = [(o.id, o.unit_id) for o in subj.objectives if o.id in available]
        weights = {u.id: u.weight for u in subj.units}
        st = new_session(
            row["mode"],
            row["subject"],
            row["max_items"],
            weights,
            los,
            graph=self.graph,
            # Evidence sharing is off: in simulation D it did not improve mastery estimates.
            share=False,
            sd_stop=row["sd_stop"] if row["mode"] == "assessment" else irt.SD_STOP,
        )
        for r in self.conn.execute("SELECT * FROM responses WHERE session_id = ? ORDER BY seq", (sid,)):
            item = self.bank.get(r["item_id"]) or {}
            wrong = item.get("choices", {}).get(r["choice"]) if not r["correct"] else None
            st.record(
                Candidate(r["item_id"], r["lo_id"], r["unit_id"], r["b"]),
                bool(r["correct"]),
                r["choice"],
                wrong,
            )
        return row, st

    # -- questions --------------------------------------------------------------------------------
    def next_question(self, sid: str) -> Question | None:
        row, st = self.state(sid)
        if st.done():
            return None
        if row["current_item"]:
            item = self.bank.get(row["current_item"])
            if item:
                return self._question(item, len(st.responses) + 1)
        rng = np.random.default_rng(int(sid, 16) + len(st.responses))
        tried: set[str] = set()
        item = None
        while item is None:
            lo = st.choose_lo(rng, available={lo for lo, _ in st.los} - tried)
            if lo is None:
                break
            tried.add(lo)
            cands = self.bank.candidates(lo, exclude=st.seen)
            by_id = {c["id"]: c for c in cands}
            pick = st.choose_item([Candidate(c["id"], c["lo_id"], c["unit_id"], c["b"]) for c in cands])
            item = by_id[pick.item_id] if pick else None
        if item is None:
            return None
        with self.lock, self.conn:
            self.conn.execute("UPDATE sessions SET current_item = ? WHERE id = ?", (item["id"], sid))
        return self._question(item, len(st.responses) + 1)

    def _question(self, item: dict, number: int) -> Question:
        lo = self.tax.lo(item["lo_id"])
        return Question(item, number, lo.text, lo.unit_name)

    def answer(self, sid: str, choice: str) -> dict:
        row, st = self.state(sid)
        if st.done():
            raise SessionFinished(sid)
        item_id = row["current_item"]
        if not item_id:
            # Either nothing was served yet or a concurrent submit already took it.
            raise AnswerConflict(sid)
        if choice not in ("A", "B", "C", "D"):
            raise ValueError("choice must be A-D")
        item = self.bank.get(item_id)
        correct = choice == item["answer"]
        cand = Candidate(item["id"], item["lo_id"], item["unit_id"], item["b"])
        theta_pre = st.theta_for(cand.unit_id)
        # Claim the pending question and store the response in one transaction. A second submit
        # for the same question (double click, retry) finds current_item already cleared and loses.
        with self.lock, self.conn:
            claimed = self.conn.execute(
                "UPDATE sessions SET current_item = NULL WHERE id = ? AND current_item = ?", (sid, item_id)
            ).rowcount
            if claimed != 1:
                raise AnswerConflict(sid)
            seq = (
                self.conn.execute("SELECT COUNT(*) FROM responses WHERE session_id = ?", (sid,)).fetchone()[0]
                + 1
            )
            self.conn.execute(
                "INSERT INTO responses (session_id, seq, item_id, lo_id, unit_id, b, choice, correct, created)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    seq,
                    item["id"],
                    item["lo_id"],
                    item["unit_id"],
                    cand.b,
                    choice,
                    int(correct),
                    time.time(),
                ),
            )
        # Item calibration runs only after the response is committed, with the pre-response EAP mean.
        self.bank.calibrate(item["id"], theta_pre, correct)
        st.record(cand, correct, choice, None if correct else item["choices"][choice])
        return {
            "correct": correct,
            "answer": item["answer"],
            "choice": choice,
            "explanation": item.get("explanation") if row["mode"] == "practice" else None,
            "grounded": item.get("grounded", True),
            "done": st.done(),
            "item": item,
        }

    # -- summaries --------------------------------------------------------------------------------
    def progress(self, sid: str) -> dict:
        row, st = self.state(sid)
        post = st.posterior()
        return {
            "id": sid,
            "mode": row["mode"],
            "subject": row["subject"],
            "subject_name": self.tax.subjects[row["subject"]].name,
            "answered": len(st.responses),
            "correct": sum(r.correct for r in st.responses),
            "max_items": row["max_items"],
            "done": st.done(),
            "theta": post.mean,
            "sd": post.sd,
            "sd_stop": row["sd_stop"],
            "units": scoring.unit_mastery(st, self.unit_names),
        }

    def report(self, sid: str) -> dict:
        row, st = self.state(sid)
        summary = scoring.summary(st)
        responses = []
        for r in st.responses:
            item = self.bank.get(r.item_id) or {}
            responses.append(
                {
                    "item_id": r.item_id,
                    "stem": item.get("stem", ""),
                    "lo_id": r.lo_id,
                    "unit": self.unit_names.get(r.unit_id, r.unit_id),
                    "correct": r.correct,
                    "choice": r.choice,
                    "answer": item.get("answer"),
                    "b": r.b,
                    "theta_after": r.theta_after,
                    "sd_after": r.sd_after,
                }
            )
        misconceptions = [
            {"lo_id": lo, "lo_text": self.tax.lo(lo).text, "picked": text}
            for lo, text in st.misconceptions.items()
        ]
        return {
            **self.progress(sid),
            "summary": summary,
            "responses": responses,
            "misconceptions": misconceptions,
            "chart": trajectory_chart(responses),
        }


def trajectory_chart(responses: list[dict], w: int = 640, h: int = 220, pad: int = 36) -> dict | None:
    """SVG geometry for theta-hat after each response with a +/- 1.96 SD band (prior at x = 0)."""
    if not responses:
        return None
    pts = [(0, 0.0, 1.0)] + [(i + 1, r["theta_after"], r["sd_after"]) for i, r in enumerate(responses)]
    lo_v, hi_v = -3.0, 3.0
    n = len(pts) - 1

    def x(i: int) -> float:
        return pad + (w - 2 * pad) * (i / max(n, 1))

    def y(v: float) -> float:
        v = min(max(v, lo_v), hi_v)
        return pad / 2 + (h - pad) * (hi_v - v) / (hi_v - lo_v)

    line = " ".join(f"{x(i):.1f},{y(t):.1f}" for i, t, _ in pts)
    upper = [f"{x(i):.1f},{y(t + 1.96 * s):.1f}" for i, t, s in pts]
    lower = [f"{x(i):.1f},{y(t - 1.96 * s):.1f}" for i, t, s in reversed(pts)]
    ticks = [{"v": v, "y": y(v)} for v in (-3, -2, -1, 0, 1, 2, 3)]
    xticks = [{"i": i, "x": x(i)} for i in range(0, n + 1, max(1, n // 6))]
    dots = [{"x": x(i), "y": y(t), "i": i, "t": t, "s": s} for i, t, s in pts[1:]]
    return {
        "w": w,
        "h": h,
        "pad": pad,
        "line": line,
        "band": " ".join(upper + lower),
        "ticks": ticks,
        "xticks": xticks,
        "dots": dots,
        "last": dots[-1],
    }


def to_json(obj) -> str:
    return json.dumps(obj, default=float)
