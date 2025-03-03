from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from eduai.config import ROOT

CURRICULUM_DIR = ROOT / "curriculum"
BLOOM_LEVELS = ("remember", "understand", "apply", "analyze", "evaluate", "create")


@dataclass(frozen=True)
class Objective:
    id: str
    text: str
    keywords: tuple[str, ...]
    bloom: str
    subject: str
    subject_name: str
    unit_id: str
    unit_name: str
    topic_id: str
    topic_name: str

    def embed_text(self) -> str:
        return (
            f"{self.subject_name} > {self.unit_name} > {self.topic_name}: {self.text} "
            f"Keywords: {', '.join(self.keywords)}"
        )


@dataclass
class Unit:
    id: str
    name: str
    weight: float
    subject: str
    objectives: list[Objective] = field(default_factory=list)


@dataclass
class Subject:
    id: str
    name: str
    units: list[Unit]

    @property
    def objectives(self) -> list[Objective]:
        return [o for u in self.units for o in u.objectives]


class TaxonomyError(ValueError):
    pass


class Taxonomy:
    def __init__(self, subjects: list[Subject]):
        self.subjects = {s.id: s for s in subjects}
        self.objectives: dict[str, Objective] = {}
        self.units: dict[str, Unit] = {}
        for s in subjects:
            for u in s.units:
                self.units[u.id] = u
                for o in u.objectives:
                    self.objectives[o.id] = o

    def lo(self, lo_id: str) -> Objective:
        return self.objectives[lo_id]

    def subject_of(self, lo_id: str) -> str:
        return self.objectives[lo_id].subject

    def objectives_for(self, subject: str) -> list[Objective]:
        return self.subjects[subject].objectives

    def lo_ids(self) -> list[str]:
        return list(self.objectives)


def _parse(doc: dict, source: str) -> Subject:
    try:
        sid, sname = doc["subject"], doc["name"]
        units = []
        for u in doc["units"]:
            unit = Unit(u["id"], u["name"], float(u["weight"]), sid)
            for t in u["topics"]:
                for o in t["objectives"]:
                    unit.objectives.append(
                        Objective(
                            id=o["id"],
                            text=o["text"].strip(),
                            keywords=tuple(k.strip().lower() for k in o["keywords"]),
                            bloom=o["bloom"],
                            subject=sid,
                            subject_name=sname,
                            unit_id=u["id"],
                            unit_name=u["name"],
                            topic_id=t["id"],
                            topic_name=t["name"],
                        )
                    )
            units.append(unit)
    except (KeyError, TypeError) as exc:
        raise TaxonomyError(f"{source}: missing field {exc}") from exc
    return Subject(sid, sname, units)


def validate(subjects: list[Subject]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for s in subjects:
        total = sum(u.weight for u in s.units)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            problems.append(f"{s.id}: unit weights sum to {total:.3f}, expected 1.0")
        for u in s.units:
            if not u.id.startswith(s.id + "."):
                problems.append(f"{u.id}: unit id does not start with subject id {s.id}")
            if not u.objectives:
                problems.append(f"{u.id}: unit has no objectives")
            for ident in [u.id, *(o.id for o in u.objectives)]:
                if ident in seen:
                    problems.append(f"duplicate id {ident}")
                seen.add(ident)
            for o in u.objectives:
                if not o.id.startswith(o.topic_id + "."):
                    problems.append(f"{o.id}: objective id does not start with topic id {o.topic_id}")
                if o.bloom not in BLOOM_LEVELS:
                    problems.append(f"{o.id}: unknown bloom level {o.bloom!r}")
                if len(o.keywords) < 3:
                    problems.append(f"{o.id}: needs at least 3 keywords")
                if len(o.text) < 30:
                    problems.append(f"{o.id}: objective text is too short")
    return problems


def load_subjects(directory: Path = CURRICULUM_DIR) -> list[Subject]:
    subjects = []
    for path in sorted(directory.glob("ap_*.yaml")):
        with open(path) as fh:
            subjects.append(_parse(yaml.safe_load(fh), path.name))
    return subjects


def load_taxonomy(directory: Path = CURRICULUM_DIR, strict: bool = True) -> Taxonomy:
    subjects = load_subjects(directory)
    problems = validate(subjects)
    if strict and problems:
        raise TaxonomyError("; ".join(problems))
    return Taxonomy(subjects)


@cache
def default_taxonomy() -> Taxonomy:
    return load_taxonomy()
