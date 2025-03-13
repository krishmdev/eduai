from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPLITS = ("train", "valid", "test")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


@dataclass
class SciqItem:
    id: str
    split: str
    question: str
    correct: str
    distractors: list[str]
    support: str
    grounded: bool = True
    tags: dict = field(default_factory=dict)

    def qa_text(self) -> str:
        from eduai.curriculum.tagger import tag_text

        return tag_text(self.question, self.correct)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SciqItem:
        return cls(**d)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return _WS.sub(" ", text).strip()


def normalize_for_hash(text: str) -> str:
    return _WS.sub(" ", _NON_ALNUM.sub(" ", clean(text).lower())).strip()


def passage_hash(support: str) -> str | None:
    norm = normalize_for_hash(support)
    if not norm:
        # Blank passages must not hash to one shared value, or every ungrounded item would
        # collapse into a single split group.
        return None
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def qa_hash(question: str, answer: str) -> str:
    return hashlib.sha1(f"{normalize_for_hash(question)}||{normalize_for_hash(answer)}".encode()).hexdigest()[
        :16
    ]


def parse_record(raw: dict, split: str, index: int) -> SciqItem:
    support = clean(raw.get("support", ""))
    return SciqItem(
        id=f"{split}-{index:05d}",
        split=split,
        question=clean(raw["question"]),
        correct=clean(raw["correct_answer"]),
        distractors=[clean(raw[f"distractor{i}"]) for i in (1, 2, 3)],
        support=support,
        grounded=bool(support),
    )


def load_split(sciq_dir: Path, split: str) -> list[SciqItem]:
    path = Path(sciq_dir) / f"{split}.json"
    with open(path) as fh:
        rows = json.load(fh)
    return [parse_record(r, split, i) for i, r in enumerate(rows)]


def load_sciq(sciq_dir: Path) -> list[SciqItem]:
    items: list[SciqItem] = []
    for split in SPLITS:
        items.extend(load_split(sciq_dir, split))
    return items


def eligibility_counts(items: list[SciqItem]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        rows = [it for it in items if it.split == split]
        out[split] = {
            "total": len(rows),
            "grounded": sum(it.grounded for it in rows),
            "blank_support": sum(not it.grounded for it in rows),
        }
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
