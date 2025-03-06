"""Small committed samples of the derived data (the full SciQ-derived files stay local)."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from eduai.config import ROOT
from eduai.data.sciq import read_jsonl, write_jsonl

SAMPLES = ROOT / "data" / "samples"
DEMO_BANK = SAMPLES / "bank_sample.jsonl"


def write_samples(data_dir: Path, per_lo: int = 5, seed: int = 11) -> None:
    rng = random.Random(seed)
    sft = read_jsonl(data_dir / "sft" / "train.jsonl")
    write_jsonl(SAMPLES / "sft_sample.jsonl", sft[:20])
    bank = read_jsonl(data_dir / "bank" / "sciq_items.jsonl")
    by_lo: dict[str, list[dict]] = defaultdict(list)
    for row in bank:
        if row["aligned"]:
            by_lo[row["lo_id"]].append(row)
    # A few ungrounded rows are kept so the demo exercises the "no explanation" path.
    ungrounded = [r for r in bank if r["ungrounded"] and r["aligned"]]
    picked = []
    for lo in sorted(by_lo):
        rows = by_lo[lo][:]
        rng.shuffle(rows)
        picked += rows[:per_lo]
    picked += rng.sample(ungrounded, min(20, len(ungrounded)))
    seen, out = set(), []
    for r in picked:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    write_jsonl(DEMO_BANK, out)
