"""Held-out eval prompts with independent LO labels and a leakage filter.

1. Candidates: grounded items whose split group landed in SciQ test, excluding the SFT test rows.
   The tagger is NOT used to pick them, so eval targets are not the tagger's own choices.
2. Leakage filter against the SFT train and valid rows the adapter actually saw: drop a candidate
   whose passage has >= 50% 8-gram containment in one training passage, or whose question+answer
   has cosine >= 0.88 with a training item that has the same answer.
3. An independent labeling pass (never shown tagger output) assigns an LO or marks the item
   off-curriculum; off-curriculum items are dropped and the first 150 labeled items are kept.

The valid-split prompts used for model selection come from the same code with split="valid": groups
assigned to the valid split, minus the SFT valid rows, filtered against the SFT train rows only, and
labeled with the tagger's own LO (only tagger-aligned items are candidates).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from eduai.curriculum.taxonomy import Taxonomy
from eduai.data import explain, leakage
from eduai.data.sciq import SciqItem, read_jsonl, write_jsonl
from eduai.prompts import GenerationRequest

PASSAGE_MARK = "Source passage:\n"


def sft_passage(row: dict) -> str:
    user = row["messages"][1]["content"]
    return user.split(PASSAGE_MARK, 1)[1].rsplit("\n\nReturn JSON", 1)[0]


def sft_qa(row: dict) -> tuple[str, str]:
    from eduai.curriculum.tagger import tag_text

    d = json.loads(row["messages"][2]["content"])
    return tag_text(d["stem"], d["choices"][d["answer"]]), d["choices"][d["answer"]]


def candidates(
    data_dir: Path, embedder, n: int, seed: int = 20260805, split: str = "test"
) -> tuple[list[dict], dict]:
    items = [SciqItem.from_dict(d) for d in read_jsonl(data_dir / "items_tagged.jsonl")]
    sft_rows = {m["id"] for m in read_jsonl(data_dir / "sft_meta.jsonl")}
    train_rows = read_jsonl(data_dir / "sft" / "train.jsonl")
    if split == "test":
        train_rows += read_jsonl(data_dir / "sft" / "valid.jsonl")
    pool = [
        it
        for it in items
        if it.grounded
        and it.tags["final_split"] == split
        and it.id not in sft_rows
        and explain.sentences(it.support)
        and (split == "test" or it.tags["aligned"])
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)

    passages = [sft_passage(r) for r in train_rows]
    shingle_idx = leakage.ShingleIndex(passages)
    qa = [sft_qa(r) for r in train_rows]
    ref_vecs = embedder.encode([q for q, _ in qa])
    stats = Counter()
    kept: list[dict] = []
    for it in pool:
        if len(kept) >= n:
            break
        stats["screened"] += 1
        passage = explain.trim_passage(it.support, it.question, it.correct)
        cont, _ = leakage.ShingleIndex.best(shingle_idx, passage)
        if cont >= leakage.CONTAINMENT_MAX:
            stats["dropped_passage_containment"] += 1
            continue
        v = embedder.encode([it.qa_text()])
        if leakage.same_answer_qa_leaks(v, [it.correct], ref_vecs, [a for _, a in qa]):
            stats["dropped_same_answer_qa"] += 1
            continue
        row = {
            "id": it.id,
            "question": it.question,
            "answer": it.correct,
            "passage": passage,
            "difficulty": it.tags["difficulty"],
            "group": it.tags["group"],
            "distractors": it.distractors,
            "max_containment": round(cont, 3),
        }
        if split != "test":
            # Test candidates go to an independent labeler and must not carry tagger output.
            row["tagger_lo_id"] = it.tags["lo_id"]
        kept.append(row)
    stats["kept"] = len(kept)
    stats["pool"] = len(pool)
    return kept, dict(stats)


def build_prompts(
    cands: list[dict],
    labels: dict[str, str | None],
    tax: Taxonomy,
    n: int,
    seed: int,
    stimulus_frac: float = 0.4,
    misconception_frac: float = 0.3,
    label_source: str = "independent",
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    labeled = [c for c in cands if labels.get(c["id"])]
    chosen = labeled[:n]
    k = len(chosen)
    stim = set(rng.sample(range(k), round(stimulus_frac * k)))
    mis = set(rng.sample(range(k), round(misconception_frac * k)))
    rows = []
    for i, c in enumerate(chosen):
        lo = tax.lo(labels[c["id"]])
        req = GenerationRequest(
            subject=lo.subject_name,
            unit=lo.unit_name,
            topic=lo.topic_name,
            lo_id=lo.id,
            lo_text=lo.text,
            difficulty=c["difficulty"],
            format="stimulus" if i in stim else "standard",
            passage=c["passage"],
            target_misconception=rng.choice(c["distractors"]) if i in mis else None,
        )
        rows.append(
            {
                "id": c["id"],
                "request": req.__dict__,
                "group": c["group"],
                "reference": {"question": c["question"], "answer": c["answer"]},
                "label_source": label_source,
            }
        )
    stats = {
        "candidates": len(cands),
        "labeled_in_curriculum": len(labeled),
        "off_curriculum": sum(1 for c in cands if c["id"] in labels and not labels[c["id"]]),
        "used": len(rows),
        "short_of_target": max(0, n - len(rows)),
        "by_subject": dict(Counter(tax.lo(labels[c["id"]]).subject for c in chosen)),
    }
    return rows, stats


def write(rows: list[dict], path: Path) -> None:
    write_jsonl(path, rows)
