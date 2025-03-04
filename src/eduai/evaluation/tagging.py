"""Evaluate the curriculum tagger on the hand-labelled gold set and calibrate tau."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eduai.curriculum.tagger import CALIBRATION_PATH, Tagger
from eduai.curriculum.taxonomy import Taxonomy
from eduai.data.sciq import read_jsonl

TAU_PERCENTILE = 20


@dataclass
class ModeResult:
    mode: str
    n_in: int
    top1: float
    top1_lenient: float
    top3: float
    subject_acc: float
    tau: float
    in_aligned: float
    off_rejected: float
    tau_cv: float
    in_aligned_cv: float
    off_rejected_cv: float


def gold_text(row: dict) -> str:
    return f"{row['question']} Answer: {row['answer']}"


def calibrate_tau(gold_scores: np.ndarray) -> float:
    return float(np.percentile(gold_scores, TAU_PERCENTILE))


def evaluate_mode(tagger: Tagger, gold: list[dict], seed: int = 0) -> ModeResult:
    texts = [gold_text(r) for r in gold]
    results = tagger.tag_texts(texts)
    in_idx = [i for i, r in enumerate(gold) if r["lo_id"]]
    off_idx = [i for i, r in enumerate(gold) if not r["lo_id"]]

    def ok(i: int, lenient: bool) -> bool:
        accept = {gold[i]["lo_id"], *(gold[i]["alt_lo_ids"] if lenient else [])}
        return results[i].lo_id in accept

    top1 = np.mean([ok(i, False) for i in in_idx])
    top1_l = np.mean([ok(i, True) for i in in_idx])
    top3 = np.mean([gold[i]["lo_id"] in results[i].top3 for i in in_idx])
    subj = np.mean([results[i].subject == gold[i]["subject"] for i in in_idx])

    gold_lo_scores = tagger.score_pair([texts[i] for i in in_idx], [gold[i]["lo_id"] for i in in_idx])
    top_scores = np.array([r.score for r in results])
    tau = calibrate_tau(gold_lo_scores)

    # Aligned = gold LO in top 3 and top score >= tau, the same rule the generation validator uses.
    def aligned_rate(idx: list[int], t: float) -> float:
        return float(np.mean([gold[i]["lo_id"] in results[i].top3 and top_scores[i] >= t for i in idx]))

    def off_reject_rate(idx: list[int], t: float) -> float:
        return float(np.mean([top_scores[i] < t for i in idx])) if idx else float("nan")

    # 2-fold cross-fitting so the reported aligned/off-curriculum rates don't use tau fit on the
    # same items.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(gold))
    folds = [set(perm[: len(gold) // 2].tolist()), set(perm[len(gold) // 2 :].tolist())]
    in_hits, off_hits, taus = [], [], []
    pos = {i: k for k, i in enumerate(in_idx)}
    for f in range(2):
        fit = [i for i in in_idx if i not in folds[f]]
        t = calibrate_tau(gold_lo_scores[[pos[i] for i in fit]])
        taus.append(t)
        in_hits += [
            gold[i]["lo_id"] in results[i].top3 and top_scores[i] >= t for i in in_idx if i in folds[f]
        ]
        off_hits += [top_scores[i] < t for i in off_idx if i in folds[f]]
    return ModeResult(
        mode=tagger.mode,
        n_in=len(in_idx),
        top1=float(top1),
        top1_lenient=float(top1_l),
        top3=float(top3),
        subject_acc=float(subj),
        tau=tau,
        in_aligned=aligned_rate(in_idx, tau),
        off_rejected=off_reject_rate(off_idx, tau),
        tau_cv=float(np.mean(taus)),
        in_aligned_cv=float(np.mean(in_hits)),
        off_rejected_cv=float(np.mean(off_hits)) if off_hits else float("nan"),
    )


def run(taxonomy: Taxonomy, gold_path: Path, taggers: list[Tagger]) -> list[ModeResult]:
    gold = read_jsonl(gold_path)
    return [evaluate_mode(t, gold) for t in taggers]


def save_calibration(results: list[ModeResult], path: Path = CALIBRATION_PATH) -> None:
    cal = {
        r.mode: {"tau": round(r.tau, 4), "percentile": TAU_PERCENTILE, "n_gold_in_curriculum": r.n_in}
        for r in results
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=2) + "\n")


def render_markdown(results: list[ModeResult], gold: list[dict], manifest_path: str) -> str:
    n_off = sum(1 for r in gold if not r["lo_id"])
    low = sum(1 for r in gold if r["confidence"] == "low")
    lines = [
        "# Curriculum tagger evaluation",
        "",
        f"Gold set: {len(gold)} SciQ items, {len(gold) - n_off} mapped to a learning objective and "
        f"{n_off} marked off-curriculum. Labels were written by an AI labeling pass that did not see "
        f"the tagger's output ({low} flagged low-confidence); they still need a human spot-check.",
        "",
        "Text tagged: question plus correct answer. Top-1 strict counts only the primary gold LO; "
        "lenient also accepts the listed alternates. Aligned means the gold LO is in the top 3 and "
        f"the top score is at least tau (the {TAU_PERCENTILE}th percentile of gold-LO scores). "
        "The CV columns fit tau on one half of the gold set and score the other half.",
        "",
        "| Mode | Top-1 strict | Top-1 lenient | Top-3 | Subject gate | tau | Aligned (CV) | Off-curriculum rejected (CV) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.mode} | {r.top1:.1%} | {r.top1_lenient:.1%} | {r.top3:.1%} | {r.subject_acc:.1%} | "
            f"{r.tau:.3f} | {r.in_aligned_cv:.1%} | {r.off_rejected_cv:.1%} |"
        )
    lines += ["", f"Run manifest: `{manifest_path}`", ""]
    return "\n".join(lines)
