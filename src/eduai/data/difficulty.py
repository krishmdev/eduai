"""Heuristic difficulty: z-scored mix of distractor similarity, question type and stem length."""

from __future__ import annotations

import re

import numpy as np

HIGHER_ORDER = re.compile(
    r"\b(why|explain|predict|would happen|most likely|best explains|infer|compare|which of the following|"
    r"results? in|because|effect of|cause[sd]?|if)\b",
    re.I,
)
RECALL = re.compile(
    r"\b(is called|are called|what is the name|known as|term for|what type|what kind)\b", re.I
)
LABELS = ("easy", "medium", "hard")
LABEL_B = {"easy": -1.0, "medium": 0.0, "hard": 1.0}


def bloom_score(question: str) -> float:
    score = 0.0
    if HIGHER_ORDER.search(question):
        score += 1.0
    if RECALL.search(question):
        score -= 1.0
    return score


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def raw_scores(questions: list[str], answer_vecs: np.ndarray, distractor_vecs: np.ndarray) -> np.ndarray:
    """answer_vecs: (n, d); distractor_vecs: (n, 3, d), all L2-normalized."""
    sim = np.einsum("nd,nkd->nk", answer_vecs, distractor_vecs).mean(axis=1)
    bloom = np.array([bloom_score(q) for q in questions])
    length = np.log1p(np.array([len(q.split()) for q in questions], dtype=float))
    return 0.5 * _z(sim) + 0.25 * _z(bloom) + 0.25 * _z(length)


def tertile_cuts(eligible_scores: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(eligible_scores, [1 / 3, 2 / 3])
    return float(lo), float(hi)


def label(score: float, cuts: tuple[float, float]) -> str:
    if score < cuts[0]:
        return "easy"
    if score < cuts[1]:
        return "medium"
    return "hard"
