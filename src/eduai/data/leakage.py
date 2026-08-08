"""Passage and question overlap between splits: 8-gram containment and Q+A cosine."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

_W = re.compile(r"[a-z0-9]+")
NGRAM = 8
CONTAINMENT_MAX = 0.5
QA_COS_MAX = 0.88


def shingles(text: str, n: int = NGRAM) -> set[str]:
    w = _W.findall(text.lower())
    return {" ".join(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


class ShingleIndex:
    def __init__(self, texts: Sequence[str], n: int = NGRAM):
        self.n = n
        self.index: dict[str, set[int]] = defaultdict(set)
        for i, t in enumerate(texts):
            for g in shingles(t, n):
                self.index[g].add(i)

    def best(self, text: str) -> tuple[float, int | None]:
        """Largest share of `text`'s n-grams found in a single indexed text, and that text's index."""
        s = shingles(text, self.n)
        if not s:
            return 0.0, None
        c = Counter(j for g in s for j in self.index.get(g, ()))
        if not c:
            return 0.0, None
        j, k = c.most_common(1)[0]
        return k / len(s), j


def containment_pairs(
    texts: Sequence[str], threshold: float = CONTAINMENT_MAX, n: int = NGRAM
) -> list[tuple[int, int]]:
    """Pairs (i, j) where one passage's n-grams are at least `threshold` contained in the other."""
    idx = ShingleIndex(texts, n)
    sh = [shingles(t, n) for t in texts]
    pairs = []
    for i, s in enumerate(sh):
        if not s:
            continue
        c = Counter(j for g in s for j in idx.index.get(g, ()) if j != i)
        for j, k in c.items():
            if k / len(s) >= threshold and i < j:
                pairs.append((i, j))
            elif k / len(s) >= threshold and j < i:
                pairs.append((j, i))
    return sorted(set(pairs))


def norm_answer(a: str) -> str:
    tokens = _W.findall(a.lower())
    if tokens and tokens[0] in ("the", "a", "an"):
        tokens = tokens[1:]
    return " ".join(tokens)


def same_answer_qa_leaks(
    q_vecs: np.ndarray,
    q_answers: Sequence[str],
    ref_vecs: np.ndarray,
    ref_answers: Sequence[str],
    threshold: float = QA_COS_MAX,
) -> list[tuple[int, int, float]]:
    """(query idx, ref idx, cos) where the answers match and Q+A cosine >= threshold."""
    sims = q_vecs @ ref_vecs.T
    ra = [norm_answer(a) for a in ref_answers]
    out = []
    for i, a in enumerate(q_answers):
        na = norm_answer(a)
        for j in np.nonzero(sims[i] >= threshold)[0]:
            if ra[j] == na:
                out.append((i, int(j), float(sims[i, j])))
    return out
