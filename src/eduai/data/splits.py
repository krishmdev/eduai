"""Leak-free split assignment.

Items are grouped with union-find over three kinds of links:
  - same support passage (grounded items only; blank passages never link),
  - same normalized question+answer text,
  - question+answer embedding cosine above a threshold (near-paraphrases).
Every group then lands in exactly one split. SciQ's own test split is kept intact for evaluation,
so a group touching test goes to test, else valid, else train; train items pulled out this way
are reported in the data card.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from eduai.data.sciq import SciqItem, passage_hash, qa_hash

NEAR_DUP_COSINE = 0.92
PRIORITY = {"test": 0, "valid": 1, "train": 2}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass
class GroupingStats:
    passage_links: int
    qa_links: int
    cosine_links: int
    groups: int
    largest_group: int
    reassigned: Counter


def near_duplicate_pairs(
    vecs: np.ndarray, threshold: float = NEAR_DUP_COSINE, block: int = 2048
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    n = len(vecs)
    for start in range(0, n, block):
        sims = vecs[start : start + block] @ vecs.T
        rows, cols = np.nonzero(sims > threshold)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            i = start + r
            if c > i:
                pairs.append((i, c))
    return pairs


def group_items(
    items: list[SciqItem],
    qa_vecs: np.ndarray | None,
    threshold: float = NEAR_DUP_COSINE,
    containment: float | None = None,
):
    """containment: if set, also link grounded items whose passages share >= this fraction of
    8-grams. Off by default because the released v1 adapter was trained on splits built without it;
    `eduai data build --containment-link` produces the v2 grouping."""
    uf = UnionFind(len(items))
    passage_links = qa_links = cos_links = 0
    first_by_passage: dict[str, int] = {}
    first_by_qa: dict[str, int] = {}
    for i, it in enumerate(items):
        ph = passage_hash(it.support) if it.grounded else None
        if ph is not None:
            if ph in first_by_passage:
                passage_links += uf.union(first_by_passage[ph], i)
            else:
                first_by_passage[ph] = i
        qh = qa_hash(it.question, it.correct)
        if qh in first_by_qa:
            qa_links += uf.union(first_by_qa[qh], i)
        else:
            first_by_qa[qh] = i
    if qa_vecs is not None:
        for a, b in near_duplicate_pairs(qa_vecs, threshold):
            cos_links += uf.union(a, b)
    if containment is not None:
        from eduai.data.leakage import containment_pairs

        grounded = [i for i, it in enumerate(items) if it.grounded]
        for a, b in containment_pairs([items[i].support for i in grounded], containment):
            passage_links += uf.union(grounded[a], grounded[b])
    roots = [uf.find(i) for i in range(len(items))]
    sizes = Counter(roots)
    return roots, (passage_links, qa_links, cos_links, len(sizes), max(sizes.values()))


def assign_splits(items: list[SciqItem], roots: list[int]) -> tuple[list[str], Counter]:
    members: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(roots):
        members[r].append(i)
    final = [it.split for it in items]
    moved: Counter = Counter()
    for idx in members.values():
        target = min((items[i].split for i in idx), key=PRIORITY.__getitem__)
        for i in idx:
            if items[i].split != target:
                moved[f"{items[i].split}->{target}"] += 1
                final[i] = target
    return final, moved


def leakage(items: list[SciqItem], roots: list[int], final: list[str]) -> int:
    """Number of groups that still span more than one split (should be 0)."""
    spans: dict[int, set[str]] = defaultdict(set)
    for r, s in zip(roots, final, strict=True):
        spans[r].add(s)
    return sum(1 for v in spans.values() if len(v) > 1)
