"""Novelty against the item bank, source-copy rate, and memorization against SFT training stems.

When an item was generated from a known source (an eval prompt), that source item and every bank
item in its near-duplicate split group are excluded from the novelty comparison; closeness to the
source is reported separately as a source copy instead of a silent duplicate failure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eduai.curriculum.embedder import Embedder, cached_encode

DUP_COSINE = 0.92


class NoveltyIndex:
    def __init__(
        self,
        embedder: Embedder,
        bank_stems: Sequence[str],
        bank_ids: Sequence[str] | None = None,
        bank_groups: Sequence[int | None] | None = None,
        train_stems: Sequence[str] | None = None,
        threshold: float = DUP_COSINE,
    ):
        self.embedder = embedder
        self.threshold = threshold
        empty = np.zeros((0, embedder.dim), np.float32)
        self.bank = cached_encode(embedder, list(bank_stems)) if bank_stems else empty
        self.ids = list(bank_ids or [""] * len(bank_stems))
        self.groups = np.array([(-1 if g is None else g) for g in (bank_groups or [None] * len(bank_stems))])
        self.train = cached_encode(embedder, list(train_stems)) if train_stems else empty
        self.accepted: list[np.ndarray] = []

    def check(
        self,
        item: dict,
        exclude_ids: Sequence[str] = (),
        exclude_group: int | None = None,
        source_stem: str | None = None,
    ) -> tuple[bool, dict]:
        v = self.embedder.encode([item["stem"]])[0]
        mask = np.ones(len(self.ids), dtype=bool)
        if exclude_ids:
            ex = set(exclude_ids)
            mask &= np.array([i not in ex for i in self.ids])
        if exclude_group is not None:
            mask &= self.groups != exclude_group
        sims = self.bank[mask] @ v if mask.any() else np.zeros(0)
        bank_cos = max(
            float(sims.max()) if len(sims) else 0.0, max((float(a @ v) for a in self.accepted), default=0.0)
        )
        train_cos = float((self.train @ v).max()) if len(self.train) else 0.0
        info = {
            "max_bank_cos": bank_cos,
            "max_train_cos": train_cos,
            "memorized": float(train_cos >= self.threshold),
        }
        if source_stem is not None:
            src = self.embedder.encode([source_stem])[0]
            info["source_cos"] = float(src @ v)
            info["source_copy"] = float(info["source_cos"] >= self.threshold)
        ok = bank_cos < self.threshold
        if ok:
            self.accepted.append(v)
        return ok, info
