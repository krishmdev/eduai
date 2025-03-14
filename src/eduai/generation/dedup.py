"""Novelty against the item bank and memorization against the SFT training stems."""

from __future__ import annotations

import numpy as np

from eduai.curriculum.embedder import Embedder, cached_encode

DUP_COSINE = 0.92


class NoveltyIndex:
    def __init__(
        self,
        embedder: Embedder,
        bank_stems: list[str],
        train_stems: list[str] | None = None,
        threshold: float = DUP_COSINE,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.bank = (
            cached_encode(embedder, bank_stems) if bank_stems else np.zeros((0, embedder.dim), np.float32)
        )
        self.train = (
            cached_encode(embedder, train_stems) if train_stems else np.zeros((0, embedder.dim), np.float32)
        )
        self.accepted: list[np.ndarray] = []

    def _max(self, mat: np.ndarray, v: np.ndarray) -> float:
        return float((mat @ v).max()) if len(mat) else 0.0

    def check(self, item: dict) -> tuple[bool, dict]:
        v = self.embedder.encode([item["stem"]])[0]
        bank_cos = max(self._max(self.bank, v), max((float(a @ v) for a in self.accepted), default=0.0))
        train_cos = self._max(self.train, v)
        ok = bank_cos < self.threshold
        if ok:
            self.accepted.append(v)
        return ok, {
            "max_bank_cos": bank_cos,
            "max_train_cos": train_cos,
            "memorized": float(train_cos >= self.threshold),
        }
