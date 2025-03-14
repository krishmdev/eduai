"""Online item difficulty calibration on the shared response model.

After a response y to item i by a student whose pre-response EAP estimate is theta:
    p = c + (1 - c) sigmoid(theta - b_i),   b_i <- b_i - K(n_i) (y - p),   K(n) = K0 / (1 + a n)
The estimate lives in the item store (SQLite in the app); sessions only read it. Until an item has
`min_n` responses the label difficulty (easy/medium/hard -> -1/0/+1) is used for selection and
scoring, because a handful of Elo steps is noisier than the label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from eduai.kt.irt import GUESS, p_correct


@dataclass(frozen=True)
class ItemCalibration:
    k0: float = 0.15
    a: float = 0.02
    min_n: int = 5
    c: float = GUESS

    def gain(self, n: int) -> float:
        return self.k0 / (1.0 + self.a * n)

    def step(self, b: float, n: int, theta_pre: float, correct: bool) -> float:
        p = float(p_correct(theta_pre, b, self.c))
        return b - self.gain(n) * ((1.0 if correct else 0.0) - p)

    def effective_b(self, b_hat: float, n: int, b_label: float) -> float:
        return b_label if n < self.min_n else b_hat


DEFAULT_CALIBRATION = ItemCalibration()


class ItemStore(Protocol):
    def item_b(self, item_id: str) -> tuple[float, int, float]:
        """(b_hat, n_responses, b_label)"""
        ...

    def calibrate(
        self, item_id: str, theta_pre: float, correct: bool, cal: ItemCalibration = ...
    ) -> float: ...


class InMemoryItemStore:
    def __init__(self, labels: dict[str, float], cal: ItemCalibration = DEFAULT_CALIBRATION):
        self.cal = cal
        self.b = dict(labels)
        self.label = dict(labels)
        self.n = dict.fromkeys(labels, 0)

    def item_b(self, item_id: str) -> tuple[float, int, float]:
        return self.b[item_id], self.n[item_id], self.label[item_id]

    def effective_b(self, item_id: str) -> float:
        b, n, lab = self.item_b(item_id)
        return self.cal.effective_b(b, n, lab)

    def calibrate(
        self, item_id: str, theta_pre: float, correct: bool, cal: ItemCalibration | None = None
    ) -> float:
        cal = cal or self.cal
        self.b[item_id] = cal.step(self.b[item_id], self.n[item_id], theta_pre, correct)
        self.n[item_id] += 1
        return self.b[item_id]
