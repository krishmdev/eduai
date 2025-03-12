"""Synthetic students and item pools for the simulation experiments.

Modelling assumptions (stated, not measured):
  - true ability theta* ~ N(0, 1), unit offsets ~ N(0, 0.5);
  - responses follow the same model the system uses, p = c + (1 - c) sigmoid(theta_unit - b*),
    with a +MASTERY_BONUS shift on LOs the student has mastered;
  - initial LO mastery is more likely for students strong in that unit, so mastery is correlated
    within a unit (this is what evidence sharing can exploit);
  - in practice mode an unmastered LO is learned with a probability that peaks when the attempt's
    success probability is near 0.7 (a "desirable difficulty" assumption).
True item difficulty b* ~ N(label b, 0.5), where label b is -1/0/+1 for easy/medium/hard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from eduai.curriculum.taxonomy import Taxonomy
from eduai.data.difficulty import LABEL_B, LABELS
from eduai.kt import irt
from eduai.kt.policy import Candidate

MASTERY_BONUS = 1.5
LEARN_MAX = 0.3
LEARN_PEAK_P = 0.7
LEARN_WIDTH = 0.15


@dataclass
class SimItem:
    item_id: str
    lo_id: str
    unit_id: str
    label: str
    b_label: float
    b_true: float

    def candidate(self, b: float | None = None) -> Candidate:
        return Candidate(self.item_id, self.lo_id, self.unit_id, self.b_label if b is None else b)


def make_pool(tax: Taxonomy, subject: str, per_lo: int, rng: np.random.Generator) -> list[SimItem]:
    items = []
    for lo in tax.objectives_for(subject):
        for k in range(per_lo):
            label = LABELS[k % 3]
            items.append(
                SimItem(
                    f"{lo.id}#{k}",
                    lo.id,
                    lo.unit_id,
                    label,
                    LABEL_B[label],
                    float(rng.normal(LABEL_B[label], 0.5)),
                )
            )
    return items


@dataclass
class Student:
    theta: float
    unit_offset: dict[str, float]
    mastered: dict[str, bool]

    def ability(self, unit: str) -> float:
        return self.theta + self.unit_offset.get(unit, 0.0)

    def p_correct(self, item: SimItem) -> float:
        bonus = MASTERY_BONUS if self.mastered.get(item.lo_id, False) else 0.0
        return float(irt.p_correct(self.ability(item.unit_id) + bonus, item.b_true))

    def answer(self, item: SimItem, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.p_correct(item))

    def maybe_learn(self, item: SimItem, p: float, rng: np.random.Generator) -> bool:
        if self.mastered.get(item.lo_id, False):
            return False
        prob = LEARN_MAX * math.exp(-((p - LEARN_PEAK_P) ** 2) / (2 * LEARN_WIDTH**2))
        if rng.random() < prob:
            self.mastered[item.lo_id] = True
            return True
        return False

    def mastery_fraction(self) -> float:
        return sum(self.mastered.values()) / max(len(self.mastered), 1)


def make_student(
    tax: Taxonomy, subject: str, rng: np.random.Generator, mastery_shift: float = -1.2
) -> Student:
    theta = float(rng.normal(0, 1))
    units = tax.subjects[subject].units
    offsets = {u.id: float(rng.normal(0, 0.5)) for u in units}
    mastered = {}
    for u in units:
        p_m = 1.0 / (1.0 + math.exp(-(theta + offsets[u.id] + mastery_shift)))
        for lo in u.objectives:
            mastered[lo.id] = bool(rng.random() < p_m)
    return Student(theta, offsets, mastered)
