"""Choose the next learning objective and item.

LO choice (both modes): UCB-weighted need, restricted to units still under their blueprint quota.
Item choice:
  - assessment: maximize Fisher information at the current EAP estimate. For the guessing-floor
    model that's the item whose predicted p is closest to p* ~= 0.683, not 0.5.
  - practice: target p = 0.7. This is a pedagogical "desirable difficulty" choice, not an
    information-maximizing one; it happens to sit close to p* but is chosen for a different reason.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from eduai.kt import irt

PRACTICE_TARGET_P = 0.7


@dataclass
class Candidate:
    item_id: str
    lo_id: str
    unit_id: str
    b: float


@dataclass
class Blueprint:
    weights: dict[str, float]
    length: int

    def cap(self, unit: str) -> int:
        return max(1, math.ceil(self.weights.get(unit, 0.0) * self.length))


def lo_score(weight: float, mastery: float, n_lo: int, t: int, c_ucb: float = 0.3) -> float:
    return weight * (1.0 - mastery) + c_ucb * math.sqrt(math.log(t + 1) / (n_lo + 1))


def choose_lo(
    los: Iterable[tuple[str, str]],
    mastery: Callable[[str], float],
    lo_counts: dict[str, int],
    unit_counts: dict[str, int],
    blueprint: Blueprint,
    t: int,
    rng: np.random.Generator | None = None,
) -> str | None:
    best, best_score = None, -math.inf
    for lo, unit in los:
        if unit_counts.get(unit, 0) >= blueprint.cap(unit):
            continue
        s = lo_score(blueprint.weights.get(unit, 0.0), mastery(lo), lo_counts.get(lo, 0), t)
        if rng is not None:
            s += 1e-6 * rng.random()
        if s > best_score:
            best, best_score = lo, s
    return best


def choose_item_assessment(
    cands: list[Candidate], theta_hat: float, c: float = irt.GUESS
) -> Candidate | None:
    if not cands:
        return None
    info = irt.fisher_information(theta_hat, np.array([x.b for x in cands]), c)
    return cands[int(np.argmax(info))]


def choose_item_practice(
    cands: list[Candidate], theta_hat: float, target_p: float = PRACTICE_TARGET_P, c: float = irt.GUESS
) -> Candidate | None:
    if not cands:
        return None
    p = irt.p_correct(theta_hat, np.array([x.b for x in cands]), c)
    return cands[int(np.argmin(np.abs(p - target_p)))]


def b_for_target(theta_hat: float, target_p: float, c: float = irt.GUESS) -> float:
    """Item difficulty that gives p = target_p at theta_hat (used when requesting a generated item)."""
    s = (target_p - c) / (1.0 - c)
    s = min(max(s, 1e-4), 1 - 1e-4)
    return theta_hat - math.log(s / (1.0 - s))


def difficulty_label_for_b(b: float) -> str:
    if b < -0.5:
        return "easy"
    if b < 0.5:
        return "medium"
    return "hard"
