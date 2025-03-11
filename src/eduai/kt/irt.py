"""The single response model used everywhere: p(theta, b) = c + (1 - c) * sigmoid(theta - b).

c is the guessing floor for four-option items. Online item calibration (elo.py), the assessment
estimator (EAP on a grid), item selection (Fisher information), and the simulator all use this
same function, so there is one likelihood in the system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GUESS = 0.25
SD_STOP = 0.3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def p_correct(theta, b, c: float = GUESS):
    return c + (1.0 - c) * sigmoid(np.asarray(theta, dtype=float) - np.asarray(b, dtype=float))


def dp_dtheta(theta, b, c: float = GUESS):
    s = sigmoid(np.asarray(theta, dtype=float) - np.asarray(b, dtype=float))
    return (1.0 - c) * s * (1.0 - s)


def fisher_information(theta, b, c: float = GUESS):
    """I(theta) = p'^2 / (p (1 - p)) for a single item."""
    p = p_correct(theta, b, c)
    return dp_dtheta(theta, b, c) ** 2 / (p * (1.0 - p))


def optimal_p(c: float = GUESS) -> float:
    """Observed success probability at which this model's item information peaks.

    Setting dI/d(theta - b) = 0 gives 2p^2 - p - c = 0, so p* = (1 + sqrt(1 + 8c)) / 4.
    That's 0.5 when c = 0 and about 0.683 when c = 0.25.
    """
    return (1.0 + math.sqrt(1.0 + 8.0 * c)) / 4.0


def optimal_offset(c: float = GUESS) -> float:
    """theta - b at which information peaks (b = theta - offset)."""
    ps = optimal_p(c)
    s = (ps - c) / (1.0 - c)
    return math.log(s / (1.0 - s))


@dataclass
class Posterior:
    mean: float
    sd: float
    n: int


class EAPGrid:
    """Expected a posteriori theta on a fixed quadrature grid with a N(mu, sigma^2) prior."""

    def __init__(
        self,
        lo: float = -5.0,
        hi: float = 5.0,
        points: int = 161,
        mu: float = 0.0,
        sigma: float = 1.0,
        c: float = GUESS,
    ):
        self.grid = np.linspace(lo, hi, points)
        self.c = c
        self.log_prior = -0.5 * ((self.grid - mu) / sigma) ** 2
        self.log_post = self.log_prior.copy()
        self.n = 0

    def update(self, b: float, correct: bool) -> Posterior:
        p = p_correct(self.grid, b, self.c)
        self.log_post += np.log(p if correct else 1.0 - p)
        self.n += 1
        return self.posterior()

    def posterior(self) -> Posterior:
        w = np.exp(self.log_post - self.log_post.max())
        w /= w.sum()
        mean = float((w * self.grid).sum())
        var = float((w * (self.grid - mean) ** 2).sum())
        return Posterior(mean, math.sqrt(max(var, 0.0)), self.n)

    def weights(self) -> np.ndarray:
        w = np.exp(self.log_post - self.log_post.max())
        return w / w.sum()


def eap(responses: list[tuple[float, bool]], **kw) -> Posterior:
    est = EAPGrid(**kw)
    for b, y in responses:
        est.update(b, y)
    return est.posterior()


def info_se(theta: float, bs: list[float], c: float = GUESS) -> float:
    """1/sqrt(sum I) at theta. Reported only as a comparison column, never used to stop."""
    total = float(np.sum(fisher_information(theta, np.asarray(bs), c)))
    return 1.0 / math.sqrt(total) if total > 0 else float("inf")


def should_stop(
    post: Posterior, n_items: int, max_items: int, sd_stop: float = SD_STOP, min_items: int = 3
) -> bool:
    """One stopping rule: EAP posterior SD below the threshold, or the item cap."""
    if n_items >= max_items:
        return True
    return n_items >= min_items and post.sd < sd_stop
