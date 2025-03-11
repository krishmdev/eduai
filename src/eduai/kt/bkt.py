"""Per-learning-objective Bayesian Knowledge Tracing.

Guess and slip shift with item difficulty so that a hard item answered correctly is stronger
evidence than an easy one. Parameters per LO can be fit by EM (Baum-Welch for the two-state HMM)
from response sequences. Evidence can be shared, damped, with similar LOs in the same unit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-6


@dataclass
class BKTParams:
    p_init: float = 0.2
    p_learn: float = 0.12
    guess: float = 0.25
    slip: float = 0.10


def adjusted_guess_slip(params: BKTParams, b: float, scale: float = 0.08) -> tuple[float, float]:
    """Harder items (larger b): less guessing, more slipping. b is on the IRT scale (~ -2..2)."""
    g = min(max(params.guess - scale * b, 0.05), 0.45)
    s = min(max(params.slip + scale * b, 0.02), 0.40)
    return g, s


def posterior_mastery(prior: float, correct: bool, g: float, s: float) -> float:
    if correct:
        num = prior * (1.0 - s)
        den = num + (1.0 - prior) * g
    else:
        num = prior * s
        den = num + (1.0 - prior) * (1.0 - g)
    return num / max(den, EPS)


def step(
    prior: float, correct: bool, params: BKTParams, b: float = 0.0, learning: bool = True
) -> tuple[float, float]:
    """Returns (posterior before learning, next prior). Learning is off in assessment mode."""
    g, s = adjusted_guess_slip(params, b)
    post = posterior_mastery(prior, correct, g, s)
    nxt = post + (1.0 - post) * params.p_learn if learning else post
    return post, nxt


def p_correct(mastery: float, params: BKTParams, b: float = 0.0) -> float:
    g, s = adjusted_guess_slip(params, b)
    return mastery * (1.0 - s) + (1.0 - mastery) * g


@dataclass
class NeighborGraph:
    """LO -> [(neighbor LO, similarity)] within the same unit."""

    edges: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    damping: float = 0.3
    # bge-small cosines between objectives in one unit are compressed (median ~0.79, all > 0.6),
    # so 0.8 keeps roughly the closer half of pairs instead of linking everything.
    min_sim: float = 0.8

    def neighbors(self, lo: str) -> list[tuple[str, float]]:
        return [(j, s) for j, s in self.edges.get(lo, []) if s > self.min_sim]


class BKTTracker:
    def __init__(
        self,
        params: dict[str, BKTParams] | None = None,
        default: BKTParams | None = None,
        graph: NeighborGraph | None = None,
        share: bool = True,
    ):
        self.params = params or {}
        self.default = default or BKTParams()
        self.graph = graph or NeighborGraph()
        self.share = share
        self.mastery: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def p(self, lo: str) -> BKTParams:
        return self.params.get(lo, self.default)

    def get(self, lo: str) -> float:
        return self.mastery.get(lo, self.p(lo).p_init)

    def update(self, lo: str, correct: bool, b: float = 0.0, learning: bool = True) -> float:
        prior = self.get(lo)
        post, nxt = step(prior, correct, self.p(lo), b, learning)
        self.mastery[lo] = nxt
        self.counts[lo] = self.counts.get(lo, 0) + 1
        if self.share:
            # Damped evidence sharing: P_j += damping * sim * (post_i - prior_i).
            delta = post - prior
            for j, sim in self.graph.neighbors(lo):
                pj = self.get(j) + self.graph.damping * sim * delta
                self.mastery[j] = min(max(pj, EPS), 1.0 - EPS)
        return nxt


def fit_em(
    sequences: Sequence[Sequence[int]], init: BKTParams | None = None, iters: int = 50, tol: float = 1e-5
) -> tuple[BKTParams, list[float]]:
    """Baum-Welch for BKT (no forgetting). sequences: lists of 0/1 responses on one LO.

    Returns the fitted parameters and the log-likelihood trace (non-decreasing up to tolerance).
    """
    prm = init or BKTParams()
    pi, T, g, s = prm.p_init, prm.p_learn, prm.guess, prm.slip
    trace: list[float] = []
    for _ in range(iters):
        # Emission: state 0 = unlearned, 1 = learned.
        A = np.array([[1 - T, T], [0.0, 1.0]])
        num_pi = 0.0
        num_T = den_T = 0.0
        num_g = den_g = num_s = den_s = 0.0
        ll = 0.0
        for seq in sequences:
            if not len(seq):
                continue
            obs = np.asarray(seq, dtype=int)
            n = len(obs)
            B = np.where(obs[:, None] == 1, np.array([g, 1 - s]), np.array([1 - g, s]))
            alpha = np.zeros((n, 2))
            scale = np.zeros(n)
            alpha[0] = np.array([1 - pi, pi]) * B[0]
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0]
            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ A) * B[t]
                scale[t] = alpha[t].sum()
                alpha[t] /= scale[t]
            beta = np.ones((n, 2))
            for t in range(n - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / scale[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            ll += float(np.log(scale).sum())
            num_pi += gamma[0, 1]
            for t in range(n - 1):
                xi = alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]
                xi /= xi.sum()
                num_T += xi[0, 1]
                den_T += gamma[t, 0]
            num_g += float((gamma[:, 0] * (obs == 1)).sum())
            den_g += float(gamma[:, 0].sum())
            num_s += float((gamma[:, 1] * (obs == 0)).sum())
            den_s += float(gamma[:, 1].sum())
        trace.append(ll)
        n_seq = sum(1 for s_ in sequences if len(s_))
        pi = _clip(num_pi / max(n_seq, 1), 0.01, 0.99)
        T = _clip(num_T / max(den_T, EPS), 0.005, 0.6)
        # Keep g < 0.5 and s < 0.5 so the "learned" state stays the one with more correct answers.
        g = _clip(num_g / max(den_g, EPS), 0.01, 0.49)
        s = _clip(num_s / max(den_s, EPS), 0.01, 0.49)
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < tol:
            break
    return BKTParams(pi, T, g, s), trace


def _clip(x: float, lo: float, hi: float) -> float:
    if math.isnan(x):
        return lo
    return min(max(x, lo), hi)
