"""Adaptive session engine shared by the simulator and the web app."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eduai.kt import irt, policy
from eduai.kt.bkt import BKTTracker, NeighborGraph
from eduai.kt.elo import EloModel
from eduai.kt.policy import Blueprint, Candidate

MODES = ("practice", "assessment")


@dataclass
class Response:
    item_id: str
    lo_id: str
    unit_id: str
    b: float
    correct: bool
    choice: str | None = None
    p_pred: float | None = None
    theta_after: float | None = None
    sd_after: float | None = None


@dataclass
class SessionState:
    mode: str
    subject: str
    max_items: int
    blueprint: Blueprint
    los: list[tuple[str, str]]
    elo: EloModel = field(default_factory=EloModel)
    bkt: BKTTracker = field(default_factory=BKTTracker)
    eap: irt.EAPGrid = field(default_factory=irt.EAPGrid)
    responses: list[Response] = field(default_factory=list)
    lo_counts: dict[str, int] = field(default_factory=dict)
    unit_counts: dict[str, int] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    # LO -> last wrong option text picked; feeds misconception-targeted regeneration.
    misconceptions: dict[str, str] = field(default_factory=dict)
    sd_stop: float = irt.SD_STOP

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

    # -- estimates --------------------------------------------------------------------------------
    def theta_for(self, unit_id: str) -> float:
        if self.mode == "assessment":
            return self.eap.posterior().mean
        return self.elo.theta.get(unit_id, self.elo.theta.get(self.subject, 0.0))

    def posterior(self) -> irt.Posterior:
        return self.eap.posterior()

    def done(self) -> bool:
        if self.mode == "assessment":
            return irt.should_stop(self.posterior(), len(self.responses), self.max_items, self.sd_stop)
        return len(self.responses) >= self.max_items

    # -- selection --------------------------------------------------------------------------------
    def choose_lo(
        self, rng: np.random.Generator | None = None, available: set[str] | None = None
    ) -> str | None:
        los = [(lo, u) for lo, u in self.los if available is None or lo in available]
        pick = policy.choose_lo(
            los, self.bkt.get, self.lo_counts, self.unit_counts, self.blueprint, len(self.responses), rng
        )
        if pick is None and los:
            # Every unit hit its cap (possible when the test runs past the blueprint length).
            pick = policy.choose_lo(
                los, self.bkt.get, self.lo_counts, {}, self.blueprint, len(self.responses), rng
            )
        return pick

    def choose_item(self, cands: list[Candidate]) -> Candidate | None:
        fresh = [c for c in cands if c.item_id not in self.seen]
        if not fresh:
            return None
        theta = self.theta_for(fresh[0].unit_id)
        if self.mode == "assessment":
            return policy.choose_item_assessment(fresh, theta)
        return policy.choose_item_practice(fresh, theta)

    def target_b(self, unit_id: str) -> float:
        theta = self.theta_for(unit_id)
        if self.mode == "assessment":
            return theta - irt.optimal_offset()
        return policy.b_for_target(theta, policy.PRACTICE_TARGET_P)

    # -- updates ----------------------------------------------------------------------------------
    def record(
        self,
        cand: Candidate,
        correct: bool,
        choice: str | None = None,
        wrong_text: str | None = None,
        calibrate_items: bool = True,
    ) -> Response:
        learning = self.mode == "practice"
        self.elo.ensure_item(cand.item_id, cand.b)
        b_now = self.elo.b.get(cand.item_id, cand.b)
        p_pred = float(irt.p_correct(self.theta_for(cand.unit_id), b_now))
        self.elo.update(cand.unit_id, cand.item_id, correct, update_item=calibrate_items)
        self.elo.update(self.subject, cand.item_id, correct, update_item=False)
        self.eap.update(b_now, correct)
        self.bkt.update(cand.lo_id, correct, b=b_now, learning=learning)
        self.lo_counts[cand.lo_id] = self.lo_counts.get(cand.lo_id, 0) + 1
        self.unit_counts[cand.unit_id] = self.unit_counts.get(cand.unit_id, 0) + 1
        self.seen.add(cand.item_id)
        if not correct and wrong_text:
            self.misconceptions[cand.lo_id] = wrong_text
        post = self.posterior()
        r = Response(
            cand.item_id, cand.lo_id, cand.unit_id, b_now, correct, choice, p_pred, post.mean, post.sd
        )
        self.responses.append(r)
        return r


def new_session(
    mode: str,
    subject: str,
    max_items: int,
    unit_weights: dict[str, float],
    los: list[tuple[str, str]],
    graph: NeighborGraph | None = None,
    share: bool = True,
    sd_stop: float = irt.SD_STOP,
) -> SessionState:
    return SessionState(
        mode=mode,
        subject=subject,
        max_items=max_items,
        blueprint=Blueprint(unit_weights, max_items),
        los=los,
        bkt=BKTTracker(graph=graph, share=share),
        sd_stop=sd_stop,
    )
