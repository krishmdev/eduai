"""Summaries for the report page. The 1-5 score is illustrative and labelled as simulated."""

from __future__ import annotations

from collections import defaultdict

from eduai.kt.state import SessionState

# Illustrative cut points on the theta scale; not an official AP score conversion.
SCORE_CUTS = (-1.0, -0.3, 0.4, 1.1)


def simulated_score(theta: float) -> int:
    return 1 + sum(theta >= c for c in SCORE_CUTS)


def unit_mastery(state: SessionState, unit_names: dict[str, str]) -> list[dict]:
    by_unit: dict[str, list[str]] = defaultdict(list)
    for lo, unit in state.los:
        by_unit[unit].append(lo)
    rows = []
    for unit, los in by_unit.items():
        answered = [r for r in state.responses if r.unit_id == unit]
        rows.append(
            {
                "unit_id": unit,
                "name": unit_names.get(unit, unit),
                "mastery": sum(state.bkt.get(lo) for lo in los) / len(los),
                "answered": len(answered),
                "correct": sum(r.correct for r in answered),
                "theta": state.elo.theta.get(unit),
            }
        )
    return sorted(rows, key=lambda r: r["unit_id"])


def summary(state: SessionState) -> dict:
    post = state.posterior()
    n = len(state.responses)
    return {
        "answered": n,
        "correct": sum(r.correct for r in state.responses),
        "theta": post.mean,
        "sd": post.sd,
        "ci95": (post.mean - 1.96 * post.sd, post.mean + 1.96 * post.sd),
        "simulated_score": simulated_score(post.mean),
        "stopped_by": (
            "precision" if state.mode == "assessment" and n < state.max_items and state.done() else "length"
        ),
    }
