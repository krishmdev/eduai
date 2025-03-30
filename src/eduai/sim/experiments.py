"""Simulation experiments A-D over synthetic students (fixed seeds).

A. Assessment: RMSE of theta-hat vs test length for adaptive (Fisher information), random and a
   fixed form; plus calibration of the EAP posterior under the SD < 0.3 stopping rule.
B. Practice: true mastery over time and steps to 80% mastery, adaptive vs random vs round-robin.
C. Item calibration: RMSE of Elo b-hat vs responses per item.
D. Evidence sharing on vs off: Brier score of BKT mastery estimates against true mastery.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from eduai.curriculum.neighbors import load_graph
from eduai.curriculum.taxonomy import Taxonomy
from eduai.kt import irt
from eduai.kt.bkt import BKTParams, BKTTracker, fit_em
from eduai.kt.elo import DEFAULT_CALIBRATION, InMemoryItemStore, ItemCalibration
from eduai.kt.state import new_session
from eduai.sim.students import SimItem, Student, make_pool, make_student

SUBJECT = "BIO"


@dataclass
class SimConfig:
    students: int = 500
    seed: int = 20260923
    subject: str = SUBJECT
    per_lo: int = 12
    a_lengths: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 60)
    a_max_stop: int = 80
    sd_stops: tuple[float, ...] = (0.3, 0.4, 0.5)
    b_steps: int = 300
    c_students: int = 2000
    c_items_per_student: int = 20
    c_pool: int = 200


def _unit_weights(tax: Taxonomy, subject: str) -> dict[str, float]:
    return {u.id: u.weight for u in tax.subjects[subject].units}


def _los(tax: Taxonomy, subject: str) -> list[tuple[str, str]]:
    return [(o.id, o.unit_id) for o in tax.objectives_for(subject)]


def _by_lo(pool: list[SimItem]) -> dict[str, list[SimItem]]:
    out: dict[str, list[SimItem]] = defaultdict(list)
    for it in pool:
        out[it.lo_id].append(it)
    return out


# -- A: assessment -------------------------------------------------------------------------------
def run_assessment(
    tax: Taxonomy,
    cfg: SimConfig,
    policy: str,
    max_items: int,
    sd_stop: float | None,
    b_known: str = "true",
    rng_seed: int = 0,
) -> list[dict]:
    """policy: adaptive (blueprint + Fisher), adaptive-free (Fisher over whole pool), random, fixed."""
    rng = np.random.default_rng(cfg.seed + rng_seed)
    pool = make_pool(tax, cfg.subject, cfg.per_lo, np.random.default_rng(cfg.seed))
    by_lo = _by_lo(pool)
    los = _los(tax, cfg.subject)
    weights = _unit_weights(tax, cfg.subject)
    fixed_rng = np.random.default_rng(cfg.seed + 99)
    fixed_form = [pool[i] for i in fixed_rng.permutation(len(pool))[:max_items]]
    results = []
    for i in range(cfg.students):
        # Students come from their own seeded stream so every policy sees the same cohort.
        stu = make_student(tax, cfg.subject, np.random.default_rng([cfg.seed, 11, i]))
        # Assessment measures the unidimensional theta; mastery bonus is off so theta* is well defined.
        stu.mastered = dict.fromkeys(stu.mastered, False)
        stu.unit_offset = dict.fromkeys(stu.unit_offset, 0.0)
        st = new_session(
            "assessment",
            cfg.subject,
            max_items,
            weights,
            los,
            share=False,
            sd_stop=sd_stop if sd_stop is not None else -1.0,
        )
        bs_used: list[float] = []
        trajectory = []
        seen: set[str] = set()
        for t in range(max_items):
            if sd_stop is not None and st.done():
                break
            if policy == "adaptive":
                lo = st.choose_lo(rng)
                cands = [it.candidate(it.b_true if b_known == "true" else None) for it in by_lo[lo]]
                cand = st.choose_item(cands)
                if cand is None:
                    cands = [
                        it.candidate(it.b_true if b_known == "true" else None)
                        for it in pool
                        if it.item_id not in seen
                    ]
                    cand = st.choose_item(cands)
            elif policy == "adaptive-free":
                cands = [
                    it.candidate(it.b_true if b_known == "true" else None)
                    for it in pool
                    if it.item_id not in seen
                ]
                cand = st.choose_item(cands)
            elif policy == "random":
                choices = [it for it in pool if it.item_id not in seen]
                pick = choices[int(rng.integers(len(choices)))]
                cand = pick.candidate(pick.b_true if b_known == "true" else None)
            elif policy == "fixed":
                pick = fixed_form[t]
                cand = pick.candidate(pick.b_true if b_known == "true" else None)
            else:
                raise ValueError(policy)
            item = next(i for i in by_lo[cand.lo_id] if i.item_id == cand.item_id)
            y = stu.answer(item, rng)
            st.record(cand, y)
            seen.add(cand.item_id)
            bs_used.append(cand.b)
            post = st.posterior()
            trajectory.append((post.mean, post.sd))
        post = st.posterior()
        results.append(
            {
                "theta": stu.theta,
                "theta_hat": post.mean,
                "sd": post.sd,
                "length": len(st.responses),
                "info_se": irt.info_se(post.mean, bs_used) if bs_used else float("inf"),
                "trajectory": trajectory,
            }
        )
    return results


def paired_ci(d: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of paired differences with a 95% bootstrap CI (resampling students)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


def experiment_a(tax: Taxonomy, cfg: SimConfig) -> dict:
    max_len = max(cfg.a_lengths)
    curves = {}
    sq_err: dict[str, dict[int, np.ndarray]] = {}
    for policy in ("adaptive", "adaptive-free", "random", "fixed"):
        res = run_assessment(tax, cfg, policy, max_len, None, rng_seed=1)
        curve = {}
        sq_err[policy] = {}
        for n in cfg.a_lengths:
            err = np.array([r["trajectory"][n - 1][0] - r["theta"] for r in res])
            curve[n] = float(np.sqrt(np.mean(np.square(err))))
            sq_err[policy][n] = err**2
        curves[policy] = curve
    # Same students in every policy, so squared errors can be compared pairwise.
    paired = {
        other: {n: paired_ci(sq_err["adaptive"][n] - sq_err[other][n], seed=cfg.seed) for n in cfg.a_lengths}
        for other in ("random", "fixed")
    }
    label_b = run_assessment(tax, cfg, "adaptive", max_len, None, b_known="label", rng_seed=1)
    curves["adaptive (label b, uncalibrated)"] = {
        n: float(np.sqrt(np.mean([(r["trajectory"][n - 1][0] - r["theta"]) ** 2 for r in label_b])))
        for n in cfg.a_lengths
    }

    stopping = []
    for sd_stop in cfg.sd_stops:
        res = run_assessment(tax, cfg, "adaptive", cfg.a_max_stop, sd_stop, rng_seed=2)
        err = np.array([r["theta_hat"] - r["theta"] for r in res])
        sds = np.array([r["sd"] for r in res])
        lengths = np.array([r["length"] for r in res])
        stopping.append(
            {
                "sd_stop": sd_stop,
                "max_items": cfg.a_max_stop,
                "coverage_95": float(np.mean(np.abs(err) <= 1.96 * sds)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mean_reported_sd": float(np.mean(sds)),
                "mean_info_se": float(np.mean([r["info_se"] for r in res])),
                "mean_length": float(lengths.mean()),
                "median_length": float(np.median(lengths)),
                "stopped_by_precision": float(np.mean(lengths < cfg.a_max_stop)),
            }
        )
    # Posterior SD vs 1/sqrt(sum I) at a fixed 20 responses.
    res20 = run_assessment(tax, cfg, "adaptive", 20, None, rng_seed=3)
    at20 = {
        "mean_posterior_sd": float(np.mean([r["sd"] for r in res20])),
        "mean_info_se": float(np.mean([r["info_se"] for r in res20])),
        "rmse": float(np.sqrt(np.mean([(r["theta_hat"] - r["theta"]) ** 2 for r in res20]))),
    }
    return {
        "rmse_by_length": curves,
        "paired_mse_diff_vs_adaptive": paired,
        "stopping": stopping,
        "at_20": at20,
        "max_item_information": float(irt.fisher_information(irt.optimal_offset(), 0.0)),
        "p_star": irt.optimal_p(),
    }


# -- B / D: practice -------------------------------------------------------------------------------
def fit_bkt_params(tax: Taxonomy, cfg: SimConfig, n_students: int = 200) -> tuple[BKTParams, list[float]]:
    """Fit pooled BKT parameters by EM on sequences from a separate random-practice cohort."""
    rng = np.random.default_rng(cfg.seed + 500)
    pool = make_pool(tax, cfg.subject, cfg.per_lo, np.random.default_rng(cfg.seed))
    by_lo = _by_lo(pool)
    seqs = []
    for _ in range(n_students):
        stu = make_student(tax, cfg.subject, rng)
        for items in by_lo.values():
            seq = []
            for _ in range(8):
                it = items[int(rng.integers(len(items)))]
                p = stu.p_correct(it)
                seq.append(int(rng.random() < p))
                stu.maybe_learn(it, p, rng)
            seqs.append(seq)
    return fit_em(seqs, BKTParams(0.3, 0.1, 0.3, 0.15))


def run_practice(
    tax: Taxonomy,
    cfg: SimConfig,
    policy: str,
    params: BKTParams,
    share: bool = True,
    rng_seed: int = 0,
    students: int | None = None,
) -> dict:
    rng = np.random.default_rng(cfg.seed + 1000 + rng_seed)
    pool = make_pool(tax, cfg.subject, cfg.per_lo, np.random.default_rng(cfg.seed))
    by_lo = _by_lo(pool)
    los = _los(tax, cfg.subject)
    lo_ids = [lo for lo, _ in los]
    weights = _unit_weights(tax, cfg.subject)
    graph = load_graph()
    n_students = students or cfg.students
    mastery_curve = np.zeros(cfg.b_steps + 1)
    brier_curve = np.zeros(cfg.b_steps + 1)
    steps_to_80 = []
    finals = []
    for i in range(n_students):
        stu: Student = make_student(tax, cfg.subject, np.random.default_rng([cfg.seed, 12, i]))
        st = new_session("practice", cfg.subject, cfg.b_steps, weights, los, graph=graph, share=share)
        st.bkt = BKTTracker(default=params, graph=graph, share=share)
        st.blueprint.length = cfg.b_steps
        mastery_curve[0] += stu.mastery_fraction()
        brier_curve[0] += np.mean([(st.bkt.get(lo) - stu.mastered[lo]) ** 2 for lo in lo_ids])
        reached = None
        for t in range(cfg.b_steps):
            if policy == "adaptive":
                lo = st.choose_lo(rng)
                cand = st.choose_item([it.candidate() for it in by_lo[lo]])
                if cand is None:
                    st.seen -= {it.item_id for it in by_lo[lo]}
                    cand = st.choose_item([it.candidate() for it in by_lo[lo]])
            elif policy == "random-lo-targeted":
                # Random objective, but the item is still picked for p ~ 0.7 (isolates LO choice).
                lo = lo_ids[int(rng.integers(len(lo_ids)))]
                cand = st.choose_item([it.candidate() for it in by_lo[lo]])
                if cand is None:
                    st.seen -= {it.item_id for it in by_lo[lo]}
                    cand = st.choose_item([it.candidate() for it in by_lo[lo]])
            else:
                lo = lo_ids[int(rng.integers(len(lo_ids)))] if policy == "random" else lo_ids[t % len(lo_ids)]
                items = by_lo[lo]
                cand = items[int(rng.integers(len(items)))].candidate()
            item = next(i for i in by_lo[cand.lo_id] if i.item_id == cand.item_id)
            p = stu.p_correct(item)
            y = bool(rng.random() < p)
            st.record(cand, y)
            stu.maybe_learn(item, p, rng)
            frac = stu.mastery_fraction()
            mastery_curve[t + 1] += frac
            brier_curve[t + 1] += np.mean([(st.bkt.get(lo) - stu.mastered[lo]) ** 2 for lo in lo_ids])
            if reached is None and frac >= 0.8:
                reached = t + 1
        steps_to_80.append(reached if reached is not None else cfg.b_steps + 1)
        finals.append(stu.mastery_fraction())
    steps = np.array(steps_to_80)
    return {
        "final_mastery_per_student": finals,
        "mastery_curve": (mastery_curve / n_students).tolist(),
        "brier_curve": (brier_curve / n_students).tolist(),
        "reached_80": float(np.mean(steps <= cfg.b_steps)),
        "median_steps_to_80": float(np.median(steps)),
        "mean_final_mastery": float(mastery_curve[-1] / n_students),
        "mean_brier": float(np.mean(brier_curve[1:] / n_students)),
    }


def experiment_b(tax: Taxonomy, cfg: SimConfig, params: BKTParams) -> dict:
    out = {
        p: run_practice(tax, cfg, p, params, rng_seed=7)
        for p in ("adaptive", "random-lo-targeted", "random", "round-robin")
    }
    base = np.array(out["adaptive"].pop("final_mastery_per_student"))
    for name in ("random-lo-targeted", "random", "round-robin"):
        other = np.array(out[name].pop("final_mastery_per_student"))
        out[name]["adaptive_minus_this"] = paired_ci(base - other, seed=cfg.seed)
    return out


def experiment_d(tax: Taxonomy, cfg: SimConfig, params: BKTParams) -> dict:
    out = {
        "sharing_on": run_practice(tax, cfg, "adaptive", params, share=True, rng_seed=9),
        "sharing_off": run_practice(tax, cfg, "adaptive", params, share=False, rng_seed=9),
    }
    for v in out.values():
        v.pop("final_mastery_per_student")
    return out


# -- C: item calibration -------------------------------------------------------------------------
def run_calibration(tax: Taxonomy, cfg: SimConfig, cal: ItemCalibration) -> dict:
    """Students answer random items; each response calibrates b using the student's pre-response
    EAP mean (computed with the effective b the store serves). Students' theta is unknown."""
    rng = np.random.default_rng(cfg.seed + 3000)
    pool = make_pool(tax, cfg.subject, cfg.per_lo, np.random.default_rng(cfg.seed))
    idx = rng.permutation(len(pool))[: cfg.c_pool]
    items = [pool[i] for i in idx]
    store = InMemoryItemStore({it.item_id: it.b_label for it in items}, cal)
    checkpoints = [0, 5, 10, 20, 40, 80, 160]

    def rmse(effective: bool) -> float:
        get = store.effective_b if effective else (lambda i: store.b[i])
        return float(np.sqrt(np.mean([(get(it.item_id) - it.b_true) ** 2 for it in items])))

    raw_at = {0: rmse(False)}
    eff_at = {0: rmse(True)}
    next_cp = 1
    for _ in range(cfg.c_students):
        theta = float(rng.normal(0, 1))
        eap = irt.EAPGrid()
        for j in rng.permutation(len(items))[: cfg.c_items_per_student]:
            it = items[j]
            b_served = store.effective_b(it.item_id)
            theta_pre = eap.posterior().mean
            y = bool(rng.random() < float(irt.p_correct(theta, it.b_true)))
            store.calibrate(it.item_id, theta_pre, y)
            eap.update(b_served, y)
        mean_n = float(np.mean(list(store.n.values())))
        while next_cp < len(checkpoints) and mean_n >= checkpoints[next_cp]:
            raw_at[checkpoints[next_cp]] = rmse(False)
            eff_at[checkpoints[next_cp]] = rmse(True)
            next_cp += 1
    return {
        "k0": cal.k0,
        "a": cal.a,
        "min_n": cal.min_n,
        "rmse_raw": raw_at,
        "rmse_effective": eff_at,
        "responses_per_item_final": float(np.mean(list(store.n.values()))),
    }


def experiment_c(tax: Taxonomy, cfg: SimConfig) -> dict:
    main = run_calibration(tax, cfg, DEFAULT_CALIBRATION)
    sweep = [
        run_calibration(tax, cfg, ItemCalibration(k0=k, a=DEFAULT_CALIBRATION.a, min_n=0))
        for k in (0.1, 0.15, 0.3, 0.6)
    ]
    pool = make_pool(tax, cfg.subject, cfg.per_lo, np.random.default_rng(cfg.seed))
    idx = np.random.default_rng(cfg.seed + 3000).permutation(len(pool))[: cfg.c_pool]
    label_rmse = float(np.sqrt(np.mean([(pool[i].b_label - pool[i].b_true) ** 2 for i in idx])))
    return {
        "rmse_by_responses": main["rmse_effective"],
        "raw_rmse_by_responses": main["rmse_raw"],
        "config": {
            "k0": DEFAULT_CALIBRATION.k0,
            "a": DEFAULT_CALIBRATION.a,
            "min_n": DEFAULT_CALIBRATION.min_n,
        },
        "k0_sweep": [{"k0": r["k0"], "rmse_raw": r["rmse_raw"]} for r in sweep],
        "label_only_rmse": label_rmse,
        "items": cfg.c_pool,
        "responses_per_item_final": main["responses_per_item_final"],
    }


def run_all(tax: Taxonomy, cfg: SimConfig) -> dict:
    t0 = time.time()
    params, trace = fit_bkt_params(tax, cfg)
    out = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "bkt_em": {"params": params.__dict__, "loglik_trace": trace},
        "A": experiment_a(tax, cfg),
        "B": experiment_b(tax, cfg, params),
        "C": experiment_c(tax, cfg),
        "D": experiment_d(tax, cfg, params),
    }
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
