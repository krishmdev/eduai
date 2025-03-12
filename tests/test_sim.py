from eduai.sim import experiments as ex


def _cfg(**kw):
    base = dict(students=80, seed=5, a_lengths=(10, 40), b_steps=200)
    base.update(kw)
    return ex.SimConfig(**base)


def test_adaptive_assessment_beats_random(taxonomy):
    # Pooled over four seeds (1,200 students per policy) so the margin isn't one seed's luck.
    ratios = []
    for seed in (5, 6, 7, 8):
        cfg = _cfg(students=300, seed=seed)
        rmse = {}
        for policy in ("adaptive", "random"):
            res = ex.run_assessment(taxonomy, cfg, policy, 40, None, rng_seed=1)
            per_len = [
                (sum((r["trajectory"][n - 1][0] - r["theta"]) ** 2 for r in res) / len(res)) ** 0.5
                for n in (10, 20, 30, 40)
            ]
            rmse[policy] = sum(per_len) / len(per_len)
        ratios.append(rmse["adaptive"] / rmse["random"])
    assert sum(ratios) / len(ratios) < 0.95


def test_adaptive_practice_beats_random(taxonomy):
    from eduai.kt.bkt import BKTParams

    cfg = _cfg()
    params = BKTParams(0.4, 0.08, 0.45, 0.12)
    adaptive = ex.run_practice(taxonomy, cfg, "adaptive", params, rng_seed=2)
    random = ex.run_practice(taxonomy, cfg, "random", params, rng_seed=2)
    assert adaptive["mean_final_mastery"] > random["mean_final_mastery"] + 0.03


def test_sd_stopping_ends_tests_and_covers(taxonomy):
    cfg = _cfg(students=60)
    res = ex.run_assessment(taxonomy, cfg, "adaptive", 80, 0.5, rng_seed=3)
    assert all(r["sd"] < 0.5 or r["length"] == 80 for r in res)
    cover = sum(abs(r["theta_hat"] - r["theta"]) <= 1.96 * r["sd"] for r in res) / len(res)
    assert cover > 0.85
