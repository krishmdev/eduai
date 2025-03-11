import math

import numpy as np
import pytest

from eduai.kt import irt


@pytest.mark.parametrize("c", [0.0, 0.2, 0.25])
@pytest.mark.parametrize("theta,b", [(-1.3, 0.4), (0.0, 0.0), (1.7, -0.6), (0.5, 2.0)])
def test_derivative_matches_finite_difference(c, theta, b):
    h = 1e-5
    fd = (irt.p_correct(theta + h, b, c) - irt.p_correct(theta - h, b, c)) / (2 * h)
    assert irt.dp_dtheta(theta, b, c) == pytest.approx(float(fd), rel=1e-6)
    p = irt.p_correct(theta, b, c)
    assert irt.fisher_information(theta, b, c) == pytest.approx(float(fd) ** 2 / (p * (1 - p)), rel=1e-5)


@pytest.mark.parametrize("c", [0.0, 0.2, 0.25])
def test_information_peaks_at_analytic_p_star(c):
    theta = 0.3
    bs = np.linspace(-6, 6, 400001)
    info = irt.fisher_information(theta, bs, c)
    b_best = bs[np.argmax(info)]
    p_at_best = float(irt.p_correct(theta, b_best, c))
    assert p_at_best == pytest.approx(irt.optimal_p(c), abs=1e-4)
    assert theta - b_best == pytest.approx(irt.optimal_offset(c), abs=1e-3)


def test_p_star_values():
    assert irt.optimal_p(0.0) == pytest.approx(0.5)
    assert irt.optimal_p(0.25) == pytest.approx((1 + math.sqrt(3)) / 4)
    assert irt.optimal_p(0.25) == pytest.approx(0.683, abs=1e-3)


def test_eap_grid_matches_fine_reference():
    rng = np.random.default_rng(3)
    bs = rng.normal(0, 1, 20)
    ys = rng.random(20) < irt.p_correct(0.8, bs)
    coarse = irt.eap(list(zip(bs, ys, strict=True)))
    fine = irt.eap(list(zip(bs, ys, strict=True)), lo=-8, hi=8, points=20001)
    assert coarse.mean == pytest.approx(fine.mean, abs=1e-3)
    assert coarse.sd == pytest.approx(fine.sd, abs=1e-3)


def test_prior_only_posterior_is_standard_normal():
    post = irt.EAPGrid(lo=-8, hi=8, points=4001).posterior()
    assert post.mean == pytest.approx(0.0, abs=1e-9)
    assert post.sd == pytest.approx(1.0, abs=1e-4)


def test_posterior_sd_and_information_se_differ():
    # The stopping rule uses the posterior SD; 1/sqrt(sum I) is only a comparison and can differ a lot.
    bs = [0.0] * 20
    ys = [True, False] * 10
    post = irt.eap(list(zip(bs, ys, strict=True)))
    se = irt.info_se(post.mean, bs)
    assert post.sd < 1.0 and se < 1.0
    assert abs(post.sd - se) > 0.01


def test_stopping_rule_uses_posterior_sd_only():
    assert irt.should_stop(irt.Posterior(0.0, 0.29, 10), 10, 40)
    assert not irt.should_stop(irt.Posterior(0.0, 0.31, 10), 10, 40)
    assert irt.should_stop(irt.Posterior(0.0, 0.9, 40), 40, 40)
    assert not irt.should_stop(irt.Posterior(0.0, 0.1, 2), 2, 40)
