"""Stationary block bootstrap. Spec §14.2."""

from __future__ import annotations

import numpy as np
import pytest

from prism.analysis.bootstrap import stationary_block_bootstrap_ci, non_overlapping


def test_ci_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, 500)
    res = stationary_block_bootstrap_ci(x, np.mean, block_length=20, n_bootstrap=1000, seed=1)
    assert res.ci_low <= res.point_estimate <= res.ci_high
    assert res.point_estimate == pytest.approx(np.mean(x))


def test_wider_block_length_does_not_crash_and_stays_sane():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    res = stationary_block_bootstrap_ci(x, np.mean, block_length=60, n_bootstrap=500, seed=2)
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
    assert res.ci_low <= res.ci_high


def test_serially_correlated_data_gets_a_wider_ci_than_iid():
    """The whole point of a BLOCK bootstrap: autocorrelated data has fewer
    effective independent observations, so its CI should be wider than an
    i.i.d. series of the same length and marginal variance."""
    rng = np.random.default_rng(0)
    n = 1000
    iid = rng.normal(size=n)

    ar = np.empty(n)
    ar[0] = rng.normal()
    phi = 0.9
    for t in range(1, n):
        ar[t] = phi * ar[t - 1] + rng.normal(0, np.sqrt(1 - phi**2))  # unit marginal variance

    res_iid = stationary_block_bootstrap_ci(iid, np.mean, block_length=20, n_bootstrap=1000, seed=3)
    res_ar = stationary_block_bootstrap_ci(ar, np.mean, block_length=20, n_bootstrap=1000, seed=3)

    width_iid = res_iid.ci_high - res_iid.ci_low
    width_ar = res_ar.ci_high - res_ar.ci_low
    assert width_ar > width_iid, "autocorrelated series should get a wider CI than i.i.d."


def test_non_overlapping_detects_clearly_different_and_rejects_identical():
    rng = np.random.default_rng(0)
    a = stationary_block_bootstrap_ci(rng.normal(0, 1, 500), np.mean, block_length=20, n_bootstrap=1000, seed=1)
    b = stationary_block_bootstrap_ci(rng.normal(5, 1, 500), np.mean, block_length=20, n_bootstrap=1000, seed=2)
    assert non_overlapping(a, b)

    same_data = rng.normal(0, 1, 500)
    c = stationary_block_bootstrap_ci(same_data, np.mean, block_length=20, n_bootstrap=1000, seed=3)
    d = stationary_block_bootstrap_ci(same_data, np.mean, block_length=20, n_bootstrap=1000, seed=4)
    assert not non_overlapping(c, d)


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="at least 2"):
        stationary_block_bootstrap_ci(np.array([1.0]), np.mean, block_length=5, n_bootstrap=10, seed=0)
    with pytest.raises(ValueError, match="ci_level"):
        stationary_block_bootstrap_ci(np.arange(10.0), np.mean, block_length=5, n_bootstrap=10, ci_level=1.5, seed=0)
    with pytest.raises(ValueError, match="1-D"):
        stationary_block_bootstrap_ci(
            np.zeros((5, 2)), np.mean, block_length=5, n_bootstrap=10, seed=0
        )


def test_reproducible_given_the_same_seed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    a = stationary_block_bootstrap_ci(x, np.mean, block_length=20, n_bootstrap=300, seed=42)
    b = stationary_block_bootstrap_ci(x, np.mean, block_length=20, n_bootstrap=300, seed=42)
    np.testing.assert_array_equal(a.replicate_statistics, b.replicate_statistics)
