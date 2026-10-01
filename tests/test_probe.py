"""Walk-forward ridge probe. Spec §13.1 (the core step 4a will extend)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.probes.probe import compare_probes, ridge_probe, select_ridge_alpha


@pytest.fixture
def split_dates():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    return dict(
        train_start=idx[0], train_end=idx[599],
        val_start=idx[600], val_end=idx[699],
        test_start=idx[700], test_end=idx[999],
    ), idx


def test_informative_features_beat_noise_on_r2(split_dates):
    bounds, idx = split_dates
    rng = np.random.default_rng(0)
    signal = rng.normal(size=1000)
    target = pd.Series(0.8 * signal + 0.3 * rng.normal(size=1000), index=idx)

    informative = pd.DataFrame({"signal": signal, "noise": rng.normal(size=1000)}, index=idx)
    pure_noise = pd.DataFrame({"n1": rng.normal(size=1000), "n2": rng.normal(size=1000)}, index=idx)

    res_informative = ridge_probe(informative, target, alpha_grid=[0.01, 0.1, 1.0, 10.0], **bounds)
    res_noise = ridge_probe(pure_noise, target, alpha_grid=[0.01, 0.1, 1.0, 10.0], **bounds)

    assert res_informative.r2 > 0.5
    assert res_noise.r2 < 0.1
    assert res_informative.r2 > res_noise.r2


def test_compare_probes_requires_identical_test_dates(split_dates):
    bounds, idx = split_dates
    rng = np.random.default_rng(0)
    target = pd.Series(rng.normal(size=1000), index=idx)
    features = pd.DataFrame({"x": rng.normal(size=1000)}, index=idx)

    res_a = ridge_probe(features, target, alpha_grid=[1.0], **bounds)
    shifted = dict(bounds)
    shifted["test_end"] = idx[998]  # one day shorter test window
    res_b = ridge_probe(features, target, alpha_grid=[1.0], **shifted)

    with pytest.raises(ValueError, match="identical test dates"):
        compare_probes(res_a, res_b, name_a="a", name_b="b")


def test_compare_probes_prefers_lower_mse(split_dates):
    bounds, idx = split_dates
    rng = np.random.default_rng(0)
    signal = rng.normal(size=1000)
    target = pd.Series(0.8 * signal + 0.2 * rng.normal(size=1000), index=idx)

    good = pd.DataFrame({"signal": signal}, index=idx)
    bad = pd.DataFrame({"noise": rng.normal(size=1000)}, index=idx)

    res_good = ridge_probe(good, target, alpha_grid=[0.01, 1.0], **bounds)
    res_bad = ridge_probe(bad, target, alpha_grid=[0.01, 1.0], **bounds)

    comparison = compare_probes(res_good, res_bad, name_a="good", name_b="bad", n_bootstrap=500)
    assert comparison.a_beats_b()
    assert not compare_probes(res_bad, res_good, name_a="bad", name_b="good", n_bootstrap=500).a_beats_b()


def test_alpha_selected_on_validation_not_train(split_dates):
    """select_ridge_alpha must actually use X_val/y_val, not just refit on train."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(200, 5))
    y_train = x_train[:, 0] + rng.normal(scale=0.01, size=200)  # near-perfect on train
    # On validation, the same coefficient does NOT generalise (different relationship).
    x_val = rng.normal(size=(50, 5))
    y_val = -x_val[:, 0] + rng.normal(scale=0.01, size=50)

    alpha = select_ridge_alpha(x_train, y_train, x_val, y_val, alpha_grid=[0.001, 0.01, 1.0, 100.0, 10000.0])
    # A large alpha (heavy shrinkage toward zero) should be selected, since
    # any aggressive fit to train's x0->y relationship hurts badly on val.
    assert alpha >= 1.0


def test_dropna_removes_rows_where_either_side_is_nan(split_dates):
    bounds, idx = split_dates
    rng = np.random.default_rng(0)
    target = pd.Series(rng.normal(size=1000), index=idx)
    features = pd.DataFrame({"x": rng.normal(size=1000)}, index=idx)
    target.iloc[700:705] = np.nan  # NaNs inside the test window

    res = ridge_probe(features, target, alpha_grid=[1.0], **bounds)
    assert res.n_test == 300 - 5


def test_empty_slice_after_dropna_raises(split_dates):
    bounds, idx = split_dates
    rng = np.random.default_rng(0)
    target = pd.Series(np.nan, index=idx)  # entirely NaN
    features = pd.DataFrame({"x": rng.normal(size=1000)}, index=idx)
    with pytest.raises(ValueError, match="empty"):
        ridge_probe(features, target, alpha_grid=[1.0], **bounds)
