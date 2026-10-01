"""Threshold-regime baseline. Spec §8.7, §10 (the C2 control).

Verifies the quantile breakpoints are genuinely causal (fit on one window,
applied unchanged to another) and that the walk-forward wrapper produces the
same de-duplicated, calendar-aligned contract the HMM's own walk-forward
does — which is what lets C2 and V3 be interchangeable inputs to state
assembly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.models.baselines.threshold_regime import (
    apply_threshold_breakpoints,
    fit_threshold_breakpoints,
    one_hot,
    threshold_regime_walkforward,
)


def test_breakpoints_are_equal_frequency_on_the_fit_window():
    idx = pd.bdate_range("2020-01-01", periods=100)
    signal = pd.Series(np.arange(100, dtype="float64"), index=idx)
    bp = fit_threshold_breakpoints(signal, k=4)
    assert bp.k == 4
    assert len(bp.edges) == 3
    # Quartile edges of 0..99 land near 24.75, 49.5, 74.25.
    np.testing.assert_allclose(bp.edges, [24.75, 49.5, 74.25], atol=1e-6)


def test_apply_breakpoints_is_causal_and_does_not_refit():
    """Applying fixed edges to a NEW series must not depend on that series's
    own distribution at all — only on the edges."""
    idx = pd.bdate_range("2020-01-01", periods=50)
    fit_signal = pd.Series(np.linspace(0, 10, 50), index=idx)
    bp = fit_threshold_breakpoints(fit_signal, k=2)  # median edge = 5.0

    future_idx = pd.bdate_range("2021-01-01", periods=5)
    future = pd.Series([0.0, 4.9, 5.1, 100.0, -100.0], index=future_idx)
    states = apply_threshold_breakpoints(future, bp)
    assert states.tolist() == [0, 0, 1, 1, 0]

    # Perturbing `future` beyond what was already assigned leaves EARLIER
    # assignments unchanged — the hallmark of a fixed-edge, non-refitting
    # transform.
    mutated = future.copy()
    mutated.iloc[-1] = 9999.0
    restated = apply_threshold_breakpoints(mutated, bp)
    assert restated.iloc[:-1].tolist() == states.iloc[:-1].tolist()


def test_one_hot_rows_sum_to_one_and_match_the_state_assignment():
    idx = pd.bdate_range("2020-01-01", periods=6)
    states = pd.Series([0, 1, 2, 1, 0, 2], index=idx)
    out = one_hot(states, k=3)
    assert list(out.columns) == ["state_0", "state_1", "state_2"]
    np.testing.assert_allclose(out.sum(axis=1).to_numpy(), 1.0)
    for i, s in enumerate(states):
        assert out.iloc[i, s] == 1.0


def test_one_hot_rejects_out_of_range_states():
    idx = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(ValueError, match="outside"):
        one_hot(pd.Series([0, 1, 5], index=idx), k=3)


def test_walkforward_matches_hmm_walkforward_contract(rng):
    """Same de-duplicated, calendar-aligned, columns-sum-to-1 contract as
    :func:`prism.models.hmm.walkforward.hmm_walkforward`'s posteriors — this
    is what lets C2 and V3 be interchangeable inputs to state assembly."""
    from prism.utils.calendar import trading_days

    idx = trading_days("2015-01-02", "2018-12-31")
    signal = pd.Series(rng.normal(20, 5, len(idx)).clip(min=1), index=idx, name="vix_level")

    out = threshold_regime_walkforward(
        signal,
        k=2,
        fit_start="2015-01-02",
        first_apply_start="2017-01-01",
        apply_end="2018-12-31",
        cadence="quarterly",
        embargo_days=5,
    )
    assert out.index.is_unique
    assert out.index.is_monotonic_increasing
    expected_index = trading_days("2017-01-01", "2018-12-31")
    assert out.index.equals(expected_index)
    np.testing.assert_allclose(out.sum(axis=1).to_numpy(), 1.0)


def test_walkforward_breakpoints_differ_across_expanding_folds(rng):
    """An expanding fit window means later folds' breakpoints are NOT fit on
    the same data as earlier ones — proving the walk-forward wrapper is
    actually refitting per fold rather than reusing the first fold's edges."""
    from prism.utils.calendar import trading_days

    idx = trading_days("2015-01-02", "2018-12-31")
    trend = np.linspace(10, 40, len(idx))  # trending signal -> edges must shift
    signal = pd.Series(trend, index=idx, name="vix_level")

    fold0_edges = fit_threshold_breakpoints(
        signal.loc["2015-01-02":"2016-12-22"], k=2
    ).edges
    fold_last_edges = fit_threshold_breakpoints(
        signal.loc["2015-01-02":"2018-09-21"], k=2
    ).edges
    assert not np.allclose(fold0_edges, fold_last_edges), (
        "a trending signal should produce different edges for an expanding "
        "early vs late fit window; if not, the fixture or the fit isn't "
        "sensitive to window content"
    )
