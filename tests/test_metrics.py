"""Metrics verified against hand-computed values.

Spec §7.5: "Sharpe, Sortino, max drawdown and turnover verified against
hand-computed values on a small fixture."

Every expected value below is derived in the test itself from first
principles — not copied from a previous run of the code under test, which
would only prove the code is self-consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.backtest.metrics import (
    annualised_return,
    annualised_vol,
    calmar_ratio,
    compute_metrics,
    conditional_value_at_risk,
    cumulative_return,
    drawdown_duration,
    drawdown_series,
    equity_curve,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    turnover,
    turnover_series,
    value_at_risk,
)
from prism.features.asset import TRADING_DAYS_PER_YEAR

#: A four-period fixture small enough to verify entirely by hand.
FIXTURE = pd.Series(
    [0.10, -0.20, 0.10, 0.05],
    index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
)
#: Equity path: 1.1, 0.88, 0.968, 1.0164
FIXTURE_EQUITY = [1.10, 0.88, 0.968, 1.0164]


def test_equity_curve_and_cumulative_return():
    """Compounding is on SIMPLE returns. Defect A10 compounded log as simple."""
    np.testing.assert_allclose(equity_curve(FIXTURE).to_numpy(), FIXTURE_EQUITY, rtol=1e-12)
    np.testing.assert_allclose(cumulative_return(FIXTURE), 1.0164 - 1.0, rtol=1e-12)


def test_annualised_return_is_geometric():
    """CAGR = total^(252/n) - 1, computed by hand."""
    expected = 1.0164 ** (TRADING_DAYS_PER_YEAR / 4) - 1.0
    np.testing.assert_allclose(annualised_return(FIXTURE), expected, rtol=1e-10)


def test_annualised_vol_uses_sample_std():
    """ddof=1, scaled by sqrt(252)."""
    r = FIXTURE.to_numpy()
    mean = r.mean()
    sample_var = ((r - mean) ** 2).sum() / (len(r) - 1)
    expected = np.sqrt(sample_var) * np.sqrt(TRADING_DAYS_PER_YEAR)
    np.testing.assert_allclose(annualised_vol(FIXTURE), expected, rtol=1e-12)


def test_sharpe_ratio_hand_computed():
    """mean / sample-std * sqrt(252), zero risk-free."""
    r = FIXTURE.to_numpy()
    mean = r.mean()  # 0.0125
    sd = np.sqrt(((r - mean) ** 2).sum() / (len(r) - 1))
    expected = mean / sd * np.sqrt(TRADING_DAYS_PER_YEAR)
    np.testing.assert_allclose(sharpe_ratio(FIXTURE), expected, rtol=1e-12)
    np.testing.assert_allclose(mean, 0.0125, rtol=1e-12)


def test_sharpe_risk_free_is_an_annual_rate():
    """A 2% risk-free rate must not be applied as 2% per day."""
    with_rf = sharpe_ratio(FIXTURE, risk_free=0.02)
    without = sharpe_ratio(FIXTURE, risk_free=0.0)
    daily = (1.02) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    assert 0 < daily < 1e-4, "the per-period conversion is not geometric"
    assert with_rf < without
    # The whole Sharpe must not collapse, which is what a per-day 2% would do.
    assert abs(with_rf - without) < 0.2 * abs(without)


def test_sortino_uses_downside_deviation_about_the_target():
    """sqrt(mean(min(r,0)^2)) over ALL periods, not the std of losers only.

    The distinction is not pedantic: with one loss in four, the std of the
    losing subset is undefined (n=1), while the downside deviation is well
    defined and is what Sortino's own papers specify.
    """
    r = FIXTURE.to_numpy()
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    expected = r.mean() / downside * np.sqrt(TRADING_DAYS_PER_YEAR)
    np.testing.assert_allclose(sortino_ratio(FIXTURE), expected, rtol=1e-12)

    # And it differs from the naive "std of the negative returns" version.
    losers = r[r < 0]
    assert len(losers) == 1
    np.testing.assert_allclose(downside, np.sqrt((0.20**2) / 4), rtol=1e-12)


def test_max_drawdown_hand_computed():
    """Peak 1.10 -> trough 0.88 is -20%."""
    np.testing.assert_allclose(max_drawdown(FIXTURE), 0.88 / 1.10 - 1.0, rtol=1e-12)
    np.testing.assert_allclose(max_drawdown(FIXTURE), -0.20, rtol=1e-12)


def test_drawdown_series_and_duration():
    """Three consecutive periods below the 1.10 peak."""
    dd = drawdown_series(FIXTURE)
    expected = [0.0, 0.88 / 1.10 - 1, 0.968 / 1.10 - 1, 1.0164 / 1.10 - 1]
    np.testing.assert_allclose(dd.to_numpy(), expected, rtol=1e-12)
    assert drawdown_duration(FIXTURE) == 3


def test_max_drawdown_is_zero_for_a_monotone_path():
    rising = pd.Series([0.01] * 20, index=pd.bdate_range("2020-01-01", periods=20))
    np.testing.assert_allclose(max_drawdown(rising), 0.0, atol=1e-15)
    assert drawdown_duration(rising) == 0


def test_calmar_is_return_over_drawdown_magnitude():
    expected = annualised_return(FIXTURE) / 0.20
    np.testing.assert_allclose(calmar_ratio(FIXTURE), expected, rtol=1e-12)


def test_var_and_cvar_are_non_positive_tail_measures():
    """VaR at 95% on this fixture is the 5th percentile of four points."""
    expected_var = np.quantile(FIXTURE.to_numpy(), 0.05)
    np.testing.assert_allclose(value_at_risk(FIXTURE, 0.95), expected_var, rtol=1e-12)
    # CVaR is the mean of everything at or below VaR, so <= VaR.
    assert conditional_value_at_risk(FIXTURE, 0.95) <= value_at_risk(FIXTURE, 0.95)


def test_hit_rate_and_tail_ratio():
    assert hit_rate(FIXTURE) == pytest.approx(3 / 4)
    left = abs(np.quantile(FIXTURE.to_numpy(), 0.05))
    right = abs(np.quantile(FIXTURE.to_numpy(), 0.95))
    np.testing.assert_allclose(tail_ratio(FIXTURE), right / left, rtol=1e-12)


# --------------------------------------------------------------------------- #
# turnover
# --------------------------------------------------------------------------- #
def test_turnover_is_one_way_and_charges_initiation():
    """0.5 * sum|dw|, averaged, with the first row treated as a move from cash.

    A complete switch from one asset to another is 100% turnover, not 200%.
    Initiating the portfolio costs turnover, otherwise buy-and-hold appears to
    have been established for free.
    """
    weights = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=["a", "b"])
    # Row 0: from cash into a  -> 0.5 * 1.0 = 0.5
    # Row 1: a fully into b    -> 0.5 * 2.0 = 1.0
    np.testing.assert_allclose(turnover(weights), (0.5 + 1.0) / 2, rtol=1e-12)


def test_zero_turnover_for_a_held_portfolio():
    """§7.5: zero turnover implies zero cost — the turnover half of that claim."""
    held = pd.DataFrame([[0.5, 0.5]] * 5, columns=["a", "b"])
    # From cash, only the initiation costs anything; the four holds are free.
    np.testing.assert_allclose(turnover(held), (0.5 * 1.0) / 5, rtol=1e-12)
    # Told what is already held, turnover is exactly zero for every period.
    np.testing.assert_allclose(
        turnover(held, initial_weights=held.iloc[0]), 0.0, atol=1e-15
    )
    assert (turnover_series(held, held.iloc[0]).to_numpy() == 0.0).all()


def test_turnover_is_monotone_in_weight_change():
    """§7.5: "cost is monotone in turnover" — the turnover side of it."""
    base = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], columns=["a", "b"])
    small = pd.DataFrame([[0.5, 0.5], [0.55, 0.45]], columns=["a", "b"])
    large = pd.DataFrame([[0.5, 0.5], [0.9, 0.1]], columns=["a", "b"])
    assert turnover(base) < turnover(small) < turnover(large)


def test_turnover_annualisation_requires_a_rate():
    weights = pd.DataFrame([[0.5, 0.5], [0.6, 0.4]], columns=["a", "b"])
    with pytest.raises(ValueError, match="periods_per_year"):
        turnover(weights, annualise=True)
    weekly = 52.0
    np.testing.assert_allclose(
        turnover(weights, annualise=True, periods_per_year=weekly),
        turnover(weights) * weekly,
        rtol=1e-12,
    )


def test_de_risking_into_cash_is_one_way_turnover():
    """Moving from 100% invested to 60% invested is 20% one-way turnover.

    |0.3-0.5| + |0.3-0.5| = 0.4 gross, halved = 0.2 one-way. The 40% that
    moved into cash is charged once, not twice.
    """
    weights = pd.DataFrame([[0.5, 0.5], [0.3, 0.3]], columns=["a", "b"])
    series = turnover_series(weights, initial_weights=weights.iloc[0])
    np.testing.assert_allclose(series.to_numpy(), [0.0, 0.2], rtol=1e-12)
    np.testing.assert_allclose(
        turnover(weights, initial_weights=weights.iloc[0]), 0.1, rtol=1e-12
    )


def test_initial_weights_prevent_a_spurious_reinitiation_at_a_fold_boundary():
    """Slicing a weight frame must not charge a fresh initiation.

    Without ``initial_weights``, evaluating fold 2 onward would bill a full
    move from cash for a portfolio that was already in place — inflating the
    reported cost of every fold after the first.
    """
    full = pd.DataFrame([[0.5, 0.5]] * 6, columns=["a", "b"])
    tail = full.iloc[3:]
    assert turnover(tail) > 0.0, "from cash, the slice looks like a fresh initiation"
    np.testing.assert_allclose(
        turnover(tail, initial_weights=full.iloc[2]), 0.0, atol=1e-15
    )


# --------------------------------------------------------------------------- #
# the aggregate
# --------------------------------------------------------------------------- #
def test_compute_metrics_reports_net_and_the_cost_drag():
    """§14.1: headline metrics are net of costs; gross is reported alongside."""
    gross = pd.Series([0.01] * 100, index=pd.bdate_range("2020-01-01", periods=100))
    cost = 0.0002
    net = gross - cost
    weights = pd.DataFrame(0.5, index=gross.index, columns=["a", "b"])

    m = compute_metrics(net, weights=weights, gross_returns=gross, rebalances_per_year=52)
    assert m.n_periods == 100
    np.testing.assert_allclose(m.annualised_return, annualised_return(net), rtol=1e-12)
    assert m.gross_annualised_return > m.annualised_return
    np.testing.assert_allclose(
        m.cost_drag_annualised,
        annualised_return(gross) - annualised_return(net),
        rtol=1e-12,
    )
    assert m.turnover is not None and m.turnover_annualised is not None
    assert set(m.as_dict()) >= {
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "drawdown_duration",
        "var_95",
        "cvar_95",
        "hit_rate",
        "tail_ratio",
        "turnover",
    }, "the §14.1 metric set is incomplete"


def test_degenerate_inputs_return_nan_not_zero():
    """A flat series has no Sharpe. Returning 0.0 would read as 'no edge'."""
    flat = pd.Series([0.0] * 10, index=pd.bdate_range("2020-01-01", periods=10))
    assert np.isnan(sharpe_ratio(flat))
    assert np.isnan(sortino_ratio(flat))
    empty = pd.Series([], dtype="float64", index=pd.DatetimeIndex([]))
    assert np.isnan(annualised_return(empty))
    assert np.isnan(max_drawdown(empty))


def test_total_loss_does_not_produce_a_complex_or_nan_cagr():
    """A path that goes to zero must report -100%, not a complex root."""
    wipeout = pd.Series([-0.5, -1.0], index=pd.bdate_range("2020-01-01", periods=2))
    assert annualised_return(wipeout) == -1.0


def test_nans_are_dropped_not_filled():
    """A missing return is absent, not zero — filling it invents a flat day."""
    with_gap = pd.Series(
        [0.01, np.nan, 0.01], index=pd.bdate_range("2020-01-01", periods=3)
    )
    np.testing.assert_allclose(cumulative_return(with_gap), 1.01 * 1.01 - 1, rtol=1e-12)


def test_a_loss_in_the_first_period_is_a_drawdown():
    """The running peak starts at the initial NAV, not at the first return."""
    from prism.backtest.metrics import max_drawdown

    r = pd.Series([-0.10, -0.10, 0.05])
    assert max_drawdown(r) == pytest.approx(0.9 * 0.9 - 1.0)
