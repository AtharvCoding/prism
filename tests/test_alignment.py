"""Index alignment and the timing convention.

Spec §7.3. Alignment bugs are the quietest class of error in this project:
they do not raise, they do not look wrong, and they change every downstream
number. The reference encoder shifted its labels by one and dropped its final
window (defect C1) — two characters of index arithmetic, invisible on
inspection, fatal to the result.

Four properties are checked:

* the encoder's windowing contract (**step 3** — ``xfail`` until then),
* the reward/return timing contract, on a synthetic path with a
  hand-computed answer,
* feature and target indices are identical after alignment,
* the state index matches (**step 3b** — ``xfail`` until then).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.backtest.engine import (
    TimingConvention,
    holding_periods,
    portfolio_period_returns,
)
from prism.config import Config
from prism.features.build import split_raw_frame
from prism.features.targets import build_targets
from prism.utils.calendar import rebalance_dates, trading_days

# --------------------------------------------------------------------------- #
# encoder windowing — step 3
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="spec §9.3 — encoder windowing is build step 3. When it lands this "
    "turns XPASS (a failure) and the marker must be removed.",
)
def test_window_count_is_len_minus_window_plus_one(cfg: Config, features_a):
    """§7.3: ``len(windows) == len(features) - window + 1``.

    The reference implementation used ``len(data) - window``, silently
    discarding the most recent window — i.e. the only one that matters for a
    live decision (defect C1).
    """
    from prism.models.encoder.dataset import make_windows

    window = cfg.encoder.window.size
    frame = features_a.frame
    windows = make_windows(frame, window)
    assert len(windows) == len(frame) - window + 1


@pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="spec §9.3 — encoder windowing is build step 3.",
)
def test_row_dated_D_is_the_window_ending_at_D(cfg: Config, features_a):
    """§7.3: "Row dated ``D`` of the latent frame equals ``encoder(window whose
    last row is ``D``)``."

    The reference labelled the window ending at row ``k+29`` with the date at
    row ``k+30``, so every latent was one day stale — and stale in the
    direction that *looks* causal while actually mislabelling the whole series.
    """
    from prism.models.encoder.dataset import make_windows

    window = cfg.encoder.window.size
    frame = features_a.frame
    windows = make_windows(frame, window)

    probe = frame.index[window + 10]
    expected_last_row = frame.loc[probe]
    actual_last_row = windows.loc[probe][-1]
    np.testing.assert_allclose(
        np.asarray(actual_last_row, dtype="float64"),
        expected_last_row.to_numpy(dtype="float64"),
    )


# --------------------------------------------------------------------------- #
# timing contract — testable now
# --------------------------------------------------------------------------- #
def test_decision_at_t_earns_the_return_starting_at_t_plus_lag(cfg: Config):
    """§7.3: "Reward for decision at ``t`` uses return from ``t -> t+1``
    (with execution lag), verified on a synthetic price path with a known
    answer."

    The path grows by exactly 1% per session, so every quantity below is
    hand-computable. The decision taken at ``t`` must earn nothing over
    ``t -> t+lag``; that stretch is still held at the previous weights.
    """
    convention = TimingConvention.from_config(cfg)
    assert convention.execution_lag_days == 1, "this test hand-computes for lag=1"

    index = trading_days("2019-01-02", "2019-06-28", cfg.data.calendar.exchange)
    growth = 1.01
    prices = pd.DataFrame({"A": growth ** np.arange(len(index))}, index=index)

    periods = holding_periods(index, convention)
    weights = pd.DataFrame(1.0, index=periods["decision_date"], columns=["A"])
    realised = portfolio_period_returns(weights, prices, convention)

    positions = index.get_indexer(pd.DatetimeIndex(periods["decision_date"]))
    for row, decision_pos in zip(periods.itertuples(index=False), positions):
        # Execution is exactly `lag` sessions after the decision.
        exec_pos = int(index.get_indexer([row.execution_date])[0])
        assert exec_pos == decision_pos + convention.execution_lag_days

        # The earning window starts at execution, not at the decision.
        assert row.return_start == row.execution_date

        # And the return is exactly the path's growth over that window.
        end_pos = int(index.get_indexer([row.return_end])[0])
        sessions_held = end_pos - exec_pos
        expected = growth**sessions_held - 1.0
        np.testing.assert_allclose(
            realised.loc[row.decision_date], expected, rtol=1e-12
        )


def test_decision_does_not_earn_the_execution_lag_period(cfg: Config):
    """The lag period must be excluded, and its exclusion must be detectable.

    A price path that jumps 50% on exactly one session — the session between
    decision and execution — makes the error unmissable: if the decision were
    credited with that session, its return would contain the jump.
    """
    convention = TimingConvention.from_config(cfg)
    index = trading_days("2019-01-02", "2019-03-29", cfg.data.calendar.exchange)

    periods = holding_periods(index, convention)
    first = periods.iloc[0]
    decision_pos = int(index.get_indexer([first.decision_date])[0])

    # Flat except for a single 50% jump on the session after the decision.
    levels = np.ones(len(index))
    levels[decision_pos + 1 :] = 1.5
    prices = pd.DataFrame({"A": levels}, index=index)

    weights = pd.DataFrame(1.0, index=periods["decision_date"], columns=["A"])
    realised = portfolio_period_returns(weights, prices, convention)

    np.testing.assert_allclose(realised.loc[first.decision_date], 0.0, atol=1e-12)


def test_holding_periods_are_contiguous_and_non_overlapping(cfg: Config):
    """Each period starts where the previous one ended — no gaps, no double-count."""
    convention = TimingConvention.from_config(cfg)
    index = trading_days("2015-01-02", "2019-12-31", cfg.data.calendar.exchange)
    periods = holding_periods(index, convention)

    starts = pd.DatetimeIndex(periods["return_start"])
    ends = pd.DatetimeIndex(periods["return_end"])
    assert (ends > starts).all(), "a holding period must span at least one session"
    # Period i+1 begins exactly where period i ended.
    assert (starts[1:] == ends[:-1]).all(), (
        "holding periods are not contiguous; returns are being double-counted or dropped"
    )
    assert pd.DatetimeIndex(periods["decision_date"]).is_unique


def test_weekly_rebalance_survives_holiday_weeks(cfg: Config):
    """A closed Friday must not silently drop that week's decision.

    Good Friday 2020-04-10 and Christmas-week closures are the cases that
    catch a naive ``dayofweek == 4`` filter. The schedule takes the last
    session on or before the target weekday instead.
    """
    index = trading_days("2020-01-01", "2020-12-31", cfg.data.calendar.exchange)
    decisions = rebalance_dates(index, "weekly", cfg.data.decision.rebalance_day)

    assert pd.Timestamp("2020-04-10") not in index, "Good Friday should be a closure"
    easter_week = [
        d for d in decisions if pd.Timestamp("2020-04-06") <= d <= pd.Timestamp("2020-04-12")
    ]
    assert len(easter_week) == 1, "the Good Friday week lost its decision"
    assert easter_week[0] == pd.Timestamp("2020-04-09"), "should fall back to Thursday"

    iso = pd.DatetimeIndex(decisions).isocalendar()
    keys = list(zip(iso["year"].to_numpy(), iso["week"].to_numpy()))
    assert len(keys) == len(set(keys)), "more than one decision in some ISO week"


# --------------------------------------------------------------------------- #
# feature / target index identity
# --------------------------------------------------------------------------- #
def test_feature_and_target_indices_are_identical_after_alignment(
    cfg: Config, raw_b: pd.DataFrame, features_b
):
    """§7.3: "Feature index, target index, and state index are identical after
    alignment."

    Targets carry trailing NaNs by construction — there is no future left to
    describe — so alignment means an inner join on the dates where *both* are
    defined, never a reindex-and-fill.
    """
    close = split_raw_frame(raw_b)["Close"]
    benchmark = cfg.data.universes["A"].benchmark
    sleeve = [t for t in cfg.data.universes["A"].equity_sectors if t in close.columns]
    targets = build_targets(close, benchmark=benchmark, sleeve=sleeve)

    features = features_b.frame
    aligned_targets = targets.dropna()
    shared = features.index.intersection(aligned_targets.index)
    assert len(shared) > 1000

    f = features.loc[shared]
    t = aligned_targets.loc[shared]
    assert f.index.equals(t.index)
    assert f.index.is_monotonic_increasing and f.index.is_unique
    assert not f.isna().to_numpy().any(), "features must be warm on the aligned index"
    assert not t.isna().to_numpy().any(), "targets must be complete on the aligned index"


def test_target_rows_describe_the_future_not_the_present(cfg: Config, raw_b: pd.DataFrame):
    """Row ``t`` of the targets must describe ``t+1 .. t+h``, inclusive of neither end wrong.

    Verified against a hand-computed answer on a deterministic path, because
    an off-by-one here shifts the probe's entire R2.
    """
    index = trading_days("2019-01-02", "2019-12-31", cfg.data.calendar.exchange)
    growth = 1.002
    prices = pd.DataFrame(
        {
            "SPY": growth ** np.arange(len(index)),
            "XLK": growth ** np.arange(len(index)),
            "XLE": (growth**0.5) ** np.arange(len(index)),
        },
        index=index,
    )
    targets = build_targets(prices, benchmark="SPY", sleeve=["XLK", "XLE"])

    # fwd_ret_5 at row t = sum of log returns over t+1..t+5 = 5 * log(growth).
    expected = 5.0 * np.log(growth)
    np.testing.assert_allclose(targets["fwd_ret_5"].iloc[10], expected, rtol=1e-12)
    # fwd_ret_20 likewise.
    np.testing.assert_allclose(
        targets["fwd_ret_20"].iloc[10], 20.0 * np.log(growth), rtol=1e-12
    )
    # A monotone path never draws down.
    np.testing.assert_allclose(targets["fwd_max_drawdown_20"].iloc[10], 0.0, atol=1e-15)
    # The last rows have no future, so they must be NaN and must not be filled.
    assert targets["fwd_ret_20"].iloc[-1:].isna().all()
    assert targets["fwd_ret_5"].iloc[-1:].isna().all()


def test_a_shifted_target_is_detectable(cfg: Config, raw_b: pd.DataFrame):
    """The alignment check must be able to fail, or it proves nothing."""
    close = split_raw_frame(raw_b)["Close"]
    benchmark = cfg.data.universes["A"].benchmark
    sleeve = [t for t in cfg.data.universes["A"].equity_sectors if t in close.columns]
    targets = build_targets(close, benchmark=benchmark, sleeve=sleeve).dropna()
    shifted = targets.shift(1).dropna()
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            targets.loc[shifted.index], shifted, check_freq=False
        )


# --------------------------------------------------------------------------- #
# state index — step 3b
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    raises=(NotImplementedError, ImportError, AttributeError),
    strict=True,
    reason="spec §10 — state assembly is build step 3b.",
)
def test_state_index_matches_feature_and_target_index(cfg: Config, features_b):
    """§7.3: the state index must equal the feature and target index."""
    from prism.state import build_state  # noqa: PLC0415

    state = build_state(features_b, cfg, variant="V1")
    assert state.frame.index.equals(features_b.frame.index)
