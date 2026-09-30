"""Forward-looking targets. Spec §5.4.

**These are deliberately non-causal.** Every quantity here looks into the
future by construction; that is what makes them targets. They exist for
supervised probes (§13.1) and for evaluation, and they are the reason the
embargo exists (§6.1).

Two safeguards keep them out of places they must never reach:

1. Every column name is prefixed ``fwd_``, and
   :func:`prism.state.assert_no_targets` rejects any state vector containing
   such a column.
2. ``tests/test_causality.py`` asserts that this module's output *fails* the
   causality test — a target that passed would mean it was not actually
   forward-looking.

The horizons here must stay consistent with
``prism.config.FORWARD_TARGET_HORIZONS``, which is what the embargo is sized
against; :func:`target_horizons` is checked against it in the test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.features.asset import TRADING_DAYS_PER_YEAR, log_returns

__all__ = ["TARGET_PREFIX", "TARGET_NAMES", "target_horizons", "build_targets"]

TARGET_PREFIX = "fwd_"

#: Columns produced by :func:`build_targets`.
TARGET_NAMES: tuple[str, ...] = (
    "fwd_vol_5",
    "fwd_vol_20",
    "fwd_ret_5",
    "fwd_ret_20",
    "fwd_max_drawdown_20",
    "fwd_corr_20",
)


def target_horizons() -> set[int]:
    """Horizons appearing in :data:`TARGET_NAMES`, parsed from the names."""
    out: set[int] = set()
    for name in TARGET_NAMES:
        tail = name.rsplit("_", 1)[-1]
        if tail.isdigit():
            out.add(int(tail))
    return out


def _forward_window_mean(values: pd.Series, horizon: int, func: str) -> pd.Series:
    """Apply a rolling ``func`` over the *next* ``horizon`` rows, exclusive of t.

    Implemented by reversing the series, rolling, and reversing back, then
    shifting so that row ``t`` holds the statistic over ``t+1 .. t+horizon``.
    Excluding ``t`` itself matters: a decision taken at the close of ``t`` is
    executed at ``t+1`` (``execution_lag_days``), so the realised outcome it
    is judged against starts at ``t+1``.
    """
    reversed_series = values.iloc[::-1]
    rolled = getattr(reversed_series.rolling(horizon), func)()
    forward = rolled.iloc[::-1]
    # ``forward`` at row t covers t .. t+horizon-1; shift up by one so it
    # covers t+1 .. t+horizon.
    return forward.shift(-1)


def build_targets(
    close: pd.DataFrame,
    *,
    benchmark: str,
    sleeve: list[str] | None = None,
    short_horizon: int = 5,
    long_horizon: int = 20,
) -> pd.DataFrame:
    """Forward risk and return targets for the benchmark and the sleeve.

    Parameters
    ----------
    close
        Cleaned close prices.
    benchmark
        The series whose forward vol, return and drawdown are targeted. The
        probes predict market-level risk, not per-asset risk, so a single
        benchmark keeps the target set small and interpretable.
    sleeve
        Equity sleeve used for ``fwd_corr_20`` (forward average pairwise
        correlation). Defaults to every column except the benchmark.

    Returns
    -------
    A frame whose row ``t`` describes the window ``t+1 .. t+horizon``. Trailing
    rows are NaN — there is no future left to describe — and callers must not
    fill them.
    """
    from prism.features.cross import average_pairwise_correlation

    if benchmark not in close.columns:
        raise KeyError(f"benchmark {benchmark!r} is not a column of close")
    members = (
        [c for c in close.columns if c != benchmark] if sleeve is None else list(sleeve)
    )
    missing = [t for t in members if t not in close.columns]
    if missing:
        raise KeyError(f"close is missing sleeve tickers {missing}")

    rets = log_returns(close[benchmark])
    out = pd.DataFrame(index=close.index)

    ann = np.sqrt(TRADING_DAYS_PER_YEAR)
    out["fwd_vol_5"] = _forward_window_mean(rets, short_horizon, "std") * ann
    out["fwd_vol_20"] = _forward_window_mean(rets, long_horizon, "std") * ann

    # Cumulative log return over the forward window.
    out["fwd_ret_5"] = _forward_window_mean(rets, short_horizon, "sum")
    out["fwd_ret_20"] = _forward_window_mean(rets, long_horizon, "sum")

    out["fwd_max_drawdown_20"] = _forward_max_drawdown(close[benchmark], long_horizon)

    # Forward realised average pairwise correlation: compute the trailing
    # version and shift it back by the horizon, so row t holds the
    # correlation realised over t+1 .. t+20.
    sleeve_rets = np.log(close[members]).diff()
    trailing_corr = average_pairwise_correlation(sleeve_rets, long_horizon)
    out["fwd_corr_20"] = trailing_corr.shift(-long_horizon)

    out = out.reindex(columns=list(TARGET_NAMES))
    out.index.name = close.index.name or "date"
    return out


def _forward_max_drawdown(price: pd.Series, horizon: int) -> pd.Series:
    """Worst peak-to-trough decline over the next ``horizon`` sessions.

    Reported as a non-positive number (0 means the path never fell below its
    running peak within the window). The window is ``t+1 .. t+horizon``; the
    running peak is taken within that window, not from ``t``, so the target
    measures the drawdown an investor who bought at ``t+1`` would have
    experienced.
    """
    values = price.to_numpy(dtype="float64")
    n = len(values)
    out = np.full(n, np.nan)
    for t in range(n):
        lo, hi = t + 1, t + 1 + horizon
        if hi > n:
            break
        window = values[lo:hi]
        if np.isnan(window).any():
            continue
        running_peak = np.maximum.accumulate(window)
        drawdowns = window / running_peak - 1.0
        out[t] = float(drawdowns.min())
    return pd.Series(out, index=price.index, name="fwd_max_drawdown_20")
