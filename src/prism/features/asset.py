"""Per-asset features. Spec §5.1.

All features are causal and, where a choice existed, expressed in stationary
form. The §3.2 hazard about the 1999-2007 regime shift (decimalisation,
RegNMS, HFT) is the reason: raw volatility *levels* are not comparable across
that span, so the feature set leans on changes and z-scoreable quantities
rather than levels wherever the economics allow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "log_returns",
    "realised_vol",
    "downside_vol",
    "momentum_skip",
    "rolling_beta",
    "dist_from_high",
    "build_asset_features",
    "ASSET_FEATURE_NAMES",
]

TRADING_DAYS_PER_YEAR = 252

#: Momentum skips the most recent week to avoid short-term reversal
#: contamination (spec §5.1, "ex-most-recent-week").
MOMENTUM_SKIP_DAYS = 5

#: Feature suffixes produced by :func:`build_asset_features`, in order.
ASSET_FEATURE_NAMES: tuple[str, ...] = (
    "return_1d",
    "vol_20",
    "vol_60",
    "dvol",
    "mom_20",
    "mom_60",
    "downside_vol_20",
    "skew_60",
    "kurt_60",
    "volume_ratio",
    "beta_60_spy",
    "dist_from_52w_high",
)


def log_returns(close: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """One-day log returns. ``log(P_t / P_{t-1})``, so row ``t`` needs ``t-1``."""
    return np.log(close).diff()


def realised_vol(returns: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    """Annualised realised volatility over a trailing ``window``."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def downside_vol(
    returns: pd.DataFrame | pd.Series, window: int
) -> pd.DataFrame | pd.Series:
    """Annualised semi-deviation: the volatility of the losses only.

    Defined as ``sqrt(mean(min(r, 0)^2))``, so it is a second moment about
    zero rather than about the mean — the convention that makes it comparable
    to :func:`realised_vol` and monotone in tail severity.
    """
    losses = returns.clip(upper=0.0)
    mean_sq = (losses**2).rolling(window).mean()
    return np.sqrt(mean_sq) * np.sqrt(TRADING_DAYS_PER_YEAR)


def momentum_skip(
    close: pd.DataFrame | pd.Series, window: int, skip: int = MOMENTUM_SKIP_DAYS
) -> pd.DataFrame | pd.Series:
    """Log momentum over ``window`` sessions, ending ``skip`` sessions ago.

    ``log(P_{t-skip} / P_{t-skip-window})``. Both terms are strictly in the
    past, and skipping the most recent week removes the short-horizon
    reversal that otherwise dominates a 20-day momentum signal.
    """
    lagged = close.shift(skip)
    return np.log(lagged) - np.log(close.shift(skip + window))


def rolling_beta(
    returns: pd.DataFrame, benchmark: pd.Series, window: int
) -> pd.DataFrame:
    """Rolling OLS beta of each column against ``benchmark``.

    Replaces the reference pipeline's ``spycorr``, which was identically 1.0
    for SPY itself and so entered the state vector as a constant (defect A7).
    Beta is 1.0 for the benchmark too, but the benchmark is not an allocatable
    asset, and :func:`build_asset_features` drops degenerate columns anyway.
    """
    bench = benchmark.reindex(returns.index)
    cov = returns.rolling(window).cov(bench)
    var = bench.rolling(window).var()
    # ``var`` can be 0 only if the benchmark was flat for the whole window,
    # which would make beta undefined rather than infinite.
    return cov.div(var.where(var > 0), axis=0)


def dist_from_high(close: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    """Fractional distance below the trailing ``window``-session maximum.

    The maximum includes today's close, which is known at the close of day
    ``t``, so the feature is causal. Range is ``(-1, 0]``.
    """
    return close / close.rolling(window).max() - 1.0


def build_asset_features(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    *,
    benchmark: str,
    tickers: list[str] | None = None,
    vol_short: int = 20,
    vol_long: int = 60,
    high_window: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Per-asset feature frame with ``{ticker}_{feature}`` columns.

    Parameters
    ----------
    close, volume
        Cleaned, calendar-aligned price and volume panels.
    benchmark
        Ticker used for ``beta_60_spy``. Must be a column of ``close``.
    tickers
        Restrict output to these tickers. The benchmark's own features are
        still produced when it appears here — ``SPY_return_1d`` is the HMM's
        primary observation (spec §8.2).

    The frame retains leading NaN rows; trimming to the warm-up period is the
    orchestrator's job (:func:`prism.features.build.build_features`), so that
    this function stays a pure transform and truncation-invariant.
    """
    if benchmark not in close.columns:
        raise KeyError(f"benchmark {benchmark!r} is not a column of close")
    cols = list(close.columns) if tickers is None else list(tickers)
    missing = [t for t in cols if t not in close.columns]
    if missing:
        raise KeyError(f"close is missing tickers {missing}")

    rets = log_returns(close)
    bench_ret = rets[benchmark]

    vol_s = realised_vol(rets, vol_short)
    vol_l = realised_vol(rets, vol_long)
    # Change in log volatility — the stationary form (spec §5.1 `dvol`).
    # ``vol_s`` is strictly positive wherever it is defined; a zero would mean
    # 20 identical closes, which the stale-price QA check catches as a soft
    # warning, so guard rather than produce -inf.
    dvol = np.log(vol_s.where(vol_s > 0)).diff()

    mom_s = momentum_skip(close, vol_short)
    mom_l = momentum_skip(close, vol_long)
    dsv = downside_vol(rets, vol_short)
    skew = rets.rolling(vol_long).skew()
    kurt = rets.rolling(vol_long).kurt()
    beta = rolling_beta(rets, bench_ret, vol_long)
    dist = dist_from_high(close, high_window)

    if volume is not None:
        vol_aligned = volume.reindex(index=close.index, columns=close.columns)
        mean_vol = vol_aligned.rolling(vol_short).mean()
        vratio = vol_aligned.div(mean_vol.where(mean_vol > 0))
    else:
        vratio = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    sources: dict[str, pd.DataFrame] = {
        "return_1d": rets,
        "vol_20": vol_s,
        "vol_60": vol_l,
        "dvol": dvol,
        "mom_20": mom_s,
        "mom_60": mom_l,
        "downside_vol_20": dsv,
        "skew_60": skew,
        "kurt_60": kurt,
        "volume_ratio": vratio,
        "beta_60_spy": beta,
        "dist_from_52w_high": dist,
    }
    assert tuple(sources) == ASSET_FEATURE_NAMES, "ASSET_FEATURE_NAMES is out of sync"

    pieces: dict[str, pd.Series] = {}
    for ticker in cols:
        for name in ASSET_FEATURE_NAMES:
            pieces[f"{ticker}_{name}"] = sources[name][ticker]

    out = pd.DataFrame(pieces, index=close.index)
    out.index.name = close.index.name or "date"
    return out
