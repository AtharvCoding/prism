"""Cross-sectional features over the equity sleeve. Spec §5.2.

These are the features the reference project lacked entirely (defect A11):
without them, a regime signal has nothing to act on beyond the level of
volatility. Average pairwise correlation and market-mode dominance are the
classic crisis indicators, and they are *not* recoverable from per-asset
features — correlation is a property of the panel, not of any one series.

The shrunk covariance matrix is returned separately rather than flattened
into the feature frame (spec §5.2): a 13x13 covariance is 91 numbers, which
would swamp the state vector while adding little the scalar summaries miss.
The allocator consumes it directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "CROSS_FEATURE_NAMES",
    "average_pairwise_correlation",
    "first_eigenvalue_share",
    "breadth_above_ma",
    "build_cross_features",
    "rolling_shrunk_covariance",
]

#: Columns produced by :func:`build_cross_features`, in order.
CROSS_FEATURE_NAMES: tuple[str, ...] = (
    "avg_pairwise_corr_60",
    "corr_dispersion_60",
    "avg_pairwise_corr_change",
    "return_dispersion",
    "breadth",
    "first_eigenvalue_share_60",
)


def _corr_summary(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mean and std of the off-diagonal correlations, per session.

    Computed from a rolling covariance rather than ``DataFrame.rolling.corr``
    on pairs, so one pass produces both moments. Rows before the window is
    full are NaN, which keeps the function truncation-invariant.
    """
    n = returns.shape[1]
    if n < 2:
        raise ValueError("cross-sectional correlation needs at least two assets")

    iu = np.triu_indices(n, k=1)
    values = returns.to_numpy(dtype="float64")
    out_mean = np.full(len(returns), np.nan)
    out_std = np.full(len(returns), np.nan)

    for end in range(window - 1, len(returns)):
        block = values[end - window + 1 : end + 1]
        if np.isnan(block).any():
            continue
        corr = np.corrcoef(block, rowvar=False)
        off = corr[iu]
        if not np.isfinite(off).all():
            continue
        out_mean[end] = off.mean()
        out_std[end] = off.std(ddof=1) if len(off) > 1 else 0.0

    return pd.DataFrame(
        {"avg_pairwise_corr_60": out_mean, "corr_dispersion_60": out_std},
        index=returns.index,
    )


def average_pairwise_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Mean off-diagonal correlation over a trailing window."""
    return _corr_summary(returns, window)["avg_pairwise_corr_60"]


def first_eigenvalue_share(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Share of total variance explained by PC1 of the *correlation* matrix.

    Market-mode dominance. Using the correlation matrix rather than the
    covariance matrix makes this scale-free, so it does not simply track the
    level of volatility — which matters, because the whole point of §8.1 is
    to stop building features that are proxies for volatility.
    """
    n = returns.shape[1]
    values = returns.to_numpy(dtype="float64")
    out = np.full(len(returns), np.nan)
    for end in range(window - 1, len(returns)):
        block = values[end - window + 1 : end + 1]
        if np.isnan(block).any():
            continue
        corr = np.corrcoef(block, rowvar=False)
        if not np.isfinite(corr).all():
            continue
        eigvals = np.linalg.eigvalsh(corr)
        out[end] = float(eigvals[-1] / n)  # trace of a correlation matrix is n
    return pd.Series(out, index=returns.index, name="first_eigenvalue_share_60")


def breadth_above_ma(close: pd.DataFrame, window: int = 50) -> pd.Series:
    """Fraction of the sleeve trading above its own ``window``-session mean."""
    ma = close.rolling(window).mean()
    above = close.gt(ma)
    valid = ma.notna() & close.notna()
    count = valid.sum(axis=1)
    return (above & valid).sum(axis=1).div(count.where(count > 0)).rename("breadth")


def build_cross_features(
    close: pd.DataFrame,
    *,
    sleeve: list[str],
    corr_window: int = 60,
    ma_window: int = 50,
) -> pd.DataFrame:
    """Cross-sectional feature frame computed over ``sleeve``.

    ``sleeve`` is the equity sleeve only. Mixing bonds and gold into the
    "average pairwise correlation" would destroy the feature's meaning: the
    crisis signal is equities correlating *with each other*, while the
    defensive assets are precisely the ones expected to decorrelate.
    """
    missing = [t for t in sleeve if t not in close.columns]
    if missing:
        raise KeyError(f"close is missing sleeve tickers {missing}")
    if len(sleeve) < 2:
        raise ValueError("cross-sectional features need at least two sleeve members")

    sub = close[list(sleeve)]
    rets = np.log(sub).diff()

    corr = _corr_summary(rets, corr_window)
    out = pd.DataFrame(index=close.index)
    out["avg_pairwise_corr_60"] = corr["avg_pairwise_corr_60"]
    out["corr_dispersion_60"] = corr["corr_dispersion_60"]
    # Stationary form, and the HMM's correlation observation (spec §8.2).
    out["avg_pairwise_corr_change"] = out["avg_pairwise_corr_60"].diff()
    out["return_dispersion"] = rets.std(axis=1, ddof=1)
    out["breadth"] = breadth_above_ma(sub, ma_window)
    out["first_eigenvalue_share_60"] = first_eigenvalue_share(rets, corr_window)

    out = out.reindex(columns=list(CROSS_FEATURE_NAMES))
    out.index.name = close.index.name or "date"
    return out


def rolling_shrunk_covariance(
    close: pd.DataFrame,
    *,
    assets: list[str],
    window: int = 60,
    dates: pd.DatetimeIndex | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Ledoit-Wolf shrunk covariance of ``assets`` returns, per date.

    Returned as a mapping rather than a flattened frame, keyed by the session
    whose trailing window produced it. Only ``dates`` are computed when given
    — the allocator needs one matrix per *rebalance* date, not per session,
    and the estimator is the expensive part of the Tier 1 allocator.

    The window ends at (and includes) the keyed date, so the matrix is usable
    for a decision taken at that close.
    """
    from sklearn.covariance import LedoitWolf

    missing = [t for t in assets if t not in close.columns]
    if missing:
        raise KeyError(f"close is missing assets {missing}")

    rets = np.log(close[list(assets)]).diff()
    index = rets.index
    targets = index if dates is None else pd.DatetimeIndex(dates)

    out: dict[pd.Timestamp, pd.DataFrame] = {}
    positions = index.get_indexer(targets)
    for date, end in zip(targets, positions):
        if end < window:  # not enough history, including the diff's first NaN
            continue
        block = rets.iloc[end - window + 1 : end + 1]
        if block.isna().to_numpy().any():
            continue
        estimator = LedoitWolf(assume_centered=False).fit(block.to_numpy(dtype="float64"))
        out[pd.Timestamp(date)] = pd.DataFrame(
            estimator.covariance_, index=list(assets), columns=list(assets)
        )
    return out
