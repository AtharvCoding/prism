"""Performance metrics. Spec §14.1.

Every headline metric is **net of costs** (§14.1); this module takes a net
return series and does not itself apply costs. Two conventions are fixed here
because getting them wrong is silent:

* **Returns are simple, not log.** ``cumulative_return`` compounds with
  ``prod(1 + r)``. Defect A10 was compounding log returns as if they were
  simple returns, which understates growth; it was cosmetic there because it
  only touched a plot, but the same error in a Sharpe denominator is not.
* **Annualisation uses 252 sessions.** Stated once, in
  :data:`prism.features.asset.TRADING_DAYS_PER_YEAR`.

Turnover is defined as **one-way**: ``0.5 * sum |w_t - w_{t-1}|``, so a
complete switch from one asset to another is 100% turnover, not 200%. The
cost model then charges ``cost_bps`` per side, which is the convention that
makes "5 bps per side" mean what a practitioner expects.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from prism.features.asset import TRADING_DAYS_PER_YEAR

__all__ = [
    "PerformanceMetrics",
    "cumulative_return",
    "equity_curve",
    "annualised_return",
    "annualised_vol",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_series",
    "drawdown_duration",
    "calmar_ratio",
    "value_at_risk",
    "conditional_value_at_risk",
    "turnover",
    "turnover_series",
    "hit_rate",
    "tail_ratio",
    "compute_metrics",
]


def _clean(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    return returns.astype("float64").dropna()


def cumulative_return(returns: pd.Series) -> float:
    """Total simple compounded return over the series."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float(np.prod(1.0 + r.to_numpy()) - 1.0)


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compounded equity path, starting at ``initial``."""
    r = _clean(returns)
    return initial * (1.0 + r).cumprod()


def annualised_return(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric annualised return (CAGR)."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    total = np.prod(1.0 + r.to_numpy())
    if total <= 0:
        return -1.0
    return float(total ** (periods / len(r)) - 1.0)


def annualised_vol(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised standard deviation, sample (ddof=1)."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio of excess returns.

    ``risk_free`` is an *annual* rate and is converted to a per-period rate
    geometrically, so passing 2% does not silently mean 2% per day.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    rf_period = (1.0 + risk_free) ** (1.0 / periods) - 1.0
    excess = r - rf_period
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio.

    The downside deviation is a second moment **about the target, over all
    periods** — ``sqrt(mean(min(r - target, 0)^2))`` — not the standard
    deviation of the losing subset. The two differ, and the former is the
    definition in Sortino's own papers.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    target_period = (1.0 + target) ** (1.0 / periods) - 1.0
    excess = r - target_period
    downside = np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2))
    if downside == 0:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each point, as a non-positive fraction of the running peak."""
    curve = equity_curve(returns)
    if curve.empty:
        return curve
    # The running peak starts at the initial NAV of 1.0, not at the first return:
    # otherwise a loss in the very first period is never counted as a drawdown.
    peak = curve.cummax().clip(lower=1.0)
    return curve / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline. Non-positive; 0.0 for a never-losing path."""
    dd = drawdown_series(returns)
    if dd.empty:
        return float("nan")
    return float(dd.min())


def drawdown_duration(returns: pd.Series) -> int:
    """Longest run of consecutive periods spent below a prior peak."""
    dd = drawdown_series(returns)
    if dd.empty:
        return 0
    under = (dd < 0).to_numpy()
    best = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised return divided by the magnitude of max drawdown."""
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(annualised_return(returns, periods) / abs(mdd))


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical VaR at ``level``, reported as a non-positive return."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float(np.quantile(r.to_numpy(), 1.0 - level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Mean of the returns at or below the VaR threshold (expected shortfall)."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    var = value_at_risk(r, level)
    tail = r[r <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())


def turnover_series(
    weights: pd.DataFrame, initial_weights: pd.Series | None = None
) -> pd.Series:
    """One-way turnover for each rebalance.

    ``0.5 * sum_i |w_{t,i} - w_{t-1,i}|``. When ``initial_weights`` is omitted
    the first row is treated as a move from **all cash**, so initiating the
    portfolio costs turnover — otherwise buy-and-hold appears to have been
    established for free.

    Pass ``initial_weights`` to state what is already held. This matters at a
    walk-forward fold boundary: slicing a weight frame and calling this on the
    tail would otherwise charge a fresh initiation for a portfolio that was
    already in place, inflating the cost of every fold after the first.
    """
    if weights.empty:
        return pd.Series(dtype="float64")
    w = weights.astype("float64").fillna(0.0)
    prior = w.shift(1)
    if initial_weights is None:
        prior.iloc[0] = 0.0
    else:
        prior.iloc[0] = (
            initial_weights.reindex(w.columns).astype("float64").fillna(0.0).to_numpy()
        )
    return 0.5 * (w - prior).abs().sum(axis=1)


def turnover(
    weights: pd.DataFrame,
    *,
    initial_weights: pd.Series | None = None,
    annualise: bool = False,
    periods_per_year: float | None = None,
) -> float:
    """Average one-way turnover per rebalance. See :func:`turnover_series`."""
    if weights.empty:
        return float("nan")
    per_period = turnover_series(weights, initial_weights)
    mean = float(per_period.mean())
    if not annualise:
        return mean
    if periods_per_year is None:
        raise ValueError("annualise=True requires periods_per_year")
    return mean * periods_per_year


def hit_rate(returns: pd.Series) -> float:
    """Fraction of strictly positive periods."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def tail_ratio(returns: pd.Series, level: float = 0.95) -> float:
    """Right tail over left tail: ``|q(level)| / |q(1-level)|``."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    left = abs(np.quantile(r.to_numpy(), 1.0 - level))
    if left == 0:
        return float("nan")
    return float(abs(np.quantile(r.to_numpy(), level)) / left)


@dataclass(frozen=True)
class PerformanceMetrics:
    """The §14.1 metric set for one return series."""

    n_periods: int
    cumulative_return: float
    annualised_return: float
    annualised_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    drawdown_duration: int
    var_95: float
    cvar_95: float
    hit_rate: float
    tail_ratio: float
    turnover: float | None = None
    turnover_annualised: float | None = None
    gross_annualised_return: float | None = None
    cost_drag_annualised: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.as_dict())


def compute_metrics(
    net_returns: pd.Series,
    *,
    weights: pd.DataFrame | None = None,
    gross_returns: pd.Series | None = None,
    periods: int = TRADING_DAYS_PER_YEAR,
    rebalances_per_year: float | None = None,
) -> PerformanceMetrics:
    """Compute the full §14.1 metric set.

    Parameters
    ----------
    net_returns
        Per-period returns **after** costs. Headline metrics are net (§14.1).
    gross_returns
        Optional pre-cost returns; supplying them populates the
        net-vs-gross comparison and the annualised cost drag.
    """
    r = _clean(net_returns)
    to = None if weights is None else turnover(weights)
    to_ann = (
        None
        if (weights is None or rebalances_per_year is None)
        else turnover(weights, annualise=True, periods_per_year=rebalances_per_year)
    )
    gross_ann = None if gross_returns is None else annualised_return(gross_returns, periods)
    net_ann = annualised_return(r, periods)
    drag = None if gross_ann is None else gross_ann - net_ann

    return PerformanceMetrics(
        n_periods=len(r),
        cumulative_return=cumulative_return(r),
        annualised_return=net_ann,
        annualised_vol=annualised_vol(r, periods),
        sharpe=sharpe_ratio(r, periods=periods),
        sortino=sortino_ratio(r, periods=periods),
        calmar=calmar_ratio(r, periods),
        max_drawdown=max_drawdown(r),
        drawdown_duration=drawdown_duration(r),
        var_95=value_at_risk(r, 0.95),
        cvar_95=conditional_value_at_risk(r, 0.95),
        hit_rate=hit_rate(r),
        tail_ratio=tail_ratio(r),
        turnover=to,
        turnover_annualised=to_ann,
        gross_annualised_return=gross_ann,
        cost_drag_annualised=drag,
    )
