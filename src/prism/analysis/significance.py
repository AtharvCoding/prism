"""Deflated Sharpe ratio. Spec §14.2; pre-registration §7.

"You are running many variants" (§14.2). A Sharpe ratio picked from a set of
N strategies is biased upward by the selection itself, so the headline figure
is deflated against the Sharpe the *best of N null strategies* would be
expected to show (Bailey & Lopez de Prado, 2014, "The Deflated Sharpe Ratio").

All Sharpe ratios here are PER-PERIOD (daily), not annualised: the
skewness/kurtosis correction and the sqrt(T-1) scaling are stated for the
sampling frequency of the returns.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["EULER_MASCHERONI", "per_period_sharpe", "expected_max_sharpe", "deflated_sharpe"]

EULER_MASCHERONI = 0.5772156649015329


def per_period_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype="float64")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """``SR0``: the expected maximum Sharpe of ``n_trials`` independent null strategies."""
    if n_trials < 2:
        raise ValueError("deflation needs at least 2 trials")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance must be non-negative")
    g = EULER_MASCHERONI
    return float(
        np.sqrt(sharpe_variance)
        * (
            (1 - g) * stats.norm.ppf(1 - 1.0 / n_trials)
            + g * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    )


def deflated_sharpe(returns: np.ndarray, *, sr0: float) -> float:
    """Probability the true Sharpe exceeds ``sr0``, given this return series.

    ``DSR = Phi( (SR - SR0) * sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )``
    with ``kurt`` the NON-excess kurtosis of the returns.
    """
    r = np.asarray(returns, dtype="float64")
    t = len(r)
    if t < 3:
        raise ValueError("need at least 3 returns")
    sr = per_period_sharpe(r)
    if not np.isfinite(sr):
        return float("nan")
    skew = float(stats.skew(r, bias=True))
    kurt = float(stats.kurtosis(r, fisher=False, bias=True))
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(t - 1) / np.sqrt(denom)))
