"""EW, 60/40, minimum-variance, risk-parity, vol-target and buy-and-hold SPY. Spec §13.3.

"Min-variance and vol-target are the honest competitors" (§13.3): the reference
project compared only against equal-weight and 60/40. Every benchmark here uses
the same decision calendar, timing contract and cost model as the variant
strategies (preregistration §7), and none of them sees a probe forecast.

Each function returns target weights indexed by decision date, over the same
lines as the variant strategies (13 risky assets, SPY, CASH).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.probes.allocator import (
    CASH, SPY, AllocatorInputs, AllocatorParams, _empty, _exposure_weights, solve_min_variance,
)

__all__ = [
    "equal_weight", "sixty_forty", "minimum_variance", "risk_parity", "vol_target_trailing",
    "buy_and_hold_spy", "capped_inverse_vol", "BENCHMARK_NAMES", "all_benchmarks",
]

BENCHMARK_NAMES = (
    "EqualWeight", "SixtyForty", "MinVariance", "RiskParity", "VolTarget", "BuyHoldSPY",
)


def equal_weight(inputs: AllocatorInputs, params: AllocatorParams) -> pd.DataFrame:
    """1/13 in each risky asset, no cash, rebalanced at every decision."""
    w = _empty(params, inputs.dates)
    for a in params.risky:
        w[a] = 1.0 / len(params.risky)
    return w


def sixty_forty(inputs: AllocatorInputs, params: AllocatorParams) -> pd.DataFrame:
    """60% SPY, 40% IEF, rebalanced at every decision."""
    w = _empty(params, inputs.dates)
    w[SPY] = 0.6
    w["IEF"] = 0.4
    return w


def minimum_variance(inputs: AllocatorInputs, params: AllocatorParams) -> pd.DataFrame:
    """Long-only minimum variance, every line (cash included) capped at ``params.cap``, fully invested."""
    w = _empty(params, inputs.dates)
    for d in inputs.dates:
        sol = solve_min_variance(inputs.cov[d], params.cap)
        w.loc[d, list(params.risky)] = sol[:-1]
        w.loc[d, CASH] = sol[-1]
    return w


def capped_inverse_vol(vol: np.ndarray, cap: float) -> np.ndarray:
    """Inverse-volatility weights, capped, with the excess redistributed pro rata to the uncapped."""
    w = 1.0 / vol
    w = w / w.sum()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            return w
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any():
            raise ValueError("risk-parity cap cannot be satisfied")
        w[free] += excess * w[free] / w[free].sum()
    raise ValueError("risk-parity cap redistribution did not converge")


def risk_parity(inputs: AllocatorInputs, params: AllocatorParams) -> pd.DataFrame:
    """Inverse trailing-60-session volatility over the risky assets, capped, no cash."""
    w = _empty(params, inputs.dates)
    for d in inputs.dates:
        w.loc[d, list(params.risky)] = capped_inverse_vol(inputs.asset_vol.loc[d].to_numpy(), params.cap)
    return w


def vol_target_trailing(inputs: AllocatorInputs, params: AllocatorParams) -> pd.DataFrame:
    """The VT leg with TRAILING realised portfolio vol in place of any probe forecast."""
    exposure = (params.target_vol / inputs.sigma_base).clip(0.0, 1.0)
    return _exposure_weights(exposure, params)


def buy_and_hold_spy(inputs: AllocatorInputs, params: AllocatorParams) -> pd.DataFrame:
    """100% SPY from the first execution, never rebalanced: ONE decision row only."""
    w = _empty(params, inputs.dates[:1])
    w[SPY] = 1.0
    return w


def all_benchmarks(inputs: AllocatorInputs, params: AllocatorParams) -> dict[str, pd.DataFrame]:
    return {
        "EqualWeight": equal_weight(inputs, params),
        "SixtyForty": sixty_forty(inputs, params),
        "MinVariance": minimum_variance(inputs, params),
        "RiskParity": risk_parity(inputs, params),
        "VolTarget": vol_target_trailing(inputs, params),
        "BuyHoldSPY": buy_and_hold_spy(inputs, params),
    }
