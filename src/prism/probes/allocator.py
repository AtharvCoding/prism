"""Classical allocators driven by each variant's probe forecast. Spec §13.1; preregistration §7.

A classical allocator does not consume a feature vector, so each variant drives
it through its own walk-forward forecast of ``fwd_vol_5`` (annualised SPY
volatility), produced by the same forecaster, with the same grid and procedure,
for every variant. Three legs per variant:

* **VT**  vol-target on the equal-weight nine-sector base portfolio;
* **RVT** the same with a regime-dependent target (variants with a regime column);
* **MV**  mean-variance with a cash sleeve, so a rescaled covariance changes
  total risky exposure (a fully-invested leg would be invariant to the rescale).

Everything computed at a decision date uses data up to and including that
date's close. Weights are decided at close ``t``, executed at close ``t+1`` and
earn from the execution close (the contract in :mod:`prism.backtest.engine`).

**Cap interpretation.** The 0.35 cap applies to the 13 risky assets in every
strategy. The cash line is the residual of the vol-target and mean-variance
legs and is not capped there (a vol-target strategy must be able to hold
mostly cash); it IS capped at 0.35 in the minimum-variance benchmark, as the
pre-registration states, where an uncapped zero-variance cash line would
otherwise absorb the whole portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

__all__ = [
    "CASH", "SPY", "ALPHA_FLOOR", "AllocatorParams", "AllocatorInputs", "prepare_inputs",
    "vol_target_weights", "regime_vol_target_weights", "mean_variance_weights",
    "solve_mean_variance", "solve_min_variance", "calibrate_gamma",
    "simulate", "net_returns", "IntegrityError", "check_weights", "mv_pairwise_distance",
    "assert_mv_weights_differ",
]

CASH = "CASH"
SPY = "SPY"
ALPHA_FLOOR = 0.05  # floor on the annualised vol forecast, pre-registration §7


class IntegrityError(RuntimeError):
    """A pre-registered fail-loud check failed. The run stops; it does not warn."""


@dataclass(frozen=True)
class AllocatorParams:
    """The fixed, pre-registered numbers (preregistration §7)."""

    risky: tuple[str, ...]
    equity_sectors: tuple[str, ...]
    cap: float = 0.35
    target_vol: float = 0.10
    regime_vol_calm: float = 0.12
    regime_vol_stress: float = 0.06
    cov_window: int = 60
    mu_window: int = 252
    mu_shrink: float = 0.5
    gamma_grid: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0)
    gamma_target_exposure: float = 0.70

    @property
    def lines(self) -> list[str]:
        return [*self.risky, SPY, CASH]


@dataclass
class AllocatorInputs:
    """Every causal input the legs need, computed once for a set of decision dates."""

    dates: pd.DatetimeIndex
    sigma_spy: pd.Series
    sigma_base: pd.Series
    asset_vol: pd.DataFrame
    mu: pd.DataFrame
    cov: dict[pd.Timestamp, np.ndarray]


def prepare_inputs(
    close: pd.DataFrame, params: AllocatorParams, dates: pd.DatetimeIndex, *, rf_column: str = "^IRX"
) -> tuple[AllocatorInputs, pd.DataFrame]:
    """Trailing volatilities, expected excess returns and Ledoit-Wolf covariances at ``dates``.

    Returns the inputs and the per-session line-return frame (risky assets and
    SPY: simple returns; ``CASH``: the PRIOR session's ``^IRX`` close, percent
    annualised, divided by 100 and by 252).
    """
    risky = list(params.risky)
    simple = close[[*risky, SPY]].astype("float64").pct_change()
    rf = (close[rf_column].astype("float64").shift(1) / 100.0 / 252.0).rename(CASH)
    line_returns = pd.concat([simple, rf], axis=1)[params.lines]

    log_spy = np.log(close[SPY].astype("float64")).diff()
    base_ret = simple[list(params.equity_sectors)].mean(axis=1)
    ann = np.sqrt(252.0)
    sigma_spy = log_spy.rolling(params.cov_window).std() * ann
    sigma_base = base_ret.rolling(params.cov_window).std() * ann
    asset_vol = simple[risky].rolling(params.cov_window).std() * ann

    excess = simple[risky].sub(rf, axis=0)
    mu_raw = excess.rolling(params.mu_window).mean() * 252.0
    cross = mu_raw.mean(axis=1)
    mu = ((1 - params.mu_shrink) * mu_raw.sub(cross, axis=0)).add(cross, axis=0)

    cov: dict[pd.Timestamp, np.ndarray] = {}
    for d in dates:
        pos = close.index.get_loc(d)
        window = simple[risky].iloc[pos - params.cov_window + 1 : pos + 1]
        if len(window) < params.cov_window or window.isna().any().any():
            raise ValueError(f"insufficient clean history for the covariance at {d.date()}")
        cov[d] = LedoitWolf().fit(window.to_numpy()).covariance_ * 252.0

    inputs = AllocatorInputs(
        dates=dates, sigma_spy=sigma_spy.reindex(dates), sigma_base=sigma_base.reindex(dates),
        asset_vol=asset_vol.reindex(dates), mu=mu.reindex(dates), cov=cov,
    )
    for name, frame in (("sigma_spy", inputs.sigma_spy), ("sigma_base", inputs.sigma_base),
                        ("asset_vol", inputs.asset_vol), ("mu", inputs.mu)):
        if frame.isna().to_numpy().any():
            raise ValueError(f"allocator input {name} has NaN on decision dates")
    return inputs, line_returns


def _empty(params: AllocatorParams, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=dates, columns=params.lines)


def _exposure_weights(exposure: pd.Series, params: AllocatorParams) -> pd.DataFrame:
    w = _empty(params, exposure.index)
    for s in params.equity_sectors:
        w[s] = exposure / len(params.equity_sectors)
    w[CASH] = 1.0 - exposure
    return w


def vol_target_weights(
    sigma_hat: pd.Series, inputs: AllocatorInputs, params: AllocatorParams
) -> pd.DataFrame:
    """VT leg. ``sigma_hat`` is the (floored) annualised SPY vol forecast at each decision date."""
    return regime_vol_target_weights(sigma_hat, None, inputs, params)


def regime_vol_target_weights(
    sigma_hat: pd.Series, p_stress: pd.Series | None, inputs: AllocatorInputs, params: AllocatorParams
) -> pd.DataFrame:
    """RVT leg. With ``p_stress=None`` the target is the plain 10%, so RVT == VT by construction."""
    s_hat = sigma_hat.reindex(inputs.dates).clip(lower=ALPHA_FLOOR)
    if s_hat.isna().any():
        raise IntegrityError("vol forecast missing on a decision date")
    forecast_b = s_hat * inputs.sigma_base / inputs.sigma_spy
    if p_stress is None:
        target = pd.Series(params.target_vol, index=inputs.dates)
    else:
        p = p_stress.reindex(inputs.dates)
        if p.isna().any():
            raise IntegrityError("regime probability missing on a decision date")
        target = (1 - p) * params.regime_vol_calm + p * params.regime_vol_stress
    exposure = (target / forecast_b).clip(0.0, 1.0)
    return _exposure_weights(exposure, params)


def solve_mean_variance(mu: np.ndarray, cov: np.ndarray, gamma: float, cap: float) -> np.ndarray:
    """``max mu'w - gamma/2 w'Cw``  s.t. ``0 <= w <= cap``, ``sum(w) <= 1`` (remainder is cash)."""
    n = len(mu)
    res = minimize(
        lambda w: -(mu @ w) + 0.5 * gamma * w @ cov @ w, np.full(n, min(0.01, cap)),
        jac=lambda w: -mu + gamma * cov @ w, bounds=[(0.0, cap)] * n,
        constraints=[{"type": "ineq", "fun": lambda w: 1.0 - w.sum(), "jac": lambda w: -np.ones(n)}],
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise IntegrityError(f"mean-variance QP did not reach an optimum: {res.message}")
    w = np.clip(res.x, 0.0, cap)
    if w.sum() > 1.0 + 1e-8:
        raise IntegrityError(f"mean-variance weights sum to {w.sum():.10f} > 1")
    return w / max(1.0, w.sum())


def mean_variance_weights(
    sigma_hat: pd.Series, inputs: AllocatorInputs, params: AllocatorParams, gamma: float
) -> pd.DataFrame:
    """MV leg: Ledoit-Wolf covariance rescaled by ``(sigma_hat / sigma_spy)^2``."""
    s_hat = sigma_hat.reindex(inputs.dates).clip(lower=ALPHA_FLOOR)
    if s_hat.isna().any():
        raise IntegrityError("vol forecast missing on a decision date")
    w = _empty(params, inputs.dates)
    for d in inputs.dates:
        scale = (s_hat[d] / inputs.sigma_spy[d]) ** 2
        wi = solve_mean_variance(inputs.mu.loc[d].to_numpy(), inputs.cov[d] * scale, gamma, params.cap)
        w.loc[d, list(params.risky)] = wi
        w.loc[d, CASH] = 1.0 - wi.sum()
    return w


def solve_min_variance(cov: np.ndarray, cap: float) -> np.ndarray:
    """Minimum variance over the risky assets PLUS a zero-variance cash line, all in ``[0, cap]``, sum 1."""
    n = cov.shape[0]
    full = np.zeros((n + 1, n + 1))
    full[:n, :n] = cov
    full[n, n] = 1e-10
    res = minimize(
        lambda w: 0.5 * w @ full @ w, np.full(n + 1, 1.0 / (n + 1)), jac=lambda w: full @ w,
        bounds=[(0.0, cap)] * (n + 1),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n + 1)}],
        method="SLSQP", options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not res.success:
        raise IntegrityError(f"minimum-variance QP did not reach an optimum: {res.message}")
    return np.clip(res.x, 0.0, cap)


def calibrate_gamma(
    sigma_hat_val: pd.Series, inputs_val: AllocatorInputs, params: AllocatorParams
) -> tuple[float, pd.DataFrame]:
    """Pick gamma from the grid so V1's mean risky exposure on VALIDATION decisions is nearest 0.70.

    Computed once, from validation data only, before any test-split allocator
    output exists; the same gamma is then used for every variant.
    """
    rows = []
    for g in params.gamma_grid:
        w = mean_variance_weights(sigma_hat_val, inputs_val, params, g)
        rows.append({"gamma": g, "mean_risky_exposure": float(1.0 - w[CASH].mean())})
    table = pd.DataFrame(rows)
    best = (table["mean_risky_exposure"] - params.gamma_target_exposure).abs().idxmin()
    return float(table.loc[best, "gamma"]), table


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def simulate(
    weights: pd.DataFrame, line_returns: pd.DataFrame, *, execution_lag: int = 1
) -> tuple[pd.Series, pd.Series]:
    """Daily GROSS portfolio returns and one-way turnover at each execution, drift-aware.

    Decision at close ``t`` -> executed at close ``t + lag``. On an execution
    date the day's return is earned with the OLD (drifted) weights, then the
    portfolio is rebalanced to target at that close and the one-way turnover
    ``0.5 * sum |target - drifted|`` is recorded. The portfolio starts in cash
    and the initiation is a trade. Costs are applied by :func:`net_returns`.
    """
    sessions = pd.DatetimeIndex(line_returns.index)
    cols = list(weights.columns)
    R = line_returns[cols].to_numpy(dtype="float64")
    exec_pos: dict[int, np.ndarray] = {}
    for d, row in zip(weights.index, weights.to_numpy(dtype="float64")):
        pos = sessions.get_loc(d) + execution_lag
        if pos >= len(sessions):
            continue
        exec_pos[pos] = row
    if not exec_pos:
        raise ValueError("no decision executes inside the return index")
    start = min(exec_pos)
    w = np.zeros(len(cols))
    w[cols.index(CASH)] = 1.0
    gross = np.full(len(sessions) - start, np.nan)
    turn = np.zeros(len(sessions) - start)
    for t in range(start, len(sessions)):
        r = R[t]
        if not np.isfinite(r).all():
            raise IntegrityError(f"non-finite line return on {sessions[t].date()}")
        port = float(w @ r)
        gross[t - start] = port
        w = w * (1.0 + r) / (1.0 + port)
        if t in exec_pos:
            target = exec_pos[t]
            turn[t - start] = 0.5 * np.abs(target - w).sum()
            w = target.copy()
    idx = sessions[start:]
    return pd.Series(gross, index=idx, name="gross"), pd.Series(turn, index=idx, name="turnover")


def net_returns(gross: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """Gross return less ``cost_bps`` per side on the day's one-way turnover."""
    return (gross - cost_bps / 1e4 * turnover.reindex(gross.index).fillna(0.0)).rename("net")


# --------------------------------------------------------------------------- #
# fail-loud checks
# --------------------------------------------------------------------------- #
def check_weights(
    name: str, weights: pd.DataFrame, params: AllocatorParams, *, cash_capped: bool = False, enforce_cap: bool = True
) -> None:
    """Long-only, risky <= cap, sum 1, finite (pre-registration §7).

    ``enforce_cap=False`` is for the 60/40 benchmark only: 60% SPY / 40% IEF is a
    definition, not an allocation under the strategies' concentration limit.
    """
    w = weights.to_numpy(dtype="float64")
    if not np.isfinite(w).all():
        raise IntegrityError(f"{name}: non-finite weights")
    if (w < -1e-9).any():
        raise IntegrityError(f"{name}: negative weight (long-only violated)")
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-8):
        raise IntegrityError(f"{name}: weights do not sum to 1")
    risky = weights[list(params.risky)].to_numpy()
    if enforce_cap and (risky > params.cap + 1e-8).any():
        raise IntegrityError(f"{name}: a risky weight exceeds the {params.cap} cap")
    if cash_capped and (weights[CASH].to_numpy() > params.cap + 1e-8).any():
        raise IntegrityError(f"{name}: the cash line exceeds the {params.cap} cap")


def mv_pairwise_distance(mv_weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean one-way weight distance between every pair of variants' MV weights."""
    names = list(mv_weights)
    out = pd.DataFrame(0.0, index=names, columns=names)
    for a in names:
        for b in names:
            out.loc[a, b] = float(0.5 * (mv_weights[a] - mv_weights[b]).abs().sum(axis=1).mean())
    return out


def assert_mv_weights_differ(distance: pd.DataFrame, *, minimum: float = 0.001) -> None:
    """Pre-registered: at least one pair of variants must differ by >= ``minimum`` mean one-way distance."""
    off = distance.to_numpy()[~np.eye(len(distance), dtype=bool)]
    if not (off >= minimum).any():
        raise IntegrityError(
            f"mean-variance weights are indistinguishable across every pair of variants "
            f"(max mean one-way distance {off.max():.2e} < {minimum}); the MV leg would be "
            "uninformative by construction"
        )
