"""Statistics for the Tier 2 policy ablation. Preregistration_tier2 §8-§10.

Same design as Tier 1 (stationary block bootstrap, common random numbers, a
comparison "passes" iff favourable on at least 3 of 4 primary metrics and
adverse on none, deflated Sharpe) adapted to a variant that is a *set of seeds*:

* a variant's statistic is the **mean over its seeds** of a metric of that
  seed's daily net return series;
* each bootstrap replicate resamples **days** (one stationary-block path shared
  by every series: common random numbers) **and seeds** (independently within
  each variant, with replacement), so both sources of uncertainty are in the CI;
* a paired difference X - Y is the difference of those two replicate statistics.

Orientation: every primary metric is oriented so larger is better
(annualised return, Sharpe, max drawdown and CVaR are all <= 0 when negative,
so "less negative" is better), hence *favourable* is a CI entirely above zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.analysis.bootstrap import bootstrap_paths
from prism.utils.seeding import derive_seed

__all__ = [
    "METRICS", "series_metrics_replicates", "point_metrics", "variant_replicates", "paired_table",
    "gate_verdict", "non_overlap_verdict",
]

METRICS = ("annualised_return", "sharpe", "max_drawdown", "cvar_95")
PERIODS = 252


def point_metrics(returns: np.ndarray) -> dict[str, float]:
    """The four primary metrics of one daily net return vector (same definitions as ``backtest.metrics``)."""
    r = np.asarray(returns, dtype="float64")
    n = len(r)
    total = np.prod(1.0 + r)
    ann = total ** (PERIODS / n) - 1.0 if total > 0 else -1.0
    sd = r.std(ddof=1)
    curve = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    mdd = float((curve / np.maximum.accumulate(curve) - 1.0).min())
    var = np.quantile(r, 0.05)
    tail = r[r <= var]
    return {
        "annualised_return": float(ann),
        "sharpe": float(r.mean() / sd * np.sqrt(PERIODS)) if sd > 0 else float("nan"),
        "max_drawdown": mdd,
        "cvar_95": float(tail.mean()) if len(tail) else float(var),
    }


def series_metrics_replicates(returns: np.ndarray, paths: np.ndarray) -> dict[str, np.ndarray]:
    """The four metrics of ``returns[path]`` for every bootstrap path (vectorised)."""
    x = np.asarray(returns, dtype="float64")[paths]                 # (B, n)
    n = x.shape[1]
    total_log = np.log1p(x).sum(axis=1)
    ann = np.expm1(total_log * PERIODS / n)
    sd = x.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, x.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(PERIODS), np.nan)
    curve = np.concatenate([np.ones((x.shape[0], 1)), np.cumprod(1.0 + x, axis=1)], axis=1)
    mdd = (curve / np.maximum.accumulate(curve, axis=1) - 1.0).min(axis=1)
    var = np.quantile(x, 0.05, axis=1, keepdims=True)
    cvar = np.where(x <= var, x, 0.0).sum(axis=1) / np.maximum((x <= var).sum(axis=1), 1)
    return {"annualised_return": ann, "sharpe": sharpe, "max_drawdown": mdd, "cvar_95": cvar}


def variant_replicates(
    series: dict[int, np.ndarray], paths: np.ndarray, *, master: int, variant: str, resample_seeds: bool = True
) -> dict[str, np.ndarray]:
    """Replicate statistics of a variant: per-seed metrics under shared day paths, then a seed resample, then the mean."""
    seeds = sorted(series)
    per_seed = [series_metrics_replicates(series[s], paths) for s in seeds]
    b = paths.shape[0]
    if resample_seeds:
        rng = np.random.default_rng(derive_seed(master, "tier2-seed-resample", variant))
        idx = rng.integers(0, len(seeds), size=(b, len(seeds)))
    else:
        idx = np.tile(np.arange(len(seeds)), (b, 1))
    out = {}
    for m in METRICS:
        mat = np.stack([ps[m] for ps in per_seed], axis=0)            # (S, B)
        out[m] = mat[idx, np.arange(b)[:, None]].mean(axis=1)
    return out


def _ci(x: np.ndarray, level: float) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(x, [a, 1.0 - a])
    return float(lo), float(hi)


def _verdict(lo: float, hi: float) -> str:
    return "favourable" if lo > 0 else "adverse" if hi < 0 else "indeterminate"


def paired_table(
    reps: dict[str, dict[str, np.ndarray]], points: dict[str, dict[str, float]],
    comparisons: list[tuple[str, str]], *, level: float = 0.95,
) -> pd.DataFrame:
    """One row per (comparison, metric): point difference, paired CI, verdict, and the spec's non-overlap verdict."""
    rows = []
    for x, y in comparisons:
        for m in METRICS:
            d = reps[x][m] - reps[y][m]
            lo, hi = _ci(d, level)
            xl, xh = _ci(reps[x][m], level)
            yl, yh = _ci(reps[y][m], level)
            non_overlap = "favourable" if xl > yh else "adverse" if xh < yl else "indeterminate"
            rows.append({
                "candidate": x, "control": y, "metric": m,
                "x": points[x][m], "y": points[y][m], "diff": points[x][m] - points[y][m],
                "ci_low": lo, "ci_high": hi, "verdict": _verdict(lo, hi),
                "x_ci_low": xl, "x_ci_high": xh, "y_ci_low": yl, "y_ci_high": yh, "nonoverlap_verdict": non_overlap,
            })
    return pd.DataFrame(rows)


def gate_verdict(table: pd.DataFrame, x: str, y: str, *, column: str = "verdict") -> dict[str, object]:
    """Pre-registered rule: pass iff favourable on >= 3 of 4 primary metrics and adverse on none."""
    t = table[(table.candidate == x) & (table.control == y)]
    fav = int((t[column] == "favourable").sum())
    adv = int((t[column] == "adverse").sum())
    return {"candidate": x, "control": y, "favourable": fav, "adverse": adv, "of": int(len(t)),
            "pass": bool(fav >= 3 and adv == 0)}


def non_overlap_verdict(table: pd.DataFrame, x: str, y: str) -> dict[str, object]:
    return gate_verdict(table, x, y, column="nonoverlap_verdict")
