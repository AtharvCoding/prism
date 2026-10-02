"""Stationary block bootstrap CIs, block ~20 days. Spec §14.2.

Financial return (and residual, and error) series are serially correlated —
an ordinary i.i.d. bootstrap would understate every confidence interval by
resampling individual days as if they were independent draws. The
**stationary bootstrap** (Politis & Romano, 1994) resamples *blocks* of
geometrically-distributed random length (mean ``block_length``) with
circular wraparound, which preserves local serial dependence while still
being a stationary resampling scheme (unlike a fixed-length moving-block
bootstrap, which has edge effects at block boundaries). This is
``configs/experiments/tier1_probes.yaml``'s ``uncertainty.bootstrap:
"stationary_block"``.

A variant "wins" (spec §14.2) only if its CI does not overlap the
comparator's — :func:`non_overlapping` is the one-line check every gate
decision in this project is supposed to use, so it exists once rather than
being re-derived ad hoc at every comparison site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "BootstrapResult",
    "stationary_block_bootstrap_ci",
    "non_overlapping",
    "bootstrap_paths",
    "path_ci",
]


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_bootstrap: int
    block_length: int
    replicate_statistics: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "n_bootstrap": self.n_bootstrap,
            "block_length": self.block_length,
        }


def _stationary_bootstrap_indices(
    n: int, block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """One resampled index path of length ``n`` from the stationary bootstrap.

    Built as a sequence of circular runs of geometrically-distributed length
    (mean ``block_length``), each starting from a uniformly random position —
    the direct, block-at-a-time construction of the Politis-Romano scheme,
    rather than a per-sample Bernoulli-restart loop (mathematically
    equivalent, but a Python loop over every one of ``n`` samples per
    replicate would dominate runtime at ``n_bootstrap`` in the thousands).
    """
    p = 1.0 / block_length
    out = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        length = min(length, n - filled)
        out[filled : filled + length] = (start + np.arange(length)) % n
        filled += length
    return out


def stationary_block_bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    block_length: int = 20,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile CI for ``statistic(values)`` under the stationary bootstrap.

    ``statistic`` is any function of a 1-D array to a scalar — a mean, an R²
    computed against a fixed set of predictions via closure, a Sharpe ratio.
    The point estimate is ``statistic`` applied to the ORIGINAL (not
    resampled) data; the interval comes from the empirical percentiles of
    the resampled replicates.
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D; got shape {values.shape}")
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values to bootstrap")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be in (0, 1)")

    point = float(statistic(values))
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        replicates[i] = statistic(values[idx])

    alpha = 1.0 - ci_level
    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        point_estimate=point, ci_low=float(lo), ci_high=float(hi), ci_level=ci_level,
        n_bootstrap=n_bootstrap, block_length=block_length, replicate_statistics=replicates,
    )


def non_overlapping(a: BootstrapResult, b: BootstrapResult) -> bool:
    """True iff ``a``'s and ``b``'s confidence intervals do not overlap.

    A variant "wins" (spec §14.2) only when this is true AND its point
    estimate is on the favourable side — checking non-overlap alone does not
    say which one is better, only that the gap is unlikely to be noise.
    """
    return a.ci_high < b.ci_low or b.ci_high < a.ci_low


def bootstrap_paths(n: int, block_length: int, n_bootstrap: int, seed: int) -> np.ndarray:
    """``(n_bootstrap, n)`` stationary-bootstrap index paths, fixed by ``(n, block_length, seed)``.

    The pre-registration (§6) fixes common random numbers: every variant,
    target and comparison on the test split reuses the SAME resampled days, so
    a difference between two variants is resampled as a genuine pair rather
    than as two independent draws. Generating the paths once, here, and
    applying them to whatever vector is being summarised is what makes that
    structural instead of a convention a caller could forget.
    """
    if n < 2:
        raise ValueError("need at least 2 observations to bootstrap")
    rng = np.random.default_rng(seed)
    return np.stack([_stationary_bootstrap_indices(n, block_length, rng) for _ in range(n_bootstrap)])


def path_ci(
    values: np.ndarray, paths: np.ndarray, *, ci_level: float = 0.95
) -> tuple[float, float, float]:
    """``(point, lo, hi)`` for the MEAN of ``values`` under pre-generated ``paths``."""
    values = np.asarray(values, dtype="float64")
    if paths.shape[1] != len(values):
        raise ValueError(f"paths are for n={paths.shape[1]} but values has {len(values)} rows")
    replicates = values[paths].mean(axis=1)
    alpha = 1.0 - ci_level
    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
