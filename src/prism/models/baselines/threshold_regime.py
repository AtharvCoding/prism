"""VIX / realised-vol quantile rules — the C2 control and the HMM's honest
competitor. Spec §8.7, §10.

"If the HMM cannot beat a two-state volatility threshold on both detection
lag and downstream probe performance, that is a finding — report it" (§8.7).
A quantile rule on a single signal is the simplest regime detector that
could plausibly work, and the HMM has to earn its complexity against it, not
merely against "no regime signal at all".

Like the HMM, this is **causal and walk-forward**: the quantile breakpoints
that define "high VIX" are fit on a trailing window and applied forward,
never on the whole sample — a threshold computed with knowledge of the
entire series' future distribution would be exactly the kind of leak the
rest of this project exists to rule out, and would make for an unfairly weak
competitor.

Output is **one-hot**, not probabilistic (spec §10: "V1 + threshold-regime
one-hot"), reflecting what this kind of rule actually is: a hard
classification, with no notion of posterior uncertainty to report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prism.splits import expanding_folds
from prism.utils.logging import get_logger

__all__ = [
    "ThresholdBreakpoints",
    "fit_threshold_breakpoints",
    "apply_threshold_breakpoints",
    "one_hot",
    "threshold_regime_walkforward",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class ThresholdBreakpoints:
    """``k - 1`` quantile edges partitioning a signal into ``k`` equal-frequency bins."""

    edges: np.ndarray
    k: int
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp


def fit_threshold_breakpoints(
    signal: pd.Series, k: int, *, scope: str = ""
) -> ThresholdBreakpoints:
    """Equal-frequency breakpoints from ``signal`` (already restricted to a fit window).

    Spec §8.4's "volatility ladder" failure mode — states that differ only in
    the level of one variable — is exactly what a quantile rule *is*, by
    construction. That is the honest baseline, not a defect to fix here: the
    question §8.7 asks is whether the HMM earns its complexity *beyond* this.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2; got {k}")
    quantiles = np.linspace(0.0, 1.0, k + 1)[1:-1]
    edges = signal.quantile(quantiles).to_numpy()
    if np.any(np.diff(edges) <= 0):
        _log.warning(
            "threshold breakpoints for %r are not strictly increasing (degenerate "
            "or heavily-tied signal); some bins may end up empty",
            scope,
        )
    idx = pd.DatetimeIndex(signal.index)
    return ThresholdBreakpoints(edges=edges, k=k, fit_start=idx[0], fit_end=idx[-1])


def apply_threshold_breakpoints(signal: pd.Series, breakpoints: ThresholdBreakpoints) -> pd.Series:
    """Assign each observation to bin ``0..k-1`` using FIXED (already-fit) edges.

    Causal by construction: the edges are a parameter fit once, elsewhere, on
    a disjoint past window, and applying them here does not look at
    ``signal``'s own distribution at all.
    """
    bins = np.searchsorted(breakpoints.edges, signal.to_numpy(dtype="float64"), side="right")
    return pd.Series(bins, index=signal.index, name="threshold_state", dtype="int64")


def one_hot(states: pd.Series, k: int) -> pd.DataFrame:
    """One-hot encode integer states ``0..k-1`` as ``state_0 .. state_{k-1}`` columns.

    Matches :func:`prism.models.hmm.walkforward.hmm_walkforward`'s posterior
    column convention exactly, so C2 (this) and V3 (the HMM) are
    interchangeable inputs to state assembly (spec §10).
    """
    values = states.to_numpy()
    if values.min() < 0 or values.max() >= k:
        raise ValueError(f"states outside [0, {k}): min={values.min()}, max={values.max()}")
    out = np.zeros((len(states), k), dtype="float64")
    out[np.arange(len(states)), values] = 1.0
    columns = [f"state_{i}" for i in range(k)]
    return pd.DataFrame(out, index=states.index, columns=columns)


def threshold_regime_walkforward(
    signal: pd.Series,
    *,
    k: int,
    fit_start: str | pd.Timestamp,
    first_apply_start: str | pd.Timestamp,
    apply_end: str | pd.Timestamp,
    cadence: str,
    embargo_days: int,
    exchange: str = "NYSE",
) -> pd.DataFrame:
    """Walk-forward one-hot threshold states, on the same fold schedule as the HMM.

    Refitting the breakpoints on the same expanding, embargoed fold schedule
    :func:`~prism.models.hmm.walkforward.hmm_walkforward` uses is what keeps
    the comparison fair: both detectors have access to exactly the same
    causal information at every date, so a difference in detection
    performance reflects the detectors themselves, not an information
    asymmetry between them.
    """
    folds = expanding_folds(
        fit_start, first_apply_start, apply_end, cadence,
        embargo_days=embargo_days, exchange=exchange,
    )
    chunks: list[pd.DataFrame] = []
    for fold in folds:
        fit_signal = signal.loc[fold.fit_start : fold.fit_end]
        apply_signal = signal.loc[fold.apply_start : fold.apply_end]
        if fit_signal.empty or apply_signal.empty:
            raise ValueError(f"fold {fold.index}: empty fit or apply slice")
        breakpoints = fit_threshold_breakpoints(fit_signal, k, scope=f"fold {fold.index}")
        states = apply_threshold_breakpoints(apply_signal, breakpoints)
        chunks.append(one_hot(states, k))

    out = pd.concat(chunks).sort_index()
    if not out.index.is_unique:
        dupes = out.index[out.index.duplicated()]
        raise AssertionError(f"threshold walk-forward produced duplicate dates, first {dupes[0]}")
    return out
