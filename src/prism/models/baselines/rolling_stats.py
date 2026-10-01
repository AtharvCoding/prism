"""Hand-picked summary statistics of the same window. Spec §9.4.

Isolates "any summary of the window" from "what the LSTM specifically
learned". Four per-feature moments — mean, std, skew, excess kurtosis —
over the trailing window: a deliberately simple, parameter-free (nothing is
fitted) encoding that a practitioner would reach for before building a
neural network at all.

Dimensionality is ``4 * n_features``, not matched to the LSTM's configured
``latent_dim`` — spec §9.4 does not ask for dimension parity here (unlike
PCA and the random encoder, which share the LSTM's own latent size), and
forcing parity would mean dropping most features' statistics or picking only
one moment, both of which would make this a weaker baseline than it should
be. The comparison that matters (spec §9.3) is downstream probe performance,
which does not require equal input width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from prism.models.encoder.dataset import window_array

__all__ = ["ROLLING_STATS_MOMENTS", "rolling_stats_encode"]

ROLLING_STATS_MOMENTS: tuple[str, ...] = ("mean", "std", "skew", "kurt")


def rolling_stats_encode(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Four moments per feature, per window — causal, parameter-free.

    Row dated ``D`` describes the window ending at ``D``, the same alignment
    contract :func:`prism.models.encoder.dataset.make_windows` guarantees for
    the LSTM encoder, so this baseline's output is drop-in comparable.
    """
    dates, arr = window_array(frame, window)  # (n_windows, window, n_features)
    mean = arr.mean(axis=1)
    std = arr.std(axis=1, ddof=1) if window > 1 else np.zeros(arr.shape[::2])
    skew = scipy_stats.skew(arr, axis=1, bias=False, nan_policy="propagate")
    kurt = scipy_stats.kurtosis(arr, axis=1, bias=False, nan_policy="propagate")
    # A near-constant window (std ~ 0) makes skew/kurtosis divide-by-zero;
    # scipy already returns 0.0 for those rather than NaN/inf, which is the
    # right convention here (no shape in a flat line), but guard explicitly
    # in case a future scipy version changes that.
    skew = np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)
    kurt = np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)

    blocks = {"mean": mean, "std": std, "skew": skew, "kurt": kurt}
    columns = [f"{col}_{moment}" for moment in ROLLING_STATS_MOMENTS for col in frame.columns]
    values = np.concatenate([blocks[m] for m in ROLLING_STATS_MOMENTS], axis=1)
    return pd.DataFrame(values, index=dates, columns=columns)
