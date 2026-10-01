"""Window construction. The alignment contract is the whole point. Spec §9.3.

The reference implementation's off-by-one (defect C1) was two lines, neither
wrong-looking on its own: the window ending at row ``k+29`` was labelled with
the date at row ``k+30`` (a stray ``+1`` in the index slice), and the dataset
was sized ``len(data) - window`` rather than ``len(data) - window + 1``,
silently dropping the most recent window — the only one that matters for a
live decision. Both are fixed here by construction: :func:`make_windows`
derives its output index directly from the input frame's own index at the
position the window ends on, and its length is asserted against the formula
spec §9.3 and ``tests/test_alignment.py`` both state explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["make_windows", "window_array", "WindowDataset"]


def _validate(frame: pd.DataFrame, window: int) -> None:
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    if len(frame) < window:
        raise ValueError(f"frame has {len(frame)} rows, shorter than window={window}")
    if frame.isna().to_numpy().any():
        raise ValueError(
            "frame contains NaN; windowing must run on an already-warm, "
            "already-imputed-or-trimmed frame (prism.features.build handles this)"
        )


def window_array(frame: pd.DataFrame, window: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """``(dates, array)`` with ``array.shape == (n_windows, window, n_features)``.

    The efficient form, built on :func:`numpy.lib.stride_tricks.sliding_window_view`
    — a *view*, not a copy, so windowing a 6,000-row x 130-column frame costs
    nothing beyond the frame's own memory, however large ``window`` is.
    ``dates[i]`` is the date of ``array[i]``'s **last** row — the date the
    window ends on, which is the row it is causally valid to be used for.

    ``len(dates) == len(frame) - window + 1`` — never ``len(frame) - window``
    (defect C1): the window ending at the very last row of ``frame`` is
    included, not dropped.
    """
    _validate(frame, window)
    values = frame.to_numpy(dtype="float64")
    raw = np.lib.stride_tricks.sliding_window_view(values, window_shape=window, axis=0)
    arr = np.moveaxis(raw, -1, 1)  # (n_windows, n_features, window) -> (n_windows, window, n_features)
    dates = pd.DatetimeIndex(frame.index[window - 1 :])
    assert len(dates) == len(frame) - window + 1
    assert arr.shape == (len(dates), window, frame.shape[1])
    return dates, arr


def make_windows(frame: pd.DataFrame, window: int) -> pd.Series:
    """Windows of length ``window``, labelled by the date they **end on**.

    Returns a ``pd.Series`` indexed by that date, each entry a
    ``(window, n_features)`` array — ``windows.loc[D][-1]`` is ``frame.loc[D]``
    itself, by construction, which is the alignment identity spec §9.3 and
    ``tests/test_alignment.py`` check.

    This is the small, obviously-correct form for tests and diagnostics.
    :func:`window_array` is the efficient form actual training uses — same
    guarantee, no per-window copy.
    """
    dates, arr = window_array(frame, window)
    return pd.Series(list(arr), index=dates, name="window")


class WindowDataset(Dataset):
    """A ``torch.utils.data.Dataset`` over windows, for training.

    ``targets``, if given, must be a frame of forward-looking targets aligned
    on the SAME index as ``frame`` (e.g. ``fwd_vol_5``) — used by the ``PRED``
    variant (spec §9.3). Rows whose target is NaN (the trailing rows with no
    future left, spec §5.4) are dropped from the dataset entirely, not
    zero-filled: training a predictive head against a fabricated zero target
    would teach it a wrong, silent lesson about the last `horizon` days of
    every window-able period.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        window: int,
        *,
        targets: pd.DataFrame | None = None,
    ) -> None:
        self.window = window
        self.feature_columns = list(frame.columns)
        dates, arr = window_array(frame, window)

        keep = np.ones(len(dates), dtype=bool)
        target_arr: np.ndarray | None = None
        if targets is not None:
            aligned = targets.reindex(dates)
            keep &= aligned.notna().all(axis=1).to_numpy()
            target_arr = aligned.to_numpy(dtype="float64")

        self.dates = dates[keep]
        self._X = torch.from_numpy(np.ascontiguousarray(arr[keep])).float()
        self._y = (
            torch.from_numpy(np.ascontiguousarray(target_arr[keep])).float()
            if target_arr is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int):
        x = self._X[idx]
        if self._y is None:
            return x, x  # autoencoding: input is its own target
        return x, self._y[idx]
