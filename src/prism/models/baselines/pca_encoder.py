"""PCA to the same latent dimension on the flattened window. Spec §9.4.

The mandatory "did a learned nonlinear encoder beat a linear one" control.
PCA is **fit** (train-window-only, via :class:`~prism.fitting.FittedArtifact`)
on the *flattened* window — ``window * n_features`` columns, one row per
window-ending-date — never on the raw frame: the comparison with the LSTM
encoder is only fair if both see the same windowed information.

**Walk-forward matters here too, and was missed in an earlier version of
the step 3 comparison script.** A PCA fit once on 1999-2006 and left
unrefit through 2023 is not a fair baseline for an LSTM that re-adapts every
year: any apparent LSTM weakness could just be the LSTM's annual refit
chasing a moving target while PCA's fixed projection happens to still work
tolerably on data two decades later, or it could cut the other way just as
easily. :func:`pca_encoder_walkforward` refits PCA on the identical
expanding, embargoed fold schedule
:mod:`prism.models.encoder.walkforward`'s ``encoder_walkforward`` uses, so a
difference in downstream performance reflects the representations
themselves, not an asymmetry in how current each one's fit is.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.decomposition import PCA

from prism.fitting import FittedArtifact
from prism.models.encoder.dataset import window_array
from prism.splits import expanding_folds

__all__ = ["flatten_windows", "PCAEncoder", "pca_encoder_walkforward"]


def flatten_windows(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Each window flattened into one row: ``window * n_features`` columns.

    Column order is ``t-{window-1}_{col}, ..., t-0_{col}`` per original
    column, i.e. time-major then feature-minor — stable and inspectable,
    not an implementation-detail reshape order.
    """
    dates, arr = window_array(frame, window)
    n_windows, w, n_features = arr.shape
    flat = arr.reshape(n_windows, w * n_features)
    columns = [f"t-{w - 1 - t}_{c}" for t in range(w) for c in frame.columns]
    return pd.DataFrame(flat, index=dates, columns=columns)


class PCAEncoder(FittedArtifact):
    """Fit PCA on a training window's flattened windows; transform any other."""

    def __init__(self, n_components: int = 8) -> None:
        # Default matches encoder.yaml's smallest configured latent_dim, so
        # PCAEncoder() is constructible with no arguments for the generic
        # FittedArtifact contract tests (tests/test_fit_scope.py). Production
        # call sites always pass an explicit n_components.
        super().__init__()
        self.n_components = n_components
        self._pca: PCA | None = None
        self._columns: list[str] | None = None

    def _fit(self, frame: pd.DataFrame) -> None:
        if frame.shape[1] < self.n_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds the flattened window's "
                f"{frame.shape[1]} columns"
            )
        self._columns = list(frame.columns)
        self._pca = PCA(n_components=self.n_components, random_state=0)
        self._pca.fit(frame.to_numpy(dtype="float64"))

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self._pca is not None and self._columns is not None
        missing = [c for c in self._columns if c not in frame.columns]
        if missing:
            raise KeyError(f"PCAEncoder was fitted on columns absent here: {missing[:5]}")
        values = frame[self._columns].to_numpy(dtype="float64")
        latent = self._pca.transform(values)
        cols = [f"latent_{i}" for i in range(self.n_components)]
        return pd.DataFrame(latent, index=frame.index, columns=cols)

    def _params(self) -> dict[str, Any]:
        assert self._pca is not None
        return {
            "n_components": self.n_components,
            "components": self._pca.components_,
            "mean": self._pca.mean_,
            "explained_variance_ratio": self._pca.explained_variance_ratio_,
        }

    @property
    def explained_variance_ratio(self):
        assert self._pca is not None, "PCAEncoder is not fitted"
        return self._pca.explained_variance_ratio_


def pca_encoder_walkforward(
    frame: pd.DataFrame,
    *,
    window: int,
    n_components: int,
    fit_start: str | pd.Timestamp,
    first_apply_start: str | pd.Timestamp,
    apply_end: str | pd.Timestamp,
    cadence: str,
    embargo_days: int,
    exchange: str = "NYSE",
) -> pd.DataFrame:
    """Walk-forward PCA latents, refit on the same fold schedule as the LSTM.

    Each fold: flatten that fold's fit-window-worth of raw frame into
    windows, fit a fresh :class:`PCAEncoder` on them, and transform the
    fold's apply-window's own flattened windows. Returns the merged,
    de-duplicated, calendar-aligned apply-window latent series — the same
    contract :func:`~prism.models.encoder.walkforward.encoder_walkforward`
    and :func:`~prism.models.baselines.threshold_regime.threshold_regime_walkforward`
    both produce.
    """
    folds = expanding_folds(
        fit_start, first_apply_start, apply_end, cadence,
        embargo_days=embargo_days, exchange=exchange,
    )
    chunks: list[pd.DataFrame] = []
    for fold in folds:
        fit_raw = frame.loc[fold.fit_start : fold.fit_end]
        apply_raw = frame.loc[fold.apply_start : fold.apply_end]
        if len(fit_raw) < window or apply_raw.empty:
            raise ValueError(f"fold {fold.index}: insufficient fit or apply data")

        # Flatten over fit+apply together so the apply window's FIRST date
        # has a fully-formed trailing window reaching back into the fit
        # period, exactly like encoder_walkforward's latent extraction.
        combined = pd.concat([fit_raw, apply_raw])
        flat = flatten_windows(combined, window)
        fit_flat = flat.loc[flat.index.intersection(fit_raw.index)]

        encoder = PCAEncoder(n_components=n_components).fit(fit_flat, scope=f"pca fold {fold.index}")
        apply_flat = flat.loc[flat.index.intersection(apply_raw.index)]
        chunks.append(encoder.transform(apply_flat))

    out = pd.concat(chunks).sort_index()
    if not out.index.is_unique:
        dupes = out.index[out.index.duplicated()]
        raise AssertionError(f"PCA walk-forward produced duplicate dates, first {dupes[0]}")
    return out
