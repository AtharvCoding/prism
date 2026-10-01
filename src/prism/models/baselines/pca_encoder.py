"""PCA to the same latent dimension on the flattened window. Spec §9.4.

The mandatory "did a learned nonlinear encoder beat a linear one" control.
PCA is **fit** (train-window-only, via :class:`~prism.fitting.FittedArtifact`)
on the *flattened* window — ``window * n_features`` columns, one row per
window-ending-date — never on the raw frame: the comparison with the LSTM
encoder is only fair if both see the same windowed information.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.decomposition import PCA

from prism.fitting import FittedArtifact
from prism.models.encoder.dataset import window_array

__all__ = ["flatten_windows", "PCAEncoder"]


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
