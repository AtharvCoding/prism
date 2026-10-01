"""Per-feature reconstruction MSE, dead units, latent drift. Spec §9.3.

Three diagnostics, each replacing a named reference defect with something
measured rather than eyeballed:

* **Reconstruction quality, out of sample, per feature family and period**
  (:func:`evaluate_reconstruction`) — the reference reported MSE on the
  first *training* batch (defect C5), which is both in-sample and a single
  batch's worth of luck.
* **Dead units** (:func:`dead_units`) — variance of each latent dimension
  across a whole evaluation period; defect C3's ``latent_0``/``latent_6``
  were identically zero for the *entire* sample, which a single-batch check
  would not reliably catch (a dead unit can have nonzero variance over 64
  random rows and still be solidly dead over 6,000).
* **Latent drift** (:func:`latent_drift`) — per-dimension KS test between
  two periods' latent distributions. Not meant to prove "the same
  representation still works" (latents are not comparable across *seeds* at
  all — rotation/permutation, spec §9.3 — so this is a within-one-fitted-
  model diagnostic, used across TIME, not across seeds).

Explicitly **not** here: PCA-by-regime separation. Defect C6 used exactly
that as "validation" — circular, since the features already contain
volatility and the regimes are themselves a function of volatility, so
separation is guaranteed regardless of whether the encoder learned anything
beyond what PCA already would. See ``tests/test_encoder.py::test_pca_by_regime_is_not_used_as_validation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from prism.models.encoder.dataset import WindowDataset

__all__ = [
    "compute_latents",
    "ReconstructionEvaluation",
    "evaluate_reconstruction",
    "DeadUnitsReport",
    "dead_units",
    "LatentDriftReport",
    "latent_drift",
]


@torch.no_grad()
def compute_latents(model: torch.nn.Module, dataset: WindowDataset, *, batch_size: int = 256) -> pd.DataFrame:
    """Latents for every window in ``dataset``, indexed by the window's end date.

    Always in ``eval()`` mode (no DAE noise, VAE returns the posterior mean
    not a sample) — this is what any downstream consumer of the latent
    actually sees, so diagnostics computed any other way would describe a
    model nobody uses.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    for x, _ in loader:
        chunks.append(model.latent(x).cpu().numpy())
    latent = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0))
    columns = [f"latent_{i}" for i in range(latent.shape[1])]
    return pd.DataFrame(latent, index=dataset.dates, columns=columns)


@dataclass
class ReconstructionEvaluation:
    mse_overall: float
    mse_per_feature: pd.Series
    mse_per_family: pd.Series
    n_windows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mse_overall": self.mse_overall,
            "n_windows": self.n_windows,
            "mse_per_family": self.mse_per_family.to_dict(),
        }


@torch.no_grad()
def evaluate_reconstruction(
    model: torch.nn.Module,
    dataset: WindowDataset,
    *,
    families: dict[str, list[str]] | None = None,
    batch_size: int = 256,
) -> ReconstructionEvaluation:
    """Reconstruction MSE overall, per feature, and per feature family.

    Requires a model exposing reconstruction (``AE``/``DAE``/``VAE``, or
    ``PRED`` with its decoder attached); raises clearly for a pure-predictive
    ``PRED`` model rather than silently returning zeros.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    columns = dataset.feature_columns
    sq_err_sum = np.zeros(len(columns))
    n_total = 0

    for x, _ in loader:
        out = model(x)
        if len(out) == 2:
            # Autoencoder / DenoisingAutoencoder / VariationalAutoencoder: (recon, latent)
            recon, _ = out
        elif len(out) == 3:
            # PredictiveAutoencoder: (pred, recon, latent)
            _, recon, _ = out
        else:  # pragma: no cover - defensive against a future variant shape
            raise ValueError(f"unrecognised forward() output of length {len(out)}")
        if recon is None:
            raise ValueError(
                f"{type(model).__name__} has no reconstruction output "
                "(a PRED model with reconstruction_weight=0 has no decoder)"
            )
        err = (recon - x).pow(2).cpu().numpy()  # (batch, window, n_features)
        sq_err_sum += err.sum(axis=(0, 1))
        n_total += err.shape[0] * err.shape[1]

    if n_total == 0:
        raise ValueError("dataset produced zero windows")
    mse_per_feature = pd.Series(sq_err_sum / n_total, index=columns, name="mse")
    mse_overall = float(mse_per_feature.mean())

    if families is None:
        mse_per_family = pd.Series({"all": mse_overall})
    else:
        rows = {}
        for family, cols in families.items():
            present = [c for c in cols if c in mse_per_feature.index]
            rows[family] = float(mse_per_feature[present].mean()) if present else float("nan")
        mse_per_family = pd.Series(rows)

    return ReconstructionEvaluation(
        mse_overall=mse_overall, mse_per_feature=mse_per_feature,
        mse_per_family=mse_per_family, n_windows=len(dataset),
    )


@dataclass
class DeadUnitsReport:
    variances: pd.Series
    dead_mask: pd.Series
    n_dead: int
    n_total: int

    @property
    def dead_columns(self) -> list[str]:
        return list(self.variances.index[self.dead_mask])

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_dead": self.n_dead, "n_total": self.n_total,
            "dead_columns": self.dead_columns,
            "variances": self.variances.to_dict(),
        }


def dead_units(latents: pd.DataFrame, *, var_threshold: float) -> DeadUnitsReport:
    """Latent dimensions whose variance, over a whole evaluation period, is
    below ``var_threshold``. Defect C3's dead units were zero for the ENTIRE
    sample — this is a period-wide statistic, not a per-batch one, for
    exactly that reason."""
    variances = latents.var(ddof=1)
    dead_mask = variances < var_threshold
    return DeadUnitsReport(
        variances=variances, dead_mask=dead_mask,
        n_dead=int(dead_mask.sum()), n_total=len(variances),
    )


@dataclass
class LatentDriftReport:
    per_dimension_pvalue: pd.Series
    n_drifted: int
    n_total: int
    alpha: float

    @property
    def drifted_dimensions(self) -> list[str]:
        return list(self.per_dimension_pvalue.index[self.per_dimension_pvalue < self.alpha])

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_drifted": self.n_drifted, "n_total": self.n_total, "alpha": self.alpha,
            "drifted_dimensions": self.drifted_dimensions,
            "per_dimension_pvalue": self.per_dimension_pvalue.to_dict(),
        }


def latent_drift(
    latents_a: pd.DataFrame, latents_b: pd.DataFrame, *, test: str = "ks", alpha: float = 0.01
) -> LatentDriftReport:
    """Per-dimension distribution drift between two periods' latents.

    ``test="ks"`` (two-sample Kolmogorov-Smirnov) is the spec-named default;
    a within-model, across-TIME diagnostic — comparing latents from two
    different seeds' models would be meaningless (spec §9.3: latents are not
    comparable across seeds).
    """
    if test != "ks":
        raise ValueError(f"unsupported test {test!r}; only 'ks' is implemented")
    shared = [c for c in latents_a.columns if c in latents_b.columns]
    if not shared:
        raise ValueError("latents_a and latents_b share no columns")
    pvalues = {}
    for col in shared:
        a = latents_a[col].dropna().to_numpy()
        b = latents_b[col].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            pvalues[col] = float("nan")
            continue
        pvalues[col] = float(stats.ks_2samp(a, b).pvalue)
    series = pd.Series(pvalues, name="pvalue")
    drifted = series < alpha
    return LatentDriftReport(
        per_dimension_pvalue=series, n_drifted=int(drifted.sum()), n_total=len(series), alpha=alpha,
    )
