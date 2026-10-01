"""Annual expanding-window refits. Spec §9.3.

Mirrors :mod:`prism.models.hmm.walkforward`'s design exactly, because the
underlying requirement is the same: "the agent must train on latents
produced the same way it will see them at test time — otherwise it learns
on clean in-sample encodings and is tested on noisier out-of-sample ones"
(spec §9.3; this is design gap D4 for the encoder specifically). Each fold
fits its own :class:`~prism.features.scaling.FeatureScaler` on its own fit
window only, trains a fresh model with its own internal validation split for
early stopping, and contributes only its apply-window latents to the merged
series — the same "fit window warms up, apply window is kept, nothing is
scored in isolation" shape as the HMM's walk-forward, adapted for a model
whose forward pass needs no running state to carry (each window is encoded
independently, so there is no cross-fold continuation question at all).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from prism.features.scaling import FeatureScaler
from prism.models.encoder.dataset import WindowDataset
from prism.models.encoder.evaluate import compute_latents
from prism.models.encoder.models import build_model
from prism.models.encoder.train import TrainingResult, train_model
from prism.splits import WalkForwardFold, expanding_folds
from prism.utils.logging import get_logger
from prism.utils.seeding import derive_seed

__all__ = ["EncoderFoldOutput", "EncoderWalkforwardResult", "encoder_walkforward"]

_log = get_logger(__name__)


@dataclass
class EncoderFoldOutput:
    fold: WalkForwardFold
    model: torch.nn.Module
    scaler: FeatureScaler
    training: TrainingResult

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.fold.as_dict(),
            "best_epoch": self.training.best_epoch,
            "best_val_loss": self.training.best_val_loss,
            "stopped_early": self.training.stopped_early,
            "n_epochs_run": self.training.n_epochs_run,
        }


@dataclass
class EncoderWalkforwardResult:
    folds: list[EncoderFoldOutput]
    latents: pd.DataFrame

    def as_table(self) -> list[dict[str, Any]]:
        return [f.as_dict() for f in self.folds]


def _internal_train_val_split(
    frame: pd.DataFrame, window: int, val_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The LAST ``val_fraction`` of ``frame`` as an internal validation set.

    Overlaps the training tail by ``window - 1`` rows so the validation
    set's first window is fully formed (not needing rows before the
    split) — the overlap is in the RAW rows feeding the window, not in
    which dates get evaluated: no date is both a training and a validation
    target.
    """
    n = len(frame)
    n_val_target = max(window, int(round(n * val_fraction)))
    if n - n_val_target < window:
        raise ValueError(
            f"fit window has {n} rows; cannot carve out both a >= {window}-row "
            f"training set and a >= {window}-row validation set at "
            f"val_fraction={val_fraction}"
        )
    split_pos = n - n_val_target
    train = frame.iloc[:split_pos]
    val = frame.iloc[split_pos - (window - 1) :]
    return train, val


def encoder_walkforward(
    frame: pd.DataFrame,
    *,
    variant: str,
    window: int,
    hidden_dim: int,
    latent_dim: int,
    fit_start: str | pd.Timestamp,
    first_apply_start: str | pd.Timestamp,
    apply_end: str | pd.Timestamp,
    cadence: str,
    embargo_days: int,
    seed_base: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 25,
    grad_clip_norm: float = 1.0,
    batch_size: int = 64,
    internal_val_fraction: float = 0.15,
    dropout: float = 0.0,
    latent_activation: str = "tanh",
    dae_noise_std: float = 0.1,
    vae_kl_weight: float = 0.001,
    targets: pd.DataFrame | None = None,
    pred_reconstruction_weight: float = 1.0,
    exchange: str = "NYSE",
    device: str | torch.device = "cpu",
) -> EncoderWalkforwardResult:
    """Expanding-window encoder refit. Spec §9.3.

    ``targets``, if given (the ``PRED`` variant), must be a date-indexed
    frame of forward targets; passed straight through to
    :class:`~prism.models.encoder.dataset.WindowDataset`, which drops any
    window whose target row is NaN (spec §5.4 — targets are never filled).
    """
    folds = expanding_folds(
        fit_start, first_apply_start, apply_end, cadence,
        embargo_days=embargo_days, exchange=exchange,
    )

    fold_outputs: list[EncoderFoldOutput] = []
    latent_chunks: list[pd.DataFrame] = []
    n_pred_targets = targets.shape[1] if targets is not None else 0

    for fold in folds:
        fit_raw = frame.loc[fold.fit_start : fold.fit_end]
        apply_raw = frame.loc[fold.apply_start : fold.apply_end]
        if len(fit_raw) < window or len(apply_raw) < 1:
            raise ValueError(
                f"fold {fold.index}: fit window has {len(fit_raw)} rows (need >= "
                f"{window}) or apply window is empty ({len(apply_raw)} rows)"
            )

        scaler = FeatureScaler().fit(fit_raw, scope=f"encoder walkforward fold {fold.index}")
        fit_scaled = scaler.transform(fit_raw)
        apply_scaled = scaler.transform(apply_raw)

        internal_train, internal_val = _internal_train_val_split(
            fit_scaled, window, internal_val_fraction
        )
        train_ds = WindowDataset(internal_train, window, targets=targets)
        val_ds = WindowDataset(internal_val, window, targets=targets)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError(
                f"fold {fold.index}: internal train/val split produced an empty "
                f"dataset (train={len(train_ds)}, val={len(val_ds)}) — likely every "
                "window's target row was NaN"
            )

        torch.manual_seed(derive_seed(seed_base, "encoder_walkforward", fold.index))
        model = build_model(
            variant, input_dim=fit_scaled.shape[1], hidden_dim=hidden_dim,
            latent_dim=latent_dim, window=window, dropout=dropout,
            latent_activation=latent_activation, dae_noise_std=dae_noise_std,
            vae_kl_weight=vae_kl_weight, n_pred_targets=n_pred_targets,
            pred_reconstruction_weight=pred_reconstruction_weight,
        )
        training = train_model(
            model,
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            lr=lr, weight_decay=weight_decay, max_epochs=max_epochs,
            patience=patience, grad_clip_norm=grad_clip_norm, device=device,
        )

        # Latents for the apply window: windowed over fit-tail + apply, so
        # the FIRST apply-window date has a fully-formed trailing window,
        # then restricted to apply-window dates only.
        combined = pd.concat([fit_scaled, apply_scaled])
        if not combined.index.is_monotonic_increasing or not combined.index.is_unique:
            raise ValueError(f"fold {fold.index}: fit/apply windows are not disjoint or sorted")
        apply_ds = WindowDataset(combined, window)
        all_latents = compute_latents(training.model, apply_ds)
        apply_latents = all_latents.loc[all_latents.index.intersection(apply_scaled.index)]

        latent_chunks.append(apply_latents)
        fold_outputs.append(
            EncoderFoldOutput(fold=fold, model=training.model, scaler=scaler, training=training)
        )
        _log.info(
            "fold %d: fit %s..%s (%d rows), apply %s..%s (%d latent rows), "
            "best_epoch=%d val_loss=%.6f stopped_early=%s",
            fold.index, fold.fit_start.date(), fold.fit_end.date(), len(fit_raw),
            fold.apply_start.date(), fold.apply_end.date(), len(apply_latents),
            training.best_epoch, training.best_val_loss, training.stopped_early,
        )

    latents = pd.concat(latent_chunks).sort_index()
    if not latents.index.is_unique:
        dupes = latents.index[latents.index.duplicated()]
        raise AssertionError(
            f"encoder walk-forward produced {len(dupes)} duplicate dates, first "
            f"{dupes[0].date()}"
        )
    if not latents.index.is_monotonic_increasing:
        raise AssertionError("encoder walk-forward latents are not calendar-ordered")

    return EncoderWalkforwardResult(folds=fold_outputs, latents=latents)
