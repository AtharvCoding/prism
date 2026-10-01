"""Validation-driven training loop with early stopping. Spec §9.3.

Defect C2, precisely: the reference had no validation set at all. "Best
weights" meant lowest *training* loss, which selects for memorisation, not
generalisation — and the training loss was still falling at epoch 100, so
the model was not even converged by that meaningless criterion either.

Both halves of the fix are structural, not a convention a caller could
forget: :func:`train_model` requires a validation loader as a positional
argument (there is no training-loss-only code path to accidentally take),
and it trains until validation loss stops improving for ``patience`` epochs
— "to convergence, not a fixed epoch count" (spec §9.3) — rather than for a
hard-coded 100.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from prism.utils.logging import get_logger

__all__ = ["TrainingCurve", "TrainingResult", "train_model"]

_log = get_logger(__name__)


@dataclass
class TrainingCurve:
    """Per-epoch loss, train and validation, every component logged."""

    train: list[dict[str, float]] = field(default_factory=list)
    val: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"train": self.train, "val": self.val}


@dataclass
class TrainingResult:
    model: torch.nn.Module
    curve: TrainingCurve
    best_epoch: int
    best_val_loss: float
    stopped_early: bool
    n_epochs_run: int


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float | None,
    device: torch.device,
) -> dict[str, float]:
    """One pass over ``loader``. Training if ``optimizer`` is given, eval otherwise."""
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: dict[str, float] = {}
    n_batches = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if train_mode:
                optimizer.zero_grad()
            out = model.loss(x, y)
            if train_mode:
                out.total.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            for key, value in out.as_floats().items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

    if n_batches == 0:
        raise ValueError("loader produced zero batches")
    return {key: value / n_batches for key, value in totals.items()}


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    grad_clip_norm: float,
    device: str | torch.device = "cpu",
    log_every: int = 10,
) -> TrainingResult:
    """Train until validation loss stops improving for ``patience`` epochs.

    "Best weights" are a deep copy of the model's state at the epoch with
    the lowest VALIDATION total loss — never the training loss, and never
    just whatever the final epoch happened to produce. Training stops at
    ``max_epochs`` only if patience never triggers first; ``stopped_early``
    on the result records which happened, so a caller can tell "converged"
    from "ran out of budget" rather than treating every run identically.
    """
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    curve = TrainingCurve()
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    patience_counter = 0
    stopped_early = False
    epoch = 0

    for epoch in range(max_epochs):
        train_metrics = _run_epoch(
            model, train_loader, optimizer=optimizer, grad_clip_norm=grad_clip_norm, device=device
        )
        val_metrics = _run_epoch(model, val_loader, optimizer=None, grad_clip_norm=None, device=device)
        curve.train.append(train_metrics)
        curve.val.append(val_metrics)

        val_loss = val_metrics["total"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % log_every == 0 or patience_counter == 0:
            _log.info(
                "epoch %4d: train_loss=%.6f val_loss=%.6f best=%.6f@%d patience=%d/%d",
                epoch + 1, train_metrics["total"], val_loss, best_val_loss, best_epoch + 1,
                patience_counter, patience,
            )

        if patience_counter >= patience:
            stopped_early = True
            _log.info("early stopping at epoch %d (no improvement for %d epochs)", epoch + 1, patience)
            break

    if best_state is None:  # pragma: no cover - only if max_epochs == 0
        raise RuntimeError("training ran zero epochs")
    model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        curve=curve,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stopped_early=stopped_early,
        n_epochs_run=epoch + 1,
    )
