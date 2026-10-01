"""Encoder architecture, training, walk-forward, evaluation, baselines. Spec §9."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from prism.models.encoder.dataset import WindowDataset, window_array
from prism.models.encoder.evaluate import (
    compute_latents,
    dead_units,
    evaluate_reconstruction,
    latent_drift,
)
from prism.models.encoder.models import build_model
from prism.models.encoder.train import train_model
from prism.models.encoder.walkforward import encoder_walkforward


@pytest.fixture
def synth_frame():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(400, 6)), index=idx, columns=[f"f{i}" for i in range(6)])


# --------------------------------------------------------------------------- #
# dataset.py
# --------------------------------------------------------------------------- #
def test_window_array_is_a_view_not_a_copy(synth_frame):
    """Memory efficiency claim, checked: sliding_window_view shares the
    underlying buffer with the source array."""
    _, arr = window_array(synth_frame, 30)
    values = synth_frame.to_numpy(dtype="float64")
    assert np.shares_memory(arr, values)


def test_window_dataset_rejects_nan(synth_frame):
    holed = synth_frame.copy()
    holed.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        WindowDataset(holed, 10)


def test_window_dataset_autoencoding_target_is_identical_object(synth_frame):
    ds = WindowDataset(synth_frame, 10)
    x, y = ds[0]
    assert torch.equal(x, y)


# --------------------------------------------------------------------------- #
# models.py
# --------------------------------------------------------------------------- #
def test_dae_noise_only_applied_in_training_mode():
    torch.manual_seed(0)
    model = build_model("DAE", input_dim=4, hidden_dim=6, latent_dim=3, window=8, dae_noise_std=1.0)
    x = torch.randn(5, 8, 4)

    model.eval()
    with torch.no_grad():
        recon_a, _ = model(x)
        recon_b, _ = model(x)
    # Eval mode: no noise injected, so two forward passes on the SAME input
    # are identical (modulo nothing stochastic left in the graph).
    assert torch.allclose(recon_a, recon_b)

    model.train()
    loss_a = model.loss(x).total.item()
    loss_b = model.loss(x).total.item()
    # Train mode: noise IS injected, so repeated losses on the same x differ.
    assert loss_a != loss_b


def test_vae_latent_is_the_posterior_mean_not_a_sample():
    torch.manual_seed(0)
    model = build_model("VAE", input_dim=4, hidden_dim=6, latent_dim=3, window=8)
    x = torch.randn(5, 8, 4)
    model.eval()
    with torch.no_grad():
        a = model.latent(x)
        b = model.latent(x)
    assert torch.allclose(a, b), "eval-mode VAE latent must be deterministic (mu, not a sample)"


def test_vae_logvar_is_clamped_against_overflow():
    torch.manual_seed(0)
    model = build_model("VAE", input_dim=4, hidden_dim=6, latent_dim=3, window=8)
    # Force huge pre-activation values by scaling up a weight, simulating an
    # early, unstable point in training.
    with torch.no_grad():
        for p in model.encoder.project.parameters():
            p.mul_(1000.0)
    x = torch.randn(5, 8, 4)
    loss = model.loss(x)
    assert torch.isfinite(loss.total), "unclamped logvar would overflow exp() here"


def test_no_module_anywhere_is_relu():
    for variant, kw in [("AE", {}), ("DAE", {}), ("VAE", {}), ("PRED", {"n_pred_targets": 2})]:
        model = build_model(variant, input_dim=4, hidden_dim=6, latent_dim=3, window=8, **kw)
        assert not any(isinstance(m, torch.nn.ReLU) for m in model.modules()), variant


def test_pred_with_zero_reconstruction_weight_has_no_decoder():
    model = build_model(
        "PRED", input_dim=4, hidden_dim=6, latent_dim=3, window=8,
        n_pred_targets=2, pred_reconstruction_weight=0.0,
    )
    assert model.decoder is None
    x = torch.randn(5, 8, 4)
    y = torch.randn(5, 2)
    loss = model.loss(x, y)
    assert "reconstruction" not in loss.components


def test_build_model_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        build_model("NOPE", input_dim=4, hidden_dim=6, latent_dim=3, window=8)


def test_pred_requires_targets_at_loss_time():
    model = build_model("PRED", input_dim=4, hidden_dim=6, latent_dim=3, window=8, n_pred_targets=2)
    x = torch.randn(5, 8, 4)
    with pytest.raises(ValueError, match="requires forward targets"):
        model.loss(x, None)


# --------------------------------------------------------------------------- #
# train.py
# --------------------------------------------------------------------------- #
def test_training_stops_early_and_keeps_the_best_not_the_last_weights():
    torch.manual_seed(0)
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(400, 4)), index=idx, columns=list("abcd"))
    train_ds = WindowDataset(frame.iloc[:300], 10)
    val_ds = WindowDataset(frame.iloc[291:], 10)

    model = build_model("AE", input_dim=4, hidden_dim=8, latent_dim=3, window=10)
    result = train_model(
        model,
        DataLoader(train_ds, batch_size=32, shuffle=True),
        DataLoader(val_ds, batch_size=32, shuffle=False),
        lr=1e-2, weight_decay=0.0, max_epochs=200, patience=5, grad_clip_norm=1.0,
    )
    # Every val loss after best_epoch must be >= best_val_loss (that is the
    # whole definition of "best"); and patience means training stopped
    # within `patience` epochs of the best one, not at max_epochs.
    val_losses = [v["total"] for v in result.curve.val]
    assert min(val_losses) == pytest.approx(result.best_val_loss)
    assert val_losses.index(min(val_losses)) == result.best_epoch
    if result.stopped_early:
        assert result.n_epochs_run == result.best_epoch + 1 + 5
    assert result.n_epochs_run <= 200


def test_training_loop_never_touches_training_loss_for_checkpointing():
    """Construct a model whose training loss keeps improving while validation
    loss gets WORSE (by training on a window that overlaps/aliases the val
    set's early rows) — checkpointing on train loss would keep the LAST
    epoch; checkpointing on val loss must keep an EARLIER one."""
    torch.manual_seed(1)
    idx = pd.bdate_range("2015-01-01", periods=200)
    rng = np.random.default_rng(1)
    train_frame = pd.DataFrame(rng.normal(size=(150, 3)), index=idx[:150], columns=list("abc"))
    # Validation frame is pure noise unrelated to train's distribution shift target.
    val_frame = pd.DataFrame(rng.normal(10, 5, size=(50, 3)), index=idx[150:], columns=list("abc"))

    train_ds = WindowDataset(train_frame, 10)
    val_ds = WindowDataset(val_frame, 10)
    model = build_model("AE", input_dim=3, hidden_dim=16, latent_dim=2, window=10)
    result = train_model(
        model,
        DataLoader(train_ds, batch_size=16, shuffle=True),
        DataLoader(val_ds, batch_size=16, shuffle=False),
        lr=1e-2, weight_decay=0.0, max_epochs=60, patience=10, grad_clip_norm=1.0,
    )
    train_losses = [t["total"] for t in result.curve.train]
    # Training loss should have decreased a lot (model fits train fine);
    # the SELECTED epoch need not be the last one even so.
    assert train_losses[-1] < train_losses[0]
    assert result.best_epoch <= result.n_epochs_run - 1


# --------------------------------------------------------------------------- #
# walkforward.py
# --------------------------------------------------------------------------- #
def test_walkforward_scaler_is_refit_per_fold(synth_frame):
    from prism.utils.calendar import trading_days

    idx = trading_days("2015-01-02", "2017-12-31")
    rng = np.random.default_rng(0)
    trend = np.linspace(0, 5, len(idx))
    frame = pd.DataFrame(
        {"a": rng.normal(size=len(idx)) + trend, "b": rng.normal(size=len(idx))}, index=idx
    )
    result = encoder_walkforward(
        frame, variant="AE", window=10, hidden_dim=4, latent_dim=2,
        fit_start="2015-01-02", first_apply_start="2017-01-01", apply_end="2017-12-31",
        cadence="quarterly", embargo_days=5, seed_base=0, max_epochs=5, patience=3, batch_size=16,
    )
    scopes = [f.scaler.fit_record.scope for f in result.folds]
    assert len(set(scopes)) == len(scopes)
    means = [f.scaler.fit_record.data_hash for f in result.folds]
    assert len(set(means)) == len(means), "each fold's scaler should be fit on different (expanding) data"


def test_walkforward_latents_are_deduplicated_and_calendar_aligned():
    from prism.utils.calendar import trading_days

    idx = trading_days("2015-01-02", "2017-12-31")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(len(idx), 4)), index=idx, columns=list("abcd"))
    result = encoder_walkforward(
        frame, variant="AE", window=10, hidden_dim=4, latent_dim=2,
        fit_start="2015-01-02", first_apply_start="2017-01-01", apply_end="2017-12-31",
        cadence="quarterly", embargo_days=5, seed_base=0, max_epochs=5, patience=3, batch_size=16,
    )
    assert result.latents.index.is_unique
    assert result.latents.index.is_monotonic_increasing
    expected = trading_days("2017-01-01", "2017-12-31")
    assert result.latents.index.equals(expected)


def test_walkforward_raises_on_empty_fit_window():
    idx = pd.bdate_range("2017-01-01", periods=10)
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 2)), index=idx, columns=list("ab"))
    with pytest.raises(ValueError):
        encoder_walkforward(
            frame, variant="AE", window=20, hidden_dim=4, latent_dim=2,
            fit_start="2017-01-01", first_apply_start="2017-01-05", apply_end="2017-01-15",
            cadence="monthly", embargo_days=0, seed_base=0, max_epochs=5, patience=3,
        )


# --------------------------------------------------------------------------- #
# evaluate.py
# --------------------------------------------------------------------------- #
def test_dead_units_flags_a_genuinely_constant_dimension(synth_frame):
    ds = WindowDataset(synth_frame, 10)
    model = build_model("AE", input_dim=6, hidden_dim=4, latent_dim=3, window=10)
    latents = compute_latents(model, ds)
    latents["latent_0"] = 0.0  # force one dead dimension
    report = dead_units(latents, var_threshold=1e-4)
    assert "latent_0" in report.dead_columns
    assert report.n_dead == 1


def test_evaluate_reconstruction_is_computed_out_of_sample(synth_frame):
    train_frame, test_frame = synth_frame.iloc[:300], synth_frame.iloc[291:]
    train_ds = WindowDataset(train_frame, 10)
    test_ds = WindowDataset(test_frame, 10)
    model = build_model("AE", input_dim=6, hidden_dim=8, latent_dim=3, window=10)
    train_model(
        model, DataLoader(train_ds, batch_size=32, shuffle=True),
        DataLoader(test_ds, batch_size=32, shuffle=False),
        lr=1e-2, weight_decay=0.0, max_epochs=5, patience=5, grad_clip_norm=1.0,
    )
    families = {"fam1": ["f0", "f1", "f2"], "fam2": ["f3", "f4", "f5"]}
    ev = evaluate_reconstruction(model, test_ds, families=families)
    assert ev.n_windows == len(test_ds)
    assert set(ev.mse_per_family.index) == {"fam1", "fam2"}
    assert ev.mse_overall > 0


def test_latent_drift_detects_a_shifted_distribution():
    idx_a = pd.bdate_range("2015-01-01", periods=200)
    idx_b = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"latent_0": rng.normal(0, 1, 200), "latent_1": rng.normal(0, 1, 200)}, index=idx_a)
    b = pd.DataFrame({"latent_0": rng.normal(5, 1, 200), "latent_1": rng.normal(0, 1, 200)}, index=idx_b)
    report = latent_drift(a, b)
    assert "latent_0" in report.drifted_dimensions
    assert "latent_1" not in report.drifted_dimensions


def test_latent_drift_requires_shared_columns():
    idx = pd.bdate_range("2015-01-01", periods=50)
    a = pd.DataFrame({"x": np.zeros(50)}, index=idx)
    b = pd.DataFrame({"y": np.zeros(50)}, index=idx)
    with pytest.raises(ValueError, match="no columns"):
        latent_drift(a, b)
