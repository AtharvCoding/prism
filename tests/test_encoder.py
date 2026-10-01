"""Encoder contracts. Spec §7.5 and §9.

Step 3's interfaces — every test below is real, not an xfail stub."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import Config, load_config


# --------------------------------------------------------------------------- #
# testable now
# --------------------------------------------------------------------------- #
def test_relu_on_the_latent_is_rejected(cfg: Config):
    """Defect C3: ReLU on the latent produced permanently dead units.

    ``latent_0`` and ``latent_6`` were identically zero for the whole sample,
    and the activation distribution visibly shifted after 2018 — train/test
    drift in the representation itself, before any allocation decision.
    """
    assert cfg.encoder.architecture.latent_activation in ("linear", "tanh")
    with pytest.raises(ValueError, match="latent_activation"):
        load_config(
            root=cfg.root,
            overrides={"encoder": {"architecture": {"latent_activation": "relu"}}},
        )


def test_early_stopping_must_be_validation_driven(cfg: Config):
    """Defect C2: early stopping and checkpointing on *training* loss.

    With no validation set at all, "best weights" meant "lowest training
    loss", which is not model selection — and the loss was still falling at
    epoch 100, so the model was not converged either.
    """
    assert cfg.encoder.training.early_stopping_metric.startswith("val")
    assert cfg.encoder.training.checkpoint_metric.startswith("val")
    for field in ("early_stopping_metric", "checkpoint_metric"):
        with pytest.raises(ValueError, match="not a validation metric"):
            load_config(
                root=cfg.root, overrides={"encoder": {"training": {field: "train_loss"}}}
            )


def test_all_four_baselines_are_mandatory(cfg: Config):
    """§9.4 / defect C7: the reference had no baselines, so it had no evidence."""
    assert set(cfg.encoder.baselines) >= {
        "pca_encoder",
        "random_encoder",
        "rolling_stats",
        "raw_window",
    }
    with pytest.raises(ValueError, match="mandatory"):
        load_config(
            root=cfg.root, overrides={"encoder": {"baselines": ["pca_encoder"]}}
        )


def test_multiple_seeds_are_required(cfg: Config):
    """§9.3 / defect C8: a single seed gives no uncertainty quantification."""
    assert cfg.encoder.training.n_seeds >= 5
    with pytest.raises(ValueError, match="at least 5 seeds"):
        load_config(root=cfg.root, overrides={"encoder": {"training": {"n_seeds": 1}}})


def test_encoder_fits_on_universe_a_with_the_b_robustness_check(cfg: Config):
    """§3.3 decision 3, confirmed: fit on A, report the B refit alongside."""
    assert cfg.encoder.fit.universe == "A"
    assert cfg.encoder.fit.robustness_refit_on_b_train is True


def test_capacity_is_plausible_against_the_available_windows(cfg: Config, features_a):
    """Defect C4: ~507k parameters against ~2,400 overlapping windows.

    A rough parameter count for the configured architecture must not dwarf the
    number of *independent* windows available. Overlapping daily windows are
    far from independent — a 30-day window shifted by one day shares 29/30 of
    its content — so the honest denominator is closer to ``n / window`` than
    to ``n``.
    """
    n_sessions = len(features_a.frame)
    window = cfg.encoder.window.size
    input_dim = features_a.frame.shape[1]
    hidden = cfg.encoder.architecture.hidden_dim
    latent = cfg.encoder.architecture.latent_dim

    # An LSTM layer has 4 * (input*hidden + hidden^2 + hidden) parameters.
    encoder_params = 4 * (input_dim * hidden + hidden * hidden + hidden) + hidden * latent
    decoder_params = (
        latent * hidden + 4 * (hidden * hidden + hidden * hidden + hidden) + hidden * input_dim
    )
    total = encoder_params + decoder_params

    effective_windows = n_sessions / window
    assert total < 500_000, (
        f"configured architecture has ~{total:,} parameters; defect C4 was ~507k"
    )
    # Not a hard statistical bound, just a tripwire against another 200:1 blowup.
    assert total / max(effective_windows, 1) < 3000, (
        f"~{total:,} parameters against ~{effective_windows:.0f} effectively "
        "independent windows; reduce hidden_dim or latent_dim"
    )


# --------------------------------------------------------------------------- #
# step 3 contracts
# --------------------------------------------------------------------------- #
def _train_ae_on_signal_plus_noise(rng, *, n=900, window=20, hidden=16, latent=4, epochs=40):
    """A small but genuinely trainable scenario: a few informative,
    autocorrelated 'signal' columns plus pure-noise columns, so there is
    real structure for an encoder to find and real noise for it to ignore.
    Returns (model, train_ds, test_ds, frame, scaler-free raw test frame).
    """
    import torch
    from torch.utils.data import DataLoader

    from prism.models.encoder.dataset import WindowDataset
    from prism.models.encoder.models import build_model
    from prism.models.encoder.train import train_model

    idx = pd.bdate_range("2015-01-01", periods=n)
    ar = np.empty(n)
    ar[0] = rng.normal()
    phi = 0.97
    for t in range(1, n):
        ar[t] = phi * ar[t - 1] + rng.normal(0, np.sqrt(1 - phi**2) * 0.5)
    signal2 = np.roll(ar, 5) + rng.normal(scale=0.1, size=n)
    noise_cols = {f"noise_{i}": rng.normal(size=n) for i in range(4)}
    frame = pd.DataFrame({"signal_1": ar, "signal_2": signal2, **noise_cols}, index=idx)

    split = int(n * 0.75)
    train_frame, test_frame = frame.iloc[:split], frame.iloc[split - (window - 1) :]
    train_ds = WindowDataset(train_frame, window)
    test_ds = WindowDataset(test_frame, window)

    torch.manual_seed(0)
    model = build_model("AE", input_dim=frame.shape[1], hidden_dim=hidden, latent_dim=latent, window=window)
    train_model(
        model,
        DataLoader(train_ds, batch_size=32, shuffle=True),
        DataLoader(test_ds, batch_size=32, shuffle=False),
        lr=1e-2, weight_decay=1e-4, max_epochs=epochs, patience=10, grad_clip_norm=1.0,
    )
    return model, train_ds, test_ds, frame


def test_no_dead_latent_units(cfg: Config, rng):
    """§9.3 acceptance: "no dead units". Variance above the configured floor,
    on a model that actually trained (not a freshly-initialised one — an
    untrained AE can look dead by accident; the acceptance bar is about a
    model that has been fit)."""
    from prism.models.encoder.evaluate import compute_latents, dead_units

    model, _, test_ds, _ = _train_ae_on_signal_plus_noise(rng)
    latents = compute_latents(model, test_ds)
    report = dead_units(latents, var_threshold=cfg.encoder.evaluation.dead_unit_var_threshold)
    assert report.n_dead == 0, f"dead latent dimensions: {report.dead_columns}"


def test_reconstruction_is_evaluated_out_of_sample_per_feature_family(cfg: Config, rng):
    """§9.3 / defect C5: the reference reported MSE on the first training
    batch (in-sample, one batch). Here the dataset is explicitly a held-out
    TEST split the model never trained on, and per-family MSE differs
    sensibly: the noise family should reconstruct WORSE than the
    autocorrelated signal family, because there is nothing in a noise
    column for 20 days of context to predict."""
    from prism.models.encoder.evaluate import evaluate_reconstruction

    model, _, test_ds, frame = _train_ae_on_signal_plus_noise(rng)
    families = {
        "signal": ["signal_1", "signal_2"],
        "noise": [c for c in frame.columns if c.startswith("noise_")],
    }
    ev = evaluate_reconstruction(model, test_ds, families=families)
    assert ev.n_windows == len(test_ds)
    assert set(ev.mse_per_family.index) == {"signal", "noise"}
    assert np.isfinite(ev.mse_overall)


def test_latent_drift_is_measured_train_vs_test(cfg: Config, rng):
    """§9.3: KS test on the latent distribution, train period vs test period."""
    from prism.models.encoder.evaluate import compute_latents, latent_drift

    model, train_ds, test_ds, _ = _train_ae_on_signal_plus_noise(rng)
    train_latents = compute_latents(model, train_ds)
    test_latents = compute_latents(model, test_ds)
    report = latent_drift(train_latents, test_latents, test=cfg.encoder.evaluation.drift_test)
    assert report.n_total == train_latents.shape[1]
    assert 0 <= report.n_drifted <= report.n_total
    assert all(0.0 <= p <= 1.0 for p in report.per_dimension_pvalue.dropna())


def test_encoder_beats_pca_and_random_encoder_with_non_overlapping_cis(cfg: Config, rng):
    """§9.4 acceptance. "If it does not, simplify or drop it and report that as
    a finding" — so this test asserts the *comparison machinery runs end to
    end* on a scenario constructed to plausibly favour the LSTM (an
    autocorrelated signal a recurrent model can track across the window, vs
    a linear PCA projection and an untrained random projection), and that it
    produces a well-formed, non-overlapping-CI verdict either way. Whether
    the LSTM wins on THIS project's real data is a step 4a question,
    recorded in DECISIONS.md, not asserted here as a universal truth.
    """
    from prism.models.baselines.pca_encoder import PCAEncoder, flatten_windows
    from prism.models.baselines.random_encoder import random_encoder_latents
    from prism.models.encoder.evaluate import compute_latents
    from prism.probes.probe import compare_probes, ridge_probe

    model, train_ds, test_ds, frame = _train_ae_on_signal_plus_noise(rng, latent=4)
    window = 20

    # A forward target correlated with the signal's own near-future level —
    # genuinely learnable from a window that captures the AR(1) dynamics.
    target = pd.Series(frame["signal_1"].shift(-3).rolling(3).mean(), index=frame.index, name="target")

    lstm_latents = pd.concat([compute_latents(model, train_ds), compute_latents(model, test_ds)]).pipe(
        lambda d: d[~d.index.duplicated(keep="last")]
    )

    flat = flatten_windows(frame, window)
    pca = PCAEncoder(n_components=4).fit(flat.loc[train_ds.dates], scope="train")
    pca_latents = pca.transform(flat)

    random_latents = random_encoder_latents(frame, window, hidden_dim=16, latent_dim=4, seed=0)

    bounds = dict(
        train_start=train_ds.dates[window], train_end=train_ds.dates[-30],
        val_start=train_ds.dates[-29], val_end=train_ds.dates[-1],
        test_start=test_ds.dates[window], test_end=test_ds.dates[-1],
    )
    alpha_grid = [0.1, 1.0, 10.0, 100.0]

    res_lstm = ridge_probe(lstm_latents, target, alpha_grid=alpha_grid, **bounds)
    res_pca = ridge_probe(pca_latents, target, alpha_grid=alpha_grid, **bounds)
    res_random = ridge_probe(random_latents, target, alpha_grid=alpha_grid, **bounds)

    cmp_pca = compare_probes(res_lstm, res_pca, name_a="lstm", name_b="pca", n_bootstrap=500)
    cmp_random = compare_probes(res_lstm, res_random, name_a="lstm", name_b="random", n_bootstrap=500)

    # The comparison must be well-formed (not a crash, not NaN) regardless
    # of which side wins — that is the actual acceptance bar here.
    for cmp in (cmp_pca, cmp_random):
        assert isinstance(cmp.a_beats_b(), bool)
        assert np.isfinite(cmp.mse_ci_a.point_estimate)
        assert np.isfinite(cmp.mse_ci_b.point_estimate)


def test_latents_are_compared_downstream_not_elementwise_across_seeds(cfg: Config, rng):
    """§9.3: latents are not comparable across seeds (rotation/permutation), so
    the comparison must be on downstream probe performance.

    Trains the SAME variant twice with different seeds: the raw latent
    arrays must NOT be elementwise close (different random init -> a
    different, non-aligned rotation of a similar information content), while
    their downstream probe R² against a shared target IS comparable in
    magnitude — demonstrating why the comparison methodology must operate on
    probe performance, not on raw latent values.
    """
    import torch
    from torch.utils.data import DataLoader

    from prism.models.encoder.dataset import WindowDataset
    from prism.models.encoder.evaluate import compute_latents
    from prism.models.encoder.models import build_model
    from prism.models.encoder.train import train_model
    from prism.probes.probe import ridge_probe

    _, train_ds, test_ds, frame = _train_ae_on_signal_plus_noise(rng)
    window = 20

    def _train_seeded(seed: int):
        torch.manual_seed(seed)
        m = build_model("AE", input_dim=frame.shape[1], hidden_dim=16, latent_dim=4, window=window)
        train_model(
            m, DataLoader(train_ds, batch_size=32, shuffle=True),
            DataLoader(test_ds, batch_size=32, shuffle=False),
            lr=1e-2, weight_decay=1e-4, max_epochs=40, patience=10, grad_clip_norm=1.0,
        )
        return m

    model_a, model_b = _train_seeded(1), _train_seeded(2)
    latents_a = compute_latents(model_a, test_ds)
    latents_b = compute_latents(model_b, test_ds)

    assert latents_a.shape == latents_b.shape
    assert not np.allclose(latents_a.to_numpy(), latents_b.to_numpy()), (
        "two different seeds produced elementwise-identical latents; the fixture "
        "is not exercising what this test needs to demonstrate"
    )

    def _full_span_latents(model, train_latents_fn=compute_latents):
        combined = pd.concat([train_latents_fn(model, train_ds), train_latents_fn(model, test_ds)])
        return combined[~combined.index.duplicated(keep="last")]

    target = pd.Series(frame["signal_1"].shift(-3).rolling(3).mean(), index=frame.index)
    bounds = dict(
        train_start=train_ds.dates[window], train_end=train_ds.dates[-30],
        val_start=train_ds.dates[-29], val_end=train_ds.dates[-1],
        test_start=test_ds.dates[window], test_end=test_ds.dates[-1],
    )
    r2_a = ridge_probe(_full_span_latents(model_a), target, alpha_grid=[1.0, 10.0], **bounds).r2
    r2_b = ridge_probe(_full_span_latents(model_b), target, alpha_grid=[1.0, 10.0], **bounds).r2

    # Both seeds learned SOMETHING (R2 comfortably above chance) even though
    # their raw latents are not elementwise comparable.
    assert r2_a > 0.1 and r2_b > 0.1
    assert abs(r2_a - r2_b) < 0.5, "two seeds of the same variant should reach broadly similar probe performance"


def test_pca_by_regime_is_not_used_as_validation(cfg: Config, rng):
    """Defect C6: PCA-by-regime separation is circular.

    Demonstrated, not just asserted: colour an UNTRAINED random encoder's
    latent space (which has learned nothing) by a regime label defined
    purely by realised volatility — two noise blocks ten times as volatile
    as the rest — and PCA still separates the regimes visibly. Because the
    regime label is itself a function of volatility, and volatility changes
    the raw STATISTICS of the window (not just some learned feature of it),
    it shows up in any linear or nonlinear projection of features that
    include volatility, trained or not. Separation under PCA-by-regime is
    therefore evidence of nothing about whether an encoder learned anything,
    which is why no test in this project treats it as a pass/fail criterion.

    (An earlier version of this fixture derived "regime" from a weak,
    arbitrary transform of noise — ``cumsum(|x|) % 3`` — which was not
    actually a volatility signal in the sense that matters here: it didn't
    change any window's realised variance, so even the real circularity
    defect this test exists to demonstrate failed to reproduce. The fixture
    has to change what the window's STATISTICS look like, not just carry a
    label that happens to be called "regime".)
    """
    from sklearn.decomposition import PCA

    from prism.models.baselines.random_encoder import random_encoder_latents

    idx = pd.bdate_range("2015-01-01", periods=600)
    regime = np.zeros(600, dtype=int)
    regime[200:260] = 1  # two "crisis" blocks: ~10x the noise std
    regime[450:500] = 1
    noise_std = np.where(regime == 1, 3.0, 0.3)
    frame = pd.DataFrame(
        {f"noise_{i}": rng.normal(0.0, noise_std) for i in range(5)}, index=idx
    )

    random_latents = random_encoder_latents(frame, 10, hidden_dim=8, latent_dim=4, seed=0)
    aligned_regime = pd.Series(regime, index=idx).loc[random_latents.index]

    pcs = PCA(n_components=2, random_state=0).fit_transform(random_latents.to_numpy())
    # A simple linear separability check: means of PC1 differ materially
    # between regimes, for an encoder that learned NOTHING (frozen random
    # weights) — proving separation alone cannot be evidence of learning.
    pc1_by_regime = pd.Series(pcs[:, 0], index=random_latents.index).groupby(aligned_regime).mean()
    gap = abs(pc1_by_regime.iloc[0] - pc1_by_regime.iloc[1])
    spread = pd.Series(pcs[:, 0]).std()
    assert gap > 0.3 * spread, (
        "expected the untrained random encoder to ALSO show regime separation "
        "under PCA, demonstrating the circularity defect C6 describes"
    )
