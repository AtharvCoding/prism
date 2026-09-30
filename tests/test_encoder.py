"""Encoder contracts. Spec §7.5 and §9.

Written against the interfaces build step 3 must provide, marked
``xfail(strict=True)`` so they convert into real guards the moment step 3
lands. The config-level guards against defects C2 and C3 are enforceable
today and are not marked.
"""

from __future__ import annotations

import pytest

from prism.config import Config, load_config

STEP_3 = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="spec §9 — encoder rebuild is build step 3. When it lands this turns "
    "XPASS (a failure) and the marker must be removed.",
)


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
@STEP_3
def test_no_dead_latent_units(cfg: Config):
    """§9.3 acceptance: "no dead units". Variance above the configured floor."""
    from prism.models.encoder.evaluate import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §9.3 — build step 3")


@STEP_3
def test_reconstruction_is_evaluated_out_of_sample_per_feature_family(cfg: Config):
    """§9.3 / defect C5: the reference reported MSE on the first training batch."""
    from prism.models.encoder.evaluate import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §9.3 — build step 3")


@STEP_3
def test_latent_drift_is_measured_train_vs_test(cfg: Config):
    """§9.3: KS or MMD on the latent distribution across periods."""
    from prism.models.encoder.evaluate import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §9.3 — build step 3")


@STEP_3
def test_encoder_beats_pca_and_random_encoder_with_non_overlapping_cis(cfg: Config):
    """§9.4 acceptance. "If it does not, simplify or drop it and report that as
    a finding" — so this test asserts the *comparison was run*, and the gate
    decision is recorded in DECISIONS.md either way."""
    from prism.models.baselines.pca_encoder import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §9.4 — build step 3")


@STEP_3
def test_latents_are_compared_downstream_not_elementwise_across_seeds(cfg: Config):
    """§9.3: latents are not comparable across seeds (rotation/permutation), so
    the comparison must be on downstream probe performance."""
    from prism.models.encoder.evaluate import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §9.3 — build step 3")


@STEP_3
def test_pca_by_regime_is_not_used_as_validation(cfg: Config):
    """Defect C6: PCA-by-regime separation is circular.

    The features contain volatility and the regimes are a function of
    volatility, so separation is guaranteed regardless of whether the encoder
    learned anything. Step 3 must not report it as evidence.
    """
    from prism.models.encoder.evaluate import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §9.2 — build step 3")
