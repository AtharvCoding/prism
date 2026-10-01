"""The three mandatory encoder baselines. Spec §9.4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.models.baselines.pca_encoder import PCAEncoder, flatten_windows
from prism.models.baselines.random_encoder import build_random_encoder, random_encoder_latents
from prism.models.baselines.rolling_stats import rolling_stats_encode


@pytest.fixture
def synth_frame():
    idx = pd.bdate_range("2015-01-01", periods=200)
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(200, 5)), index=idx, columns=[f"f{i}" for i in range(5)])


# --------------------------------------------------------------------------- #
# pca_encoder.py
# --------------------------------------------------------------------------- #
def test_flatten_windows_alignment_matches_the_lstm_convention(synth_frame):
    """Row dated D's flattened window ends with frame.loc[D]'s own values
    (time-major, feature-minor column order, as documented)."""
    flat = flatten_windows(synth_frame, 10)
    probe = synth_frame.index[50]
    last_block = flat.loc[probe].to_numpy()[-5:]  # last 5 cols = last timestep's features
    np.testing.assert_allclose(last_block, synth_frame.loc[probe].to_numpy())


def test_pca_is_fit_on_train_only_and_does_not_refit_on_transform(synth_frame):
    flat = flatten_windows(synth_frame, 10)
    train, test = flat.iloc[:100], flat.iloc[100:]
    encoder = PCAEncoder(n_components=3).fit(train, scope="train")
    before = encoder.params_hash()
    out = encoder.transform(test)
    assert encoder.params_hash() == before
    assert out.shape == (len(test), 3)


def test_pca_explained_variance_ratio_is_decreasing(synth_frame):
    flat = flatten_windows(synth_frame, 10)
    encoder = PCAEncoder(n_components=4).fit(flat, scope="all")
    ratios = encoder.explained_variance_ratio
    assert list(ratios) == sorted(ratios, reverse=True)


def test_pca_rejects_too_many_components(synth_frame):
    flat = flatten_windows(synth_frame, 10)
    with pytest.raises(ValueError, match="exceeds"):
        PCAEncoder(n_components=10_000).fit(flat, scope="train")


# --------------------------------------------------------------------------- #
# random_encoder.py
# --------------------------------------------------------------------------- #
def test_random_encoder_is_deterministic_given_a_seed(synth_frame):
    a = random_encoder_latents(synth_frame, 10, hidden_dim=8, latent_dim=4, seed=7)
    b = random_encoder_latents(synth_frame, 10, hidden_dim=8, latent_dim=4, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_random_encoder_differs_across_seeds(synth_frame):
    a = random_encoder_latents(synth_frame, 10, hidden_dim=8, latent_dim=4, seed=1)
    b = random_encoder_latents(synth_frame, 10, hidden_dim=8, latent_dim=4, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_random_encoder_weights_never_require_grad():
    encoder = build_random_encoder(input_dim=4, hidden_dim=6, latent_dim=3, seed=0)
    assert not any(p.requires_grad for p in encoder.parameters())


def test_random_encoder_alignment_matches_the_lstm_convention(synth_frame):
    """Same alignment contract as the real encoder: latent dated D comes
    from the window ending at D."""
    latents = random_encoder_latents(synth_frame, 10, hidden_dim=8, latent_dim=4, seed=0)
    expected_index = synth_frame.index[9:]
    assert latents.index.equals(expected_index)


# --------------------------------------------------------------------------- #
# rolling_stats.py
# --------------------------------------------------------------------------- #
def test_rolling_stats_mean_matches_a_hand_computed_window():
    idx = pd.bdate_range("2020-01-01", periods=10)
    frame = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=idx)
    out = rolling_stats_encode(frame, 5)
    # Window ending at row 4 (values 0..4): mean=2.0, std(ddof=1)=sqrt(2.5).
    np.testing.assert_allclose(out["x_mean"].iloc[0], 2.0)
    np.testing.assert_allclose(out["x_std"].iloc[0], np.std(np.arange(5), ddof=1))


def test_rolling_stats_constant_window_has_zero_moments_not_nan():
    idx = pd.bdate_range("2020-01-01", periods=10)
    frame = pd.DataFrame({"x": np.full(10, 3.0)}, index=idx)
    out = rolling_stats_encode(frame, 5)
    assert np.isfinite(out.to_numpy()).all()
    np.testing.assert_allclose(out["x_std"].to_numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(out["x_skew"].to_numpy(), 0.0, atol=1e-10)


def test_rolling_stats_alignment_and_shape(synth_frame):
    out = rolling_stats_encode(synth_frame, 10)
    assert out.shape == (len(synth_frame) - 10 + 1, 4 * synth_frame.shape[1])
    assert out.index.equals(synth_frame.index[9:])
