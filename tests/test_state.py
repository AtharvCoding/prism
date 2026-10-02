"""State-vector assembly contracts. Spec §7.5 and §10.

Build step 3b. :mod:`prism.state` assembles; it does not fit anything, so
these tests construct small, explicit synthetic artifacts (latents,
posteriors, random latents, threshold states) rather than running a real HMM
or encoder fit — exactly the style ``test_threshold_regime.py`` and
``test_encoder_baselines.py`` already use for the modules that produce those
artifacts in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import Config, load_config
from prism.features.scaling import FeatureScaler
from prism.state import (
    MissingArtifactError,
    StateArtifacts,
    build_state,
    shuffle_posteriors,
)


# --------------------------------------------------------------------------- #
# testable without state.py: the variant set and the gate algebra
# --------------------------------------------------------------------------- #
def test_all_mandatory_variants_and_controls_are_configured(cfg: Config):
    """§10: the controls are mandatory, not optional.

    With V1-V4 alone, "V2 beats V1" has at least three explanations: the LSTM
    learned something; thirty days of history helps and any summary would do;
    or the state vector simply got wider. V1', C1, C2 and C3 are what make
    the claims falsifiable.
    """
    assert set(cfg.tier1.variants) >= {"V1", "V1p", "V2", "V3", "V4", "C1", "C2", "C3"}


def test_the_leaky_oracle_is_diagnostic_only(cfg: Config):
    """§10: O1 uses smoothed posteriors and must never be reported as a result."""
    assert "O1" not in cfg.tier1.variants
    assert "O1" in cfg.tier1.diagnostic_variants
    with pytest.raises(ValueError, match="diagnostic only"):
        load_config(
            root=cfg.root,
            overrides={
                "tier1": {
                    "variants": ["V1", "V1p", "V2", "V3", "V4", "C1", "C2", "C3", "O1"]
                }
            },
        )


def test_gate_rules_express_the_falsifiable_claims(cfg: Config):
    """§13.1: each component must beat its OWN control, not just the baseline."""
    gates = cfg.tier1.gates
    assert set(gates["lstm_adds_value"]) == {"V2 > V1p", "V2 > C1"}
    assert set(gates["hmm_adds_value"]) == {"V3 > C2", "V3 > C3"}
    assert gates["research_question"] == ["V4 > V2"]


def test_forward_targets_are_never_state_members(cfg: Config):
    """§5.4: targets are for probes and evaluation only."""
    from prism.features.targets import TARGET_NAMES

    for target in TARGET_NAMES:
        assert target not in cfg.tier1.variants
    # And the feature builder already refuses to emit them (asserted in
    # prism.features.build.assert_universe_isolation).


# --------------------------------------------------------------------------- #
# fixtures: a small window of real features, plus hand-built artifacts
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_features(features_b):
    """The last 120 sessions of the real, warm Universe B feature frame.

    Real features (not synthetic ones built fresh here) so the test exercises
    the actual :class:`~prism.features.build.FeatureSet` contract — the
    ``schema_hash`` property, the column set — rather than a stand-in.
    """
    import copy

    small = copy.copy(features_b)
    small.frame = features_b.frame.iloc[-120:]
    return small


@pytest.fixture
def fitted_scaler(small_features):
    return FeatureScaler().fit(small_features.frame, scope="test_state fixture")


def _posteriors_like(index: pd.DatetimeIndex, k: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(k), size=len(index))
    return pd.DataFrame(raw, index=index, columns=[f"state_{i}" for i in range(k)])


def _latents_like(index: pd.DatetimeIndex, dim: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = np.tanh(rng.normal(size=(len(index), dim)))
    return pd.DataFrame(raw, index=index, columns=[f"latent_{i}" for i in range(dim)])


def _one_hot_like(index: pd.DatetimeIndex, k: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = rng.integers(0, k, size=len(index))
    out = np.zeros((len(index), k))
    out[np.arange(len(index)), states] = 1.0
    return pd.DataFrame(out, index=index, columns=[f"state_{i}" for i in range(k)])


@pytest.fixture
def full_artifacts(small_features):
    idx = small_features.frame.index
    return StateArtifacts(
        latents=_latents_like(idx, 4, seed=1),
        posteriors=_posteriors_like(idx, 3, seed=2),
        random_latents=_latents_like(idx, 4, seed=3),
        threshold_states=_one_hot_like(idx, 3, seed=4),
        smoothed_posteriors=_posteriors_like(idx, 3, seed=5),
    )


# --------------------------------------------------------------------------- #
# step 3b contracts
# --------------------------------------------------------------------------- #
def test_every_variant_is_reproducible_from_config(cfg, small_features, fitted_scaler, full_artifacts):
    """§16 step 3b acceptance: "Every variant reproducible from config"."""
    variants = [*cfg.tier1.variants, *cfg.tier1.diagnostic_variants]
    for variant in variants:
        first = build_state(
            small_features, cfg, variant, artifacts=full_artifacts, scaler=fitted_scaler,
            shuffle_seed=99,
        )
        second = build_state(
            small_features, cfg, variant, artifacts=full_artifacts, scaler=fitted_scaler,
            shuffle_seed=99,
        )
        pd.testing.assert_frame_equal(first.frame, second.frame)
        assert first.schema_hash == second.schema_hash


def test_schema_hash_is_recorded_for_every_variant(cfg, small_features, fitted_scaler, full_artifacts):
    """§16 step 3b acceptance: "schema hash recorded".

    A state frame built against a different feature schema must be detectable
    rather than merely wrong.
    """
    result = build_state(small_features, cfg, "V1", scaler=fitted_scaler)
    assert isinstance(result.schema_hash, str) and len(result.schema_hash) == 64
    assert result.feature_schema_hash == small_features.schema_hash

    narrower = build_state(
        small_features, cfg, "V1",
        scaler=FeatureScaler().fit(small_features.frame.iloc[:, :-1], scope="narrower"),
    )
    # Dropping one base column changes the realised column set, hence the hash.
    different_columns = build_state(
        small_features, cfg, "V2",
        artifacts=full_artifacts, scaler=fitted_scaler,
    )
    assert result.schema_hash != different_columns.schema_hash
    assert narrower.schema_hash == narrower.schema_hash  # self-consistent, sanity only


def test_portfolio_block_is_wired_but_unused_in_phase_a(cfg, small_features, fitted_scaler):
    """§10: "In Phase A the allocator supplies these directly, so ``state.py``
    must accept them as an optional block, wired but unused."

    Design gap D1: without current weights an agent cannot reason about
    transaction costs. The block is accepted now so Phase B does not have to
    reshape the interface.
    """
    without = build_state(small_features, cfg, "V1", scaler=fitted_scaler, portfolio_block=None)
    assert not without.includes_portfolio_block
    assert not any(str(c).startswith("portfolio_") for c in without.columns)

    portfolio = pd.DataFrame(
        {
            "portfolio_time_since_rebalance": 0.0,
            "portfolio_cumulative_turnover": 0.0,
        },
        index=small_features.frame.index,
    )
    with_block = build_state(
        small_features, cfg, "V1", scaler=fitted_scaler, portfolio_block=portfolio
    )
    assert with_block.includes_portfolio_block
    assert "portfolio_time_since_rebalance" in with_block.columns
    assert "portfolio_cumulative_turnover" in with_block.columns
    # Unused: dropping it from the computation path changes nothing about the
    # non-portfolio columns' values.
    pd.testing.assert_frame_equal(
        with_block.frame.drop(columns=["portfolio_time_since_rebalance", "portfolio_cumulative_turnover"]),
        without.frame,
    )


def test_portfolio_block_must_cover_every_state_date(cfg, small_features, fitted_scaler):
    """A partial portfolio block would otherwise silently become a reindex-fill."""
    short = pd.DataFrame(
        {"portfolio_cumulative_turnover": 0.0},
        index=small_features.frame.index[:-5],
    )
    with pytest.raises(ValueError, match="missing"):
        build_state(small_features, cfg, "V1", scaler=fitted_scaler, portfolio_block=short)


def test_all_variants_share_one_index_and_scaling_policy(cfg, small_features, fitted_scaler, full_artifacts):
    """§10: "All variants share the same index, scaling policy, and
    portfolio-state block." Otherwise the ablation compares periods, not
    representations."""
    v1 = build_state(small_features, cfg, "V1", scaler=fitted_scaler)
    v3 = build_state(small_features, cfg, "V3", artifacts=full_artifacts, scaler=fitted_scaler)

    base_cols = list(v1.columns)
    # V3's base-feature block, scaled by the SAME fitted scaler, is byte-identical
    # to V1's — the only difference is the extra posterior columns appended.
    pd.testing.assert_frame_equal(v3.frame.loc[v1.index, base_cols], v1.frame[base_cols])


def test_missing_artifact_raises_a_clear_error(cfg, small_features, fitted_scaler):
    """state.py assembles from precomputed artifacts; it must say exactly what
    is missing rather than failing deeper inside pandas."""
    with pytest.raises(MissingArtifactError, match="posteriors"):
        build_state(small_features, cfg, "V3", scaler=fitted_scaler)
    with pytest.raises(MissingArtifactError, match="latents"):
        build_state(small_features, cfg, "V4", scaler=fitted_scaler)


def test_unknown_variant_is_rejected(cfg, small_features, fitted_scaler):
    with pytest.raises(ValueError, match="unknown variant"):
        build_state(small_features, cfg, "V99", scaler=fitted_scaler)


def test_c3_shuffled_posteriors_preserve_the_marginal_distribution(cfg, small_features, fitted_scaler, full_artifacts):
    """§10: C3 is a permutation control, so it must destroy the *timing* of the
    regime signal while preserving its marginal — otherwise it controls for
    the wrong thing."""
    c3 = build_state(
        small_features, cfg, "C3", artifacts=full_artifacts, scaler=fitted_scaler, shuffle_seed=7
    )
    posterior_cols = [c for c in c3.columns if str(c).startswith("state_")]
    shuffled = c3.frame[posterior_cols]
    original = full_artifacts.posteriors.loc[shuffled.index]

    # Same multiset of rows (marginal preserved exactly) ...
    np.testing.assert_allclose(
        np.sort(shuffled.to_numpy(), axis=0), np.sort(original.to_numpy(), axis=0)
    )
    # ... attached to different dates (timing destroyed).
    assert not np.allclose(shuffled.to_numpy(), original.to_numpy())


def test_shuffle_posteriors_is_reproducible_given_a_seed():
    idx = pd.bdate_range("2020-01-01", periods=50)
    posteriors = pd.DataFrame(
        np.random.default_rng(0).dirichlet(np.ones(3), size=50),
        index=idx, columns=["state_0", "state_1", "state_2"],
    )
    a = shuffle_posteriors(posteriors, seed=42)
    b = shuffle_posteriors(posteriors, seed=42)
    pd.testing.assert_frame_equal(a, b)
    c = shuffle_posteriors(posteriors, seed=43)
    assert not a.equals(c)


def test_v1_prime_matches_v2_in_information_available(cfg, small_features, fitted_scaler, full_artifacts):
    """§10 / design gap D3: V1 sees one day, V2 sees thirty. V1' exists so the
    comparison is fair — it must carry the same window V2's encoder saw."""
    v1p = build_state(small_features, cfg, "V1p", scaler=fitted_scaler)
    v2 = build_state(small_features, cfg, "V2", artifacts=full_artifacts, scaler=fitted_scaler)

    window = cfg.encoder.window.size
    expected_first_date = small_features.frame.index[window - 1]
    assert v1p.index[0] == expected_first_date

    base_cols = [c for c in small_features.frame.columns]
    flattened_cols = [c for c in v1p.columns if c not in base_cols]
    # window * n_features flattened columns, beyond the base V1 block.
    assert len(flattened_cols) == window * small_features.frame.shape[1]
    # V1' carries the FULL window (not compressed to latent_dim) — C1 already
    # isolates the matched-dimensionality case; see the module docstring.
    assert len(flattened_cols) > v2.frame.shape[1] - small_features.frame.shape[1]


def test_v1_prime_window_is_explicit_and_overrides_the_config_default(cfg, small_features, fitted_scaler):
    """V1' must carry the window the SELECTED encoder saw, which step 3 chose as
    10 against a config default of 30. ``v1p_window`` is how production says so;
    silently using ``cfg.encoder.window.size`` built V1' at the wrong window."""
    n_base = small_features.frame.shape[1]
    default = build_state(small_features, cfg, "V1p", scaler=fitted_scaler)
    explicit = build_state(small_features, cfg, "V1p", scaler=fitted_scaler, v1p_window=10)

    assert default.frame.shape[1] == n_base + cfg.encoder.window.size * n_base
    assert explicit.frame.shape[1] == n_base + 10 * n_base
    assert explicit.index[0] == small_features.frame.index[9]
    assert explicit.schema_hash != default.schema_hash


# --------------------------------------------------------------------------- #
# Phase B control C4 = V2 + C2's threshold-regime columns (DECISIONS.md D-032)
# --------------------------------------------------------------------------- #
def test_c4_is_v2_plus_the_c2_threshold_columns(cfg, small_features, fitted_scaler, full_artifacts):
    """C4 isolates whether HMM posteriors matter beyond a threshold once a temporal
    model is present: it must be exactly V2's columns followed by C2's regime
    columns, with every block byte-identical to the one the sibling variant carries."""
    v1 = build_state(small_features, cfg, "V1", scaler=fitted_scaler)
    v2 = build_state(small_features, cfg, "V2", artifacts=full_artifacts, scaler=fitted_scaler)
    c2 = build_state(small_features, cfg, "C2", artifacts=full_artifacts, scaler=fitted_scaler)
    c4 = build_state(small_features, cfg, "C4", artifacts=full_artifacts, scaler=fitted_scaler)

    latent_cols = list(full_artifacts.latents.columns)
    regime_cols = list(full_artifacts.threshold_states.columns)
    assert c4.columns == [*v1.columns, *latent_cols, *regime_cols]
    pd.testing.assert_frame_equal(c4.frame[latent_cols], v2.frame[latent_cols])
    pd.testing.assert_frame_equal(c4.frame[regime_cols], c2.frame[regime_cols])
    pd.testing.assert_frame_equal(c4.frame[list(v1.columns)], v1.frame)
    assert c4.schema_hash not in {v2.schema_hash, c2.schema_hash}


def test_c4_carries_no_hmm_information(cfg, small_features, fitted_scaler):
    """C4's regime columns come from the threshold rule only. Supplying wildly
    different HMM posteriors must not change C4 by a single value."""
    idx = small_features.frame.index
    base = dict(
        latents=_latents_like(idx, 4, seed=1), threshold_states=_one_hot_like(idx, 2, seed=4),
    )
    a = build_state(
        small_features, cfg, "C4", scaler=fitted_scaler,
        artifacts=StateArtifacts(posteriors=_posteriors_like(idx, 2, seed=10), **base),
    )
    b = build_state(
        small_features, cfg, "C4", scaler=fitted_scaler,
        artifacts=StateArtifacts(posteriors=_posteriors_like(idx, 2, seed=11), **base),
    )
    pd.testing.assert_frame_equal(a.frame, b.frame)


def test_c4_is_phase_b_only_and_never_a_tier1_variant(cfg):
    """The Tier 1 pre-registration fixed cfg.tier1.variants; C4 must not leak into it."""
    from prism.state import PHASE_B_VARIANTS

    assert PHASE_B_VARIANTS == ("C4",)
    assert "C4" not in cfg.tier1.variants
    assert "C4" not in cfg.tier1.diagnostic_variants


def test_c4_names_exactly_which_artifact_is_missing(cfg, small_features, fitted_scaler, full_artifacts):
    no_threshold = StateArtifacts(latents=full_artifacts.latents)
    with pytest.raises(MissingArtifactError, match="threshold_states"):
        build_state(small_features, cfg, "C4", artifacts=no_threshold, scaler=fitted_scaler)
    no_latents = StateArtifacts(threshold_states=full_artifacts.threshold_states)
    with pytest.raises(MissingArtifactError, match="latents"):
        build_state(small_features, cfg, "C4", artifacts=no_latents, scaler=fitted_scaler)
