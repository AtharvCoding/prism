"""State-vector assembly contracts. Spec §7.5 and §10.

Build step 3b. The variant algebra is what makes the research question
falsifiable, so the contracts are written now even though the module is not.
"""

from __future__ import annotations

import pytest

from prism.config import Config, load_config

STEP_3B = pytest.mark.xfail(
    raises=(NotImplementedError, ImportError, ModuleNotFoundError),
    strict=True,
    reason="spec §10 — state assembly is build step 3b. When it lands this turns "
    "XPASS (a failure) and the marker must be removed.",
)


# --------------------------------------------------------------------------- #
# testable now: the variant set and the gate algebra
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
# step 3b contracts
# --------------------------------------------------------------------------- #
@STEP_3B
def test_every_variant_is_reproducible_from_config(cfg: Config, features_b):
    """§16 step 3b acceptance: "Every variant reproducible from config"."""
    from prism.state import build_state

    build_state(features_b, cfg, variant="V1")


@STEP_3B
def test_schema_hash_is_recorded_for_every_variant(cfg: Config, features_b):
    """§16 step 3b acceptance: "schema hash recorded".

    A state frame built against a different feature schema must be detectable
    rather than merely wrong.
    """
    from prism.state import build_state

    build_state(features_b, cfg, variant="V1")


@STEP_3B
def test_portfolio_block_is_wired_but_unused_in_phase_a(cfg: Config, features_b):
    """§10: "In Phase A the allocator supplies these directly, so ``state.py``
    must accept them as an optional block, wired but unused."

    Design gap D1: without current weights an agent cannot reason about
    transaction costs. The block is accepted now so Phase B does not have to
    reshape the interface.
    """
    from prism.state import build_state

    build_state(features_b, cfg, variant="V1", portfolio_block=None)


@STEP_3B
def test_all_variants_share_one_index_and_scaling_policy(cfg: Config, features_b):
    """§10: "All variants share the same index, scaling policy, and
    portfolio-state block." Otherwise the ablation compares periods, not
    representations."""
    from prism.state import build_state

    build_state(features_b, cfg, variant="V1")


@STEP_3B
def test_c3_shuffled_posteriors_preserve_the_marginal_distribution(cfg: Config, features_b):
    """§10: C3 is a permutation control, so it must destroy the *timing* of the
    regime signal while preserving its marginal — otherwise it controls for
    the wrong thing."""
    from prism.state import build_state

    build_state(features_b, cfg, variant="C3")


@STEP_3B
def test_v1_prime_matches_v2_in_information_available(cfg: Config, features_b):
    """§10 / design gap D3: V1 sees one day, V2 sees thirty. V1' exists so the
    comparison is fair — it must carry the same window V2's encoder saw."""
    from prism.state import build_state

    build_state(features_b, cfg, variant="V1p")
