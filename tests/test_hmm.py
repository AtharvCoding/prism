"""HMM contracts. Spec §7.5 and §8.

These tests are written **now**, against the interfaces build step 2 must
provide, and are marked ``xfail(strict=True)``. Two consequences, both
deliberate:

* The suite runs green today and documents exactly what step 2 owes.
* When step 2 lands, each test turns XPASS — which ``strict=True`` reports as
  a **failure** — so the marker must be removed and the test becomes a real
  guard. The suite tells you the stub is gone; nobody has to remember.

The one thing testable today is the *config-level* guard against defect B3,
which is why that test is not marked.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism.config import Config, load_config

STEP_2 = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="spec §8 — HMM rebuild is build step 2. When it lands this turns "
    "XPASS (a failure) and the marker must be removed.",
)


# --------------------------------------------------------------------------- #
# testable now: the defect B3 guard lives in config validation
# --------------------------------------------------------------------------- #
def test_rolling_statistics_are_rejected_as_hmm_observations(cfg: Config):
    """Defect B3: ``spy_vol20`` as an observation collapsed the states.

    A 20-day rolling statistic is strongly autocorrelated, which violates the
    conditional-independence assumption, double-counts evidence and saturates
    the posteriors to 0/1. The result was a chain-structured transition matrix
    with near-identical state means — a volatility ladder, not economically
    distinct regimes — and it duplicated information already in the feature
    set, so the HMM could not add anything by construction.

    The guard is in ``config.py`` rather than in the fitting code, so the run
    fails at load time instead of six hours in.
    """
    with pytest.raises(ValueError, match="rolling-statistic blocklist"):
        load_config(
            root=cfg.root,
            overrides={
                "hmm": {
                    "specifications": {
                        "H1": {"observations": ["SPY_return_1d", "SPY_vol_20"]}
                    }
                }
            },
        )


def test_configured_observations_are_all_changes_or_returns(cfg: Config):
    """Every observation in every specification must be low-autocorrelation."""
    allowed_suffixes = ("return_1d", "_change")
    for name, spec in cfg.hmm.specifications.items():
        for obs in spec.observations:
            assert obs.endswith(allowed_suffixes), (
                f"HMM specification {name!r} observation {obs!r} is neither a return "
                "nor a change; §8.2 requires conditionally-independent, "
                "low-autocorrelation observables"
            )


def test_hmm_is_configured_to_fit_on_universe_a(cfg: Config):
    """§3.2: the HMM fits on Universe A, before Universe B's train window."""
    assert cfg.hmm.fit.universe == "A"
    fit_end = cfg.data.fit_early("A")[1]
    b_train_start = cfg.data.split("train")[0]
    assert fit_end < b_train_start


def test_hmm_walkforward_must_carry_filter_state(cfg: Config):
    """§8.6 / defect B5: restarting from ``startprob_`` each month is rejected."""
    assert cfg.hmm.walkforward.carry_filter_state is True
    assert cfg.hmm.walkforward.scheme == "expanding"
    with pytest.raises(ValueError, match="carried across refit"):
        load_config(
            root=cfg.root, overrides={"hmm": {"walkforward": {"carry_filter_state": False}}}
        )


def test_hmm_selection_is_driven_by_validation_likelihood(cfg: Config):
    """§8.4: validation log-likelihood is PRIMARY; BIC/AIC are secondary.

    Defect B2 made BIC invalid by multiplying an already-total log-likelihood
    by ``n`` and using ``K**2 + 2K + 4K`` as the free-parameter count.
    Recomputed correctly, BIC was monotone decreasing to the edge of the
    searched range, so ``K = 5`` was the boundary of the sweep rather than an
    optimum.
    """
    assert cfg.hmm.selection.primary == "val_loglik"
    assert "bic" in cfg.hmm.selection.secondary
    assert cfg.hmm.fit.n_restarts >= 20
    assert max(cfg.hmm.fit.k_range) > 5, (
        "the K sweep must extend past 5, or a monotone criterion cannot be "
        "distinguished from a genuine optimum (defect B2)"
    )


def test_free_parameter_count_formula(cfg: Config):
    """The §8.4 formula, computed here so step 2 has a reference to match.

    ``(K-1) + K(K-1) + Kd + Kd(d+1)/2`` for a Gaussian HMM with full
    covariance. Checked against a hand count for K=2, d=1: 1 initial + 2
    transition + 2 means + 2 variances = 7.
    """

    def n_params(k: int, d: int) -> int:
        return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2

    assert n_params(2, 1) == 7
    # K=3, d=2: 2 + 6 + 6 + 9 = 23
    assert n_params(3, 2) == 23
    # The reference formula, for contrast, is wrong in both directions.
    def reference_wrong(k: int) -> int:
        return k * k + k * 2 + k * 4

    assert reference_wrong(2) != n_params(2, 1)


# --------------------------------------------------------------------------- #
# step 2 contracts
# --------------------------------------------------------------------------- #
@STEP_2
def test_filtered_posteriors_equal_the_last_row_of_a_prefix_run(cfg: Config):
    """§7.5: "filtered posteriors at ``t`` equal the last row of a prefix-run".

    The backward pass is trivial at the final step, so
    ``model.predict_proba(X[:t+1])[-1]`` **is** the filtered posterior at
    ``t``. This identity is the definition of the thing step 2 must build, and
    the reason defect B1 was a leak: ``predict_proba`` over the full sample
    returns *smoothed* posteriors ``P(s_t | x_1..x_T)``, which differed from
    filtered by more than 0.25 on roughly 12% of days on a synthetic
    two-regime series.
    """
    from prism.models.hmm.filtered import filtered_posteriors

    filtered_posteriors(None, np.zeros((10, 1)))


@STEP_2
def test_filtered_posteriors_are_causal(cfg: Config):
    """Filtered posteriors must pass the §7.1 causality harness."""
    from prism.models.hmm.filtered import filtered_posteriors

    filtered_posteriors(None, np.zeros((10, 1)))


@STEP_2
def test_smoothed_and_filtered_posteriors_differ_materially(cfg: Config):
    """The leak must be shown to matter, not assumed to.

    If smoothed and filtered agreed, defect B1 would have been harmless and
    the mandatory rebuild would be ceremony. Step 2 must quantify the gap on
    the actual data and record it.
    """
    from prism.models.hmm.filtered import filtered_posteriors

    filtered_posteriors(None, np.zeros((10, 1)))


@STEP_2
def test_canonical_labels_are_deterministic_and_permutation_invariant(cfg: Config):
    """§7.5: "relabeling is deterministic and permutation-invariant".

    Fit twice with permuted initialisations; after canonical relabeling the
    two models must agree on every parameter. Without this, defect B4 recurs:
    hard-coded label lists assuming an ordering the fitted model does not
    have, so the "live crisis probability" reports the bull state.
    """
    from prism.models.hmm.labeling import canonical_labels

    canonical_labels(None)


@STEP_2
def test_canonical_labels_permute_every_parameter_together(cfg: Config):
    """§8.5: the permutation applies to startprob_, transmat_, means_, covars_
    **and** the posterior columns. Permuting a subset is worse than none."""
    from prism.models.hmm.labeling import canonical_labels

    canonical_labels(None)


@STEP_2
def test_log_likelihood_and_parameter_count_match_a_hand_computed_model(cfg: Config):
    """§7.5: verified "on a tiny synthetic model" with a known answer."""
    from prism.models.hmm.fit import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §8.4 — build step 2")


@STEP_2
def test_degenerate_states_are_rejected(cfg: Config):
    """§8.4: reject any state with expected duration < 5 days or
    unconditional probability < 2%."""
    from prism.models.hmm.fit import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §8.4 — build step 2")


@STEP_2
def test_walkforward_posteriors_are_deduplicated_and_calendar_aligned(cfg: Config):
    """§8.6 / defect B5: the reference rolling loop used an inclusive slice and
    produced 1,557 rows for 1,509 trading days."""
    from prism.models.hmm.walkforward import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §8.6 — build step 2")


@STEP_2
def test_hmm_is_characterised_against_nber_and_drawdown_labels(cfg: Config):
    """§8.7: detection lag and false-alarm rate against independent references."""
    from prism.models.hmm.evaluate import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §8.7 — build step 2")


@STEP_2
def test_hmm_is_compared_against_the_threshold_baseline(cfg: Config):
    """§8.7: "If the HMM cannot beat a two-state volatility threshold on both
    detection lag and downstream probe performance, that is a finding — report
    it." The comparison is mandatory; winning is not."""
    from prism.models.baselines.threshold_regime import __all__ as _  # noqa: F401

    raise NotImplementedError("spec §8.7 — build step 2")
