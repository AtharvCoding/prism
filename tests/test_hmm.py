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
import pandas as pd
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
def _fit_two_regime_hmm(rng: np.random.Generator) -> tuple["GaussianHMM", np.ndarray]:
    """A genuinely Markov-switching two-regime series, fitted with hmmlearn directly.

    Deliberately bypasses ``prism.models.hmm.fit`` — these tests check what
    ``filtered_posteriors`` promises about ANY fitted hmmlearn model, not
    anything specific to this project's own fitting code.

    A single calm-vs-crisis segment (tried first) turned out to be the wrong
    fixture: with only two clean transitions and emissions separated enough
    for the model to be instantly confident, fewer than 2% of days showed a
    material smoothed/filtered gap — nowhere near enough to demonstrate
    defect B1. The actual source of the gap is **regime persistence
    ambiguity**: an observation just after a transition could plausibly still
    belong to the old regime, and only the next several observations resolve
    it. That needs a real Markov chain with many transitions (stay
    probability 0.97, so ~30-session expected duration — long enough to be a
    regime, short enough to transition often in 600 points) and emissions
    that overlap rather than separate cleanly. This reaches ~11% of days with
    a >0.25 gap, close to the spec's own ~12% finding on its synthetic
    two-regime series.
    """
    from hmmlearn.hmm import GaussianHMM

    n = 600
    stay_prob = 0.97
    transmat = np.array([[stay_prob, 1 - stay_prob], [1 - stay_prob, stay_prob]])
    means = np.array([0.0, 1.8])
    sds = np.array([1.0, 2.2])

    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=transmat[states[t - 1]])
    assert (np.diff(states) != 0).sum() >= 5, "fixture drew too few transitions"
    X = rng.normal(means[states], sds[states]).reshape(-1, 1)

    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=7)
    model.fit(X)
    return model, X


def test_filtered_posteriors_equal_the_last_row_of_a_prefix_run(cfg: Config, rng):
    """§7.5: "filtered posteriors at ``t`` equal the last row of a prefix-run".

    The backward pass is trivial at the final step, so
    ``model.predict_proba(X[:t+1])[-1]`` **is** the filtered posterior at
    ``t``. This identity is the definition of filtered posteriors, and the
    reason defect B1 was a leak: ``predict_proba`` over the full sample
    returns *smoothed* posteriors ``P(s_t | x_1..x_T)``, which condition on
    the future.
    """
    from prism.models.hmm.filtered import filtered_posteriors

    model, X = _fit_two_regime_hmm(rng)
    result = filtered_posteriors(model, X)

    checked = 0
    for t in range(0, len(X), 17):  # a spread of points, not every single one
        expected = model.predict_proba(X[: t + 1])[-1]
        np.testing.assert_allclose(result.posteriors[t], expected, atol=1e-10)
        checked += 1
    assert checked > 10, "the identity should be checked at more than a handful of points"

    # And the unpacked 2-tuple form the stub's docstring promises must work.
    posteriors, per_step_loglik = filtered_posteriors(model, X)
    assert posteriors.shape == (len(X), model.n_components)
    assert per_step_loglik.shape == (len(X),)


def test_filtered_posteriors_sum_to_the_models_total_log_likelihood(cfg: Config, rng):
    """``per_step_loglik.sum() == model.score(X)`` exactly.

    Pinned because the bug it guards against is real: an earlier draft of
    this module returned the CUMULATIVE log-likelihood ``log P(x_1..x_t)`` at
    each step instead of the per-step PREDICTIVE log-likelihood
    ``log P(x_t | x_1..x_{t-1})``. The two are easy to confuse — one is a
    monotone running total, the other is its first difference — and summing
    the cumulative version was off from ``model.score(X)`` by five orders of
    magnitude on a 400-observation smoke test before this test existed.
    """
    from prism.models.hmm.filtered import filtered_posteriors

    model, X = _fit_two_regime_hmm(rng)
    result = filtered_posteriors(model, X)
    np.testing.assert_allclose(result.per_step_loglik.sum(), model.score(X), rtol=1e-8)


def test_filtered_posteriors_are_causal(cfg: Config, rng):
    """Filtered posteriors must pass the §7.1 causality harness.

    Uses the harness's own truncation/perturbation machinery against a build
    function that fits fresh each call — the honest test, since a fitted
    model's *parameters* are themselves a function of the data (fit scope,
    §7.2, is a separate guarantee from causality, §7.1). What this test
    isolates is narrower and is the right scope for `filtered.py` alone: with
    the model's parameters HELD FIXED, does the filtered posterior at `t`
    depend on observations after `t`? It must not.
    """
    from prism.models.hmm.filtered import filtered_posteriors

    model, X = _fit_two_regime_hmm(rng)

    def build(obs: np.ndarray) -> np.ndarray:
        return filtered_posteriors(model, obs).posteriors

    full = build(X)
    for cut in (80, 150, 250, len(X) - 5):
        truncated = build(X[: cut + 1])
        np.testing.assert_allclose(
            full[: cut + 1], truncated, atol=1e-10,
            err_msg=f"filtered posteriors at/before {cut} changed when the series was truncated",
        )

        noisy = X.copy()
        noisy[cut + 1 :] = rng.uniform(0.5, 1.5, noisy[cut + 1 :].shape) * noisy[cut + 1 :]
        perturbed = build(noisy)
        np.testing.assert_allclose(
            full[: cut + 1], perturbed[: cut + 1], atol=1e-10,
            err_msg=f"filtered posteriors at/before {cut} changed when the future was randomised",
        )


def test_smoothed_and_filtered_posteriors_differ_materially(cfg: Config, rng):
    """The leak must be shown to matter, not assumed to.

    If smoothed and filtered agreed, defect B1 would have been harmless and
    the mandatory rebuild would be ceremony. A regime that is about to end
    looks, right up until the observation before the transition, exactly
    like one that will persist for the smoothed posterior to disagree with
    the filtered one on, because smoothing sees the transition coming and
    filtering does not. Quantified here, not assumed: a material fraction of
    days must show a gap of more than 0.25 in a state's probability.
    """
    from prism.models.hmm.filtered import filtered_posteriors

    model, X = _fit_two_regime_hmm(rng)
    filtered = filtered_posteriors(model, X).posteriors
    smoothed = model.predict_proba(X)

    max_abs_diff = np.abs(filtered - smoothed).max(axis=1)
    material_fraction = float((max_abs_diff > 0.25).mean())
    assert material_fraction > 0.05, (
        f"only {material_fraction:.1%} of days show a >0.25 smoothed/filtered gap; "
        "the synthetic series may not have enough regime persistence to demonstrate "
        "the leak (spec's own synthetic experiment found ~12%)"
    )
    # And the two must agree exactly on the LAST observation (no future to see).
    np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-10)


def _manually_permuted_clone(model, order):
    """A SECOND model: the same chain as ``model``, labelled differently.

    Constructs it by hand (not through ``apply_canonical_labels``, which is
    the function under test) so the test has an independent ground truth.
    ``order[i]`` names which of ``model``'s states becomes ``clone``'s state
    ``i`` — the exact same contract ``apply_canonical_labels`` documents —
    simulating what a second EM run with a different random initialisation
    could converge to: the identical chain, relabelled arbitrarily.
    """
    import copy

    import numpy as np

    order = np.asarray(order)
    clone = copy.deepcopy(model)
    clone.startprob_ = model.startprob_[order]
    clone.transmat_ = model.transmat_[np.ix_(order, order)]
    clone.means_ = model.means_[order]
    clone.covars_ = model.covars_[order]
    return clone


def test_canonical_labels_are_deterministic_and_permutation_invariant(cfg: Config, rng):
    """§7.5: "relabeling is deterministic and permutation-invariant".

    Build a second model that is the SAME chain as the first, with its states
    relabelled by an arbitrary permutation (simulating what a second EM run
    with a different random initialisation could converge to). After
    canonical relabeling, the two must agree on every parameter — if they
    did not, defect B4 recurs: a label meaning "crisis" in one run could mean
    "bull" in another, and nothing downstream would know.
    """
    import numpy as np

    from prism.models.hmm.labeling import canonicalize

    model, X = _fit_two_regime_hmm(rng)
    arbitrary_order = np.array([1, 0])  # the only nontrivial permutation of 2 states
    relabelled_input = _manually_permuted_clone(model, arbitrary_order)

    canonical_a, order_a, _ = canonicalize(model)
    canonical_b, order_b, _ = canonicalize(relabelled_input)

    np.testing.assert_allclose(canonical_a.startprob_, canonical_b.startprob_, atol=1e-12)
    np.testing.assert_allclose(canonical_a.transmat_, canonical_b.transmat_, atol=1e-12)
    np.testing.assert_allclose(canonical_a.means_, canonical_b.means_, atol=1e-12)
    np.testing.assert_allclose(canonical_a.covars_, canonical_b.covars_, atol=1e-12)

    # And running it twice on the SAME model must be deterministic.
    canonical_c, order_c, _ = canonicalize(model)
    np.testing.assert_array_equal(order_a, order_c)
    np.testing.assert_allclose(canonical_a.means_, canonical_c.means_, atol=1e-15)


def test_canonical_labels_permute_every_parameter_together(cfg: Config, rng):
    """§8.5: the permutation applies to startprob_, transmat_, means_, covars_
    **and** the posterior columns. Permuting a subset is worse than none: a
    transition matrix permuted on rows but not columns describes a different,
    wrong chain.
    """
    import numpy as np

    from prism.models.hmm.filtered import filtered_posteriors
    from prism.models.hmm.labeling import canonicalize

    model, X = _fit_two_regime_hmm(rng)
    posteriors = filtered_posteriors(model, X).posteriors

    relabelled, order, relabelled_posteriors = canonicalize(model, posteriors=posteriors)

    # Every array actually changed (states were genuinely reordered), except
    # where the permutation happens to be the identity.
    if list(order) != [0, 1]:
        assert not np.allclose(relabelled.means_, model.means_)
        assert not np.allclose(relabelled.transmat_, model.transmat_)

    # The relabelled TRANSITION MATRIX must still be the same chain, just
    # relabelled — both axes permuted consistently, not just rows. Checked
    # directly: relabelled.transmat_[i, j] must equal
    # model.transmat_[order[i], order[j]].
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(
                relabelled.transmat_[i, j], model.transmat_[order[i], order[j]], atol=1e-12
            )

    # The relabelled model, run through filtered_posteriors again, must give
    # the SAME posteriors as the original model's posteriors with columns
    # permuted — proving the relabelled model is the identical chain, not a
    # different one that happens to share some parameters.
    recomputed = filtered_posteriors(relabelled, X).posteriors
    np.testing.assert_allclose(recomputed, posteriors[:, order], atol=1e-8)

    # And the posterior matrix passed alongside the model was permuted the
    # same way.
    np.testing.assert_allclose(relabelled_posteriors, posteriors[:, order], atol=1e-15)

    # Canonical order itself: ascending return std.
    from prism.models.hmm.labeling import state_return_std

    stds = state_return_std(relabelled)
    assert stds[0] <= stds[1], "states are not sorted ascending by return std"


def _brute_force_loglik(startprob, transmat, means, sds, X) -> float:
    """Independent ground truth: sum P(X, S) over EVERY state path ``S``.

    Tiny ``T`` only (state-path count is ``K**T``) — this exists to be
    obviously correct by inspection, not to be fast. A HMM's log-likelihood
    is, by definition, ``log sum_S P(X, S)``; this computes exactly that
    definition, independent of any forward-algorithm implementation
    (``filtered.py``'s or ``hmmlearn``'s own), so the comparison is a real
    cross-check rather than the module testing itself.
    """
    from itertools import product

    from scipy.stats import norm

    n = len(X)
    k = len(startprob)
    total = 0.0
    for path in product(range(k), repeat=n):
        p = startprob[path[0]]
        for t in range(1, n):
            p *= transmat[path[t - 1], path[t]]
        for t in range(n):
            p *= norm.pdf(X[t, 0], means[path[t], 0], sds[path[t]])
        total += p
    return float(np.log(total))


def test_log_likelihood_and_parameter_count_match_a_hand_computed_model(cfg: Config):
    """§7.5: verified "on a tiny synthetic model" with a known answer.

    A manually-parameterised (not EM-fitted) 2-state, 1-D Gaussian HMM with a
    4-observation sequence — small enough that the brute-force sum over all
    ``2**4 = 16`` state paths is obviously correct by inspection. Checks both
    halves of §8.4's correctness claim: the log-likelihood itself, and the
    free-parameter count defect B2 got wrong.
    """
    from hmmlearn.hmm import GaussianHMM

    from prism.models.hmm.fit import n_free_params

    startprob = np.array([0.6, 0.4])
    transmat = np.array([[0.7, 0.3], [0.2, 0.8]])
    means = np.array([[0.0], [2.0]])
    sds = np.array([1.0, 1.5])
    covars = np.array([[[1.0]], [[2.25]]])
    X = np.array([[0.5], [1.8], [-0.3], [2.4]])

    model = GaussianHMM(n_components=2, covariance_type="full")
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    model.n_features = 1

    expected = _brute_force_loglik(startprob, transmat, means, sds, X)
    np.testing.assert_allclose(model.score(X), expected, rtol=1e-10)

    # K=2, d=1: (K-1) + K(K-1) + Kd + Kd(d+1)/2 = 1 + 2 + 2 + 2 = 7, matching
    # exactly the parameters set above: 1 free startprob entry, 2 free
    # transmat entries, 2 means, 2 (1x1 "full" covariance) variances.
    assert n_free_params(2, 1) == 7


def test_degenerate_states_are_rejected(cfg: Config):
    """§8.4: reject any state with expected duration < 5 days or
    unconditional probability < 2%.

    Two hand-built models: one genuinely degenerate (a state visited with
    stationary probability under 1%, and a state with a one-day expected
    duration), one not. ``is_degenerate`` must distinguish them, and the
    thresholds must be the exact ones spec §8.4 and ``configs/hmm.yaml``
    state — not approximately.
    """
    from hmmlearn.hmm import GaussianHMM

    from prism.models.hmm.fit import is_degenerate

    def make_model(startprob, transmat):
        model = GaussianHMM(n_components=len(startprob), covariance_type="full")
        model.startprob_ = np.asarray(startprob)
        model.transmat_ = np.asarray(transmat)
        model.means_ = np.zeros((len(startprob), 1))
        model.covars_ = np.ones((len(startprob), 1, 1))
        model.n_features = 1
        return model

    # A rarely-visited state with a LONG duration — degenerate on the
    # unconditional-probability criterion only, by construction, so the test
    # below can prove that criterion is actually load-bearing rather than
    # riding along on the duration check also firing.
    rare_state = make_model(
        [0.99, 0.01],
        [[0.9995, 0.0005], [0.05, 0.95]],
    )
    stationary = rare_state.get_stationary_distribution()
    assert stationary.min() < 0.02, "fixture should have a rare state to be meaningful"
    duration = 1.0 / (1.0 - np.diag(rare_state.transmat_))
    assert duration.min() >= 5.0, "fixture's rare state must NOT also be short-lived"
    assert is_degenerate(rare_state, min_expected_duration_days=5.0, min_unconditional_prob=0.02)

    # A flickering state: self-transition 0.3 -> expected duration ~1.4 days.
    flickering = make_model([0.5, 0.5], [[0.3, 0.7], [0.5, 0.5]])
    duration_state0 = 1.0 / (1.0 - 0.3)
    assert duration_state0 < 5.0, "fixture should have a short-duration state"
    assert is_degenerate(flickering, min_expected_duration_days=5.0, min_unconditional_prob=0.02)

    # A healthy two-state model: well-separated durations and mass.
    healthy = make_model([0.5, 0.5], [[0.97, 0.03], [0.04, 0.96]])
    assert not is_degenerate(
        healthy, min_expected_duration_days=5.0, min_unconditional_prob=0.02
    )

    # The thresholds must actually be used, not hardcoded inside the function:
    # a very lax threshold must accept what the default threshold rejects.
    assert not is_degenerate(
        rare_state, min_expected_duration_days=5.0, min_unconditional_prob=0.0001
    )


def _make_walkforward_observations(rng, start="2015-01-02", end="2018-12-31"):
    """A short, two-regime observation series for fast walk-forward tests."""
    from prism.utils.calendar import trading_days

    idx = trading_days(start, end)
    n = len(idx)
    transmat = np.array([[0.97, 0.03], [0.04, 0.96]])
    means, sds = np.array([0.0, 1.8]), np.array([1.0, 2.2])
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=transmat[states[t - 1]])
    X = rng.normal(means[states], sds[states])
    return pd.DataFrame({"SPY_return_1d": X}, index=idx)


def test_walkforward_posteriors_are_deduplicated_and_calendar_aligned(cfg: Config, rng):
    """§8.6 / defect B5: the reference rolling loop used an inclusive slice and
    produced 1,557 rows for 1,509 trading days.

    Quarterly cadence over four years (not the configured monthly default) —
    the property under test does not depend on cadence, and a handful of
    folds exercises the same concatenation/de-duplication logic in a fraction
    of the wall-clock time.
    """
    from prism.models.hmm.walkforward import hmm_walkforward
    from prism.utils.calendar import trading_days

    obs = _make_walkforward_observations(rng)
    result = hmm_walkforward(
        obs,
        k=2,
        fit_start="2015-01-02",
        first_apply_start="2017-01-01",
        apply_end="2018-12-31",
        cadence="quarterly",
        embargo_days=5,
        covariance_type="full",
        n_restarts=3,
        n_iter=100,
        tol=1e-3,
        seed_base=cfg.data.seeds.master,
    )

    assert len(result.folds) >= 4, "expected several quarterly folds over two years"
    assert result.posteriors.index.is_unique, "defect B5: duplicate dates in the merged series"
    assert result.posteriors.index.is_monotonic_increasing
    assert result.per_step_loglik.index.equals(result.posteriors.index)

    # Covers exactly the apply span's trading days — no gaps, no extras.
    expected_index = trading_days("2017-01-01", "2018-12-31")
    assert result.posteriors.index.equals(expected_index), (
        "walk-forward posteriors do not exactly tile the apply period"
    )

    # Every row is a valid probability distribution over 2 states.
    row_sums = result.posteriors.sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)
    assert (result.posteriors.to_numpy() >= -1e-12).all()

    # The refit date is never inside its own fit window (spec §8.6) — proven
    # structurally by WalkForwardFold's own constructor, re-checked here on
    # the actual folds this call produced.
    for fold_output in result.folds:
        assert fold_output.fold.fit_end < fold_output.fold.apply_start

    # Scalers are refit per fold, never a shared/stale instance (spec §8.6).
    scopes = [f.scaler.fit_record.scope for f in result.folds]
    assert len(set(scopes)) == len(scopes), "scaler scope is not unique per fold"
    fit_ends = [f.scaler.fit_record.end for f in result.folds]
    assert fit_ends == sorted(fit_ends) and len(set(fit_ends)) == len(fit_ends), (
        "each fold's scaler must be fit on that fold's own (expanding) window"
    )

    # Canonical labelling: every fold's model has ascending return std.
    from prism.models.hmm.labeling import state_return_std

    for fold_output in result.folds:
        stds = state_return_std(fold_output.model)
        assert stds[0] <= stds[1], f"fold {fold_output.fold.index} is not canonically labelled"


def _make_crisis_scenario(start="2015-01-02", end="2019-12-31"):
    """A synthetic price path with PERSISTENT, RECURRING regime switching.

    An earlier version of this fixture was "calm noise, then one late burst".
    That produced DEGENERATE fits in every fold whose fit window fell
    entirely before the burst (there was no two-state structure in that
    window for EM to find — not a bug in the code under test, a weakness in
    the fixture). A true two-state Markov chain (persistent, recurring
    crisis visits throughout, rather than one isolated event) gives every
    fold's fit window real regime structure, and still produces genuine,
    dateable >20% drawdowns for ``drawdown_bear_episodes`` to find
    independently of anything the HMM or the threshold baseline computes.

    Uses its OWN dedicated, hard-coded seed rather than the shared ``rng``
    pytest fixture. A first version drew from ``rng`` (seed 20260101, fixed
    project-wide) and that specific draw happened to produce one crisis so
    early and so persistent that every episode's START fell before the
    walk-forward apply window even began — meaning the "detect it inside the
    apply window" test path could never run, silently, forever, under that
    fixture's seed. Seed 16 here was chosen by checking the first 20
    candidates for one that reliably lands multiple contained episodes
    inside the ``2017-01-01..2019-12-31`` apply window used below.
    """
    from prism.utils.calendar import trading_days

    idx = trading_days(start, end)
    n = len(idx)
    rng = np.random.default_rng(16)
    transmat = np.array([[0.992, 0.008], [0.08, 0.92]])  # ~125-day calm, ~12.5-day crisis
    means = np.array([0.0006, -0.010])
    sds = np.array([0.008, 0.022])
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=transmat[states[t - 1]])
    returns = rng.normal(means[states], sds[states])
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=idx)
    observations = pd.DataFrame({"SPY_return_1d": returns}, index=idx)
    # A crude but legitimate vol proxy for the threshold baseline: large
    # moves (up or down) should read as "high vol", independent of the HMM.
    vol_proxy = pd.Series(np.abs(returns) * 100.0 + 10.0, index=idx, name="vol_proxy")
    return idx, prices, observations, vol_proxy


def test_hmm_is_characterised_against_nber_and_drawdown_labels(cfg: Config):
    """§8.7: detection lag and false-alarm rate against independent references.

    End-to-end: walk-forward HMM posteriors on a synthetic crisis scenario,
    a drawdown episode found independently from the price path alone, and
    the crisis (highest-index canonical) state's probability checked against
    it via ``detect_episodes`` — proving the whole chain from
    ``hmm_walkforward`` through ``characterise_states``/``detect_episodes``
    actually wires together, not just each piece in isolation.
    """
    from prism.models.hmm.evaluate import (
        characterise_states,
        detect_episodes,
        drawdown_bear_episodes,
        posterior_quality,
    )
    from prism.models.hmm.walkforward import hmm_walkforward

    idx, prices, observations, _ = _make_crisis_scenario()
    episodes = drawdown_bear_episodes(prices, threshold=0.20)
    assert episodes, "fixture should produce at least one >20% drawdown episode"

    wf = hmm_walkforward(
        observations, k=2,
        fit_start="2015-01-02", first_apply_start="2017-01-01", apply_end="2019-12-31",
        cadence="quarterly", embargo_days=5,
        covariance_type="full", n_restarts=3, n_iter=100, tol=1e-3, seed_base=cfg.data.seeds.master,
    )

    # Characterisation runs on the walk-forward output without error and
    # produces a plausible table: the higher-indexed canonical state (ascending
    # return std) must show materially higher volatility than the lower one.
    returns = observations["SPY_return_1d"].loc[wf.posteriors.index]
    characterisation = characterise_states(wf.posteriors, returns)
    assert (
        characterisation.table.loc[1, "volatility_annualised"]
        > characterisation.table.loc[0, "volatility_annualised"]
    ), "canonical state 1 should be the more volatile one, by construction (§8.5)"

    quality = posterior_quality(wf.posteriors, entropy_saturation_warn=0.05)
    assert quality.mean_entropy >= 0.0
    assert 0.0 <= quality.flip_rate <= 1.0

    # Detection against the independently-derived drawdown episode(s) that
    # fall inside the walk-forward apply window.
    in_window = [(s, e) for s, e in episodes if s >= wf.posteriors.index[0]]
    assert in_window, "fixture's seed should reliably land an episode inside the apply window"
    detections = detect_episodes(wf.posteriors["state_1"], in_window, threshold=0.5)
    assert len(detections) == len(in_window)
    for d in detections:
        assert isinstance(d.detected, bool)
        if d.detected:
            assert d.lag_sessions is not None


def test_hmm_is_compared_against_the_threshold_baseline(cfg: Config):
    """§8.7: "If the HMM cannot beat a two-state volatility threshold on both
    detection lag and downstream probe performance, that is a finding — report
    it." The comparison is mandatory; winning is not — this test asserts the
    comparison RUNS and produces a well-formed result, not that the HMM wins.
    """
    from prism.models.baselines.threshold_regime import threshold_regime_walkforward
    from prism.models.hmm.evaluate import compare_detection, drawdown_bear_episodes
    from prism.models.hmm.walkforward import hmm_walkforward

    idx, prices, observations, vol_proxy = _make_crisis_scenario()
    episodes = drawdown_bear_episodes(prices, threshold=0.20)
    assert episodes

    fold_kwargs = dict(
        fit_start="2015-01-02", first_apply_start="2017-01-01", apply_end="2019-12-31",
        cadence="quarterly", embargo_days=5,
    )
    wf = hmm_walkforward(
        observations, k=2, covariance_type="full", n_restarts=3, n_iter=100, tol=1e-3,
        seed_base=cfg.data.seeds.master, **fold_kwargs,
    )
    baseline = threshold_regime_walkforward(vol_proxy, k=2, **fold_kwargs)

    in_window = [(s, e) for s, e in episodes if s >= wf.posteriors.index[0]]
    assert in_window, (
        "fixture's seed should reliably land at least one episode inside the apply "
        "window; if this fires, the fixture's seed or parameters need re-checking "
        "(see _make_crisis_scenario's docstring)"
    )

    comparison = compare_detection(
        wf.posteriors["state_1"], baseline["state_1"], in_window, threshold=0.5
    )
    # The comparison must be well-formed regardless of which detector wins.
    assert len(comparison.hmm) == len(comparison.baseline) == len(in_window)
    assert 0.0 <= comparison.hmm_false_alarm_rate <= 1.0 or np.isnan(
        comparison.hmm_false_alarm_rate
    )
    assert isinstance(comparison.hmm_wins_on_detection_lag(), bool)
    table = comparison.summary_table()
    assert {"hmm_detected", "baseline_detected", "hmm_lag_sessions", "baseline_lag_sessions"} <= set(
        table.columns
    )
