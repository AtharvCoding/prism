"""CAUSAL filtered posteriors via a log-space forward recursion. Spec §8.3.

``hmmlearn``'s ``predict_proba`` runs forward-*backward* and returns
**smoothed** posteriors ``P(s_t | x_1..x_T)``, which condition on the future.
Feeding those into the state vector was defect B1: a look-ahead leak straight
into the RL state, which would have made any V3/V4 advantage an artefact.

``filtered_posteriors`` implements the forward pass only, in log space, by
hand — not by calling a private ``hmmlearn`` forward-pass method — so the
recursion is exactly the one the module docstring and the tests describe:

    log alpha_t(j) = logsumexp_i( log alpha_{t-1}(i) + log A[i, j] ) + log B[t, j]

where ``A`` is the transition matrix and ``B[t, j]`` is state ``j``'s
emission log-density at observation ``t`` (``model._compute_log_likelihood``
— the one private ``hmmlearn`` call this module does make, because
reimplementing the Gaussian emission density by hand would just be a second,
less-tested copy of what ``hmmlearn`` already validates in its own test
suite). ``log alpha_t`` is the *unnormalized* joint log-probability
``log P(x_1..x_t, s_t=j)``; everything else — the posterior, the per-step
log-likelihood, the running total — is a normalization or a difference of
this one quantity, computed in log space throughout so that 20+ years of
daily observations never has to exponentiate a number small enough to
underflow.

A found-by-testing bug, worth stating because it is easy to reintroduce: the
cumulative log-likelihood ``c_t = logsumexp_j(log alpha_t(j)) = log
P(x_1..x_t)`` is **not** the per-step log-likelihood. It is monotonically
decreasing in ``t`` (more observations, smaller joint probability), and
summing it across ``t`` does not reproduce ``model.score(X)`` — it was off
by five orders of magnitude on a 400-observation smoke test before this was
caught. The per-step **predictive** log-likelihood is the *difference*,
``delta_t = c_t - c_{t-1}`` (and ``delta_0 = c_0``); only ``sum(delta_t) ==
model.score(X)`` holds, and that identity is what
``tests/test_hmm.py::test_log_likelihood_and_parameter_count_match_a_hand_computed_model``
and the smoke tests in this module's own test file check.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

__all__ = ["FilteredResult", "filtered_posteriors", "score_causal_loglik"]


class FilteredResult(NamedTuple):
    """Output of :func:`filtered_posteriors`.

    Unpacks as a 2-tuple (``posteriors, per_step_loglik``) for the identity
    spec §8.3 states and the tests check; ``final_log_alpha`` is reached by
    name for the walk-forward continuation case (spec §8.6).
    """

    posteriors: np.ndarray
    per_step_loglik: np.ndarray
    final_log_alpha: np.ndarray

    def __iter__(self):
        # NamedTuple already supports 3-way unpacking; this override makes
        # the common 2-tuple call site (`post, ll = filtered_posteriors(...)`)
        # work too, matching the stub's documented contract exactly, while
        # `.final_log_alpha` remains reachable by name for callers that need
        # walk-forward continuation.
        return iter((self.posteriors, self.per_step_loglik))


def filtered_posteriors(
    model: Any,
    X: np.ndarray,
    *,
    initial_log_alpha: np.ndarray | None = None,
) -> FilteredResult:
    """``P(s_t | x_1..x_t)`` — the only posteriors any consumer may use.

    Parameters
    ----------
    model
        A **fitted** ``hmmlearn``-compatible Gaussian HMM: must expose
        ``startprob_``, ``transmat_`` and ``_compute_log_likelihood(X)``.
    X
        Observations, shape ``(n_samples, n_features)``.
    initial_log_alpha
        The unnormalized log-alpha vector to continue from, as returned by a
        previous call's ``.final_log_alpha``. Omit for a fresh start (the
        default; this is the case the §8.3 identity below is stated for).
        Supplying it conditions every posterior and log-likelihood in this
        call on the *entire* history including the previous chunk, without
        reprocessing it — this is how a walk-forward refit "carries the
        filter state across refit boundaries" (spec §8.6) rather than
        restarting from ``startprob_`` at every fold (defect B5).

    Returns
    -------
    A :class:`FilteredResult`. Unpack as ``posteriors, per_step_loglik`` for
    the ordinary case; the identity this function exists to satisfy, checked
    in the test suite, is::

        filtered_posteriors(model, X).posteriors[t] == model.predict_proba(X[:t+1])[-1]

    for every ``t``, when ``initial_log_alpha`` is omitted. (The backward
    pass is trivial at the final step of a truncated run, so a prefix run's
    last row *is* the filtered posterior — that is the whole reason this
    identity is true and worth testing.)

    ``per_step_loglik[t]`` is the *predictive* log-likelihood
    ``log P(x_t | x_1..x_{t-1})`` (or, with ``initial_log_alpha`` supplied,
    conditioned on that prior history too). ``per_step_loglik.sum()`` equals
    ``model.score(X)`` exactly when starting fresh, and equals the causally
    correct out-of-sample log-likelihood of ``X`` alone, given the carried
    state, when continuing — which is what validation-set K-selection (spec
    §8.4) and walk-forward scoring (§8.6) need: scoring a validation chunk
    with ``model.score(X_val)`` in isolation would implicitly restart the
    filter from ``startprob_`` at the validation boundary, discarding
    exactly the regime information carried over from training.
    """
    if not hasattr(model, "startprob_") or not hasattr(model, "transmat_"):
        raise ValueError(
            "filtered_posteriors requires a FITTED model (startprob_/transmat_ "
            "are not set). Call model.fit(...) first."
        )
    X = np.asarray(X, dtype="float64")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features); got shape {X.shape}")
    n_samples = X.shape[0]
    if n_samples == 0:
        raise ValueError("X has zero samples")

    # An exactly-zero start/transition probability is common once EM
    # converges to a near-deterministic chain (e.g. startprob_ == [1.0, 0.0]),
    # and log(0) == -inf is the mathematically correct value there — every
    # downstream logsumexp already treats -inf as "contributes zero
    # probability" correctly. Silenced locally, not globally (defect A9):
    # this is the one, specific, intended case, not warnings being hidden.
    with np.errstate(divide="ignore"):
        log_transmat = np.log(model.transmat_)
        log_startprob = np.log(model.startprob_) if initial_log_alpha is None else None

    framelogprob = model._compute_log_likelihood(X)  # (n_samples, n_components)
    n_components = framelogprob.shape[1]

    log_alpha = np.empty((n_samples, n_components))
    cum_loglik = np.empty(n_samples)

    if initial_log_alpha is None:
        log_alpha[0] = log_startprob + framelogprob[0]
    else:
        prior = np.asarray(initial_log_alpha, dtype="float64")
        if prior.shape != (n_components,):
            raise ValueError(
                f"initial_log_alpha has shape {prior.shape}, expected "
                f"({n_components},) to match the model's component count"
            )
        log_alpha[0] = (
            logsumexp(prior[:, None] + log_transmat, axis=0) + framelogprob[0]
        )
    cum_loglik[0] = logsumexp(log_alpha[0])

    for t in range(1, n_samples):
        log_alpha[t] = (
            logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0) + framelogprob[t]
        )
        cum_loglik[t] = logsumexp(log_alpha[t])

    posteriors = np.exp(log_alpha - cum_loglik[:, None])

    per_step_loglik = np.empty(n_samples)
    if initial_log_alpha is None:
        per_step_loglik[0] = cum_loglik[0]
    else:
        # cum_loglik[0] here already includes the carried prior's mass
        # (logsumexp(prior + log_transmat) contributes it), so the FIRST
        # step of a continued chunk is also a difference, against the prior
        # chunk's own final cumulative log-likelihood — which the caller has
        # not supplied and does not need to: what matters is that
        # per_step_loglik sums to log P(X | prior history), and the prior
        # chunk's mass cancels out of that sum by construction, exactly as
        # it does for every subsequent step below.
        per_step_loglik[0] = cum_loglik[0] - logsumexp(prior)
    per_step_loglik[1:] = cum_loglik[1:] - cum_loglik[:-1]

    return FilteredResult(
        posteriors=posteriors,
        per_step_loglik=per_step_loglik,
        final_log_alpha=log_alpha[-1],
    )


def score_causal_loglik(
    model: Any,
    observations: pd.DataFrame,
    score_start: str | pd.Timestamp,
    score_end: str | pd.Timestamp,
) -> float:
    """Causal log-likelihood of ``[score_start, score_end]``, honestly scored.

    "Honestly" means: conditioned on *everything in* ``observations`` from its
    own start through ``score_end`` — not just on the scored window in
    isolation. This is what spec §8.4's "validation-set (2018) log-likelihood"
    requires in practice, because the model is fit on Universe A's 1999-2006
    ``fit_early`` window while the validation window is 2018: eleven years
    (2007-2017) of unmodelled history sit between them. Scoring 2018 against
    the model's ``startprob_`` directly — as if January 2018 were the first
    day the chain ever ran — would skip the transition-matrix-driven mixing
    that should have happened across those eleven years and misstate how
    confident the filter actually is by 2018. Scoring causally through the
    whole prefix is the fix, and it is cheap: a full filtered pass over ~6,300
    daily observations takes well under half a second.

    ``observations`` must be indexed by date and contain no gaps the model's
    own fit did not already tolerate; this function does no alignment of its
    own beyond date-range slicing.
    """
    index = pd.DatetimeIndex(observations.index)
    start_ts, end_ts = pd.Timestamp(score_start), pd.Timestamp(score_end)
    through_end = index <= end_ts
    if not through_end.any():
        raise ValueError(f"no observations on or before {end_ts.date()}")
    sub_index = index[through_end]
    X = observations.loc[through_end].to_numpy(dtype="float64")

    result = filtered_posteriors(model, X)

    scored = sub_index >= start_ts
    if not scored.any():
        raise ValueError(
            f"no observations in [{start_ts.date()}, {end_ts.date()}] to score"
        )
    return float(result.per_step_loglik[np.asarray(scored)].sum())
