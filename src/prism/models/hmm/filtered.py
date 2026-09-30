"""CAUSAL filtered posteriors via a log-space forward recursion. Spec §8.3.

PHASE A, build step 2 — not yet implemented.
"""

from __future__ import annotations

__all__: list[str] = []


def filtered_posteriors(model, X):  # noqa: ANN001, ANN201
    """``P(s_t | x_1..x_t)`` — the only posteriors any consumer may use.

    Spec §8.3. ``hmmlearn``'s ``predict_proba`` runs forward-*backward* and
    returns **smoothed** posteriors ``P(s_t | x_1..x_T)``, which condition on
    the future. Feeding those into the state vector was defect B1: a
    look-ahead leak straight into the RL state, which would have made any
    V3/V4 advantage an artefact.

    Must return ``(posteriors, per_step_loglik)`` and satisfy, for every ``t``::

        filtered_posteriors(model, X)[0][t] == model.predict_proba(X[:t+1])[-1]

    (the backward pass is trivial at the final step, so a prefix run's last
    row *is* the filtered posterior). ``tests/test_hmm.py`` asserts exactly
    this identity.
    """
    raise NotImplementedError("spec §8.3 — build step 2")
