"""Deterministic canonical state relabeling. Spec §8.5.

EM state ordering is arbitrary and changes across restarts and refits
(label switching): nothing in Baum-Welch's objective cares whether "state 0"
is the calm state or the crisis state, so two fits of the identical model —
or the same fit re-run with a different ``random_state`` — can label the same
economic regime 0 in one run and 2 in another.

The reference implementation hard-coded three label lists (the HMM cell 16
live-regime readout, the LSTM notebook's PCA legend, the LSTM regime bar
chart) plus the Streamlit dashboard, all assuming an ordering that did not
match the fitted model — so the "live crisis probability" readout was
reporting the bull-state probability under a crisis label (defect B4).

``canonical_labels`` fixes this by deriving the order from a stable,
economically meaningful statistic of the fitted model itself — state-
conditional return standard deviation, ascending — so state names are a
*function of the fit*, never a list typed by hand. ``apply_canonical_labels``
then permutes every fitted array together: ``startprob_``, ``transmat_``
(both axes), ``means_``, ``covars_``, and (optionally) a posterior matrix's
columns. Permuting a subset is worse than none — a transition matrix permuted
on rows but not columns silently describes a different, wrong chain.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

__all__ = [
    "REGIME_NAMES",
    "state_return_std",
    "canonical_labels",
    "apply_canonical_labels",
    "canonicalize",
    "regime_names",
]

#: Names derived from POSITION after sorting (ascending return std), never
#: from a model's internal state index. Used only for human-readable
#: reporting (plots, tables); every numeric computation uses the integer
#: position, which these are a 1:1 labelling of.
REGIME_NAMES: dict[int, tuple[str, ...]] = {
    2: ("Calm", "Crisis"),
    3: ("Calm", "Volatile", "Crisis"),
    4: ("Calm", "Mild Volatile", "High Volatile", "Crisis"),
    5: ("Calm", "Mild Volatile", "Moderate", "High Volatile", "Crisis"),
    6: ("Calm", "Mild Volatile", "Moderate", "Elevated", "High Volatile", "Crisis"),
    7: (
        "Calm", "Mild Volatile", "Moderate", "Elevated", "High Volatile",
        "Severe", "Crisis",
    ),
    8: (
        "Calm", "Mild Volatile", "Moderate", "Elevated",
        "High Volatile", "Severe", "Extreme", "Crisis",
    ),
}


def regime_names(k: int) -> tuple[str, ...]:
    """Names for a ``k``-state model, by position after canonical sorting.

    Falls back to ``Regime 0 .. Regime k-1`` for any ``k`` not in
    :data:`REGIME_NAMES` (the K-sweep goes up to 8; spec §8.4) so a wider
    sweep never crashes reporting, it just reports less evocatively.
    """
    if k in REGIME_NAMES:
        return REGIME_NAMES[k]
    return tuple(f"Regime {i}" for i in range(k))


def state_return_std(model: Any, *, return_dim: int = 0) -> np.ndarray:
    """Per-state standard deviation of observation dimension ``return_dim``.

    ``return_dim=0`` is the project's convention throughout: both HMM
    specifications in ``configs/hmm.yaml`` (H1 and H2) place
    ``SPY_return_1d`` first, so dimension 0 is always "the return" regardless
    of which specification produced ``model``.
    """
    cov_type = getattr(model, "covariance_type", None)
    covars = model.covars_
    if cov_type == "full":
        variances = covars[:, return_dim, return_dim]
    elif cov_type == "diag":
        variances = covars[:, return_dim]
    elif cov_type == "spherical":
        variances = covars
    elif cov_type == "tied":
        raise ValueError(
            "covariance_type='tied' shares one covariance across every state, so "
            "state-conditional return std cannot distinguish them. Phase A always "
            "fits covariance_type='full' (configs/hmm.yaml); this is a guard "
            "against accidentally relaxing that."
        )
    else:
        raise ValueError(f"unsupported covariance_type {cov_type!r}")
    if np.any(variances < 0):
        raise ValueError("a fitted variance is negative; the model did not fit cleanly")
    return np.sqrt(variances)


def canonical_labels(
    model: Any,
    *,
    sort_by: str = "state_return_std",
    return_dim: int = 0,
    ascending: bool = True,
) -> np.ndarray:
    """The permutation that sorts ``model``'s states canonically.

    Returns ``order``, an array of length ``K`` where ``order[i]`` is the
    **old** state index that becomes the **new** canonical state ``i``. Pass
    this to :func:`apply_canonical_labels` to actually permute a model (and
    optionally a posterior matrix).

    Ties are broken by the original state index (``np.argsort``'s stable
    sort), so the result is deterministic even in the degenerate case of two
    states with numerically identical variance.
    """
    if sort_by != "state_return_std":
        raise ValueError(
            f"unsupported sort_by {sort_by!r}; only 'state_return_std' is "
            "implemented (spec §8.5 recommends it specifically)"
        )
    statistic = state_return_std(model, return_dim=return_dim)
    order = np.argsort(statistic, kind="stable")
    if not ascending:
        order = order[::-1]
    return order


def apply_canonical_labels(
    model: Any,
    order: np.ndarray,
    *,
    posteriors: np.ndarray | None = None,
) -> tuple[Any, np.ndarray | None]:
    """Permute every fitted parameter together, plus optionally a posterior matrix.

    Returns ``(relabeled_model, relabeled_posteriors)`` — a **deep copy** of
    ``model`` with ``startprob_``, ``transmat_``, ``means_`` and ``covars_``
    all permuted by ``order`` consistently, so the relabeled model is a
    different *labelling* of the identical chain, not a different chain.
    ``model`` itself is never mutated: a caller holding a reference to the
    pre-relabeling model (e.g. to compare against) is not surprised by it
    changing underneath them.

    ``posteriors``, if given, must have shape ``(n_samples, K)`` with columns
    in the same (pre-relabeling) state order as ``model``; its columns are
    permuted identically to the model's parameters, which is what keeps a
    posterior matrix and the model that produced it describing the same
    states after both are relabeled.
    """
    order = np.asarray(order)
    k = len(model.startprob_)
    if order.shape != (k,) or set(order.tolist()) != set(range(k)):
        raise ValueError(
            f"order must be a permutation of range({k}); got {order.tolist()}"
        )

    relabeled = copy.deepcopy(model)
    relabeled.startprob_ = model.startprob_[order]
    relabeled.transmat_ = model.transmat_[np.ix_(order, order)]
    relabeled.means_ = model.means_[order]

    cov_type = model.covariance_type
    if cov_type in ("full", "diag", "spherical"):
        relabeled.covars_ = model.covars_[order]
    elif cov_type == "tied":
        relabeled.covars_ = model.covars_  # shared; nothing to permute
    else:
        raise ValueError(f"unsupported covariance_type {cov_type!r}")

    relabeled_posteriors = None
    if posteriors is not None:
        posteriors = np.asarray(posteriors)
        if posteriors.shape[1] != k:
            raise ValueError(
                f"posteriors has {posteriors.shape[1]} columns, expected {k} to "
                "match the model's component count"
            )
        relabeled_posteriors = posteriors[:, order]

    return relabeled, relabeled_posteriors


def canonicalize(
    model: Any,
    *,
    posteriors: np.ndarray | None = None,
    sort_by: str = "state_return_std",
    return_dim: int = 0,
    ascending: bool = True,
) -> tuple[Any, np.ndarray, np.ndarray | None]:
    """Convenience wrapper: compute the order and apply it in one call.

    Returns ``(relabeled_model, order, relabeled_posteriors)``. ``order`` is
    returned too (not just applied) so a caller can record *which* original
    state became which canonical one — useful provenance when comparing a
    model across restarts or refits, where the original ordering was never
    meaningful to begin with but the mapping itself can still be logged.
    """
    order = canonical_labels(
        model, sort_by=sort_by, return_dim=return_dim, ascending=ascending
    )
    relabeled, relabeled_posteriors = apply_canonical_labels(
        model, order, posteriors=posteriors
    )
    return relabeled, order, relabeled_posteriors
