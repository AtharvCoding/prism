"""Baum-Welch fitting with restarts and validation-likelihood selection. Spec §8.4.

Three things the reference implementation got wrong, fixed here:

**BIC was invalid (defect B2).** ``model.score(X)`` already returns the
*total* log-likelihood of the sequence; the reference code multiplied it by
``n`` again, inflating it by orders of magnitude. The free-parameter count
was also wrong (``K**2 + 2*K + 4*K``, an approximation that does not match a
Gaussian HMM with full covariance). :func:`n_free_params` implements the
correct formula from spec §8.4 and is checked against a hand count in
``tests/test_hmm.py``.

**Selection was BIC/AIC-only, searched too small a range, and rewarded the
edge of the sweep rather than a real optimum.** With the corrected formula
the reference's K sweep (2-5) gave a BIC that was still monotone decreasing
at K=5 — i.e. K=5 "won" only because the sweep stopped there, not because it
was a minimum. This module's :func:`select_k` sweeps up to ``k_range``'s
configured ceiling (8 by default) and selects on **validation-set
log-likelihood primarily**, with BIC/AIC reported as secondary diagnostics
(spec §8.4) — and that validation score is computed *causally*, continuing
the filtered forward recursion from the end of training rather than scoring
the validation chunk in isolation (see :mod:`prism.models.hmm.filtered`'s
module docstring for why ``model.score(X_val)`` alone would be wrong).

**Degenerate solutions were never rejected.** A state with near-zero
unconditional probability or an expected duration of a day or two is not a
regime, it is EM exploiting a handful of outliers. :func:`is_degenerate`
implements the spec §8.4 rejection rule and :func:`select_k` prefers
non-degenerate restarts/K values, falling back to the full pool (flagged)
only if every candidate is degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prism.models.hmm.filtered import score_causal_loglik
from prism.utils.logging import get_logger
from prism.utils.seeding import derive_seed

__all__ = [
    "RestartResult",
    "KResult",
    "n_free_params",
    "DegeneracyCheck",
    "is_degenerate",
    "fit_restarts",
    "select_k",
    "select_best_k",
    "FitSweepResult",
    "fit_and_select",
]

_log = get_logger(__name__)


def n_free_params(k: int, d: int) -> int:
    """Free parameters of a Gaussian HMM, full covariance. Spec §8.4.

    ``(K-1)`` initial distribution + ``K(K-1)`` transition-matrix rows +
    ``Kd`` means + ``Kd(d+1)/2`` covariance entries (each state's full
    covariance is symmetric, so only the upper triangle is free).
    """
    if k < 1 or d < 1:
        raise ValueError(f"k and d must be positive; got k={k}, d={d}")
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


@dataclass(frozen=True)
class DegeneracyCheck:
    """Per-state detail behind :func:`is_degenerate`'s verdict.

    ``bool(check)`` reproduces the old scalar-return behaviour exactly (every
    existing ``if is_degenerate(...)`` / ``assert is_degenerate(...)`` call
    site keeps working unchanged), while :attr:`expected_duration_days` and
    :attr:`stationary_prob` expose the per-state values the verdict was
    already computed from, for diagnostics that need to say WHICH state
    failed WHICH bound rather than just that the model, overall, did.
    """

    degenerate: bool
    expected_duration_days: np.ndarray
    stationary_prob: np.ndarray
    min_expected_duration_days: float
    min_unconditional_prob: float

    def __bool__(self) -> bool:
        return self.degenerate

    def failing_states(self) -> list[dict[str, Any]]:
        """One row per (state, bound) that is checked, duration and mass both.

        Includes passing rows too (``fails=False``) so a diagnostic can audit
        every state's margin to its threshold, not only the failures.
        """
        rows: list[dict[str, Any]] = []
        for i, (dur, stat) in enumerate(zip(self.expected_duration_days, self.stationary_prob)):
            rows.append(
                {
                    "state": i,
                    "bound": "expected_duration_days",
                    "value": float(dur),
                    "threshold": self.min_expected_duration_days,
                    "fails": bool(dur < self.min_expected_duration_days),
                }
            )
            rows.append(
                {
                    "state": i,
                    "bound": "stationary_prob",
                    "value": float(stat),
                    "threshold": self.min_unconditional_prob,
                    "fails": bool(stat < self.min_unconditional_prob),
                }
            )
        return rows


def is_degenerate(
    model: Any,
    *,
    min_expected_duration_days: float = 5.0,
    min_unconditional_prob: float = 0.02,
) -> DegeneracyCheck:
    """Spec §8.4: reject states with near-zero mass or near-zero persistence.

    Expected duration of state ``i`` under a discrete-time Markov chain is
    ``1 / (1 - P(i -> i))``: the mean of a geometric distribution over how
    many consecutive days the chain stays in ``i``. Unconditional
    (stationary) probability comes from ``model.get_stationary_distribution()``
    — the left eigenvector of ``transmat_`` for eigenvalue 1, which
    ``hmmlearn`` already implements correctly; re-deriving it by hand here
    would just be a second, less-tested copy.

    Returns a :class:`DegeneracyCheck`, not a bare ``bool`` — see its
    docstring for why existing truthiness-only call sites are unaffected.
    """
    diag = np.diag(model.transmat_)
    durations = 1.0 / np.clip(1.0 - diag, 1e-12, None)
    stationary = model.get_stationary_distribution()
    degenerate = bool((durations < min_expected_duration_days).any()) or bool(
        (stationary < min_unconditional_prob).any()
    )
    return DegeneracyCheck(
        degenerate=degenerate,
        expected_duration_days=durations,
        stationary_prob=stationary,
        min_expected_duration_days=min_expected_duration_days,
        min_unconditional_prob=min_unconditional_prob,
    )


@dataclass(frozen=True)
class RestartResult:
    """One random restart's fit, for one ``K``."""

    k: int
    seed: int
    model: Any
    train_loglik: float
    converged: bool
    n_iter: int
    degenerate: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "seed": self.seed,
            "train_loglik": self.train_loglik,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "degenerate": self.degenerate,
        }


@dataclass(frozen=True)
class KResult:
    """Every restart for one ``K``, the selected best, and its scores."""

    k: int
    n_params: int
    restarts: list[RestartResult]
    best: RestartResult
    val_loglik: float
    bic: float
    aic: float
    degenerate: bool

    @property
    def train_logliks(self) -> np.ndarray:
        """The full restart log-likelihood distribution. Spec §8.4: "report
        the log-likelihood distribution, not just the best"."""
        return np.array([r.train_loglik for r in self.restarts])

    @property
    def n_converged(self) -> int:
        return sum(r.converged for r in self.restarts)

    @property
    def n_degenerate(self) -> int:
        return sum(r.degenerate for r in self.restarts)

    def as_dict(self) -> dict[str, Any]:
        ll = self.train_logliks
        return {
            "k": self.k,
            "n_params": self.n_params,
            "n_restarts": len(self.restarts),
            "n_converged": self.n_converged,
            "n_degenerate": self.n_degenerate,
            "train_loglik_best": float(ll.max()) if len(ll) else None,
            "train_loglik_mean": float(ll.mean()) if len(ll) else None,
            "train_loglik_std": float(ll.std()) if len(ll) else None,
            "train_loglik_min": float(ll.min()) if len(ll) else None,
            "val_loglik": self.val_loglik,
            "bic": self.bic,
            "aic": self.aic,
            "degenerate": self.degenerate,
        }


def fit_restarts(
    X_train: np.ndarray,
    k: int,
    *,
    n_restarts: int,
    n_iter: int,
    tol: float,
    covariance_type: str,
    seed_base: int,
) -> list[RestartResult]:
    """Fit ``k``-state Gaussian HMMs with ``n_restarts`` deterministic seeds.

    Seeds are derived from ``seed_base`` via
    :func:`prism.utils.seeding.derive_seed`, keyed on ``(k, attempt)``, so
    the whole sweep is bit-reproducible without depending on iteration order
    or any global RNG state. A restart that raises (hmmlearn can throw on a
    numerically degenerate covariance) is logged and skipped rather than
    aborting the whole sweep — one bad initialisation out of 20+ should not
    crash K selection.
    """
    from hmmlearn.hmm import GaussianHMM

    results: list[RestartResult] = []
    for attempt in range(n_restarts):
        seed = derive_seed(seed_base, "hmm_restart", k, attempt)
        model = GaussianHMM(
            n_components=k,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
        )
        try:
            model.fit(X_train)
            ll = float(model.score(X_train))
        except Exception as exc:  # pragma: no cover - depends on numerical luck
            _log.warning("K=%d attempt=%d failed to fit: %s", k, attempt, exc)
            continue
        if not np.isfinite(ll):
            _log.warning("K=%d attempt=%d produced non-finite log-likelihood", k, attempt)
            continue
        results.append(
            RestartResult(
                k=k,
                seed=seed,
                model=model,
                train_loglik=ll,
                converged=bool(model.monitor_.converged),
                n_iter=int(model.monitor_.iter),
                degenerate=bool(is_degenerate(model)),
            )
        )
    return results


def select_k(
    observations: Any,
    fit_start: Any,
    fit_end: Any,
    val_start: Any,
    val_end: Any,
    k_range: list[int],
    *,
    n_restarts: int,
    n_iter: int,
    tol: float,
    covariance_type: str,
    seed_base: int,
    min_expected_duration_days: float = 5.0,
    min_unconditional_prob: float = 0.02,
) -> dict[int, KResult]:
    """Fit every ``K`` in ``k_range`` with restarts, score each on validation.

    Parameters
    ----------
    observations
        A date-indexed ``pd.DataFrame`` spanning *at least* ``fit_start`` to
        ``val_end`` — not just the fit window. Spec §3.2's HMM schedule fits
        on Universe A's ``fit_early`` (1999-2006) and validates on Universe
        B's val split (2018); eleven years sit between them. Scoring 2018
        against ``startprob_`` directly would skip the transition-driven
        mixing that should happen across those eleven years, so validation is
        scored with :func:`~prism.models.hmm.filtered.score_causal_loglik`,
        which conditions on the model's own filtered pass through the *whole*
        prefix from ``observations``'s start through ``val_end`` — not on
        ``val_start..val_end`` in isolation.
    fit_start, fit_end
        The window EM actually fits on (``observations.loc[fit_start:fit_end]``
        becomes ``X_train``).
    val_start, val_end
        The window scored for selection (spec §8.4's primary criterion).

    For each ``K``: fit ``n_restarts`` times, prefer the highest-likelihood
    **non-degenerate** restart as that ``K``'s representative model (falling
    back to the overall best, flagged ``degenerate=True``, only if every
    restart was degenerate).
    """
    X_train = observations.loc[fit_start:fit_end].to_numpy(dtype="float64")
    d = X_train.shape[1]
    n = len(X_train)
    out: dict[int, KResult] = {}

    for k in k_range:
        restarts = fit_restarts(
            X_train,
            k,
            n_restarts=n_restarts,
            n_iter=n_iter,
            tol=tol,
            covariance_type=covariance_type,
            seed_base=seed_base,
        )
        if not restarts:
            _log.warning("K=%d: every restart failed; excluded from selection", k)
            continue

        non_degenerate = [r for r in restarts if not r.degenerate]
        pool = non_degenerate if non_degenerate else restarts
        best = max(pool, key=lambda r: r.train_loglik)

        val_loglik = score_causal_loglik(best.model, observations, val_start, val_end)

        n_params = n_free_params(k, d)
        bic = -2.0 * best.train_loglik + np.log(n) * n_params
        aic = -2.0 * best.train_loglik + 2.0 * n_params

        out[k] = KResult(
            k=k,
            n_params=n_params,
            restarts=restarts,
            best=best,
            val_loglik=val_loglik,
            bic=bic,
            aic=aic,
            degenerate=best.degenerate,
        )
        _log.info(
            "K=%d: %d/%d restarts converged, %d/%d degenerate, best train_ll=%.2f "
            "val_ll=%.2f bic=%.2f",
            k,
            out[k].n_converged,
            len(restarts),
            out[k].n_degenerate,
            len(restarts),
            best.train_loglik,
            val_loglik,
            bic,
        )

    return out


def select_best_k(results: dict[int, KResult], *, near_tie_margin: float) -> int:
    """Spec §8.4 (amended, D-025): highest validation log-likelihood, among
    non-degenerate ``K`` only, with near-ties broken by BIC.

    A ``K`` whose best restart is degenerate is **discarded outright** before
    any comparison — not merely deprioritised as the original "fall back to
    the full pool, flagged" rule did. A degenerate best restart means every
    restart EM found for that ``K`` has at least one state with an implausibly
    short expected duration or near-zero mass; there is no reading of "most
    likely on validation" that makes such a model trustworthy, so it is never
    a candidate, no matter how high its ``val_loglik``.

    Among the surviving, non-degenerate ``K`` values, the one with the
    highest ``val_loglik`` wins outright **unless** another non-degenerate
    ``K`` is within ``near_tie_margin`` log-likelihood points of it — raw
    validation log-likelihood is noisy enough at this sample size that a gap
    smaller than ``near_tie_margin`` is not a meaningful preference, and BIC
    (lower is better; it penalises the extra parameters a higher ``K`` adds)
    is the fairer tiebreaker among near-equally-likely candidates.

    Raises if every ``K`` in the sweep is degenerate: unlike the old
    fallback, there is no safe "least-bad" choice to make silently — that
    is itself the finding, and the caller (or a human) must decide what to
    do about the whole spec/``k_range`` pair, not this function.
    """
    if not results:
        raise ValueError("no K produced a usable fit; nothing to select from")
    non_degenerate = {k: r for k, r in results.items() if not r.degenerate}
    if not non_degenerate:
        raise RuntimeError(
            "every K in the sweep had a degenerate best restart; there is no "
            "non-degenerate candidate to select from. This is a finding about the "
            "observation spec itself, not a bug — investigate or drop it (spec "
            "§8.4/§13.1) rather than trusting any K produced here."
        )
    best_val = max(r.val_loglik for r in non_degenerate.values())
    near_tie = {
        k: r for k, r in non_degenerate.items() if best_val - r.val_loglik <= near_tie_margin
    }
    if len(near_tie) > 1:
        _log.info(
            "K values %s are within %.2f val_loglik points of the best (%.2f); "
            "breaking the tie by BIC (lower is better)",
            sorted(near_tie), near_tie_margin, best_val,
        )
    return min(near_tie, key=lambda k: near_tie[k].bic)


@dataclass(frozen=True)
class FitSweepResult:
    """The full K-sweep outcome: every K's result, and the selected one."""

    results: dict[int, KResult]
    selected_k: int
    observation_columns: tuple[str, ...]

    @property
    def selected(self) -> KResult:
        return self.results[self.selected_k]

    def as_table(self) -> list[dict[str, Any]]:
        return [self.results[k].as_dict() for k in sorted(self.results)]


def fit_and_select(
    observations: Any,
    fit_start: Any,
    fit_end: Any,
    val_start: Any,
    val_end: Any,
    cfg: Any,
) -> FitSweepResult:
    """Convenience wrapper reading sweep parameters from ``cfg.hmm``.

    ``observations`` is a date-indexed frame spanning at least ``fit_start``
    to ``val_end`` (see :func:`select_k`); its column order is recorded on
    the result as ``observation_columns``, so a report can label dimension 0
    as "the return" without re-deriving which specification (H1/H2) produced
    it.
    """
    fit_cfg = cfg.hmm.fit
    sel_cfg = cfg.hmm.selection
    results = select_k(
        observations,
        fit_start,
        fit_end,
        val_start,
        val_end,
        fit_cfg.k_range,
        n_restarts=fit_cfg.n_restarts,
        n_iter=fit_cfg.n_iter,
        tol=fit_cfg.tol,
        covariance_type=fit_cfg.covariance_type,
        seed_base=cfg.data.seeds.master,
        min_expected_duration_days=sel_cfg.min_expected_duration_days,
        min_unconditional_prob=sel_cfg.min_unconditional_prob,
    )
    selected_k = select_best_k(results, near_tie_margin=sel_cfg.near_tie_margin)
    return FitSweepResult(
        results=results,
        selected_k=selected_k,
        observation_columns=tuple(str(c) for c in observations.columns),
    )
