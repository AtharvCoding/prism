"""Expanding-window refit with carried filter state. Spec §8.6.

The reference rolling loop (defect B5) had five independent problems in one
~25-line cell: label switching across months (no canonical relabeling),
duplicate dates from an inclusive slice (1,557 rows for 1,509 trading days),
the filter restarted from ``startprob_`` at the start of every month's apply
slice (discarding the fit window's own warm-up), the refit date landing
inside its own fit window, and a scaler mutated in place across folds. Every
one of those is a named requirement below, with the test or assertion that
would catch its regression.

**"Restarted each month", precisely.** The defect was calling
``predict_proba`` (or an equivalent) on *just* that month's apply slice in
isolation — which makes ``hmmlearn`` implicitly treat the first day of the
month as if trading had just begun, using ``startprob_`` rather than
whatever the chain's actual belief state was at the end of the prior month.
The fix here is not to carry a raw alpha vector *across a change of fitted
parameters* (which would be mixing two different models' beliefs — not
meaningful, since alpha is defined relative to one specific parameterisation)
but to run the filtered recursion **fresh, each fold, over that fold's own
fit window followed immediately by its own apply window**, under that
fold's own (freshly fit, canonically relabelled) model, and keep only the
apply-window rows. The fit window supplies the warm-up the reference loop
skipped; nothing is restarted at the apply boundary.

**Embargo gap, addressed explicitly.** :func:`~prism.splits.expanding_folds`
embeds spec §6.1's embargo between each fold's fit and apply windows — the
same embargo every other walk-forward process in this project respects. For
an HMM this is not a leakage concern in the §6.1 sense (the embargo exists
for forward-looking *targets and rewards* overlapping a boundary; HMM
observations are same-day, zero-horizon realised values, "not leakage" by
§6.1's own definition of what a lookback may legitimately read). Using the
same embargo-respecting folds here anyway is a consistency choice, not a
leakage fix: it reuses already-tested machinery rather than inventing a
second fold scheme, and the filtered recursion within a fold simply does not
see the embargoed calendar days — equivalent to treating them as absent,
which is a minor approximation, never a leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from prism.features.scaling import FeatureScaler
from prism.models.hmm.fit import fit_restarts
from prism.models.hmm.filtered import filtered_posteriors
from prism.models.hmm.labeling import canonicalize
from prism.splits import WalkForwardFold, expanding_folds
from prism.utils.logging import get_logger
from prism.utils.seeding import derive_seed

__all__ = ["WalkforwardFoldOutput", "WalkforwardResult", "hmm_walkforward"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class WalkforwardFoldOutput:
    """One fold's fitted-and-relabelled model, plus how it got there."""

    fold: WalkForwardFold
    model: Any
    order: np.ndarray
    scaler: FeatureScaler
    n_restarts_used: int
    n_restarts_degenerate: int
    train_loglik: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.fold.as_dict(),
            "n_restarts_used": self.n_restarts_used,
            "n_restarts_degenerate": self.n_restarts_degenerate,
            "train_loglik": self.train_loglik,
        }


@dataclass(frozen=True)
class WalkforwardResult:
    """The merged, de-duplicated, calendar-aligned walk-forward output."""

    folds: list[WalkforwardFoldOutput]
    posteriors: pd.DataFrame
    per_step_loglik: pd.Series
    #: Spec §10's O1 (the leaky oracle) only — each fold's SMOOTHED
    #: (forward-backward, ``predict_proba``) posteriors, apply-window rows
    #: kept. ``None`` unless ``also_smoothed=True`` was passed. Leaky *within*
    #: a fold (it sees that fold's own apply-window future), never across
    #: folds. Never fed into anything but the O1 diagnostic — see
    #: :mod:`prism.state`.
    smoothed_posteriors: pd.DataFrame | None = None

    def as_table(self) -> list[dict[str, Any]]:
        return [f.as_dict() for f in self.folds]


def hmm_walkforward(
    observations: pd.DataFrame,
    *,
    k: int,
    fit_start: str | pd.Timestamp,
    first_apply_start: str | pd.Timestamp,
    apply_end: str | pd.Timestamp,
    cadence: str,
    embargo_days: int,
    covariance_type: str,
    n_restarts: int,
    n_iter: int,
    tol: float,
    seed_base: int,
    min_expected_duration_days: float = 5.0,
    min_unconditional_prob: float = 0.02,
    also_smoothed: bool = False,
    max_degenerate_fold_fraction: float = 0.10,
    exchange: str = "NYSE",
) -> WalkforwardResult:
    """Expanding-window HMM refit. Spec §8.6.

    Parameters
    ----------
    observations
        Date-indexed, causal, scale-free-ready observation frame (raw units
        — NOT pre-scaled: each fold fits its own :class:`FeatureScaler`,
        spec §8.6's "scalers are refit per fold on fold data only, never
        mutated in place on a shared object").
    k
        The number of states, already selected (spec §8.4's sweep is a
        separate, one-time step; this function does not re-select K).
    fit_start, first_apply_start, apply_end
        Passed straight to :func:`prism.splits.expanding_folds`.
    max_degenerate_fold_fraction
        Spec §8.6 amendment, D-026: raise once more than this fraction of
        folds come back with EVERY restart degenerate. A handful of such
        folds is tolerable noise; a spec/K pair that cannot produce a single
        clean restart across a large share of the walk-forward is not
        producing usable regimes often enough to trust, and the posteriors
        this function would otherwise still happily return are not a safe
        basis for anything downstream.

    Each fold: fit a new :class:`FeatureScaler` on the fold's own fit window;
    fit ``n_restarts`` Gaussian HMMs on the scaled fit window and keep the
    best non-degenerate one (or the overall best, flagged, if every restart
    is degenerate); canonically relabel it
    (:func:`~prism.models.hmm.labeling.canonicalize`); run
    :func:`~prism.models.hmm.filtered.filtered_posteriors` fresh over the
    fold's scaled fit window followed by its scaled apply window, and keep
    only the apply-window rows.

    Returns a single result whose ``posteriors`` and ``per_step_loglik`` are
    the concatenation of every fold's apply-window output, asserted
    de-duplicated and calendar-ordered before being returned — the property
    defect B5 (1,557 rows for 1,509 trading days) violated.
    """
    folds = expanding_folds(
        fit_start, first_apply_start, apply_end, cadence,
        embargo_days=embargo_days, exchange=exchange,
    )

    fold_outputs: list[WalkforwardFoldOutput] = []
    posterior_chunks: list[pd.DataFrame] = []
    loglik_chunks: list[pd.Series] = []
    smoothed_chunks: list[pd.DataFrame] = []
    columns = [f"state_{i}" for i in range(k)]
    n_all_degenerate_folds = 0

    for fold in folds:
        fit_raw = observations.loc[fold.fit_start : fold.fit_end]
        apply_raw = observations.loc[fold.apply_start : fold.apply_end]
        if fit_raw.empty or apply_raw.empty:
            raise ValueError(
                f"fold {fold.index}: empty fit or apply slice "
                f"({len(fit_raw)}, {len(apply_raw)} rows) — observations do not "
                "cover this fold's dates"
            )

        scaler = FeatureScaler().fit(fit_raw, scope=f"hmm walkforward fold {fold.index}")
        fit_scaled = scaler.transform(fit_raw)
        apply_scaled = scaler.transform(apply_raw)

        fold_seed = derive_seed(seed_base, "hmm_walkforward", fold.index)
        restarts = fit_restarts(
            fit_scaled.to_numpy(dtype="float64"),
            k,
            n_restarts=n_restarts,
            n_iter=n_iter,
            tol=tol,
            covariance_type=covariance_type,
            seed_base=fold_seed,
        )
        if not restarts:
            raise RuntimeError(f"fold {fold.index}: every restart failed to fit")

        non_degenerate = [r for r in restarts if not r.degenerate]
        pool = non_degenerate if non_degenerate else restarts
        best = max(pool, key=lambda r: r.train_loglik)
        if not non_degenerate:
            n_all_degenerate_folds += 1
            _log.warning(
                "fold %d (%s..%s): every restart degenerate; using the least-bad",
                fold.index, fold.apply_start.date(), fold.apply_end.date(),
            )

        relabeled, order, _ = canonicalize(
            best.model,
            sort_by="state_return_std",
            return_dim=0,
            ascending=True,
        )

        combined = pd.concat([fit_scaled, apply_scaled])
        if not combined.index.is_monotonic_increasing or not combined.index.is_unique:
            raise ValueError(f"fold {fold.index}: fit/apply windows are not disjoint or sorted")
        result = filtered_posteriors(relabeled, combined.to_numpy(dtype="float64"))

        n_apply = len(apply_scaled)
        apply_posteriors = result.posteriors[-n_apply:]
        apply_loglik = result.per_step_loglik[-n_apply:]

        posterior_chunks.append(
            pd.DataFrame(apply_posteriors, index=apply_scaled.index, columns=columns)
        )
        loglik_chunks.append(pd.Series(apply_loglik, index=apply_scaled.index, name="loglik"))

        if also_smoothed:
            # Spec §10 O1: the deliberately leaky oracle. `predict_proba` is
            # hmmlearn's own forward-BACKWARD (smoothed) posterior — exactly
            # what defect B1 used and §8.3 forbids for V3 — reused here on
            # purpose, on the SAME already-fitted fold model, so no second fit
            # is needed. Leaky only within this fold's fit+apply window, never
            # across folds (a bound, not the maximally leaky full-sample case).
            smoothed_full = relabeled.predict_proba(combined.to_numpy(dtype="float64"))
            smoothed_chunks.append(
                pd.DataFrame(smoothed_full[-n_apply:], index=apply_scaled.index, columns=columns)
            )

        fold_outputs.append(
            WalkforwardFoldOutput(
                fold=fold,
                model=relabeled,
                order=order,
                scaler=scaler,
                n_restarts_used=len(restarts),
                n_restarts_degenerate=sum(r.degenerate for r in restarts),
                train_loglik=best.train_loglik,
            )
        )
        _log.info(
            "fold %d: fit %s..%s (%d sessions), apply %s..%s (%d sessions), "
            "%d/%d restarts degenerate",
            fold.index,
            fold.fit_start.date(), fold.fit_end.date(), len(fit_raw),
            fold.apply_start.date(), fold.apply_end.date(), n_apply,
            fold_outputs[-1].n_restarts_degenerate, len(restarts),
        )

    degenerate_fold_fraction = n_all_degenerate_folds / len(folds)
    if degenerate_fold_fraction > max_degenerate_fold_fraction:
        raise RuntimeError(
            f"{n_all_degenerate_folds}/{len(folds)} folds ({degenerate_fold_fraction:.1%}) "
            f"had every restart degenerate, exceeding max_degenerate_fold_fraction="
            f"{max_degenerate_fold_fraction:.0%} (spec §8.6 amendment, D-026). This spec/K "
            "is not producing usable regimes across a meaningful share of the walk-forward "
            "— redesign or drop it (spec §8.4/§13.1) rather than trusting the posteriors "
            "this run would otherwise return."
        )

    posteriors = pd.concat(posterior_chunks).sort_index()
    per_step_loglik = pd.concat(loglik_chunks).sort_index()

    if not posteriors.index.is_unique:
        dupes = posteriors.index[posteriors.index.duplicated()]
        raise AssertionError(
            f"walk-forward produced {len(dupes)} duplicate dates, first "
            f"{dupes[0].date()} — exactly defect B5 (1,557 rows for 1,509 trading "
            "days). Folds must tile the apply period without overlap."
        )
    if not posteriors.index.is_monotonic_increasing:
        raise AssertionError("walk-forward posteriors are not calendar-ordered")

    smoothed_posteriors: pd.DataFrame | None = None
    if also_smoothed:
        smoothed_posteriors = pd.concat(smoothed_chunks).sort_index()
        if not smoothed_posteriors.index.equals(posteriors.index):
            raise AssertionError("smoothed and filtered posteriors disagree on their index")

    return WalkforwardResult(
        folds=fold_outputs,
        posteriors=posteriors,
        per_step_loglik=per_step_loglik,
        smoothed_posteriors=smoothed_posteriors,
    )
