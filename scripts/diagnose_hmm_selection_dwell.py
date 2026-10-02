#!/usr/bin/env python
"""One-off diagnostic: decoded dwell times and selection table for EVERY
(spec, K) fitted during Step 2's K-selection sweep. Spec §8.4.

    python scripts/diagnose_hmm_selection_dwell.py

Not a build step and not part of the Phase A pipeline — a follow-up to the
earlier HMM degeneracy diagnostics. Diagnostics only: refits exactly what
``scripts/02_fit_hmm.py``'s K-selection already fit (same observations, same
fit/val windows, same ``seed_base`` via :func:`prism.models.hmm.fit.select_k`
itself — not a reimplementation), no threshold or hyperparameter is changed.

For every specification (H1, H2) and every K in ``cfg.hmm.fit.k_range``:

* the selected (best) restart's model, canonically relabelled, is run
  through the causal :func:`~prism.models.hmm.filtered.filtered_posteriors`
  from the observation frame's own start through ``val_end`` (exactly the
  span :func:`~prism.models.hmm.filtered.score_causal_loglik` scores), hard-
  decoded by per-day argmax, and run-length encoded — the same method the
  earlier H2 walk-forward dwell-time analysis used, applied here to the
  one-shot K-selection fit (there is no walk-forward at this stage).
* every restart fit for that (spec, K) is canonically relabelled and checked
  with :func:`~prism.models.hmm.fit.is_degenerate`, and the per-state
  expected-duration failure rate is reported across restarts.

Writes ``reports/tables/hmm_selection_dwell.csv`` (one row per spec/K/state)
and prints the Step 2 selection table (val log-likelihood per K per spec,
and which spec/K won overall) to stdout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prism.config import load_config  # noqa: E402
from prism.models.hmm.fit import is_degenerate, select_k  # noqa: E402
from prism.models.hmm.filtered import filtered_posteriors  # noqa: E402
from prism.models.hmm.labeling import canonicalize  # noqa: E402
from prism.utils.logging import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)


def _dwell_stats(decoded: np.ndarray) -> pd.DataFrame:
    """Run-length encode a decoded state path; return per-state quartiles."""
    runs = []
    cur, length = decoded[0], 1
    for s in decoded[1:]:
        if s == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = s, 1
    runs.append((cur, length))
    df = pd.DataFrame(runs, columns=["state", "run_length"])
    return df


def build_observation_frame(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise KeyError(f"observation columns absent from features: {missing}")
    return features[columns].dropna()


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    log = configure_logging()

    features_a = pd.read_parquet(cfg.path("processed") / "A_features.parquet")
    fit_start, fit_end = cfg.data.fit_early("A")
    val_start, val_end = cfg.data.split("val")

    fit_cfg, sel_cfg = cfg.hmm.fit, cfg.hmm.selection
    rows: list[dict[str, object]] = []
    selection_table: list[dict[str, object]] = []

    for spec_name, spec in cfg.hmm.specifications.items():
        obs = build_observation_frame(features_a, list(spec.observations))
        log.info("=== %s: observations=%s ===", spec_name, list(spec.observations))

        results = select_k(
            obs, fit_start, fit_end, val_start, val_end, fit_cfg.k_range,
            n_restarts=fit_cfg.n_restarts, n_iter=fit_cfg.n_iter, tol=fit_cfg.tol,
            covariance_type=fit_cfg.covariance_type, seed_base=cfg.data.seeds.master,
            min_expected_duration_days=sel_cfg.min_expected_duration_days,
            min_unconditional_prob=sel_cfg.min_unconditional_prob,
        )

        for k, kresult in sorted(results.items()):
            selection_table.append(
                {
                    "spec": spec_name, "k": k, "val_loglik": kresult.val_loglik,
                    "bic": kresult.bic, "aic": kresult.aic,
                    "n_degenerate": kresult.n_degenerate, "n_restarts": len(kresult.restarts),
                    "best_degenerate": kresult.degenerate,
                }
            )

            # --- per-state duration fail rate across every restart -------- #
            # Canonicalizing re-sets `covars_` through hmmlearn's own setter,
            # which validates positive-definiteness; a restart whose EM run
            # converged to a numerically invalid covariance (observed at a
            # handful of (spec, K) combinations below) raises there. That is
            # itself a sign of a pathological restart, not a bug in this
            # diagnostic — skipped and counted separately rather than
            # crashing the sweep or silently mislabeling it as non-degenerate.
            per_state_fail_count = np.zeros(k, dtype=int)
            n_usable = 0
            n_uncanonicalizable = 0
            for restart in kresult.restarts:
                try:
                    relabeled, _, _ = canonicalize(
                        restart.model, sort_by="state_return_std", return_dim=0, ascending=True
                    )
                except ValueError as exc:
                    n_uncanonicalizable += 1
                    log.warning(
                        "%s K=%d: restart seed=%d not canonicalizable (%s); excluded "
                        "from the per-state fail-rate denominator",
                        spec_name, k, restart.seed, exc,
                    )
                    continue
                check = is_degenerate(relabeled)
                per_state_fail_count += (
                    check.expected_duration_days < check.min_expected_duration_days
                ).astype(int)
                n_usable += 1

            # --- decoded dwell times from the SELECTED (best) restart ----- #
            try:
                best_relabeled, _, _ = canonicalize(
                    kresult.best.model, sort_by="state_return_std", return_dim=0, ascending=True
                )
            except ValueError as exc:
                log.warning(
                    "%s K=%d: SELECTED (best) restart not canonicalizable (%s); "
                    "dwell times not computed for this (spec, K)",
                    spec_name, k, exc,
                )
                for state in range(k):
                    rows.append(
                        {
                            "spec": spec_name, "k": k, "state": state, "n_runs": None,
                            "median_dwell_days": None, "q1_dwell_days": None, "q3_dwell_days": None,
                            "pooled_median_dwell_days": None,
                            "duration_fail_rate": per_state_fail_count[state] / n_usable
                            if n_usable else None,
                            "n_restarts_checked": n_usable,
                            "n_restarts_uncanonicalizable": n_uncanonicalizable,
                            "best_restart_degenerate": kresult.best.degenerate,
                        }
                    )
                continue

            through_val = obs.loc[obs.index <= val_end]
            decoded = filtered_posteriors(
                best_relabeled, through_val.to_numpy(dtype="float64")
            ).posteriors.argmax(axis=1)
            dwell = _dwell_stats(decoded)
            quartiles = dwell.groupby("state")["run_length"].quantile([0.25, 0.5, 0.75]).unstack()
            pooled_median = float(dwell["run_length"].median())

            for state in range(k):
                if state in quartiles.index:
                    q1, med, q3 = quartiles.loc[state, [0.25, 0.5, 0.75]]
                    n_runs = int((dwell["state"] == state).sum())
                else:
                    q1 = med = q3 = float("nan")
                    n_runs = 0
                rows.append(
                    {
                        "spec": spec_name, "k": k, "state": state,
                        "n_runs": n_runs,
                        "median_dwell_days": med, "q1_dwell_days": q1, "q3_dwell_days": q3,
                        "pooled_median_dwell_days": pooled_median,
                        "duration_fail_rate": per_state_fail_count[state] / n_usable
                        if n_usable else None,
                        "n_restarts_checked": n_usable,
                        "n_restarts_uncanonicalizable": n_uncanonicalizable,
                        "best_restart_degenerate": kresult.best.degenerate,
                    }
                )
            log.info(
                "%s K=%d: pooled median dwell=%.1fd, val_ll=%.2f, best_degenerate=%s, "
                "per-state fail rate=%s%s",
                spec_name, k, pooled_median, kresult.val_loglik, kresult.best.degenerate,
                (per_state_fail_count / n_usable).round(2).tolist() if n_usable else "n/a",
                f" ({n_uncanonicalizable} restarts excluded)" if n_uncanonicalizable else "",
            )

    out_path = cfg.path("tables") / "hmm_selection_dwell.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %d rows to %s", len(rows), out_path)

    print("\n=== Step 2 selection table (val log-likelihood, primary criterion) ===")
    print(f"{'spec':<6}{'K':<4}{'val_loglik':>14}{'bic':>14}{'aic':>14}{'degenerate':>12}")
    by_spec: dict[str, list[dict]] = {}
    for row in selection_table:
        by_spec.setdefault(row["spec"], []).append(row)
    overall_best = max(
        (max(rs, key=lambda r: r["val_loglik"]) for rs in by_spec.values()),
        key=lambda r: r["val_loglik"],
    )
    for spec_name, rs in by_spec.items():
        best_row = max(rs, key=lambda r: r["val_loglik"])
        for row in sorted(rs, key=lambda r: r["k"]):
            marker = " <- spec-selected" if row is best_row else ""
            overall = " <- OVERALL WINNER" if row is overall_best else ""
            print(
                f"{row['spec']:<6}{row['k']:<4}{row['val_loglik']:>14.2f}"
                f"{row['bic']:>14.2f}{row['aic']:>14.2f}{str(row['best_degenerate']):>12}"
                f"{marker}{overall}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
