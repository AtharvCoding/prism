#!/usr/bin/env python
"""One-off diagnostic: per-state degeneracy detail for HMM walk-forward folds.

    python scripts/diagnose_hmm_degeneracy.py

Not a build step and not part of the Phase A pipeline — a follow-up to the
2022-2023 "every restart degenerate" finding surfaced while verifying step
3b's O1 oracle. Refits folds 150-203 of the winning HMM specification (same
fit windows, same derived seeds :func:`prism.models.hmm.walkforward.hmm_walkforward`
would use, so this reproduces exactly what that run already did — no
threshold or hyperparameter is changed here) and, for every fold and every
restart, records each state's expected duration and stationary probability
against :mod:`prism.models.hmm.fit`'s (unchanged) degeneracy thresholds.

Writes ``reports/tables/hmm_degeneracy_folds_150_203.csv`` and separately
walks backward from fold 180 to report the last fold with at least one clean
(non-degenerate) restart.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from prism.config import load_config  # noqa: E402
from prism.features.scaling import FeatureScaler  # noqa: E402
from prism.models.hmm.fit import fit_restarts, is_degenerate  # noqa: E402
from prism.splits import build_split_plan, expanding_folds  # noqa: E402
from prism.utils.logging import configure_logging, get_logger  # noqa: E402
from prism.utils.seeding import derive_seed  # noqa: E402

_log = get_logger(__name__)

FOLD_RANGE = range(150, 204)  # folds 150-203 inclusive
BACKWARD_FROM = 180


def _folds_and_observations(cfg):
    plan = build_split_plan(cfg)
    fit_start = cfg.data.fit_early("A")[0]
    first_apply_start = cfg.data.fit_early("A")[1] + pd.Timedelta(days=1)
    apply_end = plan["test"].declared_end
    folds = expanding_folds(
        fit_start, first_apply_start, apply_end,
        cfg.hmm.walkforward.refit_cadence,
        embargo_days=cfg.data.splits.embargo_days,
    )

    summary = json.loads(
        (cfg.path("processed") / "hmm_summary.json").read_text(encoding="utf-8")
    )
    spec_name = summary["walkforward"]["specification"]
    k = summary["walkforward"]["k"]
    obs_cols = list(cfg.hmm.specifications[spec_name].observations)
    features_a = pd.read_parquet(cfg.path("processed") / "A_features.parquet")
    obs = features_a[obs_cols].dropna()
    return folds, obs, k


def _fit_fold(cfg, obs, k, fold):
    """Refit one fold exactly as hmm_walkforward would. Returns the restart list."""
    fit_raw = obs.loc[fold.fit_start : fold.fit_end]
    scaler = FeatureScaler().fit(fit_raw, scope=f"diagnose fold {fold.index}")
    fit_scaled = scaler.transform(fit_raw)
    fold_seed = derive_seed(cfg.data.seeds.master, "hmm_walkforward", fold.index)
    return fit_restarts(
        fit_scaled.to_numpy(dtype="float64"), k,
        n_restarts=cfg.hmm.walkforward.n_restarts,
        n_iter=cfg.hmm.fit.n_iter,
        tol=cfg.hmm.fit.tol,
        covariance_type=cfg.hmm.fit.covariance_type,
        seed_base=fold_seed,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    log = configure_logging()
    folds, obs, k = _folds_and_observations(cfg)
    by_index = {f.index: f for f in folds}

    # --- CSV: every (fold, restart, state, bound) in [150, 203] ----------- #
    rows: list[dict[str, object]] = []
    fold_cache: dict[int, list] = {}
    for idx in FOLD_RANGE:
        fold = by_index[idx]
        restarts = _fit_fold(cfg, obs, k, fold)
        fold_cache[idx] = restarts
        for restart in restarts:
            check = is_degenerate(restart.model)
            for row in check.failing_states():
                rows.append(
                    {
                        "fold_index": idx,
                        "apply_start": fold.apply_start.date(),
                        "apply_end": fold.apply_end.date(),
                        "restart_seed": restart.seed,
                        "restart_degenerate": restart.degenerate,
                        **row,
                    }
                )
        n_clean = sum(not r.degenerate for r in restarts)
        log.info("fold %d (%s..%s): %d/%d restarts clean", idx, fold.apply_start.date(),
                  fold.apply_end.date(), n_clean, len(restarts))

    out_path = cfg.path("tables") / "hmm_degeneracy_folds_150_203.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %d rows to %s", len(rows), out_path)

    # --- backward search from fold 180 for the last fold with >=1 clean --- #
    idx = BACKWARD_FROM
    last_clean: int | None = None
    while idx >= 0:
        restarts = fold_cache.get(idx)
        if restarts is None:
            fold = by_index[idx]
            restarts = _fit_fold(cfg, obs, k, fold)
            fold_cache[idx] = restarts
        if any(not r.degenerate for r in restarts):
            last_clean = idx
            break
        idx -= 1

    if last_clean is None:
        log.info("no fold at or before %d had any clean restart", BACKWARD_FROM)
    else:
        fold = by_index[last_clean]
        log.info(
            "last fold at or before %d with >=1 clean restart: fold %d (apply %s..%s)",
            BACKWARD_FROM, last_clean, fold.apply_start.date(), fold.apply_end.date(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
