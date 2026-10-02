#!/usr/bin/env python
"""Fit the HMM on Universe A, select K, walk-forward, evaluate. Spec §8, §16 step 2.

    python scripts/02_fit_hmm.py                     # K-selection on the pinned spec,
                                                       # full monthly walk-forward, evaluation
    python scripts/02_fit_hmm.py --skip-walkforward   # K-selection and reporting only

Reads ``data/processed/{A,B}_features.parquet`` (Step 1's output; never the
network). Operates on a single, config-pinned specification
(``cfg.hmm.fit.specification``, currently H1 — returns only):

1. Builds the observation frame from Universe A's own columns.
2. Sweeps ``K`` with restarts (spec §8.4), fit on ``fit_early``
   (1999-2006), scored causally on Universe B's val split (2018). A ``K``
   whose best restart is degenerate is discarded outright; among the
   survivors, the highest validation log-likelihood wins, with near-ties
   (within ``cfg.hmm.selection.near_tie_margin`` points) broken by BIC
   (spec §8.4 amendment, D-025).

**Why only one spec.** An earlier version of this script fit multiple
specifications (H1, H2) and picked whichever had the higher validation
log-likelihood. That comparison is invalid: a multivariate Gaussian density
(H2, 4 observation dims) sits on a structurally different scale than a
univariate one (H1, 1 dim) on the same data, so the higher-dimensional spec
wins regardless of regime quality — and H2 was, independently, degenerate at
every ``K`` in its own sweep. See DECISIONS.md D-024. Comparing specs of
different observation dimensionality is no longer something this script does
at all; ``cfg.hmm.fit.specification`` is a pinned, human decision, not an
automatic winner.

The pinned spec's selected ``K`` is carried forward: a full expanding-window,
monthly, canonically-relabelled walk-forward (spec §8.6) from ``fit_early``'s
start through the end of the test split, producing the regime posterior
series step 3b's state assembly will consume. The walk-forward itself now
raises if more than ``cfg.hmm.walkforward.max_degenerate_fold_fraction`` of
its folds come back with every restart degenerate (spec §8.6 amendment,
D-026) — a spec/K pair that cannot produce a single clean restart across a
large share of the walk-forward is not safe to trust, and the posteriors
this script would otherwise still happily persist are not a sound basis for
anything downstream.

Evaluation (spec §8.7): state characterisation, posterior quality, and
detection against both NBER recessions and SPY drawdown episodes, compared
against a two-state VIX-threshold baseline on the same fold schedule.

The §3.2 robustness check — refit on Universe B's training window alone
(2007-2017) and compare — runs as a single K-selection sweep (not a second
full walk-forward; the spec's "run the entire Tier 1 ablation a second time"
instruction is about step 4a's probes, not a duplicate multi-decade HMM
refit here) and its state characterisation is reported alongside the primary
model's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prism.config import Config, load_config  # noqa: E402
from prism.models.hmm.evaluate import (  # noqa: E402
    characterise_states,
    compare_detection,
    drawdown_bear_episodes,
    load_nber_recessions,
    posterior_quality,
)
from prism.models.hmm.fit import FitSweepResult, fit_and_select  # noqa: E402
from prism.models.hmm.walkforward import WalkforwardResult, hmm_walkforward  # noqa: E402
from prism.models.baselines.threshold_regime import threshold_regime_walkforward  # noqa: E402
from prism.splits import build_split_plan  # noqa: E402
from prism.utils.hashing import hash_object, make_run_id, write_manifest  # noqa: E402
from prism.utils.logging import configure_logging, get_logger  # noqa: E402
from prism.utils.seeding import seed_everything  # noqa: E402

_log = get_logger(__name__)


def _config_hash(cfg: Config) -> str:
    return hash_object(cfg.model_dump(mode="json", exclude={"root"}))


def build_observation_frame(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise KeyError(f"observation columns absent from features: {missing}")
    obs = features[columns].dropna()
    if obs.empty:
        raise ValueError("observation frame is empty after dropping NaN rows")
    return obs


def run_k_sweep(cfg: Config, observations: pd.DataFrame, spec_name: str) -> FitSweepResult:
    fit_start, fit_end = cfg.data.fit_early("A")
    val_start, val_end = cfg.data.split("val")
    _log.info(
        "[%s] K-sweep: fit %s..%s, val %s..%s, observations=%s",
        spec_name, fit_start.date(), fit_end.date(), val_start.date(), val_end.date(),
        list(observations.columns),
    )
    t0 = time.time()
    result = fit_and_select(observations, fit_start, fit_end, val_start, val_end, cfg)
    _log.info(
        "[%s] K-sweep done in %.1fs. Selected K=%d (val_ll=%.2f)",
        spec_name, time.time() - t0, result.selected_k, result.selected.val_loglik,
    )
    for k in sorted(result.results):
        r = result.results[k]
        _log.info(
            "  K=%d: best_train_ll=%.2f (mean=%.2f std=%.4f over %d restarts, "
            "%d degenerate) val_ll=%.2f bic=%.2f aic=%.2f",
            k, r.best.train_loglik, r.train_logliks.mean(), r.train_logliks.std(),
            len(r.restarts), r.n_degenerate, r.val_loglik, r.bic, r.aic,
        )
    return result


def dataframe_to_markdown(df: pd.DataFrame, *, include_index: bool = True) -> str:
    """A minimal markdown-table renderer, so the project does not need the
    optional ``tabulate`` dependency just for report formatting."""
    frame = df.reset_index() if include_index else df
    headers = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_k_sweep_report(path: Path, name: str, result: FitSweepResult) -> None:
    lines = [f"# HMM K-selection — {name}", "", f"Observations: `{result.observation_columns}`", ""]
    lines += ["| K | n_params | best_train_ll | mean_train_ll | std_train_ll | "
              "n_degenerate/n_restarts | val_ll (primary) | BIC | AIC |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in sorted(result.results):
        r = result.results[k]
        marker = " **<- selected**" if k == result.selected_k else ""
        lines.append(
            f"| {k}{marker} | {r.n_params} | {r.best.train_loglik:.2f} | "
            f"{r.train_logliks.mean():.2f} | {r.train_logliks.std():.4f} | "
            f"{r.n_degenerate}/{len(r.restarts)} | {r.val_loglik:.2f} | {r.bic:.2f} | {r.aic:.2f} |"
        )
    lines.append("")
    lines.append(f"**Selected K = {result.selected_k}** on primary criterion: validation log-likelihood.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_report(
    path: Path,
    spec_name: str,
    k: int,
    wf: WalkforwardResult,
    characterisation,
    quality,
    nber_comparison,
    drawdown_comparison,
) -> None:
    lines = [f"# HMM evaluation — {spec_name} (K={k})", "", "## State characterisation", ""]
    lines.append(dataframe_to_markdown(characterisation.table.round(4)))
    lines += ["", "## Empirical transition matrix (walk-forward hard assignment)", ""]
    lines.append(dataframe_to_markdown(characterisation.transition_matrix.round(4)))
    lines += [
        "", "## Posterior quality", "",
        f"- Mean entropy: {quality.mean_entropy:.4f} "
        f"({'**SATURATED**' if quality.saturated else 'ok'})",
        f"- Median max posterior: {quality.median_max_posterior:.4f}",
        f"- Day-to-day flip rate: {quality.flip_rate:.4f}",
        "",
        "## Detection vs NBER recessions",
        "",
    ]
    for comparison, label in ((nber_comparison, "NBER"), (drawdown_comparison, "Drawdown (>=20%)")):
        lines += [f"### {label} episodes", ""]
        if comparison is None:
            lines.append("_No episodes fell inside the walk-forward apply window._")
            continue
        lines.append(dataframe_to_markdown(comparison.summary_table(), include_index=False))
        lines += [
            "",
            f"- HMM false-alarm rate: {comparison.hmm_false_alarm_rate:.4f}",
            f"- Baseline false-alarm rate: {comparison.baseline_false_alarm_rate:.4f}",
            f"- HMM beats baseline on detection lag (every episode, strictly lower mean lag): "
            f"**{comparison.hmm_wins_on_detection_lag()}**",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--skip-walkforward", action="store_true",
        help="run only the K-selection sweeps and reports, skip the (slow) walk-forward",
    )
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="skip the Universe-B-only robustness refit",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = configure_logging()
    seeds = seed_everything(cfg.data.seeds.master)

    features_a = pd.read_parquet(cfg.path("processed") / "A_features.parquet")
    features_b = pd.read_parquet(cfg.path("processed") / "B_features.parquet")

    # --- K-selection: the pinned specification, on Universe A ----------- #
    # Spec §8.2 amendment, D-024: no cross-spec comparison — see module
    # docstring. `winner`/`winner_k` names are kept (rather than renamed to
    # e.g. `spec_name`) because every downstream consumer of
    # ``hmm_summary.json`` (notably ``scripts/03b_build_states.py``'s O1
    # oracle rebuild) reads ``summary["walkforward"]["specification"]`` /
    # ``["k"]`` and nothing about that contract needs to change.
    winner = cfg.hmm.fit.specification
    spec = cfg.hmm.specifications[winner]
    obs = build_observation_frame(features_a, list(spec.observations))
    sweep_result = run_k_sweep(cfg, obs, winner)
    write_k_sweep_report(cfg.path("tables") / f"hmm_k_selection_{winner}.md", winner, sweep_result)

    winner_k = sweep_result.selected_k
    log.info("Specification: %s (pinned). Selected K=%d (val_ll=%.2f)",
              winner, winner_k, sweep_result.selected.val_loglik)

    summary: dict[str, object] = {
        "k_selection": {
            winner: {
                "selected_k": sweep_result.selected_k,
                "val_loglik": sweep_result.selected.val_loglik,
                "observations": list(sweep_result.observation_columns),
            }
        },
        "winner": winner,
        "winner_k": winner_k,
    }

    if args.skip_walkforward:
        log.info("--skip-walkforward: stopping after K-selection.")
    else:
        # --- full walk-forward for the winning specification ------------- #
        plan = build_split_plan(cfg)
        winner_obs_cols = list(cfg.hmm.specifications[winner].observations)
        obs_full = build_observation_frame(features_a, winner_obs_cols)

        fit_start = cfg.data.fit_early("A")[0]
        first_apply_start = cfg.data.fit_early("A")[1] + pd.Timedelta(days=1)
        apply_end = plan["test"].declared_end

        log.info(
            "Walk-forward (%s, K=%d): fit_start=%s first_apply_start=%s apply_end=%s cadence=%s",
            winner, winner_k, fit_start.date(), first_apply_start.date(), apply_end.date(),
            cfg.hmm.walkforward.refit_cadence,
        )
        t0 = time.time()
        wf = hmm_walkforward(
            obs_full, k=winner_k,
            fit_start=fit_start, first_apply_start=first_apply_start, apply_end=apply_end,
            cadence=cfg.hmm.walkforward.refit_cadence,
            embargo_days=cfg.data.splits.embargo_days,
            covariance_type=cfg.hmm.fit.covariance_type,
            n_restarts=cfg.hmm.walkforward.n_restarts,
            n_iter=cfg.hmm.fit.n_iter,
            tol=cfg.hmm.fit.tol,
            seed_base=cfg.data.seeds.master,
            min_expected_duration_days=cfg.hmm.selection.min_expected_duration_days,
            min_unconditional_prob=cfg.hmm.selection.min_unconditional_prob,
            max_degenerate_fold_fraction=cfg.hmm.walkforward.max_degenerate_fold_fraction,
        )
        log.info(
            "Walk-forward done in %.1fs: %d folds, %d posterior rows (%s..%s)",
            time.time() - t0, len(wf.folds), len(wf.posteriors),
            wf.posteriors.index[0].date(), wf.posteriors.index[-1].date(),
        )

        out_path = cfg.path("processed") / "hmm_posteriors.parquet"
        wf.posteriors.to_parquet(out_path)
        log.info("posteriors persisted to %s", out_path)

        # --- evaluation --------------------------------------------------- #
        returns = features_a["SPY_return_1d"].loc[wf.posteriors.index]
        characterisation = characterise_states(wf.posteriors, returns)
        quality = posterior_quality(
            wf.posteriors, entropy_saturation_warn=cfg.hmm.evaluation.entropy_saturation_warn
        )
        crisis_col = wf.posteriors.columns[-1]  # canonical: highest-index = highest return std

        # SPY_return_1d is a LOG return (features/asset.py); compounding it as
        # if simple (defect A10's mistake) would distort every drawdown. The
        # correct reconstruction is exp(cumsum(log_returns)).
        prices_a = np.exp(features_a["SPY_return_1d"].cumsum())
        nber = load_nber_recessions(cfg)
        nber_in_window = [(s, e) for s, e in nber if s >= wf.posteriors.index[0]]
        drawdown_episodes = drawdown_bear_episodes(
            prices_a.loc[wf.posteriors.index[0] :], threshold=cfg.hmm.evaluation.drawdown_bear_threshold
        )
        drawdown_in_window = [(s, e) for s, e in drawdown_episodes if s >= wf.posteriors.index[0]]

        baseline = threshold_regime_walkforward(
            features_a["vix_level"], k=2,
            fit_start=fit_start, first_apply_start=first_apply_start, apply_end=apply_end,
            cadence=cfg.hmm.walkforward.refit_cadence, embargo_days=cfg.data.splits.embargo_days,
        )
        baseline_crisis_col = baseline.columns[-1]

        margin = cfg.hmm.evaluation.detection_search_margin_sessions
        nber_comparison = (
            compare_detection(
                wf.posteriors[crisis_col], baseline[baseline_crisis_col], nber_in_window,
                search_margin_sessions=margin,
            )
            if nber_in_window else None
        )
        drawdown_comparison = (
            compare_detection(
                wf.posteriors[crisis_col], baseline[baseline_crisis_col], drawdown_in_window,
                search_margin_sessions=margin,
            )
            if drawdown_in_window else None
        )

        write_evaluation_report(
            cfg.path("tables") / f"hmm_evaluation_{winner}.md",
            winner, winner_k, wf, characterisation, quality, nber_comparison, drawdown_comparison,
        )

        summary["walkforward"] = {
            "specification": winner, "k": winner_k, "n_folds": len(wf.folds),
            "n_posterior_rows": len(wf.posteriors),
            "mean_entropy": quality.mean_entropy, "saturated": quality.saturated,
            "flip_rate": quality.flip_rate,
            "nber_episodes_in_window": len(nber_in_window),
            "drawdown_episodes_in_window": len(drawdown_in_window),
            "hmm_beats_baseline_nber": (
                nber_comparison.hmm_wins_on_detection_lag() if nber_comparison else None
            ),
            "hmm_beats_baseline_drawdown": (
                drawdown_comparison.hmm_wins_on_detection_lag() if drawdown_comparison else None
            ),
        }

    # --- §3.2 robustness: refit on Universe B's train window only -------- #
    if not args.skip_robustness:
        plan = build_split_plan(cfg)
        b_train_start, b_train_end = plan["train"].effective_start, plan["train"].effective_end
        b_val_start, b_val_end = cfg.data.split("val")

        # Spec §8.2 amendment, D-024: robustness is checked for the pinned
        # spec only — the same reason the main K-selection above no longer
        # loops over every declared specification.
        obs_b = build_observation_frame(features_b, list(spec.observations))
        log.info(
            "[robustness %s] K-sweep on B-train-only: fit %s..%s val %s..%s",
            winner, b_train_start.date(), b_train_end.date(), b_val_start.date(), b_val_end.date(),
        )
        robustness_result = fit_and_select(obs_b, b_train_start, b_train_end, b_val_start, b_val_end, cfg)
        write_k_sweep_report(
            cfg.path("tables") / f"hmm_k_selection_{winner}_robustness_B.md",
            f"{winner} (robustness: B-train-only)", robustness_result,
        )
        log.info(
            "[robustness %s] selected K=%d (val_ll=%.2f) vs A-fitted K=%d (val_ll=%.2f)",
            winner, robustness_result.selected_k, robustness_result.selected.val_loglik,
            sweep_result.selected_k, sweep_result.selected.val_loglik,
        )
        summary["robustness_b_train_only"] = {
            winner: {
                "selected_k": robustness_result.selected_k,
                "val_loglik": robustness_result.selected.val_loglik,
            }
        }

    run_id = make_run_id("02_fit_hmm")
    manifest_path = write_manifest(
        cfg.path("logs") / f"{run_id}.json",
        run_id=run_id, stage="02_fit_hmm",
        config_hash=_config_hash(cfg), snapshot_hash=None,
        seeds=seeds.as_dict(), extra=summary, root=cfg.root,
    )
    (cfg.path("processed") / "hmm_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    log.info("run manifest: %s", manifest_path)
    log.info("Step 2 summary: %s", json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
