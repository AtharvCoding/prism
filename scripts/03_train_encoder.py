#!/usr/bin/env python
"""Train the LSTM encoder on Universe A, select hyperparameters, walk-forward. Spec §9, §16 step 3.

    python scripts/03_train_encoder.py

Reads ``data/processed/{A,B}_features.parquet`` and the frozen snapshot
(never the network). Three staged hyperparameter decisions, each scored on
Universe B's 2018 validation period (same convention as the HMM's K-sweep,
§3.2) — never on the training loss itself (defect C2):

1. **Window** ∈ {10, 20, 30, 60} at the AE variant, default latent/hidden,
   2 seeds each.
2. **Variant** ∈ {AE, DAE, VAE, PRED} at the selected window, default
   latent/hidden, 5 seeds each (spec §9.3's seed floor, exactly, for the
   comparison that actually matters).
3. **latent_dim** x **hidden_dim** micro-sweep at the selected window and
   variant, 2 seeds each.

The full grid spec §9.3 describes (window x latent x hidden x variant x
seeds, 480 individual training runs) is not run in one pass — benchmarked at
~13s per run on the real data, a staged reduction like this finishes in
minutes rather than requiring roughly two hours of serial training, mirroring
the same resource-constrained-but-documented choice made for the HMM's
walk-forward restart budget (DECISIONS.md). Each stage fixes what the
previous stage selected rather than re-sweeping it, so the reduction's
assumption — that window, variant and architecture size are not sharply
interacting — is at least checked in the order most likely to matter (window
first: it changes what information is even available to later stages).

Then: the winning configuration's full annual expanding walk-forward,
evaluation (reconstruction MSE, dead units, latent drift), a beats-baseline
comparison against PCA and the random encoder via the step 3 probe
machinery, and the §3.2 Universe-B-train-only robustness refit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from prism.analysis.bootstrap import stationary_block_bootstrap_ci  # noqa: E402
from prism.config import Config, load_config  # noqa: E402
from prism.data.loaders import load_snapshot  # noqa: E402
from prism.features.build import make_raw_frame, split_raw_frame  # noqa: E402
from prism.features.scaling import FeatureScaler  # noqa: E402
from prism.features.targets import build_targets  # noqa: E402
from prism.models.baselines.pca_encoder import pca_encoder_walkforward  # noqa: E402
from prism.models.baselines.random_encoder import random_encoder_latents  # noqa: E402
from prism.models.encoder.dataset import WindowDataset  # noqa: E402
from prism.models.encoder.evaluate import (  # noqa: E402
    compute_latents,
    dead_units,
    evaluate_reconstruction,
    latent_drift,
)
from prism.models.encoder.models import build_model  # noqa: E402
from prism.models.encoder.train import train_model  # noqa: E402
from prism.models.encoder.walkforward import encoder_walkforward  # noqa: E402
from prism.probes.probe import compare_probes, ridge_probe  # noqa: E402
from prism.splits import build_split_plan  # noqa: E402
from prism.utils.hashing import hash_object, make_run_id, write_manifest  # noqa: E402
from prism.utils.logging import configure_logging, get_logger  # noqa: E402
from prism.utils.seeding import derive_seed, seed_everything  # noqa: E402

_log = get_logger(__name__)


def _config_hash(cfg: Config) -> str:
    return hash_object(cfg.model_dump(mode="json", exclude={"root"}))


@dataclass
class CandidateScore:
    label: str
    config: dict
    seed_val_losses: list[float]

    @property
    def mean_val_loss(self) -> float:
        return float(np.mean(self.seed_val_losses))

    @property
    def std_val_loss(self) -> float:
        return float(np.std(self.seed_val_losses))


def score_val_loss(model: torch.nn.Module, val_ds: WindowDataset, batch_size: int = 256) -> float:
    """Mean of ``model.loss(...).total`` over ``val_ds``, no gradient."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            total += float(model.loss(x, y).total.item()) * len(x)
            n += len(x)
    return total / n if n else float("nan")


def fit_one_candidate(
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    variant: str,
    window: int,
    hidden_dim: int,
    latent_dim: int,
    cfg: Config,
    seed: int,
    targets: pd.DataFrame | None,
    internal_val_fraction: float = 0.15,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> tuple[torch.nn.Module, float]:
    """Fit on ``fit_frame`` (own internal train/val split), score on ``val_frame``."""
    scaler = FeatureScaler().fit(fit_frame, scope=f"candidate seed={seed}")
    fit_scaled = scaler.transform(fit_frame)
    val_scaled = scaler.transform(val_frame)

    n_val = max(window, int(round(len(fit_scaled) * internal_val_fraction)))
    internal_train = fit_scaled.iloc[: len(fit_scaled) - n_val]
    internal_val = fit_scaled.iloc[len(fit_scaled) - n_val - (window - 1) :]

    train_targets = targets.reindex(internal_train.index) if targets is not None else None
    internal_val_targets = targets.reindex(internal_val.index) if targets is not None else None
    val_targets = targets.reindex(val_scaled.index) if targets is not None else None

    train_ds = WindowDataset(internal_train, window, targets=train_targets)
    internal_val_ds = WindowDataset(internal_val, window, targets=internal_val_targets)
    val_ds = WindowDataset(val_scaled, window, targets=val_targets)

    torch.manual_seed(seed)
    model = build_model(
        variant, input_dim=fit_scaled.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim,
        window=window, dropout=cfg.encoder.architecture.dropout,
        latent_activation=cfg.encoder.architecture.latent_activation,
        dae_noise_std=cfg.encoder.dae_noise_std, vae_kl_weight=cfg.encoder.vae_kl_weight,
        n_pred_targets=(targets.shape[1] if targets is not None else 0),
    )
    result = train_model(
        model,
        DataLoader(train_ds, batch_size=cfg.encoder.training.batch_size, shuffle=True),
        DataLoader(internal_val_ds, batch_size=cfg.encoder.training.batch_size, shuffle=False),
        lr=cfg.encoder.training.lr, weight_decay=cfg.encoder.architecture.weight_decay,
        max_epochs=max_epochs or cfg.encoder.training.max_epochs,
        patience=patience or cfg.encoder.training.patience,
        grad_clip_norm=cfg.encoder.training.grad_clip_norm, log_every=1000,
    )
    val_loss = score_val_loss(result.model, val_ds)
    return result.model, val_loss


def run_stage(
    label: str,
    candidates: list[dict],
    *,
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    cfg: Config,
    seed_base: int,
    n_seeds: int,
    targets: pd.DataFrame | None,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> list[CandidateScore]:
    scores = []
    for cand in candidates:
        losses = []
        for s in range(n_seeds):
            seed = derive_seed(seed_base, label, json.dumps(cand, sort_keys=True), s)
            _, val_loss = fit_one_candidate(
                fit_frame, val_frame, cfg=cfg, seed=seed, targets=targets,
                max_epochs=max_epochs, patience=patience, **cand,
            )
            losses.append(val_loss)
        scores.append(CandidateScore(label=json.dumps(cand, sort_keys=True), config=cand, seed_val_losses=losses))
        _log.info(
            "[%s] %s: val_loss mean=%.6f std=%.6f (%d seeds)",
            label, cand, scores[-1].mean_val_loss, scores[-1].std_val_loss, n_seeds,
        )
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-walkforward", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument(
        "--fast", action="store_true",
        help="cap max_epochs/patience for the hyperparameter sweeps (not the final walk-forward), "
        "for a quick smoke run",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = configure_logging()
    seeds = seed_everything(cfg.data.seeds.master)

    features_a = pd.read_parquet(cfg.path("processed") / "A_features.parquet")
    features_b = pd.read_parquet(cfg.path("processed") / "B_features.parquet")

    snapshot = load_snapshot(cfg)
    raw = make_raw_frame(snapshot.close, snapshot.volume, snapshot.macro)
    close = split_raw_frame(raw)["Close"]
    sleeve = [t for t in cfg.data.universes["A"].equity_sectors if t in close.columns]
    all_targets = build_targets(close, benchmark=cfg.data.universes["A"].benchmark, sleeve=sleeve)
    pred_targets = all_targets[cfg.encoder.pred_targets]

    fit_start, fit_end = cfg.data.fit_early("A")
    val_start, val_end = cfg.data.split("val")
    fit_frame = features_a.loc[fit_start:fit_end]
    val_frame = features_a.loc[val_start:val_end]
    log.info(
        "fit %s..%s (%d rows), val %s..%s (%d rows)",
        fit_start.date(), fit_end.date(), len(fit_frame), val_start.date(), val_end.date(), len(val_frame),
    )

    sweep_epochs = (20, 8) if args.fast else (None, None)

    # --- stage 1: window ---------------------------------------------------#
    default_latent, default_hidden = cfg.encoder.architecture.latent_dim, cfg.encoder.architecture.hidden_dim
    window_candidates = [
        {"variant": "AE", "window": w, "hidden_dim": default_hidden, "latent_dim": default_latent}
        for w in cfg.encoder.window.sweep
    ]
    t0 = time.time()
    window_scores = run_stage(
        "window", window_candidates, fit_frame=fit_frame, val_frame=val_frame, cfg=cfg,
        seed_base=cfg.data.seeds.master, n_seeds=2, targets=None,
        max_epochs=sweep_epochs[0], patience=sweep_epochs[1],
    )
    best_window = min(window_scores, key=lambda s: s.mean_val_loss).config["window"]
    log.info("stage 1 (window) done in %.1fs. Selected window=%d", time.time() - t0, best_window)

    # --- stage 2: variant ---------------------------------------------------#
    variant_candidates = [
        {"variant": v, "window": best_window, "hidden_dim": default_hidden, "latent_dim": default_latent}
        for v in cfg.encoder.variants
    ]
    t0 = time.time()
    variant_scores = run_stage(
        "variant", variant_candidates, fit_frame=fit_frame, val_frame=val_frame, cfg=cfg,
        seed_base=cfg.data.seeds.master, n_seeds=cfg.encoder.training.n_seeds,
        targets=pred_targets, max_epochs=sweep_epochs[0], patience=sweep_epochs[1],
    )
    best_variant = min(variant_scores, key=lambda s: s.mean_val_loss).config["variant"]
    log.info("stage 2 (variant) done in %.1fs. Selected variant=%s", time.time() - t0, best_variant)

    # --- stage 3: latent_dim x hidden_dim ------------------------------------#
    arch_candidates = [
        {"variant": best_variant, "window": best_window, "hidden_dim": h, "latent_dim": l}
        for l in cfg.encoder.architecture.latent_dim_sweep
        for h in cfg.encoder.architecture.hidden_dim_sweep
    ]
    t0 = time.time()
    arch_scores = run_stage(
        "architecture", arch_candidates, fit_frame=fit_frame, val_frame=val_frame, cfg=cfg,
        seed_base=cfg.data.seeds.master, n_seeds=2,
        targets=pred_targets if best_variant == "PRED" else None,
        max_epochs=sweep_epochs[0], patience=sweep_epochs[1],
    )
    best_arch = min(arch_scores, key=lambda s: s.mean_val_loss).config
    best_latent, best_hidden = best_arch["latent_dim"], best_arch["hidden_dim"]
    log.info(
        "stage 3 (architecture) done in %.1fs. Selected latent_dim=%d hidden_dim=%d",
        time.time() - t0, best_latent, best_hidden,
    )

    summary: dict = {
        "window_sweep": [{"config": s.config, "mean_val_loss": s.mean_val_loss, "std_val_loss": s.std_val_loss} for s in window_scores],
        "variant_sweep": [{"config": s.config, "mean_val_loss": s.mean_val_loss, "std_val_loss": s.std_val_loss} for s in variant_scores],
        "architecture_sweep": [{"config": s.config, "mean_val_loss": s.mean_val_loss, "std_val_loss": s.std_val_loss} for s in arch_scores],
        "selected": {
            "window": best_window, "variant": best_variant, "latent_dim": best_latent, "hidden_dim": best_hidden,
        },
    }

    with (cfg.path("tables") / "encoder_hyperparameter_sweep.md").open("w", encoding="utf-8") as fh:
        fh.write("# Encoder hyperparameter sweep\n\n")
        for stage_name, scores in (("Window", window_scores), ("Variant", variant_scores), ("Architecture", arch_scores)):
            fh.write(f"## {stage_name}\n\n| Candidate | mean val loss | std |\n| --- | --- | --- |\n")
            for s in sorted(scores, key=lambda s: s.mean_val_loss):
                fh.write(f"| `{s.config}` | {s.mean_val_loss:.6f} | {s.std_val_loss:.6f} |\n")
            fh.write("\n")
        fh.write(f"**Selected:** {summary['selected']}\n")

    final_targets = pred_targets if best_variant == "PRED" else None

    if args.skip_walkforward:
        log.info("--skip-walkforward: stopping after hyperparameter selection.")
    else:
        plan = build_split_plan(cfg)
        first_apply_start = fit_end + pd.Timedelta(days=1)
        apply_end = plan["test"].declared_end
        log.info(
            "Walk-forward (%s, window=%d, latent=%d, hidden=%d): fit_start=%s first_apply_start=%s apply_end=%s",
            best_variant, best_window, best_latent, best_hidden, fit_start.date(),
            first_apply_start.date(), apply_end.date(),
        )
        t0 = time.time()
        wf = encoder_walkforward(
            features_a, variant=best_variant, window=best_window, hidden_dim=best_hidden, latent_dim=best_latent,
            fit_start=fit_start, first_apply_start=first_apply_start, apply_end=apply_end,
            cadence=cfg.encoder.walkforward.refit_cadence, embargo_days=cfg.data.splits.embargo_days,
            seed_base=cfg.data.seeds.master, lr=cfg.encoder.training.lr,
            weight_decay=cfg.encoder.architecture.weight_decay, max_epochs=cfg.encoder.training.max_epochs,
            patience=cfg.encoder.training.patience, grad_clip_norm=cfg.encoder.training.grad_clip_norm,
            batch_size=cfg.encoder.training.batch_size, dropout=cfg.encoder.architecture.dropout,
            latent_activation=cfg.encoder.architecture.latent_activation,
            dae_noise_std=cfg.encoder.dae_noise_std, vae_kl_weight=cfg.encoder.vae_kl_weight,
            targets=final_targets,
        )
        log.info(
            "Walk-forward done in %.1fs: %d folds, %d latent rows (%s..%s)",
            time.time() - t0, len(wf.folds), len(wf.latents), wf.latents.index[0].date(), wf.latents.index[-1].date(),
        )
        out_path = cfg.path("processed") / "encoder_latents.parquet"
        wf.latents.to_parquet(out_path)
        log.info("latents persisted to %s", out_path)

        # --- evaluation ------------------------------------------------------#
        last_fold = wf.folds[-1]
        apply_frame = features_a.loc[last_fold.fold.apply_start : last_fold.fold.apply_end]
        apply_scaled = last_fold.scaler.transform(apply_frame)
        apply_targets = final_targets.reindex(apply_scaled.index) if final_targets is not None else None
        apply_ds = WindowDataset(apply_scaled, best_window, targets=apply_targets)

        du = dead_units(wf.latents, var_threshold=cfg.encoder.evaluation.dead_unit_var_threshold)

        # Family split by column-name prefix: features_a was loaded straight
        # from the persisted parquet, so FeatureSet.families (which build.py
        # computes at construction time) is not available here. Good enough
        # for a per-family MSE breakdown; the prefixes mirror
        # prism.features.build._family_map's own cross/macro membership.
        cross_prefixes = ("avg_pairwise", "corr_", "first_eigenvalue", "return_dispersion", "breadth")
        macro_prefixes = ("vix", "curve")
        families = {
            "cross": [c for c in features_a.columns if c.startswith(cross_prefixes)],
            "macro": [c for c in features_a.columns if c.startswith(macro_prefixes)],
        }
        families["asset"] = [
            c for c in features_a.columns if c not in families["cross"] and c not in families["macro"]
        ]

        recon_eval = None
        try:
            recon_eval = evaluate_reconstruction(last_fold.model, apply_ds, families=families)
        except ValueError as exc:
            log.info("reconstruction evaluation skipped (expected for a pure-PRED model): %s", exc)

        first_fold = wf.folds[0]
        first_fold_apply = features_a.loc[first_fold.fold.apply_start : first_fold.fold.apply_end]
        first_fold_ds = WindowDataset(first_fold.scaler.transform(first_fold_apply), best_window)
        first_fold_latents = compute_latents(first_fold.model, first_fold_ds)
        last_fold_latents = compute_latents(last_fold.model, apply_ds)
        drift = latent_drift(first_fold_latents, last_fold_latents, test=cfg.encoder.evaluation.drift_test)

        # --- beats-baseline ----------------------------------------------------#
        target_name = "fwd_vol_20"
        target_series = all_targets[target_name]
        plan2 = build_split_plan(cfg)
        probe_bounds = dict(
            train_start=plan2["train"].effective_start, train_end=plan2["train"].effective_end,
            val_start=plan2["val"].effective_start, val_end=plan2["val"].effective_end,
            test_start=plan2["test"].effective_start, test_end=plan2["test"].effective_end,
        )
        alpha_grid = cfg.tier1.probe.alpha_grid

        lstm_res = ridge_probe(wf.latents, target_series, alpha_grid=alpha_grid, **probe_bounds)

        # PCA refit on the SAME expanding, embargoed fold schedule as the
        # LSTM walk-forward — a PCA fit once on fit_early and left unrefit
        # through 2023 would be an unfair comparison in either direction
        # (see DECISIONS.md). pca_encoder_walkforward exists specifically so
        # this comparison and the LSTM's are on equal footing.
        pca_latents = pca_encoder_walkforward(
            features_a, window=best_window, n_components=best_latent,
            fit_start=fit_start, first_apply_start=first_apply_start, apply_end=apply_end,
            cadence=cfg.encoder.walkforward.refit_cadence, embargo_days=cfg.data.splits.embargo_days,
        ).loc[wf.latents.index]
        pca_res = ridge_probe(pca_latents, target_series, alpha_grid=alpha_grid, **probe_bounds)

        random_latents = random_encoder_latents(
            features_a, best_window, hidden_dim=best_hidden, latent_dim=best_latent, seed=cfg.data.seeds.master,
        ).loc[wf.latents.index]
        random_res = ridge_probe(random_latents, target_series, alpha_grid=alpha_grid, **probe_bounds)

        cmp_pca = compare_probes(lstm_res, pca_res, name_a="LSTM", name_b="PCA", n_bootstrap=cfg.tier1.uncertainty.n_bootstrap)
        cmp_random = compare_probes(lstm_res, random_res, name_a="LSTM", name_b="random", n_bootstrap=cfg.tier1.uncertainty.n_bootstrap)

        with (cfg.path("tables") / f"encoder_evaluation_{best_variant}.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# Encoder evaluation — {best_variant} (window={best_window}, latent={best_latent}, hidden={best_hidden})\n\n")
            fh.write(f"## Dead units\n\n- {du.n_dead}/{du.n_total} dimensions below variance threshold {cfg.encoder.evaluation.dead_unit_var_threshold}\n")
            if du.n_dead:
                fh.write(f"- dead: {du.dead_columns}\n")
            if recon_eval is not None:
                fh.write(f"\n## Out-of-sample reconstruction (last fold)\n\n- overall MSE: {recon_eval.mse_overall:.6f}\n")
                for fam, mse in recon_eval.mse_per_family.items():
                    fh.write(f"- {fam}: {mse:.6f}\n")
            fh.write(f"\n## Latent drift (first fold vs last fold), {cfg.encoder.evaluation.drift_test}\n\n- {drift.n_drifted}/{drift.n_total} dimensions drifted at alpha={drift.alpha}\n")
            fh.write(f"\n## Beats-baseline on probe (`{target_name}`)\n\n")
            fh.write(f"- LSTM R2: {lstm_res.r2:.4f}, PCA R2: {pca_res.r2:.4f}, random R2: {random_res.r2:.4f}\n")
            fh.write(f"- LSTM beats PCA (non-overlapping MSE CI, lower is better): **{cmp_pca.a_beats_b()}**\n")
            fh.write(f"- LSTM beats random encoder: **{cmp_random.a_beats_b()}**\n")
            fh.write(f"- LSTM MSE CI: {cmp_pca.mse_ci_a.as_dict()}\n")
            fh.write(f"- PCA MSE CI: {cmp_pca.mse_ci_b.as_dict()}\n")
            fh.write(f"- random MSE CI: {cmp_random.mse_ci_b.as_dict()}\n")

        summary["walkforward"] = {
            "variant": best_variant, "window": best_window, "latent_dim": best_latent, "hidden_dim": best_hidden,
            "n_folds": len(wf.folds), "n_latent_rows": len(wf.latents),
            "dead_units": du.n_dead, "latent_drift_n_drifted": drift.n_drifted,
            "lstm_r2": lstm_res.r2, "pca_r2": pca_res.r2, "random_r2": random_res.r2,
            "lstm_beats_pca": cmp_pca.a_beats_b(), "lstm_beats_random": cmp_random.a_beats_b(),
        }

    if not args.skip_robustness:
        plan3 = build_split_plan(cfg)
        b_train_start, b_train_end = plan3["train"].effective_start, plan3["train"].effective_end
        b_val_start, b_val_end = cfg.data.split("val")
        b_fit_frame = features_b.loc[b_train_start:b_train_end]
        b_val_frame = features_b.loc[b_val_start:b_val_end]
        b_targets = pred_targets.reindex(features_b.index) if best_variant == "PRED" else None

        losses = []
        for s in range(2):
            seed = derive_seed(cfg.data.seeds.master, "robustness_B", s)
            _, val_loss = fit_one_candidate(
                b_fit_frame, b_val_frame, variant=best_variant, window=best_window,
                hidden_dim=best_hidden, latent_dim=best_latent, cfg=cfg, seed=seed, targets=b_targets,
            )
            losses.append(val_loss)
        summary["robustness_b_train_only"] = {"mean_val_loss": float(np.mean(losses)), "losses": losses}
        log.info("robustness (B-train-only): mean val_loss=%.6f vs A-fitted", float(np.mean(losses)))

    run_id = make_run_id("03_train_encoder")
    manifest_path = write_manifest(
        cfg.path("logs") / f"{run_id}.json", run_id=run_id, stage="03_train_encoder",
        config_hash=_config_hash(cfg), snapshot_hash=snapshot.snapshot_hash,
        seeds=seeds.as_dict(), extra=summary, root=cfg.root,
    )
    (cfg.path("processed") / "encoder_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    log.info("run manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
