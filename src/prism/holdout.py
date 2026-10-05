"""Extend the walk-forward state pipeline past the test split. Spec §6.3, §16 step 5; preregistration_holdout §3.

Steps 1-3b were run with every date cut at the test split's end (Phase A must not
read the holdout). Evaluating the frozen Tier 2 agents on the holdout needs their
inputs, V1, V2, V4 and C4, for 2024-01-01 .. 2026-09-30, produced **the same way**:
causal features, the pinned H1/K=2 HMM walk-forward, the selected DAE encoder
walk-forward, the VIX-threshold control, the train-split scaler. Nothing is
re-selected or re-tuned: K, the specification, the encoder window/latent/hidden, the
seeds, the folds and every threshold are read from the persisted step 2-3 summaries
and configs.

:func:`build_extended_states` does it for any ``end`` date. It is **verified by
replay**: run with ``end`` = the test split's end it must reproduce the stored
``data/processed/states/{V1,V2,V4,C4}.parquet`` (:func:`replay_check`), which is
evidence that the extension adds rows and changes none. The real extension is
accepted only if its rows up to the test end equal the stored ones.

Nothing here writes into ``data/processed`` itself or can overwrite a pre-registered file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prism.config import Config
from prism.data.loaders import load_snapshot
from prism.features.build import FeatureSet, build_features, make_raw_frame
from prism.features.scaling import CorrelationPruner, FeatureScaler
from prism.models.baselines.threshold_regime import threshold_regime_walkforward
from prism.models.encoder.walkforward import encoder_walkforward
from prism.models.hmm.walkforward import hmm_walkforward
from prism.splits import build_split_plan
from prism.state import PHASE_B_VARIANTS, StateArtifacts, build_state
from prism.utils.hashing import hash_object
from prism.utils.logging import get_logger

__all__ = ["ExtendedStates", "build_extended_states", "replay_check", "VARIANTS"]

_log = get_logger(__name__)
VARIANTS = ("V1", "V2", "V4", "C4")
C2_SIGNAL_COLUMN = "vix_level"


@dataclass
class ExtendedStates:
    end: pd.Timestamp
    states: dict[str, pd.DataFrame]
    posteriors: pd.DataFrame
    latents: pd.DataFrame
    close: pd.DataFrame
    timings: dict[str, float]


def _prune(cfg: Config, fs: FeatureSet, universe: str, end: pd.Timestamp) -> pd.DataFrame:
    """The step-1 pruning, verbatim: fit on the universe's own fit window, applied to the span up to ``end``."""
    if universe == "A":
        fit_start, fit_end = cfg.data.fit_early("A")
    else:
        plan = build_split_plan(cfg)
        fit_start, fit_end = plan["train"].effective_start, plan["train"].effective_end
    cut = fs.frame.loc[:end]
    pruner = CorrelationPruner(0.95).fit(cut.loc[fit_start:fit_end], scope=f"{universe} fit window")
    return pruner.transform(cut)


def _load_set(cfg: Config, universe: str, frame: pd.DataFrame, fs: FeatureSet) -> FeatureSet:
    schema = json.loads((cfg.path("processed") / f"{universe}_schema.json").read_text(encoding="utf-8"))
    out = FeatureSet(universe=universe, frame=frame, untrimmed=frame, warm_start=fs.warm_start,
                     families={k: [] for k in schema.get("families", {})})
    if out.schema_hash != schema["schema_hash"]:
        raise AssertionError(f"universe {universe}: the extended feature schema differs from the persisted one")
    return out


def build_extended_states(cfg: Config, end: str | pd.Timestamp, *, raw: pd.DataFrame | None = None) -> ExtendedStates:
    """V1, V2, V4, C4 with the walk-forward pipeline run through ``end``."""
    end = pd.Timestamp(end)
    timings: dict[str, float] = {}
    t0 = time.time()
    if raw is None:
        snap = load_snapshot(cfg)
        raw = make_raw_frame(snap.close, snap.volume, snap.macro)
    raw = raw.loc[:end]
    plan = build_split_plan(cfg)

    fsets, close = {}, None
    for u in ("A", "B"):
        fs = build_features(raw, cfg, u)
        pruned = _prune(cfg, fs, u, end)
        fsets[u] = _load_set(cfg, u, pruned, fs)
        if u == "B":
            close = fs.close
    features_a, features_b = fsets["A"].frame, fsets["B"]
    timings["features"] = time.time() - t0

    hmm_summary = json.loads((cfg.path("processed") / "hmm_summary.json").read_text(encoding="utf-8"))
    spec = hmm_summary["walkforward"]["specification"]
    k = int(hmm_summary["walkforward"]["k"])
    obs = features_a[list(cfg.hmm.specifications[spec].observations)].dropna()
    fit_start = cfg.data.fit_early("A")[0]
    first_apply = cfg.data.fit_early("A")[1] + pd.Timedelta(days=1)

    t1 = time.time()
    wf = hmm_walkforward(
        obs, k=k, fit_start=fit_start, first_apply_start=first_apply, apply_end=end,
        cadence=cfg.hmm.walkforward.refit_cadence, embargo_days=cfg.data.splits.embargo_days,
        covariance_type=cfg.hmm.fit.covariance_type, n_restarts=cfg.hmm.walkforward.n_restarts,
        n_iter=cfg.hmm.fit.n_iter, tol=cfg.hmm.fit.tol, seed_base=cfg.data.seeds.master,
        min_expected_duration_days=cfg.hmm.selection.min_expected_duration_days,
        min_unconditional_prob=cfg.hmm.selection.min_unconditional_prob,
        max_degenerate_fold_fraction=cfg.hmm.walkforward.max_degenerate_fold_fraction,
    )
    timings["hmm"] = time.time() - t1

    enc = json.loads((cfg.path("processed") / "encoder_summary.json").read_text(encoding="utf-8"))
    sel = enc["selected"]
    t2 = time.time()
    ewf = encoder_walkforward(
        features_a, variant=sel["variant"], window=int(sel["window"]), hidden_dim=int(sel["hidden_dim"]),
        latent_dim=int(sel["latent_dim"]), fit_start=fit_start, first_apply_start=first_apply, apply_end=end,
        cadence=cfg.encoder.walkforward.refit_cadence, embargo_days=cfg.data.splits.embargo_days,
        seed_base=cfg.data.seeds.master, lr=cfg.encoder.training.lr, weight_decay=cfg.encoder.architecture.weight_decay,
        max_epochs=cfg.encoder.training.max_epochs, patience=cfg.encoder.training.patience,
        grad_clip_norm=cfg.encoder.training.grad_clip_norm, batch_size=cfg.encoder.training.batch_size,
        dropout=cfg.encoder.architecture.dropout, latent_activation=cfg.encoder.architecture.latent_activation,
        dae_noise_std=cfg.encoder.dae_noise_std, vae_kl_weight=cfg.encoder.vae_kl_weight, targets=None,
    )
    timings["encoder"] = time.time() - t2

    threshold = threshold_regime_walkforward(
        features_a[C2_SIGNAL_COLUMN], k=wf.posteriors.shape[1], fit_start=fit_start,
        first_apply_start=first_apply, apply_end=end, cadence=cfg.hmm.walkforward.refit_cadence,
        embargo_days=cfg.data.splits.embargo_days,
    )
    artifacts = StateArtifacts(latents=ewf.latents, posteriors=wf.posteriors, threshold_states=threshold)
    scaler = FeatureScaler().fit(
        features_b.frame.loc[plan["train"].effective_start : plan["train"].effective_end],
        scope="state assembly (train split)",
    )
    states = {}
    for v in VARIANTS:
        states[v] = build_state(features_b, cfg, v, artifacts=artifacts, scaler=scaler,
                                v1p_window=int(sel["window"])).frame
    timings["total"] = time.time() - t0
    _log.info("extended states through %s: %s (timings %s)", end.date(), {v: f.shape for v, f in states.items()},
              {k_: round(v_) for k_, v_ in timings.items()})
    return ExtendedStates(end=end, states=states, posteriors=wf.posteriors, latents=ewf.latents,
                          close=close.loc[:end], timings=timings)


def replay_check(cfg: Config, ext: ExtendedStates, *, up_to: str | pd.Timestamp, atol: float) -> dict[str, object]:
    """Compare the extended states with the stored ones on every row up to ``up_to``.

    Returns the per-variant maximum absolute difference and whether the index and columns are identical.
    """
    up_to = pd.Timestamp(up_to)
    out: dict[str, object] = {"up_to": str(up_to.date()), "atol": atol, "variants": {}}
    ok = True
    for v in VARIANTS:
        stored = pd.read_parquet(cfg.path("processed") / "states" / f"{v}.parquet")
        new = ext.states[v].loc[:up_to]
        same_shape = list(stored.columns) == list(new.columns) and stored.index.equals(new.index)
        diff = float(np.abs(stored.to_numpy(dtype="float64") - new.to_numpy(dtype="float64")).max()) if same_shape else float("inf")
        out["variants"][v] = {"same_index_and_columns": bool(same_shape), "max_abs_diff": diff, "rows": int(len(new))}
        ok &= bool(same_shape and diff <= atol)
    out["ok"] = bool(ok)
    out["hash"] = hash_object({k: v for k, v in out.items() if k != "hash"})
    return out
