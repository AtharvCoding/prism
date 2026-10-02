#!/usr/bin/env python
"""Assemble all nine state variants. Spec §10, §16 step 3b.

    python scripts/03b_build_states.py                # mandatory variants only
    python scripts/03b_build_states.py --with-oracle   # also O1 (diagnostic; re-fits the
                                                        # winning HMM spec with smoothing)

Reads what steps 1-3 already persisted — never refits the HMM or the encoder:

* ``data/processed/{A,B}_features.parquet`` + ``{A,B}_schema.json`` (step 1)
* ``data/processed/hmm_posteriors.parquet`` + ``hmm_summary.json`` (step 2)
* ``data/processed/encoder_latents.parquet`` + ``encoder_summary.json`` (step 3)

V1's base is Universe B (the allocatable universe); the regime signal
(latents, posteriors) is date-indexed and fit on Universe A, reused as-is —
the same cross-universe reuse step 2/3's own scripts already rely on.

C1 (random-encoder latents) and C2 (threshold-regime one-hot) are cheap,
unfitted-or-trivially-fitted controls: recomputed here, on Universe A's
features. C1 uses the window / hidden / latent dimensions step 3 SELECTED
(read from ``encoder_summary.json``, never the config defaults), and V1' uses
that same window; C2 uses the identical expanding/embargoed fold schedule
steps 2-3 used, rather than persisted anywhere upstream. C2's
one-hot cardinality is taken from the persisted HMM posteriors' own column
count, so it is directly comparable to V3 (spec §10). C3 (shuffled
posteriors) is computed inline by :func:`prism.state.build_state` itself.

O1 (the leaky oracle, spec §10) is diagnostic-only and off by default
(``--with-oracle``): producing it means re-fitting the winning HMM
specification with ``also_smoothed=True`` (the per-fold model is not
persisted by step 2, so there is nothing cheaper to reuse). It is written to
its own file, separate from the eight reportable variants, and never
produced otherwise.

Every variant's assembled frame is written to
``data/processed/states/<variant>.parquet``, and
``data/processed/states/schema.json`` records every variant's schema hash,
column count and date range — the step 3b acceptance criterion "schema hash
recorded".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from prism.config import Config, load_config  # noqa: E402
from prism.features.build import FeatureSet  # noqa: E402
from prism.features.scaling import FeatureScaler  # noqa: E402
from prism.models.baselines.random_encoder import random_encoder_latents  # noqa: E402
from prism.models.baselines.threshold_regime import threshold_regime_walkforward  # noqa: E402
from prism.models.hmm.walkforward import hmm_walkforward  # noqa: E402
from prism.splits import build_split_plan  # noqa: E402
from prism.state import StateArtifacts, StateFrame, build_state  # noqa: E402
from prism.utils.hashing import hash_object, make_run_id, write_manifest  # noqa: E402
from prism.utils.logging import configure_logging, get_logger  # noqa: E402
from prism.utils.seeding import derive_seed, seed_everything  # noqa: E402

_log = get_logger(__name__)

STEP = "3b"

#: Observation column the threshold-regime control (C2) is fit on — the same
#: series step 2's own HMM-vs-baseline evaluation uses (spec §8.7).
C2_SIGNAL_COLUMN = "vix_level"


def _config_hash(cfg: Config) -> str:
    return hash_object(cfg.model_dump(mode="json", exclude={"root"}))


def _load_feature_set(cfg: Config, universe: str) -> FeatureSet:
    """Reconstruct the step-1 :class:`FeatureSet` from its persisted parquet + schema.

    Step 1 persists the pruned ``frame`` and a schema record, not the
    dataclass itself. ``schema_hash`` is a pure function of ``universe`` and
    the frame's own column list, so reconstructing it here reproduces the
    persisted schema hash exactly — verified below rather than assumed.
    """
    processed = cfg.path("processed")
    frame = pd.read_parquet(processed / f"{universe}_features.parquet")
    schema = json.loads((processed / f"{universe}_schema.json").read_text(encoding="utf-8"))
    fs = FeatureSet(
        universe=universe,
        frame=frame,
        untrimmed=frame,
        warm_start=pd.Timestamp(schema["warm_start"]),
        families={k: [] for k in schema.get("families", {})},
    )
    if fs.schema_hash != schema["schema_hash"]:
        raise AssertionError(
            f"reconstructed {universe} FeatureSet schema_hash does not match the persisted "
            f"{universe}_schema.json — the parquet and schema record have drifted apart"
        )
    return fs


def _selected_encoder(cfg: Config) -> dict[str, int]:
    """The encoder step 3 actually selected, from ``encoder_summary.json``.

    NOT ``cfg.encoder.window.size`` / ``cfg.encoder.architecture``: those are
    the config *defaults* the sweep started from. Step 3 selected window 10 and
    latent_dim 32 against defaults of 30 and 16, and C1 and V1' must match the
    selection, not the defaults (spec §10: C1 isolates "more dimensions", V1'
    carries the window V2's encoder saw).
    """
    summary = json.loads((cfg.path("processed") / "encoder_summary.json").read_text(encoding="utf-8"))
    wf = summary["walkforward"]
    return {"window": int(wf["window"]), "latent_dim": int(wf["latent_dim"]), "hidden_dim": int(wf["hidden_dim"])}


def _build_c1(cfg: Config, features_a: pd.DataFrame, selected: dict[str, int]) -> pd.DataFrame:
    """C1: frozen random-encoder latents, same window/hidden/latent dims as the selected encoder."""
    return random_encoder_latents(
        features_a,
        selected["window"],
        hidden_dim=selected["hidden_dim"],
        latent_dim=selected["latent_dim"],
        seed=derive_seed(cfg.data.seeds.master, "state_c1_random_encoder"),
        latent_activation=cfg.encoder.architecture.latent_activation,
    )


def _build_c2(
    cfg: Config, features_a: pd.DataFrame, k: int,
    fit_start: pd.Timestamp, first_apply_start: pd.Timestamp, apply_end: pd.Timestamp,
) -> pd.DataFrame:
    """C2: threshold-regime one-hot, k matched to the HMM's own selected K."""
    return threshold_regime_walkforward(
        features_a[C2_SIGNAL_COLUMN], k=k,
        fit_start=fit_start, first_apply_start=first_apply_start, apply_end=apply_end,
        cadence=cfg.hmm.walkforward.refit_cadence, embargo_days=cfg.data.splits.embargo_days,
    )


def _build_oracle(
    cfg: Config, features_a: pd.DataFrame,
    fit_start: pd.Timestamp, first_apply_start: pd.Timestamp, apply_end: pd.Timestamp,
) -> pd.DataFrame:
    """O1: re-fit the winning HMM spec with also_smoothed=True.

    Step 2 does not persist per-fold model objects, so there is nothing
    cheaper to reuse. Fitting is deterministic in ``cfg.data.seeds.master``,
    so this reproduces step 2's own filtered posteriors exactly alongside the
    new smoothed ones — asserted below, not just assumed.
    """
    hmm_summary = json.loads(
        (cfg.path("processed") / "hmm_summary.json").read_text(encoding="utf-8")
    )
    spec_name = hmm_summary["walkforward"]["specification"]
    k = hmm_summary["walkforward"]["k"]
    obs_cols = list(cfg.hmm.specifications[spec_name].observations)
    obs = features_a[obs_cols].dropna()

    wf = hmm_walkforward(
        obs, k=k,
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
        also_smoothed=True,
    )
    persisted = pd.read_parquet(cfg.path("processed") / "hmm_posteriors.parquet")
    pd.testing.assert_frame_equal(wf.posteriors.loc[persisted.index], persisted, check_exact=False)
    assert wf.smoothed_posteriors is not None
    return wf.smoothed_posteriors


def write_schema_report(path: Path, states: dict[str, StateFrame]) -> None:
    record = {
        variant: {
            "schema_hash": sf.schema_hash,
            "feature_schema_hash": sf.feature_schema_hash,
            "n_columns": sf.frame.shape[1],
            "n_sessions": sf.frame.shape[0],
            "columns": sf.columns,
            "first_date": str(sf.index[0].date()) if len(sf.index) else None,
            "last_date": str(sf.index[-1].date()) if len(sf.index) else None,
            "includes_portfolio_block": sf.includes_portfolio_block,
        }
        for variant, sf in states.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--with-oracle", action="store_true",
        help="also build O1, the diagnostic leaky oracle (re-fits the winning HMM spec)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = configure_logging()
    seeds = seed_everything(cfg.data.seeds.master)

    features_b = _load_feature_set(cfg, "B")
    features_a_fs = _load_feature_set(cfg, "A")
    features_a = features_a_fs.frame

    posteriors = pd.read_parquet(cfg.path("processed") / "hmm_posteriors.parquet")
    latents = pd.read_parquet(cfg.path("processed") / "encoder_latents.parquet")
    k = posteriors.shape[1]

    plan = build_split_plan(cfg)
    fit_start = cfg.data.fit_early("A")[0]
    first_apply_start = cfg.data.fit_early("A")[1] + pd.Timedelta(days=1)
    apply_end = plan["test"].declared_end

    t0 = time.time()
    selected = _selected_encoder(cfg)
    random_latents = _build_c1(cfg, features_a, selected)
    if random_latents.shape[1] != latents.shape[1]:
        raise AssertionError(
            f"C1 has {random_latents.shape[1]} columns but the encoder latent has "
            f"{latents.shape[1]}: C1 must be dimension-matched to V2 (spec §10)"
        )
    log.info("encoder selection used for C1 and V1': %s", selected)
    threshold_states = _build_c2(cfg, features_a, k, fit_start, first_apply_start, apply_end)
    log.info("C1/C2 controls built in %.1fs", time.time() - t0)

    artifacts = StateArtifacts(
        latents=latents, posteriors=posteriors,
        random_latents=random_latents, threshold_states=threshold_states,
    )

    # One scaler, fit on the Universe B train split, shared across every
    # variant — spec §10's "same...scaling policy" (see prism.state's
    # module docstring).
    train_start, train_end = plan["train"].effective_start, plan["train"].effective_end
    scaler = FeatureScaler().fit(
        features_b.frame.loc[train_start:train_end], scope="state assembly (train split)"
    )

    states: dict[str, StateFrame] = {}
    for variant in cfg.tier1.variants:
        states[variant] = build_state(
            features_b, cfg, variant, artifacts=artifacts, scaler=scaler,
            v1p_window=selected["window"],
        )
        log.info(
            "variant %s: %d sessions x %d columns, schema_hash=%s",
            variant, states[variant].frame.shape[0], states[variant].frame.shape[1],
            states[variant].schema_hash[:12],
        )

    if args.with_oracle:
        smoothed = _build_oracle(cfg, features_a, fit_start, first_apply_start, apply_end)
        oracle_artifacts = StateArtifacts(
            latents=latents, posteriors=posteriors,
            random_latents=random_latents, threshold_states=threshold_states,
            smoothed_posteriors=smoothed,
        )
        states["O1"] = build_state(
            features_b, cfg, "O1", artifacts=oracle_artifacts, scaler=scaler,
            v1p_window=selected["window"],
        )
        log.info(
            "variant O1 (diagnostic, NEVER a reported result): %d sessions x %d columns",
            states["O1"].frame.shape[0], states["O1"].frame.shape[1],
        )

    out_dir = cfg.path("processed") / "states"
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, sf in states.items():
        sf.frame.to_parquet(out_dir / f"{variant}.parquet")
    write_schema_report(out_dir / "schema.json", states)
    log.info("states persisted to %s", out_dir)

    run_id = make_run_id("03b_build_states")
    manifest_path = write_manifest(
        cfg.path("logs") / f"{run_id}.json",
        run_id=run_id, stage="03b_build_states",
        config_hash=_config_hash(cfg), snapshot_hash=None,
        seeds=seeds.as_dict(),
        extra={"variants": sorted(states), "k": k},
        root=cfg.root,
    )
    log.info("run manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
