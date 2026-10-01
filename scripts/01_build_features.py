#!/usr/bin/env python
"""Build both universes' features, prune, QA, and persist. Spec §5, §16 step 1.

    python scripts/01_build_features.py

Reads the frozen snapshot named by ``configs/data.yaml`` (never the network).
For each universe (A, then B):

1. Builds the full causal feature frame (:func:`prism.features.build.build_features`),
   which also asserts universe isolation internally — Universe A's build raises
   if it ever references a Universe B-only ticker.
2. Truncates to the non-holdout span. Phase A must never touch the holdout
   (spec §0.4.5); ``build_features`` is holdout-agnostic by design (so it can
   serve live inference in Phase B), so the truncation is this script's job,
   checked with an explicit assertion rather than trusted.
3. Fits :class:`~prism.features.scaling.CorrelationPruner` on the universe's
   own fit/train window — Universe A's ``fit_early`` range, Universe B's
   embargo-purged ``train`` split — and applies it to decide the final,
   frozen feature schema. Spec §5.5: "document which survived."
4. Runs the §4.3 quality checks against the cleaned, allocatable-asset close
   panel and the final feature frame, and writes
   ``reports/tables/data_quality_<universe>.md``.
5. Persists the pruned feature frame to
   ``data/processed/<universe>_features.parquet`` and a schema record to
   ``data/processed/<universe>_schema.json``.

Aborts (non-zero exit) if any universe has a hard QA failure or a
constant/degenerate feature survives pruning — Step 1's acceptance criteria,
checked rather than asserted by eye.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from prism.config import Config, load_config  # noqa: E402
from prism.data.loaders import assert_not_holdout, load_snapshot, non_holdout_span  # noqa: E402
from prism.data.quality import run_quality_checks, write_quality_report  # noqa: E402
from prism.features.build import FeatureSet, build_features, make_raw_frame  # noqa: E402
from prism.features.scaling import CorrelationPruner  # noqa: E402
from prism.splits import build_split_plan  # noqa: E402
from prism.utils.hashing import hash_object, make_run_id, write_manifest  # noqa: E402
from prism.utils.logging import get_logger  # noqa: E402
from prism.utils.logging import configure_logging  # noqa: E402
from prism.utils.seeding import seed_everything  # noqa: E402

_log = get_logger(__name__)


def _fit_window(cfg: Config, universe: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The window a universe's correlation pruner is fit on.

    Universe A is fit over its ``fit_early`` range — the same window the HMM
    and encoder will use in steps 2 and 3 — so the schema that survives
    pruning is the one those models actually see. Universe B is fit over the
    embargo-purged training split, matching the probes and allocator in step
    4a. Fitting on anything larger (e.g. the full non-holdout span) would let
    the pruning decision see data those downstream consumers are not
    permitted to fit on.
    """
    if universe == "A":
        return cfg.data.fit_early("A")
    plan = build_split_plan(cfg)
    return plan["train"].effective_start, plan["train"].effective_end


def _config_hash(cfg: Config) -> str:
    return hash_object(cfg.model_dump(mode="json", exclude={"root"}))


def process_universe(
    cfg: Config,
    raw: pd.DataFrame,
    universe: str,
    non_holdout_end: pd.Timestamp,
) -> dict[str, object]:
    """Build, truncate, prune, QA and persist one universe. Returns a summary dict."""
    _log.info("=== universe %s ===", universe)

    fs: FeatureSet = build_features(raw, cfg, universe)
    assert fs.close is not None, "build_features did not attach a cleaned close panel"

    # --- Phase A must never touch the holdout (spec §0.4.5, §6.3) --------- #
    truncated = fs.frame.loc[:non_holdout_end]
    assert_not_holdout(cfg, truncated.index, context=f"universe {universe} features")
    _log.info(
        "universe %s: %d sessions after holdout truncation (%s .. %s)",
        universe,
        len(truncated),
        truncated.index[0].date() if len(truncated) else None,
        truncated.index[-1].date() if len(truncated) else None,
    )

    # --- correlation pruning, fit on this universe's own window ----------- #
    fit_start, fit_end = _fit_window(cfg, universe)
    fit_slice = truncated.loc[fit_start:fit_end]
    if fit_slice.empty:
        raise ValueError(
            f"universe {universe}: fit window {fit_start.date()}..{fit_end.date()} "
            "contains no rows of the truncated feature frame"
        )
    pruner = CorrelationPruner(0.95).fit(fit_slice, scope=f"{universe} fit window")
    pruned = pruner.transform(truncated)
    _log.info(
        "universe %s: correlation pruning kept %d of %d columns (threshold 0.95)",
        universe,
        len(pruner.kept),
        truncated.shape[1],
    )

    pruning_report_path = cfg.path("tables") / f"{universe}_correlation_pruning.md"
    pruning_report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = pruner.report_rows()
    lines = [
        f"# Correlation pruning — Universe {universe}",
        "",
        f"Fit window: {fit_start.date()} .. {fit_end.date()} "
        f"({len(fit_slice)} sessions). Threshold: |rho| > 0.95.",
        "",
        f"Kept {len(pruner.kept)} of {truncated.shape[1]} columns.",
        "",
    ]
    if rows.empty:
        lines.append("No columns were dropped.")
    else:
        lines += [
            "| Dropped | Kept instead | Family | Group | \\|rho\\| |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in rows.itertuples(index=False):
            lines.append(
                f"| `{row.dropped}` | `{row.kept}` | {row.family} | {row.group} | "
                f"{row.abs_corr:.4f} |"
            )
    pruning_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- quality assurance -------------------------------------------------#
    allocatable = [t for t in cfg.data.allocatable[universe] if t in fs.close.columns]
    close_qa = fs.close.loc[truncated.index, allocatable]
    available_qa = (
        fs.available.loc[truncated.index, allocatable] if fs.available is not None else None
    )
    volume_qa = (
        fs.volume.loc[truncated.index, allocatable]
        if fs.volume is not None and all(c in fs.volume.columns for c in allocatable)
        else None
    )

    report = run_quality_checks(
        close_qa,
        cfg,
        volume=volume_qa,
        available=available_qa,
        features=pruned,
        warmup_end=fs.warm_start,
        universe=universe,
    )
    qa_path = write_quality_report(report, cfg.path("tables") / f"data_quality_{universe}.md")
    _log.info(
        "universe %s: QA — %d hard failure(s), %d soft warning(s) (%s)",
        universe,
        len(report.hard_failures),
        len(report.soft_warnings),
        qa_path,
    )
    for finding in report.hard_failures:
        _log.error("  HARD FAIL [%s]: %s", finding.check, finding.detail)
    for finding in report.soft_warnings:
        _log.warning("  soft [%s]: %s", finding.check, finding.detail)

    # --- persist ------------------------------------------------------------#
    processed_dir = cfg.path("processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{universe}_features.parquet"
    pruned.to_parquet(out_path)

    schema = {
        "universe": universe,
        "schema_hash": hash_object({"universe": universe, "columns": list(pruned.columns)}),
        "columns": [str(c) for c in pruned.columns],
        "n_sessions": int(len(pruned)),
        "first_date": str(pruned.index[0].date()) if len(pruned) else None,
        "last_date": str(pruned.index[-1].date()) if len(pruned) else None,
        "warm_start": str(fs.warm_start.date()),
        "fit_window": [str(fit_start.date()), str(fit_end.date())],
        "pruned_columns": rows["dropped"].tolist() if not rows.empty else [],
        "families": {k: len(v) for k, v in fs.families.items()},
        "long_gaps": fs.long_gaps,
        "qa_hard_failures": len(report.hard_failures),
        "qa_soft_warnings": len(report.soft_warnings),
    }
    (processed_dir / f"{universe}_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
    )
    _log.info("universe %s: %d features persisted to %s", universe, pruned.shape[1], out_path)

    return {
        "universe": universe,
        "qa_ok": report.ok,
        "hard_failures": [f.check for f in report.hard_failures],
        "soft_warnings": [f.check for f in report.soft_warnings],
        "n_features_before_pruning": int(truncated.shape[1]),
        "n_features_after_pruning": int(pruned.shape[1]),
        "schema_hash": schema["schema_hash"],
        "out_path": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--universe",
        action="append",
        choices=["A", "B"],
        help="restrict to one universe; repeatable. Default: both, A then B.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = configure_logging()
    seeds = seed_everything(cfg.data.seeds.master)

    if cfg.data.snapshot.date is None:
        log.error(
            "configs/data.yaml has no snapshot.date; run scripts/00_snapshot.py first "
            "and commit the resulting date"
        )
        return 1

    snapshot = load_snapshot(cfg)
    log.info("snapshot %s (hash %s)", snapshot.path.name, snapshot.snapshot_hash[:12])

    raw = make_raw_frame(snapshot.close, snapshot.volume, snapshot.macro)
    non_holdout_start, non_holdout_end = non_holdout_span(cfg)
    log.info(
        "Phase A readable span: %s .. %s (holdout starts %s)",
        non_holdout_start.date(),
        non_holdout_end.date(),
        cfg.data.split("holdout")[0].date(),
    )

    universes = args.universe or ["A", "B"]
    summaries = [process_universe(cfg, raw, u, non_holdout_end) for u in universes]

    run_id = make_run_id("01_build_features")
    manifest_path = write_manifest(
        cfg.path("logs") / f"{run_id}.json",
        run_id=run_id,
        stage="01_build_features",
        config_hash=_config_hash(cfg),
        snapshot_hash=snapshot.snapshot_hash,
        seeds=seeds.as_dict(),
        extra={"universes": summaries},
        root=cfg.root,
    )
    log.info("run manifest: %s", manifest_path)

    failed = [s["universe"] for s in summaries if not s["qa_ok"]]
    if failed:
        log.error("hard QA failures in universe(s): %s", failed)
        return 1

    log.info("Step 1 acceptance: both universes built, QA clean, schemas frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
