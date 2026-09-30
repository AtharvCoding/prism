#!/usr/bin/env python
"""Create the immutable raw data snapshot. Spec §4.1, build step 0.

This is the **only** script that touches the network. Everything downstream
reads the snapshot path from config.

    python scripts/00_snapshot.py                 # download and freeze
    python scripts/00_snapshot.py --dry-run       # show what would be fetched
    python scripts/00_snapshot.py --synthetic     # deterministic offline panel

After a successful run, set ``snapshot.date`` in ``configs/data.yaml`` to the
printed date and commit that change. From then on every result traces to one
hash.

The script refuses to overwrite an existing snapshot. Adjusted prices are
restated whenever a new dividend is paid, so re-downloading silently changes
history — that was defect A1, and it is why the reference project's "frozen"
data was not reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from prism.config import load_config  # noqa: E402
from prism.data.download import (  # noqa: E402
    SnapshotExistsError,
    create_snapshot,
    snapshot_dir,
)
from prism.utils.hashing import make_run_id, write_manifest  # noqa: E402
from prism.utils.logging import configure_logging  # noqa: E402
from prism.utils.seeding import seed_everything  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--date", default=None, help="snapshot date; defaults to today (UTC)"
    )
    parser.add_argument(
        "--universe",
        default="B",
        help="universe whose ticker list to download; B is a superset of A",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="permit overwriting; ALSO requires snapshot.allow_overwrite in config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the tickers, range and destination, then exit without downloading",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="write a deterministic synthetic panel instead of downloading. For "
        "pipeline smoke-testing only; the manifest records synthetic=true.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = configure_logging()
    seeds = seed_everything(cfg.data.seeds.master)

    tickers = cfg.data.tickers(args.universe)
    snap_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.utcnow().normalize()
    destination = snapshot_dir(cfg, snap_date)

    log.info("universe %s: %d tickers", args.universe, len(tickers))
    log.info("tickers: %s", ", ".join(tickers))
    download_start = min(
        pd.Timestamp(cfg.data.inception[t]) for t in tickers if t in cfg.data.inception
    )
    log.info(
        "download range: %s .. %s (earliest inception to holdout end)",
        download_start.date(),
        pd.Timestamp(cfg.data.snapshot.download_end).date(),
    )
    log.info(
        "universe %s features will be restricted to %s onward",
        args.universe,
        cfg.data.start(args.universe).date(),
    )
    log.info("destination: %s", destination)
    log.info(
        "excluded by config: %s",
        ", ".join(f"{k} ({v})" for k, v in cfg.data.excluded.items()),
    )

    if args.dry_run:
        log.info("--dry-run: nothing downloaded, nothing written")
        return 0

    panels = None
    if args.synthetic:
        from prism.data.synthetic import make_synthetic_raw
        from prism.features.build import split_raw_frame

        log.warning(
            "--synthetic: generating a deterministic offline panel. This is NOT "
            "research data and the manifest will record synthetic=true."
        )
        raw = make_synthetic_raw(cfg, universe=args.universe)
        panels = split_raw_frame(raw)

    try:
        out_dir = create_snapshot(
            cfg,
            date=snap_date,
            force=args.force,
            tickers=tickers if args.synthetic else None,
            panels=panels,
        )
    except SnapshotExistsError as exc:
        log.error("%s", exc)
        return 1

    run_id = make_run_id("00_snapshot")
    manifest_path = write_manifest(
        cfg.path("logs") / f"{run_id}.json",
        run_id=run_id,
        stage="00_snapshot",
        config_hash=_config_hash(cfg),
        snapshot_hash=_snapshot_hash(out_dir),
        seeds=seeds.as_dict(),
        extra={"snapshot_dir": str(out_dir), "universe": args.universe},
        root=cfg.root,
    )

    log.info("run manifest: %s", manifest_path)
    log.info(
        "NEXT: set snapshot.date: \"%s\" in configs/data.yaml and commit it.",
        snap_date.date(),
    )
    return 0


def _config_hash(cfg) -> str:  # noqa: ANN001
    from prism.utils.hashing import hash_object

    return hash_object(cfg.model_dump(mode="json", exclude={"root"}))


def _snapshot_hash(out_dir: Path) -> str:
    import json

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    return str(manifest["snapshot_hash"])


if __name__ == "__main__":
    raise SystemExit(main())
