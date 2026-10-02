#!/usr/bin/env python
"""Regenerate every Tier 1 table and figure from STORED results. Spec §15.2, §16 step 4a.

    python scripts/07_report.py            # == `make report`

Reads ``data/processed/tier1/`` (written by ``scripts/04_tier1_ablation.py``)
and rewrites ``reports/tier1_report.md``, ``reports/tables/tier1_*.csv`` and
``reports/figures/tier1_*.png``. Nothing statistical is recomputed here.
``make tier1`` runs both steps, so a single command regenerates the whole report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism.config import load_config  # noqa: E402
from prism.reporting.report import build_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)
    path = build_report(load_config(args.config))
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
