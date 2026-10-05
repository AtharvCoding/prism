#!/usr/bin/env python
"""Regenerate the consolidated final report from STORED results. Spec §15.2, §16 step 5.

    python scripts/08_final_report.py        # = make final-report

Reads data/processed/{tier1,tier2,holdout}; recomputes nothing and never opens the holdout. If the holdout
has not been evaluated, the report says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism.config import load_config  # noqa: E402
from prism.reporting.final_report import build_final_report  # noqa: E402


def main() -> int:
    cfg = load_config(ROOT / "configs/base.yaml")
    print(build_final_report(cfg, ROOT, ROOT / "reports" / "final_report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
