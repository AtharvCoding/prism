#!/usr/bin/env python
"""Precompute the dashboard's artifacts. DASHBOARD.md §5; DECISIONS.md D-045, D-046.

    python scripts/10_dashboard_data.py            # = make dashboard-data
    python scripts/10_dashboard_data.py --check    # verify the committed artifacts, write nothing

Stage "stored" (the only stage so far; the replay and the live models are milestone D2):

    0 baseline   every source is checked against dashboard/frozen_sources.sha256; a differing file aborts the run
    1 copy       the stored tables of the test split and the holdout, the Tier 1 gates and the frozen configurations,
                 byte for byte, into dashboard/artifacts/
    2 derive     the 40 final runs' learning curves in one file; facts.json (configuration values and table summaries)
    3 verify     every headline row the pages show must be a row of reports/final_report.md, or the run fails
    4 manifest   SHA-256 of every artifact

It reads stored results only: no raw data, no state file, no model, no network, and it never opens the holdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism import dashboard_data as dd  # noqa: E402
from prism.config import load_config  # noqa: E402
from prism.utils.logging import configure_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--check", action="store_true", help="verify the artifacts against the manifest and the final report; write nothing")
    args = ap.parse_args(argv)
    log = configure_logging()
    out = ROOT / dd.ARTIFACTS
    if args.check:
        manifest = dd.verify_artifacts(out)
        check = dd.verify_against_report(out, ROOT / dd.REPORT)
        log.info("artifacts match the manifest (%d files, %s); %d rows found in %s", len(manifest["files"]), manifest["hash"][:12],
                 check["rows_checked"], dd.REPORT)
        return 0
    res = dd.build_stored(load_config(ROOT / args.config), ROOT, out)
    log.info("wrote %d artifacts to %s (manifest %s); %d rows checked against %s", len(res["manifest"]["files"]), dd.ARTIFACTS,
             res["manifest"]["hash"][:12], res["report_check"]["rows_checked"], dd.REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
