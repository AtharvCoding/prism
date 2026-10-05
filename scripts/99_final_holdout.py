#!/usr/bin/env python
"""Evaluate the final holdout, ONCE. Spec §6.3, §16 step 5; reports/tables/preregistration_holdout.md.

    PRISM_ALLOW_HOLDOUT=1 python scripts/99_final_holdout.py --i-am-sure     # the one real run
    python scripts/99_final_holdout.py --rehearse                             # never reads the holdout

What it does, in order (each step aborts the run, loudly, before the next):

    0 contract    the holdout pre-registration is committed and unmodified; every file it pins matches its SHA-256
                  (the four state files, chosen_configs.json, all 40 final checkpoints)
    1 gates       --i-am-sure AND PRISM_ALLOW_HOLDOUT=1 AND a recorded reason; the access is logged permanently in
                  reports/logs/holdout_access.jsonl; the run refuses if the holdout was already evaluated
    2 inputs      V1, V2, V4, C4 through the holdout, the same walk-forward pipeline; REPLAY CHECK: its rows up to the
                  test split's end must equal the stored states to 1e-9, else abort
    3 evaluate    the 40 frozen agents and the six benchmarks, through the env, once; daily net returns stored
    4 report      the same paired-bootstrap gates, deflated Sharpe, episodes, overfitting; gate decisions appended to
                  DECISIONS.md; nothing is retrained, re-tuned, re-selected or re-run on a second look

``--rehearse`` does steps 0, 2 and 3-4 with the pipeline's *replay* window and the VALIDATION split standing in for the
holdout, into data/processed/holdout_rehearsal: it exercises every line of code the real run uses without opening the holdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from prism import holdout as H  # noqa: E402
from prism.agents import tier2 as t2  # noqa: E402
from prism.config import load_config  # noqa: E402
from prism.data.loaders import HOLDOUT_ENV_VAR, load_holdout, load_snapshot  # noqa: E402
from prism.features.build import make_raw_frame  # noqa: E402
from prism.reporting.final_report import build_final_report, holdout_decisions_entry  # noqa: E402
from prism.reporting.tier2_report import append_decisions  # noqa: E402
from prism.splits import build_split_plan  # noqa: E402
from prism.utils.hashing import make_run_id, sha256_file, write_manifest  # noqa: E402
from prism.utils.logging import configure_logging  # noqa: E402
from prism.utils.prereg import preregistered_hashes  # noqa: E402
from prism.utils.seeding import seed_everything  # noqa: E402

PREREG = "reports/tables/preregistration_holdout.md"
REPLAY_ATOL = 1e-9
REASON = "build step 5: one confirmatory evaluation of the frozen Tier 2 agents (preregistration_holdout.md)"


def _git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def check_contract(log) -> str:  # noqa: ANN001
    try:
        _git("ls-files", "--error-unmatch", PREREG)
    except subprocess.CalledProcessError:
        raise SystemExit(f"{PREREG} is not committed; commit it first (it must precede the holdout)")
    if _git("status", "--porcelain", "--", PREREG):
        raise SystemExit(f"{PREREG} has uncommitted changes")
    commit = _git("log", "-1", "--format=%H", "--", PREREG)
    listed = preregistered_hashes(ROOT, PREREG)
    if len(listed) < 45:
        raise SystemExit(f"{PREREG} pins only {len(listed)} files; expected the 4 states, the frozen configs and 40 checkpoints")
    bad = [n for n, d in listed.items() if sha256_file(ROOT / "data" / "processed" / n) != d]
    if bad:
        raise SystemExit(f"files differ from the pre-registered SHA-256: {bad[:5]}{' ...' if len(bad) > 5 else ''}")
    log.info("contract: %s at %s; %d pinned files verified", PREREG, commit[:10], len(listed))
    return commit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--i-am-sure", action="store_true", help="required for the real run")
    ap.add_argument("--rehearse", action="store_true", help="exercise the whole path on the validation split; never reads the holdout")
    args = ap.parse_args(argv)
    if args.rehearse == args.i_am_sure:
        ap.error("pass exactly one of --i-am-sure (the real, one-time run) or --rehearse")

    cfg = load_config(ROOT / args.config)
    log = configure_logging(log_file=ROOT / "logs" / "holdout" / ("rehearsal.log" if args.rehearse else "holdout.log"))
    seeds = seed_everything(cfg.data.seeds.master)
    base = t2.Tier2Plan.from_config(cfg, ROOT)
    plan_split = build_split_plan(cfg)
    runs_dir = base.out_dir                                              # the Tier 2 agents
    sanity = json.loads((runs_dir / "sanity.json").read_text())

    if args.rehearse:
        out_dir = base.out_dir.parent / "holdout_rehearsal"
        plan = replace(base, out_dir=out_dir, runs_dir=runs_dir, eval_split="val", smoke=True)
        end = plan_split["test"].declared_end                            # the replay window: no holdout row
        commit = "rehearsal"
    else:
        commit = check_contract(log)
        out_dir = base.out_dir.parent / "holdout"
        plan = replace(base, out_dir=out_dir, runs_dir=runs_dir, eval_split="holdout")
        end = plan_split["holdout"].declared_end
        marker = out_dir / "eval_done.json"
        if marker.exists():
            raise SystemExit(f"the holdout was already evaluated ({marker}); it is used exactly once")
        if os.environ.get(HOLDOUT_ENV_VAR) != "1":
            raise SystemExit(f"set {HOLDOUT_ENV_VAR}=1 as well as --i-am-sure (two independent gates, spec §6.3)")
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = t2.load_frozen(plan)

    # ------------------------------------------------------------------ inputs
    snap = load_snapshot(cfg)
    raw = make_raw_frame(snap.close, snap.volume, snap.macro)
    if not args.rehearse:
        load_holdout(cfg, raw, final=True, reason=REASON)                # logs the access permanently
    ext_path = out_dir / "states_extended.parquet"
    ext = H.build_extended_states(cfg, end, raw=raw)
    check = H.replay_check(cfg, ext, up_to=plan_split["test"].declared_end, atol=REPLAY_ATOL)
    (out_dir / "replay_check.json").write_text(json.dumps(check, indent=2))
    log.info("replay check (rows up to %s vs stored states): %s", check["up_to"], json.dumps(check["variants"]))
    if not check["ok"]:
        raise SystemExit("REPLAY CHECK FAILED: the extended pipeline does not reproduce the stored states; the holdout is not evaluated")
    pd.concat({v: f for v, f in ext.states.items()}, axis=1).to_parquet(ext_path)

    # ------------------------------------------------------------------ evaluate + report
    run_id = make_run_id("holdout" if not args.rehearse else "holdout_rehearsal")
    write_manifest(ROOT / "logs" / "holdout" / f"manifest_{run_id}.json", run_id=run_id, stage="99_final_holdout",
                   seeds=seeds.as_dict(), extra={"prereg_commit": commit, "eval_split": plan.eval_split,
                                                 "replay": check["variants"]}, root=ROOT)
    t2.evaluate_test(plan, cfg, log, sanity, close=ext.close, states=ext.states, final_holdout=not args.rehearse)
    res = t2.analyse(plan, cfg, log)
    report = out_dir / "final_report.md" if args.rehearse else ROOT / "reports" / "final_report.md"
    build_final_report(cfg, ROOT, report, holdout_dir=out_dir, holdout_res=res, rehearsal=args.rehearse)
    log.info("wrote %s", report)
    if not args.rehearse:
        entry = holdout_decisions_entry(plan, res, _git("rev-parse", "HEAD"), check)
        log.info("holdout gate decisions %s DECISIONS.md", "appended to" if append_decisions(ROOT, entry, marker="## Gate decisions (holdout)") else "already in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
