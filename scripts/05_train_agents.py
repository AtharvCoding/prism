#!/usr/bin/env python
"""Tier 2: SAC policy ablation, end to end. Spec §12-§14, build step 4c.

    python scripts/05_train_agents.py            # = make tier2
    python scripts/05_train_agents.py --smoke    # tiny pipeline check; reads the VALIDATION split, never test

Stages, in this order (each is resumable; finished runs and finished stages are skipped):

    0 contract   the pre-registration is committed and unmodified; input state files match its hashes
    1 sanity     degenerate task + beats-random, train/validation only; ABORT before anything else if it fails
    2 tune       variants x grid x tuning seeds, validation-selected checkpoints
    3 freeze     the best configuration per variant, written to disk and never overwritten
    4 final      variants x final seeds at the frozen configuration
    5 evaluate   ONE evaluation of every final checkpoint and every benchmark on the test split
    6 report     bootstrap, gates, deflated Sharpe, episodes -> reports/tier2_report.md; gate decisions -> DECISIONS.md

Logs: ``logs/tier2/``. The holdout is never read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prism.agents import tier2 as t2  # noqa: E402
from prism.agents.sac import SacSettings  # noqa: E402
from prism.config import load_config  # noqa: E402
from prism.reporting.tier2_report import append_decisions, build_tier2_report, decisions_entry, sanity_markdown  # noqa: E402
from prism.utils.hashing import hash_object, make_run_id, sha256_file, write_manifest  # noqa: E402
from prism.utils.logging import configure_logging  # noqa: E402
from prism.utils.prereg import preregistered_hashes  # noqa: E402

PREREG_T2 = "reports/tables/preregistration_tier2.md"
STAGES = ("contract", "sanity", "tune", "freeze", "final", "evaluate", "report")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def check_contract(plan: t2.Tier2Plan, log) -> str:  # noqa: ANN001
    """The pre-registration exists in git, is unmodified, and the state files match the hashes it lists."""
    if plan.smoke:
        log.info("smoke run: contract check skipped (nothing here is a result)")
        return "smoke"
    try:
        _git("ls-files", "--error-unmatch", PREREG_T2)
    except subprocess.CalledProcessError:
        raise SystemExit(f"{PREREG_T2} is not committed; commit it before any run (it must precede the test split)")
    if _git("status", "--porcelain", "--", PREREG_T2):
        raise SystemExit(f"{PREREG_T2} has uncommitted changes; amendments are appended and committed first")
    commit = _git("log", "-1", "--format=%H", "--", PREREG_T2)
    listed = preregistered_hashes(ROOT, PREREG_T2)
    if not listed:
        raise SystemExit(f"{PREREG_T2} lists no input hashes")
    bad = [name for name, digest in listed.items() if sha256_file(ROOT / "data" / "processed" / name) != digest]
    if bad:
        raise SystemExit(f"input files differ from the pre-registered hashes: {bad}; an amendment is required first")
    log.info("contract: %s at %s; %d input hashes verified", PREREG_T2, commit[:10], len(listed))
    return commit


def run_sanity_stage(plan: t2.Tier2Plan, cfg, log) -> dict:  # noqa: ANN001
    from prism.agents.sanity import DEFAULT_SANITY_CONFIG, run_sanity

    base_settings = SacSettings(**cfg.tier2.sac.model_dump())      # the smoke plan's reduced settings never apply to the gates
    main_dir = plan.out_dir if not plan.smoke else plan.out_dir.parent / cfg.tier2.output_dir.rsplit("/", 1)[-1]
    path = main_dir / "sanity.json"
    want = {"settings": asdict(base_settings), "cfg_id": DEFAULT_SANITY_CONFIG.cfg_id, "steps": cfg.tier2.training.steps,
            "eval_every": cfg.tier2.training.eval_every,
            "sanity_config_hash": hash_object(cfg.tier2.sanity.model_dump())}
    if path.exists():
        rec = json.loads(path.read_text())
        if (rec.get("settings") == want["settings"] and rec["config"]["cfg_id"] == want["cfg_id"]
                and rec.get("steps") == want["steps"] and rec.get("eval_every") == want["eval_every"]
                and rec.get("sanity_config_hash") == want["sanity_config_hash"]):
            log.info("sanity record found (%s): %s", path, "PASS" if rec["passed"] else "FAIL")
            return rec
        log.info("sanity record is for different settings; re-running the gates")
    elif plan.smoke:
        raise SystemExit("smoke needs a passing sanity.json from the real run first (python scripts/05_train_agents.py --stage sanity)")
    log.info("running sanity gates (train/validation only)")
    rec = run_sanity(ROOT, cfg, settings=base_settings, workers=min(plan.workers, 6))
    rec["sanity_config_hash"] = want["sanity_config_hash"]
    rec["eval_every"] = want["eval_every"]
    main_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=float))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--stage", choices=("all", *STAGES), default="all", help="run one stage (earlier ones are verified, not repeated)")
    ap.add_argument("--smoke", action="store_true", help="tiny budget; evaluates the VALIDATION split into data/processed/tier2_smoke")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    plan = t2.Tier2Plan.from_config(cfg, ROOT)
    if args.smoke:
        plan = plan.smoke_plan()
    if args.workers:
        plan = replace(plan, workers=args.workers)
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    plan.log_dir.mkdir(parents=True, exist_ok=True)
    log = configure_logging(log_file=plan.log_dir / "tier2.log")
    run_id = make_run_id("tier2" + ("_smoke" if args.smoke else ""))
    want = STAGES if args.stage == "all" else (args.stage,)
    log.info("tier2 %s: stages %s, %d workers, %d steps/run, out %s", run_id, list(want), plan.workers, plan.steps, plan.out_dir)

    # The sanity gates read train/validation only, so they may run before the pre-registration is committed
    # (they are part of what it records); every later stage requires the committed contract.
    commit = check_contract(plan, log) if want != ("sanity",) else "n/a (sanity only)"

    sanity = None
    if any(s in want for s in ("sanity", "tune", "freeze", "final", "evaluate", "report")):
        sanity = run_sanity_stage(plan, cfg, log)
        if not plan.smoke:
            (ROOT / "reports" / "tables" / "tier2_sanity.md").write_text(sanity_markdown(sanity))
        if not sanity["passed"]:
            log.error("SANITY GATES FAILED (degenerate task: %s; beats random: %s). Aborting before the test split.",
                      sanity["degenerate_task"]["passed"], sanity["beats_random"]["passed"])
            return 3
    if "sanity" in want and len(want) == 1:
        return 0

    write_manifest(plan.log_dir / f"manifest_{run_id}.json", run_id=run_id, stage="tier2", config_hash=hash_object(cfg.model_dump(mode="json", exclude={"root"})),
                   seeds={"master": plan.master_seed, "tuning": list(plan.tuning_seeds), "final": list(plan.final_seeds)},
                   extra={"plan_steps": plan.steps, "workers": plan.workers, "smoke": plan.smoke, "prereg_commit": commit,
                          "eval_split": plan.eval_split}, root=ROOT)

    if "tune" in want:
        t2.run_pool(plan, t2.tuning_specs(plan), log)
    if "freeze" in want or "final" in want or "evaluate" in want or "report" in want:
        frozen = t2.freeze_configs(plan, log)
    if "final" in want:
        t2.run_pool(plan, t2.final_specs(plan, frozen), log)
    if "evaluate" in want:
        t2.evaluate_test(plan, cfg, log, sanity)
    if "report" in want:
        res = t2.analyse(plan, cfg, log)
        out = plan.out_dir / "tier2_report.md" if plan.smoke else ROOT / "reports" / "tier2_report.md"
        build_tier2_report(plan, cfg, res, frozen, sanity, out)
        log.info("wrote %s", out)
        if not plan.smoke:
            added = append_decisions(ROOT, decisions_entry(plan, res, frozen, subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
            log.info("gate decisions %s DECISIONS.md", "appended to" if added else "already present in")
    log.info("tier2 stages %s finished", list(want))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
