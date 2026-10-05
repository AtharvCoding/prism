#!/usr/bin/env python
"""Precompute the dashboard's artifacts. DASHBOARD.md §5; DECISIONS.md D-045 to D-047.

    python scripts/10_dashboard_data.py                    # = make dashboard-data: every stage that is NOT gated
    python scripts/10_dashboard_data.py --check            # verify the committed artifacts, write nothing
    python scripts/10_dashboard_data.py --stage <name>     # one stage

    PRISM_ALLOW_HOLDOUT=1 python scripts/10_dashboard_data.py --stage holdout-replay --i-am-sure
                                                           # the one gated stage; typed by hand (Amendment 1)

Stages (each verifies its sources against dashboard/frozen_sources.sha256 and writes only under dashboard/artifacts
and data/live):

    stored          the stored tables, byte for byte; facts.json; every headline row checked against final_report.md
    derived         per-seed deflated Sharpe, regime and latent series, the SPY line, K selection; each checked against
                    a stored table
    replay-test     the 40 frozen agents and 6 benchmarks replayed on the TEST split to record weekly weights; every
                    daily series must equal data/processed/tier2/eval_daily.parquet to 1e-9
    folds-test      the walk-forward re-run to the test split's end (about 6 minutes) for the per-fold HMM parameters;
                    must reproduce the stored states to 1e-9. Also rehearses persisting and verifying the live models
    live-check      the live path's functions against stored results, on the test split: the frozen (rehearsal) models
                    re-derive the stored state rows of their last month; the continued-episode rollout reproduces the
                    recorded weekly weights and the final decision's observation for all 40 agents
    holdout-replay  GATED. The same two things on the HOLDOUT window, plus the frozen last-fold models for the live view.
                    Needs --i-am-sure and PRISM_ALLOW_HOLDOUT=1, the committed Amendment 1, and is logged in
                    reports/logs/holdout_access.jsonl. It computes no statistic and writes nothing to data/processed.

None of the first four stages reads a holdout row from the raw data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from prism import dashboard_data as dd  # noqa: E402
from prism.config import load_config  # noqa: E402
from prism.utils.hashing import sha256_file  # noqa: E402
from prism.utils.logging import configure_logging  # noqa: E402

UNGATED = ("stored", "derived", "replay-test", "folds-test", "live-check")
REASON = ("dashboard: descriptive replay of the frozen agents to record weekly weights and persist the last-fold models; "
          "no statistic is computed (preregistration_holdout.md Amendment 1)")


def _git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def _summaries(cfg) -> tuple[str, int, dict]:  # noqa: ANN001
    hmm = json.loads((cfg.path("processed") / "hmm_summary.json").read_text())["walkforward"]
    enc = json.loads((cfg.path("processed") / "encoder_summary.json").read_text())["selected"]
    return hmm["specification"], int(hmm["k"]), enc


def _write_weights(out: Path, window: str, res: dict) -> dict:
    files = {}
    for name in ("agents", "benchmarks"):
        rel = f"weights/{window}_{name}.parquet"
        files[rel] = {"kind": "replay", "sha256": dd.write_frame(res[name], out / rel)}
    rel = f"weights/{window}_check.json"
    files[rel] = {"kind": "check", "sha256": dd._write_json(out / rel, res["check"])}
    rel = f"weights/{window}_last_observations.json"          # what each agent saw at its final recorded decision (the what-if lab's input)
    files[rel] = {"kind": "replay", "sha256": dd._write_json(out / rel, res["last_observations"])}
    return files


def stage_replay_test(cfg, out: Path, log) -> None:  # noqa: ANN001
    from prism import dashboard_replay as dr
    from prism.agents import tier2 as t2
    from prism.agents.data import load_close

    dr.verify_agents(ROOT)
    baseline = dd.read_baseline(ROOT)
    plan = t2.Tier2Plan.from_config(cfg, ROOT)
    daily = pd.read_parquet(dd._checked(ROOT, f"{dd.PROCESSED}/tier2/eval_daily.parquet", baseline))
    meta = json.loads(dd._checked(ROOT, f"{dd.PROCESSED}/tier2/eval_done.json", baseline).read_text())
    res = dr.replay_window(cfg, plan, "test", load_close(cfg, "test"), None, daily, meta)
    dd.write_manifest(out, _write_weights(out, "test", res), sha256_file(ROOT / dd.BASELINE))
    log.info("replay-test: %d series reproduce the stored daily returns (max |diff| %.2e); %d decisions, %s .. %s",
             res["check"]["series"], res["check"]["max_abs_diff"], res["check"]["decisions"], res["check"]["first_decision"],
             res["check"]["last_decision"])


def _save_and_verify_models(cfg, ext, features_b: pd.DataFrame, target: Path, stored: dict, log) -> tuple[dict, dict]:  # noqa: ANN001
    """Persist the last fold's models to a scratch directory, reproduce the stored states from them, then move into place."""
    from prism import live
    from prism.holdout import C2_SIGNAL_COLUMN

    spec, k, enc = _summaries(cfg)
    scratch = target.with_name(target.name + ".tmp")
    if scratch.exists():
        shutil.rmtree(scratch)
    manifest = live.save_models(cfg, ext, scratch, spec=spec, k=k, encoder_selection=enc, threshold_column=C2_SIGNAL_COLUMN)
    models = live.load_models(scratch, expected=manifest)
    check = live.verify_models(models, ext.features_a, features_b, stored)
    if target.exists():
        shutil.rmtree(target)
    scratch.rename(target)
    log.info("live models (%s): HMM fold %d fit to %s, encoder fold %d fit to %s; reproduce the stored states: %s", target.name,
             manifest["hmm"]["fold"], manifest["hmm"]["fit_end"], manifest["encoder"]["fold"], manifest["encoder"]["fit_end"],
             {m: f"{check[m]['max_abs_diff']:.1e}" for m in ("hmm", "encoder", "threshold", "state_scaler")})
    return manifest, check


def _features_b(cfg, raw: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:  # noqa: ANN001
    from prism.features.build import build_features
    from prism.holdout import _load_set, _prune

    fs = build_features(raw.loc[:end], cfg, "B")
    return _load_set(cfg, "B", _prune(cfg, fs, "B", end), fs).frame


def stage_folds_test(cfg, out: Path, log) -> None:  # noqa: ANN001
    from prism import dashboard_replay as dr
    from prism import live
    from prism.data.loaders import assert_not_holdout, load_snapshot
    from prism.features.build import make_raw_frame
    from prism.splits import build_split_plan

    end = build_split_plan(cfg)["test"].declared_end
    snap = load_snapshot(cfg)
    raw = make_raw_frame(snap.close, snap.volume, snap.macro).loc[:end]
    assert_not_holdout(cfg, raw.index, context="dashboard folds-test")
    stored = dr.stored_states(ROOT, "test")
    ext, check = dr.walkforward(cfg, end, raw, stored)
    files = {}
    rel = "regimes/hmm_folds_test.csv"
    table = dr.hmm_fold_table(ext)
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / rel, index=False, lineterminator="\n")
    files[rel] = {"kind": "derived", "sha256": sha256_file(out / rel), "folds": int(len(table))}
    manifest, mcheck = _save_and_verify_models(cfg, ext, _features_b(cfg, raw, end), ROOT / (live.MODELS_DIR + "_rehearsal"), stored, log)
    rel = "live/rehearsal_check.json"
    files[rel] = {"kind": "check", "sha256": dd._write_json(out / rel, {"walkforward": check, "models": manifest, "reproduce": mcheck})}
    dd.write_manifest(out, files, sha256_file(ROOT / dd.BASELINE))
    log.info("folds-test: %d HMM folds; the re-run walk-forward reproduces the stored states: %s", len(table),
             {v: f"{c['max_abs_diff']:.1e}" for v, c in check["variants"].items()})


def stage_live_check(cfg, out: Path, log) -> None:  # noqa: ANN001
    """Verify prism.live on real, already-exposed data. Reads nothing after the test split's end."""
    import numpy as np

    from prism import dashboard_replay as dr
    from prism import live
    from prism.agents import tier2 as t2
    from prism.agents.data import load_close
    from prism.agents.jobs import job_dir
    from prism.agents.sac import greedy_policy, load_agent
    from prism.data.loaders import assert_not_holdout, load_snapshot
    from prism.features.build import make_raw_frame
    from prism.splits import build_split_plan

    dr.verify_agents(ROOT)
    split = build_split_plan(cfg)["test"]
    snap = load_snapshot(cfg)
    raw = make_raw_frame(snap.close, snap.volume, snap.macro).loc[: split.declared_end]
    assert_not_holdout(cfg, raw.index, context="dashboard live-check")
    stored = dr.stored_states(ROOT, "test")
    models = live.load_models(ROOT / (live.MODELS_DIR + "_rehearsal"))
    fresh = live.live_states(cfg, raw, models, {v: [str(c) for c in f.columns] for v, f in stored.items()},
                             start=pd.Timestamp(models.hmm["apply_start"]))
    states_diff = {v: float(np.abs(f.to_numpy() - stored[v].loc[f.index].to_numpy()).max()) for v, f in fresh.items()}
    rows = {v: int(len(f)) for v, f in fresh.items()}
    if max(states_diff.values()) > dr.REPLAY_ATOL or min(rows.values()) == 0:
        raise dr.ReplayError(f"live_states does not reproduce the stored states: {states_diff}")

    plan = t2.Tier2Plan.from_config(cfg, ROOT)
    frozen = t2.load_frozen(plan)
    close = load_close(cfg, "test")
    recorded = pd.read_parquet(out / "weights" / "test_agents.parquet")
    last_obs = json.loads((out / "weights" / "test_last_observations.json").read_text())
    last_decision = pd.Timestamp(last_obs["decision_date"])
    worst = {"weights": 0.0, "pre_trade": 0.0, "turnover": 0.0, "observation": 0.0, "what_if_identity": 0.0}
    for v in plan.variants:
        cols = last_obs["columns"][v]
        for s in plan.final_seeds:
            policy = greedy_policy(load_agent(job_dir(plan.runs, "final", v, frozen[v].cfg_id, s) / "best.zip"))
            want = recorded[(recorded.variant == v) & (recorded.seed == s)].reset_index(drop=True)
            full = live.rollout(cfg, stored[v], close, policy, start=split.effective_start, end=split.effective_end)
            got = full["decisions"]
            if not (got.decision_date.to_numpy() == want.decision_date.to_numpy()).all():
                raise dr.ReplayError(f"{v} seed {s}: the live rollout's decision dates differ from the recorded ones")
            for key, prefix in (("weights", "w_"), ("pre_trade", "p_")):
                c = [x for x in want.columns if x.startswith(prefix)]
                worst[key] = max(worst[key], float(np.abs(got[c].to_numpy() - want[c].to_numpy()).max()))
            worst["turnover"] = max(worst["turnover"], float(np.abs(got.turnover.to_numpy() - want.turnover.to_numpy()).max()))
            # ending the episode at the final decision's close: the terminal observation is that decision's observation
            at_decision = live.rollout(cfg, stored[v], close, policy, start=split.effective_start, end=last_decision)["latest"]
            seen = np.asarray(last_obs["observations"][f"{v}|s{s}"], dtype="float32")
            worst["observation"] = max(worst["observation"], float(np.abs(at_decision["observation"] - seen).max()))
            target = want[want.decision_date == last_decision][[x for x in want.columns if x.startswith("w_")]].to_numpy()[0]
            worst["weights"] = max(worst["weights"], float(np.abs(at_decision["weights"].to_numpy() - target).max()))
            same = live.what_if_weights(cfg, policy, seen, cols, None)
            worst["what_if_identity"] = max(worst["what_if_identity"], float(np.abs(same - target).max()))
    if max(worst.values()) > dr.REPLAY_ATOL:
        raise dr.ReplayError(f"the live rollout does not reproduce the recorded replay: {worst}")
    check = {"window": "test", "atol": dr.REPLAY_ATOL, "live_states": {"start": models.hmm["apply_start"], "rows": rows, "max_abs_diff": states_diff},
             "rollout": {"agents": len(plan.variants) * len(plan.final_seeds), "decisions": int(recorded.decision_date.nunique()),
                         "max_abs_diff": worst}, "ok": True}
    files = {"live/live_check.json": {"kind": "check", "sha256": dd._write_json(out / "live" / "live_check.json", check)}}
    dd.write_manifest(out, files, sha256_file(ROOT / dd.BASELINE))
    log.info("live-check: live_states reproduce the stored states %s; rollout reproduces the recorded weights %s", states_diff, worst)


def stage_holdout_replay(cfg, out: Path, log, *, sure: bool) -> None:  # noqa: ANN001
    from prism import dashboard_replay as dr
    from prism import holdout as H
    from prism import live
    from prism.agents import tier2 as t2
    from prism.data.loaders import HOLDOUT_ENV_VAR, load_holdout, load_snapshot
    from prism.features.build import make_raw_frame
    from prism.splits import build_split_plan

    # 0 gates, before any file is read
    if not sure:
        raise SystemExit("the holdout replay needs --i-am-sure (and PRISM_ALLOW_HOLDOUT=1); it is typed by hand")
    if os.environ.get(HOLDOUT_ENV_VAR) != "1":
        raise SystemExit(f"set {HOLDOUT_ENV_VAR}=1 as well as --i-am-sure (two independent gates)")
    # 1 contract: the amendment is committed, unmodified and says what this run does; every pinned file matches
    try:
        _git("ls-files", "--error-unmatch", dr.PREREG_HOLDOUT)
    except subprocess.CalledProcessError:
        raise SystemExit(f"{dr.PREREG_HOLDOUT} is not committed")
    if _git("status", "--porcelain", "--", dr.PREREG_HOLDOUT):
        raise SystemExit(f"{dr.PREREG_HOLDOUT} has uncommitted changes")
    if dr.AMENDMENT_MARK not in (ROOT / dr.PREREG_HOLDOUT).read_text(encoding="utf-8"):
        raise SystemExit(f"{dr.PREREG_HOLDOUT} has no Amendment 1; the replay is not pre-registered")
    dr.verify_agents(ROOT)
    baseline = dd.read_baseline(ROOT)
    plan_split = build_split_plan(cfg)
    plan = t2.Tier2Plan.from_config(cfg, ROOT)
    plan = replace(plan, out_dir=plan.out_dir.parent / "holdout", runs_dir=plan.out_dir, eval_split="holdout")
    daily = pd.read_parquet(dd._checked(ROOT, f"{dd.PROCESSED}/holdout/eval_daily.parquet", baseline))
    meta = json.loads(dd._checked(ROOT, f"{dd.PROCESSED}/holdout/eval_done.json", baseline).read_text())
    dd._checked(ROOT, f"{dd.PROCESSED}/holdout/states_extended.parquet", baseline)
    stored = dr.stored_states(ROOT, "holdout")

    snap = load_snapshot(cfg)
    raw = make_raw_frame(snap.close, snap.volume, snap.macro)
    load_holdout(cfg, raw, final=True, reason=REASON)                      # logs the access permanently
    end = plan_split["holdout"].declared_end
    # 2 inputs: the walk-forward must reproduce the holdout run's states on every row, and the Tier 2 states to 2023
    ext, check = dr.walkforward(cfg, end, raw, stored)
    tier2_check = H.replay_check(cfg, ext, up_to=plan_split["test"].declared_end, atol=dr.REPLAY_ATOL)
    if not tier2_check["ok"]:
        raise SystemExit("REPLAY CHECK FAILED against the Tier 2 state files; nothing is recorded")
    # 3 the replay; raises before anything is written if a single series differs
    res = dr.replay_window(cfg, plan, "holdout", ext.close, stored, daily, meta, final_holdout=True)
    # 4 record
    files = _write_weights(out, "holdout", res)
    first_holdout_fold = sum(f.fold.apply_start <= plan_split["test"].declared_end for f in ext.hmm.folds)
    rel = "regimes/hmm_folds_holdout.csv"
    table = dr.hmm_fold_table(ext, first_fold=first_holdout_fold)
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / rel, index=False, lineterminator="\n")
    files[rel] = {"kind": "derived", "sha256": sha256_file(out / rel), "folds": int(len(table))}
    manifest, mcheck = _save_and_verify_models(cfg, ext, _features_b(cfg, raw, end), ROOT / live.MODELS_DIR, stored, log)
    files["live/models_manifest.json"] = {"kind": "manifest", "sha256": dd._write_json(out / "live" / "models_manifest.json", manifest)}
    files["live/models_check.json"] = {"kind": "check", "sha256": dd._write_json(
        out / "live" / "models_check.json", {"walkforward": check, "tier2_states": tier2_check["variants"], "reproduce": mcheck})}
    dd.write_manifest(out, files, sha256_file(ROOT / dd.BASELINE))
    log.info("holdout-replay: %d series reproduce the stored daily returns (max |diff| %.2e); %d decisions, %s .. %s; %d HMM folds",
             res["check"]["series"], res["check"]["max_abs_diff"], res["check"]["decisions"], res["check"]["first_decision"],
             res["check"]["last_decision"], len(table))
    log.info("nothing was written to data/processed; commit dashboard/artifacts")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--stage", choices=[*UNGATED, "holdout-replay", "all"], default="all", help="'all' runs every stage that is not gated")
    ap.add_argument("--check", action="store_true", help="verify the artifacts against the manifest and the final report; write nothing")
    ap.add_argument("--i-am-sure", action="store_true", help="required, with PRISM_ALLOW_HOLDOUT=1, for --stage holdout-replay")
    args = ap.parse_args(argv)
    log = configure_logging()
    out = ROOT / dd.ARTIFACTS
    if args.check:
        manifest = dd.verify_artifacts(out)
        check = dd.verify_against_report(out, ROOT / dd.REPORT)
        log.info("artifacts match the manifest (%d files, %s); %d rows found in %s", len(manifest["files"]), manifest["hash"][:12],
                 check["rows_checked"], dd.REPORT)
        return 0
    cfg = load_config(ROOT / args.config)
    if args.stage == "holdout-replay":
        stage_holdout_replay(cfg, out, log, sure=args.i_am_sure)
        dd.verify_artifacts(out)
        return 0
    if args.i_am_sure:
        ap.error("--i-am-sure only applies to --stage holdout-replay")
    for stage in (UNGATED if args.stage == "all" else (args.stage,)):
        if stage == "stored":
            res = dd.build_stored(cfg, ROOT, out)
            log.info("stored: %d rows checked against %s", res["report_check"]["rows_checked"], dd.REPORT)
        elif stage == "derived":
            res = dd.build_derived(cfg, ROOT, out)
            log.info("derived: %s", json.dumps(res["checks"]))
        elif stage == "replay-test":
            stage_replay_test(cfg, out, log)
        elif stage == "folds-test":
            stage_folds_test(cfg, out, log)
        elif stage == "live-check":
            stage_live_check(cfg, out, log)
    manifest = dd.verify_artifacts(out)
    log.info("dashboard/artifacts: %d files, manifest %s", len(manifest["files"]), manifest["hash"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
