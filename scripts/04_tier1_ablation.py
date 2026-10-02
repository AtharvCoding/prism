#!/usr/bin/env python
"""Run the Tier 1 representation ablation and the allocator comparison. Spec §13.1, §16 step 4a.

    PYTHONHASHSEED=0 python scripts/04_tier1_ablation.py     # == `make tier1`

Runs EXACTLY what ``reports/tables/preregistration.md`` fixes, in the order it
fixes, so that the test split is touched last:

  0. verify every input file's SHA-256 against the pre-registration (abort on mismatch)
  1. alpha selection on validation folds only                      (no test rows)
  2. gamma calibration for the mean-variance leg on validation     (no test rows)
  3. FREEZE: alphas and gamma are written to ``phase1.json`` before any test-split step
  4. test-split walk-forward probes, paired differences, gates
  5. allocator legs, benchmarks, costs, episodes, deflated Sharpe
  6. regenerate the report from the stored results (``prism.reporting.report``)

The holdout is never read: the raw panel is truncated to the end of the test
split BEFORE any feature or target is built, and every frame is asserted clear
of the holdout window.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prism.analysis.bootstrap import bootstrap_paths, path_ci  # noqa: E402
from prism.analysis.episodes import episode_performance, find_drawdown_episodes  # noqa: E402
from prism.analysis.significance import deflated_sharpe, expected_max_sharpe, per_period_sharpe  # noqa: E402
from prism.backtest import benchmarks as bm  # noqa: E402
from prism.backtest.metrics import compute_metrics  # noqa: E402
from prism.config import Config, load_config  # noqa: E402
from prism.data.loaders import assert_not_holdout, load_snapshot  # noqa: E402
from prism.features.build import build_features, make_raw_frame  # noqa: E402
from prism.features.targets import build_targets  # noqa: E402
from prism.probes import allocator as al  # noqa: E402
from prism.probes import tier1 as t1  # noqa: E402
from prism.splits import build_split_plan  # noqa: E402
from prism.utils.calendar import rebalance_dates  # noqa: E402
from prism.utils.hashing import hash_object, make_run_id, sha256_file, write_manifest  # noqa: E402
from prism.utils.prereg import PREREG, preregistered_hashes  # noqa: E402
from prism.utils.logging import configure_logging, get_logger  # noqa: E402
from prism.utils.seeding import seed_everything  # noqa: E402

_log = get_logger(__name__)

PREREG_SEED = 20260101
REGIME_VARIANTS = ("V3", "V4", "C2", "C3", "O1")
REPORTABLE = ("V1", "V1p", "V2", "V3", "V4", "C1", "C2", "C3")
DIAGNOSTIC = ("O1",)
LEGS = ("VT", "RVT", "MV")
COSTS_BPS = (0.0, 5.0, 10.0, 20.0)
HEADLINE_BPS = 5.0
CONTRASTS = [("V2", "V1p"), ("V2", "C1"), ("V3", "C2"), ("V3", "C3"), ("V4", "V2")]
N_BOOT = 2000


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------- #
# 0. inputs
# --------------------------------------------------------------------------- #
#: O1's smoothed-HMM refit is not byte-reproducible (hmmlearn forward-backward
#: differs between runs at ~2e-11), and the original file was overwritten.
#: Pre-registration Amendment 1 replaces its exact hash with this check: O1 must
#: match a frozen reference copy (whose own SHA-256 IS verified exactly) to
#: within O1_TOLERANCE. O1 is diagnostic-only and in no gate.
O1_NAME = "states/O1.parquet"
O1_REFERENCE = "states/O1.amendment1.parquet"
O1_TOLERANCE = 1e-9


def _verify_o1(processed: Path) -> str:
    cur, ref = pd.read_parquet(processed / O1_NAME), pd.read_parquet(processed / O1_REFERENCE)
    if not (cur.index.equals(ref.index) and list(cur.columns) == list(ref.columns)):
        raise RuntimeError("O1 differs from its Amendment 1 reference in index or columns")
    worst = float(np.abs(cur.to_numpy("float64") - ref.to_numpy("float64")).max())
    if not worst <= O1_TOLERANCE:
        raise RuntimeError(f"O1 differs from its Amendment 1 reference by {worst:.3e} > {O1_TOLERANCE:.0e}")
    return sha256_file(processed / O1_NAME)


def verify_inputs(cfg: Config) -> dict[str, str]:
    """Abort unless every input matches the pre-registration.

    Exact SHA-256 for every listed file except O1 (see :data:`O1_NAME`), which is
    checked numerically against its frozen Amendment 1 reference.
    """
    expected = preregistered_hashes(cfg.root)
    if len(expected) < 14:
        raise RuntimeError(f"found only {len(expected)} input hashes in {PREREG}; expected 14")
    if O1_REFERENCE not in expected:
        raise RuntimeError(f"{PREREG} lists no hash for {O1_REFERENCE} (Amendment 1)")
    processed = cfg.path("processed")
    mismatched = []
    for name, want in expected.items():
        got = _verify_o1(processed) if name == O1_NAME else sha256_file(processed / name)
        if name == O1_NAME:
            expected[name] = got
        elif got != want:
            mismatched.append((name, want[:12], got[:12]))
    if mismatched:
        raise RuntimeError(
            "input files differ from the pre-registered hashes — the state variants were rebuilt "
            f"after pre-registration, which requires an amendment first: {mismatched}"
        )
    return expected


def preregistration_commit(cfg: Config) -> str | None:
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", PREREG],
            cwd=cfg.root, capture_output=True, text=True, check=True,
        ).stdout.split()
        return out[-1] if out else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# --------------------------------------------------------------------------- #
# replicate statistics on daily returns (shared bootstrap paths)
# --------------------------------------------------------------------------- #
def _stat_point_and_reps(r: np.ndarray, paths: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
    def stats(R: np.ndarray) -> dict[str, np.ndarray]:
        R = np.atleast_2d(R)
        n = R.shape[1]
        mean, sd = R.mean(axis=1), R.std(axis=1, ddof=1)
        curve = np.concatenate([np.ones((R.shape[0], 1)), np.cumprod(1.0 + R, axis=1)], axis=1)
        mdd = (curve[:, 1:] / np.maximum.accumulate(curve, axis=1)[:, 1:] - 1.0).min(axis=1)
        q = np.quantile(R, 0.05, axis=1)
        tail = R <= q[:, None]
        cvar = (R * tail).sum(axis=1) / tail.sum(axis=1)
        ann = curve[:, -1] ** (252.0 / n) - 1.0
        return {"sharpe": mean / sd * np.sqrt(252.0), "max_drawdown": mdd, "cvar_95": cvar, "annualised_return": ann}

    point = {k: float(v[0]) for k, v in stats(r).items()}
    reps = stats(r[paths])
    return {k: (point[k], reps[k]) for k in point}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-report", action="store_true", help="store results but do not rebuild the report")
    parser.add_argument("--skip-auc", action="store_true", help="DEVELOPMENT ONLY: skip the secondary AUC probe")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = configure_logging()
    seeds = seed_everything(cfg.data.seeds.master)
    if cfg.data.seeds.master != PREREG_SEED:
        raise RuntimeError("master seed differs from the pre-registered 20260101")
    out_dir = cfg.path("processed") / "tier1"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------ 0. inputs
    input_hashes = verify_inputs(cfg)
    log.info("input hashes verified against the pre-registration (%d files)", len(input_hashes))
    plan = build_split_plan(cfg)
    val_start, val_end = plan["val"].effective_start, plan["val"].effective_end
    test_start, test_end = plan["test"].effective_start, plan["test"].effective_end
    train_start, train_end = plan["train"].effective_start, plan["train"].effective_end
    non_holdout_end = plan["test"].declared_end

    snapshot = load_snapshot(cfg)
    raw = make_raw_frame(snapshot.close, snapshot.volume, snapshot.macro).loc[:non_holdout_end]
    assert_not_holdout(cfg, raw.index, context="tier1 raw panel")
    fs = build_features(raw, cfg, "B")
    close = fs.close.loc[:non_holdout_end]
    assert_not_holdout(cfg, close.index, context="tier1 close")
    sleeve = [t for t in cfg.data.universes["A"].equity_sectors if t in close.columns]
    targets = build_targets(close, benchmark=cfg.data.universes["A"].benchmark, sleeve=sleeve)

    states = {
        v: pd.read_parquet(cfg.path("processed") / "states" / f"{v}.parquet")
        for v in (*REPORTABLE, *DIAGNOSTIC)
    }
    for v, frame in states.items():
        assert_not_holdout(cfg, frame.index, context=f"state {v}")
    common = t1.common_evaluation_index(states, targets)
    assert_not_holdout(cfg, common, context="common index")
    test_idx = common[(common >= test_start) & (common <= test_end)]
    log.info("common index: %d sessions (%s..%s); test split %d sessions", len(common),
             common[0].date(), common[-1].date(), len(test_idx))

    embargo = cfg.data.splits.embargo_days
    val_folds = t1.make_folds(
        common, pd.Timestamp(year=cfg.tier1.probe.validation_first_apply_year, month=1, day=1), val_end, embargo
    )
    test_folds = t1.make_folds(common, test_start, test_end, embargo)
    assert [f.apply_start.year for f in val_folds] == [2015, 2016, 2017, 2018], val_folds
    assert [f.apply_start.year for f in test_folds] == [2019, 2020, 2021, 2022, 2023], test_folds
    alpha_grid = list(cfg.tier1.probe.alpha_grid)

    # ------------------------------------------------------------------ 1. alpha on validation
    validation: dict[str, t1.ValidationResult] = {}
    for v, frame in states.items():
        t0 = time.time()
        validation[v] = t1.select_alphas(v, frame, targets, common, val_folds, alpha_grid)
        log.info("[validation] %-4s alpha=%s (%.1fs)", v, validation[v].alpha, time.time() - t0)
        if validation[v].alpha_at_edge:
            log.warning("[validation] %s selected alpha at the grid edge: %s", v, validation[v].alpha_at_edge)

    # ------------------------------------------------------------------ 2. gamma on validation
    params = al.AllocatorParams(
        risky=tuple(cfg.data.allocatable["B"]),
        equity_sectors=tuple(cfg.data.universes["A"].equity_sectors),
        cap=cfg.data.allocation.weight_max,
    )
    val_sessions = close.index[(close.index >= val_start) & (close.index <= val_end)]
    val_decisions = rebalance_dates(val_sessions, "weekly", "FRI")
    inputs_val, _ = al.prepare_inputs(close, params, val_decisions)
    sigma_val = validation["V1"].validation_predictions["fwd_vol_5"].reindex(val_decisions)
    gamma, gamma_table = al.calibrate_gamma(sigma_val, inputs_val, params)
    log.info("[validation] gamma=%s (V1 mean risky exposure target %.2f)\n%s",
             gamma, params.gamma_target_exposure, gamma_table.to_string(index=False))

    # ------------------------------------------------------------------ 3. FREEZE
    phase1 = {
        "frozen_utc": pd.Timestamp.now("UTC").isoformat(),
        "alpha": {v: validation[v].alpha for v in validation},
        "alpha_at_edge": {v: validation[v].alpha_at_edge for v in validation},
        "gamma": gamma,
        "gamma_calibration": gamma_table.to_dict("records"),
        "alpha_grid": alpha_grid,
        "validation_mse": {v: validation[v].validation_mse.to_dict() for v in validation},
        "input_hashes": input_hashes,
        "preregistration_commit": preregistration_commit(cfg),
    }
    (out_dir / "phase1.json").write_text(json.dumps(phase1, indent=2, default=_jsonable), encoding="utf-8")
    log.info("FROZEN: alphas and gamma written to %s; the test split is touched from here on", out_dir / "phase1.json")

    # ------------------------------------------------------------------ 4. test-split probes
    probes: dict[str, t1.VariantProbe] = {}
    for v, frame in states.items():
        t0 = time.time()
        probes[v] = t1.predict_test_walkforward(validation[v], frame, targets, common, test_folds)
        log.info("[test] %-4s walk-forward done (%.1fs)", v, time.time() - t0)
    ref_index = probes["V1"].predictions.index
    assert all(p.predictions.index.equals(ref_index) for p in probes.values()), "variants scored on different days"
    assert ref_index.equals(test_idx), "predictions do not tile the common test index"
    y_test = targets.loc[ref_index, list(t1.ALL_TARGETS)]
    n = len(ref_index)
    paths = {b: bootstrap_paths(n, b, N_BOOT, PREREG_SEED) for b in (10, 20, 40)}

    metrics_tbl = t1.variant_metric_table(probes, y_test, paths[20])
    gate_pairs = sorted({pair for pairs in t1.GATES.values() for pair in pairs})
    paired = {b: t1.paired_comparison_table(probes, y_test, gate_pairs, paths[b]) for b in (10, 20, 40)}
    diagnostic_paired = t1.paired_comparison_table(probes, y_test, [("O1", "V3"), ("O1", "V1")], paths[20])
    gates = {
        "paired_block20": t1.evaluate_gates(paired[20]),
        "spec_nonoverlap_block20": t1.evaluate_gates(paired[20], column="spec_verdict"),
        "paired_block10": t1.evaluate_gates(paired[10]),
        "paired_block40": t1.evaluate_gates(paired[40]),
    }
    for key, res in gates.items():
        log.info("[gates:%s] %s", key, {g: r["passes"] for g, r in res.items()})

    metrics_tbl.to_csv(out_dir / "probe_metrics.csv", index=False)
    for b in (10, 20, 40):
        paired[b].to_csv(out_dir / f"paired_block{b}.csv", index=False)
    diagnostic_paired.to_csv(out_dir / "paired_diagnostic_O1.csv", index=False)
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2, default=_jsonable), encoding="utf-8")
    pd.concat({v: p.predictions for v, p in probes.items()}, axis=1).to_parquet(out_dir / "probe_predictions.parquet")
    pd.concat({v: p.fit_mean for v, p in probes.items()}, axis=1).to_parquet(out_dir / "probe_fit_mean.parquet")

    # secondary: AUC on a binary high-vol label (quantile fixed on the TRAIN split only)
    auc_rows = []
    if not args.skip_auc:
        in_train = common[(common >= train_start) & (common <= train_end)]
        q80 = float(targets.loc[in_train, "fwd_vol_20"].quantile(0.80))
        label = (targets["fwd_vol_20"] > q80).astype(int)
        for v, frame in states.items():
            t0 = time.time()
            lp = t1.run_logistic_probe(v, frame, label, common, val_folds, test_folds, alpha_grid, paths[20])
            auc_rows.append(vars(lp) | {"q80": q80})
            log.info("[auc] %-4s AUC=%.3f [%.3f, %.3f] alpha=%g warnings=%d (%.1fs)", v, lp.auc, lp.auc_lo,
                     lp.auc_hi, lp.alpha, lp.n_convergence_warnings, time.time() - t0)
    pd.DataFrame(auc_rows).to_csv(out_dir / "auc.csv", index=False)

    # ------------------------------------------------------------------ 5. allocators
    sessions = close.loc[:test_end].index
    test_sessions = sessions[sessions >= test_start]
    decisions = rebalance_dates(test_sessions, "weekly", "FRI")
    decisions = decisions[[sessions.get_loc(d) + 1 < len(sessions) for d in decisions]]  # drop non-executing tail
    inputs, line_returns = al.prepare_inputs(close.loc[:test_end], params, decisions)

    weights: dict[str, pd.DataFrame] = {}
    for v in states:
        sigma_hat = probes[v].predictions["fwd_vol_5"].reindex(decisions)
        p = states[v]["state_1"] if v in REGIME_VARIANTS else None
        weights[f"{v}|VT"] = al.vol_target_weights(sigma_hat, inputs, params)
        weights[f"{v}|RVT"] = al.regime_vol_target_weights(sigma_hat, p, inputs, params)
        weights[f"{v}|MV"] = al.mean_variance_weights(sigma_hat, inputs, params, gamma)
    benchmark_weights = bm.all_benchmarks(inputs, params)
    for name, w in weights.items():
        al.check_weights(name, w, params)
    for name, w in benchmark_weights.items():
        al.check_weights(f"BM|{name}", w, params, cash_capped=(name == "MinVariance"), enforce_cap=(name != "SixtyForty"))

    # fail-loud integrity checks (preregistration §7)
    mv_dist = al.mv_pairwise_distance({v: weights[f"{v}|MV"] for v in REPORTABLE})
    al.assert_mv_weights_differ(mv_dist)
    for v in ("V1", "V1p", "V2", "C1"):
        pd.testing.assert_frame_equal(weights[f"{v}|RVT"], weights[f"{v}|VT"])
    for other in ("C3", "C2"):
        if np.allclose(weights["V3|RVT"].to_numpy(), weights[f"{other}|RVT"].to_numpy()):
            raise al.IntegrityError(f"V3 and {other} regime-conditional weights are identical; the control is vacuous")
    log.info("integrity checks passed; MV pairwise distance max %.4f", mv_dist.to_numpy().max())

    gross, turn, net = {}, {}, {c: {} for c in COSTS_BPS}
    for name, w in [*weights.items(), *((f"BM|{k}", v) for k, v in benchmark_weights.items())]:
        g, tr = al.simulate(w, line_returns)
        gross[name], turn[name] = g, tr
        for c in COSTS_BPS:
            net[c][name] = al.net_returns(g, tr, c)
    gross_df, turn_df = pd.DataFrame(gross), pd.DataFrame(turn)
    net_df = {c: pd.DataFrame(net[c]) for c in COSTS_BPS}
    assert gross_df.notna().all().all(), "strategies do not share one return calendar"
    daily = net_df[HEADLINE_BPS]
    n_days = len(daily)
    assert_not_holdout(cfg, daily.index, context="strategy returns")

    paths_d = {b: bootstrap_paths(n_days, b, N_BOOT, PREREG_SEED) for b in (20,)}
    reps: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
    rows = []
    for name in daily.columns:
        r = daily[name].to_numpy()
        reps[name] = _stat_point_and_reps(r, paths_d[20])
        m = compute_metrics(daily[name], gross_returns=gross_df[name], rebalances_per_year=52)
        row = {"strategy": name} | m.as_dict()
        row.pop("turnover", None)
        # one-way turnover per year, initiation included (a per-rebalance mean would be meaningless for buy-and-hold)
        row["turnover_annualised"] = float(turn_df[name].sum() / (n_days / 252.0))
        for stat in ("sharpe", "max_drawdown", "cvar_95", "annualised_return"):
            point, rep = reps[name][stat]
            lo, hi = np.quantile(rep, [0.025, 0.975])
            row[f"{stat}_ci_lo"], row[f"{stat}_ci_hi"] = float(lo), float(hi)
        # consistency of my replicate statistics with backtest.metrics, loudly
        for stat, ref in (("sharpe", m.sharpe), ("max_drawdown", m.max_drawdown), ("cvar_95", m.cvar_95)):
            if not np.isclose(reps[name][stat][0], ref, rtol=1e-6, atol=1e-9):
                raise al.IntegrityError(f"{name}: bootstrap point {stat} {reps[name][stat][0]} != metrics {ref}")
        rows.append(row)
    alloc_metrics = pd.DataFrame(rows)

    # cost sensitivity
    sens = []
    for c in COSTS_BPS:
        for name in daily.columns:
            m = compute_metrics(net_df[c][name])
            sens.append({"strategy": name, "cost_bps": c, "sharpe": m.sharpe,
                         "annualised_return": m.annualised_return, "max_drawdown": m.max_drawdown})
    sensitivity = pd.DataFrame(sens)

    # deflated Sharpe: N = 27 trials = 9 variants x 3 legs (preregistration §7)
    trial_names = [f"{v}|{leg}" for v in states for leg in LEGS]
    assert len(trial_names) == 27
    trial_sr = np.array([per_period_sharpe(daily[s].to_numpy()) for s in trial_names])
    sr_var = float(np.var(trial_sr, ddof=1))
    sr0 = {27: expected_max_sharpe(27, sr_var), 9: expected_max_sharpe(9, sr_var)}
    dsr = pd.DataFrame([{
        "strategy": s, "per_period_sharpe": per_period_sharpe(daily[s].to_numpy()),
        "dsr_n27": deflated_sharpe(daily[s].to_numpy(), sr0=sr0[27]),
        "dsr_n9": deflated_sharpe(daily[s].to_numpy(), sr0=sr0[9]),
        "sr0_n27": sr0[27], "sr0_n9": sr0[9], "trial": s in trial_names,
    } for s in daily.columns])
    dsr_lookup = dsr.set_index("strategy")["dsr_n27"]

    # pre-specified paired allocator contrasts (secondary, descriptive)
    contrast_rows = []
    higher_is_better = {"sharpe": True, "max_drawdown": True, "cvar_95": True}
    for leg in LEGS:
        for x, y in CONTRASTS:
            for stat in higher_is_better:
                px, rx = reps[f"{x}|{leg}"][stat]
                py, ry = reps[f"{y}|{leg}"][stat]
                diff_rep = rx - ry
                lo, hi = np.quantile(diff_rep, [0.025, 0.975])
                favourable = lo > 0
                dsr_x = float(dsr_lookup[f"{x}|{leg}"])
                contrast_rows.append({
                    "leg": leg, "x": x, "y": y, "metric": stat, "diff": px - py, "ci_lo": float(lo), "ci_hi": float(hi),
                    "ci_excludes_zero_favourably": bool(favourable), "dsr_x": dsr_x,
                    "claimable": bool(favourable and dsr_x >= 0.95),
                    "identical_by_construction": bool(leg == "RVT" and x not in REGIME_VARIANTS and y not in REGIME_VARIANTS),
                })
    contrasts = pd.DataFrame(contrast_rows)

    # episodes and calendar years
    spy_window = close["SPY"].loc[(close.index >= test_start) & (close.index <= test_end)]
    episodes = find_drawdown_episodes(spy_window, threshold=0.10)
    ep_rows = []
    for i, ep in enumerate(episodes):
        for name in daily.columns:
            ep_rows.append({"episode": i + 1, **ep.as_dict(), "strategy": name, **episode_performance(daily[name], ep)})
    episodes_tbl = pd.DataFrame(ep_rows)
    cal_rows = []
    for year in range(test_start.year, test_end.year + 1):
        for name in daily.columns:
            r = daily[name][daily.index.year == year]
            if len(r):
                m = compute_metrics(r)
                cal_rows.append({"strategy": name, "year": year, "return": m.cumulative_return,
                                 "max_drawdown": m.max_drawdown, "n_sessions": len(r)})
    calendar_tbl = pd.DataFrame(cal_rows)

    exposure = pd.DataFrame({k: 1.0 - w[al.CASH] for k, w in {**weights, **{f"BM|{a}": b for a, b in benchmark_weights.items()}}.items()})

    # ------------------------------------------------------------------ store
    alloc_metrics.to_csv(out_dir / "allocator_metrics.csv", index=False)
    sensitivity.to_csv(out_dir / "cost_sensitivity.csv", index=False)
    dsr.to_csv(out_dir / "dsr.csv", index=False)
    contrasts.to_csv(out_dir / "allocator_contrasts.csv", index=False)
    episodes_tbl.to_csv(out_dir / "episodes.csv", index=False)
    calendar_tbl.to_csv(out_dir / "calendar.csv", index=False)
    mv_dist.to_csv(out_dir / "mv_pairwise_distance.csv")
    exposure.to_parquet(out_dir / "risky_exposure.parquet")
    daily.to_parquet(out_dir / "strategy_net_5bps.parquet")
    gross_df.to_parquet(out_dir / "strategy_gross.parquet")
    turn_df.to_parquet(out_dir / "strategy_turnover.parquet")
    wdir = out_dir / "weights"
    wdir.mkdir(exist_ok=True)
    for name, w in {**weights, **{f"BM|{a}": b for a, b in benchmark_weights.items()}}.items():
        w.to_parquet(wdir / f"{name.replace('|', '__')}.parquet")

    meta = {
        "n_test_days": n, "n_strategy_days": n_days, "n_decisions": len(decisions),
        "first_decision": str(decisions[0].date()), "last_decision": str(decisions[-1].date()),
        "strategy_first_day": str(daily.index[0].date()), "strategy_last_day": str(daily.index[-1].date()),
        "common_index": [str(common[0].date()), str(common[-1].date()), len(common)],
        "test_window": [str(test_start.date()), str(test_end.date())],
        "gamma": gamma, "sr_variance_across_27": sr_var, "sr0": {str(k): v for k, v in sr0.items()},
        "n_episodes": len(episodes), "runtime_s": time.time() - t_start,
        "preregistration_commit": phase1["preregistration_commit"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_jsonable), encoding="utf-8")

    run_id = make_run_id("04_tier1_ablation")
    write_manifest(
        cfg.path("logs") / f"{run_id}.json", run_id=run_id, stage="04_tier1_ablation",
        config_hash=hash_object(cfg.model_dump(mode="json", exclude={"root"})),
        snapshot_hash=snapshot.snapshot_hash, seeds=seeds.as_dict(),
        extra={"input_hashes": input_hashes, "meta": meta, "gates": {k: {g: r["passes"] for g, r in v.items()} for k, v in gates.items()}},
        root=cfg.root,
    )
    log.info("stored results in %s (%.0fs)", out_dir, time.time() - t_start)

    if not args.skip_report:
        from prism.reporting.report import build_report

        report_path = build_report(cfg)
        log.info("report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
