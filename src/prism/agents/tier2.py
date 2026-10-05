"""The Tier 2 pipeline: tuning -> freeze -> final training -> single test evaluation -> statistics.

Spec §12-§14; ``reports/tables/preregistration_tier2.md`` is the contract. Each
stage is a function that is **resumable** (finished runs are skipped, a stage
whose output exists is not repeated) and **ordered**: nothing here reads the
test split except :func:`evaluate_test`, which refuses to run unless the
sanity record passed, the chosen configurations are frozen on disk, every final
run is finished, and it has not already run.

``Tier2Plan.smoke`` builds a reduced plan whose "test" stage evaluates the
**validation** split into a separate directory, so the whole pipeline can be
exercised end to end without reading the test split.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.agents.evaluate import benchmark_weight_frames, run_benchmark, run_policy
from prism.agents.jobs import is_done, job_dir, make_spec, read_result, run_job
from prism.agents.sac import SacConfig, SacSettings, greedy_policy, load_agent
from prism.analysis.bootstrap import bootstrap_paths
from prism.analysis.episodes import episode_performance, find_drawdown_episodes
from prism.analysis.significance import deflated_sharpe, expected_max_sharpe, per_period_sharpe
from prism.analysis.tier2 import METRICS, gate_verdict, non_overlap_verdict, paired_table, point_metrics, variant_replicates
from prism.backtest.metrics import compute_metrics
from prism.env.costs import CostModel
from prism.env.portfolio_env import make_env
from prism.utils.hashing import hash_object, utc_now_iso

__all__ = [
    "Tier2Plan", "grid_configs", "run_pool", "tuning_table", "freeze_configs", "load_frozen",
    "final_specs", "tuning_specs", "evaluate_test", "analyse", "BENCHMARKS",
]

BENCHMARKS = ("EqualWeight", "SixtyForty", "MinVariance", "RiskParity", "VolTarget", "BuyHoldSPY")
COST_LEVELS = (0.0, 5.0, 10.0, 20.0)
HEADLINE_BPS = 5.0


@dataclass(frozen=True)
class Tier2Plan:
    root: Path
    variants: tuple[str, ...]
    comparisons: tuple[tuple[str, str], ...]
    configs: tuple[SacConfig, ...]
    tuning_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    steps: int
    eval_every: int
    workers: int
    settings: SacSettings
    n_bootstrap: int
    block: int
    block_sensitivity: tuple[int, ...]
    level: float
    master_seed: int
    dsr_trials_headline: int
    dsr_trials_all_runs: int
    out_dir: Path
    log_dir: Path
    eval_split: str = "test"
    smoke: bool = False
    #: Where the trained runs and the frozen configs live, if not ``out_dir`` (the holdout run reads the
    #: Tier 2 agents from here and writes its own outputs to ``out_dir``).
    runs_dir: Path | None = None

    @property
    def runs(self) -> Path:
        return self.runs_dir if self.runs_dir is not None else self.out_dir

    @classmethod
    def from_config(cls, cfg, root: Path | None = None) -> Tier2Plan:  # noqa: ANN001
        t = cfg.tier2
        root = Path(root) if root is not None else cfg.root
        return cls(
            root=root, variants=tuple(t.variants), comparisons=tuple(tuple(c) for c in t.comparisons),
            configs=tuple(grid_configs(cfg)), tuning_seeds=tuple(t.tuning_seeds), final_seeds=tuple(t.final_seeds),
            steps=t.training.steps, eval_every=t.training.eval_every, workers=t.training.workers,
            settings=SacSettings(**t.sac.model_dump()),
            n_bootstrap=t.uncertainty.n_bootstrap, block=t.uncertainty.block_length_days,
            block_sensitivity=tuple(t.uncertainty.block_length_sensitivity), level=t.uncertainty.ci_level,
            master_seed=cfg.data.seeds.master, dsr_trials_headline=t.dsr_trials_headline,
            dsr_trials_all_runs=t.dsr_trials_all_runs,
            out_dir=root / t.output_dir, log_dir=root / t.log_dir,
        )

    def smoke_plan(self) -> Tier2Plan:
        """A tiny plan whose evaluation stage reads the VALIDATION split, in its own directory."""
        return replace(
            self, configs=(self.configs[0], self.configs[-1]), tuning_seeds=self.tuning_seeds[:2],
            final_seeds=self.final_seeds[:3], steps=2400, eval_every=800,
            settings=replace(self.settings, learning_starts=300), n_bootstrap=200, block_sensitivity=(),
            out_dir=self.out_dir.parent / "tier2_smoke", log_dir=self.log_dir.parent / "tier2_smoke",
            eval_split="val", smoke=True,
        )


def grid_configs(cfg) -> list[SacConfig]:  # noqa: ANN001
    g = cfg.tier2.grid
    return [SacConfig(gamma, tuple(h), lr) for gamma, h, lr in itertools.product(g.gamma, g.hidden, g.lr)]


# --------------------------------------------------------------------------- #
# training stages
# --------------------------------------------------------------------------- #
def tuning_specs(plan: Tier2Plan) -> list[dict[str, Any]]:
    return [
        make_spec(plan.root, "tune", v, c, s, plan.steps, plan.eval_every, plan.settings, plan.log_dir, plan.out_dir)
        for c, v, s in itertools.product(plan.configs, plan.variants, plan.tuning_seeds)
    ]


def final_specs(plan: Tier2Plan, frozen: dict[str, SacConfig]) -> list[dict[str, Any]]:
    return [
        make_spec(plan.root, "final", v, frozen[v], s, plan.steps, plan.eval_every, plan.settings, plan.log_dir, plan.out_dir)
        for v, s in itertools.product(plan.variants, plan.final_seeds)
    ]


def _spec_dir(plan: Tier2Plan, spec: dict[str, Any]) -> Path:
    c = spec["config"]
    cid = SacConfig(c["gamma"], tuple(c["hidden"]), c["lr"]).cfg_id
    return job_dir(plan.out_dir, spec["stage"], spec["variant"], cid, spec["seed"])


def run_pool(plan: Tier2Plan, specs: list[dict[str, Any]], log) -> int:  # noqa: ANN001
    """Run unfinished specs on a process pool; return how many were run. Finished runs are skipped."""
    import multiprocessing as mp

    todo = [s for s in specs if not is_done(_spec_dir(plan, s))]
    log.info("%d of %d runs to do (%d finished, skipped)", len(todo), len(specs), len(specs) - len(todo))
    if not todo:
        return 0
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=plan.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_job, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            r = fut.result()          # a failed run raises here and aborts the stage: no silent gaps
            elapsed = time.time() - t0
            log.info("[%d/%d] %s %s %s seed %s: best val %+.5f at step %d (%.0fs; elapsed %.1fh, eta %.1fh)",
                     i, len(todo), s["stage"], s["variant"], r["config"]["cfg_id"], s["seed"],
                     r["best"]["val_mean_log_return"], r["best"]["step"], r["wall_seconds"], elapsed / 3600,
                     elapsed / i * (len(todo) - i) / 3600)
    return len(todo)


def _results(plan: Tier2Plan, stage: str) -> pd.DataFrame:
    rows = []
    for f in sorted((plan.runs / "runs" / stage).glob("*/*/seed*/result.json")):
        r = json.loads(f.read_text())
        b, fin = r["best"], r["final"]
        rows.append({
            "stage": stage, "variant": r["variant"], "cfg_id": r["config"]["cfg_id"], "gamma": r["config"]["gamma"],
            "hidden": "x".join(map(str, r["config"]["hidden"])), "lr": r["config"]["lr"], "seed": r["data_seed"],
            "best_step": b["step"], "val_best": b["val_mean_log_return"], "train_at_best": b["train_mean_log_return"],
            "val_sharpe_at_best": b["val_weekly_sharpe"], "train_sharpe_at_best": b["train_weekly_sharpe"],
            "val_final": fin["val_mean_log_return"], "train_final": fin["train_mean_log_return"],
            "wall_seconds": r["wall_seconds"], "steps_per_second": r["steps_per_second"],
        })
    return pd.DataFrame(rows)


def tuning_table(plan: Tier2Plan) -> pd.DataFrame:
    """Per (variant, config): mean over the tuning seeds of the best-checkpoint validation score. The selection statistic."""
    runs = _results(plan, "tune")
    expected = len(plan.variants) * len(plan.configs) * len(plan.tuning_seeds)
    if len(runs) != expected:
        raise RuntimeError(f"tuning is incomplete: {len(runs)} of {expected} runs finished")
    order = {c.cfg_id: i for i, c in enumerate(plan.configs)}
    g = runs.groupby(["variant", "cfg_id"], sort=False).agg(
        val_best_mean=("val_best", "mean"), val_best_std=("val_best", "std"), train_at_best_mean=("train_at_best", "mean"),
        val_final_mean=("val_final", "mean"), best_step_mean=("best_step", "mean"), n_seeds=("seed", "count"),
    ).reset_index()
    g["grid_index"] = g["cfg_id"].map(order)
    return g.sort_values(["variant", "val_best_mean", "grid_index"], ascending=[True, False, True]).reset_index(drop=True)


def freeze_configs(plan: Tier2Plan, log) -> dict[str, SacConfig]:  # noqa: ANN001
    """Choose each variant's best configuration on validation and write it to disk; refuse to change a frozen choice."""
    table = tuning_table(plan)
    by_id = {c.cfg_id: c for c in plan.configs}
    chosen = {}
    record: dict[str, Any] = {"selection_metric": "val_mean_log_return (mean over tuning seeds of the best checkpoint)", "variants": {}}
    for v in plan.variants:
        top = table[table.variant == v].iloc[0]
        chosen[v] = by_id[top["cfg_id"]]
        record["variants"][v] = {
            "cfg_id": top["cfg_id"], "gamma": chosen[v].gamma, "hidden": list(chosen[v].hidden), "lr": chosen[v].lr,
            "val_best_mean": float(top["val_best_mean"]),
        }
    record["hash"] = hash_object(record["variants"])
    path = plan.out_dir / "chosen_configs.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["hash"] != record["hash"]:
            raise RuntimeError("chosen_configs.json exists with a different selection; the frozen choice is never overwritten")
    else:
        record["frozen_utc"] = utc_now_iso()
        path.write_text(json.dumps(record, indent=2))
        log.info("froze configurations: %s", {v: c.cfg_id for v, c in chosen.items()})
    table.to_csv(plan.out_dir / "tuning_table.csv", index=False)
    return chosen


def load_frozen(plan: Tier2Plan) -> dict[str, SacConfig]:
    path = plan.runs / "chosen_configs.json"
    if not path.exists():
        raise RuntimeError("chosen configurations are not frozen on disk; the test split stays closed")
    rec = json.loads(path.read_text())
    if hash_object(rec["variants"]) != rec["hash"]:
        raise RuntimeError("chosen_configs.json was edited after it was frozen")
    return {v: SacConfig(r["gamma"], tuple(r["hidden"]), r["lr"]) for v, r in rec["variants"].items()}


# --------------------------------------------------------------------------- #
# the single evaluation of the test split
# --------------------------------------------------------------------------- #
def evaluate_test(plan: Tier2Plan, cfg, log, sanity: dict | None, *, close: pd.DataFrame | None = None,  # noqa: ANN001
                  states: dict[str, pd.DataFrame] | None = None, final_holdout: bool = False) -> Path:
    """Score every final checkpoint and every benchmark ONCE on the evaluation split; write daily net returns."""
    from prism.agents.data import load_close, variant_env_data
    from prism.splits import build_split_plan

    out = plan.out_dir / "eval_daily.parquet"
    marker = plan.out_dir / "eval_done.json"
    if marker.exists():
        log.info("the %s split was already evaluated (%s); not repeating", plan.eval_split, marker)
        return out
    if sanity is None or not sanity.get("passed"):
        raise RuntimeError("sanity gates have not passed; refusing to read the test split")
    frozen = load_frozen(plan)
    for v in plan.variants:
        for s in plan.final_seeds:
            if not is_done(job_dir(plan.runs, "final", v, frozen[v].cfg_id, s)):
                raise RuntimeError(f"final run {v} seed {s} is not finished; refusing to evaluate")

    log.info("EVALUATING THE %s SPLIT (once)", plan.eval_split.upper())
    split_plan = build_split_plan(cfg)
    if close is None:
        close = load_close(cfg, plan.eval_split)
    elif plan.eval_split != "holdout" and not plan.smoke:
        raise RuntimeError("injected inputs are for the holdout evaluation (or a rehearsal) only")
    base_cost = CostModel.from_config(cfg)
    cols: dict[str, pd.Series] = {}
    meta: dict[str, Any] = {"weights": {}, "turnover": {}}
    for v in plan.variants:
        data = variant_env_data(cfg, v, close, plan.eval_split, plan=split_plan,
                                state=None if states is None else states[v], final_holdout=final_holdout)
        for s in plan.final_seeds:
            model = load_agent(job_dir(plan.runs, "final", v, frozen[v].cfg_id, s) / "best.zip")
            policy = greedy_policy(model)
            for bps in COST_LEVELS:
                env = make_env(cfg, data, mode="eval", cost_model=base_cost.scaled(bps))
                res = run_policy(env, policy)
                cols[f"{v}|s{s}|{bps:g}"] = res["daily"]
                if bps == HEADLINE_BPS:
                    meta["turnover"][f"{v}|s{s}"] = float(res["turnover"].mean())
                    meta["weights"][f"{v}|s{s}"] = res["weights"].mean().round(6).to_dict()
        log.info("evaluated %s (%d seeds)", v, len(plan.final_seeds))
    # Benchmarks re-run through the same env and cost model (D-034): lines = 13 risky, SPY, CASH.
    risky = [*cfg.data.allocatable[cfg.env.universe], "SPY"]
    bdata = variant_env_data(cfg, "V1", close, plan.eval_split, plan=split_plan, risky=risky,
                             state=None if states is None else states["V1"], final_holdout=final_holdout)
    frames = benchmark_weight_frames(cfg, close, bdata)
    for name in BENCHMARKS:
        for bps in COST_LEVELS:
            res = run_benchmark(cfg, bdata, frames, name, base_cost.scaled(bps))
            cols[f"BM|{name}|{bps:g}"] = res["daily"]
            if bps == HEADLINE_BPS:
                meta["turnover"][f"BM|{name}"] = float(res["turnover"].mean())
    frame = pd.DataFrame(cols)
    if frame.isna().any().any():
        raise RuntimeError("evaluation series are not on a common index")
    frame.to_parquet(out)
    spy = close["SPY"]
    spy.loc[(spy.index >= split_plan[plan.eval_split].effective_start) & (spy.index <= split_plan[plan.eval_split].effective_end)] \
        .to_frame("SPY").to_parquet(plan.out_dir / "eval_spy.parquet")
    marker.write_text(json.dumps({
        "split": plan.eval_split, "evaluated_utc": utc_now_iso(), "n_sessions": int(len(frame)),
        "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()),
        "benchmark_lines": risky, **meta,
    }, indent=2, default=float))
    if plan.eval_split == "test":
        with (plan.root / "reports" / "logs" / "test_access_tier2.jsonl").open("a") as fh:
            fh.write(json.dumps({"utc": utc_now_iso(), "split": "test", "stage": "tier2 evaluate"}) + "\n")
    log.info("wrote %s (%d sessions, %d columns)", out, len(frame), frame.shape[1])
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _series(frame: pd.DataFrame, plan: Tier2Plan, variant: str, bps: float = HEADLINE_BPS) -> dict[int, np.ndarray]:
    return {s: frame[f"{variant}|s{s}|{bps:g}"].to_numpy() for s in plan.final_seeds}


def analyse(plan: Tier2Plan, cfg, log) -> dict[str, Any]:  # noqa: ANN001
    """Every table of the report, from the stored evaluation series. Writes CSVs under ``tables/`` and returns them."""
    frame = pd.read_parquet(plan.out_dir / "eval_daily.parquet")
    spy = pd.read_parquet(plan.out_dir / "eval_spy.parquet")["SPY"]
    tdir = plan.out_dir / "tables"
    tdir.mkdir(exist_ok=True)
    n = len(frame)
    seeds = plan.final_seeds
    res: dict[str, Any] = {"n_sessions": n, "first": str(frame.index[0].date()), "last": str(frame.index[-1].date())}

    # --- headline metrics per variant (seed band) and per benchmark ---------------------------------
    full = {}
    for v in plan.variants:
        per_seed = pd.DataFrame({s: compute_metrics(frame[f"{v}|s{s}|5"]).as_dict() for s in seeds}).T
        full[v] = per_seed
        per_seed.to_csv(tdir / f"seed_metrics_{v}.csv")
    band_rows = []
    for v in plan.variants:
        for m in ("annualised_return", "annualised_vol", "sharpe", "sortino", "calmar", "max_drawdown", "cvar_95"):
            x = full[v][m].astype(float)
            band_rows.append({"variant": v, "metric": m, "mean": x.mean(), "std": x.std(ddof=1), "min": x.min(),
                              "median": x.median(), "max": x.max()})
    band = pd.DataFrame(band_rows)
    band.to_csv(tdir / "seed_band.csv", index=False)
    bm_metrics = pd.DataFrame({b: compute_metrics(frame[f"BM|{b}|5"]).as_dict() for b in BENCHMARKS}).T
    bm_metrics.to_csv(tdir / "benchmark_metrics.csv")
    res["band"], res["benchmarks"] = band, bm_metrics

    # --- turnover and cost sensitivity ---------------------------------------------------------------
    meta = json.loads((plan.out_dir / "eval_done.json").read_text())
    turn = pd.DataFrame([{"variant": v, "mean_turnover_per_step": float(np.mean([meta["turnover"][f"{v}|s{s}"] for s in seeds]))}
                         for v in plan.variants] + [{"variant": f"BM|{b}", "mean_turnover_per_step": meta["turnover"][f"BM|{b}"]}
                                                    for b in BENCHMARKS])
    turn.to_csv(tdir / "turnover.csv", index=False)
    cost_rows = []
    for v in plan.variants:
        for bps in COST_LEVELS:
            ms = pd.DataFrame({s: compute_metrics(frame[f"{v}|s{s}|{bps:g}"]).as_dict() for s in seeds}).T
            cost_rows.append({"strategy": v, "bps": bps, "annualised_return": ms.annualised_return.mean(), "sharpe": ms.sharpe.mean()})
    for b in BENCHMARKS:
        for bps in COST_LEVELS:
            m = compute_metrics(frame[f"BM|{b}|{bps:g}"])
            cost_rows.append({"strategy": f"BM|{b}", "bps": bps, "annualised_return": m.annualised_return, "sharpe": m.sharpe})
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(tdir / "cost_sensitivity.csv", index=False)
    res["turnover"], res["cost"] = turn, cost

    # --- paired bootstrap: shared day paths, seeds resampled ------------------------------------------
    pairs = [tuple(c) for c in plan.comparisons]
    points = {v: {m: float(np.mean([point_metrics(x)[m] for x in _series(frame, plan, v).values()])) for m in METRICS}
              for v in plan.variants}
    gate_tables: dict[int, pd.DataFrame] = {}
    reps_main = None
    for block in (plan.block, *plan.block_sensitivity):
        paths = bootstrap_paths(n, block, plan.n_bootstrap, plan.master_seed)
        reps = {v: variant_replicates(_series(frame, plan, v), paths, master=plan.master_seed, variant=v) for v in plan.variants}
        gate_tables[block] = paired_table(reps, points, pairs, level=plan.level)
        gate_tables[block].to_csv(tdir / f"paired_block{block}.csv", index=False)
        if block == plan.block:
            reps_main = reps
            bm_paths = paths
    gates = {}
    for block, t in gate_tables.items():
        gates[block] = {f"{x}>{y}": {"paired": gate_verdict(t, x, y), "non_overlap": non_overlap_verdict(t, x, y)} for x, y in pairs}
    res["paired"], res["gates"] = gate_tables, gates
    res["points"] = points

    # --- deflated Sharpe -----------------------------------------------------------------------------
    daily_sr = {f"{v}|s{s}": per_period_sharpe(frame[f"{v}|s{s}|5"].to_numpy()) for v in plan.variants for s in seeds}
    var_sr = float(np.var(list(daily_sr.values()), ddof=1))
    sr0 = {plan.dsr_trials_headline: expected_max_sharpe(plan.dsr_trials_headline, var_sr),
           plan.dsr_trials_all_runs: expected_max_sharpe(plan.dsr_trials_all_runs, var_sr)}
    dsr_rows = []
    for v in plan.variants:
        d = {k: np.array([deflated_sharpe(frame[f"{v}|s{s}|5"].to_numpy(), sr0=sr0[k]) for s in seeds]) for k in sr0}
        row = {"variant": v}
        for k, arr in d.items():
            row[f"dsr_median_n{k}"] = float(np.nanmedian(arr))
            row[f"dsr_min_n{k}"] = float(np.nanmin(arr))
            row[f"seeds_ge_0.95_n{k}"] = int((arr >= 0.95).sum())
        dsr_rows.append(row)
    for b in BENCHMARKS:
        row = {"variant": f"BM|{b}"}
        for k in sr0:
            row[f"dsr_median_n{k}"] = float(deflated_sharpe(frame[f"BM|{b}|5"].to_numpy(), sr0=sr0[k]))
        dsr_rows.append(row)
    dsr = pd.DataFrame(dsr_rows)
    dsr.to_csv(tdir / "dsr.csv", index=False)
    res["dsr"], res["sr0"], res["sharpe_variance"] = dsr, sr0, var_sr
    claim = []
    for x, y in pairs:
        t = gate_tables[plan.block]
        fav = int(((t.candidate == x) & (t.control == y) & (t.verdict == "favourable")).sum())
        med = float(dsr.loc[dsr.variant == x, f"dsr_median_n{plan.dsr_trials_headline}"].iloc[0])
        gate_pass = gate_verdict(t, x, y)["pass"]
        claim.append({"candidate": x, "control": y, "gate_pass": gate_pass, "favourable_metrics": fav,
                      "dsr_median_candidate": med, "claimable": bool(gate_pass and med >= 0.95)})
    res["claims"] = pd.DataFrame(claim)
    res["claims"].to_csv(tdir / "claims.csv", index=False)

    # --- benchmark CIs on the same day paths (no seed resampling) -------------------------------------
    from prism.analysis.tier2 import series_metrics_replicates
    rows = []
    for b in BENCHMARKS:
        rep = series_metrics_replicates(frame[f"BM|{b}|5"].to_numpy(), bm_paths)
        pt = point_metrics(frame[f"BM|{b}|5"].to_numpy())
        for m in METRICS:
            lo, hi = np.quantile(rep[m], [(1 - plan.level) / 2, 1 - (1 - plan.level) / 2])
            rows.append({"benchmark": b, "metric": m, "point": pt[m], "ci_low": float(lo), "ci_high": float(hi)})
    pd.DataFrame(rows).to_csv(tdir / "benchmark_ci.csv", index=False)
    res["benchmark_ci"] = pd.DataFrame(rows)

    # --- variants vs benchmarks (descriptive): Sharpe and drawdown, same paths ------------------------
    # --- episodes and calendar years ------------------------------------------------------------------
    episodes = find_drawdown_episodes(spy, threshold=0.10)
    ep_rows = []
    for i, ep in enumerate(episodes):
        for v in plan.variants:
            parts = pd.DataFrame([episode_performance(frame[f"{v}|s{s}|5"], ep) for s in seeds])
            ep_rows.append({"episode": i + 1, **ep.as_dict(), "strategy": v,
                            **{f"{c}_mean": parts[c].mean() for c in parts}, **{f"{c}_std": parts[c].std(ddof=1) for c in parts}})
        for b in BENCHMARKS:
            p = episode_performance(frame[f"BM|{b}|5"], ep)
            ep_rows.append({"episode": i + 1, **ep.as_dict(), "strategy": f"BM|{b}", **{f"{c}_mean": p[c] for c in p}})
    episodes_tbl = pd.DataFrame(ep_rows)
    episodes_tbl.to_csv(tdir / "episodes.csv", index=False)
    cal_rows = []
    for year in sorted(set(frame.index.year)):
        mask = frame.index.year == year
        for v in plan.variants:
            ms = pd.DataFrame({s: compute_metrics(frame.loc[mask, f"{v}|s{s}|5"]).as_dict() for s in seeds}).T
            cal_rows.append({"year": year, "strategy": v, "annualised_return": ms.annualised_return.mean(),
                             "sharpe": ms.sharpe.mean(), "max_drawdown": ms.max_drawdown.mean()})
        for b in BENCHMARKS:
            m = compute_metrics(frame.loc[mask, f"BM|{b}|5"])
            cal_rows.append({"year": year, "strategy": f"BM|{b}", "annualised_return": m.annualised_return,
                             "sharpe": m.sharpe, "max_drawdown": m.max_drawdown})
    pd.DataFrame(cal_rows).to_csv(tdir / "calendar.csv", index=False)
    res["episodes"], res["calendar"] = episodes_tbl, pd.DataFrame(cal_rows)
    res["episode_list"] = [e.as_dict() for e in episodes]

    # --- audit triggers (spec §15.1) ------------------------------------------------------------------
    sharpe_max = float(max(full[v]["sharpe"].max() for v in plan.variants))
    res["audit"] = {"max_seed_sharpe_net": sharpe_max, "sharpe_above_2": bool(sharpe_max > 2.0)}
    log.info("analysis done: gates %s", {k: {kk: vv["paired"]["pass"] for kk, vv in g.items()} for k, g in gates.items()})
    return res


def overfitting_table(plan: Tier2Plan, frozen: dict[str, SacConfig]) -> pd.DataFrame:
    """Train vs validation, per seed, at the selected checkpoint and at the last one (no test data)."""
    runs = _results(plan, "final")
    runs = runs[runs.apply(lambda r: r.cfg_id == frozen[r.variant].cfg_id, axis=1)].copy()
    runs["gap_at_best"] = runs.train_at_best - runs.val_best
    runs["gap_final"] = runs.train_final - runs.val_final
    return runs.sort_values(["variant", "seed"]).reset_index(drop=True)
