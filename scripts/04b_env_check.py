#!/usr/bin/env python
"""Acceptance check for build step 4b on the real data. Spec §11, §16 step 4b.

    python scripts/04b_env_check.py

Builds the environment for V1, V2, V4 and C4 on the **train split only** (the
raw panel is cut at the train split's last session before any feature is built;
validation, test and holdout rows are never loaded) and checks, against
independent calculations:

* observations are finite, float32, of the expected width, through full episodes;
* the episode has one step per complete weekly holding period, counted from the calendar;
* a constant policy reproduces the analytic constant-mix return computed from
  the price path alone, and the Tier 1 simulator's compounded return;
* the cost model's effect over the 0 / 5 / 10 / 20 bps grid, at two turnover levels;
* a random-start training episode stays inside the split.

Writes ``reports/tables/env_check.md``. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prism.config import load_config  # noqa: E402
from prism.data.loaders import assert_not_holdout, load_snapshot  # noqa: E402
from prism.env.actions import action_to_weights  # noqa: E402
from prism.env.costs import CostModel  # noqa: E402
from prism.env.data import build_env_data  # noqa: E402
from prism.env.portfolio_env import make_env, run_episode  # noqa: E402
from prism.features.build import build_features, make_raw_frame  # noqa: E402
from prism.probes import allocator as al  # noqa: E402
from prism.splits import build_split_plan  # noqa: E402
from prism.utils.hashing import hash_object, make_run_id, sha256_file, write_manifest  # noqa: E402
from prism.utils.logging import configure_logging  # noqa: E402
from prism.utils.seeding import seed_everything  # noqa: E402

VARIANTS = ("V1", "V2", "V4", "C4")  # the Phase B variants (DECISIONS.md D-032)
SPLIT = "train"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    log = configure_logging()
    seeds = seed_everything(cfg.data.seeds.master)
    plan = build_split_plan(cfg)
    train_end = plan[SPLIT].effective_end
    t0 = time.time()

    snapshot = load_snapshot(cfg)
    raw = make_raw_frame(snapshot.close, snapshot.volume, snapshot.macro).loc[:train_end]
    assert_not_holdout(cfg, raw.index, context="env check raw panel")
    close = build_features(raw, cfg, "B").close
    assert close.index.max() <= train_end, "the check must never see beyond the train split"

    states_dir = cfg.path("processed") / "states"
    lines: list[str] = []
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        lines.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
        log.info("%s: %s %s", name, "PASS" if ok else "FAIL", detail)
        if not ok:
            failures.append(name)

    rng = np.random.default_rng(cfg.data.seeds.master)
    base_cost = CostModel.from_config(cfg)
    n_assets = len(cfg.data.allocatable[cfg.env.universe])
    cost_rows: list[str] = []
    widths: dict[str, int] = {}
    first_data = None
    for variant in VARIANTS:
        state = pd.read_parquet(states_dir / f"{variant}.parquet")
        data = build_env_data(cfg, state, close, SPLIT, plan=plan)
        first_data = first_data or data
        assert data.sessions[-1] <= train_end
        env = make_env(cfg, data, mode="eval", cost_model=base_cost)
        widths[variant] = env.observation_space.shape[0]

        # (1) finite observations of the declared shape through a full random episode
        obs, _ = env.reset(seed=0)
        ok, steps = True, 0
        while True:
            obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape).astype("float32"))
            ok &= bool(np.isfinite(obs).all() and np.isfinite(r) and obs.shape == env.observation_space.shape
                       and obs.dtype == np.float32)
            steps += 1
            if term or trunc:
                break
        check(f"{variant}: finite observations and rewards over a full train episode", ok,
              f"{steps} steps, observation width {widths[variant]} (state {data.states.shape[1]} + "
              f"{n_assets + 1} weights + 1 turnover)")

        # (2) episode length against an independent count of weekly holding periods
        # Weekly decisions fall on the last session of each calendar week (a Thursday when Friday is a
        # holiday, e.g. Good Friday): counted here from ISO weeks, not from the engine's own schedule.
        idx = data.sessions
        iso = idx.isocalendar()
        last_of_week = pd.Series(np.arange(len(idx)), index=idx).groupby(
            [iso["year"].to_numpy(), iso["week"].to_numpy()]
        ).max().to_numpy()
        expected = sum(
            1 for j, p in enumerate(last_of_week)
            if p + 1 < len(idx)
            and (last_of_week[j + 1] + 1 if j + 1 < len(last_of_week) else len(idx) - 1) > p + 1
        )
        check(f"{variant}: episode length equals the number of complete weekly holding periods",
              steps == expected == data.n_decisions, f"{steps} steps; calendar count {expected}")

        # (3) constant policy vs the analytic constant-mix return and the Tier 1 simulator (zero cost)
        action = rng.uniform(-1, 1, n_assets + 1).astype("float32")
        free = make_env(cfg, data, mode="eval", cost_model=base_cost.scaled(0.0))
        w = action_to_weights(action, free._upper, free.logit_scale)
        df = run_episode(free, lambda o: action, seed=0)
        prices = np.cumprod(1.0 + np.nan_to_num(data.line_returns), axis=0)
        analytic = float(np.prod([w @ (prices[data.end_pos[k]] / prices[data.exec_pos[k]])
                                  for k in range(data.n_decisions)]))
        dates = data.sessions[data.decision_pos]
        weights = pd.DataFrame(np.tile(w, (len(dates), 1)), index=dates, columns=list(data.lines))
        lr = pd.DataFrame(data.line_returns, index=data.sessions, columns=list(data.lines))
        gross, _ = al.simulate(weights, lr, execution_lag=cfg.data.decision.execution_lag_days)
        sim = float((1.0 + gross.iloc[1:]).prod())  # simulate() earns the first execution day in cash
        nav = float(df["nav"].iloc[-1])
        check(f"{variant}: constant policy reproduces the analytic constant-mix NAV",
              abs(nav / analytic - 1) < 1e-10, f"env {nav:.10f}, analytic {analytic:.10f}")
        check(f"{variant}: ... and the Tier 1 simulator's compounded return",
              abs(nav / sim - 1) < 1e-10, f"env {nav:.10f}, simulator {sim:.10f}")

        # (4) cost sensitivity at two turnover levels (V1 only; the cost model is variant-independent)
        if variant == "V1":
            for label, policy in (
                ("constant mix (low turnover)", lambda o: action),
                ("random each week (high turnover)", None),
            ):
                for bps, model in base_cost.sensitivity(cfg.env.costs.sensitivity_bps).items():
                    e = make_env(cfg, data, mode="eval", cost_model=model)
                    r2 = np.random.default_rng(1)
                    pol = policy or (lambda o: r2.uniform(-1, 1, n_assets + 1).astype("float32"))
                    res = run_episode(e, pol, seed=0)
                    cost_rows.append(
                        f"| {label} | {bps:g} | {res['nav'].iloc[-1]:.4f} | {res['cost'].sum() * 1e4:.1f} | "
                        f"{res['turnover'].mean():.3f} |"
                    )
            by = {}
            for row in cost_rows:
                c = [x.strip() for x in row.strip("|").split("|")]
                by.setdefault(c[0], []).append(float(c[2]))
            check("V1: NAV falls monotonically as cost rises, at both turnover levels",
                  all(all(b < a for a, b in zip(v, v[1:])) for v in by.values()), str(by))

        # (5) a random-start training episode stays inside the split
        tr = make_env(cfg, data, mode="train", cost_model=base_cost)
        inside = True
        for s in range(20):
            tr.reset(seed=s)
            inside &= 0 <= tr._first and tr._last < data.n_decisions
            inside &= data.sessions[data.end_pos[tr._last]] <= train_end
        check(f"{variant}: training episodes stay inside the train split", bool(inside),
              f"{cfg.env.episode.train_length_decisions} decisions each, 20 seeds")

    out = cfg.path("tables") / "env_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    first = first_data
    text = [
        "# Environment acceptance check (build step 4b)",
        "",
        f"Generated by `scripts/04b_env_check.py`. **Train split only**: effective range "
        f"{plan[SPLIT].effective_start.date()} .. {plan[SPLIT].effective_end.date()}; "
        f"{len(first.sessions)} sessions, {first.n_decisions} weekly decisions. "
        "No validation, test or holdout row is loaded.",
        "",
        f"Cost model: {base_cost.per_side_bps:g} bps per side on each risky leg's traded notional, plus "
        f"{base_cost.slippage_vol_coef:g} x trailing {cfg.env.costs.vol_window}-session daily vol per unit traded "
        f"(DECISIONS.md D-034). Reward: `{cfg.env.reward.name}`.",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
        *lines,
        "",
        "## Cost sensitivity on the train split (V1 state, random action weights)",
        "",
        "Net asset value at the end of the train split, total cost paid (bps of NAV, summed over steps) and mean "
        "one-way turnover per step. Cost level scales both the proportional and the slippage term.",
        "",
        "| Policy | bps per side | Final NAV | Total cost (bps) | Mean turnover |",
        "|---|---|---|---|---|",
        *cost_rows,
        "",
        "Observation widths: " + ", ".join(f"{v} = {w}" for v, w in widths.items()) + ".",
        "",
    ]
    out.write_text("\n".join(text), encoding="utf-8")
    log.info("wrote %s (%.1fs)", out, time.time() - t0)

    run_id = make_run_id("04b_env_check")
    write_manifest(
        cfg.path("logs") / f"{run_id}.json", run_id=run_id, stage="04b_env_check",
        config_hash=hash_object(cfg.model_dump(mode="json", exclude={"root"})),
        snapshot_hash=None, seeds=seeds.as_dict(),
        extra={"split": SPLIT, "variants": list(VARIANTS), "failures": failures,
               "state_hashes": {v: sha256_file(states_dir / f"{v}.parquet") for v in VARIANTS}},
        root=cfg.root,
    )
    if failures:
        log.error("FAILED: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
