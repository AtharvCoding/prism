"""Sanity gates for the SAC pipeline. Spec §12; preregistration_tier2 §5.

Run on **train and validation only**. A failure aborts the Tier 2 run before the
test split is read.

1. **Degenerate task.** A synthetic panel in which one asset earns a constant
   positive return every day and the others are zero-mean noise, with the real
   cost model and cap. The optimum holds the edge asset at the cap and the rest
   in cash. The agent passes if its deterministic policy, on a fresh noise path,
   holds the edge asset at (nearly) the cap and earns (nearly) the oracle's log return.
2. **Beats random.** SAC on the real V1 train split, trained as the pipeline trains
   (validation-selected checkpoint), against the distribution of i.i.d.-uniform-action
   policies on the real validation split. The gate is the median random policy; the 95th
   percentile and the final checkpoint are reported (preregistration_tier2 §5 says why).
   Both gates train for ``tier2.training.steps``, the production budget.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.agents.sac import (
    SacConfig, SacSettings, ScaledReward, evaluate_deterministic, greedy_policy, make_sac, random_policy_scores, train_run,
)
from prism.backtest.engine import TimingConvention
from prism.env.data import CASH, EnvData
from prism.env.portfolio_env import PortfolioEnv, make_env

__all__ = ["synthetic_edge_data", "run_degenerate_seed", "run_beat_random_seed", "run_sanity", "DEFAULT_SANITY_CONFIG"]

#: The configuration the sanity gates train with; the same for every gate and every fix attempt.
DEFAULT_SANITY_CONFIG = SacConfig(gamma=0.97, hidden=(64, 64), lr=3e-4)


def synthetic_edge_data(cfg, *, seed: int, risky: list[str] | None = None, n_state: int = 4) -> EnvData:  # noqa: ANN001
    """A panel where ``edge_asset`` returns a constant positive amount daily; the rest are zero-mean noise.

    The state is a constant, deliberately: with random state features a network can use them as session
    identifiers and memorise the one noise path it trains on (exploration runs showed the gate getting
    *worse* with more training for exactly that reason). That is overfitting, which the overfitting report
    measures on real data; this gate asks only whether the optimiser and the policy head can learn.
    """
    s = cfg.tier2.sanity
    risky = list(risky) if risky is not None else list(cfg.data.allocatable[cfg.env.universe])
    rng = np.random.default_rng(seed)
    n = s.n_sessions
    sessions = pd.bdate_range(s.start_date, periods=n)
    rets = rng.normal(0.0, s.noise_daily_vol, size=(n, len(risky)))
    rets[:, risky.index(s.edge_asset)] = s.edge_daily_return
    lines = np.column_stack([rets, np.zeros(n)])
    sigma = np.full((n, len(risky)), s.noise_daily_vol)
    states = np.zeros((n, n_state), dtype="float32")
    return EnvData.from_arrays(
        sessions, states, lines, sigma, lines=(*risky, CASH), convention=TimingConvention.from_config(cfg)
    )


def _env(cfg, data: EnvData, mode: str) -> PortfolioEnv:  # noqa: ANN001
    return make_env(cfg, data, mode=mode)


def run_degenerate_seed(root: str, seed: int, settings: dict[str, Any], config: dict[str, Any],
                        steps: int | None = None) -> dict[str, Any]:
    """Train on a synthetic edge panel, evaluate on a fresh one."""
    from prism.config import load_config

    cfg = load_config(Path(root) / "configs" / "base.yaml")
    s = cfg.tier2.sanity
    sc = SacConfig(config["gamma"], tuple(config["hidden"]), config["lr"])
    st = SacSettings(**settings)
    train = synthetic_edge_data(cfg, seed=seed)
    fresh = synthetic_edge_data(cfg, seed=seed + 500_000)
    risky = list(fresh.lines[:-1])
    edge = risky.index(s.edge_asset)
    steps = cfg.tier2.training.steps if steps is None else steps
    t0 = time.time()
    model = make_sac(ScaledReward(_env(cfg, train, "train"), st.reward_scale), sc, st, seed, total_steps=steps)
    model.learn(total_timesteps=steps)
    env = _env(cfg, fresh, "eval")
    weights: list[np.ndarray] = []
    policy = greedy_policy(model)
    obs, _ = env.reset(seed=0)
    rewards = []
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        weights.append(info["weights"])
        rewards.append(r)
        if term or trunc:
            break
    w = np.asarray(weights)
    # Oracle: edge asset at the cap, the rest in cash, from the first decision (initiation paid once).
    cap = cfg.data.allocation.weight_max
    oracle_w = np.zeros(len(risky) + 1)
    oracle_w[edge] = cap
    oracle_w[-1] = 1.0 - cap
    env_o = _env(cfg, fresh, "eval")
    env_o.reset(seed=0)
    oracle = []
    while True:
        _, r, term, trunc, _ = env_o.step_weights(oracle_w)
        oracle.append(r)
        if term or trunc:
            break
    mean_w = float(w[:, edge].mean())
    frac_oracle = float(np.mean(rewards) / np.mean(oracle))
    return {
        "seed": seed,
        "steps": steps,
        "edge_weight_mean": mean_w,
        "edge_weight_fraction_of_cap": mean_w / cap,
        "agent_mean_log_return": float(np.mean(rewards)),
        "oracle_mean_log_return": float(np.mean(oracle)),
        "log_return_fraction_of_oracle": frac_oracle,
        "wall_seconds": time.time() - t0,
        "passed": bool(mean_w / cap >= s.min_edge_weight_fraction_of_cap
                       and frac_oracle >= s.min_log_return_fraction_of_oracle),
    }


def run_beat_random_seed(root: str, seed: int, settings: dict[str, Any], config: dict[str, Any],
                         steps: int | None = None) -> dict[str, Any]:
    """Train V1 on the real train split as the pipeline does; score the selected and the final checkpoint on validation."""
    from prism.agents.data import load_close, variant_env_data
    from prism.config import load_config
    from prism.splits import build_split_plan

    cfg = load_config(Path(root) / "configs" / "base.yaml")
    s = cfg.tier2.sanity
    sc = SacConfig(config["gamma"], tuple(config["hidden"]), config["lr"])
    st = SacSettings(**settings)
    steps = cfg.tier2.training.steps if steps is None else steps
    plan = build_split_plan(cfg)
    close = load_close(cfg, "val")
    tr = variant_env_data(cfg, s.beat_random_variant, close, "train", plan=plan)
    va = variant_env_data(cfg, s.beat_random_variant, close, "val", plan=plan)
    t0 = time.time()
    res = train_run(make_env(cfg, tr, mode="train"), make_env(cfg, va, mode="eval"), sc, st, seed=seed,
                    total_steps=steps, eval_every=cfg.tier2.training.eval_every,
                    train_eval_env=make_env(cfg, tr, mode="eval"))
    best, final = res["best"], res["final"]
    pick = lambda row, split: {k[len(split) + 1:]: v for k, v in row.items() if k.startswith(split + "_")}  # noqa: E731
    return {"seed": seed, "steps": steps, "selected": {"step": best["step"], "val": pick(best, "val"), "train": pick(best, "train")},
            "final": {"step": final["step"], "val": pick(final, "val"), "train": pick(final, "train")},
            "wall_seconds": time.time() - t0}


def run_sanity(root: Path, cfg, *, settings: SacSettings, config: SacConfig = DEFAULT_SANITY_CONFIG,  # noqa: ANN001
               workers: int = 6) -> dict[str, Any]:
    """Both gates; returns the record written to ``sanity.json``."""
    import multiprocessing as mp

    from prism.agents.data import load_close, variant_env_data
    from prism.splits import build_split_plan

    s = cfg.tier2.sanity
    sd, cd = asdict(settings), asdict(config)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        deg = [pool.submit(run_degenerate_seed, str(root), seed, sd, cd) for seed in s.seeds]
        rnd = [pool.submit(run_beat_random_seed, str(root), seed, sd, cd) for seed in s.seeds]
        degenerate = [f.result() for f in deg]
        beat = [f.result() for f in rnd]

    plan = build_split_plan(cfg)
    close = load_close(cfg, "val")
    va = variant_env_data(cfg, s.beat_random_variant, close, "val", plan=plan)
    random_scores = random_policy_scores(make_env(cfg, va, mode="eval"), s.random_policies, seed=cfg.data.seeds.master)
    threshold = float(np.quantile(random_scores, s.random_quantile))
    threshold_reported = float(np.quantile(random_scores, s.random_quantile_reported))
    flat = evaluate_deterministic(make_env(cfg, va, mode="eval"), lambda o: np.zeros(va.n_risky + 1, dtype="float32"))
    for b in beat:
        sel, fin = b["selected"]["val"]["mean_log_return"], b["final"]["val"]["mean_log_return"]
        b["beats_gate_quantile"] = bool(sel > threshold)
        b["selected_percentile_rank"] = float((random_scores < sel).mean())
        b["final_percentile_rank"] = float((random_scores < fin).mean())
        b["selected_beats_reported_quantile"] = bool(sel > threshold_reported)
    gate1 = all(d["passed"] for d in degenerate)
    gate2 = sum(b["beats_gate_quantile"] for b in beat) >= s.beat_random_min_seeds
    return {
        "config": cd | {"cfg_id": config.cfg_id},
        "settings": sd,
        "steps": cfg.tier2.training.steps,
        "degenerate_task": {"passed": bool(gate1), "seeds": degenerate},
        "beats_random": {
            "passed": bool(gate2),
            "gate_quantile": s.random_quantile,
            "gate_threshold_mean_log_return": threshold,
            "reported_quantile": s.random_quantile_reported,
            "reported_threshold_mean_log_return": threshold_reported,
            "random_median": float(np.median(random_scores)),
            "random_max": float(random_scores.max()),
            "equal_weight_constant_policy_val": flat["mean_log_return"],
            "equal_weight_constant_percentile_rank": float((random_scores < flat["mean_log_return"]).mean()),
            "seeds": beat,
        },
        "passed": bool(gate1 and gate2),
    }
