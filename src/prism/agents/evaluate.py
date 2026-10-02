"""Evaluate policies and benchmarks through the environment. Spec §11, §13.3; preregistration_tier2 §7-§8.

Every strategy, agent or benchmark, is run by the same environment code with the
same cost model, so net numbers are comparable (D-034: the Tier 1 allocator used
a different cost convention, which is why the benchmarks are re-run here).

``daily_net_returns`` turns an environment episode into a daily net return
series: the cost paid at an execution close is charged on the first session of
the holding period it opens, so the compounded series equals the env's NAV.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from prism.backtest import benchmarks as bm
from prism.env.costs import CostModel
from prism.env.data import EnvData
from prism.env.portfolio_env import PortfolioEnv, make_env
from prism.probes import allocator as al

__all__ = ["episode_daily", "run_policy", "run_weights", "benchmark_weight_frames", "run_benchmark"]


def _daily_from_steps(data: EnvData, infos: list[dict]) -> pd.Series:
    parts = []
    for k, info in enumerate(infos):
        e, end = int(data.exec_pos[k]), int(data.end_pos[k])
        r = np.asarray(info["daily_returns"], dtype="float64").copy()
        r[0] = (1.0 - info["cost"]) * (1.0 + r[0]) - 1.0   # the trade is paid at the open of the period
        parts.append(pd.Series(r, index=data.sessions[e + 1 : end + 1]))
    out = pd.concat(parts)
    if out.index.has_duplicates:
        raise RuntimeError("holding periods overlap")
    return out.rename("net")


def episode_daily(env: PortfolioEnv, step_fn: Callable[[np.ndarray, PortfolioEnv], tuple]) -> dict:
    """Run one eval episode; ``step_fn(obs, env)`` returns the env's ``step`` tuple."""
    obs, _ = env.reset(seed=0)
    infos, rewards = [], []
    while True:
        obs, r, term, trunc, info = step_fn(obs, env)
        infos.append(info)
        rewards.append(r)
        if term or trunc:
            break
    daily = _daily_from_steps(env.data, infos)
    weights = pd.DataFrame(
        [i["weights"] for i in infos], index=env.data.sessions[env.data.decision_pos[: len(infos)]],
        columns=list(env.data.lines),
    )
    turnover = np.array([i["turnover"] for i in infos])
    cost = np.array([i["cost"] for i in infos])
    nav = float(np.prod(1.0 + daily.to_numpy()))
    if abs(nav / infos[-1]["nav"] - 1.0) > 1e-9:
        raise RuntimeError("daily series does not compound to the environment's NAV")
    return {"daily": daily, "weights": weights, "turnover": turnover, "cost": cost, "rewards": np.asarray(rewards)}


def run_policy(env: PortfolioEnv, policy: Callable[[np.ndarray], np.ndarray]) -> dict:
    return episode_daily(env, lambda obs, e: e.step(policy(obs)))


def run_weights(env: PortfolioEnv, weights_fn: Callable[[PortfolioEnv], np.ndarray]) -> dict:
    return episode_daily(env, lambda obs, e: e.step_weights(weights_fn(e)))


def benchmark_weight_frames(cfg, close: pd.DataFrame, data: EnvData) -> dict[str, pd.DataFrame]:  # noqa: ANN001
    """The six spec §13.3 benchmarks' target weights at the env's own decision dates."""
    params = al.AllocatorParams(
        risky=tuple(cfg.data.allocatable[cfg.env.universe]),
        equity_sectors=tuple(cfg.data.universes["A"].equity_sectors),
        cap=cfg.data.allocation.weight_max,
    )
    dates = data.sessions[data.decision_pos]
    inputs, _ = al.prepare_inputs(close, params, dates)
    frames = bm.all_benchmarks(inputs, params)
    for name, w in frames.items():
        al.check_weights(f"BM|{name}", w, params, cash_capped=(name == "MinVariance"),
                         enforce_cap=(name != "SixtyForty"))
    return frames


def run_benchmark(cfg, data: EnvData, frames: dict[str, pd.DataFrame], name: str,  # noqa: ANN001
                  cost_model: CostModel) -> dict:
    """One benchmark through the env. ``data`` has lines (13 risky, SPY, CASH)."""
    env = make_env(cfg, data, mode="eval", cost_model=cost_model)
    w = frames[name]
    cols = list(data.lines)
    if name == "BuyHoldSPY":
        first = w.iloc[0][cols].to_numpy(dtype="float64")
        return run_weights(env, lambda e: first if e._steps == 0 else e.drifted_weights)
    return run_weights(env, lambda e: w.loc[e.current_decision_date, cols].to_numpy(dtype="float64"))
