"""Soft Actor-Critic via stable-baselines3, CPU only, small networks. Spec §12.

Everything that makes one training run is here: the SAC factory, the reward-scale
wrapper (training only; evaluation always reads the env's own reward), the
deterministic evaluation of a policy over a whole split, and :func:`train_run`,
which trains for a fixed number of environment steps and evaluates the
deterministic policy on the train and validation splits at every checkpoint.
The checkpoint with the best **validation** score is kept; the test split is
never an argument of anything here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from prism.env.portfolio_env import PortfolioEnv

__all__ = [
    "SacSettings", "SacConfig", "ScaledReward", "make_sac", "evaluate_deterministic",
    "greedy_policy", "random_policy_scores", "train_run", "load_agent",
]


@dataclass(frozen=True)
class SacSettings:
    """Fixed for every variant and every run (preregistration_tier2 §4)."""

    batch_size: int = 256
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    reward_scale: float = 1.0


@dataclass(frozen=True)
class SacConfig:
    """One point of the tuning grid."""

    gamma: float
    hidden: tuple[int, ...]
    lr: float

    @property
    def cfg_id(self) -> str:
        return f"g{self.gamma}_n{'x'.join(map(str, self.hidden))}_lr{self.lr:g}"


class ScaledReward(gym.RewardWrapper):
    """Multiply the reward by a constant. Training only (evaluation reads the env's reward)."""

    def __init__(self, env: gym.Env, scale: float) -> None:
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        return reward * self.scale


def make_sac(env: gym.Env, config: SacConfig, settings: SacSettings, seed: int, *, total_steps: int) -> SAC:
    """SAC on CPU with ``config.hidden`` for actor and both critics."""
    torch.set_num_threads(1)
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=config.lr,
        gamma=config.gamma,
        batch_size=settings.batch_size,
        buffer_size=min(settings.buffer_size, max(total_steps, settings.batch_size * 2)),
        learning_starts=settings.learning_starts,
        tau=settings.tau,
        train_freq=settings.train_freq,
        gradient_steps=settings.gradient_steps,
        ent_coef="auto",
        policy_kwargs={"net_arch": list(config.hidden)},
        seed=seed,
        device="cpu",
        verbose=0,
    )


def greedy_policy(model: SAC):
    def act(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return act


def evaluate_deterministic(env: PortfolioEnv, policy, *, seed: int = 0) -> dict[str, Any]:
    """Run ``policy`` over one full eval-mode episode; return scores and the step table."""
    obs, _ = env.reset(seed=seed)
    rewards, net, cost, turn = [], [], [], []
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        rewards.append(r)
        net.append(info["net_return"])
        cost.append(info["cost"])
        turn.append(info["turnover"])
        if term or trunc:
            break
    r, n = np.asarray(rewards), np.asarray(net)
    return {
        "mean_log_return": float(r.mean()),
        "total_log_return": float(r.sum()),
        "weekly_sharpe": float(n.mean() / n.std(ddof=1) * np.sqrt(52)) if n.std(ddof=1) > 0 else float("nan"),
        "final_nav": float(np.prod(1.0 + n)),
        "mean_turnover": float(np.mean(turn)),
        "mean_cost_bps": float(np.mean(cost) * 1e4),
        "n_steps": int(len(r)),
    }


def random_policy_scores(env: PortfolioEnv, n: int, seed: int) -> np.ndarray:
    """Mean per-step log net return of ``n`` i.i.d.-uniform-action policies (the random baseline)."""
    out = np.empty(n)
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        pol = lambda o: rng.uniform(-1.0, 1.0, env.action_space.shape).astype("float32")  # noqa: E731
        out[i] = evaluate_deterministic(env, pol, seed=0)["mean_log_return"]
    return out


class _CheckpointCallback(BaseCallback):
    """Every ``eval_every`` steps: score the deterministic policy on train and validation; keep the best on validation."""

    def __init__(self, train_env: PortfolioEnv, val_env: PortfolioEnv, eval_every: int, out_dir: Path | None,
                 on_checkpoint=None) -> None:  # noqa: ANN001
        super().__init__()
        self.on_checkpoint = on_checkpoint
        self.train_env, self.val_env, self.eval_every = train_env, val_env, eval_every
        self.out_dir = out_dir
        self.curve: list[dict[str, Any]] = []
        self.best_val = -np.inf
        self.best_row: dict[str, Any] | None = None

    def _score(self) -> None:
        pol = greedy_policy(self.model)
        tr = evaluate_deterministic(self.train_env, pol)
        va = evaluate_deterministic(self.val_env, pol)
        row = {"step": int(self.num_timesteps),
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        self.curve.append(row)
        if self.on_checkpoint is not None:
            self.on_checkpoint(row)
        if va["mean_log_return"] > self.best_val:
            self.best_val, self.best_row = va["mean_log_return"], row
            if self.out_dir is not None:
                self.model.save(self.out_dir / "best")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0 and self.num_timesteps >= self.model.learning_starts:
            self._score()
        return True


def train_run(
    train_env: PortfolioEnv,
    val_env: PortfolioEnv,
    config: SacConfig,
    settings: SacSettings,
    *,
    seed: int,
    total_steps: int,
    eval_every: int,
    out_dir: Path | None = None,
    train_eval_env: PortfolioEnv | None = None,
    on_checkpoint=None,  # noqa: ANN001
) -> dict[str, Any]:
    """Train for ``total_steps`` env steps; return the curve and the best-on-validation checkpoint.

    ``train_env`` is a train-mode (random-start) env; ``train_eval_env`` an eval-mode env on the same
    split for scoring the policy (defaults to ``val_env``'s sibling built by the caller). The best
    checkpoint is the one with the highest validation mean log net return; ties keep the earlier one.
    """
    if train_eval_env is None:
        raise ValueError("train_eval_env (an eval-mode env on the train split) is required")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scaled = ScaledReward(train_env, settings.reward_scale)
    model = make_sac(scaled, config, settings, seed, total_steps=total_steps)
    cb = _CheckpointCallback(train_eval_env, val_env, eval_every, out_dir, on_checkpoint)
    model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)
    wall = time.time() - t0
    if not cb.curve:
        raise RuntimeError("no checkpoint was scored; total_steps is below learning_starts + eval_every")
    curve = pd.DataFrame(cb.curve)
    result = {
        "config": asdict(config) | {"cfg_id": config.cfg_id},
        "settings": asdict(settings),
        "seed": int(seed),
        "total_steps": int(total_steps),
        "eval_every": int(eval_every),
        "wall_seconds": wall,
        "steps_per_second": total_steps / wall,
        "best": cb.best_row,
        "final": cb.curve[-1],
        "n_checkpoints": len(cb.curve),
    }
    if out_dir is not None:
        curve.to_csv(out_dir / "curve.csv", index=False)
    return result


def load_agent(path: str | Path) -> SAC:
    return SAC.load(str(path), device="cpu")
