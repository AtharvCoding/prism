"""Gymnasium-compatible portfolio environment. Spec §11, DECISIONS.md D-034 .. D-036.

One step is one weekly decision (D-001). At step ``k``::

    observe at close d_k   state vector, drifted weights, mean turnover so far
    act                    a in [-1, 1]^(n+1) -> target weights (see actions.py)
    execute at close e_k   trade from the weights drifted to e_k; pay the cost
    earn                   the target weights drift through (e_k, e_{k+1}], day by day
    reward                 from that window, net of the cost (see rewards.py)

Two weight vectors are tracked because the drift between ``d_k`` and ``e_k``
belongs to the previous step: ``w_obs`` (drifted to ``d_k``, what the agent sees)
and ``w_exec`` (drifted to ``e_k``, what the trade starts from).

**Observation** = the variant's state vector, then the portfolio block: the
``n + 1`` current weights and the mean one-way turnover per step so far in the
episode. Time-since-last-rebalance (spec §10) is omitted: with a decision every
week it is constant. Cumulative turnover is divided by the step count so it
stays bounded and carries no episode-position information (D-035).

**Episodes.** ``mode="train"``: a random start inside the data and
``episode_length`` decisions (truncated, so the next observation exists for
bootstrapping). ``mode="eval"``: one fixed episode over the whole data, ending
in ``terminated`` at its last decision. The portfolio starts in cash and the
initiation is a trade.

**No look-ahead.** Row ``t`` of any array is only read for ``t <= `` the close the
step is at; ``tests/test_env.py`` perturbs everything after it and checks nothing
changes, and checks that the same test fails on a deliberately leaky env.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from prism.env.actions import action_to_weights, upper_bounds
from prism.env.costs import CostModel, one_way_turnover
from prism.env.data import EnvData
from prism.env.rewards import Reward, StepRecord, make_reward

__all__ = ["PortfolioEnv", "make_env", "run_episode"]


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: EnvData,
        cost_model: CostModel,
        reward: Reward,
        *,
        cap: float,
        logit_scale: float,
        mode: str = "eval",
        episode_length: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if data.n_decisions < 2:
            raise ValueError("need at least 2 decisions")
        if mode == "train" and (episode_length is None or episode_length < 2):
            raise ValueError("train mode needs episode_length >= 2")
        if not 0.0 < cap <= 1.0:
            raise ValueError("cap must be in (0, 1]")
        self.data, self.cost_model, self.reward_fn = data, cost_model, reward
        self.mode = mode
        self.episode_length = (
            min(int(episode_length), data.n_decisions) if mode == "train" else data.n_decisions
        )
        self.logit_scale = float(logit_scale)
        self._upper = upper_bounds(data.n_risky, cap)
        self._n = data.n_risky + 1
        obs_dim = data.states.shape[1] + self._n + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self._n,), dtype=np.float32)
        self._k = self._first = self._last = 0
        self._steps = 0
        self._w_obs = self._w_exec = np.zeros(self._n)
        self._nav = self._peak = 1.0
        self._cum_turnover = 0.0

    # ------------------------------------------------------------------ helpers
    @property
    def n_risky(self) -> int:
        return self.data.n_risky

    @property
    def current_decision_date(self) -> pd.Timestamp:
        return self.data.sessions[self.data.decision_pos[self._k]]

    def _cash(self) -> np.ndarray:
        w = np.zeros(self._n)
        w[-1] = 1.0
        return w

    def _observation(self, state_pos: int, weights: np.ndarray) -> np.ndarray:
        mean_turn = self._cum_turnover / self._steps if self._steps else 0.0
        obs = np.concatenate([self.data.states[state_pos], weights, [mean_turn]]).astype(np.float32)
        if not np.isfinite(obs).all():
            raise RuntimeError("non-finite observation")
        return obs

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        n_dec = self.data.n_decisions
        if self.mode == "train":
            self._first = int(self.np_random.integers(0, n_dec - self.episode_length + 1))
        else:
            self._first = 0
        self._last = self._first + self.episode_length - 1
        self._k = self._first
        self._steps = 0
        self._w_obs = self._w_exec = self._cash()
        self._nav = self._peak = 1.0
        self._cum_turnover = 0.0
        self.reward_fn.reset()
        obs = self._observation(int(self.data.decision_pos[self._k]), self._w_obs)
        return obs, {"decision_date": self.current_decision_date, "nav": self._nav}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self.step_weights(action_to_weights(action, self._upper, self.logit_scale))

    @property
    def drifted_weights(self) -> np.ndarray:
        """The weights as drifted to the next execution close: what a pure hold would keep."""
        return self._w_exec.copy()

    def step_weights(self, target: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """One step with explicit target weights (``step`` is this after the action map).

        Used by the Tier 2 benchmarks, whose weights (an SPY line, a hold) are not
        reachable through the action map. The cost, drift and reward are the same code path.
        """
        target = np.asarray(target, dtype="float64")
        if target.shape != (self._n,) or not np.isfinite(target).all():
            raise ValueError("target weights must be a finite vector over the risky lines and cash")
        d, k = self.data, self._k
        e_pos, end_pos = int(d.exec_pos[k]), int(d.end_pos[k])
        has_next = k < d.n_decisions - 1
        next_d_pos = int(d.decision_pos[k + 1]) if has_next else end_pos

        sigma = d.sigma[e_pos]
        cost = self.cost_model.trade_cost(self._w_exec, target, sigma)
        turnover = one_way_turnover(self._w_exec, target)
        traded = self.cost_model.traded_notional(self._w_exec, target)

        # The target weights earn (e_k, e_{k+1}], drifting day by day.
        w = target.copy()
        w_at_next_decision = w.copy() if next_d_pos <= e_pos else None
        daily = np.empty(end_pos - e_pos)
        for i, pos in enumerate(range(e_pos + 1, end_pos + 1)):
            r = d.line_returns[pos]
            port = float(w @ r)
            daily[i] = port
            w = w * (1.0 + r) / (1.0 + port)
            if pos == next_d_pos:
                w_at_next_decision = w.copy()
        if w_at_next_decision is None:
            raise RuntimeError("the next decision falls before the previous execution; check the lag")
        gross = float(np.prod(1.0 + daily) - 1.0)
        net = (1.0 - cost) * (1.0 + gross) - 1.0
        self._nav *= 1.0 + net
        self._peak = max(self._peak, self._nav)
        reward = self.reward_fn(
            StepRecord(gross, cost, net, daily, self._nav, self._peak)
        )
        if not np.isfinite(reward):
            raise RuntimeError("non-finite reward")

        self._cum_turnover += turnover
        self._steps += 1
        self._w_exec = w
        self._w_obs = w_at_next_decision
        info = {
            "decision_date": d.sessions[d.decision_pos[k]],
            "execution_date": d.sessions[e_pos],
            "window_end": d.sessions[end_pos],
            "weights": target,
            "cost": cost,
            "turnover": turnover,
            "traded_notional": traded,
            "gross_return": gross,
            "daily_returns": daily,
            "net_return": net,
            "nav": self._nav,
        }
        terminated = not has_next or (self.mode == "eval" and k == self._last)
        truncated = (not terminated) and k == self._last
        if terminated or truncated:
            self._k = k
        else:
            self._k = k + 1
        state_pos = next_d_pos  # the next decision close, or the window's last close at the data end
        return self._observation(state_pos, self._w_obs), float(reward), terminated, truncated, info


def make_env(cfg, data: EnvData, *, mode: str = "eval", cost_model: CostModel | None = None) -> PortfolioEnv:  # noqa: ANN001
    """The environment as configured in ``configs/env.yaml`` and ``configs/data.yaml``."""
    return PortfolioEnv(
        data,
        cost_model if cost_model is not None else CostModel.from_config(cfg),
        make_reward(cfg.env.reward),
        cap=cfg.data.allocation.weight_max,
        logit_scale=cfg.env.action.logit_scale,
        mode=mode,
        episode_length=cfg.env.episode.train_length_decisions,
    )


def run_episode(
    env: PortfolioEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run one episode and return one row per step (date, weights, cost, returns, NAV, reward)."""
    obs, _ = env.reset(seed=seed)
    rows = []
    while True:
        obs_in = obs
        obs, reward, terminated, truncated, info = env.step(policy(obs_in))
        rows.append({**{k: v for k, v in info.items() if k not in ("weights", "daily_returns")}, "reward": reward})
        rows[-1].update({f"w_{name}": w for name, w in zip(env.data.lines, info["weights"])})
        if terminated or truncated:
            break
    return pd.DataFrame(rows).set_index("decision_date")
