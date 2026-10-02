"""Reward variants. Spec §11, DECISIONS.md D-036.

Reward choice is a documented experimental factor, not a silent default: the
default is ``log_return_net`` and the others are selected by name in
``configs/env.yaml``. None of the parameters below was tuned.

Every reward is a function of one holding period's :class:`StepRecord`: the
period is ``(execution close, next execution close]``, so it starts strictly
after the close on which the decision was observed (spec §11: "assert in tests
that the reward window does not overlap the observation window").

* ``log_return_net`` — ``log(1 + net return)``; net of the cost paid to reach
  the weights, so costs are in the reward by construction.
* ``dsr`` — Moody & Saffell's differential Sharpe ratio on the net period
  return: with exponential moments ``A, B`` of ``R`` and ``R^2`` (rate ``eta``),
  ``D = (B*dA - A*dB/2) / (B - A^2)^(3/2)``, evaluated with the moments *before*
  this step's update. Zero while the variance estimate is degenerate.
* ``mv_penalty`` — ``log net return - lambda * sum_t (r_t - mean r)^2`` over the
  period's daily portfolio returns (realised variance, not annualised).
* ``drawdown_penalty`` — ``log net return - lambda * DD`` with ``DD = 1 - NAV /
  peak NAV`` after the step, within the episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["StepRecord", "Reward", "make_reward", "REWARD_NAMES"]

REWARD_NAMES = ("log_return_net", "dsr", "mv_penalty", "drawdown_penalty")
_DSR_EPS = 1e-12


@dataclass(frozen=True)
class StepRecord:
    """One holding period, as the reward sees it."""

    gross_return: float        # simple return of the period, before the cost
    cost: float                # cost paid at the period's start, fraction of NAV
    net_return: float          # (1 - cost) * (1 + gross) - 1
    daily_returns: np.ndarray  # daily portfolio returns inside the period
    nav: float                 # NAV at the period's end
    peak_nav: float            # running peak of NAV, this period included


class Reward:
    """Stateful over an episode; :meth:`reset` is called at every episode start."""

    name = ""

    def reset(self) -> None:
        pass

    def __call__(self, rec: StepRecord) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class LogReturnNet(Reward):
    name = "log_return_net"

    def __call__(self, rec: StepRecord) -> float:
        return float(np.log1p(rec.net_return))


class DifferentialSharpe(Reward):
    name = "dsr"

    def __init__(self, eta: float) -> None:
        if not 0.0 < eta < 1.0:
            raise ValueError("eta must be in (0, 1)")
        self.eta = eta
        self.reset()

    def reset(self) -> None:
        self.a = 0.0
        self.b = 0.0

    def __call__(self, rec: StepRecord) -> float:
        r = rec.net_return
        d_a, d_b = r - self.a, r * r - self.b
        var = self.b - self.a * self.a
        value = 0.0 if var <= _DSR_EPS else (self.b * d_a - 0.5 * self.a * d_b) / var**1.5
        self.a += self.eta * d_a
        self.b += self.eta * d_b
        return float(value)


class MeanVariancePenalty(Reward):
    name = "mv_penalty"

    def __init__(self, lam: float) -> None:
        self.lam = lam

    def __call__(self, rec: StepRecord) -> float:
        d = np.asarray(rec.daily_returns, "float64")
        return float(np.log1p(rec.net_return) - self.lam * ((d - d.mean()) ** 2).sum())


class DrawdownPenalty(Reward):
    name = "drawdown_penalty"

    def __init__(self, lam: float) -> None:
        self.lam = lam

    def __call__(self, rec: StepRecord) -> float:
        return float(np.log1p(rec.net_return) - self.lam * (1.0 - rec.nav / rec.peak_nav))


def make_reward(cfg_reward) -> Reward:  # noqa: ANN001
    """Build the configured reward from ``cfg.env.reward``."""
    name = cfg_reward.name
    if name == "log_return_net":
        return LogReturnNet()
    if name == "dsr":
        return DifferentialSharpe(cfg_reward.dsr_eta)
    if name == "mv_penalty":
        return MeanVariancePenalty(cfg_reward.mv_lambda)
    if name == "drawdown_penalty":
        return DrawdownPenalty(cfg_reward.drawdown_lambda)
    raise ValueError(f"unknown reward {name!r}; known: {REWARD_NAMES}")
