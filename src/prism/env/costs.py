"""Transaction cost: proportional per-side cost plus volatility-scaled slippage. Spec §11.

**Convention (DECISIONS.md D-034).** The cost of moving from weights ``w_from``
to ``w_to`` is a fraction of portfolio value::

    cost = sum over risky assets i of |w_to_i - w_from_i| * (per_side_bps / 1e4
                                                             + slippage_vol_coef * sigma_i)

* ``per_side_bps`` is charged on **each leg's traded notional**: a purchase and
  a sale each pay it. A round trip into and out of a position therefore costs
  ``2 * per_side_bps`` of that position (10 bps at the 5 bps default), which is
  the spec §7.5 statement "a full round trip costs 10 bps of the traded notional".
* The cash line is free to trade; it is the residual of the other trades.
* ``sigma_i`` is asset ``i``'s trailing daily return standard deviation known at
  the execution close, so slippage is ``slippage_vol_coef`` times one daily
  standard deviation per unit traded. It rises in stress, which is the point.

This is **not** the convention of the Tier 1 allocator (``prism.probes.allocator
.net_returns``: ``bps * 0.5 * sum|dw|`` with cash included), which charges a
100% risky-to-risky swap 5 bps where this charges it 10. Strategies compared
with Tier 1 numbers must be re-run through this model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CostModel", "one_way_turnover"]


def one_way_turnover(w_from: np.ndarray, w_to: np.ndarray) -> float:
    """``0.5 * sum |w_to - w_from|`` over all lines including cash.

    The definition of :func:`prism.backtest.metrics.turnover_series`, reported
    alongside the cost so turnover is comparable across the project. It is not
    what the cost is charged on.
    """
    return float(0.5 * np.abs(np.asarray(w_to, "float64") - np.asarray(w_from, "float64")).sum())


@dataclass(frozen=True)
class CostModel:
    """Weights are vectors over ``n`` risky assets followed by one cash line."""

    per_side_bps: float
    slippage_vol_coef: float

    def __post_init__(self) -> None:
        if self.per_side_bps < 0 or self.slippage_vol_coef < 0:
            raise ValueError("cost parameters must be non-negative")

    @classmethod
    def from_config(cls, cfg) -> CostModel:  # noqa: ANN001
        c = cfg.env.costs
        return cls(per_side_bps=c.per_side_bps, slippage_vol_coef=c.slippage_vol_coef)

    def traded_notional(self, w_from: np.ndarray, w_to: np.ndarray) -> float:
        """``sum |dw|`` over the risky assets: what the per-side cost is charged on."""
        dw = np.asarray(w_to, "float64") - np.asarray(w_from, "float64")
        return float(np.abs(dw[:-1]).sum())

    def trade_cost(self, w_from: np.ndarray, w_to: np.ndarray, sigma: np.ndarray) -> float:
        """Cost of the trade as a fraction of portfolio value. ``sigma`` is per risky asset."""
        w_from, w_to = np.asarray(w_from, "float64"), np.asarray(w_to, "float64")
        sigma = np.asarray(sigma, "float64")
        if w_from.shape != w_to.shape or sigma.shape != (w_to.shape[0] - 1,):
            raise ValueError(
                f"shapes: w_from {w_from.shape}, w_to {w_to.shape}, sigma {sigma.shape} "
                "(sigma must have one entry per risky asset, i.e. one fewer than the weights)"
            )
        if not np.isfinite(sigma).all():
            raise ValueError("non-finite volatility passed to the cost model")
        dw = np.abs(w_to[:-1] - w_from[:-1])
        return float(dw @ (self.per_side_bps / 1e4 + self.slippage_vol_coef * sigma))

    def scaled(self, bps: float) -> CostModel:
        """The model at ``bps`` per side, with slippage scaled by the same factor.

        ``bps = 0`` is frictionless, ``bps = 2 * per_side_bps`` doubles both
        terms. This is how the spec §11 sensitivity grid (0/5/10/20 bps) is
        reported without a separate slippage knob.
        """
        if bps < 0:
            raise ValueError("bps must be non-negative")
        if self.per_side_bps == 0:
            if bps != 0:
                raise ValueError("cannot scale up from a zero-cost model")
            return self
        factor = bps / self.per_side_bps
        return CostModel(per_side_bps=bps, slippage_vol_coef=self.slippage_vol_coef * factor)

    def sensitivity(self, grid: list[float]) -> dict[float, CostModel]:
        return {float(b): self.scaled(b) for b in grid}
