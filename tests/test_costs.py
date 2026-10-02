"""Transaction-cost model. Spec §7.5, §11; DECISIONS.md D-034.

Every expected value below is computed by hand from the stated convention:
``cost = sum_risky |dw_i| * (per_side_bps/1e4 + slippage_vol_coef * sigma_i)``,
cash free, as a fraction of portfolio value.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism.config import load_config
from prism.env.costs import CostModel, one_way_turnover

BPS = 1e-4
W_CASH = np.array([0.0, 0.0, 0.0, 1.0])            # 3 risky assets then cash
SIGMA = np.array([0.01, 0.02, 0.005])


def _model(per_side: float = 5.0, coef: float = 0.02) -> CostModel:
    return CostModel(per_side_bps=per_side, slippage_vol_coef=coef)


def test_zero_turnover_implies_zero_cost():
    """§7.5. Holding any portfolio, however risky, costs nothing."""
    w = np.array([0.3, 0.2, 0.1, 0.4])
    assert _model().trade_cost(w, w, SIGMA) == 0.0
    assert _model().trade_cost(W_CASH, W_CASH, SIGMA) == 0.0


def test_cost_is_monotone_in_turnover():
    """§7.5. Trading more of the same trade never costs less."""
    target = np.array([0.35, 0.25, 0.1, 0.3])
    costs = [
        _model().trade_cost(W_CASH, W_CASH + s * (target - W_CASH), SIGMA)
        for s in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert costs[0] == 0.0
    assert all(b > a for a, b in zip(costs, costs[1:]))


def test_round_trip_cost_matches_the_analytic_value():
    """§7.5: at 5 bps per side a round trip costs 10 bps of the traded position, nothing else."""
    pos = 0.35
    into = np.array([pos, 0.0, 0.0, 1.0 - pos])
    model = _model(per_side=5.0, coef=0.0)           # no slippage: the proportional term alone
    buy = model.trade_cost(W_CASH, into, SIGMA)
    sell = model.trade_cost(into, W_CASH, SIGMA)
    assert buy == pytest.approx(pos * 5 * BPS, abs=1e-15)
    assert buy + sell == pytest.approx(pos * 10 * BPS, abs=1e-15)


def test_cash_trades_are_free():
    """Only the risky legs are charged; moving between cash and cash does nothing."""
    model = _model()
    assert model.traded_notional(np.array([0, 0, 0, 1.0]), np.array([0, 0, 0, 1.0])) == 0.0
    # Buying 0.2 of asset 0 funded from cash charges 0.2 of notional, not 0.4.
    w = np.array([0.2, 0.0, 0.0, 0.8])
    assert model.traded_notional(W_CASH, w) == pytest.approx(0.2)


def test_slippage_is_volatility_scaled_and_additive():
    """Analytic: 0.1 traded in an asset with daily vol 2% at coef 0.02 -> 0.1 * (5e-4 + 0.02*0.02)."""
    w = np.array([0.0, 0.1, 0.0, 0.9])
    expected = 0.1 * (5 * BPS + 0.02 * 0.02)
    assert _model().trade_cost(W_CASH, w, SIGMA) == pytest.approx(expected, abs=1e-15)
    # Slippage rises with volatility, and is zero at zero volatility.
    calm, stressed = np.full(3, 0.005), np.full(3, 0.04)
    assert _model().trade_cost(W_CASH, w, stressed) > _model().trade_cost(W_CASH, w, calm)
    assert _model().trade_cost(W_CASH, w, np.zeros(3)) == pytest.approx(0.1 * 5 * BPS, abs=1e-15)


def test_the_cost_is_linear_in_the_trade():
    a, b = np.array([0.1, 0.0, 0.0, 0.9]), np.array([0.0, 0.2, 0.0, 0.8])
    model = _model()
    both = np.array([0.1, 0.2, 0.0, 0.7])
    assert model.trade_cost(W_CASH, both, SIGMA) == pytest.approx(
        model.trade_cost(W_CASH, a, SIGMA) + model.trade_cost(W_CASH, b, SIGMA), abs=1e-15
    )


def test_a_swap_costs_twice_the_tier1_allocator_convention():
    """D-034: the env charges both legs; the Tier 1 allocator charged 0.5 * sum|dw| (cash included).

    Swapping 0.3 from asset 0 to asset 1 is 0.3 sold and 0.3 bought, 0.6 of
    notional: 3 bps here at 5 bps per side, against 0.5 * (0.3 + 0.3) * 5 bps =
    1.5 bps in the allocator. (A 100% swap is 10 bps against 5.)
    """
    a, b = np.array([0.3, 0.0, 0.0, 0.7]), np.array([0.0, 0.3, 0.0, 0.7])
    env_cost = _model(coef=0.0).trade_cost(a, b, SIGMA)
    allocator_cost = 5 * BPS * one_way_turnover(a, b)
    assert env_cost == pytest.approx(3 * BPS, abs=1e-15)
    assert allocator_cost == pytest.approx(1.5 * BPS, abs=1e-15)
    assert env_cost == pytest.approx(2 * allocator_cost, rel=1e-12)
    # ...whereas a pure cash <-> risky trade is charged identically by both.
    buy = np.array([0.3, 0.0, 0.0, 0.7])
    assert _model(coef=0.0).trade_cost(W_CASH, buy, SIGMA) == pytest.approx(
        5 * BPS * one_way_turnover(W_CASH, buy), abs=1e-15
    )


def test_cost_sensitivity_is_reported_at_0_5_10_20_bps():
    """§11: the grid scales both terms by bps / per_side_bps; 0 is frictionless."""
    cfg = load_config()
    base = CostModel.from_config(cfg)
    grid = base.sensitivity(cfg.env.costs.sensitivity_bps)
    assert sorted(grid) == [0.0, 5.0, 10.0, 20.0]
    w = np.array([0.2, 0.1, 0.0, 0.7])
    costs = {b: m.trade_cost(W_CASH, w, SIGMA) for b, m in grid.items()}
    assert costs[0.0] == 0.0
    assert costs[5.0] == pytest.approx(base.trade_cost(W_CASH, w, SIGMA), abs=1e-15)
    assert costs[10.0] == pytest.approx(2 * costs[5.0], rel=1e-12)
    assert costs[20.0] == pytest.approx(4 * costs[5.0], rel=1e-12)
    assert grid[10.0].per_side_bps == 10.0 and grid[10.0].slippage_vol_coef == pytest.approx(
        2 * base.slippage_vol_coef
    )


def test_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError):
        CostModel(per_side_bps=-1.0, slippage_vol_coef=0.0)
    with pytest.raises(ValueError, match="sigma must have one entry per risky asset"):
        _model().trade_cost(W_CASH, W_CASH, np.zeros(4))
    with pytest.raises(ValueError, match="non-finite"):
        _model().trade_cost(W_CASH, W_CASH, np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(ValueError, match="cannot scale up"):
        _model(per_side=0.0).scaled(5.0)
