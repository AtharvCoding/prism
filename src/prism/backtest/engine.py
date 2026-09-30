"""Backtest timing and the weight -> return path.

Spec §14. The full engine — costs, slippage, benchmark comparison — is build
step 4a. What is implemented **now** is the part §7.3 requires a Step 0 test
for: the timing contract.

The timing contract
-------------------
Getting this wrong is the quietest possible way to manufacture a result, so
it is stated once, here, and asserted against a hand-computed answer in
``tests/test_alignment.py``::

    observe at close t  ->  decide w_t  ->  execute at close t + lag
                        ->  earn the return from t + lag to the next decision

With ``execution_lag_days: 1``, a decision taken at Friday's close is
executed at Monday's close, and the return it earns begins from Monday's
close. The decision at ``t`` therefore earns **nothing** over ``t -> t+1``:
that period is still held at the previous weights. Charging a decision with
the return of the day it was merely *decided* on is a one-day look-ahead
worth several points of annual Sharpe on daily data.

The reference design left reward alignment entirely to downstream code
(defect A5) — there was no aligned reward series at all, only feature splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prism.utils.calendar import rebalance_dates

__all__ = [
    "TimingConvention",
    "decision_schedule",
    "holding_periods",
    "simple_returns",
    "portfolio_period_returns",
]


@dataclass(frozen=True)
class TimingConvention:
    """The observe -> decide -> execute -> earn convention, as data."""

    frequency: str
    rebalance_day: str
    execution_lag_days: int

    @classmethod
    def from_config(cls, cfg) -> TimingConvention:  # noqa: ANN001
        d = cfg.data.decision
        return cls(
            frequency=d.frequency,
            rebalance_day=d.rebalance_day,
            execution_lag_days=d.execution_lag_days,
        )


def decision_schedule(
    index: pd.DatetimeIndex, convention: TimingConvention
) -> pd.DatetimeIndex:
    """Sessions on which a decision is taken, drawn from ``index``."""
    return rebalance_dates(index, convention.frequency, convention.rebalance_day)


def holding_periods(
    index: pd.DatetimeIndex, convention: TimingConvention
) -> pd.DataFrame:
    """One row per decision: when it was decided, executed, and what it earns.

    Columns
    -------
    ``decision_date``
        The close at which the state was observed and the weights chosen.
    ``execution_date``
        ``decision_date`` plus ``execution_lag_days`` sessions. Weights are
        in force from this close.
    ``return_start`` / ``return_end``
        The half-open holding period, ``(return_start, return_end]``. The
        return earned is the portfolio's growth from the close of
        ``return_start`` to the close of ``return_end``, so ``return_start``
        equals ``execution_date``.

    The final decision is dropped when its execution date would fall outside
    ``index``, rather than being silently held to the end of the sample.
    """
    idx = pd.DatetimeIndex(index)
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("index must be sorted and unique")
    decisions = decision_schedule(idx, convention)
    lag = convention.execution_lag_days

    positions = idx.get_indexer(decisions)
    rows = []
    for i, (date, pos) in enumerate(zip(decisions, positions)):
        exec_pos = pos + lag
        if exec_pos >= len(idx):
            break
        # The period runs until the next decision is executed.
        if i + 1 < len(decisions):
            next_exec = int(positions[i + 1]) + lag
            end_pos = min(next_exec, len(idx) - 1)
        else:
            end_pos = len(idx) - 1
        if end_pos <= exec_pos:
            continue
        rows.append(
            {
                "decision_date": date,
                "execution_date": idx[exec_pos],
                "return_start": idx[exec_pos],
                "return_end": idx[end_pos],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("no complete holding periods fit inside this index")
    return out


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (not log) one-period returns.

    Portfolio arithmetic is linear in *simple* returns — a weighted sum of
    log returns is not the log return of the weighted portfolio. Compounding
    log returns as if simple was defect A10.
    """
    return prices.astype("float64").pct_change()


def portfolio_period_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    convention: TimingConvention,
    *,
    cash_return: float = 0.0,
) -> pd.Series:
    """Gross portfolio return per holding period, honouring the timing contract.

    Parameters
    ----------
    weights
        Indexed by **decision date**, one column per asset. Any weight not
        summing to 1 is treated as held in cash earning ``cash_return`` per
        period, which is how the long-only-with-cash constraint (§3.3
        decision 2) is expressed.
    prices
        Total-return prices, indexed by session.

    Returns
    -------
    A series indexed by **decision date**, whose value is the return realised
    over that decision's holding period. Indexing by the decision date rather
    than the earning date is deliberate: it makes the pairing of a state
    observation with its outcome explicit, and the index shift is then
    impossible to lose track of.
    """
    periods = holding_periods(pd.DatetimeIndex(prices.index), convention)
    assets = [c for c in weights.columns if c in prices.columns]
    missing = [c for c in weights.columns if c not in prices.columns]
    if missing:
        raise KeyError(f"weights name assets absent from prices: {missing}")

    out: dict[pd.Timestamp, float] = {}
    for row in periods.itertuples(index=False):
        if row.decision_date not in weights.index:
            continue
        w = weights.loc[row.decision_date, assets].astype("float64").fillna(0.0)
        p0 = prices.loc[row.return_start, assets].astype("float64")
        p1 = prices.loc[row.return_end, assets].astype("float64")
        growth = (p1 / p0).to_numpy()
        if not np.isfinite(growth).all():
            continue
        cash_weight = 1.0 - float(w.sum())
        out[row.decision_date] = float(
            (w.to_numpy() * growth).sum() + cash_weight * (1.0 + cash_return) - 1.0
        )
    return pd.Series(out, name="gross_return").sort_index()
