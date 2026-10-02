"""The pre-built arrays the environment runs on. Spec §11, DECISIONS.md D-035.

"The env holds a pre-built state array and a pre-built return array; it may
never index beyond the current step" (§11). :class:`EnvData` is those arrays,
plus the holding-period schedule from :mod:`prism.backtest.engine` (the single
statement of the observe -> decide -> execute -> earn contract).

Timing, with decisions on Fridays and ``execution_lag_days = 1``::

    d_k   decision close: the state and the drifted weights are observed here
    e_k   execution close = d_k + lag: weights are traded to the target here
    reward window  (e_k, e_{k+1}]: the return the target weights earn

The window starts strictly after ``d_k`` whenever ``lag >= 1``, so the reward
never contains a return the observation already used. The return between
``d_k`` and ``e_k`` is still earned by the *previous* weights and belongs to the
previous step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prism.backtest.engine import TimingConvention, holding_periods

__all__ = ["EnvData", "line_returns", "build_env_data"]

CASH = "CASH"


@dataclass(frozen=True)
class EnvData:
    """Everything the environment reads, as aligned arrays over ``sessions``."""

    sessions: pd.DatetimeIndex
    states: np.ndarray            # (T, d) float32 — the variant's state vector
    state_columns: tuple[str, ...]
    lines: tuple[str, ...]        # risky assets, then CASH
    line_returns: np.ndarray      # (T, n+1) float64 simple returns; row t is close t-1 -> t
    sigma: np.ndarray             # (T, n) float64 trailing daily vol, known at close t
    decision_pos: np.ndarray      # (N,) int positions of the decision closes d_k
    exec_pos: np.ndarray          # (N,) positions of the execution closes e_k
    end_pos: np.ndarray           # (N,) positions of the last close of each reward window

    @property
    def n_risky(self) -> int:
        return len(self.lines) - 1

    @property
    def n_decisions(self) -> int:
        return len(self.decision_pos)

    @classmethod
    def from_arrays(
        cls,
        sessions: pd.DatetimeIndex,
        states: np.ndarray,
        line_returns: np.ndarray,
        sigma: np.ndarray,
        *,
        lines: tuple[str, ...],
        state_columns: tuple[str, ...] | None = None,
        convention: TimingConvention,
    ) -> EnvData:
        sessions = pd.DatetimeIndex(sessions)
        states = np.asarray(states, dtype="float32")
        line_returns = np.asarray(line_returns, dtype="float64")
        sigma = np.asarray(sigma, dtype="float64")
        t = len(sessions)
        if lines[-1] != CASH:
            raise ValueError("the last line must be CASH")
        if states.shape[0] != t or line_returns.shape != (t, len(lines)) or sigma.shape != (t, len(lines) - 1):
            raise ValueError(
                f"inconsistent shapes: sessions {t}, states {states.shape}, "
                f"line_returns {line_returns.shape}, sigma {sigma.shape}, lines {len(lines)}"
            )
        if not np.isfinite(states).all():
            raise ValueError("the state array contains NaN/inf")
        periods = holding_periods(sessions, convention)
        pos = sessions.get_indexer
        data = cls(
            sessions=sessions,
            states=states,
            state_columns=tuple(state_columns) if state_columns is not None
            else tuple(f"x{i}" for i in range(states.shape[1])),
            lines=tuple(lines),
            line_returns=line_returns,
            sigma=sigma,
            decision_pos=pos(pd.DatetimeIndex(periods["decision_date"])),
            exec_pos=pos(pd.DatetimeIndex(periods["execution_date"])),
            end_pos=pos(pd.DatetimeIndex(periods["return_end"])),
        )
        for k in range(data.n_decisions):
            window = slice(data.exec_pos[k] + 1, data.end_pos[k] + 1)
            if not np.isfinite(line_returns[window]).all():
                raise ValueError(f"non-finite line return inside the reward window of decision {k}")
            if not np.isfinite(sigma[data.exec_pos[k]]).all():
                raise ValueError(f"non-finite volatility at the execution close of decision {k}")
        return data


def line_returns(close: pd.DataFrame, risky: list[str], rf_column: str = "^IRX") -> pd.DataFrame:
    """Simple per-session returns of the risky assets and the cash line.

    The same construction as ``prism.probes.allocator.prepare_inputs`` (checked
    against it in the tests): cash earns the PRIOR session's ``rf_column`` close,
    quoted in percent annualised, divided by 100 and by 252, so no rate is used
    before it was published.
    """
    simple = close[risky].astype("float64").pct_change()
    rf = (close[rf_column].astype("float64").shift(1) / 100.0 / 252.0).rename(CASH)
    return pd.concat([simple, rf], axis=1)[[*risky, CASH]]


def build_env_data(
    cfg,  # noqa: ANN001 - prism.config.Config
    state: pd.DataFrame,
    close: pd.DataFrame,
    split: str,
    *,
    plan=None,  # noqa: ANN001 - prism.splits.SplitPlan
    risky: list[str] | None = None,
) -> EnvData:
    """Arrays for one variant's ``state`` frame, restricted to one split.

    ``split`` is the split whose *effective* (embargo-purged) range the
    sessions are cut to, so no reward window crosses into an embargo or into
    the next split. The holdout is refused outright. ``risky`` overrides the
    asset list (the Tier 2 benchmarks add SPY, which is a benchmark and not an
    agent asset, D-035); the default is ``data.allocatable``.
    """
    from prism.data.loaders import assert_not_holdout
    from prism.splits import build_split_plan

    if split == "holdout":
        raise PermissionError("the holdout is locked until build step 5 (spec §6.3)")
    plan = plan if plan is not None else build_split_plan(cfg)
    risky = list(risky) if risky is not None else list(cfg.data.allocatable[cfg.env.universe])

    lr_full = line_returns(close, risky)
    vol_full = lr_full[risky].rolling(cfg.env.costs.vol_window).std()

    sessions = pd.DatetimeIndex(state.index).intersection(close.index).sort_values()
    sessions = sessions[(sessions >= plan[split].effective_start) & (sessions <= plan[split].effective_end)]
    assert_not_holdout(cfg, sessions, context=f"env data ({split})")
    if len(sessions) < 2:
        raise ValueError(f"split {split!r} has fewer than 2 sessions in the state frame")
    # Returns are per-session: the sessions must be consecutive in the price panel.
    gaps = np.diff(close.index.get_indexer(sessions))
    if (gaps != 1).any():
        raise ValueError("state sessions are not consecutive in the price panel; returns would span gaps")

    return EnvData.from_arrays(
        sessions,
        state.loc[sessions].to_numpy(),
        lr_full.reindex(sessions).to_numpy(),
        vol_full.reindex(sessions).to_numpy(),
        lines=(*risky, CASH),
        state_columns=tuple(str(c) for c in state.columns),
        convention=TimingConvention.from_config(cfg),
    )
