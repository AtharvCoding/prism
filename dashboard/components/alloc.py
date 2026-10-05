"""Weights by strategy, sleeve and week, from the recorded replay. Descriptive transforms only; no statistic for the verdict.

An *ensemble* here is the plain average of a variant's ten agents' target weights on each decision date. It describes
what the ten held on average; no such averaged portfolio was evaluated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BENCHMARKS = {"Equal weight": "EqualWeight", "60/40": "SixtyForty", "Risk parity": "RiskParity", "Minimum variance": "MinVariance",
              "Vol target": "VolTarget", "Buy and hold SPY": "BuyHoldSPY"}


def strategies(variants: list[str]) -> list[str]:
    ordered = [v for v in ("V4", "V2", "V1", "C4") if v in variants]
    return [f"{v} ensemble" for v in ordered] + list(BENCHMARKS)


def lines(universe: dict[str, Any], *, with_benchmark: bool = False) -> list[str]:
    spy = [universe["benchmark"]] if with_benchmark else []
    return [*universe["sectors"], *spy, *universe["bonds"], *universe["gold"], universe["cash"]]


def sleeve_of(universe: dict[str, Any]) -> dict[str, str]:
    """Line -> display sleeve. The S&P 500 fund, which only the 60/40 and buy-and-hold benchmarks hold, counts as Equity."""
    out = {t: "Equity" for t in [*universe["sectors"], universe["benchmark"]]}
    out.update({t: "Bonds" for t in universe["bonds"]})
    out.update({t: "Gold" for t in universe["gold"]})
    out[universe["cash"]] = "Cash"
    return out


def _wide(frame: pd.DataFrame, prefix: str = "w_") -> pd.DataFrame:
    cols = [c for c in frame.columns if c.startswith(prefix)]
    return frame.set_index("decision_date")[cols].rename(columns=lambda c: c[len(prefix):])


def strategy_weights(agents: pd.DataFrame, benchmarks: pd.DataFrame, strategy: str) -> dict[str, pd.DataFrame | None]:
    """Target weights per decision date for one strategy: ``mean`` and, for an ensemble, the seeds' ``low`` and ``high``."""
    if strategy in BENCHMARKS:
        return {"mean": _wide(benchmarks[benchmarks.benchmark == BENCHMARKS[strategy]]), "low": None, "high": None, "turnover": None}
    variant = strategy.split()[0]
    sub = agents[agents.variant == variant]
    cols = [c for c in sub.columns if c.startswith("w_")]
    by_date = sub.groupby("decision_date")[cols]
    strip = lambda f: f.rename(columns=lambda c: c[2:])  # noqa: E731
    return {"mean": strip(by_date.mean()), "low": strip(by_date.min()), "high": strip(by_date.max()),
            "turnover": sub.groupby("decision_date").turnover.mean()}


def sleeves(weights: pd.DataFrame, universe: dict[str, Any]) -> pd.DataFrame:
    """Weights summed into Equity, Bonds, Gold and Cash, in that order."""
    mapping = sleeve_of(universe)
    out = weights.T.groupby(lambda line: mapping[line]).sum().T
    return out.reindex(columns=["Equity", "Bonds", "Gold", "Cash"], fill_value=0.0)


def defensive_share(weights: pd.DataFrame, universe: dict[str, Any]) -> pd.Series:
    """Bonds + gold + cash: everything that is not equity."""
    s = sleeves(weights, universe)
    return (s["Bonds"] + s["Gold"] + s["Cash"]).rename("defensive")


def correlation(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation of two aligned weekly series (descriptive)."""
    both = pd.concat([x, y], axis=1).dropna()
    return float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])


def equity_curves(daily: pd.DataFrame, names: list[str], bps: float) -> pd.DataFrame:
    """Growth of 1 from the stored daily net returns, one column per series name (``V4|s3`` or ``BM|EqualWeight``)."""
    return (1.0 + daily[[f"{n}|{bps:g}" for n in names]]).cumprod().set_axis(names, axis=1)


def interpolate_costs(cost: pd.DataFrame, bps: float, column: str = "sharpe") -> pd.Series:
    """Each strategy's stored value at ``bps``: exact at a stored cost level, linear between two neighbouring levels."""
    out = {}
    for strategy, g in cost.groupby("strategy"):
        g = g.sort_values("bps")
        if not g.bps.min() <= bps <= g.bps.max():
            raise ValueError(f"{bps} bps is outside the stored cost levels")
        out[strategy] = float(np.interp(bps, g.bps.to_numpy(), g[column].to_numpy()))
    return pd.Series(out, name=column).sort_values(ascending=False)
