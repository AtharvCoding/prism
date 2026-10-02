"""Drawdown-episode segmentation (peak -> trough -> recovery). Spec §14.3.

"Calendar blocks mislead": a calendar window can contain a violent crash and
the bull market that followed it. Episodes are identified on ONE reference
price series (SPY, in the pre-registration) before any strategy result is
consulted, and every strategy is then scored over the same dated phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Episode", "find_drawdown_episodes", "episode_performance"]


@dataclass(frozen=True)
class Episode:
    peak: pd.Timestamp
    trough: pd.Timestamp
    #: First session at or above the prior peak; ``None`` if not recovered by the window end.
    recovery: pd.Timestamp | None
    depth: float
    window_end: pd.Timestamp

    @property
    def end(self) -> pd.Timestamp:
        return self.recovery if self.recovery is not None else self.window_end

    def as_dict(self) -> dict[str, object]:
        return {
            "peak": self.peak.date(), "trough": self.trough.date(),
            "recovery": None if self.recovery is None else self.recovery.date(),
            "depth": self.depth, "recovered": self.recovery is not None,
        }


def find_drawdown_episodes(prices: pd.Series, *, threshold: float = 0.10) -> list[Episode]:
    """Peak-to-trough declines of at least ``threshold``, with their recovery dates.

    The running peak starts at the first observation. A decline phase opens
    when the price first falls below its running peak and closes when it
    returns to that peak (the recovery) or the series ends; it is kept only if
    its depth reaches ``threshold``.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    px = prices.dropna().astype("float64")
    if len(px) < 2:
        return []
    idx, v = px.index, px.to_numpy()
    episodes: list[Episode] = []
    peak_i = 0
    trough_i: int | None = None
    for i in range(1, len(v)):
        if v[i] >= v[peak_i]:
            if trough_i is not None:
                depth = v[trough_i] / v[peak_i] - 1.0
                if depth <= -threshold:
                    episodes.append(Episode(idx[peak_i], idx[trough_i], idx[i], float(depth), idx[-1]))
            peak_i, trough_i = i, None
        elif trough_i is None or v[i] < v[trough_i]:
            trough_i = i
    if trough_i is not None:
        depth = v[trough_i] / v[peak_i] - 1.0
        if depth <= -threshold:
            episodes.append(Episode(idx[peak_i], idx[trough_i], None, float(depth), idx[-1]))
    return episodes


def _compound(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0) if len(r) else float("nan")


def _max_drawdown(r: pd.Series) -> float:
    if not len(r):
        return float("nan")
    curve = np.concatenate([[1.0], (1.0 + r).cumprod().to_numpy()])  # start from NAV 1, not the first return
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


def episode_performance(net_returns: pd.Series, episode: Episode) -> dict[str, float]:
    """A strategy's net return over the decline, over the recovery, and its worst drawdown inside.

    The decline is ``(peak, trough]`` and the recovery ``(trough, end]``; the
    first of these starts the session AFTER the reference peak because the
    return dated on the peak session was earned getting to the peak.
    """
    r = net_returns.dropna()
    decline = r.loc[(r.index > episode.peak) & (r.index <= episode.trough)]
    recovery = r.loc[(r.index > episode.trough) & (r.index <= episode.end)]
    whole = r.loc[(r.index > episode.peak) & (r.index <= episode.end)]
    return {
        "decline_return": _compound(decline),
        "recovery_return": _compound(recovery),
        "max_drawdown_in_episode": _max_drawdown(whole),
        "n_decline_sessions": float(len(decline)),
    }
