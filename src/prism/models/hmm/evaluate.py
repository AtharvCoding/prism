"""State characterisation against NBER/drawdown labels and the threshold baseline. Spec §8.7.

Four kinds of evidence, each a function below:

* **State characterisation** (:func:`characterise_states`) — mean return,
  volatility, drawdown behaviour, duration and the empirical transition
  matrix of each canonical state, from a **hard** (argmax) assignment of the
  walk-forward posterior series. Spec §8.1's complaint about the reference
  HMM was that its states differed *only* in volatility level — a "volatility
  ladder", not economically distinct regimes. This table is what lets that
  question be answered from data rather than assumed.
* **Posterior quality** (:func:`posterior_quality`) — mean entropy (a
  saturation check: entropy near zero everywhere means the conditional-
  independence violation that produced the volatility ladder has returned,
  §15.1's "what would indicate something is wrong"), and the day-to-day
  stability of the hard assignment.
* **Economic validity** (:func:`drawdown_bear_episodes`,
  :func:`load_nber_recessions`, :func:`detect_episodes`) — overlap with NBER
  recessions and SPY drawdown episodes, reported as detection lag (signed:
  negative means the model's crisis state rose before the reference episode
  is dated to have begun) and a false-alarm rate.
* **Beats-baseline, the detection half** (:func:`compare_detection`) — spec
  §8.7's "if the HMM cannot beat a two-state volatility threshold on both
  detection lag and downstream probe performance, that is a finding". This
  module covers detection lag and false-alarm rate. The other half —
  downstream *probe* performance — is the §13.1 Tier 1 ablation's V3-vs-C2
  gate, which needs the probe harness (step 4a) and does not belong here;
  ``configs/experiments/tier1_probes.yaml``'s ``gates.hmm_adds_value``
  already names it. Conflating the two here would just be a second, worse
  copy of what step 4a is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from prism.backtest.metrics import drawdown_series
from prism.utils.logging import get_logger

__all__ = [
    "StateCharacterisation",
    "characterise_states",
    "PosteriorQuality",
    "posterior_quality",
    "drawdown_bear_episodes",
    "load_nber_recessions",
    "EpisodeDetection",
    "detect_episodes",
    "false_alarm_rate",
    "DetectionComparison",
    "compare_detection",
]

_log = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# state characterisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateCharacterisation:
    """Per-state economic profile plus the empirical transition matrix."""

    table: pd.DataFrame
    transition_matrix: pd.DataFrame
    hard_assignment: pd.Series

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table.to_dict(orient="index"),
            "transition_matrix": self.transition_matrix.to_dict(orient="index"),
        }


def characterise_states(
    posteriors: pd.DataFrame, returns: pd.Series, names: list[str] | None = None
) -> StateCharacterisation:
    """Characterise each canonical state from a hard (argmax) assignment.

    Parameters
    ----------
    posteriors
        Walk-forward posterior series, columns ``state_0..state_{k-1}``
        (canonically ordered — ascending return std, spec §8.5).
    returns
        Daily benchmark (SPY) log returns, aligned to ``posteriors``'s index.
    """
    common = posteriors.index.intersection(returns.index)
    if len(common) < len(posteriors):
        _log.warning(
            "characterise_states: %d of %d posterior dates have no matching return",
            len(posteriors) - len(common),
            len(posteriors),
        )
    post = posteriors.loc[common]
    rets = returns.loc[common]
    k = post.shape[1]
    hard = pd.Series(post.to_numpy().argmax(axis=1), index=common, name="state")

    dd = drawdown_series(rets)

    rows = []
    for state in range(k):
        mask = hard == state
        n_days = int(mask.sum())
        state_returns = rets[mask]
        state_dd = dd[mask]
        rows.append(
            {
                "state": state,
                "n_days": n_days,
                "unconditional_share": n_days / len(hard) if len(hard) else np.nan,
                "mean_return_annualised": (
                    float(state_returns.mean() * TRADING_DAYS_PER_YEAR) if n_days else np.nan
                ),
                "volatility_annualised": (
                    float(state_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
                    if n_days > 1
                    else np.nan
                ),
                "mean_drawdown_while_in_state": float(state_dd.mean()) if n_days else np.nan,
                "worst_drawdown_while_in_state": float(state_dd.min()) if n_days else np.nan,
                "mean_duration_days": _mean_run_length(hard, state),
            }
        )
    table = pd.DataFrame(rows).set_index("state")
    if names is not None:
        if len(names) != k:
            raise ValueError(f"names has {len(names)} entries, expected {k}")
        table.insert(0, "name", names)

    transition_matrix = _empirical_transition_matrix(hard, k)
    return StateCharacterisation(table=table, transition_matrix=transition_matrix, hard_assignment=hard)


def _mean_run_length(hard: pd.Series, state: int) -> float:
    """Average number of consecutive days spent in ``state`` per visit."""
    values = (hard == state).to_numpy()
    if not values.any():
        return float("nan")
    padded = np.concatenate(([False], values, [False]))
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return float((ends - starts).mean())


def _empirical_transition_matrix(hard: pd.Series, k: int) -> pd.DataFrame:
    """Row-normalised empirical transition counts from the hard-assigned sequence.

    This is an **aggregate, observed** transition matrix over the whole
    walk-forward posterior series — it will not exactly match any single
    fold's fitted ``transmat_`` (each fold refits on an expanding window), and
    is not meant to; it summarises what the deployed, month-by-month sequence
    of models actually produced.
    """
    values = hard.to_numpy()
    counts = np.zeros((k, k))
    for a, b in zip(values[:-1], values[1:]):
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    labels = [f"state_{i}" for i in range(k)]
    return pd.DataFrame(probs, index=labels, columns=labels)


# --------------------------------------------------------------------------- #
# posterior quality
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PosteriorQuality:
    """Saturation and stability diagnostics. Spec §8.7, §15.1."""

    mean_entropy: float
    entropy_series: pd.Series
    saturated: bool
    median_max_posterior: float
    flip_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_entropy": self.mean_entropy,
            "saturated": self.saturated,
            "median_max_posterior": self.median_max_posterior,
            "flip_rate": self.flip_rate,
        }


def posterior_quality(posteriors: pd.DataFrame, *, entropy_saturation_warn: float) -> PosteriorQuality:
    """Entropy, max-posterior distribution, and hard-assignment stability.

    Spec §15.1: "Perfectly separated HMM posteriors (entropy near zero) -> the
    conditional-independence violation has returned" — the exact failure
    mode of defect B3 (feeding a rolling statistic as an observation).
    ``saturated`` flags that condition directly rather than leaving it to be
    eyeballed off a plot.
    """
    p = posteriors.to_numpy()
    p_safe = np.clip(p, 1e-300, 1.0)  # avoid log(0); a true 0 contributes 0 to entropy
    entropy = -np.sum(np.where(p > 0, p_safe * np.log(p_safe), 0.0), axis=1)
    entropy_series = pd.Series(entropy, index=posteriors.index, name="entropy")

    hard = p.argmax(axis=1)
    flips = (hard[1:] != hard[:-1]).mean() if len(hard) > 1 else float("nan")

    return PosteriorQuality(
        mean_entropy=float(entropy_series.mean()),
        entropy_series=entropy_series,
        saturated=bool(entropy_series.mean() < entropy_saturation_warn),
        median_max_posterior=float(np.median(p.max(axis=1))),
        flip_rate=float(flips),
    )


# --------------------------------------------------------------------------- #
# economic validity: episodes
# --------------------------------------------------------------------------- #
def drawdown_bear_episodes(
    price: pd.Series, *, threshold: float = 0.20
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Episodes where ``price`` is ``>= threshold`` below its running peak.

    Episode start is the date of the running peak immediately preceding the
    drawdown crossing ``threshold`` (not the date the threshold is crossed
    itself) — matching NBER's own convention of dating a recession from the
    peak, so the two kinds of episode are comparable.
    """
    values = price.to_numpy(dtype="float64")
    idx = pd.DatetimeIndex(price.index)
    n = len(values)
    if n == 0:
        return []

    episodes: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    peak_val, peak_date = values[0], idx[0]
    in_episode = False
    ep_start: pd.Timestamp | None = None

    for t in range(n):
        if values[t] > peak_val:
            peak_val, peak_date = values[t], idx[t]
        dd = values[t] / peak_val - 1.0
        if dd <= -threshold and not in_episode:
            in_episode = True
            ep_start = peak_date
        elif dd > -threshold and in_episode:
            in_episode = False
            episodes.append((ep_start, idx[t]))  # type: ignore[arg-type]
    if in_episode:
        episodes.append((ep_start, idx[-1]))  # type: ignore[arg-type]
    return episodes


def load_nber_recessions(cfg: Any) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """NBER recession (peak, trough) pairs from ``configs/reference/nber_recessions.yaml``.

    Published historical facts, not a project split boundary — kept in
    ``configs/`` anyway so the "no hard-coded date literals outside configs/"
    grep test (``tests/test_splits.py``) stays a clean, exception-free rule.
    """
    path = Path(cfg.root) / "configs" / "reference" / "nber_recessions.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for name, bounds in data["recessions"].items():
        out.append((pd.Timestamp(bounds["peak"]), pd.Timestamp(bounds["trough"])))
    return sorted(out)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EpisodeDetection:
    """One episode's detection outcome against a crisis-probability series."""

    episode_start: pd.Timestamp
    episode_end: pd.Timestamp
    detected: bool
    detection_date: pd.Timestamp | None
    lag_sessions: int | None  # negative = detected BEFORE episode_start

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_start": str(self.episode_start.date()),
            "episode_end": str(self.episode_end.date()),
            "detected": self.detected,
            "detection_date": str(self.detection_date.date()) if self.detection_date else None,
            "lag_sessions": self.lag_sessions,
        }


def detect_episodes(
    crisis_signal: pd.Series,
    episodes: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    threshold: float = 0.5,
    search_margin_sessions: int = 60,
) -> list[EpisodeDetection]:
    """First crossing of ``threshold`` for each episode, with signed lag.

    Searches ``[episode_start - search_margin_sessions, episode_end]`` on
    ``crisis_signal``'s own index — the margin lets a model that leads the
    reference date (crisis probability rising *before* the dated peak, which
    is plausible and would be a point in its favour) register a **negative**
    lag rather than being scored as having "missed" the episode entirely.
    """
    index = pd.DatetimeIndex(crisis_signal.index)
    out: list[EpisodeDetection] = []
    for start, end in episodes:
        start_pos = int(index.searchsorted(start))
        margin_pos = max(0, start_pos - search_margin_sessions)
        window = crisis_signal.iloc[margin_pos:].loc[:end]
        hits = window[window > threshold]
        if hits.empty:
            out.append(
                EpisodeDetection(
                    episode_start=start, episode_end=end, detected=False,
                    detection_date=None, lag_sessions=None,
                )
            )
            continue
        detection_date = pd.Timestamp(hits.index[0])
        detection_pos = int(index.searchsorted(detection_date))
        lag = detection_pos - start_pos
        out.append(
            EpisodeDetection(
                episode_start=start, episode_end=end, detected=True,
                detection_date=detection_date, lag_sessions=int(lag),
            )
        )
    return out


def false_alarm_rate(
    crisis_signal: pd.Series,
    episodes: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    threshold: float = 0.5,
) -> float:
    """Fraction of non-episode days where the crisis signal exceeds ``threshold``."""
    index = pd.DatetimeIndex(crisis_signal.index)
    in_episode = np.zeros(len(index), dtype=bool)
    for start, end in episodes:
        in_episode |= (index >= start) & (index <= end)
    outside = crisis_signal.to_numpy()[~in_episode]
    if len(outside) == 0:
        return float("nan")
    return float((outside > threshold).mean())


# --------------------------------------------------------------------------- #
# beats-baseline comparison (detection half only — see module docstring)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectionComparison:
    """HMM vs. threshold-baseline detection performance, episode by episode."""

    hmm: list[EpisodeDetection]
    baseline: list[EpisodeDetection]
    hmm_false_alarm_rate: float
    baseline_false_alarm_rate: float

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for h, b in zip(self.hmm, self.baseline):
            assert h.episode_start == b.episode_start
            rows.append(
                {
                    "episode_start": h.episode_start,
                    "episode_end": h.episode_end,
                    "hmm_detected": h.detected,
                    "hmm_lag_sessions": h.lag_sessions,
                    "baseline_detected": b.detected,
                    "baseline_lag_sessions": b.lag_sessions,
                    "hmm_faster": (
                        h.lag_sessions < b.lag_sessions
                        if h.detected and b.detected
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows)

    def hmm_wins_on_detection_lag(self) -> bool:
        """Spec §8.7's beats-baseline test, the detection-lag half.

        Both detectors must have detected every episode, and the HMM's mean
        lag across those episodes must be strictly lower. A model that
        detects fewer episodes than the baseline has already lost regardless
        of its lag on the ones it did catch.
        """
        if any(not h.detected for h in self.hmm) or any(not b.detected for b in self.baseline):
            return False
        hmm_mean = float(np.mean([h.lag_sessions for h in self.hmm]))
        baseline_mean = float(np.mean([b.lag_sessions for b in self.baseline]))
        return hmm_mean < baseline_mean


def compare_detection(
    hmm_crisis_signal: pd.Series,
    baseline_crisis_signal: pd.Series,
    episodes: list[tuple[pd.Timestamp, pd.Timestamp]],
    *,
    threshold: float = 0.5,
    search_margin_sessions: int = 60,
) -> DetectionComparison:
    """Run :func:`detect_episodes` for both the HMM and the baseline, paired."""
    hmm_detections = detect_episodes(
        hmm_crisis_signal, episodes, threshold=threshold, search_margin_sessions=search_margin_sessions
    )
    baseline_detections = detect_episodes(
        baseline_crisis_signal, episodes, threshold=threshold, search_margin_sessions=search_margin_sessions
    )
    return DetectionComparison(
        hmm=hmm_detections,
        baseline=baseline_detections,
        hmm_false_alarm_rate=false_alarm_rate(hmm_crisis_signal, episodes, threshold=threshold),
        baseline_false_alarm_rate=false_alarm_rate(
            baseline_crisis_signal, episodes, threshold=threshold
        ),
    )
