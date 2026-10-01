"""State characterisation, posterior quality, and episode detection. Spec §8.7.

Covers the parts of ``prism.models.hmm.evaluate`` most likely to have a
sign error or an off-by-one: the empirical transition matrix, entropy, and
detection lag (which is explicitly signed — a model that leads the reference
date must register negative, not be clamped to zero or treated as a miss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import Config
from prism.models.hmm.evaluate import (
    characterise_states,
    compare_detection,
    detect_episodes,
    drawdown_bear_episodes,
    false_alarm_rate,
    load_nber_recessions,
    posterior_quality,
)


# --------------------------------------------------------------------------- #
# state characterisation
# --------------------------------------------------------------------------- #
def test_characterise_states_hand_computed():
    """A deterministic 2-state hard assignment with known returns."""
    idx = pd.bdate_range("2020-01-01", periods=8)
    # State 0 on days 0-2 and 6-7, state 1 on days 3-5 (one contiguous run of 3).
    posteriors = pd.DataFrame(
        {
            "state_0": [1, 1, 1, 0, 0, 0, 1, 1],
            "state_1": [0, 0, 0, 1, 1, 1, 0, 0],
        },
        index=idx,
        dtype="float64",
    )
    returns = pd.Series([0.01, 0.01, 0.01, -0.05, -0.05, -0.05, 0.01, 0.01], index=idx)

    result = characterise_states(posteriors, returns)
    table = result.table

    assert table.loc[0, "n_days"] == 5
    assert table.loc[1, "n_days"] == 3
    np.testing.assert_allclose(table.loc[0, "unconditional_share"], 5 / 8)
    # Mean duration: state 0 has two runs (length 3, then length 2) -> mean 2.5.
    np.testing.assert_allclose(table.loc[0, "mean_duration_days"], 2.5)
    # State 1 has one run of length 3.
    np.testing.assert_allclose(table.loc[1, "mean_duration_days"], 3.0)


def test_transition_matrix_hand_computed():
    """A short, fully-determined sequence: 0,0,1,1,0,1 -> counts by hand."""
    idx = pd.bdate_range("2020-01-01", periods=6)
    hard_seq = [0, 0, 1, 1, 0, 1]
    posteriors = pd.DataFrame(
        {
            "state_0": [1.0 if s == 0 else 0.0 for s in hard_seq],
            "state_1": [1.0 if s == 1 else 0.0 for s in hard_seq],
        },
        index=idx,
    )
    returns = pd.Series(np.zeros(6), index=idx)
    result = characterise_states(posteriors, returns)
    tm = result.transition_matrix

    # Transitions: 0->0, 0->1, 1->1, 1->0, 0->1 => from 0: [0->0:1, 0->1:2]/3;
    # from 1: [1->0:1, 1->1:1]/2.
    np.testing.assert_allclose(tm.loc["state_0", "state_0"], 1 / 3)
    np.testing.assert_allclose(tm.loc["state_0", "state_1"], 2 / 3)
    np.testing.assert_allclose(tm.loc["state_1", "state_0"], 1 / 2)
    np.testing.assert_allclose(tm.loc["state_1", "state_1"], 1 / 2)
    # Every row sums to 1.
    np.testing.assert_allclose(tm.sum(axis=1).to_numpy(), 1.0)


def test_characterise_states_names_must_match_k():
    idx = pd.bdate_range("2020-01-01", periods=4)
    posteriors = pd.DataFrame({"state_0": [1, 1, 0, 0], "state_1": [0, 0, 1, 1]}, index=idx, dtype="float64")
    returns = pd.Series(np.zeros(4), index=idx)
    with pytest.raises(ValueError, match="names has"):
        characterise_states(posteriors, returns, names=["only_one"])


# --------------------------------------------------------------------------- #
# posterior quality
# --------------------------------------------------------------------------- #
def test_entropy_is_zero_for_a_fully_saturated_posterior():
    idx = pd.bdate_range("2020-01-01", periods=5)
    posteriors = pd.DataFrame(
        {"state_0": [1.0] * 5, "state_1": [0.0] * 5}, index=idx
    )
    result = posterior_quality(posteriors, entropy_saturation_warn=0.05)
    np.testing.assert_allclose(result.mean_entropy, 0.0, atol=1e-12)
    assert result.saturated


def test_entropy_is_maximal_for_a_uniform_posterior():
    """A K=2 uniform posterior has entropy log(2)."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    posteriors = pd.DataFrame({"state_0": [0.5] * 5, "state_1": [0.5] * 5}, index=idx)
    result = posterior_quality(posteriors, entropy_saturation_warn=0.05)
    np.testing.assert_allclose(result.mean_entropy, np.log(2), rtol=1e-10)
    assert not result.saturated


def test_flip_rate_counts_hard_assignment_changes():
    idx = pd.bdate_range("2020-01-01", periods=6)
    # Hard sequence: 0,0,1,1,1,0 -> 2 flips out of 5 transitions.
    posteriors = pd.DataFrame(
        {
            "state_0": [0.9, 0.9, 0.1, 0.1, 0.1, 0.9],
            "state_1": [0.1, 0.1, 0.9, 0.9, 0.9, 0.1],
        },
        index=idx,
    )
    result = posterior_quality(posteriors, entropy_saturation_warn=0.05)
    np.testing.assert_allclose(result.flip_rate, 2 / 5)


# --------------------------------------------------------------------------- #
# drawdown episodes
# --------------------------------------------------------------------------- #
def test_drawdown_bear_episodes_hand_computed():
    """A price path with one clean 25% drawdown, peak dated correctly."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    # Peak at day 2 (value 100), trough at day 5 (value 75, a 25% drawdown),
    # recovers above -20% at day 7.
    prices = pd.Series([90, 95, 100, 90, 80, 75, 76, 81, 85, 90], index=idx, dtype="float64")
    episodes = drawdown_bear_episodes(prices, threshold=0.20)

    assert len(episodes) == 1
    start, end = episodes[0]
    assert start == idx[2], "episode should start at the PEAK date, not the threshold-crossing date"
    # dd at day6 (76/100-1=-0.24) still <= -0.20; day7 (81/100-1=-0.19) recovers.
    assert end == idx[7]


def test_drawdown_bear_episodes_ongoing_at_series_end():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.Series([100, 90, 70, 65, 60], index=idx, dtype="float64")  # never recovers
    episodes = drawdown_bear_episodes(prices, threshold=0.20)
    assert len(episodes) == 1
    assert episodes[0] == (idx[0], idx[-1])


def test_drawdown_bear_episodes_none_below_threshold():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.Series([100, 98, 99, 97, 100], index=idx, dtype="float64")
    assert drawdown_bear_episodes(prices, threshold=0.20) == []


def test_drawdown_bear_episodes_detects_two_separate_crashes():
    idx = pd.bdate_range("2020-01-01", periods=16)
    prices = pd.Series(
        [100, 100, 70, 100, 100, 100, 65, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        index=idx, dtype="float64",
    )
    episodes = drawdown_bear_episodes(prices, threshold=0.20)
    assert len(episodes) == 2


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def test_detect_episodes_positive_lag():
    """Crisis signal crosses threshold 3 sessions AFTER the episode starts."""
    idx = pd.bdate_range("2020-01-01", periods=20)
    signal = pd.Series(0.1, index=idx)
    signal.iloc[8:] = 0.9  # crosses threshold at position 8
    episode_start = idx[5]
    episode_end = idx[15]

    detections = detect_episodes(signal, [(episode_start, episode_end)], threshold=0.5)
    assert len(detections) == 1
    d = detections[0]
    assert d.detected
    assert d.detection_date == idx[8]
    assert d.lag_sessions == 3


def test_detect_episodes_negative_lag_when_signal_leads():
    """The model's crisis probability rises BEFORE the reference episode start
    — this must register as a NEGATIVE lag, not be clamped to zero or treated
    as a miss. A model that leads the reference is, if anything, better."""
    idx = pd.bdate_range("2020-01-01", periods=20)
    signal = pd.Series(0.1, index=idx)
    signal.iloc[4:] = 0.9  # crosses threshold at position 4
    episode_start = idx[8]  # reference episode starts LATER, at position 8
    episode_end = idx[15]

    detections = detect_episodes(
        signal, [(episode_start, episode_end)], threshold=0.5, search_margin_sessions=10
    )
    d = detections[0]
    assert d.detected
    assert d.detection_date == idx[4]
    assert d.lag_sessions == -4


def test_detect_episodes_reports_a_miss():
    idx = pd.bdate_range("2020-01-01", periods=20)
    signal = pd.Series(0.1, index=idx)  # never crosses threshold
    detections = detect_episodes(signal, [(idx[5], idx[15])], threshold=0.5)
    d = detections[0]
    assert not d.detected
    assert d.detection_date is None
    assert d.lag_sessions is None


def test_false_alarm_rate_excludes_episode_days():
    idx = pd.bdate_range("2020-01-01", periods=10)
    # High signal only inside the episode [idx[4], idx[6]] -> zero false alarms.
    signal = pd.Series(0.1, index=idx)
    signal.iloc[4:7] = 0.9
    rate = false_alarm_rate(signal, [(idx[4], idx[6])], threshold=0.5)
    assert rate == 0.0

    # Now add one false alarm outside the episode.
    signal.iloc[0] = 0.9
    rate2 = false_alarm_rate(signal, [(idx[4], idx[6])], threshold=0.5)
    assert rate2 == pytest.approx(1 / 7)  # 7 days outside the episode, 1 is a false alarm


def test_compare_detection_prefers_lower_lag():
    idx = pd.bdate_range("2020-01-01", periods=20)
    hmm_signal = pd.Series(0.1, index=idx)
    hmm_signal.iloc[6:] = 0.9  # detects at lag 1
    baseline_signal = pd.Series(0.1, index=idx)
    baseline_signal.iloc[10:] = 0.9  # detects at lag 5
    episode_start = idx[5]

    comparison = compare_detection(
        hmm_signal, baseline_signal, [(episode_start, idx[15])], threshold=0.5
    )
    assert comparison.hmm_wins_on_detection_lag()


def test_compare_detection_loses_if_a_single_episode_is_missed():
    """A model that detects every episode except one must not win overall,
    even if its average lag on the ones it did catch is excellent."""
    idx = pd.bdate_range("2020-01-01", periods=20)
    hmm_signal = pd.Series(0.1, index=idx)  # never crosses: misses everything
    baseline_signal = pd.Series(0.1, index=idx)
    baseline_signal.iloc[6:] = 0.9

    comparison = compare_detection(
        hmm_signal, baseline_signal, [(idx[5], idx[15])], threshold=0.5
    )
    assert not comparison.hmm_wins_on_detection_lag()


# --------------------------------------------------------------------------- #
# NBER reference data
# --------------------------------------------------------------------------- #
def test_nber_recessions_are_ordered_and_within_project_range(cfg: Config):
    recessions = load_nber_recessions(cfg)
    assert len(recessions) == 3, "expected dot-com, GFC and COVID in the 1999-2023 span"
    for peak, trough in recessions:
        assert peak < trough
    starts = [p for p, _ in recessions]
    assert starts == sorted(starts)

    project_start = cfg.data.start("A")
    project_end = cfg.data.split("test")[1]
    for peak, _ in recessions:
        assert project_start <= peak <= project_end


def test_nber_recessions_match_published_dates(cfg: Config):
    """Pinned against the well-known, published NBER dates — not just
    internally consistent, actually correct."""
    recessions = load_nber_recessions(cfg)
    expected = {
        (pd.Timestamp("2001-03-01"), pd.Timestamp("2001-11-01")),
        (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01")),
        (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")),
    }
    assert set(recessions) == expected
