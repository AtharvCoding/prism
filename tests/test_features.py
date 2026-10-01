"""Numerical correctness of the §5 feature definitions.

Causality is proved in ``test_causality.py``; this file checks the features
are the quantities the spec says they are. A causal feature computing the
wrong statistic passes every causality test.

Each expected value is derived from first principles in the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import Config
from prism.features.asset import (
    TRADING_DAYS_PER_YEAR,
    MOMENTUM_SKIP_DAYS,
    build_asset_features,
    dist_from_high,
    downside_vol,
    log_returns,
    momentum_skip,
    realised_vol,
    rolling_beta,
)
from prism.features.cross import (
    average_pairwise_correlation,
    breadth_above_ma,
    build_cross_features,
    first_eigenvalue_share,
    rolling_shrunk_covariance,
)
from prism.features.macro import build_macro_features
from prism.features.scaling import CorrelationPruner, FeatureScaler, Winsoriser


@pytest.fixture
def idx() -> pd.DatetimeIndex:
    from prism.utils.calendar import trading_days

    return trading_days("2015-01-02", "2019-12-31")


# --------------------------------------------------------------------------- #
# per-asset — §5.1
# --------------------------------------------------------------------------- #
def test_log_returns_are_log_not_simple(idx):
    prices = pd.DataFrame({"A": 1.05 ** np.arange(len(idx))}, index=idx)
    r = log_returns(prices)
    np.testing.assert_allclose(r["A"].iloc[1:], np.log(1.05), rtol=1e-12)
    assert np.isnan(r["A"].iloc[0]), "the first return is undefined, not zero"


def test_realised_vol_is_annualised_sample_std(idx):
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    out = realised_vol(r, 20)
    window = r.iloc[30 - 20 : 30]
    expected = window.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    np.testing.assert_allclose(out.iloc[29], expected, rtol=1e-12)
    assert out.iloc[:19].isna().all(), "vol must be NaN before the window fills"


def test_downside_vol_is_a_second_moment_about_zero(idx):
    """sqrt(mean(min(r,0)^2)) over ALL periods, annualised.

    Not the std of the losing subset: that is a different quantity, and it is
    undefined when a window contains a single loss.
    """
    r = pd.Series([0.02, -0.01, 0.03, -0.02], index=idx[:4])
    out = downside_vol(r, 4)
    expected = np.sqrt((0.0 + 0.01**2 + 0.0 + 0.02**2) / 4) * np.sqrt(TRADING_DAYS_PER_YEAR)
    np.testing.assert_allclose(out.iloc[3], expected, rtol=1e-12)


def test_downside_vol_is_zero_when_nothing_falls(idx):
    r = pd.Series([0.01] * 30, index=idx[:30])
    np.testing.assert_allclose(downside_vol(r, 20).iloc[25], 0.0, atol=1e-15)


def test_momentum_skips_the_most_recent_week(idx):
    """§5.1: momentum is ex-most-recent-week, to avoid reversal contamination.

    ``mom_20`` at ``t`` is ``log(P[t-5] / P[t-25])`` — so it is completely
    unaffected by the last five sessions.
    """
    prices = pd.Series(1.01 ** np.arange(len(idx)), index=idx, name="A")
    out = momentum_skip(prices, 20)
    expected = 20 * np.log(1.01)
    np.testing.assert_allclose(out.iloc[40], expected, rtol=1e-12)

    # Perturbing only the last week must not change today's momentum.
    tampered = prices.copy()
    tampered.iloc[36:41] *= 3.0
    np.testing.assert_allclose(
        momentum_skip(tampered, 20).iloc[40], expected, rtol=1e-12
    )
    assert MOMENTUM_SKIP_DAYS == 5


def test_momentum_without_the_skip_would_be_contaminated(idx):
    """The converse: a no-skip momentum *is* moved by the last week.

    Without this, the test above would pass for a momentum that read nothing
    at all.
    """
    prices = pd.Series(1.01 ** np.arange(len(idx)), index=idx)
    tampered = prices.copy()
    tampered.iloc[36:41] *= 3.0
    no_skip = np.log(tampered.iloc[40] / tampered.iloc[20])
    clean = np.log(prices.iloc[40] / prices.iloc[20])
    assert not np.isclose(no_skip, clean)


def test_rolling_beta_is_one_for_the_benchmark_itself(idx):
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    rets = pd.DataFrame({"SPY": market, "LEVERED": 2.0 * market}, index=idx)
    beta = rolling_beta(rets, market, 60)
    np.testing.assert_allclose(beta["SPY"].iloc[100], 1.0, rtol=1e-10)
    np.testing.assert_allclose(beta["LEVERED"].iloc[100], 2.0, rtol=1e-10)


def test_rolling_beta_replaces_the_degenerate_spycorr(idx, cfg: Config, features_b):
    """Defect A7: ``SPY_spycorr`` was identically 1.0 and entered V1 as a constant.

    Beta is 1.0 for the benchmark too, but the benchmark is not allocatable,
    and every column must still have non-zero variance after warm-up — which
    the QA ``constant_feature`` hard check enforces.
    """
    assert "SPY_beta_60_spy" in features_b.columns
    assert not any("spycorr" in c for c in features_b.columns)
    stds = features_b.frame.std(numeric_only=True)
    constant = sorted(stds.index[stds.fillna(0.0) == 0.0].astype(str))
    assert not constant, f"constant features present: {constant}"


def test_dist_from_high_is_non_positive_and_zero_at_a_new_high(idx):
    prices = pd.Series(1.01 ** np.arange(len(idx)), index=idx)
    out = dist_from_high(prices, 252)
    # A monotone path is always at its own high.
    np.testing.assert_allclose(out.iloc[300], 0.0, atol=1e-15)

    # After a 20% fall from the peak, it reads -20%.
    fallen = prices.copy()
    fallen.iloc[300:] = prices.iloc[299] * 0.8
    np.testing.assert_allclose(dist_from_high(fallen, 252).iloc[305], -0.2, rtol=1e-10)
    assert (dist_from_high(fallen, 252).dropna() <= 1e-15).all()


def test_dvol_is_the_change_in_log_volatility(cfg: Config, features_b):
    """§5.1: ``dvol`` is the stationary form — a change, not a level."""
    col = [c for c in features_b.columns if c.endswith("_dvol")][0]
    ticker = col[: -len("_dvol")]
    vol = features_b.frame[f"{ticker}_vol_20"]

    # Compare from row 1 onward. `features_b.frame` is trimmed to the warm
    # period, so its first `dvol` was computed against the session *before*
    # the trim — recomputing `.diff()` on the trimmed column has no
    # predecessor there and yields NaN. That difference is the trim, not a
    # disagreement about the definition.
    expected = np.log(vol).diff().to_numpy()[1:]
    np.testing.assert_allclose(
        features_b.frame[col].to_numpy()[1:], expected, rtol=1e-9
    )
    assert np.isfinite(features_b.frame[col].iloc[0]), (
        "the first dvol should be a real value carried over from before the trim"
    )
    # A change series must be far less autocorrelated than the level it came
    # from — that is the entire point, and the reason defect B3 mattered.
    assert abs(features_b.frame[col].autocorr(1)) < 0.5 < abs(vol.autocorr(1))


def test_asset_features_require_a_valid_benchmark(cfg: Config, idx):
    prices = pd.DataFrame({"A": np.ones(len(idx))}, index=idx)
    with pytest.raises(KeyError, match="benchmark"):
        build_asset_features(prices, None, benchmark="SPY")


# --------------------------------------------------------------------------- #
# cross-sectional — §5.2
# --------------------------------------------------------------------------- #
def test_average_pairwise_correlation_on_a_known_matrix(idx):
    """Two perfectly correlated series and one independent one.

    Off-diagonal correlations are (1, rho_13, rho_23); with series 3
    independent, the mean tends to 1/3.
    """
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.01, len(idx))
    rets = pd.DataFrame(
        {"A": base, "B": base, "C": rng.normal(0, 0.01, len(idx))}, index=idx
    )
    out = average_pairwise_correlation(rets, 250)
    assert 0.2 < out.iloc[300] < 0.45

    # Perfectly correlated sleeve -> exactly 1.
    perfect = pd.DataFrame({"A": base, "B": base, "C": base}, index=idx)
    np.testing.assert_allclose(
        average_pairwise_correlation(perfect, 60).iloc[100], 1.0, rtol=1e-10
    )


def test_first_eigenvalue_share_is_one_when_everything_moves_together(idx):
    """PC1 of a correlation matrix of identical series explains everything."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, len(idx))
    perfect = pd.DataFrame({c: base for c in "ABCD"}, index=idx)
    np.testing.assert_allclose(
        first_eigenvalue_share(perfect, 60).iloc[100], 1.0, rtol=1e-8
    )

    # Independent series: PC1 share tends to 1/n.
    independent = pd.DataFrame(
        {c: rng.normal(0, 0.01, len(idx)) for c in "ABCD"}, index=idx
    )
    share = first_eigenvalue_share(independent, 250).iloc[300]
    assert 0.25 <= share < 0.45


def test_first_eigenvalue_share_is_scale_free(idx):
    """It uses the CORRELATION matrix, so multiplying a series by 10 changes nothing.

    This matters: §8.1's whole complaint is about features that are proxies
    for the level of volatility. A covariance-based version would be one.
    """
    rng = np.random.default_rng(4)
    rets = pd.DataFrame({c: rng.normal(0, 0.01, len(idx)) for c in "ABC"}, index=idx)
    scaled = rets.copy()
    scaled["A"] *= 10.0
    np.testing.assert_allclose(
        first_eigenvalue_share(rets, 60).iloc[100],
        first_eigenvalue_share(scaled, 60).iloc[100],
        rtol=1e-10,
    )


def test_breadth_is_the_fraction_above_the_moving_average(idx):
    rising = 1.01 ** np.arange(len(idx))
    falling = 0.99 ** np.arange(len(idx))
    prices = pd.DataFrame(
        {"UP1": rising, "UP2": rising, "DOWN": falling}, index=idx
    )
    out = breadth_above_ma(prices, 50)
    np.testing.assert_allclose(out.iloc[100], 2 / 3, rtol=1e-12)
    assert (out.dropna() >= 0).all() and (out.dropna() <= 1).all()


def test_cross_features_reject_a_one_asset_sleeve(idx):
    prices = pd.DataFrame({"A": np.ones(len(idx))}, index=idx)
    with pytest.raises(ValueError, match="at least two"):
        build_cross_features(prices, sleeve=["A"])


def test_cross_features_reject_a_missing_sleeve_member(idx):
    prices = pd.DataFrame({"A": np.ones(len(idx)), "B": np.ones(len(idx))}, index=idx)
    with pytest.raises(KeyError, match="sleeve tickers"):
        build_cross_features(prices, sleeve=["A", "MISSING"])


# --------------------------------------------------------------------------- #
# shrunk covariance — §5.2, consumed by the allocator
# --------------------------------------------------------------------------- #
def test_shrunk_covariance_is_symmetric_and_positive_definite(cfg: Config, raw_b):
    """Ledoit-Wolf shrinkage exists so the allocator gets an invertible matrix.

    A 60-day sample covariance of 13 assets is nearly singular — 60
    observations for 91 free parameters — and mean-variance optimisation on a
    singular matrix produces arbitrary weights.
    """
    from prism.features.build import split_raw_frame

    close = split_raw_frame(raw_b)["Close"]
    assets = [t for t in cfg.data.allocatable["B"] if t in close.columns]
    dates = close.index[-200::40]

    mats = rolling_shrunk_covariance(close, assets=assets, window=60, dates=dates)
    assert mats, "no covariance matrices were produced"

    for date, mat in mats.items():
        assert list(mat.index) == assets and list(mat.columns) == assets
        values = mat.to_numpy()
        np.testing.assert_allclose(values, values.T, rtol=1e-12)
        eigvals = np.linalg.eigvalsh(values)
        assert eigvals.min() > 0, f"{date.date()}: not positive definite"
        # And invertible in practice, which the sample covariance would not be.
        np.linalg.inv(values)


def test_shrunk_covariance_beats_the_sample_covariance_on_conditioning(cfg: Config, raw_b):
    """The converse: the raw sample covariance is far worse conditioned.

    Otherwise the shrinkage is doing nothing and could be dropped.
    """
    from prism.features.build import split_raw_frame

    close = split_raw_frame(raw_b)["Close"]
    assets = [t for t in cfg.data.allocatable["B"] if t in close.columns]
    date = close.index[-50]

    mats = rolling_shrunk_covariance(close, assets=assets, window=60, dates=[date])
    shrunk = mats[date].to_numpy()

    rets = np.log(close[assets]).diff()
    end = int(close.index.get_indexer([date])[0])
    sample = np.cov(rets.iloc[end - 59 : end + 1].to_numpy(), rowvar=False)

    assert np.linalg.cond(shrunk) < np.linalg.cond(sample), (
        "shrinkage did not improve conditioning; it is not earning its place"
    )


def test_shrunk_covariance_window_ends_at_the_keyed_date(cfg: Config, raw_b):
    """The matrix keyed at ``D`` must be usable for a decision taken at ``D``.

    So its window ends at and includes ``D`` — and, critically, contains no
    data after ``D``.
    """
    from prism.features.build import split_raw_frame

    close = split_raw_frame(raw_b)["Close"]
    assets = [t for t in cfg.data.allocatable["B"] if t in close.columns][:4]
    date = close.index[-100]

    baseline = rolling_shrunk_covariance(close, assets=assets, window=60, dates=[date])
    # Corrupting data strictly after `date` must leave the matrix unchanged.
    tampered = close.copy()
    tampered.loc[tampered.index > date, assets] *= 5.0
    after = rolling_shrunk_covariance(tampered, assets=assets, window=60, dates=[date])
    np.testing.assert_allclose(baseline[date].to_numpy(), after[date].to_numpy(), rtol=1e-12)

    # Corrupting data inside the window must change it.
    inside = close.copy()
    lo = close.index[int(close.index.get_indexer([date])[0]) - 30]
    inside.loc[(inside.index >= lo) & (inside.index <= date), assets] *= 5.0
    changed = rolling_shrunk_covariance(inside, assets=assets, window=60, dates=[date])
    assert not np.allclose(baseline[date].to_numpy(), changed[date].to_numpy())


def test_shrunk_covariance_is_not_flattened_into_the_feature_frame(features_b):
    """§5.2: "stored separately for the allocator, not flattened into the state".

    A 13x13 covariance is 91 numbers; adding them would roughly halve the
    signal-to-dimension ratio of the whole state vector.
    """
    assert not any("cov_" in c for c in features_b.columns)


# --------------------------------------------------------------------------- #
# macro — §5.3
# --------------------------------------------------------------------------- #
def test_macro_requires_vix_and_the_curve_legs(idx):
    with pytest.raises(KeyError, match=r"\^VIX"):
        build_macro_features(pd.DataFrame({"^TNX": np.ones(len(idx))}, index=idx))
    with pytest.raises(KeyError, match=r"\^TNX and \^IRX"):
        build_macro_features(pd.DataFrame({"^VIX": np.ones(len(idx)) * 20}, index=idx))


def test_curve_slope_is_an_arithmetic_difference_of_yields(idx):
    """§5.3 / D-006: yields are never log-transformed."""
    macro = pd.DataFrame(
        {
            "^VIX": np.full(len(idx), 18.0),
            "^TNX": np.full(len(idx), 3.5),
            "^IRX": np.full(len(idx), 1.25),
            "^FVX": np.full(len(idx), 2.75),
        },
        index=idx,
    )
    out = build_macro_features(macro)
    np.testing.assert_allclose(out["curve_slope_10y_3m"].iloc[10], 3.5 - 1.25, rtol=1e-12)
    np.testing.assert_allclose(out["curve_slope_10y_5y"].iloc[10], 3.5 - 2.75, rtol=1e-12)
    np.testing.assert_allclose(out["curve_change"].iloc[10], 0.0, atol=1e-15)


def test_zero_and_negative_yields_do_not_produce_infinities(idx):
    """`^IRX` sat at zero for years; a log would have produced -inf."""
    macro = pd.DataFrame(
        {
            "^VIX": np.full(len(idx), 18.0),
            "^TNX": np.full(len(idx), 1.0),
            "^IRX": np.zeros(len(idx)),
            "^FVX": np.full(len(idx), -0.1),
        },
        index=idx,
    )
    out = build_macro_features(macro)
    finite = out[["curve_slope_10y_3m", "curve_slope_10y_5y", "curve_change"]].dropna()
    assert np.isfinite(finite.to_numpy()).all()
    np.testing.assert_allclose(finite["curve_slope_10y_5y"].iloc[0], 1.1, rtol=1e-12)


def test_credit_proxy_is_the_log_ratio_and_its_changes(idx):
    """§5.3: log(HYG/LQD), plus a 1d change (HMM) and a 20d change (features)."""
    macro = pd.DataFrame(
        {
            "^VIX": np.full(len(idx), 18.0),
            "^TNX": np.full(len(idx), 3.5),
            "^IRX": np.full(len(idx), 1.25),
            "HYG": 80.0 * 0.999 ** np.arange(len(idx)),
            "LQD": np.full(len(idx), 110.0),
        },
        index=idx,
    )
    out = build_macro_features(macro)
    np.testing.assert_allclose(
        out["credit_proxy"].iloc[50], np.log(80.0 * 0.999**50 / 110.0), rtol=1e-12
    )
    np.testing.assert_allclose(out["credit_proxy_change"].iloc[50], np.log(0.999), rtol=1e-10)
    np.testing.assert_allclose(
        out["credit_proxy_change_20"].iloc[50], 20 * np.log(0.999), rtol=1e-10
    )


def test_optional_macro_features_are_omitted_not_nan_filled(idx):
    """A B-only series absent from the panel means the column is absent."""
    macro = pd.DataFrame(
        {
            "^VIX": np.full(len(idx), 18.0),
            "^TNX": np.full(len(idx), 3.5),
            "^IRX": np.full(len(idx), 1.25),
        },
        index=idx,
    )
    out = build_macro_features(macro)
    for column in ("credit_proxy", "vix_term_structure", "oil_vol_20", "dollar_return_20"):
        assert column not in out.columns


def test_vix_change_is_a_log_change(idx):
    macro = pd.DataFrame(
        {
            "^VIX": 15.0 * 1.02 ** np.arange(len(idx)),
            "^TNX": np.full(len(idx), 3.5),
            "^IRX": np.full(len(idx), 1.25),
        },
        index=idx,
    )
    out = build_macro_features(macro)
    np.testing.assert_allclose(out["vix_change"].iloc[10], np.log(1.02), rtol=1e-12)


def test_oil_vol_tolerates_a_negative_print(idx):
    """D-014: WTI printed -$37.63 on 2020-04-20. ``oil_vol_20`` must not gap.

    A log return is undefined across a sign change and a mask-to-positive
    (the yield convention, D-006) turns the one bad day into a ~20-session
    hole in the rolling std. ``oil_vol_20`` uses a simple return instead,
    which stays finite and reports a real, elevated volatility for the month
    after the event — not a gap.
    """
    rng = np.random.default_rng(9)
    n = len(idx)
    oil = 40.0 + rng.normal(0, 1.0, n)
    # A dramatic plunge through zero to a negative print, then a recovery —
    # the real shape of the 2020-04-20 event, compressed into a short window.
    crash_at = 200
    oil[crash_at] = -37.63
    oil[crash_at + 1 : crash_at + 10] = np.linspace(-10.0, 35.0, 9)
    macro = pd.DataFrame(
        {
            "^VIX": np.full(n, 18.0),
            "^TNX": np.full(n, 3.5),
            "^IRX": np.full(n, 1.25),
            "CL=F": oil,
        },
        index=idx,
    )
    out = build_macro_features(macro)

    # No NaN/inf anywhere past the warm-up window — including through and
    # after the crash.
    warm = out["oil_vol_20"].iloc[30:]
    assert np.isfinite(warm.to_numpy()).all(), (
        "oil_vol_20 produced a non-finite value across the negative print"
    )
    # And the volatility spike is real, not smoothed away: the 20 days
    # following the crash must show materially higher realised vol than the
    # calm period before it.
    before = out["oil_vol_20"].iloc[100:190].mean()
    after = out["oil_vol_20"].iloc[crash_at : crash_at + 20].mean()
    assert after > 3 * before, "the crash should dominate the rolling realised vol"


# --------------------------------------------------------------------------- #
# scaling and pruning — §5.5
# --------------------------------------------------------------------------- #
def test_winsoriser_clips_to_fit_window_quantiles(idx):
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"x": rng.normal(size=len(idx))}, index=idx)
    train = frame.iloc[:500]
    w = Winsoriser(0.01, 0.99).fit(train, scope="train")
    out = w.transform(frame)

    lo, hi = train["x"].quantile(0.01), train["x"].quantile(0.99)
    assert out["x"].min() >= lo - 1e-12
    assert out["x"].max() <= hi + 1e-12
    # A later extreme value is clipped to the TRAIN bound, not its own.
    assert frame["x"].max() > hi


def test_scaler_standardises_on_the_fit_window_only(idx):
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"x": rng.normal(5.0, 3.0, len(idx))}, index=idx)
    train = frame.iloc[:500]
    s = FeatureScaler("standard").fit(train, scope="train")
    out = s.transform(frame)

    np.testing.assert_allclose(out["x"].iloc[:500].mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(out["x"].iloc[:500].std(ddof=0), 1.0, rtol=1e-10)
    # The full sample is NOT standardised — that would mean a full-sample fit.
    assert abs(out["x"].mean()) < 0.5


def test_robust_scaler_uses_median_and_iqr(idx):
    frame = pd.DataFrame({"x": np.arange(len(idx), dtype="float64")}, index=idx)
    train = frame.iloc[:500]
    s = FeatureScaler("robust").fit(train, scope="train")
    out = s.transform(frame)
    np.testing.assert_allclose(out["x"].iloc[:500].median(), 0.0, atol=1e-10)


def test_constant_column_does_not_become_infinite(idx):
    frame = pd.DataFrame({"flat": np.ones(len(idx)), "x": np.arange(len(idx.values))}, index=idx)
    s = FeatureScaler().fit(frame.iloc[:500], scope="train")
    out = s.transform(frame)
    assert np.isfinite(out["flat"].to_numpy()).all()


def test_correlation_pruning_is_deterministic_and_within_family(idx):
    """§5.5: prune within a family only, and pick the survivor deterministically."""
    rng = np.random.default_rng(7)
    base = rng.normal(size=len(idx))
    frame = pd.DataFrame(
        {
            "XLK_vol_20": base,
            "XLK_vol_60": base * 1.0001 + 1e-9 * rng.normal(size=len(idx)),
            "XLK_mom_20": base,  # same values, DIFFERENT family
            "XLE_vol_20": rng.normal(size=len(idx)),
        },
        index=idx,
    )
    first = CorrelationPruner(0.95).fit(frame, scope="train")
    second = CorrelationPruner(0.95).fit(frame.copy(), scope="train")
    assert first.kept == second.kept, "pruning is not deterministic"

    # vol_20 precedes vol_60 in FEATURE_FAMILIES, so vol_20 survives.
    assert "XLK_vol_20" in first.kept
    assert "XLK_vol_60" not in first.kept
    # mom_20 is a different family and is untouched despite identical values.
    assert "XLK_mom_20" in first.kept
    # XLE's family is pruned independently of XLK's.
    assert "XLE_vol_20" in first.kept

    rows = first.report_rows()
    assert len(rows) == 1
    assert rows.iloc[0]["dropped"] == "XLK_vol_60"
    assert rows.iloc[0]["kept"] == "XLK_vol_20"
    assert rows.iloc[0]["abs_corr"] > 0.95


def test_pruning_keeps_everything_when_nothing_is_redundant(idx):
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        {
            "XLK_vol_20": rng.normal(size=len(idx)),
            "XLK_vol_60": rng.normal(size=len(idx)),
            "XLK_mom_20": rng.normal(size=len(idx)),
        },
        index=idx,
    )
    pruner = CorrelationPruner(0.95).fit(frame, scope="train")
    assert set(pruner.kept) == set(frame.columns)
    assert pruner.report_rows().empty


def test_pruning_on_real_features_documents_what_survived(cfg: Config, features_b):
    """§5.5: "document which survived". The report must be produced either way."""
    from prism.splits import build_split_plan

    plan = build_split_plan(cfg)
    train = features_b.frame.loc[
        plan["train"].effective_start : plan["train"].effective_end
    ]
    pruner = CorrelationPruner(0.95).fit(train, scope="train")
    rows = pruner.report_rows()
    assert set(rows.columns) == {"dropped", "kept", "family", "group", "abs_corr"}
    assert len(pruner.kept) <= features_b.frame.shape[1]
    # Pruning must not be able to empty a family entirely.
    assert len(pruner.kept) > 0.5 * features_b.frame.shape[1], (
        f"pruning removed {features_b.frame.shape[1] - len(pruner.kept)} of "
        f"{features_b.frame.shape[1]} columns; the threshold is too aggressive"
    )
