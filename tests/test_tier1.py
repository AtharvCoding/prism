"""Tier 1 machinery: probes, paired bootstrap, gates, DSR, episodes, allocator. Step 4a.

Each test pins a property the pre-registration relies on, with a hand-computable
answer where one exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from prism.analysis.bootstrap import bootstrap_paths, path_ci
from prism.analysis.episodes import episode_performance, find_drawdown_episodes
from prism.analysis.significance import deflated_sharpe, expected_max_sharpe, per_period_sharpe
from prism.backtest.benchmarks import capped_inverse_vol
from prism.probes import allocator as al
from prism.probes.tier1 import (
    GATES, PRIMARY_RISK_TARGETS, aggregate_comparison, common_evaluation_index, evaluate_gates,
    ridge_fold_predictions,
)


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def test_closed_form_ridge_matches_sklearn_for_every_alpha_and_target():
    rng = np.random.default_rng(0)
    X_fit, X_apply = rng.normal(size=(120, 15)), rng.normal(size=(30, 15))
    X_fit[:, 3] = 5.0  # a zero-spread column must be left unscaled, not blow up
    Y_fit = rng.normal(size=(120, 3)) + X_fit[:, :3]
    alphas = [0.01, 1.0, 100.0, 1e6]
    preds, ybar = ridge_fold_predictions(X_fit, Y_fit, X_apply, alphas)

    mean, std = X_fit.mean(0), X_fit.std(0)
    std = np.where(std > 0, std, 1.0)
    Zf, Za = (X_fit - mean) / std, (X_apply - mean) / std
    for a in alphas:
        expected = Ridge(alpha=a, fit_intercept=True).fit(Zf, Y_fit).predict(Za)
        np.testing.assert_allclose(preds[a], expected, atol=1e-8)
    np.testing.assert_allclose(ybar, Y_fit.mean(0))


def test_strong_regularisation_collapses_to_the_historical_mean():
    """At a huge alpha the probe forecasts the fit-window mean, i.e. R^2 = 0 by construction."""
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(80, 50)), rng.normal(size=(80, 1))
    preds, ybar = ridge_fold_predictions(X, Y, rng.normal(size=(10, 50)), [1e12])
    np.testing.assert_allclose(preds[1e12], np.tile(ybar, (10, 1)), atol=1e-6)


def test_common_index_is_the_intersection_and_drops_rows_missing_any_target():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.DataFrame({"x": 1.0}, index=idx)
    b = pd.DataFrame({"x": 1.0}, index=idx[3:])  # a shorter variant, like V1'
    targets = pd.DataFrame({"fwd_vol_5": 1.0, "fwd_ret_20": 1.0}, index=idx)
    targets.iloc[-2:, 1] = np.nan  # trailing rows have no future
    out = common_evaluation_index({"A": a, "B": b}, targets, ("fwd_vol_5", "fwd_ret_20"))
    assert out.equals(idx[3:-2])


# --------------------------------------------------------------------------- #
# bootstrap pairing
# --------------------------------------------------------------------------- #
def test_paths_are_fixed_by_n_block_and_seed_only():
    a, b = bootstrap_paths(200, 20, 50, seed=7), bootstrap_paths(200, 20, 50, seed=7)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, bootstrap_paths(200, 20, 50, seed=8))
    assert a.min() >= 0 and a.max() < 200


def test_paired_difference_is_resampled_as_a_pair():
    """Replicate means of the differential equal the difference of replicate means,
    which is true only if both variants were resampled on the same days."""
    rng = np.random.default_rng(2)
    ex2, ey2 = rng.gamma(2, size=300), rng.gamma(2, size=300)
    paths = bootstrap_paths(300, 20, 200, seed=3)
    d_rep = (ex2 - ey2)[paths].mean(axis=1)
    np.testing.assert_allclose(d_rep, ex2[paths].mean(axis=1) - ey2[paths].mean(axis=1), atol=1e-12)
    point, lo, hi = path_ci(ex2 - ey2, paths)
    assert lo < point < hi


# --------------------------------------------------------------------------- #
# gate aggregation
# --------------------------------------------------------------------------- #
def _table(x, y, verdicts, column="verdict"):
    return pd.DataFrame({
        "x": x, "y": y, "target": list(PRIMARY_RISK_TARGETS), column: verdicts, "primary": True,
    })


def test_a_comparison_needs_three_favourable_and_no_adverse():
    f, i, a = "favourable", "indeterminate", "adverse"
    assert aggregate_comparison(_table("X", "Y", [f, f, f, i]), "X", "Y")["passes"]
    assert aggregate_comparison(_table("X", "Y", [f, f, f, f]), "X", "Y")["passes"]
    assert not aggregate_comparison(_table("X", "Y", [f, f, i, i]), "X", "Y")["passes"]
    assert not aggregate_comparison(_table("X", "Y", [f, f, f, a]), "X", "Y")["passes"], "one adverse blocks the pass"


def test_a_gate_needs_both_of_its_comparisons():
    f, i = "favourable", "indeterminate"
    rows = []
    for x, y in GATES["lstm_adds_value"] + GATES["hmm_adds_value"] + GATES["research_question"]:
        verdicts = [f, f, f, f] if (x, y) != ("V2", "C1") else [f, i, i, i]
        rows.append(_table(x, y, verdicts))
    gates = evaluate_gates(pd.concat(rows, ignore_index=True))
    assert not gates["lstm_adds_value"]["passes"], "V2 beats V1' but not C1: the LSTM gate fails"
    assert gates["hmm_adds_value"]["passes"]


# --------------------------------------------------------------------------- #
# deflated Sharpe
# --------------------------------------------------------------------------- #
def test_expected_max_sharpe_grows_with_the_number_of_trials():
    assert expected_max_sharpe(27, 1e-4) > expected_max_sharpe(9, 1e-4) > 0
    assert expected_max_sharpe(27, 0.0) == 0.0


def test_deflated_sharpe_rewards_a_clear_edge_and_punishes_noise():
    rng = np.random.default_rng(4)
    strong = rng.normal(0.002, 0.005, 1500)  # per-period Sharpe ~0.4
    noise = rng.normal(0.0, 0.01, 1500)
    sr0 = expected_max_sharpe(27, 1e-4)
    assert deflated_sharpe(strong, sr0=sr0) > 0.99
    assert deflated_sharpe(noise, sr0=sr0) < 0.5
    assert per_period_sharpe(strong) > per_period_sharpe(noise)
    assert deflated_sharpe(strong, sr0=0.0) > deflated_sharpe(strong, sr0=0.2)


# --------------------------------------------------------------------------- #
# episodes
# --------------------------------------------------------------------------- #
def test_episodes_find_peak_trough_and_recovery():
    idx = pd.bdate_range("2020-01-01", periods=12)
    px = pd.Series([100, 105, 110, 99, 88, 95, 105, 111, 120, 108, 100, 96], index=idx, dtype=float)
    eps = find_drawdown_episodes(px, threshold=0.10)
    assert len(eps) == 2
    first, second = eps
    assert (first.peak, first.trough, first.recovery) == (idx[2], idx[4], idx[7])
    assert first.depth == pytest.approx(88 / 110 - 1)
    assert second.peak == idx[8] and second.trough == idx[11] and second.recovery is None
    assert second.end == idx[-1]


def test_a_shallow_dip_is_not_an_episode():
    idx = pd.bdate_range("2020-01-01", periods=5)
    assert find_drawdown_episodes(pd.Series([100, 95, 97, 100, 102], index=idx, dtype=float)) == []


def test_episode_performance_splits_decline_and_recovery():
    idx = pd.bdate_range("2020-01-01", periods=12)
    px = pd.Series([100, 105, 110, 99, 88, 95, 105, 111, 120, 108, 100, 96], index=idx, dtype=float)
    ep = find_drawdown_episodes(px)[0]
    r = pd.Series(0.0, index=idx)
    r.loc[idx[3]], r.loc[idx[4]] = -0.10, -0.10   # inside (peak, trough]
    r.loc[idx[5]] = 0.05                          # inside (trough, recovery]
    r.loc[idx[2]] = 0.50                          # ON the peak: must NOT count
    perf = episode_performance(r, ep)
    assert perf["decline_return"] == pytest.approx(0.9 * 0.9 - 1)
    assert perf["recovery_return"] == pytest.approx(0.05)
    assert perf["max_drawdown_in_episode"] == pytest.approx(0.9 * 0.9 - 1)


# --------------------------------------------------------------------------- #
# allocator
# --------------------------------------------------------------------------- #
def test_mean_variance_exposure_responds_to_a_rescaled_covariance():
    """The reason the MV leg has a cash sleeve: scaling Sigma changes TOTAL risky exposure."""
    mu = np.array([0.06, 0.05, 0.04])
    cov = np.diag([0.04, 0.03, 0.02])
    low = al.solve_mean_variance(mu, cov, gamma=8.0, cap=0.35).sum()
    high = al.solve_mean_variance(mu, cov * 4.0, gamma=8.0, cap=0.35).sum()
    assert high < low, "a higher forecast volatility must cut risky exposure"
    w = al.solve_mean_variance(mu, cov, gamma=8.0, cap=0.35)
    assert (w >= 0).all() and (w <= 0.35 + 1e-9).all() and w.sum() <= 1 + 1e-9


def test_mv_weights_must_differ_across_variants_or_the_run_aborts():
    idx = pd.bdate_range("2021-01-01", periods=4)
    same = pd.DataFrame(0.1, index=idx, columns=["A", "B"])
    dist = al.mv_pairwise_distance({"V1": same, "V2": same.copy()})
    with pytest.raises(al.IntegrityError, match="indistinguishable"):
        al.assert_mv_weights_differ(dist)
    other = same.copy()
    other["A"] = 0.12
    al.assert_mv_weights_differ(al.mv_pairwise_distance({"V1": same, "V2": other}))


def test_weight_checks_fail_loudly():
    params = al.AllocatorParams(risky=("A", "B", "C"), equity_sectors=("A", "B"))
    idx = pd.bdate_range("2021-01-01", periods=2)
    good = pd.DataFrame({"A": 0.3, "B": 0.3, "C": 0.3, "SPY": 0.0, "CASH": 0.1}, index=idx)
    al.check_weights("ok", good, params)
    over = good.copy()
    over["A"], over["CASH"] = 0.5, -0.1
    with pytest.raises(al.IntegrityError):
        al.check_weights("bad", over, params)
    short = good.copy()
    short["CASH"] = 0.0
    with pytest.raises(al.IntegrityError, match="sum to 1"):
        al.check_weights("bad", short, params)


def test_capped_inverse_vol_respects_the_cap_and_sums_to_one():
    w = capped_inverse_vol(np.array([0.005, 0.15, 0.16, 0.14, 0.12, 0.2]), cap=0.35)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.35 + 1e-9
    assert w.argmax() == 0, "the lowest-vol asset is the one that hits the cap"


def test_simulation_matches_a_hand_computed_path_with_drift_and_costs():
    """Two lines, one decision. Decide Mon close, execute Tue close, earn from Tue."""
    idx = pd.bdate_range("2021-03-01", periods=5)  # Mon..Fri
    rets = pd.DataFrame({"A": [0.0, 0.10, 0.10, -0.10, 0.0], "CASH": 0.0}, index=idx)
    w = pd.DataFrame({"A": [1.0], "CASH": [0.0]}, index=idx[:1])
    gross, turn = al.simulate(w, rets, execution_lag=1)
    assert gross.index[0] == idx[1], "the first return belongs to the execution session"
    assert gross.iloc[0] == pytest.approx(0.0), "executed AT this close: the day itself was held in cash"
    assert turn.iloc[0] == pytest.approx(1.0), "cash -> 100% A is a full one-way turn"
    # afterwards fully invested in A: returns +10%, -10%, 0
    assert gross.iloc[1] == pytest.approx(0.10) and gross.iloc[2] == pytest.approx(-0.10)
    net = al.net_returns(gross, turn, cost_bps=10.0)
    assert net.iloc[0] == pytest.approx(-0.001), "10 bps on one-way turnover of 1.0"
    assert (net.iloc[1:] == gross.iloc[1:]).all(), "no cost on days without a trade"


def test_drifted_weights_are_what_get_charged_at_a_rebalance():
    idx = pd.bdate_range("2021-03-01", periods=6)
    rets = pd.DataFrame({"A": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "B": 0.0, "CASH": 0.0}, index=idx)
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5], "CASH": 0.0}, index=[idx[0], idx[3]])
    gross, turn = al.simulate(w, rets, execution_lag=1)
    # A doubles on idx[2]: 0.5/0.5 drifts to 2/3 and 1/3, so the rebalance executed idx[4] trades 1/6.
    assert turn.loc[idx[4]] == pytest.approx(1.0 / 6.0)


def test_regime_leg_equals_plain_vol_target_when_there_is_no_regime_column():
    dates = pd.bdate_range("2021-01-01", periods=3)
    assets = ["A", "B", "C", "D"]
    params = al.AllocatorParams(risky=tuple(assets), equity_sectors=tuple(assets))
    inputs = al.AllocatorInputs(
        dates=dates, sigma_spy=pd.Series(0.16, index=dates), sigma_base=pd.Series(0.15, index=dates),
        asset_vol=pd.DataFrame(0.2, index=dates, columns=assets),
        mu=pd.DataFrame(0.05, index=dates, columns=assets), cov={d: np.eye(4) * 0.04 for d in dates},
    )
    sigma_hat = pd.Series([0.12, 0.2, 0.3], index=dates)
    vt = al.vol_target_weights(sigma_hat, inputs, params)
    rvt_none = al.regime_vol_target_weights(sigma_hat, None, inputs, params)
    pd.testing.assert_frame_equal(vt, rvt_none)
    stressed = al.regime_vol_target_weights(sigma_hat, pd.Series(1.0, index=dates), inputs, params)
    assert (stressed["CASH"] >= vt["CASH"]).all() and (stressed["CASH"] > vt["CASH"]).any()
    al.check_weights("vt", vt, params)


# --------------------------------------------------------------------------- #
# pre-registration integrity (DECISIONS.md D-033)
# --------------------------------------------------------------------------- #
def _load_script(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_cfg(root, variants=("V1", "V2"), diagnostic=("O1",)):
    from types import SimpleNamespace

    return SimpleNamespace(
        root=root, path=lambda key: root / "data" / "processed",
        tier1=SimpleNamespace(variants=list(variants), diagnostic_variants=list(diagnostic)),
    )


def _write_prereg(root, entries: dict[str, str]):
    path = root / "reports" / "tables" / "preregistration.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{h}  {name}" for name, h in entries.items()]
    path.write_text("```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")


def test_preregistered_hashes_parses_the_hash_table(tmp_path):
    from prism.utils.prereg import preregistered_hashes

    _write_prereg(tmp_path, {"states/V1.parquet": "a" * 64, "hmm_posteriors.parquet": "b" * 64})
    assert preregistered_hashes(tmp_path) == {
        "states/V1.parquet": "a" * 64, "hmm_posteriors.parquet": "b" * 64,
    }


def test_state_builder_refuses_to_overwrite_a_preregistered_file(tmp_path):
    build = _load_script("03b_build_states.py")
    states = tmp_path / "data" / "processed" / "states"
    states.mkdir(parents=True)
    (states / "V1.parquet").write_bytes(b"x")
    (states / "V2.parquet").write_bytes(b"x")
    _write_prereg(tmp_path, {"states/V1.parquet": "a" * 64})
    cfg = _stub_cfg(tmp_path)
    # V1 is pinned and present; V2 is present but not pinned; schema.json is absent.
    assert build.protected_targets(cfg, with_oracle=False) == ["states/V1.parquet"]
    # O1 is only in scope with --with-oracle.
    _write_prereg(tmp_path, {"states/V1.parquet": "a" * 64, "states/O1.parquet": "c" * 64})
    (states / "O1.parquet").write_bytes(b"x")
    assert "states/O1.parquet" not in build.protected_targets(cfg, with_oracle=False)
    assert "states/O1.parquet" in build.protected_targets(cfg, with_oracle=True)
    # A file that does not exist yet cannot be overwritten, pinned or not.
    (states / "V1.parquet").unlink()
    assert "states/V1.parquet" not in build.protected_targets(cfg, with_oracle=False)


def _o1_frame(shift: float = 0.0) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame({"f": np.arange(6.0), "state_0": np.linspace(0.1, 0.9, 6) + shift}, index=idx)


def _write_o1_pair(tmp_path, shift):
    processed = tmp_path / "data" / "processed" / "states"
    processed.mkdir(parents=True)
    _o1_frame().to_parquet(processed / "O1.amendment1.parquet")
    _o1_frame(shift).to_parquet(processed / "O1.parquet")
    return tmp_path / "data" / "processed"


def test_o1_check_tolerates_float_noise_but_not_a_real_change(tmp_path):
    t1s = _load_script("04_tier1_ablation.py")
    processed = _write_o1_pair(tmp_path, shift=1e-11)  # the observed run-to-run noise
    t1s._verify_o1(processed)
    processed = _write_o1_pair(tmp_path / "b", shift=1e-6)
    with pytest.raises(RuntimeError, match="differs from its Amendment 1 reference by"):
        t1s._verify_o1(processed)


def test_o1_check_rejects_a_different_shape(tmp_path):
    t1s = _load_script("04_tier1_ablation.py")
    processed = _write_o1_pair(tmp_path, shift=0.0)
    _o1_frame().rename(columns={"state_0": "other"}).to_parquet(processed / "states" / "O1.parquet")
    with pytest.raises(RuntimeError, match="index or columns"):
        t1s._verify_o1(processed)
