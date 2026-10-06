"""Dashboard milestone D2: the replay records what the stored evaluation did, the gate holds, the frozen models reproduce.

DASHBOARD.md §4.2, §5.3, §5.4; preregistration_holdout.md Amendment 1. No test opens the holdout, loads a trained
agent or reads ``data/processed``: the replay's output is checked through the committed artifacts, and the model
functions on a small synthetic walk-forward.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from prism import dashboard_data as dd  # noqa: E402
from prism import dashboard_replay as dr  # noqa: E402
from prism import live  # noqa: E402
from prism.data.loaders import HOLDOUT_ACCESS_LOG, HOLDOUT_ENV_VAR  # noqa: E402
from prism.env.costs import one_way_turnover  # noqa: E402
from prism.features.scaling import FeatureScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / dd.ARTIFACTS


def _script():
    spec = importlib.util.spec_from_file_location("dashboard_data_script", ROOT / "scripts" / "10_dashboard_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- the gate
def test_the_holdout_replay_refuses_without_both_keys_and_logs_nothing(monkeypatch):
    script, log = _script(), ROOT / HOLDOUT_ACCESS_LOG
    before = log.read_text() if log.exists() else None
    monkeypatch.delenv(HOLDOUT_ENV_VAR, raising=False)
    with pytest.raises(SystemExit, match="--i-am-sure"):
        script.main(["--stage", "holdout-replay"])
    with pytest.raises(SystemExit, match=f"{HOLDOUT_ENV_VAR}=1"):
        script.main(["--stage", "holdout-replay", "--i-am-sure"])
    monkeypatch.setenv(HOLDOUT_ENV_VAR, "1")
    with pytest.raises(SystemExit, match="--i-am-sure"):
        script.main(["--stage", "holdout-replay"])
    assert (log.read_text() if log.exists() else None) == before


def test_the_ungated_stages_cannot_be_given_the_holdout_flag_and_all_excludes_the_gated_stage():
    script = _script()
    assert "holdout-replay" not in script.UNGATED
    with pytest.raises(SystemExit):
        script.main(["--stage", "stored", "--i-am-sure"])


def test_the_amendment_is_appended_and_the_pinned_hashes_still_parse():
    from prism.utils.prereg import preregistered_hashes

    text = (ROOT / dr.PREREG_HOLDOUT).read_text(encoding="utf-8")
    head, _, amendments = text.partition("## Amendments")
    assert dr.AMENDMENT_MARK in amendments and dr.AMENDMENT_MARK not in head
    assert "Holdout output seen at this point: yes" in amendments and "no statistic, test or decision" in amendments
    assert len(preregistered_hashes(ROOT, dr.PREREG_HOLDOUT)) == 46


# --------------------------------------------------------------------------- the reproduce check
def test_one_series_off_by_more_than_the_tolerance_fails_the_replay():
    names = ["V1|s0|5", "BM|EqualWeight|5"]
    dr.require_reproduced({"V1|s0|5": 0.0, "BM|EqualWeight|5": 9e-10}, names, 1e-9)
    with pytest.raises(dr.ReplayError, match="1 of 2 daily series differ"):
        dr.require_reproduced({"V1|s0|5": 0.0, "BM|EqualWeight|5": 2e-9}, names, 1e-9)
    with pytest.raises(dr.ReplayError, match="REPLAY FAILED"):
        dr.require_reproduced({"V1|s0|5": float("nan"), "BM|EqualWeight|5": 0.0}, names, 1e-9)
    with pytest.raises(dr.ReplayError, match="did not cover exactly"):
        dr.require_reproduced({"V1|s0|5": 0.0}, names, 1e-9)


def test_a_replayed_series_on_another_index_is_a_failure_not_a_reindex():
    days = pd.bdate_range("2019-01-08", periods=3)
    with pytest.raises(dr.ReplayError, match="not on the stored index"):
        dr._compare("x", pd.Series(0.0, index=days), pd.Series(0.0, index=days.shift(1)), {})


# --------------------------------------------------------------------------- the recorded weights (test window)
@pytest.fixture(scope="module")
def recorded() -> dict:
    return {"agents": pd.read_parquet(ARTIFACTS / "weights" / "test_agents.parquet"),
            "benchmarks": pd.read_parquet(ARTIFACTS / "weights" / "test_benchmarks.parquet"),
            "check": json.loads((ARTIFACTS / "weights" / "test_check.json").read_text()),
            "meta": json.loads((ARTIFACTS / "test" / "eval_done.json").read_text())}


def test_the_test_window_replay_reproduced_every_stored_series(recorded, cfg):
    check = recorded["check"]
    n_series = (len(cfg.tier2.variants) * len(cfg.tier2.final_seeds) + 6) * len(cfg.env.costs.sensitivity_bps)
    assert check["ok"] and check["series"] == n_series == 184 and check["max_abs_diff"] <= dr.REPLAY_ATOL
    assert (check["first_return_day"], check["last_return_day"]) == (recorded["meta"]["first"], recorded["meta"]["last"])
    assert pd.Timestamp(check["first_decision"]) < pd.Timestamp(check["first_return_day"])


def test_every_agent_has_one_row_per_decision_and_valid_weights(recorded, cfg):
    a = recorded["agents"]
    w = a[[c for c in a.columns if c.startswith("w_")]].to_numpy()
    p = a[[c for c in a.columns if c.startswith("p_")]].to_numpy()
    assert a.groupby(["variant", "seed"]).ngroups == 40 and (a.groupby(["variant", "seed"]).size() == recorded["check"]["decisions"]).all()
    assert a.groupby(["variant", "seed"]).decision_date.apply(lambda d: tuple(d)).nunique() == 1          # one shared calendar
    assert (a.execution_date > a.decision_date).all()
    cap = cfg.data.allocation.weight_max
    for x in (w, p):
        assert x.min() >= -1e-12 and np.abs(x.sum(axis=1) - 1.0).max() < 1e-9
    assert w[:, :-1].max() <= cap + 1e-9                                  # the cap binds risky assets, not cash
    first = a[a.decision_date == a.decision_date.min()]
    assert (first["p_CASH"] == 1.0).all()                                 # every episode starts in cash
    sample = a.sample(200, random_state=0)
    cols_w, cols_p = [c for c in a.columns if c.startswith("w_")], [c for c in a.columns if c.startswith("p_")]
    for _, row in sample.iterrows():
        assert row.turnover == pytest.approx(one_way_turnover(row[cols_p].to_numpy(float), row[cols_w].to_numpy(float)), abs=1e-12)


def test_the_recorded_weights_are_the_ones_the_stored_evaluation_summarised(recorded, cfg):
    """Mean weights per agent and mean turnover per variant equal what the single stored evaluation wrote down."""
    a = recorded["agents"]
    cols = [c for c in a.columns if c.startswith("w_")]
    for (v, s), g in a.groupby(["variant", "seed"]):
        want = pd.Series(recorded["meta"]["weights"][f"{v}|s{s}"])
        got = g[cols].mean().rename(lambda c: c[2:])
        assert np.abs(got[want.index] - want).max() < 1e-6 + 1e-9
        assert g.turnover.mean() == pytest.approx(recorded["meta"]["turnover"][f"{v}|s{s}"], abs=1e-9)
    stored = pd.read_csv(ARTIFACTS / "test" / "tables" / "turnover.csv").set_index("variant")["mean_turnover_per_step"]
    by_variant = a.groupby(["variant", "seed"]).turnover.mean().groupby("variant").mean()
    for v in cfg.tier2.variants:
        assert by_variant[v] == pytest.approx(stored[v], abs=1e-9)
    b = recorded["benchmarks"].groupby("benchmark").turnover.mean()
    for name, value in b.items():
        assert value == pytest.approx(stored[f"BM|{name}"], abs=1e-9)


def test_the_benchmark_weights_are_their_definitions(recorded, cfg):
    b = recorded["benchmarks"]
    risky = [f"w_{t}" for t in cfg.data.allocatable["B"]]
    ew = b[b.benchmark == "EqualWeight"]
    assert np.allclose(ew[risky], 1 / len(risky)) and (ew[["w_SPY", "w_CASH"]] == 0).all().all()
    sf = b[b.benchmark == "SixtyForty"]
    assert np.allclose(sf.w_SPY, 0.6) and np.allclose(sf.w_IEF, 0.4)
    rp = b[b.benchmark == "RiskParity"]
    assert np.allclose(rp[risky].sum(axis=1), 1.0) and rp[risky].to_numpy().max() <= cfg.data.allocation.weight_max + 1e-9


# --------------------------------------------------------------------------- fold parameters
def test_the_fold_table_reports_state_parameters_in_return_units():
    class Scaler:
        def _params(self):
            return {"centre": pd.Series([0.0004]), "scale": pd.Series([0.012])}

    model = SimpleNamespace(n_components=2, means_=np.array([[0.1], [-0.2]]), covars_=np.array([[[0.25]], [[4.0]]]),
                            transmat_=np.array([[0.98, 0.02], [0.05, 0.95]]))
    fold = SimpleNamespace(index=7, fit_start=pd.Timestamp("1999-01-04"), fit_end=pd.Timestamp("2010-01-04"),
                           apply_start=pd.Timestamp("2010-02-01"), apply_end=pd.Timestamp("2010-02-26"))
    ext = SimpleNamespace(hmm=SimpleNamespace(folds=[SimpleNamespace(fold=fold, model=model, scaler=Scaler(), n_restarts_used=5,
                                                                    n_restarts_degenerate=1)]))
    [row] = dr.hmm_fold_table(ext).to_dict("records")
    assert row["vol_0"] == pytest.approx(0.5 * 0.012) and row["vol_1"] == pytest.approx(2.0 * 0.012)
    assert row["mean_1"] == pytest.approx(-0.2 * 0.012 + 0.0004)
    assert row["dwell_0"] == pytest.approx(50.0) and row["dwell_1"] == pytest.approx(20.0) and row["p_01"] == 0.02
    assert dr.hmm_fold_table(ext, first_fold=8).empty


def test_the_stored_fold_table_covers_every_fold_to_the_test_end_in_volatility_order(cfg):
    table = pd.read_csv(ARTIFACTS / "regimes" / "hmm_folds_test.csv", parse_dates=["apply_start", "apply_end", "fit_end"])
    summary = json.loads((ARTIFACTS / "facts.json").read_text())["models"]["hmm"]
    assert len(table) == summary["folds_through_test"] and table.fold.tolist() == list(range(len(table)))
    assert (table.vol_1 > table.vol_0).all()                              # state_1 is the Volatile state in every fold
    assert (table.fit_end < table.apply_start).all() and table.apply_start.is_monotonic_increasing
    assert np.allclose(table.p_00 + table.p_01, 1.0) and np.allclose(table.dwell_0, 1 / (1 - table.stay_0))


# --------------------------------------------------------------------------- the frozen models, on a synthetic walk-forward
@pytest.fixture(scope="module")
def tiny(cfg, features_a, tmp_path_factory):
    """A three-fold HMM walk-forward and a one-fold encoder on synthetic features, persisted with ``save_models``."""
    from prism.models.baselines.threshold_regime import threshold_regime_walkforward
    from prism.models.encoder.walkforward import encoder_walkforward
    from prism.models.hmm.walkforward import hmm_walkforward

    spec = cfg.hmm.fit.specification
    obs_col = list(cfg.hmm.specifications[spec].observations)
    extra = [c for c in features_a.frame.columns if c not in obs_col and c != "vix_level"][:6]
    frame = features_a.frame[[*obs_col, "vix_level", *extra]].dropna().iloc[-700:]
    start, end = frame.index[0], frame.index[-1]
    hmm_apply = frame.index[-64]
    wf = hmm_walkforward(frame[obs_col], k=2, fit_start=start, first_apply_start=hmm_apply, apply_end=end, cadence="monthly",
                         embargo_days=5, covariance_type=cfg.hmm.fit.covariance_type, n_restarts=2, n_iter=60, tol=1e-3,
                         seed_base=1, min_expected_duration_days=1.0, min_unconditional_prob=0.0, max_degenerate_fold_fraction=1.0)
    enc_apply = pd.Timestamp(year=end.year, month=1, day=1)
    ewf = encoder_walkforward(frame, variant="DAE", window=5, hidden_dim=8, latent_dim=4, fit_start=start, first_apply_start=enc_apply,
                              apply_end=end, cadence="annual", embargo_days=5, seed_base=1, max_epochs=2, patience=1,
                              dropout=cfg.encoder.architecture.dropout, latent_activation=cfg.encoder.architecture.latent_activation,
                              dae_noise_std=cfg.encoder.dae_noise_std, vae_kl_weight=cfg.encoder.vae_kl_weight)
    threshold = threshold_regime_walkforward(frame["vix_level"], k=2, fit_start=start, first_apply_start=hmm_apply, apply_end=end,
                                             cadence="monthly", embargo_days=5)
    scaler = FeatureScaler().fit(frame.iloc[:300], scope="test")
    ext = SimpleNamespace(hmm=wf, encoder=ewf, scaler=scaler, features_a=frame, end=end)
    stored = {"V1": scaler.transform(frame), "V2": ewf.latents, "V4": wf.posteriors, "C4": threshold}
    out = tmp_path_factory.mktemp("models")
    manifest = live.save_models(cfg, ext, out, spec=spec, k=2, encoder_selection={"variant": "DAE", "window": 5, "hidden_dim": 8, "latent_dim": 4},
                                threshold_column="vix_level")
    return SimpleNamespace(frame=frame, wf=wf, ewf=ewf, stored=stored, dir=out, manifest=manifest, obs_col=obs_col)


def test_the_persisted_last_fold_reproduces_the_stored_states_on_its_own_apply_window(tiny):
    models = live.load_models(tiny.dir, expected=tiny.manifest)
    check = live.verify_models(models, tiny.frame, tiny.frame, tiny.stored)
    assert check["ok"] and check["hmm"]["carry_matches"]
    for part in ("hmm", "encoder", "threshold", "state_scaler"):
        assert check[part]["rows"] > 0 and check[part]["max_abs_diff"] <= 1e-9
    assert models.hmm["fold"] == tiny.wf.folds[-1].fold.index and models.hmm["carry_date"] == str(tiny.frame.index[-1].date())
    assert check["hmm"]["rows"] == len(tiny.wf.posteriors.loc[tiny.wf.folds[-1].fold.apply_start:])


def test_continuing_the_filter_from_the_carried_state_equals_filtering_the_whole_history(tiny):
    """The live path steps forward from the persisted filter state; that must equal a fresh run over everything."""
    models = live.load_models(tiny.dir)
    h, fold = models.hmm, tiny.wf.folds[-1].fold
    obs = tiny.frame[tiny.obs_col]
    history = pd.concat([obs.loc[fold.fit_start: fold.fit_end], obs.loc[fold.apply_start: fold.apply_end]])
    cut = len(history) - 7
    _, carried = live.filter_hmm(h, history.iloc[:cut])
    continued, final = live.filter_hmm(h, history.iloc[cut:], initial_log_alpha=carried)
    whole, _ = live.filter_hmm(h, history)
    assert np.abs(continued.to_numpy() - whole.iloc[cut:].to_numpy()).max() < 1e-12
    assert np.allclose(final, h["carry_log_alpha"], rtol=0, atol=1e-9)
    assert np.allclose(continued.sum(axis=1), 1.0)


def test_the_live_forward_passes_do_not_read_the_future(tiny):
    models = live.load_models(tiny.dir)
    obs = tiny.frame[tiny.obs_col].iloc[-30:]
    garbled = obs.copy()
    garbled.iloc[20:] = 9.9
    a, _ = live.filter_hmm(models.hmm, obs, initial_log_alpha=np.array(models.hmm["carry_log_alpha"]))
    b, _ = live.filter_hmm(models.hmm, garbled, initial_log_alpha=np.array(models.hmm["carry_log_alpha"]))
    assert a.iloc[:20].equals(b.iloc[:20]) and not a.iloc[20:].equals(b.iloc[20:])
    frame = tiny.frame.copy()
    frame.iloc[-3:] = 5.0
    lat_a, lat_b = live.encode(models, tiny.frame), live.encode(models, frame)
    assert lat_a.iloc[:-3].equals(lat_b.iloc[:-3]) and not lat_a.iloc[-3:].equals(lat_b.iloc[-3:])


def test_changed_model_files_and_changed_parameters_are_refused(tiny, tmp_path):
    import shutil

    copy = tmp_path / "models"
    shutil.copytree(tiny.dir, copy)
    with pytest.raises(live.LiveModelError, match="differs from the committed one"):
        live.load_models(copy, expected={**tiny.manifest, "end": "1999-01-04"})
    models = live.load_models(copy)
    bad = live.LiveModels(meta={**models.meta, "hmm": {**models.hmm, "transmat": [[0.5, 0.5], [0.5, 0.5]]}}, encoder_state=models.encoder_state)
    with pytest.raises(live.LiveModelError, match="do not reproduce the stored states"):
        live.verify_models(bad, tiny.frame, tiny.frame, tiny.stored)
    with (copy / "encoder.npz").open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(live.LiveModelError, match="encoder.npz differs"):
        live.load_models(copy)
    with pytest.raises(live.LiveModelError, match="no frozen live models"):
        live.load_models(tmp_path / "absent")


def test_threshold_states_are_a_lookup_on_fixed_edges():
    params = {"edges": [20.0], "k": 2}
    out = live.threshold_states(params, pd.Series([12.0, 20.0, 35.0], index=pd.bdate_range("2026-10-01", periods=3)))
    assert out.to_numpy().tolist() == [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]] and list(out.columns) == ["state_0", "state_1"]


# --------------------------------------------------------------------------- the live path (DASHBOARD.md §4): sessions and the splice
def test_a_session_counts_only_after_its_close():
    days = pd.bdate_range("2026-10-05", periods=3)                       # Mon, Tue, Wed
    during = pd.Timestamp("2026-10-07 15:00", tz="America/New_York")     # Wednesday, market open
    after = pd.Timestamp("2026-10-07 16:45", tz="America/New_York")
    assert list(live.completed_sessions(days, during)) == list(days[:2])
    assert list(live.completed_sessions(days, after)) == list(days)
    assert list(live.completed_sessions(days, pd.Timestamp("2026-10-07 16:10", tz="America/New_York"))) == list(days[:2])   # inside the buffer
    weekend = pd.DatetimeIndex(["2026-10-09", "2026-10-10", "2026-10-11"])
    assert list(live.completed_sessions(weekend, pd.Timestamp("2026-10-12 12:00", tz="UTC"))) == [pd.Timestamp("2026-10-09")]


def _panels(n: int = 90, new: int = 4):
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2026-06-01", periods=n + new)
    etf = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n + new)), index=days)
    vix = pd.Series(18 + rng.normal(0, 1, n + new).cumsum() * 0.1, index=days)
    close = pd.DataFrame({"XLK": etf, "^VIX": vix})
    volume = pd.DataFrame({"XLK": rng.integers(1_000_000, 2_000_000, n + new).astype(float)}, index=days)
    return close.iloc[:n], volume.iloc[:n], close, volume


def test_the_splice_rescales_restated_prices_and_leaves_levels_alone():
    frozen_c, frozen_v, fresh_c, fresh_v = _panels()
    restated = fresh_c.assign(XLK=fresh_c.XLK * 0.991)                    # a dividend was paid since the snapshot
    close, volume, report = live.splice(frozen_c, frozen_v, restated, fresh_v)
    assert report["ok"] and report["new_sessions"] == 4 and report["anchor"] == str(frozen_c.index[-1].date())
    assert close.iloc[:90].equals(frozen_c)                               # the frozen rows are untouched
    assert report["tickers"]["XLK"]["scale"] == pytest.approx(1 / 0.991) and report["tickers"]["^VIX"]["scale"] == 1.0
    assert np.allclose(close.XLK.pct_change().iloc[-4:], fresh_c.XLK.pct_change().iloc[-4:])      # returns are continuous at the seam
    assert close["^VIX"].iloc[-4:].equals(fresh_c["^VIX"].iloc[-4:])
    assert volume.iloc[-4:].equals(fresh_v.iloc[-4:])


def test_the_splice_refuses_data_that_disagrees_with_the_snapshot():
    frozen_c, frozen_v, fresh_c, fresh_v = _panels()
    bad = fresh_c.copy()
    bad.loc[bad.index[70], "XLK"] *= 1.01                                 # one overlapping return off by 1%
    with pytest.raises(live.SeamError, match="data seam check failed for XLK"):
        live.splice(frozen_c, frozen_v, bad, fresh_v)
    shifted = fresh_c.assign(**{"^VIX": fresh_c["^VIX"] + 0.5})           # a level series must agree in level, not be rescaled
    with pytest.raises(live.SeamError, match=r"\^VIX"):
        live.splice(frozen_c, frozen_v, shifted, fresh_v)
    with pytest.raises(live.SeamError, match="overlap"):
        live.splice(frozen_c, frozen_v, fresh_c.iloc[-10:], fresh_v.iloc[-10:])
    with pytest.raises(live.SeamError, match="lacks"):
        live.splice(frozen_c, frozen_v, fresh_c[["XLK"]], fresh_v)


def test_a_split_rescales_price_and_volume_together():
    frozen_c, frozen_v, fresh_c, fresh_v = _panels()
    split_c, split_v = fresh_c.assign(XLK=fresh_c.XLK / 2), fresh_v * 2    # a 2-for-1 split restates the whole history
    close, volume, report = live.splice(frozen_c, frozen_v, split_c, split_v)
    assert report["tickers"]["XLK"]["scale"] == pytest.approx(2.0)
    assert np.allclose(close.XLK.iloc[-4:], fresh_c.XLK.iloc[-4:]) and np.allclose(volume.XLK.iloc[-4:], fresh_v.XLK.iloc[-4:])


def test_level_series_are_recognised_by_ticker(cfg):
    macro = [*cfg.data.universes["A"].macro, *cfg.data.universes["B"].macro_extra]
    assert {t for t in macro if live.is_level_series(t)} == {"^VIX", "^TNX", "^FVX", "^IRX", "^VIX3M", "DX-Y.NYB", "CL=F"}
    assert not any(live.is_level_series(t) for t in cfg.data.allocatable["B"])


# --------------------------------------------------------------------------- the live path: decisions, previews, the rollout
def test_a_decision_needs_a_finished_week():
    from prism.utils.calendar import trading_days

    sessions = trading_days("2026-09-21", "2026-10-09")
    assert live.decision_calendar(sessions) == (pd.Timestamp("2026-10-09"), True)                 # ends on a Friday
    assert live.decision_calendar(sessions[:-2]) == (pd.Timestamp("2026-10-02"), False)           # ends on a Wednesday: a preview
    assert live.decision_calendar(sessions[:-5]) == (pd.Timestamp("2026-10-02"), True)
    good_friday = trading_days("2026-03-23", "2026-04-02")                                        # Friday 3 April 2026 is a holiday
    assert live.decision_calendar(good_friday) == (pd.Timestamp("2026-04-02"), True)


@pytest.fixture(scope="module")
def toy(cfg):
    """A random-walk price panel and a 3-column state, enough to run the real environment."""
    from prism.utils.calendar import trading_days

    rng = np.random.default_rng(11)
    days = trading_days("2026-01-02", "2026-10-09")
    risky = list(cfg.data.allocatable["B"])
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(days), len(risky))), axis=0), index=days, columns=risky)
    close["^IRX"] = 4.0
    states = pd.DataFrame(rng.normal(size=(len(days), 3)), index=days, columns=["x0", "x1", "x2"]).iloc[30:]
    policy = lambda obs: np.tanh(obs[:14] * 0.7 + obs[3:17])  # noqa: E731 - depends on the state AND on the agent's own weights
    return SimpleNamespace(close=close, states=states, policy=policy, start=states.index[0])


def test_the_weights_chosen_at_a_decision_do_not_change_when_more_data_arrives(toy, cfg):
    """The terminal observation of an episode ended at a decision close is that decision's observation in any longer episode."""
    friday = pd.Timestamp("2026-09-25")
    short = live.rollout(cfg, toy.states, toy.close, toy.policy, start=toy.start, end=friday)
    long = live.rollout(cfg, toy.states, toy.close, toy.policy, start=toy.start, end=toy.states.index[-1])
    row = long["decisions"].set_index("decision_date").loc[friday]
    assert np.allclose(short["latest"]["weights"].to_numpy(), row[[c for c in row.index if c.startswith("w_")]].to_numpy(float), atol=1e-12)
    assert np.allclose(short["latest"]["held"].sum(), 1.0) and short["latest"]["date"] == friday
    common = short["decisions"].set_index("decision_date").index
    cols = [c for c in short["decisions"].columns if c.startswith("w_")]
    assert np.allclose(short["decisions"].set_index("decision_date")[cols], long["decisions"].set_index("decision_date").loc[common, cols], atol=1e-12)


def test_the_rollout_does_not_read_past_its_end(toy, cfg):
    friday = pd.Timestamp("2026-09-25")
    a = live.rollout(cfg, toy.states, toy.close, toy.policy, start=toy.start, end=friday)
    close, states = toy.close.copy(), toy.states.copy()
    close.loc[close.index > friday] *= 3.0
    states.loc[states.index > friday] = 99.0
    b = live.rollout(cfg, states, close, toy.policy, start=toy.start, end=friday)
    assert a["latest"]["weights"].equals(b["latest"]["weights"]) and a["decisions"].equals(b["decisions"])


def test_mid_week_gives_the_last_decision_and_a_separate_preview(toy, cfg):
    policies = {("V1", 0): toy.policy, ("V1", 1): lambda obs: -toy.policy(obs)}
    wednesday = pd.Timestamp("2026-10-07")
    out = live.latest_weights(cfg, {"V1": toy.states.loc[:wednesday]}, toy.close.loc[:wednesday], policies, start=toy.start)
    assert (out["decision_date"], out["as_of"], out["is_preview"]) == (pd.Timestamp("2026-10-02"), wednesday, True)
    assert out["preview"] is not None and not np.allclose(out["preview"].filter(like="w_"), out["decision"].filter(like="w_"))
    full = live.latest_weights(cfg, {"V1": toy.states}, toy.close, policies, start=toy.start)
    assert full["is_preview"] is False and full["preview"] is None and full["decision_date"] == full["as_of"] == toy.states.index[-1]
    ens = live.ensemble(full["decision"])
    assert np.allclose(ens["mean"].sum(), 1.0) and (ens["min"] <= ens["mean"]).all() and (ens["spread"] == ens["max"] - ens["min"]).all()


# --------------------------------------------------------------------------- what-if
def test_what_if_changes_only_the_two_regime_inputs_and_is_flat_without_them(cfg):
    lines = [f"weight_{t}" for t in [*cfg.data.allocatable["B"], "CASH"]]
    with_regime = ["f0", "f1", "state_0", "state_1", *lines, "mean_turnover"]
    without = ["f0", "f1", *lines, "mean_turnover"]
    seen = []

    def policy(obs):
        seen.append(obs.copy())
        return np.tanh(np.resize(obs, 14))

    obs = np.linspace(-1, 1, len(with_regime)).astype("float32")
    base = live.what_if_weights(cfg, policy, obs, with_regime, None)
    assert np.array_equal(seen[-1], obs)
    moved = live.what_if_weights(cfg, policy, obs, with_regime, 0.8)
    assert seen[-1][2] == np.float32(0.2) and seen[-1][3] == np.float32(0.8)
    changed = np.flatnonzero(seen[-1] != obs)
    assert set(changed) <= set(live.regime_positions(with_regime)) and not np.allclose(base, moved)
    assert np.allclose(moved.sum(), 1.0) and moved[:-1].max() <= cfg.data.allocation.weight_max + 1e-12
    flat = [live.what_if_weights(cfg, policy, obs[: len(without)], without, p) for p in (None, 0.0, 0.5, 1.0)]
    assert all(np.array_equal(flat[0], f) for f in flat)                   # V1 and V2: the slider changes nothing, by construction
    with pytest.raises(ValueError):
        live.what_if_weights(cfg, policy, obs, with_regime, 1.2)
    with pytest.raises(ValueError):
        live.regime_positions(without)


def test_the_live_path_was_verified_against_the_recorded_test_window():
    check = json.loads((ARTIFACTS / "live" / "live_check.json").read_text())
    assert check["ok"] and check["rollout"]["agents"] == 40 and check["rollout"]["decisions"] == 255
    assert max(check["rollout"]["max_abs_diff"].values()) <= dr.REPLAY_ATOL
    assert max(check["live_states"]["max_abs_diff"].values()) <= dr.REPLAY_ATOL and min(check["live_states"]["rows"].values()) > 0
    seen = json.loads((ARTIFACTS / "weights" / "test_last_observations.json").read_text())
    assert len(seen["observations"]) == 40 and {v: len(c) for v, c in seen["columns"].items()} == {"V1": 199, "V2": 231, "V4": 233, "C4": 233}
    assert seen["columns"]["V4"][216:218] == ["state_0", "state_1"] == seen["columns"]["C4"][216:218]


def test_a_refresh_refuses_weights_whose_episode_does_not_contain_the_recorded_one(toy, cfg):
    """Every refresh re-derives the recorded decisions on the way to the latest one; a mismatch stops it."""
    policies = {("V1", 0): toy.policy}
    friday = pd.Timestamp("2026-09-25")
    recorded = live.rollout(cfg, toy.states, toy.close, toy.policy, start=toy.start, end=friday)["decisions"].assign(variant="V1", seed=0)
    out = live.latest_weights(cfg, {"V1": toy.states}, toy.close, policies, start=toy.start, recorded=recorded)
    assert out["recorded_max_abs_diff"] == 0.0
    tampered = recorded.copy()
    tampered.loc[tampered.index[10], "w_GLD"] += 1e-6
    with pytest.raises(live.LiveModelError, match="differs from the recorded one"):
        live.latest_weights(cfg, {"V1": toy.states}, toy.close, policies, start=toy.start, recorded=tampered)
    with pytest.raises(live.LiveModelError, match="does not contain the recorded decisions"):
        live.latest_weights(cfg, {"V1": toy.states}, toy.close, policies, start=toy.start, recorded=recorded.assign(seed=5))


holdout_recorded = pytest.mark.skipif(not (ARTIFACTS / "weights" / "holdout_check.json").exists(),
                                      reason="the gated holdout replay's artifacts are not in this checkout")


@holdout_recorded
def test_the_holdout_replay_reproduced_the_stored_evaluation_and_the_frozen_models_reproduce_the_states(cfg):
    check = json.loads((ARTIFACTS / "weights" / "holdout_check.json").read_text())
    meta = json.loads((ARTIFACTS / "holdout" / "eval_done.json").read_text())
    assert check["ok"] and check["series"] == 184 and check["max_abs_diff"] <= dr.REPLAY_ATOL and check["agents"] == 40
    assert (check["first_return_day"], check["last_return_day"]) == (meta["first"], meta["last"])
    agents = pd.read_parquet(ARTIFACTS / "weights" / "holdout_agents.parquet")
    assert agents.decision_date.nunique() == check["decisions"] and (agents[agents.decision_date == agents.decision_date.min()].p_CASH == 1.0).all()
    for (v, s), g in agents.groupby(["variant", "seed"]):
        assert g.turnover.mean() == pytest.approx(meta["turnover"][f"{v}|s{s}"], abs=1e-9)
    models = json.loads((ARTIFACTS / "live" / "models_check.json").read_text())
    manifest = json.loads((ARTIFACTS / "live" / "models_manifest.json").read_text())
    assert all(v["max_abs_diff"] <= dr.REPLAY_ATOL for v in models["walkforward"]["variants"].values())
    assert all(models["reproduce"][k]["max_abs_diff"] <= dr.REPLAY_ATOL and models["reproduce"][k]["rows"] > 0
               for k in ("hmm", "encoder", "threshold", "state_scaler"))
    assert manifest["end"] == meta["last"] and manifest["hmm"]["fit_end"] < manifest["hmm"]["apply_start"]
    folds = pd.read_csv(ARTIFACTS / "regimes" / "hmm_folds_holdout.csv")
    first = json.loads((ARTIFACTS / "facts.json").read_text())["models"]["hmm"]["folds_through_test"]
    assert folds.fold.tolist() == list(range(first, manifest["hmm"]["fold"] + 1)) and (folds.vol_1 > folds.vol_0).all()
