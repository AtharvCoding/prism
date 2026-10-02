"""Tier 2 (build step 4c): SAC wiring, evaluation through the env, statistics, and the run contract.

Everything runs on synthetic data and never reads the test split: the one
behavioural guard on that (``evaluate_test`` refuses before loading anything) is
tested by making the loader raise.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from prism.agents import evaluate as ev  # noqa: E402
from prism.agents import jobs, tier2 as t2  # noqa: E402
from prism.agents.sac import SacConfig, SacSettings, greedy_policy, train_run  # noqa: E402
from prism.agents.sanity import synthetic_edge_data  # noqa: E402
from prism.analysis import tier2 as st  # noqa: E402
from prism.backtest.metrics import compute_metrics  # noqa: E402
from prism.config import Tier2Config  # noqa: E402
from prism.env.costs import CostModel  # noqa: E402
from prism.env.portfolio_env import make_env  # noqa: E402
from prism.reporting.tier2_report import DECISIONS_MARKER, append_decisions  # noqa: E402


@pytest.fixture(scope="module")
def data(cfg):
    return synthetic_edge_data(cfg, seed=3)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_tier2_config_is_the_preregistered_design(cfg):
    t = cfg.tier2
    assert t.variants == ["V1", "V2", "V4", "C4"]
    assert t.reward == "log_return_net"
    assert sorted(t.grid.gamma) == [0.9, 0.97] and sorted(t.grid.lr) == [0.0003, 0.001]
    assert [list(h) for h in t.grid.hidden] == [[64, 64], [128, 128]] and t.grid.n_configs == 8
    assert len(t.tuning_seeds) == 3 and len(t.final_seeds) == 10
    assert set(t.tuning_seeds).isdisjoint(t.final_seeds)
    assert [tuple(c) for c in t.comparisons] == [("V4", "V2"), ("V4", "C4"), ("V2", "V1")]


def test_tuning_and_final_seeds_must_be_disjoint(cfg):
    raw = cfg.tier2.model_dump()
    raw["final_seeds"] = [*raw["final_seeds"][:-1], raw["tuning_seeds"][0]]
    with pytest.raises(ValueError, match="disjoint"):
        Tier2Config.model_validate(raw)


def test_fewer_than_ten_final_seeds_is_refused(cfg):
    raw = cfg.tier2.model_dump()
    raw["final_seeds"] = raw["final_seeds"][:9]
    with pytest.raises(ValueError, match="10 seeds"):
        Tier2Config.model_validate(raw)


def test_the_grid_has_eight_distinct_configs(cfg):
    ids = [c.cfg_id for c in t2.grid_configs(cfg)]
    assert len(ids) == len(set(ids)) == 8


def test_smoke_plan_evaluates_validation_in_its_own_directory(cfg):
    plan = t2.Tier2Plan.from_config(cfg, cfg.root)
    smoke = plan.smoke_plan()
    assert plan.eval_split == "test" and smoke.eval_split == "val" and smoke.smoke
    assert smoke.out_dir != plan.out_dir and smoke.log_dir != plan.log_dir
    assert smoke.steps < plan.steps


def test_training_code_cannot_reach_the_test_split():
    """The worker cuts the close panel at validation; nothing in training names the test split."""
    src = inspect.getsource(jobs) + inspect.getsource(__import__("prism.agents.sac", fromlist=["x"]))
    assert 'load_close(cfg, "val")' in src
    assert '"test"' not in src and "'test'" not in src


# --------------------------------------------------------------------------- #
# SAC wiring
# --------------------------------------------------------------------------- #
def test_train_run_scores_checkpoints_and_keeps_the_best_on_validation(cfg, data, tmp_path):
    train = make_env(cfg, data, mode="train")
    train_eval = make_env(cfg, data, mode="eval")
    val = make_env(cfg, synthetic_edge_data(cfg, seed=4), mode="eval")
    res = train_run(train, val, SacConfig(0.9, (16, 16), 1e-3), SacSettings(learning_starts=100, batch_size=32),
                    seed=1, total_steps=600, eval_every=200, out_dir=tmp_path, train_eval_env=train_eval)
    curve = pd.read_csv(tmp_path / "curve.csv")
    assert list(curve.step) == [200, 400, 600] and res["n_checkpoints"] == 3
    assert res["best"]["val_mean_log_return"] == pytest.approx(curve.val_mean_log_return.max())
    assert (tmp_path / "best.zip").exists()
    assert np.isfinite(curve.drop(columns="step").to_numpy()).all()


def test_the_trained_policy_is_deterministic_and_loadable(cfg, data, tmp_path):
    from prism.agents.sac import load_agent

    train = make_env(cfg, data, mode="train")
    ev_env = make_env(cfg, data, mode="eval")
    train_run(train, ev_env, SacConfig(0.9, (16, 16), 1e-3), SacSettings(learning_starts=100, batch_size=32),
              seed=2, total_steps=300, eval_every=300, out_dir=tmp_path, train_eval_env=ev_env)
    a = ev.run_policy(make_env(cfg, data, mode="eval"), greedy_policy(load_agent(tmp_path / "best.zip")))
    b = ev.run_policy(make_env(cfg, data, mode="eval"), greedy_policy(load_agent(tmp_path / "best.zip")))
    pd.testing.assert_series_equal(a["daily"], b["daily"])


def test_same_seed_gives_the_same_run(cfg, data, tmp_path):
    out = []
    for i in range(2):
        d = tmp_path / str(i)
        out.append(train_run(make_env(cfg, data, mode="train"), make_env(cfg, data, mode="eval"),
                             SacConfig(0.9, (16, 16), 1e-3), SacSettings(learning_starts=100, batch_size=32), seed=5,
                             total_steps=300, eval_every=300, out_dir=d, train_eval_env=make_env(cfg, data, mode="eval"))["best"])
    assert out[0] == out[1]


# --------------------------------------------------------------------------- #
# evaluation through the env
# --------------------------------------------------------------------------- #
def test_step_weights_is_what_step_does_after_the_action_map(cfg, data):
    from prism.env.actions import action_to_weights

    rng = np.random.default_rng(0)
    actions = [rng.uniform(-1, 1, data.n_risky + 1).astype("float32") for _ in range(8)]
    e1, e2 = make_env(cfg, data, mode="eval"), make_env(cfg, data, mode="eval")
    e1.reset(seed=0)
    e2.reset(seed=0)
    for a in actions:
        o1, r1, *_ = e1.step(a)
        o2, r2, *_ = e2.step_weights(action_to_weights(a, e2._upper, e2.logit_scale))
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2


def test_daily_net_series_compounds_to_the_env_nav(cfg, data):
    rng = np.random.default_rng(1)
    res = ev.run_policy(make_env(cfg, data, mode="eval"), lambda o: rng.uniform(-1, 1, data.n_risky + 1).astype("float32"))
    assert res["cost"].sum() > 0                      # the series is net of a real cost
    assert res["daily"].index.is_monotonic_increasing and not res["daily"].index.has_duplicates


def test_a_constant_mix_through_the_env_matches_the_analytic_return(cfg, data):
    w = np.zeros(data.n_risky + 1)
    w[0], w[-1] = 0.3, 0.7
    env = make_env(cfg, data, mode="eval", cost_model=CostModel(0.0, 0.0))
    res = ev.run_weights(env, lambda e: w)
    prices = np.cumprod(1.0 + data.line_returns, axis=0)
    expected = np.prod([w @ (prices[data.end_pos[k]] / prices[data.exec_pos[k]]) for k in range(data.n_decisions)])
    got = np.prod(1.0 + res["daily"].to_numpy())
    assert got == pytest.approx(expected, rel=1e-10)    # weekly rebalancing to a constant mix, drift inside the week
    assert res["turnover"][1:].max() < 0.05


def test_holding_costs_nothing_after_the_initiation(cfg, data):
    w = np.zeros(data.n_risky + 1)
    w[0], w[-1] = 0.3, 0.7
    env = make_env(cfg, data, mode="eval")
    res = ev.run_weights(env, lambda e: w if e._steps == 0 else e.drifted_weights)
    assert res["cost"][0] > 0 and np.allclose(res["cost"][1:], 0.0, atol=1e-15)


def test_every_benchmark_runs_through_the_env_and_keeps_its_constraints(cfg):
    from prism.data.synthetic import make_synthetic_raw
    from prism.features.build import build_features

    raw = make_synthetic_raw(cfg, seed=0, universe="B")
    close = build_features(raw, cfg, "B").close
    sessions = close.index[(close.index >= "2013-01-01") & (close.index <= "2014-12-31")]
    sub = close.loc[: sessions[-1]]
    from prism.env.data import build_env_data

    risky = [*cfg.data.allocatable[cfg.env.universe], "SPY"]
    states = pd.DataFrame(np.zeros((len(sessions), 2)), index=sessions)
    # a tiny panel: cut the effective range by hand through EnvData.from_arrays instead of a split
    from prism.env.data import EnvData, line_returns, CASH
    from prism.backtest.engine import TimingConvention

    lr = line_returns(sub, risky)
    vol = lr[risky].rolling(20).std()
    data = EnvData.from_arrays(sessions, states.to_numpy(), lr.reindex(sessions).to_numpy(), vol.reindex(sessions).to_numpy(),
                               lines=(*risky, CASH), convention=TimingConvention.from_config(cfg))
    frames = ev.benchmark_weight_frames(cfg, sub, data)
    assert set(frames) == set(t2.BENCHMARKS)
    out = {n: ev.run_benchmark(cfg, data, frames, n, CostModel.from_config(cfg)) for n in t2.BENCHMARKS}
    idx = out["EqualWeight"]["daily"].index
    for n, r in out.items():
        assert r["daily"].index.equals(idx), n
        assert np.isfinite(r["daily"].to_numpy()).all()
    spy = out["BuyHoldSPY"]
    assert spy["cost"][0] > 0 and np.allclose(spy["cost"][1:], 0.0, atol=1e-15)   # bought once, never rebalanced
    assert out["EqualWeight"]["turnover"][1:].mean() < 0.2


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_point_metrics_match_the_backtest_module():
    r = np.random.default_rng(0).normal(0.0004, 0.01, 900)
    p = st.point_metrics(r)
    m = compute_metrics(pd.Series(r))
    assert p["annualised_return"] == pytest.approx(m.annualised_return)
    assert p["sharpe"] == pytest.approx(m.sharpe)
    assert p["max_drawdown"] == pytest.approx(m.max_drawdown)
    assert p["cvar_95"] == pytest.approx(m.cvar_95)


def test_replicates_on_the_identity_path_equal_the_point_estimate():
    r = np.random.default_rng(1).normal(0.0004, 0.01, 400)
    rep = st.series_metrics_replicates(r, np.arange(400)[None, :])
    p = st.point_metrics(r)
    for m in st.METRICS:
        assert rep[m][0] == pytest.approx(p[m])


def test_variant_replicates_are_deterministic_and_share_day_paths():
    from prism.analysis.bootstrap import bootstrap_paths

    rng = np.random.default_rng(2)
    series = {s: rng.normal(0.0005, 0.01, 300) for s in range(4)}
    paths = bootstrap_paths(300, 20, 50, 7)
    a = st.variant_replicates(series, paths, master=1, variant="V")
    b = st.variant_replicates(series, paths, master=1, variant="V")
    for m in st.METRICS:
        np.testing.assert_array_equal(a[m], b[m])
    c = st.variant_replicates(series, paths, master=1, variant="W")    # another variant: another seed resample
    assert not np.array_equal(a["sharpe"], c["sharpe"])


def _table(diffs: dict[str, tuple[float, float]]):
    rows = []
    for m, (lo, hi) in diffs.items():
        v = "favourable" if lo > 0 else "adverse" if hi < 0 else "indeterminate"
        rows.append({"candidate": "X", "control": "Y", "metric": m, "verdict": v, "nonoverlap_verdict": "indeterminate"})
    return pd.DataFrame(rows)


def test_the_gate_rule_is_three_of_four_favourable_and_none_adverse():
    m = list(st.METRICS)
    good = _table({m[0]: (0.1, 0.2), m[1]: (0.1, 0.2), m[2]: (0.1, 0.2), m[3]: (-0.1, 0.1)})
    assert st.gate_verdict(good, "X", "Y")["pass"]
    two = _table({m[0]: (0.1, 0.2), m[1]: (0.1, 0.2), m[2]: (-0.1, 0.1), m[3]: (-0.1, 0.1)})
    assert not st.gate_verdict(two, "X", "Y")["pass"]
    adverse = _table({m[0]: (0.1, 0.2), m[1]: (0.1, 0.2), m[2]: (0.1, 0.2), m[3]: (-0.3, -0.1)})
    assert not st.gate_verdict(adverse, "X", "Y")["pass"]          # 3 favourable but one adverse


def test_a_better_candidate_gets_favourable_verdicts_and_the_reverse_adverse():
    from prism.analysis.bootstrap import bootstrap_paths

    rng = np.random.default_rng(3)
    n = 500
    noise = rng.normal(0, 0.008, n)
    good = {s: noise + 0.003 + rng.normal(0, 0.0005, n) for s in range(5)}
    base = {s: noise + 0.0 + rng.normal(0, 0.0005, n) for s in range(5)}
    paths = bootstrap_paths(n, 20, 400, 11)
    reps = {"G": st.variant_replicates(good, paths, master=1, variant="G"), "B": st.variant_replicates(base, paths, master=1, variant="B")}
    pts = {k: {m: float(np.mean([st.point_metrics(x)[m] for x in v.values()])) for m in st.METRICS}
           for k, v in (("G", good), ("B", base))}
    t = st.paired_table(reps, pts, [("G", "B"), ("B", "G")])
    assert st.gate_verdict(t, "G", "B")["favourable"] >= 3 and st.gate_verdict(t, "G", "B")["pass"]
    assert st.gate_verdict(t, "B", "G")["adverse"] >= 3 and not st.gate_verdict(t, "B", "G")["pass"]


# --------------------------------------------------------------------------- #
# selection, freezing, resuming
# --------------------------------------------------------------------------- #
def _fake_result(plan, stage, variant, cfgc, seed, val_best, *, done=True):
    d = jobs.job_dir(plan.out_dir, stage, variant, cfgc.cfg_id, seed)
    d.mkdir(parents=True, exist_ok=True)
    row = {"step": 100, "train_mean_log_return": 0.01, "train_weekly_sharpe": 1.0, "val_mean_log_return": val_best, "val_weekly_sharpe": 0.1}
    if done:
        (d / "result.json").write_text(json.dumps({
            "variant": variant, "config": {"gamma": cfgc.gamma, "hidden": list(cfgc.hidden), "lr": cfgc.lr, "cfg_id": cfgc.cfg_id},
            "data_seed": seed, "best": row, "final": row, "wall_seconds": 1.0, "steps_per_second": 100.0}))


@pytest.fixture()
def plan(cfg, tmp_path):
    p = t2.Tier2Plan.from_config(cfg, tmp_path)
    return replace(p, out_dir=tmp_path / "out", log_dir=tmp_path / "log")


def test_tuning_picks_the_best_mean_over_seeds_and_breaks_ties_by_grid_order(plan):
    for v in plan.variants:
        for i, c in enumerate(plan.configs):
            for s in plan.tuning_seeds:
                _fake_result(plan, "tune", v, c, s, 0.001 if i != 3 else 0.002)
    # two tied winners for V2: indices 3 and 5 equal -> earlier wins
    for s in plan.tuning_seeds:
        _fake_result(plan, "tune", "V2", plan.configs[5], s, 0.002)
    tbl = t2.tuning_table(plan)
    assert tbl[tbl.variant == "V4"].iloc[0].cfg_id == plan.configs[3].cfg_id
    assert tbl[tbl.variant == "V2"].iloc[0].cfg_id == plan.configs[3].cfg_id


def test_tuning_refuses_to_select_from_an_incomplete_grid(plan):
    _fake_result(plan, "tune", "V1", plan.configs[0], plan.tuning_seeds[0], 0.1)
    with pytest.raises(RuntimeError, match="incomplete"):
        t2.tuning_table(plan)


def test_a_frozen_choice_is_never_overwritten_and_edits_are_detected(plan):
    import logging

    log = logging.getLogger("t")
    for v in plan.variants:
        for i, c in enumerate(plan.configs):
            for s in plan.tuning_seeds:
                _fake_result(plan, "tune", v, c, s, 0.001 + 0.0001 * (i == 2))
    frozen = t2.freeze_configs(plan, log)
    assert all(c.cfg_id == plan.configs[2].cfg_id for c in frozen.values())
    assert t2.load_frozen(plan) == frozen
    for v in plan.variants:                                            # the data changes: another config now wins
        for s in plan.tuning_seeds:
            _fake_result(plan, "tune", v, plan.configs[6], s, 0.5)
    with pytest.raises(RuntimeError, match="different selection"):
        t2.freeze_configs(plan, log)
    path = plan.out_dir / "chosen_configs.json"
    rec = json.loads(path.read_text())
    rec["variants"]["V1"]["lr"] = 0.5
    path.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="edited"):
        t2.load_frozen(plan)


def test_a_finished_run_is_skipped_and_an_unfinished_one_is_not_resumed(plan):
    c = plan.configs[0]
    _fake_result(plan, "tune", "V1", c, 1000, 0.1)
    spec = jobs.make_spec(plan.root, "tune", "V1", c, 1000, plan.steps, plan.eval_every, plan.settings, plan.log_dir, plan.out_dir)
    assert jobs.run_job(spec)["best"]["val_mean_log_return"] == 0.1          # returned from disk, nothing trained
    d = jobs.job_dir(plan.out_dir, "tune", "V1", c.cfg_id, 1001)
    d.mkdir(parents=True)
    (d / "best.zip").write_text("partial")
    assert not jobs.is_done(d)                                                # no result.json: unfinished
    assert len(t2.tuning_specs(plan)) == 4 * 8 * 3 and len(t2.final_specs(plan, {v: c for v in plan.variants})) == 40


def test_the_test_split_stays_closed_until_everything_upstream_is_done(cfg, plan, monkeypatch):
    """evaluate_test raises before loading any data if sanity failed, the configs are not frozen, or a final run is missing."""
    import prism.agents.data as agent_data

    def boom(*a, **k):
        raise AssertionError("the evaluation split was read")

    monkeypatch.setattr(agent_data, "load_close", boom)
    import logging

    log = logging.getLogger("t")
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="sanity"):
        t2.evaluate_test(plan, cfg, log, {"passed": False})
    with pytest.raises(RuntimeError, match="not frozen"):
        t2.evaluate_test(plan, cfg, log, {"passed": True})
    for v in plan.variants:
        for i, c in enumerate(plan.configs):
            for s in plan.tuning_seeds:
                _fake_result(plan, "tune", v, c, s, 0.001)
    t2.freeze_configs(plan, log)
    with pytest.raises(RuntimeError, match="not finished"):
        t2.evaluate_test(plan, cfg, log, {"passed": True})


def test_a_second_evaluation_is_a_no_op(cfg, plan):
    import logging

    plan.out_dir.mkdir(parents=True, exist_ok=True)
    (plan.out_dir / "eval_done.json").write_text("{}")
    assert t2.evaluate_test(plan, cfg, logging.getLogger("t"), None) == plan.out_dir / "eval_daily.parquet"


def test_decisions_are_appended_once(tmp_path):
    (tmp_path / "DECISIONS.md").write_text("# DECISIONS\n")
    assert append_decisions(tmp_path, f"{DECISIONS_MARKER}\n\nbody\n")
    assert not append_decisions(tmp_path, f"{DECISIONS_MARKER}\n\nagain\n")
    assert (tmp_path / "DECISIONS.md").read_text().count(DECISIONS_MARKER) == 1


# --------------------------------------------------------------------------- #
# sanity task
# --------------------------------------------------------------------------- #
def test_the_degenerate_task_has_exactly_one_deterministic_edge(cfg):
    s = cfg.tier2.sanity
    d = synthetic_edge_data(cfg, seed=0)
    risky = list(d.lines[:-1])
    edge = risky.index(s.edge_asset)
    assert np.allclose(d.line_returns[1:, edge], s.edge_daily_return)
    others = np.delete(d.line_returns[:, :-1], edge, axis=1)
    assert abs(others.mean()) < 5e-4 and others.std() == pytest.approx(s.noise_daily_vol, rel=0.05)
    assert np.allclose(d.line_returns[:, -1], 0.0)
    other = synthetic_edge_data(cfg, seed=1)
    assert not np.allclose(d.line_returns[:, 0 if edge else 1], other.line_returns[:, 0 if edge else 1])  # noise differs by seed


def test_the_sanity_record_renders():
    from prism.reporting.tier2_report import sanity_markdown

    deg = {"seed": 1, "edge_weight_mean": 0.35, "edge_weight_fraction_of_cap": 1.0, "agent_mean_log_return": 0.003,
           "oracle_mean_log_return": 0.0035, "log_return_fraction_of_oracle": 0.9, "wall_seconds": 10.0, "passed": True}
    row = {"step": 5000, "val": {"mean_log_return": -0.0002}, "train": {"mean_log_return": 0.001}}
    beat = {"seed": 1, "selected": row, "final": row, "beats_gate_quantile": True, "selected_percentile_rank": 0.9,
            "final_percentile_rank": 0.5, "selected_beats_reported_quantile": False}
    rec = {"config": {"cfg_id": "c"}, "settings": {"reward_scale": 10.0}, "steps": 100, "eval_every": 10, "passed": True,
           "degenerate_task": {"passed": True, "seeds": [deg]},
           "beats_random": {"passed": True, "random_median": -0.0015, "random_max": 0.001, "reported_threshold_mean_log_return": 0.0,
                            "equal_weight_constant_policy_val": -0.0006, "equal_weight_constant_percentile_rank": 0.8, "seeds": [beat]}}
    text = sanity_markdown(rec)
    assert "Overall: PASS" in text and "Attempt history" in text and "| 1 |" in text
