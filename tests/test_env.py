"""Portfolio environment. Spec §7.5, §11; DECISIONS.md D-034 .. D-036.

Synthetic paths with known answers: the expected value of every timing and
accounting test is computed from the price path directly, not from the code under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from scipy.optimize import minimize  # noqa: E402

from prism.backtest.engine import TimingConvention  # noqa: E402
from prism.env.actions import action_to_weights, project_capped_simplex, softmax, upper_bounds  # noqa: E402
from prism.env.costs import CostModel  # noqa: E402
from prism.env.data import EnvData, build_env_data, line_returns  # noqa: E402
from prism.env.portfolio_env import PortfolioEnv, run_episode  # noqa: E402
from prism.env.rewards import (  # noqa: E402
    DifferentialSharpe, DrawdownPenalty, LogReturnNet, MeanVariancePenalty, StepRecord,
)

CONVENTION = TimingConvention(frequency="weekly", rebalance_day="FRI", execution_lag_days=1)
NO_COST = CostModel(per_side_bps=0.0, slippage_vol_coef=0.0)
LINES = ("A", "B", "CASH")


def make_data(
    n_sessions: int = 80, *, ret_a: float = 0.01, ret_b: float = 0.0, cash: float = 0.0,
    d: int = 3, seed: int = 0, sigma: float = 0.01, spike: tuple[int, float] | None = None,
    lines: tuple[str, ...] = LINES,
) -> EnvData:
    """Two risky assets with constant daily returns (and optional one-session spike in A), plus cash."""
    sessions = pd.bdate_range("2020-01-06", periods=n_sessions)  # starts on a Monday
    n = len(lines) - 1
    r = np.zeros((n_sessions, n + 1))
    r[:, 0], r[:, 1], r[:, -1] = ret_a, ret_b, cash
    r[0] = 0.0
    if spike is not None:
        r[spike[0], 0] = spike[1]
    states = np.random.default_rng(seed).normal(size=(n_sessions, d))
    return EnvData.from_arrays(
        sessions, states, r, np.full((n_sessions, n), sigma), lines=lines, convention=CONVENTION
    )


def make_env(data: EnvData, *, cost: CostModel = NO_COST, mode: str = "eval", cap: float = 0.5,
             logit_scale: float = 3.0, reward=None, episode_length: int | None = None, env_cls=PortfolioEnv):
    return env_cls(
        data, cost, reward if reward is not None else LogReturnNet(), cap=cap, logit_scale=logit_scale,
        mode=mode, episode_length=episode_length,
    )


def zero_action(env):
    return lambda obs: np.zeros(env.action_space.shape, dtype="float32")


# --------------------------------------------------------------------------- #
# actions: weights by construction
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one_and_respect_bounds():
    """§7.5. Long-only with cash, each risky weight in [0, cap], enforced by projection."""
    rng = np.random.default_rng(0)
    n, cap = 13, 0.35
    upper = upper_bounds(n, cap)
    for _ in range(500):
        action = rng.uniform(-1, 1, n + 1)
        w = action_to_weights(action, upper, logit_scale=rng.uniform(0.5, 12.0))
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert (w >= 0).all()
        assert (w[:-1] <= cap + 1e-12).all()


def test_projection_is_the_identity_on_a_feasible_point():
    upper = upper_bounds(3, 0.5)
    p = np.array([0.3, 0.2, 0.1, 0.4])
    np.testing.assert_allclose(project_capped_simplex(p, upper), p, atol=1e-12)


def test_projection_is_the_euclidean_projection():
    """Against a generic constrained optimiser on points that violate the cap."""
    rng = np.random.default_rng(1)
    upper = upper_bounds(4, 0.3)
    for _ in range(25):
        p = softmax(rng.normal(scale=3.0, size=5))
        got = project_capped_simplex(p, upper)
        res = minimize(
            lambda w: 0.5 * ((w - p) ** 2).sum(), np.full(5, 0.2), method="SLSQP",
            bounds=[(0.0, u) for u in upper], constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"ftol": 1e-14, "maxiter": 500},
        )
        np.testing.assert_allclose(got, res.x, atol=1e-6)


def test_the_cap_binds_and_the_excess_goes_to_the_other_lines():
    """A near-certain bet on one asset is held at the cap. The Euclidean projection hands the
    excess to every line with slack in equal additive amounts (cash included, not cash alone)."""
    upper = upper_bounds(3, 0.35)
    p = softmax(12.0 * np.array([1.0, -1.0, -1.0, -1.0]))
    w = action_to_weights(np.array([1.0, -1.0, -1.0, -1.0]), upper, logit_scale=12.0)
    assert w[0] == pytest.approx(0.35, abs=1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    shift = (w[1:] - p[1:])
    np.testing.assert_allclose(shift, np.full(3, shift[0]), atol=1e-9)
    assert shift[0] == pytest.approx((p[0] - 0.35) / 3, abs=1e-9)


def test_the_zero_action_is_equal_weight_over_all_lines():
    w = action_to_weights(np.zeros(4), upper_bounds(3, 0.5), logit_scale=3.0)
    np.testing.assert_allclose(w, np.full(4, 0.25))


def test_infeasible_or_non_finite_input_fails_loudly():
    with pytest.raises(ValueError, match="infeasible"):
        project_capped_simplex(np.array([0.5, 0.5]), np.array([0.3, 0.3]))
    with pytest.raises(ValueError, match="non-finite"):
        action_to_weights(np.array([np.nan, 0.0, 0.0]), upper_bounds(2, 0.5), 3.0)


# --------------------------------------------------------------------------- #
# episode structure
# --------------------------------------------------------------------------- #
def test_episode_length_matches_the_calendar():
    """§7.5. One step per complete weekly holding period, counted independently."""
    data = make_data(80)
    idx = data.sessions
    fridays = [i for i, d in enumerate(idx) if d.dayofweek == 4]
    expected = 0
    for j, p in enumerate(fridays):
        exec_p = p + 1
        end_p = fridays[j + 1] + 1 if j + 1 < len(fridays) else len(idx) - 1
        if exec_p < len(idx) and min(end_p, len(idx) - 1) > exec_p:
            expected += 1
    env = make_env(data)
    df = run_episode(env, zero_action(env))
    assert len(df) == expected == data.n_decisions
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert all(d.dayofweek == 4 for d in df.index)


def test_eval_terminates_at_the_last_decision_and_train_truncates():
    data = make_data(200)
    ev = make_env(data, mode="eval")
    ev.reset()
    flags = [ev.step(np.zeros(3, "float32"))[2:4] for _ in range(data.n_decisions)]
    assert flags[-1] == (True, False) and all(f == (False, False) for f in flags[:-1])

    tr = make_env(data, mode="train", episode_length=10)
    tr.reset(seed=0)
    out = [tr.step(np.zeros(3, "float32")) for _ in range(10)]
    assert (out[-1][2], out[-1][3]) == (False, True), "a time limit inside the data is a truncation"
    assert not any(o[2] or o[3] for o in out[:-1])


def test_random_start_episodes_stay_inside_the_data_and_are_seeded():
    data = make_data(300)
    env = make_env(data, mode="train", episode_length=12)
    firsts = set()
    for seed in range(40):
        env.reset(seed=seed)
        firsts.add(env._first)
        assert 0 <= env._first and env._last < data.n_decisions and env._last - env._first == 11
    assert len(firsts) > 5, "random starts should actually vary"
    env.reset(seed=7)
    a = env._first
    env.reset(seed=7)
    assert env._first == a


def test_no_nan_observations():
    """§7.5: finite observations of the declared shape and dtype through a full episode."""
    data = make_data(120)
    env = make_env(data)
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 3).astype("float32"))
        assert np.isfinite(obs).all() and np.isfinite(r)
        assert obs.shape == env.observation_space.shape
        done = term or trunc


def test_the_gymnasium_env_checker_passes():
    from gymnasium.utils.env_checker import check_env

    check_env(make_env(make_data(60)), skip_render_check=True)


def test_portfolio_block_carries_weights_and_mean_turnover():
    """D-035: the observation ends with the n+1 drifted weights and the mean one-way turnover per step."""
    data = make_data(60, ret_a=0.0, ret_b=0.0)  # no drift, so weights are exactly the targets
    env = make_env(data)
    obs, _ = env.reset(seed=0)
    d = data.states.shape[1]
    np.testing.assert_allclose(obs[d:d + 3], [0.0, 0.0, 1.0])  # starts in cash
    assert obs[-1] == 0.0
    obs, *_ = env.step(np.zeros(3, "float32"))                  # equal weight 1/3 each
    np.testing.assert_allclose(obs[d:d + 3], [1 / 3] * 3, atol=1e-6)
    assert obs[-1] == pytest.approx(2 / 3, abs=1e-6)            # 0.5 * (1/3 + 1/3 + 2/3) from all cash
    obs, *_ = env.step(np.zeros(3, "float32"))                  # no change -> mean turnover halves
    assert obs[-1] == pytest.approx(1 / 3, abs=1e-6)


# --------------------------------------------------------------------------- #
# accounting against hand-computed answers
# --------------------------------------------------------------------------- #
def test_first_step_return_matches_a_hand_computation():
    """A +1%/day, B and cash flat, equal weight 1/3: one 5-day window grows (1.01^5 + 2) / 3."""
    data = make_data(60)
    env = make_env(data, cap=0.5)
    env.reset(seed=0)
    _, reward, *_ , info = env.step(np.zeros(3, "float32"))
    growth = (1.01**5 + 2.0) / 3.0
    assert info["gross_return"] == pytest.approx(growth - 1.0, abs=1e-12)
    assert reward == pytest.approx(np.log(growth), abs=1e-12)
    assert info["nav"] == pytest.approx(growth, abs=1e-12)


def test_buy_and_hold_reproduces_the_analytic_benchmark_return():
    """§7.5. A constant policy is a constant-mix portfolio; its NAV is the product of each
    window's weighted asset growth, computed here from the price path alone."""
    for ret_a, ret_b, cash in [(0.01, -0.004, 0.0), (0.0005, 0.002, 0.0001)]:
        data = make_data(120, ret_a=ret_a, ret_b=ret_b, cash=cash)
        action = np.array([0.4, -0.2, 0.1], "float32")
        env = make_env(data)
        w = action_to_weights(action, env._upper, env.logit_scale)
        df = run_episode(env, lambda obs: action)

        prices = np.cumprod(1.0 + data.line_returns, axis=0)  # close of each line, base 1 at session 0
        nav = 1.0
        for k in range(data.n_decisions):
            e, end = data.exec_pos[k], data.end_pos[k]
            nav *= float(w @ (prices[end] / prices[e]))
        assert df["nav"].iloc[-1] == pytest.approx(nav, rel=1e-12)


def test_the_env_reproduces_the_tier1_allocator_simulation():
    """With no costs the env's NAV equals the Tier 1 simulator's compounded gross return for the same weights."""
    from prism.probes import allocator as al

    data = make_data(150, ret_a=0.0, ret_b=0.0)
    rng = np.random.default_rng(3)
    data = EnvData.from_arrays(
        data.sessions, data.states, rng.normal(0.0004, 0.01, size=data.line_returns.shape) * [1, 1, 0] + [0, 0, 0.0001],
        data.sigma, lines=LINES, convention=CONVENTION,
    )
    action = np.array([0.7, -0.3, 0.2], "float32")
    env = make_env(data)
    w = action_to_weights(action, env._upper, env.logit_scale)
    df = run_episode(env, lambda obs: action)

    dates = data.sessions[data.decision_pos]
    weights = pd.DataFrame(np.tile(w, (len(dates), 1)), index=dates, columns=list(LINES))
    lr = pd.DataFrame(data.line_returns, index=data.sessions, columns=list(LINES))
    gross, _ = al.simulate(weights, lr, execution_lag=1)
    # simulate() also earns the first execution close's day return while still in cash; the env
    # starts earning after its first trade, so compare from the following session.
    assert df["nav"].iloc[-1] == pytest.approx(float((1.0 + gross.iloc[1:]).prod()), rel=1e-12)


def test_the_cost_enters_the_reward_exactly_once_per_trade():
    """No returns at all, so net = -cost. Step 0 trades from cash; step 1 holds and pays nothing."""
    data = make_data(60, ret_a=0.0, ret_b=0.0, sigma=0.01)
    cost = CostModel(per_side_bps=5.0, slippage_vol_coef=0.02)
    env = make_env(data, cost=cost)
    env.reset(seed=0)
    _, r0, *_, i0 = env.step(np.zeros(3, "float32"))
    expected = 2 * (1 / 3) * (5e-4 + 0.02 * 0.01)  # buying 1/3 of each risky asset; cash is free
    assert i0["cost"] == pytest.approx(expected, abs=1e-15)
    assert r0 == pytest.approx(np.log(1.0 - expected), abs=1e-12)
    _, r1, *_, i1 = env.step(np.zeros(3, "float32"))
    assert i1["cost"] == 0.0 and r1 == 0.0


def test_costs_lower_the_return_and_more_cost_lowers_it_more():
    data = make_data(120, ret_a=0.002, ret_b=-0.001)
    rng = np.random.default_rng(5)
    navs = {}
    for bps in (0.0, 5.0, 10.0, 20.0):
        env = make_env(data, cost=CostModel(5.0, 0.02).scaled(bps))
        actions = iter(rng.uniform(-1, 1, size=(data.n_decisions, 3)).astype("float32"))
        rng = np.random.default_rng(5)  # identical action sequence for every cost level
        seq = rng.uniform(-1, 1, size=(data.n_decisions, 3)).astype("float32")
        it = iter(seq)
        navs[bps] = run_episode(env, lambda obs: next(it))["nav"].iloc[-1]
    assert navs[0.0] > navs[5.0] > navs[10.0] > navs[20.0]


# --------------------------------------------------------------------------- #
# timing: the reward window does not overlap the observation window
# --------------------------------------------------------------------------- #
def test_reward_window_does_not_overlap_the_observation_window():
    """§11. A one-session return spike lands in exactly one step's reward: the step whose
    window (e_k, e_{k+1}] contains it. Spikes on the decision close d_k and on the execution
    close e_k belong to the PREVIOUS step, never to the decision made on d_k."""
    base_data = make_data(80, ret_a=0.0)
    base = run_episode(make_env(base_data), lambda obs: np.zeros(3, "float32"))["reward"].to_numpy()
    d = base_data
    for s in range(1, len(d.sessions)):
        spiked = make_data(80, ret_a=0.0, spike=(s, 0.10))
        got = run_episode(make_env(spiked), lambda obs: np.zeros(3, "float32"))["reward"].to_numpy()
        changed = np.flatnonzero(~np.isclose(got, base, atol=1e-12))
        owners = [k for k in range(d.n_decisions) if d.exec_pos[k] < s <= d.end_pos[k]]
        if owners:
            assert list(changed) == owners, f"spike at session {s}: changed {changed}, expected {owners}"
        else:
            assert len(changed) == 0, f"spike at session {s} lies in no window but changed {changed}"
    # The explicit statement of the property for every decision.
    assert (d.exec_pos > d.decision_pos).all()
    assert (d.exec_pos[1:] >= d.exec_pos[:-1] + 1).all() and (d.end_pos[:-1] == d.exec_pos[1:]).all()


def test_the_observation_at_a_decision_uses_only_data_up_to_that_close():
    """The state row and drifted weights seen at d_k are the ones at d_k, not later."""
    data = make_data(80)
    env = make_env(data)
    obs, info = env.reset(seed=0)
    d = data.states.shape[1]
    np.testing.assert_allclose(obs[:d], data.states[data.decision_pos[0]])
    for k in range(1, data.n_decisions):
        obs, *_ = env.step(np.zeros(3, "float32"))
        np.testing.assert_allclose(obs[:d], data.states[data.decision_pos[k]])
        if k >= 5:
            break


# --------------------------------------------------------------------------- #
# no look-ahead: poison everything after the current step and nothing may change
# --------------------------------------------------------------------------- #
class LeakyEnv(PortfolioEnv):
    """Deliberately broken: the observation reads a state row three sessions into the future."""

    def _observation(self, state_pos: int, weights: np.ndarray) -> np.ndarray:
        return super()._observation(min(state_pos + 3, len(self.data.sessions) - 1), weights)


def _poisoned(data: EnvData, *, keep_state: int, keep_returns: int, keep_sigma: int, seed: int = 99) -> EnvData:
    rng = np.random.default_rng(seed)
    states, rets, sig = data.states.copy(), data.line_returns.copy(), data.sigma.copy()
    states[keep_state + 1:] = rng.normal(scale=50.0, size=states[keep_state + 1:].shape)
    rets[keep_returns + 1:] = rng.normal(scale=0.2, size=rets[keep_returns + 1:].shape)
    sig[keep_sigma + 1:] = rng.uniform(0.0, 0.5, size=sig[keep_sigma + 1:].shape)
    return EnvData(
        sessions=data.sessions, states=states, state_columns=data.state_columns, lines=data.lines,
        line_returns=rets, sigma=sig, decision_pos=data.decision_pos, exec_pos=data.exec_pos,
        end_pos=data.end_pos,
    )


def _run_steps(env: PortfolioEnv, n_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=0)
    trace = [obs]
    for _ in range(n_steps):
        obs, r, *_ = env.step(rng.uniform(-1, 1, 3).astype("float32"))
        trace.append(np.append(obs, r))
    return trace


def _unchanged_by_the_future(env_cls, steps: int = 12) -> bool:
    rng = np.random.default_rng(11)
    clean = make_data(200)
    clean = EnvData.from_arrays(
        clean.sessions, clean.states, rng.normal(0.0003, 0.01, size=clean.line_returns.shape) * [1, 1, 0],
        np.full(clean.sigma.shape, 0.01), lines=LINES, convention=CONVENTION,
    )
    k = steps - 1  # the last step taken; it shows the next decision's observation
    bad = _poisoned(
        clean, keep_state=int(clean.decision_pos[k + 1]), keep_returns=int(clean.end_pos[k]),
        keep_sigma=int(clean.exec_pos[k]),
    )
    a = _run_steps(make_env(clean, cost=CostModel(5.0, 0.02), env_cls=env_cls), steps, seed=4)
    b = _run_steps(make_env(bad, cost=CostModel(5.0, 0.02), env_cls=env_cls), steps, seed=4)
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_env_never_indexes_beyond_the_current_step():
    """§11: "No look-ahead in the env". Replace every state, return and volatility row the first
    ``steps`` steps may legitimately read's successors with garbage: observations and rewards
    must be bit-identical."""
    assert _unchanged_by_the_future(PortfolioEnv)


def test_the_look_ahead_test_fails_on_an_injected_leak():
    """A look-ahead test that never fails is worthless (spec §7.1): the same check on an env
    that reads three sessions ahead must catch it."""
    assert not _unchanged_by_the_future(LeakyEnv)


# --------------------------------------------------------------------------- #
# rewards
# --------------------------------------------------------------------------- #
def _rec(net, daily=None, nav=1.0, peak=1.0, cost=0.0):
    daily = np.array([net]) if daily is None else np.asarray(daily, float)
    return StepRecord(gross_return=net, cost=cost, net_return=net, daily_returns=daily, nav=nav, peak_nav=peak)


def test_log_return_net_is_log_of_one_plus_net_return():
    assert LogReturnNet()(_rec(0.02)) == pytest.approx(np.log(1.02))


def test_dsr_matches_a_hand_computation_of_the_moody_saffell_formula():
    eta = 0.5
    r = DifferentialSharpe(eta)
    assert r(_rec(0.10)) == 0.0  # degenerate variance on the first step
    # After step 1: A = 0.05, B = 0.005.
    # Step 2 with R = 0.02: dA = -0.03, dB = 0.0004 - 0.005 = -0.0046
    a, b = 0.05, 0.005
    expected = (b * (0.02 - a) - 0.5 * a * (0.02**2 - b)) / (b - a * a) ** 1.5
    assert r(_rec(0.02)) == pytest.approx(expected, rel=1e-12)
    r.reset()
    assert r(_rec(0.10)) == 0.0, "reset must clear the moments"


def test_mv_penalty_and_drawdown_penalty_by_hand():
    daily = np.array([0.01, -0.01, 0.03])
    ss = ((daily - daily.mean()) ** 2).sum()
    net = float(np.prod(1 + daily) - 1)
    assert MeanVariancePenalty(2.0)(_rec(net, daily)) == pytest.approx(np.log1p(net) - 2.0 * ss)
    assert DrawdownPenalty(0.5)(_rec(net, nav=0.9, peak=1.2)) == pytest.approx(np.log1p(net) - 0.5 * 0.25)
    assert DrawdownPenalty(0.5)(_rec(net, nav=1.2, peak=1.2)) == pytest.approx(np.log1p(net))


def test_each_reward_variant_runs_through_the_env():
    data = make_data(100, ret_a=0.002)
    for reward in (LogReturnNet(), DifferentialSharpe(0.05), MeanVariancePenalty(1.0), DrawdownPenalty(0.02)):
        env = make_env(data, reward=reward, cost=CostModel(5.0, 0.02))
        rng = np.random.default_rng(0)
        df = run_episode(env, lambda obs: rng.uniform(-1, 1, 3).astype("float32"), seed=0)
        assert np.isfinite(df["reward"]).all()


# --------------------------------------------------------------------------- #
# data construction
# --------------------------------------------------------------------------- #
def test_line_returns_match_the_tier1_allocator_construction(cfg, features_b):
    """The cash line and the asset returns are built exactly as prepare_inputs builds them."""
    from prism.probes import allocator as al

    close = features_b.close
    risky = list(cfg.data.allocatable["B"])
    params = al.AllocatorParams(
        risky=tuple(risky), equity_sectors=tuple(cfg.data.universes["A"].equity_sectors), cap=0.35
    )
    dates = features_b.frame.index[100:103]  # inside Universe B, where every asset exists
    _, theirs = al.prepare_inputs(close, params, dates)
    ours = line_returns(close, risky)
    window = close.index[close.index.get_loc(dates[0]) - 5 : close.index.get_loc(dates[-1]) + 1]
    pd.testing.assert_frame_equal(
        ours.loc[window], theirs.loc[window, [*risky, "CASH"]], check_names=False
    )


def test_the_builder_refuses_the_holdout_and_respects_split_edges(cfg, features_b, plan):
    state = features_b.frame
    with pytest.raises(PermissionError, match="holdout"):
        build_env_data(cfg, state, features_b.close, "holdout")
    data = build_env_data(cfg, state, features_b.close, "train", plan=plan)
    assert data.sessions[0] >= plan["train"].effective_start
    assert data.sessions[-1] <= plan["train"].effective_end
    assert data.lines[-1] == "CASH" and len(data.lines) == len(cfg.data.allocatable["B"]) + 1
    # No reward window reaches beyond the split's last session.
    assert data.end_pos.max() <= len(data.sessions) - 1
    assert np.isfinite(data.states).all()


def test_the_builder_rejects_a_state_with_gaps(cfg, features_b, plan):
    state = features_b.frame
    holey = state.drop(state.index[500:503])
    with pytest.raises(ValueError, match="not consecutive"):
        build_env_data(cfg, holey, features_b.close, "train", plan=plan)


def test_a_trade_starts_from_the_weights_drifted_to_the_execution_close():
    """The drift between the decision close d_k and the execution close e_k belongs to the previous
    step, and the trade at e_k starts from the weights AFTER it. A +1%/day, B and cash flat, equal
    weight 1/3, 5 bps per side, no slippage. After the first 5-day window the weights are
    ((1.01^5)/3, 1/3, 1/3) / g with g = (1.01^5 + 2)/3, and re-targeting 1/3 each costs
    5 bps * (|1/3 - a| + |1/3 - b|). Starting from the weights seen at d_1 (one day earlier, 4 days
    of drift) would give a different, wrong number."""
    data = make_data(60)
    env = make_env(data, cost=CostModel(per_side_bps=5.0, slippage_vol_coef=0.0), cap=0.5)
    env.reset(seed=0)
    env.step(np.zeros(3, "float32"))
    obs, _, _, _, info = env.step(np.zeros(3, "float32"))
    g = (1.01**5 + 2.0) / 3.0
    a, b = (1.01**5 / 3.0) / g, (1.0 / 3.0) / g
    assert info["cost"] == pytest.approx(5e-4 * (abs(1 / 3 - a) + abs(1 / 3 - b)), rel=1e-10)
    wrong_a = (1.01**4 / 3.0) / ((1.01**4 + 2.0) / 3.0)
    assert abs(info["cost"] - 5e-4 * 2 * abs(1 / 3 - wrong_a)) > 1e-7


def test_a_holiday_friday_moves_the_decision_to_thursday():
    """The decision is the last session of each calendar week on or before Friday, so a Friday
    holiday (Good Friday) gives a Thursday decision; the env follows that schedule."""
    sessions = pd.bdate_range("2020-01-06", periods=60)
    holiday = pd.Timestamp("2020-02-14")                  # a Friday, treated as closed
    sessions = sessions.drop(holiday)
    n = len(sessions)
    r = np.zeros((n, 3))
    data = EnvData.from_arrays(sessions, np.zeros((n, 2)), r, np.full((n, 2), 0.01), lines=LINES, convention=CONVENTION)
    decided = data.sessions[data.decision_pos]
    assert pd.Timestamp("2020-02-13") in decided and holiday not in decided
    assert all(d.dayofweek in (3, 4) for d in decided)
    assert len(decided) == len(set(decided.isocalendar().week.astype(int) + 100 * decided.year))  # one per week
