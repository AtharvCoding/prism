"""The dashboard precompute's model stages: weekly weights by deterministic replay, fold parameters, frozen live models.

DASHBOARD.md §5.3, §5.4; ``reports/tables/preregistration_holdout.md`` Amendment 1; DECISIONS.md D-044, D-047.

Three things were never stored and are produced here, each behind a reproduce-the-stored-result check:

* **Weekly weights** (:func:`replay_window`): the 40 frozen agents and the six benchmarks are run again through the
  environment on a window that was already evaluated, to record target weights, the weights drifted to the execution
  close, turnover and cost. Every daily net return series, at every cost level, must equal its stored column to 1e-9,
  or the function raises and nothing is written. No statistic is computed.
* **Per-fold HMM parameters** (:func:`hmm_fold_table`) from a re-run of the walk-forward, which must reproduce the
  stored states to 1e-9 (:func:`walkforward`).
* **The last fold's models** for the live view (:func:`prism.live.save_models`), verified on that fold's apply window.

The same code runs on the test window (not gated: it was already exposed) and on the holdout window (gated: two keys,
typed by hand, logged). The test-window run is therefore the rehearsal of the holdout run. Nothing here writes into
``data/processed``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.dashboard_data import PROCESSED, ArtifactError, FrozenSourceError
from prism.utils.hashing import sha256_file

__all__ = ["REPLAY_ATOL", "PREREG_HOLDOUT", "AMENDMENT_MARK", "verify_agents", "replay_window", "walkforward", "stored_states",
           "hmm_fold_table", "require_reproduced", "ReplayError"]

REPLAY_ATOL = 1e-9
PREREG_HOLDOUT = "reports/tables/preregistration_holdout.md"
AMENDMENT_MARK = "### Amendment 1"
VARIANT_FILES = ("V1", "V2", "V4", "C4")


class ReplayError(ArtifactError):
    """A replay did not reproduce the stored result; nothing may be recorded from it."""


def verify_agents(root: Path) -> dict[str, str]:
    """Every file the holdout pre-registration pins (states, frozen configs, 40 checkpoints) must match its SHA-256."""
    from prism.utils.prereg import preregistered_hashes

    listed = preregistered_hashes(root, PREREG_HOLDOUT)
    if len(listed) < 45:
        raise FrozenSourceError(f"{PREREG_HOLDOUT} pins only {len(listed)} files")
    bad = [n for n, d in listed.items() if sha256_file(Path(root) / PROCESSED / n) != d]
    if bad:
        raise FrozenSourceError(f"files differ from the pre-registered SHA-256: {bad[:5]}")
    return listed


def stored_states(root: Path, window: str) -> dict[str, pd.DataFrame]:
    """The state frames the stored evaluation of ``window`` ran on: the Tier 2 files, or the holdout run's extension."""
    processed = Path(root) / PROCESSED
    if window == "holdout":
        frame = pd.read_parquet(processed / "holdout" / "states_extended.parquet")
        return {v: frame[v].copy() for v in VARIANT_FILES}
    return {v: pd.read_parquet(processed / "states" / f"{v}.parquet") for v in VARIANT_FILES}


def _compare(name: str, new: pd.Series, stored: pd.Series, worst: dict[str, float]) -> None:
    if not new.index.equals(stored.index):
        raise ReplayError(f"{name}: the replayed series is not on the stored index")
    worst[name] = float(np.abs(new.to_numpy() - stored.to_numpy()).max())


def require_reproduced(worst: dict[str, float], expected: list[str], atol: float) -> None:
    """Raise unless every stored series was replayed and none differs by more than ``atol`` (NaN counts as a difference)."""
    if set(worst) != set(expected):
        raise ReplayError("the replay did not cover exactly the stored series")
    over = {k: d for k, d in worst.items() if not d <= atol}
    if over:
        k = max(over, key=lambda name: float("inf") if over[name] != over[name] else over[name])
        raise ReplayError(f"REPLAY FAILED: {len(over)} of {len(worst)} daily series differ from the stored evaluation "
                          f"by more than {atol:g} (worst {k}: {over[k]:.3e}); nothing is recorded")


def replay_window(cfg, plan, split: str, close: pd.DataFrame, states: dict[str, pd.DataFrame] | None,  # noqa: ANN001
                  stored_daily: pd.DataFrame, stored_meta: dict[str, Any], *, final_holdout: bool = False,
                  atol: float = REPLAY_ATOL) -> dict[str, Any]:
    """Replay every frozen agent and benchmark on ``split``; return their weekly weights and the reproduce check.

    ``states=None`` reads the stored Tier 2 state files (the test window); the holdout passes the stored extended
    states. ``plan`` is the Tier 2 plan whose ``runs`` hold the checkpoints. Raises :class:`ReplayError` if any
    daily series, the mean weights or the mean turnover differ from what the single stored evaluation recorded.
    """
    from prism.agents import tier2 as t2
    from prism.agents.data import variant_env_data
    from prism.agents.evaluate import benchmark_weight_frames, run_benchmark, run_policy
    from prism.agents.jobs import job_dir
    from prism.agents.sac import greedy_policy, load_agent
    from prism.env.costs import CostModel
    from prism.env.portfolio_env import make_env
    from prism.splits import build_split_plan

    frozen, split_plan, base = t2.load_frozen(plan), build_split_plan(cfg), CostModel.from_config(cfg)
    worst: dict[str, float] = {}
    agent_rows, bench_rows = [], []
    last_obs: dict[str, list[float]] = {}
    obs_columns: dict[str, list[str]] = {}
    for v in plan.variants:
        data = variant_env_data(cfg, v, close, split, plan=split_plan, state=None if states is None else states[v],
                                final_holdout=final_holdout)
        decisions = data.sessions[data.decision_pos]
        executions = data.sessions[data.exec_pos]
        for s in plan.final_seeds:
            policy = greedy_policy(load_agent(job_dir(plan.runs, "final", v, frozen[v].cfg_id, s) / "best.zip"))
            for bps in t2.COST_LEVELS:
                env = make_env(cfg, data, mode="eval", cost_model=base.scaled(bps))
                pre_trade: list[np.ndarray] = []
                seen: list[np.ndarray] = []

                def recording(obs: np.ndarray, env=env, pre_trade=pre_trade, seen=seen) -> np.ndarray:
                    pre_trade.append(env.drifted_weights)       # what the trade at the execution close starts from
                    seen[:] = [obs.copy()]                      # keep only the latest: the final decision's observation
                    return policy(obs)

                res = run_policy(env, recording)
                _compare(f"{v}|s{s}|{bps:g}", res["daily"], stored_daily[f"{v}|s{s}|{bps:g}"], worst)
                if bps != t2.HEADLINE_BPS:
                    continue
                w = res["weights"]
                if not w.index.equals(decisions[: len(w)]) or len(w) != len(decisions):
                    raise ReplayError(f"{v} seed {s}: the episode did not cover every decision")
                frame = w.add_prefix("w_")
                frame[[f"p_{c}" for c in w.columns]] = np.vstack(pre_trade)
                frame.insert(0, "cost", res["cost"])
                frame.insert(0, "turnover", res["turnover"])
                frame.insert(0, "execution_date", executions)
                frame.insert(0, "seed", s)
                frame.insert(0, "variant", v)
                agent_rows.append(frame.rename_axis("decision_date").reset_index())
                key = f"{v}|s{s}"
                last_obs[key] = [float(x) for x in seen[0]]
                obs_columns[v] = [*data.state_columns, *(f"weight_{c}" for c in data.lines), "mean_turnover"]
                want = pd.Series(stored_meta["weights"][key])
                if float(np.abs(w.mean().round(6)[want.index] - want).max()) > 1e-6 + 1e-12:
                    raise ReplayError(f"{key}: mean weights differ from those the stored evaluation recorded")
                if abs(float(res["turnover"].mean()) - stored_meta["turnover"][key]) > atol:
                    raise ReplayError(f"{key}: mean turnover differs from the stored evaluation")

    risky = [*cfg.data.allocatable[cfg.env.universe], "SPY"]
    bdata = variant_env_data(cfg, "V1", close, split, plan=split_plan, risky=risky,
                             state=None if states is None else states["V1"], final_holdout=final_holdout)
    frames = benchmark_weight_frames(cfg, close, bdata)
    for name in t2.BENCHMARKS:
        for bps in t2.COST_LEVELS:
            res = run_benchmark(cfg, bdata, frames, name, base.scaled(bps))
            _compare(f"BM|{name}|{bps:g}", res["daily"], stored_daily[f"BM|{name}|{bps:g}"], worst)
            if bps != t2.HEADLINE_BPS:
                continue
            frame = res["weights"].add_prefix("w_")
            frame.insert(0, "cost", res["cost"])
            frame.insert(0, "turnover", res["turnover"])
            frame.insert(0, "benchmark", name)
            bench_rows.append(frame.rename_axis("decision_date").reset_index())
            if abs(float(res["turnover"].mean()) - stored_meta["turnover"][f"BM|{name}"]) > atol:
                raise ReplayError(f"BM|{name}: mean turnover differs from the stored evaluation")

    require_reproduced(worst, list(stored_daily.columns), atol)
    agents = pd.concat(agent_rows, ignore_index=True)
    check = {
        "split": split, "atol": atol, "series": len(worst), "max_abs_diff": max(worst.values()),
        "first_return_day": str(stored_daily.index[0].date()), "last_return_day": str(stored_daily.index[-1].date()),
        "decisions": int(agents.decision_date.nunique()), "first_decision": str(agents.decision_date.min().date()),
        "last_decision": str(agents.decision_date.max().date()), "agents": int(agents.groupby(["variant", "seed"]).ngroups),
        "ok": True,
    }
    observations = {"decision_date": check["last_decision"], "columns": obs_columns, "observations": last_obs}
    return {"agents": agents, "benchmarks": pd.concat(bench_rows, ignore_index=True), "check": check, "last_observations": observations}


# --------------------------------------------------------------------------- #
# the walk-forward, re-run for its fold objects
# --------------------------------------------------------------------------- #
def walkforward(cfg, end: pd.Timestamp, raw: pd.DataFrame, reference: dict[str, pd.DataFrame], *, atol: float = REPLAY_ATOL):  # noqa: ANN001, ANN201
    """Re-run the state pipeline through ``end`` and require it to reproduce ``reference`` on every row up to ``end``.

    Returns ``(ExtendedStates, check)``. ``reference`` is :func:`stored_states` of the window that ends at ``end``.
    """
    from prism import holdout as H

    ext = H.build_extended_states(cfg, end, raw=raw)
    check: dict[str, Any] = {"end": str(pd.Timestamp(end).date()), "atol": atol, "variants": {}}
    ok = True
    for v, ref in reference.items():
        new = ext.states[v]
        ref = ref.loc[:end]
        same = list(ref.columns) == list(new.columns) and ref.index.equals(new.index)
        diff = float(np.abs(ref.to_numpy(dtype="float64") - new.to_numpy(dtype="float64")).max()) if same else float("inf")
        check["variants"][v] = {"same_index_and_columns": bool(same), "rows": int(len(new)), "max_abs_diff": diff}
        ok &= bool(same and diff <= atol)
    check["ok"] = bool(ok)
    if not ok:
        raise ReplayError(f"the re-run walk-forward does not reproduce the stored states: {json.dumps(check['variants'])}")
    return ext, check


def hmm_fold_table(ext, *, first_fold: int = 0) -> pd.DataFrame:  # noqa: ANN001
    """One row per HMM fold: its dates and its parameters in return units (states already in canonical order).

    The HMM is fit on returns standardised by the fold's own scaler, so a state's mean and volatility are mapped
    back with that scaler. ``dwell_i`` is the expected stay in state ``i``, ``1 / (1 - p_ii)`` sessions.
    """
    rows = []
    for f in ext.hmm.folds:
        if f.fold.index < first_fold:
            continue
        p = f.scaler._params()
        centre, scale = float(p["centre"].iloc[0]), float(p["scale"].iloc[0])
        k = f.model.n_components
        trans = np.asarray(f.model.transmat_, dtype="float64")
        variances = np.array([np.asarray(f.model.covars_[i]).reshape(-1)[0] for i in range(k)])
        row: dict[str, Any] = {
            "fold": int(f.fold.index), "fit_start": f.fold.fit_start.date(), "fit_end": f.fold.fit_end.date(),
            "apply_start": f.fold.apply_start.date(), "apply_end": f.fold.apply_end.date(),
            "restarts": int(f.n_restarts_used), "restarts_degenerate": int(f.n_restarts_degenerate),
        }
        for i in range(k):
            row[f"mean_{i}"] = float(f.model.means_[i, 0]) * scale + centre
            row[f"vol_{i}"] = float(np.sqrt(variances[i])) * scale
            row[f"stay_{i}"] = float(trans[i, i])
            row[f"dwell_{i}"] = float(1.0 / (1.0 - trans[i, i])) if trans[i, i] < 1 else float("inf")
            for j in range(k):
                row[f"p_{i}{j}"] = float(trans[i, j])
        rows.append(row)
    return pd.DataFrame(rows)
