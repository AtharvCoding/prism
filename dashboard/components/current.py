"""The most recent agent weights available, where they came from, and the refresh control. DASHBOARD.md §4, §6.

Two sources, never mixed and always labelled:

* **live**: the last successful :func:`prism.live.refresh` (fresh prices through the frozen models), read from its cache.
  It exists only once the gated holdout replay has written the frozen last-fold models.
* **recorded**: the last weekly decision in the replay artifacts (the holdout's if it has been recorded, else the test
  split's). Nothing is fetched and no model runs.

If a refresh fails, the previous result stays on screen with a red badge. Nothing here invents a number.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from components import data, ui
from prism import live

BENCHMARKS = {"Equal weight": "EqualWeight", "60/40": "SixtyForty", "Risk parity": "RiskParity"}


@st.cache_resource(show_spinner=False)
def _config():  # noqa: ANN202
    from prism.config import load_config

    return load_config(data.ROOT / "configs" / "base.yaml")


@st.cache_resource(show_spinner="Loading the 40 frozen agents…")
def policies() -> dict[tuple[str, int], Any] | None:
    """The frozen agents as deterministic policies; ``None`` where the checkpoints are not on this machine."""
    try:
        return live.load_policies(_config(), data.ROOT)
    except Exception:  # noqa: BLE001 - absent or altered checkpoints: the pages say so instead of crashing
        return None


def _recorded() -> dict[str, Any]:
    window = data.weight_windows()[-1]
    agents, bench = data.weights("agents"), data.weights("benchmarks")
    agents = agents[agents.window == window]
    last = agents.decision_date.max()
    seen = json.loads((data.ARTIFACTS / "weights" / f"{window}_last_observations.json").read_text(encoding="utf-8"))
    b = bench[(bench.window == window) & (bench.decision_date == last) & bench.benchmark.isin(BENCHMARKS.values())]
    v4 = seen["columns"]["V4"]
    obs = seen["observations"]["V4|s0"]
    return {
        "source": "recorded", "status": "stored", "window": window, "as_of": last.date(), "decision_date": last.date(), "is_preview": False,
        "decision": agents[agents.decision_date == last].drop(columns=["window", "decision_date", "execution_date", "cost"]),
        "preview": None, "benchmarks": b.drop(columns=["window", "decision_date", "turnover", "cost"]),
        "columns": seen["columns"], "observations": seen["observations"], "models": None,
        "p_volatile": float(obs[v4.index("state_1")]), "latent": [x for c, x in zip(v4, obs) if c.startswith("latent_")],
        "p_history": data.regimes().p_volatile.loc[:last].iloc[-260:],
    }


def view() -> dict[str, Any]:
    """What the weight pages show right now. See the module docstring."""
    manifest = data.live_models_manifest()
    cache = live.read_cache(data.ROOT / live.CACHE_DIR) if manifest is not None else None
    if cache is None:
        return {**_recorded(), "can_refresh": manifest is not None}
    history = pd.Series(cache["p_volatile"])
    history.index = pd.to_datetime(history.index)
    return {
        "source": "live", "can_refresh": True, "status": st.session_state.get("live_status", "cached"), "window": cache["window"],
        "as_of": date.fromisoformat(cache["as_of"]), "decision_date": date.fromisoformat(cache["decision_date"]), "is_preview": cache["is_preview"],
        "decision": cache["decision"], "preview": cache["preview"], "benchmarks": cache["benchmarks"], "columns": cache["columns"],
        "observations": cache["observations"], "models": cache["models"], "p_volatile": float(history.iloc[-1]) if not cache["is_preview"]
        else float(history.loc[: cache["decision_date"]].iloc[-1]), "latent": cache["latent"], "p_history": history,
        "message": st.session_state.get("live_message"),
    }


def refresh_now() -> None:
    """Run one refresh. On failure keep the cached result and record why, for the badge."""
    try:
        window = data.weight_windows()[-1]
        recorded = pd.read_parquet(data.ARTIFACTS / "weights" / f"{window}_agents.parquet") if window == "holdout" else None
        live.refresh(_config(), data.ROOT, now=pd.Timestamp.now(tz="UTC"), expected_manifest=data.live_models_manifest(), policies=policies(),
                     recorded=recorded)
        st.session_state["live_status"], st.session_state["live_message"] = "live", None
    except live.SeamError as exc:
        st.session_state["live_status"], st.session_state["live_message"] = "seam-check-failed", str(exc)
    except Exception as exc:  # noqa: BLE001 - network down, vendor error, missing file: show the last good result as stale
        st.session_state["live_status"], st.session_state["live_message"] = "stale", f"{type(exc).__name__}: {exc}"
    st.session_state["live_refreshed_at"] = pd.Timestamp.now(tz="UTC")


def controls(v: dict[str, Any], key: str) -> None:
    """The freshness badge and the refresh button, or the reason there is no live view yet."""
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        models = v["models"]
        ui.freshness_badge(v["status"], v["as_of"], today=date.today() if v["source"] == "live" else None,
                           models_refit=date.fromisoformat(models["hmm_fit_end"]) if models else None)
        if v["source"] == "recorded":
            what = "holdout" if v["window"] == "holdout" else "test split"
            st.caption(f"Not live. These are the weights at the last decision recorded from the {what} replay ({v['decision_date']:%d %b %Y}). "
                       + ("Press Refresh to fetch the latest closes and run them through the frozen models." if v["can_refresh"] else
                          "A live view needs the frozen last-fold models, which are written by the gated holdout replay; that has "
                          "not been run in this copy."))
            if st.session_state.get("live_message"):
                st.caption(f":red[{st.session_state['live_message']}]")
        elif v.get("message"):
            st.caption(f":red[{v['message']}] The last good result is shown.")
    with right:
        last = st.session_state.get("live_refreshed_at")
        waiting = last is not None and (pd.Timestamp.now(tz="UTC") - last).total_seconds() < 60
        st.button("Refresh", icon=":material/refresh:", key=f"refresh_{key}", width="stretch", on_click=refresh_now,
                  disabled=not v["can_refresh"] or waiting or policies() is None,
                  help="Fetches the latest daily closes and runs them through the frozen models. At most once a minute.")


def strategy_weights(v: dict[str, Any], strategy: str, lines: list[str], *, preview: bool = False) -> dict[str, pd.Series | None]:
    """Mean, lowest-seed and highest-seed weight per holding for one strategy in the current view."""
    if strategy in BENCHMARKS:
        row = v["benchmarks"][v["benchmarks"].benchmark == BENCHMARKS[strategy]].iloc[0]
        w = pd.Series({c[2:]: float(row[c]) for c in v["benchmarks"].columns if c.startswith("w_")})
        return {"mean": w[w.index.isin([*lines, "SPY"])], "min": None, "max": None, "turnover": None, "traded": None}
    frame = v["preview"] if preview else v["decision"]
    sub = frame[frame.variant == strategy.split()[0]]
    ens = live.ensemble(sub).loc[lines]
    traded = sum((sub[f"w_{c}"] - sub[f"p_{c}"]).abs() for c in lines[:-1]).mean()      # risky legs only: what the cost is charged on
    return {"mean": ens["mean"], "min": ens["min"], "max": ens["max"], "turnover": float(sub.turnover.mean()), "traded": float(traded)}
