"""Cached readers of ``dashboard/artifacts``. Pages read stored results through these and nowhere else.

Nothing here computes a statistic. The artifacts are verified against their manifest once per session
(``manifest``); a mismatch stops the app rather than showing a number that cannot be traced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from prism import dashboard_data as dd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / dd.ARTIFACTS


@st.cache_data(show_spinner=False)
def manifest() -> dict[str, Any]:
    return dd.verify_artifacts(ARTIFACTS)


@st.cache_data(show_spinner=False)
def facts() -> dict[str, Any]:
    return json.loads((ARTIFACTS / "facts.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def report_check() -> dict[str, Any]:
    return json.loads((ARTIFACTS / "report_check.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def results(window: str) -> dict[str, Any]:
    """One window's stored tables (``test`` or ``holdout``), as the final report reads them."""
    return dd.load_results(ARTIFACTS, window)


@st.cache_data(show_spinner=False)
def headline_tables(window: str) -> dict[str, pd.DataFrame]:
    """The final report's own rows for one window: gates, paired differences, variants and benchmarks, costs."""
    res = dd.load_results(ARTIFACTS, window)
    return {"gates": dd.gate_table(res), "differences": dd.difference_table(res), "variants": dd.variant_table(res),
            "costs": dd.cost_table(res), "episodes": dd.episode_table(res)}


@st.cache_data(show_spinner=False)
def tier1_gates() -> pd.DataFrame:
    return dd.tier1_table(json.loads((ARTIFACTS / "tier1" / "gates.json").read_text(encoding="utf-8")))


@st.cache_data(show_spinner=False)
def learning_curves() -> pd.DataFrame:
    """Mean over the final seeds, per variant and checkpoint, of the stored per-decision log net return."""
    curves = pd.read_csv(ARTIFACTS / "tier2" / "learning_curves.csv")
    return curves.groupby(["variant", "step"], as_index=False)[["train_mean_log_return", "val_mean_log_return"]].mean()


# --------------------------------------------------------------------------- regimes and latents (derived stage)
@st.cache_data(show_spinner=False)
def regimes() -> pd.DataFrame:
    """Daily P(Calm), P(Volatile) and the VIX-threshold state (NaN before the state files begin)."""
    return pd.read_parquet(ARTIFACTS / "regimes" / "daily.parquet")


@st.cache_data(show_spinner=False)
def regime_summary() -> dict[str, Any]:
    return json.loads((ARTIFACTS / "regimes" / "summary.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def spy() -> pd.Series:
    return pd.read_parquet(ARTIFACTS / "regimes" / "spy.parquet")["spy_index"]


@st.cache_data(show_spinner=False)
def hmm_folds() -> pd.DataFrame:
    """Per-fold HMM parameters: every fold to the test split's end, plus the holdout folds once the gated replay has run."""
    parts = [pd.read_csv(p, parse_dates=["fit_start", "fit_end", "apply_start", "apply_end"])
             for p in (ARTIFACTS / "regimes" / "hmm_folds_test.csv", ARTIFACTS / "regimes" / "hmm_folds_holdout.csv") if p.exists()]
    return pd.concat(parts, ignore_index=True)


@st.cache_data(show_spinner=False)
def k_selection() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "regimes" / "k_selection.csv")


@st.cache_data(show_spinner=False)
def selection_dwell() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "regimes" / "selection_dwell.csv")


@st.cache_data(show_spinner=False)
def latent_map() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "latents" / "pca.parquet")


@st.cache_data(show_spinner=False)
def latent_folds() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "latents" / "folds.csv", parse_dates=["start", "end", "fit_end"])


# --------------------------------------------------------------------------- returns and weights
@st.cache_data(show_spinner=False)
def eval_daily(window: str) -> pd.DataFrame:
    """The stored daily net returns of one window: every seed, variant and benchmark at every cost level."""
    return pd.read_parquet(ARTIFACTS / window / "eval_daily.parquet")


@st.cache_data(show_spinner=False)
def seed_dsr(window: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / window / "seed_dsr.csv")


def weight_windows() -> list[str]:
    """Windows whose weekly weights have been recorded. The holdout appears once the gated replay has been run."""
    return [w for w in ("test", "holdout") if (ARTIFACTS / "weights" / f"{w}_agents.parquet").exists()]


@st.cache_data(show_spinner=False)
def weights(kind: str) -> pd.DataFrame:
    """Weekly weights of the agents (``kind="agents"``) or the benchmarks, over every recorded window."""
    parts = []
    for w in weight_windows():
        f = pd.read_parquet(ARTIFACTS / "weights" / f"{w}_{kind}.parquet")
        f.insert(0, "window", w)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


@st.cache_data(show_spinner=False)
def live_models_manifest() -> dict[str, Any] | None:
    path = ARTIFACTS / "live" / "models_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def seed_metrics(window: str) -> pd.DataFrame:
    """The stored per-seed metrics of every variant in one window (one row per variant and seed)."""
    parts = []
    for v in facts()["agents"]["variants"]:
        f = pd.read_csv(ARTIFACTS / window / "tables" / f"seed_metrics_{v}.csv", index_col=0).rename_axis("seed").reset_index()
        f.insert(0, "variant", v)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


@st.cache_data(show_spinner=False)
def benchmark_metrics(window: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / window / "tables" / "benchmark_metrics.csv", index_col=0)
