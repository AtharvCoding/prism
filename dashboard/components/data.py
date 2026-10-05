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
