"""PRISM dashboard: the frame. DASHBOARD.md §3, §7.

    make dashboard        # = streamlit run dashboard/app.py, from the repository root

This file sets the page up, builds the navigation from ``components.registry`` and draws the header that every
page shares. Pages read ``dashboard/artifacts`` through ``components.data``; nothing in milestone D1 touches the
network, a model or a frozen data file.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

from components import data, ui  # noqa: E402
from components.registry import PAGES  # noqa: E402
from prism.dashboard_data import ArtifactError  # noqa: E402

st.set_page_config(page_title="PRISM", page_icon=":material/analytics:", layout="wide")

page = st.navigation([st.Page(p.script, title=p.title, icon=p.icon, url_path=p.key, default=i == 0) for i, p in enumerate(PAGES)])

with st.sidebar:
    st.markdown("### PRISM")
    st.caption("Probabilistic Regime-Informed Systematic Management. A pre-registered test of whether market-regime "
               "conditioning helps a reinforcement-learning portfolio allocator. It did not.")

try:
    data.manifest()
except (ArtifactError, FileNotFoundError) as exc:
    st.error(f"The dashboard's stored results could not be verified, so nothing is shown. {exc}")
    st.stop()

ui.frame_header()
page.run()
