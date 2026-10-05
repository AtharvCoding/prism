"""Page furniture: the frame header, takeaways, "How to read this", the permanent caption, badges, the verified footer.

The wording rules of DASHBOARD.md §0 and §6 live here so that every page gets them the same way.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from components import data
from components.registry import BY_KEY

#: DASHBOARD.md §0: the framing sentence for Home and the Verdict page.
FRAMING = "We built the full system end to end, and tested it rigorously enough to know what it does and doesn't do."
#: DASHBOARD.md §0: on every page that shows live or historical weights.
WEIGHTS_CAPTION = ("Demonstration of the frozen system. Not a recommendation; the agents did not beat these benchmarks "
                   "after costs.")
HOW_TO_READ = "How to read this"
#: Freshness states of a live element (DASHBOARD.md §6) and of the stored results: label, st.badge colour, icon.
FRESHNESS = {
    "stored": ("Stored results", "gray", ":material/inventory_2:"),
    "live": ("Live", "green", ":material/sensors:"),
    "cached": ("Cached", "blue", ":material/cached:"),
    "stale": ("Stale", "red", ":material/warning:"),
    "seam-check-failed": ("Data seam check failed", "red", ":material/error:"),
}


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def pct_range(lo_hi: list[float], nd: int = 0) -> str:
    return f"{100 * lo_hi[0]:.{nd}f}–{100 * lo_hi[1]:.{nd}f}%"


def takeaway_values(facts: dict[str, Any]) -> dict[str, str]:
    """The numbers the takeaway templates quote, across both evaluation windows."""
    w = facts["windows"]
    both = lambda k: [min(w["test"][k][0], w["holdout"][k][0]), max(w["test"][k][1], w["holdout"][k][1])]  # noqa: E731
    agree = data.regime_summary()["agreement"]["all"]
    return {
        "regime_agreement": pct(agree["agreement"]),
        "volatile_overlap": pct(agree["both_volatile"] / (agree["both_volatile"] + agree["hmm_only"])),
        "encoder_window": str(facts["models"]["encoder"]["window"]), "latent_dim": str(facts["models"]["encoder"]["latent_dim"]),
        "agents_turnover": pct_range(both("turnover_agents")),
        "bench_turnover": pct_range(both("turnover_benchmarks"), 1),
        "passed_test": str(w["test"]["comparisons_passing"]),
        "passed_holdout": str(w["holdout"]["comparisons_passing"]),
    }


def takeaway_text(key: str, facts: dict[str, Any] | None = None) -> str:
    return BY_KEY[key].takeaway.format(**takeaway_values(facts if facts is not None else data.facts()))


def page_header(key: str) -> dict[str, Any]:
    """Title and the one-line takeaway. Returns the facts, which most pages need next."""
    facts = data.facts()
    page = BY_KEY[key]
    st.title(page.title)
    st.markdown(f"**{takeaway_text(key, facts)}**")
    if page.shows_weights:
        weights_caption()
    return facts


def table(frame, *, index: bool = False) -> None:  # noqa: ANN001
    """A static table: every row visible, long text wrapped. Used for the stored tables, which are short."""
    st.table(frame, border="horizontal", hide_index=not index)


def how_to_read(markdown: str) -> None:
    with st.expander(HOW_TO_READ, icon=":material/menu_book:"):
        st.markdown(markdown)


def weights_caption() -> None:
    st.caption(f"*{WEIGHTS_CAPTION}*")


def freshness_label(status: str, as_of: date, *, today: date | None = None, models_refit: date | None = None) -> str:
    """The badge text for a live element: state, as-of date, data age, and the age of the frozen models."""
    name = FRESHNESS[status][0]
    parts = [name, f"as of {as_of.isoformat()}"]
    if today is not None:
        age = (today - as_of).days
        parts.append("today" if age == 0 else f"{age} day{'s' if age != 1 else ''} old")
    if models_refit is not None:
        parts.append(f"models last refit {models_refit.isoformat()}")
    return " · ".join(parts)


def freshness_badge(status: str, as_of: date, *, today: date | None = None, models_refit: date | None = None) -> None:
    _, color, icon = FRESHNESS[status]
    st.badge(freshness_label(status, as_of, today=today, models_refit=models_refit), color=color, icon=icon)


@st.dialog("What this is, and what it isn't")
def about_dialog() -> None:
    f = data.facts()
    h = f["windows"]["holdout"]
    st.markdown(
        f"""
**What this is.** A walk through PRISM, a research project that asked whether an explicit market-regime
model (an HMM) helps a reinforcement-learning portfolio allocator beyond what a learned summary of recent
history (an LSTM) already gives it. The design, the comparisons and the pass rule were written down and
committed before the results were seen. These pages show those stored results and how the system works.

**What it found.** Nothing detectable. None of the three comparisons passed, on the test split or on the
holdout ({h['first']} to {h['last']}), which was evaluated exactly once. The agents did not beat an
equal-weight, a 60/40 or a risk-parity portfolio after costs.

**What it isn't.** Not a recommendation, not a forecast and not a trading system. Where weights appear,
they demonstrate a frozen system; nothing here says what anyone should hold.

**How the numbers got here.** Every figure is read from the tables behind `reports/final_report.md`.
The dashboard computes no new statistic for the verdict and tunes nothing.
"""
    )


def frame_header() -> None:
    """Above every page: what kind of numbers these are, the holdout's status, and the about dialog."""
    f = data.facts()
    evaluated = f["windows"]["holdout"]["evaluated_utc"][:10]
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        name, color, icon = FRESHNESS["stored"]
        st.markdown(
            f":{color}-badge[{icon} {name}: nothing on this page is live] "
            f":orange-badge[:material/lock_open: Holdout spent: evaluated once, {evaluated}]"
        )
    with right:
        if st.button("What this is / isn't", icon=":material/info:", key="about", width="stretch"):
            about_dialog()


def verified_footer() -> None:
    """DASHBOARD.md §6: the numbers are the stored ones, and that was checked when the artifacts were built."""
    check, manifest = data.report_check(), data.manifest()
    st.divider()
    st.caption(
        f":material/verified: All numbers on this page are read from the source tables of `{check['report']}`. "
        f"At build time {check['rows_checked']} table rows built from them were matched against that report "
        f"(report {check['report_sha256'][:12]}, artifacts {manifest['hash'][:12]})."
    )
