"""Page 11, Verdict and what's next. DASHBOARD.md §7.11."""

from __future__ import annotations

import streamlit as st

from components import data, ui

facts = ui.page_header("verdict")
agents, windows, stats, u = facts["agents"], facts["windows"], facts["statistics"], facts["universe"]
models = facts["models"]
diff = {w: data.headline_tables(w)["differences"] for w in ("test", "holdout")}
gates = {w: data.headline_tables(w)["gates"].set_index("comparison") for w in ("test", "holdout")}

st.markdown(f"> *{ui.FRAMING}*")

# --------------------------------------------------------------------------- the verdict
st.subheader("The null result, replicated on unseen data")
ASKS = {("V4", "V2"): "Does the HMM add anything beyond the LSTM summary? (the research question)",
        ("V4", "C4"): "Is the HMM better than a simple VIX threshold?",
        ("V2", "V1"): "Does the LSTM summary add anything to the raw features?"}


def _sharpe(window: str, x: str, y: str) -> str:
    d = diff[window]
    return d[(d.candidate == x) & (d.control == y) & (d.metric == "Sharpe")].iloc[0]["difference [95% CI]"]


for x, y in (tuple(p) for p in agents["comparisons"]):
    name = f"{x} vs {y}"
    with st.container(border=True):
        st.markdown(f"**{name}** · {ASKS[(x, y)]}")
        a, b = st.columns(2)
        a.markdown(f"Test split: **{gates['test'].loc[name, 'paired rule']}** · Sharpe difference {_sharpe('test', x, y)}")
        b.markdown(f"Holdout: **{gates['holdout'].loc[name, 'paired rule']}** · Sharpe difference {_sharpe('holdout', x, y)}")
counts = {(g.loc[n, "favourable"], g.loc[n, "adverse"]) for g in gates.values() for n in g.index}
st.caption(f"Differences are candidate minus control with their {stats['ci_level']:.0%} intervals."
           + (f" On both windows every comparison was favourable on {next(iter(counts))[0]} measures and adverse on "
              f"{next(iter(counts))[1]}: not better, not worse, indistinguishable." if len(counts) == 1 else
              " The favourable and adverse counts for each comparison are on the Results page."))

# --------------------------------------------------------------------------- shows / does not show
st.subheader("What this does and doesn't show")
yes, no = st.columns(2)
with yes.container(border=True):
    st.markdown(
        """
**It shows**

* No detectable benefit from the HMM's regime estimates for this agent, in this setup.
* No detectable benefit from the trained LSTM summary either.
* Simple benchmarks (equal weight, 60/40, risk parity) are hard to beat once trading costs are counted.
* The same answer on data nobody had looked at, under rules fixed in advance.
"""
    )
with no.container(border=True):
    st.markdown(
        """
**It does not show**

* That regime information is worthless in general.
* That reinforcement learning cannot work for allocation.
* That a different algorithm, universe, reward or trading frequency would give the same answer.
* Anything about what to hold. Nothing here is a recommendation.
"""
    )

# --------------------------------------------------------------------------- limitations
st.subheader("Limitations")
hw_t, hw_h = windows["test"]["sharpe_ci_half_width"], windows["holdout"]["sharpe_ci_half_width"]
st.markdown(
    f"""
* **Statistical power.** The intervals on a Sharpe-ratio difference are about ±{hw_t[0]:.2f} to ±{hw_t[1]:.2f} on the test
  split and ±{hw_h[0]:.2f} to ±{hw_h[1]:.2f} on the shorter holdout. An effect smaller than that could exist and go unseen.
* **A narrow universe.** {len(u['sectors'])} US sector funds, {len(u['bonds'])} Treasury funds and gold.
* **One algorithm, one frequency.** SAC only, weekly decisions only.
* **Little training data.** About {agents['train_decisions']} weekly decisions containing one major crisis; the agents
  overfit heavily (see the learning curves on the Agent page).
* **A simple regime model.** {models['hmm']['k']} states, estimated from S&P 500 daily returns alone.
* **Two evaluation windows.** One had been seen beforehand (exploratory) and the other is now spent.
"""
)

# --------------------------------------------------------------------------- what next
st.subheader("What would come next")
st.markdown(
    """
* An agent that is penalised for turnover, since these agents trade heavily and pay for it.
* A broader universe and a longer history.
* Other algorithms and decision frequencies.

Each of these would be a new study: it needs **fresh data and a fresh plan written down in advance**. The holdout used
here cannot be reused.
"""
)

# --------------------------------------------------------------------------- documents
st.subheader("The full record")
DOCS = (
    ("reports/final_report.md", "Final report", "the verdict and every table"),
    ("reports/tier1_report.md", "Tier 1 report", "the probes and the classical allocator"),
    ("reports/tier2_report.md", "Tier 2 report", "the agents on the test split"),
    ("reports/tables/preregistration.md", "Pre-registration, Tier 1", "committed before Tier 1 was scored"),
    ("reports/tables/preregistration_tier2.md", "Pre-registration, Tier 2", "committed before any agent was trained"),
    ("reports/tables/preregistration_holdout.md", "Pre-registration, holdout", "committed before the holdout was opened"),
    ("DECISIONS.md", "Decision log", "every non-obvious choice and why"),
)
for path, title, what in DOCS:
    file = data.ROOT / path
    left, right = st.columns([4, 1], vertical_alignment="center")
    left.markdown(f"**{title}** · {what} · `{path}`")
    if file.exists():
        right.download_button("Download", file.read_bytes(), file_name=file.name, mime="text/markdown", key=f"doc_{file.stem}",
                              width="stretch", on_click="ignore")
    else:
        right.caption("not found")

ui.how_to_read(
    """
* **Null result**: the test found no effect. It does not prove the effect is exactly zero; it says that if there is
  one, it is too small for this study to see.
* **Replicated**: the same answer came out twice, the second time on data that had been kept locked.
* **Pre-registered**: the comparisons, the measures and the pass rule were written down and committed before the
  results existed, so they could not be adjusted to suit the outcome.
* **FAIL** here means "did not pass the pre-registered rule". It is the finding, reported as it came out.
"""
)
ui.verified_footer()
