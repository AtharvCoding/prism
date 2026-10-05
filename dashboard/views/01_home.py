"""Page 1, Home: the machine, with its real values, and the verdict up front. DASHBOARD.md §7.1."""

from __future__ import annotations

import streamlit as st

from components import alloc, charts, current, data, theme, ui

facts = ui.page_header("home")
u, agents, windows, models = facts["universe"], facts["agents"], facts["windows"], facts["models"]
c = theme.colors()
v = current.view()
st.markdown(f"> *{ui.FRAMING}*")
current.controls(v, "home")

lines = alloc.lines(u)
w = current.strategy_weights(v, "V4 ensemble", lines)
sl = alloc.sleeves(w["mean"].to_frame().T, u).iloc[0]
p = v["p_volatile"]
b = agents["blocks"]["V4"]

st.subheader("The machine")
steps = [
    ("Prices", f"{len(lines) - 1 + 1 + len(u['signals_a']) + len(u['signals_b'])} series<br>to {v['as_of']:%d %b %Y}"),
    ("Features", f"{b['features']} numbers"),
    ("HMM", f"P(Volatile) {p:.0%}"),
    ("LSTM latent", f"{b['latent']} numbers"),
    ("State", f"{b['observation_width']} inputs"),
    (f"SAC × {agents['seeds']} seeds", "one policy each"),
    (f"{len(lines)} weights", f"equity {sl['Equity']:.0%}"),
]
st.plotly_chart(charts.pipeline(steps, c), config=theme.PLOT_CONFIG, key="pipeline")
a, bcol, d = st.columns([2, 3, 2])
with a:
    st.markdown("**Regime**")
    st.progress(p, text=f"P(Volatile) {p:.0%}: " + ("Volatile" if p > 0.5 else "Calm"))
    st.caption("Calm on the left, Volatile on the right.")
with bcol:
    st.markdown(f"**LSTM latent: {len(v['latent'])} numbers summarising the last {models['encoder']['window']} days**")
    st.plotly_chart(charts.latent_spark(v["latent"], c), config=theme.PLOT_CONFIG, key="latent_spark")
with d:
    st.markdown("**V4 ensemble, by sleeve**")
    st.plotly_chart(charts.sleeve_donut(sl, c, height=230), config=theme.PLOT_CONFIG, key="home_donut")

m = st.columns(4)
m[0].metric("As of", f"{v['decision_date']:%d %b %Y}", help="The weekly decision shown. See the badge above for whether it is live or recorded.")
m[1].metric("Regime", "Volatile" if p > 0.5 else "Calm", help=f"P(Volatile) = {p:.2f} on that day.")
m[2].metric("Equity / defensive", f"{sl['Equity']:.0%} / {1 - sl['Equity']:.0%}", help="Equity sectors against bonds, gold and cash; V4 ensemble.")
m[3].metric("Turnover from last week", ui.pct(w["turnover"], 1), help="Share of the portfolio traded, one way, averaged over the ten V4 agents.")

st.subheader("What the test found")
k = st.columns(3)
k[0].metric("Comparisons passed, holdout", f"{windows['holdout']['comparisons_passing']} of {len(agents['comparisons'])}")
k[1].metric("Comparisons passed, test split", f"{windows['test']['comparisons_passing']} of {len(agents['comparisons'])}")
k[2].metric("Agents' turnover per week", ui.pct_range(windows["holdout"]["turnover_agents"]),
            help=f"Against {ui.pct_range(windows['holdout']['turnover_benchmarks'], 1)} for the benchmarks, on the holdout.")
st.markdown(
    """
The system above was built end to end and then tested under rules fixed in advance. **Regime information from the HMM gave
the agent no detectable benefit, and neither did the LSTM summary**; the same answer came back on a holdout that was opened
exactly once. The agents did not beat an equal-weight, a 60/40 or a risk-parity portfolio after trading costs.

That is a result, and a clean one: the comparisons, the measures and the pass rule could not be adjusted to suit the outcome.
"""
)
st.page_link("views/11_verdict.py", label="Read the verdict and its limits", icon=":material/gavel:")
st.page_link("views/10_results.py", label="See every number behind it", icon=":material/fact_check:")

ui.how_to_read(
    f"""
* **The row of nodes** is the path from market data to portfolio weights. Each node shows a real value for the date in the
  badge above. Press *Run the pipeline* to light them in order.
* **HMM**: a model that estimates, each day, the probability that the market is in a Volatile regime.
* **LSTM latent**: {models['encoder']['latent_dim']} numbers a small neural network uses to summarise the last
  {models['encoder']['window']} trading days. The bars are those numbers; they are not individually meaningful.
* **SAC × {agents['seeds']} seeds**: the allocation agent, trained {agents['seeds']} separate times. The weights shown are the average of
  those {agents['seeds']} agents.
* **Comparisons passed**: how many of the three pre-registered tests found a reliable benefit. None did.
"""
)
