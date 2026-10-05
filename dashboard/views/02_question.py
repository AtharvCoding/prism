"""Page 2, The question. DASHBOARD.md §7.2."""

from __future__ import annotations

import streamlit as st

from components import data, ui

facts = ui.page_header("question")
agents = facts["agents"]
blocks = agents["blocks"]
encoder, n_weights = facts["models"]["encoder"], facts["universe"]["n_weights"]

st.markdown(
    """
> Does explicit probabilistic regime conditioning (a hidden Markov model) improve portfolio allocation beyond what a
> temporal representation model (an LSTM autoencoder) and raw features already capture?

Put plainly: a portfolio agent already sees today's market features and a compressed summary of the last few weeks.
If we also tell it *"the market is probably calm"* or *"probably volatile"*, does it allocate better?
"""
)

# --------------------------------------------------------------------------- three comparisons
st.subheader("Three comparisons, fixed in advance")
ASKS = {
    ("V4", "V2"): ("The research question", "Does the HMM's regime estimate add anything once the agent has the LSTM summary?"),
    ("V4", "C4"): ("HMM against a simple rule", "Is the HMM any better than just asking whether the VIX is above or below a threshold?"),
    ("V2", "V1"): ("The LSTM summary", "Does the LSTM summary add anything to the raw features?"),
}
gates = {w: data.headline_tables(w)["gates"].set_index("comparison") for w in ("test", "holdout")}
cols = st.columns(3)
for col, (x, y) in zip(cols, (tuple(c) for c in agents["comparisons"])):
    title, ask = ASKS[(x, y)]
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{x} vs {y}** · {title}")
        st.write(ask)
        st.caption(f"Test split: {gates['test'].loc[f'{x} vs {y}', 'paired rule']} · "
                   f"Holdout: {gates['holdout'].loc[f'{x} vs {y}', 'paired rule']}")
st.caption("The first-named variant is the candidate, the second the control. A comparison passes only if the candidate is "
           "reliably better on at least 3 of 4 measures and reliably worse on none. Details are on the Results page.")

# --------------------------------------------------------------------------- what each variant sees
st.subheader("What each variant sees")
st.write(f"Four versions of the same agent, {agents['seeds']} independently trained copies (seeds) of each, "
         f"{agents['n_agents']} agents in all. They differ only in their inputs.")
labels = [f"{x} vs {y}" for x, y in agents["comparisons"]]
choice = st.segmented_control("Highlight what differs between", labels, default=labels[0], key="pair")
pair = tuple((choice or labels[0]).split(" vs "))

BLOCKS = (
    ("features", "Market features", "today's returns, volatility, momentum, rates, credit"),
    ("latent", "LSTM summary", f"a {encoder['latent_dim']}-number summary of the last {encoder['window']} sessions"),
    ("hmm", "HMM regime", "P(Calm), P(Volatile)"),
    ("vix_threshold", "VIX threshold", "VIX low / VIX high, as 0 or 1"),
    ("portfolio", "Own portfolio", f"current {n_weights} weights and average turnover"),
)


def _count(variant: str, block: str) -> int:
    b = blocks[variant]
    if block in ("hmm", "vix_threshold"):
        return b["regime"] if b["regime_source"] == block else 0
    return b[block]


differs = {key for key, _, _ in BLOCKS if _count(pair[0], key) != _count(pair[1], key)}
header = st.columns([1.1, *([2] * len(BLOCKS))])
header[0].caption("Variant")
for col, (_, name, what) in zip(header[1:], BLOCKS):
    col.markdown(f"**{name}**")
    col.caption(what)
for v in agents["variants"]:
    row = st.columns([1.1, *([2] * len(BLOCKS))])
    selected = v in pair
    row[0].markdown(f"**{v}**" if selected else f":gray[{v}]")
    for col, (key, _, _) in zip(row[1:], BLOCKS):
        n = _count(v, key)
        text = f"{n} inputs" if n else "not given"
        if selected and key in differs:
            col.markdown(f":blue-background[**{text}**]")
        elif selected:
            col.markdown(text)
        else:
            col.markdown(f":gray[{text}]")
if differs:
    names = ", ".join(name for key, name, _ in BLOCKS if key in differs)
    st.caption(f"{pair[0]} and {pair[1]} differ only in: {names}. Everything else they see is identical, so any reliable "
               "difference in results would be down to that input.")

# --------------------------------------------------------------------------- two tiers
st.subheader("Two tiers, so that a null result means something")
st.markdown(
    """
**Tier 1, no agent.** Simple forecasting probes ask whether each component carries information the raw features lack.
**Tier 2, with the agent.** Trained SAC agents ask whether a policy can turn that information into better allocations.

Tier 1 exists because training an agent is noisy. If Tier 2 finds nothing, Tier 1 tells us whether there was anything
to find in the first place, or whether the agent simply failed to use it.
"""
)
GATE = {"lstm_adds_value": "The LSTM adds value", "hmm_adds_value": "The HMM adds value",
        "research_question": "Research question (V4 beyond V2)"}
tier1 = data.tier1_gates().assign(
    gate=lambda d: d.gate.map(GATE),
    comparison=lambda d: d.comparison.replace("(gate)", "the gate as a whole"),
    result=lambda d: d.result.str.replace("**", "", regex=False),
)
ui.table(tier1)
st.caption(
    "Tier 1 results, as stored. A gate passes only if both of its comparisons pass. The HMM was no better than a "
    "VIX threshold (V3 vs C2) and the trained LSTM was no better than an untrained random one (V2 vs C1), so both gates "
    "failed. The research-question comparison (V4 vs V2) did pass at this stage, by a small margin (the sizes are in "
    "`reports/tier1_report.md`). Because the HMM could not be told apart from the threshold, that gain could not be "
    "credited to the HMM, and the control C4 was added to Tier 2 to test exactly that."
)

ui.how_to_read(
    f"""
* **Regime**: a broad market condition. Here there are two, *Calm* and *Volatile*.
* **HMM (hidden Markov model)**: a statistical model that looks at daily S&P 500 returns and estimates, each day, the
  probability that the market is in each regime. It uses only data up to that day.
* **LSTM summary (latent)**: {encoder['latent_dim']} numbers produced by a small neural network that has learned to
  compress the last {encoder['window']} trading days of market features.
* **SAC agent**: a reinforcement-learning algorithm (Soft Actor-Critic). It learns, by trial and error on historical
  data, a rule that turns what it sees into {n_weights} portfolio weights.
* **Variant**: one choice of inputs for the agent (V1, V2, V4, C4). **Control**: a variant built to rule out an
  alternative explanation; C4 swaps the HMM for a crude VIX rule.
* **Seed**: one training run. The same variant trained {agents['seeds']} times gives {agents['seeds']} somewhat different agents; the spread
  between them is noise that any real effect has to exceed.
* **PASS / FAIL** under each comparison is the pre-registered verdict, read from the stored results.
"""
)
