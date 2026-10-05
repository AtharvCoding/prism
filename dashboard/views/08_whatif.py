"""Page 8, What-if lab. DASHBOARD.md §7.8. Change the regime input of the latest observation; everything else stays fixed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import alloc, charts, current, theme, ui
from prism import live

facts = ui.page_header("whatif")
u, agents = facts["universe"], facts["agents"]
c = theme.colors()
v = current.view()
current.controls(v, "whatif")
policies = current.policies()
if policies is None:
    st.error("The frozen agents' checkpoints are not available on this machine, so the lab cannot run them.")
    st.stop()
cfg = current._config()
lines, sleeve_of = alloc.lines(u), alloc.sleeve_of(u)
seeds = sorted({s for (_, s) in policies})
today = v["p_volatile"]


@st.cache_data(show_spinner=False)
def weights_at(variant: str, p: float | None, decision_key: str) -> pd.DataFrame:
    """One row per seed: the weights that variant's agents choose with the regime inputs set to (1 - p, p)."""
    cols = v["columns"][variant]
    rows = [live.what_if_weights(cfg, policies[(variant, s)], np.asarray(v["observations"][f"{variant}|s{s}"], dtype="float32"), cols, p) for s in seeds]
    return pd.DataFrame(rows, index=seeds, columns=lines)


key = f"{v['source']}|{v['decision_date']}"
variant = st.segmented_control("Agent", ["V4", "C4"], default="V4", key="whatif_variant",
                               format_func=lambda x: {"V4": "V4 · sees the HMM's P(Volatile)", "C4": "C4 · sees VIX low / high"}[x]) or "V4"
st.subheader("Set the regime the agent is told")
if variant == "V4":
    b1, b2, b3, _ = st.columns([1, 1, 1, 4])
    if b1.button("Calm (0)", key="preset_calm", width="stretch"):
        st.session_state["whatif_p"] = 0.0
    if b2.button("Volatile (1)", key="preset_volatile", width="stretch"):
        st.session_state["whatif_p"] = 1.0
    if b3.button("Today", key="preset_today", width="stretch"):
        st.session_state["whatif_p"] = round(today * 20) / 20
        st.session_state["whatif_today"] = True
    p = st.slider("P(Volatile) fed to the agent", 0.0, 1.0, round(today * 20) / 20, 0.05, key="whatif_p")
    use_today = st.session_state.get("whatif_today", True) and p == round(today * 20) / 20
    st.session_state["whatif_today"] = use_today
    setting = None if use_today else p
    st.caption(f"The actual value on {v['decision_date']:%d %b %Y} was {today:.3f}. " + ("**Showing the actual input, untouched.**" if use_today else
               f"Showing P(Volatile) = {p:.2f}; the other {len(v['columns']['V4']) - 2} inputs are exactly as they were."))
else:
    actual_high = float(v["observations"][f"C4|s{seeds[0]}"][v["columns"]["C4"].index("state_1")]) == 1.0
    choice = st.radio("VIX state fed to the agent", ["As it was", "VIX low", "VIX high"], horizontal=True, key="whatif_c4")
    setting = None if choice == "As it was" else (0.0 if choice == "VIX low" else 1.0)
    p = (1.0 if actual_high else 0.0) if setting is None else setting
    st.caption(f"This control is binary: C4 is told only whether the VIX is below or above its threshold. On {v['decision_date']:%d %b %Y} "
               f"it was **{'high' if actual_high else 'low'}**.")

base = weights_at(variant, None, key)
now = weights_at(variant, setting, key)
change = now.mean() - base.mean()
sl_base, sl_now = alloc.sleeves(base.mean().to_frame().T, u).iloc[0], alloc.sleeves(now.mean().to_frame().T, u).iloc[0]

st.subheader("What the ten agents choose")
m = st.columns(4)
for col, name in zip(m, theme.SLEEVES):
    col.metric(name, ui.pct(sl_now[name], 1), delta=f"{100 * (sl_now[name] - sl_base[name]):+.1f} pp", delta_color="off")
left, right = st.columns(2)
with left:
    st.markdown("**Weights at this setting**")
    st.plotly_chart(charts.weights_bar(now.mean(), now.min(), now.max(), sleeve_of, facts["rules"]["cap"], c), config=theme.PLOT_CONFIG, key="whatif_bar")
with right:
    st.markdown("**Change from the actual input, by holding**")
    st.plotly_chart(charts.change_bars(change, sleeve_of, c), config=theme.PLOT_CONFIG, key="whatif_change")

st.subheader("The whole range: sleeves against the regime input")
grid = [round(x, 2) for x in np.linspace(0, 1, 21)] if variant == "V4" else [0.0, 1.0]
sweep = pd.DataFrame({g: alloc.sleeves(weights_at(variant, g, key).mean().to_frame().T, u).iloc[0] for g in grid}).T
flat = {f"{x} equity (no regime input)": float(alloc.sleeves(live.ensemble(v["decision"][v["decision"].variant == x])["mean"][lines].to_frame().T, u).iloc[0]["Equity"])
        for x in ("V1", "V2")}
st.plotly_chart(charts.sweep_lines(sweep, flat, p if variant == "C4" else today, c), config=theme.PLOT_CONFIG, key="whatif_sweep")
effect = float((sweep.max() - sweep.min()).max())
which = (sweep.max() - sweep.min()).idxmax()
per_seed = pd.DataFrame({s: alloc.sleeves(base.loc[[s]], u).iloc[0] for s in seeds}).T
disagreement = float(per_seed[which].max() - per_seed[which].min())
e1, e2 = st.columns(2)
e1.metric("Regime effect", f"{100 * effect:.1f} pp", help=f"The most any sleeve's average share moves across the whole range of the input ({which}).")
e2.metric("Seed disagreement", f"{100 * disagreement:.1f} pp", help=f"The gap between the highest and lowest seed's {which} share at the actual input.")
st.markdown(
    f"Moving the regime input across its whole range shifts the {which.lower()} sleeve by **{100 * effect:.1f} percentage points** on "
    f"average. At the actual input, the ten agents already differ from each other by **{100 * disagreement:.1f} points** on the same sleeve. "
    + ("**The regime input moves the allocation less than the choice of random seed does.** A flat or wandering line here is the finding "
       "in visible form: the agent makes little systematic use of the regime it is told." if effect < disagreement else
       "Here the regime input moves the allocation more than the seeds differ. That describes this one observation; the tests on the "
       "Results page are what measure whether it helped, and they found it did not.")
)
st.caption("V1 and V2 are given no regime input, so by construction the slider changes nothing for them: they are the flat dotted lines. "
           "An input the agent never met in training alongside these features (P(Volatile) = 1 on a calm day, say) is a counterfactual. "
           "The chart shows sensitivity to one input. It is not a forecast of what the agent would do in a real Volatile regime, where "
           "every other input would differ too.")

ui.how_to_read(
    f"""
* **What changes**: only the two regime numbers in what the agent sees. Prices, features, the LSTM summary and the agent's own
  current portfolio stay exactly as they were on {v['decision_date']:%d %b %Y}.
* **pp**: percentage points of the portfolio.
* **Whiskers**: the lowest and highest of the {len(seeds)} seeds.
* **Regime effect vs seed disagreement**: two sizes on the same scale. If the first is smaller, changing what the agent is told
  about the regime matters less than which training run you happened to pick.
* **Flat dotted lines**: variants that are not told the regime at all.
"""
)
