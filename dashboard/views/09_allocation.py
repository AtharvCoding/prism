"""Page 9, Allocation through time (descriptive). DASHBOARD.md §7.9. Reads the weekly weights recorded by the replay (D-047)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import alloc, charts, data, theme, ui

facts = ui.page_header("allocation")
u, rules, agents_f = facts["universe"], facts["rules"], facts["agents"]
c = theme.colors()
windows = data.weight_windows()
agents, bench = data.weights("agents"), data.weights("benchmarks")
if "holdout" not in windows:
    st.info("These charts stop at the end of the test split. The holdout weeks (2024 to 2026) are recorded by a separate, "
            "gated replay of the frozen agents that has not been run in this copy.", icon=":material/lock:")

strategy = st.selectbox("Whose allocation", alloc.strategies(agents_f["variants"]), key="alloc_strategy")
w = alloc.strategy_weights(agents, bench, strategy)
is_ensemble = w["low"] is not None
order = [n for n in alloc.lines(u, with_benchmark=True) if n in w["mean"].columns]
weights = w["mean"][order]
sleeve_of = alloc.sleeve_of(u)
p_vol = data.regimes().p_volatile.reindex(weights.index)
holdout_start = agents[agents.window == "holdout"].decision_date.min() if "holdout" in windows else None

# --------------------------------------------------------------------------- sleeves through time
st.subheader("The four sleeves, every weekly decision")
sl = alloc.sleeves(weights, u)
st.plotly_chart(charts.allocation_area(sl, p_vol, holdout_start, c), config=theme.PLOT_CONFIG, key="allocation_area")
if is_ensemble:
    st.caption(f"The average target weights of the {agents_f['seeds']} {strategy.split()[0]} agents at each of {len(weights)} weekly decisions, "
               "added up by sleeve. The strip above is the HMM's P(Volatile) on each decision day (blue Calm, red Volatile). "
               "An average of ten agents is a summary of what they held; no averaged portfolio was evaluated.")
else:
    st.caption(f"The {strategy} benchmark's target weights at each of {len(weights)} weekly decisions, added up by sleeve. The strip "
               "above is the HMM's P(Volatile) on each decision day; a benchmark does not see it.")

# --------------------------------------------------------------------------- one week
st.subheader("One week in detail")
dates = list(weights.index)
week = st.select_slider("Decision date", options=dates, value=dates[-1], format_func=lambda d: f"{d:%d %b %Y}", key="alloc_week")
row = weights.loc[week]
cols = st.columns(5)
for col, name in zip(cols, theme.SLEEVES):
    col.metric(name, ui.pct(sl.loc[week, name], 1))
cols[4].metric("Turnover that week", ui.pct(w["turnover"].loc[week], 1) if is_ensemble else "n/a",
               help="Average over the seeds of the share of the portfolio traded to reach these weights (one-way).")
low, high = (w["low"].loc[week, order], w["high"].loc[week, order]) if is_ensemble else (None, None)
st.plotly_chart(charts.weights_bar(row, low, high, sleeve_of, rules["cap"], c), config=theme.PLOT_CONFIG, key="weights_bar")
st.caption((f"Bars are the average of the {agents_f['seeds']} seeds; the whiskers run from the lowest seed to the highest. Wide whiskers "
            "mean the ten agents, trained identically but for their random seed, disagreed about that holding. " if is_ensemble else "")
           + f"P(Volatile) on this day was {p_vol.loc[week]:.2f}.")
with st.expander("The numbers for this week"):
    table = pd.DataFrame({"sleeve": [sleeve_of[n] for n in order], "target weight": row.to_numpy()}, index=pd.Index(order, name="holding"))
    if is_ensemble:
        table["lowest seed"], table["highest seed"] = low.to_numpy(), high.to_numpy()
    ui.table(table.style.format({k: "{:.1%}" for k in table.columns if k != "sleeve"}), index=True)

# --------------------------------------------------------------------------- defensive share against the regime
st.subheader("Did the allocation turn defensive when the regime turned Volatile?")
defensive = alloc.defensive_share(weights, u)
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(charts.defensive_scatter(defensive, p_vol, c), config=theme.PLOT_CONFIG, key="defensive_scatter")
with right:
    corr = alloc.correlation(defensive, p_vol)
    st.metric("Correlation across weeks", f"{corr:+.2f}", help="Pearson correlation between the non-equity share and P(Volatile), one point per weekly decision.")
    st.write("Each dot is one weekly decision. If regime information drove the allocation, the dots would climb to the "
             "right: more bonds, gold and cash when the HMM says Volatile.")
    st.caption("A description of these weeks, with no test behind it. Weeks close together are not independent, and "
               + ("only V4 is given the HMM's estimate; the other variants can only react to what moves with it."
                  if is_ensemble else "a benchmark does not see the regime at all."))

# --------------------------------------------------------------------------- duration ladder
st.subheader("Inside the bond sleeve: short, mid and long maturities")
episodes = data.results("test")["episodes"].drop_duplicates("episode")
rate_year = episodes[pd.to_datetime(episodes.peak).dt.year == 2022]
span = (rate_year.iloc[0].peak, rate_year.iloc[0].trough) if len(rate_year) else None
st.plotly_chart(charts.duration_ladder(weights, u["bonds"], span, c), config=theme.PLOT_CONFIG, key="ladder")
if span is not None:
    e = data.results("test")["episodes"]
    sf = e[(e.peak == span[0]) & (e.strategy == "BM|SixtyForty")].iloc[0]
    st.caption(f"{u['bonds'][0]} holds short-dated Treasuries, {u['bonds'][-1]} long-dated ones, which move most when interest rates "
               f"change. The shaded span is the {span[0][:4]} fall in the S&P 500 ({span[0]} to {span[1]}, "
               f"{ui.pct(rate_year.iloc[0].depth, 1)}), when bonds did not cushion equities: the 60/40 benchmark lost "
               f"{ui.pct(abs(sf.decline_return_mean), 1)} over the same weeks.")

ui.how_to_read(
    f"""
* **Sleeve**: equity is the {len(u['sectors'])} sector funds (and the S&P 500 fund for the two benchmarks that hold it); bonds are
  {', '.join(u['bonds'])}; gold is {u['gold'][0]}; cash earns the Treasury bill rate.
* **Target weight**: what the strategy asked for at that week's last close. It is traded at the next close.
* **Ensemble**: the plain average of a variant's {agents_f['seeds']} agents. **Whiskers** show how far apart those agents were.
* **Defensive share**: bonds + gold + cash, everything that is not equity.
* **Correlation**: +1 would mean the defensive share always rose with P(Volatile), 0 no relation, −1 the opposite.
* **Descriptive** means exactly that. These charts show what was held. The Results page shows what it earned, and
  that it did not beat the benchmarks.
"""
)
