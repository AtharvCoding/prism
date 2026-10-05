"""Page 7, Live weights. DASHBOARD.md §7.7. The frozen system's latest weekly decision: a demonstration, not a recommendation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import alloc, charts, current, data, theme, ui

facts = ui.page_header("live")
u, rules, agents, stats = facts["universe"], facts["rules"], facts["agents"], facts["statistics"]
c = theme.colors()
v = current.view()
current.controls(v, "live")
lines, sleeve_of = alloc.lines(u), alloc.sleeve_of(u)

options = [*(f"{x} ensemble" for x in ("V4", "V2", "V1", "C4") if x in agents["variants"]), *current.BENCHMARKS]
strategy = st.segmented_control("Whose weights", options, default=options[0], key="live_strategy") or options[0]
w = current.strategy_weights(v, strategy, lines)
order = [n for n in alloc.lines(u, with_benchmark=True) if n in w["mean"].index]
mean = w["mean"][order]
sl = alloc.sleeves(mean.to_frame().T, u).iloc[0]

st.subheader(f"The weekly decision of {v['decision_date']:%d %b %Y}")
m = st.columns(4)
m[0].metric("Equity", ui.pct(sl["Equity"], 1))
m[1].metric("Bonds + gold + cash", ui.pct(1 - sl["Equity"], 1))
if w["turnover"] is not None:
    m[2].metric("Turnover from last week", ui.pct(w["turnover"], 1), help="Share of the portfolio traded to reach these weights, one way, averaged over the seeds.")
    m[3].metric(f"Cost of that trade at {stats['headline_bps']:g} bps a side", f"{w['traded'] * stats['headline_bps']:.1f} bps",
                help="Basis points of the whole portfolio: the per-side cost on the risky holdings traded, before volatility-scaled slippage.")
else:
    m[2].metric("Turnover from last week", "n/a", help="Recorded for the agents only.")
left, right = st.columns([2, 5])
with left:
    st.plotly_chart(charts.sleeve_donut(sl, c), config=theme.PLOT_CONFIG, key="live_donut")
with right:
    st.plotly_chart(charts.weights_bar(mean, None if w["min"] is None else w["min"][order], None if w["max"] is None else w["max"][order],
                                       sleeve_of, rules["cap"], c), config=theme.PLOT_CONFIG, key="live_bar")
if w["min"] is not None:
    spread = (w["max"] - w["min"])[order]
    st.caption(f"Bars are the average of the {agents['seeds']} {strategy.split()[0]} agents; whiskers run from the lowest seed to the highest. "
               f"The seeds disagree by {ui.pct(spread.mean(), 1)} of the portfolio on a typical holding and by {ui.pct(spread.max(), 1)} on "
               f"{spread.idxmax()}: that gap is how unsettled this \"decision\" is.")
else:
    st.caption(f"{strategy}, by its stored definition" + (" (the S&P 500 fund is a benchmark holding only; the agents cannot hold it)."
                                                          if "SPY" in order else "."))

if v["is_preview"] and v["preview"] is not None and w["min"] is not None:
    st.subheader(f"Preview as of the close of {v['as_of']:%d %b %Y}")
    pw = current.strategy_weights(v, strategy, lines, preview=True)
    st.warning("This is **not a decision**. The week has not ended. It is what these agents would choose if it ended at the latest "
               "close, shown so the page is not a week stale; the next decision is taken at the week's last close.", icon=":material/schedule:")
    st.plotly_chart(charts.weights_bar(pw["mean"][order], pw["min"][order], pw["max"][order], sleeve_of, rules["cap"], c),
                    config=theme.PLOT_CONFIG, key="live_preview")

st.subheader("How this number is produced")
models = v["models"]
first = facts["splits"]["holdout" if v["window"] == "holdout" else "test"]["effective"][0]
st.markdown(
    f"""
* **One continuous episode.** Each agent's weights depend on what it already holds. So every agent starts in cash at the first
  decision of the {'holdout' if v['window'] == 'holdout' else 'test'} window (the week of {first}) and is stepped forward week by week; the
  latest step is what you see. Up to the end of the stored evaluation this is the same episode that produced the stored results.
* **Frozen models.** """
    + (f"The regime model was last fitted on data to {models['hmm_fit_end']} and the LSTM on data to {models['encoder_fit_end']}. The "
       "research pipeline would refit them monthly and yearly; **the live view does not refit anything**." if models else
       "This copy is showing recorded weights, so no model is running here.")
    + f"""
* **Weekly.** A decision is taken at the last close of the week and traded at the next close. Mid-week, the page shows the
  last decision and, separately, a labelled preview.
* **Checks.** Fresh prices are joined to the frozen data only if they agree over the last 60 common days; today's unfinished
  bar is never used.
"""
)
with st.expander("The numbers"):
    table = pd.DataFrame({"sleeve": [sleeve_of[n] for n in order], "weight": mean.to_numpy()}, index=pd.Index(order, name="holding"))
    if w["min"] is not None:
        table["lowest seed"], table["highest seed"] = w["min"][order].to_numpy(), w["max"][order].to_numpy()
    ui.table(table.style.format({k: "{:.1%}" for k in table.columns if k != "sleeve"}), index=True)

ui.how_to_read(
    f"""
* **Ensemble**: the plain average of a variant's {agents['seeds']} independently trained agents. **Whiskers** show the lowest and
  highest of them for each holding.
* **Turnover**: how much of the portfolio changes hands to get from last week's (drifted) holdings to these weights.
* **Preview**: a what-if for an unfinished week. It is never traded and never called a decision.
* **Stored / Live / Cached / Stale**: where the numbers came from. *Stored* means recorded results; *Live* a refresh made in
  this session; *Cached* an earlier refresh; *Stale* that the last attempt to refresh failed.
* None of this is advice. On the test split and on the holdout these agents did not beat equal weight, 60/40 or risk parity
  after costs.
"""
)
