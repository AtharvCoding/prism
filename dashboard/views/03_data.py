"""Page 3, Data and universe. DASHBOARD.md §7.3."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts, theme, ui

facts = ui.page_header("data")
u, splits, rules = facts["universe"], facts["splits"], facts["rules"]
hold = facts["windows"]["holdout"]
c = theme.colors()

# --------------------------------------------------------------------------- timeline
st.subheader("Two universes, four periods")
st.plotly_chart(charts.timeline(facts, c), config=theme.PLOT_CONFIG, key="timeline")
st.markdown(
    f"""
**Universe A** (from {u['a_start'][:4]}) holds the sector funds, the S&P 500 and the main rates and volatility series.
The regime model and the LSTM are fitted on it, so they have seen the dot-com crash, 2008 and 2020.
**Universe B** (from {u['b_start']}, when the last of its series began trading) adds bonds, gold, credit and the
dollar. It is what the agent actually allocates across.

The gaps between the periods are deliberate: {facts['embargo_days']} trading days are dropped from the end of each
period, so nothing computed near a boundary can look into the next one.
"""
)
ROLE = {
    "train": "The agents learn here.",
    "val": "Settings and the checkpoint to keep are chosen here.",
    "test": "Exploratory: these years had already been looked at in Tier 1 before the agents were built.",
    "holdout": "Confirmatory. Kept locked until everything was frozen, then evaluated exactly once.",
}
ui.table(pd.DataFrame([{"Period": charts.SPLIT_LABEL[k], "From": splits[k]["effective"][0], "To": splits[k]["effective"][1],
                        "Trading days": splits[k]["sessions"], "What it is for": ROLE[k]} for k in ROLE]))

# --------------------------------------------------------------------------- the holdout
st.subheader("The holdout: sealed, then opened once")
a, b, c3 = st.columns(3)
a.metric("Status", "Spent", help="It has been used for its one evaluation and cannot be used to choose or tune anything.")
b.metric("Opened", hold["evaluated_utc"][:10], help="The date of the single pre-registered evaluation.")
c3.metric("Trading days evaluated", f"{hold['n_sessions']}", help=f"Daily returns from {hold['first']} to {hold['last']}.")
st.markdown(
    """
:material/lock: **Sealed** while the models, the agents, the costs and the pass rule were being built and frozen.
:material/arrow_forward: :material/lock_open: **Opened once**, by hand, behind two separate safeguards, with the
analysis plan already committed. Nothing was changed afterwards, and it will not be evaluated again: any new idea
needs new data and a new plan written in advance.
"""
)

# --------------------------------------------------------------------------- assets and signals
st.subheader("What can be held, and what is only watched")
NAMES = {
    "XLK": "Technology", "XLE": "Energy", "XLV": "Health care", "XLF": "Financials", "XLU": "Utilities",
    "XLI": "Industrials", "XLP": "Consumer staples", "XLY": "Consumer discretionary", "XLB": "Materials",
    "SHY": "US Treasuries, 1 to 3 years (short)", "IEF": "US Treasuries, 7 to 10 years (mid)",
    "TLT": "US Treasuries, 20+ years (long)", "GLD": "Gold", "CASH": "Cash, earning the 13-week Treasury bill rate",
    "SPY": "S&P 500: benchmark and signal, not tradable", "^VIX": "VIX, expected 30-day volatility",
    "^VIX3M": "VIX3M, expected 3-month volatility", "^TNX": "10-year Treasury yield", "^FVX": "5-year Treasury yield",
    "^IRX": "13-week Treasury bill yield", "HYG": "High-yield bonds (credit proxy, with LQD)",
    "LQD": "Investment-grade bonds (credit proxy, with HYG)", "DX-Y.NYB": "US dollar index", "CL=F": "Crude oil futures",
}
tradable = ([("Equity", t) for t in u["sectors"]] + [("Bonds", t) for t in u["bonds"]] + [("Gold", t) for t in u["gold"]]
            + [("Cash", u["cash"])])
signals = [u["benchmark"], *u["signals_a"], *u["signals_b"]]
left, right = st.columns(2)
with left:
    st.markdown(f"**Tradable: the {u['n_weights']} weights**")
    ui.table(pd.DataFrame([{"Sleeve": s, "Ticker": t, "What it is": NAMES.get(t, "")} for s, t in tradable]))
with right:
    st.markdown("**Signal only: the agent sees these but cannot hold them**")
    ui.table(pd.DataFrame([{"Ticker": t, "What it is": NAMES.get(t, "")} for t in signals]))
    st.markdown("**Left out**")
    ui.table(pd.DataFrame([{"Ticker": t, "Why": why} for t, why in u["excluded"].items()]))
st.caption("The S&P 500 fund is deliberately not a holding: it overlaps the nine sector funds. It is the yardstick, "
           "and its returns feed the regime model.")

# --------------------------------------------------------------------------- rules
st.subheader("The allocation rules")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Direction", "Long-only", help="No short positions and no leverage.")
r2.metric("Cap per risky asset", ui.pct(rules["cap"]), help="Cash is not capped, so the agent can always step aside.")
r3.metric("Weights sum to", "100%", help="Cash included.")
r4.metric("Decisions", "Weekly", help="Decided at the last close of the week, traded at the next close.")
st.caption(f"Every trade costs {rules['per_side_bps']:g} basis points per side on each risky asset traded, plus slippage "
           f"that grows with that asset's recent volatility ({rules['slippage_vol_coef']:g} times its {rules['vol_window']}-day "
           "daily volatility, per unit traded). A basis point is one hundredth of one percent.")

ui.how_to_read(
    """
* **The timeline** shows which years each stage could see. The four coloured blocks are separate, in time order:
  nothing used to build or choose the agents comes from a later block.
* **Sleeve**: a group of similar assets (equity sectors, bonds, gold, cash). Later pages add weights up by sleeve.
* **Exploratory vs confirmatory**: a result on data you have already looked at can be flattered by the choices you
  made after looking. The test split is in that position. The holdout is not, which is why it carries the verdict.
* **Spent**: the holdout's one use has happened. It is shown here but can never again be used to decide anything.
"""
)
