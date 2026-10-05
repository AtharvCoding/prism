"""Page 4, Regimes. DASHBOARD.md §7.4. Everything here is read from stored results; no model runs on this page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import charts, data, theme, ui

facts = ui.page_header("regimes")
hmm, u = facts["models"]["hmm"], facts["universe"]
c = theme.colors()
regimes, summary, folds = data.regimes(), data.regime_summary(), data.hmm_folds()
agree = summary["agreement"]

# --------------------------------------------------------------------------- price with shading
st.subheader("Two regimes, estimated one day at a time")
show_nber = st.toggle("Mark NBER recessions", value=False, key="nber")
st.plotly_chart(charts.regime_price(data.spy(), regimes, u["fit_early"], summary["nber_recessions"] if show_nber else None, c),
                config=theme.PLOT_CONFIG, key="regime_price")
st.caption(
    f"The model looks only at the S&P 500's daily return. It is refitted every month on all history up to then, and each "
    f"day is shaded by the estimate made with data up to that day, never later. The grey span, {u['fit_early'][0][:4]} to "
    f"{u['fit_early'][1][:4]}, is the first fit window: there is no out-of-sample estimate for it. Drag the strip under the "
    "chart, or use the buttons, to zoom."
)

# --------------------------------------------------------------------------- latest state, transition matrix, dwell
st.subheader("The model as it stood at the end of the stored results")
last_day, p_vol = regimes.index[-1], float(regimes.p_volatile.iloc[-1])
fold = folds.iloc[-1]
left, mid, right = st.columns([1, 1.2, 1])
with left:
    st.metric(f"P(Volatile) on {last_day:%d %b %Y}", f"{p_vol:.0%}", help="The last day in the stored results. A live reading is a later milestone.")
    st.progress(p_vol, text="Calm  ←  →  Volatile")
with mid:
    st.markdown("**Chance of moving between regimes from one day to the next**")
    matrix = pd.DataFrame([[fold.p_00, fold.p_01], [fold.p_10, fold.p_11]], index=["from Calm", "from Volatile"],
                          columns=["to Calm", "to Volatile"])
    ui.table(matrix.rename_axis("today → tomorrow").style.format("{:.1%}"), index=True)
with right:
    st.metric("Expected stay in Calm", f"{fold.dwell_0:.0f} trading days", help="1 / (1 − chance of staying), from the matrix.")
    st.metric("Expected stay in Volatile", f"{fold.dwell_1:.0f} trading days")
note = "" if "holdout" in data.weight_windows() else (" The refits for 2024 onwards are recorded by the gated holdout replay, "
                                                      "which has not been run in this copy.")
st.caption(f"From refit number {int(fold.fold)}, fitted on data to {fold.fit_end:%d %b %Y} and applied to "
           f"{fold.apply_start:%b %Y}: the most recent refit whose parameters are stored here.{note}")

# --------------------------------------------------------------------------- volatility across folds
st.subheader("How the two regimes' volatility estimates moved as history accumulated")
st.plotly_chart(charts.fold_volatility(folds, c), config=theme.PLOT_CONFIG, key="fold_vol")
st.caption(f"One point per monthly refit ({len(folds)} of them). The Volatile state's estimate jumps when a crisis enters the "
           "history (2008, 2020) and then settles, because every refit uses all the data so far.")

# --------------------------------------------------------------------------- HMM vs VIX
st.subheader("The HMM against a plain VIX threshold")
st.plotly_chart(charts.regime_strips(regimes, c), config=theme.PLOT_CONFIG, key="strips")
a = agree["all"]
overlap = a["both_volatile"] / (a["both_volatile"] + a["hmm_only"])
m1, m2, m3 = st.columns(3)
m1.metric("Days the two agree", ui.pct(a["agreement"], 1), help=f"{a['days']:,} trading days, {a['start']} to {a['end']}.")
m2.metric("VIX high when the HMM says Volatile", ui.pct(overlap), help=f"{a['both_volatile']:,} of {a['both_volatile'] + a['hmm_only']:,} days.")
m3.metric("Share of days flagged", f"{ui.pct(a['hmm_volatile_share'])} vs {ui.pct(a['vix_high_share'])}", help="HMM Volatile against VIX high.")
st.markdown(
    f"""
The VIX threshold is the control in this project: split the VIX into a low half and a high half and call that the
regime. Almost every day the HMM calls Volatile, the VIX is also high. The disagreement is nearly all one way: the
threshold says "high" on many days the HMM still calls Calm ({a['vix_only']:,} days, against {a['hmm_only']:,} the other
way). So the HMM behaves like a stricter version of the same signal. That is the likeliest reason it added nothing: an
agent that can see the VIX already has most of what the HMM would tell it.
"""
)
NAMES = {"train": "Train", "val": "Validation", "test": "Test", "holdout": "Holdout", "all": "All days"}
ui.table(pd.DataFrame([{"Period": NAMES[k], "From": v["start"], "To": v["end"], "Days": v["days"], "Agree": ui.pct(v["agreement"], 1),
                        "Both Calm / low": v["both_calm"], "Both Volatile / high": v["both_volatile"],
                        "HMM Volatile, VIX low": v["hmm_only"], "HMM Calm, VIX high": v["vix_only"]} for k, v in ((k, agree[k]) for k in NAMES)]))
st.caption("Counts of days. The HMM's call is Volatile when P(Volatile) is above one half. The agreement is lower on the "
           "test split and the holdout than on the training years.")

# --------------------------------------------------------------------------- why two states
with st.expander("Why two states?", icon=":material/quiz:"):
    k = data.k_selection()
    usable = k[k.restarts_degenerate < k.restarts]
    best, chosen = usable.loc[usable.val_loglik.idxmax()], k[k.selected].iloc[0]
    st.plotly_chart(charts.k_selection(k, hmm["near_tie_margin"], c), config=theme.PLOT_CONFIG, key="k_selection")
    st.markdown(
        f"""
Models with 2 to 8 states were fitted on the first window and scored on a later validation year.

* With **{int(k[~k.index.isin(usable.index)].k.min())} or more states** every fit was *degenerate*: at least one state lasted about a day or
  two, which is a shock, not a regime. Those were discarded outright (greyed).
* Between the two that remained, **K = {int(best.k)} scored {best.val_loglik - chosen.val_loglik:.2f} points higher** on validation
  log-likelihood. A gap under {hmm['near_tie_margin']:g} points was declared a tie in advance, to be broken by BIC, a score
  that penalises extra parameters. **K = {int(chosen.k)} has the better BIC**, so two states were chosen.
* K = {int(chosen.k)} was also the only choice whose regimes typically lasted more than a week.
"""
    )
    dwell = data.selection_dwell()
    h1 = dwell[dwell.spec == hmm["specification"]].groupby("k", as_index=False).agg(
        typical_stay=("pooled_median_dwell_days", "first"), discarded=("best_restart_degenerate", "first"))
    ui.table(h1.assign(discarded=lambda d: d.discarded.map({True: "yes", False: "no"}))
             .rename(columns={"k": "States (K)", "typical_stay": "Typical stay in a regime (trading days, median)",
                              "discarded": "Best fit degenerate"}))
    st.caption("A richer version of the model, which also watched changes in the VIX, the yield curve and cross-asset "
               "correlation, was tried and dropped: at every K its states collapsed into one-or-two-day shocks. "
               "Details: DECISIONS.md D-024 and D-025.")

ui.how_to_read(
    f"""
* **Regime**: one of two market conditions the model distinguishes, *Calm* (low day-to-day variability) and *Volatile*.
  In every refit the higher-volatility state is labelled Volatile, so the labels mean the same thing throughout.
* **P(Volatile)**: the model's probability, on that day and using only data up to that day, that the market is in the
  Volatile regime. A day is shaded Volatile when it is above one half.
* **Transition matrix**: if today is Calm, the chance tomorrow is Calm or Volatile, and likewise from Volatile.
  **Expected stay** follows from it: a 98% chance of staying means a typical stay of about 50 days.
* **VIX**: the market's own gauge of expected volatility. The **VIX threshold** calls a day "high" when the VIX is above
  the middle of its past values; like the HMM, it is re-estimated every month on the past only.
* **Refit**: the model is re-estimated {hmm['refit_cadence']} on all history so far, which is why its volatility estimates
  drift over time.
* **Log-likelihood** and **BIC**: two scores of how well a model fits; BIC also charges for complexity.
"""
)
