"""Page 10, Results (pre-registered). DASHBOARD.md §7.10: the stored tables and the headline forest plot.

The interactive parts (metric and block-length selectors, per-seed plots, equity curves, the cost slider) are milestone D4.
"""

from __future__ import annotations

import streamlit as st

from components import charts, data, theme, ui

facts = ui.page_header("results")
stats, agents, windows = facts["statistics"], facts["agents"], facts["windows"]
c = theme.colors()
LABEL = {"holdout": "Holdout (confirmatory, one use)", "test": "Test split (exploratory)"}
res = {w: data.results(w) for w in LABEL}
tables = {w: data.headline_tables(w) for w in LABEL}
n_cmp = len(agents["comparisons"])

# --------------------------------------------------------------------------- headline
k1, k2, k3, k4 = st.columns(4)
k1.metric("Comparisons passed, holdout", f"{windows['holdout']['comparisons_passing']} of {n_cmp}")
k2.metric("Comparisons passed, test split", f"{windows['test']['comparisons_passing']} of {n_cmp}")
k3.metric("Results that can be claimed", f"{windows['holdout']['claimable'] + windows['test']['claimable']}",
          help="A claim needs a passed comparison and a deflated Sharpe ratio of at least "
               f"{stats['dsr_bar']:g} for the candidate.")
k4.metric(f"Agents above the {stats['dsr_bar']:g} deflated-Sharpe bar",
          f"{windows['holdout']['seeds_ge_dsr_bar']} of {agents['n_agents']}", help="On the holdout. The test split gives the same count: "
          f"{windows['test']['seeds_ge_dsr_bar']} of {agents['n_agents']}.")
st.markdown(
    f"**The rule, fixed before any result was seen.** A comparison passes only if the candidate is reliably better than its "
    f"control (the whole {stats['ci_level']:.0%} interval of the difference above zero) on at least 3 of the 4 measures, and "
    "reliably worse on none. An interval that contains zero counts as *indeterminate*, not as a pass."
)

# --------------------------------------------------------------------------- forest
st.subheader("The three comparisons, on both windows")
metric = "sharpe"
st.plotly_chart(charts.forest(res["test"]["paired_block20"], res["holdout"]["paired_block20"], metric, c),
                config=theme.PLOT_CONFIG, key="forest")
flips = facts["sign_flips"][metric]
flipped = [f"{f['candidate']} vs {f['control']}" for f in flips if f["flipped"]]
st.markdown(
    f"""
Each mark is the difference in Sharpe ratio between a variant and its control, with its {stats['ci_level']:.0%} interval.
**Every interval crosses zero, on both windows.** For {len(flipped)} of the {n_cmp} comparisons ({', '.join(flipped)}) the
point estimate also **changed sign** between the test split and the holdout. That is what noise looks like: when the
interval contains zero and the sign will not hold still, the point estimate carries no information about direction.
"""
)

# --------------------------------------------------------------------------- per window tables
st.subheader("The stored tables")
for tab, w in zip(st.tabs(list(LABEL.values())), LABEL):
    with tab:
        m, t = windows[w], tables[w]
        st.caption(f"Daily returns from {m['first']} to {m['last']} ({m['n_sessions']} trading days); the same {agents['n_agents']} "
                   f"frozen agents; costs of {stats['headline_bps']:g} basis points per side plus slippage."
                   + (" These years had been looked at in Tier 1 before the agents were built, so this window is exploratory."
                      if w == "test" else " Evaluated exactly once."))
        st.markdown("**Verdict for each comparison**")
        ui.table(t["gates"].assign(claimable=lambda d: d.claimable.map({True: "yes", False: "no"})))
        st.caption(f"*claimable* is the last column: it needs the paired rule to pass and the candidate's median deflated Sharpe "
                   f"ratio to reach {stats['dsr_bar']:g}. *spec non-overlap* is a stricter version of the rule, reported beside it.")
        st.markdown("**Every difference, candidate minus control (larger is better on every measure)**")
        ui.table(t["differences"])
        st.markdown(f"**The variants (average of {agents['seeds']} seeds ± the spread between seeds) and the benchmarks**")
        ui.table(t["variants"].assign(variant=lambda d: d.variant.map(lambda s: charts.strategy_label(s.replace("BM ", "BM|")))))
        lo, hi = m["seed_sharpe_std"]
        gaps = [abs(f[w]) for f in flips]
        st.caption(f"Seed noise against the effect: across seeds of one variant the Sharpe ratio varies by {lo:.2f} to {hi:.2f} "
                   f"(standard deviation), while the three between-variant differences are {min(gaps):.2f} to {max(gaps):.2f}. "
                   "The last six rows are the simple benchmarks, run through the same environment and costs.")
        st.markdown("**Net Sharpe ratio by cost level (basis points per side)**")
        ui.table(t["costs"].rename(index=charts.strategy_label, columns=lambda b: f"{b:g}").rename_axis("strategy").style.format("{:.2f}"), index=True)
        st.caption(f"The agents trade {ui.pct_range(m['turnover_agents'])} of the portfolio a week against "
                   f"{ui.pct_range(m['turnover_benchmarks'], 1)} for the benchmarks, so their results fall quickly as costs rise. "
                   "The Agent page shows this as a chart.")
        st.markdown("**Deflated Sharpe ratio**")
        dsr = res[w]["dsr"].copy()
        dsr["variant"] = dsr["variant"].map(charts.strategy_label)
        n40, n136 = stats["dsr_trials"]
        ui.table(
            dsr.rename(columns={"variant": "strategy", "dsr_median_n40": f"median ({n40} trials)", "dsr_min_n40": f"lowest seed ({n40} trials)",
                                "seeds_ge_0.95_n40": f"seeds at or above {stats['dsr_bar']:g}", "dsr_median_n136": f"median ({n136} trials)",
                                "dsr_min_n136": f"lowest seed ({n136} trials)"}).drop(columns=["seeds_ge_0.95_n136"])
            .style.format(lambda x: "" if x != x else f"{x:.3f}", subset=[f"median ({n40} trials)", f"lowest seed ({n40} trials)",
                                                                           f"median ({n136} trials)", f"lowest seed ({n136} trials)"])
            .format(lambda x: "" if x != x else f"{x:.0f}", subset=[f"seeds at or above {stats['dsr_bar']:g}"])
        )
        st.caption(f"The highest median among the variants is {m['dsr_median_max_variant']:.2f}; the bar is {stats['dsr_bar']:g}. "
                   "A benchmark is a single series, so it has one value for each trial count and no seed columns.")

ui.how_to_read(
    f"""
* **Candidate minus control**: for *V4 vs V2*, V4 is the candidate. A positive difference favours the candidate.
* **{stats['ci_level']:.0%} interval**: the range of differences consistent with the data, from re-sampling both the days
  (in blocks of about {stats['block']} trading days, {stats['n_bootstrap']:,} times) and the seeds. If it contains zero,
  the data cannot tell the two variants apart.
* **The four measures**: annualised return, Sharpe ratio (return per unit of variability), maximum drawdown (the worst
  peak-to-trough fall) and CVaR 95% (the average of the worst 5% of days). All are oriented so that larger is better.
* **favourable / adverse / indeterminate**: the interval is entirely above zero, entirely below, or contains it.
* **Deflated Sharpe ratio**: the probability that a Sharpe ratio is genuinely above what the best of many tries would
  reach by luck alone. {stats['dsr_trials'][0]} tries are counted (every final agent); counting all
  {stats['dsr_trials'][1]} training runs is shown as a check. {stats['dsr_bar']:g} is the bar.
* **Exploratory vs confirmatory**: the test split had been seen before the agents were designed; the holdout had not,
  and was used once. The holdout is the one that decides.
* The forest plot shows the Sharpe ratio; the other three measures are in the second table of each tab.
"""
)
ui.verified_footer()
