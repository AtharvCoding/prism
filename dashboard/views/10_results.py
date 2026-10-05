"""Page 10, Results (pre-registered). DASHBOARD.md §7.10.

Every number is a stored one. The selectors choose which stored table or series is drawn; the cost slider interpolates
between the four stored cost levels and says so. Nothing here is recomputed for the verdict.
"""

from __future__ import annotations

import streamlit as st

from components import alloc, charts, data, theme, ui
from prism import dashboard_data as dd

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
f1, f2 = st.columns([3, 2])
metric = f1.segmented_control("Measure", stats["metrics"], format_func=lambda m: charts.METRIC_LABEL[m][0].upper() + charts.METRIC_LABEL[m][1:], default="sharpe",
                              key="forest_metric") or "sharpe"
blocks = sorted({stats["block"], *stats["block_sensitivity"]})
block = f2.segmented_control("Bootstrap block length (trading days)", blocks, default=stats["block"], key="forest_block",
                             help=f"{stats['block']} is the pre-registered setting; the others are its pre-registered sensitivity checks.") or stats["block"]
paired = {w: res[w][f"paired_block{block}"] for w in LABEL}
st.plotly_chart(charts.forest(paired["test"], paired["holdout"], metric, c), config=theme.PLOT_CONFIG, key="forest")
flips = dd.sign_flips(paired["test"], paired["holdout"], metric).to_dict("records")
flipped = [f"{f['candidate']} vs {f['control']}" for f in flips if f["flipped"]]
crossing = sum(int(r.ci_low < 0 < r.ci_high) for w in LABEL for r in paired[w][paired[w].metric == metric].itertuples())
st.markdown(
    f"""
Each mark is the difference in {charts.METRIC_LABEL[metric]} between a variant and its control, with its {stats['ci_level']:.0%}
interval. **{crossing} of the {2 * n_cmp} intervals shown cross zero.** """
    + (f"For {len(flipped)} of the {n_cmp} comparisons ({', '.join(flipped)}) the point estimate also **changed sign** between the "
       "test split and the holdout. That is what noise looks like: when the interval contains zero and the sign will not "
       "hold still, the point estimate carries no information about direction." if flipped else
       "No point estimate changed sign between the two windows on this measure; with every interval containing zero, "
       "that agreement in sign is not evidence of an effect.")
)

# --------------------------------------------------------------------------- seeds against the effect
st.subheader("Seed noise against the size of the effect")
window = st.segmented_control("Evaluation window", list(LABEL), format_func=LABEL.get, default="holdout", key="results_window") or "holdout"
sm, bm, m = data.seed_metrics(window), data.benchmark_metrics(window), windows[window]
left, right = st.columns(2)
with left:
    st.markdown("**Sharpe ratio of each of the 40 agents**")
    st.plotly_chart(charts.seed_strip({v: sm[sm.variant == v].sharpe.to_numpy() for v in agents["variants"]},
                                      {f"BM|{k}": float(v) for k, v in bm.sharpe.items()}, "net Sharpe ratio", c),
                    config=theme.PLOT_CONFIG, key="seed_strip")
with right:
    st.markdown("**The same thing as sizes**")
    band = res[window]["seed_band"]
    spread = {v: float(band[(band.variant == v) & (band.metric == "sharpe")]["std"].iloc[0]) for v in agents["variants"]}
    p20 = res[window][f"paired_block{stats['block']}"]
    gaps = {f"{r.candidate} vs {r.control}": abs(float(r.diff)) for r in p20[p20.metric == "sharpe"].itertuples()}
    st.plotly_chart(charts.size_bars(spread, gaps, c), config=theme.PLOT_CONFIG, key="size_bars")
st.caption(f"Left: one dot per trained agent, the black line the average of its variant, the grey ticks the six benchmarks. Right: grey "
           f"bars are how much agents of the *same* variant differ from each other (standard deviation, {min(spread.values()):.2f} to "
           f"{max(spread.values()):.2f}); black bars are the differences *between* variants ({min(gaps.values()):.2f} to {max(gaps.values()):.2f}). "
           "Re-running the same recipe with another random seed moves the result more than changing the recipe does.")

st.markdown("**Deflated Sharpe ratio of each agent**")
sd = data.seed_dsr(window)
n40 = stats["dsr_trials"][0]
st.plotly_chart(charts.seed_strip({v: sd[sd.strategy == v][f"dsr_n{n40}"].to_numpy() for v in agents["variants"]},
                                  {r.strategy: float(getattr(r, f"dsr_n{n40}")) for r in sd[sd.seed < 0].itertuples()},
                                  "deflated Sharpe ratio", c, bar=stats["dsr_bar"]), config=theme.PLOT_CONFIG, key="dsr_strip")
st.caption(f"The deflated Sharpe ratio asks how likely a result is to be more than the luck of the best of {n40} tries. {stats['dsr_bar']:g} "
           f"is the bar. On this window {m['seeds_ge_dsr_bar']} of {agents['n_agents']} agents clear it; the highest agent is at "
           f"{sd[sd.seed >= 0][f'dsr_n{n40}'].max():.2f}.")

# --------------------------------------------------------------------------- equity curves
st.subheader("Growth of 1, against the benchmarks")
e1, e2, e3 = st.columns([2, 3, 3])
shown = e1.pills("Variants", agents["variants"], selection_mode="multi", default=["V4"], key="equity_variants") or []
bench_all = list(bm.index)
bench_shown = e2.pills("Benchmarks", bench_all, selection_mode="multi", default=[b for b in ("EqualWeight", "SixtyForty", "RiskParity") if b in bench_all],
                       format_func=charts.strategy_label, key="equity_benchmarks") or []
ep = res[window]["episodes"].drop_duplicates("episode")
zooms = {"Whole window": None} | {f"{e.peak} to {e.recovery if isinstance(e.recovery, str) else m['last']} (S&P 500 fell {ui.pct(abs(e.depth))})":
                                  (e.peak, e.recovery if isinstance(e.recovery, str) else m["last"]) for e in ep.itertuples()}
zoom = e3.selectbox("Zoom to a drawdown episode", list(zooms), key="equity_zoom")
seeds = sorted(sm.seed.unique().tolist())
names = [f"{v}|s{s}" for v in agents["variants"] for s in seeds] + [f"BM|{b}" for b in bench_all]
curves = alloc.equity_curves(data.eval_daily(window), names, stats["headline_bps"])
st.plotly_chart(charts.equity_chart(curves, list(shown), seeds, list(bench_shown), c, zooms[zoom]), config=theme.PLOT_CONFIG, key="equity")
st.caption(f"Cumulated from the stored daily returns at {stats['headline_bps']:g} basis points per side. For a variant, the line is the "
           f"median of its {len(seeds)} agents on each day and the shaded band runs from its lowest agent to its highest. Episodes are the "
           "S&P 500's falls of 10% or more inside this window, from peak to recovery.")

# --------------------------------------------------------------------------- cost slider
st.subheader("Ranking by net Sharpe ratio as trading gets more expensive")
levels = sorted(res[window]["cost_sensitivity"].bps.unique().tolist())
bps = st.slider("Cost per side (basis points)", float(levels[0]), float(levels[-1]), float(stats["headline_bps"]), 0.5, key="cost_bps")
ranked = alloc.interpolate_costs(res[window]["cost_sensitivity"], bps)
st.plotly_chart(charts.cost_rank(ranked, c), config=theme.PLOT_CONFIG, key="cost_rank")
exact = bps in levels
best_variant = next(k for k in ranked.index if k in agents["variants"])
st.caption((f"Stored values at {bps:g} basis points. " if exact else
            f"**Interpolated**: {bps:g} basis points is between two stored cost levels ({', '.join(f'{x:g}' for x in levels)}), and the bars "
            "are a straight line between the stored values on either side, not a re-run. ")
           + f"Coloured bars are the variants (average of seeds), grey the benchmarks. The highest-ranked variant here, {best_variant}, is "
           f"number {list(ranked.index).index(best_variant) + 1} of {len(ranked)}.")

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
* **Block length**: returns on neighbouring days are related, so the re-sampling draws blocks of days, not single days.
  {stats['block']} was fixed in advance; the other lengths show the verdict does not depend on that choice.
* **Seed**: one training run of a variant. Each variant was trained {agents['seeds']} times.
* **Growth of 1**: what one unit invested at the start of the window would be worth, after costs.
"""
)
ui.verified_footer()
