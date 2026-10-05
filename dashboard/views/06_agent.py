"""Page 6, The agent. DASHBOARD.md §7.6 (static parts and the action-map interactive; no trained model is loaded here)."""

from __future__ import annotations

import numpy as np
import streamlit as st

from components import charts, data, theme, ui
from prism.env.actions import action_to_weights, upper_bounds

facts = ui.page_header("agent")
u, rules, agents, windows = facts["universe"], facts["rules"], facts["agents"], facts["windows"]
c = theme.colors()
lines = [*u["sectors"], *u["bonds"], *u["gold"], u["cash"]]
n_risky = len(lines) - 1

# --------------------------------------------------------------------------- how it decides
st.subheader("How the agent turns what it sees into weights")
obs = agents["blocks"]
s1, s2, s3, s4 = st.columns(4)
with s1.container(border=True, height="stretch"):
    st.markdown("**1 · State**")
    st.write(f"What it sees at the week's last close: {obs['V1']['observation_width']} to {obs['V4']['observation_width']} "
             "numbers, depending on the variant, including its own current weights.")
with s2.container(border=True, height="stretch"):
    st.markdown("**2 · Actor**")
    st.write(f"A small neural network (two layers of {agents['configs']['V4']['hidden'][0]} units) proposes a range of "
             f"plausible actions: {len(lines)} numbers between -1 and +1, one per holding.")
with s3.container(border=True, height="stretch"):
    st.markdown("**3 · Mean action**")
    st.write("While learning it samples from that range to explore. When it is evaluated it takes the middle of the "
             "range, so the same inputs always give the same answer.")
with s4.container(border=True, height="stretch"):
    st.markdown("**4 · Weights**")
    st.write(f"The {len(lines)} numbers are turned into positive weights that sum to 100%, then adjusted so no risky "
             f"asset exceeds {ui.pct(rules['cap'])}.")

st.markdown("**Try step 4.** Move a few of the action numbers and watch the weights. This is the environment's own "
            "rule; no trained agent is involved.")
picks = [lines[0], u["bonds"][-1], u["gold"][0], u["cash"]]
action = np.zeros(len(lines))
for col, name in zip(st.columns(len(picks)), picks):
    action[lines.index(name)] = col.slider(f"Action for {name}", -1.0, 1.0, 0.0, 0.05, key=f"action_{name}")
weights = action_to_weights(action, upper_bounds(n_risky, rules["cap"]), rules["logit_scale"])
st.plotly_chart(charts.action_weights(lines, weights, rules["cap"], c), config=theme.PLOT_CONFIG, key="action_weights")
st.caption(f"The other {len(lines) - len(picks)} actions are held at zero. With every action at zero, each holding gets an equal share "
           f"({ui.pct(1 / len(lines), 1)}). The rule is: multiply each action by {rules['logit_scale']:g}, apply a softmax, "
           "then move the result to the nearest weights that respect the cap.")

st.subheader("What a weight means")
st.write("A weight is the share of the portfolio the agent wants in that holding for the coming week. It is decided at "
         "the last close of the week and traded at the next close; between decisions the shares drift as prices move.")

# --------------------------------------------------------------------------- horizons
st.subheader("Two horizons")
st.write("The agent decides one week at a time, but it is trained to value rewards further out, discounted by a factor "
         "γ (gamma) each week. A rough planning horizon is 1 / (1 − γ) weeks. The γ for each variant was chosen on the "
         "validation split and then frozen.")
for col, v in zip(st.columns(len(agents["variants"])), agents["variants"]):
    cfg = agents["configs"][v]
    col.metric(f"{v} · γ = {cfg['gamma']:g}", f"≈ {cfg['horizon_weeks']:.0f} weeks", help="Planning horizon, 1 / (1 − γ), in weekly decisions.")

# --------------------------------------------------------------------------- turnover and costs
st.subheader("Turnover, and why it hurts")
LABEL = {"holdout": "Holdout (confirmatory)", "test": "Test split (exploratory)"}
window = st.segmented_control("Evaluation window", list(LABEL), format_func=LABEL.get, default="holdout", key="agent_window") or "holdout"
res, w = data.results(window), windows[window]
left, right = st.columns(2)
with left:
    st.markdown("**How much is traded each week**")
    st.plotly_chart(charts.turnover_bars(res["turnover"], c), config=theme.PLOT_CONFIG, key="turnover")
with right:
    st.markdown("**What trading costs do to the Sharpe ratio**")
    st.plotly_chart(charts.cost_lines(res["cost_sensitivity"], c), config=theme.PLOT_CONFIG, key="costs")
cost = res["cost_sensitivity"].set_index(["strategy", "bps"])
free, head = min(rules["cost_levels_bps"]), facts["statistics"]["headline_bps"]
drop = {v: (cost.loc[(v, free), "annualised_return"], cost.loc[(v, head), "annualised_return"]) for v in agents["variants"]}
worst = max(drop, key=lambda v: drop[v][0] - drop[v][1])
bench = [s for s in cost.index.get_level_values("strategy").unique() if s.startswith("BM|")]
bench_drop = max(cost.loc[(s, free), "annualised_return"] - cost.loc[(s, head), "annualised_return"] for s in bench)
st.markdown(
    f"""
On this window the agents traded **{ui.pct_range(w['turnover_agents'])}** of the portfolio at each weekly decision; the
benchmarks traded **{ui.pct_range(w['turnover_benchmarks'], 1)}**. The policy re-weights every week in response to inputs
that are mostly noise from one week to the next, and every re-weighting is paid for. Going from no costs to the
{head:g} basis points used for the headline results takes {worst}'s annual return from {ui.pct(drop[worst][0], 1)} to
{ui.pct(drop[worst][1], 1)}; no benchmark loses more than {ui.pct(bench_drop, 1)} a year to the same costs.
"""
)
bench_names = {"BM|EqualWeight": "equal weight", "BM|SixtyForty": "60/40", "BM|RiskParity": "risk parity"}
best_agent_free = max(cost.loc[(v, free), "sharpe"] for v in agents["variants"])
beaten_free = [n for b, n in bench_names.items() if cost.loc[(b, free), "sharpe"] < best_agent_free]
best_agent_head = max(cost.loc[(v, head), "sharpe"] for v in agents["variants"])
beaten_head = [n for b, n in bench_names.items() if cost.loc[(b, head), "sharpe"] < best_agent_head]
describe = lambda names: "none of equal weight, 60/40 and risk parity" if not names else ", ".join(names)  # noqa: E731
st.caption(
    f"Read from the stored cost table for this window. With no costs at all, the best variant's average Sharpe ratio "
    f"({best_agent_free:.2f}) was above: {describe(beaten_free)}. At {head:g} basis points ({best_agent_head:.2f}) it was above: "
    f"{describe(beaten_head)}. These are point estimates with wide intervals, not tested differences; they show that costs "
    "account for part of the gap to the benchmarks, not necessarily all of it."
)
with st.expander("The numbers behind these two charts"):
    t = res["turnover"].assign(strategy=lambda d: d.variant.map(charts.strategy_label))
    ui.table(t[["strategy", "mean_turnover_per_step"]].rename(columns={"mean_turnover_per_step": "mean one-way turnover per decision"})
             .style.format({"mean one-way turnover per decision": "{:.1%}"}))
    st.caption("Net Sharpe ratio by cost level (basis points per side), as in the final report:")
    table = data.headline_tables(window)["costs"]
    ui.table(table.rename(index=charts.strategy_label, columns=lambda b: f"{b:g} bps").rename_axis("strategy").style.format("{:.2f}"), index=True)

# --------------------------------------------------------------------------- learning curves
st.subheader("Learning curves: the overfitting picture")
curves = data.learning_curves()
st.plotly_chart(charts.learning_curves(curves, c), config=theme.PLOT_CONFIG, key="curves")
last = curves[curves.step == curves.step.max()]
st.markdown(
    f"""
Average over the {agents['seeds']} seeds of each variant, checked every {agents['eval_every']:,} training steps. On the
{agents['train_decisions']} weekly decisions it trains on, the agent's result keeps rising, to about
{last.train_mean_log_return.mean():+.3f} per decision by the end: it is memorising its training data. On the
{agents['val_decisions']} validation decisions it has not seen, the result stays flat, at or slightly below zero
({last.val_mean_log_return.min():+.4f} to {last.val_mean_log_return.max():+.4f} at the end). Both panels share one scale.
That gap is why each agent is kept at the checkpoint that did best on validation, not at the end of training.
"""
)

ui.how_to_read(
    f"""
* **Turnover** is the share of the portfolio that changes hands at a decision, counted one way (moving 10% of the
  portfolio from one asset to another counts as 10%).
* **Basis point (bps)**: one hundredth of one percent. The headline results use {head:g} bps per side; the cost chart
  re-runs the same decisions at {', '.join(f'{b:g}' for b in rules['cost_levels_bps'])} bps.
* **Sharpe ratio**: average return divided by its variability, annualised. Higher is better; it lets a calm strategy
  and a volatile one be compared.
* **Coloured lines and bars** are the four variants (each the average of its {agents['seeds']} seeds). **Grey** is the six
  simple benchmarks; hover to see which is which.
* **Log net return per decision**: roughly the percentage gained in one week after costs (0.001 is about 0.1%).
* The agents were trained with the reward multiplied by {agents['reward_scale']:g} to help the algorithm learn; every
  number shown here is on the original scale.
"""
)
