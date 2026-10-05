"""Chart builders: plain functions from stored frames to Plotly figures. No statistic is computed here.

Every figure takes the palette ``c`` from :mod:`components.theme`, so light and dark are both selected, not
flipped. A chart with two or more series has a legend; the page shows the table the chart was drawn from.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components import theme

SPLIT_LABEL = {"train": "Train", "val": "Validation", "test": "Test (exploratory)", "holdout": "Holdout (spent)"}
COMPARISON_LABEL = {("V4", "V2"): "V4 vs V2<br>HMM beyond the LSTM", ("V4", "C4"): "V4 vs C4<br>HMM vs a VIX threshold",
                    ("V2", "V1"): "V2 vs V1<br>the LSTM latent"}
METRIC_LABEL = {"annualised_return": "annualised return", "sharpe": "Sharpe ratio", "max_drawdown": "maximum drawdown",
                "cvar_95": "CVaR 95%"}
BENCHMARK_LABEL = {"EqualWeight": "Equal weight", "SixtyForty": "60/40", "MinVariance": "Minimum variance",
                   "RiskParity": "Risk parity", "VolTarget": "Vol target", "BuyHoldSPY": "Buy and hold SPY"}


def strategy_label(name: str) -> str:
    return BENCHMARK_LABEL.get(name.removeprefix("BM|"), name)


def _span(fig: go.Figure, row: str, start: str, end: str, *, color: str, name: str, legend: bool, hover: str) -> None:
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    fig.add_trace(go.Bar(
        y=[row], x=[(b - a) / pd.Timedelta(milliseconds=1)], base=[a], orientation="h", width=0.42, name=name,
        marker={"color": color, "line": {"width": 0}}, showlegend=legend, hovertemplate=hover + "<extra></extra>",
    ))


def timeline(facts: dict[str, Any], c: dict[str, Any]) -> go.Figure:
    """Universe A, Universe B and the four splits on one time axis; the embargo shows as the gaps between splits."""
    u, splits = facts["universe"], facts["splits"]
    end = splits["holdout"]["declared"][1]
    rows = ("Splits (Universe B)", "Universe B: allocation", "Universe A: model fitting")
    fig = go.Figure()
    _span(fig, rows[2], u["a_start"], end, color=c["context"], name="Data available", legend=True,
          hover=f"Universe A<br>{u['a_start']} to {end}")
    _span(fig, rows[1], u["b_start"], end, color=c["context"], name="Data available", legend=False,
          hover=f"Universe B<br>{u['b_start']} to {end}")
    for (key, label), color in zip(SPLIT_LABEL.items(), c["ordinal"]):
        s = splits[key]
        _span(fig, rows[0], s["effective"][0], s["effective"][1], color=color, name=label, legend=True,
              hover=f"{label}<br>{s['effective'][0]} to {s['effective'][1]}<br>{s['sessions']} sessions")
    _span(fig, rows[2], u["fit_early"][0], u["fit_early"][1], color=c["muted"], name="First model fit", legend=True,
          hover=f"First HMM and encoder fit<br>{u['fit_early'][0]} to {u['fit_early'][1]}")
    fig.add_annotation(x=u["fit_early"][0], y=rows[2], yshift=26, xanchor="left", showarrow=False, font={"color": c["ink_2"], "size": 12},
                       text=f"first HMM and encoder fit: {u['fit_early'][0][:4]} to {u['fit_early'][1][:4]}")
    fig.update_layout(barmode="overlay", bargap=0.3)
    fig.update_yaxes(categoryorder="array", categoryarray=list(rows), showgrid=False)
    fig.update_xaxes(type="date")
    return theme.style(fig, c, height=250)


def action_weights(lines: list[str], weights: np.ndarray, cap: float, c: dict[str, Any]) -> go.Figure:
    """The 14 weights the environment's action map returns for one action vector."""
    fig = go.Figure(go.Bar(
        x=lines, y=weights, marker={"color": c["accent"], "line": {"width": 0}}, width=0.5,
        text=[f"{100 * w:.0f}%" for w in weights], textposition="outside", textfont={"color": c["ink_2"], "size": 12},
        hovertemplate="%{x}: %{y:.1%}<extra></extra>", cliponaxis=False,
    ))
    fig.add_hline(y=cap, line={"color": c["muted"], "width": 1}, annotation_text=f"cap on a risky asset: {cap:.0%}",
                  annotation_position="top left", annotation_font={"color": c["ink_2"], "size": 12})
    fig.update_yaxes(tickformat=".0%", range=[0, max(cap * 1.25, float(weights.max()) * 1.15)], title_text="weight")
    return theme.style(fig, c, height=320, legend=False)


def turnover_bars(turnover: pd.DataFrame, c: dict[str, Any]) -> go.Figure:
    """Mean one-way turnover per weekly decision: the four variants in their colours, the benchmarks in grey."""
    t = turnover.sort_values("mean_turnover_per_step")
    names = [strategy_label(v) for v in t.variant]
    cols = [c["variants"].get(v, c["context"]) for v in t.variant]
    fig = go.Figure(go.Bar(
        y=names, x=t.mean_turnover_per_step, orientation="h", width=0.55, marker={"color": cols, "line": {"width": 0}},
        text=[f"{100 * x:.1f}%" for x in t.mean_turnover_per_step], textposition="outside",
        textfont={"color": c["ink_2"], "size": 12}, cliponaxis=False, hovertemplate="%{y}: %{x:.1%} a week<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%", title_text="portfolio traded per weekly decision (one-way)",
                     range=[0, float(t.mean_turnover_per_step.max()) * 1.15])
    fig.update_yaxes(showgrid=False)
    return theme.style(fig, c, height=340, legend=False)


def cost_lines(cost: pd.DataFrame, c: dict[str, Any]) -> go.Figure:
    """Net Sharpe at each stored cost level. Variants in colour (mean over seeds), benchmarks in grey."""
    fig = go.Figure()
    bench = [s for s in cost.strategy.unique() if s.startswith("BM|")]
    for i, s in enumerate(bench):
        g = cost[cost.strategy == s]
        fig.add_trace(go.Scatter(
            x=g.bps, y=g.sharpe, mode="lines", name="Benchmarks", legendgroup="bm", showlegend=i == 0,
            line={"color": c["context"], "width": 1.5}, hovertemplate=f"{strategy_label(s)}<br>%{{x:g}} bps: %{{y:.2f}}<extra></extra>",
        ))
    for v in theme.VARIANTS:
        g = cost[cost.strategy == v]
        fig.add_trace(go.Scatter(
            x=g.bps, y=g.sharpe, mode="lines+markers", name=v,
            line={"color": c["variants"][v], "width": 2, "dash": theme.VARIANT_DASH[v]},
            marker={"size": 9, "symbol": theme.VARIANT_SYMBOL[v], "line": {"color": c["surface"], "width": 2}},
            hovertemplate=f"{v} (mean of seeds)<br>%{{x:g}} bps: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_xaxes(title_text="cost per side (basis points); slippage scales with it", tickvals=sorted(cost.bps.unique()))
    fig.update_yaxes(title_text="net Sharpe ratio")
    return theme.style(fig, c, height=360)


def learning_curves(curves: pd.DataFrame, c: dict[str, Any]) -> go.Figure:
    """Mean per-decision log net return at each checkpoint, train and validation on one shared scale."""
    panels = (("train_mean_log_return", "Train split (data the agent learns from)"), ("val_mean_log_return", "Validation split (unseen)"))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[p[1] for p in panels], horizontal_spacing=0.04)
    for col, (column, _) in enumerate(panels, start=1):
        for v in theme.VARIANTS:
            g = curves[curves.variant == v]
            fig.add_trace(go.Scatter(
                x=g.step, y=g[column], mode="lines", name=v, legendgroup=v, showlegend=col == 1,
                line={"color": c["variants"][v], "width": 2, "dash": theme.VARIANT_DASH[v]},
                hovertemplate=f"{v}<br>step %{{x:,}}: %{{y:+.4f}}<extra></extra>",
            ), row=1, col=col)
    fig.update_xaxes(title_text="environment steps", tickformat="~s")
    fig.update_yaxes(title_text="mean log net return per decision", tickformat="+.3f", row=1, col=1)
    fig.update_annotations(font={"size": 13, "color": c["ink_2"]})
    fig = theme.style(fig, c, height=370)
    return fig.update_layout(margin={"t": 64}, legend={"y": 1.14})


def forest(test: pd.DataFrame, holdout: pd.DataFrame, metric: str, c: dict[str, Any]) -> go.Figure:
    """Paired difference (candidate minus control) with its stored 95% interval, test split beside holdout."""
    pairs = list(COMPARISON_LABEL)
    fig = go.Figure()
    fmt = ".2f" if metric == "sharpe" else ".1%"
    windows = (("Test split (exploratory)", test, 0.16, c["muted"], "circle-open"),
               ("Holdout (confirmatory, one use)", holdout, -0.16, c["ink"], "circle"))
    for name, table, offset, color, symbol in windows:
        rows = [table[(table.candidate == x) & (table.control == y) & (table.metric == metric)].iloc[0] for x, y in pairs]
        diff = np.array([r["diff"] for r in rows])
        lo, hi = np.array([r["ci_low"] for r in rows]), np.array([r["ci_high"] for r in rows])
        fig.add_trace(go.Scatter(
            x=diff, y=np.arange(len(pairs))[::-1] + offset, mode="markers", name=name,
            marker={"size": 11, "color": color, "symbol": symbol, "line": {"color": color, "width": 2}},
            error_x={"type": "data", "symmetric": False, "array": hi - diff, "arrayminus": diff - lo, "color": color,
                     "thickness": 2, "width": 5},
            customdata=np.stack([lo, hi], axis=1),
            hovertemplate=f"{name}<br>difference %{{x:{fmt}}}<br>95% interval %{{customdata[0]:{fmt}}} to %{{customdata[1]:{fmt}}}<extra></extra>",
        ))
    fig.add_vline(x=0, line={"color": c["ink_2"], "width": 1})
    fig.update_yaxes(tickvals=list(range(len(pairs)))[::-1], ticktext=[COMPARISON_LABEL[p] for p in pairs], showgrid=False,
                     range=[-0.6, len(pairs) - 0.4], tickfont={"color": c["ink_2"], "size": 13})
    fig.update_xaxes(title_text=f"difference in {METRIC_LABEL[metric]}, candidate minus control (right of zero favours the candidate)",
                     tickformat=fmt)
    return theme.style(fig, c, height=330)


# --------------------------------------------------------------------------- regimes (page 4)
def _runs(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, bool]]:
    """Consecutive runs of a boolean daily series as (first day, last day, value)."""
    change = flag.ne(flag.shift()).cumsum()
    return [(g.index[0], g.index[-1], bool(g.iloc[0])) for _, g in flag.groupby(change)]


def _legend_key(fig: go.Figure, name: str, color: str, opacity: float = 1.0) -> None:
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, hoverinfo="skip",
                             marker={"size": 11, "symbol": "square", "color": color, "opacity": opacity}))


def regime_price(spy: pd.Series, regimes: pd.DataFrame, fit_early: list[str], recessions: dict[str, Any] | None,
                 c: dict[str, Any]) -> go.Figure:
    """The S&P 500 (indexed, log scale) with each day shaded by the regime the HMM assigned to it at the time."""
    day = pd.Timedelta(days=1)
    shapes = [{"type": "rect", "xref": "x", "yref": "paper", "x0": fit_early[0], "x1": fit_early[1], "y0": 0, "y1": 1,
               "fillcolor": c["muted"], "opacity": 0.14, "line": {"width": 0}, "layer": "below"}]
    for start, end, volatile in _runs(regimes.p_volatile > 0.5):
        shapes.append({"type": "rect", "xref": "x", "yref": "paper", "x0": start, "x1": end + day, "y0": 0, "y1": 1,
                       "fillcolor": c["volatile"] if volatile else c["calm"], "opacity": 0.30 if volatile else 0.10,
                       "line": {"width": 0}, "layer": "below"})
    fig = go.Figure(go.Scatter(x=spy.index, y=spy, mode="lines", name="S&P 500 (indexed to 100)",
                               line={"color": c["ink"], "width": 1.5}, hovertemplate="%{x|%d %b %Y}<br>%{y:.0f}<extra></extra>"))
    _legend_key(fig, "Calm", c["calm"], 0.45)
    _legend_key(fig, "Volatile", c["volatile"], 0.6)
    _legend_key(fig, "First fit window (no regime estimate)", c["muted"], 0.5)
    if recessions:
        for r in recessions.values():
            shapes.append({"type": "rect", "xref": "x", "yref": "paper", "x0": r["peak"], "x1": r["trough"], "y0": 0, "y1": 0.035,
                           "fillcolor": c["ink"], "opacity": 0.85, "line": {"width": 0}})
        _legend_key(fig, "NBER recession", c["ink"], 0.85)
    fig.update_layout(shapes=shapes)
    fig.update_yaxes(type="log", title_text="S&P 500, indexed (log scale)", showgrid=False)
    fig.update_xaxes(showgrid=False, rangeslider={"visible": True, "thickness": 0.07},
                     rangeselector={"buttons": [{"count": 2, "label": "2y", "step": "year", "stepmode": "backward"},
                                                {"count": 5, "label": "5y", "step": "year", "stepmode": "backward"},
                                                {"count": 10, "label": "10y", "step": "year", "stepmode": "backward"},
                                                {"step": "all", "label": "All"}],
                                    "bgcolor": "rgba(0,0,0,0)", "activecolor": c["grid"], "font": {"color": c["ink_2"]}, "y": 1.12})
    fig = theme.style(fig, c, height=460)
    return fig.update_layout(margin={"t": 70}, legend={"y": 1.02, "x": 0.22})


def regime_strips(regimes: pd.DataFrame, c: dict[str, Any]) -> go.Figure:
    """Two rows on one calendar: the HMM's call and the VIX-threshold state, each day Calm or Volatile / low or high."""
    f = regimes.dropna(subset=["vix_high"])
    z = np.vstack([(f.vix_high == 1).to_numpy(dtype=float), (f.p_volatile > 0.5).to_numpy(dtype=float)])
    text = np.vstack([np.where(z[0] == 1, "VIX high", "VIX low"), np.where(z[1] == 1, "Volatile", "Calm")])
    fig = go.Figure(go.Heatmap(
        x=f.index, y=["VIX threshold", "HMM"], z=z, text=text, zmin=0, zmax=1, showscale=False, ygap=8, opacity=0.8,
        colorscale=[[0, c["calm"]], [1, c["volatile"]]], hovertemplate="%{x|%d %b %Y}<br>%{y}: %{text}<extra></extra>",
    ))
    _legend_key(fig, "Calm / VIX low", c["calm"])
    _legend_key(fig, "Volatile / VIX high", c["volatile"])
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    return theme.style(fig, c, height=190)


def fold_volatility(folds: pd.DataFrame, c: dict[str, Any]) -> go.Figure:
    """Each monthly refit's estimate of the two states' volatility, annualised."""
    fig = go.Figure()
    for i, (name, color) in enumerate((("Calm", c["calm"]), ("Volatile", c["volatile"]))):
        fig.add_trace(go.Scatter(x=folds.apply_start, y=folds[f"vol_{i}"] * np.sqrt(252), mode="lines", name=f"{name} state",
                                 line={"color": color, "width": 2}, hovertemplate=f"{name}, fold applied from %{{x|%b %Y}}<br>%{{y:.1%}} a year<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title_text="state volatility, annualised", rangemode="tozero")
    fig.update_xaxes(title_text="month the refit model was applied to")
    return theme.style(fig, c, height=320)


def k_selection(table: pd.DataFrame, margin: float, c: dict[str, Any]) -> go.Figure:
    """Validation log-likelihood and BIC for each number of states; K values whose fits were degenerate are greyed."""
    usable = table.restarts_degenerate < table.restarts
    colors = [c["accent"] if s else (c["ink_2"] if u else c["context"]) for s, u in zip(table.selected, usable)]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=["Validation log-likelihood (higher is better)", "BIC (lower is better)"])
    for col, (column, fmt) in enumerate((("val_loglik", ".2f"), ("bic", ",.0f")), start=1):
        fig.add_trace(go.Bar(x=table.k, y=table[column], marker={"color": colors, "line": {"width": 0}}, width=0.45, showlegend=False,
                             text=[f"{v:{fmt}}" for v in table[column]], textposition="outside", cliponaxis=False,
                             textfont={"color": c["ink_2"], "size": 11}, hovertemplate=f"K = %{{x}}<br>%{{y:{fmt}}}<extra></extra>"),
                      row=1, col=col)
        lo, hi = float(table[column].min()), float(table[column].max())
        pad = (hi - lo) * 0.35
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=col)
    _legend_key(fig, "Selected", c["accent"])
    _legend_key(fig, "Usable, not selected", c["ink_2"])
    _legend_key(fig, "Every restart degenerate: discarded", c["context"])
    fig.update_xaxes(title_text="number of states, K", tickvals=list(table.k))
    fig.update_annotations(font={"size": 13, "color": c["ink_2"]})
    fig = theme.style(fig, c, height=330)
    return fig.update_layout(margin={"t": 70}, legend={"y": 1.16})


# --------------------------------------------------------------------------- latent map (page 5)
def latent_map(points: pd.DataFrame, c: dict[str, Any]) -> go.Figure:
    """One encoder fold's latents on its own two principal components, coloured by P(Volatile); play to move through the year."""
    months = sorted(points.index.to_period("M").unique())
    marker = {"size": 8, "colorscale": theme.regime_scale(c), "cmin": 0, "cmax": 1, "line": {"color": c["surface"], "width": 1},
              "colorbar": {"title": {"text": "P(Volatile)", "side": "right"}, "thickness": 12, "len": 0.8, "tickvals": [0, 0.5, 1],
                           "ticktext": ["0 Calm", "0.5", "1 Volatile"]}}

    def trace(upto) -> go.Scatter:  # noqa: ANN001
        shown = points[points.index.to_period("M") <= upto]
        return go.Scatter(x=shown.pc1, y=shown.pc2, mode="markers", marker={**marker, "color": shown.p_volatile},
                          customdata=np.stack([shown.index.strftime("%d %b %Y"), shown.p_volatile], axis=1),
                          hovertemplate="%{customdata[0]}<br>P(Volatile) %{customdata[1]:.2f}<extra></extra>", showlegend=False)

    fig = go.Figure(trace(months[-1]))
    fig.frames = [go.Frame(data=[trace(m)], name=str(m)) for m in months]
    pad_x, pad_y = (points.pc1.max() - points.pc1.min()) * 0.06, (points.pc2.max() - points.pc2.min()) * 0.06
    fig.update_xaxes(range=[points.pc1.min() - pad_x, points.pc1.max() + pad_x], title_text="first principal component of this year's latents")
    fig.update_yaxes(range=[points.pc2.min() - pad_y, points.pc2.max() + pad_y], title_text="second principal component")
    fig = theme.style(fig, c, height=470, legend=False)
    return fig.update_layout(
        margin={"b": 90},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": -0.2, "xanchor": "left", "direction": "left",
                      "bgcolor": "rgba(0,0,0,0)", "font": {"color": c["ink_2"]},
                      "buttons": [{"label": "▶ Play", "method": "animate",
                                   "args": [None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": False, "transition": {"duration": 0}}]},
                                  {"label": "Pause", "method": "animate",
                                   "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}]}],
        sliders=[{"active": len(months) - 1, "x": 0.16, "len": 0.84, "y": -0.16, "currentvalue": {"prefix": "through ", "xanchor": "right", "font": {"color": c["ink_2"]}},
                  "font": {"color": c["muted"]},
                  "steps": [{"label": m.strftime("%b"), "method": "animate",
                             "args": [[str(m)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]} for m in months]}],
    )


# --------------------------------------------------------------------------- allocation through time (page 9)
def allocation_area(sleeves: pd.DataFrame, p_volatile: pd.Series, holdout_start: pd.Timestamp | None, c: dict[str, Any]) -> go.Figure:
    """Stacked sleeves per weekly decision, with the HMM's P(Volatile) on those dates as a strip above (never behind)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.09, 0.91], vertical_spacing=0.03)
    fig.add_trace(go.Heatmap(x=p_volatile.index, y=["P(Volatile)"], z=[p_volatile.to_numpy()], zmin=0, zmax=1, showscale=False,
                             colorscale=theme.regime_scale(c), hovertemplate="%{x|%d %b %Y}<br>P(Volatile) %{z:.2f}<extra></extra>"), row=1, col=1)
    for name in theme.SLEEVES:
        fig.add_trace(go.Scatter(x=sleeves.index, y=sleeves[name], name=name, mode="lines", stackgroup="one",
                                 line={"width": 1, "color": c["surface"]}, fillcolor=c["sleeves"][name],
                                 hovertemplate=f"{name} %{{y:.1%}}<extra></extra>"), row=2, col=1)
    if holdout_start is not None:
        fig.add_vline(x=holdout_start, line={"color": c["ink"], "width": 1}, row=2, col=1)
        fig.add_annotation(x=holdout_start, y=1.0, yref="y2", xanchor="left", yanchor="bottom", showarrow=False, xshift=4,
                           text="holdout from here", font={"color": c["ink_2"], "size": 12})
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="share of the portfolio", row=2, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False)
    fig = theme.style(fig, c, height=430)
    return fig.update_layout(hovermode="x unified", legend={"y": 1.04, "traceorder": "normal"})


def weights_bar(weights: pd.Series, low: pd.Series | None, high: pd.Series | None, sleeve_of: dict[str, str], cap: float,
                c: dict[str, Any]) -> go.Figure:
    """One week's weights by holding, coloured by sleeve; whiskers span the lowest and highest seed."""
    error = None
    if low is not None and high is not None:
        error = {"type": "data", "symmetric": False, "array": (high - weights).clip(lower=0).to_numpy(),
                 "arrayminus": (weights - low).clip(lower=0).to_numpy(), "color": c["ink_2"], "thickness": 1.5, "width": 4}
    fig = go.Figure()
    for sleeve in theme.SLEEVES:
        names = [n for n in weights.index if sleeve_of[n] == sleeve]
        if not names:
            continue
        idx = [list(weights.index).index(n) for n in names]
        err = None if error is None else {**error, "array": error["array"][idx], "arrayminus": error["arrayminus"][idx]}
        fig.add_trace(go.Bar(x=names, y=weights[names], name=sleeve, width=0.55, marker={"color": c["sleeves"][sleeve], "line": {"width": 0}},
                             error_y=err, hovertemplate="%{x}: %{y:.1%}<extra>" + sleeve + "</extra>"))
    fig.add_hline(y=cap, line={"color": c["muted"], "width": 1}, annotation_text=f"cap on a risky asset: {cap:.0%}",
                  annotation_position="top left", annotation_font={"color": c["ink_2"], "size": 12})
    top = float(max(cap * 1.2, (high.max() if high is not None else weights.max()) * 1.1))
    fig.update_yaxes(tickformat=".0%", range=[0, top], title_text="target weight")
    fig.update_xaxes(categoryorder="array", categoryarray=list(weights.index))
    return theme.style(fig, c, height=340)


def defensive_scatter(defensive: pd.Series, p_volatile: pd.Series, c: dict[str, Any]) -> go.Figure:
    """One dot per weekly decision: the non-equity share against the HMM's P(Volatile) that day."""
    both = pd.concat([p_volatile.rename("p"), defensive.rename("d")], axis=1).dropna()
    fig = go.Figure(go.Scatter(x=both.p, y=both.d, mode="markers", showlegend=False,
                               marker={"size": 8, "color": c["accent"], "opacity": 0.55, "line": {"color": c["surface"], "width": 1}},
                               customdata=both.index.strftime("%d %b %Y"),
                               hovertemplate="%{customdata}<br>P(Volatile) %{x:.2f}<br>defensive share %{y:.1%}<extra></extra>"))
    fig.update_xaxes(title_text="P(Volatile) on the decision day", range=[-0.03, 1.03])
    fig.update_yaxes(title_text="bonds + gold + cash", tickformat=".0%", range=[0, 1])
    return theme.style(fig, c, height=340, legend=False)


def duration_ladder(weights: pd.DataFrame, bonds: list[str], highlight: tuple[str, str] | None, c: dict[str, Any]) -> go.Figure:
    """The bond sleeve split by maturity, short to long, one blue ramp light to dark."""
    fig = go.Figure()
    for name, color in zip(bonds, c["ordinal"][:len(bonds)] if c["mode"] == "light" else c["ordinal"][-len(bonds):][::-1]):
        fig.add_trace(go.Scatter(x=weights.index, y=weights[name], name=name, mode="lines", stackgroup="one",
                                 line={"width": 1, "color": c["surface"]}, fillcolor=color, hovertemplate=f"{name} %{{y:.1%}}<extra></extra>"))
    if highlight is not None:
        fig.add_vrect(x0=highlight[0], x1=highlight[1], fillcolor=c["ink"], opacity=0.07, line_width=0,
                      annotation_text=highlight[0][:4], annotation_position="top left", annotation_font={"color": c["ink_2"], "size": 12})
    fig.update_yaxes(tickformat=".0%", title_text="share of the portfolio", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    fig = theme.style(fig, c, height=300)
    return fig.update_layout(hovermode="x unified")


# --------------------------------------------------------------------------- results interactivity (page 10)
def seed_strip(values: dict[str, np.ndarray], benchmarks: dict[str, float], title: str, c: dict[str, Any], *, bar: float | None = None,
               fmt: str = ".2f") -> go.Figure:
    """One dot per seed, a column per variant; the benchmarks' single values as grey ticks; optionally a bar to clear."""
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for i, v in enumerate(theme.VARIANTS):
        y = np.asarray(values[v], dtype=float)
        fig.add_trace(go.Scatter(x=i + rng.uniform(-0.12, 0.12, len(y)), y=y, mode="markers", name=v,
                                 marker={"size": 9, "color": c["variants"][v], "symbol": theme.VARIANT_SYMBOL[v], "line": {"color": c["surface"], "width": 1.5}},
                                 customdata=np.arange(len(y)), hovertemplate=f"{v} seed %{{customdata}}: %{{y:{fmt}}}<extra></extra>"))
        fig.add_shape(type="line", x0=i - 0.25, x1=i + 0.25, y0=float(y.mean()), y1=float(y.mean()), line={"color": c["ink"], "width": 2})
    if benchmarks:
        fig.add_trace(go.Scatter(x=[len(theme.VARIANTS)] * len(benchmarks), y=list(benchmarks.values()), mode="markers", name="Benchmarks",
                                 marker={"size": 16, "color": c["context"], "symbol": "line-ew", "line": {"color": c["ink_2"], "width": 2}},
                                 customdata=[strategy_label(k) for k in benchmarks], hovertemplate=f"%{{customdata}}: %{{y:{fmt}}}<extra></extra>"))
    if bar is not None:
        fig.add_hline(y=bar, line={"color": c["ink_2"], "width": 1}, annotation_text=f"bar to clear: {bar:g}", annotation_position="top left",
                      annotation_font={"color": c["ink_2"], "size": 12})
    labels = [*theme.VARIANTS, "Benchmarks"] if benchmarks else list(theme.VARIANTS)
    fig.update_xaxes(tickvals=list(range(len(labels))), ticktext=labels, range=[-0.5, len(labels) - 0.5], showgrid=False)
    fig.update_yaxes(title_text=title, tickformat=fmt)
    return theme.style(fig, c, height=340)


def size_bars(spread: dict[str, float], gaps: dict[str, float], c: dict[str, Any]) -> go.Figure:
    """On one Sharpe-ratio axis: how much seeds of one variant differ, against how much variants differ from each other."""
    names = [*(f"{k} seeds" for k in spread), *gaps]
    vals = [*spread.values(), *gaps.values()]
    cols = [*(c["context"] for _ in spread), *(c["ink"] for _ in gaps)]
    fig = go.Figure(go.Bar(y=names[::-1], x=vals[::-1], orientation="h", width=0.5, marker={"color": cols[::-1], "line": {"width": 0}},
                           text=[f"{v:.2f}" for v in vals[::-1]], textposition="outside", cliponaxis=False, textfont={"color": c["ink_2"], "size": 12},
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Sharpe ratio units", range=[0, max(vals) * 1.2])
    fig.update_yaxes(showgrid=False)
    return theme.style(fig, c, height=340, legend=False)


def equity_chart(curves: pd.DataFrame, variants: list[str], seeds: list[int], benchmarks: list[str], c: dict[str, Any],
                 span: tuple[str, str] | None = None) -> go.Figure:
    """Growth of 1: for each chosen variant the band between its lowest and highest seed and the median seed; benchmarks as lines."""
    fig = go.Figure()
    for b in benchmarks:
        fig.add_trace(go.Scatter(x=curves.index, y=curves[f"BM|{b}"], mode="lines", name=strategy_label(b), line={"color": c["context"], "width": 1.5},
                                 hovertemplate=f"{strategy_label(b)} %{{y:.2f}}<extra></extra>"))
    for v in variants:
        block = curves[[f"{v}|s{s}" for s in seeds]]
        lo, hi, mid = block.min(axis=1), block.max(axis=1), block.median(axis=1)
        color = c["variants"][v]
        rgba = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.16)"
        fig.add_trace(go.Scatter(x=curves.index, y=hi, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip", legendgroup=v))
        fig.add_trace(go.Scatter(x=curves.index, y=lo, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=rgba, showlegend=False,
                                 hoverinfo="skip", legendgroup=v))
        fig.add_trace(go.Scatter(x=curves.index, y=mid, mode="lines", name=f"{v}: median seed, band = all seeds", legendgroup=v,
                                 line={"color": color, "width": 2, "dash": theme.VARIANT_DASH[v]},
                                 customdata=np.stack([lo, hi], axis=1),
                                 hovertemplate=f"{v} median %{{y:.2f}} (seeds %{{customdata[0]:.2f}} to %{{customdata[1]:.2f}})<extra></extra>"))
    if span is not None:
        fig.update_xaxes(range=list(span))
        window = curves.loc[span[0]: span[1]]
        if len(window):
            fig.update_yaxes(range=[float(window.min().min()) * 0.97, float(window.max().max()) * 1.03])
    fig.update_yaxes(title_text="growth of 1, after costs")
    fig.update_xaxes(showgrid=False)
    fig = theme.style(fig, c, height=420)
    return fig.update_layout(hovermode="x unified")


def cost_rank(values: pd.Series, c: dict[str, Any]) -> go.Figure:
    """Strategies sorted by net Sharpe at one cost level: variants in colour (mean of seeds), benchmarks in grey."""
    v = values.sort_values()
    cols = [c["variants"].get(k, c["context"]) for k in v.index]
    fig = go.Figure(go.Bar(y=[strategy_label(k) for k in v.index], x=v.to_numpy(), orientation="h", width=0.55,
                           marker={"color": cols, "line": {"width": 0}}, text=[f"{x:.2f}" for x in v], textposition="outside",
                           cliponaxis=False, textfont={"color": c["ink_2"], "size": 12}, hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="net Sharpe ratio", zeroline=True)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(transition={"duration": 300})
    return theme.style(fig, c, height=360, legend=False)


# --------------------------------------------------------------------------- live weights, home, what-if (pages 7, 1, 8)
def sleeve_donut(sleeves: pd.Series, c: dict[str, Any], *, height: int = 300) -> go.Figure:
    """Part-to-whole at a glance: the four sleeves of one allocation."""
    names = [n for n in theme.SLEEVES if sleeves.get(n, 0.0) > 0]
    fig = go.Figure(go.Pie(labels=names, values=[float(sleeves[n]) for n in names], hole=0.62, sort=False, direction="clockwise",
                           marker={"colors": [c["sleeves"][n] for n in names], "line": {"color": c["surface"], "width": 2}},
                           textinfo="percent", textposition="inside", insidetextorientation="horizontal",
                           hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.update_layout(transition={"duration": 400})
    fig = theme.style(fig, c, height=height)
    return fig.update_layout(margin={"l": 8, "r": 8, "t": 40, "b": 8}, legend={"y": 1.08})


def change_bars(change: pd.Series, sleeve_of: dict[str, str], c: dict[str, Any]) -> go.Figure:
    """Change in each holding's weight, in percentage points, from the unmodified observation."""
    fig = go.Figure(go.Bar(x=list(change.index), y=change.to_numpy() * 100, width=0.55,
                           marker={"color": [c["sleeves"][sleeve_of[n]] for n in change.index], "line": {"width": 0}},
                           hovertemplate="%{x}: %{y:+.1f} pp<extra></extra>"))
    span = max(1.0, float(np.abs(change).max()) * 115)
    fig.update_yaxes(title_text="change in weight (percentage points)", range=[-span, span], zeroline=True, zerolinewidth=1)
    fig.update_layout(transition={"duration": 300})
    return theme.style(fig, c, height=300, legend=False)


def sweep_lines(sweep: pd.DataFrame, references: dict[str, float], today: float, c: dict[str, Any]) -> go.Figure:
    """Each sleeve's share (average of the seeds) as P(Volatile) is set from 0 to 1; flat references for variants with no regime input."""
    fig = go.Figure()
    for name in theme.SLEEVES:
        fig.add_trace(go.Scatter(x=sweep.index, y=sweep[name], mode="lines", name=name, line={"color": c["sleeves"][name], "width": 2},
                                 hovertemplate=f"{name} %{{y:.1%}} at P(Volatile) %{{x:.2f}}<extra></extra>"))
    for name, value in references.items():
        fig.add_trace(go.Scatter(x=[0, 1], y=[value, value], mode="lines", name=name, line={"color": c["muted"], "width": 1.5, "dash": "dot"},
                                 hovertemplate=f"{name}: %{{y:.1%}} whatever the slider says<extra></extra>"))
    fig.add_vline(x=today, line={"color": c["ink_2"], "width": 1}, annotation_text="today", annotation_position="top",
                  annotation_font={"color": c["ink_2"], "size": 12})
    fig.update_xaxes(title_text="P(Volatile) fed to the agent", range=[0, 1])
    fig.update_yaxes(title_text="share of the portfolio", tickformat=".0%", range=[0, 1])
    return theme.style(fig, c, height=360)


def latent_spark(latent: list[float], c: dict[str, Any]) -> go.Figure:
    """The latent's numbers as a strip of small bars: a picture of 'a vector', not something to read values from."""
    fig = go.Figure(go.Bar(x=list(range(len(latent))), y=latent, marker={"color": c["accent"], "line": {"width": 0}},
                           hovertemplate="latent %{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-1.05, 1.05])
    fig = theme.style(fig, c, height=90, legend=False)
    return fig.update_layout(margin={"l": 0, "r": 0, "t": 4, "b": 4}, bargap=0.25)


def pipeline(steps: list[tuple[str, str]], c: dict[str, Any]) -> go.Figure:
    """The pipeline as a row of nodes, each with its real value; play to light them in order (about two seconds)."""
    n = len(steps)
    x = list(range(n))

    def nodes(lit: int) -> go.Scatter:
        return go.Scatter(x=x, y=[0] * n, mode="markers+text", text=[f"<b>{a}</b><br>{b}" for a, b in steps], textposition="bottom center",
                          textfont={"color": c["ink_2"], "size": 12}, hoverinfo="skip", showlegend=False,
                          marker={"size": 26, "symbol": "circle", "color": [c["accent"] if i <= lit else c["grid"] for i in range(n)],
                                  "line": {"color": c["surface"], "width": 2}})

    fig = go.Figure([go.Scatter(x=[0, n - 1], y=[0, 0], mode="lines", line={"color": c["axis"], "width": 2}, hoverinfo="skip", showlegend=False),
                     nodes(n - 1)])
    fig.frames = [go.Frame(data=[fig.data[0], nodes(i)], name=str(i)) for i in range(n)]
    fig.update_xaxes(visible=False, range=[-0.6, n - 0.4])
    fig.update_yaxes(visible=False, range=[-1.6, 0.6])
    fig = theme.style(fig, c, height=190, legend=False)
    return fig.update_layout(
        margin={"l": 0, "r": 0, "t": 34, "b": 0},
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": 1.25, "xanchor": "left", "bgcolor": "rgba(0,0,0,0)",
                      "font": {"color": c["ink_2"]},
                      "buttons": [{"label": "▶ Run the pipeline", "method": "animate",
                                   "args": [None, {"frame": {"duration": 280, "redraw": True}, "fromcurrent": False, "transition": {"duration": 0}}]}]}])
