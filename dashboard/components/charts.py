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
