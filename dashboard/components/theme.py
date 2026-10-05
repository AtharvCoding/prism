"""Colours and the Plotly look, for light and dark. DASHBOARD.md §7 ("one colour per variant, consistently across pages").

Colour is assigned by entity and never by rank, so a variant keeps its hue on every page:

* **Variants** V1, V2, V4, C4 = blue, orange, aqua, violet, in that fixed order. The set was run through the
  palette validator on both surfaces: adjacent pairs clear the colour-vision-deficiency and normal-vision
  floors in light and dark. No four-hue set clears them for *every* pair (in dark, blue and violet are close
  under protanopia), so C4, the control, also gets a dashed line and a diamond marker, and every chart with
  variants has a legend and a table beside it.
* **Benchmarks** are context: one grey, named in the legend and on hover.
* **Windows** (test split, holdout) are not hues: the holdout is drawn in the primary ink, filled; the test
  split in muted grey, open. Shape carries it as well as tone.
* **Ordered categories** (the four splits) use one blue ramp, light to dark in time order.
* **Regimes**: Calm is the cool pole (blue) and Volatile the warm pole (red), on every page; a probability of
  Volatile is the diverging scale between them through a neutral grey at 0.5. The VIX-threshold state uses the
  same two colours, because it is the same kind of thing.
* **Sleeves**: Equity violet, Bonds aqua, Gold yellow, Cash grey. Both sets pass the validator for every pair in
  both themes. Regimes and sleeves never share a plot area: where both appear the regime is a strip above.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

VARIANTS = ("V1", "V2", "V4", "C4")

_LIGHT = {
    "mode": "light", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "context": "#a8a69f",
    "variants": {"V1": "#2a78d6", "V2": "#eb6834", "V4": "#1baf7a", "C4": "#4a3aa7"},
    "accent": "#2a78d6",
    "ordinal": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    "calm": "#2a78d6", "volatile": "#e34948", "neutral": "#f0efec",
    "sleeves": {"Equity": "#4a3aa7", "Bonds": "#1baf7a", "Gold": "#eda100", "Cash": "#a8a69f"},
}
_DARK = {
    "mode": "dark", "surface": "#1a1a19", "ink": "#ffffff", "ink_2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "context": "#6f6e69",
    "variants": {"V1": "#3987e5", "V2": "#d95926", "V4": "#199e70", "C4": "#9085e9"},
    "accent": "#3987e5",
    "ordinal": ["#184f95", "#256abf", "#5598e7", "#9ec5f4"],
    "calm": "#3987e5", "volatile": "#e66767", "neutral": "#383835",
    "sleeves": {"Equity": "#9085e9", "Bonds": "#199e70", "Gold": "#c98500", "Cash": "#6f6e69"},
}
#: Secondary encoding for the variants, so identity never rests on hue alone.
VARIANT_DASH = {"V1": "solid", "V2": "solid", "V4": "solid", "C4": "dash"}
VARIANT_SYMBOL = {"V1": "circle", "V2": "square", "V4": "triangle-up", "C4": "diamond"}
SLEEVES = ("Equity", "Bonds", "Gold", "Cash")
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def regime_scale(c: dict[str, Any]) -> list[list[Any]]:
    """Diverging colour scale for P(Volatile): Calm at 0, neutral grey at 0.5, Volatile at 1."""
    return [[0.0, c["calm"]], [0.5, c["neutral"]], [1.0, c["volatile"]]]


def colors() -> dict[str, Any]:
    """The palette for the viewer's current theme (light unless Streamlit reports dark)."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - no browser context (tests, bare mode)
        dark = False
    return _DARK if dark else _LIGHT


def palette(mode: str) -> dict[str, Any]:
    return _DARK if mode == "dark" else _LIGHT


def style(fig: go.Figure, c: dict[str, Any], *, height: int = 360, legend: bool = True) -> go.Figure:
    """Thin marks, hairline solid grid, transparent surface (the app's own background shows through)."""
    fig.update_layout(
        height=height, margin={"l": 8, "r": 16, "t": 28, "b": 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": c["ink_2"], "size": 13}, hoverlabel={"font_size": 13}, showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "title_text": "",
                "font": {"color": c["ink_2"]}},
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=c["grid"], gridwidth=1, linecolor=c["axis"], zerolinecolor=c["axis"], zerolinewidth=1,
                     tickfont={"color": c["muted"]}, title_font={"color": c["ink_2"]}, griddash="solid")
    fig.update_yaxes(gridcolor=c["grid"], gridwidth=1, linecolor=c["axis"], zerolinecolor=c["axis"], zerolinewidth=1,
                     tickfont={"color": c["muted"]}, title_font={"color": c["ink_2"]}, griddash="solid")
    return fig
