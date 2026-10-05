"""Page 5, LSTM latent. DASHBOARD.md §7.5. The map is precomputed, one PCA per encoder fold (DECISIONS.md D-044 item 8, D-047)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts, data, theme, ui

facts = ui.page_header("latent")
enc = facts["models"]["encoder"]
c = theme.colors()
coords, folds = data.latent_map(), data.latent_folds()

# --------------------------------------------------------------------------- what it is
st.subheader("What the LSTM does")
s1, s2, s3 = st.columns(3)
with s1.container(border=True, height="stretch"):
    st.markdown(f"**1 · Look at {enc['window']} days**")
    st.write(f"The network reads the last {enc['window']} trading days of market features, with a little random noise added "
             "while it is learning.")
with s2.container(border=True, height="stretch"):
    st.markdown(f"**2 · Squeeze to {enc['latent_dim']} numbers**")
    st.write(f"It must pass everything it wants to keep through a bottleneck of {enc['latent_dim']} numbers: the *latent*.")
with s3.container(border=True, height="stretch"):
    st.markdown("**3 · Rebuild the original**")
    st.write(f"From those {enc['latent_dim']} numbers alone it tries to reconstruct the clean {enc['window']} days. What "
             "survives the squeeze is its summary of recent history.")
st.caption(f"This is a denoising autoencoder. It is retrained once a year on all history so far, and the agent variants V2, "
           f"V4 and C4 are given its {enc['latent_dim']} numbers every day alongside the raw features.")

# --------------------------------------------------------------------------- map
st.subheader("A map of the latent, one year at a time")
stress: dict[int, str] = {}
for r in data.regime_summary()["nber_recessions"].values():
    for y in range(pd.Timestamp(r["peak"]).year, pd.Timestamp(r["trough"]).year + 1):
        stress[y] = "recession"
for w in ("test", "holdout"):
    ep = data.results(w)["episodes"].drop_duplicates("episode")
    for _, e in ep.iterrows():
        for y in range(pd.Timestamp(e.peak).year, pd.Timestamp(e.trough).year + 1):
            stress.setdefault(y, "S&P 500 fall of 10% or more")
years = {int(row.start.year): int(row.fold) for row in folds.itertuples()}
label = lambda y: f"{y} · {stress[y]}" if y in stress else str(y)  # noqa: E731
default = max(y for y in years if y in stress and stress[y] == "recession")
year = st.select_slider("Year (each year has its own, separately trained, network)", options=list(years), value=default,
                        format_func=label, key="latent_year")
points = coords[coords.fold == years[year]]
info = folds[folds.fold == years[year]].iloc[0]
st.plotly_chart(charts.latent_map(points, c), config=theme.PLOT_CONFIG, key="latent_map")
st.caption(
    f"Each dot is one trading day of {year}: its {enc['latent_dim']} latent numbers flattened to the two directions along which "
    f"that year's latents vary most ({info.explained_1 + info.explained_2:.0%} of their variation). Colour is the HMM's "
    f"P(Volatile) for the same day. The network used in {year} was trained on data to {info.fit_end:%d %b %Y}. Press play to "
    "watch the year unfold month by month."
)
st.info(
    "**Positions are not comparable between years.** Every January a freshly trained network replaces the old one and "
    "lays the same information out differently, so a dot's place on the 2020 map says nothing about its place on the 2021 "
    "map. That is also what the agent experienced: its latent inputs were rearranged once a year.",
    icon=":material/info:",
)

# --------------------------------------------------------------------------- tier 1
st.subheader("Did training the LSTM matter?")
t1 = data.tier1_gates()
row = t1[t1.comparison == "V2 vs C1"].iloc[0]
st.markdown(
    f"""
In Tier 1 the trained latent was compared with the output of an **untrained** network of exactly the same shape: random
weights, never fitted (the control called C1). On four forecasting targets, adding the trained latent was reliably better
on **{row.favourable}** and reliably worse on **{row.adverse}**. The gate needed 3 of 4 with none worse, so it **failed**.

What that means: passing recent history through *a* recurrent network gave the forecasts something, but the training
did not add to it. The value was in the shape of the map, not in what the network had learned.
"""
)

ui.how_to_read(
    f"""
* **Latent**: the {enc['latent_dim']} numbers in the middle of the network. They are not individually meaningful; together
  they are its compressed description of the last {enc['window']} days.
* **Principal components**: a standard way to draw many-dimensional points on a flat page, using the two directions
  that spread the points out most. The axes have no units and no meaning beyond that.
* **Colour**: blue days are ones the HMM called Calm, red ones Volatile, grey in between. If the dots sort by colour,
  the latent and the HMM are picking up related things in that year.
* **Why one year at a time**: the network is retrained every year, and each retraining produces its own layout.
* **Untrained control (C1)**: a network with random weights. If a trained network cannot beat it, the training is not
  what is doing the work.
"""
)
