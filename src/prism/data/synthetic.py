"""Deterministic synthetic market data. **Testing only.**

No research code path imports this module. It exists so that the test suite —
and in particular the causality and leak-injection tests — runs against data
whose properties are known exactly, with no network dependency and no
sensitivity to whatever the vendor returns today.

The generator is not a toy. To exercise the pipeline honestly it has to
produce a panel with the same structural awkwardnesses as the real one:

* **A three-state regime process** with distinct mean, volatility *and*
  correlation structure per state, so a fitted HMM has something to find
  beyond a volatility ladder (the failure mode of defect B3).
* **Staggered inception dates** taken from config, so pre-inception blanks
  are distinguishable from missing sessions and ``clean.py``'s availability
  logic is actually tested.
* **Yields in percentage points** that can approach zero, so any code that
  takes a log of a yield breaks loudly in tests rather than in production.
* **The NYSE calendar**, so a builder that assumes contiguous business days
  fails here first.

Returns are clipped at ±25% so the §4.3 hard extreme-move check (>30%) passes
on synthetic data; the soft check (>15%) may legitimately fire during a
generated crisis, which is the intended behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.config import Config
from prism.features.build import make_raw_frame
from prism.utils.calendar import trading_days
from prism.utils.seeding import make_rng

__all__ = [
    "SyntheticSpec",
    "make_synthetic_raw",
    "make_small_raw",
    "synthetic_true_states",
]

#: Hard clip on generated daily log returns, below the §4.3 hard threshold.
RETURN_CLIP = 0.25


class SyntheticSpec:
    """Regime parameters. Three states: calm, choppy, crisis."""

    #: Annualised market drift per state.
    drift = np.array([0.10, 0.02, -0.35])
    #: Annualised market volatility per state.
    vol = np.array([0.11, 0.19, 0.42])
    #: Idiosyncratic volatility as a multiple of market volatility. Lower in
    #: crisis, which is what makes cross-sectional correlation *rise*.
    idio_mult = np.array([1.30, 0.95, 0.45])
    #: Row-stochastic transition matrix. Expected durations ~100/25/15 days.
    transmat = np.array(
        [
            [0.990, 0.008, 0.002],
            [0.030, 0.960, 0.010],
            [0.010, 0.057, 0.933],
        ]
    )

    def __init__(self) -> None:
        assert np.allclose(self.transmat.sum(axis=1), 1.0)


def _simulate_states(n: int, rng: np.random.Generator, spec: SyntheticSpec) -> np.ndarray:
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(3, p=spec.transmat[states[t - 1]])
    return states


def _ou(
    n: int,
    rng: np.random.Generator,
    *,
    level: float,
    speed: float,
    sigma: float,
    floor: float | None = None,
) -> np.ndarray:
    """Discrete Ornstein-Uhlenbeck path, optionally floored."""
    out = np.empty(n)
    out[0] = level
    shocks = rng.normal(size=n) * sigma
    for t in range(1, n):
        out[t] = out[t - 1] + speed * (level - out[t - 1]) + shocks[t]
        if floor is not None:
            out[t] = max(out[t], floor)
    return out


def make_synthetic_raw(
    cfg: Config,
    *,
    seed: int = 0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    universe: str = "B",
) -> pd.DataFrame:
    """A full synthetic raw panel for ``universe``, as a ``(field, ticker)`` frame.

    ``start`` defaults to the earliest inception among the universe's
    **tradable** tickers, so the generator carries no date literal of its own
    (every date in the project comes from config, which ``test_splits.py``
    enforces with a source grep). Anchoring on tradable rather than macro
    inception matters: ``^VIX`` goes back to 1990, but a panel that opens nine
    years before any equity exists produces no warm features for those years
    and only slows the suite down.

    ``end`` defaults to the end of the **test** split, so the fixture never
    spans the holdout window: a test that accidentally reads the holdout
    should get an empty slice, not plausible-looking data.
    """
    spec = SyntheticSpec()
    # Two independent substreams. The regime chain gets its own so that
    # `synthetic_true_states` can reproduce it without re-simulating the
    # whole panel, and so that adding a ticker does not shift the regime path.
    rng = make_rng(cfg.data.seeds.master, "synthetic", seed, "observations")
    state_rng = make_rng(cfg.data.seeds.master, "synthetic", seed, "states")

    end_ts = pd.Timestamp(end) if end is not None else cfg.data.split("test")[1]
    if start is None:
        tradable = set(cfg.data.allocatable[universe]) | {
            cfg.data.universes["A"].benchmark
        }
        start_ts = min(
            pd.Timestamp(cfg.data.inception[t])
            for t in cfg.data.tickers(universe)
            if t in tradable
        )
    else:
        start_ts = pd.Timestamp(start)
    index = trading_days(start_ts, end_ts, cfg.data.calendar.exchange)
    n = len(index)

    states = _simulate_states(n, state_rng, spec)
    dt = 1.0 / 252.0
    mkt_mu = spec.drift[states] * dt
    mkt_sd = spec.vol[states] * np.sqrt(dt)
    market = rng.normal(mkt_mu, mkt_sd)

    tickers = cfg.data.tickers(universe)
    macro_set = set(cfg.data.universes["A"].macro) | set(
        cfg.data.universes["B"].macro_extra
    )
    assets = [t for t in tickers if t not in macro_set]
    macros = [t for t in tickers if t in macro_set]

    # -- assets ----------------------------------------------------------- #
    # Betas spread around 1 for equities; the defensive sleeve is given
    # negative or near-zero beta so it can actually function as a hedge.
    defensive = set(cfg.data.universes["B"].defensive)
    close = pd.DataFrame(index=index, columns=assets, dtype="float64")
    volume = pd.DataFrame(index=index, columns=assets, dtype="float64")

    for i, ticker in enumerate(assets):
        if ticker in defensive:
            beta = {"SHY": 0.02, "IEF": -0.15, "TLT": -0.30, "GLD": 0.05}.get(ticker, 0.0)
            idio_scale = 0.25
        elif ticker == cfg.data.universes["A"].benchmark:
            beta, idio_scale = 1.0, 0.10
        else:
            beta = 0.75 + 0.5 * ((i * 7) % 9) / 8.0
            idio_scale = 1.0

        idio_sd = idio_scale * spec.idio_mult[states] * mkt_sd
        rets = beta * market + rng.normal(0.0, idio_sd)
        rets = np.clip(rets, -RETURN_CLIP, RETURN_CLIP)

        prices = 50.0 * np.exp(np.cumsum(rets))
        series = pd.Series(prices, index=index)

        # Volume: lognormal, elevated in stress.
        base = 1e6 * (1.0 + (i % 5))
        vol_series = base * np.exp(
            rng.normal(0.0, 0.25, size=n) + 0.6 * (states > 0).astype(float)
        )

        inception = pd.Timestamp(cfg.data.inception[ticker])
        live = index >= inception
        close.loc[live, ticker] = series[live].to_numpy()
        volume.loc[live, ticker] = np.round(vol_series[live])

    # -- macro ------------------------------------------------------------ #
    macro = pd.DataFrame(index=index, columns=macros, dtype="float64")
    realised_vol = pd.Series(spec.vol[states], index=index)

    for ticker in macros:
        inception = pd.Timestamp(cfg.data.inception[ticker])
        live = index >= inception
        if ticker == "^VIX":
            values = 100.0 * realised_vol.to_numpy() * np.exp(
                rng.normal(0.0, 0.12, size=n)
            ) + 2.0
        elif ticker == "^VIX3M":
            # Term premium in calm, inversion in stress — the real shape.
            premium = np.where(states == 2, -0.10, 0.08)
            values = (
                100.0 * realised_vol.to_numpy() * np.exp(premium + rng.normal(0.0, 0.08, size=n))
                + 2.5
            )
        elif ticker == "^TNX":
            values = _ou(n, rng, level=3.6, speed=0.004, sigma=0.055, floor=0.35)
        elif ticker == "^FVX":
            values = _ou(n, rng, level=3.0, speed=0.005, sigma=0.055, floor=0.20)
        elif ticker == "^IRX":
            # Can get very close to zero (2009-2015, 2020-2021). Deliberate:
            # any log() of a yield must fail in tests.
            values = _ou(n, rng, level=2.0, speed=0.004, sigma=0.060, floor=0.01)
        elif ticker in ("HYG", "LQD"):
            # HYG carries credit risk (loads on the crisis state), LQD mostly
            # duration, so log(HYG/LQD) is a genuine credit-stress proxy.
            load = -0.55 if ticker == "HYG" else -0.10
            credit = load * np.where(states == 2, 1.0, 0.0) * 0.004
            rets = np.clip(
                0.00012 + credit + 0.25 * market + rng.normal(0.0, 0.0035, size=n),
                -RETURN_CLIP,
                RETURN_CLIP,
            )
            values = 80.0 * np.exp(np.cumsum(rets))
        elif ticker == "DX-Y.NYB":
            rets = np.clip(rng.normal(0.0, 0.004, size=n) - 0.12 * market, -RETURN_CLIP, RETURN_CLIP)
            values = 95.0 * np.exp(np.cumsum(rets))
        elif ticker == "CL=F":
            rets = np.clip(
                rng.normal(0.0, 0.021, size=n) + 0.45 * market, -RETURN_CLIP, RETURN_CLIP
            )
            values = 60.0 * np.exp(np.cumsum(rets))
        else:  # pragma: no cover - a new macro ticker without a generator
            raise KeyError(f"no synthetic generator for macro ticker {ticker!r}")
        macro.loc[live, ticker] = np.asarray(values)[live]

    # NOTE: the latent state path is deliberately NOT attached to the frame's
    # `attrs`. pandas propagates `attrs` through concat/slicing and compares
    # them with `==`; a Series there makes that comparison elementwise and
    # every downstream concat raises "truth value of a Series is ambiguous".
    # Use `synthetic_true_states` instead, which regenerates the same path
    # from its own RNG substream.
    return make_raw_frame(close, volume, macro)


def synthetic_true_states(
    cfg: Config,
    index: pd.DatetimeIndex,
    *,
    seed: int = 0,
) -> pd.Series:
    """The latent regime path underlying a synthetic panel.

    Reproduces the chain :func:`make_synthetic_raw` used, drawn from its own
    RNG substream so that it does not depend on how many tickers were
    generated. Step 2 scores recovered posteriors against this.
    """
    spec = SyntheticSpec()
    state_rng = make_rng(cfg.data.seeds.master, "synthetic", seed, "states")
    states = _simulate_states(len(index), state_rng, spec)
    return pd.Series(states, index=pd.DatetimeIndex(index), name="true_state")


def make_small_raw(
    cfg: Config,
    *,
    seed: int = 0,
    sessions: int = 900,
    universe: str = "A",
) -> pd.DataFrame:
    """A short panel for fast tests, still long enough to be warm.

    Starts at the **latest** inception among the universe's tickers, so every
    series is live from row 0 and all ``sessions`` rows are usable. Using the
    earliest inception instead would spend most of a short panel on
    pre-inception blanks and leave too few warm rows to test anything.

    ``sessions=900`` gives roughly 3.5 years: past the 252-session warm-up
    with enough left over for a meaningful number of rows.
    """
    latest = max(
        pd.Timestamp(cfg.data.inception[t]) for t in cfg.data.tickers(universe)
    )
    full = make_synthetic_raw(cfg, seed=seed, universe=universe, start=latest)
    return full.iloc[:sessions].copy()
