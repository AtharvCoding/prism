"""Feature-build orchestrator. Spec §5, and the target of the causality tests.

:func:`build_features` is a **pure causal transform**: one raw panel in, one
feature frame out, nothing fitted, nothing learned. That is what makes it
truncation- and perturbation-invariant, and therefore what makes
``tests/test_causality.py`` able to prove it causal rather than assert it by
inspection. Winsorising, scaling and correlation pruning are fitted objects
and live in :mod:`prism.features.scaling`.

Universe isolation (§3.2 hazard 1)
----------------------------------
``build_features(universe="A")`` and ``build_features(universe="B")`` are
separate calls producing separate frames. Isolation is enforced by
construction, not by convention: the raw panel is *filtered down to the
universe's own ticker list as the very first step*, so the Universe A build
physically cannot read HYG — the column is gone before any feature code runs.
:func:`assert_universe_isolation` then re-checks the output, and
``test_fit_scope.py`` checks it against a panel where the B-only columns have
been replaced with values that would visibly corrupt any feature touching them.

Warm-up: two consequences worth stating plainly
-----------------------------------------------
``dist_from_52w_high`` needs 252 sessions, which is the longest lookback in
the feature set, so ``warmup_days`` is 252 rather than the 60 in the §3.3
config sketch (60 predates that feature). This has two knock-on effects that
are properties of the data, not choices:

* **Universe A features begin ~1999-12-16, not 1999-01-04.** The nine sector
  ETFs launched 1998-12-16, so a trailing 52-week high does not exist before
  late 1999. The effective ``fit_early`` window is therefore about 7.0 years
  and still contains the whole dot-com episode.
* **Universe B features begin ~2007-05-04, not 2007-04-04.** HYG's first
  trade *is* 2007-04-04, so ``credit_proxy_change_20`` cannot exist until 20
  sessions later. Universe B's equity and macro features are warm long
  before this; the credit proxy is the binding constraint.

Both are reported by :attr:`FeatureSet.warm_start` and written into the run
manifest, rather than being silently absorbed by a ``dropna()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from prism.config import Config
from prism.data.clean import clean_panel
from prism.features.asset import ASSET_FEATURE_NAMES, build_asset_features
from prism.features.cross import CROSS_FEATURE_NAMES, build_cross_features
from prism.features.macro import (
    MACRO_FEATURE_NAMES,
    MACRO_OPTIONAL_FEATURE_NAMES,
    build_macro_features,
)
from prism.features.targets import TARGET_PREFIX
from prism.utils.hashing import hash_object
from prism.utils.logging import get_logger

__all__ = [
    "FeatureSet",
    "build_features",
    "make_raw_frame",
    "split_raw_frame",
    "assert_universe_isolation",
    "RAW_FIELDS",
]

_log = get_logger(__name__)

#: Fields carried in the flat raw frame. ``Close`` holds both asset prices and
#: macro levels; ``Volume`` is asset-only.
RAW_FIELDS: tuple[str, ...] = ("Close", "Volume")


# --------------------------------------------------------------------------- #
# raw panel representation
# --------------------------------------------------------------------------- #
def make_raw_frame(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pack prices, volume and macro levels into one ``(field, ticker)`` frame.

    A single frame — rather than a dict of frames — is what lets the causality
    test express "truncate the input" as ``raw.loc[:t]`` and "randomise the
    future" as an in-place multiply, with no stage-specific plumbing.
    """
    parts: dict[str, pd.DataFrame] = {}
    price = close if macro is None else pd.concat([close, macro], axis=1)
    if price.columns.has_duplicates:
        dupes = price.columns[price.columns.duplicated()].unique().tolist()
        raise ValueError(f"duplicate tickers across close and macro: {dupes}")
    parts["Close"] = price
    parts["Volume"] = (
        pd.DataFrame(index=price.index, columns=price.columns, dtype="float64")
        if volume is None
        else volume.reindex(index=price.index)
    )
    out = pd.concat(parts, axis=1, names=["field", "ticker"])
    out.index = pd.DatetimeIndex(out.index).normalize()
    out.index.name = "date"
    return out.sort_index()


def split_raw_frame(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Unpack a ``(field, ticker)`` frame back into a dict of frames."""
    if not isinstance(raw.columns, pd.MultiIndex):
        raise TypeError("raw frame must have a (field, ticker) MultiIndex on columns")
    return {
        str(field): raw[field].copy()
        for field in raw.columns.get_level_values(0).unique()
    }


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSet:
    """A built feature frame plus everything needed to verify and reproduce it."""

    universe: str
    #: Warm, universe-restricted features. This is the deliverable.
    frame: pd.DataFrame
    #: Untrimmed frame including leading NaN warm-up rows, for diagnostics.
    untrimmed: pd.DataFrame
    #: First session on which every feature is finite.
    warm_start: pd.Timestamp
    #: Column names grouped by family, for the report and for V1' assembly.
    families: dict[str, list[str]] = field(default_factory=dict)
    #: Availability mask from cleaning, carried for QA.
    available: pd.DataFrame | None = None
    #: Sessions forward-filled per ticker, carried for QA.
    filled_sessions: pd.Series | None = None
    long_gaps: list[dict[str, object]] = field(default_factory=list)

    @cached_property
    def schema_hash(self) -> str:
        """Hash of the ordered column list. Spec §5.5.

        State-vector assembly records this, so a state frame built against a
        different feature schema is detectable rather than merely wrong.
        """
        return hash_object(
            {"universe": self.universe, "columns": [str(c) for c in self.frame.columns]}
        )

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index)

    def as_dict(self) -> dict[str, object]:
        return {
            "universe": self.universe,
            "schema_hash": self.schema_hash,
            "n_features": self.frame.shape[1],
            "n_sessions": self.frame.shape[0],
            "warm_start": str(self.warm_start.date()),
            "first_date": str(self.index[0].date()) if len(self.index) else None,
            "last_date": str(self.index[-1].date()) if len(self.index) else None,
            "families": {k: len(v) for k, v in self.families.items()},
            "long_gaps": self.long_gaps,
        }


# --------------------------------------------------------------------------- #
# the builder
# --------------------------------------------------------------------------- #
def build_features(
    raw: pd.DataFrame,
    cfg: Config,
    universe: str = "B",
    *,
    trim_to_warm: bool = True,
    restrict_to_universe_start: bool = True,
) -> FeatureSet:
    """Build the causal feature frame for ``universe`` from a raw panel.

    Parameters
    ----------
    raw
        ``(field, ticker)`` frame from :func:`make_raw_frame`. May contain
        tickers outside ``universe``; they are dropped before any feature code
        runs, which is how universe isolation is guaranteed.
    trim_to_warm
        Drop leading rows where any feature is still NaN. On by default;
        ``False`` is for diagnosing *why* a feature is not warm.
    restrict_to_universe_start
        Clip the output to ``universes[universe].start`` onward.

    Notes
    -----
    Nothing in this function is fitted, and nothing reads a statistic over the
    whole sample. Every operation is either row-local or a trailing rolling
    window. That is the invariant the causality test verifies.
    """
    spec = cfg.data._universe(universe)
    tickers = cfg.data.tickers(universe)

    # --- universe isolation, enforced first ------------------------------- #
    panels_all = split_raw_frame(raw)
    present = [t for t in tickers if t in panels_all["Close"].columns]
    absent = [t for t in tickers if t not in panels_all["Close"].columns]
    if absent:
        _log.warning("universe %s: tickers absent from the raw panel: %s", universe, absent)
    if not present:
        raise KeyError(f"raw panel contains none of universe {universe}'s tickers")
    panels = {fld: frame.reindex(columns=present) for fld, frame in panels_all.items()}

    # --- clean -------------------------------------------------------------#
    macro_names = set(spec.macro) | set(spec.macro_extra)
    if spec.inherits:
        parent = cfg.data.universes[spec.inherits]
        macro_names |= set(parent.macro) | set(parent.macro_extra)
    asset_cols = [t for t in present if t not in macro_names]
    macro_cols = [t for t in present if t in macro_names]

    cleaned = clean_panel(panels, cfg, tickers=present)
    close = cleaned.close
    volume = cleaned.panels.get("Volume")

    benchmark = spec.benchmark or (
        cfg.data.universes[spec.inherits].benchmark if spec.inherits else None
    )
    if benchmark is None or benchmark not in close.columns:
        raise KeyError(f"universe {universe} has no usable benchmark (looked for {benchmark!r})")

    # --- features ---------------------------------------------------------#
    sleeve = [t for t in cfg.data.allocatable[universe] if t in close.columns]
    equity_sleeve = [t for t in spec.equity_sectors if t in close.columns]
    if spec.inherits:
        parent_sectors = cfg.data.universes[spec.inherits].equity_sectors
        equity_sleeve = [t for t in parent_sectors if t in close.columns] or equity_sleeve

    asset_frame = build_asset_features(
        close[asset_cols],
        None if volume is None else volume[asset_cols],
        benchmark=benchmark,
        tickers=asset_cols,
    )
    cross_frame = build_cross_features(close, sleeve=equity_sleeve)
    macro_frame = build_macro_features(close[macro_cols], index=close.index)

    frame = pd.concat([asset_frame, cross_frame, macro_frame], axis=1)
    if frame.columns.has_duplicates:
        dupes = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"duplicate feature names across builders: {dupes}")

    # --- drop degenerate columns (spec §5.5) ------------------------------ #
    # `SPY_spycorr == 1.0` was defect A7: a constant carried into V1. This
    # drops any all-NaN column, which is the only *structural* degeneracy
    # detectable without fitting; constant-after-warmup is a QA hard check.
    all_nan = [c for c in frame.columns if frame[c].isna().all()]
    if all_nan:
        _log.info("dropping %d all-NaN feature columns: %s", len(all_nan), all_nan[:8])
        frame = frame.drop(columns=all_nan)

    # --- warm-up ----------------------------------------------------------#
    finite = frame.notna().all(axis=1) & np.isfinite(
        frame.to_numpy(dtype="float64", na_value=np.nan)
    ).all(axis=1)
    if not finite.any():
        raise ValueError(
            f"universe {universe}: no session has every feature finite. "
            "Inspect with trim_to_warm=False."
        )
    warm_start = pd.Timestamp(frame.index[finite.to_numpy().argmax()])

    out = frame
    if trim_to_warm:
        out = out.loc[out.index >= warm_start]
    if restrict_to_universe_start:
        out = out.loc[out.index >= pd.Timestamp(spec.start)]
    out.index.name = "date"

    families = _family_map(out.columns, asset_cols)
    result = FeatureSet(
        universe=universe,
        frame=out,
        untrimmed=frame,
        warm_start=warm_start,
        families=families,
        available=cleaned.available,
        filled_sessions=cleaned.filled_sessions,
        long_gaps=cleaned.long_gaps,
    )
    assert_universe_isolation(result, cfg, universe)
    _log.info(
        "universe %s: %d features x %d sessions, %s .. %s (warm from %s)",
        universe,
        out.shape[1],
        out.shape[0],
        out.index[0].date() if len(out) else None,
        out.index[-1].date() if len(out) else None,
        warm_start.date(),
    )
    return result


def _family_map(columns: pd.Index, asset_cols: list[str]) -> dict[str, list[str]]:
    """Group feature columns into asset / cross / macro families."""
    out: dict[str, list[str]] = {"asset": [], "cross": [], "macro": []}
    cross = set(CROSS_FEATURE_NAMES)
    macro = set(MACRO_FEATURE_NAMES) | set(MACRO_OPTIONAL_FEATURE_NAMES)
    asset_suffixes = set(ASSET_FEATURE_NAMES)
    for col in columns:
        name = str(col)
        if name in cross:
            out["cross"].append(name)
        elif name in macro:
            out["macro"].append(name)
        elif any(name.endswith(f"_{s}") for s in asset_suffixes) and any(
            name.startswith(f"{t}_") for t in asset_cols
        ):
            out["asset"].append(name)
        else:  # pragma: no cover - a new builder without a family
            out.setdefault("other", []).append(name)
    return out


def assert_universe_isolation(
    features: FeatureSet, cfg: Config, universe: str
) -> None:
    """Assert the built frame references no ticker outside ``universe``.

    Spec §3.2 hazard 1. For Universe A this is the check that it "provably
    never touches a B-only ticker" (§16 step 1 acceptance).
    """
    allowed = set(cfg.data.tickers(universe))
    forbidden = set(cfg.data.tickers("B")) - allowed if universe == "A" else set()
    # Any ticker in the project that is not in this universe.
    for other in cfg.data.universes:
        forbidden |= set(cfg.data.tickers(other)) - allowed
    forbidden |= set(cfg.data.excluded)

    offending = sorted(
        {t for t in forbidden for c in features.columns if str(c).startswith(f"{t}_")}
    )
    assert not offending, (
        f"universe {universe} feature frame references out-of-universe tickers "
        f"{offending} (spec §3.2 hazard 1)"
    )
    targets = [c for c in features.columns if str(c).startswith(TARGET_PREFIX)]
    assert not targets, (
        f"forward-looking targets leaked into the {universe} feature frame: {targets}. "
        "Targets are for probes and evaluation only (spec §5.4)."
    )
