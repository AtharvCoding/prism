"""Macro features: volatility, term structure, credit, dollar, oil. Spec §5.3.

Two substitutions are forced by what the snapshot can actually supply, and
are recorded here rather than buried:

**The 10y-2y slope is not obtainable.** Yahoo's Treasury yield indices are
``^IRX`` (13-week), ``^FVX`` (5-year), ``^TNX`` (10-year) and ``^TYX``
(30-year). There is no 2-year series. The primary slope is therefore
**10y minus 3m** (``curve_slope_10y_3m``), which is the other classical
recession signal and is in fact the better-documented one for that purpose;
10y-5y is reported alongside. See DECISIONS.md.

**VIX term structure requires ``^VIX3M``**, whose history begins 2002-12 and
so does not cover Universe A's 1999 start. It is therefore a **Universe
B-only** feature (``macro_extra``), consistent with §3.1's design rather than
an exception to it. The builder degrades gracefully with a warning if the
series is absent from the snapshot, so a vendor outage at Step 1 does not
abort the run.

**Yields are not prices.** ``^TNX`` and friends are quoted in percentage
points and can be zero or negative. Every yield-derived quantity here uses
*arithmetic* differences; taking a log return of a yield would be both
dimensionally wrong and undefined at the zero bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prism.utils.logging import get_logger

__all__ = [
    "MACRO_FEATURE_NAMES",
    "MACRO_OPTIONAL_FEATURE_NAMES",
    "YIELD_TICKERS",
    "build_macro_features",
]

_log = get_logger(__name__)

#: Macro series that are yields in percentage points, not prices.
YIELD_TICKERS: tuple[str, ...] = ("^IRX", "^FVX", "^TNX", "^TYX")

#: Always produced when ``^VIX``, ``^TNX`` and ``^IRX`` are present.
MACRO_FEATURE_NAMES: tuple[str, ...] = (
    "vix_level",
    "vix_change",
    "curve_slope_10y_3m",
    "curve_slope_10y_5y",
    "curve_change",
)

#: Produced only when the underlying series is in the universe. Every one of
#: these is Universe B-only by construction (§3.1).
MACRO_OPTIONAL_FEATURE_NAMES: tuple[str, ...] = (
    "vix_term_structure",
    "credit_proxy",
    "credit_proxy_change",
    "credit_proxy_change_20",
    "dollar_return_20",
    "oil_vol_20",
)


def _need(frame: pd.DataFrame, ticker: str) -> pd.Series | None:
    if ticker not in frame.columns:
        return None
    return frame[ticker]


def build_macro_features(
    macro: pd.DataFrame,
    *,
    index: pd.DatetimeIndex | None = None,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Macro feature frame from a panel of macro *levels*.

    Parameters
    ----------
    macro
        Columns are macro tickers (``^VIX``, ``^TNX``, ``HYG``, ...), rows are
        sessions. Yield columns are in percentage points; ``HYG``/``LQD`` are
        total-return prices; ``^VIX`` is an index level.
    index
        Reindex the result onto this index. Passing the panel's own trading
        calendar keeps macro features aligned with the asset features even
        where a macro series has a different session set.

    The frame retains leading NaN rows; trimming is the orchestrator's job.
    """
    src = macro if index is None else macro.reindex(pd.DatetimeIndex(index))
    out = pd.DataFrame(index=src.index)

    vix = _need(src, "^VIX")
    if vix is None:
        raise KeyError("^VIX is required for macro features but is absent from the panel")
    positive_vix = vix.where(vix > 0)
    out["vix_level"] = vix
    # Log change: VIX is a positive, right-skewed, mean-reverting level, so its
    # log difference is much closer to white noise than its arithmetic
    # difference. This is the series the HMM consumes (§8.2), where low
    # autocorrelation is the whole requirement (defect B3).
    out["vix_change"] = np.log(positive_vix).diff()

    tnx, irx, fvx = _need(src, "^TNX"), _need(src, "^IRX"), _need(src, "^FVX")
    if tnx is None or irx is None:
        raise KeyError("^TNX and ^IRX are required for the yield-curve slope")
    out["curve_slope_10y_3m"] = tnx - irx
    out["curve_slope_10y_5y"] = (tnx - fvx) if fvx is not None else np.nan
    # Primary slope's daily change — the HMM's `curve_change` observation.
    out["curve_change"] = out["curve_slope_10y_3m"].diff()

    vix3m = _need(src, "^VIX3M")
    if vix3m is not None:
        out["vix_term_structure"] = np.log(positive_vix) - np.log(vix3m.where(vix3m > 0))
    else:
        _log.info(
            "^VIX3M absent from the macro panel; vix_term_structure not produced "
            "(Universe B-only feature, spec §5.3)"
        )

    hyg, lqd = _need(src, "HYG"), _need(src, "LQD")
    if hyg is not None and lqd is not None:
        # The ETF-ratio credit proxy. Spec §5.3 warns against FRED's ICE BofA
        # high-yield OAS series, whose history is restricted; the HYG/LQD
        # ratio isolates credit spread from duration because both legs carry
        # broadly similar interest-rate exposure.
        proxy = np.log(hyg.where(hyg > 0)) - np.log(lqd.where(lqd > 0))
        out["credit_proxy"] = proxy
        out["credit_proxy_change"] = proxy.diff()        # 1d — HMM observation
        out["credit_proxy_change_20"] = proxy.diff(20)   # 20d — spec §5.3
    else:
        _log.info("HYG/LQD absent; credit proxy not produced (Universe B-only)")

    dollar = _need(src, "DX-Y.NYB")
    if dollar is not None:
        pos = dollar.where(dollar > 0)
        out["dollar_return_20"] = np.log(pos) - np.log(pos.shift(vol_window))
    else:
        _log.info("DX-Y.NYB absent; dollar_return_20 not produced (Universe B-only)")

    oil = _need(src, "CL=F")
    if oil is not None:
        pos = oil.where(oil > 0)
        oil_ret = np.log(pos).diff()
        out["oil_vol_20"] = oil_ret.rolling(vol_window).std() * np.sqrt(252)
    else:
        _log.info("CL=F absent; oil_vol_20 not produced (Universe B-only)")

    ordered = [c for c in (*MACRO_FEATURE_NAMES, *MACRO_OPTIONAL_FEATURE_NAMES) if c in out]
    out = out.reindex(columns=ordered)
    out.index.name = src.index.name or "date"
    return out
