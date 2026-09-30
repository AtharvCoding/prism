"""Calendar alignment and cleaning. No bfill, ever.

Spec §4.2. Three rules, each of which the reference pipeline broke or was one
step away from breaking:

1. **Reindex to the exchange calendar**, not to ``bdate_range``. Business-day
   ranges include exchange holidays; treating those as missing sessions and
   filling them invents prices for days the market was shut.
2. **Forward-fill only, capped at 3 consecutive sessions.** A ``bfill``
   writes the future into the past — it was sitting in the reference QA cell
   (defect A2) waiting to fire the first time a series had a gap.
3. **Never interpolate prices**, and never forward-fill across an inception
   date. Assets in this universe have inception dates spanning 1993 to 2007;
   filling backwards from XLK's first trade to 1993 would fabricate nine
   years of history.

The distinction between "pre-inception" and "missing session" is therefore
explicit and carried forward as an availability mask, rather than resolved by
``dropna()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from prism.config import Config
from prism.utils.calendar import trading_days
from prism.utils.logging import get_logger

__all__ = ["CleanResult", "clean_panel", "availability_mask", "align_to_calendar"]

_log = get_logger(__name__)


@dataclass
class CleanResult:
    """A cleaned panel plus everything needed to audit how it was cleaned."""

    #: Field -> cleaned frame, indexed on the exchange calendar.
    panels: dict[str, pd.DataFrame]
    #: True where the asset had traded by that session (post-inception AND
    #: the series has started). Never inferred from NaN alone.
    available: pd.DataFrame
    #: Per-ticker count of sessions filled forward.
    filled_sessions: pd.Series
    #: Gaps longer than the cap, flagged for manual review rather than filled.
    long_gaps: list[dict[str, object]] = field(default_factory=list)

    @property
    def close(self) -> pd.DataFrame:
        return self.panels["Close"]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def as_dict(self) -> dict[str, object]:
        return {
            "sessions": len(self.index),
            "first_date": str(self.index[0].date()) if len(self.index) else None,
            "last_date": str(self.index[-1].date()) if len(self.index) else None,
            "filled_sessions": {k: int(v) for k, v in self.filled_sessions.items()},
            "long_gaps": self.long_gaps,
        }


def align_to_calendar(
    frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    exchange: str = "NYSE",
) -> pd.DataFrame:
    """Reindex ``frame`` onto the exchange calendar for ``[start, end]``.

    Duplicate and unsorted index entries are hard errors, not something to
    quietly de-duplicate: they mean the upstream download is wrong. (The
    reference rolling HMM loop produced 1,557 rows for 1,509 trading days by
    using an inclusive slice — defect B5.)
    """
    idx = pd.DatetimeIndex(frame.index)
    if idx.has_duplicates:
        dupes = idx[idx.duplicated()].unique()
        raise ValueError(
            f"index has {len(dupes)} duplicate dates, first {dupes[0].date()}; "
            "refusing to guess which row is correct"
        )
    if not idx.is_monotonic_increasing:
        raise ValueError("index is not sorted ascending")

    calendar = trading_days(start, end, exchange)
    extra = idx.difference(calendar)
    if len(extra) > 0:
        # Data on a non-session is a download bug (or a calendar mismatch);
        # dropping it silently would hide it.
        _log.warning(
            "%d dated rows fall outside the %s calendar and are dropped, first %s",
            len(extra),
            exchange,
            extra[0].date(),
        )
    return frame.reindex(calendar)


def availability_mask(
    close: pd.DataFrame,
    inception: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """True where a ticker is genuinely tradable on that session.

    A ticker is available from the later of (a) its configured inception date
    and (b) its first non-null observation. Both bounds matter: the config
    date is the ground truth about when the fund existed, while the first
    observation catches a data vendor whose history starts later than the
    fund's actual launch.
    """
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    for ticker in close.columns:
        series = close[ticker]
        observed = series.first_valid_index()
        if observed is None:
            _log.warning("ticker %s has no observations at all", ticker)
            continue
        declared = inception.get(ticker)
        start = observed if declared is None else max(pd.Timestamp(declared), observed)
        mask.loc[mask.index >= start, ticker] = True
    return mask


def clean_panel(
    panels: dict[str, pd.DataFrame],
    cfg: Config,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tickers: list[str] | None = None,
) -> CleanResult:
    """Align to the exchange calendar and forward-fill within availability.

    Returns a :class:`CleanResult`; NaNs outside each ticker's availability
    window are left in place deliberately so that feature construction can
    decide what to do about them rather than inheriting a silently filled
    panel.
    """
    exchange = cfg.data.calendar.exchange
    cap = cfg.data.calendar.max_ffill_days
    inception = {k: pd.Timestamp(v) for k, v in cfg.data.inception.items()}

    close = panels["Close"]
    cols = list(close.columns) if tickers is None else [t for t in tickers if t in close.columns]
    start_ts = pd.Timestamp(start) if start is not None else pd.DatetimeIndex(close.index)[0]
    end_ts = pd.Timestamp(end) if end is not None else pd.DatetimeIndex(close.index)[-1]

    aligned = {
        fld: align_to_calendar(frame.reindex(columns=cols), start_ts, end_ts, exchange)
        for fld, frame in panels.items()
    }
    available = availability_mask(aligned["Close"], inception)

    filled = pd.Series(0, index=cols, dtype=int, name="filled_sessions")
    long_gaps: list[dict[str, object]] = []
    out: dict[str, pd.DataFrame] = {}

    for fld, frame in aligned.items():
        frame = frame.copy()
        # Only fill inside the availability window. Outside it the value is
        # not missing, it does not exist.
        masked = frame.where(available)
        filled_frame = masked.ffill(limit=cap)
        filled_frame = filled_frame.where(available)

        if fld == "Close":
            was_na = masked.isna() & available
            now_ok = filled_frame.notna() & available
            filled = (was_na & now_ok).sum().reindex(cols).fillna(0).astype(int)
            filled.name = "filled_sessions"
            # Anything still NaN inside the availability window is a gap the
            # cap refused to bridge: report it, do not widen the cap.
            residual = filled_frame.isna() & available
            for ticker in cols:
                gaps = _runs(residual[ticker])
                for gap_start, gap_end, length in gaps:
                    long_gaps.append(
                        {
                            "ticker": ticker,
                            "start": str(gap_start.date()),
                            "end": str(gap_end.date()),
                            "sessions": int(length),
                            "note": f"exceeds max_ffill_days={cap}; flagged for review",
                        }
                    )
        out[fld] = filled_frame

    if long_gaps:
        _log.warning(
            "%d gaps exceed the %d-session forward-fill cap and were left as NaN",
            len(long_gaps),
            cap,
        )

    return CleanResult(
        panels=out, available=available, filled_sessions=filled, long_gaps=long_gaps
    )


def _runs(flags: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Contiguous runs of True in ``flags``, as (start, end, length)."""
    values = flags.to_numpy(dtype=bool)
    if not values.any():
        return []
    idx = flags.index
    padded = np.concatenate(([False], values, [False]))
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1) - 1
    return [(idx[s], idx[e], int(e - s + 1)) for s, e in zip(starts, ends)]
