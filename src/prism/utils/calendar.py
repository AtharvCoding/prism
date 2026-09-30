"""NYSE trading-calendar helpers.

Spec §4.2: the panel is reindexed to the *exchange* calendar, not to
``bdate_range``. Business-day ranges include exchange holidays, which then
look like missing sessions and get forward-filled — quietly inventing prices
for days the market was shut.

All timestamps returned here are tz-naive midnight, matching the convention
``yfinance`` uses for daily bars.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

__all__ = [
    "trading_days",
    "n_sessions_between",
    "shift_sessions",
    "rebalance_dates",
    "session_offset_map",
]


@lru_cache(maxsize=32)
def _calendar(exchange: str):
    import pandas_market_calendars as mcal

    return mcal.get_calendar(exchange)


@lru_cache(maxsize=256)
def _valid_days_cached(exchange: str, start: str, end: str) -> tuple[pd.Timestamp, ...]:
    days = _calendar(exchange).valid_days(start_date=start, end_date=end)
    idx = pd.DatetimeIndex(days)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return tuple(idx.normalize())


def trading_days(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    exchange: str = "NYSE",
) -> pd.DatetimeIndex:
    """Sessions on which ``exchange`` was open, inclusive of both endpoints."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError(f"end {end_ts.date()} precedes start {start_ts.date()}")
    days = _valid_days_cached(exchange, str(start_ts.date()), str(end_ts.date()))
    return pd.DatetimeIndex(days, name="date")


def n_sessions_between(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    exchange: str = "NYSE",
    inclusive: str = "neither",
) -> int:
    """Count sessions strictly between ``start`` and ``end`` by default.

    ``inclusive="neither"`` is the right default for embargo checks: the gap
    between two splits is the number of sessions that belong to *neither*.
    """
    days = trading_days(start, end, exchange)
    if inclusive == "both":
        return len(days)
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    mask = pd.Series(True, index=days)
    if inclusive in ("neither", "right"):
        mask &= days != start_ts
    if inclusive in ("neither", "left"):
        mask &= days != end_ts
    return int(mask.sum())


def shift_sessions(
    date: str | pd.Timestamp,
    n: int,
    exchange: str = "NYSE",
) -> pd.Timestamp:
    """The session ``n`` trading days after ``date`` (negative shifts back).

    ``date`` need not itself be a session; it is first snapped forward (for
    ``n >= 0``) or backward (for ``n < 0``) to the nearest one.
    """
    ts = pd.Timestamp(date).normalize()
    # A generous window: 10 calendar days per session covers holiday clusters.
    pad = pd.Timedelta(days=max(30, abs(n) * 10))
    days = trading_days(ts - pad, ts + pad, exchange)
    if n >= 0:
        pos = days.searchsorted(ts, side="left")
    else:
        pos = days.searchsorted(ts, side="right") - 1
    target = pos + n
    if target < 0 or target >= len(days):
        raise ValueError(
            f"shifting {ts.date()} by {n} sessions leaves the padded calendar window"
        )
    return days[target]


def rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: str,
    weekday: str = "FRI",
) -> pd.DatetimeIndex:
    """Decision dates within ``index``.

    For ``frequency="weekly"`` this is the *last session of each calendar week*
    that falls on or before ``weekday``. Using the last available session
    rather than requiring the literal weekday matters: Good Friday and
    Christmas-week closures would otherwise silently drop a decision.

    ``index`` must be sorted and unique — the caller's panel index, not a
    generated range.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("index must be sorted and unique")
    if frequency == "daily":
        return index.copy()
    if frequency != "weekly":
        raise ValueError(f"unsupported frequency {frequency!r}")

    target = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"].index(weekday.upper())
    eligible = index[index.dayofweek <= target]
    if len(eligible) == 0:
        return pd.DatetimeIndex([], name=index.name)
    # ISO week keys so the year boundary does not merge two distinct weeks.
    iso = eligible.isocalendar()
    keys = pd.Series(
        list(zip(iso["year"].to_numpy(), iso["week"].to_numpy())), index=eligible
    )
    last_per_week = keys.groupby(keys.to_numpy(), sort=False).apply(lambda s: s.index[-1])
    return pd.DatetimeIndex(sorted(last_per_week.to_numpy()), name=index.name)


def session_offset_map(index: pd.DatetimeIndex) -> pd.Series:
    """Map each session to its 0-based position. Used by alignment assertions."""
    return pd.Series(range(len(index)), index=index, name="session")
