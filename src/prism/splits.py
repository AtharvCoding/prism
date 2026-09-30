"""Split objects, embargo, and walk-forward folds.

Spec §6.

Embargo semantics — a decision the spec leaves implicit
-------------------------------------------------------
``configs/data.yaml`` declares splits that are *contiguous* in calendar terms
(train ends 2017-12-31, val starts 2018-01-01) while also declaring
``embargo_days: 25``. Taken literally those two statements contradict each
other: there are zero trading sessions between the two boundaries.

The spec's intent (§6.1) is unambiguous about *why* the embargo exists: the
lookback features may legitimately read across a boundary, but the
**forward-looking targets** computed for days near the end of one split
overlap the next split. So the embargo is applied by **purging the end of the
earlier split**, never by moving the later split's start:

    declared:   train [2007-04-04 .. 2017-12-31]  val [2018-01-01 .. 2018-12-31]
    effective:  train [2007-04-04 .. 2017-11-22]  val [2018-01-01 .. 2018-12-31]
                                     ^^^^^^^^^^ last 25 sessions purged

This preserves the reported evaluation windows exactly as specified — val is
the 2018 calendar year, test is 2019-2023, holdout is 2024 onward — and pays
the whole cost of the embargo out of *training* data, which is the correct
side to pay on. Every consumer must use ``Split.sessions`` (effective), never
the declared range.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from prism.config import Config
from prism.utils.calendar import n_sessions_between, trading_days

__all__ = ["Split", "SplitPlan", "WalkForwardFold", "build_split_plan", "expanding_folds"]

#: Splits in chronological order. The embargo is applied between consecutive
#: members of this sequence.
SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test", "holdout")


@dataclass(frozen=True)
class Split:
    """One evaluation window, with declared and embargo-purged bounds."""

    name: str
    declared_start: pd.Timestamp
    declared_end: pd.Timestamp
    effective_start: pd.Timestamp
    effective_end: pd.Timestamp
    exchange: str = "NYSE"
    #: Number of sessions purged from the end by the embargo.
    purged_sessions: int = 0

    @cached_property
    def sessions(self) -> pd.DatetimeIndex:
        """Trading sessions in the *effective* range."""
        if self.effective_end < self.effective_start:
            return pd.DatetimeIndex([], name="date")
        return trading_days(self.effective_start, self.effective_end, self.exchange)

    def __len__(self) -> int:
        return len(self.sessions)

    def contains(self, date: str | pd.Timestamp) -> bool:
        return self.effective_start <= pd.Timestamp(date).normalize() <= self.effective_end

    def mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """Boolean mask selecting this split's sessions out of ``index``."""
        ts = pd.DatetimeIndex(index)
        return pd.Series(
            (ts >= self.effective_start) & (ts <= self.effective_end), index=ts, name=self.name
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "declared": [str(self.declared_start.date()), str(self.declared_end.date())],
            "effective": [str(self.effective_start.date()), str(self.effective_end.date())],
            "sessions": len(self),
            "purged_sessions": self.purged_sessions,
        }


@dataclass(frozen=True)
class SplitPlan:
    """The full set of splits plus the embargo that separates them."""

    splits: dict[str, Split]
    embargo_days: int
    exchange: str = "NYSE"

    def __getitem__(self, name: str) -> Split:
        try:
            return self.splits[name]
        except KeyError:
            raise KeyError(f"unknown split {name!r}; known: {list(self.splits)}") from None

    def __iter__(self):
        return iter(self.splits.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.splits)

    def gap_sessions(self, earlier: str, later: str) -> int:
        """Sessions belonging to neither split, between the two."""
        return n_sessions_between(
            self[earlier].effective_end, self[later].effective_start, self.exchange
        )

    def validate(self) -> None:
        """Assert the §7.4 properties. Raises ``AssertionError`` on violation."""
        present = [n for n in SPLIT_ORDER if n in self.splits]
        for name in present:
            split = self[name]
            assert split.effective_start <= split.effective_end, (
                f"split {name!r} is empty after embargo purging: "
                f"{split.effective_start.date()} > {split.effective_end.date()}"
            )
            assert len(split) > 0, f"split {name!r} contains no trading sessions"
        for earlier, later in zip(present, present[1:]):
            lo, hi = self[earlier], self[later]
            assert lo.effective_end < hi.effective_start, (
                f"splits {earlier!r} and {later!r} overlap: {lo.effective_end.date()} "
                f">= {hi.effective_start.date()}"
            )
            gap = self.gap_sessions(earlier, later)
            assert gap >= self.embargo_days, (
                f"embargo violation between {earlier!r} and {later!r}: {gap} sessions "
                f"separate them, {self.embargo_days} required"
            )
            overlap = lo.sessions.intersection(hi.sessions)
            assert len(overlap) == 0, f"{earlier!r} and {later!r} share {len(overlap)} sessions"

    def as_dict(self) -> dict[str, object]:
        return {
            "embargo_days": self.embargo_days,
            "exchange": self.exchange,
            "splits": [s.as_dict() for s in self],
            "gaps": {
                f"{a}->{b}": self.gap_sessions(a, b)
                for a, b in zip(self.names, self.names[1:])
            },
        }


def build_split_plan(cfg: Config, *, include_holdout: bool = True) -> SplitPlan:
    """Construct the embargo-purged split plan from config.

    ``include_holdout=False`` omits the holdout entirely, which is the right
    default for anything that is not ``scripts/99_final_holdout.py``. Note
    that omitting it also removes the embargo purge at the end of ``test``;
    pass ``include_holdout=True`` (the default) so ``test`` is purged against
    the holdout boundary even though the holdout is never read.
    """
    exchange = cfg.data.calendar.exchange
    embargo = cfg.data.splits.embargo_days
    names = [n for n in SPLIT_ORDER if include_holdout or n != "holdout"]

    declared = {n: cfg.data.split(n) for n in names}
    # A calendar spanning every split, used to locate the purge boundary.
    span_start = min(s for s, _ in declared.values())
    span_end = max(e for _, e in declared.values())
    calendar = trading_days(span_start, span_end, exchange)

    splits: dict[str, Split] = {}
    for i, name in enumerate(names):
        start, end = declared[name]
        eff_end, purged = end, 0
        if i + 1 < len(names):
            next_start = declared[names[i + 1]][0]
            # Index of the first session on/after the next split's start.
            j = int(calendar.searchsorted(next_start, side="left"))
            target = j - embargo - 1
            if target < 0:
                raise ValueError(
                    f"embargo of {embargo} sessions before {next_start.date()} runs off the "
                    "start of the calendar"
                )
            boundary = calendar[target]
            if boundary < eff_end:
                # Count how many of this split's sessions are being dropped.
                own = trading_days(start, end, exchange)
                purged = int((own > boundary).sum())
                eff_end = boundary
        splits[name] = Split(
            name=name,
            declared_start=start,
            declared_end=end,
            effective_start=start,
            effective_end=eff_end,
            exchange=exchange,
            purged_sessions=purged,
        )

    plan = SplitPlan(splits=splits, embargo_days=embargo, exchange=exchange)
    plan.validate()
    return plan


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window refit.

    Spec §6.2: "Each fold records ``(fit_start, fit_end, apply_start,
    apply_end)`` and asserts ``fit_end < apply_start``." The assertion lives in
    ``__post_init__`` so an invalid fold cannot be constructed at all.
    """

    index: int
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    apply_start: pd.Timestamp
    apply_end: pd.Timestamp
    embargo_days: int = 0
    exchange: str = "NYSE"

    def __post_init__(self) -> None:
        if not self.fit_start <= self.fit_end:
            raise ValueError(f"fold {self.index}: fit_start after fit_end")
        if not self.apply_start <= self.apply_end:
            raise ValueError(f"fold {self.index}: apply_start after apply_end")
        if not self.fit_end < self.apply_start:
            raise ValueError(
                f"fold {self.index}: fit_end {self.fit_end.date()} is not before "
                f"apply_start {self.apply_start.date()} — the refit would see the data "
                "it is applied to (spec §6.2)"
            )
        if self.embargo_days:
            gap = n_sessions_between(self.fit_end, self.apply_start, self.exchange)
            if gap < self.embargo_days:
                raise ValueError(
                    f"fold {self.index}: only {gap} sessions between fit_end and "
                    f"apply_start, {self.embargo_days} required"
                )

    @cached_property
    def fit_sessions(self) -> pd.DatetimeIndex:
        return trading_days(self.fit_start, self.fit_end, self.exchange)

    @cached_property
    def apply_sessions(self) -> pd.DatetimeIndex:
        return trading_days(self.apply_start, self.apply_end, self.exchange)

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "fit_start": str(self.fit_start.date()),
            "fit_end": str(self.fit_end.date()),
            "apply_start": str(self.apply_start.date()),
            "apply_end": str(self.apply_end.date()),
            "fit_sessions": len(self.fit_sessions),
            "apply_sessions": len(self.apply_sessions),
        }


_CADENCE_ALIAS = {"monthly": "MS", "quarterly": "QS", "annual": "YS"}


def expanding_folds(
    fit_start: str | pd.Timestamp,
    first_apply_start: str | pd.Timestamp,
    apply_end: str | pd.Timestamp,
    cadence: str,
    *,
    embargo_days: int = 0,
    exchange: str = "NYSE",
) -> list[WalkForwardFold]:
    """Expanding-window walk-forward folds.

    Spec §6.2 requires an *expanding* window, not a rolling one: "A fixed
    5-year window starting in 2013 contains no crisis, which cripples the
    HMM's crisis state" (defect B7). Every fold therefore shares ``fit_start``
    and extends ``fit_end`` forward.

    The fit window ends ``embargo_days`` sessions before the fold's apply
    window begins, so forward-looking targets computed at the end of the fit
    window cannot reach into the data the fitted object is applied to.
    """
    try:
        freq = _CADENCE_ALIAS[cadence]
    except KeyError:
        raise ValueError(
            f"unsupported cadence {cadence!r}; known: {sorted(_CADENCE_ALIAS)}"
        ) from None

    fit_start_ts = pd.Timestamp(fit_start).normalize()
    first_apply = pd.Timestamp(first_apply_start).normalize()
    end_ts = pd.Timestamp(apply_end).normalize()
    if first_apply <= fit_start_ts:
        raise ValueError("first_apply_start must be after fit_start")

    calendar = trading_days(fit_start_ts, end_ts, exchange)
    # Period starts on/after the first apply date; each becomes one fold.
    boundaries = pd.date_range(start=first_apply, end=end_ts, freq=freq, normalize=True)
    if len(boundaries) == 0 or boundaries[0] > first_apply:
        boundaries = pd.DatetimeIndex([first_apply]).append(boundaries)

    folds: list[WalkForwardFold] = []
    for i, boundary in enumerate(boundaries):
        # First session on/after the boundary.
        j = int(calendar.searchsorted(boundary, side="left"))
        if j >= len(calendar):
            break
        apply_from = calendar[j]
        if i + 1 < len(boundaries):
            k = int(calendar.searchsorted(boundaries[i + 1], side="left"))
            if k <= j:
                continue
            apply_to = calendar[k - 1]
        else:
            apply_to = calendar[-1]

        target = j - embargo_days - 1
        if target < 0:
            continue
        fold_fit_end = calendar[target]
        if fold_fit_end <= fit_start_ts:
            continue
        folds.append(
            WalkForwardFold(
                index=len(folds),
                fit_start=fit_start_ts,
                fit_end=fold_fit_end,
                apply_start=apply_from,
                apply_end=apply_to,
                embargo_days=embargo_days,
                exchange=exchange,
            )
        )
    if not folds:
        raise ValueError("no walk-forward folds could be constructed from these bounds")
    return folds
