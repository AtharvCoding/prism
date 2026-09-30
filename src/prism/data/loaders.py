"""Snapshot loading and the holdout lock.

Spec §6.3: ``load_holdout()`` raises unless called with ``final=True`` **and**
the environment variable ``PRISM_ALLOW_HOLDOUT=1``. Two independent gates,
because one is an accident waiting to happen: a default argument gets flipped
during a refactor, or an env var gets exported in a shell profile and
forgotten. Requiring both means the holdout can only be read by someone who
did two deliberate things at once, and every such read is logged.

The holdout is 2024-01-01 .. 2026-09-30 and is touched exactly once, at the
very end of Phase B (§16 step 5). Not in Phase A.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prism.config import Config
from prism.data.download import snapshot_dir
from prism.splits import build_split_plan
from prism.utils.hashing import sha256_file, utc_now_iso
from prism.utils.logging import get_logger

__all__ = [
    "HoldoutLockError",
    "Snapshot",
    "load_snapshot",
    "load_holdout",
    "HOLDOUT_ENV_VAR",
    "HOLDOUT_ACCESS_LOG",
]

_log = get_logger(__name__)

HOLDOUT_ENV_VAR = "PRISM_ALLOW_HOLDOUT"
HOLDOUT_ACCESS_LOG = "reports/logs/holdout_access.jsonl"


class HoldoutLockError(RuntimeError):
    """Raised on any attempt to read the holdout without both explicit gates."""


@dataclass(frozen=True)
class Snapshot:
    """An immutable raw snapshot, with its manifest and verified hashes."""

    path: Path
    manifest: dict[str, object]
    panels: dict[str, pd.DataFrame]
    macro: pd.DataFrame | None

    @property
    def snapshot_hash(self) -> str:
        return str(self.manifest["snapshot_hash"])

    @property
    def close(self) -> pd.DataFrame:
        return self.panels["Close"]

    @property
    def volume(self) -> pd.DataFrame | None:
        return self.panels.get("Volume")

    def prices(self, tickers: list[str]) -> pd.DataFrame:
        """Close prices for ``tickers``, drawing from assets or macro as needed.

        This is the one accessor feature builders should use: it makes the
        universe restriction explicit at the call site, so a builder for
        Universe A cannot accidentally reach a Universe B-only ticker just
        because the snapshot happens to contain it.
        """
        frames = []
        asset_cols = [t for t in tickers if t in self.close.columns]
        if asset_cols:
            frames.append(self.close[asset_cols])
        if self.macro is not None:
            macro_cols = [t for t in tickers if t in self.macro.columns]
            if macro_cols:
                frames.append(self.macro[macro_cols])
        if not frames:
            raise KeyError(f"none of {tickers} are present in the snapshot")
        out = pd.concat(frames, axis=1)
        missing = [t for t in tickers if t not in out.columns]
        if missing:
            raise KeyError(f"snapshot does not contain {missing}")
        return out.reindex(columns=tickers)


def load_snapshot(
    cfg: Config,
    *,
    date: str | pd.Timestamp | None = None,
    verify_hashes: bool = True,
) -> Snapshot:
    """Load the raw snapshot named by config (or ``date``) and verify it.

    Hash verification is on by default. A snapshot whose files no longer match
    its manifest is not a warning — every result traced to that hash is now
    untraceable, so it is an error.
    """
    path = snapshot_dir(cfg, date)
    manifest_path = path / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no MANIFEST.json in {path}. Run scripts/00_snapshot.py first; downstream "
            "code never reads the network (spec §4.1)."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if verify_hashes:
        for name, meta in manifest.get("files", {}).items():
            file_path = path / name
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path} is named in the manifest but absent")
            actual = sha256_file(file_path)
            expected = meta["sha256"]
            if actual != expected:
                raise ValueError(
                    f"{file_path} hash {actual[:12]} does not match manifest "
                    f"{expected[:12]}. The snapshot has been modified; results traced to "
                    "this hash are no longer reproducible."
                )

    ohlcv = pd.read_parquet(path / "ohlcv.parquet")
    if not isinstance(ohlcv.columns, pd.MultiIndex):
        raise ValueError("ohlcv.parquet must have a (field, ticker) MultiIndex on columns")
    panels = {
        str(field): ohlcv[field].copy() for field in ohlcv.columns.get_level_values(0).unique()
    }
    for frame in panels.values():
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame.index.name = "date"

    macro_path = path / "macro.parquet"
    macro = None
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
        macro.index = pd.DatetimeIndex(macro.index).normalize()
        macro.index.name = "date"

    _log.info(
        "loaded snapshot %s (%s), %d sessions",
        path.name,
        str(manifest["snapshot_hash"])[:12],
        len(panels["Close"]),
    )
    return Snapshot(path=path, manifest=manifest, panels=panels, macro=macro)


def load_holdout(
    cfg: Config,
    frame: pd.DataFrame | None = None,
    *,
    final: bool = False,
    reason: str = "",
) -> pd.DataFrame:
    """Return holdout-period rows. Locked behind two independent gates.

    Parameters
    ----------
    frame
        Any date-indexed frame to slice. If omitted, the snapshot's close
        prices are used.
    final
        Must be ``True``. Spec §6.3.
    reason
        Free text recorded in the access log. Required, so that the log says
        *why* the holdout was opened, not merely that it was.

    Raises
    ------
    HoldoutLockError
        If ``final`` is not True, if ``PRISM_ALLOW_HOLDOUT`` is not ``"1"``,
        or if no reason was given.
    """
    if not final:
        raise HoldoutLockError(
            "the holdout is locked: load_holdout(..., final=True) is required. "
            "It is evaluated exactly once, at the end of Phase B (§16 step 5), "
            "after every decision is frozen. This is Phase A."
        )
    if os.environ.get(HOLDOUT_ENV_VAR) != "1":
        raise HoldoutLockError(
            f"the holdout is locked: environment variable {HOLDOUT_ENV_VAR}=1 is "
            "required in addition to final=True (spec §6.3). Two gates, so that "
            "neither a refactored default nor a stale shell export can open it alone."
        )
    if not reason.strip():
        raise HoldoutLockError(
            "load_holdout requires a non-empty reason; it is written to "
            f"{HOLDOUT_ACCESS_LOG} permanently."
        )

    start, end = cfg.data.split("holdout")
    _record_holdout_access(cfg, reason=reason, start=start, end=end)
    _log.warning(
        "HOLDOUT OPENED %s .. %s — reason: %s", start.date(), end.date(), reason
    )

    if frame is None:
        frame = load_snapshot(cfg).close
    return frame.loc[start:end]


def _record_holdout_access(
    cfg: Config, *, reason: str, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    """Append an immutable line to the holdout access log."""
    from prism.utils.hashing import git_commit

    path = cfg.root / HOLDOUT_ACCESS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "utc": utc_now_iso(),
        "reason": reason,
        "range": [str(start.date()), str(end.date())],
        "git": git_commit(cfg.root),
        "pid": os.getpid(),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def assert_not_holdout(cfg: Config, index: pd.DatetimeIndex, *, context: str) -> None:
    """Raise if ``index`` intersects the holdout window.

    Called by every Phase A stage that writes a processed artefact, so that a
    mis-specified date range cannot pull holdout rows into a feature frame by
    accident. This is the cheap guard that makes the expensive one unnecessary.
    """
    start, end = cfg.data.split("holdout")
    ts = pd.DatetimeIndex(index)
    leaked = ts[(ts >= start) & (ts <= end)]
    if len(leaked):
        raise HoldoutLockError(
            f"{context}: {len(leaked)} rows fall inside the holdout window "
            f"({start.date()} .. {end.date()}), first {leaked[0].date()}. "
            "Phase A must never read the holdout (spec §0.4.5)."
        )


def non_holdout_span(cfg: Config) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The full date range Phase A may legitimately read: start of A to test end."""
    plan = build_split_plan(cfg)
    return cfg.data.start("A"), plan["test"].declared_end
