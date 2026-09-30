"""Write-once snapshot creation.

Spec §4.1. The reference pipeline re-downloaded and overwrote its "frozen"
data on every run with no hash, date or library version recorded (defect A1).
Because ``auto_adjust=True`` prices are restated whenever a new dividend is
paid, that silently rewrote history: two runs a month apart were not
comparing the same dataset.

This module downloads once, into ``data/raw/snapshot_<YYYYMMDD>/``, and
refuses to overwrite. Everything downstream reads the snapshot path from
config and never touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from prism.config import Config
from prism.utils.hashing import (
    hash_object,
    library_versions,
    pip_freeze,
    sha256_file,
    utc_now_iso,
)
from prism.utils.logging import get_logger

__all__ = ["SnapshotExistsError", "snapshot_dir", "create_snapshot", "download_panel"]

_log = get_logger(__name__)

#: Fields kept from the OHLCV download. ``Close`` is already total-return
#: adjusted because ``auto_adjust=True`` (verified correct in the reference).
OHLCV_FIELDS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


class SnapshotExistsError(RuntimeError):
    """Raised when a snapshot directory already exists and overwrite is off."""


def snapshot_dir(cfg: Config, date: str | pd.Timestamp | None = None) -> Path:
    """Path of the snapshot directory for ``date`` (config's date by default)."""
    snap_date = date if date is not None else cfg.data.snapshot.date
    if snap_date is None:
        raise ValueError(
            "no snapshot date: pass one explicitly, or set snapshot.date in "
            "configs/data.yaml after running scripts/00_snapshot.py"
        )
    stamp = pd.Timestamp(snap_date).strftime("%Y%m%d")
    return cfg.root / cfg.data.snapshot.dir / f"snapshot_{stamp}"


def download_panel(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Download daily bars for ``tickers`` as one frame per OHLCV field.

    ``auto_adjust=True`` yields split- and dividend-adjusted prices, i.e.
    total returns. ``end`` is inclusive here, unlike ``yfinance``'s
    half-open convention, so the snapshot contains the date the caller asked
    for.

    This is the only function in the project that touches the network.
    """
    import yfinance as yf

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    _log.info(
        "downloading %d tickers, %s .. %s", len(tickers), start_ts.date(), end_ts.date()
    )
    raw = yf.download(
        tickers,
        start=str(start_ts.date()),
        end=str((end_ts + pd.Timedelta(days=1)).date()),  # yfinance end is exclusive
        auto_adjust=True,
        progress=False,
        actions=False,
        group_by="column",
        threads=True,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data; refusing to write an empty snapshot")

    if not isinstance(raw.columns, pd.MultiIndex):
        # Single ticker: yfinance drops the ticker level.
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    out: dict[str, pd.DataFrame] = {}
    available = set(raw.columns.get_level_values(0))
    for field in OHLCV_FIELDS:
        if field not in available:
            _log.warning("field %s absent from download; skipping", field)
            continue
        frame = raw[field].copy()
        missing = [t for t in tickers if t not in frame.columns]
        if missing:
            _log.warning("field %s missing tickers %s", field, missing)
        frame = frame.reindex(columns=[t for t in tickers if t in frame.columns])
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame.index.name = "date"
        frame = frame.sort_index()
        out[field] = frame
    if "Close" not in out:
        raise RuntimeError("download produced no Close prices")
    return out


def create_snapshot(
    cfg: Config,
    *,
    date: str | pd.Timestamp | None = None,
    force: bool = False,
    tickers: list[str] | None = None,
    panels: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Create a snapshot directory with a MANIFEST.json and refuse to clobber.

    Parameters
    ----------
    panels
        Pre-fetched panels, keyed by OHLCV field. Supplying these skips the
        network entirely, which is how the test suite exercises this code
        path against deterministic synthetic data.
    force
        Permit writing into an existing directory. Requires
        ``snapshot.allow_overwrite: true`` in config as well, so a single
        careless CLI flag cannot destroy a snapshot that results depend on.
    """
    snap_date = pd.Timestamp(date) if date is not None else pd.Timestamp.utcnow().normalize()
    out_dir = snapshot_dir(cfg, snap_date)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SnapshotExistsError(
                f"{out_dir} already exists. A snapshot is write-once (spec §4.1): "
                "re-downloading restates adjusted prices and silently changes history. "
                "Use a new snapshot date, or --force with snapshot.allow_overwrite=true."
            )
        if not cfg.data.snapshot.allow_overwrite:
            raise SnapshotExistsError(
                f"--force was given but snapshot.allow_overwrite is false in config; "
                f"refusing to overwrite {out_dir}"
            )
        _log.warning("overwriting existing snapshot at %s", out_dir)

    ticker_list = tickers if tickers is not None else cfg.data.tickers("B")
    # Download from the earliest inception among the requested tickers, not
    # from the universe's declared start. A universe start is a *feature*
    # boundary; the snapshot should hold everything that exists, so that a
    # later change to warm-up or to a lookback window does not require a
    # re-download — which would restate every adjusted price (§4.1).
    download_start = min(
        pd.Timestamp(cfg.data.inception[t]) for t in ticker_list if t in cfg.data.inception
    )
    if panels is None:
        panels = download_panel(
            ticker_list, download_start, cfg.data.snapshot.download_end
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Macro series are stored separately from tradable OHLCV so that the
    # A/B universe split is visible in the file layout, not just in code.
    macro_tickers = [
        t
        for t in ticker_list
        if t in set(cfg.data.universes["A"].macro) | set(cfg.data.universes["B"].macro_extra)
    ]
    asset_tickers = [t for t in ticker_list if t not in macro_tickers]

    written: dict[str, dict[str, Any]] = {}

    def _write(name: str, frame: pd.DataFrame) -> None:
        path = out_dir / name
        frame.to_parquet(path, index=True)
        written[name] = {
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "first_date": str(frame.index[0].date()) if len(frame) else None,
            "last_date": str(frame.index[-1].date()) if len(frame) else None,
            "na_count": int(frame.isna().sum().sum()),
        }

    ohlcv = pd.concat(
        {field: panels[field].reindex(columns=asset_tickers) for field in panels},
        axis=1,
        names=["field", "ticker"],
    )
    _write("ohlcv.parquet", ohlcv)

    if macro_tickers:
        macro = panels["Close"].reindex(columns=macro_tickers)
        _write("macro.parquet", macro)

    manifest: dict[str, Any] = {
        "snapshot_date": str(snap_date.date()),
        "downloaded_utc": utc_now_iso(),
        "download_range": [
            str(download_start.date()),
            str(pd.Timestamp(cfg.data.snapshot.download_end).date()),
        ],
        "auto_adjust": True,
        "tickers": {"assets": asset_tickers, "macro": macro_tickers},
        "files": written,
        "libraries": library_versions(),
        "pip_freeze": pip_freeze(),
        "synthetic": panels is not None and tickers is not None,
    }
    manifest["snapshot_hash"] = hash_object(
        {name: meta["sha256"] for name, meta in sorted(written.items())}
    )
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _log.info(
        "snapshot written to %s (hash %s)", out_dir, manifest["snapshot_hash"][:12]
    )
    return out_dir
