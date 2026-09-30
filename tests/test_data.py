"""Snapshot write-once behaviour, cleaning rules, and QA thresholds.

Spec §4. Step 0 covers "Repo, config, snapshot, splits, tests", and the
snapshot's central promise — that it cannot be silently rewritten — is a
property of code and therefore testable offline.

Every test here runs against a deterministic synthetic panel, so the suite
never touches the network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prism.config import Config, load_config
from prism.data.clean import availability_mask, clean_panel
from prism.data.download import SnapshotExistsError, create_snapshot, snapshot_dir
from prism.data.loaders import load_snapshot
from prism.data.quality import (
    EXTREME_MOVE_HARD,
    EXTREME_MOVE_SOFT,
    KNOWN_ARTEFACTS,
    run_quality_checks,
    write_quality_report,
)
from prism.features.build import split_raw_frame


# --------------------------------------------------------------------------- #
# snapshot: write-once
# --------------------------------------------------------------------------- #
def _write_snapshot(cfg: Config, raw: pd.DataFrame, date: str = "2026-09-30"):
    panels = split_raw_frame(raw)
    return create_snapshot(
        cfg, date=date, tickers=cfg.data.tickers("B"), panels=panels
    )


def test_snapshot_writes_a_manifest_with_per_file_hashes(tmp_repo: Config, raw_b):
    """§4.1: per-file SHA-256, row counts, dates, library versions, pip freeze."""
    out = _write_snapshot(tmp_repo, raw_b)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))

    assert manifest["auto_adjust"] is True, "total-return prices are required"
    assert manifest["snapshot_hash"]
    assert "ohlcv.parquet" in manifest["files"]
    for meta in manifest["files"].values():
        assert len(meta["sha256"]) == 64
        assert meta["rows"] > 0
        assert meta["first_date"] and meta["last_date"]
    assert manifest["libraries"]["pandas"]
    assert manifest["pip_freeze"], "pip freeze is part of the reproducibility record"
    assert manifest["downloaded_utc"]


def test_snapshot_refuses_to_overwrite(tmp_repo: Config, raw_b):
    """§4.1 / defect A1: re-downloading restates adjusted prices.

    The reference pipeline overwrote its "frozen" data on every run, so two
    runs a month apart silently compared different datasets.
    """
    _write_snapshot(tmp_repo, raw_b)
    with pytest.raises(SnapshotExistsError, match="write-once"):
        _write_snapshot(tmp_repo, raw_b)


def test_force_alone_does_not_permit_overwrite(tmp_repo: Config, raw_b):
    """``--force`` requires ``snapshot.allow_overwrite`` in config as well.

    One careless CLI flag must not be able to destroy a snapshot that every
    committed result traces to.
    """
    _write_snapshot(tmp_repo, raw_b)
    panels = split_raw_frame(raw_b)
    with pytest.raises(SnapshotExistsError, match="allow_overwrite"):
        create_snapshot(
            tmp_repo,
            date="2026-09-30",
            force=True,
            tickers=tmp_repo.data.tickers("B"),
            panels=panels,
        )


def test_overwrite_succeeds_only_with_both_gates(tmp_path, cfg: Config, raw_b):
    """Both gates together do permit it, so the escape hatch exists."""
    import shutil

    shutil.copytree(cfg.root / "configs", tmp_path / "configs")
    permissive = load_config(
        root=tmp_path, overrides={"data": {"snapshot": {"allow_overwrite": True}}}
    )
    panels = split_raw_frame(raw_b)
    create_snapshot(
        permissive, date="2026-09-30", tickers=permissive.data.tickers("B"), panels=panels
    )
    create_snapshot(
        permissive,
        date="2026-09-30",
        force=True,
        tickers=permissive.data.tickers("B"),
        panels=panels,
    )


def test_loading_a_tampered_snapshot_raises(tmp_repo: Config, raw_b):
    """A snapshot whose files no longer match its manifest is an error.

    Not a warning: every result traced to that hash has become untraceable.
    """
    out = _write_snapshot(tmp_repo, raw_b)
    loaded = load_snapshot(tmp_repo, date="2026-09-30")
    assert loaded.snapshot_hash

    from prism.utils.hashing import sha256_file

    before = sha256_file(out / "ohlcv.parquet")
    frame = pd.read_parquet(out / "ohlcv.parquet")
    # Tamper with a cell that actually holds a value. The first cell of the
    # frame is pre-inception NaN, and NaN * 2 is NaN — which would leave the
    # file byte-identical and make this test silently vacuous.
    target = ("Close", "SPY")
    assert pd.notna(frame.iloc[3000][target]), "picked a NaN cell to tamper with"
    frame.iloc[3000, frame.columns.get_loc(target)] *= 2.0
    frame.to_parquet(out / "ohlcv.parquet")
    assert sha256_file(out / "ohlcv.parquet") != before, "the tamper did not change the file"

    with pytest.raises(ValueError, match="does not match manifest"):
        load_snapshot(tmp_repo, date="2026-09-30")


def test_snapshot_round_trips_prices_and_macro(tmp_repo: Config, raw_b):
    """What comes back must be what went in."""
    _write_snapshot(tmp_repo, raw_b)
    loaded = load_snapshot(tmp_repo, date="2026-09-30")
    original = split_raw_frame(raw_b)["Close"]

    assert loaded.macro is not None
    combined = loaded.prices(tmp_repo.data.tickers("B"))
    pd.testing.assert_frame_equal(
        original.reindex(columns=combined.columns).astype("float64"),
        combined.astype("float64"),
        check_freq=False,
    )


def test_prices_accessor_refuses_an_unknown_ticker(tmp_repo: Config, raw_b):
    """The universe restriction must fail loudly, not return a short frame."""
    _write_snapshot(tmp_repo, raw_b)
    loaded = load_snapshot(tmp_repo, date="2026-09-30")
    with pytest.raises(KeyError):
        loaded.prices(["SPY", "NOT_A_TICKER"])


def test_missing_snapshot_directs_the_user_to_the_script(tmp_repo: Config):
    with pytest.raises(FileNotFoundError, match="00_snapshot"):
        load_snapshot(tmp_repo, date="2026-09-30")


def test_snapshot_dir_requires_a_date(cfg: Config):
    """``snapshot.date`` is null until a snapshot has been taken."""
    assert cfg.data.snapshot.date is None, (
        "snapshot.date should stay null until scripts/00_snapshot.py has run"
    )
    with pytest.raises(ValueError, match="no snapshot date"):
        snapshot_dir(cfg)


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def test_availability_distinguishes_pre_inception_from_a_missing_session(cfg: Config, raw_b):
    """§4.2: assets have different inception dates; never fill across one."""
    close = split_raw_frame(raw_b)["Close"]
    inception = {k: pd.Timestamp(v) for k, v in cfg.data.inception.items()}
    mask = availability_mask(close, inception)

    hyg_start = inception["HYG"]
    assert not mask.loc[mask.index < hyg_start, "HYG"].any(), (
        "HYG is marked available before its 2007-04-04 inception"
    )
    assert mask.loc[mask.index >= hyg_start, "HYG"].all()
    # SPY has the longest history in the universe and must be available first.
    assert mask["SPY"].sum() > mask["HYG"].sum()


def test_forward_fill_is_capped_and_never_backward(cfg: Config, raw_b):
    """§4.2 / defect A2: forward-fill only, capped at 3 sessions.

    A gap wider than the cap is left as NaN and flagged for review, not
    bridged by widening the cap and certainly not by a ``bfill``.
    """
    panels = split_raw_frame(raw_b)
    holed = {k: v.copy() for k, v in panels.items()}
    cap = cfg.data.calendar.max_ffill_days

    # A short gap (within the cap) and a long one (beyond it).
    short_rows = slice(3000, 3000 + cap)
    long_rows = slice(3500, 3500 + cap + 10)
    holed["Close"].iloc[short_rows, holed["Close"].columns.get_loc("SPY")] = np.nan
    holed["Close"].iloc[long_rows, holed["Close"].columns.get_loc("SPY")] = np.nan

    result = clean_panel(holed, cfg, tickers=list(holed["Close"].columns))

    assert result.filled_sessions["SPY"] >= cap, "the short gap was not forward-filled"
    assert result.long_gaps, "the long gap was not flagged"
    flagged = [g for g in result.long_gaps if g["ticker"] == "SPY"]
    assert flagged, "SPY's long gap is missing from the report"
    assert all(g["sessions"] > 0 for g in flagged)

    # The value filled into the short gap must come from BEFORE it.
    col = result.close["SPY"]
    gap_start = holed["Close"].index[short_rows.start]
    before = holed["Close"]["SPY"].loc[: gap_start - pd.Timedelta(days=1)].dropna().iloc[-1]
    np.testing.assert_allclose(col.loc[gap_start], before, rtol=1e-12)


def test_duplicate_index_is_a_hard_error(cfg: Config, raw_b):
    """§4.2 / defect B5: 1,557 rows for 1,509 trading days.

    Refusing to guess which duplicate is correct is the point; de-duplicating
    silently is how the reference rolling loop hid its bug.
    """
    panels = split_raw_frame(raw_b)
    doubled = {
        k: pd.concat([v, v.iloc[[100]]]).sort_index() for k, v in panels.items()
    }
    with pytest.raises(ValueError, match="duplicate dates"):
        clean_panel(doubled, cfg, tickers=list(panels["Close"].columns))


def test_unsorted_index_is_a_hard_error(cfg: Config, raw_b):
    panels = split_raw_frame(raw_b)
    shuffled = {k: v.iloc[::-1] for k, v in panels.items()}
    with pytest.raises(ValueError, match="not sorted"):
        clean_panel(shuffled, cfg, tickers=list(panels["Close"].columns))


def test_cleaning_reindexes_onto_the_exchange_calendar(cfg: Config, raw_b):
    """§4.2: the NYSE calendar, not ``bdate_range``.

    A business-day range includes exchange holidays, which would then look
    like missing sessions and be forward-filled — inventing prices for days
    the market was shut.
    """
    from prism.utils.calendar import trading_days

    panels = split_raw_frame(raw_b)
    result = clean_panel(panels, cfg, tickers=list(panels["Close"].columns))
    expected = trading_days(result.index[0], result.index[-1], cfg.data.calendar.exchange)
    assert result.index.equals(expected)

    business = pd.bdate_range(result.index[0], result.index[-1])
    assert len(business) > len(expected), (
        "business days should exceed exchange sessions; if not, the calendar is "
        "not being applied"
    )
    # Independence Day 2019 fell on a Thursday: a business day, not a session.
    assert pd.Timestamp("2019-07-04") in business
    assert pd.Timestamp("2019-07-04") not in result.index


# --------------------------------------------------------------------------- #
# quality assurance
# --------------------------------------------------------------------------- #
def test_clean_synthetic_data_passes_the_hard_checks(cfg: Config, raw_b, features_b):
    """A clean panel must produce no hard failures, or QA is unusable."""
    panels = split_raw_frame(raw_b)
    cleaned = clean_panel(panels, cfg, tickers=list(panels["Close"].columns))
    assets = [t for t in cfg.data.allocatable["B"] if t in cleaned.close.columns]

    report = run_quality_checks(
        cleaned.close[assets],
        cfg,
        volume=None if cleaned.panels.get("Volume") is None else cleaned.panels["Volume"][assets],
        available=cleaned.available[assets],
        features=features_b.frame,
        warmup_end=features_b.warm_start,
        universe="B",
    )
    assert report.ok, f"hard failures on clean data: {[f.detail for f in report.hard_failures]}"
    report.raise_if_failed()


def test_qa_thresholds_match_the_spec(cfg: Config):
    """§4.3: 15% soft, 30% hard. Defect A8 used 50%, which catches nothing."""
    assert EXTREME_MOVE_SOFT == 0.15
    assert EXTREME_MOVE_HARD == 0.30
    assert EXTREME_MOVE_SOFT < EXTREME_MOVE_HARD < 0.50


# The `feature_nan_or_inf_after_warmup` case injects an inf on purpose, and
# numpy warns when it reaches a variance computation. Scoped to this test only
# — the project-wide policy is that warnings stay visible (defect A9).
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize(
    "check",
    [
        "extreme_move_hard",
        "nonpositive_prices",
        "constant_feature",
        "feature_nan_or_inf_after_warmup",
        "missing_sessions_vs_calendar",
    ],
)
def test_each_hard_check_can_actually_fire(check: str, cfg: Config, raw_b, features_b):
    """A QA suite that never fails is as worthless as a causality test that
    never fails. Each hard check is provoked individually."""
    panels = split_raw_frame(raw_b)
    cleaned = clean_panel(panels, cfg, tickers=list(panels["Close"].columns))
    assets = [t for t in cfg.data.allocatable["B"] if t in cleaned.close.columns]
    close = cleaned.close[assets].copy()
    available = cleaned.available[assets].copy()
    features = features_b.frame

    if check == "extreme_move_hard":
        close.iloc[3000, 0] = close.iloc[2999, 0] * 2.0
    elif check == "nonpositive_prices":
        close.iloc[3000, 0] = -1.0
    elif check == "constant_feature":
        features = features.copy()
        features.iloc[:, 0] = 1.0
    elif check == "feature_nan_or_inf_after_warmup":
        features = features.copy()
        features.iloc[10, 0] = np.inf
    elif check == "missing_sessions_vs_calendar":
        close = close.drop(index=close.index[3000])
        available = available.drop(index=available.index[3000])

    report = run_quality_checks(
        close,
        cfg,
        available=available,
        features=features,
        warmup_end=features_b.warm_start,
        universe="B",
    )
    failed = {f.check for f in report.hard_failures}
    assert check in failed, (
        f"the {check!r} hard check did not fire when provoked; QA is decorative. "
        f"Fired instead: {sorted(failed)}"
    )
    assert not report.ok
    with pytest.raises(ValueError, match="hard QA failure"):
        report.raise_if_failed()


def test_quality_report_is_written_as_markdown(tmp_path, cfg: Config, raw_b, features_b):
    """§4.3: the report lands in ``reports/tables/data_quality.md``."""
    panels = split_raw_frame(raw_b)
    cleaned = clean_panel(panels, cfg, tickers=list(panels["Close"].columns))
    assets = [t for t in cfg.data.allocatable["B"] if t in cleaned.close.columns]
    report = run_quality_checks(
        cleaned.close[assets],
        cfg,
        available=cleaned.available[assets],
        features=features_b.frame,
        warmup_end=features_b.warm_start,
        universe="B",
    )
    out = write_quality_report(report, tmp_path / "data_quality.md")
    text = out.read_text(encoding="utf-8")

    assert "# Data quality report" in text
    assert "| Check | Severity | Threshold | Status | Detail |" in text
    for finding in report.findings:
        assert f"`{finding.check}`" in text
    # §4.3: known artefacts are documented, not silently fixed.
    assert "Known artefacts" in text
    assert "GICS" in text
    assert len(KNOWN_ARTEFACTS) >= 3


def test_known_artefacts_are_documented_not_corrected():
    """§4.3: the 2018 GICS reclassification and constituent drift are recorded."""
    joined = " ".join(KNOWN_ARTEFACTS)
    assert "GICS" in joined
    assert "constituent drift" in joined.lower()
    assert "XLRE" in joined and "XLC" in joined
    for artefact in KNOWN_ARTEFACTS:
        assert "not corrected" in artefact.lower() or "excluded" in artefact.lower()
