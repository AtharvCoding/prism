"""Quality assurance with hard failures and soft warnings.

Spec §4.3. The reference QA (defect A8) used a 50% extreme-return threshold
and checked neither the trading calendar nor stale prices, so it was too
permissive to catch anything real. This module implements the §4.3 table
exactly: hard checks abort the pipeline, soft checks are logged and land in
``reports/tables/data_quality.md`` for inspection.

Known artefacts are *documented, not silently fixed* (spec §4.3): the 2018
GICS reclassification moved telecom into communication services, changing
XLK and XLY constituents, and ETF constituents drift continuously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from prism.config import Config
from prism.utils.calendar import trading_days
from prism.utils.logging import get_logger

__all__ = ["Severity", "Finding", "QAReport", "run_quality_checks", "write_quality_report"]

_log = get_logger(__name__)

Severity = Literal["hard", "soft"]

#: Spec §4.3 thresholds. Named so the report can cite them.
EXTREME_MOVE_SOFT = 0.15
EXTREME_MOVE_HARD = 0.30
STALE_RUN_SOFT = 3
SPY_COVERAGE_DIVERGENCE_SOFT = 1

#: Artefacts that are real, known and deliberately not corrected.
KNOWN_ARTEFACTS: tuple[str, ...] = (
    "2018 GICS reclassification: telecom moved to communication services, "
    "changing XLK and XLY constituents. Not corrected — a sector ETF's "
    "definition genuinely changed, and splicing it would fabricate a series "
    "no investor could have held.",
    "ETF constituent drift: index rebalances continuously change holdings. "
    "Not corrected; the tradable series is the one the ETF actually delivered.",
    "XLRE (2015-10) and XLC (2018-06) are excluded for insufficient history, "
    "so post-2018 sector coverage is incomplete by construction.",
)


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    passed: bool
    detail: str
    threshold: str = ""
    n_affected: int = 0

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.severity == "hard" else "WARN"


@dataclass
class QAReport:
    findings: list[Finding] = field(default_factory=list)
    context: dict[str, object] = field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        if not finding.passed:
            log = _log.error if finding.severity == "hard" else _log.warning
            log("QA %s [%s]: %s", finding.status, finding.check, finding.detail)

    @property
    def hard_failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "hard" and not f.passed]

    @property
    def soft_warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "soft" and not f.passed]

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        lines = [f"  - {f.check}: {f.detail}" for f in self.hard_failures]
        raise ValueError(
            f"{len(self.hard_failures)} hard QA failure(s):\n" + "\n".join(lines)
        )


def run_quality_checks(
    close: pd.DataFrame,
    cfg: Config,
    *,
    volume: pd.DataFrame | None = None,
    available: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    warmup_end: pd.Timestamp | None = None,
    universe: str = "B",
) -> QAReport:
    """Run the §4.3 checks over a cleaned price panel and optional features."""
    report = QAReport(
        context={
            "universe": universe,
            "tickers": [str(c) for c in close.columns],
            "sessions": int(len(close)),
            "first_date": str(close.index[0].date()) if len(close) else None,
            "last_date": str(close.index[-1].date()) if len(close) else None,
        }
    )
    exchange = cfg.data.calendar.exchange
    mask = available if available is not None else close.notna()

    # -- hard: index integrity --------------------------------------------- #
    idx = pd.DatetimeIndex(close.index)
    report.add(
        Finding(
            "duplicate_index",
            "hard",
            not idx.has_duplicates,
            f"{int(idx.duplicated().sum())} duplicate index entries",
            threshold="any",
            n_affected=int(idx.duplicated().sum()),
        )
    )
    report.add(
        Finding(
            "monotonic_index",
            "hard",
            bool(idx.is_monotonic_increasing),
            "index is sorted ascending" if idx.is_monotonic_increasing else "index is unsorted",
            threshold="any",
        )
    )

    # -- hard: every NYSE session present --------------------------------- #
    if len(idx):
        expected = trading_days(idx[0], idx[-1], exchange)
        missing = expected.difference(idx)
        report.add(
            Finding(
                "missing_sessions_vs_calendar",
                "hard",
                len(missing) == 0,
                (
                    "all expected sessions present"
                    if len(missing) == 0
                    else f"{len(missing)} {exchange} sessions absent, first {missing[0].date()}"
                ),
                threshold="any",
                n_affected=len(missing),
            )
        )

    # -- hard: prices --------------------------------------------------- #
    nonpositive = int(((close <= 0) & mask).sum().sum())
    report.add(
        Finding(
            "nonpositive_prices",
            "hard",
            nonpositive == 0,
            f"{nonpositive} zero or negative prices inside the availability window",
            threshold="any",
            n_affected=nonpositive,
        )
    )
    # NaN inside availability means the ffill cap refused to bridge a gap.
    na_in_window = int((close.isna() & mask).sum().sum())
    report.add(
        Finding(
            "price_gaps_in_availability_window",
            "hard",
            na_in_window == 0,
            f"{na_in_window} sessions still missing a price after capped forward-fill",
            threshold="any",
            n_affected=na_in_window,
        )
    )

    # -- returns ---------------------------------------------------------- #
    returns = np.log(close.where(mask)).diff()
    abs_ret = returns.abs()
    hard_extreme = (abs_ret > EXTREME_MOVE_HARD).sum()
    soft_extreme = ((abs_ret > EXTREME_MOVE_SOFT) & (abs_ret <= EXTREME_MOVE_HARD)).sum()
    report.add(
        Finding(
            "extreme_move_hard",
            "hard",
            int(hard_extreme.sum()) == 0,
            _describe_counts(hard_extreme, "|r| > 30%"),
            threshold=f"|r| > {EXTREME_MOVE_HARD:.0%}",
            n_affected=int(hard_extreme.sum()),
        )
    )
    report.add(
        Finding(
            "extreme_move_soft",
            "soft",
            int(soft_extreme.sum()) == 0,
            _describe_counts(soft_extreme, "15% < |r| <= 30%"),
            threshold=f"|r| > {EXTREME_MOVE_SOFT:.0%}",
            n_affected=int(soft_extreme.sum()),
        )
    )

    # -- soft: stale prices ---------------------------------------------- #
    stale_counts = {}
    for ticker in close.columns:
        series = close[ticker].where(mask[ticker]).dropna()
        if len(series) < STALE_RUN_SOFT:
            continue
        unchanged = series.diff().eq(0)
        runs = _max_run(unchanged.to_numpy(dtype=bool))
        if runs >= STALE_RUN_SOFT:
            stale_counts[ticker] = runs
    report.add(
        Finding(
            "stale_price_run",
            "soft",
            not stale_counts,
            (
                "no stale runs"
                if not stale_counts
                else "longest identical-close runs: "
                + ", ".join(f"{k}={v}" for k, v in sorted(stale_counts.items()))
            ),
            threshold=f">= {STALE_RUN_SOFT} identical closes",
            n_affected=len(stale_counts),
        )
    )

    # -- soft: zero volume ------------------------------------------------ #
    if volume is not None:
        vol = volume.reindex_like(close).where(mask)
        zero_vol = (vol == 0).sum()
        report.add(
            Finding(
                "zero_volume",
                "soft",
                int(zero_vol.sum()) == 0,
                _describe_counts(zero_vol, "zero-volume sessions"),
                threshold="any",
                n_affected=int(zero_vol.sum()),
            )
        )

    # -- soft: coverage vs SPY ------------------------------------------- #
    benchmark = cfg.data.universes[universe].benchmark or cfg.data.universes["A"].benchmark
    if benchmark in close.columns:
        spy_sessions = int(mask[benchmark].sum())
        divergent = {}
        for ticker in close.columns:
            if ticker == benchmark:
                continue
            # Compare only where both are past inception.
            both = mask[benchmark] & mask[ticker]
            gap = int(mask[benchmark].sum() - both.sum())
            since = mask[ticker].idxmax() if mask[ticker].any() else None
            if since is not None:
                gap = int((mask[benchmark] & (mask.index >= since) & ~mask[ticker]).sum())
            if gap > SPY_COVERAGE_DIVERGENCE_SOFT:
                divergent[ticker] = gap
        report.add(
            Finding(
                "index_coverage_vs_benchmark",
                "soft",
                not divergent,
                (
                    f"all tickers track {benchmark}'s {spy_sessions} sessions"
                    if not divergent
                    else "post-inception session shortfall vs "
                    f"{benchmark}: " + ", ".join(f"{k}={v}" for k, v in sorted(divergent.items()))
                ),
                threshold=f"> {SPY_COVERAGE_DIVERGENCE_SOFT} day divergence",
                n_affected=len(divergent),
            )
        )

    # -- features --------------------------------------------------------- #
    if features is not None:
        warm = features if warmup_end is None else features.loc[features.index > warmup_end]
        stds = warm.std(numeric_only=True)
        constant = sorted(stds.index[stds.fillna(0.0) == 0.0].astype(str))
        report.add(
            Finding(
                "constant_feature",
                "hard",
                not constant,
                (
                    "no constant features"
                    if not constant
                    else f"{len(constant)} constant after warm-up: {constant[:8]}"
                ),
                threshold="std == 0",
                n_affected=len(constant),
            )
        )
        bad = warm.isna() | np.isinf(warm.to_numpy(dtype="float64", na_value=np.nan))
        n_bad = int(np.asarray(bad).sum())
        cols_bad = sorted(warm.columns[np.asarray(bad).any(axis=0)].astype(str))
        report.add(
            Finding(
                "feature_nan_or_inf_after_warmup",
                "hard",
                n_bad == 0,
                (
                    "no NaN or inf after warm-up"
                    if n_bad == 0
                    else f"{n_bad} non-finite values in {len(cols_bad)} columns: {cols_bad[:8]}"
                ),
                threshold="any",
                n_affected=n_bad,
            )
        )
        report.add(_ks_drift_finding(features, cfg))

    return report


def _describe_counts(counts: pd.Series, label: str) -> str:
    hits = counts[counts > 0]
    if hits.empty:
        return f"no {label}"
    return f"{int(hits.sum())} {label}: " + ", ".join(
        f"{k}={int(v)}" for k, v in hits.sort_values(ascending=False).head(8).items()
    )


def _max_run(flags: np.ndarray) -> int:
    if not flags.any():
        return 0
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    # A run of n identical diffs means n+1 identical closes.
    return best + 1 if best else 0


def _ks_drift_finding(features: pd.DataFrame, cfg: Config) -> Finding:
    """Soft check: KS test on the return distribution, train vs test."""
    from scipy import stats

    train_start, train_end = cfg.data.split("train")
    test_start, test_end = cfg.data.split("test")
    ret_cols = [c for c in features.columns if str(c).endswith("return_1d")]
    if not ret_cols:
        return Finding(
            "return_distribution_drift",
            "soft",
            True,
            "no return_1d columns present to test",
            threshold="KS train vs test",
        )
    train = features.loc[train_start:train_end, ret_cols]
    test = features.loc[test_start:test_end, ret_cols]
    if train.empty or test.empty:
        return Finding(
            "return_distribution_drift",
            "soft",
            True,
            "train or test window absent from this frame; drift not testable",
            threshold="KS train vs test",
        )
    drifted = {}
    for col in ret_cols:
        a = train[col].dropna().to_numpy()
        b = test[col].dropna().to_numpy()
        if len(a) < 30 or len(b) < 30:
            continue
        res = stats.ks_2samp(a, b)
        if res.pvalue < 0.01:
            drifted[str(col)] = float(res.pvalue)
    return Finding(
        "return_distribution_drift",
        "soft",
        not drifted,
        (
            "no significant train/test return drift at p<0.01"
            if not drifted
            else f"{len(drifted)} series drifted (p<0.01): "
            + ", ".join(f"{k}={v:.1e}" for k, v in sorted(drifted.items())[:6])
            + ". Expected for 2020 and 2022; documented, not corrected."
        ),
        threshold="KS test, p < 0.01",
        n_affected=len(drifted),
    )


def write_quality_report(report: QAReport, path: str | Path) -> Path:
    """Render the report to markdown at ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Data quality report",
        "",
        "Generated by `prism.data.quality`. Thresholds are those of spec §4.3.",
        "",
        "## Context",
        "",
    ]
    for key, value in report.context.items():
        if isinstance(value, list) and len(value) > 12:
            value = f"{len(value)} items: {value[:12]} ..."
        lines.append(f"- **{key}**: {value}")

    lines += [
        "",
        "## Checks",
        "",
        "| Check | Severity | Threshold | Status | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for f in report.findings:
        detail = f.detail.replace("|", "\\|")
        lines.append(
            f"| `{f.check}` | {f.severity} | {f.threshold or '—'} | **{f.status}** | {detail} |"
        )

    lines += [
        "",
        f"**{len(report.hard_failures)} hard failure(s), "
        f"{len(report.soft_warnings)} soft warning(s).**",
        "",
        "## Known artefacts — documented, not corrected",
        "",
    ]
    for artefact in KNOWN_ARTEFACTS:
        lines.append(f"- {artefact}")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    _log.info("quality report written to %s", out)
    return out
