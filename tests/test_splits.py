"""Splits, embargo, walk-forward folds, and the holdout lock.

Spec §7.4.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pytest

from prism.config import FORWARD_TARGET_HORIZONS, Config, load_config
from prism.data.loaders import (
    HOLDOUT_ENV_VAR,
    HoldoutLockError,
    assert_not_holdout,
    load_holdout,
    non_holdout_span,
)
from prism.splits import SPLIT_ORDER, WalkForwardFold, build_split_plan, expanding_folds


# --------------------------------------------------------------------------- #
# ordering, disjointness, embargo
# --------------------------------------------------------------------------- #
def test_splits_are_ordered_disjoint_and_embargoed(plan, cfg: Config):
    """§7.4: ordered, disjoint, and separated by at least the embargo."""
    plan.validate()  # raises on any violation

    names = [n for n in SPLIT_ORDER if n in plan.splits]
    assert names == list(SPLIT_ORDER), "a split is missing from the plan"

    for earlier, later in zip(names, names[1:]):
        gap = plan.gap_sessions(earlier, later)
        assert gap >= cfg.data.splits.embargo_days, (
            f"{earlier} -> {later} gap is {gap} sessions, "
            f"{cfg.data.splits.embargo_days} required"
        )
        assert plan[earlier].effective_end < plan[later].effective_start
        assert plan[earlier].sessions.intersection(plan[later].sessions).empty


def test_embargo_is_applied_by_purging_the_earlier_split(plan, cfg: Config):
    """The declared splits are calendar-contiguous, so the embargo must bite.

    ``configs/data.yaml`` declares train ending 2017-12-31 and val starting
    2018-01-01 — zero sessions apart — while also declaring a 25-session
    embargo. ``splits.py`` resolves this by purging the **end** of the
    earlier split, which preserves the reported val/test/holdout windows and
    pays the cost out of training data.

    This test pins that resolution down. If someone later reads the embargo as
    "shift the later split's start forward" instead, the reported evaluation
    windows would silently stop being the 2018 calendar year and 2019-2023,
    and every table in the report would describe a different period than it
    claims to.
    """
    embargo = cfg.data.splits.embargo_days
    for name in ("train", "val", "test"):
        split = plan[name]
        assert split.effective_start == split.declared_start, (
            f"{name}: the embargo must never move a split's START — the reported "
            "evaluation windows are defined by the declared starts"
        )
        assert split.purged_sessions == embargo, (
            f"{name}: expected exactly {embargo} sessions purged, got "
            f"{split.purged_sessions}"
        )
        assert split.effective_end < split.declared_end

    # The holdout is last, so nothing follows it to embargo against.
    assert plan["holdout"].purged_sessions == 0
    assert plan["holdout"].effective_end == plan["holdout"].declared_end


def test_embargo_covers_the_longest_forward_horizon(cfg: Config):
    """§6.1: size the embargo to the longest forward-looking horizon plus margin."""
    longest = max((*FORWARD_TARGET_HORIZONS, cfg.data.decision.rebalance_horizon_days))
    assert cfg.data.splits.embargo_days >= longest

    from prism.features.targets import target_horizons

    assert target_horizons() <= set(FORWARD_TARGET_HORIZONS), (
        "targets.py produces a horizon the embargo was not sized against"
    )


def test_config_rejects_an_understated_embargo(cfg: Config, tmp_path):
    """The embargo check must be able to fail at config load, not at run time."""
    with pytest.raises(ValueError, match="forward target horizon"):
        load_config(
            root=cfg.root,
            overrides={"data": {"splits": {"embargo_days": 1}}},
        )


def test_config_rejects_overlapping_splits(cfg: Config):
    """Overlapping splits must be rejected before any data is touched."""
    with pytest.raises(ValueError, match="overlap or touch"):
        load_config(
            root=cfg.root,
            overrides={"data": {"splits": {"val": ["2017-06-01", "2018-12-31"]}}},
        )


def test_every_split_is_non_empty(plan):
    """A split purged into nothing is a silent disaster; assert it cannot happen."""
    for split in plan:
        assert len(split) > 0, f"split {split.name} has no sessions after purging"
    assert len(plan["train"]) > 2000, "training window is implausibly short"
    assert len(plan["val"]) > 150, "validation window is too short for model selection"
    assert len(plan["test"]) > 1000


# --------------------------------------------------------------------------- #
# walk-forward folds
# --------------------------------------------------------------------------- #
def test_every_fold_satisfies_fit_end_before_apply_start(plan, cfg: Config):
    """§7.4: "Walk-forward folds satisfy ``fit_end < apply_start`` for every fold"."""
    for cadence in ("monthly", "quarterly", "annual"):
        folds = expanding_folds(
            plan["train"].effective_start,
            plan["val"].declared_start,
            plan["test"].effective_end,
            cadence,
            embargo_days=cfg.data.splits.embargo_days,
        )
        assert folds, f"no folds for cadence {cadence}"
        for fold in folds:
            assert fold.fit_end < fold.apply_start
            assert fold.fit_sessions.intersection(fold.apply_sessions).empty


def test_folds_are_expanding_not_rolling(plan, cfg: Config):
    """§6.2 / defect B7: a fixed 5-year window starting in 2013 has no crisis.

    Every fold must share the same ``fit_start`` and extend ``fit_end``
    forward, so that later folds retain the GFC in their fit window.
    """
    folds = expanding_folds(
        plan["train"].effective_start,
        plan["val"].declared_start,
        plan["test"].effective_end,
        "annual",
        embargo_days=cfg.data.splits.embargo_days,
    )
    assert len({f.fit_start for f in folds}) == 1, "fit_start moved: this is a rolling window"
    fit_ends = [f.fit_end for f in folds]
    assert fit_ends == sorted(fit_ends), "fit_end must advance monotonically"
    assert len(folds[-1].fit_sessions) > len(folds[0].fit_sessions), "window is not expanding"


def test_folds_cover_the_apply_period_without_gaps_or_overlap(plan, cfg: Config):
    """Union of apply windows must tile the evaluation period exactly once."""
    folds = expanding_folds(
        plan["train"].effective_start,
        plan["val"].declared_start,
        plan["test"].effective_end,
        "annual",
        embargo_days=cfg.data.splits.embargo_days,
    )
    covered: list[pd.Timestamp] = []
    for fold in folds:
        covered.extend(fold.apply_sessions)
    index = pd.DatetimeIndex(covered)
    assert index.is_unique, "walk-forward folds double-count sessions (defect B5)"
    assert index.is_monotonic_increasing


def test_fold_construction_rejects_an_invalid_fold():
    """An invalid fold must be impossible to construct, not merely detectable."""
    with pytest.raises(ValueError, match="is not before"):
        WalkForwardFold(
            index=0,
            fit_start=pd.Timestamp("2010-01-04"),
            fit_end=pd.Timestamp("2012-01-04"),
            apply_start=pd.Timestamp("2011-01-04"),  # inside the fit window
            apply_end=pd.Timestamp("2013-01-04"),
        )
    with pytest.raises(ValueError, match="sessions between fit_end"):
        WalkForwardFold(
            index=0,
            fit_start=pd.Timestamp("2010-01-04"),
            fit_end=pd.Timestamp("2012-01-04"),
            apply_start=pd.Timestamp("2012-01-05"),  # one session later
            apply_end=pd.Timestamp("2013-01-04"),
            embargo_days=25,
        )


def test_refit_date_is_never_inside_its_own_fit_window(plan, cfg: Config):
    """§8.6 / defect B5: "The refit date must not be inside its own fit window"."""
    folds = expanding_folds(
        plan["train"].effective_start,
        plan["val"].declared_start,
        plan["test"].effective_end,
        "monthly",
        embargo_days=cfg.data.splits.embargo_days,
    )
    for fold in folds:
        assert fold.apply_start not in fold.fit_sessions


# --------------------------------------------------------------------------- #
# holdout lock
# --------------------------------------------------------------------------- #
def test_load_holdout_raises_without_the_explicit_flag(cfg: Config, monkeypatch):
    """§7.4: "``load_holdout()`` raises without the explicit flag"."""
    monkeypatch.delenv(HOLDOUT_ENV_VAR, raising=False)
    with pytest.raises(HoldoutLockError, match="final=True"):
        load_holdout(cfg)


def test_load_holdout_raises_without_the_environment_variable(cfg: Config, monkeypatch):
    """§6.3: both gates are required, not either."""
    monkeypatch.delenv(HOLDOUT_ENV_VAR, raising=False)
    with pytest.raises(HoldoutLockError, match=HOLDOUT_ENV_VAR):
        load_holdout(cfg, final=True, reason="test")


def test_load_holdout_raises_without_a_reason(cfg: Config, monkeypatch):
    """The access log must say *why*, not merely that it happened."""
    monkeypatch.setenv(HOLDOUT_ENV_VAR, "1")
    with pytest.raises(HoldoutLockError, match="non-empty reason"):
        load_holdout(cfg, final=True, reason="   ")


def test_wrong_environment_variable_value_does_not_unlock(cfg: Config, monkeypatch):
    """Only the exact value ``"1"`` unlocks; ``true``/``yes`` must not."""
    for value in ("0", "true", "TRUE", "yes", ""):
        monkeypatch.setenv(HOLDOUT_ENV_VAR, value)
        with pytest.raises(HoldoutLockError):
            load_holdout(cfg, final=True, reason="test")


def test_holdout_unlocks_only_with_both_gates_and_logs_the_access(
    tmp_repo: Config, monkeypatch
):
    """With both gates and a reason it opens — and writes an audit line.

    Run against a temporary repo root so the real access log is never
    touched by the test suite.
    """
    import json

    monkeypatch.setenv(HOLDOUT_ENV_VAR, "1")
    start, end = tmp_repo.data.split("holdout")
    index = pd.date_range(start - pd.Timedelta(days=60), end, freq="B")
    frame = pd.DataFrame({"x": range(len(index))}, index=index)

    out = load_holdout(tmp_repo, frame, final=True, reason="unit test of the lock")
    assert not out.empty
    assert out.index.min() >= start and out.index.max() <= end

    log_path = tmp_repo.root / "reports" / "logs" / "holdout_access.jsonl"
    assert log_path.exists(), "a holdout read must leave a permanent record"
    entry = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert entry["reason"] == "unit test of the lock"
    assert entry["range"] == [str(start.date()), str(end.date())]


def test_phase_a_span_excludes_the_holdout(cfg: Config):
    """Phase A's readable range must end before the holdout begins."""
    span_start, span_end = non_holdout_span(cfg)
    holdout_start, _ = cfg.data.split("holdout")
    assert span_end < holdout_start
    assert span_start == cfg.data.start("A")


def test_assert_not_holdout_catches_a_leaked_index(cfg: Config):
    """The cheap guard that makes the expensive one unnecessary."""
    holdout_start, holdout_end = cfg.data.split("holdout")
    clean = pd.date_range("2019-01-02", "2019-12-31", freq="B")
    assert_not_holdout(cfg, clean, context="clean index")

    leaked = clean.append(pd.DatetimeIndex([holdout_start + pd.Timedelta(days=5)]))
    with pytest.raises(HoldoutLockError, match="inside the holdout window"):
        assert_not_holdout(cfg, leaked, context="leaked index")


def test_synthetic_fixtures_never_span_the_holdout(cfg: Config, raw_b):
    """The test data itself must stop before the holdout.

    If a fixture spanned it, a test that accidentally read the holdout would
    get plausible-looking numbers instead of an empty slice — and would pass.
    """
    assert_not_holdout(cfg, pd.DatetimeIndex(raw_b.index), context="raw_b fixture")


def test_holdout_data_directory_is_empty(cfg: Config):
    """Nothing may be written into ``data/holdout/`` during Phase A."""
    holdout_dir = cfg.path("holdout")
    if not holdout_dir.exists():
        return
    contents = [p for p in holdout_dir.iterdir() if p.name not in {".gitkeep", ".DS_Store"}]
    assert not contents, f"Phase A wrote into the holdout directory: {contents}"


# --------------------------------------------------------------------------- #
# no hard-coded dates outside config
# --------------------------------------------------------------------------- #
DATE_LITERAL = re.compile(r"""['"]\d{4}-\d{2}-\d{2}['"]""")


def test_no_hard_coded_date_strings_outside_config(cfg: Config):
    """§7.4: "No hard-coded date strings anywhere outside config (grep test)".

    Scoped to ``src/`` and ``scripts/``. ``tests/`` is deliberately exempt:
    several tests hand-compute an expected answer on a specific synthetic path
    and need a literal window to do it — that is the point of a fixture, and
    those dates are not research parameters.

    The reference pipeline hard-coded ``'2008':'2017'`` in its overlap check
    (defect A3), which is why that check proved nothing: it compared the
    config's splits against a date range typed in by hand.
    """
    offenders: list[str] = []
    for root in ("src", "scripts"):
        base = cfg.root / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            stripped = _strip_strings_and_comments_keeping_code(source)
            for match in DATE_LITERAL.finditer(stripped):
                line = stripped[: match.start()].count("\n") + 1
                offenders.append(
                    f"{path.relative_to(cfg.root)}:{line}: {match.group(0)}"
                )
    assert not offenders, (
        "hard-coded date literals found outside configs/:\n  "
        + "\n  ".join(offenders)
        + "\nEvery date belongs in configs/data.yaml (spec §7.4)."
    )


def _strip_strings_and_comments_keeping_code(source: str) -> str:
    """Blank out docstrings and comments, keeping code byte positions intact.

    Dates inside docstrings and comments are documentation — the §3.2
    discussion of the 1999-2007 regime shift is *supposed* to name those
    years. What the grep test looks for is a date literal a code path
    actually reads.

    Docstrings are identified with :mod:`ast` rather than by guessing from the
    token stream. An earlier token-based version tried to infer "is this
    string a statement or an expression?" from the preceding token and got the
    condition backwards, so it blanked ``TRAIN_END = "2017-12-31"`` — the
    single most important thing the test exists to catch. ``ast`` answers the
    question exactly: a statement-level string is an ``ast.Expr`` wrapping a
    string constant, and nothing else is.

    Newlines are preserved so reported line numbers stay correct.
    """
    import ast
    import io
    import tokenize

    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def flat(row: int, col: int) -> int:
        return offsets[row - 1] + col

    out = list(source)

    def blank(start: int, end: int) -> None:
        for i in range(max(0, start), min(end, len(out))):
            if out[i] != "\n":
                out[i] = " "

    # Statement-level strings: docstrings and prose blocks.
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return source
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.end_lineno is not None
            and node.end_col_offset is not None
        ):
            blank(
                flat(node.lineno, node.col_offset),
                flat(node.end_lineno, node.end_col_offset),
            )

    # Comments.
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):  # pragma: no cover
        tokens = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            blank(flat(*tok.start), flat(*tok.end))

    return "".join(out)


def test_the_grep_test_can_actually_fail(tmp_path, cfg: Config):
    """A grep test that cannot fail is decoration. Prove it fires."""
    offending = tmp_path / "src" / "prism"
    offending.mkdir(parents=True)
    (offending / "bad.py").write_text(
        'TRAIN_END = "2017-12-31"  # hard-coded, must be caught\n', encoding="utf-8"
    )
    (tmp_path / "scripts").mkdir()

    stripped = _strip_strings_and_comments_keeping_code(
        (offending / "bad.py").read_text(encoding="utf-8")
    )
    assert DATE_LITERAL.search(stripped), (
        "the date-literal regex failed to match an obvious offender"
    )


def test_docstring_dates_are_not_flagged():
    """Documentation may and should name dates; only code paths are policed."""
    sample = '''
"""Universe A runs 1999-01-04 to 2006-12-31."""
# also 2018-01-01 in a comment
x = cfg.data.split("train")
'''
    stripped = _strip_strings_and_comments_keeping_code(sample)
    assert not DATE_LITERAL.search(stripped)
