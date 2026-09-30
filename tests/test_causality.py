"""Causality: the output for day ``t`` must not depend on any data after ``t``.

Spec §7.1, and the most important file in the repository.

The test has two halves, because either alone is insufficient:

**Truncation invariance.** Rebuilding on data up to ``t`` must reproduce the
rows up to ``t`` exactly. This catches anything computed over the whole
sample — a full-sample scaler, a global quantile, a ``dropna`` whose effect
propagates backwards.

**Perturbation invariance.** Randomising every value *after* ``t`` must leave
the rows up to ``t`` byte-identical. This catches the leaks truncation
misses: a ``shift(-1)``, a ``bfill``, a centred rolling window. Truncation
alone would not notice a ``shift(-1)``, because the shifted value is simply
absent from the truncated frame and both sides end up NaN in the same place.

**A causality test that never fails is worthless** (§7.1). So this file does
not only test the real builder; it constructs three *known-bad* builders —
a full-sample scaler, a ``bfill``, and a ``shift(-1)`` — and asserts the
harness rejects each one. If those assertions ever pass silently, the harness
has stopped working and every PASS above it is meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import Config
from prism.features.build import build_features, split_raw_frame
from prism.features.targets import build_targets

# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #
CausalityError = AssertionError


def assert_causal(
    build_fn,
    raw_panel: pd.DataFrame,
    cut_dates,
    *,
    rtol: float = 1e-8,
    rng: np.random.Generator | None = None,
    label: str = "build_fn",
) -> None:
    """Assert ``build_fn`` output for ``t`` is invariant to data after ``t``.

    Parameters
    ----------
    build_fn
        Callable taking a raw panel and returning a date-indexed frame.
    raw_panel
        The full input. Must be date-indexed and sliceable with ``.loc[:t]``.
    cut_dates
        Dates at which to truncate and perturb. Choose several, spread
        through the sample, and none of them inside the warm-up period.

    Raises
    ------
    AssertionError
        With the stage, the cut date, and which half of the test failed.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    full = build_fn(raw_panel)

    for cut in cut_dates:
        cut_ts = pd.Timestamp(cut)
        expected = full.loc[:cut_ts]
        if expected.empty:
            raise AssertionError(
                f"{label}: cut date {cut_ts.date()} precedes the warm-up period, so "
                "the comparison would be vacuous. Choose a later cut date."
            )

        # (a) truncation: rebuilding on data up to t reproduces rows up to t.
        truncated = build_fn(raw_panel.loc[:cut_ts])
        try:
            pd.testing.assert_frame_equal(
                expected, truncated, rtol=rtol, check_freq=False
            )
        except AssertionError as exc:
            raise AssertionError(
                f"{label}: TRUNCATION invariance failed at {cut_ts.date()}. "
                f"Rebuilding on data up to {cut_ts.date()} changed rows at or before "
                f"{cut_ts.date()}, so something is computed over the whole sample "
                f"(a global scaler, a global quantile, a backward-propagating "
                f"dropna).\n{exc}"
            ) from exc

        # (b) perturbation: randomising the future leaves the past unchanged.
        noisy = raw_panel.copy()
        future = noisy.index > cut_ts
        if not future.any():
            continue
        block = noisy.loc[future]
        multiplier = generator.uniform(0.5, 1.5, size=block.shape)
        noisy.loc[future] = block.to_numpy(dtype="float64") * multiplier
        try:
            pd.testing.assert_frame_equal(
                expected, build_fn(noisy).loc[:cut_ts], rtol=rtol, check_freq=False
            )
        except AssertionError as exc:
            raise AssertionError(
                f"{label}: PERTURBATION invariance failed at {cut_ts.date()}. "
                f"Randomising data after {cut_ts.date()} changed rows at or before it, "
                f"so a value from the future is reaching into the past (a negative "
                f"shift, a bfill, or a centred window).\n{exc}"
            ) from exc


def _cut_dates(index: pd.DatetimeIndex, n: int = 4) -> list[pd.Timestamp]:
    """``n`` cut dates spread across the back two-thirds of ``index``."""
    if len(index) < 10:
        raise ValueError("index too short to choose cut dates")
    lo = int(len(index) * 0.35)
    positions = np.linspace(lo, len(index) - 2, n, dtype=int)
    return [pd.Timestamp(index[p]) for p in dict.fromkeys(positions)]


# --------------------------------------------------------------------------- #
# the real stages
# --------------------------------------------------------------------------- #
def test_build_features_is_causal_universe_a(cfg: Config, raw_small: pd.DataFrame, rng):
    """§7.1: ``build_features`` must PASS. Universe A, short panel."""
    def build(raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw, cfg, "A").frame

    full = build(raw_small)
    assert not full.empty, "no warm sessions in the short panel"
    assert_causal(build, raw_small, _cut_dates(full.index), rng=rng, label="build_features[A]")


@pytest.mark.slow
def test_build_features_is_causal_universe_b(cfg: Config, raw_b: pd.DataFrame, rng):
    """§7.1: ``build_features`` must PASS. Universe B, full panel.

    Universe B is the case that matters: it carries the credit proxy, the
    defensive sleeve and the VIX term structure, so it exercises every
    builder including the ones that only exist for B.
    """
    def build(raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw, cfg, "B").frame

    full = build(raw_b)
    assert_causal(build, raw_b, _cut_dates(full.index, 3), rng=rng, label="build_features[B]")


def test_every_feature_family_is_exercised(features_b):
    """A causality PASS is only meaningful if it covered every builder."""
    families = features_b.families
    for name in ("asset", "cross", "macro"):
        assert families.get(name), f"no {name} features were built, so none were tested"
    # B-only macro features must be present, or the B causality test is
    # silently no stronger than the A one.
    for column in ("credit_proxy_change_20", "vix_term_structure", "oil_vol_20"):
        assert column in features_b.columns, f"{column} absent; B coverage is incomplete"


def test_forward_targets_are_not_causal(cfg: Config, raw_small: pd.DataFrame, rng):
    """§5.4: the targets must FAIL the causality test.

    This is not a paradox. Forward targets look into the future by definition;
    a target that passed would not be forward-looking, and the probes would be
    predicting the past. Asserting the failure pins the direction of the
    dependency and doubles as a live check that the harness detects leakage in
    real code, not only in the synthetic bad builders below.
    """
    panels = split_raw_frame(raw_small)
    benchmark = cfg.data.universes["A"].benchmark
    sleeve = cfg.data.universes["A"].equity_sectors

    def build(raw: pd.DataFrame) -> pd.DataFrame:
        close = split_raw_frame(raw)["Close"]
        members = [t for t in sleeve if t in close.columns]
        out = build_targets(close, benchmark=benchmark, sleeve=members)
        return out.dropna()

    del panels
    full = build(raw_small)
    with pytest.raises(AssertionError, match="invariance failed"):
        assert_causal(build, raw_small, _cut_dates(full.index, 2), rng=rng, label="targets")


# --------------------------------------------------------------------------- #
# validation against known-bad implementations — §7.1
# --------------------------------------------------------------------------- #
def _leaky_global_scaler(cfg: Config, cut_dates, universe: str = "A"):
    """Known-bad: standardise using moments of the WHOLE sample.

    The single most common leak in a research pipeline, and the mechanism the
    reference notebooks used for both the HMM and the LSTM. Caught by
    truncation (the moments change) and by perturbation (likewise).
    """
    del cut_dates

    def build(raw: pd.DataFrame) -> pd.DataFrame:
        frame = build_features(raw, cfg, universe).frame
        return (frame - frame.mean()) / frame.std(ddof=0).replace(0.0, 1.0)

    return build


def _leaky_bfill(cfg: Config, cut_dates, universe: str = "A"):
    """Known-bad: backward-fill. Defect A2 — writes the future into the past.

    The injection is built *against the cut dates* deliberately. A ``bfill``
    is only a leak when the gap it fills **straddles** the boundary: a hole
    entirely in the past is filled from the past and nothing crosses ``t``.
    An earlier version of this test punched its hole at a fixed row far
    before every cut date, so no future value ever reached backwards and the
    harness correctly reported no leak — the injection, not the harness, was
    wrong.

    So the hole is placed at each cut date and the two sessions before it,
    which is precisely the real-world failure: a gap at a split boundary
    filled from the other side of the boundary.
    """
    holes = [pd.Timestamp(c) for c in cut_dates]

    def build(raw: pd.DataFrame) -> pd.DataFrame:
        holed = raw.copy()
        index = pd.DatetimeIndex(holed.index)
        for hole in holes:
            end = int(index.searchsorted(hole, side="right"))
            start = max(0, end - 3)
            if start < len(holed):
                holed.iloc[start:end] = np.nan
        holed = holed.bfill()
        return build_features(holed, cfg, universe).frame

    return build


def _leaky_negative_shift(cfg: Config, cut_dates, universe: str = "A"):
    """Known-bad: ``shift(-1)``. Tomorrow's feature value, labelled today.

    Defect C1's mechanism, and the one truncation invariance alone cannot
    see — which is exactly why perturbation invariance is not optional.
    """
    del cut_dates

    def build(raw: pd.DataFrame) -> pd.DataFrame:
        frame = build_features(raw, cfg, universe).frame
        return frame.shift(-1).dropna()

    return build


def _leaky_centred_window(cfg: Config, cut_dates, universe: str = "A"):
    """Known-bad: a centred rolling window straddles ``t``."""
    del cut_dates

    def build(raw: pd.DataFrame) -> pd.DataFrame:
        frame = build_features(raw, cfg, universe).frame
        return frame.rolling(21, center=True, min_periods=1).mean()

    return build


def _leaky_future_quantile_winsorise(cfg: Config, cut_dates, universe: str = "A"):
    """Known-bad: winsorise at quantiles of the full sample. Spec §5.5.

    Included because it is the subtlest of the five: the clip bounds move by a
    fraction of a percent when the future changes, so the leak is real but
    small. It is here to confirm the harness's tolerance is tight enough to
    see a leak that does not announce itself.
    """
    del cut_dates

    def build(raw: pd.DataFrame) -> pd.DataFrame:
        frame = build_features(raw, cfg, universe).frame
        lo, hi = frame.quantile(0.01), frame.quantile(0.99)
        return frame.clip(lower=lo, upper=hi, axis=1)

    return build


KNOWN_BAD = {
    "global_scaler": _leaky_global_scaler,
    "bfill": _leaky_bfill,
    "negative_shift": _leaky_negative_shift,
    "centred_window": _leaky_centred_window,
    "future_quantile_winsorise": _leaky_future_quantile_winsorise,
}


@pytest.mark.parametrize("name", sorted(KNOWN_BAD))
def test_harness_rejects_known_bad_builders(
    name: str, cfg: Config, raw_small: pd.DataFrame, rng
):
    """§7.1: the harness must FAIL on each injected leak.

    "A causality test that never fails is worthless." Each of these is a real
    defect from Appendix A or its direct mechanism. If any of them passes,
    ``assert_causal`` is broken and every other result in this file is void.
    """
    clean = build_features(raw_small, cfg, "A").frame
    cuts = _cut_dates(clean.index, 3)
    build = KNOWN_BAD[name](cfg, cuts)

    with pytest.raises(AssertionError, match="invariance failed") as excinfo:
        assert_causal(build, raw_small, cuts, rng=rng, label=f"KNOWN-BAD[{name}]")
    assert "KNOWN-BAD" in str(excinfo.value)


def _truncation_detects(build, raw: pd.DataFrame, cut: pd.Timestamp) -> bool:
    """Whether the truncation half alone flags ``build`` at ``cut``."""
    expected = build(raw).loc[:cut]
    try:
        pd.testing.assert_frame_equal(
            expected, build(raw.loc[:cut]), rtol=1e-8, check_freq=False
        )
    except AssertionError:
        return True
    return False


def _perturbation_detects(
    build, raw: pd.DataFrame, cut: pd.Timestamp, rng: np.random.Generator
) -> bool:
    """Whether the perturbation half alone flags ``build`` at ``cut``."""
    expected = build(raw).loc[:cut]
    noisy = raw.copy()
    future = noisy.index > cut
    block = noisy.loc[future]
    noisy.loc[future] = block.to_numpy(dtype="float64") * rng.uniform(
        0.5, 1.5, size=block.shape
    )
    try:
        pd.testing.assert_frame_equal(
            expected, build(noisy).loc[:cut], rtol=1e-8, check_freq=False
        )
    except AssertionError:
        return True
    return False


@pytest.mark.parametrize("name", sorted(KNOWN_BAD))
def test_each_half_independently_detects_each_leak(
    name: str, cfg: Config, raw_small: pd.DataFrame, rng
):
    """Neither half of the harness is dead code.

    ``assert_causal`` runs truncation and perturbation in sequence, so a
    suite-level PASS cannot tell you whether *both* halves are working — one
    could be silently vacuous and every test would stay green. This test
    exercises each half on its own against each injected leak and requires
    both to fire.

    It also settles a question worth recording rather than guessing at. It is
    tempting to claim truncation is blind to a ``shift(-1)`` and that
    perturbation is what saves you. That is **not** true here: the truncated
    rebuild has no ``t+1`` to shift from, so its boundary row differs and
    truncation fires too. For a one-step shift both halves in fact detect the
    leak only at the boundary row — truncation through the row count, 
    perturbation through that row's values. The two halves are kept because
    they are independent detectors of the same property, not because either
    covers a blind spot of the other.
    """
    clean = build_features(raw_small, cfg, "A").frame
    cuts = _cut_dates(clean.index, 3)
    build = KNOWN_BAD[name](cfg, cuts)
    cut = cuts[1]

    assert _truncation_detects(build, raw_small, cut), (
        f"truncation invariance did NOT detect the {name!r} leak at {cut.date()}; "
        "that half of the harness is not doing its job"
    )
    assert _perturbation_detects(build, raw_small, cut, rng), (
        f"perturbation invariance did NOT detect the {name!r} leak at {cut.date()}; "
        "that half of the harness is not doing its job"
    )


def test_each_half_passes_the_real_builder(cfg: Config, raw_small: pd.DataFrame, rng):
    """The converse: neither half is a false positive on causal code.

    A harness that flagged everything would also be worthless, and would be
    indistinguishable from a working one when every test asserts failure.
    """
    def build(raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw, cfg, "A").frame

    cut = _cut_dates(build(raw_small).index, 3)[1]
    assert not _truncation_detects(build, raw_small, cut)
    assert not _perturbation_detects(build, raw_small, cut, rng)


def test_cut_date_inside_warmup_is_rejected(cfg: Config, raw_small: pd.DataFrame):
    """A cut date before the warm-up makes the comparison vacuous, so it raises.

    Without this guard, a badly chosen cut date would compare two empty
    frames and report PASS — the exact way a causality suite rots into
    decoration.
    """
    def build(raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw, cfg, "A").frame

    early = pd.Timestamp(raw_small.index[5])
    with pytest.raises(AssertionError, match="precedes the warm-up"):
        assert_causal(build, raw_small, [early], label="warmup-guard")
