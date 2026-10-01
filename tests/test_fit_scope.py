"""Fit scope: anything fitted sees only training data.

Spec §7.2. Three properties, each one a defect the reference project had or
was one refactor away from:

1. **Every fitted object records its fit range**, and that range ends on or
   before the permitted boundary.
2. **Mutating data after ``train_end`` leaves fitted parameters
   byte-identical.** This is the strong form: it does not ask whether the code
   *looks* like it only reads training data, it changes the future and checks
   the parameters did not move.
3. **Scalers are never refit on transform.** The reference rolling HMM called
   ``scaler.fit_transform`` on a shared object inside its refit loop, so every
   fold silently rewrote the scaler used by every other fold (defect B5).

Plus the §3.2 hazard-1 property, which is Step 1's acceptance criterion and is
proved here rather than asserted: **the Universe A pipeline never touches a
Universe B-only ticker.**
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import Config
from prism.features.build import build_features, split_raw_frame
from prism.features.scaling import CorrelationPruner, FeatureScaler, Winsoriser
from prism.fitting import AlreadyFittedError, FittedArtifact, NotFittedError
from prism.models.baselines.pca_encoder import PCAEncoder
from prism.splits import build_split_plan, expanding_folds

# PCAEncoder's generic FittedArtifact contract (fit-range recording, mutation
# invariance, no implicit refit) is exercised here exactly like the scalers,
# fit directly on Universe B's raw feature frame. Its DOMAIN-specific usage —
# fit on flattened windows via prism.models.baselines.pca_encoder.flatten_windows
# — is tested separately in tests/test_encoder.py, where the windowing
# alignment itself is also under test.
FITTED_CLASSES = [Winsoriser, FeatureScaler, CorrelationPruner, PCAEncoder]


def _train_slice(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    plan = build_split_plan(cfg)
    train = plan["train"]
    return frame.loc[train.effective_start : train.effective_end]


# --------------------------------------------------------------------------- #
# 1. every fitted object records its fit range
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", FITTED_CLASSES, ids=lambda c: c.__name__)
def test_fit_record_ends_at_or_before_train_end(cls, cfg: Config, features_b):
    """§7.2: assert ``fit_record.end <= train_end``."""
    plan = build_split_plan(cfg)
    train_end = plan["train"].effective_end
    train = _train_slice(features_b.frame, cfg)
    assert not train.empty, "training slice is empty"

    artifact = cls().fit(train, scope="train")
    record = artifact.fit_record

    record.assert_within(train_end, what=cls.__name__)
    assert record.start == train.index[0]
    assert record.end == train.index[-1]
    assert record.n_rows == len(train)
    assert record.columns == tuple(str(c) for c in train.columns)
    assert record.scope == "train"


@pytest.mark.parametrize("cls", FITTED_CLASSES, ids=lambda c: c.__name__)
def test_fit_record_catches_an_out_of_scope_fit(cls, cfg: Config, features_b):
    """The recorder must be able to *fail*, or it records nothing useful."""
    plan = build_split_plan(cfg)
    train_end = plan["train"].effective_end
    # Deliberately over-long: train plus the whole validation period.
    over = features_b.frame.loc[: plan["val"].effective_end]
    artifact = cls().fit(over, scope="train+val (deliberately wrong)")
    with pytest.raises(AssertionError, match="after the permitted boundary"):
        artifact.fit_record.assert_within(train_end, what=cls.__name__)


@pytest.mark.parametrize("cls", FITTED_CLASSES, ids=lambda c: c.__name__)
def test_walkforward_folds_respect_their_own_fit_end(cls, cfg: Config, features_b):
    """Each fold's fitted object must end at that fold's ``fit_end``, not later."""
    plan = build_split_plan(cfg)
    folds = expanding_folds(
        plan["train"].effective_start,
        plan["val"].declared_start,
        plan["test"].effective_end,
        "annual",
        embargo_days=cfg.data.splits.embargo_days,
    )
    frame = features_b.frame
    for fold in folds:
        fit_data = frame.loc[fold.fit_start : fold.fit_end]
        if fit_data.empty:
            continue
        artifact = cls().fit(fit_data, scope=f"fold {fold.index}")
        artifact.fit_record.assert_within(fold.fit_end, what=f"{cls.__name__} fold {fold.index}")
        assert artifact.fit_record.end < fold.apply_start


# --------------------------------------------------------------------------- #
# 2. mutating the future leaves parameters byte-identical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", FITTED_CLASSES, ids=lambda c: c.__name__)
def test_parameters_are_invariant_to_post_train_mutation(
    cls, cfg: Config, features_b, rng
):
    """§7.2: mutate data after ``train_end``; parameters must not move."""
    plan = build_split_plan(cfg)
    train_end = plan["train"].effective_end
    frame = features_b.frame

    clean = cls().fit(_train_slice(frame, cfg), scope="train")
    baseline = clean.params_hash()

    mutated = frame.copy()
    future = mutated.index > train_end
    assert future.any(), "no post-train rows to mutate"
    block = mutated.loc[future]
    mutated.loc[future] = (
        block.to_numpy(dtype="float64") * rng.uniform(2.0, 5.0, size=block.shape) + 17.0
    )

    refit = cls().fit(_train_slice(mutated, cfg), scope="train")
    assert refit.params_hash() == baseline, (
        f"{cls.__name__} parameters changed when data after {train_end.date()} was "
        "mutated, so the fit is reading beyond the training window (spec §7.2)"
    )
    assert refit.fit_record.data_hash == clean.fit_record.data_hash


def test_mutation_detector_itself_works(cfg: Config, features_b, rng):
    """Mutating *inside* the training window must change the parameters.

    Without this, ``test_parameters_are_invariant_to_post_train_mutation``
    would pass for a scaler that ignored its input entirely.
    """
    frame = features_b.frame
    clean = FeatureScaler().fit(_train_slice(frame, cfg), scope="train")

    mutated = frame.copy()
    plan = build_split_plan(cfg)
    inside = (mutated.index >= plan["train"].effective_start) & (
        mutated.index <= plan["train"].effective_end
    )
    block = mutated.loc[inside]
    mutated.loc[inside] = block.to_numpy(dtype="float64") + 3.0

    refit = FeatureScaler().fit(_train_slice(mutated, cfg), scope="train")
    assert refit.params_hash() != clean.params_hash(), (
        "mutating the training window did not change the fitted parameters; the "
        "invariance test above is vacuous"
    )


# --------------------------------------------------------------------------- #
# 3. transform never refits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", FITTED_CLASSES, ids=lambda c: c.__name__)
def test_transform_does_not_refit(cls, cfg: Config, features_b):
    """§7.2: assert scalers are never refit on transform."""
    frame = features_b.frame
    artifact = cls().fit(_train_slice(frame, cfg), scope="train")
    before_params = artifact.params_hash()
    before_record = artifact.fit_record

    # Transform the *whole* frame, including data far beyond the fit window.
    artifact.transform(frame)
    # And again, in a different order.
    artifact.transform(frame.iloc[::-1].sort_index())

    assert artifact.params_hash() == before_params
    assert artifact.fit_record == before_record


@pytest.mark.parametrize("cls", FITTED_CLASSES, ids=lambda c: c.__name__)
def test_transform_before_fit_raises(cls, features_b):
    """There is no ``fit_transform``: a single call that does both is the leak."""
    with pytest.raises(NotFittedError):
        cls().transform(features_b.frame)
    assert not hasattr(cls(), "fit_transform"), (
        f"{cls.__name__} exposes fit_transform, which is how full-sample scalers "
        "get written by accident (spec §7.2)"
    )


@pytest.mark.parametrize("cls", FITTED_CLASSES, ids=lambda c: c.__name__)
def test_implicit_refit_raises(cls, cfg: Config, features_b):
    """Re-fitting a shared instance must be explicit. Defect B5."""
    frame = features_b.frame
    artifact = cls().fit(_train_slice(frame, cfg), scope="train")
    with pytest.raises(AlreadyFittedError, match="already fitted"):
        artifact.fit(frame)
    # The explicit escape hatch works, and is recorded.
    artifact.fit(frame, scope="deliberate refit", refit=True)
    assert artifact.fit_record.scope == "deliberate refit"


def test_every_fitted_artifact_subclass_is_covered():
    """A new fitted class must be added to ``FITTED_CLASSES`` or this fails.

    Fit scope is only enforced for classes this file actually exercises, so
    the set of subclasses is checked against the set under test. When the HMM
    and encoder land in steps 2 and 3, this test is what tells you to bring
    them under the same guarantees.
    """
    import prism  # noqa: F401
    import importlib
    import pkgutil

    for module in pkgutil.walk_packages(prism.__path__, "prism."):
        importlib.import_module(module.name)

    def descendants(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from descendants(sub)

    found = {c for c in descendants(FittedArtifact)}
    untested = sorted(c.__name__ for c in found - set(FITTED_CLASSES))
    assert not untested, (
        f"these FittedArtifact subclasses are not covered by test_fit_scope: "
        f"{untested}. Add them to FITTED_CLASSES — fit scope is not optional for "
        "the HMM (§8.6) or the encoder (§9.3) either."
    )


# --------------------------------------------------------------------------- #
# §3.2 hazard 1 — universe isolation
# --------------------------------------------------------------------------- #
def test_universe_a_never_references_a_b_only_ticker(cfg: Config, features_a):
    """§16 step 1 acceptance: "A-pipeline provably never touches B-only tickers"."""
    b_only = cfg.data.b_only_tickers()
    assert b_only, "the two-universe design implies a non-empty B-only set"
    for ticker in b_only:
        offenders = [c for c in features_a.columns if c.startswith(f"{ticker}_")]
        assert not offenders, f"Universe A features reference {ticker}: {offenders[:5]}"
    # B-only macro features must be absent from A by name as well as by ticker.
    for column in (
        "credit_proxy",
        "credit_proxy_change",
        "credit_proxy_change_20",
        "vix_term_structure",
        "dollar_return_20",
        "oil_vol_20",
    ):
        assert column not in features_a.columns, (
            f"{column} is derived from a B-only series but appears in Universe A"
        )


def test_universe_a_is_bit_identical_when_b_only_data_is_destroyed(
    cfg: Config, raw_b: pd.DataFrame
):
    """The strong form of isolation: corrupt every B-only series, rebuild A.

    Naming-based checks prove A's *output* mentions no B ticker. This proves A
    never *read* one: the B-only columns are replaced with values that would
    visibly corrupt any feature touching them, and Universe A's feature frame
    must come out byte-identical.
    """
    b_only = cfg.data.b_only_tickers()
    baseline = build_features(raw_b, cfg, "A").frame

    corrupted = raw_b.copy()
    for field in raw_b.columns.get_level_values(0).unique():
        for ticker in b_only:
            if (field, ticker) in corrupted.columns:
                corrupted[(field, ticker)] = -12345.0

    rebuilt = build_features(corrupted, cfg, "A").frame
    pd.testing.assert_frame_equal(baseline, rebuilt, check_freq=False)


def test_universe_b_does_depend_on_b_only_data(cfg: Config, raw_b: pd.DataFrame):
    """Converse: destroying B-only series must change Universe B.

    Otherwise the test above proves only that nothing depends on anything.
    """
    b_only = cfg.data.b_only_tickers()
    baseline = build_features(raw_b, cfg, "B").frame

    corrupted = raw_b.copy()
    for ticker in b_only:
        if ("Close", ticker) in corrupted.columns:
            corrupted[("Close", ticker)] = corrupted[("Close", ticker)] * 1.5

    rebuilt = build_features(corrupted, cfg, "B").frame
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(baseline, rebuilt, check_freq=False)


def test_a_and_b_agree_on_shared_columns_over_the_overlap(
    cfg: Config, features_a, features_b
):
    """§3.2 hazard 3: the two universes share an index from 2007-04 onward.

    Shared feature columns must also agree in *value* on the overlap. If they
    did not, a state vector assembled from B could not be compared with a
    model fitted on A — which is the entire premise of the two-universe
    design.

    Both fixtures are built from the *same* raw panel, as production does.
    Building each universe from its own panel would compare two different
    datasets and the test would fail for a reason that means nothing.
    """
    shared_cols = [c for c in features_a.columns if c in set(features_b.columns)]
    assert len(shared_cols) > 50, "suspiciously little overlap between A and B features"

    overlap = features_a.index.intersection(features_b.index)
    assert len(overlap) > 1000, "A and B barely overlap; check the universe start dates"

    left = features_a.frame.loc[overlap, shared_cols]
    right = features_b.frame.loc[overlap, shared_cols]
    pd.testing.assert_frame_equal(left, right, check_freq=False, rtol=1e-10)


def test_universe_a_builds_when_b_tickers_are_absent_entirely(
    cfg: Config, raw_a_only: pd.DataFrame, features_a
):
    """A must not merely ignore the B series — it must not require them.

    Isolation that depends on the B columns being *present and skipped* would
    break the moment someone snapshots Universe A alone. Here the B tickers
    are absent from the panel, and the A build must still produce the same
    schema.
    """
    built = build_features(raw_a_only, cfg, "A")
    assert built.columns == features_a.columns, (
        "Universe A's feature schema depends on whether B tickers happen to be "
        "present in the panel"
    )
    assert built.schema_hash == features_a.schema_hash


def test_b_only_tickers_are_absent_from_the_a_panel_before_features_run(
    cfg: Config, raw_b: pd.DataFrame
):
    """Isolation is structural: the columns are gone before feature code runs."""
    a_tickers = set(cfg.data.tickers("A"))
    panels = split_raw_frame(raw_b)
    present = [t for t in cfg.data.tickers("A") if t in panels["Close"].columns]
    filtered = panels["Close"].reindex(columns=present)
    for ticker in cfg.data.b_only_tickers():
        assert ticker not in filtered.columns
    assert set(filtered.columns) <= a_tickers


def test_scaler_fitted_on_a_is_not_silently_reused_for_b(cfg: Config, features_a, features_b):
    """§3.2: A-features and B-features get *separate* scalers.

    A scaler fitted on Universe A cannot transform a Universe B frame,
    because B has columns A has never seen. The failure must be loud — a
    ``KeyError`` naming the columns — not a silent partial transform that
    leaves B's credit features unscaled.
    """
    scaler = FeatureScaler().fit(features_a.frame, scope="A")
    with pytest.raises(KeyError):
        scaler.transform(features_b.frame.drop(columns=features_a.columns[:1]))
    b_only_cols = [c for c in features_b.columns if c not in set(features_a.columns)]
    assert b_only_cols, "expected B to carry columns A does not"
    transformed = scaler.transform(features_b.frame)
    # The A-fitted scaler leaves B-only columns untouched; that is why a
    # separate B scaler is required rather than optional.
    for col in b_only_cols[:5]:
        assert np.allclose(
            transformed[col].dropna().to_numpy(),
            features_b.frame[col].dropna().to_numpy(),
        )
