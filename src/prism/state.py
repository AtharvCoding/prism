"""State vector assembly for the ablation variants (nine in §10, plus Phase B's C4). Spec §10, §16 step 3b.

Spec §10: "All variants share the same index, scaling policy, and
portfolio-state block." This module *assembles*; it does not fit or train
anything. Every expensive, learned component — HMM posteriors, encoder
latents, the random-encoder control, the threshold-regime control, the leaky
smoothed oracle — is produced upstream by its own walk-forward module
(:mod:`prism.models.hmm.walkforward`, :mod:`prism.models.encoder.walkforward`,
:mod:`prism.models.baselines.random_encoder`,
:mod:`prism.models.baselines.threshold_regime`) and handed in here as a
precomputed, already date-aligned frame via :class:`StateArtifacts`. The two
things this module computes itself — V1's windowed counterpart (V1') and C3's
row-shuffle of the HMM posteriors — are pure, unfitted reshape/permutation
operations, not model fits, which is why they live here rather than upstream.

**V1' carries the raw window, not a PCA compression.** C1 (the random
encoder) already isolates "more dimensions" at matched ``latent_dim``; if V1'
were also compressed to ``latent_dim`` it would duplicate C1's job and weaken
the "more history, unlearned" claim V1' exists to make (gate:
``V2 > V1p and V2 > C1`` are two separate falsifications, not one). V1' is
therefore the full window flattened — ``window * n_features`` columns, via
:func:`~prism.models.baselines.pca_encoder.flatten_windows`.

**O1 (the leaky oracle)** is built from **per-fold smoothed** posteriors —
each HMM walk-forward fold's already-fitted, canonically-relabelled model,
run through ``hmmlearn``'s own forward-backward (``predict_proba``) over that
fold's fit+apply window, apply rows kept. This is leaky only *within* a fold
(an upper bound on regime knowledge, not a global one), reusing
:func:`~prism.models.hmm.walkforward.hmm_walkforward`'s fold structure rather
than inventing a second full-sample fit. O1 is diagnostic only — spec §10,
enforced at the config layer (``Tier1Config`` rejects it from ``variants``)
and asserted again at the end of this module.

**Scaling policy.** Only the base (V1) feature block is scaled — the other
blocks (latents, posteriors, random latents, threshold one-hot) are already
bounded/standardised by their own producing pipeline, and scaling a
probability simplex a second time would be meaningless. The *same*
:class:`~prism.features.scaling.FeatureScaler` must be reused across every
variant's call for the comparison to be fair (spec §10's "same ... scaling
policy"): production call sites (:mod:`scripts.03b_build_states`) fit one
scaler on the train split and pass it explicitly to every ``build_state``
call. Omitting ``scaler`` fits a fresh one on whatever frame is given, which
is fine for a single self-contained call (e.g. a test) but not across
variants that must be compared against each other.

**Portfolio block.** Phase B's allocator state — current weights, time since
last rebalance, cumulative turnover — is accepted as an optional, already
date-indexed frame and concatenated unchanged. Spec §10: "In Phase A the
allocator supplies these directly, so ``state.py`` must accept them as an
optional block, wired but unused." Nothing here reads its columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prism.config import Config
from prism.features.build import FeatureSet
from prism.features.scaling import FeatureScaler
from prism.models.baselines.pca_encoder import flatten_windows
from prism.utils.hashing import hash_object
from prism.utils.seeding import derive_seed

__all__ = [
    "StateArtifacts",
    "StateFrame",
    "MissingArtifactError",
    "PHASE_B_VARIANTS",
    "build_state",
    "shuffle_posteriors",
]


class MissingArtifactError(ValueError):
    """Raised when a variant's required precomputed artifact was not supplied."""


@dataclass(frozen=True)
class StateArtifacts:
    """Precomputed, walk-forward-produced component frames. Spec §10.

    Every frame here is the output of some OTHER module's own fit/walk-forward
    pipeline — this module does not fit, train or walk-forward anything; it
    only reads these and the base features. Each must already be date-indexed
    and causal; :func:`build_state` intersects indices but does no alignment
    beyond that (no fill, no reindex-with-nearest).
    """

    #: V2, V4 — LSTM encoder latents, ``prism.models.encoder.walkforward``.
    latents: pd.DataFrame | None = None
    #: V3, V4, C3 — HMM filtered posteriors, ``prism.models.hmm.walkforward``.
    posteriors: pd.DataFrame | None = None
    #: C1 — frozen random-encoder latents, ``prism.models.baselines.random_encoder``.
    random_latents: pd.DataFrame | None = None
    #: C2 — one-hot threshold-regime states, ``prism.models.baselines.threshold_regime``.
    threshold_states: pd.DataFrame | None = None
    #: O1 only — deliberately leaky, per-fold smoothed HMM posteriors. Never
    #: fed into anything but the diagnostic variant.
    smoothed_posteriors: pd.DataFrame | None = None


#: Variants added for Phase B (DECISIONS.md D-032). Deliberately NOT in
#: ``cfg.tier1.variants``: the Tier 1 pre-registration fixed that list and its
#: config, and changing either would change the code version behind the Tier 1
#: results (D-031). C4 = V2 + C2's threshold-regime columns. It isolates whether
#: the HMM's probabilistic posteriors matter beyond a threshold once a temporal
#: model is present, i.e. V4 > C4 where V4 > V2 could not attribute the gain.
PHASE_B_VARIANTS: tuple[str, ...] = ("C4",)

#: Which StateArtifacts fields each variant needs, beyond the base features.
_REQUIRES: dict[str, tuple[str, ...]] = {
    "V1": (),
    "V1p": (),
    "V2": ("latents",),
    "V3": ("posteriors",),
    "V4": ("latents", "posteriors"),
    "C1": ("random_latents",),
    "C2": ("threshold_states",),
    "C3": ("posteriors",),
    "O1": ("smoothed_posteriors",),
    "C4": ("latents", "threshold_states"),
}


@dataclass(frozen=True)
class StateFrame:
    """One variant's assembled state matrix, plus its provenance. Spec §16 step 3b."""

    variant: str
    frame: pd.DataFrame
    #: :attr:`~prism.features.build.FeatureSet.schema_hash` of the base
    #: features this was assembled from — detects a feature-schema drift
    #: independent of a change to this variant's own extra columns.
    feature_schema_hash: str
    #: Hash of this variant's full, realised column set. Spec §16 step 3b:
    #: "schema hash recorded" — a state frame built against a different
    #: feature or artifact schema is detectable rather than merely wrong.
    schema_hash: str
    includes_portfolio_block: bool

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index)

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]


def shuffle_posteriors(posteriors: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """C3: permute WHICH DATE each posterior row is attached to. Spec §10.

    A permutation control must destroy the regime signal's *timing* while
    preserving its *marginal distribution exactly* — not approximately. The
    only operation with that property is permuting row order: the same
    K-dimensional probability vectors that occurred over the sample are still
    exactly the values present (same multiset, same per-column mean/variance,
    same simplex membership), just attached to the wrong dates. The state
    vector therefore sees a real draw from the HMM's output distribution that
    carries no information about the regime actually prevailing on that date.
    """
    if len(posteriors) < 2:
        raise ValueError("shuffle_posteriors needs at least 2 rows to be a meaningful control")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(posteriors))
    shuffled = posteriors.to_numpy()[order]
    return pd.DataFrame(shuffled, index=posteriors.index, columns=posteriors.columns)


def _missing_artifacts(variant: str, artifacts: StateArtifacts) -> list[str]:
    return [name for name in _REQUIRES[variant] if getattr(artifacts, name) is None]


def build_state(
    features: FeatureSet,
    cfg: Config,
    variant: str,
    *,
    artifacts: StateArtifacts | None = None,
    scaler: FeatureScaler | None = None,
    portfolio_block: pd.DataFrame | None = None,
    shuffle_seed: int | None = None,
    v1p_window: int | None = None,
) -> StateFrame:
    """Assemble ``variant``'s state matrix. Spec §10, §16 step 3b.

    Parameters
    ----------
    features
        A built :class:`~prism.features.build.FeatureSet` (the V1 baseline).
    variant
        One of ``cfg.tier1.variants``, ``cfg.tier1.diagnostic_variants`` or
        :data:`PHASE_B_VARIANTS`.
    artifacts
        Precomputed component frames this variant needs (see
        :data:`_REQUIRES`); raises :class:`MissingArtifactError` naming
        exactly which are absent, rather than failing deeper inside pandas.
    scaler
        A :class:`~prism.features.scaling.FeatureScaler` already fit on the
        intended scope (e.g. the train split). If omitted, one is fit here on
        ``features.frame`` itself — only acceptable when this call does not
        need to be compared against a sibling call for a different variant
        (see the module docstring's "Scaling policy").
    portfolio_block
        Optional, already date-indexed frame of Phase B allocator state
        (current weights, time since rebalance, cumulative turnover). Wired
        but unused in Phase A (spec §10): concatenated as-is, never read.
    shuffle_seed
        Overrides the derived default seed for C3's permutation. Mostly for
        tests that need a specific, reproducible shuffle.
    v1p_window
        V1′'s window length. It must equal the window the *selected* encoder
        was trained on (spec §10, design gap D3: V1′ carries the information
        V2's encoder saw). Production passes the step-3 selection from
        ``encoder_summary.json``; the fallback, ``cfg.encoder.window.size``, is
        only the config default and is NOT necessarily what step 3 selected
        (it selected 10 against a default of 30).

    Returns
    -------
    A :class:`StateFrame` whose index is the intersection of the base
    features' index with every supplied artifact's index (and, if given, the
    portfolio block's), sorted and de-duplicated, containing no NaN/inf.
    """
    known_variants = (
        set(cfg.tier1.variants) | set(cfg.tier1.diagnostic_variants) | set(PHASE_B_VARIANTS)
    )
    if variant not in known_variants:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(known_variants)}")

    artifacts = artifacts if artifacts is not None else StateArtifacts()
    missing = _missing_artifacts(variant, artifacts)
    if missing:
        raise MissingArtifactError(
            f"variant {variant!r} requires StateArtifacts.{', '.join(missing)}, which "
            "were not supplied. state.py assembles from precomputed walk-forward "
            "output; it does not fit models itself (spec §10)."
        )

    fitted_scaler = scaler
    if fitted_scaler is None:
        fitted_scaler = FeatureScaler().fit(features.frame, scope=f"state assembly ({variant})")
    base = fitted_scaler.transform(features.frame)

    extra_blocks: list[pd.DataFrame] = []
    if variant == "V1":
        pass
    elif variant == "V1p":
        extra_blocks.append(
            flatten_windows(base, v1p_window if v1p_window is not None else cfg.encoder.window.size)
        )
    elif variant == "V2":
        extra_blocks.append(artifacts.latents)
    elif variant == "V3":
        extra_blocks.append(artifacts.posteriors)
    elif variant == "V4":
        extra_blocks.append(artifacts.latents)
        extra_blocks.append(artifacts.posteriors)
    elif variant == "C1":
        extra_blocks.append(artifacts.random_latents)
    elif variant == "C2":
        extra_blocks.append(artifacts.threshold_states)
    elif variant == "C3":
        seed = (
            shuffle_seed
            if shuffle_seed is not None
            else derive_seed(cfg.data.seeds.master, "state_c3_shuffle")
        )
        extra_blocks.append(shuffle_posteriors(artifacts.posteriors, seed=seed))
    elif variant == "O1":
        extra_blocks.append(artifacts.smoothed_posteriors)
    elif variant == "C4":
        extra_blocks.append(artifacts.latents)
        extra_blocks.append(artifacts.threshold_states)
    else:  # pragma: no cover - guarded by the known_variants check above
        raise AssertionError(f"unhandled variant {variant!r}")

    index = pd.DatetimeIndex(base.index)
    for block in extra_blocks:
        index = index.intersection(block.index)
    if len(index) == 0:
        raise ValueError(
            f"variant {variant!r}: no overlapping dates between the base features and "
            "the supplied artifacts"
        )
    index = index.sort_values()

    pieces = [base.loc[index], *(block.loc[index] for block in extra_blocks)]
    frame = pd.concat(pieces, axis=1)
    if frame.columns.has_duplicates:
        dupes = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"variant {variant!r}: duplicate columns across blocks: {dupes}")

    includes_portfolio_block = portfolio_block is not None
    if portfolio_block is not None:
        absent = index.difference(portfolio_block.index)
        if len(absent):
            raise ValueError(
                f"portfolio_block is missing {len(absent)} date(s) the state frame "
                f"needs, first {absent[0].date()}"
            )
        frame = pd.concat([frame, portfolio_block.loc[index]], axis=1)
        if frame.columns.has_duplicates:
            dupes = frame.columns[frame.columns.duplicated()].unique().tolist()
            raise ValueError(f"variant {variant!r}: portfolio_block collides on columns: {dupes}")

    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise AssertionError(f"variant {variant!r}: assembled index is not sorted/unique")
    if not np.isfinite(frame.to_numpy(dtype="float64")).all():
        raise ValueError(f"variant {variant!r}: assembled state contains NaN/inf")
    if variant in cfg.tier1.diagnostic_variants and variant != "O1":  # pragma: no cover
        raise AssertionError(f"unexpected diagnostic variant {variant!r}")
    if variant == "O1" and variant in cfg.tier1.variants:
        raise AssertionError("O1 must never be a reportable variant (spec §10)")

    schema_hash = hash_object(
        {
            "variant": variant,
            "feature_schema_hash": features.schema_hash,
            "columns": [str(c) for c in frame.columns],
            "includes_portfolio_block": includes_portfolio_block,
        }
    )

    return StateFrame(
        variant=variant,
        frame=frame,
        feature_schema_hash=features.schema_hash,
        schema_hash=schema_hash,
        includes_portfolio_block=includes_portfolio_block,
    )
