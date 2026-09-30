"""Typed configuration loader with validation.

Spec §3. The YAML files under ``configs/`` are the single source of truth for
every date, ticker and hyperparameter in the project. This module parses them
into frozen pydantic models and *enforces the spec's non-negotiable
constraints at load time* rather than trusting downstream code to respect
them. In particular:

* Splits must be ordered, disjoint and separated by at least the embargo.
* The embargo must cover the longest forward-target horizon.
* HMM observations must not be rolling statistics (defect B3).
* The encoder latent activation must not be ReLU (defect C3).
* Encoder early stopping must be driven by validation loss (defect C2).
* Universe A must not reference any Universe B-only ticker (§3.2 hazard 1).

A config that violates any of these raises at import of the experiment, not
halfway through a six-hour run.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "Config",
    "DataConfig",
    "EncoderConfig",
    "HMMConfig",
    "Tier1Config",
    "UniverseSpec",
    "load_config",
    "FORWARD_TARGET_HORIZONS",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Horizons of every forward-looking target in ``features/targets.py``. The
#: embargo is validated against ``max`` of these (spec §6.1).
FORWARD_TARGET_HORIZONS: tuple[int, ...] = (5, 20)

_DateRange = tuple[dt.date, dt.date]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #
class SnapshotConfig(_Frozen):
    dir: str
    date: dt.date | None = None
    allow_overwrite: bool = False
    download_end: dt.date


class CalendarConfig(_Frozen):
    exchange: str = "NYSE"
    max_ffill_days: int = Field(ge=0, le=5)


class UniverseSpec(_Frozen):
    start: dt.date
    equity_sectors: list[str] = Field(default_factory=list)
    benchmark: str | None = None
    macro: list[str] = Field(default_factory=list)
    defensive: list[str] = Field(default_factory=list)
    macro_extra: list[str] = Field(default_factory=list)
    inherits: str | None = None
    fit_early: _DateRange | None = None

    @property
    def own_tickers(self) -> list[str]:
        """Tickers declared directly on this universe (not inherited)."""
        out = [*self.equity_sectors, *self.macro, *self.defensive, *self.macro_extra]
        if self.benchmark:
            out.append(self.benchmark)
        return out


class SplitsConfig(_Frozen):
    train: _DateRange
    val: _DateRange
    test: _DateRange
    holdout: _DateRange
    embargo_days: int = Field(ge=0)

    @property
    def named(self) -> dict[str, _DateRange]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "holdout": self.holdout,
        }

    @model_validator(mode="after")
    def _ordered_and_embargoed(self) -> SplitsConfig:
        order = ["train", "val", "test", "holdout"]
        ranges = self.named
        for name, (start, end) in ranges.items():
            if start >= end:
                raise ValueError(f"split {name!r}: start {start} is not before end {end}")
        for earlier, later in zip(order, order[1:]):
            if ranges[earlier][1] >= ranges[later][0]:
                raise ValueError(
                    f"splits {earlier!r} and {later!r} overlap or touch: "
                    f"{earlier} ends {ranges[earlier][1]}, {later} starts {ranges[later][0]}"
                )
        min_embargo = max(FORWARD_TARGET_HORIZONS)
        if self.embargo_days < min_embargo:
            raise ValueError(
                f"embargo_days={self.embargo_days} is shorter than the longest forward "
                f"target horizon ({min_embargo}); forward targets would straddle the "
                "split boundary (spec §6.1)"
            )
        return self


class DecisionConfig(_Frozen):
    frequency: Literal["daily", "weekly"]
    rebalance_day: Literal["MON", "TUE", "WED", "THU", "FRI"]
    execution_lag_days: int = Field(ge=0, le=5)
    rebalance_horizon_days: int = Field(ge=1)


class AllocationConfig(_Frozen):
    long_only: bool
    include_cash: bool
    weight_min: float
    weight_max: float
    weights_sum_to: float

    @model_validator(mode="after")
    def _feasible(self) -> AllocationConfig:
        if self.long_only and self.weight_min < 0:
            raise ValueError("long_only=True is inconsistent with weight_min < 0")
        if self.weight_max <= self.weight_min:
            raise ValueError("weight_max must exceed weight_min")
        return self

    def max_assets_needed(self) -> int:
        """Smallest number of positions that can satisfy ``weights_sum_to``."""
        import math

        return math.ceil(self.weights_sum_to / self.weight_max)


class SeedsConfig(_Frozen):
    master: int
    runs: list[int]


class DataConfig(_Frozen):
    snapshot: SnapshotConfig
    calendar: CalendarConfig
    universes: dict[str, UniverseSpec]
    allocatable: dict[str, list[str]]
    inception: dict[str, dt.date]
    excluded: dict[str, str]
    splits: SplitsConfig
    decision: DecisionConfig
    allocation: AllocationConfig
    warmup_days: int = Field(ge=1)
    seeds: SeedsConfig

    # -- universe resolution ------------------------------------------------ #
    def tickers(self, universe: str) -> list[str]:
        """All tickers in ``universe``, resolving ``inherits``, deduplicated.

        Order is stable (declaration order, parents first) so that any hash
        computed over the ticker list is reproducible.
        """
        spec = self._universe(universe)
        out: list[str] = []
        if spec.inherits:
            out.extend(self.tickers(spec.inherits))
        for t in spec.own_tickers:
            if t not in out:
                out.append(t)
        return out

    def b_only_tickers(self) -> list[str]:
        """Tickers present in B but not in A — the set A must never touch."""
        a = set(self.tickers("A"))
        return [t for t in self.tickers("B") if t not in a]

    def start(self, universe: str) -> pd.Timestamp:
        return pd.Timestamp(self._universe(universe).start)

    def fit_early(self, universe: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        spec = self._universe(universe)
        if spec.fit_early is None:
            raise ValueError(f"universe {universe!r} declares no fit_early range")
        return pd.Timestamp(spec.fit_early[0]), pd.Timestamp(spec.fit_early[1])

    def split(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        try:
            start, end = self.splits.named[name]
        except KeyError:
            raise KeyError(f"unknown split {name!r}; known: {list(self.splits.named)}") from None
        return pd.Timestamp(start), pd.Timestamp(end)

    def _universe(self, universe: str) -> UniverseSpec:
        try:
            return self.universes[universe]
        except KeyError:
            raise KeyError(
                f"unknown universe {universe!r}; known: {sorted(self.universes)}"
            ) from None

    @model_validator(mode="after")
    def _universes_consistent(self) -> DataConfig:
        for name in self.universes:
            for t in self.tickers(name):
                if t in self.excluded:
                    raise ValueError(
                        f"universe {name!r} includes {t!r}, which is on the excluded "
                        f"list: {self.excluded[t]}"
                    )
                if t not in self.inception:
                    raise ValueError(
                        f"universe {name!r} includes {t!r} with no inception date; "
                        "clean.py cannot distinguish pre-inception from a missing session"
                    )
        # §3.2 hazard 1: the A pipeline must never see a B-only ticker.
        a_tickers = set(self.tickers("A"))
        b_only = set(self.b_only_tickers())
        if a_tickers & b_only:
            raise ValueError(f"Universe A leaks B-only tickers: {sorted(a_tickers & b_only)}")
        for name, alloc in self.allocatable.items():
            universe_tickers = set(self.tickers(name))
            missing = [t for t in alloc if t not in universe_tickers]
            if missing:
                raise ValueError(f"allocatable[{name!r}] names non-members: {missing}")
        # Universe B must start no earlier than the latest inception among its
        # own tickers, otherwise its panel opens with pre-inception blanks.
        for name, spec in self.universes.items():
            own = [t for t in spec.own_tickers if t in self.inception]
            if not own:
                continue
            latest = max(self.inception[t] for t in own)
            if spec.start < latest:
                raise ValueError(
                    f"universe {name!r} starts {spec.start} but its own ticker with the "
                    f"latest inception begins {latest}"
                )
        # The allocator must be able to reach weights_sum_to under the cap.
        for name, alloc in self.allocatable.items():
            n_slots = len(alloc) + (1 if self.allocation.include_cash else 0)
            if n_slots < self.allocation.max_assets_needed():
                raise ValueError(
                    f"allocatable[{name!r}] has {n_slots} slots but weight_max="
                    f"{self.allocation.weight_max} requires at least "
                    f"{self.allocation.max_assets_needed()}"
                )
        return self

    @model_validator(mode="after")
    def _horizons_covered(self) -> DataConfig:
        need = max((*FORWARD_TARGET_HORIZONS, self.decision.rebalance_horizon_days))
        if self.splits.embargo_days < need:
            raise ValueError(
                f"embargo_days={self.splits.embargo_days} < max(forward horizon, "
                f"rebalance horizon)={need} (spec §6.1)"
            )
        if self.decision.frequency == "weekly" and self.decision.rebalance_horizon_days != 5:
            raise ValueError("weekly rebalancing implies rebalance_horizon_days=5")
        if self.decision.frequency == "daily" and self.decision.rebalance_horizon_days != 1:
            raise ValueError("daily rebalancing implies rebalance_horizon_days=1")
        return self


# --------------------------------------------------------------------------- #
# hmm.yaml
# --------------------------------------------------------------------------- #
class HMMSpecification(_Frozen):
    observations: list[str] = Field(min_length=1)


class HMMFitConfig(_Frozen):
    universe: str
    covariance_type: Literal["full", "diag", "tied", "spherical"]
    k_range: list[int] = Field(min_length=1)
    n_restarts: int = Field(ge=1)
    n_iter: int = Field(ge=1)
    tol: float = Field(gt=0)

    @field_validator("n_restarts")
    @classmethod
    def _enough_restarts(cls, v: int) -> int:
        if v < 20:
            raise ValueError("spec §8.4 requires at least 20 random restarts per K")
        return v


class HMMSelectionConfig(_Frozen):
    primary: Literal["val_loglik"]
    secondary: list[str]
    min_expected_duration_days: float = Field(gt=0)
    min_unconditional_prob: float = Field(gt=0, lt=1)


class HMMLabelingConfig(_Frozen):
    sort_by: str
    ascending: bool


class HMMWalkforwardConfig(_Frozen):
    scheme: Literal["expanding"]
    refit_cadence: Literal["monthly", "quarterly", "annual"]
    carry_filter_state: bool

    @field_validator("carry_filter_state")
    @classmethod
    def _must_carry(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "spec §8.6 requires the filter state to be carried across refit "
                "boundaries; restarting from startprob_ each month was defect B5"
            )
        return v


class HMMRobustnessConfig(_Frozen):
    refit_on_b_train: bool


class HMMEvaluationConfig(_Frozen):
    drawdown_bear_threshold: float = Field(gt=0, lt=1)
    entropy_saturation_warn: float = Field(gt=0)


class HMMConfig(_Frozen):
    observation_blocklist_substrings: list[str]
    specifications: dict[str, HMMSpecification]
    fit: HMMFitConfig
    selection: HMMSelectionConfig
    labeling: HMMLabelingConfig
    walkforward: HMMWalkforwardConfig
    robustness: HMMRobustnessConfig
    evaluation: HMMEvaluationConfig

    @model_validator(mode="after")
    def _observations_are_not_rolling_stats(self) -> HMMConfig:
        """Defect B3 guard.

        ``spy_vol20`` as an HMM observation is what collapsed the reference
        model into a volatility ladder: rolling statistics are strongly
        autocorrelated, which violates the conditional-independence assumption
        and double-counts evidence. Any observation whose name contains a
        rolling-window marker is rejected outright.
        """
        for spec_name, spec in self.specifications.items():
            for obs in spec.observations:
                lowered = obs.lower()
                for bad in self.observation_blocklist_substrings:
                    if bad.lower() in lowered:
                        raise ValueError(
                            f"HMM specification {spec_name!r} uses observation {obs!r}, "
                            f"which matches the rolling-statistic blocklist entry {bad!r}. "
                            "Rolling statistics violate conditional independence "
                            "(defect B3, spec §8.1). Use a change/difference instead."
                        )
        if any(k < 2 for k in self.fit.k_range):
            raise ValueError("k_range must contain only K >= 2")
        return self


# --------------------------------------------------------------------------- #
# encoder.yaml
# --------------------------------------------------------------------------- #
class EncoderFitConfig(_Frozen):
    universe: str
    robustness_refit_on_b_train: bool


class EncoderWindowConfig(_Frozen):
    size: int = Field(ge=2)
    sweep: list[int]


class EncoderArchConfig(_Frozen):
    latent_dim: int = Field(ge=1)
    latent_dim_sweep: list[int]
    hidden_dim: int = Field(ge=1)
    hidden_dim_sweep: list[int]
    num_layers: int = Field(ge=1)
    dropout: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    latent_activation: Literal["linear", "tanh"]

    @field_validator("latent_activation")
    @classmethod
    def _no_relu(cls, v: str) -> str:
        # Literal already excludes it; this exists so the *reason* is in the
        # traceback if someone widens the Literal.
        if v == "relu":
            raise ValueError(
                "ReLU on the latent produced permanently dead units and post-2018 "
                "activation drift (defect C3, spec §9.2). Use linear or tanh."
            )
        return v


class EncoderTrainingConfig(_Frozen):
    batch_size: int = Field(ge=1)
    lr: float = Field(gt=0)
    max_epochs: int = Field(ge=1)
    patience: int = Field(ge=1)
    early_stopping_metric: str
    checkpoint_metric: str
    grad_clip_norm: float = Field(gt=0)
    n_seeds: int = Field(ge=1)

    @model_validator(mode="after")
    def _validation_driven(self) -> EncoderTrainingConfig:
        """Defect C2 guard: selection must never be driven by training loss."""
        for field in ("early_stopping_metric", "checkpoint_metric"):
            value = getattr(self, field)
            if not value.startswith("val"):
                raise ValueError(
                    f"{field}={value!r} is not a validation metric. Early stopping and "
                    "checkpointing on training loss made model selection meaningless "
                    "(defect C2, spec §9.2)."
                )
        if self.n_seeds < 5:
            raise ValueError("spec §9.3 requires at least 5 seeds")
        return self


class EncoderWalkforwardConfig(_Frozen):
    scheme: Literal["expanding"]
    refit_cadence: Literal["monthly", "quarterly", "annual"]


class EncoderEvaluationConfig(_Frozen):
    dead_unit_var_threshold: float = Field(gt=0)
    drift_test: Literal["ks", "mmd"]


class EncoderConfig(_Frozen):
    fit: EncoderFitConfig
    window: EncoderWindowConfig
    architecture: EncoderArchConfig
    variants: list[str]
    dae_noise_std: float = Field(ge=0)
    vae_kl_weight: float = Field(ge=0)
    pred_targets: list[str]
    training: EncoderTrainingConfig
    walkforward: EncoderWalkforwardConfig
    evaluation: EncoderEvaluationConfig
    baselines: list[str]

    @model_validator(mode="after")
    def _baselines_present(self) -> EncoderConfig:
        required = {"pca_encoder", "random_encoder", "rolling_stats", "raw_window"}
        missing = required - set(self.baselines)
        if missing:
            raise ValueError(
                f"spec §9.4 makes these baselines mandatory; missing: {sorted(missing)}"
            )
        if self.window.size not in self.window.sweep:
            raise ValueError("window.size must be a member of window.sweep")
        if self.architecture.latent_dim not in self.architecture.latent_dim_sweep:
            raise ValueError("architecture.latent_dim must be a member of its sweep")
        return self


# --------------------------------------------------------------------------- #
# experiments/tier1_probes.yaml
# --------------------------------------------------------------------------- #
class ProbeConfig(_Frozen):
    regression: str
    classification: str
    alpha_grid: list[float] = Field(min_length=1)
    tune_on: Literal["validation_folds"]


class AllocatorConfig(_Frozen):
    methods: list[str]
    covariance: str
    covariance_window: int = Field(ge=2)
    cost_bps_per_side: float = Field(ge=0)
    cost_sensitivity_bps: list[float]


class UncertaintyConfig(_Frozen):
    bootstrap: str
    block_length_days: int = Field(ge=1)
    n_bootstrap: int = Field(ge=100)
    ci_level: float = Field(gt=0, lt=1)


class Tier1Config(_Frozen):
    universe: str
    walkforward: EncoderWalkforwardConfig
    variants: list[str]
    diagnostic_variants: list[str]
    targets: list[str]
    probe: ProbeConfig
    allocator: AllocatorConfig
    uncertainty: UncertaintyConfig
    gates: dict[str, list[str]]

    @model_validator(mode="after")
    def _controls_present(self) -> Tier1Config:
        """Spec §10: the controls are mandatory, not optional."""
        required = {"V1", "V1p", "V2", "V3", "V4", "C1", "C2", "C3"}
        missing = required - set(self.variants)
        if missing:
            raise ValueError(
                f"spec §10/§13.1 makes these variants mandatory; missing: {sorted(missing)}"
            )
        if "O1" in self.variants:
            raise ValueError(
                "O1 uses smoothed (leaky) posteriors and is diagnostic only. It belongs "
                "in diagnostic_variants and must never be reported as a result (§10)."
            )
        return self


# --------------------------------------------------------------------------- #
# base.yaml
# --------------------------------------------------------------------------- #
class DeterminismConfig(_Frozen):
    pythonhashseed: int
    torch_deterministic: bool
    warn_on_nondeterminism: bool


class Config(_Frozen):
    project: str
    phase: Literal["A", "B"]
    paths: dict[str, str]
    determinism: DeterminismConfig
    data: DataConfig
    hmm: HMMConfig
    encoder: EncoderConfig
    tier1: Tier1Config
    #: Absolute path of the repository these configs were loaded from.
    root: Path

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    def path(self, key: str) -> Path:
        try:
            return self.root / self.paths[key]
        except KeyError:
            raise KeyError(f"unknown path key {key!r}; known: {sorted(self.paths)}") from None

    @model_validator(mode="after")
    def _phase_a_scope(self) -> Config:
        """Spec §0.3: Phase A must not depend on Phase B packages."""
        if self.phase != "A":
            return self
        import importlib.util

        for forbidden in ("gymnasium", "stable_baselines3"):
            if importlib.util.find_spec(forbidden) is not None:
                raise ValueError(
                    f"{forbidden!r} is installed but phase=A. Phase B dependencies must "
                    "not be present until the §0.3 exit criteria are met."
                )
        return self

    @model_validator(mode="after")
    def _cross_component(self) -> Config:
        if self.hmm.fit.universe not in self.data.universes:
            raise ValueError(f"hmm.fit.universe={self.hmm.fit.universe!r} is not a known universe")
        if self.encoder.fit.universe not in self.data.universes:
            raise ValueError(
                f"encoder.fit.universe={self.encoder.fit.universe!r} is not a known universe"
            )
        if self.tier1.universe not in self.data.universes:
            raise ValueError(f"tier1.universe={self.tier1.universe!r} is not a known universe")
        # §3.2: the models are fitted on a window that ends before the
        # allocation universe's training data begins.
        fit_end = self.data.fit_early(self.hmm.fit.universe)[1]
        b_train_start = self.data.split("train")[0]
        if fit_end >= b_train_start:
            raise ValueError(
                f"fit_early ends {fit_end.date()} which is not before Universe B's train "
                f"start {b_train_start.date()}; the §3.2 no-overlap property is lost"
            )
        # The encoder needs a full window inside the fit range.
        fit_start = self.data.fit_early(self.encoder.fit.universe)[0]
        approx_sessions = int((fit_end - fit_start).days * 252 / 365)
        if approx_sessions <= self.data.warmup_days + max(self.encoder.window.sweep):
            raise ValueError(
                "encoder fit range is too short for the warm-up plus the largest swept window"
            )
        targets = set(self.tier1.targets) | set(self.encoder.pred_targets)
        for t in targets:
            horizon = t.rsplit("_", 1)[-1]
            if horizon.isdigit() and int(horizon) > max(FORWARD_TARGET_HORIZONS):
                raise ValueError(
                    f"target {t!r} has horizon {horizon} but FORWARD_TARGET_HORIZONS "
                    f"tops out at {max(FORWARD_TARGET_HORIZONS)}; the embargo check in "
                    "SplitsConfig would be understated"
                )
        return self


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return loaded


def load_config(
    path: str | Path = "configs/base.yaml",
    root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load, compose and validate the project configuration.

    Parameters
    ----------
    path
        Path to the master config, relative to ``root``.
    root
        Repository root. Defaults to the directory two levels above this file,
        so the loader works from any working directory.
    overrides
        Optional nested mapping merged over the composed config *before*
        validation. Used by tests to build deliberately-invalid configs; not
        used by production code paths, which must read the committed YAML.
    """
    root_path = Path(root).resolve() if root is not None else REPO_ROOT
    base_path = root_path / path
    base = _read_yaml(base_path)
    includes = base.pop("includes", {})

    composed: dict[str, Any] = dict(base)
    for key, rel in includes.items():
        composed[key] = _read_yaml(root_path / rel)
    composed["root"] = root_path

    if overrides:
        composed = _deep_merge(composed, overrides)
    return Config.model_validate(composed)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
