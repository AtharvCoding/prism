"""Walk-forward ridge/logistic probes from each feature set to forward targets. Spec §13.1.

Built now, at step 3, because the encoder's own acceptance criterion (spec
§9.4, §16 step 3: "beats PCA and random encoder on the Tier-1 probe with
non-overlapping confidence intervals") needs a probe to exist before step
4a's full ablation does. This module is the core step 4a reuses, not a
throwaway: :func:`ridge_probe` fits on one window and scores causally on a
later one with the SAME alpha-selection discipline §13.1 asks for (grid
search, validation-only, the same grid for every variant — "unequal tuning
invalidates the ablation").

What step 4a adds on top of this is breadth, not a different mechanism:
every one of the eight (plus diagnostic) state variants, every forward
target, and a walk-forward refit schedule rather than the single
train/apply split step 3 uses for its narrower V2-vs-baselines question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from prism.analysis.bootstrap import BootstrapResult, non_overlapping, stationary_block_bootstrap_ci

__all__ = ["ProbeResult", "select_ridge_alpha", "ridge_probe", "ProbeComparison", "compare_probes"]


@dataclass(frozen=True)
class ProbeResult:
    """One feature set's out-of-sample probe outcome against one target."""

    predictions: pd.Series
    actuals: pd.Series
    r2: float
    alpha: float
    n_train: int
    n_test: int

    @property
    def residuals(self) -> pd.Series:
        return self.actuals - self.predictions

    def as_dict(self) -> dict[str, Any]:
        return {
            "r2": self.r2, "alpha": self.alpha, "n_train": self.n_train, "n_test": self.n_test,
        }


def select_ridge_alpha(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, alpha_grid: list[float],
) -> float:
    """The alpha in ``alpha_grid`` with the lowest validation MSE.

    A plain train/val split, not cross-validation: the data is a time
    series, and k-fold CV would shuffle future folds into the selection of
    a model applied to the past. Spec §13.1: "Probe hyperparameters (ridge
    α) tuned on validation folds only, with the same grid for every
    variant."
    """
    best_alpha, best_mse = alpha_grid[0], np.inf
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        mse = float(np.mean((model.predict(X_val) - y_val) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha


def ridge_probe(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    val_start: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    alpha_grid: list[float],
) -> ProbeResult:
    """Fit on train, select alpha on val, score (causally) on test.

    ``features`` and ``target`` must share an index; rows with a NaN in
    either are dropped — a target's trailing NaN (no future left, spec
    §5.4) must never be imputed, and dropping is how that's enforced here
    rather than trusted to the caller.

    The predictor is standardised using **train-only** moments (spec
    §5.5's rule applies here identically to how it applies to feature
    construction): fitting a scaler on train+val+test would leak the
    future into the coefficients exactly as fitting the ridge itself on
    future data would.
    """
    common = features.index.intersection(target.index)
    frame = features.loc[common].copy()
    frame["__target__"] = target.loc[common]
    frame = frame.dropna()

    def _slice(start, end) -> pd.DataFrame:
        return frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    train, val, test = _slice(train_start, train_end), _slice(val_start, val_end), _slice(test_start, test_end)
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise ValueError(f"{name} slice is empty after dropping NaN rows")

    feature_cols = [c for c in frame.columns if c != "__target__"]
    mean, std = train[feature_cols].mean(), train[feature_cols].std(ddof=0).replace(0.0, 1.0)

    def _xy(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = ((part[feature_cols] - mean) / std).to_numpy(dtype="float64")
        y = part["__target__"].to_numpy(dtype="float64")
        return x, y

    X_train, y_train = _xy(train)
    X_val, y_val = _xy(val)
    X_test, y_test = _xy(test)

    alpha = select_ridge_alpha(X_train, y_train, X_val, y_val, alpha_grid)
    model = Ridge(alpha=alpha)
    model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    predictions = model.predict(X_test)
    ss_res = float(np.sum((y_test - predictions) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return ProbeResult(
        predictions=pd.Series(predictions, index=test.index, name="prediction"),
        actuals=pd.Series(y_test, index=test.index, name="actual"),
        r2=r2, alpha=alpha, n_train=len(train) + len(val), n_test=len(test),
    )


@dataclass(frozen=True)
class ProbeComparison:
    """Two feature sets' probe outcomes on the same target, with bootstrap CIs
    on mean squared error (lower is better) — the §13.1/§10 gate test."""

    name_a: str
    name_b: str
    result_a: ProbeResult
    result_b: ProbeResult
    mse_ci_a: BootstrapResult
    mse_ci_b: BootstrapResult

    def a_beats_b(self) -> bool:
        """``a``'s CI does not overlap ``b``'s AND ``a``'s MSE point estimate is lower.

        Non-overlap alone does not say which direction the gap runs;
        checking the point estimate too is what makes this a one-line
        answer to "does A beat B", not just "are A and B different".
        """
        return non_overlapping(self.mse_ci_a, self.mse_ci_b) and (
            self.mse_ci_a.point_estimate < self.mse_ci_b.point_estimate
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name_a": self.name_a, "name_b": self.name_b,
            "r2_a": self.result_a.r2, "r2_b": self.result_b.r2,
            "mse_a": self.mse_ci_a.as_dict(), "mse_b": self.mse_ci_b.as_dict(),
            "a_beats_b": self.a_beats_b(),
        }


def compare_probes(
    result_a: ProbeResult,
    result_b: ProbeResult,
    *,
    name_a: str,
    name_b: str,
    block_length: int = 20,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> ProbeComparison:
    """Bootstrap-CI comparison of two probe results' mean squared error.

    Requires ``result_a`` and ``result_b`` to have been scored on the SAME
    test dates (same target, same split) — otherwise a lower MSE could
    simply mean an easier test period, not a better feature set. Checked,
    not assumed.
    """
    if not result_a.residuals.index.equals(result_b.residuals.index):
        raise ValueError(
            "result_a and result_b must be scored on identical test dates for "
            "the comparison to isolate the feature set rather than the period"
        )
    mse_a = stationary_block_bootstrap_ci(
        result_a.residuals.to_numpy() ** 2, np.mean,
        block_length=block_length, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed,
    )
    mse_b = stationary_block_bootstrap_ci(
        result_b.residuals.to_numpy() ** 2, np.mean,
        block_length=block_length, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed + 1,
    )
    return ProbeComparison(
        name_a=name_a, name_b=name_b, result_a=result_a, result_b=result_b,
        mse_ci_a=mse_a, mse_ci_b=mse_b,
    )
