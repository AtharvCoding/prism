"""Tier 1 probe procedure, exactly as pre-registered. Spec §13.1; preregistration §3-§6.

``prism.probes.probe`` (step 3) is a lean single train/val/test split built to
answer one narrow question. This module is the pre-registered Tier 1 procedure:

* one **common evaluation index** for every variant (fit AND score);
* **annual expanding walk-forward** refits with a 25-session embargo;
* ridge ``alpha`` chosen **once** per (variant, target) on validation folds
  (apply years 2015-2018, all inside train+validation) from ONE shared grid,
  then held fixed through every test fold;
* OOS R-squared against the fold's own historical-mean forecast;
* paired loss differentials on **shared bootstrap paths** (common random
  numbers) and the pre-registered gate aggregation.

Ridge is solved in closed form through one SVD of the standardised fit matrix
per fold, which yields every alpha on the grid (and every target) for the price
of a single decomposition. :func:`ridge_fold_predictions` is checked against
``sklearn.linear_model.Ridge`` in the test suite.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from prism.analysis.bootstrap import path_ci
from prism.splits import WalkForwardFold, expanding_folds

__all__ = [
    "PRIMARY_RISK_TARGETS",
    "SECONDARY_TARGETS",
    "ALL_TARGETS",
    "common_evaluation_index",
    "ridge_fold_predictions",
    "make_folds",
    "ValidationResult",
    "VariantProbe",
    "select_alphas",
    "predict_test_walkforward",
    "paired_comparison_table",
    "aggregate_comparison",
    "evaluate_gates",
    "GATES",
    "variant_metric_table",
    "LogisticProbe",
    "run_logistic_probe",
]

PRIMARY_RISK_TARGETS: tuple[str, ...] = (
    "fwd_vol_5", "fwd_vol_20", "fwd_max_drawdown_20", "fwd_corr_20",
)
SECONDARY_TARGETS: tuple[str, ...] = ("fwd_ret_20",)
ALL_TARGETS: tuple[str, ...] = PRIMARY_RISK_TARGETS + SECONDARY_TARGETS

#: Pre-registration §4. Each gate passes iff BOTH its comparisons pass.
GATES: dict[str, list[tuple[str, str]]] = {
    "lstm_adds_value": [("V2", "V1p"), ("V2", "C1")],
    "hmm_adds_value": [("V3", "C2"), ("V3", "C3")],
    "research_question": [("V4", "V2")],
}

#: Minimum number of primary risk targets on which a comparison must be favourable.
MIN_FAVOURABLE = 3


def common_evaluation_index(
    states: dict[str, pd.DataFrame], targets: pd.DataFrame, target_names: Iterable[str] = ALL_TARGETS
) -> pd.DatetimeIndex:
    """Intersection of every state frame's dates, restricted to rows where every target exists.

    Pre-registration §5: V1' has fewer sessions than the other variants (its
    first ``window - 1`` have no full window). Every variant is fitted AND
    scored on the intersection, so no variant gains or loses days.
    """
    idx: pd.DatetimeIndex | None = None
    for name, frame in states.items():
        idx = pd.DatetimeIndex(frame.index) if idx is None else idx.intersection(frame.index)
    assert idx is not None
    tgt = targets.loc[:, list(target_names)]
    ok = tgt.notna().all(axis=1) & np.isfinite(tgt.to_numpy(dtype="float64")).all(axis=1)
    idx = idx.intersection(tgt.index[ok.to_numpy()])
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()
    return idx


def ridge_fold_predictions(
    X_fit: np.ndarray, Y_fit: np.ndarray, X_apply: np.ndarray, alphas: Iterable[float]
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """Ridge-with-intercept predictions for every alpha and every target column, via one SVD.

    Features are standardised with the FIT window's moments only (zero-spread
    columns are left unscaled). The intercept is the fit window's target mean,
    which is exactly what ``Ridge(fit_intercept=True)`` returns once X is
    centred. Returns ``({alpha: predictions (n_apply, n_targets)}, ybar)``.
    """
    mean = X_fit.mean(axis=0)
    std = X_fit.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = (X_fit - mean) / std
    ybar = Y_fit.mean(axis=0)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    UtY = U.T @ (Y_fit - ybar)
    ZaV = ((X_apply - mean) / std) @ Vt.T
    out: dict[float, np.ndarray] = {}
    for alpha in alphas:
        shrink = (s / (s**2 + alpha))[:, None]
        out[float(alpha)] = ZaV @ (shrink * UtY) + ybar
    return out, ybar


def make_folds(
    index: pd.DatetimeIndex, first_apply_start: str | pd.Timestamp, apply_end: str | pd.Timestamp,
    embargo_days: int,
) -> list[WalkForwardFold]:
    """Annual expanding folds on the common index's calendar, embargoed (pre-registration §5)."""
    return expanding_folds(
        index[0], first_apply_start, apply_end, "annual", embargo_days=embargo_days,
    )


def _masks(index: pd.DatetimeIndex, fold: WalkForwardFold) -> tuple[np.ndarray, np.ndarray]:
    fit = (index >= fold.fit_start) & (index <= fold.fit_end)
    apply = (index >= fold.apply_start) & (index <= fold.apply_end)
    return np.asarray(fit), np.asarray(apply)


@dataclass
class ValidationResult:
    """Phase 1 output: alpha per target, chosen on validation folds only. No test row is touched."""

    name: str
    target_names: tuple[str, ...]
    alpha: dict[str, float]
    #: Pooled validation MSE, rows = alpha grid, columns = targets.
    validation_mse: pd.DataFrame
    #: Validation-fold predictions at the selected alpha (used only for gamma calibration).
    validation_predictions: pd.DataFrame
    n_features: int
    alpha_at_edge: dict[str, str] = field(default_factory=dict)


@dataclass
class VariantProbe:
    """One variant's probe outcome across all five targets."""

    name: str
    target_names: tuple[str, ...]
    alpha: dict[str, float]
    validation_mse: pd.DataFrame
    #: Test-split out-of-sample predictions (rows = test dates, columns = targets).
    predictions: pd.DataFrame
    #: Per-row fit-window mean of each target (the historical-mean benchmark forecast).
    fit_mean: pd.DataFrame
    validation_predictions: pd.DataFrame
    n_features: int
    alpha_at_edge: dict[str, str] = field(default_factory=dict)


def _matrices(frame, targets, common_index, target_names):
    X = frame.loc[common_index].to_numpy(dtype="float64")
    if not np.isfinite(X).all():
        raise ValueError("non-finite values in the state frame on the common index")
    Y = targets.loc[common_index, list(target_names)].to_numpy(dtype="float64")
    return X, Y


def select_alphas(
    name: str,
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    common_index: pd.DatetimeIndex,
    val_folds: list[WalkForwardFold],
    alpha_grid: list[float],
    target_names: tuple[str, ...] = ALL_TARGETS,
) -> ValidationResult:
    """Phase 1. Alpha per target = the grid value with the lowest POOLED validation MSE; ties go to the larger alpha."""
    grid = sorted(float(a) for a in alpha_grid)
    X, Y = _matrices(frame, targets, common_index, target_names)
    n_targets = len(target_names)
    sse = {a: np.zeros(n_targets) for a in grid}
    count = 0
    val_rows: list[pd.DatetimeIndex] = []
    val_pred_chunks: dict[float, list[np.ndarray]] = {a: [] for a in grid}
    for fold in val_folds:
        fit_m, apply_m = _masks(common_index, fold)
        if not fit_m.any() or not apply_m.any():
            raise ValueError(f"validation fold {fold.index}: empty fit or apply window")
        preds, _ = ridge_fold_predictions(X[fit_m], Y[fit_m], X[apply_m], grid)
        for a in grid:
            sse[a] += ((preds[a] - Y[apply_m]) ** 2).sum(axis=0)
            val_pred_chunks[a].append(preds[a])
        count += int(apply_m.sum())
        val_rows.append(common_index[apply_m])
    val_index = val_rows[0].append(val_rows[1:]) if len(val_rows) > 1 else val_rows[0]
    mse_table = pd.DataFrame({a: sse[a] / count for a in grid}, index=list(target_names)).T
    mse_table.index.name = "alpha"

    alpha_star: dict[str, float] = {}
    at_edge: dict[str, str] = {}
    for j, tgt in enumerate(target_names):
        best_a, best_mse = grid[0], np.inf
        for a in grid:
            m = sse[a][j] / count
            if m <= best_mse:
                best_a, best_mse = a, m
        alpha_star[tgt] = best_a
        if best_a == grid[0]:
            at_edge[tgt] = "lower"
        elif best_a == grid[-1]:
            at_edge[tgt] = "upper"
    val_pred = np.column_stack(
        [np.concatenate(val_pred_chunks[alpha_star[t]], axis=0)[:, j] for j, t in enumerate(target_names)]
    )
    return ValidationResult(
        name=name, target_names=tuple(target_names), alpha=alpha_star, validation_mse=mse_table,
        validation_predictions=pd.DataFrame(val_pred, index=val_index, columns=list(target_names)),
        n_features=X.shape[1], alpha_at_edge=at_edge,
    )


def predict_test_walkforward(
    validation: ValidationResult,
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    common_index: pd.DatetimeIndex,
    test_folds: list[WalkForwardFold],
) -> VariantProbe:
    """Phase 2. Annual walk-forward over the test split with alpha FROZEN from phase 1."""
    target_names = validation.target_names
    X, Y = _matrices(frame, targets, common_index, target_names)
    needed = sorted(set(validation.alpha.values()))
    pred_chunks, ybar_chunks, row_chunks = [], [], []
    for fold in test_folds:
        fit_m, apply_m = _masks(common_index, fold)
        if not fit_m.any() or not apply_m.any():
            raise ValueError(f"test fold {fold.index}: empty fit or apply window")
        preds, ybar = ridge_fold_predictions(X[fit_m], Y[fit_m], X[apply_m], needed)
        pred_chunks.append(
            np.column_stack([preds[validation.alpha[t]][:, j] for j, t in enumerate(target_names)])
        )
        ybar_chunks.append(np.tile(ybar, (int(apply_m.sum()), 1)))
        row_chunks.append(common_index[apply_m])
    rows = row_chunks[0].append(row_chunks[1:]) if len(row_chunks) > 1 else row_chunks[0]
    if not rows.is_unique or not rows.is_monotonic_increasing:
        raise AssertionError(f"variant {validation.name}: test folds do not tile the test dates exactly once")
    return VariantProbe(
        name=validation.name, target_names=tuple(target_names), alpha=validation.alpha,
        validation_mse=validation.validation_mse,
        predictions=pd.DataFrame(np.vstack(pred_chunks), index=rows, columns=list(target_names)),
        fit_mean=pd.DataFrame(np.vstack(ybar_chunks), index=rows, columns=list(target_names)),
        validation_predictions=validation.validation_predictions, n_features=validation.n_features,
        alpha_at_edge=validation.alpha_at_edge,
    )


# --------------------------------------------------------------------------- #
# metrics and paired differences
# --------------------------------------------------------------------------- #
def _verdict(lo: float, hi: float) -> str:
    if hi < 0:
        return "favourable"
    if lo > 0:
        return "adverse"
    return "indeterminate"


def variant_metric_table(
    probes: dict[str, VariantProbe], y_test: pd.DataFrame, paths: np.ndarray, *, ci_level: float = 0.95,
) -> pd.DataFrame:
    """OOS R-squared and MSE with CIs from the shared bootstrap paths (pre-registration §5, §6)."""
    rows = []
    for name, probe in probes.items():
        for tgt in probe.target_names:
            y = y_test[tgt].to_numpy()
            e2 = (y - probe.predictions[tgt].to_numpy()) ** 2
            s2 = (y - probe.fit_mean[tgt].to_numpy()) ** 2
            mse, mse_lo, mse_hi = path_ci(e2, paths, ci_level=ci_level)
            num, den = e2[paths].sum(axis=1), s2[paths].sum(axis=1)
            r2_rep = 1.0 - num / den
            alpha = 1.0 - ci_level
            r2_lo, r2_hi = np.quantile(r2_rep, [alpha / 2, 1 - alpha / 2])
            rows.append({
                "variant": name, "target": tgt, "n_test": len(y),
                "oos_r2": 1.0 - e2.sum() / s2.sum(), "r2_lo": float(r2_lo), "r2_hi": float(r2_hi),
                "mse": mse, "mse_lo": mse_lo, "mse_hi": mse_hi,
                "alpha": probe.alpha[tgt], "alpha_at_edge": probe.alpha_at_edge.get(tgt, ""),
                "n_features": probe.n_features,
            })
    return pd.DataFrame(rows)


def paired_comparison_table(
    probes: dict[str, VariantProbe], y_test: pd.DataFrame, comparisons: list[tuple[str, str]],
    paths: np.ndarray, *, ci_level: float = 0.95,
) -> pd.DataFrame:
    """For every (X, Y) and target: the paired loss-differential CI and the spec's non-overlap version.

    ``d_t = e^2(X,t) - e^2(Y,t)``; negative means X forecasts better. Both
    variants are resampled on the SAME days (``paths``).
    """
    rows = []
    for x, y_name in comparisons:
        for tgt in probes[x].target_names:
            y = y_test[tgt].to_numpy()
            ex2 = (y - probes[x].predictions[tgt].to_numpy()) ** 2
            ey2 = (y - probes[y_name].predictions[tgt].to_numpy()) ** 2
            s2 = (y - probes[x].fit_mean[tgt].to_numpy()) ** 2
            d = ex2 - ey2
            mean_d, lo, hi = path_ci(d, paths, ci_level=ci_level)
            x_mse, x_lo, x_hi = path_ci(ex2, paths, ci_level=ci_level)
            y_mse, y_lo, y_hi = path_ci(ey2, paths, ci_level=ci_level)
            if x_hi < y_lo:
                spec = "favourable"
            elif x_lo > y_hi:
                spec = "adverse"
            else:
                spec = "indeterminate"
            denom = float(s2.mean())
            rows.append({
                "x": x, "y": y_name, "target": tgt,
                "mean_d": mean_d, "ci_lo": lo, "ci_hi": hi, "verdict": _verdict(lo, hi),
                "delta_r2": -mean_d / denom, "delta_r2_lo": -hi / denom, "delta_r2_hi": -lo / denom,
                "x_mse": x_mse, "x_mse_lo": x_lo, "x_mse_hi": x_hi,
                "y_mse": y_mse, "y_mse_lo": y_lo, "y_mse_hi": y_hi,
                "spec_verdict": spec, "primary": tgt in PRIMARY_RISK_TARGETS,
            })
    return pd.DataFrame(rows)


def aggregate_comparison(table: pd.DataFrame, x: str, y: str, *, column: str = "verdict") -> dict[str, object]:
    """Pre-registration §4: pass iff favourable on >= 3 of the 4 primary risk targets AND adverse on none."""
    sub = table[(table["x"] == x) & (table["y"] == y) & table["primary"]]
    if len(sub) != len(PRIMARY_RISK_TARGETS):
        raise ValueError(f"expected {len(PRIMARY_RISK_TARGETS)} primary targets for {x} vs {y}, got {len(sub)}")
    n_fav = int((sub[column] == "favourable").sum())
    n_adv = int((sub[column] == "adverse").sum())
    return {
        "x": x, "y": y, "n_favourable": n_fav, "n_adverse": n_adv,
        "n_indeterminate": len(sub) - n_fav - n_adv,
        "passes": n_fav >= MIN_FAVOURABLE and n_adv == 0,
        "per_target": dict(zip(sub["target"], sub[column])),
    }


def evaluate_gates(table: pd.DataFrame, *, column: str = "verdict") -> dict[str, dict[str, object]]:
    """Each gate passes iff every one of its comparisons passes."""
    out: dict[str, dict[str, object]] = {}
    for gate, comps in GATES.items():
        parts = [aggregate_comparison(table, x, y, column=column) for x, y in comps]
        out[gate] = {"passes": all(p["passes"] for p in parts), "comparisons": parts}
    return out


# --------------------------------------------------------------------------- #
# secondary: logistic AUC
# --------------------------------------------------------------------------- #
@dataclass
class LogisticProbe:
    name: str
    alpha: float
    auc: float
    auc_lo: float
    auc_hi: float
    n_test: int
    n_convergence_warnings: int


def run_logistic_probe(
    name: str, frame: pd.DataFrame, label: pd.Series, common_index: pd.DatetimeIndex,
    val_folds: list[WalkForwardFold], test_folds: list[WalkForwardFold], alpha_grid: list[float],
    paths: np.ndarray, *, ci_level: float = 0.95, max_iter: int = 1000,
) -> LogisticProbe:
    """Secondary descriptive AUC (pre-registration §3): L2 logistic, ``C = 1/alpha``, alpha by validation log-loss."""
    grid = sorted(float(a) for a in alpha_grid)
    X = frame.loc[common_index].to_numpy(dtype="float64")
    y = label.loc[common_index].to_numpy(dtype="int64")
    n_warn = 0

    def _fit(Xf, yf, a):
        nonlocal n_warn
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LogisticRegression(C=1.0 / a, max_iter=max_iter)
            model.fit(Xf, yf)
        n_warn += sum(issubclass(w.category, Warning) and "converge" in str(w.message).lower() for w in caught)
        return model

    def _std(X_fit, X_apply):
        mean, std = X_fit.mean(axis=0), X_fit.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        return (X_fit - mean) / std, (X_apply - mean) / std

    logloss = {a: 0.0 for a in grid}
    n_val = 0
    for fold in val_folds:
        fit_m, apply_m = _masks(common_index, fold)
        Zf, Za = _std(X[fit_m], X[apply_m])
        for a in grid:
            p = np.clip(_fit(Zf, y[fit_m], a).predict_proba(Za)[:, 1], 1e-12, 1 - 1e-12)
            ya = y[apply_m]
            logloss[a] += float(-(ya * np.log(p) + (1 - ya) * np.log(1 - p)).sum())
        n_val += int(apply_m.sum())
    best_a, best = grid[0], np.inf
    for a in grid:
        if logloss[a] / n_val <= best:
            best_a, best = a, logloss[a] / n_val

    scores, labels = [], []
    for fold in test_folds:
        fit_m, apply_m = _masks(common_index, fold)
        Zf, Za = _std(X[fit_m], X[apply_m])
        scores.append(_fit(Zf, y[fit_m], best_a).decision_function(Za))
        labels.append(y[apply_m])
    s, l = np.concatenate(scores), np.concatenate(labels)
    auc = float(roc_auc_score(l, s))
    reps = np.array([roc_auc_score(l[p], s[p]) if len(np.unique(l[p])) == 2 else np.nan for p in paths])
    alpha_q = 1.0 - ci_level
    lo, hi = np.nanquantile(reps, [alpha_q / 2, 1 - alpha_q / 2])
    return LogisticProbe(name, best_a, auc, float(lo), float(hi), len(l), n_warn)
