"""Winsorising, scaling and correlation pruning — all fitted, all train-only.

Spec §5.5. These three operations are deliberately kept *out* of
:func:`prism.features.build.build_features`, for a reason that is easy to
miss: they are the only parts of feature construction that *learn* something
from the data.

If winsorisation quantiles or scaler moments were computed inside the feature
builder, the builder would no longer be truncation-invariant — rebuilding on
data up to ``t`` would produce different values for rows before ``t``, because
the quantiles would differ. The causality test would catch it, and the fix
would be exactly this separation. So the separation is the design, not a
patch: ``build_features`` is a pure causal transform, and everything fitted
lives here behind :class:`~prism.fitting.FittedArtifact`, where fit scope is
enforced and recorded.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from prism.fitting import FittedArtifact
from prism.utils.logging import get_logger

__all__ = ["Winsoriser", "FeatureScaler", "CorrelationPruner", "FEATURE_FAMILIES"]

_log = get_logger(__name__)

#: Families within which correlation pruning is applied (spec §5.5). Pruning
#: only ever happens *within* a family: ``vol_20`` and ``mom_20`` may well be
#: correlated in a given sample without either being redundant.
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "volatility": ("vol_20", "vol_60", "downside_vol_20", "volume_ratio"),
    "momentum": ("mom_20", "mom_60"),
    "shape": ("skew_60", "kurt_60"),
    "correlation": ("avg_pairwise_corr_60", "corr_dispersion_60", "first_eigenvalue_share_60"),
}


def _family_of(column: str) -> str | None:
    """Family a column belongs to, matched on its feature suffix."""
    for family, suffixes in FEATURE_FAMILIES.items():
        for suffix in suffixes:
            if column == suffix or column.endswith(f"_{suffix}"):
                return family
    return None


class Winsoriser(FittedArtifact):
    """Clip to training-set quantiles. Spec §5.5.

    The quantiles are learned on the fit window and applied unchanged to every
    split. Recomputing them per split would let the test period define its own
    idea of "extreme", which both leaks and defeats the purpose: the point is
    to bound the influence of values the *training* data considered outliers.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        super().__init__()
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("require 0 <= lower < upper <= 1")
        self.lower = lower
        self.upper = upper
        self._lo: pd.Series | None = None
        self._hi: pd.Series | None = None

    def _fit(self, frame: pd.DataFrame) -> None:
        numeric = frame.select_dtypes(include=[np.number])
        self._lo = numeric.quantile(self.lower)
        self._hi = numeric.quantile(self.upper)

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self._lo is not None and self._hi is not None
        missing = [c for c in self._lo.index if c not in frame.columns]
        if missing:
            raise KeyError(f"Winsoriser was fitted on columns absent here: {missing[:8]}")
        out = frame.copy()
        cols = list(self._lo.index)
        out[cols] = out[cols].clip(lower=self._lo, upper=self._hi, axis=1)
        return out

    def _params(self) -> dict[str, Any]:
        return {"lower_q": self.lower, "upper_q": self.upper, "lo": self._lo, "hi": self._hi}


class FeatureScaler(FittedArtifact):
    """Standardise features using fit-window moments only. Spec §5.5.

    ``kind="robust"`` uses the median and IQR, which is the safer default for
    a panel containing 2008 and 2020: a standard deviation computed over a
    window containing the GFC is itself a crisis-contaminated statistic.
    """

    def __init__(self, kind: Literal["standard", "robust"] = "standard") -> None:
        super().__init__()
        self.kind = kind
        self._centre: pd.Series | None = None
        self._scale: pd.Series | None = None

    def _fit(self, frame: pd.DataFrame) -> None:
        numeric = frame.select_dtypes(include=[np.number])
        if self.kind == "standard":
            centre = numeric.mean()
            scale = numeric.std(ddof=0)
        else:
            centre = numeric.median()
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            scale = (q3 - q1) / 1.349  # IQR -> sigma for a normal
        # A zero scale means a constant column, which the QA `constant_feature`
        # hard check should already have rejected. Guard anyway rather than
        # emitting inf, and say which columns.
        degenerate = sorted(scale.index[~(scale > 0)].astype(str))
        if degenerate:
            _log.warning(
                "%d columns have zero spread on the fit window and are left uncentred: %s",
                len(degenerate),
                degenerate[:8],
            )
        self._centre = centre
        self._scale = scale.where(scale > 0, 1.0)

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self._centre is not None and self._scale is not None
        missing = [c for c in self._centre.index if c not in frame.columns]
        if missing:
            raise KeyError(f"FeatureScaler was fitted on columns absent here: {missing[:8]}")
        out = frame.copy()
        cols = list(self._centre.index)
        out[cols] = (out[cols] - self._centre[cols]) / self._scale[cols]
        return out

    def _params(self) -> dict[str, Any]:
        return {"kind": self.kind, "centre": self._centre, "scale": self._scale}


class CorrelationPruner(FittedArtifact):
    """Drop within-family features correlated above ``threshold``. Spec §5.5.

    Which member survives is decided deterministically — by the declared
    order in :data:`FEATURE_FAMILIES`, earlier suffixes winning — so the
    surviving column set does not depend on dict iteration order, the sample,
    or the platform. :attr:`dropped` records every decision and its
    correlation, for the report.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        super().__init__()
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self._keep: list[str] | None = None
        self.dropped: list[dict[str, Any]] = []

    def _fit(self, frame: pd.DataFrame) -> None:
        numeric = frame.select_dtypes(include=[np.number])
        keep = list(numeric.columns)
        dropped: list[dict[str, Any]] = []

        # Group by (asset prefix, family) so XLK's vol family is pruned
        # independently of XLE's — they are different series.
        groups: dict[tuple[str, str], list[str]] = {}
        for col in numeric.columns:
            family = _family_of(col)
            if family is None:
                continue
            suffixes = FEATURE_FAMILIES[family]
            suffix = next(
                (s for s in suffixes if col == s or col.endswith(f"_{s}")), None
            )
            prefix = col[: -len(suffix) - 1] if suffix and col != suffix else ""
            groups.setdefault((prefix, family), []).append(col)

        for (prefix, family), members in sorted(groups.items()):
            if len(members) < 2:
                continue
            order = FEATURE_FAMILIES[family]

            def rank(col: str, _order: tuple[str, ...] = order) -> int:
                for i, suffix in enumerate(_order):
                    if col == suffix or col.endswith(f"_{suffix}"):
                        return i
                return len(_order)

            ordered = sorted(members, key=lambda c: (rank(c), c))
            corr = numeric[ordered].corr()
            survivors: list[str] = []
            for col in ordered:
                clash = next(
                    (
                        s
                        for s in survivors
                        if np.isfinite(corr.loc[col, s])
                        and abs(corr.loc[col, s]) > self.threshold
                    ),
                    None,
                )
                if clash is None:
                    survivors.append(col)
                else:
                    keep.remove(col)
                    dropped.append(
                        {
                            "dropped": col,
                            "kept": clash,
                            "family": family,
                            "group": prefix or "(global)",
                            "abs_corr": float(abs(corr.loc[col, clash])),
                        }
                    )
        self._keep = keep
        self.dropped = dropped
        if dropped:
            _log.info(
                "correlation pruning dropped %d of %d columns at |rho| > %.2f",
                len(dropped),
                numeric.shape[1],
                self.threshold,
            )

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self._keep is not None
        missing = [c for c in self._keep if c not in frame.columns]
        if missing:
            raise KeyError(f"CorrelationPruner kept columns absent here: {missing[:8]}")
        non_numeric = [c for c in frame.columns if c not in frame.select_dtypes(include=[np.number]).columns]
        return frame[[*self._keep, *non_numeric]]

    @property
    def kept(self) -> list[str]:
        assert self._keep is not None, "pruner is not fitted"
        return list(self._keep)

    def _params(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "keep": self._keep}

    def report_rows(self) -> pd.DataFrame:
        """Pruning decisions as a frame, for ``reports/tables``."""
        if not self.dropped:
            return pd.DataFrame(columns=["dropped", "kept", "family", "group", "abs_corr"])
        return pd.DataFrame(self.dropped).sort_values(
            ["family", "group", "abs_corr"], ascending=[True, True, False]
        )
