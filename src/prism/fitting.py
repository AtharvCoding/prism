"""Fit-scope bookkeeping for every fitted object in the project.

Spec §0.4.2 and §7.2. "Anything *fitted* (scalers, HMM parameters, encoder
weights, policies) sees only training data." That is a property of the code,
and properties of code are enforced by tests, not by reading it. For a test to
check it, every fitted object must be able to answer three questions:

* **What data did you see?** :attr:`FittedArtifact.fit_record` carries the
  first and last date, the row count, the column set, and a hash of the
  actual values.
* **Are your parameters what they were?** :meth:`FittedArtifact.params_hash`
  digests the fitted parameters, so a test can mutate data after
  ``train_end``, refit, and assert the hash is unchanged.
* **Did you refit when you should have transformed?** ``transform`` raises
  if called before ``fit``, and ``fit`` on an already-fitted object raises
  unless ``refit=True`` is passed explicitly. The reference rolling HMM
  mutated a shared scaler in place on every fold (defect B5); that cannot be
  expressed through this interface by accident.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd

from prism.utils.hashing import hash_object

__all__ = ["FitRecord", "FittedArtifact", "NotFittedError", "AlreadyFittedError"]


class NotFittedError(RuntimeError):
    """Raised when ``transform`` is called before ``fit``."""


class AlreadyFittedError(RuntimeError):
    """Raised when ``fit`` would silently discard an existing fit."""


@dataclass(frozen=True)
class FitRecord:
    """Provenance of one fit: exactly which rows and columns were seen."""

    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int
    columns: tuple[str, ...]
    #: SHA-256 of the fitted-on values. Two fits over the same rows produce
    #: the same digest; a single changed value changes it.
    data_hash: str
    #: Free-form label, e.g. ``"fold 3"`` or ``"train"``.
    scope: str = ""

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, scope: str = "") -> FitRecord:
        if len(frame) == 0:
            raise ValueError("cannot record a fit over an empty frame")
        idx = pd.DatetimeIndex(frame.index)
        values = np.ascontiguousarray(frame.to_numpy(dtype="float64"))
        return cls(
            start=idx[0],
            end=idx[-1],
            n_rows=len(frame),
            columns=tuple(str(c) for c in frame.columns),
            data_hash=hash_object(
                {
                    "values": np.round(values, 12).tobytes().hex(),
                    "index": [str(d) for d in idx],
                    "columns": [str(c) for c in frame.columns],
                }
            ),
            scope=scope,
        )

    def assert_within(self, limit: pd.Timestamp, *, what: str = "fit") -> None:
        """Assert this fit ended on or before ``limit``.

        Raises ``AssertionError`` so it reads naturally inside a test, and
        names both dates so the failure is actionable.
        """
        limit_ts = pd.Timestamp(limit)
        assert self.end <= limit_ts, (
            f"{what} saw data through {self.end.date()}, which is after the permitted "
            f"boundary {limit_ts.date()} (spec §0.4.2)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "n_rows": self.n_rows,
            "n_columns": len(self.columns),
            "data_hash": self.data_hash,
            "scope": self.scope,
        }


class FittedArtifact(ABC):
    """Base class for anything that learns parameters from data.

    Subclasses implement :meth:`_fit` and :meth:`_transform` and expose their
    learned parameters through :meth:`_params`. The public ``fit`` / 
    ``transform`` wrappers handle the bookkeeping so no subclass can forget it.
    """

    def __init__(self) -> None:
        self._fit_record: FitRecord | None = None

    # -- public API -------------------------------------------------------- #
    @property
    def fit_record(self) -> FitRecord:
        if self._fit_record is None:
            raise NotFittedError(f"{type(self).__name__} has not been fitted")
        return self._fit_record

    @property
    def is_fitted(self) -> bool:
        return self._fit_record is not None

    def fit(self, frame: pd.DataFrame, *, scope: str = "", refit: bool = False) -> Self:
        """Learn parameters from ``frame`` and record what was seen.

        ``refit=True`` is required to re-fit an already-fitted object. Walk-
        forward refits are legitimate (§0.4.2) but must construct a *new*
        object per fold, or pass ``refit=True`` deliberately — never mutate a
        shared instance implicitly.
        """
        if self.is_fitted and not refit:
            raise AlreadyFittedError(
                f"{type(self).__name__} is already fitted on "
                f"{self._fit_record.start.date()}..{self._fit_record.end.date()}. "  # type: ignore[union-attr]
                "Construct a new instance per walk-forward fold, or pass refit=True "
                "if replacing the fit is genuinely intended (spec §7.2)."
            )
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("fit requires a DatetimeIndex so the fit range can be recorded")
        if not frame.index.is_monotonic_increasing:
            raise ValueError("fit requires a sorted index")
        self._fit(frame)
        self._fit_record = FitRecord.from_frame(frame, scope=scope)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned parameters. Never updates them."""
        if not self.is_fitted:
            raise NotFittedError(
                f"{type(self).__name__}.transform called before fit. There is no "
                "fit_transform on this class by design: a single call that does both "
                "is how full-sample scalers get written."
            )
        before = self.params_hash()
        out = self._transform(frame)
        after = self.params_hash()
        if before != after:
            raise RuntimeError(
                f"{type(self).__name__}.transform mutated its own parameters "
                f"({before[:12]} -> {after[:12]}). transform must be pure (spec §7.2)."
            )
        return out

    def params_hash(self) -> str:
        """Stable digest of the learned parameters."""
        return hash_object(_digestible(self._params()))

    # -- subclass hooks ---------------------------------------------------- #
    @abstractmethod
    def _fit(self, frame: pd.DataFrame) -> None: ...

    @abstractmethod
    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def _params(self) -> dict[str, Any]:
        """The learned parameters, as a JSON-digestible mapping."""


def _digestible(obj: Any) -> Any:
    """Convert numpy/pandas objects into something ``hash_object`` can digest."""
    if isinstance(obj, dict):
        return {str(k): _digestible(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_digestible(v) for v in obj]
    if isinstance(obj, pd.Series):
        return {"index": [str(i) for i in obj.index], "values": _digestible(obj.to_numpy())}
    if isinstance(obj, pd.DataFrame):
        return {
            "index": [str(i) for i in obj.index],
            "columns": [str(c) for c in obj.columns],
            "values": _digestible(obj.to_numpy()),
        }
    if isinstance(obj, np.ndarray):
        # Round before digesting so that a change below float64's meaningful
        # precision does not register as a different parameter set.
        return np.round(np.asarray(obj, dtype="float64"), 12).tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
