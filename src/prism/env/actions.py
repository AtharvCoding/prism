"""Action -> portfolio weights. Spec §11, DECISIONS.md D-001 and D-035.

The agent emits ``a`` in ``[-1, 1]^(n+1)`` (``n`` risky assets, then cash). The
weights are ``project(softmax(logit_scale * a))``, where ``project`` is the
Euclidean projection onto

    { w : 0 <= w_i <= cap for risky i,  0 <= w_cash <= 1,  sum w = 1 }

The cap applies to the risky assets and not to cash, the same interpretation as
the Tier 1 allocator (``prism.probes.allocator`` module docstring): a policy
must be able to hold mostly cash, and with ``n`` risky assets at ``cap`` the
set is non-empty whenever ``n * cap + 1 >= 1``. Long-only, the cap and full
investment (cash included) hold **by construction**, not by penalty.

The softmax keeps the map smooth where no bound binds (the projection is then
the identity); the projection only acts when a risky weight would exceed the cap.
"""

from __future__ import annotations

import numpy as np

__all__ = ["softmax", "project_capped_simplex", "upper_bounds", "action_to_weights"]


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, "float64")
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def upper_bounds(n_risky: int, cap: float) -> np.ndarray:
    """``cap`` on every risky asset, 1 on the cash line."""
    return np.concatenate([np.full(n_risky, float(cap)), [1.0]])


def project_capped_simplex(p: np.ndarray, upper: np.ndarray, *, iterations: int = 200) -> np.ndarray:
    """Euclidean projection of ``p`` onto ``{0 <= w <= upper, sum w = 1}``.

    The solution is ``w = clip(p - tau, 0, upper)`` for the ``tau`` at which the
    weights sum to one; ``sum clip(p - tau, 0, upper)`` is continuous and
    non-increasing in ``tau``, so bisection finds it.
    """
    p = np.asarray(p, "float64")
    upper = np.asarray(upper, "float64")
    if p.shape != upper.shape:
        raise ValueError("p and upper must have the same shape")
    if upper.sum() < 1.0 - 1e-12:
        raise ValueError("infeasible: the upper bounds sum to less than 1")
    if not np.isfinite(p).all():
        raise ValueError("non-finite input to the projection")
    lo, hi = float((p - upper).min()), float(p.max())  # sum is sum(upper) >= 1 at lo, 0 at hi
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if np.clip(p - mid, 0.0, upper).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    w = np.clip(p - 0.5 * (lo + hi), 0.0, upper)
    return w


def action_to_weights(action: np.ndarray, upper: np.ndarray, logit_scale: float) -> np.ndarray:
    """Map a raw action to feasible weights (see the module docstring)."""
    a = np.asarray(action, "float64")
    if a.shape != upper.shape:
        raise ValueError(f"action shape {a.shape} != weights shape {upper.shape}")
    if not np.isfinite(a).all():
        raise ValueError("non-finite action")
    return project_capped_simplex(softmax(logit_scale * np.clip(a, -1.0, 1.0)), upper)
