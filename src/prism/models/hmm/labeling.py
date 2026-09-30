"""Deterministic canonical state relabeling. Spec §8.5.

PHASE A, build step 2 — not yet implemented.
"""

from __future__ import annotations

__all__: list[str] = []


def canonical_labels(model):  # noqa: ANN001, ANN201
    """Permutation sorting states by conditional return std, ascending.

    Spec §8.5. EM state ordering is arbitrary and changes across restarts and
    refits (label switching). The permutation must be applied to
    ``startprob_``, ``transmat_``, ``means_``, ``covars_`` **and** the
    posterior columns together — and names must be derived from position
    after sorting, never hard-coded. The reference code carried three
    hard-coded label lists assuming an ordering the fitted model did not have,
    so its "live crisis probability" readout was reporting the bull state
    (defect B4).
    """
    raise NotImplementedError("spec §8.5 — build step 2")
