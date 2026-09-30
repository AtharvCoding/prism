"""Window construction. The alignment contract is the whole point. Spec §9.3.

PHASE A, build step 3 — not yet implemented.
"""

from __future__ import annotations

__all__: list[str] = []


def make_windows(frame, window):  # noqa: ANN001, ANN201
    """Windows of length ``window``, labelled by the date they **end on**.

    Spec §9.3, and the defect this replaces (C1): the reference implementation
    labelled the window ending at row ``k+window-1`` with the date at row
    ``k+window``, and sized the dataset at ``len(data) - window``, dropping
    the final window. Both errors are one line and neither is visible by
    inspection, which is why ``tests/test_alignment.py`` checks:

    * ``len(windows) == len(frame) - window + 1``
    * the row dated ``D`` is the window whose **last** row is ``D``
    """
    raise NotImplementedError("spec §9.3 — build step 3")
