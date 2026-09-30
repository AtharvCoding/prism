#!/usr/bin/env python
"""Evaluate the final holdout. **PHASE B, DEFERRED** — spec §6.3, §16 step 5.

The holdout (2024-01-01 .. 2026-09-30 per config) is touched **exactly once**,
at the very end, after every decision is frozen. Not in Phase A.

Three gates must all be satisfied:

1. ``--i-am-sure`` on the command line,
2. ``PRISM_ALLOW_HOLDOUT=1`` in the environment,
3. ``final=True`` at the ``load_holdout`` call site.

Every invocation is logged permanently to
``reports/logs/holdout_access.jsonl`` with a reason and the git commit.
"""

from __future__ import annotations

import sys

PHASE = "B"


def main(argv: list[str] | None = None) -> int:
    print(
        "scripts/99_final_holdout.py is PHASE B (spec §16 step 5). The holdout is "
        "locked and stays locked: it is evaluated once, after Phase B's results are "
        "final. Running it now would spend the only unbiased estimate this project "
        "has.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
