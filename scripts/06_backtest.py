#!/usr/bin/env python
"""Backtest the allocator variants against all six benchmarks.

Spec §14, §16 step 4a — build step 4a. Not yet implemented.

Step 0 delivers the repository, the validated config, the splits and the test
suite. This script is a declared entry point so that the Makefile and the
``§16`` build order are visible from the start; implementing it is step 4a.
"""

from __future__ import annotations

import sys

STEP = "4a"


def main(argv: list[str] | None = None) -> int:
    print(
        f"scripts/06_backtest.py is build step {STEP} and is not yet implemented "
        "(spec §14, §16 step 4a).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
