#!/usr/bin/env python
"""Fit the HMM on Universe A, select K on validation, emit filtered posteriors.

Spec §8, §16 step 2 — build step 2. Not yet implemented.

Step 0 delivers the repository, the validated config, the splits and the test
suite. This script is a declared entry point so that the Makefile and the
``§16`` build order are visible from the start; implementing it is step 2.
"""

from __future__ import annotations

import sys

STEP = "2"


def main(argv: list[str] | None = None) -> int:
    print(
        f"scripts/02_fit_hmm.py is build step {STEP} and is not yet implemented "
        "(spec §8, §16 step 2).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
