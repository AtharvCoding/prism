#!/usr/bin/env python
"""Build both universes' features, run QA, write the quality report.

Spec §5, §16 step 1 — build step 1. Not yet implemented.

Step 0 delivers the repository, the validated config, the splits and the test
suite. This script is a declared entry point so that the Makefile and the
``§16`` build order are visible from the start; implementing it is step 1.
"""

from __future__ import annotations

import sys

STEP = "1"


def main(argv: list[str] | None = None) -> int:
    print(
        f"scripts/01_build_features.py is build step {STEP} and is not yet implemented "
        "(spec §5, §16 step 1).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
