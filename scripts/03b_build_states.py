#!/usr/bin/env python
"""Assemble all nine state variants.

Spec §10, §16 step 3b — build step 3b. Not yet implemented.

Step 0 delivers the repository, the validated config, the splits and the test
suite. This script is a declared entry point so that the Makefile and the
``§16`` build order are visible from the start; implementing it is step 3b.
"""

from __future__ import annotations

import sys

STEP = "3b"


def main(argv: list[str] | None = None) -> int:
    print(
        f"scripts/03b_build_states.py is build step {STEP} and is not yet implemented "
        "(spec §10, §16 step 3b).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
