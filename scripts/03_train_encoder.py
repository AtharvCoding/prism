#!/usr/bin/env python
"""Train the LSTM encoder on Universe A with walk-forward refits.

Spec §9, §16 step 3 — build step 3. Not yet implemented.

Step 0 delivers the repository, the validated config, the splits and the test
suite. This script is a declared entry point so that the Makefile and the
``§16`` build order are visible from the start; implementing it is step 3.
"""

from __future__ import annotations

import sys

STEP = "3"


def main(argv: list[str] | None = None) -> int:
    print(
        f"scripts/03_train_encoder.py is build step {STEP} and is not yet implemented "
        "(spec §9, §16 step 3).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
