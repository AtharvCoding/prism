#!/usr/bin/env python
"""Train SAC agents. **PHASE B, DEFERRED** — spec §0.3, §12, §16 step 4c.

Not implemented, and must not be until Phase A's §0.3 exit criteria are met
and reviewed. ``gymnasium`` and ``stable-baselines3`` are not installed, and
``prism.config`` raises at load time if either appears while ``phase: A``.
"""

from __future__ import annotations

import sys

PHASE = "B"


def main(argv: list[str] | None = None) -> int:
    print(
        "scripts/05_train_agents.py is PHASE B (spec §0.3). Phase A must end and be "
        "reviewed against the §0.3 exit criteria before any agent code is written.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
