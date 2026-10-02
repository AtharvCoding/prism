"""Read the SHA-256 table out of ``reports/tables/preregistration.md``.

The pre-registration lists the state files and inputs the Tier 1 results were
produced from, one ``<sha256>  <path relative to data/processed>`` per line.
Two things depend on that table: the Tier 1 script verifies its inputs against
it, and the state builder refuses to overwrite a listed file (DECISIONS.md
D-033) so a rebuild cannot silently invalidate the pre-registered hashes.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["PREREG", "preregistered_hashes"]

PREREG = "reports/tables/preregistration.md"

_LINE = re.compile(r"^([0-9a-f]{64})  (\S+)$", flags=re.M)


def preregistered_hashes(root: str | Path, prereg: str = PREREG) -> dict[str, str]:
    """``{path relative to data/processed: sha256}`` for every listed file."""
    text = (Path(root) / prereg).read_text(encoding="utf-8")
    return {name: digest for digest, name in _LINE.findall(text)}
