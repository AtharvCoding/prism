"""Content hashing and run manifests.

Spec §1: "Every script writes a run manifest to ``reports/logs/<run_id>.json``
containing the config hash, the data snapshot hash, the git commit, the seeds,
and the library versions."

Spec §4.1: the raw snapshot carries per-file SHA-256 so that every result
traces to exactly one dataset. Adjusted prices are restated whenever a
dividend is paid, so a re-download silently changes history; the hash is the
only thing that makes that visible.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "sha256_file",
    "sha256_bytes",
    "hash_object",
    "git_commit",
    "pip_freeze",
    "library_versions",
    "utc_now_iso",
    "write_manifest",
    "make_run_id",
]

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_object(obj: Any) -> str:
    """Stable SHA-256 of any JSON-serialisable object.

    Keys are sorted and non-JSON types are stringified, so the digest depends
    on content rather than on dict insertion order or ``PYTHONHASHSEED``.
    """
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def git_commit(root: str | Path | None = None) -> dict[str, Any]:
    """Current commit and whether the tree is dirty.

    A dirty tree is recorded, not corrected: a result produced from
    uncommitted code must say so.
    """
    cwd = str(root) if root is not None else None
    out: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        out["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        out["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
        out["dirty"] = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return out


def pip_freeze() -> list[str]:
    """Full ``pip freeze`` of the active interpreter."""
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return []
    return [line for line in res.stdout.splitlines() if line.strip()]


def library_versions() -> dict[str, str]:
    """Versions of the libraries whose behaviour can change a result."""
    import importlib.metadata as md

    names = [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "hmmlearn",
        "torch",
        "yfinance",
        "pandas-market-calendars",
        "statsmodels",
    ]
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:  # pragma: no cover
            out[name] = "not installed"
    return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(stage: str) -> str:
    """``<stage>_<utc timestamp>`` — sorts chronologically, unique per second."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}_{stamp}"


def write_manifest(
    path: str | Path,
    *,
    run_id: str,
    stage: str,
    config_hash: str | None = None,
    snapshot_hash: str | None = None,
    seeds: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> Path:
    """Write a run manifest and return its path."""
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "written_utc": utc_now_iso(),
        "config_hash": config_hash,
        "snapshot_hash": snapshot_hash,
        "git": git_commit(root),
        "seeds": seeds or {},
        "libraries": library_versions(),
        "argv": sys.argv,
    }
    if extra:
        manifest.update(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out
