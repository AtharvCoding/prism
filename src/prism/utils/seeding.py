"""Deterministic seeding.

Spec §2. Determinism is a stated non-negotiable principle (§0.4.3). Where full
determinism is not achievable — notably cuDNN convolution/RNN kernels — this
module *logs* the fact rather than silently proceeding.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

from prism.utils.logging import get_logger

__all__ = ["SeedState", "seed_everything", "derive_seed", "make_rng"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class SeedState:
    """What was actually seeded, for the run manifest."""

    master: int
    python_hash_seed: str | None
    numpy: int
    torch: int | None
    torch_deterministic: bool
    nondeterminism_notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "master": self.master,
            "python_hash_seed": self.python_hash_seed,
            "numpy": self.numpy,
            "torch": self.torch,
            "torch_deterministic": self.torch_deterministic,
            "nondeterminism_notes": list(self.nondeterminism_notes),
        }


def derive_seed(master: int, *tags: object) -> int:
    """Derive a stable child seed from ``master`` and arbitrary tags.

    Uses SHA-256 rather than ``hash()`` so the result does not depend on
    ``PYTHONHASHSEED`` and is identical across processes and platforms.
    """
    import hashlib

    payload = "|".join([str(master), *(str(t) for t in tags)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def make_rng(master: int, *tags: object) -> np.random.Generator:
    """A fresh ``numpy`` Generator seeded from ``master`` and ``tags``."""
    return np.random.default_rng(derive_seed(master, *tags))


def seed_everything(master: int, *, torch_deterministic: bool = True) -> SeedState:
    """Seed every RNG this project touches and report what could not be fixed."""
    notes: list[str] = []

    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None:
        notes.append(
            "PYTHONHASHSEED was not set before interpreter start; it cannot be set "
            "retroactively. Use `make` targets or `PYTHONHASHSEED=0 python ...` for "
            "bit-identical set/dict iteration order."
        )

    random.seed(master)
    np.random.seed(master % (2**32))

    torch_seed: int | None = None
    deterministic = False
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        notes.append("torch not importable; torch RNGs unseeded")
    else:
        torch_seed = master
        torch.manual_seed(master)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(master)
        if torch_deterministic:
            try:
                torch.use_deterministic_algorithms(True)
                deterministic = True
            except Exception as exc:  # pragma: no cover - platform dependent
                notes.append(f"torch.use_deterministic_algorithms(True) failed: {exc}")
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:  # pragma: no cover
                pass
            if torch.backends.cudnn.is_available():
                notes.append(
                    "cuDNN is available: LSTM kernels may still be non-deterministic "
                    "even with deterministic algorithms requested (spec §2)."
                )

    for note in notes:
        _log.warning("determinism: %s", note)

    return SeedState(
        master=master,
        python_hash_seed=hash_seed,
        numpy=master % (2**32),
        torch=torch_seed,
        torch_deterministic=deterministic,
        nondeterminism_notes=tuple(notes),
    )
