"""Logging setup.

Deliberately plain: a single ``rich``-backed handler on the ``prism`` logger,
configured once, never at import time of a library module. No module calls
``warnings.filterwarnings('ignore')`` — hiding convergence and deprecation
warnings was defect A9.
"""

from __future__ import annotations

import logging
from pathlib import Path

__all__ = ["get_logger", "configure_logging"]

_ROOT = "prism"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``prism`` namespace."""
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure the ``prism`` logger. Idempotent; safe to call from scripts.

    Library modules must *not* call this — only entry points in ``scripts/``.
    """
    global _configured
    root = logging.getLogger(_ROOT)
    if not _configured:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                rich_tracebacks=True, show_path=False, omit_repeated_times=False
            )
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        except ImportError:  # pragma: no cover
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
            )
        root.addHandler(handler)
        root.propagate = False
        _configured = True

    root.setLevel(level)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = {
            getattr(h, "baseFilename", None) for h in root.handlers if isinstance(h, logging.FileHandler)
        }
        if str(path.resolve()) not in existing:
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
            )
            root.addHandler(fh)
    return root
