"""Shared fixtures.

Every fixture is deterministic and offline. Nothing in the suite touches the
network: the snapshot is write-once and its creation is exercised against
synthetic panels (:mod:`prism.data.synthetic`), so the tests assert properties
of the *code* rather than properties of whatever the vendor served today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.config import Config, load_config
from prism.data.synthetic import make_small_raw, make_synthetic_raw
from prism.features.build import build_features
from prism.splits import build_split_plan

pytest_plugins: list[str] = []


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The committed project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def plan(cfg: Config):
    """The embargo-purged split plan."""
    return build_split_plan(cfg)


@pytest.fixture(scope="session")
def raw_b(cfg: Config) -> pd.DataFrame:
    """Full synthetic Universe B raw panel, 1998-12-16 .. end of test split."""
    return make_synthetic_raw(cfg, seed=0, universe="B")


@pytest.fixture(scope="session")
def raw_a_only(cfg: Config) -> pd.DataFrame:
    """A panel containing **only** Universe A tickers.

    Production always builds both universes from one snapshot, so this is not
    the fixture to use for cross-universe comparisons — it is a different
    dataset (fewer tickers means a different draw order from the generator).
    Its purpose is narrower and structural: prove the Universe A pipeline runs
    when the B-only series are not merely ignored but absent entirely.
    """
    return make_synthetic_raw(cfg, seed=0, universe="A")


@pytest.fixture(scope="session")
def raw_small(cfg: Config) -> pd.DataFrame:
    """Short Universe A panel (~900 sessions) for the fast causality sweeps."""
    return make_small_raw(cfg, seed=0, sessions=900, universe="A")


@pytest.fixture(scope="session")
def features_b(cfg: Config, raw_b: pd.DataFrame):
    return build_features(raw_b, cfg, "B")


@pytest.fixture(scope="session")
def features_a(cfg: Config, raw_b: pd.DataFrame):
    """Universe A features built from the **same** panel as ``features_b``.

    This mirrors production: one snapshot, two ``build_features`` calls that
    filter it differently. Building A from its own panel would make the two
    frames incomparable for no reason.
    """
    return build_features(raw_b, cfg, "A")


@pytest.fixture
def rng() -> np.random.Generator:
    """A generator with a fixed seed, fresh per test."""
    return np.random.default_rng(20260101)


@pytest.fixture
def tmp_repo(tmp_path, cfg: Config) -> Config:
    """A config rooted at a temporary directory, for tests that write files."""
    import shutil

    shutil.copytree(cfg.root / "configs", tmp_path / "configs")
    (tmp_path / "reports" / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    return load_config(root=tmp_path)
