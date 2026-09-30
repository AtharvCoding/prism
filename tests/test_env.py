"""RL environment. **PHASE B, DEFERRED** — spec §0.3, §7.5, §11.

Not implemented in Phase A, deliberately: per §0.3, "Do not implement
``src/prism/env/``, ``src/prism/agents/``, or any SAC code in Phase A."

The §7.5 contracts are recorded so step 4b has a written target. One of them —
that a deterministic buy-and-hold policy reproduces the analytic benchmark
return — is already satisfied at the *timing* level by
``test_alignment.py::test_decision_at_t_earns_the_return_starting_at_t_plus_lag``,
which verifies the observe/decide/execute/earn convention against a
hand-computed path. The environment will wrap that convention rather than
reimplement it.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="PHASE B, DEFERRED (spec §0.3). src/prism/env/ contains docstring stubs "
    "only; gymnasium is not installed and prism.config raises if it appears while "
    "phase: A. The environment is build step 4b."
)


def test_weights_sum_to_one_and_respect_bounds():
    """§7.5. Long-only with cash, each weight in [0, 0.35] (§3.3 decision 2),
    enforced by simplex projection rather than by penalty (§11)."""
    raise NotImplementedError("spec §11 — build step 4b")


def test_episode_length_matches_the_calendar():
    """§7.5."""
    raise NotImplementedError("spec §11 — build step 4b")


def test_no_nan_observations():
    """§7.5."""
    raise NotImplementedError("spec §11 — build step 4b")


def test_buy_and_hold_reproduces_the_analytic_benchmark_return():
    """§7.5. Timing-level half already verified in test_alignment.py."""
    raise NotImplementedError("spec §11 — build step 4b")


def test_reward_window_does_not_overlap_the_observation_window():
    """§11: asserted in tests, not assumed."""
    raise NotImplementedError("spec §11 — build step 4b")


def test_env_never_indexes_beyond_the_current_step():
    """§11: "No look-ahead in the env"."""
    raise NotImplementedError("spec §11 — build step 4b")
