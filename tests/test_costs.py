"""Transaction-cost model. **PHASE B, DEFERRED** — spec §0.3, §7.5, §11.

Not implemented in Phase A. The contracts §7.5 specifies are recorded here so
that step 4b has a written target:

* zero turnover implies zero cost,
* cost is monotone in turnover,
* a round-trip cost matches an analytic value.

The first two are already verified at the *turnover* level in
``test_metrics.py`` (``test_zero_turnover_for_a_held_portfolio`` and
``test_turnover_is_monotone_in_weight_change``), because turnover is the
input the cost model multiplies. What remains for step 4b is the cost model
itself: the bps-per-side convention and the volatility-scaled slippage term.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="PHASE B, DEFERRED (spec §0.3). src/prism/env/costs.py is a docstring "
    "stub; the cost model is build step 4b, which must not begin until Phase A's "
    "§0.3 exit criteria are met and reviewed."
)


def test_zero_turnover_implies_zero_cost():
    """§7.5. Turnover-level half already verified in test_metrics.py."""
    raise NotImplementedError("spec §11 — build step 4b")


def test_cost_is_monotone_in_turnover():
    """§7.5. Turnover-level half already verified in test_metrics.py."""
    raise NotImplementedError("spec §11 — build step 4b")


def test_round_trip_cost_matches_the_analytic_value():
    """§7.5: at 5 bps per side, a full round trip costs 10 bps of the traded
    notional — and nothing else."""
    raise NotImplementedError("spec §11 — build step 4b")


def test_cost_sensitivity_is_reported_at_0_5_10_20_bps():
    """§11: "Report results at 0 / 5 / 10 / 20 bps as a sensitivity"."""
    raise NotImplementedError("spec §11 — build step 4b")
