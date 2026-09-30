"""RL environment — **PHASE B, DEFERRED**. Spec §0.3, §11.

Not implemented, deliberately. Per §0.3: "Do not implement
``src/prism/env/``, ``src/prism/agents/``, or any SAC code in Phase A."

This package exists so that Phase A's interfaces anticipate the environment
rather than having to be reshaped for it later. Two interface commitments are
already honoured upstream:

* :mod:`prism.state` accepts an optional **portfolio block** (current weights,
  time since last rebalance, cumulative turnover) — wired but unused in Phase
  A. Without current weights an agent cannot reason about transaction costs,
  which was design gap D1.
* Every state variant shares one index and one scaling policy, so the env can
  hold a pre-built state array and a pre-built return array and never index
  beyond the current step (§11).

Phase A must end and be reviewed against the §0.3 exit criteria before any of
this is written.
"""

PHASE = "B"
STATUS = "deferred"
