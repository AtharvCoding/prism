"""SAC agents and the Tier 2 pipeline — **PHASE B**, build step 4c. Spec §12-§14.

* :mod:`prism.agents.sac` — SAC via stable-baselines3 (CPU), checkpointed training, deterministic evaluation
* :mod:`prism.agents.jobs` — one resumable training run as a picklable job
* :mod:`prism.agents.sanity` — the sanity gates (degenerate task, beats random)
* :mod:`prism.agents.data` — train/validation-only env data; the test split only when asked for
* :mod:`prism.agents.evaluate` — agents and benchmarks through the environment
* :mod:`prism.agents.tier2` — tuning, freezing, the single test evaluation, the statistics

The contract is ``reports/tables/preregistration_tier2.md``; the entry point is ``scripts/05_train_agents.py`` (``make tier2``).
"""

PHASE = "B"
STATUS = "implemented"
