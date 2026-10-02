"""Multi-seed training with a shared hyperparameter budget. Spec §12.

The implementation lives in :mod:`prism.agents.jobs` (one run) and :mod:`prism.agents.tier2` (the grid, the
seeds, the freeze); this module is kept so the spec §1 layout still resolves.
"""

from __future__ import annotations

from prism.agents.jobs import run_job
from prism.agents.tier2 import final_specs, tuning_specs

__all__ = ["run_job", "tuning_specs", "final_specs"]
