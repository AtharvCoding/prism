"""RL environment and cost model — **PHASE B**. Spec §11, build step 4b.

* :mod:`prism.env.costs` — per-side proportional cost plus volatility-scaled slippage
* :mod:`prism.env.actions` — action -> long-only, capped, fully-invested weights
* :mod:`prism.env.rewards` — ``log_return_net`` (default), ``dsr``, ``mv_penalty``, ``drawdown_penalty``
* :mod:`prism.env.data` — the pre-built state / return arrays and the holding-period schedule
* :mod:`prism.env.portfolio_env` — the Gymnasium environment

``PortfolioEnv`` is imported lazily by name (``from prism.env.portfolio_env
import PortfolioEnv``) so that importing the cost model does not require
``gymnasium``.
"""

PHASE = "B"
STATUS = "implemented"
