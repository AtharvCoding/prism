"""Environment data for the Phase B variants, per split. Spec §11-§12, build step 4c.

The training stages read the **train and validation splits only**: the raw panel
is cut at the validation split's last session before any feature is built, and
every state frame is sliced to the same bound the moment it is read. The test
split is built by :func:`load_close` / :func:`variant_env_data` only when the
caller asks for it, which the Tier 2 runner does once, in its evaluation stage
(``reports/tables/preregistration_tier2.md`` §9). The holdout is refused by
``build_env_data`` and by ``assert_not_holdout`` here.
"""

from __future__ import annotations

import pandas as pd

from prism.data.loaders import assert_not_holdout, load_snapshot
from prism.env.data import EnvData, build_env_data
from prism.features.build import build_features, make_raw_frame
from prism.splits import SplitPlan, build_split_plan

__all__ = ["load_close", "variant_env_data", "split_bounds"]


def split_bounds(cfg, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:  # noqa: ANN001
    """Effective (embargo-purged) first and last session of a split."""
    s = build_split_plan(cfg)[split]
    return s.effective_start, s.effective_end


def load_close(cfg, last_split: str) -> pd.DataFrame:  # noqa: ANN001
    """Cleaned close panel for Universe B, with no row after ``last_split``'s last session."""
    if last_split == "holdout":
        raise PermissionError("the holdout is locked until build step 5 (spec §6.3)")
    _, end = split_bounds(cfg, last_split)
    snapshot = load_snapshot(cfg)
    raw = make_raw_frame(snapshot.close, snapshot.volume, snapshot.macro).loc[:end]
    assert_not_holdout(cfg, raw.index, context=f"tier2 raw panel (through {last_split})")
    close = build_features(raw, cfg, "B").close
    if close.index.max() > end:
        raise RuntimeError("the close panel extends past the requested split")
    return close


def variant_env_data(
    cfg,  # noqa: ANN001
    variant: str,
    close: pd.DataFrame,
    split: str,
    *,
    plan: SplitPlan | None = None,
    risky: list[str] | None = None,
) -> EnvData:
    """One variant's state frame, restricted to ``split``'s effective range, as env arrays."""
    plan = plan if plan is not None else build_split_plan(cfg)
    state = pd.read_parquet(cfg.path("processed") / "states" / f"{variant}.parquet")
    state = state.loc[: min(plan[split].effective_end, close.index.max())]
    return build_env_data(cfg, state, close, split, plan=plan, risky=risky)
