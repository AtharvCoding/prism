"""Step 5: the holdout stays locked, and the pieces of the one real run work. No test opens the holdout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from prism import holdout as H  # noqa: E402
from prism.agents.data import load_close  # noqa: E402
from prism.env.data import build_env_data  # noqa: E402
from prism.reporting.tier2_report import append_decisions  # noqa: E402


def _tiny_state(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(np.zeros((len(index), 2)), index=index)


def test_the_env_builder_refuses_the_holdout_without_both_gates(cfg, monkeypatch):
    close = pd.DataFrame(index=pd.bdate_range("2024-01-02", periods=30))
    state = _tiny_state(close.index)
    monkeypatch.delenv("PRISM_ALLOW_HOLDOUT", raising=False)
    with pytest.raises(PermissionError, match="locked"):
        build_env_data(cfg, state, close, "holdout")
    with pytest.raises(PermissionError, match="locked"):
        build_env_data(cfg, state, close, "holdout", final_holdout=True)      # flag without the environment variable
    monkeypatch.setenv("PRISM_ALLOW_HOLDOUT", "1")
    with pytest.raises(PermissionError, match="locked"):
        build_env_data(cfg, state, close, "holdout")                           # environment variable without the flag


def test_load_close_refuses_the_holdout(cfg):
    with pytest.raises(PermissionError):
        load_close(cfg, "holdout")


def test_the_holdout_script_needs_exactly_one_mode(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("s99", Path(__file__).resolve().parents[1] / "scripts" / "99_final_holdout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.main([])
    with pytest.raises(SystemExit):
        mod.main(["--i-am-sure", "--rehearse"])


def test_replay_check_flags_a_difference_and_a_shape_change(cfg, tmp_path, monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=10)
    stored = {v: pd.DataFrame(np.arange(20.0).reshape(10, 2), index=idx, columns=["a", "b"]) for v in H.VARIANTS}
    monkeypatch.setattr(pd, "read_parquet", lambda path, *a, **k: stored[Path(path).stem].copy())
    ext_states = {v: f.copy() for v, f in stored.items()}
    ext = H.ExtendedStates(end=idx[-1], states=ext_states, posteriors=pd.DataFrame(), latents=pd.DataFrame(), close=pd.DataFrame(), timings={})
    assert H.replay_check(cfg, ext, up_to=idx[-1], atol=1e-9)["ok"]
    ext_states["V2"] = ext_states["V2"] + 1e-6
    bad = H.replay_check(cfg, ext, up_to=idx[-1], atol=1e-9)
    assert not bad["ok"] and bad["variants"]["V2"]["max_abs_diff"] == pytest.approx(1e-6)
    ext_states["V2"] = stored["V2"].iloc[:-1]
    shape = H.replay_check(cfg, ext, up_to=idx[-1], atol=1e-9)
    assert not shape["ok"] and shape["variants"]["V2"]["max_abs_diff"] == float("inf")


def test_a_holdout_decisions_entry_is_appended_once(tmp_path):
    (tmp_path / "DECISIONS.md").write_text("# D\n")
    assert append_decisions(tmp_path, "## Gate decisions (holdout)\n\nx\n", marker="## Gate decisions (holdout)")
    assert not append_decisions(tmp_path, "## Gate decisions (holdout)\n\ny\n", marker="## Gate decisions (holdout)")


def test_the_final_report_renders_from_stored_results_without_a_holdout(cfg, tmp_path):
    root = Path(cfg.root)
    if not (root / "data" / "processed" / "tier2" / "eval_done.json").exists():
        pytest.skip("Tier 2 results are not stored on this machine")
    from prism.reporting.final_report import build_final_report

    out = build_final_report(cfg, root, tmp_path / "final.md", holdout_dir=tmp_path / "no_holdout_here")
    text = out.read_text()
    assert "## 1. Verdict" in text and "not yet evaluated" in text and "Reproducibility statement" in text
