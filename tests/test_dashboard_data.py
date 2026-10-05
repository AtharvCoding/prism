"""The dashboard's stored-results stage: copies are the frozen bytes, and every shown number is in the final report.

DASHBOARD.md §2 (frozen artifacts are immutable), §5.1, §6. These tests read the committed
``dashboard/artifacts`` and ``reports/final_report.md``; none reads ``data/processed``, a model or the holdout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism import dashboard_data as dd
from prism.utils.hashing import sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / dd.ARTIFACTS


@pytest.fixture()
def artifacts_copy(tmp_path) -> Path:
    out = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS, out)
    return out


# --------------------------------------------------------------------------- frozen sources
def _source(tmp_path: Path, text: str = "a,b\n1,2\n") -> tuple[dd.CopySpec, dict[str, str]]:
    src = tmp_path / "data" / "processed" / "x" / "t.csv"
    src.parent.mkdir(parents=True)
    src.write_text(text)
    return dd.CopySpec("data/processed/x/t.csv", "x/t.csv"), {"data/processed/x/t.csv": sha256_bytes(text.encode())}


def test_a_verified_source_is_copied_byte_for_byte(tmp_path):
    spec, baseline = _source(tmp_path)
    rec = dd.copy_verified(tmp_path, spec, baseline, tmp_path / "out")
    assert (tmp_path / "out" / "x" / "t.csv").read_bytes() == (tmp_path / spec.source).read_bytes()
    assert rec == {"kind": "copy", "source": spec.source, "sha256": baseline[spec.source]}


def test_a_source_that_differs_from_the_baseline_is_refused_and_nothing_is_written(tmp_path):
    spec, baseline = _source(tmp_path)
    (tmp_path / spec.source).write_text("a,b\n1,3\n")           # one digit changed after the baseline was taken
    with pytest.raises(dd.FrozenSourceError, match="frozen result was modified"):
        dd.copy_verified(tmp_path, spec, baseline, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_source_outside_the_baseline_is_refused(tmp_path):
    spec, _ = _source(tmp_path)
    with pytest.raises(dd.FrozenSourceError, match="not in the frozen baseline"):
        dd.copy_verified(tmp_path, spec, {}, tmp_path / "out")


def test_the_build_refuses_to_write_inside_data_processed(cfg, tmp_path):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    with pytest.raises(dd.ArtifactError, match="may not be inside data/processed"):
        dd.build_stored(cfg, tmp_path, tmp_path / "data" / "processed" / "dashboard")


def test_every_copied_artifact_is_the_frozen_file(cfg):
    """A copy's hash is the baseline hash of its source: the tables shown are the stored tables, not a re-export."""
    baseline = dd.read_baseline(ROOT)
    manifest = dd.verify_artifacts(ARTIFACTS)
    copies = {name: rec for name, rec in manifest["files"].items() if rec["kind"] == "copy"}
    assert {rec["source"] for rec in copies.values()} >= {s.source for s in dd.stored_copies(tuple(cfg.tier2.variants))}
    assert {"test/eval_daily.parquet", "holdout/eval_daily.parquet"} <= set(copies)
    for name, rec in copies.items():
        assert rec["sha256"] == baseline[rec["source"]], name


def test_the_learning_curves_are_a_consolidation_that_changes_no_value(tmp_path):
    cols = list(dd.CURVE_COLUMNS)
    sources, baseline = {}, {}
    for seed in (0, 1):
        rel = f"data/processed/tier2/runs/final/V1/c/seed{seed}/curve.csv"
        path = tmp_path / rel
        path.parent.mkdir(parents=True)
        pd.DataFrame({c: [seed + 0.123456789012345, seed + 2.0] for c in cols} | {"unused": [9, 9]}).to_csv(path, index=False)
        sources[("V1", seed)], baseline[rel] = rel, sha256_bytes(path.read_bytes())
    out = dd.learning_curves(tmp_path, sources, baseline)
    assert list(out.columns) == ["variant", "seed", *cols] and len(out) == 4
    for seed in (0, 1):
        stored = pd.read_csv(tmp_path / sources[("V1", seed)])[cols]
        pd.testing.assert_frame_equal(out[out.seed == seed][cols].reset_index(drop=True), stored)


# --------------------------------------------------------------------------- manifest
def test_the_committed_artifacts_match_their_manifest():
    manifest = dd.verify_artifacts(ARTIFACTS)
    assert manifest["baseline"] == dd.BASELINE and len(manifest["files"]) >= 40


def test_a_changed_artifact_an_extra_file_and_an_edited_manifest_are_each_detected(artifacts_copy):
    table = artifacts_copy / "holdout" / "tables" / "turnover.csv"
    original = table.read_text()
    table.write_text(original.replace("0.23", "0.13", 1))
    with pytest.raises(dd.ArtifactError, match="differs from the manifest"):
        dd.verify_artifacts(artifacts_copy)
    table.write_text(original)
    dd.verify_artifacts(artifacts_copy)

    (artifacts_copy / "extra.csv").write_text("x\n")
    with pytest.raises(dd.ArtifactError, match="not in the manifest"):
        dd.verify_artifacts(artifacts_copy)
    (artifacts_copy / "extra.csv").unlink()

    path = artifacts_copy / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["facts.json"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(dd.ArtifactError, match="manifest.json was edited"):
        dd.verify_artifacts(artifacts_copy)


# --------------------------------------------------------------------------- numbers against the final report
def test_every_headline_row_the_dashboard_shows_is_a_row_of_the_final_report():
    check = dd.verify_against_report(ARTIFACTS, ROOT / dd.REPORT)
    assert check["ok"] and check["rows_checked"] >= 100
    lines = dd.report_lines(ARTIFACTS)
    for window in dd.WINDOWS:                       # every comparison on every metric, and every variant and benchmark
        res = dd.load_results(ARTIFACTS, window)
        assert len(dd.gate_table(res)) == 3 and len(dd.difference_table(res)) == 12 and len(dd.variant_table(res)) == 10
    assert sum("indeterminate" in line for line in lines) == 24


@pytest.mark.parametrize("window", list(dd.WINDOWS))
def test_one_changed_number_fails_the_report_check(artifacts_copy, window):
    path = artifacts_copy / window / "tables" / "paired_block20.csv"
    table = pd.read_csv(path)
    table.loc[(table.candidate == "V4") & (table.control == "V2") & (table.metric == "sharpe"), "diff"] += 0.02
    table.to_csv(path, index=False)
    with pytest.raises(dd.ArtifactError, match="not in final_report.md"):
        dd.verify_against_report(artifacts_copy, ROOT / dd.REPORT)


def test_a_verdict_that_differs_from_the_report_fails_the_check(artifacts_copy):
    path = artifacts_copy / "holdout" / "tables" / "paired_block20.csv"
    table = pd.read_csv(path)
    table.loc[(table.candidate == "V4") & (table.control == "V2"), "verdict"] = "favourable"   # would read as PASS
    table.to_csv(path, index=False)
    assert dd.gate_table(dd.load_results(artifacts_copy, "holdout")).iloc[0]["paired rule"] == "PASS"
    with pytest.raises(dd.ArtifactError):
        dd.verify_against_report(artifacts_copy, ROOT / dd.REPORT)


# --------------------------------------------------------------------------- facts quoted in the copy
@pytest.fixture(scope="module")
def facts() -> dict:
    return json.loads((ARTIFACTS / "facts.json").read_text())


@pytest.mark.parametrize("window", list(dd.WINDOWS))
def test_the_facts_restate_the_stored_tables(facts, cfg, window):
    res, f = dd.load_results(ARTIFACTS, window), facts["windows"][window]
    turn = res["turnover"].set_index("variant")["mean_turnover_per_step"]
    agents = turn.loc[list(cfg.tier2.variants)]
    bench = turn.drop(index=list(cfg.tier2.variants))
    assert f["turnover_agents"] == pytest.approx([agents.min(), agents.max()], abs=1e-12)
    assert f["turnover_benchmarks"] == pytest.approx([bench.min(), bench.max()], abs=1e-12)
    assert f["comparisons_passing"] == int((dd.gate_table(res)["paired rule"] == "PASS").sum())
    assert (f["first"], f["last"], f["n_sessions"]) == (res["meta"]["first"], res["meta"]["last"], res["meta"]["n_sessions"])
    sharpe = res["paired_block20"].query("metric == 'sharpe'")
    assert f["sharpe_ci_half_width"] == pytest.approx([((sharpe.ci_high - sharpe.ci_low) / 2).min(), ((sharpe.ci_high - sharpe.ci_low) / 2).max()])


def test_the_facts_match_the_configuration(facts, cfg, plan):
    assert facts["universe"]["n_weights"] == len(cfg.data.allocatable["B"]) + 1
    assert facts["universe"]["sectors"] + facts["universe"]["bonds"] + facts["universe"]["gold"] == list(cfg.data.allocatable["B"])
    assert facts["rules"]["cap"] == cfg.data.allocation.weight_max and facts["rules"]["per_side_bps"] == cfg.env.costs.per_side_bps
    assert facts["splits"]["holdout"] == plan["holdout"].as_dict()
    assert facts["agents"]["n_agents"] == len(cfg.tier2.variants) * len(cfg.tier2.final_seeds)
    blocks = facts["agents"]["blocks"]
    assert blocks["V4"]["regime_source"] == "hmm" and blocks["C4"]["regime_source"] == "vix_threshold"
    assert blocks["V1"]["latent"] == 0 and blocks["V2"]["regime"] == 0 and blocks["V4"]["state_width"] == blocks["C4"]["state_width"]
    for v, c in facts["agents"]["configs"].items():
        assert c["horizon_weeks"] == pytest.approx(1 / (1 - c["gamma"]))


def test_the_page_takeaways_hold_in_the_stored_numbers(facts):
    """The Results takeaway says no comparison passed and the variant gaps are inside the seed spread; check both."""
    for window in dd.WINDOWS:
        f = facts["windows"][window]
        assert f["comparisons_passing"] == 0 and f["claimable"] == 0 and f["seeds_ge_dsr_bar"] == 0
        gaps = [abs(row[window]) for row in facts["sign_flips"]["sharpe"]]
        assert max(gaps) < f["seed_sharpe_std"][0]
        assert f["turnover_agents"][0] > 5 * f["turnover_benchmarks"][1]


def test_sign_flips_reads_the_sign_of_the_stored_difference():
    def table(v4v2, v4c4, v2v1):
        return pd.DataFrame({"candidate": ["V4", "V4", "V2"], "control": ["V2", "C4", "V1"], "metric": "sharpe", "diff": [v4v2, v4c4, v2v1]})

    out = dd.sign_flips(table(-0.1, 0.2, 0.3), table(0.1, 0.4, -0.3), "sharpe")
    assert out.flipped.tolist() == [True, False, True] and out.test.tolist() == [-0.1, 0.2, 0.3]


# --------------------------------------------------------------------------- derived series (DASHBOARD.md §5.2)
@pytest.mark.parametrize("window", list(dd.WINDOWS))
def test_per_seed_deflated_sharpe_has_the_stored_median_minimum_and_count(cfg, window):
    table = pd.read_csv(ARTIFACTS / window / "seed_dsr.csv")
    stored = pd.read_csv(ARTIFACTS / window / "tables" / "dsr.csv").set_index("variant")
    daily = pd.read_parquet(ARTIFACTS / window / "eval_daily.parquet")
    trials = (cfg.tier2.dsr_trials_headline, cfg.tier2.dsr_trials_all_runs)
    fresh = dd.seed_dsr(daily, tuple(cfg.tier2.variants), tuple(cfg.tier2.final_seeds), tuple(n[3:] for n in stored.index if n.startswith("BM|")),
                        trials, 5.0)
    pd.testing.assert_frame_equal(fresh, table, check_exact=False, rtol=0, atol=1e-12)
    for v in cfg.tier2.variants:
        mine = table[table.strategy == v]
        assert len(mine) == len(cfg.tier2.final_seeds)
        for k in trials:
            assert mine[f"dsr_n{k}"].median() == pytest.approx(stored.loc[v, f"dsr_median_n{k}"], abs=1e-12)
            assert mine[f"dsr_n{k}"].min() == pytest.approx(stored.loc[v, f"dsr_min_n{k}"], abs=1e-12)
            assert int((mine[f"dsr_n{k}"] >= 0.95).sum()) == int(stored.loc[v, f"seeds_ge_0.95_n{k}"])
    for name in stored.index[stored.index.str.startswith("BM|")]:
        assert table[table.strategy == name][f"dsr_n{trials[0]}"].iloc[0] == pytest.approx(stored.loc[name, f"dsr_median_n{trials[0]}"], abs=1e-12)


@pytest.mark.parametrize("window", list(dd.WINDOWS))
def test_an_equity_curve_from_the_stored_daily_returns_ends_at_the_stored_annualised_return(cfg, window):
    daily = pd.read_parquet(ARTIFACTS / window / "eval_daily.parquet")
    assert dd._check_nav(daily, ARTIFACTS, window, tuple(cfg.tier2.variants), 5.0) == 46
    broken = daily.copy()
    broken.iloc[5, broken.columns.get_loc("V4|s3|5")] += 1e-4
    with pytest.raises(dd.ArtifactError, match="V4 seed 3"):
        dd._check_nav(broken, ARTIFACTS, window, tuple(cfg.tier2.variants), 5.0)


def test_the_agreement_rate_is_reproducible_from_the_daily_regime_series(plan):
    """D3 acceptance: the HMM-vs-VIX agreement shown is a count over the stored daily states, checked against C2 at build time."""
    daily = pd.read_parquet(ARTIFACTS / "regimes" / "daily.parquet")
    summary = json.loads((ARTIFACTS / "regimes" / "summary.json").read_text())
    assert summary["checks"]["vix_state_equals_C2"] and summary["checks"]["posteriors_max_abs_diff"] <= 1e-9
    assert np.allclose(daily.p_calm + daily.p_volatile, 1.0) and set(daily.vix_high.dropna().unique()) == {0.0, 1.0}
    for name in ("train", "val", "test", "holdout"):
        f = daily.loc[plan[name].effective_start: plan[name].effective_end].dropna(subset=["vix_high"])
        got = summary["agreement"][name]
        assert got["days"] == len(f) and got["agreement"] == pytest.approx(((f.p_volatile > 0.5) == (f.vix_high == 1)).mean())
        assert got["both_calm"] + got["both_volatile"] + got["hmm_only"] + got["vix_only"] == len(f)
    everything = dd.regime_summary(daily, {"all": (daily.dropna().index[0], daily.index[-1])})["all"]
    assert everything == summary["agreement"]["all"]


def test_regime_summary_counts_a_hand_made_series():
    days = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame({"p_calm": [0.9, 0.2, 0.4, 0.8, 0.1], "p_volatile": [0.1, 0.8, 0.6, 0.2, 0.9], "vix_high": [0.0, 1.0, 0.0, 1.0, np.nan]}, index=days)
    out = dd.regime_summary(frame, {"w": (days[0], days[-1])})["w"]
    assert (out["days"], out["both_calm"], out["both_volatile"], out["hmm_only"], out["vix_only"]) == (4, 1, 1, 1, 1)
    assert out["agreement"] == 0.5 and out["hmm_volatile_share"] == 0.5


def test_the_k_selection_table_is_the_step_2_reports():
    """D3 acceptance: the K-selection chart's numbers are those of reports/tables/hmm_k_selection_H1.md."""
    text = (ROOT / "reports" / "tables" / "hmm_k_selection_H1.md").read_text()
    table = pd.read_csv(ARTIFACTS / "regimes" / "k_selection.csv")
    pd.testing.assert_frame_equal(table, dd.parse_k_selection(text))
    assert table.k.tolist() == list(range(2, 9)) and table.selected.tolist() == [True] + [False] * 6
    for _, row in table.iterrows():
        assert f"| {row.val_loglik:.2f} | {row.bic:.2f} |" in text
    usable = table[table.restarts_degenerate < table.restarts]
    assert usable.k.tolist() == [2, 3] and usable.val_loglik.diff().iloc[-1] < 2.0 and usable.bic.idxmin() == 0


def test_the_latent_map_is_fitted_one_encoder_fold_at_a_time():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=120)
    latents = pd.DataFrame(rng.normal(size=(120, 6)), index=days, columns=[f"latent_{i}" for i in range(6)])
    latents.iloc[60:] = latents.iloc[60:] @ np.linalg.qr(rng.normal(size=(6, 6)))[0] + 5.0      # the refit lays the space out differently
    fold_of = pd.Series([0] * 60 + [1] * 60, index=days)
    coords, summary = dd.latent_pca(latents, fold_of)
    assert coords.fold.tolist() == fold_of.tolist() and summary.days.tolist() == [60, 60]
    for fold in (0, 1):
        block = coords[coords.fold == fold]
        assert abs(block.pc1.mean()) < 1e-9 and abs(block.pc2.mean()) < 1e-9          # centred within its own fold, not across folds
        assert block.pc1.var() >= block.pc2.var()
    alone, _ = dd.latent_pca(latents.iloc[:60], fold_of.iloc[:60])
    assert np.allclose(alone[["pc1", "pc2"]], coords.iloc[:60][["pc1", "pc2"]])            # fold 0 does not depend on fold 1


def test_the_stored_latent_map_has_one_fold_per_year_through_the_holdout():
    coords = pd.read_parquet(ARTIFACTS / "latents" / "pca.parquet")
    folds = pd.read_csv(ARTIFACTS / "latents" / "folds.csv", parse_dates=["start", "end", "fit_end"])
    assert len(folds) == 20 and (folds.fit_end < folds.start).all() and folds.days.sum() == len(coords)
    assert (coords.groupby("fold").apply(lambda g: g.index.year.nunique()) == 1).all()
    assert coords.p_volatile.between(0, 1).all()
