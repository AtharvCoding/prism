"""The dashboard's stored-results stage: copies are the frozen bytes, and every shown number is in the final report.

DASHBOARD.md §2 (frozen artifacts are immutable), §5.1, §6. These tests read the committed
``dashboard/artifacts`` and ``reports/final_report.md``; none reads ``data/processed``, a model or the holdout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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
    assert {rec["source"] for rec in copies.values()} == {s.source for s in dd.stored_copies(tuple(cfg.tier2.variants))}
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
