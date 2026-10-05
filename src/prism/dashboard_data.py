"""The dashboard's precompute, stage "stored": copy the stored results the pages display. DASHBOARD.md §5.1, §6.

The dashboard computes no statistic for the verdict. This module copies the stored,
pre-registered tables **byte for byte** into ``dashboard/artifacts/`` (small, committed),
after checking each source against the SHA-256 baseline of the frozen files
(``dashboard/frozen_sources.sha256``, DECISIONS.md D-045), and derives three things that
are not statistics: the learning curves of the 40 final runs in one file, a ``facts.json``
of the configuration values and table summaries the page copy quotes, and a check that
every headline row the dashboard shows is a row of ``reports/final_report.md``
(:func:`verify_against_report`; the precompute fails on any mismatch).

Nothing here reads the raw snapshot, a state file, a model or the network, and nothing
writes outside the artifact directory. The stages that replay the agents and persist the
live models are milestone D2 and are gated separately.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from prism.analysis.tier2 import METRICS
from prism.reporting import final_report as fr
from prism.reporting.report import md_table
from prism.utils.hashing import hash_object, sha256_file

__all__ = [
    "BASELINE", "ARTIFACTS", "REPORT", "WINDOWS", "FrozenSourceError", "ArtifactError", "CopySpec",
    "read_baseline", "stored_copies", "curve_sources", "copy_verified", "learning_curves", "load_results",
    "gate_table", "difference_table", "variant_table", "cost_table", "episode_table", "tier1_table", "sign_flips",
    "build_facts", "report_lines", "verify_against_report", "write_manifest", "verify_artifacts", "build_stored",
]

BASELINE = "dashboard/frozen_sources.sha256"
ARTIFACTS = "dashboard/artifacts"
REPORT = "reports/final_report.md"
PROCESSED = "data/processed"
#: Artifact window name -> the directory under ``data/processed`` that holds its evaluation.
WINDOWS = {"test": "tier2", "holdout": "holdout"}
TABLES = (
    "benchmark_ci", "benchmark_metrics", "calendar", "claims", "cost_sensitivity", "dsr", "episodes",
    "paired_block10", "paired_block20", "paired_block40", "seed_band", "turnover",
)
#: Display sleeves (DASHBOARD.md §1): the defensive assets that are not bonds, and where each variant's two
#: regime columns come from (both are named ``state_0``, ``state_1`` in the state files; D-044 item 11).
GOLD = ("GLD",)
DSR_BAR = 0.95  # preregistration_tier2 §10: "claimable" needs a median deflated Sharpe at or above this
REGIME_SOURCE = {"V4": "hmm", "C4": "vix_threshold"}
CURVE_COLUMNS = ("step", "train_mean_log_return", "val_mean_log_return", "train_mean_turnover", "val_mean_turnover",
                 "train_n_steps", "val_n_steps")


class FrozenSourceError(RuntimeError):
    """A stored source is missing from the baseline or no longer matches it."""


class ArtifactError(RuntimeError):
    """An artifact differs from the manifest, or a displayed number is not in the final report."""


@dataclass(frozen=True)
class CopySpec:
    source: str    # relative to the repository root
    artifact: str  # relative to the artifact directory


# --------------------------------------------------------------------------- #
# frozen sources -> artifacts
# --------------------------------------------------------------------------- #
def read_baseline(root: Path, baseline: str = BASELINE) -> dict[str, str]:
    """``{path relative to the repository root: sha256}`` from the ``shasum``-format baseline."""
    out: dict[str, str] = {}
    for line in (Path(root) / baseline).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            out[name] = digest
    return out


def stored_copies(variants: tuple[str, ...]) -> list[CopySpec]:
    """Every stored file copied unchanged (DASHBOARD.md §5.1, the part milestone D1 displays)."""
    specs = []
    for window, src in WINDOWS.items():
        specs.append(CopySpec(f"{PROCESSED}/{src}/eval_done.json", f"{window}/eval_done.json"))
        names = [*TABLES, *(f"seed_metrics_{v}" for v in variants)]
        specs += [CopySpec(f"{PROCESSED}/{src}/tables/{n}.csv", f"{window}/tables/{n}.csv") for n in names]
    specs.append(CopySpec(f"{PROCESSED}/holdout/replay_check.json", "holdout/replay_check.json"))
    specs.append(CopySpec(f"{PROCESSED}/tier1/gates.json", "tier1/gates.json"))
    specs.append(CopySpec(f"{PROCESSED}/tier2/chosen_configs.json", "tier2/chosen_configs.json"))
    return specs


def _checked(root: Path, source: str, baseline: dict[str, str]) -> Path:
    path = Path(root) / source
    if source not in baseline:
        raise FrozenSourceError(f"{source} is not in the frozen baseline; it cannot be shown as a stored result")
    if not path.exists():
        raise FrozenSourceError(f"{source} is missing")
    actual = sha256_file(path)
    if actual != baseline[source]:
        raise FrozenSourceError(f"{source} has SHA-256 {actual[:12]}, the frozen baseline says {baseline[source][:12]}; "
                                "a frozen result was modified, nothing is copied")
    return path


def copy_verified(root: Path, spec: CopySpec, baseline: dict[str, str], out_dir: Path) -> dict[str, str]:
    """Copy one stored file unchanged, after checking it against the baseline. Returns its manifest record."""
    src = _checked(root, spec.source, baseline)
    dst = Path(out_dir) / spec.artifact
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    digest = sha256_file(dst)
    if digest != baseline[spec.source]:
        raise ArtifactError(f"{spec.artifact} was not copied intact")
    return {"kind": "copy", "source": spec.source, "sha256": digest}


def curve_sources(root: Path, chosen: dict[str, Any], seeds: tuple[int, ...]) -> dict[tuple[str, int], str]:
    """``{(variant, seed): curve.csv path relative to the root}`` for the final runs at the frozen configurations."""
    return {
        (v, s): f"{PROCESSED}/tier2/runs/final/{v}/{rec['cfg_id']}/seed{s}/curve.csv"
        for v, rec in chosen["variants"].items() for s in seeds
    }


def learning_curves(root: Path, sources: dict[tuple[str, int], str], baseline: dict[str, str]) -> pd.DataFrame:
    """The 40 final runs' checkpoint curves in one long frame (a consolidation; no value is changed)."""
    frames = []
    for (variant, seed), source in sources.items():
        f = pd.read_csv(_checked(root, source, baseline))[list(CURVE_COLUMNS)]
        f.insert(0, "seed", seed)
        f.insert(0, "variant", variant)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# the tables the pages show: the final report's own rows, from the artifacts
# --------------------------------------------------------------------------- #
def load_results(artifacts: Path, window: str) -> dict[str, Any]:
    """One window's stored tables, read exactly as ``final_report`` reads them."""
    res = fr._load(Path(artifacts) / window)
    if res is None:
        raise ArtifactError(f"no stored evaluation for window {window!r} under {artifacts}")
    return res


def gate_table(res: dict[str, Any]) -> pd.DataFrame:
    return fr._gate_rows(res)


def difference_table(res: dict[str, Any]) -> pd.DataFrame:
    return fr._diff_rows(res)


def variant_table(res: dict[str, Any]) -> pd.DataFrame:
    return fr._variant_table(res)


def cost_table(res: dict[str, Any]) -> pd.DataFrame:
    return res["cost_sensitivity"].pivot(index="strategy", columns="bps", values="sharpe").round(2)


def episode_table(res: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for (i, s), g in res["episodes"].groupby(["episode", "strategy"], sort=False):
        g = g.iloc[0]
        rows.append({"episode": f"{i} ({g['peak']} -> {g['trough']}, {fr._pct(g['depth'])})", "strategy": s,
                     "decline": fr._pct(g["decline_return_mean"]), "recovery": fr._pct(g["recovery_return_mean"]),
                     "max dd inside": fr._pct(g["max_drawdown_in_episode_mean"])})
    return pd.DataFrame(rows)


def tier1_table(gates: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, g in gates["paired_block20"].items():
        for c in g["comparisons"]:
            rows.append({"gate": name, "comparison": f"{c['x']} vs {c['y']}", "favourable": f"{c['n_favourable']}/4",
                         "adverse": f"{c['n_adverse']}/4", "result": "pass" if c["passes"] else "fail"})
        rows.append({"gate": name, "comparison": "(gate)", "favourable": "", "adverse": "",
                     "result": "**PASS**" if g["passes"] else "**FAIL**"})
    return pd.DataFrame(rows)


def sign_flips(test: pd.DataFrame, holdout: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per comparison: the stored point difference on each window and whether its sign changed."""
    rows = []
    for x, y in fr.PAIRS:
        a = test[(test.candidate == x) & (test.control == y) & (test.metric == metric)].iloc[0]
        b = holdout[(holdout.candidate == x) & (holdout.control == y) & (holdout.metric == metric)].iloc[0]
        rows.append({"candidate": x, "control": y, "test": float(a["diff"]), "holdout": float(b["diff"]),
                     "flipped": bool((a["diff"] > 0) != (b["diff"] > 0))})
    return pd.DataFrame(rows)


def report_lines(artifacts: Path) -> list[str]:
    """Every markdown row the dashboard's headline tables correspond to, formatted as the final report formats them."""
    artifacts = Path(artifacts)
    lines = md_table(tier1_table(json.loads((artifacts / "tier1" / "gates.json").read_text()))).splitlines()
    for window in WINDOWS:
        res = load_results(artifacts, window)
        for table in (gate_table(res), difference_table(res), variant_table(res), episode_table(res)):
            lines += md_table(table).splitlines()
        lines += md_table(cost_table(res), index=True).splitlines()
    return lines


def verify_against_report(artifacts: Path, report: Path) -> dict[str, Any]:
    """Every headline row built from the artifacts must be a line of the committed final report."""
    have = set(Path(report).read_text(encoding="utf-8").splitlines())
    lines = report_lines(artifacts)
    missing = [line for line in lines if line not in have]
    if missing:
        raise ArtifactError(f"{len(missing)} dashboard rows are not in {Path(report).name}; first: {missing[0]}")
    return {"report": REPORT, "report_sha256": sha256_file(report), "rows_checked": len(lines), "ok": True}


# --------------------------------------------------------------------------- #
# facts quoted in the page copy
# --------------------------------------------------------------------------- #
def _range(values) -> list[float]:  # noqa: ANN001
    return [float(min(values)), float(max(values))]


def _window_facts(res: dict[str, Any], variants: tuple[str, ...]) -> dict[str, Any]:
    turn = res["turnover"].set_index("variant")["mean_turnover_per_step"]
    band = res["seed_band"]
    paired = res["paired_block20"]
    sharpe = paired[paired.metric == "sharpe"]
    dsr = res["dsr"].set_index("variant")
    meta = res["meta"]
    return {
        "first": meta["first"], "last": meta["last"], "n_sessions": int(meta["n_sessions"]), "evaluated_utc": meta["evaluated_utc"],
        "turnover_agents": _range(turn.loc[list(variants)]),
        "turnover_benchmarks": _range(turn.loc[[i for i in turn.index if i.startswith("BM|")]]),
        "seed_sharpe_std": _range(band[(band.metric == "sharpe") & band.variant.isin(variants)]["std"]),
        "sharpe_ci_half_width": _range((sharpe.ci_high - sharpe.ci_low) / 2.0),
        "dsr_median_max_variant": float(dsr.loc[list(variants), "dsr_median_n40"].max()),
        "seeds_ge_dsr_bar": int(dsr.loc[list(variants), "seeds_ge_0.95_n40"].sum()),
        "comparisons_passing": int(res["claims"].gate_pass.sum()),
        "claimable": int(res["claims"].claimable.sum()),
    }


def build_facts(cfg, root: Path, artifacts: Path, curves: pd.DataFrame, baseline: dict[str, str]) -> dict[str, Any]:  # noqa: ANN001
    """Configuration values and table summaries the page copy quotes, so no page hard-codes a research number."""
    from prism.agents.tier2 import HEADLINE_BPS
    from prism.splits import build_split_plan

    artifacts, d, t = Path(artifacts), cfg.data, cfg.tier2
    variants = tuple(t.variants)
    ua, ub = d.universes["A"], d.universes["B"]
    risky = list(d.allocatable[cfg.env.universe])
    sectors = list(ua.equity_sectors)
    defensive = [a for a in risky if a not in sectors]
    gold = [a for a in defensive if a in GOLD]
    plan = build_split_plan(cfg)
    chosen = json.loads((artifacts / "tier2" / "chosen_configs.json").read_text())["variants"]
    stored = lambda rel: json.loads(_checked(root, f"{PROCESSED}/{rel}", baseline).read_text(encoding="utf-8"))  # noqa: E731
    schema = {**stored("states/schema.json"), **stored("states/schema_phase_b.json")}
    encoder, hmm = stored("encoder_summary.json")["selected"], stored("hmm_summary.json")["walkforward"]
    blocks = {}
    for v in variants:
        cols = schema[v]["columns"]
        latent = sum(c.startswith("latent_") for c in cols)
        regime = sum(c.startswith("state_") for c in cols)
        blocks[v] = {"features": len(cols) - latent - regime, "latent": latent, "regime": regime,
                     "regime_source": REGIME_SOURCE.get(v) if regime else None, "portfolio": len(risky) + 2,
                     "state_width": len(cols), "observation_width": len(cols) + len(risky) + 2}
    res = {w: load_results(artifacts, w) for w in WINDOWS}
    fit_start, fit_end = d.fit_early("A")
    return {
        "universe": {
            "sectors": sectors, "bonds": [a for a in defensive if a not in gold], "gold": gold, "cash": "CASH", "n_weights": len(risky) + 1,
            "benchmark": ua.benchmark, "signals_a": list(ua.macro), "signals_b": list(ub.macro_extra),
            "excluded": dict(d.excluded), "a_start": str(ua.start), "b_start": str(ub.start),
            "fit_early": [str(fit_start.date()), str(fit_end.date())],
        },
        "rules": {
            "cap": d.allocation.weight_max, "long_only": d.allocation.long_only, "frequency": d.decision.frequency,
            "rebalance_day": d.decision.rebalance_day, "execution_lag_days": d.decision.execution_lag_days,
            "per_side_bps": cfg.env.costs.per_side_bps, "slippage_vol_coef": cfg.env.costs.slippage_vol_coef,
            "vol_window": cfg.env.costs.vol_window, "cost_levels_bps": list(cfg.env.costs.sensitivity_bps),
            "logit_scale": cfg.env.action.logit_scale, "reward": cfg.env.reward.name,
        },
        "models": {
            "hmm": {"specification": hmm["specification"], "k": int(hmm["k"]), "folds_through_test": int(hmm["n_folds"])},
            "encoder": {"variant": encoder["variant"], "window": int(encoder["window"]), "latent_dim": int(encoder["latent_dim"]),
                        "hidden_dim": int(encoder["hidden_dim"])},
        },
        "splits": {name: plan[name].as_dict() for name in ("train", "val", "test", "holdout")},
        "embargo_days": d.splits.embargo_days,
        "agents": {
            "variants": list(variants), "comparisons": [list(c) for c in t.comparisons], "seeds": len(t.final_seeds),
            "n_agents": len(variants) * len(t.final_seeds), "steps": t.training.steps, "eval_every": t.training.eval_every,
            "reward_scale": t.sac.reward_scale, "train_decisions": int(curves.train_n_steps.iloc[0]),
            "val_decisions": int(curves.val_n_steps.iloc[0]), "blocks": blocks,
            "configs": {v: {"gamma": c["gamma"], "hidden": c["hidden"], "lr": c["lr"],
                            "horizon_weeks": 1.0 / (1.0 - c["gamma"])} for v, c in chosen.items()},
        },
        "statistics": {
            "metrics": list(METRICS), "n_bootstrap": t.uncertainty.n_bootstrap, "block": t.uncertainty.block_length_days,
            "block_sensitivity": list(t.uncertainty.block_length_sensitivity), "ci_level": t.uncertainty.ci_level,
            "dsr_trials": [t.dsr_trials_headline, t.dsr_trials_all_runs], "dsr_bar": DSR_BAR, "headline_bps": HEADLINE_BPS,
        },
        "windows": {w: _window_facts(r, variants) for w, r in res.items()},
        "sign_flips": {m: sign_flips(res["test"]["paired_block20"], res["holdout"]["paired_block20"], m).to_dict("records")
                       for m in METRICS},
    }


# --------------------------------------------------------------------------- #
# manifest and the build
# --------------------------------------------------------------------------- #
def write_manifest(out_dir: Path, files: dict[str, dict[str, Any]], baseline_sha256: str) -> dict[str, Any]:
    """``manifest.json``: one record per artifact. No timestamp or commit, so a rebuild of unchanged inputs is byte-identical."""
    manifest = {"baseline": BASELINE, "baseline_sha256": baseline_sha256, "files": dict(sorted(files.items()))}
    manifest["hash"] = hash_object(manifest["files"])
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def verify_artifacts(artifacts: Path) -> dict[str, Any]:
    """Every artifact must exist and match the manifest; no unlisted file may sit beside them."""
    artifacts = Path(artifacts)
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    if hash_object(manifest["files"]) != manifest["hash"]:
        raise ArtifactError("manifest.json was edited")
    for name, rec in manifest["files"].items():
        path = artifacts / name
        if not path.exists() or sha256_file(path) != rec["sha256"]:
            raise ArtifactError(f"artifact {name} is missing or differs from the manifest; run `make dashboard-data`")
    present = {str(p.relative_to(artifacts)) for p in artifacts.rglob("*") if p.is_file() and p.name != ".DS_Store"}
    extra = present - set(manifest["files"]) - {"manifest.json"}
    if extra:
        raise ArtifactError(f"files not in the manifest: {sorted(extra)[:5]}")
    return manifest


def _write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def build_stored(cfg, root: Path, out_dir: Path | None = None) -> dict[str, Any]:  # noqa: ANN001
    """Run the stage: copy, consolidate, derive the facts, check against the report, write the manifest."""
    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / ARTIFACTS
    if (root / PROCESSED).resolve() in [out_dir.resolve(), *out_dir.resolve().parents]:
        raise ArtifactError("the artifact directory may not be inside data/processed")
    baseline = read_baseline(root)
    variants, seeds = tuple(cfg.tier2.variants), tuple(cfg.tier2.final_seeds)
    files: dict[str, dict[str, Any]] = {}
    for spec in stored_copies(variants):
        files[spec.artifact] = copy_verified(root, spec, baseline, out_dir)

    chosen = json.loads((out_dir / "tier2" / "chosen_configs.json").read_text())
    sources = curve_sources(root, chosen, seeds)
    curves = learning_curves(root, sources, baseline)
    path = out_dir / "tier2" / "learning_curves.csv"
    curves.to_csv(path, index=False, lineterminator="\n")
    files["tier2/learning_curves.csv"] = {"kind": "derived", "sha256": sha256_file(path), "n_sources": len(sources),
                                          "sources_hash": hash_object({s: baseline[s] for s in sources.values()})}

    files["facts.json"] = {"kind": "derived", "sha256": _write_json(out_dir / "facts.json", build_facts(cfg, root, out_dir, curves, baseline))}
    check = verify_against_report(out_dir, root / REPORT)
    files["report_check.json"] = {"kind": "check", "sha256": _write_json(out_dir / "report_check.json", check)}
    manifest = write_manifest(out_dir, files, sha256_file(root / BASELINE))
    verify_artifacts(out_dir)
    return {"manifest": manifest, "report_check": check}
