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
    "seed_dsr", "regime_frame", "regime_summary", "latent_pca", "parse_k_selection", "build_derived", "write_frame",
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
            "hmm": {"specification": hmm["specification"], "k": int(hmm["k"]), "folds_through_test": int(hmm["n_folds"]),
                    "observations": list(cfg.hmm.specifications[hmm["specification"]].observations),
                    "refit_cadence": cfg.hmm.walkforward.refit_cadence, "near_tie_margin": cfg.hmm.selection.near_tie_margin,
                    "min_expected_duration_days": cfg.hmm.selection.min_expected_duration_days},
            "encoder": {"variant": encoder["variant"], "window": int(encoder["window"]), "latent_dim": int(encoder["latent_dim"]),
                        "hidden_dim": int(encoder["hidden_dim"]), "refit_cadence": cfg.encoder.walkforward.refit_cadence},
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
    """``manifest.json``: one record per artifact; a stage adds or replaces its own records and keeps the others.

    No timestamp or commit, so a rebuild of unchanged inputs is byte-identical.
    """
    path = Path(out_dir) / "manifest.json"
    previous = json.loads(path.read_text(encoding="utf-8"))["files"] if path.exists() else {}
    manifest = {"baseline": BASELINE, "baseline_sha256": baseline_sha256, "files": dict(sorted({**previous, **files}.items()))}
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


# --------------------------------------------------------------------------- #
# stage "derived": series the charts need, each checked against a stored table. DASHBOARD.md §5.2
# --------------------------------------------------------------------------- #
def write_frame(frame: pd.DataFrame, path: Path) -> str:
    """Write a parquet artifact; returns its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return sha256_file(path)


def seed_dsr(daily: pd.DataFrame, variants: tuple[str, ...], seeds: tuple[int, ...], benchmarks: tuple[str, ...],
             trials: tuple[int, ...], bps: float) -> pd.DataFrame:
    """Per-seed deflated Sharpe ratio, exactly as ``prism.agents.tier2.analyse`` computes it before taking the median."""
    import numpy as np

    from prism.analysis.significance import deflated_sharpe, expected_max_sharpe, per_period_sharpe

    col = lambda name: daily[f"{name}|{bps:g}"].to_numpy()  # noqa: E731
    per_period = {(v, s): per_period_sharpe(col(f"{v}|s{s}")) for v in variants for s in seeds}
    var_sr = float(np.var(list(per_period.values()), ddof=1))
    sr0 = {k: expected_max_sharpe(k, var_sr) for k in trials}
    rows = [{"strategy": v, "seed": s, "per_period_sharpe": per_period[(v, s)],
             **{f"dsr_n{k}": deflated_sharpe(col(f"{v}|s{s}"), sr0=sr0[k]) for k in trials}} for v in variants for s in seeds]
    rows += [{"strategy": f"BM|{b}", "seed": -1, "per_period_sharpe": per_period_sharpe(col(f"BM|{b}")),
              **{f"dsr_n{k}": deflated_sharpe(col(f"BM|{b}"), sr0=sr0[k]) for k in trials}} for b in benchmarks]
    out = pd.DataFrame(rows)
    for k in trials:
        out[f"sr0_n{k}"] = sr0[k]
    return out


def _check_dsr(table: pd.DataFrame, stored: pd.DataFrame, trials: tuple[int, ...], window: str) -> None:
    for _, row in stored.iterrows():
        mine = table[table.strategy == row.variant]
        for k in trials:
            got = float(mine[f"dsr_n{k}"].median())
            if abs(got - row[f"dsr_median_n{k}"]) > 1e-12:
                raise ArtifactError(f"{window}: per-seed deflated Sharpe for {row.variant} (N={k}) has median {got}, stored {row[f'dsr_median_n{k}']}")


def _check_nav(daily: pd.DataFrame, artifacts: Path, window: str, variants: tuple[str, ...], bps: float) -> int:
    """An equity curve cumulated from the stored daily returns must end at the stored annualised return."""
    n, checked = len(daily), 0
    ann = lambda name: float((1.0 + daily[f"{name}|{bps:g}"]).prod() ** (252.0 / n) - 1.0)  # noqa: E731
    for v in variants:
        stored = pd.read_csv(Path(artifacts) / window / "tables" / f"seed_metrics_{v}.csv", index_col=0)["annualised_return"]
        for seed, want in stored.items():
            if abs(ann(f"{v}|s{seed}") - want) > 1e-9:
                raise ArtifactError(f"{window}: the equity curve of {v} seed {seed} does not end at the stored annualised return")
            checked += 1
    bench = pd.read_csv(Path(artifacts) / window / "tables" / "benchmark_metrics.csv", index_col=0)["annualised_return"]
    for name, want in bench.items():
        if abs(ann(f"BM|{name}") - want) > 1e-9:
            raise ArtifactError(f"{window}: the equity curve of benchmark {name} does not end at the stored annualised return")
        checked += 1
    return checked


def regime_frame(states: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily P(Calm), P(Volatile) from V4's HMM columns and the VIX-threshold state from C4's (both named ``state_*``)."""
    hmm, vix = states["V4"][["state_0", "state_1"]], states["C4"][["state_0", "state_1"]]
    if not hmm.index.equals(vix.index):
        raise ArtifactError("V4 and C4 are not on one index")
    return pd.DataFrame({"p_calm": hmm["state_0"], "p_volatile": hmm["state_1"], "vix_high": vix["state_1"].astype("float64")})


def regime_summary(frame: pd.DataFrame, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict[str, Any]:
    """How often the HMM's call (P(Volatile) > 0.5) and the VIX-threshold state are the same, per window. Descriptive."""
    out = {}
    for name, (start, end) in windows.items():
        f = frame.loc[start:end].dropna(subset=["vix_high"])
        volatile = f.p_volatile > 0.5
        high = f.vix_high == 1
        out[name] = {
            "start": str(f.index[0].date()), "end": str(f.index[-1].date()), "days": int(len(f)),
            "agreement": float((volatile == high).mean()), "hmm_volatile_share": float(volatile.mean()),
            "vix_high_share": float(high.mean()), "both_volatile": int((volatile & high).sum()),
            "hmm_only": int((volatile & ~high).sum()), "vix_only": int((~volatile & high).sum()), "both_calm": int((~volatile & ~high).sum()),
        }
    return out


def latent_pca(latents: pd.DataFrame, fold_of: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two principal components of the latent, fitted **separately for each encoder fold** (DECISIONS.md D-044 item 8).

    The encoder is refit every year and each refit lays the latent space out differently, so coordinates are only
    comparable within a fold. Returns the per-date coordinates and the per-fold explained variance.
    """
    from sklearn.decomposition import PCA

    coords, summary = [], []
    for fold, idx in fold_of.groupby(fold_of).groups.items():
        block = latents.loc[idx]
        pca = PCA(n_components=2, svd_solver="full").fit(block.to_numpy(dtype="float64"))
        xy = pca.transform(block.to_numpy(dtype="float64"))
        coords.append(pd.DataFrame({"fold": int(fold), "pc1": xy[:, 0], "pc2": xy[:, 1]}, index=block.index))
        summary.append({"fold": int(fold), "start": str(block.index[0].date()), "end": str(block.index[-1].date()), "days": int(len(block)),
                        "explained_1": float(pca.explained_variance_ratio_[0]), "explained_2": float(pca.explained_variance_ratio_[1])})
    return pd.concat(coords).sort_index(), pd.DataFrame(summary)


def parse_k_selection(markdown: str) -> pd.DataFrame:
    """The K-selection table of ``reports/tables/hmm_k_selection_H1.md`` as a frame (K, validation log-likelihood, BIC, ...)."""
    rows = [line for line in markdown.splitlines() if line.startswith("|")]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    out = []
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rec = dict(zip(header, cells))
        degenerate, restarts = rec["n_degenerate/n_restarts"].split("/")
        out.append({"k": int(rec["K"].split()[0]), "selected": "selected" in rec["K"], "n_params": int(rec["n_params"]),
                    "val_loglik": float(rec["val_ll (primary)"]), "bic": float(rec["BIC"]), "aic": float(rec["AIC"]),
                    "restarts": int(restarts), "restarts_degenerate": int(degenerate)})
    return pd.DataFrame(out)


def build_derived(cfg, root: Path, out_dir: Path | None = None) -> dict[str, Any]:  # noqa: ANN001
    """Stage "derived": everything the regime, latent and results charts need that the stored tables do not hold.

    Reads stored results (daily returns, state files) and, for the price line only, the snapshot's SPY closes up to
    the test split's end. Each derived series is checked against a stored table before it is written.
    """
    import numpy as np
    import yaml

    from prism.agents.tier2 import BENCHMARKS, HEADLINE_BPS
    from prism.dashboard_replay import stored_states
    from prism.data.loaders import assert_not_holdout, load_snapshot
    from prism.splits import build_split_plan, expanding_folds

    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / ARTIFACTS
    baseline = read_baseline(root)
    variants, seeds = tuple(cfg.tier2.variants), tuple(cfg.tier2.final_seeds)
    trials = (cfg.tier2.dsr_trials_headline, cfg.tier2.dsr_trials_all_runs)
    plan = build_split_plan(cfg)
    files: dict[str, dict[str, Any]] = {}
    checks: dict[str, Any] = {}

    # daily returns (copied) -> per-seed deflated Sharpe, and the equity-curve end-point check
    for window, src in WINDOWS.items():
        for name in ("eval_daily.parquet", "eval_spy.parquet"):
            spec = CopySpec(f"{PROCESSED}/{src}/{name}", f"{window}/{name}")
            files[spec.artifact] = copy_verified(root, spec, baseline, out_dir)
        daily = pd.read_parquet(out_dir / window / "eval_daily.parquet")
        res = load_results(out_dir, window)
        table = seed_dsr(daily, variants, seeds, BENCHMARKS, trials, HEADLINE_BPS)
        _check_dsr(table, res["dsr"], trials, window)
        path = out_dir / window / "seed_dsr.csv"
        table.to_csv(path, index=False, lineterminator="\n")
        files[f"{window}/seed_dsr.csv"] = {"kind": "derived", "sha256": sha256_file(path), "checked_against": f"{window}/tables/dsr.csv"}
        checks[f"{window}_equity_curves"] = _check_nav(daily, out_dir, window, variants, HEADLINE_BPS)

    # regimes and latents, from the state files the evaluations ran on
    for v in ("V2", "V4", "C4"):
        _checked(root, f"{PROCESSED}/states/{v}.parquet", baseline)
    _checked(root, f"{PROCESSED}/holdout/states_extended.parquet", baseline)
    states = stored_states(root, "holdout")                       # 2007 .. the holdout's end, one consistent source
    regimes = regime_frame(states)
    posteriors = pd.read_parquet(_checked(root, f"{PROCESSED}/hmm_posteriors.parquet", baseline))
    common = regimes.index.intersection(posteriors.index)
    d_post = float(np.abs(regimes.loc[common, "p_volatile"] - posteriors.loc[common, "state_1"]).max())
    c2 = pd.read_parquet(_checked(root, f"{PROCESSED}/states/C2.parquet", baseline))
    common_c2 = regimes.index.intersection(c2.index)
    same_c2 = bool((regimes.loc[common_c2, "vix_high"].to_numpy() == c2.loc[common_c2, "state_1"].to_numpy()).all())
    if d_post > 1e-9 or not same_c2:
        raise ArtifactError("the regime series do not match hmm_posteriors.parquet / the C2 states")
    checks["regimes"] = {"posteriors_max_abs_diff": d_post, "posterior_days_checked": int(len(common)),
                         "vix_state_equals_C2": same_c2, "vix_days_checked": int(len(common_c2))}
    # The posteriors start a few months before the state files do (the features' warm-up); keep those days for the
    # price chart's shading. They have no VIX-threshold state, so they take no part in the agreement rate.
    head = posteriors.loc[posteriors.index < regimes.index[0]]
    regimes = pd.concat([pd.DataFrame({"p_calm": head["state_0"], "p_volatile": head["state_1"], "vix_high": float("nan")}), regimes])
    files["regimes/daily.parquet"] = {"kind": "derived", "sha256": write_frame(regimes, out_dir / "regimes" / "daily.parquet")}
    windows = {name: (plan[name].effective_start, plan[name].effective_end) for name in ("train", "val", "test", "holdout")}
    windows["all"] = (regimes.dropna().index[0], regimes.index[-1])
    nber = yaml.safe_load((root / "configs" / "reference" / "nber_recessions.yaml").read_text(encoding="utf-8"))["recessions"]
    summary = {"agreement": regime_summary(regimes, windows), "nber_recessions": nber, "checks": checks["regimes"]}
    files["regimes/summary.json"] = {"kind": "derived", "sha256": _write_json(out_dir / "regimes" / "summary.json", summary)}

    # the price line: the snapshot's SPY closes to the test split's end, then the stored holdout SPY series
    snap = load_snapshot(cfg).close["SPY"].dropna()
    head = snap.loc[pd.Timestamp(cfg.data.universes["A"].start): plan["test"].declared_end]
    assert_not_holdout(cfg, head.index, context="dashboard SPY line")
    tail = pd.read_parquet(out_dir / "holdout" / "eval_spy.parquet")["SPY"]
    overlap = pd.read_parquet(out_dir / "test" / "eval_spy.parquet")["SPY"]
    if float(np.abs(head.loc[overlap.index] - overlap).max()) > 1e-9:
        raise ArtifactError("the snapshot's SPY closes differ from the stored evaluation's")
    spy = pd.concat([head, tail])
    if not spy.index.is_monotonic_increasing or spy.index.has_duplicates:
        raise ArtifactError("the SPY line is not a clean calendar")
    spy = (100.0 * spy / spy.iloc[0]).rename("spy_index").to_frame()
    files["regimes/spy.parquet"] = {"kind": "derived", "sha256": write_frame(spy, out_dir / "regimes" / "spy.parquet")}

    # latent map, one PCA per encoder fold
    latent_cols = [c for c in states["V2"].columns if c.startswith("latent_")]
    latents = states["V2"][latent_cols]
    stored_latents = pd.read_parquet(_checked(root, f"{PROCESSED}/encoder_latents.parquet", baseline))
    common = latents.index.intersection(stored_latents.index)
    d_lat = float(np.abs(latents.loc[common].to_numpy() - stored_latents.loc[common, latent_cols].to_numpy()).max())
    if d_lat > 1e-9:
        raise ArtifactError("the latent series do not match encoder_latents.parquet")
    fit_start, fit_end = cfg.data.fit_early("A")
    folds = expanding_folds(fit_start, fit_end + pd.Timedelta(days=1), plan["holdout"].declared_end, cfg.encoder.walkforward.refit_cadence,
                            embargo_days=cfg.data.splits.embargo_days)
    fold_of = pd.Series(-1, index=latents.index)
    for f in folds:
        fold_of.loc[f.apply_start: f.apply_end] = f.index
    if (fold_of < 0).any():
        raise ArtifactError("a latent date belongs to no encoder fold")
    coords, explained = latent_pca(latents, fold_of)
    coords["p_volatile"] = regimes["p_volatile"]
    files["latents/pca.parquet"] = {"kind": "derived", "sha256": write_frame(coords, out_dir / "latents" / "pca.parquet")}
    explained["fit_end"] = [str(folds[i].fit_end.date()) for i in explained.fold]
    path = out_dir / "latents" / "folds.csv"
    explained.to_csv(path, index=False, lineterminator="\n")
    files["latents/folds.csv"] = {"kind": "derived", "sha256": sha256_file(path)}
    checks["latents"] = {"max_abs_diff": d_lat, "days_checked": int(len(common)), "folds": int(len(explained))}

    # K selection, from the committed step-2 report
    k_table = parse_k_selection((root / "reports" / "tables" / "hmm_k_selection_H1.md").read_text(encoding="utf-8"))
    path = out_dir / "regimes" / "k_selection.csv"
    k_table.to_csv(path, index=False, lineterminator="\n")
    files["regimes/k_selection.csv"] = {"kind": "derived", "sha256": sha256_file(path), "source": "reports/tables/hmm_k_selection_H1.md"}
    dwell = out_dir / "regimes" / "selection_dwell.csv"
    shutil.copyfile(root / "reports" / "tables" / "hmm_selection_dwell.csv", dwell)
    files["regimes/selection_dwell.csv"] = {"kind": "copy-report", "sha256": sha256_file(dwell), "source": "reports/tables/hmm_selection_dwell.csv"}

    files["derived_check.json"] = {"kind": "check", "sha256": _write_json(out_dir / "derived_check.json", checks)}
    manifest = write_manifest(out_dir, files, sha256_file(root / BASELINE))
    verify_artifacts(out_dir)
    return {"manifest": manifest, "checks": checks}
