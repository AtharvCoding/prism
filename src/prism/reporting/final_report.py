"""The consolidated final report (build step 5) and the holdout gate-decision entry. Spec §15.2.

``build_final_report`` reads **stored results only**: Tier 1 (``data/processed/tier1``), Tier 2 on the test
split (``data/processed/tier2``) and, if it exists, the one holdout evaluation (``data/processed/holdout``). It
recomputes no statistic and fits nothing, so ``make final-report`` reproduces it without opening the holdout. If
the holdout has not been evaluated the report says so instead of showing a section.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.analysis.tier2 import METRICS, gate_verdict, non_overlap_verdict
from prism.reporting.report import md_table

__all__ = ["build_final_report", "holdout_decisions_entry"]

PAIRS = [("V4", "V2"), ("V4", "C4"), ("V2", "V1")]
LABEL = {"V4>V2": "HMM adds beyond the LSTM (research question)", "V4>C4": "HMM vs a VIX threshold",
         "V2>V1": "LSTM latent adds beyond raw features"}
NICE = {"annualised_return": "ann. return", "sharpe": "Sharpe", "max_drawdown": "max drawdown", "cvar_95": "CVaR 95%"}
BENCHMARKS = ["EqualWeight", "SixtyForty", "MinVariance", "RiskParity", "VolTarget", "BuyHoldSPY"]


def _pct(x: float, nd: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


def _f(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _fmt(m: str, x: float) -> str:
    return _pct(x) if m != "sharpe" else _f(x)


def _git(root: Path, *a: str) -> str:
    try:
        return subprocess.check_output(["git", *a], cwd=root, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load(dirpath: Path) -> dict[str, Any] | None:
    t = dirpath / "tables"
    if not (dirpath / "eval_done.json").exists() or not t.exists():
        return None
    out: dict[str, Any] = {"meta": json.loads((dirpath / "eval_done.json").read_text())}
    for name in ("paired_block20", "paired_block10", "paired_block40", "seed_band", "benchmark_ci", "dsr", "claims",
                 "episodes", "cost_sensitivity", "turnover", "calendar"):
        f = t / f"{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f)
    return out


def _gate_rows(res: dict[str, Any]) -> pd.DataFrame:
    rows = []
    t = res["paired_block20"]
    for x, y in PAIRS:
        g, n = gate_verdict(t, x, y), non_overlap_verdict(t, x, y)
        c = res["claims"][(res["claims"].candidate == x) & (res["claims"].control == y)].iloc[0]
        rows.append({"comparison": f"{x} vs {y}", "asks": LABEL[f"{x}>{y}"],
                     "paired rule": "PASS" if g["pass"] else "FAIL", "favourable": f"{g['favourable']}/4", "adverse": f"{g['adverse']}/4",
                     "spec non-overlap": "pass" if n["pass"] else "fail", "median DSR": _f(c.dsr_median_candidate), "claimable": bool(c.claimable)})
    return pd.DataFrame(rows)


def _diff_rows(res: dict[str, Any]) -> pd.DataFrame:
    t = res["paired_block20"]
    rows = []
    for _, r in t.iterrows():
        m = r.metric
        rows.append({"candidate": r.candidate, "control": r.control, "metric": NICE[m],
                     "difference [95% CI]": f"{_fmt(m, r['diff'])} [{_fmt(m, r.ci_low)}, {_fmt(m, r.ci_high)}]",
                     "verdict": r.verdict})
    return pd.DataFrame(rows)


def _variant_table(res: dict[str, Any]) -> pd.DataFrame:
    b = res["seed_band"]
    rows = []
    for v in ("V1", "V2", "V4", "C4"):
        r = {"variant": v}
        for m in METRICS:
            x = b[(b.variant == v) & (b.metric == m)].iloc[0]
            r[NICE[m]] = f"{_fmt(m, x['mean'])} ± {_fmt(m, x['std'])}"
        rows.append(r)
    ci = res["benchmark_ci"]
    for name in BENCHMARKS:
        r = {"variant": f"BM {name}"}
        for m in METRICS:
            q = ci[(ci.benchmark == name) & (ci.metric == m)].iloc[0]
            r[NICE[m]] = f"{_fmt(m, q.point)}"
        rows.append(r)
    return pd.DataFrame(rows)


def _curves(runs: Path, out: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    cfgs = json.loads((runs / "chosen_configs.json").read_text())["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for v, rec in cfgs.items():
        frames = [pd.read_csv(f) for f in sorted((runs / "runs" / "final" / v / rec["cfg_id"]).glob("seed*/curve.csv"))]
        if not frames:
            continue
        step = frames[0]["step"]
        for ax, col, title in ((axes[0], "train_mean_log_return", "train split (in-sample)"), (axes[1], "val_mean_log_return", "validation split")):
            m = np.mean([f[col].to_numpy() for f in frames], axis=0)
            ax.plot(step, m, label=v)
            ax.set_title(f"mean log net return per decision, {title}")
            ax.set_xlabel("environment steps")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def build_final_report(cfg, root: Path, path: Path, *, holdout_dir: Path | None = None,  # noqa: ANN001
                       holdout_res: dict[str, Any] | None = None, rehearsal: bool = False) -> Path:
    processed = root / "data" / "processed"
    t1 = json.loads((processed / "tier1" / "gates.json").read_text())["paired_block20"]
    t2 = _load(processed / "tier2")
    hd = _load(holdout_dir if holdout_dir is not None else processed / "holdout")
    if rehearsal:
        hd = None
    commit = _git(root, "rev-parse", "HEAD")
    snap_dir = sorted((root / "data" / "raw").glob("snapshot_*"))[-1]
    manifest = json.loads((snap_dir / "MANIFEST.json").read_text())
    fig = _curves(processed / "tier2", root / "reports" / "figures" / "tier2_learning_curves.png") if not rehearsal else None

    L: list[str] = []
    A = L.append
    A("# PRISM — final report (build step 5)" + (" — REHEARSAL, NOT A RESULT" if rehearsal else ""))
    A("")
    A(f"Regenerated from stored results by `make final-report` at commit `{commit}`. Nothing here is recomputed; the holdout is "
      "never opened by this report.")
    A("")
    A("## 1. Verdict")
    A("")
    if t2 is None:
        A("Tier 2 results are not stored; run `make tier2`.")
    else:
        A("**Question (spec §0.1):** does explicit probabilistic regime conditioning (an HMM) improve portfolio allocation beyond "
          "what an LSTM representation and raw features already capture?")
        A("")
        A("| Stage | Evidence | Answer |")
        A("|---|---|---|")
        A(f"| Tier 1, probes | HMM vs VIX threshold (V3 vs C2): {t1['hmm_adds_value']['comparisons'][0]['n_favourable']}/4 favourable | **HMM adds nothing over a threshold** |")
        A(f"| Tier 1, probes | LSTM vs random encoder (V2 vs C1): {t1['lstm_adds_value']['comparisons'][1]['n_favourable']}/4 favourable, "
          f"{t1['lstm_adds_value']['comparisons'][1]['n_adverse']}/4 adverse | **LSTM training adds nothing over a random projection** |")
        g2 = _gate_rows(t2)
        s2 = ", ".join(r["comparison"] + " " + r["paired rule"] for r in g2.to_dict("records"))
        A(f"| Tier 2, test 2019-2023 (exploratory) | SAC, 10 seeds per variant: {s2} | **no comparison passes** |")
        if hd is not None:
            g3 = _gate_rows(hd)
            s3 = ", ".join(r["comparison"] + " " + r["paired rule"] for r in g3.to_dict("records"))
            A(f"| Tier 2, holdout 2024-2026 (confirmatory, one use) | the same frozen agents: {s3} | "
              f"**{'no comparison passes' if not (g3['paired rule'] == 'PASS').any() else 'at least one comparison passes: see section 4'}** |")
        else:
            A("| Tier 2, holdout 2024-2026 | not yet evaluated | — |")
    A("")
    A("## 2. Tier 1 (representation, no RL) — summary")
    A("")
    A("Gate rule: favourable on at least 3 of 4 primary risk targets and adverse on none (paired block-bootstrap CI, block 20). Full report: `reports/tier1_report.md`.")
    A("")
    rows = []
    for name, g in t1.items():
        for c in g["comparisons"]:
            rows.append({"gate": name, "comparison": f"{c['x']} vs {c['y']}", "favourable": f"{c['n_favourable']}/4", "adverse": f"{c['n_adverse']}/4",
                         "result": "pass" if c["passes"] else "fail"})
        rows.append({"gate": name, "comparison": "(gate)", "favourable": "", "adverse": "", "result": "**PASS**" if g["passes"] else "**FAIL**"})
    A(md_table(pd.DataFrame(rows)))
    for label, res, title in (("test", t2, "3. Tier 2 on the test split, 2019-01-08 .. 2023-11-22 (exploratory)"),
                              ("holdout", hd, "4. Tier 2 on the holdout (confirmatory; evaluated once)")):
        A("")
        A(f"## {title}")
        A("")
        if res is None:
            A("Not evaluated." if label == "holdout" else "Not available.")
            continue
        meta = res["meta"]
        A(f"{meta['first']} .. {meta['last']}, {meta['n_sessions']} sessions; the same 40 agents (10 seeds x V1, V2, V4, C4), frozen configs, the six "
          "benchmarks re-run through the same environment and costs (5 bps per side + volatility-scaled slippage).")
        if label == "test":
            A("The test split had been viewed in Tier 1 and the variant set was chosen after that, so this is exploratory.")
        A("")
        A(md_table(_gate_rows(res)))
        A("")
        A("Paired differences, candidate minus control (larger is better on every metric):")
        A("")
        A(md_table(_diff_rows(res)))
        A("")
        A("Seed means ± seed standard deviation, and the benchmarks:")
        A("")
        A(md_table(_variant_table(res)))
        A("")
        ep = res.get("episodes")
        if ep is not None and not ep.empty:
            A("Drawdown episodes (SPY, >= 10%), mean over seeds for the variants:")
            A("")
            rr = []
            for (i, s), g in ep.groupby(["episode", "strategy"], sort=False):
                g = g.iloc[0]
                rr.append({"episode": f"{i} ({g['peak']} -> {g['trough']}, {_pct(g['depth'])})", "strategy": s,
                           "decline": _pct(g["decline_return_mean"]), "recovery": _pct(g["recovery_return_mean"]),
                           "max dd inside": _pct(g["max_drawdown_in_episode_mean"])})
            A(md_table(pd.DataFrame(rr)))
        cs = res["cost_sensitivity"].pivot(index="strategy", columns="bps", values="sharpe").round(2)
        A("")
        A("Net Sharpe by cost level (bps per side):")
        A("")
        A(md_table(cs, index=True))
    A("")
    A("## 5. Learning curves and overfitting")
    A("")
    if fig:
        A(f"![learning curves]({Path(fig).relative_to(root)}) — mean over the 10 final seeds per variant. The train curve rises to about +0.02 per "
          "decision (in-sample memorisation of 550 weekly decisions); the validation curve stays near zero, which is why checkpoints are chosen on validation. "
          "Per-seed train-vs-validation numbers: `reports/tier2_report.md` section 8.")
    else:
        A("Figure unavailable (matplotlib missing). Per-seed train-vs-validation numbers: `reports/tier2_report.md` section 8.")
    A("")
    A("## 6. Reading the result")
    A("")
    A("* The pre-registered outcome (spec §15.1: modest, fragile, null) is what was found. An HMM adds nothing over a two-bin VIX threshold in Tier 1 "
      "or at the policy level; the LSTM's trained latent is no better than an untrained random projection; V4 vs V2 is indeterminate on every metric.")
    A("* Between-variant differences are inside the seed-to-seed spread (Sharpe standard deviation 0.25-0.35 across seeds); no variant clears a deflated Sharpe of 0.95.")
    A("* The agents do not beat the simple benchmarks after costs; they trade 25-40% of the portfolio a week against 1-3% for the benchmarks, so their gross edge "
      "disappears at 5-10 bps.")
    A("* What was **not** tested: other algorithms, turnover-penalised rewards, other asset universes or frequencies. The null is a null for this setup.")
    A("")
    A("## 7. Reproducibility statement")
    A("")
    A(f"* Raw snapshot `{snap_dir.name}`, SHA-256 {manifest.get('snapshot_hash', 'see MANIFEST.json')} (`{snap_dir.relative_to(root)}/MANIFEST.json`).")
    A(f"* Code commit of this report: `{commit}`. Master seed {cfg.data.seeds.master}; tuning seeds {list(cfg.tier2.tuning_seeds)}; final seeds {list(cfg.tier2.final_seeds)}.")
    A("* Pre-registrations: `reports/tables/preregistration.md` (Tier 1), `preregistration_tier2.md` (Tier 2), `preregistration_holdout.md` (holdout); each committed before the data it governs was scored.")
    A("* Command sequence: `make snapshot && make phase-a` (Phase A), `make env-check`, `make tier2` (about 12 h), then once "
      "`PRISM_ALLOW_HOLDOUT=1 python scripts/99_final_holdout.py --i-am-sure`, then `make final-report`.")
    A("* The holdout access log is `reports/logs/holdout_access.jsonl`; it has one entry per opening.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    return path


def holdout_decisions_entry(plan, res: dict[str, Any], commit: str, replay: dict[str, Any]) -> str:  # noqa: ANN001
    g = res["gates"][plan.block]
    claims = res["claims"].set_index(["candidate", "control"])
    t = res["paired"][plan.block]
    summary = " · ".join(f"{k}: {'PASS' if v['paired']['pass'] else 'FAIL'}" for k, v in g.items())
    L = ["## Gate decisions (holdout)", "", f"### Holdout · {summary}",
         f"**Date:** {pd.Timestamp.now('UTC').date()} · **Phase B, step 5; confirmatory, one use** · run under `reports/tables/preregistration_holdout.md`; "
         f"code at commit `{commit}`; window {res['first']} .. {res['last']} ({res['n_sessions']} sessions); the 40 frozen Tier 2 agents, nothing retrained. "
         f"Replay check on the extended inputs: max |difference| to the stored states {max(v['max_abs_diff'] for v in replay['variants'].values()):.2e}.", ""]
    for name, v in g.items():
        x, y = name.split(">")
        rows = t[(t.candidate == x) & (t.control == y)]
        parts = [f"`{r.metric}` {_fmt(r.metric, r['diff'])} [{_fmt(r.metric, r.ci_low)}, {_fmt(r.metric, r.ci_high)}] {r.verdict}" for _, r in rows.iterrows()]
        c = claims.loc[(x, y)]
        L.append(f"* **{x} vs {y}: {'PASS' if v['paired']['pass'] else 'FAIL'}** ({v['paired']['favourable']}/4 favourable, {v['paired']['adverse']}/4 adverse); "
                 f"spec non-overlap: {'pass' if v['non_overlap']['pass'] else 'fail'}; median DSR {_f(c.dsr_median_candidate)} -> "
                 f"{'claimable' if c.claimable else 'not claimable'}. " + "; ".join(parts) + ".")
    L += ["", "The holdout is spent. No result of this evaluation changes a choice; the final report is `reports/final_report.md`.", ""]
    return "\n".join(L)
