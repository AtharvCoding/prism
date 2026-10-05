# PRISM — holdout pre-registration (build step 5)

**Spec:** §0.4.5, §6.3, §13.2, §16 step 5. **Written:** 2026-10-05. **Status:** committed before the holdout is read by any
step, in any form. The git commit timestamp of this file is the ordering evidence. This file never contains its own hash.

**Amendment rule.** Nothing below is edited in place after the commit. A deviation is appended under "Amendments", dated, saying
what changed, why, and whether any holdout output had been seen. `scripts/99_final_holdout.py` refuses to run unless this file is
committed and unmodified and every file pinned in section 1 matches its SHA-256.

**Decided by the principal investigator (2026-10-05):** the holdout is spent on the **frozen Tier 2 agents**, as a confirmatory
evaluation, rather than held back for a redesigned candidate. Everything else below is my choice, made before any holdout row is read.

---

## 0. What has been seen (disclosure)

* **The holdout (2024-01-01 .. 2026-09-30) has never been read** by any step: not by Phase A, not by Tier 1, not by Tier 2.
  `reports/logs/holdout_access.jsonl` has no entry at the time of writing. The snapshot file contains those dates (it is one download), and the
  data loader returns the full snapshot; every analysis stage cut it at the test split's end and asserted so.
* **The Tier 2 result is known and null** (`reports/tier2_report.md`; DECISIONS.md "Gate decisions (Tier 2)"): on the test split no comparison
  passed (0 of 4 metrics favourable and 0 of 4 adverse in all three, at block lengths 10, 20 and 40), every variant's deflated Sharpe is about 0.5,
  and no agent beats the equal-weight, 60/40, risk-parity or SPY benchmarks on Sharpe after costs. That result was exploratory (the test split had been viewed in Tier 1).
* **Nothing about the agents, their configurations, their checkpoints, the cost model, the metrics or the gate rule may change in light of the holdout.**

---

## 1. What is evaluated, and what is frozen

The **40 trained agents** of Tier 2: variants V1, V2, V4, C4 x seeds 0-9, each at its **validation-selected checkpoint** (`best.zip`), at the
configuration frozen in `data/processed/tier2/chosen_configs.json` before the final runs. **Nothing is retrained, re-tuned, re-selected or
fine-tuned.** The six benchmarks (equal-weight, 60/40, minimum-variance, risk-parity, vol-target on trailing realised vol, buy-and-hold SPY) are
re-run through the same environment and cost model, as in Tier 2. SHA-256 of every pinned file, **verified by `99_final_holdout.py` before it
reads the holdout** (the four state files and `schema_phase_b.json` are the Tier 2 inputs; the holdout rows are produced separately, section 2):

```
11ea038692d1fe701ce42d40a167dee90eff37476fc964d59af6ce488ff04841  states/V1.parquet
4a7e0e2c651ea777d3488cb9ed8e436c1092774174532f002cba5ada5203bbad  states/V2.parquet
6d7dbd35b96a51cba114aaf775c836689c63556e2550d55c8d949c3ae5739611  states/V4.parquet
e7bec62bd291389a460c8b95927cd83699502952a78e8bbc1fc740948c729144  states/C4.parquet
b2388fa1dc392ccf7d5f6c8862ca928d816eb56319eba466dd6798393691c8c3  states/schema_phase_b.json
cdcc63dfcf6372124aca6e29db7a936423c9e148a6b184a428c0fcda94afaaf2  tier2/chosen_configs.json
79cd0c2c976fc682d8d52ad5c2afa44029a64c3c8053e4ab30e66689128814fa  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed0/best.zip
01b95019ddefeb201bfbe4e8d4b9d0ce5fb4dacb2560582efead98d5aafa28ad  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed1/best.zip
59e1c74d7571f7baf0a140d92db834abb530f60602706b8d9966b5c158a44a74  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed2/best.zip
61b3d5beaf655dfb4ddfa592cf54df63a82a9cc663301f5809f64b86815a3ed2  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed3/best.zip
a9e2092f369ddd7fe8c8b35769e8f7f7df18f8bc5af039bfb4934dc04fbff5ca  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed4/best.zip
4fae4519ba1801c075f5d8f96017dcd3161b4e8219fc1bf71174709d29e18806  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed5/best.zip
f480e5908a7caf024b70c2d09233ddce20aae4602e79a994ee01e1227fd65a54  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed6/best.zip
4a03a4674fc91b9baa2cdfee684450b06f2875b197664a67e3fbf13cbdf3b662  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed7/best.zip
d8eed3abfce8bcb94f9578fcacbea1b3398667c4533a9c21b1038975f38f7785  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed8/best.zip
173baeec0b57becce612fdd2834a8c5ef80b608ea8c657bb58fbf2f00b01608b  tier2/runs/final/C4/g0.9_n64x64_lr0.001/seed9/best.zip
bd2adbac1da8cf5236bc07e132719ace4b67c840a64a90d553c85ec013847676  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed0/best.zip
6de093bdeb534e04ce0a35d8044aced84e2d502a814e3372a503057d5c9925f2  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed1/best.zip
e789440c21295ce2581dc8d82746f2ae91b846dbfe4406a58c441febf597b8ad  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed2/best.zip
ddd27987a8630a08dc4451150112a7bd209e8e3f77b8da5a3d333172abae6967  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed3/best.zip
83166388617c5f6ba7063e7ba0306eeb7243e8113011fc8bc89bf2fb07eaae6d  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed4/best.zip
a325bb9f2f471647c3a318bdf301396d1f8ed696333e77b89778f29569b4d2d4  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed5/best.zip
0454538891d88da323f9bc6241a5fb2b1b83aee0a9672fbda13ad6f96b62a3e6  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed6/best.zip
c0bad3772e62328612b830e274969e2d90cb15af9d3680669381eccbed7a7e77  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed7/best.zip
8ef5f4b3974ae0119b36f4832283b6f8f6287ae02492b25e5617eee73d3667cb  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed8/best.zip
4b7b44667be91aca884146f0ddced84f181195223ce53249ce454f9e13a684cb  tier2/runs/final/V1/g0.97_n64x64_lr0.001/seed9/best.zip
fc71f1636433bc8d672d82833836aa2c2cccf968aaa99ac7c394434a3ad6230f  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed0/best.zip
555b2e01254397e5405f90ff78499729e148fdf45c53001297c5f16578874a5f  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed1/best.zip
81d6576479e0b9a299bdb639c19e93583788b292b7cef2118c7b3c81731dd0de  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed2/best.zip
1f7409264e80c7a4e5dfa2218a952f2d4d603bdb614457f0be3bd75c5054e4f6  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed3/best.zip
d209768432463b869bc2d74223c3c826d640ef1644dc9d47363c44d7a7ad8cca  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed4/best.zip
5ec6f93d8a0104d8c9cea05507458d924f9ec085f73c2ae61f5b47dfb22bf4ee  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed5/best.zip
f5c33de38ae206b322f6440dae287d99aa2b1e21c9fcfd4530792a498c38cf5b  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed6/best.zip
c539e7839f601c8e6f8e34c4f779f2ce63c4bc8af859202b971ca5561e4bc0a6  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed7/best.zip
4d84ccc8c2afdcd66da620f111eaf15ce1d403881ec1ee5aa89ce67ee39d02a0  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed8/best.zip
31b203e53714d519ff7e20128c30527369162266730c352b816c3ca299fb45a8  tier2/runs/final/V2/g0.97_n64x64_lr0.0003/seed9/best.zip
06d4e2a29cda3282fd646dde7f3e61d34cc1bec0fbefb8184b4fbad4d4c2ed87  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed0/best.zip
a3869b5dc7fd4be11378f5fdd2055f5f1aa093b793587d6b629fde65184553ac  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed1/best.zip
42601638653ef290955121a2b38c79869f1fea0487db9d3ebeec06b01b38908b  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed2/best.zip
e2f4a37aa8541d6ad22bf172cb4a72ccbdd785a7cda78758fe6ee801c4426542  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed3/best.zip
94c022eb7861f1fc2184f1646202bdce7741b8bd28a124d13eea6406a2503408  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed4/best.zip
6e815352b5eb994ebcba4dc41da45190447ba0c8db475951fb51ec1812e19f30  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed5/best.zip
20f0920ea0fce4b63f1aab55095b81f2775e049497b8dbc61b6c0a143dc6f645  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed6/best.zip
11938ba4de4c2908cc661c97b8457ffcb86c12e66038cf2b1648d0c080ceea01  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed7/best.zip
1f60dfa7281af02fd92ffd7d9751640112fa1f74e503523e15134f5ad08dbc77  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed8/best.zip
f16c034c9ec2759c46d28b0d4cbc0916c5fa74f9cf5b8318a457774fe166d509  tier2/runs/final/V4/g0.97_n64x64_lr0.0003/seed9/best.zip
```

---

## 2. Inputs for 2024-2026, and the replay check

Steps 1-3b were run with every date cut at the test split's end. The holdout inputs are produced **the same way, extended**: causal features
(Universes A and B), the pinned H1 / K = 2 HMM walk-forward, the selected DAE encoder walk-forward (window 10, latent 32, hidden 64), the
VIX-threshold control, the Universe-B train-split scaler, `state.py`'s assembly of V1, V2, V4 and C4. K, the HMM specification, the encoder
selection, the seeds, the fold cadences (HMM monthly, encoder annual, embargo 25), the pruning fit windows and every threshold are read from the persisted step 2-3
summaries and the committed configs; **nothing is re-selected.** `src/prism/holdout.py` implements it.

**Replay check (a hard gate, pre-registered tolerance).** The extended pipeline is run once, with the holdout included, and its rows **up to the
test split's end (2023-12-31)** must equal the stored `states/{V1,V2,V4,C4}.parquet` to **1e-9 maximum absolute difference**, with identical index and columns.
If they do not, the run aborts before any agent is evaluated and the holdout is **not evaluated**; there is no fallback and no widened tolerance.
Evidence it can pass: the same pipeline run with its end at 2023-12-31 (reading no holdout row) reproduced all four stored state files with a maximum difference of
3.2e-11 (V1, V2 and C4 exactly 0; V4 differed in the last digits, from the HMM posteriors it contains), in 5 minutes. The extension writes to `data/processed/holdout/`; no pre-registered file is touched.

---

## 3. The evaluation

Holdout window: 2024-01-01 .. 2026-09-30 (the embargo purges the end of the test split, not the start of the holdout; the last session available in the
snapshot ends it). The environment, the action map, the cost model (5 bps per side per risky leg + 0.02 x trailing-20-session daily volatility
per unit traded), the reward, the cash rate and the portfolio start (cash, initiation charged) are exactly those of Tier 2. Each checkpoint is run deterministically through one
eval-mode episode over the whole window at 0, 5, 10 and 20 bps (the policy's actions do not depend on the cost); the headline is 5 bps. The output is daily net
returns, stored once. **The holdout is evaluated exactly once**: the run writes `data/processed/holdout/eval_done.json`, and a run with that marker refuses to start. A run that
fails *before* that marker (an exception, the replay check, the contract check) evaluated nothing and may be repeated after the fault is fixed; the fix and the reason are
appended here first. Every opening of the holdout is logged permanently in `reports/logs/holdout_access.jsonl`.

---

## 4. Comparisons, priors, and the decision rule

The same three comparisons, **unchanged**, candidate first: **V4 vs V2** (the research question: HMM posteriors beyond the LSTM latent), **V4 vs C4** (the HMM vs a VIX threshold,
both on the latent), **V2 vs V1** (the LSTM latent beyond raw features).

**Priors, stated before the holdout.** All three comparisons fail. The Tier 2 point estimates were: V4 - V2 Sharpe -0.08 [-0.42, +0.27], V4 - C4 +0.02 [-0.38, +0.41],
V2 - V1 +0.11 [-0.25, +0.51]. I expect the holdout intervals to be as wide (about +-0.35 Sharpe) and to contain zero; a pass on any of the three would be surprising.
Expected for the agents against the benchmarks: no better after costs, high turnover (25-40% a week against 1-3%), large train-validation gaps (these are the same agents; no new information on them arrives except
their out-of-sample behaviour in 2024-2026).

**Decision rule (identical to Tier 2).** The four primary metrics (annualised net return, net Sharpe, maximum drawdown, CVaR 95%; all larger-is-better); a variant's statistic is the mean over its 10 seeds;
paired 95% percentile CI from the stationary block bootstrap (mean block 20, 2000 replicates, seed 20260101, day paths shared by every series, seeds resampled within each variant);
a comparison **passes** iff favourable on at least 3 of 4 metrics and adverse on none. Block lengths 10 and 40 and the spec's non-overlapping-CI version are reported. Deflated Sharpe as Tier 2
(per-seed daily series; SR0 from the 40 strategies' Sharpe variance; N = 40 headline and N = 136 sensitivity); "claimable" needs the gate **and** a median deflated Sharpe >= 0.95.

**How the result is read, in advance.**
* *All three fail on the holdout:* the Tier 2 null is **replicated out of sample**. This is the expected outcome and is reported as a finding.
* *A comparison passes on the holdout but failed on the test split:* **not a replication and not a discovery.** It is one pass among three comparisons with a prior of failure, on agents that were never
  selected for it; the report states it as an unconfirmed signal, requiring its own pre-registration and fresh data. No claim is made on it.
* *A comparison passes on both:* reported as such, subject to the claimable (deflated Sharpe) condition and the multiplicity caveat (three comparisons).
* *An adverse result* (a control beats the candidate with the CI excluding zero) is reported exactly as a favourable one would be.

**Audit triggers (spec §15.1):** a net Sharpe above 2.0 for any seed or benchmark over the full window; a daily series that does not compound to the environment's NAV (asserted); a replay-check difference above 1e-9
(a hard abort). Any of these means re-audit before reporting; none permits a re-run of the evaluation on the holdout.

---

## 5. What the report contains

`reports/final_report.md` (regenerated by `make final-report` from stored results, never opening the holdout): the verdict table across Tier 1, Tier 2 on the test split and the holdout; every paired difference with its CI on both
windows; seed distributions and benchmarks; deflated Sharpe; drawdown episodes (SPY, >= 10%, identified on SPY alone before any agent result is read) and calendar years; cost sensitivity at 0/5/10/20 bps; learning curves and the
overfitting summary; limitations; the reproducibility statement. Reported even if unfavourable: every comparison on every metric, and any failed or aborted attempt.

---

## 6. Not done, by decision

No retraining, tuning, checkpoint re-selection, variant change, cost or reward change, or gate change after the holdout is opened. No second evaluation. No reading of a holdout result to decide anything
about Tier 2. Whatever the result, the project's conclusion is the one these pre-registered rules produce.

---

## Amendments

*(none)*

### Amendment 1 — 2026-10-06 · Descriptive replay of the frozen agents for the dashboard; no second evaluation

**Holdout output seen at this point: yes, all of it.** The holdout was evaluated once, on 2026-10-05 (the single entry in
`reports/logs/holdout_access.jsonl`), and the result is known and reported: all three comparisons fail, the Tier 2 null is
replicated (`reports/final_report.md`; DECISIONS.md "Gate decisions (holdout)"). This amendment is written with that
knowledge. It changes no result and no choice, and it adds no statistic, test or decision.

**What is added.** A dashboard (`DASHBOARD.md`) displays the stored results and demonstrates the frozen system. Two things it
shows were never stored and need holdout-period rows to be read again:

1. **A one-off descriptive replay.** The 40 frozen agents (the checkpoints pinned in section 1) and the six benchmarks are run
   again, deterministically, through the same environment on the same holdout window, **solely to record what they held**: each
   week's target weights, the weights drifted to the execution close, turnover and cost. The same run re-runs the extended
   walk-forward of section 2 to persist model parameters that steps 2-3 never stored: every fold's HMM parameters, and the last
   fold's HMM, encoder, VIX-threshold breakpoints and scalers.
   * It goes through the same two keys as the evaluation (`final_holdout=True` and `PRISM_ALLOW_HOLDOUT=1`) and an explicit flag,
     is typed by hand by the principal investigator, and is logged in `reports/logs/holdout_access.jsonl` with its reason.
   * **Hard gates, checked before anything is written:** the extended states must equal the stored `states/{V1,V2,V4,C4}.parquet`
     up to 2023-12-31 and the stored `data/processed/holdout/states_extended.parquet` on every row, to 1e-9; and every replayed
     daily net return series (40 agents and 6 benchmarks, at 0, 5, 10 and 20 bps) must equal its column of
     `data/processed/holdout/eval_daily.parquet` to 1e-9. If either fails the run aborts and records nothing; there is no
     widened tolerance.
   * Because the policies, inputs and environment are identical and deterministic, the returns are the stored ones. Nothing is
     written to `data/processed/holdout/`; `eval_done.json` and every stored table are untouched; no metric, interval, test or
     gate is recomputed from the replay.
2. **The live demonstration re-reads holdout-period inputs.** To show the frozen system on data after 2026-09-30, the
   dashboard continues the *same* episode: it re-runs each agent from cash at the holdout's first decision and steps forward
   through the present, with the last-fold models frozen (no refit). Every refresh therefore reads holdout-period prices,
   features and states again. This is outside the two-key gate by construction (sessions after the holdout's end belong to no
   split, so the environment arrays are built directly) and is **not** logged per refresh. It is descriptive: the live view
   reports weights, never a performance statistic for the holdout period, and it is labelled on every page as a demonstration
   of a frozen system and not as a recommendation.

**What is unchanged.** Sections 1-6 stand as written. The holdout has been evaluated exactly once and is not evaluated again:
"evaluated" means producing a result that is compared, tested or reported, and nothing here does. Nothing about the agents,
their configurations, their checkpoints, the cost model, the metrics or the gate rule changes. The holdout remains spent: it
cannot be used to select, tune or judge anything, and any new candidate needs fresh data and its own pre-registration.

**Code.** `src/prism/holdout.py` gains output fields that expose objects it already computed (the two walk-forward results, the
train-split scaler, the Universe A features); no computed value changes, and the 1e-9 gates above are the evidence.
`src/prism/env/`, `src/prism/agents/` and `src/prism/analysis/tier2.py` are not edited. The replay is
`scripts/10_dashboard_data.py --stage holdout-replay`; decisions are in DECISIONS.md D-043 to D-047.
