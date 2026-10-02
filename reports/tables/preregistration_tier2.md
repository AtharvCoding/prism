# PRISM — Tier 2 pre-registration (build step 4c)

**Spec:** §12, §13.2, §14, §15.1. **Written:** 2026-10-03. **Status:** committed before any tuning run, any final
run, and before the test split is read by this step. The git commit timestamp of this file is the ordering
evidence. This file never contains its own hash.

**Amendment rule.** Nothing below is edited in place after the commit. A deviation is appended under
"Amendments" as a dated entry that states what changed, why, and whether any test-split output of Tier 2 had been
seen at that point. A result produced under a deviation is labelled as such in the report. `make tier2` refuses to
run unless this file is committed and unmodified, and verifies the input hashes in §2.

---

## 0. What has already been seen (disclosure)

The test split (2019-01-01 .. 2023-11-22) is **not unseen**.

* **Tier 1 scored it** (`reports/tier1_report.md`): probes and allocators for V1, V1′, V2, V3, V4, C1, C2, C3, O1,
  the six benchmarks, and the V4-vs-V2 comparison, which passed Tier 1's paired rule at a half-point of R². Step 3's
  latent-only probe had scored it before that (`reports/tables/preregistration.md` §0).
* **The Phase B variant set was chosen after that.** V1, V2, V4 and the control C4 are the variants the research
  question needs plus the control Tier 1's V4-vs-V2 result motivated (D-032). That choice was informed by test-split
  output.
* **No agent has been scored on any split by this step.** The environment's acceptance check (step 4b) ran on the
  train split only. SAC has been trained for sanity gates and timing on train/validation data only (§5, §6).

**Consequence.** Every Tier 2 number on the test split is **exploratory**, not confirmatory, and the report says so
at the top. The only data never used in any decision is the holdout (2024-01-01 .. 2026-09-30), **reserved for
build step 5 under its own pre-registration**; nothing in step 4c reads it.

---

## 1. Question, comparisons and priors

The research question (spec §0.1, H5): *does explicit probabilistic regime conditioning add beyond a temporal
representation?* At the policy level, with a SAC agent, the three pre-specified comparisons are (candidate first):

| Comparison | Isolates | Prior |
|---|---|---|
| **V4 vs V2** | HMM posteriors on top of V1 + LSTM latent (the research question, as written) | Uncertain. Tier 1 passed it on probes at ≤ 0.6 points of R² and could not attribute it to the HMM. |
| **V4 vs C4** | HMM posteriors vs a two-column VIX threshold, both on top of V1 + LSTM latent | **Likely indeterminate.** Tier 1: V3 was indistinguishable from the threshold (C2) on every risk target. |
| **V2 vs V1** | the DAE latent on top of V1 | **Likely indeterminate or adverse.** Tier 1: an untrained random-encoder latent (C1) forecast as well as the trained one; a latent adds 32 columns to an agent that already has 184. |

**Most probable overall outcome, stated plainly:** none of the three comparisons passes; seed-to-seed variance is
larger than any between-variant difference; the agents are no better than simple benchmarks after costs, and train
performance exceeds validation performance by a wide margin (≈ 550 weekly decisions of training data, one crisis).
A null on every comparison is a conclusion, not a failure of the pipeline. Variants V1′, V3, C1, C2, C3 and O1 are
Tier 1 variants and are **not** trained here. O1 is never trained.

**Audit triggers (spec §15.1) — any of these means "re-audit for leakage before reporting":** a net Sharpe above
2.0 on the test window for any seed or benchmark; a variant's advantage over its control that is larger than the
seed spread yet traces to a variable with no information (e.g. V2 beating V1 by more than C1-type controls could
explain); the daily series not compounding to the environment's NAV (asserted in code); test-split performance far above
validation performance for the selected checkpoints.

---

## 2. Data, inputs and identity

* Raw snapshot hash (`data/raw/snapshot_20261001/MANIFEST.json`):
  `9de525958b076f93d8355029c2f489f1aae9ac93428a26a23e4e2285f2d585f1`.
* State variants: V1, V2, V4 are the files pre-registered for Tier 1 (same SHA-256); C4 was built at commit `415947d`
  (`03b_build_states.py --phase-b-only`; D-032, D-037). SHA-256 of every input, **verified by `make tier2` at run
  time** (the run aborts on any difference; a different hash means the variants were rebuilt and an amendment is
  required first):

```
11ea038692d1fe701ce42d40a167dee90eff37476fc964d59af6ce488ff04841  states/V1.parquet
4a7e0e2c651ea777d3488cb9ed8e436c1092774174532f002cba5ada5203bbad  states/V2.parquet
6d7dbd35b96a51cba114aaf775c836689c63556e2550d55c8d949c3ae5739611  states/V4.parquet
e7bec62bd291389a460c8b95927cd83699502952a78e8bbc1fc740948c729144  states/C4.parquet
b2388fa1dc392ccf7d5f6c8862ca928d816eb56319eba466dd6798393691c8c3  states/schema_phase_b.json
```

* State widths: V1 184, V2 216, V4 218, C4 218; the environment appends 14 weights and 1 mean-turnover input
  (observation widths 199 / 231 / 233 / 233, as `reports/tables/env_check.md`).
* Splits, embargo-purged effective ranges (Universe B): train 2007-04-04 .. 2017-11-22 (550 weekly decisions),
  validation 2018-01-01 .. 2018-11-21 (46), **test 2019-01-01 .. 2023-11-22**. Holdout 2024-01-01 .. 2026-09-30 is
  refused by the env builder and by `assert_not_holdout`; no step reads it.
* Seeds: master seed 20260101 (`data.seeds.master`). Each run's SAC seed is `derive_seed(master, "sac", variant,
  config id, run seed)`. Every bootstrap uses the master seed.

---

## 3. Environment

Fixed by step 4b (D-034, D-035, D-036) and not changed here: one step = one weekly decision (Friday close, a Thursday
when Friday is a holiday), executed at the next close; reward from the holding period strictly after the decision
close; 13 risky assets + cash, long-only, risky weight ≤ 0.35, full investment by projection; cost **5 bps per side
on each risky leg's traded notional plus 0.02 × trailing-20-session daily vol per unit traded**; cash earns the prior
session's `^IRX`; the portfolio starts in cash and the initiation is a trade. Training episodes: random start in the
train split, 104 decisions, truncated; evaluation episodes: the whole split, fixed.

**Reward: `log_return_net` only.** No reward sensitivity runs: `dsr`, `mv_penalty` and `drawdown_penalty` exist in the
environment and are **not trained** in Tier 2. (Cost sensitivity at 0 / 5 / 10 / 20 bps is *evaluation* of a fixed
policy under a different cost scale, not training; it is reported, §10.)

---

## 4. Agent

Soft Actor-Critic, `stable-baselines3` 2.9.0, CPU only, one torch thread per process. Identical for every variant
and every run except the three tuned values in §7:

| Setting | Value |
|---|---|
| actor and both critics | MLP, two hidden layers, width per config (§7), ReLU |
| batch size / buffer / tau | 256 / 100 000 / 0.005 |
| learning starts / train freq / gradient steps | 1 000 / 1 / 1 |
| entropy coefficient | automatic (SB3 default target) |
| **reward scale** | **10** — the reward is multiplied by 10 **during training only** (a positive scalar does not change the optimal policy); every evaluation, checkpoint score and report reads the unscaled reward |
| observations | the variant's state as standardised by step 3b's train-split scaler, plus the portfolio block; **no online normalisation** |
| actions | `Box[-1, 1]^14`, mapped to weights by the env's softmax-and-project map (`logit_scale` 3) |

---

## 5. Sanity gates

Run on train and validation only, **before any tuning run**. `make tier2` repeats them (skipping if the record in
`data/processed/tier2/sanity.json` matches the current settings) and **aborts before the test split if either fails**.
Both train for the production budget (`training.steps`, §6; attempt 3 below) with the sanity configuration γ = 0.97, hidden [64, 64],
lr 3e-4, reward scale 10, three seeds (9000, 9001, 9002) disjoint from the tuning and final seeds.

1. **Degenerate task.** A synthetic panel (3 000 sessions, ≈ 600 weekly decisions), the real 13-asset universe, real
   cost model and cap; XLK returns a constant +0.2% per day, the other twelve are zero-mean noise (1% daily), cash
   earns 0, the **state is a constant**. The optimum holds XLK at the 0.35 cap and the rest in cash. The agent is
   evaluated deterministically on a fresh noise path. **Pass iff, for every seed, the mean XLK weight ≥ 0.85 × cap and
   the mean log return ≥ 0.75 × the oracle's.**
2. **Beats random.** SAC trained as the pipeline trains (V1, the real train split, checkpoints every 5 000 steps,
   the best on validation kept). **Pass iff the validation-selected checkpoint's mean log net return exceeds the
   median of 200 i.i.d.-uniform-action random policies on the real validation split, for at least 2 of the 3 seeds.**
   Also reported, not gating: the percentile rank, the final checkpoint, the 95th-percentile comparison and the score
   of a constant equal-weight mix.

### Attempt record (each logged; nothing deleted)

Up to three standard fixes were allowed (observation normalisation, reward scaling, learning rate). They are listed
in the order run. All runs used the sanity configuration above unless stated; a "−" means not run.

| # | Change | Degenerate task | Beats random | Note |
|---|---|---|---|---|
| 0 | baseline: reward scale 1, 20 000 steps, random 16-dim states, 95th-percentile bar, final checkpoint | **fail** (XLK weight 0.26–0.35 of cap; 0.06–0.34 of oracle) | **fail** (val −0.0005 to −0.0006 per decision, 95th pct of random +0.00001) | the agent was barely off equal weight |
| 1 | reward scale 100 | **fail** (0.44–0.58 of cap) | **fail** (val −0.0018 to −0.0029; heavily overfit: train +0.011) | scale 100 learned more but overfit the real train split |
| — | exploratory, synthetic data only: scale {10, 30, 100} × lr {3e-4, 1e-3} at 20 000 steps; scale {1, 10, 30} at 60 000 and 150 000 steps | all fail | − | the gate **got worse with more training** (scale 1: 0.84 of cap at 60k steps, 0.79 at 150k). Cause: random state features let the network memorise the single noise path it trains on. A **defect in the gate's design**, not an optimiser fault. |
| design fix | gate 1: constant state, 3 000 sessions (a test-design correction, **not** one of the three fixes). Exploratory runs on it, one seed: scale 1: pass at 30k, **fail at 100k (0.60 of cap)**, pass at 200k; scale 10: pass at 30k, 100k and 200k | | | scale 10 was the stable setting |
| design fix | gate 2: the bar moves from the 95th percentile to the **median**, and the checkpoint from the final one to the validation-selected one (the pipeline's product). Reason, seen at attempt 0 and not at production length: a constant equal-weight mix scores −0.00059 per decision on validation, **below the 95th percentile of random policies (+0.00001)**, because 2018 was a falling year; a 95th-percentile bar therefore measures the market, not the learner. Gate 1 carries the burden of showing learning. This was fixed **before** the production-length run below. | | | |
| 2 | reward scale 10, 300 000 steps, checkpoint every 10 000, both gates as redesigned | **pass** (0.92–1.00 of cap; 0.84–1.05 of oracle) | **pass** (selected-checkpoint ranks 0.93 / 0.82 / 0.88 among 200 random policies) | two of three selected checkpoints were the first one; the budget was then cut to 250 000 (§6) and the cadence refined |
| 3 | **reward scale 10, the production budget: 250 000 steps, checkpoint every 5 000** | **pass** (1.00 / 1.00 / 1.00 of cap; 0.98 / 0.83 / 0.92 of oracle) | **pass** (selected-checkpoint ranks 0.93 / 0.82 / 0.92, at steps 190 000 / 10 000 / 15 000; none above the 95th percentile; final checkpoints rank 0.75 / 0.40 / 0.68) | the record `make tier2` checks (`reports/tables/tier2_sanity.md`) |

Learning rate and observation normalisation were not needed. Reward scale 10 was chosen from the exploratory runs on
the synthetic gate (which pass at 30k / 100k / 200k steps) before the production-length run. Two things the passing
record does **not** show, stated so they are not read into it: no selected checkpoint beat the 95th percentile of
random policies (the validation window is 46 decisions in a falling year, and an equal-weight mix itself ranks 0.83),
and the final checkpoints fit the train split (mean log return +0.020 per decision) far better than they predict
validation (−0.0008 to −0.0017). **Heavy overfitting is the expected behaviour of this setup, and the overfitting
report (§10) measures it for every final run.**

---

## 6. Training budget and timing

Measured on this MacBook Air M4 (10 cores), V4 state unless stated, SAC as in §4, checkpoint evaluation included:

| Setting | Steps per second per process |
|---|---|
| 1 process, [128, 128] | 412 |
| 4 processes | 162–188 |
| **6 processes** | **156** ([128, 128]) / **173** ([64, 64]) → ≈ 990 aggregate |
| 8 processes | 95–104 (79–92 with BLAS threads pinned) → ≈ 660–800 aggregate |
| 6 processes, a full 250 000-step run (V1, [64, 64], 50 checkpoints; the sanity gate) | 137 (1 818–1 833 s per run) |

Six worker processes are used (`training.workers`). Planned work: tuning 4 variants × 8 configs × 3 seeds = 96 runs,
final 4 variants × 10 seeds = 40 runs, **136 runs**. **Steps per run: 250 000** (checkpoint every 5 000: 50 checkpoints;
one checkpoint evaluation costs about 0.3 s). A run takes about 30–34 minutes with six in parallel (the full-run
measurement above, plus a [128, 128] allowance); 136 runs in 23 rounds of six ≈ **12.3 hours**, plus the sanity gates
(about 35 minutes, done) and the test evaluation and report (< 10 minutes). With a 25% derating for sustained thermal
throttling of a fanless machine, **≈ 15.4 hours**, inside the 16-hour target. The first choice of 300 000 steps was
cut to 250 000 for that margin. At 250 000 steps with a 100 000-transition buffer a run sees every train decision about
450 times; checkpoint selection on validation exists for exactly that. The estimate is replaced by measured numbers
in the final report.

---

## 7. Tuning (validation only)

**One grid, identical for every variant** (spec §12), 8 configurations: γ ∈ {0.9, 0.97} × network ∈ {[64, 64],
[128, 128]} × learning rate ∈ {3e-4, 1e-3}; everything else fixed (§4). **3 tuning seeds** per (variant,
configuration): 1000, 1001, 1002 — disjoint from the final seeds.

* A run's score is the **best-checkpoint validation score**: the highest mean per-decision `log_return_net` of the
  deterministic policy over the whole validation split, over the 50 checkpoints.
* A configuration's score is the **mean of its three seeds' scores**; the variant's configuration is the one with the
  highest mean (ties: the earlier in grid order γ → network → lr). The test split plays no part.
* The chosen configurations are written to `data/processed/tier2/chosen_configs.json` with a hash, **before any final
  run starts**; the runner refuses to overwrite a frozen choice or to read an edited one.
* The validation split is one year (46 weekly decisions); this selection is noisy and the report says so. The
  tuning table is reported in full.

---

## 8. Final runs

Each variant's frozen configuration, **10 seeds** (0 .. 9, `data.seeds.runs`), 250 000 steps. **Checkpoint selection on
validation only**: the checkpoint with the highest validation mean log net return (ties: the earlier). The final
checkpoint is also scored on train and validation for the overfitting report. The final seeds never chose a
configuration.

---

## 9. The single test evaluation, and the order of execution

`make tier2` runs, in order, and each stage is resumable (finished runs and stages are skipped; a run killed midway
is retrained from scratch):

1. **contract** — this file committed and unmodified; §2 hashes match.
2. **sanity** — §5; abort on failure, before anything reads the test split.
3. **tune** — 96 runs. 4. **freeze** — chosen configs to disk.
5. **final** — 40 runs.
6. **evaluate** — *the only stage that reads the test split*: it refuses unless sanity passed, the configurations are
   frozen and all 40 final runs are finished, and it **runs once** (a marker file stops a repeat). Each selected
   checkpoint (deterministic policy) and each benchmark is run through the environment on the test split at 0, 5, 10
   and 20 bps (the policy's actions do not depend on the cost, so the cost levels are replays of one pass). The
   outputs are daily net returns; the test split is never read again.
7. **report** — bootstrap, gates, deflated Sharpe, episodes, overfitting, `reports/tier2_report.md`, gate decisions
   appended to `DECISIONS.md`.

**Daily net returns.** The cost paid at an execution close is charged on the first session of the holding period it
opens, so the daily series compounds exactly to the environment's NAV (asserted).

**Benchmarks, re-run through the environment** with its cost model (D-034; Tier 1's allocator numbers use another
convention and are not comparable): equal-weight (1/13), 60/40 (SPY/IEF), minimum-variance, risk-parity, vol-target
(trailing realised vol, no forecast), buy-and-hold SPY — the Tier 1 weight definitions, on the environment's decision
dates. SPY is a benchmark leg only; it is not an agent asset (D-035).

---

## 10. Metrics, comparisons, the gate rule and uncertainty

**Primary metrics**, from each seed's daily net return series (headline cost 5 bps): annualised net return
(geometric, 252), net Sharpe (annualised, return not in excess of cash, as in Tier 1), maximum drawdown, and CVaR 95%
(mean of the worst 5% of days). All four are oriented so larger is better. Also reported: Sortino, Calmar, volatility,
turnover, per-seed and seed-band tables.

**A variant's statistic is the mean over its 10 seeds** of a metric. Its uncertainty has two sources, both resampled:
**days** — a stationary block bootstrap (mean block 20 sessions, circular), **2 000 replicates, seed 20260101,
shared by every variant, seed and comparison** (common random numbers) — and **seeds** — within each replicate, each
variant's 10 seeds are resampled with replacement (independently per variant, generator seeded by the master seed and
the variant name). The paired difference is the difference of the two replicate statistics.

**Gate rule (as Tier 1).** A comparison X vs Y *passes* iff the paired 95% percentile CI is **favourable on at least 3
of the 4 primary metrics** (CI entirely above zero) **and adverse on none** (CI entirely below zero). "Indeterminate"
is not a pass. The spec's non-overlapping-CI version (each variant's own CI, disjoint and favourable; same 3-of-4
aggregation) is reported alongside, in its own column; **the paired rule decides**. Block-length sensitivity (10 and
40) is reported. There is no combined "Phase B gate": the three comparisons are reported separately.

**Deflated Sharpe** (Bailey–López de Prado) per seed's daily series, per-period Sharpe. `SR₀` from the variance of the
40 final strategies' per-period Sharpe ratios, with **N = 40** (every final run is a trial) as the headline and **N = 136**
(every run trained, tuning included) as the sensitivity. Reported per variant as the median across seeds, the minimum, and
the number of seeds ≥ 0.95. **No result is described as outperforming unless the gate passes and the candidate's median
deflated Sharpe is ≥ 0.95** (the `claimable` column).

**Episodes (spec §14.3).** SPY drawdown episodes (≥ 10%), identified on SPY alone over the test window before any
agent result is consulted, with decline return, recovery return and in-episode maximum drawdown per variant (mean over
seeds, and the spread) and per benchmark; calendar years alongside. Descriptive; no CI beyond the full-window bootstrap.

**Overfitting report.** Per seed and per variant, at the validation-selected checkpoint and at the last one: mean per-decision
log net return of the deterministic policy over the **whole train split** and over validation, and their gap. The test
split plays no part.

**Cost sensitivity.** Seed-mean Sharpe and return, and the benchmarks', at 0 / 5 / 10 / 20 bps (both cost terms scale).

**Reported even if unfavourable:** the full tuning table, every seed, every comparison on every metric, the sanity
attempts, and any run that failed.

---

## 11. Not done, by decision

No reward sensitivity runs; no variants beyond V1, V2, V4, C4; no per-variant tuning budget different from the grid
above; no change to the grid, the steps, the seeds, the selection rules or the gate after the first tuning run; no
second look at the test split; no holdout. A result that is negative, or null, is reported as such.

---

## Amendments

*(none)*
