# PRISM — Tier 1 pre-registration (build step 4a)

**Spec:** §13.1, §14, §15.1. **Written:** 2026-10-02. **Status:** committed before any state variant is
evaluated on the test split (2019-01-01 .. 2023-11-22). The git commit timestamp of this file is the
ordering evidence. This file never contains its own hash.

**Amendment rule.** Nothing below is edited in place after the commit. A deviation is appended under
"Amendments" as a dated entry that states what changed, why, and whether any test-split output had been
seen at that point. A result produced under a deviation is labelled as such in the report.

---

## 0. What has already been seen (disclosure)

This pre-registration is not written from a blank slate. Before it, the following were viewed on data that
includes the test split. None of them changed a choice below except where stated.

| Seen | Where | Test-split exposure |
|---|---|---|
| Step 3 latent-only probe of `fwd_vol_20`, scored on the test split with the single-split `ridge_probe`: DAE latent R² −0.049, PRED latent −0.203, walk-forward PCA 0.179, random encoder 0.214 | DECISIONS.md, "The encoder does not beat PCA or the random encoder on the probe" | **Yes — one target, latents alone, no raw features.** This is the closest existing preview of the V2-vs-C1 gate. The encoder's configuration (DAE, window 10, latent 32, hidden 64) was frozen by validation reconstruction loss *before* that probe ran; the result did not alter it. |
| Step 2 HMM characterisation, detection lag vs NBER / drawdown episodes, posterior entropy, flip rate | `reports/tables/hmm_evaluation_H1.md`, `data/processed/hmm_summary.json` | **Yes — the walk-forward apply window runs through 2023-11-22.** Known: the HMM does not beat the VIX-threshold baseline on detection lag against NBER or drawdown episodes. Known: 0/204 folds fully degenerate; mean decoded run ≈ 16.65 days, max 350 (D-027). |
| HMM K-selection tables for H1 and H2, dwell-time diagnostics | `reports/tables/hmm_k_selection_*.md`, `hmm_selection_dwell.csv` | No. Fit on 1999–2006, validated on 2018. |

**Never seen:** any V1, V1′, V2, V3, V4, C1, C2, C3 or O1 probe result; any allocator or benchmark result on
the test split; anything from the holdout (2024-01-01 ..), which stays locked.

**Consequence for priors.** The latent-only preview and the HMM detection result both lean negative for the
LSTM and for the HMM. Priors in §1 are stated with that in view, not as if they were unexposed.

---

## 1. Hypotheses (spec §15.1, restated for the actual setup)

The HMM is **specification H1 (returns only, one observation dimension), K = 2** (D-024, D-025). Variants
V3, V4, C2, C3 and O1 therefore carry **2 regime columns** (`state_0`, `state_1`; canonical order: ascending
state-conditional return standard deviation, so `state_1` is the higher-volatility state). V1 has 184 base
features; V2 adds the 32-dim DAE latent; C1 adds a 32-dim frozen random-encoder latent with the **same window
(10), hidden (64) and latent (32) dimensions** as the selected encoder; V1′ is the raw **10-session** window
the encoder saw (184 + 10×184 = 2024 columns). C1 and V1′ were corrected to the encoder step 3 selected
(DECISIONS.md D-028) before this file was committed.

Spec hypotheses are numbered **Hyp-1 … Hyp-7** here to avoid colliding with the HMM specification names H1/H2.

| # | Hypothesis (spec §15.1) | Restated for this setup | Prior | Decision use |
|---|---|---|---|---|
| Hyp-1 | HMM states differ in more than volatility level | The model only observes SPY returns, so emission-space differences are mean and variance by construction. "More than volatility" can only be tested **out of model**: whether the high-vol state also differs in average pairwise sector correlation and in mean return. | Likely a calm/turbulent split with little beyond volatility. | Descriptive; no gate. |
| Hyp-2 | HMM beats a VIX/vol-threshold rule on detection lag and on probe performance | Detection-lag arm already known to **fail** (§0). Probe arm = the V3-vs-C2 comparison in the HMM gate. | Roughly even on probes; detection arm already negative. | Probe arm feeds the HMM gate (§4). |
| Hyp-3 | LSTM latent beats PCA and the random encoder on risk probes | V2 vs V1′ and V2 vs C1, on `fwd_vol_5`, `fwd_vol_20`, `fwd_max_drawdown_20`, `fwd_corr_20`. | **More likely to fail than pass**, given the latent-only preview (§0). | LSTM gate (§4). |
| Hyp-4 | LSTM latent beats baselines on return probes | V2 vs V1′ and C1 on `fwd_ret_20`. | Likely no; near-zero OOS R² is the normal outcome. | Secondary; never in a gate. |
| Hyp-5 | **V4 > V2** — regime conditioning adds beyond temporal (the research question) | Same paired rule as the gates (§4), reported as the research-question outcome. | Genuinely uncertain. | Reported; not a Phase B entry gate. |
| Hyp-6 | Regime-conditioned allocation improves risk metrics (max drawdown, CVaR) more than return metrics | Regime-conditional vol-target vs plain vol-target for the five regime-carrying variants (§7). | Likely yes if the HMM carries any signal. | Secondary, descriptive. |
| Hyp-7 | Any advantage concentrates in drawdown episodes | Per-episode allocator results (§8). | Likely yes. | Secondary, descriptive. |

**Most probable overall outcome, stated plainly:** modest, statistically fragile differences, visible in risk
metrics inside drawdown episodes and invisible in full-period Sharpe; and at least one component (probably
the LSTM) failing its gate. A null on either gate is a conclusion, not a failure of the pipeline.

**Audit triggers (spec §15.1) — any of these means "re-audit for leakage before reporting":**
OOS R² > 0.15 on `fwd_ret_20` for any variant; net-of-cost Sharpe > 2.0 on the test split for any strategy;
a regime variant's advantage that survives the C3 shuffle (V3 not beating C3 while beating V1); HMM posterior
entropy near zero.

---

## 2. Data, inputs and identity

* Raw snapshot hash (`data/raw/snapshot_20261001/MANIFEST.json`):
  `9de525958b076f93d8355029c2f489f1aae9ac93428a26a23e4e2285f2d585f1`.
* State variants are those built by `scripts/03b_build_states.py --with-oracle` against the accepted H1/K=2
  posteriors, after the C1/V1′ correction (D-028). File SHA-256 of every input:

```
63fde2e4b95bdfa63a6f8a1ffbd6ea66330c3adeb5df590ae421cdd2901de69a  hmm_posteriors.parquet
9216b146106cebb32f14233be2432e42fe734309dd2d4bf5a5024560c3cf827a  encoder_latents.parquet
ee8fe1fa9270ce1d23d2f7d34d5c122a38e67843a362d4fa8f321d3db82ed9e1  A_features.parquet
b6cee5ee6a2df93458f60c05bf7fa432545962a3b700659fd9a962566ec38b9a  B_features.parquet
11ea038692d1fe701ce42d40a167dee90eff37476fc964d59af6ce488ff04841  states/V1.parquet
a625d125edda4b246da2d04e5b1bf68c654e7e78b34bfc4b63f5a661039aff03  states/V1p.parquet
4a7e0e2c651ea777d3488cb9ed8e436c1092774174532f002cba5ada5203bbad  states/V2.parquet
7da1f972097fad68718fb65388b269e40fea0c431fb7e9c14e21cbd6eb32321b  states/V3.parquet
6d7dbd35b96a51cba114aaf775c836689c63556e2550d55c8d949c3ae5739611  states/V4.parquet
b353aefaeceae723a043221051000dbf97df5e47d039326480a4b120cf532da4  states/C1.parquet
8c3e44c83f9d34eedb1737437d037c8e03c0a61f3889299cb5728e551735488f  states/C2.parquet
3eb9738f1265f9d06c0ad171d14be71a73fb8527bd4af2566282cbe0a5a34ab3  states/C3.parquet
79cf1166095dfdc00a17d97890444de8aadb8ea3b323349e5c763272a3b486ca  states/O1.parquet
c3ce1ced5331bd4649539deaf2bf69fb62be7f9d6a09d000ab14fec8fadccb14  states/schema.json
```

  The report re-hashes these at run time and aborts if any differ. A different hash means the variants were
  rebuilt, and a new pre-registration entry (an amendment) is required first.
* Universe B throughout. Splits (embargo-purged effective ranges): train 2007-04-04 .. 2017-11-22,
  validation 2018-01-01 .. 2018-11-21, **test 2019-01-01 .. 2023-11-22**. Holdout 2024-01-01 .. 2026-09-30
  is not read by any step of 4a; loaders assert it.
* Seeds: master seed 20260101 (`data.seeds.master`). Every bootstrap in this file uses it.

---

## 3. Targets

Defined by `prism.features.targets.build_targets` on the cleaned close panel, benchmark SPY, equity-sleeve
correlation over the nine sector ETFs. Row *t* describes the window *t+1 .. t+h*.

| Target | Definition | Role |
|---|---|---|
| `fwd_vol_5` | annualised std of SPY daily log returns over *t+1..t+5* | **Primary (risk)** |
| `fwd_vol_20` | same, over *t+1..t+20* | **Primary (risk)** |
| `fwd_max_drawdown_20` | worst peak-to-trough decline of SPY over *t+1..t+20* (≤ 0) | **Primary (risk)** |
| `fwd_corr_20` | average pairwise correlation of the nine sector ETFs over *t+1..t+20* | **Primary (risk)** |
| `fwd_ret_20` | cumulative SPY log return over *t+1..t+20* | **Secondary.** Near-zero or negative OOS R² is expected and is not evidence of a broken pipeline. Never used in a gate. |

`fwd_vol_5` and `fwd_vol_20` overlap heavily in information. The gate rule in §4 is built to tolerate that.

**Secondary descriptive AUC.** Binary label `1[fwd_vol_20 > q80]`, with q80 the 80th percentile of
`fwd_vol_20` over the effective *train* split only. Classifier: L2 logistic regression, `C = 1/α` on the same
α grid as §5, α chosen by validation-fold log-loss. AUC with a block-bootstrap CI. Not used in any gate.

---

## 4. Gate rule

**Comparison unit.** For a candidate X and control Y, target τ, test-split days *t* on the common index (§5):
`d_t = e²(X,t) − e²(Y,t)`, the difference in squared forecast error, with both forecasts from the
walk-forward probe of §5. A negative mean means X forecasts better.

**Paired-difference CI (governs the decision).** The mean of `d_t` with a 95% percentile CI from the
**stationary block bootstrap** (§6), resampling **the same days for X and Y**.

* *Favourable* on τ: the CI lies entirely below zero.
* *Adverse* on τ: the CI lies entirely above zero.
* Otherwise: *indeterminate* on τ.

**A comparison passes** iff it is *favourable on at least 3 of the 4 primary risk targets* **and *adverse* on
none of the 4.** `fwd_ret_20` plays no part.

**Component gates** (config `tier1.gates`):

| Gate | Passes iff both comparisons pass |
|---|---|
| **LSTM adds value** | V2 vs V1′ **and** V2 vs C1 |
| **HMM adds value** | V3 vs C2 **and** V3 vs C3 |
| *Research question (reported, not a Phase B entry gate)* | V4 vs V2 |

A component that does not pass is **redesigned or dropped before Phase B** (spec §13.1), and the decision is
written to DECISIONS.md either way. "Indeterminate" counts as not passing.

**Spec's non-overlapping-CI version, reported alongside.** For each variant and target, a 95% CI on its own
MSE from the same bootstrap index paths. "X beats Y" in the spec's sense means the two MSE CIs are disjoint
and X's point estimate is lower. The same 3-of-4 / none-adverse aggregation is applied to it and shown in a
separate column. It is stricter and usually harder to satisfy. **The paired rule decides.** If the two
disagree, the decision text states both, and a component that passes the paired rule but fails the spec's
version is recorded as exactly that.

**Reported with every comparison, not used to decide:** ΔR² (same bootstrap, same transformation), the
per-target verdict, and CI sensitivity at mean block lengths 10 and 40.

**O1 is diagnostic only. It is never an X or a Y in any gate, never ranked against reportable variants, and
never cited as a result.** It appears in tables in a visibly separate block.

---

## 5. Probe procedure

* **Model.** Ridge regression with intercept, one per (variant, target). Features standardised with the
  moments of that fit window only. Forecasts are made at close *t*; the target starts at *t+1*.
* **Common evaluation index.** Let `I` be the intersection of the date indices of all nine state frames,
  restricted to dates on which all five targets exist. V1′ has 4182 sessions against 4191 for the others
  (its first 9 sessions have no full 10-session window), so `I` drops those 9 early sessions for everyone.
  **Every variant is fitted and scored on `I`**, so that no variant gains or loses days. Scoring is on the
  test-split dates of `I` only.
* **Walk-forward.** Annual expanding refits using `prism.splits.expanding_folds` (cadence `annual`,
  `embargo_days` 25, so the last 25 sessions before each apply window are purged from its fit). Test folds
  have apply years 2019 … 2023 (the last ending 2023-11-22). Identical fold schedule for every variant and
  target.
* **α grid — one grid for every variant and target:**
  `α ∈ {1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8}`
  (11 decades; `configs/experiments/tier1_probes.yaml`). The earlier grid topped out at 1e3, which is too
  weak for V1′ (2024 standardised columns against roughly 2,000 – 3,000 training rows, so p ≈ n). The range is set to be
  wide enough that the optimum should be interior for every variant. If a selected α sits on the grid edge for
  any variant, that is **reported as a limitation**; the grid is not widened after test-split output has been
  seen.
* **α selection — validation folds only.** The same expanding annual schedule with apply years 2015, 2016,
  2017 and 2018 (the 2018 fold ends 2018-11-21), all inside train+validation. For each (variant, target) the α
  is the grid value minimising the pooled MSE across those four folds' out-of-sample predictions; ties go to
  the larger α. **α is chosen once and held fixed through every test fold.** No test-split row enters α
  selection.
* **OOS R².** `1 − Σ(y − ŷ)² / Σ(y − ȳ_fit)²`, where `ȳ_fit` is the mean of the target over that fold's own
  fit window (the historical-mean forecast). This differs from step 3's single-split R², which centred on the
  test-split mean; the two are not directly comparable and the report says so. Gate decisions use MSE
  differences, not R².
* **Reported per variant × target:** OOS R², MSE, each with a 95% CI (§6), the selected α, and the number of
  test days.

---

## 6. Uncertainty

* **Method:** stationary block bootstrap (Politis–Romano), **mean block length 20 sessions**, 2000
  replicates, 95% percentile intervals, circular wrap, **seed 20260101**.
* **Common random numbers.** The resampled index paths depend only on (n, block length, seed). Every variant,
  target and comparison on the test split reuses the same paths, which is what makes the paired difference a
  genuine paired resample.
* **Caveat committed in advance.** Targets with horizon 20 have 20-session overlapping windows, so the loss
  differentials are serially correlated at roughly the same scale as the block length. The block-length
  sensitivity (10, 40) in §4 exists to show whether any verdict depends on that choice.
* No per-test p-values are used to make decisions. The gate is the aggregation rule in §4.

---

## 7. Allocator

Economic counterpart of the probe (spec §13.1). The classical allocators do not consume a feature vector
directly; each variant drives them **through its own probe forecast**, with the same forecaster, same α grid
and same procedure for every variant, so only the inputs differ.

**Universe and constraints.** Allocatable risky assets: XLK, XLE, XLV, XLF, XLU, XLI, XLP, XLY, XLB, SHY, IEF,
TLT, GLD (13) plus a cash line. Long-only, each weight ≤ 0.35, weights sum to 1 including cash.
**Cash accrues the prior session's ^IRX close** (percent, annualised) `/100/252` per session.

**Timing.** Weekly, decision at Friday close, execution at the next close (`execution_lag_days` 1), earned
from the execution close until the next execution (the contract in `backtest/engine.py`). First decision: the
first Friday on or after 2019-01-01. Last: the final Friday whose execution falls on or before 2023-11-22. All
strategies and benchmarks share one window. The portfolio starts in cash and the initiation is charged.

**Cost model.** Proportional cost of **5 bps per side** on one-way turnover (`0.5·Σ|Δw|`), headline net of
cost; sensitivity at 0 / 5 / 10 / 20 bps. **No volatility-scaled slippage term in Phase A** (that is the
Phase B environment's cost model, spec §11). Gross and net returns both reported.

**Forecast input.** `σ̂_v,t` = variant *v*'s walk-forward probe forecast of `fwd_vol_5` (annualised SPY
volatility), floored at 0.05. Also used: trailing 60-session realised vol of SPY, `σ_SPY,t`, and of the base
portfolio, `σ_B,t`.

**Three legs per variant (27 strategies including O1):**

1. **Vol-target (VT).** Base portfolio B = equal-weight over the nine equity sectors. Forecast portfolio vol
   `σ̂_B,v,t = σ̂_v,t · σ_B,t / σ_SPY,t`. Exposure `e_t = clip(0.10 / σ̂_B,v,t, 0, 1)`; weights = `e_t·B`
   with the remainder in cash. Target vol **10%**.
2. **Regime-conditional vol-target (RVT).** As VT, but the target is
   `σ*_t = (1 − p_t)·12% + p_t·6%`, with `p_t` = the variant's `state_1` value at *t* (the filtered
   posterior of the high-vol state for V3, V4, C3; the 0/1 indicator for C2; the *smoothed* posterior for
   O1). **For V1, V1′, V2 and C1, which carry no regime column, `σ* = 10%`, so RVT is identical to VT by
   construction.** The report states this and does not present those four RVT rows as separate evidence.
3. **Mean-variance with a cash sleeve (MV).** Maximise `μ'w − (γ/2)·w'Σ_v w` over the 13 risky weights subject
   to `0 ≤ w_i ≤ 0.35` and `Σw ≤ 1`, the remainder being cash. This is **not** fully invested, so rescaling
   the covariance changes total risky exposure. (A fully-invested mean-variance leg would have the same
   weights across variants, because a scalar rescale of Σ cancels after normalisation; that is why the cash
   sleeve exists.)
   * `Σ_v,t` = Ledoit–Wolf shrunk covariance of the 13 assets over the trailing 60 sessions, **multiplied by
     `(σ̂_v,t / σ_SPY,t)²`**.
   * `μ_t` is the same for every variant: trailing 252-session mean daily excess return over cash,
     annualised, shrunk halfway to its cross-sectional mean.
   * Risk aversion `γ` is **fixed for all variants** from `{2, 4, 8, 16, 32}`: the value for which V1's mean
     risky exposure over validation-split Friday decisions (2018-01-01 .. 2018-11-21, using V1's validation-fold
     forecasts) is closest to 0.70. It is computed once, from validation data only, before any test-split
     allocator output exists.
   * Solved by a deterministic QP solver to tolerance 1e-9; any non-optimal status aborts the run.

**Fail-loud integrity checks (the run stops, it does not warn):**

* *MV weights must differ across variants.* Over test-split decision dates, at least one pair of reportable
  variants must have mean one-way MV weight distance ≥ 0.001. Otherwise abort: the leg would be uninformative
  by construction. The full pairwise distance matrix is reported either way.
* *C3 must not be vacuous:* V3's and C3's RVT exposure series must not be identical, and V3's and C2's
  must not be identical.
* *No-regime RVT ≡ VT:* asserted equal for V1, V1′, V2, C1.
* Weight constraints (long-only, ≤ 0.35, sum 1), no NaN weights or returns, one shared decision calendar, the
  timing contract, and the holdout lock are asserted on every strategy.

**Benchmarks (spec §13.3), same window, same cost model, same weekly schedule:**

| Benchmark | Definition |
|---|---|
| Equal-weight | 1/13 in each risky asset, no cash, rebalanced weekly |
| 60/40 | 60% SPY, 40% IEF, rebalanced weekly |
| Minimum-variance | long-only, ≤ 0.35 each, fully invested including the cash line, Ledoit–Wolf 60-session covariance, no forecast |
| Risk-parity | inverse trailing-60-session volatility weights over the 13 risky assets, capped at 0.35, no cash |
| Vol-target | the VT leg with **trailing** realised portfolio vol in place of any probe forecast — the honest forecast-free competitor |
| Buy-and-hold SPY | 100% SPY from the first execution date, no rebalancing |

**Metrics (spec §14.1), all headline figures net of cost:** annualised return, annualised volatility, Sharpe,
Sortino, Calmar, maximum drawdown, drawdown duration, VaR and CVaR (95%), annualised turnover, net-vs-gross
return, hit rate, tail ratio (`backtest/metrics.py`). CIs on Sharpe, max drawdown, CVaR and annualised return
from the same bootstrap (§6) on daily net returns.

**Pre-specified allocator contrasts (secondary, descriptive; no allocator result is a gate):** within each leg,
V2−V1′, V2−C1, V3−C2, V3−C3, V4−V2, each on Sharpe, max drawdown and CVaR, with paired-difference CIs from the
shared resamples.

**Multiple-testing control (spec §14.2).** Deflated Sharpe ratio (Bailey–López de Prado, 2014) for every
strategy. The trial count is **N = 27** (9 variants × 3 legs, O1 included, deliberately conservative: many of
the 27 are correlated or identical by construction). `SR₀` is the expected maximum Sharpe of N independent
null trials, using the cross-sectional variance of the 27 per-period Sharpe estimates; skewness and kurtosis
come from each strategy's own daily net returns. A sensitivity at N = 9 is also reported. **No allocator
result is described as outperforming unless its paired-difference CI excludes zero *and* its DSR ≥ 0.95;** an
allocator result below that bar is reported as "not distinguishable from selection among N trials". Calibration
of γ (validation only) and the benchmarks are not counted as trials.

---

## 8. Episode and calendar reporting (spec §14.3)

* **Drawdown episodes** are identified **on SPY alone**, over 2019-01-01 .. 2023-11-22, before any strategy
  result is consulted: a peak, the trough, and recovery to the prior peak (or the window end if unrecovered),
  kept when the peak-to-trough decline is ≥ 10%. Per episode, per strategy and benchmark: net return over the
  decline (peak→trough), net return over the recovery (trough→recovery), and maximum drawdown inside the
  episode.
* **Calendar years** 2019 … 2023 are also reported, because §14.3 notes that calendar blocks mislead; they
  are context for the episode table, not a substitute for it.
* Per-episode counts are small. Per-episode numbers are descriptive, with no CIs claimed beyond the full-window
  bootstrap, and Hyp-7 is assessed qualitatively from them.

---

## 9. What the run must produce

1. Variant × target probe table (OOS R² and MSE with CIs; AUC secondary) and, for every gate comparison, the
   paired-difference table with the spec's non-overlapping version beside it.
2. Allocator results net of cost for all 27 strategies and the six benchmarks: full window, calendar years,
   drawdown episodes, cost sensitivity, turnover, DSR.
3. LSTM and HMM gate decisions, **pass or fail**, written to DECISIONS.md with the per-target verdicts, whether
   the spec's stricter version agrees, and V4-vs-V2.
4. The whole report regenerates from one command, with input hashes (§2), git commit, config hash and seeds
   recorded in the output. O1 appears only in a separate diagnostic block.

**Order of execution inside the run (so the test split is touched last):** α selection on validation folds →
γ calibration on validation → freeze both → test-split walk-forward probes → allocator → bootstrap and gates.

**Test-split reuse.** The test split has been looked at once already (§0) and is about to be scored. If a gate
fails and a component is redesigned, evaluating the redesign on this same test split is **exploratory**, not
confirmatory, and must say so. Confirmatory evidence for a redesigned component needs a new pre-registration
and a split that has not been scored; the only such data is the holdout, reserved for step 5.

---

## Amendments

### Amendment 1 — 2026-10-02 · O1's exact SHA-256 replaced by a numeric check

**What changed.** The §2 hash for `states/O1.parquet` (`79cf1166…`) is no longer verified byte for byte. Every
other §2 hash stays an exact SHA-256 check. `scripts/04_tier1_ablation.py` now verifies O1 as follows: it must
have the same index and columns as, and agree to within **1e-9** (maximum absolute difference) with, the frozen
reference copy `states/O1.amendment1.parquet`, whose own SHA-256 is verified exactly:

```
a631c8ab60bb83dfbf607c0afcb8352561df6ed5b850d9e1b8bda418238296ef  states/O1.amendment1.parquet
```

**Why.** O1 is built from per-fold *smoothed* HMM posteriors, which `scripts/03b_build_states.py --with-oracle`
re-fits (step 2 persists no model objects). That re-fit is not byte-reproducible: two consecutive runs on
identical inputs and seeds produced different bytes, differing by at most 2.3e-11 in value (every other state
file, including the filtered posteriors V3/V4/C2/C3 depend on, reproduced byte for byte). The 03b rebuild needed
to add the Phase B control C4 overwrote the original O1 file, so the pre-registered bytes cannot be restored; the
second rebuild's output is the reference (the first differs from it by 2.3e-11). The original is not recoverable, so the check cannot be made against it:
the evidence that the rebuilt O1 is equivalent to the original is that the stored Tier 1 results re-generate from
it (DECISIONS.md D-033).

**What is unaffected.** O1 is diagnostic-only (spec §10) and enters no gate, no pre-registered contrast and no
headline number; the eight reportable variants' files are byte-identical to the pre-registered ones.

**Test-split output seen at this point:** yes. The Tier 1 run (commit `77cc6bcd2754ec636bde4e92b98238f2a790e0b5`)
had already been scored and reported. This amendment changes nothing that run depends on except the tolerance
applied to a diagnostic input.
