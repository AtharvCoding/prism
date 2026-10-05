# PRISM — final report (build step 5)

Regenerated from stored results by `make final-report` at commit `442e9db10297ee8692ee98599d9806091723c169`. Nothing here is recomputed; the holdout is never opened by this report.

## 1. Verdict

**Question (spec §0.1):** does explicit probabilistic regime conditioning (an HMM) improve portfolio allocation beyond what an LSTM representation and raw features already capture?

| Stage | Evidence | Answer |
|---|---|---|
| Tier 1, probes | HMM vs VIX threshold (V3 vs C2): 0/4 favourable | **HMM adds nothing over a threshold** |
| Tier 1, probes | LSTM vs random encoder (V2 vs C1): 1/4 favourable, 1/4 adverse | **LSTM training adds nothing over a random projection** |
| Tier 2, test 2019-2023 (exploratory) | SAC, 10 seeds per variant: V4 vs V2 FAIL, V4 vs C4 FAIL, V2 vs V1 FAIL | **no comparison passes** |
| Tier 2, holdout 2024-2026 (confirmatory, one use) | the same frozen agents: V4 vs V2 FAIL, V4 vs C4 FAIL, V2 vs V1 FAIL | **no comparison passes** |

## 2. Tier 1 (representation, no RL) — summary

Gate rule: favourable on at least 3 of 4 primary risk targets and adverse on none (paired block-bootstrap CI, block 20). Full report: `reports/tier1_report.md`.

| gate | comparison | favourable | adverse | result |
| --- | --- | --- | --- | --- |
| lstm_adds_value | V2 vs V1p | 3/4 | 0/4 | pass |
| lstm_adds_value | V2 vs C1 | 1/4 | 1/4 | fail |
| lstm_adds_value | (gate) |  |  | **FAIL** |
| hmm_adds_value | V3 vs C2 | 0/4 | 0/4 | fail |
| hmm_adds_value | V3 vs C3 | 3/4 | 0/4 | pass |
| hmm_adds_value | (gate) |  |  | **FAIL** |
| research_question | V4 vs V2 | 4/4 | 0/4 | pass |
| research_question | (gate) |  |  | **PASS** |

## 3. Tier 2 on the test split, 2019-01-08 .. 2023-11-22 (exploratory)

2019-01-08 .. 2023-11-22, 1229 sessions; the same 40 agents (10 seeds x V1, V2, V4, C4), frozen configs, the six benchmarks re-run through the same environment and costs (5 bps per side + volatility-scaled slippage).
The test split had been viewed in Tier 1 and the variant set was chosen after that, so this is exploratory.

| comparison | asks | paired rule | favourable | adverse | spec non-overlap | median DSR | claimable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V4 vs V2 | HMM adds beyond the LSTM (research question) | FAIL | 0/4 | 0/4 | fail | 0.53 | False |
| V4 vs C4 | HMM vs a VIX threshold | FAIL | 0/4 | 0/4 | fail | 0.53 | False |
| V2 vs V1 | LSTM latent adds beyond raw features | FAIL | 0/4 | 0/4 | fail | 0.55 | False |

Paired differences, candidate minus control (larger is better on every metric):

| candidate | control | metric | difference [95% CI] | verdict |
| --- | --- | --- | --- | --- |
| V4 | V2 | ann. return | -1.0% [-5.6%, 3.6%] | indeterminate |
| V4 | V2 | Sharpe | -0.08 [-0.42, 0.27] | indeterminate |
| V4 | V2 | max drawdown | -2.1% [-7.3%, 5.5%] | indeterminate |
| V4 | V2 | CVaR 95% | 0.0% [-0.2%, 0.3%] | indeterminate |
| V4 | C4 | ann. return | 0.6% [-4.7%, 5.9%] | indeterminate |
| V4 | C4 | Sharpe | 0.02 [-0.38, 0.41] | indeterminate |
| V4 | C4 | max drawdown | -1.1% [-7.5%, 7.8%] | indeterminate |
| V4 | C4 | CVaR 95% | 0.0% [-0.2%, 0.3%] | indeterminate |
| V2 | V1 | ann. return | 1.3% [-4.6%, 6.7%] | indeterminate |
| V2 | V1 | Sharpe | 0.11 [-0.25, 0.51] | indeterminate |
| V2 | V1 | max drawdown | 1.9% [-5.8%, 8.7%] | indeterminate |
| V2 | V1 | CVaR 95% | 0.0% [-0.2%, 0.3%] | indeterminate |

Seed means ± seed standard deviation, and the benchmarks:

| variant | ann. return | Sharpe | max drawdown | CVaR 95% |
| --- | --- | --- | --- | --- |
| V1 | 7.2% ± 4.7% | 0.59 ± 0.32 | -23.1% ± 5.0% | -1.9% ± 0.2% |
| V2 | 8.5% ± 3.5% | 0.71 ± 0.28 | -21.2% ± 5.5% | -1.8% ± 0.3% |
| V4 | 7.4% ± 3.3% | 0.62 ± 0.25 | -23.3% ± 4.9% | -1.8% ± 0.2% |
| C4 | 6.8% ± 4.2% | 0.60 ± 0.35 | -22.2% ± 5.1% | -1.8% ± 0.3% |
| BM EqualWeight | 10.1% | 0.74 | -25.7% | -2.1% |
| BM SixtyForty | 8.7% | 0.72 | -21.2% | -1.9% |
| BM MinVariance | 1.6% | 0.57 | -6.7% | -0.4% |
| BM RiskParity | 5.4% | 0.71 | -13.2% | -1.2% |
| BM VolTarget | 5.9% | 0.56 | -22.7% | -1.8% |
| BM BuyHoldSPY | 14.5% | 0.75 | -33.7% | -3.2% |

Drawdown episodes (SPY, >= 10%), mean over seeds for the variants:

| episode | strategy | decline | recovery | max dd inside |
| --- | --- | --- | --- | --- |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | V1 | -14.4% | 29.1% | -18.1% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | V2 | -15.6% | 31.2% | -18.7% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | V4 | -18.7% | 33.9% | -19.5% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | C4 | -15.4% | 25.9% | -17.5% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | BM|EqualWeight | -25.7% | 37.8% | -25.7% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | BM|SixtyForty | -19.4% | 29.8% | -19.4% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | BM|MinVariance | -3.2% | 5.1% | -4.8% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | BM|RiskParity | -13.2% | 18.2% | -13.2% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | BM|VolTarget | -22.7% | 9.8% | -22.7% |
| 1 (2020-02-19 -> 2020-03-23, -33.7%) | BM|BuyHoldSPY | -33.7% | 51.2% | -33.7% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | V1 | -15.5% | 8.0% | -18.5% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | V2 | -13.4% | 9.0% | -16.1% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | V4 | -13.6% | 8.1% | -16.1% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | C4 | -17.5% | 6.8% | -20.0% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | BM|EqualWeight | -14.2% | 14.9% | -15.2% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | BM|SixtyForty | -20.7% | 17.3% | -20.8% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | BM|MinVariance | -5.8% | 5.2% | -6.5% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | BM|RiskParity | -11.4% | 10.0% | -12.0% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | BM|VolTarget | -8.7% | 10.3% | -10.0% |
| 2 (2022-01-03 -> 2022-10-12, -24.5%) | BM|BuyHoldSPY | -24.5% | 29.6% | -24.5% |

Net Sharpe by cost level (bps per side):

| strategy | 0.0 | 5.0 | 10.0 | 20.0 |
| --- | --- | --- | --- | --- |
| BM|BuyHoldSPY | 0.75 | 0.75 | 0.74 | 0.74 |
| BM|EqualWeight | 0.75 | 0.74 | 0.74 | 0.73 |
| BM|MinVariance | 0.64 | 0.57 | 0.5 | 0.36 |
| BM|RiskParity | 0.72 | 0.71 | 0.7 | 0.67 |
| BM|SixtyForty | 0.73 | 0.72 | 0.72 | 0.71 |
| BM|VolTarget | 0.57 | 0.56 | 0.55 | 0.53 |
| C4 | 0.81 | 0.6 | 0.39 | -0.03 |
| V1 | 0.81 | 0.59 | 0.38 | -0.06 |
| V2 | 0.89 | 0.71 | 0.52 | 0.14 |
| V4 | 0.76 | 0.62 | 0.49 | 0.22 |

## 4. Tier 2 on the holdout (confirmatory; evaluated once)

2024-01-09 .. 2026-09-30, 684 sessions; the same 40 agents (10 seeds x V1, V2, V4, C4), frozen configs, the six benchmarks re-run through the same environment and costs (5 bps per side + volatility-scaled slippage).

| comparison | asks | paired rule | favourable | adverse | spec non-overlap | median DSR | claimable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V4 vs V2 | HMM adds beyond the LSTM (research question) | FAIL | 0/4 | 0/4 | fail | 0.74 | False |
| V4 vs C4 | HMM vs a VIX threshold | FAIL | 0/4 | 0/4 | fail | 0.74 | False |
| V2 vs V1 | LSTM latent adds beyond raw features | FAIL | 0/4 | 0/4 | fail | 0.59 | False |

Paired differences, candidate minus control (larger is better on every metric):

| candidate | control | metric | difference [95% CI] | verdict |
| --- | --- | --- | --- | --- |
| V4 | V2 | ann. return | 1.7% [-3.3%, 6.3%] | indeterminate |
| V4 | V2 | Sharpe | 0.18 [-0.31, 0.71] | indeterminate |
| V4 | V2 | max drawdown | -0.2% [-4.5%, 4.1%] | indeterminate |
| V4 | V2 | CVaR 95% | -0.0% [-0.3%, 0.2%] | indeterminate |
| V4 | C4 | ann. return | 0.6% [-4.5%, 5.3%] | indeterminate |
| V4 | C4 | Sharpe | 0.15 [-0.33, 0.64] | indeterminate |
| V4 | C4 | max drawdown | 0.4% [-3.1%, 4.1%] | indeterminate |
| V4 | C4 | CVaR 95% | 0.1% [-0.1%, 0.3%] | indeterminate |
| V2 | V1 | ann. return | -1.0% [-5.7%, 3.2%] | indeterminate |
| V2 | V1 | Sharpe | -0.14 [-0.57, 0.33] | indeterminate |
| V2 | V1 | max drawdown | -1.8% [-6.0%, 2.4%] | indeterminate |
| V2 | V1 | CVaR 95% | -0.1% [-0.3%, 0.2%] | indeterminate |

Seed means ± seed standard deviation, and the benchmarks:

| variant | ann. return | Sharpe | max drawdown | CVaR 95% |
| --- | --- | --- | --- | --- |
| V1 | 8.0% ± 3.5% | 0.91 ± 0.28 | -8.5% ± 2.5% | -1.3% ± 0.2% |
| V2 | 7.0% ± 3.3% | 0.78 ± 0.34 | -10.4% ± 3.1% | -1.3% ± 0.2% |
| V4 | 8.7% ± 3.1% | 0.95 ± 0.35 | -10.5% ± 3.4% | -1.3% ± 0.2% |
| C4 | 8.1% ± 3.5% | 0.81 ± 0.27 | -11.0% ± 2.4% | -1.4% ± 0.1% |
| BM EqualWeight | 12.6% | 1.29 | -10.2% | -1.3% |
| BM SixtyForty | 12.7% | 1.27 | -10.6% | -1.4% |
| BM MinVariance | 4.7% | 1.70 | -2.5% | -0.4% |
| BM RiskParity | 7.7% | 1.30 | -5.2% | -0.8% |
| BM VolTarget | 11.2% | 1.08 | -13.1% | -1.5% |
| BM BuyHoldSPY | 20.5% | 1.27 | -18.8% | -2.2% |

Drawdown episodes (SPY, >= 10%), mean over seeds for the variants:

| episode | strategy | decline | recovery | max dd inside |
| --- | --- | --- | --- | --- |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | V1 | -5.4% | 8.3% | -6.9% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | V2 | -6.1% | 8.6% | -7.8% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | V4 | -8.2% | 9.3% | -9.4% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | C4 | -7.8% | 9.1% | -9.1% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | BM|EqualWeight | -10.1% | 12.9% | -10.1% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | BM|SixtyForty | -10.6% | 14.4% | -10.6% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | BM|MinVariance | -0.4% | 2.9% | -1.6% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | BM|RiskParity | -5.1% | 6.7% | -5.2% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | BM|VolTarget | -12.3% | 7.8% | -12.3% |
| 1 (2025-02-19 -> 2025-04-08, -18.8%) | BM|BuyHoldSPY | -18.8% | 23.6% | -18.8% |

Net Sharpe by cost level (bps per side):

| strategy | 0.0 | 5.0 | 10.0 | 20.0 |
| --- | --- | --- | --- | --- |
| BM|BuyHoldSPY | 1.28 | 1.27 | 1.27 | 1.27 |
| BM|EqualWeight | 1.3 | 1.29 | 1.28 | 1.27 |
| BM|MinVariance | 1.78 | 1.7 | 1.61 | 1.43 |
| BM|RiskParity | 1.32 | 1.3 | 1.28 | 1.25 |
| BM|SixtyForty | 1.27 | 1.27 | 1.26 | 1.25 |
| BM|VolTarget | 1.09 | 1.08 | 1.07 | 1.04 |
| C4 | 1.06 | 0.81 | 0.56 | 0.06 |
| V1 | 1.19 | 0.91 | 0.64 | 0.09 |
| V2 | 0.99 | 0.78 | 0.56 | 0.14 |
| V4 | 1.11 | 0.95 | 0.8 | 0.49 |

## 5. Learning curves and overfitting

![learning curves](reports/figures/tier2_learning_curves.png) — mean over the 10 final seeds per variant. The train curve rises to about +0.02 per decision (in-sample memorisation of 550 weekly decisions); the validation curve stays near zero, which is why checkpoints are chosen on validation. Per-seed train-vs-validation numbers: `reports/tier2_report.md` section 8.

## 6. Reading the result

* The pre-registered outcome (spec §15.1: modest, fragile, null) is what was found. An HMM adds nothing over a two-bin VIX threshold in Tier 1 or at the policy level; the LSTM's trained latent is no better than an untrained random projection; V4 vs V2 is indeterminate on every metric.
* Between-variant differences are inside the seed-to-seed spread (Sharpe standard deviation 0.25-0.35 across seeds); no variant clears a deflated Sharpe of 0.95.
* The agents do not beat the simple benchmarks after costs; they trade 25-40% of the portfolio a week against 1-3% for the benchmarks, so their gross edge disappears at 5-10 bps.
* What was **not** tested: other algorithms, turnover-penalised rewards, other asset universes or frequencies. The null is a null for this setup.

## 7. Reproducibility statement

* Raw snapshot `snapshot_20261001`, SHA-256 9de525958b076f93d8355029c2f489f1aae9ac93428a26a23e4e2285f2d585f1 (`data/raw/snapshot_20261001/MANIFEST.json`).
* Code commit of this report: `442e9db10297ee8692ee98599d9806091723c169`. Master seed 20260101; tuning seeds [1000, 1001, 1002]; final seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
* Pre-registrations: `reports/tables/preregistration.md` (Tier 1), `preregistration_tier2.md` (Tier 2), `preregistration_holdout.md` (holdout); each committed before the data it governs was scored.
* Command sequence: `make snapshot && make phase-a` (Phase A), `make env-check`, `make tier2` (about 12 h), then once `PRISM_ALLOW_HOLDOUT=1 python scripts/99_final_holdout.py --i-am-sure`, then `make final-report`.
* The holdout access log is `reports/logs/holdout_access.jsonl`; it has one entry per opening.
