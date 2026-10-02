# PRISM — Tier 1 report (step 4a, Phase A)

Generated from stored results in `data/processed/tier1/`. Pre-registration commit `242053811b9e04a667e541394b031f0c96a20035` (`reports/tables/preregistration.md`); the run followed it exactly. Test window 2019-01-01 .. 2023-11-22 (1233 sessions on the common index). The holdout was not read.

Frozen before any test-split step (`phase1.json`, 2026-10-02T08:02:34 UTC): ridge alpha per variant and target from validation folds, and mean-variance gamma = 32. Input files were verified against the pre-registered SHA-256 values (14 files).

## 1. Gate decisions

Rule (pre-registration §4): a comparison passes iff the paired-difference 95% CI is favourable on at least 3 of the 4 primary risk targets and adverse on none; a gate passes iff both its comparisons pass. `fwd_ret_20` and O1 take no part.

**Paired-difference rule (governs the decision), block 20**

- **lstm_adds_value: FAIL** (V2 vs V1p: favourable 3/4, adverse 0/4 -> pass; V2 vs C1: favourable 1/4, adverse 1/4 -> fail)
- **hmm_adds_value: FAIL** (V3 vs C2: favourable 0/4, adverse 0/4 -> fail; V3 vs C3: favourable 3/4, adverse 0/4 -> pass)
- **research_question: PASS** (V4 vs V2: favourable 4/4, adverse 0/4 -> pass)

**Spec §13.1 non-overlapping-CI version, reported alongside**

- **lstm_adds_value: FAIL** (V2 vs V1p: favourable 0/4, adverse 0/4 -> fail; V2 vs C1: favourable 0/4, adverse 0/4 -> fail)
- **hmm_adds_value: FAIL** (V3 vs C2: favourable 0/4, adverse 0/4 -> fail; V3 vs C3: favourable 0/4, adverse 0/4 -> fail)
- **research_question: FAIL** (V4 vs V2: favourable 0/4, adverse 0/4 -> fail)

**Sensitivity: paired rule, block 10**

- **lstm_adds_value: FAIL** (V2 vs V1p: favourable 3/4, adverse 0/4 -> pass; V2 vs C1: favourable 1/4, adverse 1/4 -> fail)
- **hmm_adds_value: FAIL** (V3 vs C2: favourable 0/4, adverse 0/4 -> fail; V3 vs C3: favourable 3/4, adverse 0/4 -> pass)
- **research_question: PASS** (V4 vs V2: favourable 4/4, adverse 0/4 -> pass)

**Sensitivity: paired rule, block 40**

- **lstm_adds_value: FAIL** (V2 vs V1p: favourable 3/4, adverse 0/4 -> pass; V2 vs C1: favourable 1/4, adverse 1/4 -> fail)
- **hmm_adds_value: FAIL** (V3 vs C2: favourable 0/4, adverse 0/4 -> fail; V3 vs C3: favourable 3/4, adverse 0/4 -> pass)
- **research_question: PASS** (V4 vs V2: favourable 4/4, adverse 0/4 -> pass)

## 2. Probe results: variant x target

OOS R-squared against each fold's own historical-mean forecast, with 95% stationary-block-bootstrap CIs (block 20, 2000 replicates, shared resampled days). Risk targets are primary; `fwd_ret_20` is secondary and near-zero or negative R-squared there is expected.

| index | fwd_vol_5 | fwd_vol_20 | fwd_max_drawdown_20 | fwd_corr_20 | fwd_ret_20 |
| --- | --- | --- | --- | --- | --- |
| V1 | 0.443 [0.201, 0.540] | 0.216 [-0.057, 0.379] | 0.014 [-0.112, 0.090] | -0.317 [-0.943, 0.083] | -0.000 [-0.000, 0.000] |
| V1p | 0.410 [-0.011, 0.546] | 0.168 [-0.088, 0.331] | -0.018 [-0.147, 0.062] | -0.019 [-0.376, 0.228] | -0.001 [-0.001, 0.000] |
| V2 | 0.443 [0.194, 0.542] | 0.237 [-0.025, 0.397] | 0.022 [-0.100, 0.096] | 0.109 [-0.111, 0.268] | -0.000 [-0.000, 0.000] |
| V3 | 0.446 [0.208, 0.542] | 0.221 [-0.044, 0.386] | 0.017 [-0.109, 0.095] | -0.313 [-0.938, 0.086] | -0.000 [-0.000, 0.000] |
| V4 | 0.446 [0.202, 0.544] | 0.241 [-0.016, 0.404] | 0.025 [-0.097, 0.101] | 0.115 [-0.102, 0.274] | -0.000 [-0.000, 0.000] |
| C1 | 0.461 [0.241, 0.557] | 0.247 [0.103, 0.457] | 0.018 [-0.114, 0.092] | -0.018 [-0.318, 0.198] | -0.000 [-0.000, 0.000] |
| C2 | 0.449 [0.217, 0.546] | 0.222 [-0.038, 0.388] | 0.016 [-0.111, 0.093] | -0.323 [-0.944, 0.071] | -0.000 [-0.000, 0.000] |
| C3 | 0.443 [0.201, 0.540] | 0.216 [-0.056, 0.380] | 0.014 [-0.112, 0.090] | -0.317 [-0.944, 0.083] | -0.000 [-0.000, 0.000] |
| O1 (diagnostic) | 0.464 [0.254, 0.558] | 0.238 [-0.013, 0.406] | 0.021 [-0.104, 0.099] | -0.240 [-0.799, 0.133] | -0.000 [-0.000, 0.000] |

Selected ridge alpha (validation folds only):

| variant | fwd_vol_5 | fwd_vol_20 | fwd_max_drawdown_20 | fwd_corr_20 | fwd_ret_20 |
| --- | --- | --- | --- | --- | --- |
| V1 | 10000 | 10000 | 100000 | 1000 | 1e+08 |
| V1p | 10000 | 100000 | 1e+06 | 100000 | 1e+08 |
| V2 | 10000 | 10000 | 100000 | 10000 | 1e+08 |
| V3 | 10000 | 10000 | 100000 | 1000 | 1e+08 |
| V4 | 10000 | 10000 | 100000 | 10000 | 1e+08 |
| C1 | 10000 | 100000 | 100000 | 10000 | 1e+08 |
| C2 | 10000 | 10000 | 100000 | 1000 | 1e+08 |
| C3 | 10000 | 10000 | 100000 | 1000 | 1e+08 |
| O1 | 10000 | 10000 | 100000 | 1000 | 1e+08 |

**Limitation (pre-registered):** alpha at the grid edge for V1/fwd_ret_20 (upper), V1p/fwd_ret_20 (upper), V2/fwd_ret_20 (upper), V3/fwd_ret_20 (upper), V4/fwd_ret_20 (upper), C1/fwd_ret_20 (upper), C2/fwd_ret_20 (upper), C3/fwd_ret_20 (upper), O1/fwd_ret_20 (upper).

Secondary descriptive AUC for `1[fwd_vol_20 > train-split q80]` (L2 logistic; not used in any gate):

| variant | AUC [95% CI] | alpha | n_convergence_warnings |
| --- | --- | --- | --- |
| V1 | 0.718 [0.585, 0.828] | 10000.0 | 0 |
| V1p | 0.702 [0.563, 0.820] | 100000.0 | 0 |
| V2 | 0.726 [0.596, 0.833] | 10000.0 | 0 |
| V3 | 0.729 [0.597, 0.835] | 10000.0 | 0 |
| V4 | 0.736 [0.605, 0.840] | 10000.0 | 0 |
| C1 | 0.741 [0.619, 0.844] | 10000.0 | 0 |
| C2 | 0.729 [0.602, 0.835] | 10000.0 | 0 |
| C3 | 0.718 [0.585, 0.828] | 10000.0 | 0 |
| O1 | 0.747 [0.621, 0.850] | 10000.0 | 0 |

## 3. Paired differences for every gate comparison

`d = squared error(X) - squared error(Y)`, both resampled on the same days; negative favours X. dR2 = R2(X) - R2(Y). Verdicts: favourable = CI entirely below zero; adverse = entirely above.

### V2 vs C1

| target | dR2 [95% CI] | paired verdict | MSE V2 [CI] | MSE C1 [CI] | spec non-overlap verdict |
| --- | --- | --- | --- | --- | --- |
| fwd_vol_5 | -0.0184 [-0.0377, -0.0014] | adverse | 0.01091 [0.00484, 0.02268] | 0.01055 [0.00464, 0.02205] | indeterminate |
| fwd_vol_20 | -0.0100 [-0.1099, 0.1149] | indeterminate | 0.01135 [0.00363, 0.02593] | 0.01120 [0.00298, 0.02704] | indeterminate |
| fwd_max_drawdown_20 | 0.0047 [-0.0028, 0.0121] | indeterminate | 0.00159 [0.00059, 0.00336] | 0.00159 [0.00060, 0.00336] | indeterminate |
| fwd_corr_20 | 0.1271 [0.0419, 0.2238] | favourable | 0.03207 [0.02427, 0.04056] | 0.03664 [0.02736, 0.04665] | indeterminate |
| fwd_ret_20 (secondary) | -0.0000 [-0.0000, 0.0000] | indeterminate | 0.00310 [0.00163, 0.00574] | 0.00310 [0.00163, 0.00574] | indeterminate |

### V2 vs V1p

| target | dR2 [95% CI] | paired verdict | MSE V2 [CI] | MSE V1p [CI] | spec non-overlap verdict |
| --- | --- | --- | --- | --- | --- |
| fwd_vol_5 | 0.0326 [-0.0490, 0.1138] | indeterminate | 0.01091 [0.00484, 0.02268] | 0.01155 [0.00568, 0.02197] | indeterminate |
| fwd_vol_20 | 0.0687 [0.0056, 0.1795] | favourable | 0.01135 [0.00363, 0.02593] | 0.01237 [0.00383, 0.02838] | indeterminate |
| fwd_max_drawdown_20 | 0.0406 [0.0056, 0.1010] | favourable | 0.00159 [0.00059, 0.00336] | 0.00165 [0.00060, 0.00352] | indeterminate |
| fwd_corr_20 | 0.1280 [0.0351, 0.2418] | favourable | 0.03207 [0.02427, 0.04056] | 0.03668 [0.02674, 0.04753] | indeterminate |
| fwd_ret_20 (secondary) | 0.0005 [-0.0001, 0.0013] | indeterminate | 0.00310 [0.00163, 0.00574] | 0.00310 [0.00163, 0.00574] | indeterminate |

### V3 vs C2

| target | dR2 [95% CI] | paired verdict | MSE V3 [CI] | MSE C2 [CI] | spec non-overlap verdict |
| --- | --- | --- | --- | --- | --- |
| fwd_vol_5 | -0.0027 [-0.0059, 0.0004] | indeterminate | 0.01084 [0.00471, 0.02266] | 0.01078 [0.00468, 0.02264] | indeterminate |
| fwd_vol_20 | -0.0010 [-0.0040, 0.0016] | indeterminate | 0.01158 [0.00362, 0.02655] | 0.01157 [0.00359, 0.02655] | indeterminate |
| fwd_max_drawdown_20 | 0.0013 [-0.0004, 0.0030] | indeterminate | 0.00159 [0.00059, 0.00338] | 0.00160 [0.00059, 0.00338] | indeterminate |
| fwd_corr_20 | 0.0099 [-0.0066, 0.0276] | indeterminate | 0.04727 [0.03349, 0.06358] | 0.04763 [0.03383, 0.06381] | indeterminate |
| fwd_ret_20 (secondary) | -0.0000 [-0.0000, 0.0000] | indeterminate | 0.00310 [0.00163, 0.00574] | 0.00310 [0.00163, 0.00574] | indeterminate |

### V3 vs C3

| target | dR2 [95% CI] | paired verdict | MSE V3 [CI] | MSE C3 [CI] | spec non-overlap verdict |
| --- | --- | --- | --- | --- | --- |
| fwd_vol_5 | 0.0032 [0.0014, 0.0052] | favourable | 0.01084 [0.00471, 0.02266] | 0.01090 [0.00477, 0.02271] | indeterminate |
| fwd_vol_20 | 0.0049 [0.0020, 0.0081] | favourable | 0.01158 [0.00362, 0.02655] | 0.01165 [0.00369, 0.02662] | indeterminate |
| fwd_max_drawdown_20 | 0.0029 [0.0003, 0.0059] | favourable | 0.00159 [0.00059, 0.00338] | 0.00160 [0.00059, 0.00338] | indeterminate |
| fwd_corr_20 | 0.0039 [-0.0020, 0.0105] | indeterminate | 0.04727 [0.03349, 0.06358] | 0.04741 [0.03364, 0.06386] | indeterminate |
| fwd_ret_20 (secondary) | -0.0000 [-0.0000, -0.0000] | adverse | 0.00310 [0.00163, 0.00574] | 0.00310 [0.00163, 0.00574] | indeterminate |

### V4 vs V2

| target | dR2 [95% CI] | paired verdict | MSE V4 [CI] | MSE V2 [CI] | spec non-overlap verdict |
| --- | --- | --- | --- | --- | --- |
| fwd_vol_5 | 0.0031 [0.0015, 0.0050] | favourable | 0.01085 [0.00480, 0.02263] | 0.01091 [0.00484, 0.02268] | indeterminate |
| fwd_vol_20 | 0.0044 [0.0018, 0.0074] | favourable | 0.01129 [0.00358, 0.02583] | 0.01135 [0.00363, 0.02593] | indeterminate |
| fwd_max_drawdown_20 | 0.0028 [0.0003, 0.0059] | favourable | 0.00158 [0.00059, 0.00335] | 0.00159 [0.00059, 0.00336] | indeterminate |
| fwd_corr_20 | 0.0055 [0.0020, 0.0098] | favourable | 0.03187 [0.02414, 0.04032] | 0.03207 [0.02427, 0.04056] | indeterminate |
| fwd_ret_20 (secondary) | -0.0000 [-0.0000, -0.0000] | adverse | 0.00310 [0.00163, 0.00574] | 0.00310 [0.00163, 0.00574] | indeterminate |

### Diagnostic only (O1 is never used in a gate)

| comparison | target | dR2 [95% CI] | paired verdict |
| --- | --- | --- | --- |
| O1 vs V3 | fwd_vol_5 | 0.0175 [0.0117, 0.0241] | favourable |
| O1 vs V3 | fwd_vol_20 | 0.0171 [0.0100, 0.0256] | favourable |
| O1 vs V3 | fwd_max_drawdown_20 | 0.0034 [0.0015, 0.0058] | favourable |
| O1 vs V3 | fwd_corr_20 | 0.0736 [0.0124, 0.1406] | favourable |
| O1 vs V3 | fwd_ret_20 | 0.0000 [-0.0000, 0.0000] | indeterminate |
| O1 vs V1 | fwd_vol_5 | 0.0210 [0.0135, 0.0292] | favourable |
| O1 vs V1 | fwd_vol_20 | 0.0224 [0.0124, 0.0345] | favourable |
| O1 vs V1 | fwd_max_drawdown_20 | 0.0064 [0.0021, 0.0117] | favourable |
| O1 vs V1 | fwd_corr_20 | 0.0772 [0.0116, 0.1486] | favourable |
| O1 vs V1 | fwd_ret_20 | -0.0000 [-0.0000, 0.0000] | indeterminate |

## 4. Allocator results, net of costs

Weekly Friday decisions, executed at the next close, 5 bps per side on one-way turnover (headline), 255 decisions (2019-01-04 .. 2023-11-17); strategy returns 2019-01-07 .. 2023-11-22 (1230 sessions). Sharpe is computed on total net returns (the T-bill rate is not subtracted), consistent with `backtest/metrics.py`.

### Leg: VT

| strategy | ann. return | ann. vol | Sharpe [CI] | Sortino | Calmar | max DD [CI] | DD days | CVaR95 | turnover/yr | net-vs-gross drag | hit | tail ratio | DSR (N=27) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 6.7% | 11.6% | 0.62 [-0.26, 1.52] | 0.83 | 0.40 | -16.9% [-36.0%, -9.6%] | 416 | -1.84% | 4.98 | 0.27% | 55% | 0.87 | 0.80 |
| V1p | 6.7% | 12.3% | 0.59 [-0.31, 1.48] | 0.80 | 0.40 | -16.6% [-39.3%, -10.8%] | 416 | -1.95% | 5.26 | 0.28% | 55% | 0.96 | 0.78 |
| V2 | 6.5% | 11.5% | 0.61 [-0.27, 1.51] | 0.82 | 0.39 | -16.8% [-36.1%, -9.5%] | 416 | -1.81% | 4.94 | 0.26% | 55% | 0.88 | 0.80 |
| V3 | 6.7% | 11.6% | 0.62 [-0.27, 1.52] | 0.84 | 0.40 | -16.9% [-35.9%, -9.6%] | 416 | -1.83% | 4.98 | 0.27% | 55% | 0.87 | 0.80 |
| V4 | 6.5% | 11.4% | 0.61 [-0.27, 1.51] | 0.82 | 0.39 | -16.8% [-35.9%, -9.5%] | 416 | -1.81% | 4.94 | 0.26% | 55% | 0.88 | 0.80 |
| C1 | 6.1% | 10.9% | 0.60 [-0.29, 1.51] | 0.81 | 0.37 | -16.5% [-35.1%, -9.1%] | 474 | -1.73% | 4.92 | 0.26% | 55% | 0.86 | 0.79 |
| C2 | 6.6% | 11.5% | 0.61 [-0.27, 1.51] | 0.83 | 0.39 | -17.0% [-36.0%, -9.5%] | 416 | -1.81% | 4.95 | 0.26% | 55% | 0.88 | 0.80 |
| C3 | 6.7% | 11.6% | 0.62 [-0.26, 1.52] | 0.84 | 0.40 | -16.8% [-35.9%, -9.6%] | 416 | -1.83% | 4.97 | 0.27% | 55% | 0.88 | 0.80 |

### Leg: RVT

| strategy | ann. return | ann. vol | Sharpe [CI] | Sortino | Calmar | max DD [CI] | DD days | CVaR95 | turnover/yr | net-vs-gross drag | hit | tail ratio | DSR (N=27) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1  (= VT: no regime column) | 6.7% | 11.6% | 0.62 [-0.26, 1.52] | 0.83 | 0.40 | -16.9% [-36.0%, -9.6%] | 416 | -1.84% | 4.98 | 0.27% | 55% | 0.87 | 0.80 |
| V1p  (= VT: no regime column) | 6.7% | 12.3% | 0.59 [-0.31, 1.48] | 0.80 | 0.40 | -16.6% [-39.3%, -10.8%] | 416 | -1.95% | 5.26 | 0.28% | 55% | 0.96 | 0.78 |
| V2  (= VT: no regime column) | 6.5% | 11.5% | 0.61 [-0.27, 1.51] | 0.82 | 0.39 | -16.8% [-36.1%, -9.5%] | 416 | -1.81% | 4.94 | 0.26% | 55% | 0.88 | 0.80 |
| V3 | 7.2% | 10.7% | 0.70 [-0.19, 1.62] | 0.96 | 0.50 | -14.3% [-31.2%, -8.3%] | 384 | -1.67% | 6.16 | 0.33% | 55% | 0.96 | 0.85 |
| V4 | 7.1% | 10.6% | 0.70 [-0.19, 1.62] | 0.96 | 0.50 | -14.3% [-30.9%, -8.3%] | 384 | -1.66% | 6.20 | 0.33% | 55% | 0.95 | 0.85 |
| C1  (= VT: no regime column) | 6.1% | 10.9% | 0.60 [-0.29, 1.51] | 0.81 | 0.37 | -16.5% [-35.1%, -9.1%] | 474 | -1.73% | 4.92 | 0.26% | 55% | 0.86 | 0.79 |
| C2 | 4.7% | 9.2% | 0.55 [-0.36, 1.50] | 0.74 | 0.32 | -14.8% [-32.6%, -7.3%] | 373 | -1.44% | 5.45 | 0.29% | 55% | 0.83 | 0.76 |
| C3 | 5.2% | 12.0% | 0.48 [-0.40, 1.39] | 0.65 | 0.27 | -19.0% [-41.5%, -9.8%] | 327 | -1.91% | 8.43 | 0.44% | 55% | 0.93 | 0.71 |

### Leg: MV

| strategy | ann. return | ann. vol | Sharpe [CI] | Sortino | Calmar | max DD [CI] | DD days | CVaR95 | turnover/yr | net-vs-gross drag | hit | tail ratio | DSR (N=27) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 2.8% | 7.3% | 0.42 [-0.31, 1.20] | 0.56 | 0.27 | -10.5% [-21.9%, -6.2%] | 811 | -1.24% | 10.06 | 0.52% | 58% | 0.85 | 0.66 |
| V1p | 3.3% | 8.0% | 0.44 [-0.31, 1.20] | 0.59 | 0.30 | -11.0% [-23.9%, -7.2%] | 402 | -1.37% | 11.31 | 0.59% | 57% | 0.86 | 0.68 |
| V2 | 2.7% | 7.1% | 0.41 [-0.32, 1.20] | 0.54 | 0.26 | -10.3% [-21.6%, -6.1%] | 811 | -1.21% | 9.89 | 0.51% | 58% | 0.83 | 0.66 |
| V3 | 2.9% | 7.3% | 0.43 [-0.30, 1.22] | 0.57 | 0.27 | -10.6% [-21.8%, -6.2%] | 811 | -1.24% | 10.11 | 0.52% | 58% | 0.86 | 0.67 |
| V4 | 2.8% | 7.1% | 0.42 [-0.31, 1.21] | 0.56 | 0.27 | -10.3% [-21.5%, -6.1%] | 811 | -1.21% | 9.93 | 0.51% | 58% | 0.84 | 0.66 |
| C1 | 3.0% | 6.6% | 0.49 [-0.25, 1.27] | 0.66 | 0.35 | -8.7% [-19.1%, -5.4%] | 811 | -1.10% | 9.69 | 0.50% | 58% | 0.91 | 0.72 |
| C2 | 2.7% | 7.2% | 0.40 [-0.33, 1.19] | 0.53 | 0.26 | -10.1% [-21.7%, -6.2%] | 811 | -1.22% | 10.17 | 0.52% | 58% | 0.88 | 0.65 |
| C3 | 2.8% | 7.3% | 0.42 [-0.30, 1.21] | 0.56 | 0.27 | -10.5% [-21.8%, -6.2%] | 811 | -1.24% | 10.03 | 0.52% | 58% | 0.85 | 0.67 |

### Benchmarks (same window, same costs)

| strategy | ann. return | ann. vol | Sharpe [CI] | Sortino | Calmar | max DD [CI] | DD days | CVaR95 | turnover/yr | net-vs-gross drag | hit | tail ratio | DSR (N=27) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EqualWeight | 10.2% | 14.4% | 0.74 [-0.11, 1.75] | 1.04 | 0.39 | -25.7% [-43.5%, -9.5%] | 328 | -2.14% | 0.60 | 0.03% | 55% | 0.94 | - |
| SixtyForty | 8.7% | 12.6% | 0.72 [-0.14, 1.72] | 1.01 | 0.41 | -21.2% [-37.7%, -9.6%] | 480 | -1.92% | 0.46 | 0.02% | 54% | 0.92 | - |
| MinVariance | 1.7% | 2.8% | 0.61 [-0.30, 1.65] | 0.86 | 0.26 | -6.6% [-10.8%, -2.6%] | 514 | -0.41% | 1.52 | 0.08% | 55% | 0.97 | - |
| RiskParity | 5.4% | 7.8% | 0.72 [-0.15, 1.72] | 1.00 | 0.41 | -13.2% [-26.5%, -6.1%] | 476 | -1.16% | 0.86 | 0.05% | 56% | 0.95 | - |
| VolTarget | 5.9% | 11.3% | 0.57 [-0.38, 1.60] | 0.76 | 0.26 | -22.6% [-42.8%, -8.3%] | 314 | -1.76% | 1.54 | 0.08% | 56% | 0.93 | - |
| BuyHoldSPY | 14.4% | 21.1% | 0.75 [-0.04, 1.72] | 1.04 | 0.43 | -33.7% [-55.0%, -13.9%] | 475 | -3.23% | 0.20 | 0.01% | 55% | 0.93 | - |

### Diagnostic only: O1 (smoothed, deliberately leaky; never a result)

| strategy | ann. return | ann. vol | Sharpe [CI] | Sortino | Calmar | max DD [CI] | DD days | CVaR95 | turnover/yr | net-vs-gross drag | hit | tail ratio | DSR (N=27) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O1|VT | 7.1% | 11.5% | 0.66 [-0.23, 1.57] | 0.89 | 0.43 | -16.6% [-34.8%, -9.3%] | 416 | -1.80% | 4.84 | 0.26% | 55% | 0.87 | 0.83 |
| O1|RVT | 8.6% | 10.2% | 0.86 [-0.03, 1.74] | 1.19 | 0.68 | -12.6% [-26.9%, -7.9%] | 362 | -1.57% | 5.15 | 0.28% | 55% | 0.98 | 0.92 |
| O1|MV | 3.1% | 7.2% | 0.46 [-0.28, 1.24] | 0.61 | 0.29 | -10.7% [-21.3%, -6.3%] | 811 | -1.23% | 10.16 | 0.52% | 58% | 0.87 | 0.69 |

### Cost sensitivity (net Sharpe)

| strategy | 0 bps | 5 bps | 10 bps | 20 bps |
| --- | --- | --- | --- | --- |
| V1|VT | 0.64 | 0.62 | 0.59 | 0.55 |
| V1|RVT | 0.64 | 0.62 | 0.59 | 0.55 |
| V1|MV | 0.49 | 0.42 | 0.35 | 0.21 |
| V1p|VT | 0.61 | 0.59 | 0.57 | 0.52 |
| V1p|RVT | 0.61 | 0.59 | 0.57 | 0.52 |
| V1p|MV | 0.51 | 0.44 | 0.37 | 0.23 |
| V2|VT | 0.63 | 0.61 | 0.58 | 0.54 |
| V2|RVT | 0.63 | 0.61 | 0.58 | 0.54 |
| V2|MV | 0.48 | 0.41 | 0.34 | 0.20 |
| V3|VT | 0.64 | 0.62 | 0.60 | 0.56 |
| V3|RVT | 0.73 | 0.70 | 0.67 | 0.62 |
| V3|MV | 0.50 | 0.43 | 0.36 | 0.22 |
| V4|VT | 0.63 | 0.61 | 0.59 | 0.54 |
| V4|RVT | 0.73 | 0.70 | 0.67 | 0.61 |
| V4|MV | 0.49 | 0.42 | 0.35 | 0.21 |
| C1|VT | 0.62 | 0.60 | 0.58 | 0.53 |
| C1|RVT | 0.62 | 0.60 | 0.58 | 0.53 |
| C1|MV | 0.56 | 0.49 | 0.42 | 0.27 |
| C2|VT | 0.63 | 0.61 | 0.59 | 0.55 |
| C2|RVT | 0.57 | 0.55 | 0.52 | 0.46 |
| C2|MV | 0.47 | 0.40 | 0.33 | 0.19 |
| C3|VT | 0.64 | 0.62 | 0.60 | 0.55 |
| C3|RVT | 0.52 | 0.48 | 0.45 | 0.38 |
| C3|MV | 0.49 | 0.42 | 0.35 | 0.22 |
| BM|EqualWeight | 0.75 | 0.74 | 0.74 | 0.74 |
| BM|SixtyForty | 0.73 | 0.72 | 0.72 | 0.72 |
| BM|MinVariance | 0.64 | 0.61 | 0.59 | 0.53 |
| BM|RiskParity | 0.72 | 0.72 | 0.71 | 0.70 |
| BM|VolTarget | 0.57 | 0.57 | 0.56 | 0.54 |
| BM|BuyHoldSPY | 0.75 | 0.75 | 0.75 | 0.74 |

### Average risky exposure (1 - cash)

| index | mean risky exposure |
| --- | --- |
| V1|VT | 0.76 |
| V1|RVT | 0.76 |
| V1|MV | 0.68 |
| V1p|VT | 0.78 |
| V1p|RVT | 0.78 |
| V1p|MV | 0.68 |
| V2|VT | 0.74 |
| V2|RVT | 0.74 |
| V2|MV | 0.67 |
| V3|VT | 0.76 |
| V3|RVT | 0.73 |
| V3|MV | 0.68 |
| V4|VT | 0.74 |
| V4|RVT | 0.72 |
| V4|MV | 0.67 |
| C1|VT | 0.72 |
| C1|RVT | 0.72 |
| C1|MV | 0.67 |
| C2|VT | 0.75 |
| C2|RVT | 0.62 |
| C2|MV | 0.68 |
| C3|VT | 0.75 |
| C3|RVT | 0.75 |
| C3|MV | 0.68 |
| BM|EqualWeight | 1.00 |
| BM|SixtyForty | 1.00 |
| BM|MinVariance | 0.65 |
| BM|RiskParity | 1.00 |
| BM|VolTarget | 0.67 |
| BM|BuyHoldSPY | 1.00 |
| O1|VT | 0.76 |
| O1|RVT | 0.73 |
| O1|MV | 0.68 |

### Pre-specified paired contrasts (secondary, descriptive)

Differences of the statistic under shared resamples. `claimable` requires the CI to exclude zero favourably AND DSR >= 0.95 for X's strategy (N=27 trials; sensitivity at N=9 in `tier1_dsr.csv`). SR0(N=27) = 0.0143, SR0(N=9) = 0.0107 per period.

| leg | x | y | metric | diff [95% CI] | ci_excludes_zero_favourably | DSR x | claimable | identical_by_construction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VT | V2 | V1p | sharpe | 0.0180 [-0.1259, 0.1829] | False | 0.80 | False | False |
| VT | V2 | V1p | max_drawdown | -0.0020 [-0.0115, 0.0648] | False | 0.80 | False | False |
| VT | V2 | V1p | cvar_95 | 0.0013 [0.0002, 0.0027] | True | 0.80 | False | False |
| VT | V2 | C1 | sharpe | 0.0068 [-0.0482, 0.0592] | False | 0.80 | False | False |
| VT | V2 | C1 | max_drawdown | -0.0030 [-0.0164, 0.0058] | False | 0.80 | False | False |
| VT | V2 | C1 | cvar_95 | -0.0008 [-0.0013, -0.0003] | False | 0.80 | False | False |
| VT | V3 | C2 | sharpe | 0.0081 [-0.0124, 0.0266] | False | 0.80 | False | False |
| VT | V3 | C2 | max_drawdown | 0.0009 [-0.0046, 0.0038] | False | 0.80 | False | False |
| VT | V3 | C2 | cvar_95 | -0.0002 [-0.0004, 0.0000] | False | 0.80 | False | False |
| VT | V3 | C3 | sharpe | 0.0028 [-0.0062, 0.0124] | False | 0.80 | False | False |
| VT | V3 | C3 | max_drawdown | -0.0003 [-0.0016, 0.0024] | False | 0.80 | False | False |
| VT | V3 | C3 | cvar_95 | 0.0000 [-0.0002, 0.0002] | False | 0.80 | False | False |
| VT | V4 | V2 | sharpe | 0.0037 [-0.0034, 0.0109] | False | 0.80 | False | False |
| VT | V4 | V2 | max_drawdown | -0.0001 [-0.0011, 0.0021] | False | 0.80 | False | False |
| VT | V4 | V2 | cvar_95 | 0.0000 [-0.0001, 0.0002] | False | 0.80 | False | False |
| RVT | V2 | V1p | sharpe | 0.0180 [-0.1259, 0.1829] | False | 0.80 | False | True |
| RVT | V2 | V1p | max_drawdown | -0.0020 [-0.0115, 0.0648] | False | 0.80 | False | True |
| RVT | V2 | V1p | cvar_95 | 0.0013 [0.0002, 0.0027] | True | 0.80 | False | True |
| RVT | V2 | C1 | sharpe | 0.0068 [-0.0482, 0.0592] | False | 0.80 | False | True |
| RVT | V2 | C1 | max_drawdown | -0.0030 [-0.0164, 0.0058] | False | 0.80 | False | True |
| RVT | V2 | C1 | cvar_95 | -0.0008 [-0.0013, -0.0003] | False | 0.80 | False | True |
| RVT | V3 | C2 | sharpe | 0.1567 [-0.0927, 0.3807] | False | 0.85 | False | False |
| RVT | V3 | C2 | max_drawdown | 0.0048 [-0.0445, 0.0389] | False | 0.85 | False | False |
| RVT | V3 | C2 | cvar_95 | -0.0023 [-0.0039, -0.0009] | False | 0.85 | False | False |
| RVT | V3 | C3 | sharpe | 0.2212 [-0.0140, 0.4414] | False | 0.85 | False | False |
| RVT | V3 | C3 | max_drawdown | 0.0464 [-0.0067, 0.1162] | False | 0.85 | False | False |
| RVT | V3 | C3 | cvar_95 | 0.0024 [0.0004, 0.0047] | True | 0.85 | False | False |
| RVT | V4 | V2 | sharpe | 0.0942 [-0.1093, 0.3045] | False | 0.85 | False | False |
| RVT | V4 | V2 | max_drawdown | 0.0248 [-0.0180, 0.0743] | False | 0.85 | False | False |
| RVT | V4 | V2 | cvar_95 | 0.0015 [0.0000, 0.0032] | True | 0.85 | False | False |
| MV | V2 | V1p | sharpe | -0.0342 [-0.2508, 0.2037] | False | 0.66 | False | False |
| MV | V2 | V1p | max_drawdown | 0.0068 [-0.0241, 0.0560] | False | 0.66 | False | False |
| MV | V2 | V1p | cvar_95 | 0.0016 [0.0007, 0.0027] | True | 0.66 | False | False |
| MV | V2 | C1 | sharpe | -0.0793 [-0.1675, 0.0084] | False | 0.66 | False | False |
| MV | V2 | C1 | max_drawdown | -0.0162 [-0.0344, 0.0005] | False | 0.66 | False | False |
| MV | V2 | C1 | cvar_95 | -0.0010 [-0.0015, -0.0006] | False | 0.66 | False | False |
| MV | V3 | C2 | sharpe | 0.0267 [-0.0129, 0.0667] | False | 0.67 | False | False |
| MV | V3 | C2 | max_drawdown | -0.0047 [-0.0063, 0.0084] | False | 0.67 | False | False |
| MV | V3 | C2 | cvar_95 | -0.0002 [-0.0004, 0.0000] | False | 0.67 | False | False |
| MV | V3 | C3 | sharpe | 0.0059 [-0.0085, 0.0210] | False | 0.67 | False | False |
| MV | V3 | C3 | max_drawdown | -0.0008 [-0.0020, 0.0025] | False | 0.67 | False | False |
| MV | V3 | C3 | cvar_95 | -0.0000 [-0.0002, 0.0001] | False | 0.67 | False | False |
| MV | V4 | V2 | sharpe | 0.0099 [-0.0006, 0.0202] | False | 0.66 | False | False |
| MV | V4 | V2 | max_drawdown | -0.0006 [-0.0008, 0.0028] | False | 0.66 | False | False |
| MV | V4 | V2 | cvar_95 | 0.0000 [-0.0001, 0.0002] | False | 0.66 | False | False |

Claimable contrasts: 0 of 45.

## 5. Drawdown episodes (§14.3) and calendar years

Episodes identified on SPY alone, >= 10% peak-to-trough, before any strategy result was consulted:

| episode | peak | trough | recovery | depth | recovered |
| --- | --- | --- | --- | --- | --- |
| 1 | 2020-02-19 | 2020-03-23 | 2020-08-10 | -33.7% | True |
| 2 | 2022-01-03 | 2022-10-12 | not recovered by window end | -24.5% | False |

### Episode 1

| strategy | decline (peak->trough) | recovery (trough->end) | max DD inside |
| --- | --- | --- | --- |
| V1|VT | -16.7% | 10.8% | -16.7% |
| V1|RVT | -16.7% | 10.8% | -16.7% |
| V1|MV | -7.0% | 8.3% | -7.7% |
| V1p|VT | -16.6% | 10.0% | -16.6% |
| V1p|RVT | -16.6% | 10.0% | -16.6% |
| V1p|MV | -6.4% | 8.6% | -7.8% |
| V2|VT | -16.6% | 10.8% | -16.6% |
| V2|RVT | -16.6% | 10.8% | -16.6% |
| V2|MV | -6.9% | 8.0% | -7.7% |
| V3|VT | -16.8% | 10.9% | -16.8% |
| V3|RVT | -14.2% | 9.8% | -14.2% |
| V3|MV | -6.9% | 8.3% | -7.7% |
| V4|VT | -16.6% | 10.8% | -16.6% |
| V4|RVT | -14.1% | 9.6% | -14.1% |
| V4|MV | -6.9% | 8.0% | -7.7% |
| C1|VT | -16.3% | 10.5% | -16.3% |
| C1|RVT | -16.3% | 10.5% | -16.3% |
| C1|MV | -6.8% | 7.3% | -7.7% |
| C2|VT | -16.8% | 10.8% | -16.8% |
| C2|RVT | -14.1% | 6.4% | -14.1% |
| C2|MV | -6.9% | 8.1% | -7.7% |
| C3|VT | -16.7% | 10.9% | -16.7% |
| C3|RVT | -19.0% | 10.3% | -19.0% |
| C3|MV | -7.0% | 8.3% | -7.7% |
| BM|EqualWeight | -25.7% | 37.9% | -25.7% |
| BM|SixtyForty | -19.4% | 29.8% | -19.4% |
| BM|MinVariance | -3.2% | 5.1% | -4.8% |
| BM|RiskParity | -13.1% | 18.3% | -13.2% |
| BM|VolTarget | -22.6% | 9.8% | -22.6% |
| BM|BuyHoldSPY | -33.7% | 51.2% | -33.7% |
| O1|VT | -16.5% | 11.2% | -16.5% |
| O1|RVT | -12.4% | 10.5% | -12.4% |
| O1|MV | -7.0% | 8.4% | -7.8% |

### Episode 2

| strategy | decline (peak->trough) | recovery (trough->end) | max DD inside |
| --- | --- | --- | --- |
| V1|VT | -11.7% | 10.1% | -12.9% |
| V1|RVT | -11.7% | 10.1% | -12.9% |
| V1|MV | 1.3% | 3.3% | -6.0% |
| V1p|VT | -15.1% | 12.4% | -16.5% |
| V1p|RVT | -15.1% | 12.4% | -16.5% |
| V1p|MV | -0.5% | 5.5% | -8.8% |
| V2|VT | -11.6% | 9.8% | -12.9% |
| V2|RVT | -11.6% | 9.8% | -12.9% |
| V2|MV | 1.4% | 3.2% | -6.0% |
| V3|VT | -11.6% | 10.1% | -12.7% |
| V3|RVT | -8.4% | 8.3% | -9.4% |
| V3|MV | 1.3% | 3.3% | -6.0% |
| V4|VT | -11.5% | 9.9% | -12.7% |
| V4|RVT | -8.3% | 8.2% | -9.3% |
| V4|MV | 1.6% | 3.2% | -5.9% |
| C1|VT | -11.4% | 9.5% | -12.3% |
| C1|RVT | -11.4% | 9.5% | -12.3% |
| C1|MV | 1.6% | 3.7% | -5.6% |
| C2|VT | -11.5% | 10.2% | -12.7% |
| C2|RVT | -6.9% | 8.7% | -7.9% |
| C2|MV | 1.3% | 3.5% | -5.9% |
| C3|VT | -11.7% | 10.0% | -12.9% |
| C3|RVT | -9.8% | 10.5% | -11.5% |
| C3|MV | 1.2% | 3.3% | -6.1% |
| BM|EqualWeight | -14.1% | 14.9% | -15.2% |
| BM|SixtyForty | -20.6% | 17.3% | -20.8% |
| BM|MinVariance | -5.7% | 5.3% | -6.5% |
| BM|RiskParity | -11.4% | 10.1% | -11.9% |
| BM|VolTarget | -8.6% | 10.4% | -10.0% |
| BM|BuyHoldSPY | -24.5% | 29.6% | -24.5% |
| O1|VT | -11.2% | 10.6% | -12.3% |
| O1|RVT | -7.2% | 10.0% | -9.2% |
| O1|MV | 1.8% | 3.6% | -5.5% |

### Calendar years (net return)

| strategy | 2019 | 2020 | 2021 | 2022 | 2023 |
| --- | --- | --- | --- | --- | --- |
| V1|VT | 18.2% | -0.5% | 19.6% | -6.7% | 4.6% |
| V1|RVT | 18.2% | -0.5% | 19.6% | -6.7% | 4.6% |
| V1|MV | 9.1% | 2.4% | -1.7% | 0.7% | 3.7% |
| V1p|VT | 20.1% | -1.5% | 21.2% | -8.2% | 4.3% |
| V1p|RVT | 20.1% | -1.5% | 21.2% | -8.2% | 4.3% |
| V1p|MV | 8.8% | 2.9% | -0.4% | 1.0% | 3.9% |
| V2|VT | 17.5% | -0.6% | 19.3% | -6.6% | 4.4% |
| V2|RVT | 17.5% | -0.6% | 19.3% | -6.6% | 4.4% |
| V2|MV | 8.8% | 2.0% | -1.9% | 0.7% | 3.7% |
| V3|VT | 18.2% | -0.5% | 19.6% | -6.6% | 4.7% |
| V3|RVT | 18.5% | -0.6% | 19.7% | -5.1% | 5.0% |
| V3|MV | 9.2% | 2.4% | -1.6% | 0.7% | 3.7% |
| V4|VT | 17.6% | -0.6% | 19.3% | -6.6% | 4.5% |
| V4|RVT | 18.3% | -0.8% | 19.7% | -5.0% | 4.9% |
| V4|MV | 8.8% | 2.1% | -1.8% | 0.9% | 3.7% |
| C1|VT | 17.8% | -0.9% | 17.6% | -6.8% | 4.6% |
| C1|RVT | 17.8% | -0.9% | 17.6% | -6.8% | 4.6% |
| C1|MV | 9.2% | 2.1% | -1.2% | 1.3% | 3.8% |
| C2|VT | 18.2% | -0.7% | 18.8% | -6.5% | 4.7% |
| C2|RVT | 15.7% | -4.3% | 11.2% | -3.4% | 5.1% |
| C2|MV | 9.1% | 2.2% | -2.5% | 0.7% | 3.8% |
| C3|VT | 18.2% | -0.5% | 19.7% | -6.7% | 4.5% |
| C3|RVT | 14.0% | -6.0% | 19.1% | -5.3% | 5.8% |
| C3|MV | 9.1% | 2.4% | -1.6% | 0.6% | 3.7% |
| BM|EqualWeight | 21.1% | 12.9% | 18.9% | -7.4% | 6.4% |
| BM|SixtyForty | 20.5% | 16.5% | 15.1% | -16.6% | 11.6% |
| BM|MinVariance | 6.6% | 2.3% | 0.6% | -4.1% | 3.3% |
| BM|RiskParity | 13.3% | 8.1% | 8.6% | -6.9% | 4.4% |
| BM|VolTarget | 16.0% | -7.1% | 21.5% | -3.7% | 5.1% |
| BM|BuyHoldSPY | 28.9% | 18.3% | 28.7% | -18.2% | 20.3% |

## 6. Integrity checks

All passed (the run aborts otherwise): input hashes; MV weights differ across variants; V3's regime-conditional exposure differs from C3's and C2's; no-regime RVT equals VT for V1, V1', V2, C1; long-only, 0.35 cap, sum-to-one and no-NaN on every strategy; bootstrap statistics agree with `backtest/metrics.py`.

Mean one-way MV weight distance between variants (largest pairs):

| pair | mean one-way distance |
| --- | --- |
| V1p / C1 | 0.1285 |
| V1p / V2 | 0.1121 |
| V1p / V4 | 0.1120 |
| V1p / C2 | 0.1084 |
| V1 / V1p | 0.1077 |
| V1p / C3 | 0.1077 |

## 7. How to read these results, and what was not as pre-registered

* **Effect sizes are tiny even where the paired CI excludes zero.** V4 vs V2 differs in R-squared by +0.0028 to +0.0055 across the four risk targets and V3 vs C3 by +0.0029 to +0.0049: under strong ridge shrinkage the two forecasts are nearly identical, so their loss differential is very low-variance and a half-point of R-squared is 'significant'. The spec's stricter non-overlapping rule passes none of these comparisons.
* **`fwd_ret_20` is a mean forecast for every variant.** The validation-selected alpha is the top of the grid (1e8) for every variant, so each probe predicts the fold's historical mean and R-squared is 0 to three decimals. Paired 'adverse' verdicts on `fwd_ret_20` are differences at the 1e-7 level and are numerical noise, not findings. This is the expected null for return prediction and it is not used in any gate.
* **Mean-variance gamma sits at the top of its pre-registered grid.** gamma = 32 is the grid maximum, and V1's validation exposure there was 0.82 against the 0.70 target, so the grid could not reach the target. It was not widened after test-split output existed; the MV leg's mean test-period risky exposure is about 0.67-0.68.
* **No allocator contrast is claimable** (0 of 45): none has both a paired CI excluding zero in the favourable direction and DSR >= 0.95. All Sharpe CIs overlap heavily over a 1230-session window; the three allocator legs are descriptive.
* **Conventions.** The 0.35 cap applies to the 13 risky assets in every strategy; the cash line is the uncapped residual in VT/RVT/MV and is capped at 0.35 only in the minimum-variance benchmark; 60/40 is 60% SPY / 40% IEF by definition and is exempt from the cap. Cash earns the prior session's ^IRX / 100 / 252. Sharpe is on total net returns (the T-bill rate is not subtracted). Annualised turnover is total one-way turnover per year, initiation included. Costs are charged on turnover measured against drift-adjusted pre-trade weights.
* **Run history.** The pipeline was executed three times against the pre-registered design: two development runs, the first of which stopped at a weight-constraint check that wrongly applied the 0.35 cap to the 60/40 benchmark (an error in the check, not a design change; fixed before the second run), then this final run through `make tier1`. The runs are deterministic, the second run's gate verdicts equal this run's, and the development runs did print gate verdicts to the log. No pre-registered choice (grid, folds, rule, seeds, allocator parameters) was changed after any test-split output existed. The state files and probes use the encoder step 3 selected (window 10, 32 dims) after the C1/V1' correction recorded as D-028, before pre-registration.

## 8. Figures

![tier1_equity_drawdown.png](figures/tier1_equity_drawdown.png)
![tier1_probe_r2.png](figures/tier1_probe_r2.png)

