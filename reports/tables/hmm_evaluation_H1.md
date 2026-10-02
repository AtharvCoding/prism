# HMM evaluation — H1 (K=2)

## State characterisation

| state | n_days | unconditional_share | mean_return_annualised | volatility_annualised | mean_drawdown_while_in_state | worst_drawdown_while_in_state | mean_duration_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3011 | 0.7038 | 0.1921 | 0.1084 | -0.0775 | -0.4606 | 23.3411 |
| 1 | 1267 | 0.2962 | -0.1505 | 0.3299 | -0.2104 | -0.5958 | 9.8984 |

## Empirical transition matrix (walk-forward hard assignment)

| index | state_0 | state_1 |
| --- | --- | --- |
| state_0 | 0.9575 | 0.0425 |
| state_1 | 0.101 | 0.899 |

## Posterior quality

- Mean entropy: 0.1899 (ok)
- Median max posterior: 0.9840
- Day-to-day flip rate: 0.0599

## Detection vs NBER recessions

### NBER episodes

| episode_start | episode_end | hmm_detected | hmm_lag_sessions | baseline_detected | baseline_lag_sessions | hmm_faster |
| --- | --- | --- | --- | --- | --- | --- |
| 2007-12-01 00:00:00 | 2009-06-01 00:00:00 | True | -60 | True | -60 | False |
| 2020-02-01 00:00:00 | 2020-04-01 00:00:00 | True | 14 | True | -5 | False |

- HMM false-alarm rate: 0.2301
- Baseline false-alarm rate: 0.3855
- HMM beats baseline on detection lag (every episode, strictly lower mean lag): **False**

### Drawdown (>=20%) episodes

| episode_start | episode_end | hmm_detected | hmm_lag_sessions | baseline_detected | baseline_lag_sessions | hmm_faster |
| --- | --- | --- | --- | --- | --- | --- |
| 2007-10-09 00:00:00 | 2008-07-16 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2008-09-10 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2008-09-19 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2010-04-05 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2010-10-13 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2010-10-20 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-08-09 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-08-11 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-08-23 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-09-26 00:00:00 | True | -52 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-10-06 00:00:00 | True | -52 | True | -54 | False |
| 2020-02-19 00:00:00 | 2020-04-08 00:00:00 | True | 3 | True | -16 | False |
| 2022-01-03 00:00:00 | 2022-06-24 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-07-01 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-07-15 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-10-04 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-10-24 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-11-07 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-11-10 00:00:00 | True | -25 | True | -60 | False |

- HMM false-alarm rate: 0.1671
- Baseline false-alarm rate: 0.2937
- HMM beats baseline on detection lag (every episode, strictly lower mean lag): **False**

