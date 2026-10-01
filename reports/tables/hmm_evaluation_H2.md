# HMM evaluation — H2 (K=4)

## State characterisation

| state | n_days | unconditional_share | mean_return_annualised | volatility_annualised | mean_drawdown_while_in_state | worst_drawdown_while_in_state | mean_duration_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2360 | 0.5517 | 0.4807 | 0.086 | -0.0712 | -0.4941 | 4.2523 |
| 1 | 843 | 0.1971 | -0.8078 | 0.1956 | -0.1131 | -0.4991 | 1.8168 |
| 2 | 850 | 0.1987 | 0.1371 | 0.2252 | -0.2225 | -0.5958 | 3.4137 |
| 3 | 225 | 0.0526 | -0.8097 | 0.5823 | -0.2116 | -0.5724 | 1.6071 |

## Empirical transition matrix (walk-forward hard assignment)

| index | state_0 | state_1 | state_2 | state_3 |
| --- | --- | --- | --- | --- |
| state_0 | 0.7648 | 0.1699 | 0.0542 | 0.011 |
| state_1 | 0.4608 | 0.4501 | 0.0392 | 0.0499 |
| state_2 | 0.1753 | 0.0329 | 0.7071 | 0.0847 |
| state_3 | 0.0756 | 0.1556 | 0.3911 | 0.3778 |

## Posterior quality

- Mean entropy: 0.3743 (ok)
- Median max posterior: 0.9228
- Day-to-day flip rate: 0.3290

## Detection vs NBER recessions

### NBER episodes

| episode_start | episode_end | hmm_detected | hmm_lag_sessions | baseline_detected | baseline_lag_sessions | hmm_faster |
| --- | --- | --- | --- | --- | --- | --- |
| 2007-12-01 00:00:00 | 2009-06-01 00:00:00 | True | -51 | True | -60 | False |
| 2020-02-01 00:00:00 | 2020-04-01 00:00:00 | True | 20 | True | -5 | False |

- HMM false-alarm rate: 0.0332
- Baseline false-alarm rate: 0.3855
- HMM beats baseline on detection lag (every episode, strictly lower mean lag): **False**

### Drawdown (>=20%) episodes

| episode_start | episode_end | hmm_detected | hmm_lag_sessions | baseline_detected | baseline_lag_sessions | hmm_faster |
| --- | --- | --- | --- | --- | --- | --- |
| 2007-10-09 00:00:00 | 2008-07-16 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2008-09-10 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2008-09-19 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2010-04-05 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2010-10-13 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2010-10-20 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-08-09 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-08-11 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-08-23 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-09-26 00:00:00 | True | -42 | True | -54 | False |
| 2007-10-09 00:00:00 | 2011-10-06 00:00:00 | True | -42 | True | -54 | False |
| 2020-02-19 00:00:00 | 2020-04-08 00:00:00 | True | 9 | True | -16 | False |
| 2022-01-03 00:00:00 | 2022-06-24 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-07-01 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-07-15 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-10-04 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-10-24 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-11-07 00:00:00 | True | -25 | True | -60 | False |
| 2022-01-03 00:00:00 | 2022-11-10 00:00:00 | True | -25 | True | -60 | False |

- HMM false-alarm rate: 0.0255
- Baseline false-alarm rate: 0.2937
- HMM beats baseline on detection lag (every episode, strictly lower mean lag): **False**

