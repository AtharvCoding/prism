# Encoder evaluation — DAE (window=10, latent=32, hidden=64)

## Dead units

- 0/32 dimensions below variance threshold 0.0001

## Out-of-sample reconstruction (last fold)

- overall MSE: 0.415605
- cross: 0.462593
- macro: 0.754702
- asset: 0.399518

## Latent drift (first fold vs last fold), ks

- 31/32 dimensions drifted at alpha=0.01

## Beats-baseline on probe (`fwd_vol_20`)

**Corrected 2026-10-01**: PCA is now refit on the SAME expanding, embargoed
walk-forward fold schedule as the LSTM (`pca_encoder_walkforward`), not fit
once on 1999-2006 and left unrefit through 2023. The static-fit version
(kept here for the record) showed PCA R2=0.296; properly refit, PCA R2
drops to 0.1786 — part of PCA's earlier apparent edge was the
comparison, not the baseline. The LSTM still does not beat it.

- LSTM R2: -0.0488, PCA (walk-forward) R2: 0.1786, random R2: 0.2139
- LSTM beats PCA (non-overlapping MSE CI, lower is better): **False**
- LSTM beats random encoder: **False**
- LSTM MSE CI: {'point_estimate': 0.015211419476275764, 'ci_low': 0.0063704282106737875, 'ci_high': 0.031678353463880024, 'ci_level': 0.95, 'n_bootstrap': 2000, 'block_length': 20}
- PCA MSE CI: {'point_estimate': 0.01191377938992998, 'ci_low': 0.0036663055621490538, 'ci_high': 0.027426845382482218, 'ci_level': 0.95, 'n_bootstrap': 2000, 'block_length': 20}
- random MSE CI: {'point_estimate': 0.011401638028273783, 'ci_low': 0.004216788861131463, 'ci_high': 0.024676680607603023, 'ci_level': 0.95, 'n_bootstrap': 2000, 'block_length': 20}

See also the purely-predictive PRED variant (`pred_reconstruction_weight=0`,
same window/latent/hidden), run as a targeted follow-up rather than through
the full sweep: R2=-0.2028 (worse than DAE's own -0.0488), which also does
not beat PCA or random. Both a reconstruction-selected variant and a directly-supervised one
underperform an untrained linear projection on this target — see
DECISIONS.md for the full finding.
