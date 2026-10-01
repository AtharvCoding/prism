# Encoder hyperparameter sweep

## Window

| Candidate | mean val loss | std |
| --- | --- | --- |
| `{'variant': 'AE', 'window': 10, 'hidden_dim': 64, 'latent_dim': 16}` | 1.098305 | 0.008418 |
| `{'variant': 'AE', 'window': 20, 'hidden_dim': 64, 'latent_dim': 16}` | 1.224031 | 0.013353 |
| `{'variant': 'AE', 'window': 30, 'hidden_dim': 64, 'latent_dim': 16}` | 1.301223 | 0.011547 |
| `{'variant': 'AE', 'window': 60, 'hidden_dim': 64, 'latent_dim': 16}` | 1.320200 | 0.014776 |

## Variant

| Candidate | mean val loss | std |
| --- | --- | --- |
| `{'variant': 'DAE', 'window': 10, 'hidden_dim': 64, 'latent_dim': 16}` | 1.161196 | 0.041607 |
| `{'variant': 'AE', 'window': 10, 'hidden_dim': 64, 'latent_dim': 16}` | 1.189394 | 0.029017 |
| `{'variant': 'PRED', 'window': 10, 'hidden_dim': 64, 'latent_dim': 16}` | 1.254489 | 0.023271 |
| `{'variant': 'VAE', 'window': 10, 'hidden_dim': 64, 'latent_dim': 16}` | 1.321946 | 0.024401 |

## Architecture

| Candidate | mean val loss | std |
| --- | --- | --- |
| `{'variant': 'DAE', 'window': 10, 'hidden_dim': 64, 'latent_dim': 32}` | 1.103038 | 0.010873 |
| `{'variant': 'DAE', 'window': 10, 'hidden_dim': 64, 'latent_dim': 16}` | 1.175430 | 0.059191 |
| `{'variant': 'DAE', 'window': 10, 'hidden_dim': 64, 'latent_dim': 8}` | 1.181812 | 0.007954 |
| `{'variant': 'DAE', 'window': 10, 'hidden_dim': 32, 'latent_dim': 16}` | 1.204029 | 0.024813 |
| `{'variant': 'DAE', 'window': 10, 'hidden_dim': 32, 'latent_dim': 32}` | 1.205908 | 0.039467 |
| `{'variant': 'DAE', 'window': 10, 'hidden_dim': 32, 'latent_dim': 8}` | 1.266353 | 0.014752 |

**Selected:** {'window': 10, 'variant': 'DAE', 'latent_dim': 32, 'hidden_dim': 64}
