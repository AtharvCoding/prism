"""Untrained LSTM with frozen random weights — the C1 control. Spec §9.4, §10.

Isolates "more dimensions" from "a learned representation" (spec §10's V2
vs C1 claim: *LSTM adds value ⟺ V2 > V1′ and V2 > C1*). An LSTM with random,
never-trained weights still performs a nonlinear, recurrent projection of
the window — if that alone beat V1′ (more raw history, same dimensionality
as C1), the LSTM's *training* would not be where the value came from.

Deterministic: the random weights come from a fixed, recorded seed, not
whatever the global RNG state happens to be — two calls with the same seed
must produce the identical frozen encoder.
"""

from __future__ import annotations

import pandas as pd
import torch

from prism.models.encoder.dataset import WindowDataset
from prism.models.encoder.models import LSTMEncoder

__all__ = ["build_random_encoder", "random_encoder_latents"]


def build_random_encoder(
    *,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    seed: int,
    latent_activation: str = "tanh",
) -> LSTMEncoder:
    """A frozen, never-trained :class:`LSTMEncoder`, deterministic in ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    encoder = LSTMEncoder(
        input_dim, hidden_dim, latent_dim, latent_activation=latent_activation
    )
    with torch.no_grad():
        for param in encoder.parameters():
            param.copy_(torch.empty_like(param).uniform_(-0.1, 0.1, generator=generator))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def random_encoder_latents(
    frame: pd.DataFrame,
    window: int,
    *,
    hidden_dim: int,
    latent_dim: int,
    seed: int,
    latent_activation: str = "tanh",
    batch_size: int = 256,
) -> pd.DataFrame:
    """Latents from a frozen random encoder over every window in ``frame``.

    No fit/transform split: there is nothing to fit, by design (that is the
    entire point of the control). Still causal — each window only reads its
    own trailing data — so it is computed in one pass over the whole frame
    rather than needing a train/apply distinction.
    """
    encoder = build_random_encoder(
        input_dim=frame.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim,
        seed=seed, latent_activation=latent_activation,
    )
    dataset = WindowDataset(frame, window)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    for x, _ in loader:
        chunks.append(encoder(x).cpu().numpy())
    import numpy as np

    latent = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, latent_dim))
    columns = [f"latent_{i}" for i in range(latent_dim)]
    return pd.DataFrame(latent, index=dataset.dates, columns=columns)
