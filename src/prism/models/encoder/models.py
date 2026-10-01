"""AE / DAE / VAE / predictive-head architectures. No ReLU on the latent. Spec §9.3.

Four defects fixed structurally, not by convention:

**C3 — ReLU on the latent.** ``latent_0`` and ``latent_6`` were identically
zero for the whole sample, with a visible activation-distribution shift
after 2018. ``LSTMEncoder`` accepts only ``"linear"`` or ``"tanh"`` as its
latent activation — the same guard ``prism.config`` already enforces at the
YAML level, repeated here so a caller building a model directly (bypassing
config) cannot reintroduce it either.

**C4 — capacity mismatch.** ~507k parameters against ~2,400 overlapping
windows. ``tests/test_encoder.py::test_capacity_is_plausible_against_the_available_windows``
already pins the *configured* architecture below that; this module's job is
just to not silently add parameters config didn't ask for.

**The window dimension.** LSTM output at the final timestep already
summarises the whole window causally (each hidden state depends only on
timesteps up to and including its own); nothing here looks past the window's
last row.

**Multi-task loss (PRED).** The predictive head and the reconstruction head
share the encoder but are trained jointly with separate loss terms, never
with the predictive head silently dominating reconstruction (or vice versa)
through an unstated weighting — ``PredictiveLoss`` takes its weights as
explicit, logged arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

__all__ = [
    "LSTMEncoder",
    "LSTMDecoder",
    "Autoencoder",
    "DenoisingAutoencoder",
    "VariationalAutoencoder",
    "PredictiveAutoencoder",
    "LossOutput",
    "build_model",
]

LatentActivation = Literal["linear", "tanh"]


def _latent_activation(name: LatentActivation) -> nn.Module:
    if name == "linear":
        return nn.Identity()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"unsupported latent_activation {name!r}; only 'linear' or 'tanh' "
        "(ReLU on the latent produced permanently dead units — defect C3)"
    )


class LSTMEncoder(nn.Module):
    """Window -> latent. The final hidden state, projected and (optionally) squashed."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        latent_activation: LatentActivation = "tanh",
        latent_multiplier: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.project = nn.Linear(hidden_dim, latent_dim * latent_multiplier)
        self.activation = _latent_activation(latent_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_layer_hidden = hidden[-1]  # (batch, hidden_dim) — final timestep, top layer
        return self.activation(self.project(last_layer_hidden))


class LSTMDecoder(nn.Module):
    """Latent -> reconstructed window. No access to the original window at all."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.expand = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        seed = torch.tanh(self.expand(latent)).unsqueeze(1).repeat(1, self.seq_len, 1)
        hidden, _ = self.lstm(seed)
        return self.output(hidden)


@dataclass
class LossOutput:
    """A loss plus its named components, so a training loop can log each term
    separately without every variant needing its own logging code."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def as_floats(self) -> dict[str, float]:
        return {k: float(v.detach().cpu().item()) for k, v in {"total": self.total, **self.components}.items()}


class Autoencoder(nn.Module):
    """Plain reconstruction. Spec §9.3 ``AE``."""

    variant = "AE"

    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

    def loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> LossOutput:
        recon, _ = self(x)
        mse = nn.functional.mse_loss(recon, x)
        return LossOutput(total=mse, components={"reconstruction": mse})

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DenoisingAutoencoder(Autoencoder):
    """Gaussian input noise during training only. Spec §9.3 ``DAE``.

    Noise is added to the ENCODER's input, never to the reconstruction
    target: the model must learn to recover the clean window from a noisy
    view of it, not merely copy noise through. At inference (``.eval()``)
    no noise is added — :meth:`latent` and :meth:`loss` in eval mode both
    see the clean window, matching how the fitted model is actually used
    downstream (state assembly never sees artificially noised features).
    """

    variant = "DAE"

    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder, *, noise_std: float) -> None:
        super().__init__(encoder, decoder)
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        self.noise_std = noise_std

    def loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> LossOutput:
        noisy = x + torch.randn_like(x) * self.noise_std if self.training else x
        latent = self.encoder(noisy)
        recon = self.decoder(latent)
        mse = nn.functional.mse_loss(recon, x)  # reconstruct the CLEAN window
        return LossOutput(total=mse, components={"reconstruction": mse})


class VariationalAutoencoder(nn.Module):
    """A small KL weight for a smoother, better-behaved latent. Spec §9.3 ``VAE``.

    The encoder's projection head outputs ``2 * latent_dim`` values, split
    into ``(mu, logvar)``; the reported "latent" at inference is ``mu`` (the
    posterior mean), not a sampled draw — a downstream consumer wants a
    deterministic representation, not fresh noise every call. Sampling (the
    reparameterisation trick) is used only inside the training loss.
    """

    variant = "VAE"

    def __init__(
        self, encoder: LSTMEncoder, decoder: LSTMDecoder, *, latent_dim: int, kl_weight: float
    ) -> None:
        super().__init__()
        if kl_weight < 0:
            raise ValueError("kl_weight must be >= 0")
        self.encoder = encoder  # projects to 2*latent_dim (mu, logvar)
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(x)
        mu, logvar = out[:, : self.latent_dim], out[:, self.latent_dim :]
        # Clamp logvar: unconstrained, early training can send it to +-inf,
        # which makes exp(logvar) overflow/underflow long before the
        # optimiser has a gradient signal to correct it.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self._encode(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        recon = self.decoder(z)
        return recon, mu

    def loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> LossOutput:
        mu, logvar = self._encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        recon_loss = nn.functional.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.kl_weight * kl
        return LossOutput(total=total, components={"reconstruction": recon_loss, "kl": kl})

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self._encode(x)
        return mu


class PredictiveAutoencoder(nn.Module):
    """Predictive head(s), optionally multi-task with reconstruction. Spec §9.3 ``PRED``.

    ``reconstruction_weight=0`` gives a purely predictive encoder (the
    hypothesis being tested is "reconstruction is the wrong objective";
    spec §15.1 H3's "if refuted" branch names this as the pivot).
    """

    variant = "PRED"

    def __init__(
        self,
        encoder: LSTMEncoder,
        decoder: LSTMDecoder | None,
        *,
        latent_dim: int,
        n_targets: int,
        reconstruction_weight: float = 1.0,
        prediction_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if decoder is None and reconstruction_weight != 0:
            raise ValueError("reconstruction_weight must be 0 when decoder is None")
        self.encoder = encoder
        self.decoder = decoder
        self.head = nn.Linear(latent_dim, n_targets)
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        pred = self.head(latent)
        recon = self.decoder(latent) if self.decoder is not None else None
        return pred, recon, latent

    def loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> LossOutput:
        if y is None:
            raise ValueError("PredictiveAutoencoder.loss requires forward targets y")
        pred, recon, _ = self(x)
        pred_loss = nn.functional.mse_loss(pred, y)
        components = {"prediction": pred_loss}
        total = self.prediction_weight * pred_loss
        if recon is not None and self.reconstruction_weight > 0:
            recon_loss = nn.functional.mse_loss(recon, x)
            components["reconstruction"] = recon_loss
            total = total + self.reconstruction_weight * recon_loss
        return LossOutput(total=total, components=components)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_model(
    variant: str,
    *,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    window: int,
    num_layers: int = 1,
    dropout: float = 0.0,
    latent_activation: LatentActivation = "tanh",
    dae_noise_std: float = 0.1,
    vae_kl_weight: float = 0.001,
    n_pred_targets: int = 0,
    pred_reconstruction_weight: float = 1.0,
) -> nn.Module:
    """Construct a model by variant name, reading the shared architecture knobs
    from one place so every variant is built with the same encoder/decoder
    shape unless the variant itself requires otherwise (VAE's doubled
    projection head; PRED's optional decoder)."""
    if variant == "AE":
        enc = LSTMEncoder(
            input_dim, hidden_dim, latent_dim, num_layers=num_layers, dropout=dropout,
            latent_activation=latent_activation,
        )
        dec = LSTMDecoder(latent_dim, hidden_dim, input_dim, window, num_layers=num_layers, dropout=dropout)
        return Autoencoder(enc, dec)
    if variant == "DAE":
        enc = LSTMEncoder(
            input_dim, hidden_dim, latent_dim, num_layers=num_layers, dropout=dropout,
            latent_activation=latent_activation,
        )
        dec = LSTMDecoder(latent_dim, hidden_dim, input_dim, window, num_layers=num_layers, dropout=dropout)
        return DenoisingAutoencoder(enc, dec, noise_std=dae_noise_std)
    if variant == "VAE":
        enc = LSTMEncoder(
            input_dim, hidden_dim, latent_dim, num_layers=num_layers, dropout=dropout,
            latent_activation=latent_activation, latent_multiplier=2,
        )
        dec = LSTMDecoder(latent_dim, hidden_dim, input_dim, window, num_layers=num_layers, dropout=dropout)
        return VariationalAutoencoder(enc, dec, latent_dim=latent_dim, kl_weight=vae_kl_weight)
    if variant == "PRED":
        if n_pred_targets < 1:
            raise ValueError("PRED variant requires n_pred_targets >= 1")
        enc = LSTMEncoder(
            input_dim, hidden_dim, latent_dim, num_layers=num_layers, dropout=dropout,
            latent_activation=latent_activation,
        )
        dec = (
            LSTMDecoder(latent_dim, hidden_dim, input_dim, window, num_layers=num_layers, dropout=dropout)
            if pred_reconstruction_weight > 0
            else None
        )
        return PredictiveAutoencoder(
            enc, dec, latent_dim=latent_dim, n_targets=n_pred_targets,
            reconstruction_weight=pred_reconstruction_weight,
        )
    raise ValueError(f"unknown variant {variant!r}; expected one of AE, DAE, VAE, PRED")
