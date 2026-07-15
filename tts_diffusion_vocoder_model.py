#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperparams_base as hp


def hp_get(name: str, default):
    return getattr(hp, name, default)


def get_n_mels() -> int:
    if hasattr(hp, "n_mels"):
        return int(hp.n_mels)
    if hasattr(hp, "num_mels"):
        return int(hp.num_mels)
    raise AttributeError("hyperparams must define hp.n_mels or hp.num_mels")


class DiffusionEmbedding(nn.Module):
    """
    Sinusoidal diffusion-step embedding followed by a small MLP.
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
            nn.SiLU(),
        )

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        if diffusion_step.ndim == 0:
            diffusion_step = diffusion_step[None]
        diffusion_step = diffusion_step.float()

        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=diffusion_step.device,
            dtype=torch.float32,
        ) / max(1, half_dim - 1)
        angles = diffusion_step[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.proj(emb)


class MelUpsampler(nn.Module):
    """
    Upsample normalized mel features from frame rate to waveform sample rate.

    The product of upsample rates should match hop_length. For this project:
    5 * 5 * 11 = 275.
    """

    def __init__(
        self,
        n_mels: int,
        hidden_channels: int,
        rates: list[int],
        kernels: list[int],
        conditioner_layers: int,
    ) -> None:
        super().__init__()
        if len(rates) != len(kernels):
            raise ValueError("rates and kernels must have the same length")

        conditioner_layers = max(1, int(conditioner_layers))
        conditioner: list[nn.Module] = [
            nn.Conv1d(n_mels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.4),
        ]
        for _ in range(conditioner_layers - 1):
            conditioner.extend(
                [
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.4),
                ]
            )
        self.conditioner = nn.Sequential(*conditioner)

        layers: list[nn.Module] = []
        in_channels = hidden_channels
        for rate, kernel in zip(rates, kernels):
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    hidden_channels,
                    kernel_size=int(kernel),
                    stride=int(rate),
                    padding=(int(kernel) - int(rate)) // 2,
                )
            )
            layers.append(nn.LeakyReLU(0.4))
            in_channels = hidden_channels
        self.upsample = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        x = self.conditioner(mel)
        x = self.upsample(x)
        if target_length is not None:
            if x.size(-1) > target_length:
                x = x[..., :target_length]
            elif x.size(-1) < target_length:
                x = F.pad(x, (0, target_length - x.size(-1)))
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        conditioner_channels: int,
        diffusion_embedding_dim: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, residual_channels)
        self.conditioner_projection = nn.Conv1d(conditioner_channels, 2 * residual_channels, kernel_size=1)
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        conditioner: torch.Tensor,
        diffusion_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diffusion = self.diffusion_projection(diffusion_embedding).unsqueeze(-1)
        y = x + diffusion
        y = self.dilated_conv(y) + self.conditioner_projection(conditioner)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class DiffusionVocoder(nn.Module):
    """
    Compact DiffWave-like vocoder.

    Args:
        noisy_audio: (B, 1, T_audio)
        mel:         (B, T_mel, n_mels) or (B, n_mels, T_mel)
        diffusion_step: integer tensor (B,)

    Returns:
        predicted clean waveform x0: (B, 1, T_audio)
    """

    def __init__(self) -> None:
        super().__init__()
        n_mels = get_n_mels()
        residual_channels = int(hp_get("diffusion_vocoder_residual_channels", 64))
        residual_layers = int(hp_get("diffusion_vocoder_residual_layers", 24))
        dilation_cycle = int(hp_get("diffusion_vocoder_dilation_cycle", 10))
        diffusion_embedding_dim = int(hp_get("diffusion_vocoder_embedding_dim", 128))
        conditioner_channels = int(hp_get("diffusion_vocoder_conditioner_channels", 128))
        conditioner_layers = int(hp_get("diffusion_vocoder_conditioner_layers", 1))
        upsample_rates = list(hp_get("diffusion_vocoder_upsample_rates", [5, 5, 11]))
        upsample_kernels = list(hp_get("diffusion_vocoder_upsample_kernel_sizes", [10, 10, 22]))

        self.n_mels = n_mels
        self.input_projection = nn.Conv1d(1, residual_channels, kernel_size=1)
        self.diffusion_embedding = DiffusionEmbedding(dim=diffusion_embedding_dim)
        self.mel_upsampler = MelUpsampler(
            n_mels=n_mels,
            hidden_channels=conditioner_channels,
            rates=upsample_rates,
            kernels=upsample_kernels,
            conditioner_layers=conditioner_layers,
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    conditioner_channels=conditioner_channels,
                    diffusion_embedding_dim=diffusion_embedding_dim,
                    dilation=2 ** (i % dilation_cycle),
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.output_projection = nn.Conv1d(residual_channels, 1, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_audio: torch.Tensor,
        mel: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_audio.ndim != 3 or noisy_audio.size(1) != 1:
            raise ValueError(f"Expected noisy_audio with shape (B, 1, T), got {tuple(noisy_audio.shape)}")
        if mel.ndim != 3:
            raise ValueError(f"Expected mel with shape (B, T, C) or (B, C, T), got {tuple(mel.shape)}")
        if mel.size(1) != self.n_mels:
            mel = mel.transpose(1, 2)

        x = self.input_projection(noisy_audio)
        x = F.relu(x)
        diffusion_embedding = self.diffusion_embedding(diffusion_step)
        conditioner = self.mel_upsampler(mel, target_length=noisy_audio.size(-1))

        skip_sum = None
        for layer in self.residual_layers:
            x, skip = layer(x, conditioner, diffusion_embedding)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        assert skip_sum is not None
        x = skip_sum / math.sqrt(len(self.residual_layers))
        x = F.relu(self.skip_projection(x))
        return torch.tanh(self.output_projection(x))


def get_param_size(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def run_model_smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionVocoder().to(device)
    batch_size = 2
    mel_frames = 64
    audio_len = int(hp.hop_length) * mel_frames
    noisy_audio = torch.randn(batch_size, 1, audio_len, device=device)
    mel = torch.randn(batch_size, mel_frames, get_n_mels(), device=device)
    diffusion_step = torch.randint(0, 1000, (batch_size,), device=device)
    pred = model(noisy_audio, mel, diffusion_step)
    print("=" * 80)
    print("DiffusionVocoder smoke test")
    print("=" * 80)
    print(f"noisy_audio.shape : {tuple(noisy_audio.shape)}")
    print(f"mel.shape         : {tuple(mel.shape)}")
    print(f"pred.shape        : {tuple(pred.shape)}")
    print(f"params            : {get_param_size(model)}")
    print("=" * 80)


if __name__ == "__main__":
    run_model_smoke_test()
