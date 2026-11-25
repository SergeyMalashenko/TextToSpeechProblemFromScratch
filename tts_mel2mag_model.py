#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperparams_mel2mag as hp


def hp_get(name: str, default):
    return getattr(hp, name, default)


def get_n_mels() -> int:
    if hasattr(hp, "n_mels"):
        return int(hp.n_mels)
    if hasattr(hp, "num_mels"):
        return int(hp.num_mels)
    raise AttributeError("hyperparams must define hp.n_mels or hp.num_mels")


def get_n_freq() -> int:
    if hasattr(hp, "n_freq"):
        return int(hp.n_freq)
    if hasattr(hp, "num_freq"):
        return int(hp.num_freq)
    if hasattr(hp, "n_fft"):
        return int(hp.n_fft // 2 + 1)
    raise AttributeError("hyperparams must define hp.n_freq or hp.n_fft")


N_MELS = get_n_mels()
N_FREQ = get_n_freq()

HIDDEN = hp_get("mel2mag_hidden_dim", 512)
KERNEL = hp_get("mel2mag_kernel_size", 5)
N_CONVS = hp_get("mel2mag_n_convolutions", 5)
DROPOUT = hp_get("mel2mag_dropout", 0.5)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, activation: bool = True) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = torch.tanh(x)
        return x


class LegacyMelToMagModel(nn.Module):
    """
    Simple mel -> magnitude spectrogram predictor.

    Input:
        mel: (B, T, n_mels)

    Output:
        mag: (B, T, n_freq)
    """

    def __init__(self) -> None:
        super().__init__()

        if N_CONVS < 2:
            raise ValueError("mel2mag_n_convolutions must be >= 2")

        layers = [ConvBlock(N_MELS, HIDDEN, KERNEL, activation=True)]

        for _ in range(N_CONVS - 2):
            layers.append(ConvBlock(HIDDEN, HIDDEN, KERNEL, activation=True))

        layers.append(ConvBlock(HIDDEN, N_FREQ, KERNEL, activation=False))
        self.convs = nn.ModuleList(layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.transpose(1, 2)  # (B, n_mels, T)

        for index, layer in enumerate(self.convs):
            x = layer(x)
            if index < len(self.convs) - 1:
                x = F.dropout(x, p=0.5, training=self.training)

        x = x.transpose(1, 2)  # (B, T, n_freq)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel,
                              padding=dilation * (kernel - 1) // 2, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x.transpose(1, 2)).transpose(1, 2) * mask
        hidden = self.conv(hidden)
        return (x + self.dropout(F.silu(hidden))) * mask


class MelToMagModel(nn.Module):
    """Residual temporal predictor; inputs/outputs use (batch, time, channels)."""
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = dict(config) if config is not None else {
            "n_mels": N_MELS, "n_freq": N_FREQ,
            "hidden": int(hp_get("mel2mag_hidden_dim", 256)),
            "kernel": int(hp_get("mel2mag_kernel_size", 3)),
            "dilations": list(hp_get("mel2mag_dilations", [1, 2, 4, 8, 16, 1, 2, 4])),
            "dropout": float(hp_get("mel2mag_dropout", 0.1)),
            "output_activation": str(hp_get("mel2mag_output_activation", "sigmoid")),
        }
        c = self.config
        if c["kernel"] <= 0 or c["kernel"] % 2 != 1:
            raise ValueError("MelToMag kernel must be positive and odd")
        if not c["dilations"] or any(d <= 0 for d in c["dilations"]):
            raise ValueError("MelToMag dilations must be positive and non-empty")
        if c["output_activation"] not in {"linear", "sigmoid"}:
            raise ValueError("output_activation must be linear or sigmoid")
        self.input_projection = nn.Conv1d(c["n_mels"], c["hidden"], 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(c["hidden"], c["kernel"], d, c["dropout"])
            for d in c["dilations"]
        ])
        self.output_projection = nn.Conv1d(c["hidden"], c["n_freq"], 1)

    def forward(self, mel: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if mel.ndim != 3 or mel.size(-1) != self.config["n_mels"]:
            raise ValueError("Expected mel with shape (B, T, n_mels)")
        if lengths is None:
            mask = mel.new_ones(mel.size(0), 1, mel.size(1))
        else:
            mask = (torch.arange(mel.size(1), device=mel.device)[None, :] <
                    lengths[:, None]).unsqueeze(1).to(mel.dtype)
        x = self.input_projection(mel.transpose(1, 2) * mask) * mask
        for block in self.blocks:
            x = block(x, mask)
        x = self.output_projection(x)
        if self.config["output_activation"] == "sigmoid":
            x = torch.sigmoid(x)
        return (x * mask).transpose(1, 2)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, strict: bool = True) -> nn.Module:
        state = checkpoint.get("model", checkpoint)
        if any(key.startswith("convs.") for key in state):
            # Infer legacy dimensions rather than depend on new training defaults.
            model = LegacyMelToMagModel.__new__(LegacyMelToMagModel)
            nn.Module.__init__(model)
            indices = sorted({int(key.split(".")[1]) for key in state if key.startswith("convs.")})
            layers = []
            for index in indices:
                weight = state[f"convs.{index}.conv.weight"]
                layers.append(ConvBlock(weight.size(1), weight.size(0), weight.size(2),
                                        activation=index != indices[-1]))
            model.convs = nn.ModuleList(layers)
        else:
            config = checkpoint.get("model_config")
            if config is None:
                raise ValueError("Residual MelToMag checkpoint must include model_config")
            model = cls(config)
        model.load_state_dict(state, strict=strict)
        return model


def get_param_size(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def run_model_smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelToMagModel().to(device)

    B = 2
    T = 120
    mel = torch.randn(B, T, N_MELS, device=device)
    mag = model(mel)

    print("=" * 80)
    print("MelToMagModel smoke test")
    print("=" * 80)
    print(f"N_MELS            : {N_MELS}")
    print(f"N_FREQ            : {N_FREQ}")
    print(f"mel.shape         : {tuple(mel.shape)}")
    print(f"mag.shape         : {tuple(mag.shape)}")
    print(f"params            : {get_param_size(model)}")
    print("=" * 80)


if __name__ == "__main__":
    run_model_smoke_test()
