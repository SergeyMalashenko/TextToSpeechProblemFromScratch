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


class MelToMagModel(nn.Module):
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

        for layer in self.convs:
            x = layer(x)
            x = F.dropout(x, p=DROPOUT, training=self.training)

        x = x.transpose(1, 2)  # (B, T, n_freq)
        return x


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
