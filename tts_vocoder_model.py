#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperparams_base as hp


def get_n_mels() -> int:
    if hasattr(hp, "n_mels"):
        return int(hp.n_mels)
    if hasattr(hp, "num_mels"):
        return int(hp.num_mels)
    raise AttributeError("hyperparams must define hp.n_mels or hp.num_mels")


class SimpleVocoder(nn.Module):
    """
    A lightweight mel -> waveform vocoder baseline.

    It first upsamples along the time axis with linear interpolation to
    T_audio = T_mel * hop_length and then refines the signal with 1D convolutions.
    """

    def __init__(self) -> None:
        super().__init__()
        n_mels = get_n_mels()
        hidden = 256

        self.hop_length = int(hp.hop_length)

        self.pre = nn.Conv1d(n_mels, hidden, kernel_size=7, padding=3)
        self.blocks = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden // 2, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.post = nn.Conv1d(hidden // 2, 1, kernel_size=7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T_mel, n_mels) or (B, n_mels, T_mel)

        Returns:
            wav: (B, 1, T_audio)
        """
        if mel.ndim != 3:
            raise ValueError(f"Expected 3D mel tensor, got {tuple(mel.shape)}")

        if mel.shape[1] != get_n_mels():
            mel = mel.transpose(1, 2)

        t_audio = mel.shape[-1] * self.hop_length
        x = F.interpolate(mel, size=t_audio, mode="linear", align_corners=False)
        x = self.pre(x)
        x = self.blocks(x)
        x = torch.tanh(self.post(x))
        return x
