#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperparams_base as hp


LRELU_SLOPE = 0.1


def hp_get(name: str, default):
    return getattr(hp, name, default)


def get_n_mels() -> int:
    if hasattr(hp, "n_mels"):
        return int(hp.n_mels)
    if hasattr(hp, "num_mels"):
        return int(hp.num_mels)
    raise AttributeError("hyperparams must define hp.n_mels or hp.num_mels")


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int, int, int] = (1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )
            )
            for d in dilations
        ])

        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
            )
            for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)


class Generator(nn.Module):
    """
    HiFi-GAN style generator.
    Input:  mel (B, n_mels, T_mel)
    Output: wav (B, 1, T_audio)
    """

    def __init__(self) -> None:
        super().__init__()

        self.n_mels = get_n_mels()
        self.upsample_rates = hp_get("hifigan_upsample_rates", [5, 5, 11])
        self.upsample_kernel_sizes = hp_get("hifigan_upsample_kernel_sizes", [10, 10, 22])
        self.upsample_initial_channel = hp_get("hifigan_upsample_initial_channel", 512)
        self.resblock_kernel_sizes = hp_get("hifigan_resblock_kernel_sizes", [3, 7, 11])
        self.resblock_dilation_sizes = hp_get(
            "hifigan_resblock_dilation_sizes",
            [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        )

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(self.n_mels, self.upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        channels = self.upsample_initial_channel
        for u, k in zip(self.upsample_rates, self.upsample_kernel_sizes):
            out_ch = channels // 2
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        channels,
                        out_ch,
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                    )
                )
            )

            for rk, rd in zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(out_ch, kernel_size=rk, dilations=rd))

            channels = out_ch

        self.num_kernels = len(self.resblock_kernel_sizes)
        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)

        rb_idx = 0
        for up in self.ups:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)

            xs = 0.0
            for _ in range(self.num_kernels):
                xs = xs + self.resblocks[rb_idx](x)
                rb_idx += 1
            x = xs / self.num_kernels

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        nn.utils.remove_weight_norm(self.conv_pre)
        for up in self.ups:
            nn.utils.remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)


class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1,   32, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            t = t + pad_len

        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for d in self.discriminators:
            y_r, fmap_r = d(y)
            y_g, fmap_g = d(y_hat)
            y_d_rs.append(y_r)
            y_d_gs.append(y_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        for i, d in enumerate(self.discriminators):
            if i > 0:
                pool = self.pooling[i - 1]
                y = pool(y)
                y_hat = pool(y_hat)

            y_r, fmap_r = d(y)
            y_g, fmap_g = d(y_hat)

            y_d_rs.append(y_r)
            y_d_gs.append(y_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2.0


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    real_losses = []
    generated_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + (r_loss + g_loss)
        real_losses.append(float(r_loss.item()))
        generated_losses.append(float(g_loss.item()))

    return loss, real_losses, generated_losses


def generator_loss(disc_outputs):
    loss = 0.0
    gen_losses = []

    for dg in disc_outputs:
        l = torch.mean((1.0 - dg) ** 2)
        gen_losses.append(float(l.item()))
        loss = loss + l

    return loss, gen_losses
