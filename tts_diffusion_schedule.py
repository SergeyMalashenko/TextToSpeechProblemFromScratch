#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch

import hyperparams_base as hp
from tts_diffusion_vocoder_model import DiffusionVocoder


def hp_get(name: str, default):
    return getattr(hp, name, default)


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device) -> None:
        self.timesteps = int(timesteps)
        if self.timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        beta_start = float(hp_get("diffusion_vocoder_beta_start", 1e-4))
        beta_end = float(hp_get("diffusion_vocoder_beta_end", 0.02))
        betas = torch.linspace(beta_start, beta_end, self.timesteps, device=device, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def add_noise(
        self,
        clean_audio: torch.Tensor,
        diffusion_step: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bars[diffusion_step].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[diffusion_step].view(-1, 1, 1)
        return sqrt_ab * clean_audio + sqrt_omab * noise

    @torch.no_grad()
    def sample(
        self,
        model: DiffusionVocoder,
        mel: torch.Tensor,
        audio_length: int,
        inference_steps: int,
    ) -> torch.Tensor:
        model.eval()
        device = mel.device
        x = torch.randn(mel.size(0), 1, audio_length, device=device)
        inference_steps = max(1, min(int(inference_steps), self.timesteps))
        step_indices = torch.linspace(
            self.timesteps - 1,
            0,
            steps=inference_steps,
            device=device,
        ).long()

        for idx, step in enumerate(step_indices):
            t = torch.full((mel.size(0),), int(step.item()), device=device, dtype=torch.long)
            pred_x0 = model(x, mel, t).clamp(-1.0, 1.0)
            if idx == len(step_indices) - 1:
                x = pred_x0
                break

            next_step = step_indices[idx + 1]
            alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
            alpha_bar_next = self.alpha_bars[next_step].view(1, 1, 1)
            pred_noise = (x - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1.0 - alpha_bar_t).clamp_min(1e-8)
            x = torch.sqrt(alpha_bar_next) * pred_x0 + torch.sqrt(1.0 - alpha_bar_next) * pred_noise

        model.train()
        return x.clamp(-1.0, 1.0)
