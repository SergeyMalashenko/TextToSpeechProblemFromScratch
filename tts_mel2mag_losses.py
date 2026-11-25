"""Masked losses for normalized dB magnitude predictions."""
from __future__ import annotations

import torch
from torch import nn


class MelToMagLoss(nn.Module):
    def __init__(self, *, max_db=100.0, ref_db=20.0, mel_basis=None,
                 mel_weight=0.1, temporal_weight=0.1, frequency_weight=0.1,
                 spectral_weight=0.0):
        super().__init__()
        if mel_weight and mel_basis is None:
            raise ValueError("mel_basis is required for mel consistency")
        self.register_buffer("mel_basis", mel_basis)
        self.max_db, self.ref_db = float(max_db), float(ref_db)
        self.weights = {"mel": mel_weight, "temporal": temporal_weight,
                        "frequency": frequency_weight, "spectral": spectral_weight}

    def amplitude(self, x):
        return torch.pow(10.0, (x.clamp(0, 1) * self.max_db - self.max_db + self.ref_db) / 20.0)

    def forward(self, pred, target, mel, lengths):
        # Keep exponential conversion and all reductions out of autocast/FP16.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target, mel = pred.float(), target.float(), mel.float()
            mask = torch.arange(pred.size(1), device=pred.device)[None, :] < lengths[:, None]
            def mean(values, valid):
                valid = valid.unsqueeze(-1).to(values.dtype)
                return (values * valid).sum() / (valid.sum() * values.size(-1)).clamp_min(1)
            zero = pred.sum() * 0
            losses = {"l1": mean((pred - target).abs(), mask),
                      "temporal": zero, "frequency": zero, "mel": zero, "spectral": zero}
            if self.weights["temporal"]:
                losses["temporal"] = mean(((pred[:, 1:] - pred[:, :-1]) -
                    (target[:, 1:] - target[:, :-1])).abs(), mask[:, 1:] & mask[:, :-1])
            if self.weights["frequency"]:
                losses["frequency"] = mean(((pred[..., 1:] - pred[..., :-1]) -
                    (target[..., 1:] - target[..., :-1])).abs(), mask)
            if self.weights["mel"] or self.weights["spectral"]:
                amplitude = self.amplitude(pred)
            if self.weights["mel"]:
                mel_amp = amplitude @ self.mel_basis.float().T
                mel_pred = ((20 * torch.log10(mel_amp.clamp_min(1e-5)) - self.ref_db +
                             self.max_db) / self.max_db).clamp(0, 1)
                losses["mel"] = mean((mel_pred - mel).abs(), mask)
            if self.weights["spectral"]:
                valid = mask.unsqueeze(-1)
                real = self.amplitude(target) * valid
                diff = (amplitude - self.amplitude(target)) * valid
                losses["spectral"] = (diff.norm(dim=(1, 2)) /
                                      real.norm(dim=(1, 2)).clamp_min(1e-5)).mean()
            losses["loss"] = losses["l1"] + sum(self.weights[k] * losses[k] for k in self.weights)
            return losses
