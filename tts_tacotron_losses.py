#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn



def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Returns:
        mask: (B, T) with True for valid positions.
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) < lengths.unsqueeze(1)


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    pred, target: (B, T, C)
    lengths: (B,)
    """
    mask = sequence_mask(lengths, max_len=pred.size(1)).unsqueeze(-1).to(pred.dtype)
    loss = torch.abs(pred - target) * mask
    denom = mask.sum() * pred.size(-1)
    return loss.sum() / denom.clamp_min(1.0)


def gate_bce_loss(
    gate_logits: torch.Tensor,
    gate_target: torch.Tensor,
    pos_weight_value: float,
) -> torch.Tensor:
    pos_weight = torch.tensor([pos_weight_value], device=gate_logits.device, dtype=gate_logits.dtype)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion(gate_logits, gate_target)


def guided_attention_map(
    mel_len: int,
    text_len: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns:
        (T_mel, T_text)
    """
    t = torch.arange(mel_len, device=device, dtype=dtype) / max(float(mel_len), 1.0)
    n = torch.arange(text_len, device=device, dtype=dtype) / max(float(text_len), 1.0)

    tt = t.unsqueeze(1)  # (T_mel, 1)
    nn_ = n.unsqueeze(0)  # (1, T_text)

    w = 1.0 - torch.exp(-((tt - nn_) ** 2) / (2.0 * sigma * sigma))
    return w


def guided_attention_loss(
    alignments: torch.Tensor,
    text_lengths: torch.Tensor,
    mel_lengths: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    alignments: (B, T_mel, T_text)
    text_lengths: (B,)
    mel_lengths: (B,)

    We only compute over valid alignment area.
    """
    B, T_mel_max, T_text_max = alignments.shape
    total_loss = alignments.new_tensor(0.0)
    total_weight = alignments.new_tensor(0.0)

    for b in range(B):
        t_len = int(mel_lengths[b].item())
        n_len = int(text_lengths[b].item())

        if t_len <= 0 or n_len <= 0:
            continue

        A = alignments[b, :t_len, :n_len]
        W = guided_attention_map(
            mel_len=t_len,
            text_len=n_len,
            sigma=sigma,
            device=alignments.device,
            dtype=alignments.dtype,
        )
        total_loss = total_loss + (A * W).sum()
        total_weight = total_weight + torch.tensor(float(t_len * n_len), device=alignments.device, dtype=alignments.dtype)

    return total_loss / total_weight.clamp_min(1.0)


class Tacotron2Loss(nn.Module):
    def __init__(
        self,
        gate_pos_weight: float = 5.0,
        guided_attn_weight: float = 1.0,
        guided_attn_sigma: float = 0.4,
    ) -> None:
        super().__init__()
        self.gate_pos_weight = gate_pos_weight
        self.guided_attn_weight = guided_attn_weight
        self.guided_attn_sigma = guided_attn_sigma

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mel_before = outputs["mel_before"]      # (B, T, n_mels)
        mel_after = outputs["mel_after"]        # (B, T, n_mels)
        gate_logits = outputs["gate"]           # (B, T)
        alignments = outputs["alignments"]      # (B, T, T_text)

        mel_target = batch["mel_target"]        # (B, T, n_mels)
        gate_target = batch["gate_target"]      # (B, T)
        text_lengths = batch["text_lengths"]    # (B,)
        output_lengths = batch["output_lengths"]# (B,)

        mel_before_loss = masked_l1_loss(mel_before, mel_target, output_lengths)
        mel_after_loss = masked_l1_loss(mel_after, mel_target, output_lengths)
        mel_loss = mel_before_loss + mel_after_loss

        gate_loss = gate_bce_loss(
            gate_logits=gate_logits,
            gate_target=gate_target,
            pos_weight_value=self.gate_pos_weight,
        )

        attn_loss = guided_attention_loss(
            alignments=alignments,
            text_lengths=text_lengths,
            mel_lengths=output_lengths,
            sigma=self.guided_attn_sigma,
        )

        total_loss = mel_loss + gate_loss + self.guided_attn_weight * attn_loss

        return {
            "loss": total_loss,
            "mel_loss": mel_loss,
            "gate_loss": gate_loss,
            "attn_loss": attn_loss,
            "mel_before_loss": mel_before_loss,
            "mel_after_loss": mel_after_loss,
        }




# =============================================================================
# Smoke tests
# =============================================================================

def _smoke_test_sequence_mask() -> None:
    lengths = torch.tensor([2, 4], dtype=torch.long)
    mask = sequence_mask(lengths)
    expected = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask.cpu(), expected), f"Unexpected mask:\n{mask}"


def _smoke_test_masked_l1_loss() -> None:
    pred = torch.tensor([[[1.0], [2.0], [100.0]]])
    target = torch.tensor([[[1.0], [4.0], [0.0]]])
    lengths = torch.tensor([2], dtype=torch.long)

    loss = masked_l1_loss(pred, target, lengths)
    expected = torch.tensor(1.0)
    assert torch.isclose(loss.cpu(), expected), f"Expected {expected.item()}, got {loss.item()}"


def _smoke_test_guided_attention_map() -> None:
    w = guided_attention_map(
        mel_len=10,
        text_len=5,
        sigma=0.4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert w.shape == (10, 5), f"Unexpected shape: {tuple(w.shape)}"
    assert torch.isfinite(w).all(), "Guided attention map contains non-finite values"
    assert float(w.min()) >= 0.0, "Guided attention map has negative values"
    assert float(w.max()) <= 1.0, "Guided attention map has values greater than 1"


def _smoke_test_guided_attention_loss() -> None:
    mel_len = 12
    text_len = 6

    diag = torch.zeros(1, mel_len, text_len)
    for t in range(mel_len):
        j = min(text_len - 1, int(round(t * (text_len - 1) / max(mel_len - 1, 1))))
        diag[0, t, j] = 1.0

    offdiag = torch.zeros_like(diag)
    offdiag[:, :, -1] = 1.0

    text_lengths = torch.tensor([text_len], dtype=torch.long)
    mel_lengths = torch.tensor([mel_len], dtype=torch.long)

    diag_loss = guided_attention_loss(diag, text_lengths, mel_lengths, sigma=0.4)
    offdiag_loss = guided_attention_loss(offdiag, text_lengths, mel_lengths, sigma=0.4)

    assert torch.isfinite(diag_loss), "Diagonal guided attention loss is non-finite"
    assert torch.isfinite(offdiag_loss), "Off-diagonal guided attention loss is non-finite"
    assert diag_loss < offdiag_loss, (
        f"Diagonal alignment should have smaller loss than off-diagonal alignment: "
        f"diag={diag_loss.item():.6f}, offdiag={offdiag_loss.item():.6f}"
    )


def _smoke_test_tacotron2_loss_forward() -> None:
    batch_size = 2
    mel_steps = 20
    text_steps = 12
    n_mels = 80

    outputs = {
        "mel_before": torch.randn(batch_size, mel_steps, n_mels),
        "mel_after": torch.randn(batch_size, mel_steps, n_mels),
        "gate": torch.randn(batch_size, mel_steps),
        "alignments": torch.softmax(torch.randn(batch_size, mel_steps, text_steps), dim=-1),
    }
    batch = {
        "mel_target": torch.randn(batch_size, mel_steps, n_mels),
        "gate_target": torch.zeros(batch_size, mel_steps),
        "text_lengths": torch.tensor([text_steps, text_steps - 2], dtype=torch.long),
        "output_lengths": torch.tensor([mel_steps, mel_steps - 3], dtype=torch.long),
    }

    criterion = Tacotron2Loss(
        gate_pos_weight=5.0,
        guided_attn_weight=1.0,
        guided_attn_sigma=0.4,
    )
    loss_dict = criterion(outputs=outputs, batch=batch)

    required_keys = {
        "loss",
        "mel_loss",
        "gate_loss",
        "attn_loss",
        "mel_before_loss",
        "mel_after_loss",
    }
    missing = required_keys.difference(loss_dict.keys())
    assert not missing, f"Missing loss keys: {sorted(missing)}"

    for key in required_keys:
        value = loss_dict[key]
        assert torch.is_tensor(value), f"{key} is not a tensor"
        assert value.ndim == 0, f"{key} should be a scalar tensor, got shape {tuple(value.shape)}"
        assert torch.isfinite(value), f"{key} is non-finite: {value}"


def run_smoke_tests() -> None:
    print("Running tts_tacotron_losses.py smoke tests...")
    _smoke_test_sequence_mask()
    print("  OK: sequence_mask")
    _smoke_test_masked_l1_loss()
    print("  OK: masked_l1_loss")
    _smoke_test_guided_attention_map()
    print("  OK: guided_attention_map")
    _smoke_test_guided_attention_loss()
    print("  OK: guided_attention_loss")
    _smoke_test_tacotron2_loss_forward()
    print("  OK: Tacotron2Loss.forward")
    print("All tts_tacotron_losses.py smoke tests passed.")


if __name__ == "__main__":
    run_smoke_tests()
