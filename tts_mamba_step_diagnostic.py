#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare full-sequence Mamba decoding with stateful step-by-step decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tts_mamba_model import MambaTacotron2


def load_checkpoint_state(path: str | Path, device: torch.device):
    state = torch.load(path, map_location=device)
    return state["model"] if isinstance(state, dict) and "model" in state else state


def make_random_batch(
    model: MambaTacotron2,
    batch_size: int,
    text_len: int,
    mel_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_lengths = torch.linspace(text_len, max(1, text_len - batch_size + 1), batch_size)
    text_lengths = text_lengths.round().long().clamp(min=1, max=text_len).to(device)

    text = torch.zeros(batch_size, text_len, dtype=torch.long, device=device)
    for b, length in enumerate(text_lengths.tolist()):
        text[b, :length] = torch.randint(
            low=1,
            high=max(2, model.n_symbols),
            size=(length,),
            device=device,
        )

    mel_input = torch.randn(batch_size, mel_len, model.n_mels, device=device)
    return text, text_lengths, mel_input


@torch.no_grad()
def compare_full_and_step(
    model: MambaTacotron2,
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    mel_input: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    memory = model.encode(text, text_lengths)
    context_feedback = bool(getattr(model, "mamba_step_context_feedback", False))
    if context_feedback:
        print(
            "Note: mamba_step_context_feedback=True, so full-sequence decode "
            "and step decode use intentionally different decoder inputs."
        )

    full_mel, full_gate, full_align = model.decode_sequence(
        memory=memory,
        text_lengths=text_lengths,
        mel_input=mel_input,
    )
    step_mel, step_gate, step_align = model.decode_sequence_step_teacher_forced(
        memory=memory,
        text_lengths=text_lengths,
        mel_input=mel_input,
    )

    full_pos = full_align.argmax(dim=-1)
    step_pos = step_align.argmax(dim=-1)
    pos_mismatch = (full_pos != step_pos).float().mean()

    return {
        "mel_max_abs": float((full_mel - step_mel).abs().max().item()),
        "mel_mean_abs": float((full_mel - step_mel).abs().mean().item()),
        "gate_max_abs": float((full_gate - step_gate).abs().max().item()),
        "gate_mean_abs": float((full_gate - step_gate).abs().mean().item()),
        "align_max_abs": float((full_align - step_align).abs().max().item()),
        "align_mean_abs": float((full_align - step_align).abs().mean().item()),
        "align_argmax_mismatch": float(pos_mismatch.item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose equivalence of full Mamba decoder and Mamba.step decoder"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional Mamba checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--text_len", type=int, default=64)
    parser.add_argument("--mel_len", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--strict", action="store_true", help="Load checkpoint with strict=True")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = MambaTacotron2().to(device).eval()
    if model.decoder_type not in {"mamba", "mamba_step"}:
        raise ValueError(
            "This diagnostic expects mamba_decoder_type='mamba' or 'mamba_step', "
            f"got {model.decoder_type!r}"
        )

    if args.checkpoint:
        state = load_checkpoint_state(args.checkpoint, device)
        model.load_state_dict(state, strict=args.strict)
        model.eval()
        print(f"Loaded checkpoint: {args.checkpoint}")

    text, text_lengths, mel_input = make_random_batch(
        model=model,
        batch_size=args.batch_size,
        text_len=args.text_len,
        mel_len=args.mel_len,
        device=device,
    )

    metrics = compare_full_and_step(model, text, text_lengths, mel_input)

    print(f"Decoder type: {model.decoder_type}")
    print(f"Batch: batch_size={args.batch_size}, text_len={args.text_len}, mel_len={args.mel_len}")
    for key, value in metrics.items():
        print(f"{key}: {value:.8e}")


if __name__ == "__main__":
    main()
