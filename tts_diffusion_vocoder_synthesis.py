#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from scipy.io.wavfile import write

import hyperparams_base as hp
from tts_diffusion_schedule import DiffusionSchedule
from tts_diffusion_vocoder_model import DiffusionVocoder


def hp_get(name: str, default):
    return getattr(hp, name, default)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_state(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path), map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def resolve_checkpoint(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    epoch = hp_get("restore_diffusion_vocoder_epoch", None)
    if epoch is None:
        raise ValueError("Diffusion vocoder checkpoint is not provided. Use --checkpoint.")
    checkpoint_dir = Path(hp_get("diffusion_vocoder_checkpoint_path", "./outputs/checkpoints/diffusion_vocoder"))
    return checkpoint_dir / f"checkpoint_diffusion_vocoder_epoch_{int(epoch):04d}.pth.tar"


def de_emphasis(wav: np.ndarray) -> np.ndarray:
    return signal.lfilter([1], [1, -float(hp.preemphasis)], wav).astype(np.float32)


def save_wav(path: str | Path, wav: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    write(str(path), int(sample_rate), wav)


@torch.no_grad()
def synthesize_mel(
    mel_path: str | Path,
    checkpoint_path: str | Path,
    out_path: str | Path,
    inference_steps: int,
    device: torch.device,
) -> Path:
    mel_np = np.load(mel_path).astype(np.float32, copy=False)
    if mel_np.ndim != 2:
        raise ValueError(f"Expected mel npy with shape (T, n_mels), got {mel_np.shape}")

    mel = torch.from_numpy(mel_np).unsqueeze(0).to(device)
    audio_length = int(mel.size(1)) * int(hp.hop_length)

    model = DiffusionVocoder().to(device).eval()
    model.load_state_dict(load_checkpoint_state(checkpoint_path, device=device), strict=True)
    schedule = DiffusionSchedule(
        timesteps=int(hp_get("diffusion_vocoder_train_timesteps", 1000)),
        device=device,
    )
    wav_t = schedule.sample(
        model=model,
        mel=mel,
        audio_length=audio_length,
        inference_steps=int(inference_steps),
    )
    wav = wav_t[0, 0].detach().cpu().numpy()
    wav = de_emphasis(wav)
    out_path = Path(out_path)
    save_wav(out_path, wav, hp.sr)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone diffusion vocoder synthesis from mel .npy")
    parser.add_argument("--mel_path", type=str, required=True, help="Path to normalized mel .npy with shape (T, n_mels)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to diffusion vocoder checkpoint")
    parser.add_argument("--out_path", type=str, default="outputs/synthesis/diffusion_vocoder/sample.wav")
    parser.add_argument("--inference_steps", type=int, default=int(hp_get("diffusion_vocoder_inference_steps", 50)))
    args = parser.parse_args()

    device = get_device()
    checkpoint = resolve_checkpoint(args.checkpoint)
    out_path = synthesize_mel(
        mel_path=args.mel_path,
        checkpoint_path=checkpoint,
        out_path=args.out_path,
        inference_steps=args.inference_steps,
        device=device,
    )
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Mel        : {args.mel_path}")
    print(f"WAV        : {out_path}")


if __name__ == "__main__":
    main()
