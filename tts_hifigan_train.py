#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams_base as hp
from tts_dataset import get_hifigan_dataset, collate_fn_hifigan
from tts_hifigan_model import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)


def hp_get(name: str, default):
    return getattr(hp, name, default)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_num_workers() -> int:
    return int(hp_get("num_workers", 4))


def get_val_ratio() -> float:
    return float(hp_get("val_ratio", 0.02))


def get_seed() -> int:
    return int(hp_get("seed", 42))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mel_basis(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
    return torch.tensor(basis, device=device, dtype=dtype)


def wav_to_mel(wav: torch.Tensor, mel_basis: torch.Tensor) -> torch.Tensor:
    """
    wav: (B, T)
    returns: (B, n_mels, T_mel)
    """
    window = torch.hann_window(hp.win_length, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window=window,
        return_complex=True,
        center=True,
    )
    mag = torch.abs(spec).clamp_min(1e-5)
    mel = torch.matmul(mel_basis.unsqueeze(0), mag)
    mel = torch.log(mel.clamp_min(1e-5))
    return mel


@torch.no_grad()
def validate(generator, dataloader, device, mel_basis) -> dict[str, float]:
    generator.eval()
    total_l1 = 0.0
    total_mel = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="validation", leave=False):
        mel = batch["mel"].to(device)
        wav = batch["wav"].to(device)
        y = wav.unsqueeze(1)
        y_hat = generator(mel.transpose(1, 2))

        min_len = min(y.size(-1), y_hat.size(-1))
        y = y[..., :min_len]
        y_hat = y_hat[..., :min_len]

        l1 = torch.mean(torch.abs(y - y_hat))
        mel_real = wav_to_mel(y.squeeze(1), mel_basis)
        mel_fake = wav_to_mel(y_hat.squeeze(1), mel_basis)
        mel_loss = torch.mean(torch.abs(mel_real - mel_fake))

        total_l1 += float(l1.item())
        total_mel += float(mel_loss.item())
        n_batches += 1

    generator.train()

    if n_batches == 0:
        return {"l1": 0.0, "mel": 0.0}
    return {"l1": total_l1 / n_batches, "mel": total_mel / n_batches}


def save_checkpoint(generator, mpd, msd, optim_g, optim_d, epoch, global_step, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def main() -> None:
    set_seed(get_seed())
    device = get_device()

    full_dataset = get_hifigan_dataset()

    val_size = max(1, int(len(full_dataset) * get_val_ratio()))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(get_seed()),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(hp_get("hifigan_batch_size", 16)),
        shuffle=True,
        collate_fn=collate_fn_hifigan,
        drop_last=True,
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(hp_get("hifigan_batch_size", 16)),
        shuffle=False,
        collate_fn=collate_fn_hifigan,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=torch.cuda.is_available(),
    )

    generator = Generator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), lr=float(hp_get("hifigan_lr", 2e-4)), betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=float(hp_get("hifigan_lr", 2e-4)),
        betas=(0.8, 0.99),
    )

    mel_basis = build_mel_basis(device=device, dtype=torch.float32)

    checkpoint_dir = Path(hp_get("hifigan_checkpoint_path", "./outputs/checkpoints/hifigan"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lambda_fm = float(hp_get("hifigan_lambda_fm", 2.0))
    lambda_mel = float(hp_get("hifigan_lambda_mel", 45.0))
    save_every = int(hp_get("hifigan_save_step", 5000))
    val_every = int(hp_get("hifigan_val_step", 2000))
    global_step = 0

    for epoch in range(int(hp_get("hifigan_epochs", 10000))):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            global_step += 1

            mel = batch["mel"].to(device)               # (B, T_mel, n_mels)
            wav = batch["wav"].to(device)               # (B, T_audio)
            y = wav.unsqueeze(1)                        # (B, 1, T_audio)
            y_hat = generator(mel.transpose(1, 2))      # (B, 1, T_audio)

            min_len = min(y.size(-1), y_hat.size(-1))
            y = y[..., :min_len]
            y_hat = y_hat[..., :min_len]

            # Discriminators
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_hat.detach())
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_hat.detach())

            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc = loss_disc_f + loss_disc_s

            optim_d.zero_grad(set_to_none=True)
            loss_disc.backward()
            optim_d.step()

            # Generator
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_hat)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            mel_real = wav_to_mel(y.squeeze(1), mel_basis)
            mel_fake = wav_to_mel(y_hat.squeeze(1), mel_basis)
            loss_mel = torch.mean(torch.abs(mel_real - mel_fake))

            loss_gen = loss_gen_f + loss_gen_s + lambda_fm * (loss_fm_f + loss_fm_s) + lambda_mel * loss_mel

            optim_g.zero_grad(set_to_none=True)
            loss_gen.backward()
            optim_g.step()

            pbar.set_postfix(g=f"{float(loss_gen.item()):.4f}", d=f"{float(loss_disc.item()):.4f}", mel=f"{float(loss_mel.item()):.4f}")

            if global_step % val_every == 0:
                stats = validate(generator, val_loader, device, mel_basis)
                print(f"[VAL step={global_step}] l1={stats['l1']:.6f} mel={stats['mel']:.6f}")

            if global_step % save_every == 0:
                ckpt = checkpoint_dir / f"checkpoint_hifigan_{global_step}.pth.tar"
                save_checkpoint(generator, mpd, msd, optim_g, optim_d, epoch, global_step, ckpt)
                print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
