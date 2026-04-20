#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams_base as hp
from tts_dataset import get_vocoder_dataset, collate_fn_vocoder
from tts_vocoder_model import SimpleVocoder


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


def save_checkpoint(model, optimizer, epoch: int, global_step: int, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


@torch.no_grad()
def validate(model, dataloader, device) -> float:
    model.eval()
    total = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="validation", leave=False):
        mel = batch["mel"].to(device)
        wav = batch["wav"].to(device).unsqueeze(1)
        pred = model(mel)
        min_len = min(pred.size(-1), wav.size(-1))
        loss = F.l1_loss(pred[..., :min_len], wav[..., :min_len])
        total += float(loss.item())
        count += 1

    model.train()
    return total / max(1, count)


def main() -> None:
    set_seed(get_seed())
    device = get_device()

    dataset = get_vocoder_dataset()

    val_size = max(1, int(len(dataset) * get_val_ratio()))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(get_seed()),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(hp_get("vocoder_batch_size", 16)),
        shuffle=True,
        collate_fn=collate_fn_vocoder,
        drop_last=True,
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(hp_get("vocoder_batch_size", 16)),
        shuffle=False,
        collate_fn=collate_fn_vocoder,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleVocoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hp_get("vocoder_lr", 2e-4)), betas=(0.8, 0.99))

    checkpoint_dir = Path(hp_get("vocoder_checkpoint_path", "./checkpoint_vocoder"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_every = int(hp_get("vocoder_save_step", 5000))
    global_step = 0

    for epoch in range(int(hp_get("vocoder_epochs", 10000))):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            global_step += 1

            mel = batch["mel"].to(device)
            wav = batch["wav"].to(device).unsqueeze(1)

            pred = model(mel)
            min_len = min(pred.size(-1), wav.size(-1))
            pred = pred[..., :min_len]
            wav = wav[..., :min_len]

            loss = F.l1_loss(pred, wav)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

            if global_step % save_every == 0:
                ckpt = checkpoint_dir / f"checkpoint_vocoder_{global_step}.pth.tar"
                save_checkpoint(model, optimizer, epoch, global_step, ckpt)
                val_l1 = validate(model, val_loader, device)
                print(f"[VAL step={global_step}] l1={val_l1:.6f}")
                print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
