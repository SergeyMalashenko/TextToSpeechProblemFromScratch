#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams as hp
from tts_dataset import get_mel2mag_dataset, collate_fn_mel2mag
from tts_mel2mag_model import MelToMagModel


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def hp_get(name: str, default):
    return getattr(hp, name, default)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_num_workers() -> int:
    return int(hp_get("num_workers", 4))


def get_val_ratio() -> float:
    return float(hp_get("val_ratio", 0.02))


def get_seed() -> int:
    return int(hp_get("seed", 42))


def get_checkpoint_every() -> int:
    return int(hp_get("save_step", 2000))


def get_max_checkpoints_to_keep() -> int:
    return int(hp_get("max_checkpoints_to_keep", 5))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) < lengths.unsqueeze(1)


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    step_num: int,
    warmup_step: int = 4000,
) -> float:
    lr = hp.lr * (warmup_step ** 0.5) * min(
        step_num * (warmup_step ** -1.5),
        step_num ** -0.5,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = sequence_mask(lengths, max_len=pred.size(1)).unsqueeze(-1).to(pred.dtype)
    loss = torch.abs(pred - target) * mask
    denom = mask.sum() * pred.size(-1)
    return loss.sum() / denom.clamp_min(1.0)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def cleanup_old_checkpoints(checkpoint_dir: str | Path, keep_last_n: int) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_mel2mag_*.pth.tar"),
        key=lambda p: p.stat().st_mtime,
    )

    if len(checkpoints) <= keep_last_n:
        return

    for ckpt in checkpoints[:-keep_last_n]:
        try:
            ckpt.unlink()
        except OSError:
            pass


def maybe_resume_from_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    resume_path: Optional[str | Path],
    device: torch.device,
) -> tuple[int, int]:
    if not resume_path:
        return 0, 0

    resume_path = Path(resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    print(f"Resumed from checkpoint: {resume_path}")
    print(f"Start epoch: {start_epoch}, global step: {global_step}")
    return start_epoch, global_step


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    amp_device_type: str,
) -> Dict[str, float]:
    batch = move_batch_to_device(batch, device)

    with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
        mag_pred = model(batch["mel"])
        loss = masked_l1_loss(
            pred=mag_pred,
            target=batch["mag"],
            lengths=batch["lengths"],
        )

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return {"loss": float(loss.item())}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_device_type: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="validation", leave=False):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
            mag_pred = model(batch["mel"])
            loss = masked_l1_loss(pred=mag_pred, target=batch["mag"], lengths=batch["lengths"])
        total_loss += float(loss.item())
        n_batches += 1

    model.train()
    if n_batches == 0:
        return {"loss": 0.0}
    return {"loss": total_loss / n_batches}


def main() -> None:
    set_seed(get_seed())

    device = get_device()
    amp_device_type = get_amp_device_type()

    print(f"Using device: {device}")
    print(f"AMP device type: {amp_device_type}")

    full_dataset = get_mel2mag_dataset()

    val_size = max(1, int(len(full_dataset) * get_val_ratio()))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(get_seed()),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=collate_fn_mel2mag,
        drop_last=True,
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=collate_fn_mel2mag,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = MelToMagModel().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    scaler = torch.amp.GradScaler(amp_device_type)
    writer = SummaryWriter()

    resume_path = hp_get("resume_mel2mag_checkpoint", None)
    start_epoch, global_step = maybe_resume_from_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resume_path=resume_path,
        device=device,
    )

    checkpoint_dir = Path(hp.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, hp.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for _, batch in enumerate(pbar):
            global_step += 1

            if global_step < 400000:
                current_lr = adjust_learning_rate(optimizer, global_step)
            else:
                current_lr = float(optimizer.param_groups[0]["lr"])

            stats = train_one_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                device=device,
                amp_device_type=amp_device_type,
            )

            pbar.set_postfix(loss=f"{stats['loss']:.4f}", lr=f"{current_lr:.6f}")
            writer.add_scalar("train/loss", stats["loss"], global_step)
            writer.add_scalar("train/lr", current_lr, global_step)

            if global_step % get_checkpoint_every() == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_mel2mag_{global_step}.pth.tar"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    path=ckpt_path,
                )
                cleanup_old_checkpoints(checkpoint_dir, keep_last_n=get_max_checkpoints_to_keep())

                val_stats = validate(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    amp_device_type=amp_device_type,
                )
                writer.add_scalar("val/loss", val_stats["loss"], global_step)

    writer.close()


if __name__ == "__main__":
    main()
