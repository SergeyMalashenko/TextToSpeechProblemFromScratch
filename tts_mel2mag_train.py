#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams_mel2mag as hp
from tts_dataset import get_mel2mag_dataset, collate_fn_mel2mag
from tts_mel2mag_model import MelToMagModel
from tts_mel2mag_losses import MelToMagLoss
from tts_seed import make_torch_generator, seed_worker, set_seed


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


def get_checkpoint_every_epoch() -> int:
    interval = int(hp_get("mel2mag_save_every_epoch", 1))
    if interval < 1:
        raise ValueError("mel2mag_save_every_epoch must be at least 1")
    return interval


def get_max_checkpoints_to_keep() -> int:
    return int(hp_get("max_checkpoints_to_keep", 5))


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
    best_val_loss: float = float("inf"),
    batch_in_epoch: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "model_config": unwrap_model(model).config,
            "training_config": {key: value for key, value in vars(hp).items()
                                if not key.startswith("_") and isinstance(value, (str, int, float, bool, list, tuple, type(None)))},
            "best_val_loss": best_val_loss,
            "batch_in_epoch": batch_in_epoch,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def cleanup_old_checkpoints(checkpoint_dir: str | Path, keep_last_n: int) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if keep_last_n <= 0 or not checkpoint_dir.exists():
        return

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_mel2mag_epoch_*.pth.tar"),
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
    if checkpoint.get("model_config") != unwrap_model(model).config:
        raise ValueError("Resume requires matching residual model_config; legacy weights are inference-only.")
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
    criterion: MelToMagLoss,
) -> Dict[str, float]:
    batch = move_batch_to_device(batch, device)

    with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
        mel_input = batch["mel"]
        noise_std = float(hp_get("mel2mag_input_noise_std", 0.0))
        if noise_std > 0:
            mel_input = (mel_input + torch.randn_like(mel_input) * noise_std).clamp(0, 1)
        mag_pred = model(mel_input, batch["lengths"])
        losses = criterion(mag_pred, batch["mag"], batch["mel"], batch["lengths"])
        loss = losses["loss"]

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return {key: float(value.detach().item()) for key, value in losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_device_type: str,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_elements = 0
    outside_elements = 0
    try:
        for batch in tqdm(dataloader, desc="validation", leave=False):
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
                mag_pred = model(batch["mel"], batch["lengths"])
            # Select best by the unchanged, valid-bin L1 metric across ablations.
            loss = masked_l1_loss(mag_pred.float(), batch["mag"], batch["lengths"])
            count = int(batch["lengths"].sum().item()) * mag_pred.size(-1)
            total_loss += float(loss.item()) * count
            total_elements += count
            valid = sequence_mask(batch["lengths"], mag_pred.size(1)).unsqueeze(-1)
            outside_elements += int((((mag_pred < 0) | (mag_pred > 1)) & valid).sum().item())
    finally:
        model.train(was_training)
    if total_elements == 0:
        raise ValueError("Validation dataset has no valid frames")
    return {"loss": total_loss / total_elements, "outside_range": outside_elements / total_elements}


@torch.no_grad()
def save_validation_sample(model, dataloader, device, epoch):
    import numpy as np
    from scipy.io.wavfile import write
    from data_pipeline import spectrogram_to_wav

    was_training = model.training
    model.eval()
    try:
        batch = move_batch_to_device(next(iter(dataloader)), device)
        length = int(batch["lengths"][0].item())
        # Always use the first validation utterance for listening comparisons.
        mel = batch["mel"][:1, :length]
        prediction = model(mel)[0].float().cpu().numpy()
        target = batch["mag"][0, :length].cpu().numpy()
        directory = Path(hp_get("sample_path", "./outputs/samples/mel2mag"))
        directory.mkdir(parents=True, exist_ok=True)
        prefix = directory / f"mel2mag_epoch_{epoch:04d}"
        np.save(f"{prefix}_mel.npy", mel[0].cpu().numpy())
        for name, magnitude in (("predicted", prediction), ("target", target)):
            np.save(f"{prefix}_{name}.npy", magnitude)
            wav = spectrogram_to_wav(magnitude)
            write(f"{prefix}_{name}.wav", hp.sr, wav.astype(np.float32))
    finally:
        model.train(was_training)


def main() -> None:
    checkpoint_every = get_checkpoint_every_epoch()
    seed = get_seed()
    set_seed(seed)

    device = get_device()
    amp_device_type = get_amp_device_type()

    print(f"Using device: {device}")
    print(f"AMP device type: {amp_device_type}")

    full_dataset = get_mel2mag_dataset()

    if len(full_dataset) < 2 or not 0 < get_val_ratio() < 1:
        raise ValueError("MelToMag needs at least two examples and 0 < val_ratio < 1")
    val_size = max(1, int(len(full_dataset) * get_val_ratio()))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=make_torch_generator(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=collate_fn_mel2mag,
        drop_last=False,
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=collate_fn_mel2mag,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )

    model = MelToMagModel().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    mel_basis = None
    mel_weight = float(hp_get("mel2mag_mel_consistency_weight", 0.1))
    if mel_weight:
        import librosa
        mel_basis = torch.from_numpy(librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels))
    criterion = MelToMagLoss(
        max_db=hp.max_db, ref_db=hp.ref_db, mel_basis=mel_basis, mel_weight=mel_weight,
        temporal_weight=float(hp_get("mel2mag_temporal_weight", 0.1)),
        frequency_weight=float(hp_get("mel2mag_frequency_weight", 0.1)),
        spectral_weight=float(hp_get("mel2mag_spectral_convergence_weight", 0.0)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    scaler = torch.amp.GradScaler(amp_device_type)
    writer = SummaryWriter(log_dir=str(Path(hp_get("mel2mag_log_dir", "./logs"))))

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

    best_val_loss = float("inf")
    resume_batch = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        resume_batch = int(checkpoint.get("batch_in_epoch", 0))

    def evaluate_and_save(epoch):
        nonlocal best_val_loss
        val_stats = validate(model, val_loader, device, amp_device_type)
        for key, value in val_stats.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        improved = val_stats["loss"] < best_val_loss
        best_val_loss = min(best_val_loss, val_stats["loss"])
        kwargs = dict(model=model, optimizer=optimizer, scaler=scaler, epoch=epoch,
                      global_step=global_step, best_val_loss=best_val_loss,
                      batch_in_epoch=0)
        if improved:
            save_checkpoint(**kwargs, path=checkpoint_dir / "mel2mag_best.pth.tar")
        save_checkpoint(**kwargs, path=checkpoint_dir / "mel2mag_last.pth.tar")
        if epoch % checkpoint_every == 0 or epoch == int(hp.epochs):
            save_checkpoint(**kwargs, path=checkpoint_dir / f"checkpoint_mel2mag_epoch_{epoch:04d}.pth.tar")
            cleanup_old_checkpoints(checkpoint_dir, get_max_checkpoints_to_keep())

    for epoch in range(start_epoch, hp.epochs):
        train_loader.generator.manual_seed(seed + epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{int(hp.epochs)}")

        for batch_index, batch in enumerate(pbar):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            global_step += 1

            current_lr = adjust_learning_rate(
                optimizer, global_step, int(hp_get("mel2mag_warmup_steps", 4000)))

            stats = train_one_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                device=device,
                amp_device_type=amp_device_type,
                criterion=criterion,
            )

            pbar.set_postfix(loss=f"{stats['loss']:.4f}", lr=f"{current_lr:.6f}")
            for key, value in stats.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            writer.add_scalar("train/lr", current_lr, global_step)

        # Validate and save only after a completed epoch. global_step remains
        # the optimizer/LR counter, and old mid-epoch checkpoints can still resume.
        evaluate_and_save(epoch + 1)
        sample_every = int(hp_get("mel2mag_sample_every_epoch", 5))
        if sample_every > 0 and (epoch + 1) % sample_every == 0:
            save_validation_sample(model, val_loader, device, epoch + 1)

    writer.close()


if __name__ == "__main__":
    main()
