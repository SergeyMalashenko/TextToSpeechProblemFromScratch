#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training script for the independent MambaTacotron2 model."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams_mamba as hp

from tts_dataset import (
    get_tacotron_dataset, collate_fn_tacotron,
    compute_dataset_lengths, LengthBucketBatchSampler
)

from tts_mamba_model import MambaTacotron2
from tts_tacotron_losses import Tacotron2Loss, sequence_mask

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# =============================================================================
# Config helpers
# =============================================================================

def hp_get(name: str, default):
    return getattr(hp, name, default)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_num_workers() -> int:
    return int(hp_get("num_workers", 16))


def get_val_ratio() -> float:
    return float(hp_get("val_ratio", 0.02))


def get_guided_attn_weight() -> float:
    return float(hp_get("guided_attn_weight", 1.0))


def get_guided_attn_sigma() -> float:
    return float(hp_get("guided_attn_sigma", 0.4))


def get_gate_pos_weight() -> float:
    return float(hp_get("gate_pos_weight", 5.0))


def get_val_every_epoch() -> int:
    return int(hp_get("val_every_epoch", hp_get("validate_every_epoch", 1)))


def get_image_every_epoch() -> int:
    return int(hp_get("image_every_epoch", hp_get("log_alignment_every_epoch", 1)))


def get_checkpoint_every_epoch() -> int:
    return int(hp_get("checkpoint_every_epoch", hp_get("save_every_epoch", 1)))


def get_checkpoint_dir() -> Path:
    return Path(hp_get("checkpoint_path", "./checkpoint"))


def get_log_dir() -> Path:
    return Path(hp_get("mamba_log_dir", "./logs/mamba_tacotron"))


def get_samples_dir() -> Path:
    return Path(hp_get("sample_path", "./samples/mamba_tacotron"))


def get_max_checkpoints_to_keep() -> int:
    return int(hp_get("max_checkpoints_to_keep", 5))


def get_seed() -> int:
    return int(hp_get("seed", 42))


def get_validate_every_epoch() -> int:
    return int(hp_get("validate_every_epoch", 1))


def get_save_every_epoch() -> int:
    return int(hp_get("save_every_epoch", 1))


def get_sample_every_epoch() -> int:
    return int(hp_get("sample_every_epoch", 1))


def get_log_alignment_every_epoch() -> int:
    return int(hp_get("log_alignment_every_epoch", 1))


def get_lr_schedule_type() -> str:
    return str(hp_get("lr_schedule_type", "warmup_invsqrt_by_epoch"))


def get_lr_warmup_epochs() -> int:
    return int(hp_get("lr_warmup_epochs", 20))


def get_lr_hold_epochs() -> int:
    return int(hp_get("lr_hold_epochs", 0))


def get_lr_min() -> float:
    return float(hp_get("lr_min", 1e-5))


def get_lr_decay_gamma() -> float:
    return float(hp_get("lr_decay_gamma", 0.98))


def build_train_dataloader(train_dataset) -> DataLoader:
    """Builds the training DataLoader using the same strategy as tts_rnn_train_updated.py."""
    use_bucket_sampler = bool(hp_get("use_bucket_sampler", True))
    pin_memory = bool(hp_get("pin_memory", torch.cuda.is_available()))

    if use_bucket_sampler:
        bucket_size = int(hp_get("bucket_size", int(hp.batch_size) * 20))
        bucket_drop_last = bool(hp_get("bucket_drop_last", False))
        print("[DataLoader] Using LengthBucketBatchSampler")
        print(f"[DataLoader] batch_size={int(hp.batch_size)}, bucket_size={bucket_size}, drop_last={bucket_drop_last}")

        mel_lengths = compute_dataset_lengths(train_dataset)
        batch_sampler = LengthBucketBatchSampler(
            lengths=mel_lengths,
            batch_size=int(hp.batch_size),
            bucket_size=bucket_size,
            shuffle=True,
            drop_last=bucket_drop_last,
        )
        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=get_num_workers(),
            pin_memory=pin_memory,
            collate_fn=collate_fn_tacotron,
        )

    print("[DataLoader] Using plain random sampler")
    return DataLoader(
        train_dataset,
        batch_size=int(hp.batch_size),
        shuffle=True,
        collate_fn=collate_fn_tacotron,
        drop_last=True,
        num_workers=get_num_workers(),
        pin_memory=pin_memory,
    )


# =============================================================================
# Utility functions
# =============================================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    step_num: int,
    warmup_step: int = 4000,
) -> float:
    """Noam-style step schedule used by the updated RNN trainer."""
    lr = float(hp.lr) * (warmup_step ** 0.5) * min(
        step_num * (warmup_step ** -1.5),
        step_num ** -0.5,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def get_lr_by_step(
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

def get_lr_by_epoch(
    optimizer: torch.optim.Optimizer,
    epoch_num: int,
    warmup_epoch: int = 20,
) -> float:
    lr = hp.lr * (warmup_epoch ** 0.5) * min(
        epoch_num * (warmup_epoch ** -1.5),
        epoch_num ** -0.5,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)

def get_clip_grad_norm() -> float:
    return float(hp_get("clip_grad_norm", 1.0))


def get_weight_decay() -> float:
    return float(hp_get("weight_decay", 0.0))


def average_metric_dict(metric_list: list[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    totals = defaultdict(float)
    for metrics in metric_list:
        for key, value in metrics.items():
            totals[key] += float(value)
    n = float(len(metric_list))
    return {key: value / n for key, value in totals.items()}


class LinearScheduler:
    """
    Generic linear scheduler with a constant start and constant end.
    Epoch is 0-based.
    """
    def __init__(
        self,
        start_value: float,
        end_value: float,
        decay_start_epoch: int,
        decay_end_epoch: int,
    ) -> None:
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.decay_start_epoch = int(decay_start_epoch)
        self.decay_end_epoch = int(decay_end_epoch)

    def get_value(self, epoch: int) -> float:
        if epoch < self.decay_start_epoch:
            return self.start_value
        if epoch >= self.decay_end_epoch:
            return self.end_value

        denom = max(1, self.decay_end_epoch - self.decay_start_epoch)
        progress = float(epoch - self.decay_start_epoch) / float(denom)
        return self.start_value + progress * (self.end_value - self.start_value)


class GuidedAttentionScheduler(LinearScheduler):
    def __init__(self) -> None:
        super().__init__(
            start_value=float(hp_get("guided_attn_weight_start", get_guided_attn_weight())),
            end_value=float(hp_get("guided_attn_weight_end", 0.01)),
            decay_start_epoch=int(hp_get("guided_attn_decay_start_epoch", get_lr_warmup_epochs())),
            decay_end_epoch=int(hp_get("guided_attn_decay_end_epoch", int(hp.epochs))),
        )

    def get_weight(self, epoch: int) -> float:
        return self.get_value(epoch)


class TeacherForcingScheduler(LinearScheduler):
    def __init__(self) -> None:
        super().__init__(
            start_value=float(hp_get("teacher_forcing_start", 1.0)),
            end_value=float(hp_get("teacher_forcing_end", 0.2)),
            decay_start_epoch=int(hp_get("teacher_forcing_decay_start_epoch", get_lr_warmup_epochs())),
            decay_end_epoch=int(hp_get("teacher_forcing_decay_end_epoch", int(hp.epochs))),
        )

    def get_ratio(self, epoch: int) -> float:
        return self.get_value(epoch)


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def compute_gate_metrics(
    gate_logits: torch.Tensor,
    gate_target: torch.Tensor,
    output_lengths: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(gate_logits)
    pred = probs > threshold
    target = gate_target > 0.5

    tp = (pred & target).sum().item()
    fp = (pred & (~target)).sum().item()
    fn = ((~pred) & target).sum().item()
    tn = ((~pred) & (~target)).sum().item()

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)

    B, _ = probs.shape
    b_idx = torch.arange(B, device=probs.device)

    last_idx = (output_lengths - 1).clamp(min=0)
    prev_idx = (output_lengths - 2).clamp(min=0)

    p_last_mean = float(probs[b_idx, last_idx].mean().item())
    p_prev_mean = float(probs[b_idx, prev_idx].mean().item())

    early = []
    for b in range(B):
        L = int(output_lengths[b].item())
        if L <= 1:
            early.append(0.0)
            continue
        early_pred = pred[b, :L - 1].any().item()
        early.append(1.0 if early_pred else 0.0)

    early_stop_rate = float(sum(early) / max(1, len(early)))

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "early_stop_rate": float(early_stop_rate),
        "p_prev_mean": float(p_prev_mean),
        "p_last_mean": float(p_last_mean),
    }


@torch.no_grad()
def compute_attention_metrics(
    alignments: torch.Tensor,
    text_lengths: torch.Tensor,
    mel_lengths: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    total_entropy = 0.0
    total_peak = 0.0
    total_diag_mass = 0.0
    total_frames = 0

    for b in range(alignments.size(0)):
        t_len = int(mel_lengths[b].item())
        n_len = int(text_lengths[b].item())
        if t_len <= 0 or n_len <= 0:
            continue

        A = alignments[b, :t_len, :n_len].float()
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(eps)

        entropy = -(A * torch.log(A.clamp_min(eps))).sum(dim=-1)
        peak = A.max(dim=-1).values

        mel_pos = torch.arange(t_len, device=A.device, dtype=A.dtype) / max(float(t_len), 1.0)
        txt_pos = torch.arange(n_len, device=A.device, dtype=A.dtype) / max(float(n_len), 1.0)
        dist = torch.abs(mel_pos.unsqueeze(1) - txt_pos.unsqueeze(0))
        band = (dist <= 0.10).to(A.dtype)
        diag_mass = (A * band).sum(dim=-1)

        total_entropy += float(entropy.sum().item())
        total_peak += float(peak.sum().item())
        total_diag_mass += float(diag_mass.sum().item())
        total_frames += t_len

    if total_frames == 0:
        return {
            "attention_entropy": 0.0,
            "attention_peak_mean": 0.0,
            "attention_diag_mass": 0.0,
            "attention_sharpness": 0.0,
        }

    attention_entropy = total_entropy / total_frames
    attention_peak_mean = total_peak / total_frames
    attention_diag_mass = total_diag_mass / total_frames
    attention_sharpness = attention_peak_mean / max(attention_entropy, 1e-8)

    return {
        "attention_entropy": float(attention_entropy),
        "attention_peak_mean": float(attention_peak_mean),
        "attention_diag_mass": float(attention_diag_mass),
        "attention_sharpness": float(attention_sharpness),
    }


@torch.no_grad()
def compute_mel_metrics(
    mel_before: torch.Tensor,
    mel_after: torch.Tensor,
    mel_target: torch.Tensor,
    output_lengths: torch.Tensor,
) -> Dict[str, float]:
    mel_before = mel_before.float()
    mel_after = mel_after.float()
    mel_target = mel_target.float()

    mask = sequence_mask(output_lengths, max_len=mel_after.size(1)).unsqueeze(-1).to(torch.float32)

    mel_before_valid = mel_before * mask
    mel_after_valid = mel_after * mask
    mel_target_valid = mel_target * mask

    denom = mask.sum().clamp_min(1.0)
    norm = float(denom.item() * mel_after.size(-1))

    return {
        "mel_before_mean": float(mel_before_valid.sum().item() / norm),
        "mel_after_mean": float(mel_after_valid.sum().item() / norm),
        "mel_target_mean": float(mel_target_valid.sum().item() / norm),
        "mel_before_energy": float(mel_before_valid.abs().sum().item() / norm),
        "mel_after_energy": float(mel_after_valid.abs().sum().item() / norm),
        "mel_target_energy": float(mel_target_valid.abs().sum().item() / norm),
    }


@torch.no_grad()
def check_finite_tensor(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} contains non-finite values")


# =============================================================================
# Checkpoint helpers
# =============================================================================

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def set_guided_attn_weight(criterion: Tacotron2Loss, weight: float) -> None:
    criterion.guided_attn_weight = float(weight)


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
        checkpoint_dir.glob("checkpoint_mamba_tacotron2_epoch_*.pth.tar"),
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
    """
    Returns:
        start_epoch, global_step
    """
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


# =============================================================================
# Logging helpers
# =============================================================================

def log_alignment_image(
    writer: SummaryWriter,
    alignments: torch.Tensor,
    epoch_index: int,
    tag: str = "mamba_train_epoch/alignment",
) -> None:
    """
    alignments: (B, T_mel, T_text)
    """
    if alignments.ndim != 3 or alignments.size(0) == 0:
        return

    A = alignments[0].detach().float().cpu().unsqueeze(0)
    writer.add_image(tag, A, epoch_index)


@torch.no_grad()
def save_validation_sample(
    model: nn.Module,
    dataset,
    device: torch.device,
    save_dir: str | Path,
    epoch_index: int,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[0]

    text = torch.from_numpy(sample["text"]).long().unsqueeze(0).to(device)
    text_lengths = torch.tensor([sample["text_length"]], dtype=torch.long, device=device)

    model_w = unwrap_model(model)
    model_w.eval()
    outputs = model_w.inference(text=text, text_lengths=text_lengths)
    model_w.train()

    mel = outputs["mel_after"][0].detach().cpu()
    align = outputs["alignments"][0].detach().cpu()

    torch.save(mel, save_dir / f"mamba_sample_mel_epoch_{epoch_index:04d}.pt")
    torch.save(align, save_dir / f"mamba_sample_align_epoch_{epoch_index:04d}.pt")


# =============================================================================
# Train / validate
# =============================================================================

def train_one_step(
    model: nn.Module,
    criterion: Tacotron2Loss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    amp_device_type: str,
    guided_attn_weight: float,
    teacher_forcing_ratio: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    batch = move_batch_to_device(batch, device)
    set_guided_attn_weight(criterion, guided_attn_weight)

    with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
        outputs = model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            mel_input=batch["mel_input"],
            output_lengths=batch.get("output_lengths", None),
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        # RNN-equivalent Tacotron2Loss stores guided_attn_weight inside criterion.
        # Therefore forward signature is criterion(outputs, batch).
        loss_dict = criterion(outputs=outputs, batch=batch)

    check_finite_tensor("mel_before", outputs["mel_before"])
    check_finite_tensor("mel_after", outputs["mel_after"])
    check_finite_tensor("gate", outputs["gate"])
    check_finite_tensor("alignments", outputs["alignments"])
    check_finite_tensor("loss", loss_dict["loss"])

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss_dict["loss"]).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), get_clip_grad_norm())
    scaler.step(optimizer)
    scaler.update()

    stats = {
        "loss": float(loss_dict["loss"].item()),
        "mel_loss": float(loss_dict["mel_loss"].item()),
        "gate_loss": float(loss_dict["gate_loss"].item()),
        "attn_loss": float(loss_dict["attn_loss"].item()),
        "mel_before_loss": float(loss_dict["mel_before_loss"].item()),
        "mel_after_loss": float(loss_dict["mel_after_loss"].item()),
        "guided_attn_weight": float(guided_attn_weight),
        "teacher_forcing_ratio": float(teacher_forcing_ratio),
    }
    stats.update(compute_gate_metrics(outputs["gate"].detach(), batch["gate_target"], batch["output_lengths"]))
    stats.update(compute_attention_metrics(outputs["alignments"].detach(), batch["text_lengths"], batch["output_lengths"]))
    stats.update(compute_mel_metrics(outputs["mel_before"].detach(), outputs["mel_after"].detach(), batch["mel_target"], batch["output_lengths"]))
    return outputs, stats


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: Tacotron2Loss,
    dataloader: DataLoader,
    device: torch.device,
    amp_device_type: str,
    guided_attn_weight: float,
) -> Dict[str, float]:
    model.eval()
    set_guided_attn_weight(criterion, guided_attn_weight)
    totals = defaultdict(float)
    n_batches = 0

    for batch in tqdm(dataloader, desc="validation", leave=False):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
            outputs = model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                mel_input=batch["mel_input"],
                output_lengths=batch.get("output_lengths", None),
                teacher_forcing_ratio=1.0,
            )
            # RNN-equivalent Tacotron2Loss stores guided_attn_weight inside criterion.
            losses = criterion(outputs=outputs, batch=batch)

        check_finite_tensor("val/mel_before", outputs["mel_before"])
        check_finite_tensor("val/mel_after", outputs["mel_after"])
        check_finite_tensor("val/gate", outputs["gate"])
        check_finite_tensor("val/alignments", outputs["alignments"])
        check_finite_tensor("val/loss", losses["loss"])

        metrics = {
            "loss": float(losses["loss"].item()),
            "mel_loss": float(losses["mel_loss"].item()),
            "gate_loss": float(losses["gate_loss"].item()),
            "attn_loss": float(losses["attn_loss"].item()),
            "mel_before_loss": float(losses["mel_before_loss"].item()),
            "mel_after_loss": float(losses["mel_after_loss"].item()),
            "guided_attn_weight": float(guided_attn_weight),
            "teacher_forcing_ratio": 1.0,
        }
        metrics.update(compute_gate_metrics(outputs["gate"], batch["gate_target"], batch["output_lengths"]))
        metrics.update(compute_attention_metrics(outputs["alignments"], batch["text_lengths"], batch["output_lengths"]))
        metrics.update(compute_mel_metrics(outputs["mel_before"], outputs["mel_after"], batch["mel_target"], batch["output_lengths"]))

        for key, value in metrics.items():
            totals[key] += float(value)
        n_batches += 1

    model.train()
    if n_batches == 0:
        return {"loss": 0.0}
    return {key: value / n_batches for key, value in totals.items()}


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    set_seed(get_seed())

    device = get_device()
    amp_device_type = get_amp_device_type()

    print(f"Using device: {device}")
    print(f"AMP device type: {amp_device_type}")
    print("Mamba backend   : mamba_ssm")

    full_dataset = get_tacotron_dataset()

    val_size = max(1, int(len(full_dataset) * get_val_ratio()))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(get_seed()),
    )
   
    train_loader = build_train_dataloader(train_dataset)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(hp.batch_size),
        shuffle=False,
        collate_fn=collate_fn_tacotron,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=bool(hp_get("pin_memory", torch.cuda.is_available())),
    )
    
    model = MambaTacotron2().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp.lr))
    scaler = torch.amp.GradScaler(amp_device_type)

    criterion = Tacotron2Loss(
        gate_pos_weight=get_gate_pos_weight(),
        guided_attn_weight=float(hp_get("guided_attn_weight_start", get_guided_attn_weight())),
        guided_attn_sigma=get_guided_attn_sigma(),
    ).to(device)

    guided_attn_scheduler = GuidedAttentionScheduler()
    teacher_forcing_scheduler = TeacherForcingScheduler()

    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    resume_path = hp_get("resume_mamba_checkpoint", hp_get("resume_rnn_tacotron_path", None))
    start_epoch, global_step = maybe_resume_from_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resume_path=resume_path,
        device=device,
    )

    checkpoint_dir = get_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = get_samples_dir()
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size  : {len(val_dataset)}")
    print(f"Log dir           : {log_dir}")
    print(f"Optimizer         : Adam")
    print(f"Base LR           : {float(hp.lr):.2e}")
    print(f"Clip grad norm    : {get_clip_grad_norm():.2f}")
    print(f"Guided attn w     : {guided_attn_scheduler.start_value:.3f} -> {guided_attn_scheduler.end_value:.3f}")
    print(f"Guided attn sigma : {get_guided_attn_sigma():.3f}")
    print(f"Teacher forcing   : {teacher_forcing_scheduler.start_value:.3f} -> {teacher_forcing_scheduler.end_value:.3f}")
    print(f"Noam warmup steps : {int(hp_get('warmup_step', 4000))}")

    for epoch in range(start_epoch, hp.epochs):
        model.train()
        epoch_index = epoch + 1
        
        current_lr = get_lr_by_epoch(
            optimizer,
            epoch_index
        )
        
        current_guided_attn_weight = guided_attn_scheduler.get_weight(epoch)
        current_teacher_forcing_ratio = teacher_forcing_scheduler.get_ratio(epoch)

        pbar = tqdm(train_loader, desc=f"epoch {epoch_index}/{hp.epochs}")
        train_epoch_metrics = []
        last_outputs = None

        for batch in pbar:
            global_step += 1

            outputs, stats = train_one_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                device=device,
                amp_device_type=amp_device_type,
                guided_attn_weight=current_guided_attn_weight,
                teacher_forcing_ratio=current_teacher_forcing_ratio,
            )
            last_outputs = outputs
            train_epoch_metrics.append(stats)

            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                mel=f"{stats['mel_loss']:.4f}",
                gate=f"{stats['gate_loss']:.4f}",
                attn=f"{stats['attn_loss']:.4f}",
                sharp=f"{stats['attention_sharpness']:.4f}",
                melE=f"{stats['mel_after_energy']:.4f}",
                gaw=f"{stats['guided_attn_weight']:.3f}",
                tf=f"{stats['teacher_forcing_ratio']:.3f}",
                lr=f"{current_lr:.2e}",
            )

        train_stats = average_metric_dict(train_epoch_metrics)
        print(
            f"[TRAIN epoch={epoch_index}] "
            f"loss={train_stats['loss']:.4f} "
            f"mel={train_stats['mel_loss']:.4f} "
            f"gate={train_stats['gate_loss']:.4f} "
            f"attn={train_stats['attn_loss']:.4f} "
            f"sharp={train_stats['attention_sharpness']:.4f} "
            f"melE={train_stats['mel_after_energy']:.4f} "
            f"gaw={train_stats['guided_attn_weight']:.3f} "
            f"tf={train_stats['teacher_forcing_ratio']:.3f} "
            f"lr={current_lr:.2e}"
        )

        for key, value in train_stats.items():
            writer.add_scalar(f"train/{key}", value, epoch_index)
        writer.add_scalar("train/lr", current_lr, epoch_index)

        if (epoch_index % get_log_alignment_every_epoch() == 0) and (last_outputs is not None):
            log_alignment_image(
                writer=writer,
                alignments=last_outputs["alignments"].detach(),
                epoch_index=epoch_index,
                tag="attention/mamba_alignment",
            )

        if epoch_index % get_validate_every_epoch() == 0:
            val_stats = validate(
                model=model,
                criterion=criterion,
                dataloader=val_loader,
                device=device,
                amp_device_type=amp_device_type,
                guided_attn_weight=current_guided_attn_weight,
            )
            print(
                f"[VAL epoch={epoch_index}] "
                f"loss={val_stats['loss']:.4f} "
                f"mel={val_stats['mel_loss']:.4f} "
                f"gate={val_stats['gate_loss']:.4f} "
                f"attn={val_stats['attn_loss']:.4f} "
                f"sharp={val_stats['attention_sharpness']:.4f} "
                f"melE={val_stats['mel_after_energy']:.4f} "
                f"gaw={val_stats['guided_attn_weight']:.3f} "
                f"tf={val_stats['teacher_forcing_ratio']:.3f} "
                f"acc={val_stats['accuracy']:.4f}"
            )
            for key, value in val_stats.items():
                writer.add_scalar(f"val/{key}", value, epoch_index)

        if epoch_index % get_sample_every_epoch() == 0:
            save_validation_sample(
                model=model,
                dataset=val_dataset,
                device=device,
                save_dir=sample_dir,
                epoch_index=epoch_index,
            )

        if epoch_index % get_save_every_epoch() == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_mamba_tacotron2_epoch_{epoch_index:04d}.pth.tar"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch_index,
                global_step=global_step,
                path=ckpt_path,
            )
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=get_max_checkpoints_to_keep())
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    main()
