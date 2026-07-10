#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams_transformer as hp
from tts_dataset import (
    get_tacotron_dataset, collate_fn_tacotron,
    compute_dataset_lengths, LengthBucketBatchSampler
)


from tts_transformer_model import TransformerTacotron2
from tts_tacotron_losses import Tacotron2Loss, sequence_mask

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
    return int(hp_get("num_workers", 8))


def get_val_ratio() -> float:
    return float(hp_get("val_ratio", 0.02))


def get_gate_pos_weight() -> float:
    return float(hp_get("gate_pos_weight", 10.0))


def get_guided_attn_sigma() -> float:
    return float(hp_get("guided_attn_sigma", 0.2))


def get_val_every_epoch() -> int:
    return int(hp_get("val_every_epoch", 1))


def get_image_every_epoch() -> int:
    return int(hp_get("image_every_epoch", 1))


def get_sample_every_epoch() -> int:
    return int(hp_get("sample_every_epoch", 5))


def get_checkpoint_every_epoch() -> int:
    return int(hp_get("checkpoint_every_epoch", 5))


def get_max_checkpoints_to_keep() -> int:
    return int(hp_get("max_checkpoints_to_keep", 5))


def get_seed() -> int:
    return int(hp_get("seed", 42))


def get_checkpoint_dir() -> Path:
    return Path(hp_get("checkpoint_path", "./checkpoint"))


def get_log_dir() -> Path:
    return Path(hp_get("transformer_log_dir", hp_get("log_dir", "./outputs/logs/transformer")))


def get_samples_dir() -> Path:
    return Path(hp_get("sample_path", "./outputs/samples/transformer"))


def _safe_experiment_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    safe_chars = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_")


def create_experiment_log_dir(base_dir: str | Path, prefix: str) -> Path:
    """
    Creates a separate TensorBoard directory for each experiment run.

    Example:
        ./logs/transformer_tacotron/transformer_20260616_173245/
        ./logs/transformer_tacotron/transformer_20260616_173245_my_experiment/
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = _safe_experiment_name(str(hp_get("experiment_name", "")))

    run_name = f"{prefix}_{timestamp}"
    if experiment_name:
        run_name = f"{run_name}_{experiment_name}"

    run_dir = base_dir / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{run_name}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def get_lr_by_epoch(epoch: int) -> float:
    """
    Epoch-based piecewise LR schedule.

    epoch is 0-based. All thresholds in hyperparams are 1-based human epoch
    numbers. This keeps training behavior independent of batch size and
    DataLoader length.
    """
    e = epoch + 1
    base_lr = float(hp.lr)

    warmup_epochs = int(hp_get("lr_warmup_epochs", 20))
    decay_epoch_1 = int(hp_get("lr_decay_epoch_1", 150))
    decay_epoch_2 = int(hp_get("lr_decay_epoch_2", 300))
    decay_factor_1 = float(hp_get("lr_decay_factor_1", 0.5))
    decay_factor_2 = float(hp_get("lr_decay_factor_2", 0.25))
    lr_min = float(hp_get("lr_min", 1e-6))

    if warmup_epochs > 0 and e <= warmup_epochs:
        lr = base_lr * e / float(warmup_epochs)
    elif e <= decay_epoch_1:
        lr = base_lr
    elif e <= decay_epoch_2:
        lr = base_lr * decay_factor_1
    else:
        lr = base_lr * decay_factor_2

    return max(float(lr), lr_min)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def get_clip_grad_norm() -> float:
    return float(hp_get("clip_grad_norm", 1.0))


def get_weight_decay() -> float:
    return float(hp_get("weight_decay", 1e-4))


def average_metric_dict(metric_list: list[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    totals = defaultdict(float)
    for metrics in metric_list:
        for key, value in metrics.items():
            totals[key] += float(value)
    n = float(len(metric_list))
    return {key: value / n for key, value in totals.items()}


class GuidedAttentionScheduler:
    def __init__(self) -> None:
        self.weight_start  = float(hp_get("guided_attn_weight_start" , 2.0))
        self.weight_end    = float(hp_get("guided_attn_weight_end"   , 0.1))
        self.warmup_epochs = int  (hp_get("guided_attn_warmup_epochs", 30 ))
        self.decay_epochs  = int  (hp_get("guided_attn_decay_epochs" , 60 ))

    def get_weight(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.weight_start
        decay_epoch = epoch - self.warmup_epochs
        if decay_epoch >= self.decay_epochs:
            return self.weight_end
        progress = decay_epoch / float(self.decay_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        weight = self.weight_end + (self.weight_start - self.weight_end) * cosine
        return float(weight)



@torch.no_grad()
def compute_gate_metrics(gate_logits: torch.Tensor, gate_target: torch.Tensor, output_lengths: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
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

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "early_stop_rate": float(sum(early) / max(1, len(early))),
        "p_prev_mean": float(p_prev_mean),
        "p_last_mean": float(p_last_mean),
    }


@torch.no_grad()
def compute_attention_metrics(alignments: torch.Tensor, text_lengths: torch.Tensor, mel_lengths: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
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
def compute_mel_metrics(mel_before: torch.Tensor, mel_after: torch.Tensor, mel_target: torch.Tensor, output_lengths: torch.Tensor) -> Dict[str, float]:
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


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def set_guided_attn_weight(criterion: Tacotron2Loss, weight: float) -> None:
    criterion.guided_attn_weight = float(weight)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, epoch: int, global_step: int, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)


def cleanup_old_checkpoints(checkpoint_dir: str | Path, keep_last_n: int) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_transformer_tacotron2_epoch_*.pth.tar"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(checkpoints) <= keep_last_n:
        return
    for ckpt in checkpoints[:-keep_last_n]:
        try:
            ckpt.unlink()
        except OSError:
            pass


def maybe_resume_from_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, resume_path: Optional[str | Path], device: torch.device) -> tuple[int, int]:
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


def log_alignment_image(writer: SummaryWriter, alignments: torch.Tensor, epoch: int, tag: str = "attention/alignment") -> None:
    if alignments.ndim != 3 or alignments.size(0) == 0:
        return
    A = alignments[0].detach().float().cpu().unsqueeze(0)
    writer.add_image(tag, A, epoch)


@torch.no_grad()
def save_validation_sample(model: nn.Module, dataset, device: torch.device, save_dir: str | Path, epoch: int) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset[0]
    text = torch.from_numpy(sample["text"]).long().unsqueeze(0).to(device)
    text_lengths = torch.tensor([sample["text_length"]], dtype=torch.long, device=device)

    model_w = unwrap_model(model)
    model_w.eval()
    outputs = model_w.inference(text=text, text_lengths=text_lengths)
    model_w.train()

    torch.save(outputs["mel_after"][0].detach().cpu(), save_dir / f"transformer_sample_mel_epoch_{epoch:04d}.pt")
    torch.save(outputs["alignments"][0].detach().cpu(), save_dir / f"transformer_sample_align_epoch_{epoch:04d}.pt")


def train_one_step(model: nn.Module, criterion: Tacotron2Loss, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, batch: Dict[str, torch.Tensor], device: torch.device, amp_device_type: str, guided_attn_weight: float) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    batch = move_batch_to_device(batch, device)
    set_guided_attn_weight(criterion, guided_attn_weight)

    with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
        outputs = model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            mel_input=batch["mel_input"],
            output_lengths=batch["output_lengths"],
        )
        losses = criterion(outputs=outputs, batch=batch)

    check_finite_tensor("mel_before", outputs["mel_before"])
    check_finite_tensor("mel_after", outputs["mel_after"])
    check_finite_tensor("gate", outputs["gate"])
    check_finite_tensor("alignments", outputs["alignments"])
    check_finite_tensor("loss", losses["loss"])

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(losses["loss"]).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), get_clip_grad_norm())
    scaler.step(optimizer)
    scaler.update()

    metrics = {
        "loss": float(losses["loss"].item()),
        "mel_loss": float(losses["mel_loss"].item()),
        "gate_loss": float(losses["gate_loss"].item()),
        "attn_loss": float(losses["attn_loss"].item()),
        "mel_before_loss": float(losses["mel_before_loss"].item()),
        "mel_after_loss": float(losses["mel_after_loss"].item()),
        "guided_attn_weight": float(guided_attn_weight),
    }
    metrics.update(compute_gate_metrics(outputs["gate"].detach(), batch["gate_target"], batch["output_lengths"]))
    metrics.update(compute_attention_metrics(outputs["alignments"].detach(), batch["text_lengths"], batch["output_lengths"]))
    metrics.update(compute_mel_metrics(outputs["mel_before"].detach(), outputs["mel_after"].detach(), batch["mel_target"], batch["output_lengths"]))
    log_outputs = {
        "alignments": outputs["alignments"].detach().cpu(),
    }
    return log_outputs, metrics


@torch.no_grad()
def compute_autoregressive_metrics(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    model_w = unwrap_model(model)
    text = batch["text"][:1].to(device)
    text_lengths = batch["text_lengths"][:1].to(device)
    target_frames = int(batch["output_lengths"][0].item())

    outputs = model_w.inference(text=text, text_lengths=text_lengths)
    alignments = outputs["alignments"][0]
    gate = outputs["gate"][0]
    generated_frames = int(alignments.size(0))
    text_length = int(text_lengths[0].item())

    valid_alignments = alignments[:, :text_length]
    positions = valid_alignments.argmax(dim=-1)
    backward_jumps = (positions[1:] < positions[:-1]).float().mean() if positions.numel() > 1 else positions.new_tensor(0.0)
    final_position = int(positions[-1].item()) if positions.numel() > 0 else 0
    coverage = (final_position + 1) / max(1, text_length)

    return {
        "ar_generated_frames": float(generated_frames),
        "ar_target_frames": float(target_frames),
        "ar_length_ratio": float(generated_frames / max(1, target_frames)),
        "ar_gate_last_prob": float(torch.sigmoid(gate[-1]).item()) if gate.numel() > 0 else 0.0,
        "ar_text_coverage": float(coverage),
        "ar_backward_jump_rate": float(backward_jumps.item()),
    }


@torch.no_grad()
def validate(model: nn.Module, criterion: Tacotron2Loss, dataloader: DataLoader, device: torch.device, amp_device_type: str, guided_attn_weight: float) -> Dict[str, float]:
    model.eval()
    set_guided_attn_weight(criterion, guided_attn_weight)
    totals = defaultdict(float)
    n_batches = 0
    autoregressive_batch = None

    for batch in tqdm(dataloader, desc="validation", leave=False):
        batch = move_batch_to_device(batch, device)
        if autoregressive_batch is None:
            autoregressive_batch = {
                "text": batch["text"][:1].detach().cpu(),
                "text_lengths": batch["text_lengths"][:1].detach().cpu(),
                "output_lengths": batch["output_lengths"][:1].detach().cpu(),
            }
        with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
            outputs = model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                mel_input=batch["mel_input"],
                output_lengths=batch["output_lengths"],
            )
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
    metrics = {key: value / n_batches for key, value in totals.items()}
    if autoregressive_batch is not None:
        metrics.update(compute_autoregressive_metrics(model, autoregressive_batch, device))
    return metrics


def main() -> None:
    set_seed(get_seed())
    device = get_device()
    amp_device_type = get_amp_device_type()

    print(f"Using device: {device}")
    print(f"AMP device type: {amp_device_type}")

    full_dataset = get_tacotron_dataset()
    val_size = max(1, int(len(full_dataset) * get_val_ratio()))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(get_seed()),
    )
    
    use_bucket_sampler = bool(getattr(hp, "use_bucket_sampler", True))

    if use_bucket_sampler :
        print("[DataLoader] Using LengthBucketBatchSampler")
        mel_lengths = compute_dataset_lengths(train_dataset)
        
        batch_sampler = LengthBucketBatchSampler(
            lengths=mel_lengths,
            batch_size=hp.batch_size,
            bucket_size=hp.batch_size * 20,
            shuffle=True,
            drop_last=False,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=hp.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_tacotron,
        )
    else:
        print("[DataLoader] Using plain random sampler")
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(hp.batch_size),
            shuffle=True,
            collate_fn=collate_fn_tacotron,
            drop_last=True,
            num_workers=get_num_workers(),
            pin_memory=bool(hp_get("pin_memory", torch.cuda.is_available())),
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(hp.batch_size),
        shuffle=False,
        collate_fn=collate_fn_tacotron,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=bool(hp_get("pin_memory", torch.cuda.is_available())),
    )

    model = TransformerTacotron2().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp.lr),
        betas=(float(hp_get("adam_beta1", 0.9)), float(hp_get("adam_beta2", 0.98))),
        eps=float(hp_get("adam_eps", 1e-9)),
        weight_decay=get_weight_decay(),
    )
    scaler = torch.amp.GradScaler(amp_device_type)
    criterion = Tacotron2Loss(
        gate_pos_weight=get_gate_pos_weight(),
        guided_attn_weight=float(hp_get("guided_attn_weight_start", 2.0)),
        guided_attn_sigma=get_guided_attn_sigma(),
    ).to(device)

    guided_attn_scheduler = GuidedAttentionScheduler()

    checkpoint_dir = get_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = create_experiment_log_dir(get_log_dir(), prefix="transformer")
    writer = SummaryWriter(log_dir=str(log_dir))
    sample_dir = get_samples_dir()
    sample_dir.mkdir(parents=True, exist_ok=True)

    resume_path = hp_get("resume_transformer_tacotron_path", None)
    start_epoch, global_step = maybe_resume_from_checkpoint(model, optimizer, scaler, resume_path, device)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size  : {len(val_dataset)}")
    print(f"Log dir           : {log_dir}")
    print(f"Checkpoint dir    : {checkpoint_dir}")
    print(f"Sample dir        : {sample_dir}")
    print(f"Optimizer         : AdamW")
    print(f"Base LR           : {float(hp.lr):.2e}")
    print(f"Weight decay      : {get_weight_decay():.2e}")
    print(f"Clip grad norm    : {get_clip_grad_norm():.2f}")
    print(f"Seed              : {get_seed()}")

    writer.add_text("config/log_dir", str(log_dir), 0)
    writer.add_text("config/seed", str(get_seed()), 0)
    writer.add_text("config/optimizer", "AdamW", 0)
    writer.add_text("config/base_lr", f"{float(hp.lr):.6e}", 0)
    writer.add_text("config/weight_decay", f"{get_weight_decay():.6e}", 0)

    for epoch in range(start_epoch, int(hp.epochs)):
        epoch_idx = epoch + 1

        current_lr = get_lr_by_epoch(epoch)
        set_optimizer_lr(optimizer, current_lr)

        current_guided_attn_weight = guided_attn_scheduler.get_weight(epoch)

        pbar = tqdm(train_loader, desc=f"epoch {epoch_idx}/{int(hp.epochs)}")
        train_epoch_metrics = []
        last_outputs = None

        for batch in pbar:
            global_step += 1

            outputs, metrics = train_one_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                device=device,
                amp_device_type=amp_device_type,
                guided_attn_weight=current_guided_attn_weight,
            )

            last_outputs = outputs
            train_epoch_metrics.append(metrics)

            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                mel=f"{metrics['mel_loss']:.4f}",
                gate=f"{metrics['gate_loss']:.4f}",
                attn=f"{metrics['attn_loss']:.4f}",
                sharp=f"{metrics['attention_sharpness']:.4f}",
                melE=f"{metrics['mel_after_energy']:.4f}",
                gaw=f"{metrics['guided_attn_weight']:.3f}",
                lr=f"{current_lr:.2e}",
            )

        train_metrics = average_metric_dict(train_epoch_metrics)
        print(
            f"[TRAIN epoch={epoch_idx}] "
            f"loss={train_metrics['loss']:.4f} "
            f"mel={train_metrics['mel_loss']:.4f} "
            f"gate={train_metrics['gate_loss']:.4f} "
            f"attn={train_metrics['attn_loss']:.4f} "
            f"sharp={train_metrics['attention_sharpness']:.4f} "
            f"melE={train_metrics['mel_after_energy']:.4f} "
            f"gaw={train_metrics['guided_attn_weight']:.3f} "
            f"lr={current_lr:.2e}"
        )

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch_idx)
        writer.add_scalar("train/lr", current_lr, epoch_idx)

        if (epoch + 1) % get_image_every_epoch() == 0 and last_outputs is not None:
            log_alignment_image(writer, last_outputs["alignments"].detach(), epoch_idx, tag="attention/transformer_alignment")

        if (epoch + 1) % get_sample_every_epoch() == 0:
            save_validation_sample(model, val_dataset, device, sample_dir, epoch_idx)

        if (epoch + 1) % get_checkpoint_every_epoch() == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_transformer_tacotron2_epoch_{epoch_idx:04d}.pth.tar"
            save_checkpoint(model, optimizer, scaler, epoch_idx, global_step, ckpt_path)
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=get_max_checkpoints_to_keep())
            print(f"Saved checkpoint: {ckpt_path}")

        if (epoch + 1) % get_val_every_epoch() == 0:
            val_guided_attn_weight = guided_attn_scheduler.get_weight(epoch)
            val_metrics = validate(model, criterion, val_loader, device, amp_device_type, val_guided_attn_weight)

            print(
                f"[VAL epoch={epoch_idx}] "
                f"loss={val_metrics['loss']:.4f} "
                f"mel={val_metrics['mel_loss']:.4f} "
                f"gate={val_metrics['gate_loss']:.4f} "
                f"attn={val_metrics['attn_loss']:.4f} "
                f"sharp={val_metrics['attention_sharpness']:.4f} "
                f"melE={val_metrics['mel_after_energy']:.4f} "
                f"gaw={val_metrics['guided_attn_weight']:.3f} "
                f"acc={val_metrics['accuracy']:.4f} "
                f"ar_len={val_metrics['ar_length_ratio']:.3f} "
                f"ar_cov={val_metrics['ar_text_coverage']:.3f} "
                f"ar_back={val_metrics['ar_backward_jump_rate']:.3f}"
            )

            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch_idx)

    writer.close()


if __name__ == "__main__":
    main()
