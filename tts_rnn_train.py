#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams_rnn as hp
from tts_dataset import get_tacotron_dataset, collate_fn_tacotron
from tts_rnn_model import Tacotron2
from tts_tacotron_losses import Tacotron2Loss, sequence_mask

from tts_seed import set_seed, seed_worker


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
    return float(hp_get("guided_attn_weight", 2.0))


def get_guided_attn_sigma() -> float:
    return float(hp_get("guided_attn_sigma", 0.4))


def get_gate_pos_weight() -> float:
    return float(hp_get("gate_pos_weight", 5.0))


def get_log_alignment_every() -> int:
    return int(hp_get("image_step", 500))


def get_checkpoint_every() -> int:
    return int(hp_get("save_step", 2000))


def get_sample_every() -> int:
    return int(hp_get("sample_step", get_checkpoint_every()))


def get_max_checkpoints_to_keep() -> int:
    return int(hp_get("max_checkpoints_to_keep", 5))


def get_seed() -> int:
    return int(hp_get("seed", 42))


def get_log_dir() -> Path:
    """
    Base directory for TensorBoard logs. Every launch creates its own
    timestamped subdirectory inside this folder.
    """
    return Path(hp_get("rnn_log_dir", "./logs/rnn_tacotron"))


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
        ./logs/rnn_tacotron/rnn_20260616_173245/
        ./logs/rnn_tacotron/rnn_20260616_173245_my_experiment/
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


# =============================================================================
# Utility functions
# =============================================================================

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def get_lr_warmup_epochs() -> int:
    """
    Epoch-based warmup length.

    Historical equivalence for this project:
        4000 optimizer steps ~= 20 epochs

    Therefore, the default value is 20 epochs.
    """
    return int(hp_get("lr_warmup_epochs", 20))


def get_lr_min() -> float:
    return float(hp_get("lr_min", 1e-5))


def get_lr_by_epoch(
    epoch: int,
    base_lr: float,
    warmup_epochs: int = 20,
    min_lr: float = 1e-5,
) -> float:
    """
    Epoch-based analogue of the Noam/inverse-square-root schedule.

    Args:
        epoch: 0-based epoch index.
        base_lr: peak/base learning rate from hyperparams.
        warmup_epochs: number of warmup epochs.
        min_lr: lower bound for LR.

    Notes:
        With warmup_epochs=20 this matches the old convention:
        4000 optimizer steps ~= 20 epochs.

        Formula:
            lr = base_lr * min(e / W, sqrt(W / e))

        where e = epoch + 1 and W = warmup_epochs.
    """
    e = max(int(epoch) + 1, 1)
    W = max(int(warmup_epochs), 1)

    lr = float(base_lr) * min(
        e / float(W),
        (float(W) / float(e)) ** 0.5,
    )
    return max(float(lr), float(min_lr))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def average_metric_dict(metric_list: list[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    totals = defaultdict(float)
    for metrics in metric_list:
        for key, value in metrics.items():
            totals[key] += float(value)
    n = float(len(metric_list))
    return {key: value / n for key, value in totals.items()}


def get_val_every_epoch() -> int:
    return int(hp_get("val_every_epoch", 1))


def get_image_every_epoch() -> int:
    return int(hp_get("image_every_epoch", 1))


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
    eps: float = 1.0e-8,
) -> Dict[str, float]:
    """
    Computes diagnostic metrics for Tacotron-style attention.

    alignments: (B, T_mel, T_text)
    text_lengths: (B,)
    mel_lengths: (B,)

    Metrics:
        attention_entropy: lower means sharper attention over text positions.
        attention_peak_mean: mean max attention value per decoder step.
        attention_diag_mass: attention mass near normalized diagonal.
        attention_sharpness: peak / entropy, higher means sharper alignment.
        attention_monotonicity: fraction of non-decreasing argmax positions.
        attention_coverage: fraction of text tokens selected by argmax at least once.
    """
    if alignments.ndim != 3 or alignments.size(0) == 0:
        return {
            "attention_entropy": 0.0,
            "attention_peak_mean": 0.0,
            "attention_diag_mass": 0.0,
            "attention_sharpness": 0.0,
            "attention_monotonicity": 0.0,
            "attention_coverage": 0.0,
        }

    entropy_values = []
    peak_values = []
    diag_mass_values = []
    monotonicity_values = []
    coverage_values = []

    B = alignments.size(0)

    for b in range(B):
        mel_len = int(mel_lengths[b].item())
        text_len = int(text_lengths[b].item())

        if mel_len <= 0 or text_len <= 0:
            continue

        A = alignments[b, :mel_len, :text_len].detach().float()
        row_sum = A.sum(dim=-1, keepdim=True).clamp_min(eps)
        A = A / row_sum

        entropy = -(A * (A + eps).log()).sum(dim=-1)
        peak = A.max(dim=-1).values

        # Diagonal band in normalized coordinates. A width of one text token
        # plus one mel frame is tolerant enough for short and long samples.
        mel_pos = torch.arange(mel_len, device=A.device, dtype=A.dtype) / max(float(mel_len - 1), 1.0)
        text_pos = torch.arange(text_len, device=A.device, dtype=A.dtype) / max(float(text_len - 1), 1.0)
        dist = (mel_pos.unsqueeze(1) - text_pos.unsqueeze(0)).abs()
        band_width = (1.0 / max(float(mel_len), 1.0)) + (1.0 / max(float(text_len), 1.0))
        diag_mask = dist <= band_width
        diag_mass = (A * diag_mask.to(A.dtype)).sum(dim=-1)

        argmax_pos = A.argmax(dim=-1)
        if mel_len > 1:
            monotonicity = (argmax_pos[1:] >= argmax_pos[:-1]).float().mean()
        else:
            monotonicity = A.new_tensor(1.0)

        coverage = torch.unique(argmax_pos).numel() / max(1, text_len)

        entropy_values.append(entropy.mean())
        peak_values.append(peak.mean())
        diag_mass_values.append(diag_mass.mean())
        monotonicity_values.append(monotonicity)
        coverage_values.append(A.new_tensor(float(coverage)))

    if not entropy_values:
        return {
            "attention_entropy": 0.0,
            "attention_peak_mean": 0.0,
            "attention_diag_mass": 0.0,
            "attention_sharpness": 0.0,
            "attention_monotonicity": 0.0,
            "attention_coverage": 0.0,
        }

    entropy_mean = torch.stack(entropy_values).mean()
    peak_mean = torch.stack(peak_values).mean()
    diag_mass_mean = torch.stack(diag_mass_values).mean()
    sharpness = peak_mean / entropy_mean.clamp_min(eps)

    return {
        "attention_entropy": float(entropy_mean.item()),
        "attention_peak_mean": float(peak_mean.item()),
        "attention_diag_mass": float(diag_mass_mean.item()),
        "attention_sharpness": float(sharpness.item()),
        "attention_monotonicity": float(torch.stack(monotonicity_values).mean().item()),
        "attention_coverage": float(torch.stack(coverage_values).mean().item()),
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


# =============================================================================
# Checkpoint helpers
# =============================================================================

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

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_tacotron2_*.pth.tar"), key=lambda p: p.stat().st_mtime)
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
    global_step: int,
    tag: str = "attention/alignment",
) -> None:
    """
    alignments: (B, T_mel, T_text)
    """
    if alignments.ndim != 3 or alignments.size(0) == 0:
        return

    A = alignments[0].detach().float().cpu().unsqueeze(0)  # (1, T_mel, T_text)
    writer.add_image(tag, A, global_step)


@torch.no_grad()
def save_validation_sample(
    model: nn.Module,
    dataset,
    device: torch.device,
    save_dir: str | Path,
    global_step: int,
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

    torch.save(mel, save_dir / f"sample_mel_step_{global_step}.pt")
    torch.save(align, save_dir / f"sample_align_step_{global_step}.pt")


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
) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    batch = move_batch_to_device(batch, device)

    with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
        outputs = model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            mel_input=batch["mel_input"],
        )
        loss_dict = criterion(outputs=outputs, batch=batch)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss_dict["loss"]).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    gate_metrics = compute_gate_metrics(
        gate_logits=outputs["gate"].detach(),
        gate_target=batch["gate_target"],
        output_lengths=batch["output_lengths"],
        threshold=0.5,
    )
    attention_metrics = compute_attention_metrics(
        alignments=outputs["alignments"].detach(),
        text_lengths=batch["text_lengths"],
        mel_lengths=batch["output_lengths"],
    )
    mel_metrics = compute_mel_metrics(
        mel_before=outputs["mel_before"].detach(),
        mel_after=outputs["mel_after"].detach(),
        mel_target=batch["mel_target"],
        output_lengths=batch["output_lengths"],
    )

    stats = {
        "loss": float(loss_dict["loss"].item()),
        "mel_loss": float(loss_dict["mel_loss"].item()),
        "gate_loss": float(loss_dict["gate_loss"].item()),
        "attn_loss": float(loss_dict["attn_loss"].item()),
        "mel_before_loss": float(loss_dict["mel_before_loss"].item()),
        "mel_after_loss": float(loss_dict["mel_after_loss"].item()),
        "guided_attn_weight": float(get_guided_attn_weight()),
        **gate_metrics,
        **attention_metrics,
        **mel_metrics,
    }
    return outputs, stats


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: Tacotron2Loss,
    dataloader: DataLoader,
    device: torch.device,
    amp_device_type: str,
) -> Dict[str, float]:
    model.eval()

    totals = defaultdict(float)
    n_batches = 0

    for batch in tqdm(dataloader, desc="validation", leave=False):
        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
            outputs = model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                mel_input=batch["mel_input"],
            )
            loss_dict = criterion(outputs=outputs, batch=batch)

        metrics = {
            "loss": float(loss_dict["loss"].item()),
            "mel_loss": float(loss_dict["mel_loss"].item()),
            "gate_loss": float(loss_dict["gate_loss"].item()),
            "attn_loss": float(loss_dict["attn_loss"].item()),
            "mel_before_loss": float(loss_dict["mel_before_loss"].item()),
            "mel_after_loss": float(loss_dict["mel_after_loss"].item()),
            "guided_attn_weight": float(get_guided_attn_weight()),
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
    seed = get_seed()
    set_seed(seed)

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
        generator=torch.Generator().manual_seed(seed),
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=collate_fn_tacotron,
        drop_last=True,
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=collate_fn_tacotron,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )

    model = Tacotron2().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    scaler = torch.amp.GradScaler(amp_device_type)

    criterion = Tacotron2Loss(
        gate_pos_weight=get_gate_pos_weight(),
        guided_attn_weight=get_guided_attn_weight(),
        guided_attn_sigma=get_guided_attn_sigma(),
    )

    log_dir = create_experiment_log_dir(get_log_dir(), prefix="rnn")
    writer = SummaryWriter(log_dir=str(log_dir))

    resume_path = hp_get("resume_checkpoint", None)
    start_epoch, global_step = maybe_resume_from_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resume_path=resume_path,
        device=device,
    )

    checkpoint_dir = Path(hp.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = Path(hp.sample_path)
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size  : {len(val_dataset)}")
    print(f"Log dir           : {log_dir}")
    print(f"Checkpoint dir    : {checkpoint_dir}")
    print(f"Sample dir        : {sample_dir}")
    print(f"Optimizer         : Adam")
    print(f"Base LR           : {float(hp.lr):.2e}")
    print(f"LR schedule       : epoch_noam_inverse_sqrt")
    print(f"LR warmup epochs  : {get_lr_warmup_epochs()}")
    print(f"LR min            : {get_lr_min():.2e}")
    print(f"Seed              : {seed}")

    writer.add_text("config/log_dir", str(log_dir), 0)
    writer.add_text("config/seed", str(seed), 0)
    writer.add_text("config/optimizer", "Adam", 0)
    writer.add_text("config/base_lr", f"{float(hp.lr):.6e}", 0)
    writer.add_text("config/lr_schedule", "epoch_noam_inverse_sqrt", 0)
    writer.add_text("config/lr_warmup_epochs", str(get_lr_warmup_epochs()), 0)
    writer.add_text("config/lr_min", f"{get_lr_min():.6e}", 0)

    for epoch in range(start_epoch, hp.epochs):
        epoch_idx = epoch + 1

        current_lr = get_lr_by_epoch(
            epoch=epoch,
            base_lr=float(hp.lr),
            warmup_epochs=get_lr_warmup_epochs(),
            min_lr=get_lr_min(),
        )
        set_optimizer_lr(optimizer, current_lr)

        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch_idx}/{int(hp.epochs)}")
        train_epoch_metrics: list[Dict[str, float]] = []
        last_outputs = None

        for batch in pbar:
            global_step += 1
            current_lr = float(optimizer.param_groups[0]["lr"])

            outputs, metrics = train_one_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                device=device,
                amp_device_type=amp_device_type,
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

            if global_step % get_checkpoint_every() == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_tacotron2_{global_step}.pth.tar"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    path=ckpt_path,
                )
                cleanup_old_checkpoints(checkpoint_dir, keep_last_n=get_max_checkpoints_to_keep())
                print(f"Saved checkpoint: {ckpt_path}")

            if global_step % get_sample_every() == 0:
                save_validation_sample(
                    model=model,
                    dataset=full_dataset,
                    device=device,
                    save_dir=sample_dir,
                    global_step=global_step,
                )

        train_metrics = average_metric_dict(train_epoch_metrics)
        if train_metrics:
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
            log_alignment_image(
                writer=writer,
                alignments=last_outputs["alignments"].detach(),
                global_step=epoch_idx,
                tag="attention/rnn_alignment",
            )

        if (epoch + 1) % get_val_every_epoch() == 0:
            val_metrics = validate(
                model=model,
                criterion=criterion,
                dataloader=val_loader,
                device=device,
                amp_device_type=amp_device_type,
            )

            print(
                f"[VAL epoch={epoch_idx}] "
                f"loss={val_metrics['loss']:.4f} "
                f"mel={val_metrics['mel_loss']:.4f} "
                f"gate={val_metrics['gate_loss']:.4f} "
                f"attn={val_metrics['attn_loss']:.4f} "
                f"sharp={val_metrics['attention_sharpness']:.4f} "
                f"melE={val_metrics['mel_after_energy']:.4f} "
                f"gaw={val_metrics['guided_attn_weight']:.3f} "
                f"acc={val_metrics['accuracy']:.4f}"
            )

            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch_idx)

    writer.close()


if __name__ == "__main__":
    main()
