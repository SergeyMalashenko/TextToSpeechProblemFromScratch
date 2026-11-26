#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import copy
import hashlib
import json
from dataclasses import asdict
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

import hyperparams_transformer as hp
from tts_dataset import (
    get_tacotron_dataset, collate_fn_tacotron,
    compute_dataset_lengths, LengthBucketBatchSampler
)


from tts_transformer_model import TransformerTacotron2
from tts_transformer_config import build_transformer_tacotron_config, check_checkpoint_config
from tts_tacotron_losses import Tacotron2Loss, sequence_mask
from tts_seed import make_torch_generator, seed_worker, set_seed
from tts_transformer_sampling import (
    COMPONENTS, TeacherForcingScheduler, build_optimizer_groups, optimizer_group_layout,
    set_frozen_eval, component_gradient_norms, capture_rng_state, restore_rng_state,
)

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
        group["lr"] = float(lr) * float(group.get("lr_multiplier", 1.0))


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
def compute_gate_metrics(gate_logits: torch.Tensor, gate_target: torch.Tensor, output_lengths: torch.Tensor, threshold: Optional[float] = None) -> Dict[str, float]:
    if threshold is None:
        threshold = float(hp_get("gate_threshold", 0.5))
    probs = torch.sigmoid(gate_logits)
    valid = sequence_mask(output_lengths, max_len=gate_logits.size(1))
    pred = probs > threshold
    target = gate_target > 0.5
    tp = (pred & target & valid).sum().item()
    fp = (pred & (~target) & valid).sum().item()
    fn = ((~pred) & target & valid).sum().item()
    tn = ((~pred) & (~target) & valid).sum().item()
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


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, epoch: int, global_step: int, path: str | Path, training_state: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "model_config": asdict(build_transformer_tacotron_config()),
        "optimizer_groups": optimizer_group_layout(optimizer),
        "training_state": training_state or {},
        "rng_state": capture_rng_state(),
    }, path)


def cleanup_old_checkpoints(checkpoint_dir: str | Path, keep_last_n: int) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if keep_last_n <= 0 or not checkpoint_dir.exists():
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


def maybe_resume_from_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, resume_path: Optional[str | Path], device: torch.device,
                                 sampling_scheduler=None, weights_only: bool = False,
                                 freeze_config: Optional[dict] = None) -> tuple[int, int]:
    if not resume_path:
        return 0, 0
    resume_path = Path(resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    check_checkpoint_config(checkpoint.get("model_config"))
    model_state = checkpoint.get("model", checkpoint)
    consume_prefix_in_state_dict_if_present(model_state, "module.")
    unwrap_model(model).load_state_dict(model_state)
    if weights_only:
        print(f"Loaded model weights for a new training stage: {resume_path}")
        return 0, 0
    if "optimizer" not in checkpoint:
        raise ValueError("Checkpoint has no optimizer state; use --weights-only")
    old_groups = checkpoint["optimizer"]["param_groups"]
    same_sizes = len(old_groups) == len(optimizer.param_groups) and all(
        len(old["params"]) == len(new["params"])
        for old, new in zip(old_groups, optimizer.param_groups)
    )
    if not same_sizes or ("optimizer_groups" in checkpoint and
                          checkpoint["optimizer_groups"] != optimizer_group_layout(optimizer)):
        raise ValueError("Optimizer groups differ from checkpoint; use resume_weights_only=True or --weights-only")
    state = checkpoint.get("training_state", {})
    if freeze_config is not None and "freeze" in state and state["freeze"] != freeze_config:
        raise ValueError("Freeze configuration differs from checkpoint; use --weights-only")
    if sampling_scheduler is not None:
        saved_scheduler = state.get("sampling_scheduler")
        if saved_scheduler is not None:
            sampling_scheduler.load_state_dict(saved_scheduler)
        elif sampling_scheduler.config.get("enabled", False):
            raise ValueError("Checkpoint has no TF scheduler state; use --weights-only for a new stage")
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])

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
    was_training = model_w.training
    model_w.eval()
    try:
        outputs = diagnostic_inference(model_w, text, text_lengths, device)
    finally:
        model_w.train(was_training)

    torch.save(outputs["mel_after"][0].detach().cpu(), save_dir / f"transformer_sample_mel_epoch_{epoch:04d}.pt")
    torch.save(outputs["alignments"][0].detach().cpu(), save_dir / f"transformer_sample_align_epoch_{epoch:04d}.pt")


def train_one_step(model: nn.Module, criterion: Tacotron2Loss, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, batch: Dict[str, torch.Tensor], device: torch.device, amp_device_type: str, guided_attn_weight: float,
                   teacher_forcing_ratio: float = 1.0, freeze_config: Optional[dict] = None) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    batch = move_batch_to_device(batch, device)
    set_guided_attn_weight(criterion, guided_attn_weight)
    set_frozen_eval(unwrap_model(model), freeze_config or {})

    with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
        outputs = model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            mel_input=batch["mel_input"],
            output_lengths=batch["output_lengths"],
            teacher_forcing_ratio=teacher_forcing_ratio,
            return_sampling_stats=True,
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
    gradient_norms = component_gradient_norms(unwrap_model(model))
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
    valid_frames = int(outputs["sampling_valid_frames"].sum().item())
    teacher_frames = int(outputs["sampling_teacher_frames"].sum().item())
    effective_ratio = teacher_frames / valid_frames if valid_frames else 1.0
    metrics.update({
        "teacher_forcing_ratio": float(teacher_forcing_ratio),
        "effective_teacher_forcing_ratio": effective_ratio,
        "replaced_frames_ratio": 1.0 - effective_ratio,
        "_sampling_teacher_frames": float(teacher_frames),
        "_sampling_valid_frames": float(valid_frames),
        "postnet_loss": metrics["mel_after_loss"],
        "guided_attention_loss": metrics["attn_loss"],
        **gradient_norms,
    })
    for name in COMPONENTS:
        metrics[f"lr_{name}"] = 0.0  # frozen components have no optimizer group
    for group in optimizer.param_groups:
        if "name" in group:
            metrics[f"lr_{group['name']}"] = float(group["lr"])
    metrics.update(compute_gate_metrics(outputs["gate"].detach(), batch["gate_target"], batch["output_lengths"]))
    metrics.update(compute_attention_metrics(outputs["alignments"].detach(), batch["text_lengths"], batch["output_lengths"]))
    metrics.update(compute_mel_metrics(outputs["mel_before"].detach(), outputs["mel_after"].detach(), batch["mel_target"], batch["output_lengths"]))
    log_outputs = {
        "alignments": outputs["alignments"].detach().cpu(),
    }
    return log_outputs, metrics


@torch.no_grad()
def compute_attention_path_metrics(alignments: torch.Tensor) -> Dict[str, float]:
    """Describe the argmax attention path, without claiming to measure pronunciation."""
    positions = alignments.argmax(dim=-1)
    steps = positions[1:] - positions[:-1]
    return {
        "ar_text_coverage": float(positions.unique().numel() / alignments.size(1)),
        "ar_final_position_ratio": float((positions[-1].item() + 1) / alignments.size(1)),
        "ar_backward_jump_rate": float((steps < 0).float().mean().item()) if steps.numel() else 0.0,
        "ar_forward_skip_rate": float((steps > 1).float().mean().item()) if steps.numel() else 0.0,
    }


@torch.no_grad()
def diagnostic_inference(model, text, text_lengths, device):
    # Prenet deliberately keeps dropout in eval. A local seed makes comparisons
    # repeatable without consuming the training RNG state.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(get_seed())
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(get_seed())
        return model.inference(text=text, text_lengths=text_lengths)


@torch.no_grad()
def compute_autoregressive_metrics(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    model_w = unwrap_model(model)
    was_training = model_w.training
    model_w.eval()

    text = batch["text"][:1].to(device)
    text_lengths = batch["text_lengths"][:1].to(device)
    target_frames = int(batch["output_lengths"][0].item())

    try:
        outputs = diagnostic_inference(model_w, text, text_lengths, device)
        alignments = outputs["alignments"][0]
        gate = outputs["gate"][0]
        generated_frames = int(alignments.size(0))
        text_length = int(text_lengths[0].item())

        stopped_by_gate = bool(gate.numel() and torch.sigmoid(gate[-1]).item() > float(hp.gate_threshold))

        return {
            "ar_generated_frames": float(generated_frames),
            "ar_target_frames": float(target_frames),
            "ar_length_ratio": float(generated_frames / max(1, target_frames)),
            "ar_gate_last_prob": float(torch.sigmoid(gate[-1]).item()) if gate.numel() > 0 else 0.0,
            **compute_attention_path_metrics(alignments[:, :text_length]),
            "ar_hit_max_steps": float(generated_frames >= int(hp.max_decoder_steps) and not stopped_by_gate),
        }
    finally:
        if was_training:
            model_w.train()


@torch.no_grad()
def validate(model: nn.Module, criterion: Tacotron2Loss, dataloader: DataLoader, device: torch.device, amp_device_type: str, guided_attn_weight: float,
             teacher_forcing_ratio: float = 1.0, sampling_validation: bool = False) -> Dict[str, float]:
    model.eval()
    set_guided_attn_weight(criterion, guided_attn_weight)
    totals = defaultdict(float)
    n_batches = 0
    total_frames = 0
    mel_loss_sums = defaultdict(float)
    autoregressive_batches = []
    sampling_loss_totals = defaultdict(float)

    for batch_index, batch in enumerate(tqdm(dataloader, desc="validation", leave=False)):
        batch = move_batch_to_device(batch, device)
        for index in range(batch["text"].size(0)):
            if len(autoregressive_batches) >= int(hp_get("autoregressive_val_samples", 4)):
                break
            autoregressive_batches.append({
                key: batch[key][index:index + 1].detach().cpu()
                for key in ("text", "text_lengths", "output_lengths")
            })
        with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
            outputs = model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                mel_input=batch["mel_input"],
                output_lengths=batch["output_lengths"],
            )
            losses = criterion(outputs=outputs, batch=batch)

        if sampling_validation:
            if teacher_forcing_ratio == 1.0:
                sampled_losses = losses  # identical one-pass TF, including prenet dropout
            else:
                devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(get_seed() + batch_index)
                    if devices:
                        torch.cuda.manual_seed_all(get_seed() + batch_index)
                    with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
                        sampled_outputs = model(
                            text=batch["text"], text_lengths=batch["text_lengths"],
                            mel_input=batch["mel_input"], output_lengths=batch["output_lengths"],
                            teacher_forcing_ratio=teacher_forcing_ratio, apply_scheduled_sampling=True,
                        )
                        sampled_losses = criterion(outputs=sampled_outputs, batch=batch)
                check_finite_tensor("val/scheduled_sampling_loss", sampled_losses["loss"])
            frames = int(batch["output_lengths"].sum().item())
            for key in ("mel_loss", "gate_loss", "attn_loss"):
                weight = frames if key == "mel_loss" else 1
                sampling_loss_totals[key] += float(sampled_losses[key].item()) * weight

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
        frames = int(batch["output_lengths"].sum().item())
        for key in ("mel_before_loss", "mel_after_loss", "mel_loss"):
            mel_loss_sums[key] += metrics[key] * frames
        total_frames += frames
        n_batches += 1

    if n_batches == 0:
        model.train()
        return {"loss": 0.0}
    metrics = {key: value / n_batches for key, value in totals.items()}
    for key, value in mel_loss_sums.items():
        metrics[key] = value / max(1, total_frames)
    metrics["loss"] = metrics["mel_loss"] + metrics["gate_loss"] + guided_attn_weight * metrics["attn_loss"]
    if autoregressive_batches:
        metrics.update(average_metric_dict([
            compute_autoregressive_metrics(model, batch, device) for batch in autoregressive_batches
        ]))
    metrics["teacher_forced_loss"] = metrics["loss"]
    if sampling_validation:
        metrics["scheduled_sampling_loss"] = (sampling_loss_totals["mel_loss"] / max(1, total_frames)
            + sampling_loss_totals["gate_loss"] / n_batches
            + guided_attn_weight * sampling_loss_totals["attn_loss"] / n_batches)
    if "ar_text_coverage" in metrics:
        metrics["attention_coverage"] = metrics["ar_text_coverage"]
        metrics["attention_backward"] = metrics["ar_backward_jump_rate"]
        metrics["attention_ar_len"] = metrics["ar_length_ratio"]
    model.train()
    return metrics


def main() -> None:
    build_transformer_tacotron_config().validate()
    seed = get_seed()
    set_seed(seed)
    device = get_device()
    amp_device_type = get_amp_device_type()

    print(f"Using device: {device}")
    print(f"AMP device type: {amp_device_type}")

    sampling_config = copy.deepcopy(hp_get("scheduled_sampling", {"enabled": False}))
    sampling_scheduler = TeacherForcingScheduler(sampling_config)
    freeze_config = copy.deepcopy(hp_get("freeze", {}))
    weights_only = bool(hp_get("resume_weights_only", False))
    resume_path = hp_get("resume_transformer_tacotron_path", None)
    saved_checkpoint = torch.load(resume_path, map_location="cpu") if resume_path else {}
    previous_state = saved_checkpoint.get("training_state", {}) if not weights_only else {}

    full_dataset = get_tacotron_dataset()
    dataset_fingerprint = None
    if hasattr(full_dataset, "df"):
        dataset_fingerprint = hashlib.sha256(full_dataset.df.to_csv(index=False).encode()).hexdigest()
    if len(full_dataset) < 2 or not 0 < get_val_ratio() < 1:
        raise ValueError("Training needs at least two examples and 0 < val_ratio < 1")
    val_size = max(1, int(len(full_dataset) * get_val_ratio()))
    train_size = len(full_dataset) - val_size

    if "split" in previous_state:
        split = previous_state["split"]
        indices = split["train"] + split["val"]
        if sorted(indices) != list(range(len(full_dataset))) or not split["train"] or not split["val"]:
            raise ValueError("Saved split does not match the dataset; use --weights-only for a new stage")
        if previous_state.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Dataset metadata differs from checkpoint")
        train_dataset = Subset(full_dataset, split["train"])
        val_dataset = Subset(full_dataset, split["val"])
    else:
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=make_torch_generator(seed),
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
            seed=seed,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            generator=make_torch_generator(seed),
            num_workers=hp.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn_tacotron,
            worker_init_fn=seed_worker,
        )
    else:
        print("[DataLoader] Using plain random sampler")
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(hp.batch_size),
            shuffle=True,
            collate_fn=collate_fn_tacotron,
            drop_last=False,
            num_workers=get_num_workers(),
            pin_memory=bool(hp_get("pin_memory", torch.cuda.is_available())),
            worker_init_fn=seed_worker,
            generator=make_torch_generator(seed),
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(hp.batch_size),
        shuffle=False,
        collate_fn=collate_fn_tacotron,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=bool(hp_get("pin_memory", torch.cuda.is_available())),
        worker_init_fn=seed_worker,
    )

    model = TransformerTacotron2().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    parameter_groups = build_optimizer_groups(
        unwrap_model(model), float(hp.lr), freeze_config, hp_get("lr_multipliers", {}))
    optimizer = torch.optim.AdamW(
        parameter_groups,
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

    log_dir = create_experiment_log_dir(get_log_dir(), prefix="transformer")
    checkpoint_dir = get_checkpoint_dir() / log_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    sample_dir = get_samples_dir() / log_dir.name
    sample_dir.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(hp).items()
              if not key.startswith("_") and isinstance(value, (str, int, float, bool, list, tuple, dict, type(None)))}
    (log_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (log_dir / "split.json").write_text(json.dumps({"train": train_dataset.indices, "val": val_dataset.indices}), encoding="utf-8")

    start_epoch, global_step = maybe_resume_from_checkpoint(
        model, optimizer, scaler, resume_path, device, sampling_scheduler=sampling_scheduler,
        weights_only=weights_only, freeze_config=freeze_config,
    )

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

    best_mel_loss = float(previous_state.get("best_mel_loss", float("inf")))

    def training_state():
        return {"sampling_scheduler": sampling_scheduler.state_dict(), "freeze": freeze_config,
                "split": {"train": train_dataset.indices, "val": val_dataset.indices},
                "dataset_fingerprint": dataset_fingerprint, "best_mel_loss": best_mel_loss}

    for epoch in range(start_epoch, int(hp.epochs)):
        epoch_idx = epoch + 1
        if use_bucket_sampler:
            batch_sampler.set_epoch(epoch)
        train_loader.generator.manual_seed(seed + epoch)
        model.train()
        set_frozen_eval(unwrap_model(model), freeze_config)

        current_lr = get_lr_by_epoch(epoch)
        set_optimizer_lr(optimizer, current_lr)

        current_guided_attn_weight = guided_attn_scheduler.get_weight(epoch)

        pbar = tqdm(train_loader, desc=f"epoch {epoch_idx}/{int(hp.epochs)}")
        train_epoch_metrics = []
        last_outputs = None
        current_teacher_forcing_ratio = sampling_scheduler.get_ratio(epoch, global_step)

        for batch in pbar:
            current_teacher_forcing_ratio = sampling_scheduler.get_ratio(epoch, global_step)
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
                teacher_forcing_ratio=current_teacher_forcing_ratio,
                freeze_config=freeze_config,
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
                tf=f"{current_teacher_forcing_ratio:.3f}",
            )

        train_metrics = average_metric_dict(train_epoch_metrics)
        teacher_frames = sum(metrics["_sampling_teacher_frames"] for metrics in train_epoch_metrics)
        valid_frames = sum(metrics["_sampling_valid_frames"] for metrics in train_epoch_metrics)
        train_metrics.pop("_sampling_teacher_frames")
        train_metrics.pop("_sampling_valid_frames")
        train_metrics["effective_teacher_forcing_ratio"] = teacher_frames / valid_frames if valid_frames else 1.0
        train_metrics["replaced_frames_ratio"] = 1.0 - train_metrics["effective_teacher_forcing_ratio"]
        print(
            f"[TRAIN epoch={epoch_idx}] "
            f"loss={train_metrics['loss']:.4f} "
            f"mel={train_metrics['mel_loss']:.4f} "
            f"gate={train_metrics['gate_loss']:.4f} "
            f"attn={train_metrics['attn_loss']:.4f} "
            f"sharp={train_metrics['attention_sharpness']:.4f} "
            f"melE={train_metrics['mel_after_energy']:.4f} "
            f"gaw={train_metrics['guided_attn_weight']:.3f} "
            f"lr={current_lr:.2e} "
            f"tf={train_metrics['teacher_forcing_ratio']:.3f} "
            f"effective_tf={train_metrics['effective_teacher_forcing_ratio']:.3f}"
        )

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch_idx)
        writer.add_scalar("train/lr", current_lr, epoch_idx)

        if (epoch + 1) % get_image_every_epoch() == 0 and last_outputs is not None:
            log_alignment_image(writer, last_outputs["alignments"].detach(), epoch_idx, tag="attention/transformer_alignment")

        if (epoch + 1) % get_sample_every_epoch() == 0:
            save_validation_sample(model, val_dataset, device, sample_dir, epoch_idx)

        if (epoch + 1) % get_val_every_epoch() == 0 or epoch_idx == int(hp.epochs):
            val_guided_attn_weight = guided_attn_scheduler.get_weight(epoch)
            val_metrics = validate(
                model, criterion, val_loader, device, amp_device_type, val_guided_attn_weight,
                teacher_forcing_ratio=current_teacher_forcing_ratio,
                sampling_validation=bool(sampling_config.get("enabled", False) and sampling_config.get("validation_enabled", False)),
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
                f"acc={val_metrics['accuracy']:.4f} "
                f"ar_len={val_metrics['ar_length_ratio']:.3f} "
                f"ar_cov={val_metrics['ar_text_coverage']:.3f} "
                f"ar_back={val_metrics['ar_backward_jump_rate']:.3f}"
            )

            # Use the same mel objective even while the guided-attention weight changes.
            if val_metrics["mel_after_loss"] < best_mel_loss:
                best_mel_loss = val_metrics["mel_after_loss"]
                save_checkpoint(model, optimizer, scaler, epoch_idx, global_step,
                                checkpoint_dir / "transformer_best.pth.tar", training_state())
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch_idx)

        save_checkpoint(model, optimizer, scaler, epoch_idx, global_step,
                        checkpoint_dir / "transformer_last.pth.tar", training_state())
        if epoch_idx % get_checkpoint_every_epoch() == 0 or epoch_idx == int(hp.epochs):
            ckpt_path = checkpoint_dir / f"checkpoint_transformer_tacotron2_epoch_{epoch_idx:04d}.pth.tar"
            save_checkpoint(model, optimizer, scaler, epoch_idx, global_step, ckpt_path, training_state())
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=get_max_checkpoints_to_keep())
            print(f"Saved checkpoint: {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    main()
