#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams as hp
from tts_dataset import get_tacotron_dataset, collate_fn_tacotron
from tts_rnn_model import Tacotron2


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


# =============================================================================
# Utility functions
# =============================================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Returns:
        mask: (B, T) with True for valid positions
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) < lengths.unsqueeze(1)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


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


# =============================================================================
# Losses
# =============================================================================

def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    pred, target: (B, T, C)
    lengths: (B,)
    """
    mask = sequence_mask(lengths, max_len=pred.size(1)).unsqueeze(-1).to(pred.dtype)
    loss = torch.abs(pred - target) * mask
    denom = mask.sum() * pred.size(-1)
    return loss.sum() / denom.clamp_min(1.0)


def gate_bce_loss(
    gate_logits: torch.Tensor,
    gate_target: torch.Tensor,
    pos_weight_value: float,
) -> torch.Tensor:
    pos_weight = torch.tensor([pos_weight_value], device=gate_logits.device, dtype=gate_logits.dtype)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion(gate_logits, gate_target)


def guided_attention_map(
    mel_len: int,
    text_len: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns:
        (T_mel, T_text)
    """
    t = torch.arange(mel_len, device=device, dtype=dtype) / max(float(mel_len), 1.0)
    n = torch.arange(text_len, device=device, dtype=dtype) / max(float(text_len), 1.0)

    tt = t.unsqueeze(1)  # (T_mel, 1)
    nn_ = n.unsqueeze(0)  # (1, T_text)

    w = 1.0 - torch.exp(-((tt - nn_) ** 2) / (2.0 * sigma * sigma))
    return w


def guided_attention_loss(
    alignments: torch.Tensor,
    text_lengths: torch.Tensor,
    mel_lengths: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    alignments: (B, T_mel, T_text)
    text_lengths: (B,)
    mel_lengths: (B,)

    We only compute over valid alignment area.
    """
    B, T_mel_max, T_text_max = alignments.shape
    total_loss = alignments.new_tensor(0.0)
    total_weight = alignments.new_tensor(0.0)

    for b in range(B):
        t_len = int(mel_lengths[b].item())
        n_len = int(text_lengths[b].item())

        if t_len <= 0 or n_len <= 0:
            continue

        A = alignments[b, :t_len, :n_len]
        W = guided_attention_map(
            mel_len=t_len,
            text_len=n_len,
            sigma=sigma,
            device=alignments.device,
            dtype=alignments.dtype,
        )
        total_loss = total_loss + (A * W).sum()
        total_weight = total_weight + torch.tensor(float(t_len * n_len), device=alignments.device, dtype=alignments.dtype)

    return total_loss / total_weight.clamp_min(1.0)


class Tacotron2Loss(nn.Module):
    def __init__(
        self,
        gate_pos_weight: float = 5.0,
        guided_attn_weight: float = 1.0,
        guided_attn_sigma: float = 0.4,
    ) -> None:
        super().__init__()
        self.gate_pos_weight = gate_pos_weight
        self.guided_attn_weight = guided_attn_weight
        self.guided_attn_sigma = guided_attn_sigma

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mel_before = outputs["mel_before"]      # (B, T, n_mels)
        mel_after = outputs["mel_after"]        # (B, T, n_mels)
        gate_logits = outputs["gate"]           # (B, T)
        alignments = outputs["alignments"]      # (B, T, T_text)

        mel_target = batch["mel_target"]        # (B, T, n_mels)
        gate_target = batch["gate_target"]      # (B, T)
        text_lengths = batch["text_lengths"]    # (B,)
        output_lengths = batch["output_lengths"]# (B,)

        mel_before_loss = masked_l1_loss(mel_before, mel_target, output_lengths)
        mel_after_loss = masked_l1_loss(mel_after, mel_target, output_lengths)
        mel_loss = mel_before_loss + mel_after_loss

        gate_loss = gate_bce_loss(
            gate_logits=gate_logits,
            gate_target=gate_target,
            pos_weight_value=self.gate_pos_weight,
        )

        attn_loss = guided_attention_loss(
            alignments=alignments,
            text_lengths=text_lengths,
            mel_lengths=output_lengths,
            sigma=self.guided_attn_sigma,
        )

        total_loss = mel_loss + gate_loss + self.guided_attn_weight * attn_loss

        return {
            "loss": total_loss,
            "mel_loss": mel_loss,
            "gate_loss": gate_loss,
            "attn_loss": attn_loss,
            "mel_before_loss": mel_before_loss,
            "mel_after_loss": mel_after_loss,
        }


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

    stats = {
        "loss": float(loss_dict["loss"].item()),
        "mel_loss": float(loss_dict["mel_loss"].item()),
        "gate_loss": float(loss_dict["gate_loss"].item()),
        "attn_loss": float(loss_dict["attn_loss"].item()),
        "mel_before_loss": float(loss_dict["mel_before_loss"].item()),
        "mel_after_loss": float(loss_dict["mel_after_loss"].item()),
        **gate_metrics,
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

    agg = {
        "loss": 0.0,
        "mel_loss": 0.0,
        "gate_loss": 0.0,
        "attn_loss": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "early_stop_rate": 0.0,
        "p_prev_mean": 0.0,
        "p_last_mean": 0.0,
    }

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

        gate_metrics = compute_gate_metrics(
            gate_logits=outputs["gate"],
            gate_target=batch["gate_target"],
            output_lengths=batch["output_lengths"],
            threshold=0.5,
        )

        agg["loss"] += float(loss_dict["loss"].item())
        agg["mel_loss"] += float(loss_dict["mel_loss"].item())
        agg["gate_loss"] += float(loss_dict["gate_loss"].item())
        agg["attn_loss"] += float(loss_dict["attn_loss"].item())

        for k, v in gate_metrics.items():
            agg[k] += float(v)

        n_batches += 1

    model.train()

    if n_batches == 0:
        return {k: 0.0 for k in agg}

    return {k: v / n_batches for k, v in agg.items()}


# =============================================================================
# Main
# =============================================================================

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=collate_fn_tacotron,
        drop_last=True,
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=collate_fn_tacotron,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=torch.cuda.is_available(),
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

    writer = SummaryWriter()

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

    for epoch in range(start_epoch, hp.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            global_step += 1

            if global_step < 400000:
                current_lr = adjust_learning_rate(optimizer, global_step)
            else:
                current_lr = float(optimizer.param_groups[0]["lr"])

            outputs, stats = train_one_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                device=device,
                amp_device_type=amp_device_type,
            )

            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                mel=f"{stats['mel_loss']:.4f}",
                gate=f"{stats['gate_loss']:.4f}",
                attn=f"{stats['attn_loss']:.4f}",
                lr=f"{current_lr:.6f}",
            )

            writer.add_scalars(
                "train/loss",
                {
                    "total": stats["loss"],
                    "mel": stats["mel_loss"],
                    "gate": stats["gate_loss"],
                    "guided_attn": stats["attn_loss"],
                    "mel_before": stats["mel_before_loss"],
                    "mel_after": stats["mel_after_loss"],
                },
                global_step,
            )

            writer.add_scalars(
                "train/stop",
                {
                    "accuracy": stats["accuracy"],
                    "precision": stats["precision"],
                    "recall": stats["recall"],
                    "early_stop_rate": stats["early_stop_rate"],
                    "p_prev_mean": stats["p_prev_mean"],
                    "p_last_mean": stats["p_last_mean"],
                },
                global_step,
            )

            writer.add_scalar("train/lr", current_lr, global_step)

            if global_step % get_log_alignment_every() == 1:
                log_alignment_image(
                    writer=writer,
                    alignments=outputs["alignments"],
                    global_step=global_step,
                    tag="train/alignment",
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

                val_stats = validate(
                    model=model,
                    criterion=criterion,
                    dataloader=val_loader,
                    device=device,
                    amp_device_type=amp_device_type,
                )

                writer.add_scalars(
                    "val/loss",
                    {
                        "total": val_stats["loss"],
                        "mel": val_stats["mel_loss"],
                        "gate": val_stats["gate_loss"],
                        "guided_attn": val_stats["attn_loss"],
                    },
                    global_step,
                )

                writer.add_scalars(
                    "val/stop",
                    {
                        "accuracy": val_stats["accuracy"],
                        "precision": val_stats["precision"],
                        "recall": val_stats["recall"],
                        "early_stop_rate": val_stats["early_stop_rate"],
                        "p_prev_mean": val_stats["p_prev_mean"],
                        "p_last_mean": val_stats["p_last_mean"],
                    },
                    global_step,
                )

                save_validation_sample(
                    model=model,
                    dataset=full_dataset,
                    device=device,
                    save_dir=sample_dir,
                    global_step=global_step,
                )

    writer.close()


if __name__ == "__main__":
    main()
