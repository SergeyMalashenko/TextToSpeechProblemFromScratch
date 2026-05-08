#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import hyperparams_rnn as hp
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


def adjust_learning_rate_by_epoch(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    base_lr: float,
    schedule_type: str = "warmup_invsqrt_by_epoch",
    warmup_epochs: int = 30,
    hold_epochs: int = 0,
    min_lr: float = 1e-5,
    decay_gamma: float = 0.98,
) -> float:
    """
    Epoch-based learning-rate schedule.

    Supported modes:
      - warmup_invsqrt_by_epoch
      - exponential_by_epoch
      - constant

    epoch is 0-based.
    """
    epoch_1based = epoch + 1

    if schedule_type == "constant":
        lr = float(base_lr)

    elif schedule_type == "exponential_by_epoch":
        lr = float(base_lr) * (float(decay_gamma) ** max(0, epoch))

    else:
        if warmup_epochs > 0 and epoch_1based <= warmup_epochs:
            lr = float(base_lr) * float(epoch_1based) / float(warmup_epochs)
        elif epoch_1based <= warmup_epochs + hold_epochs:
            lr = float(base_lr)
        else:
            decay_epoch = epoch_1based - warmup_epochs - hold_epochs
            lr = float(base_lr) / (float(decay_epoch) ** 0.5)

    lr = max(float(min_lr), float(lr))

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

    tt = t.unsqueeze(1)
    nn_ = n.unsqueeze(0)

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
    """
    total_loss = alignments.new_tensor(0.0)
    total_weight = alignments.new_tensor(0.0)

    for b in range(alignments.size(0)):
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
        mel_before = outputs["mel_before"]
        mel_after = outputs["mel_after"]
        gate_logits = outputs["gate"]
        alignments = outputs["alignments"]

        mel_target = batch["mel_target"]
        gate_target = batch["gate_target"]
        text_lengths = batch["text_lengths"]
        output_lengths = batch["output_lengths"]

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

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_tacotron2_epoch_*.pth.tar"),
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
    tag: str = "train_epoch/alignment",
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

    torch.save(mel, save_dir / f"sample_mel_epoch_{epoch_index:06d}.pt")
    torch.save(align, save_dir / f"sample_align_epoch_{epoch_index:06d}.pt")


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
        persistent_workers=True,
        #prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=collate_fn_tacotron,
        drop_last=False,
        #num_workers=max(0, get_num_workers() // 2),
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        #prefetch_factor=2
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

    writer = SummaryWriter(log_dir=str(Path(hp_get("rnn_log_dir", "./logs"))))

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
        epoch_index = epoch + 1

        current_lr = adjust_learning_rate_by_epoch(
            optimizer=optimizer,
            epoch=epoch,
            base_lr=hp.lr,
            schedule_type=get_lr_schedule_type(),
            warmup_epochs=get_lr_warmup_epochs(),
            hold_epochs=get_lr_hold_epochs(),
            min_lr=get_lr_min(),
            decay_gamma=get_lr_decay_gamma(),
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch_index}/{hp.epochs}")

        epoch_sum = {
            "loss": 0.0,
            "mel_loss": 0.0,
            "gate_loss": 0.0,
            "attn_loss": 0.0,
            "mel_before_loss": 0.0,
            "mel_after_loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "early_stop_rate": 0.0,
            "p_prev_mean": 0.0,
            "p_last_mean": 0.0,
        }
        n_batches = 0
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
            )

            last_outputs = outputs
            for key in epoch_sum:
                epoch_sum[key] += float(stats[key])
            n_batches += 1

            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                mel=f"{stats['mel_loss']:.4f}",
                gate=f"{stats['gate_loss']:.4f}",
                attn=f"{stats['attn_loss']:.4f}",
                lr=f"{current_lr:.6f}",
            )

        if n_batches == 0:
            train_stats = {k: 0.0 for k in epoch_sum}
        else:
            train_stats = {k: v / n_batches for k, v in epoch_sum.items()}

        writer.add_scalars(
            "train_epoch/loss",
            {
                "total": train_stats["loss"],
                "mel": train_stats["mel_loss"],
                "gate": train_stats["gate_loss"],
                "guided_attn": train_stats["attn_loss"],
                "mel_before": train_stats["mel_before_loss"],
                "mel_after": train_stats["mel_after_loss"],
            },
            epoch_index,
        )

        writer.add_scalars(
            "train_epoch/stop",
            {
                "accuracy": train_stats["accuracy"],
                "precision": train_stats["precision"],
                "recall": train_stats["recall"],
                "early_stop_rate": train_stats["early_stop_rate"],
                "p_prev_mean": train_stats["p_prev_mean"],
                "p_last_mean": train_stats["p_last_mean"],
            },
            epoch_index,
        )

        writer.add_scalar("train_epoch/lr", current_lr, epoch_index)

        if (epoch_index % get_log_alignment_every_epoch() == 0) and (last_outputs is not None):
            log_alignment_image(
                writer=writer,
                alignments=last_outputs["alignments"],
                epoch_index=epoch_index,
                tag="train_epoch/alignment",
            )

        if epoch_index % get_validate_every_epoch() == 0:
            val_stats = validate(
                model=model,
                criterion=criterion,
                dataloader=val_loader,
                device=device,
                amp_device_type=amp_device_type,
            )

            writer.add_scalars(
                "val_epoch/loss",
                {
                    "total": val_stats["loss"],
                    "mel": val_stats["mel_loss"],
                    "gate": val_stats["gate_loss"],
                    "guided_attn": val_stats["attn_loss"],
                },
                epoch_index,
            )

            writer.add_scalars(
                "val_epoch/stop",
                {
                    "accuracy": val_stats["accuracy"],
                    "precision": val_stats["precision"],
                    "recall": val_stats["recall"],
                    "early_stop_rate": val_stats["early_stop_rate"],
                    "p_prev_mean": val_stats["p_prev_mean"],
                    "p_last_mean": val_stats["p_last_mean"],
                },
                epoch_index,
            )

        if epoch_index % get_sample_every_epoch() == 0:
            save_validation_sample(
                model=model,
                dataset=full_dataset,
                device=device,
                save_dir=sample_dir,
                epoch_index=epoch_index,
            )

        if epoch_index % get_save_every_epoch() == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_tacotron2_epoch_{epoch_index:06d}.pth.tar"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch_index,
                global_step=global_step,
                path=ckpt_path,
            )
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=get_max_checkpoints_to_keep())

    writer.close()


if __name__ == "__main__":
    main()
