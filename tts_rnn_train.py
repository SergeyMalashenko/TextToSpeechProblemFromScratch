#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import hyperparams as hp

from tts_dataset import get_tacotron_dataset, collate_fn_tacotron
from tts_rnn_model    import Tacotron2, Tacotron2Loss


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# =============================================================================
# Helpers
# =============================================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_num_workers() -> int:
    if hasattr(hp, "num_workers"):
        return int(hp.num_workers)
    return 4


def adjust_learning_rate(optimizer: torch.optim.Optimizer, step_num: int, warmup_step: int = 4000) -> float:
    """
    Noam-style LR schedule from the old code.
    """
    lr = hp.lr * (warmup_step ** 0.5) * min(
        step_num * (warmup_step ** -1.5),
        step_num ** -0.5,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return float(lr)


@torch.no_grad()
def compute_gate_metrics(
    gate_logits: torch.Tensor,
    gate_target: torch.Tensor,
    output_lengths: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    gate_logits:   (B, T)
    gate_target:   (B, T) float {0,1}
    output_lengths:(B,)
    """
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

    B, T = probs.shape
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
        early_pred = pred[b, : L - 1].any().item()
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    torch.save(
        {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# =============================================================================
# Train step
# =============================================================================

def train_step(
    model: nn.Module,
    criterion: Tacotron2Loss,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    global_step: int,
) -> Dict[str, float]:
    batch = move_batch_to_device(batch, device)

    outputs = model(
        text=batch["text"],
        text_lengths=batch["text_lengths"],
        mel_input=batch["mel_input"],
    )

    loss_dict = criterion(
        outputs=outputs,
        mel_target=batch["mel_target"],
        gate_target=batch["gate_target"],
    )

    optimizer.zero_grad(set_to_none=True)
    loss_dict["loss"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

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
        **gate_metrics,
    }

    return stats


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    dataset = get_tacotron_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=collate_fn_tacotron,
        drop_last=True,
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )

    model = Tacotron2().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    criterion = Tacotron2Loss()

    writer = SummaryWriter()

    checkpoint_dir = Path(hp.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(hp.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            global_step += 1

            if global_step < 400000:
                current_lr = adjust_learning_rate(optimizer, global_step)
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            stats = train_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                batch=batch,
                device=device,
                global_step=global_step,
            )

            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                mel=f"{stats['mel_loss']:.4f}",
                gate=f"{stats['gate_loss']:.4f}",
                lr=f"{current_lr:.6f}",
            )

            writer.add_scalars(
                "training_loss",
                {
                    "total": stats["loss"],
                    "mel_loss": stats["mel_loss"],
                    "gate_loss": stats["gate_loss"],
                },
                global_step,
            )

            writer.add_scalars(
                "stop_stats",
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

            writer.add_scalar("learning_rate", current_lr, global_step)

            if global_step % getattr(hp, "image_step", 500) == 1:
                # Log first sample's alignment
                outputs = None
                with torch.no_grad():
                    batch_on_device = move_batch_to_device(batch, device)
                    outputs = model(
                        text=batch_on_device["text"],
                        text_lengths=batch_on_device["text_lengths"],
                        mel_input=batch_on_device["mel_input"],
                    )

                alignments = outputs["alignments"]  # (B, T_out, T_text)
                if alignments.ndim == 3 and alignments.size(0) > 0:
                    attn_img = alignments[0].unsqueeze(0)  # (1, T_out, T_text)
                    writer.add_image("attention/alignment", attn_img, global_step)

            if global_step % hp.save_step == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_tacotron2_{global_step}.pth.tar"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=checkpoint_path,
                )

    writer.close()


if __name__ == "__main__":
    main()
