#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.io.wavfile import write
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import hyperparams_base as hp
from tts_dataset import get_hifigan_dataset, collate_fn_hifigan
from tts_diffusion_schedule import DiffusionSchedule
from tts_diffusion_vocoder_model import DiffusionVocoder

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


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
    np.random.seed(seed)


def average_metric_dict(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    totals: dict[str, float] = defaultdict(float)
    for item in metrics:
        for key, value in item.items():
            totals[key] += float(value)
    return {key: value / len(metrics) for key, value in totals.items()}


def de_emphasis(wav: np.ndarray) -> np.ndarray:
    return signal.lfilter([1], [1, -float(hp.preemphasis)], wav).astype(np.float32)


def save_wav(path: str | Path, wav: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    write(str(path), int(sample_rate), wav)


def build_mel_basis(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
    return torch.tensor(basis, device=device, dtype=dtype)


def wav_to_log_mel(wav: torch.Tensor, mel_basis: torch.Tensor) -> torch.Tensor:
    """
    wav: (B, T)
    returns: (B, n_mels, T_mel)
    """
    window = torch.hann_window(hp.win_length, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window=window,
        return_complex=True,
        center=True,
    )
    mag = torch.abs(spec).clamp_min(1e-5)
    mel = torch.matmul(mel_basis.unsqueeze(0), mag)
    return torch.log(mel.clamp_min(1e-5))


def mel_reconstruction_loss(pred_audio: torch.Tensor, target_audio: torch.Tensor, mel_basis: torch.Tensor) -> torch.Tensor:
    pred_mel = wav_to_log_mel(pred_audio.squeeze(1), mel_basis)
    target_mel = wav_to_log_mel(target_audio.squeeze(1), mel_basis)
    min_frames = min(pred_mel.size(-1), target_mel.size(-1))
    return F.l1_loss(pred_mel[..., :min_frames], target_mel[..., :min_frames])


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution spectral loss for waveform reconstruction.

    Combines spectral convergence and log-magnitude L1 across several STFT
    resolutions. This gives the diffusion vocoder a direct spectral training
    signal, which plain waveform L1/MSE lacks.
    """

    def __init__(self, resolutions: list[tuple[int, int, int]]) -> None:
        super().__init__()
        self.resolutions = [(int(n_fft), int(hop), int(win)) for n_fft, hop, win in resolutions]
        for idx, (_n_fft, _hop, win_length) in enumerate(self.resolutions):
            self.register_buffer(f"window_{idx}", torch.hann_window(win_length), persistent=False)

    def _magnitude(self, wav: torch.Tensor, idx: int) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolutions[idx]
        window = getattr(self, f"window_{idx}").to(device=wav.device, dtype=wav.dtype)
        spec = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        return torch.abs(spec).clamp_min(1e-7)

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = pred_audio.squeeze(1)
        target = target_audio.squeeze(1)
        total_sc = pred.new_tensor(0.0)
        total_log = pred.new_tensor(0.0)

        for idx in range(len(self.resolutions)):
            pred_mag = self._magnitude(pred, idx)
            target_mag = self._magnitude(target, idx)
            sc = torch.linalg.vector_norm(target_mag - pred_mag) / torch.linalg.vector_norm(target_mag).clamp_min(1e-7)
            log_mag = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
            total_sc = total_sc + sc
            total_log = total_log + log_mag

        denom = float(len(self.resolutions))
        total_sc = total_sc / denom
        total_log = total_log / denom
        return total_sc + total_log, total_sc, total_log


def cleanup_old_checkpoints(checkpoint_dir: str | Path, keep_last_n: int) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if keep_last_n <= 0 or not checkpoint_dir.exists():
        return
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_diffusion_vocoder_epoch_*.pth.tar"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(checkpoints) <= keep_last_n:
        return
    for ckpt in checkpoints[:-keep_last_n]:
        try:
            ckpt.unlink()
        except OSError:
            pass


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
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


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
        raise FileNotFoundError(f"Diffusion vocoder checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))
    print(f"Resumed diffusion vocoder from checkpoint: {resume_path}")
    print(f"Start epoch: {start_epoch}, global step: {global_step}")
    return start_epoch, global_step


def train_one_step(
    model: DiffusionVocoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    schedule: DiffusionSchedule,
    stft_loss_fn: MultiResolutionSTFTLoss,
    mel_basis: torch.Tensor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    amp_device_type: str,
) -> dict[str, float]:
    mel = batch["mel"].to(device, non_blocking=True)
    clean_audio = batch["wav"].to(device, non_blocking=True).unsqueeze(1)
    diffusion_step = torch.randint(0, schedule.timesteps, (clean_audio.size(0),), device=device)
    noise = torch.randn_like(clean_audio)
    noisy_audio = schedule.add_noise(clean_audio, diffusion_step, noise)

    with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
        pred_audio = model(noisy_audio, mel, diffusion_step)

    pred_audio_f = pred_audio.float()
    clean_audio_f = clean_audio.float()
    loss_l1 = F.l1_loss(pred_audio_f, clean_audio_f)
    loss_mse = F.mse_loss(pred_audio_f, clean_audio_f)
    loss_stft, loss_stft_sc, loss_stft_log = stft_loss_fn(pred_audio_f, clean_audio_f)
    loss_mel = mel_reconstruction_loss(pred_audio_f, clean_audio_f, mel_basis)
    stft_weight = float(hp_get("diffusion_vocoder_stft_weight", 1.0))
    mel_weight = float(hp_get("diffusion_vocoder_mel_loss_weight", 5.0))
    loss = loss_l1 + loss_mse + stft_weight * loss_stft + mel_weight * loss_mel

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), float(hp_get("diffusion_vocoder_clip_grad_norm", 1.0)))
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.item()),
        "l1": float(loss_l1.item()),
        "mse": float(loss_mse.item()),
        "stft": float(loss_stft.item()),
        "stft_sc": float(loss_stft_sc.item()),
        "stft_log": float(loss_stft_log.item()),
        "mel": float(loss_mel.item()),
    }


@torch.no_grad()
def validate(
    model: DiffusionVocoder,
    dataloader: DataLoader,
    schedule: DiffusionSchedule,
    stft_loss_fn: MultiResolutionSTFTLoss,
    mel_basis: torch.Tensor,
    device: torch.device,
    amp_device_type: str,
) -> dict[str, float]:
    model.eval()
    metrics = []
    max_batches = int(hp_get("diffusion_vocoder_val_batches", 0))

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="validation", leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        mel = batch["mel"].to(device, non_blocking=True)
        clean_audio = batch["wav"].to(device, non_blocking=True).unsqueeze(1)
        diffusion_step = torch.randint(0, schedule.timesteps, (clean_audio.size(0),), device=device)
        noise = torch.randn_like(clean_audio)
        noisy_audio = schedule.add_noise(clean_audio, diffusion_step, noise)

        with torch.amp.autocast(amp_device_type, enabled=torch.cuda.is_available()):
            pred_audio = model(noisy_audio, mel, diffusion_step)

        pred_audio_f = pred_audio.float()
        clean_audio_f = clean_audio.float()
        loss_l1 = F.l1_loss(pred_audio_f, clean_audio_f)
        loss_mse = F.mse_loss(pred_audio_f, clean_audio_f)
        loss_stft, loss_stft_sc, loss_stft_log = stft_loss_fn(pred_audio_f, clean_audio_f)
        loss_mel = mel_reconstruction_loss(pred_audio_f, clean_audio_f, mel_basis)
        stft_weight = float(hp_get("diffusion_vocoder_stft_weight", 1.0))
        mel_weight = float(hp_get("diffusion_vocoder_mel_loss_weight", 5.0))
        loss = loss_l1 + loss_mse + stft_weight * loss_stft + mel_weight * loss_mel
        metrics.append({
            "loss": float(loss.item()),
            "l1": float(loss_l1.item()),
            "mse": float(loss_mse.item()),
            "stft": float(loss_stft.item()),
            "stft_sc": float(loss_stft_sc.item()),
            "stft_log": float(loss_stft_log.item()),
            "mel": float(loss_mel.item()),
        })

    model.train()
    return average_metric_dict(metrics) if metrics else {"loss": 0.0}


@torch.no_grad()
def save_validation_sample(
    model: DiffusionVocoder,
    dataloader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    sample_dir: str | Path,
    epoch_index: int,
) -> None:
    batch = next(iter(dataloader))
    mel = batch["mel"][:1].to(device, non_blocking=True)
    wav = batch["wav"][:1].to(device, non_blocking=True)
    audio_len = int(wav.size(-1))
    inference_steps = int(hp_get("diffusion_vocoder_inference_steps", 50))
    pred = schedule.sample(model, mel, audio_length=audio_len, inference_steps=inference_steps)

    target_np = wav[0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()
    sample_dir = Path(sample_dir)
    save_wav(sample_dir / f"diffusion_target_epoch_{epoch_index:04d}.wav", de_emphasis(target_np), hp.sr)
    save_wav(sample_dir / f"diffusion_generated_epoch_{epoch_index:04d}.wav", de_emphasis(pred_np), hp.sr)
    np.save(sample_dir / f"diffusion_mel_epoch_{epoch_index:04d}.npy", mel[0].detach().cpu().numpy())


def main() -> None:
    set_seed(get_seed())
    device = get_device()
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    segment_size = int(hp_get("diffusion_vocoder_segment_size", hp.hop_length * 64))
    train_source_dataset = get_hifigan_dataset(segment_size=segment_size, random_segments=True)
    val_source_dataset = get_hifigan_dataset(segment_size=segment_size, random_segments=False)

    dataset_size = len(train_source_dataset)
    val_size = max(1, int(dataset_size * get_val_ratio()))
    train_size = dataset_size - val_size
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(get_seed())).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_dataset = Subset(train_source_dataset, train_indices)
    val_dataset = Subset(val_source_dataset, val_indices)

    batch_size = int(hp_get("diffusion_vocoder_batch_size", 16))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_hifigan,
        drop_last=True,
        num_workers=get_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_hifigan,
        drop_last=False,
        num_workers=max(0, get_num_workers() // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = DiffusionVocoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp_get("diffusion_vocoder_lr", 2e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(hp_get("diffusion_vocoder_weight_decay", 1e-6)),
    )
    scaler = torch.amp.GradScaler(amp_device_type)
    schedule = DiffusionSchedule(
        timesteps=int(hp_get("diffusion_vocoder_train_timesteps", 1000)),
        device=device,
    )
    stft_loss_fn = MultiResolutionSTFTLoss(
        resolutions=list(hp_get("diffusion_vocoder_stft_resolutions", [(512, 128, 512), (1024, 256, 1024)]))
    ).to(device)
    mel_basis = build_mel_basis(device=device, dtype=torch.float32)

    checkpoint_dir = Path(hp_get("diffusion_vocoder_checkpoint_path", "./outputs/checkpoints/diffusion_vocoder"))
    log_dir = Path(hp_get("diffusion_vocoder_log_dir", "./outputs/logs/diffusion_vocoder"))
    sample_dir = Path(hp_get("diffusion_vocoder_sample_path", "./outputs/samples/diffusion_vocoder"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    resume_path = hp_get("resume_diffusion_vocoder_checkpoint", None)
    start_epoch, global_step = maybe_resume_from_checkpoint(model, optimizer, scaler, resume_path, device)

    validate_every_epoch = int(hp_get("diffusion_vocoder_validate_every_epoch", 1))
    save_every_epoch = int(hp_get("diffusion_vocoder_save_every_epoch", 1))
    sample_every_epoch = int(hp_get("diffusion_vocoder_sample_every_epoch", 5))
    max_checkpoints_to_keep = int(hp_get("diffusion_vocoder_max_checkpoints_to_keep", 0))

    print(f"Using device       : {device}")
    print(f"AMP device type    : {amp_device_type}")
    print(f"Train dataset size : {train_size}")
    print(f"Val dataset size   : {val_size}")
    print(f"Checkpoint dir     : {checkpoint_dir}")
    print(f"Log dir            : {log_dir}")
    print(f"Sample dir         : {sample_dir}")
    print(f"Segment size       : {segment_size} samples")
    print(f"Train timesteps    : {schedule.timesteps}")
    print(f"Inference steps    : {int(hp_get('diffusion_vocoder_inference_steps', 50))}")
    print(f"STFT loss weight   : {float(hp_get('diffusion_vocoder_stft_weight', 1.0)):.3f}")
    print(f"STFT resolutions   : {list(hp_get('diffusion_vocoder_stft_resolutions', []))}")
    print(f"Mel loss weight    : {float(hp_get('diffusion_vocoder_mel_loss_weight', 5.0)):.3f}")
    print("Target wav         : trimmed + preemphasized waveform")

    for epoch in range(start_epoch, int(hp_get("diffusion_vocoder_epochs", 10000))):
        epoch_index = epoch + 1
        pbar = tqdm(train_loader, desc=f"epoch {epoch_index}/{int(hp_get('diffusion_vocoder_epochs', 10000))}")
        train_metrics = []

        for batch in pbar:
            global_step += 1
            stats = train_one_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                schedule=schedule,
                stft_loss_fn=stft_loss_fn,
                mel_basis=mel_basis,
                batch=batch,
                device=device,
                amp_device_type=amp_device_type,
            )
            train_metrics.append(stats)
            pbar.set_postfix(
                loss=f"{stats['loss']:.5f}",
                l1=f"{stats['l1']:.5f}",
                mse=f"{stats['mse']:.5f}",
                stft=f"{stats['stft']:.5f}",
                mel=f"{stats['mel']:.5f}",
            )

        train_stats = average_metric_dict(train_metrics)
        print(
            f"[TRAIN epoch={epoch_index}] "
            f"loss={train_stats['loss']:.6f} "
            f"l1={train_stats['l1']:.6f} "
            f"mse={train_stats['mse']:.6f} "
            f"stft={train_stats['stft']:.6f} "
            f"mel={train_stats['mel']:.6f}"
        )
        for key, value in train_stats.items():
            writer.add_scalar(f"train/{key}", value, epoch_index)

        if epoch_index % validate_every_epoch == 0:
            val_stats = validate(
                model=model,
                dataloader=val_loader,
                schedule=schedule,
                stft_loss_fn=stft_loss_fn,
                mel_basis=mel_basis,
                device=device,
                amp_device_type=amp_device_type,
            )
            print(
                f"[VAL epoch={epoch_index}] "
                f"loss={val_stats['loss']:.6f} "
                f"l1={val_stats['l1']:.6f} "
                f"mse={val_stats['mse']:.6f} "
                f"stft={val_stats['stft']:.6f} "
                f"mel={val_stats['mel']:.6f}"
            )
            for key, value in val_stats.items():
                writer.add_scalar(f"val/{key}", value, epoch_index)

        if epoch_index % sample_every_epoch == 0:
            save_validation_sample(
                model=model,
                dataloader=val_loader,
                schedule=schedule,
                device=device,
                sample_dir=sample_dir,
                epoch_index=epoch_index,
            )
            print(f"Saved sample artifacts: {sample_dir}")

        if epoch_index % save_every_epoch == 0:
            ckpt = checkpoint_dir / f"checkpoint_diffusion_vocoder_epoch_{epoch_index:04d}.pth.tar"
            save_checkpoint(model, optimizer, scaler, epoch_index, global_step, ckpt)
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n=max_checkpoints_to_keep)
            print(f"Saved checkpoint: {ckpt}")

    writer.close()


if __name__ == "__main__":
    main()
