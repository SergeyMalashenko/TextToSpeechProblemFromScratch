# data_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy import signal
from torch.utils.data import Dataset
from tqdm import tqdm

import hyperparams as hp

EPS = 1e-5

# =============================================================================
# Audio processing
# =============================================================================
def load_audio(path: str | Path, sr: int) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=sr)
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


def apply_preemphasis(wav: np.ndarray, coeff: float) -> np.ndarray:
    if len(wav) == 0:
        return wav.astype(np.float32)
    emphasized = np.append(wav[0], wav[1:] - coeff * wav[:-1])
    return emphasized.astype(np.float32)


def amplitude_to_db(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(eps, x))


def normalize_db(x_db: np.ndarray, ref_db: float, max_db: float) -> np.ndarray:
    x = (x_db - ref_db + max_db) / max_db
    return np.clip(x, 1e-8, 1.0).astype(np.float32)


def denormalize_db(x_norm: np.ndarray, ref_db: float, max_db: float) -> np.ndarray:
    return (np.clip(x_norm, 0.0, 1.0) * max_db) - max_db + ref_db


def build_mel_basis(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float32)


def wav_to_spectrograms(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert wav file to normalized mel and magnitude spectrograms.

    Returns:
        mel: (T, n_mels), float32
        mag: (T, 1 + n_fft // 2), float32
    """
    wav = load_audio(path, sr=hp.sr)
    wav = apply_preemphasis(wav, coeff=hp.preemphasis)

    stft = librosa.stft(
        y=wav,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window="hann",
    )
    mag = np.abs(stft).astype(np.float32)

    mel_basis = build_mel_basis(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
    mel = mel_basis @ mag

    mel_db = amplitude_to_db(mel)
    mag_db = amplitude_to_db(mag)

    mel_norm = normalize_db(mel_db, ref_db=hp.ref_db, max_db=hp.max_db)
    mag_norm = normalize_db(mag_db, ref_db=hp.ref_db, max_db=hp.max_db)

    return mel_norm.T, mag_norm.T


def invert_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    return librosa.istft(
        spectrogram,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window="hann",
    )


def griffin_lim(magnitude: np.ndarray, n_iter: int) -> np.ndarray:
    """
    magnitude: (freq, time)
    """
    complex_spec = magnitude.astype(np.complex64)

    for _ in range(n_iter):
        wav = invert_spectrogram(complex_spec)
        estimate = librosa.stft(
            wav,
            n_fft=hp.n_fft,
            hop_length=hp.hop_length,
            win_length=hp.win_length,
            window="hann",
        )
        phase = estimate / np.maximum(1e-8, np.abs(estimate))
        complex_spec = magnitude * phase

    wav = invert_spectrogram(complex_spec)
    return np.real(wav).astype(np.float32)


def spectrogram_to_wav(mag_norm_t: np.ndarray) -> np.ndarray:
    """
    Reconstruct waveform from normalized magnitude spectrogram.

    Args:
        mag_norm_t: (T, 1 + n_fft // 2)

    Returns:
        wav: 1D float32 waveform
    """
    mag_norm = mag_norm_t.T
    mag_db = denormalize_db(mag_norm, ref_db=hp.ref_db, max_db=hp.max_db)
    mag_amp = np.power(10.0, mag_db * 0.05)

    wav = griffin_lim(mag_amp ** hp.power, n_iter=hp.n_iter)
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


# =============================================================================
# Metadata and feature preparation
# =============================================================================
def read_metadata(csv_file: str | Path) -> pd.DataFrame:
    """
    Expected format:
        id|text|normalized_text
    """
    df = pd.read_csv(csv_file, sep="|", header=None)
    if df.shape[1] < 1:
        raise ValueError(f"Metadata file {csv_file} is empty or malformed.")
    return df.rename(columns={0: "utt_id"})


def iter_wav_paths(metadata: pd.DataFrame, wav_dir: str | Path) -> Iterable[tuple[str, Path]]:
    wav_dir = Path(wav_dir)
    for utt_id in metadata["utt_id"].astype(str):
        yield utt_id, wav_dir / f"{utt_id}.wav"


def save_features(out_dir: str | Path, utt_id: str, mel: np.ndarray, mag: np.ndarray) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{utt_id}.mel.npy", mel)
    np.save(out_dir / f"{utt_id}.mag.npy", mag)


def process_dataset(
    metadata_csv: str | Path,
    wav_dir: str | Path,
    out_dir: str | Path,
    skip_existing: bool = True,
) -> None:
    metadata = read_metadata(metadata_csv)
    out_dir = Path(out_dir)

    for utt_id, wav_path in tqdm(iter_wav_paths(metadata, wav_dir), total=len(metadata), desc="Processing audio"):
        mel_path = out_dir / f"{utt_id}.mel.npy"
        mag_path = out_dir / f"{utt_id}.mag.npy"

        if skip_existing and mel_path.exists() and mag_path.exists():
            continue

        if not wav_path.exists():
            print(f"[WARNING] Missing wav file: {wav_path}")
            continue

        try:
            mel, mag = wav_to_spectrograms(wav_path)
            save_features(out_dir, utt_id, mel, mag)
        except Exception as exc:
            print(f"[ERROR] Failed on {wav_path}: {exc}")


# =============================================================================
# Dataset for training
# =============================================================================
class SpectrogramDataset(Dataset):
    """
    Dataset that loads precomputed mel and magnitude features.
    """

    def __init__(self, metadata_csv: str | Path, features_dir: str | Path):
        self.df = read_metadata(metadata_csv)
        self.features_dir = Path(features_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        utt_id = str(self.df.iloc[idx]["utt_id"])
        mel = np.load(self.features_dir / f"{utt_id}.mel.npy")
        mag = np.load(self.features_dir / f"{utt_id}.mag.npy")

        return {
            "utt_id": utt_id,
            "mel": mel.astype(np.float32),
            "mag": mag.astype(np.float32),
        }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    data_root = Path(hp.data_path)
    metadata_csv = data_root / "metadata.csv"
    wav_dir = data_root / "wavs"
    out_dir = data_root / "features"

    process_dataset(
        metadata_csv=metadata_csv,
        wav_dir=wav_dir,
        out_dir=out_dir,
        skip_existing=True,
    )

    print(f"Done. Features saved to: {out_dir}")
