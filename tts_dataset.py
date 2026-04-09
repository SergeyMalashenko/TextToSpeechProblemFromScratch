#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import hyperparams as hp
from text import text_to_sequence


# =============================================================================
# Metadata utilities
# =============================================================================

def read_metadata(csv_file: str | Path) -> pd.DataFrame:
    """
    Expected format:
        utt_id|text|normalized_text

    Only the first two columns are required here.
    """
    df = pd.read_csv(csv_file, sep="|", header=None)

    if df.shape[1] < 2:
        raise ValueError(
            f"Metadata file {csv_file} must contain at least 2 columns: utt_id|text"
        )

    return df.rename(columns={0: "utt_id", 1: "text"})


def normalize_utt_id(value: Any) -> str:
    """
    Converts an entry like 'LJ001-0001' or 'LJ001-0001.wav' to 'LJ001-0001'.
    """
    return Path(str(value).strip()).stem


def get_default_paths() -> tuple[Path, Path]:
    data_root = Path(hp.data_path)
    metadata_csv = data_root / "metadata.csv"
    features_dir = data_root / "features"
    return metadata_csv, features_dir


def make_feature_paths(features_dir: str | Path, utt_id: str) -> tuple[Path, Path]:
    """
    Expected feature files:
        <features_dir>/<utt_id>.mel.npy
        <features_dir>/<utt_id>.mag.npy
    """
    features_dir = Path(features_dir)
    mel_path = features_dir / f"{utt_id}.mel.npy"
    mag_path = features_dir / f"{utt_id}.mag.npy"
    return mel_path, mag_path


# =============================================================================
# Feature helpers
# =============================================================================

def load_npy(path: str | Path, dtype: np.dtype) -> np.ndarray:
    arr = np.load(path)
    return arr.astype(dtype, copy=False)


def make_mel_input(mel: np.ndarray) -> np.ndarray:
    """
    Teacher-forcing input for Tacotron decoder.

    Input:
        mel: (T, n_mels)

    Output:
        mel_input: (T, n_mels), where first frame is zeros and
                   frames 1..T-1 are mel[0..T-2]
    """
    if mel.ndim != 2:
        raise ValueError(f"Expected mel with shape (T, C), got {mel.shape}")

    num_mels = mel.shape[1]
    zero_frame = np.zeros((1, num_mels), dtype=np.float32)
    return np.concatenate([zero_frame, mel[:-1]], axis=0)


def make_gate_target(mel_length: int) -> np.ndarray:
    """
    Tacotron stop/gate target.

    Convention:
      - 0 for frames before the final frame
      - 1 from the final frame onward

    For a single utterance before padding, this means:
      [0, 0, ..., 0, 1]
    """
    if mel_length <= 0:
        raise ValueError(f"mel_length must be positive, got {mel_length}")

    gate = np.zeros((mel_length,), dtype=np.float32)
    gate[mel_length - 1:] = 1.0
    return gate


def pad_length_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return length
    remainder = length % multiple
    if remainder == 0:
        return length
    return length + (multiple - remainder)


def get_outputs_per_step() -> int:
    if hasattr(hp, "outputs_per_step"):
        return int(hp.outputs_per_step)
    if hasattr(hp, "r"):
        return int(hp.r)
    return 1


# =============================================================================
# Padding helpers
# =============================================================================

def pad_1d(arr: np.ndarray, target_length: int, pad_value: int | float = 0) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError(f"pad_1d expects a 1D array, got {arr.shape}")
    if arr.shape[0] >= target_length:
        return arr
    pad_width = target_length - arr.shape[0]
    return np.pad(arr, (0, pad_width), mode="constant", constant_values=pad_value)


def pad_2d_time(arr: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad (T, C) to (target_length, C)
    """
    if arr.ndim != 2:
        raise ValueError(f"pad_2d_time expects a 2D array, got {arr.shape}")
    if arr.shape[0] >= target_length:
        return arr
    pad_width = target_length - arr.shape[0]
    return np.pad(
        arr,
        ((0, pad_width), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def stack_padded_1d(
    arrays: Sequence[np.ndarray],
    pad_value: int | float = 0,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    if len(arrays) == 0:
        raise ValueError("stack_padded_1d received an empty sequence")
    max_len = max(arr.shape[0] for arr in arrays)
    out = np.stack([pad_1d(arr, max_len, pad_value=pad_value) for arr in arrays], axis=0)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def stack_padded_2d(
    arrays: Sequence[np.ndarray],
    pad_value: float = 0.0,
    target_length: int | None = None,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    if len(arrays) == 0:
        raise ValueError("stack_padded_2d received an empty sequence")
    if target_length is None:
        target_length = max(arr.shape[0] for arr in arrays)
    out = np.stack(
        [pad_2d_time(arr, target_length, pad_value=pad_value) for arr in arrays],
        axis=0,
    )
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


# =============================================================================
# Base dataset
# =============================================================================

class BaseTTSDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        features_dir: str | Path,
        cleaners: str | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.df = read_metadata(csv_file)
        self.features_dir = Path(features_dir)

        if cleaners is None:
            cleaners = hp.cleaners

        if isinstance(cleaners, str):
            self.cleaners = [cleaners]
        else:
            self.cleaners = list(cleaners)

    def __len__(self) -> int:
        return len(self.df)

    def get_row(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]

    def get_utt_id(self, idx: int) -> str:
        return normalize_utt_id(self.get_row(idx)["utt_id"])

    def get_text_raw(self, idx: int) -> str:
        return str(self.get_row(idx)["text"])

    def encode_text(self, text: str) -> np.ndarray:
        seq = text_to_sequence(text, self.cleaners)
        return np.asarray(seq, dtype=np.int32)

    def load_mel(self, utt_id: str) -> np.ndarray:
        mel_path, _ = make_feature_paths(self.features_dir, utt_id)
        if not mel_path.exists():
            raise FileNotFoundError(
                f"Missing mel file: {mel_path} (run prepare_data first)"
            )
        return load_npy(mel_path, np.float32)

    def load_mag(self, utt_id: str) -> np.ndarray:
        _, mag_path = make_feature_paths(self.features_dir, utt_id)
        if not mag_path.exists():
            raise FileNotFoundError(
                f"Missing mag file: {mag_path} (run prepare_data first)"
            )
        return load_npy(mag_path, np.float32)


# =============================================================================
# Tacotron dataset
# =============================================================================

class TacotronDataset(BaseTTSDataset):
    """
    Dataset for Tacotron-style RNN model training.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt_id = self.get_utt_id(idx)
        text = self.encode_text(self.get_text_raw(idx))
        mel = self.load_mel(utt_id)
        mel_input = make_mel_input(mel)

        text_length = int(text.shape[0])
        mel_length = int(mel.shape[0])

        return {
            "utt_id": utt_id,
            "text": text,
            "mel": mel,
            "mel_input": mel_input,
            "text_length": text_length,
            "mel_length": mel_length,
        }


class TacotronPostnetDataset(BaseTTSDataset):
    """
    Dataset for mel -> mag training.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt_id = self.get_utt_id(idx)
        mel = self.load_mel(utt_id)
        mag = self.load_mag(utt_id)

        return {
            "utt_id": utt_id,
            "mel": mel,
            "mag": mag,
            "mel_length": int(mel.shape[0]),
        }


# =============================================================================
# Collate functions
# =============================================================================

def collate_fn_tacotron(batch: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Returns:
      text            : (B, T_text)
      text_lengths    : (B,)
      mel_input       : (B, T_mel, n_mels)
      mel_target      : (B, T_mel, n_mels)
      gate_target     : (B, T_mel)
      output_lengths  : (B,)
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")
    if not isinstance(batch[0], Mapping):
        raise TypeError(f"batch must contain dict-like objects, got {type(batch[0])}")

    batch = sorted(batch, key=lambda x: int(x["text_length"]), reverse=True)

    text_lengths = np.asarray([int(x["text_length"]) for x in batch], dtype=np.int64)
    mel_lengths = np.asarray([int(x["mel_length"]) for x in batch], dtype=np.int64)

    r = get_outputs_per_step()
    max_mel_len = max(int(x["mel_length"]) for x in batch)
    max_mel_len = pad_length_to_multiple(max_mel_len, r)

    text = stack_padded_1d(
        [x["text"] for x in batch],
        pad_value=0,
        dtype=np.int64,
    )

    mel_input = stack_padded_2d(
        [x["mel_input"] for x in batch],
        pad_value=0.0,
        target_length=max_mel_len,
        dtype=np.float32,
    )

    mel_target = stack_padded_2d(
        [x["mel"] for x in batch],
        pad_value=0.0,
        target_length=max_mel_len,
        dtype=np.float32,
    )

    gate_target = stack_padded_1d(
        [make_gate_target(int(x["mel_length"])) for x in batch],
        pad_value=1.0,
        dtype=np.float32,
    )

    if gate_target.shape[1] < max_mel_len:
        gate_target = np.pad(
            gate_target,
            ((0, 0), (0, max_mel_len - gate_target.shape[1])),
            mode="constant",
            constant_values=1.0,
        )

    return {
        "text": torch.from_numpy(text).long(),
        "text_lengths": torch.from_numpy(text_lengths).long(),
        "mel_input": torch.from_numpy(mel_input).float(),
        "mel_target": torch.from_numpy(mel_target).float(),
        "gate_target": torch.from_numpy(gate_target).float(),
        "output_lengths": torch.from_numpy(mel_lengths).long(),
    }


def collate_fn_postnet(batch: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    if len(batch) == 0:
        raise ValueError("Empty batch")
    if not isinstance(batch[0], Mapping):
        raise TypeError(f"batch must contain dict-like objects, got {type(batch[0])}")

    mel_lengths = np.asarray([int(x["mel_length"]) for x in batch], dtype=np.int64)
    max_mel_len = int(max(mel_lengths))

    mel = stack_padded_2d(
        [x["mel"] for x in batch],
        pad_value=0.0,
        target_length=max_mel_len,
        dtype=np.float32,
    )

    mag = stack_padded_2d(
        [x["mag"] for x in batch],
        pad_value=0.0,
        target_length=max_mel_len,
        dtype=np.float32,
    )

    return {
        "mel": torch.from_numpy(mel).float(),
        "mag": torch.from_numpy(mag).float(),
        "lengths": torch.from_numpy(mel_lengths).long(),
    }


# =============================================================================
# Convenience builders
# =============================================================================

def get_tacotron_dataset(
    csv_file: str | Path | None = None,
    features_dir: str | Path | None = None,
) -> TacotronDataset:
    if csv_file is None or features_dir is None:
        default_csv, default_features = get_default_paths()
        csv_file = default_csv if csv_file is None else csv_file
        features_dir = default_features if features_dir is None else features_dir
    return TacotronDataset(csv_file=csv_file, features_dir=features_dir)


def get_postnet_dataset(
    csv_file: str | Path | None = None,
    features_dir: str | Path | None = None,
) -> TacotronPostnetDataset:
    if csv_file is None or features_dir is None:
        default_csv, default_features = get_default_paths()
        csv_file = default_csv if csv_file is None else csv_file
        features_dir = default_features if features_dir is None else features_dir
    return TacotronPostnetDataset(csv_file=csv_file, features_dir=features_dir)


def get_param_size(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# =============================================================================
# Debug / smoke-test main
# =============================================================================

def _get_num_workers() -> int:
    if hasattr(hp, "num_workers"):
        return int(hp.num_workers)
    return 0


def _get_batch_size() -> int:
    if hasattr(hp, "batch_size"):
        return int(hp.batch_size)
    return 4


def run_one_pass_over_tacotron_dataset() -> None:
    dataset = get_tacotron_dataset()

    print("=" * 80)
    print("Tacotron dataset smoke test")
    print("=" * 80)
    print(f"Dataset size     : {len(dataset)}")
    print(f"Metadata path    : {get_default_paths()[0]}")
    print(f"Features dir     : {get_default_paths()[1]}")
    print(f"Batch size       : {_get_batch_size()}")
    print(f"Num workers      : {_get_num_workers()}")
    print(f"Outputs per step : {get_outputs_per_step()}")
    print("-" * 80)

    loader = DataLoader(
        dataset,
        batch_size=_get_batch_size(),
        shuffle=False,
        num_workers=_get_num_workers(),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_tacotron,
        drop_last=False,
    )

    total_batches = 0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        total_batches += 1
        batch_size = int(batch["text"].shape[0])
        total_samples += batch_size

        print(f"[Batch {batch_idx}]")
        print(f"  text.shape          : {tuple(batch['text'].shape)}")
        print(f"  text_lengths.shape  : {tuple(batch['text_lengths'].shape)}")
        print(f"  mel_input.shape     : {tuple(batch['mel_input'].shape)}")
        print(f"  mel_target.shape    : {tuple(batch['mel_target'].shape)}")
        print(f"  gate_target.shape   : {tuple(batch['gate_target'].shape)}")
        print(f"  output_lengths.shape: {tuple(batch['output_lengths'].shape)}")

        print(f"  text_lengths        : {batch['text_lengths'].tolist()}")
        print(f"  output_lengths      : {batch['output_lengths'].tolist()}")

    print("-" * 80)
    print(f"Done. Iterated over {total_batches} batches / {total_samples} samples.")
    print("=" * 80)


if __name__ == "__main__":
    run_one_pass_over_tacotron_dataset()
