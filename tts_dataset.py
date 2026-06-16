#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import hyperparams_base as hp
from text import text_to_sequence

try:
    import soundfile as sf
except ImportError:  # optional fallback
    sf = None

try:
    from scipy.io import wavfile as scipy_wavfile
except ImportError:  # optional fallback
    scipy_wavfile = None

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

def get_default_wavs_dir() -> Path:
    """
    Default waveform directory for vocoder training.

    Priority:
        1. hp.wavs_dir, if defined
        2. <hp.data_path>/wavs
    """
    if hasattr(hp, "wavs_dir"):
        return Path(hp.wavs_dir)
    return Path(hp.data_path) / "wavs"

def make_wav_path(wavs_dir: str | Path, utt_id: str) -> Path:
    """
    Expected waveform file:
        <wavs_dir>/<utt_id>.wav
    """
    return Path(wavs_dir) / f"{utt_id}.wav"

# =============================================================================
# Feature helpers
# =============================================================================

def load_npy(path: str | Path, dtype: np.dtype) -> np.ndarray:
    arr = np.load(path)
    return arr.astype(dtype, copy=False)

def load_wav(path: str | Path) -> np.ndarray:
    """
    Load a waveform as float32 in approximately [-1, 1].

    Requires soundfile or scipy. soundfile is preferred because it returns
    normalized floating-point audio directly for standard PCM WAV files.
    """
    path = Path(path)
    if sf is not None:
        wav, _sample_rate = sf.read(path, dtype="float32", always_2d=False)
    elif scipy_wavfile is not None:
        _sample_rate, wav = scipy_wavfile.read(path)
        if np.issubdtype(wav.dtype, np.integer):
            max_value = float(np.iinfo(wav.dtype).max)
            wav = wav.astype(np.float32) / max_value
        else:
            wav = wav.astype(np.float32, copy=False)
    else:
        raise ImportError("Install soundfile or scipy to load wav files for vocoder training")

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.ndim != 1:
        raise ValueError(f"Expected mono waveform or stereo waveform, got shape {wav.shape}")
    return wav.astype(np.float32, copy=False)

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
    Tacotron stop/gate target with a positive region covering the last few frames.

    This usually stabilizes stop-token learning better than a single positive frame.
    """
    if mel_length <= 0:
        raise ValueError(f"mel_length must be positive, got {mel_length}")

    gate = np.zeros((mel_length,), dtype=np.float32)

    #positive_region = int(getattr(hp, "gate_positive_region", 1))
    #start_idx = max(0, mel_length - positive_region)
    gate[mel_length-1:] = 1.0
    
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
# Length-aware batch sampling
# =============================================================================

class LengthBucketBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups examples with approximately similar lengths.

    This reduces padding waste for variable-length TTS training while preserving
    enough randomness between epochs.

    Typical Tacotron usage:
        - use mel lengths as `lengths`
        - set bucket_size to batch_size * 20 or batch_size * 30
        - pass this sampler to DataLoader as `batch_sampler`
        - do not pass `batch_size` or `shuffle` to DataLoader
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(lengths) == 0:
            raise ValueError("lengths must be non-empty")

        self.lengths = np.asarray(lengths, dtype=np.int64)
        if np.any(self.lengths <= 0):
            raise ValueError("all lengths must be positive")

        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size or batch_size * 20)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self.epoch = 0

        if self.bucket_size < self.batch_size:
            raise ValueError(
                f"bucket_size must be >= batch_size, got bucket_size={self.bucket_size}, "
                f"batch_size={self.batch_size}"
            )

    def set_epoch(self, epoch: int) -> None:
        """
        Optional helper for deterministic epoch-dependent shuffling.

        If you use DistributedDataParallel later, call sampler.set_epoch(epoch)
        at the beginning of each epoch.
        """
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(None if self.seed is None else self.seed + self.epoch)

        indices = np.arange(len(self.lengths), dtype=np.int64)

        # A small pre-shuffle makes examples with identical/similar lengths vary
        # before the stable sorting step.
        if self.shuffle:
            rng.shuffle(indices)

        indices = indices[np.argsort(self.lengths[indices], kind="stable")]

        buckets: List[np.ndarray] = [
            indices[i : i + self.bucket_size]
            for i in range(0, len(indices), self.bucket_size)
        ]

        if self.shuffle:
            rng.shuffle(buckets)

        batches: List[List[int]] = []
        for bucket in buckets:
            bucket = bucket.copy()
            if self.shuffle:
                rng.shuffle(bucket)

            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append([int(i) for i in batch])

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return int(np.ceil(len(self.lengths) / self.batch_size))


def compute_dataset_lengths(
    dataset: Dataset,
    mode: str = "mel",
    cache_path: str | Path | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Compute or load per-example sequence lengths.

    Supported modes:
        "mel"  - length of mel spectrogram, preferred for Tacotron batching
        "text" - length of encoded text
        "wav"  - waveform length, useful for vocoder batching

    The function works with the dataset classes defined in this file. For other
    datasets, it falls back to reading __getitem__(idx) and checking common keys.
    """
    mode = str(mode).lower().strip()
    if mode not in {"mel", "text", "wav"}:
        raise ValueError(f"Unsupported length mode: {mode!r}. Use 'mel', 'text', or 'wav'.")

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            lengths = np.load(cache_path)
            if len(lengths) != len(dataset):
                raise ValueError(
                    f"Length cache size mismatch: cache has {len(lengths)} entries, "
                    f"dataset has {len(dataset)} entries. Recompute the cache."
                )
            return lengths.astype(np.int64, copy=False)

    lengths: List[int] = []

    for idx in range(len(dataset)):
        if mode == "mel" and hasattr(dataset, "load_mel") and hasattr(dataset, "get_utt_id"):
            utt_id = dataset.get_utt_id(idx)  # type: ignore[attr-defined]
            value = int(dataset.load_mel(utt_id).shape[0])  # type: ignore[attr-defined]
        elif mode == "text" and hasattr(dataset, "encode_text") and hasattr(dataset, "get_text_raw"):
            text = dataset.encode_text(dataset.get_text_raw(idx))  # type: ignore[attr-defined]
            value = int(text.shape[0])
        elif mode == "wav" and hasattr(dataset, "load_wav_from_dir") and hasattr(dataset, "get_utt_id") and hasattr(dataset, "wavs_dir"):
            utt_id = dataset.get_utt_id(idx)  # type: ignore[attr-defined]
            wav = dataset.load_wav_from_dir(utt_id, dataset.wavs_dir)  # type: ignore[attr-defined]
            value = int(wav.shape[0])
        else:
            item = dataset[idx]  # type: ignore[index]
            key = f"{mode}_length"
            if key in item:
                value = int(item[key])
            elif mode in item:
                value = int(item[mode].shape[0])
            else:
                raise KeyError(
                    f"Cannot infer {mode!r} length for item {idx}. "
                    f"Expected key {key!r} or {mode!r}."
                )

        if value <= 0:
            raise ValueError(f"Non-positive length for item {idx}: {value}")
        lengths.append(value)

    lengths_arr = np.asarray(lengths, dtype=np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, lengths_arr)

    return lengths_arr


def make_length_bucketed_loader(
    dataset: Dataset,
    collate_fn,
    batch_size: int | None = None,
    length_mode: str = "mel",
    bucket_size: int | None = None,
    lengths: Sequence[int] | None = None,
    lengths_cache_path: str | Path | None = None,
    force_recompute_lengths: bool = False,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    seed: int | None = None,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """
    Build a DataLoader that uses LengthBucketBatchSampler.

    Important:
        DataLoader receives `batch_sampler`, therefore do not pass `batch_size`,
        `shuffle`, `sampler`, or `drop_last` through dataloader_kwargs.
    """
    if batch_size is None:
        batch_size = _get_batch_size()
    if num_workers is None:
        num_workers = _get_num_workers()
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    forbidden = {"batch_size", "shuffle", "sampler", "batch_sampler", "drop_last"}
    bad_keys = forbidden.intersection(dataloader_kwargs.keys())
    if bad_keys:
        raise ValueError(
            "Do not pass these arguments with batch_sampler: " + ", ".join(sorted(bad_keys))
        )

    if lengths is None:
        lengths = compute_dataset_lengths(
            dataset,
            mode=length_mode,
            cache_path=lengths_cache_path,
            force_recompute=force_recompute_lengths,
        )

    batch_sampler = LengthBucketBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        bucket_size=bucket_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


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

    def load_wav_from_dir(self, utt_id: str, wavs_dir: str | Path) -> np.ndarray:
        wav_path = make_wav_path(wavs_dir, utt_id)
        if not wav_path.exists():
            raise FileNotFoundError(
                f"Missing wav file: {wav_path} (expected <wavs_dir>/<utt_id>.wav)"
            )
        return load_wav(wav_path)


# =============================================================================
# Tacotron dataset
# =============================================================================

class TacotronDataset(BaseTTSDataset):
    """
    Dataset for Tacotron-style training.
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


class TacotronMel2MagDataset(BaseTTSDataset):
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

class VocoderDataset(BaseTTSDataset):
    """
    Dataset for mel -> waveform vocoder training.

    Returns:
        mel: (T_mel, n_mels)
        wav: (T_audio,)
    """

    def __init__(
        self,
        csv_file: str | Path,
        features_dir: str | Path,
        wavs_dir: str | Path,
        cleaners: str | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(csv_file=csv_file, features_dir=features_dir, cleaners=cleaners)
        self.wavs_dir = Path(wavs_dir)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt_id = self.get_utt_id(idx)
        mel = self.load_mel(utt_id)
        wav = self.load_wav_from_dir(utt_id, self.wavs_dir)

        return {
            "utt_id": utt_id,
            "mel": mel,
            "wav": wav,
            "mel_length": int(mel.shape[0]),
            "wav_length": int(wav.shape[0]),
        }

# =============================================================================
# Collate functions
# =============================================================================

def collate_fn_tacotron(batch: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
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


def collate_fn_mel2mag(batch: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
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

def collate_fn_vocoder(batch: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    if len(batch) == 0:
        raise ValueError("Empty batch")
    if not isinstance(batch[0], Mapping):
        raise TypeError(f"batch must contain dict-like objects, got {type(batch[0])}")

    mel_lengths = np.asarray([int(x["mel_length"]) for x in batch], dtype=np.int64)
    wav_lengths = np.asarray([int(x["wav_length"]) for x in batch], dtype=np.int64)

    max_mel_len = int(max(mel_lengths))
    max_wav_len = int(max(wav_lengths))

    mel = stack_padded_2d(
        [x["mel"] for x in batch],
        pad_value=0.0,
        target_length=max_mel_len,
        dtype=np.float32,
    )

    wav = stack_padded_1d(
        [x["wav"] for x in batch],
        pad_value=0.0,
        dtype=np.float32,
    )

    if wav.shape[1] < max_wav_len:
        wav = np.pad(
            wav,
            ((0, 0), (0, max_wav_len - wav.shape[1])),
            mode="constant",
            constant_values=0.0,
        )

    return {
        "mel": torch.from_numpy(mel).float(),
        "wav": torch.from_numpy(wav).float(),
        "mel_lengths": torch.from_numpy(mel_lengths).long(),
        "wav_lengths": torch.from_numpy(wav_lengths).long(),
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


def get_mel2mag_dataset(
    csv_file: str | Path | None = None,
    features_dir: str | Path | None = None,
) -> TacotronMel2MagDataset:
    if csv_file is None or features_dir is None:
        default_csv, default_features = get_default_paths()
        csv_file = default_csv if csv_file is None else csv_file
        features_dir = default_features if features_dir is None else features_dir
    return TacotronMel2MagDataset(csv_file=csv_file, features_dir=features_dir)

def get_vocoder_dataset(
    csv_file: str | Path | None = None,
    features_dir: str | Path | None = None,
    wavs_dir: str | Path | None = None,
) -> VocoderDataset:
    if csv_file is None or features_dir is None:
        default_csv, default_features = get_default_paths()
        csv_file = default_csv if csv_file is None else csv_file
        features_dir = default_features if features_dir is None else features_dir
    if wavs_dir is None:
        wavs_dir = get_default_wavs_dir()
    return VocoderDataset(csv_file=csv_file, features_dir=features_dir, wavs_dir=wavs_dir)

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


def _print_tacotron_dataset_smoke_test_header(
    title: str,
    dataset: Dataset,
    *,
    bucket_sampler_enabled: bool,
    bucket_size: int | None = None,
) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"Dataset size     : {len(dataset)}")
    print(f"Metadata path    : {get_default_paths()[0]}")
    print(f"Features dir     : {get_default_paths()[1]}")
    print(f"Batch size       : {_get_batch_size()}")
    print(f"Num workers      : {_get_num_workers()}")
    print(f"Outputs per step : {get_outputs_per_step()}")
    print(f"Bucket sampler   : {'enabled' if bucket_sampler_enabled else 'disabled'}")
    if bucket_size is not None:
        print(f"Bucket size      : {bucket_size}")
    print("-" * 80)


def _run_one_pass_over_tacotron_loader(loader: DataLoader) -> None:
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


def run_one_pass_over_tacotron_dataset_plain_sampler() -> None:
    """
    Smoke test for the baseline Tacotron DataLoader.

    This uses the standard DataLoader batching logic:
        - batch_size is passed directly to DataLoader
        - shuffle is disabled for deterministic inspection
        - no length bucketing is used
    """
    dataset = get_tacotron_dataset()

    _print_tacotron_dataset_smoke_test_header(
        "Tacotron dataset smoke test: plain sampler",
        dataset,
        bucket_sampler_enabled=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=_get_batch_size(),
        shuffle=False,
        num_workers=_get_num_workers(),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_tacotron,
        drop_last=False,
    )

    _run_one_pass_over_tacotron_loader(loader)


def run_one_pass_over_tacotron_dataset_bucket_sampler() -> None:
    """
    Smoke test for the length-bucketed Tacotron DataLoader.

    This uses LengthBucketBatchSampler:
        - batch_size is controlled by the batch sampler
        - DataLoader receives batch_sampler=...
        - DataLoader must not receive batch_size, shuffle, or drop_last
    """
    dataset = get_tacotron_dataset()
    bucket_size = int(getattr(hp, "bucket_size", _get_batch_size() * 20))

    _print_tacotron_dataset_smoke_test_header(
        "Tacotron dataset smoke test: length bucket sampler",
        dataset,
        bucket_sampler_enabled=True,
        bucket_size=bucket_size,
    )

    loader = make_length_bucketed_loader(
        dataset=dataset,
        collate_fn=collate_fn_tacotron,
        batch_size=_get_batch_size(),
        bucket_size=bucket_size,
        shuffle=True,
        drop_last=False,
        num_workers=_get_num_workers(),
        pin_memory=torch.cuda.is_available(),
    )

    _run_one_pass_over_tacotron_loader(loader)



if __name__ == "__main__":
    run_one_pass_over_tacotron_dataset_plain_sampler()
    run_one_pass_over_tacotron_dataset_bucket_sampler()
