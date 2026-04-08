#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hyperparams as hp
import pandas      as pd
import numpy       as np
import torch       as t

import librosa
import math
import os

from torch.utils.data import Dataset
from collections.abc  import Mapping
from text             import text_to_sequence

# =========================
# Datasets
# =========================

class LJDatasets(Dataset):
    """LJSpeech-style dataset for Transformer-TTS training."""
    def __init__(self, csv_file: str, root_dir: str):
        self.landmarks_frame = pd.read_csv(csv_file, sep="|", header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx: int):
        wav_id = str(self.landmarks_frame.iloc[idx, 0]).strip()
        wav_id = os.path.splitext(wav_id)[0]
        wav_name = os.path.join(self.root_dir, f"{wav_id}.wav")

        text = self.landmarks_frame.iloc[idx, 1]
        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)

        mel_path = wav_name[:-4] + ".pt.npy"
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Missing mel file: {mel_path} (run prepare_data first)")

        mel = np.load(mel_path).astype(np.float32)  # (T, n_mels)
        mel_length = int(mel.shape[0])

        mel_input = np.concatenate(
            [np.zeros([1, hp.num_mels], dtype=np.float32), mel[:-1, :]],
            axis=0
        )

        text_length = int(text.shape[0])
        pos_text    = np.arange(1, text_length + 1, dtype=np.int32)
        pos_mel     = np.arange(1, mel_length  + 1, dtype=np.int32)

        return {
            "text"       : text       , "mel"        : mel,
            "text_length": text_length, "mel_length" : mel_length,
            "pos_mel"    : pos_mel    , "pos_text"   : pos_text,
        }


class PostDatasets(Dataset):
    """Dataset for postnet/vocoder training (mel -> mag)."""

    def __init__(self, csv_file: str, root_dir: str):
        self.landmarks_frame = pd.read_csv(csv_file, sep="|", header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx: int):
        wav_id = str(self.landmarks_frame.iloc[idx, 0]).strip()
        wav_id = os.path.splitext(wav_id)[0]
        wav_name = os.path.join(self.root_dir, f"{wav_id}.wav")

        mel_path = wav_name[:-4] + ".pt.npy"
        mag_path = wav_name[:-4] + ".mag.npy"

        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Missing mel file: {mel_path} (run prepare_data first)")
        if not os.path.exists(mag_path):
            raise FileNotFoundError(f"Missing mag file: {mag_path} (run prepare_data first)")

        mel = np.load(mel_path).astype(np.float32)
        mag = np.load(mag_path).astype(np.float32)

        return {"mel": mel, "mag": mag}


# =========================
# Collate functions
# =========================
def collate_fn_transformer(batch):
    if not isinstance(batch[0], Mapping):
        raise TypeError(f"batch must contain dicts; found {type(batch[0])}")

    # Sort by mel length (decoder cost dominates)
    mel_lengths = np.array([int(d["mel_length"]) for d in batch], dtype=np.int32)
    order       = np.argsort(-mel_lengths)

    batch       = [batch[i] for i in order]
    mel_lengths = mel_lengths[order]

    text_lengths = np.array([int(d["text_length"]) for d in batch], dtype=np.int32)

    text      = [d["text"    ] for d in batch]
    mel       = [d["mel"     ] for d in batch]
    pos_text  = [d["pos_text"] for d in batch]
    pos_mel   = [d["pos_mel" ] for d in batch]

    # pad
    text      = _prepare_data(text).astype(np.int32)         # (B, Tmax_text)
    pos_text  = _prepare_data(pos_text).astype(np.int32)     # (B, Tmax_text)

    mel       = _pad_mel(mel).astype(np.float32)            # (B, Tmax_mel, 80)
    pos_mel   = _prepare_data(pos_mel).astype(np.int32)     # (B, Tmax_mel)

    return (
        t.LongTensor (text        ), t.FloatTensor(mel        ),
        t.LongTensor (pos_text    ), t.LongTensor (pos_mel    ),
        t.LongTensor (text_lengths), t.LongTensor (mel_lengths),
    )

def collate_fn_postnet(batch):
    if not isinstance(batch[0], Mapping):
        raise TypeError(f"batch must contain dicts; found {type(batch[0])}")

    mel = _pad_mel([d["mel"] for d in batch]).astype(np.float32)
    mag = _pad_mel([d["mag"] for d in batch]).astype(np.float32)
    return t.FloatTensor(mel), t.FloatTensor(mag)


# =========================
# Padding helpers
# =========================

def _pad_data(x: np.ndarray, length: int):
    """Pad 1D array to `length` with zeros."""
    if x.shape[0] >= length:
        return x
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=0)


def _prepare_data(inputs):
    """Pad list of 1D arrays to the same length and stack."""
    max_len = max(len(x) for x in inputs)
    return np.stack([_pad_data(x, max_len) for x in inputs], axis=0)


def _pad_mel(inputs):
    """Pad list of (T, C) arrays to (B, Tmax, C)."""
    def _pad_one(x, max_len):
        t_len = x.shape[0]
        if t_len >= max_len:
            return x
        return np.pad(x, [[0, max_len - t_len], [0, 0]], mode="constant", constant_values=0)

    max_len = max(x.shape[0] for x in inputs)
    return np.stack([_pad_one(x, max_len) for x in inputs], axis=0)


def _pad_per_step(inputs: np.ndarray):
    """Pad mel/mag timesteps to be multiple of outputs_per_step (if you use it)."""
    timesteps = inputs.shape[-1]
    r = hp.outputs_per_step
    if r <= 1:
        return inputs
    pad = (r - (timesteps % r)) % r
    if pad == 0:
        return inputs
    return np.pad(inputs, [[0, 0], [0, 0], [0, pad]], mode="constant", constant_values=0.0)


# =========================
# Convenience
# =========================

def get_param_size(model) -> int:
    params = 0
    for p in model.parameters():
        n = 1
        for x in p.size():
            n *= x
        params += n
    return int(params)


def get_dataset():
    return LJDatasets(os.path.join(hp.data_path, "metadata.csv"),
                      os.path.join(hp.data_path, "wavs"))


def get_post_dataset():
    return PostDatasets(os.path.join(hp.data_path, "metadata.csv"),
                        os.path.join(hp.data_path, "wavs"))
