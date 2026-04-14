#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import librosa
import torch
import copy
import json

from typing  import Any, Dict, List
from pathlib import Path
from scipy   import signal

import hyperparams as hp
import numpy       as np

from scipy.io.wavfile import write
from text import text_to_sequence

#from utils import spectrogram2wav

from tts_rnn_model     import Tacotron2
from tts_postnet_model import ModelPostNet

def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        #est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        est = librosa.stft(
         X_t,
         n_fft=hp.n_fft,
         hop_length=hp.hop_length,
         win_length=hp.win_length,
         window="hann",      # keep consistent with istft
         center=True,        # optional; keep librosa default unless your pipeline expects otherwise
        )
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    #return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")
    return librosa.istft(
        spectrogram,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window="hann",
    )
    





# =============================================================================
# Helpers
# =============================================================================

LETTER_MAP = {
    "A": "alpha",
    "B": "bravo",
    "C": "charlie",
    "E": "echo",
    "H": "hotel",
    "K": "kilo",
    "M": "mike",
    "O": "oscar",
    "P": "papa",
    "T": "tango",
    "X": "x-ray",
    "Y": "yankee",
}

DIGIT_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def hp_get(name: str, default: Any) -> Any:
    return getattr(hp, name, default)


def get_n_mels() -> int:
    if hasattr(hp, "n_mels"):
        return int(hp.n_mels)
    if hasattr(hp, "num_mels"):
        return int(hp.num_mels)
    raise AttributeError("hyperparams must define hp.n_mels or hp.num_mels")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def save_npy(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_spectrogram_png(path: str | Path, spec: np.ndarray, title: str = "") -> None:
    """
    spec: (T, C)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(spec.T, aspect="auto", origin="lower")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Bins")
        if title:
            ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to save PNG {path}: {exc}")


def save_alignment_png(path: str | Path, align: np.ndarray, title: str = "") -> None:
    """
    align: (T_mel, T_text)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(align, aspect="auto", origin="lower")
        ax.set_xlabel("Text positions")
        ax.set_ylabel("Mel frames")
        if title:
            ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Failed to save alignment PNG {path}: {exc}")


def spell_plate(text: str, sep: str = " ") -> str:
    words: List[str] = []
    for ch in text.upper():
        if ch in LETTER_MAP:
            words.append(LETTER_MAP[ch])
        elif ch in DIGIT_MAP:
            words.append(DIGIT_MAP[ch])
    
    print("----------------------------")
    print(words)
    

    return sep.join(words)


def normalize_input_text(text: str, spell_as_plate: bool) -> str:
    if spell_as_plate:
        return spell_plate(text)
    return text


def sanitize_filename(text: str) -> str:
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "sample"


def encode_text(text: str) -> torch.Tensor:
    seq = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
    return torch.from_numpy(seq).long().unsqueeze(0)


def load_checkpoint_state(path: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def resolve_tacotron_checkpoint(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    step = hp_get("restore_step1", None)
    if step is None:
        raise ValueError("Tacotron checkpoint is not provided. Use --tacotron_ckpt.")
    return Path(hp.checkpoint_path) / f"checkpoint_tacotron2_{step}.pth.tar"


def resolve_postnet_checkpoint(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    step = hp_get("restore_step2", None)
    if step is None:
        raise ValueError("Postnet checkpoint is not provided. Use --postnet_ckpt.")
    return Path(hp.checkpoint_path) / f"checkpoint_postnet_{step}.pth.tar"


def load_models(
    tacotron_ckpt: str | Path,
    postnet_ckpt: str | Path,
    device: torch.device,
    strict: bool = True,
) -> tuple[Tacotron2, ModelPostNet]:
    tacotron = Tacotron2().to(device).eval()
    postnet = ModelPostNet().to(device).eval()

    tacotron_sd = load_checkpoint_state(tacotron_ckpt, device=device)
    postnet_sd = load_checkpoint_state(postnet_ckpt, device=device)

    tacotron.load_state_dict(tacotron_sd, strict=strict)
    postnet.load_state_dict(postnet_sd, strict=strict)

    return tacotron, postnet


def apply_wav_postprocessing(
    wav: np.ndarray,
    wav_gain: float = 1.0,
    peak_norm: bool = False,
    peak_target: float = 0.95,
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)

    if wav_gain != 1.0:
        wav = wav * float(wav_gain)

    if peak_norm:
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-8:
            wav = float(peak_target) * wav / peak

    wav = np.clip(wav, -1.0, 1.0)
    return wav.astype(np.float32)


# =============================================================================
# Core synthesis
# =============================================================================

@torch.no_grad()
def synthesize_one(
    source_text: str,
    tacotron: Tacotron2,
    postnet_model: ModelPostNet,
    out_dir: str | Path,
    device: torch.device,
    spell_as_plate: bool = True,
    save_png: bool = False,
    save_alignment: bool = False,
    save_mag_png: bool = False,
    sample_rate: int | None = None,
    wav_gain: float = 1.0,
    peak_norm: bool = False,
    peak_target: float = 0.95,
) -> Dict[str, str]:
    out_dir = Path(out_dir)

    wav_dir = ensure_dir(out_dir / "wav")
    mel_dir = ensure_dir(out_dir / "mel")
    mag_dir = ensure_dir(out_dir / "mag")
    attn_dir = ensure_dir(out_dir / "attn")
    meta_dir = ensure_dir(out_dir / "meta")

    spoken_text = normalize_input_text(source_text, spell_as_plate=spell_as_plate)
    file_stem = sanitize_filename(source_text)

    text_tensor = encode_text(spoken_text).to(device)
    text_lengths = torch.tensor([text_tensor.size(1)], dtype=torch.long, device=device)

    tacotron.eval()
    postnet_model.eval()

    outputs = tacotron.inference(text=text_tensor, text_lengths=text_lengths)

    mel_before = outputs["mel_before"]       # (1, T, n_mels)
    mel_after = outputs["mel_after"]         # (1, T, n_mels)
    gate = outputs["gate"]                   # (1, T_out)
    alignments = outputs["alignments"]       # (1, T_mel, T_text)

    mag_pred = postnet_model(mel_after)      # (1, T, n_freq)

    mel_before_np = to_numpy(mel_before.squeeze(0))
    mel_after_np = to_numpy(mel_after.squeeze(0))
    mag_np = to_numpy(mag_pred.squeeze(0))
    align_np = to_numpy(alignments.squeeze(0))
    gate_np = to_numpy(gate.squeeze(0))

    wav = spectrogram2wav(mag_np)
    wav = apply_wav_postprocessing(
        wav=wav,
        wav_gain=wav_gain,
        peak_norm=peak_norm,
        peak_target=peak_target,
    )

    sr = int(sample_rate if sample_rate is not None else hp.sr)

    wav_path = wav_dir / f"{file_stem}.wav"
    mel_before_path = mel_dir / f"{file_stem}.mel_before.npy"
    mel_after_path = mel_dir / f"{file_stem}.mel_after.npy"
    mag_path = mag_dir / f"{file_stem}.mag.npy"
    align_path = attn_dir / f"{file_stem}.align.npy"
    meta_path = meta_dir / f"{file_stem}.json"

    write(str(wav_path), sr, wav)
    save_npy(mel_before_path, mel_before_np)
    save_npy(mel_after_path, mel_after_np)
    save_npy(mag_path, mag_np)
    save_npy(align_path, align_np)

    if save_png:
        save_spectrogram_png(
            mel_dir / f"{file_stem}.mel_before.png",
            mel_before_np,
            title=f"{source_text} mel_before",
        )
        save_spectrogram_png(
            mel_dir / f"{file_stem}.mel_after.png",
            mel_after_np,
            title=f"{source_text} mel_after",
        )

    if save_mag_png:
        save_spectrogram_png(
            mag_dir / f"{file_stem}.mag.png",
            mag_np,
            title=f"{source_text} magnitude",
        )

    if save_alignment:
        save_alignment_png(
            attn_dir / f"{file_stem}.align.png",
            align_np,
            title=f"{source_text} alignment",
        )

    wav_peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    wav_rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size > 0 else 0.0

    meta = {
        "source_text": source_text,
        "spoken_text": spoken_text,
        "text_length": int(text_tensor.size(1)),
        "generated_frames": int(mel_after_np.shape[0]),
        "n_mels": int(mel_after_np.shape[1]),
        "n_freq": int(mag_np.shape[1]),
        "wav_gain": float(wav_gain),
        "peak_norm": bool(peak_norm),
        "peak_target": float(peak_target),
        "wav_peak": wav_peak,
        "wav_rms": wav_rms,
        "gate_min": float(gate_np.min()) if gate_np.size > 0 else None,
        "gate_max": float(gate_np.max()) if gate_np.size > 0 else None,
        "gate_last": float(gate_np[-1]) if gate_np.size > 0 else None,
        "wav_path": str(wav_path),
        "mel_before_path": str(mel_before_path),
        "mel_after_path": str(mel_after_path),
        "mag_path": str(mag_path),
        "align_path": str(align_path),
    }
    save_json(meta_path, meta)

    return {
        "wav": str(wav_path),
        "mel_before": str(mel_before_path),
        "mel_after": str(mel_after_path),
        "mag": str(mag_path),
        "align": str(align_path),
        "meta": str(meta_path),
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tacotron2 + PostNet inference")

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to synthesize",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Path to text file with one utterance per line",
    )
    parser.add_argument(
        "--spell_plate",
        action="store_true",
        help="Convert input like plate symbols to spoken NATO-style words",
    )

    parser.add_argument(
        "--tacotron_ckpt",
        type=str,
        default=None,
        help="Path to Tacotron2 checkpoint",
    )
    parser.add_argument(
        "--postnet_ckpt",
        type=str,
        default=None,
        help="Path to PostNet checkpoint",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading checkpoints",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=getattr(hp, "sample_path", "./samples"),
        help="Output directory",
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="Save mel spectrogram PNGs",
    )
    parser.add_argument(
        "--save_mag_png",
        action="store_true",
        help="Save magnitude spectrogram PNGs",
    )
    parser.add_argument(
        "--save_alignment",
        action="store_true",
        help="Save alignment PNG",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Optional output sample rate override",
    )

    parser.add_argument(
        "--wav_gain",
        type=float,
        default=1.0,
        help="Linear gain applied to waveform before saving, e.g. 2.0",
    )
    parser.add_argument(
        "--peak_norm",
        action="store_true",
        help="Normalize waveform peak after gain",
    )
    parser.add_argument(
        "--peak_target",
        type=float,
        default=0.95,
        help="Target absolute peak when --peak_norm is enabled",
    )

    return parser.parse_args()


def collect_input_texts(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []

    if args.text:
        texts.append(args.text)

    if args.text_file:
        path = Path(args.text_file)
        if not path.exists():
            raise FileNotFoundError(f"text_file not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        texts.extend([line.strip() for line in lines if line.strip()])

    if not texts:
        raise ValueError("No input text provided. Use --text or --text_file.")

    return texts


def main() -> None:
    args = parse_args()
    device = get_device()

    tacotron_ckpt = resolve_tacotron_checkpoint(args.tacotron_ckpt)
    postnet_ckpt = resolve_postnet_checkpoint(args.postnet_ckpt)

    print(f"Using device       : {device}")
    print(f"Tacotron checkpoint: {tacotron_ckpt}")
    print(f"PostNet checkpoint : {postnet_ckpt}")
    print(f"WAV gain           : {args.wav_gain}")
    print(f"Peak norm          : {args.peak_norm}")
    print(f"Peak target        : {args.peak_target}")

    tacotron, postnet_model = load_models(
        tacotron_ckpt=tacotron_ckpt,
        postnet_ckpt=postnet_ckpt,
        device=device,
        strict=args.strict,
    )

    texts = collect_input_texts(args)

    for text in texts:
        result = synthesize_one(
            source_text=text,
            tacotron=tacotron,
            postnet_model=postnet_model,
            out_dir=args.out_dir,
            device=device,
            spell_as_plate=args.spell_plate,
            save_png=args.save_png,
            save_alignment=args.save_alignment,
            save_mag_png=args.save_mag_png,
            sample_rate=args.sr,
            wav_gain=args.wav_gain,
            peak_norm=args.peak_norm,
            peak_target=args.peak_target,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
