#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from scipy.io.wavfile import write

import hyperparams_transformer as hp
from text import text_to_sequence

from tts_transformer_model import TransformerTacotron2

MEL2MAG_CLASS = None
SIMPLE_VOCODER_CLASS = None
HIFIGAN_CLASS = None

try:
    from tts_mel2mag_model import MelToMagModel
    MEL2MAG_CLASS = MelToMagModel
except Exception:
    MEL2MAG_CLASS = None

try:
    from tts_vocoder_model import SimpleVocoder
    SIMPLE_VOCODER_CLASS = SimpleVocoder
except Exception:
    SIMPLE_VOCODER_CLASS = None

try:
    from tts_hifigan_model import Generator as HiFiGANGenerator
    HIFIGAN_CLASS = HiFiGANGenerator
except Exception:
    HIFIGAN_CLASS = None


LETTER_MAP = {
    "A": "alpha", "B": "bravo", "C": "charlie", "E": "echo",
    "H": "hotel", "K": "kilo", "M": "mike", "O": "oscar",
    "P": "papa", "T": "tango", "X": "x-ray", "Y": "yankee",
}
DIGIT_MAP = {str(i): w for i, w in enumerate(["zero","one","two","three","four","five","six","seven","eight","nine"])}

def hp_get(name: str, default: Any) -> Any:
    return getattr(hp, name, default)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

def save_npy(path: str | Path, array: np.ndarray) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); np.save(str(path), array)

def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def sanitize_filename(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-","_")) else "_" for ch in text).strip("_") or "sample"

def encode_text(text: str) -> torch.Tensor:
    seq = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
    return torch.from_numpy(seq).long().unsqueeze(0)

def spell_plate(text: str, sep: str = " ") -> str:
    words: List[str] = []
    for ch in text.upper():
        if ch in LETTER_MAP: words.append(LETTER_MAP[ch])
        elif ch in DIGIT_MAP: words.append(DIGIT_MAP[ch])
    return sep.join(words)

def normalize_input_text(text: str, spell_as_plate: bool) -> str:
    return spell_plate(text) if spell_as_plate else text

def save_spectrogram_png(path: str | Path, spec: np.ndarray, title: str = "") -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(spec.T, aspect="auto", origin="lower")
    ax.set_xlabel("Frames"); ax.set_ylabel("Bins")
    if title: ax.set_title(title)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(str(path), dpi=150); plt.close(fig)

def save_alignment_png(path: str | Path, align: np.ndarray, title: str = "") -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(align, aspect="auto", origin="lower")
    ax.set_xlabel("Text positions"); ax.set_ylabel("Mel frames")
    if title: ax.set_title(title)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(str(path), dpi=150); plt.close(fig)

def load_checkpoint_state(path: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."): key = key[len("module."):]
        cleaned[key] = value
    return cleaned

def resolve_transformer_checkpoint(arg_value: str | None) -> Path:
    if arg_value: return Path(arg_value)
    step = hp_get("restore_transformer_step", None)
    if step is None: raise ValueError("Transformer checkpoint is not provided. Use --transformer_ckpt.")
    return Path(hp.checkpoint_path) / f"checkpoint_transformer_tacotron2_{step}.pth.tar"

def resolve_mel2mag_checkpoint(arg_value: str | None) -> Path:
    if arg_value: return Path(arg_value)
    step = hp_get("restore_step2", None)
    if step is None: raise ValueError("Mel2Mag checkpoint is not provided. Use --mel2mag_ckpt.")
    return Path(hp_get("mel2mag_checkpoint_path", "./outputs/checkpoints/mel2mag")) / f"checkpoint_mel2mag_{step}.pth.tar"

def resolve_simple_vocoder_checkpoint(arg_value: str | None) -> Path:
    if arg_value: return Path(arg_value)
    step = hp_get("restore_simple_vocoder_step", hp_get("restore_vocoder_step", None))
    if step is None: raise ValueError("Simple vocoder checkpoint is not provided. Use --vocoder_ckpt.")
    return Path(hp_get("simple_vocoder_checkpoint_path", "./outputs/checkpoints/vocoder")) / f"checkpoint_vocoder_{step}.pth"

def resolve_hifigan_checkpoint(arg_value: str | None) -> Path:
    if arg_value: return Path(arg_value)
    epoch = hp_get("restore_hifigan_epoch", None)
    if epoch is None: raise ValueError("HiFi-GAN checkpoint is not provided. Use --hifigan_ckpt.")
    return Path(hp_get("hifigan_checkpoint_path", "./outputs/checkpoints/hifigan")) / f"checkpoint_hifigan_epoch_{int(epoch):04d}.pth.tar"

def load_transformer_model(transformer_ckpt: str | Path, device: torch.device, strict: bool = True) -> TransformerTacotron2:
    model = TransformerTacotron2().to(device).eval()
    sd = load_checkpoint_state(transformer_ckpt, device=device)
    model.load_state_dict(sd, strict=strict)
    return model

def load_mel2mag_model(mel2mag_ckpt: str | Path, device: torch.device, strict: bool = True):
    if MEL2MAG_CLASS is None: raise ImportError("MelToMag model implementation was not found.")
    model = MEL2MAG_CLASS().to(device).eval()
    sd = load_checkpoint_state(mel2mag_ckpt, device=device)
    model.load_state_dict(sd, strict=strict)
    return model

def load_simple_vocoder(ckpt_path: str | Path, device: torch.device, strict: bool = True):
    if SIMPLE_VOCODER_CLASS is None: raise ImportError("SimpleVocoder implementation was not found.")
    model = SIMPLE_VOCODER_CLASS().to(device).eval()
    sd = load_checkpoint_state(ckpt_path, device=device)
    model.load_state_dict(sd, strict=strict)
    return model

def load_hifigan_generator(ckpt_path: str | Path, device: torch.device, strict: bool = True):
    if HIFIGAN_CLASS is None: raise ImportError("HiFi-GAN Generator implementation was not found.")
    state = torch.load(Path(ckpt_path), map_location=device)
    sd = state["generator"] if isinstance(state, dict) and "generator" in state else state
    cleaned = {}
    for key, value in sd.items():
        if key.startswith("module."): key = key[len("module."):]
        cleaned[key] = value
    model = HIFIGAN_CLASS().to(device).eval()
    model.load_state_dict(cleaned, strict=strict)
    if hasattr(model, "remove_weight_norm"):
        try: model.remove_weight_norm()
        except Exception: pass
    return model

def invert_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    return librosa.istft(spectrogram, hop_length=hp.hop_length, win_length=hp.win_length, window="hann")

def griffin_lim(spectrogram: np.ndarray) -> np.ndarray:
    x_best = copy.deepcopy(spectrogram)
    for _ in range(hp.n_iter):
        x_t = invert_spectrogram(x_best)
        est = librosa.stft(x_t, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length, window="hann", center=True)
        phase = est / np.maximum(1e-8, np.abs(est))
        x_best = spectrogram * phase
    return np.real(invert_spectrogram(x_best))

def spectrogram2wav(mag: np.ndarray) -> np.ndarray:
    mag = mag.T
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
    mag = np.power(10.0, mag * 0.05)
    wav = griffin_lim(mag ** hp.power)
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    #wav, _ = librosa.effects.trim(wav)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)

def apply_wav_postprocessing(wav: np.ndarray, wav_gain: float = 1.0, peak_norm: bool = False, peak_target: float = 0.95) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav_gain != 1.0: wav = wav * float(wav_gain)
    if peak_norm:
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-8: wav = float(peak_target) * wav / peak
    return np.clip(wav, -1.0, 1.0).astype(np.float32)

def collect_input_texts(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.text is not None: texts.append(args.text)
    if args.text_file is not None:
        lines = Path(args.text_file).read_text(encoding="utf-8").splitlines()
        texts.extend([line.strip() for line in lines if line.strip()])
    if not texts: raise ValueError("No input text provided. Use --text or --text_file.")
    return texts

@torch.no_grad()
def synthesize_one(source_text: str, transformer_model: TransformerTacotron2, out_dir: str | Path, device: torch.device,
                   backend: str = "griffinlim", mel2mag_model=None, simple_vocoder=None, hifigan_model=None,
                   spell_as_plate: bool = False, save_png: bool = False, save_alignment: bool = False,
                   save_mag_png: bool = False, sample_rate: Optional[int] = None, wav_gain: float = 1.0,
                   peak_norm: bool = False, peak_target: float = 0.95) -> Dict[str, str]:
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

    outputs = transformer_model.inference(text=text_tensor, text_lengths=text_lengths)
    mel_before = outputs["mel_before"]; mel_after = outputs["mel_after"]; gate = outputs["gate"]; alignments = outputs["alignments"]
    mel_before_np = to_numpy(mel_before.squeeze(0)); mel_after_np = to_numpy(mel_after.squeeze(0))
    gate_np = to_numpy(gate.squeeze(0)); align_np = to_numpy(alignments.squeeze(0))
    mag_np = None

    if backend == "griffinlim":
        if mel2mag_model is None: raise ValueError("mel2mag_model is required for griffinlim backend")
        mag_pred = mel2mag_model(mel_after)
        mag_np = to_numpy(mag_pred.squeeze(0))
        wav = spectrogram2wav(mag_np)
    elif backend == "simple_vocoder":
        if simple_vocoder is None: raise ValueError("simple_vocoder is required for simple_vocoder backend")
        wav = to_numpy(simple_vocoder(mel_after).squeeze(0))
    elif backend == "hifigan":
        if hifigan_model is None: raise ValueError("hifigan_model is required for hifigan backend")
        wav = to_numpy(hifigan_model(mel_after.transpose(1, 2)).squeeze(0).squeeze(0))
        wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    wav = apply_wav_postprocessing(wav, wav_gain, peak_norm, peak_target)
    sr = int(sample_rate if sample_rate is not None else hp.sr)

    wav_path = wav_dir / f"{file_stem}.wav"
    mel_before_path = mel_dir / f"{file_stem}.mel_before.npy"
    mel_after_path = mel_dir / f"{file_stem}.mel_after.npy"
    align_path = attn_dir / f"{file_stem}.align.npy"
    gate_path = attn_dir / f"{file_stem}.gate.npy"
    meta_path = meta_dir / f"{file_stem}.json"

    write(str(wav_path), sr, (wav * 32767.0).astype(np.int16))
    save_npy(mel_before_path, mel_before_np); save_npy(mel_after_path, mel_after_np); save_npy(align_path, align_np); save_npy(gate_path, gate_np)

    mag_path = None
    if mag_np is not None:
        mag_path = mag_dir / f"{file_stem}.mag.npy"; save_npy(mag_path, mag_np)

    if save_png:
        save_spectrogram_png(mel_dir / f"{file_stem}.mel_after.png", mel_after_np, title=f"{source_text} mel_after")
        save_spectrogram_png(mel_dir / f"{file_stem}.mel_before.png", mel_before_np, title=f"{source_text} mel_before")
    if save_alignment:
        save_alignment_png(attn_dir / f"{file_stem}.align.png", align_np, title=f"{source_text} alignment")
    if save_mag_png and mag_np is not None:
        save_spectrogram_png(mag_dir / f"{file_stem}.mag.png", mag_np, title=f"{source_text} magnitude")

    meta = {
        "source_text": source_text, "spoken_text": spoken_text, "backend": backend, "sample_rate": sr,
        "n_mels": int(mel_after_np.shape[1]), "n_frames": int(mel_after_np.shape[0]),
        "n_freq": int(mag_np.shape[1]) if mag_np is not None else None,
        "wav_gain": float(wav_gain), "peak_norm": bool(peak_norm), "peak_target": float(peak_target),
    }
    save_json(meta_path, meta)

    return {
        "wav_path": str(wav_path), "mel_before_path": str(mel_before_path), "mel_after_path": str(mel_after_path),
        "align_path": str(align_path), "gate_path": str(gate_path), "mag_path": str(mag_path) if mag_path is not None else "",
        "meta_path": str(meta_path),
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech synthesis using Transformer Tacotron")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--backend", type=str, default="griffinlim", choices=["griffinlim", "simple_vocoder", "hifigan"])
    parser.add_argument("--transformer_ckpt", type=str, default=None)
    parser.add_argument("--mel2mag_ckpt", type=str, default=None)
    parser.add_argument("--vocoder_ckpt", type=str, default=None)
    parser.add_argument("--hifigan_ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=getattr(hp, "synthesis_path", getattr(hp, "sample_path", "./outputs/synthesis/transformer")))
    parser.add_argument("--spell_plate", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--save_alignment", action="store_true")
    parser.add_argument("--save_mag_png", action="store_true")
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--wav_gain", type=float, default=1.0)
    parser.add_argument("--peak_norm", action="store_true")
    parser.add_argument("--peak_target", type=float, default=0.95)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = get_device()
    transformer_ckpt = resolve_transformer_checkpoint(args.transformer_ckpt)
    print(f"Using device          : {device}")
    print(f"Transformer checkpoint: {transformer_ckpt}")
    print(f"Backend               : {args.backend}")

    transformer_model = load_transformer_model(transformer_ckpt, device, args.strict)

    mel2mag_model = None; simple_vocoder = None; hifigan_model = None
    if args.backend == "griffinlim":
        mel2mag_ckpt = resolve_mel2mag_checkpoint(args.mel2mag_ckpt)
        print(f"Mel2Mag checkpoint    : {mel2mag_ckpt}")
        mel2mag_model = load_mel2mag_model(mel2mag_ckpt, device, args.strict)
    elif args.backend == "simple_vocoder":
        vocoder_ckpt = resolve_simple_vocoder_checkpoint(args.vocoder_ckpt)
        print(f"Simple vocoder ckpt   : {vocoder_ckpt}")
        simple_vocoder = load_simple_vocoder(vocoder_ckpt, device, args.strict)
    elif args.backend == "hifigan":
        hifigan_ckpt = resolve_hifigan_checkpoint(args.hifigan_ckpt)
        print(f"HiFi-GAN checkpoint   : {hifigan_ckpt}")
        hifigan_model = load_hifigan_generator(hifigan_ckpt, device, args.strict)

    texts = collect_input_texts(args)
    for text in texts:
        result = synthesize_one(text, transformer_model, args.out_dir, device, args.backend, mel2mag_model, simple_vocoder, hifigan_model,
                                args.spell_plate, args.save_png, args.save_alignment, args.save_mag_png, args.sr, args.wav_gain,
                                args.peak_norm, args.peak_target)
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
