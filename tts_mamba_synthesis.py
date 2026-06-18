#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from scipy import signal
from scipy.io.wavfile import write

import hyperparams_mamba as hp
from text import text_to_sequence

from tts_hifigan_model import Generator as HiFiGANGenerator
from tts_mel2mag_model import MelToMagModel
from tts_mamba_model import MambaTacotron2
from tts_vocoder_model import SimpleVocoder


def spectrogram2wav(mag: np.ndarray) -> np.ndarray:
    mag = mag.T
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db
    mag = np.power(10.0, mag * 0.05)
    wav = griffin_lim(mag ** hp.power)
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


def griffin_lim(spectrogram: np.ndarray) -> np.ndarray:
    x_best = copy.deepcopy(spectrogram)
    for _ in range(hp.n_iter):
        x_t = invert_spectrogram(x_best)
        est = librosa.stft(
            x_t,
            n_fft=hp.n_fft,
            hop_length=hp.hop_length,
            win_length=hp.win_length,
            window="hann",
            center=True,
        )
        phase = est / np.maximum(1e-8, np.abs(est))
        x_best = spectrogram * phase
    x_t = invert_spectrogram(x_best)
    return np.real(x_t)


def invert_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    return librosa.istft(
        spectrogram,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window="hann",
    )


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


def load_checkpoint_state(path: str | Path, device: torch.device, key: str | None = "model") -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    if key is not None and isinstance(state, dict) and key in state:
        state_dict = state[key]
    else:
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def resolve_mamba_checkpoint(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)

    restore_epoch = hp_get("restore_mamba_epoch", hp_get("restore_epoch1", None))
    if restore_epoch is not None:
        return Path(hp.checkpoint_path) / f"checkpoint_mamba_tacotron2_epoch_{int(restore_epoch):04d}.pth.tar"

    raise ValueError("Mamba checkpoint is not provided. Use --mamba_ckpt or set restore_mamba_epoch / restore_epoch1.")


def resolve_mel2mag_checkpoint(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    step = hp_get("restore_step2", None)
    if step is None:
        raise ValueError("MelToMag checkpoint is not provided. Use --mel2mag_ckpt.")
    checkpoint_dir = Path(hp_get("mel2mag_checkpoint_path", "./checkpoint"))
    return checkpoint_dir / f"checkpoint_mel2mag_{step}.pth.tar"


def resolve_vocoder_checkpoint(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    step = hp_get("restore_simple_vocoder_step", hp_get("restore_vocoder_step", None))
    if step is None:
        raise ValueError("Simple vocoder checkpoint is not provided. Use --vocoder_ckpt.")
    return Path(hp_get("simple_vocoder_checkpoint_path", "./checkpoint_vocoder")) / f"checkpoint_vocoder_{step}.pth.tar"


def resolve_hifigan_checkpoint(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    step = hp_get("restore_hifigan_step", None)
    if step is None:
        raise ValueError("HiFi-GAN checkpoint is not provided. Use --hifigan_ckpt.")
    return Path(hp_get("hifigan_checkpoint_path", "./checkpoint_hifigan")) / f"checkpoint_hifigan_{step}.pth.tar"


def load_mamba(path: str | Path, device: torch.device, strict: bool) -> MambaTacotron2:
    model = MambaTacotron2().to(device).eval()
    model.load_state_dict(load_checkpoint_state(path, device=device), strict=strict)
    return model


def load_mel2mag(path: str | Path, device: torch.device, strict: bool) -> MelToMagModel:
    model = MelToMagModel().to(device).eval()
    model.load_state_dict(load_checkpoint_state(path, device=device), strict=strict)
    return model


def load_simple_vocoder(path: str | Path, device: torch.device, strict: bool) -> SimpleVocoder:
    model = SimpleVocoder().to(device).eval()
    model.load_state_dict(load_checkpoint_state(path, device=device), strict=strict)
    return model


def load_hifigan(path: str | Path, device: torch.device, strict: bool) -> HiFiGANGenerator:
    model = HiFiGANGenerator().to(device).eval()
    model.load_state_dict(load_checkpoint_state(path, device=device, key="generator"), strict=strict)
    model.remove_weight_norm()
    return model


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


@torch.no_grad()
def synthesize_one(
    source_text: str,
    mamba: MambaTacotron2,
    out_dir: str | Path,
    device: torch.device,
    backend: str = "griffinlim",
    mel2mag_model: MelToMagModel | None = None,
    vocoder_model: SimpleVocoder | None = None,
    hifigan_model: HiFiGANGenerator | None = None,
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

    mamba.eval()
    outputs = mamba.inference(text=text_tensor, text_lengths=text_lengths)

    mel_before = outputs["mel_before"]
    mel_after = outputs["mel_after"]
    gate = outputs["gate"]
    alignments = outputs["alignments"]

    mel_before_np = to_numpy(mel_before.squeeze(0))
    mel_after_np = to_numpy(mel_after.squeeze(0))
    align_np = to_numpy(alignments.squeeze(0))
    gate_np = to_numpy(gate.squeeze(0))

    mag_np = None
    if backend == "griffinlim":
        if mel2mag_model is None:
            raise ValueError("mel2mag_model is required for griffinlim backend")
        mel2mag_model.eval()
        mag_pred = mel2mag_model(mel_after)
        mag_np = to_numpy(mag_pred.squeeze(0))
        wav = spectrogram2wav(mag_np)

    elif backend == "simple_vocoder":
        if vocoder_model is None:
            raise ValueError("vocoder_model is required for simple_vocoder backend")
        vocoder_model.eval()
        wav_t = vocoder_model(mel_after)
        wav = to_numpy(wav_t.squeeze(0).squeeze(0))

    elif backend == "hifigan":
        if hifigan_model is None:
            raise ValueError("hifigan_model is required for hifigan backend")
        hifigan_model.eval()
        wav_t = hifigan_model(mel_after.transpose(1, 2))
        wav = to_numpy(wav_t.squeeze(0).squeeze(0))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    wav = apply_wav_postprocessing(wav, wav_gain=wav_gain, peak_norm=peak_norm, peak_target=peak_target)
    sr = int(sample_rate if sample_rate is not None else hp.sr)

    wav_path = wav_dir / f"{file_stem}.wav"
    mel_before_path = mel_dir / f"{file_stem}.mel_before.npy"
    mel_after_path = mel_dir / f"{file_stem}.mel_after.npy"
    align_path = attn_dir / f"{file_stem}.align.npy"
    meta_path = meta_dir / f"{file_stem}.json"

    write(str(wav_path), sr, wav)
    save_npy(mel_before_path, mel_before_np)
    save_npy(mel_after_path, mel_after_np)
    save_npy(align_path, align_np)

    mag_path = None
    if mag_np is not None:
        mag_path = mag_dir / f"{file_stem}.mag.npy"
        save_npy(mag_path, mag_np)

    if save_png:
        save_spectrogram_png(mel_dir / f"{file_stem}.mel_before.png", mel_before_np, title=f"{source_text} mel_before")
        save_spectrogram_png(mel_dir / f"{file_stem}.mel_after.png", mel_after_np, title=f"{source_text} mel_after")

    if save_mag_png and mag_np is not None:
        save_spectrogram_png(mag_dir / f"{file_stem}.mag.png", mag_np, title=f"{source_text} magnitude")

    if save_alignment:
        save_alignment_png(attn_dir / f"{file_stem}.align.png", align_np, title=f"{source_text} alignment")

    wav_peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    wav_rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size > 0 else 0.0

    meta = {
        "source_text": source_text,
        "spoken_text": spoken_text,
        "backend": backend,
        "text_length": int(text_tensor.size(1)),
        "generated_frames": int(mel_after_np.shape[0]),
        "n_mels": int(mel_after_np.shape[1]),
        "n_freq": int(mag_np.shape[1]) if mag_np is not None else None,
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
        "mag_path": str(mag_path) if mag_path is not None else None,
        "align_path": str(align_path),
    }
    save_json(meta_path, meta)

    return {
        "wav": str(wav_path),
        "mel_before": str(mel_before_path),
        "mel_after": str(mel_after_path),
        "mag": str(mag_path) if mag_path is not None else "",
        "align": str(align_path),
        "meta": str(meta_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MambaTacotron2 synthesis with epoch-based checkpoint naming")

    parser.add_argument("--text", type=str, default=None, help="Single text to synthesize")
    parser.add_argument("--text_file", type=str, default=None, help="Path to text file with one utterance per line")
    parser.add_argument("--spell_plate", action="store_true", help="Convert input like plate symbols to spoken NATO-style words")

    parser.add_argument("--backend", type=str, default="griffinlim", choices=["griffinlim", "simple_vocoder", "hifigan"], help="Waveform backend")
    parser.add_argument("--mamba_ckpt", type=str, default=None, help="Path to MambaTacotron2 checkpoint")
    parser.add_argument("--mel2mag_ckpt", type=str, default=None, help="Path to MelToMag checkpoint")
    parser.add_argument("--vocoder_ckpt", type=str, default=None, help="Path to simple neural vocoder checkpoint")
    parser.add_argument("--hifigan_ckpt", type=str, default=None, help="Path to HiFi-GAN checkpoint")

    parser.add_argument("--strict", action="store_true", help="Use strict=True when loading checkpoints")
    parser.add_argument("--out_dir", type=str, default=getattr(hp, "sample_path", "./samples"), help="Output directory")
    parser.add_argument("--save_png", action="store_true", help="Save mel spectrogram PNGs")
    parser.add_argument("--save_mag_png", action="store_true", help="Save magnitude spectrogram PNGs")
    parser.add_argument("--save_alignment", action="store_true", help="Save alignment PNG")
    parser.add_argument("--sr", type=int, default=None, help="Optional output sample rate override")
    parser.add_argument("--wav_gain", type=float, default=1.0, help="Linear gain applied to waveform before saving")
    parser.add_argument("--peak_norm", action="store_true", help="Normalize waveform peak after gain")
    parser.add_argument("--peak_target", type=float, default=0.95, help="Target absolute peak when --peak_norm is enabled")

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

    mamba_ckpt = resolve_mamba_checkpoint(args.mamba_ckpt)

    print(f"Using device       : {device}")
    print(f"Backend            : {args.backend}")
    print(f"Mamba checkpoint   : {mamba_ckpt}")
    print(f"WAV gain           : {args.wav_gain}")
    print(f"Peak norm          : {args.peak_norm}")
    print(f"Peak target        : {args.peak_target}")

    mamba = load_mamba(mamba_ckpt, device=device, strict=args.strict)

    mel2mag_model = None
    vocoder_model = None
    hifigan_model = None

    if args.backend == "griffinlim":
        mel2mag_ckpt = resolve_mel2mag_checkpoint(args.mel2mag_ckpt)
        print(f"MelToMag checkpoint: {mel2mag_ckpt}")
        mel2mag_model = load_mel2mag(mel2mag_ckpt, device=device, strict=args.strict)

    elif args.backend == "simple_vocoder":
        vocoder_ckpt = resolve_vocoder_checkpoint(args.vocoder_ckpt)
        print(f"Simple vocoder ckpt: {vocoder_ckpt}")
        vocoder_model = load_simple_vocoder(vocoder_ckpt, device=device, strict=args.strict)

    elif args.backend == "hifigan":
        hifigan_ckpt = resolve_hifigan_checkpoint(args.hifigan_ckpt)
        print(f"HiFi-GAN checkpoint: {hifigan_ckpt}")
        hifigan_model = load_hifigan(hifigan_ckpt, device=device, strict=args.strict)

    texts = collect_input_texts(args)

    for text in texts:
        result = synthesize_one(
            source_text=text,
            mamba=mamba,
            out_dir=args.out_dir,
            device=device,
            backend=args.backend,
            mel2mag_model=mel2mag_model,
            vocoder_model=vocoder_model,
            hifigan_model=hifigan_model,
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
