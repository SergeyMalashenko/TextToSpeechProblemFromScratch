# TTS Project Bundle

Research-oriented text-to-speech project for LJSpeech-style data. The repository
contains several acoustic model experiments, feature preparation, vocoder
training, and standalone synthesis scripts.

## Contents

### Acoustic models

The repository contains three Tacotron-style acoustic models. Each model maps
normalized text tokens to mel spectrograms, predicts a stop/gate signal, exposes
attention alignments, and can be used with the same synthesis backends.

| Model | Main idea | Entry points | Configuration |
| --- | --- | --- | --- |
| RNN Tacotron2 | Classic autoregressive Tacotron2 with recurrent decoder and location-sensitive attention. | `tts_rnn_train.py`, `tts_rnn_synthesis.py` | `hyperparams_rnn.py` |
| Transformer Tacotron | Tacotron-style encoder/decoder with separate Transformer decoder layers, SwiGLU FFN, and configurable self-attention positional encoding. | `tts_transformer_train.py`, `tts_transformer_synthesis.py` | `hyperparams_transformer.py` |
| Mamba3 Tacotron-style model | Mamba3-based acoustic model for comparison with the RNN and Transformer paths. | `tts_mamba_train.py`, `tts_mamba_synthesis.py` | `hyperparams_mamba.py` |

### Vocoders and waveform reconstruction

The repository provides three waveform reconstruction paths. They all consume
mel spectrograms, but differ in target representation, quality, speed, and
training complexity.

| Vocoder path | Main idea | Entry points | Configuration |
| --- | --- | --- | --- |
| MelToMag + Griffin-Lim | Predicts magnitude spectrograms from mel spectrograms and reconstructs waveform with Griffin-Lim. | `tts_mel2mag_train.py`; acoustic synthesis scripts with `--backend griffinlim` | `hyperparams_mel2mag.py` |
| HiFi-GAN | Neural GAN vocoder that maps mel spectrograms directly to waveform samples. | `tts_hifigan_train.py`; acoustic synthesis scripts with `--backend hifigan` | `hyperparams_hifigan.py` |
| Diffusion vocoder | DiffWave-like neural vocoder that samples waveform from mel conditioning through a diffusion schedule. | `tts_diffusion_vocoder_train.py`, `tts_diffusion_vocoder_synthesis.py`; acoustic synthesis scripts with `--backend diffusion` | `hyperparams_diffusion_vocoder.py` |

### Shared infrastructure

- Dataset loading and length-aware batching
- Text normalization and symbol processing
- Shared Tacotron-style losses and validation metrics
- Reproducibility helpers from `tts_seed.py`

Current Transformer behavior:

- FFN blocks use `transformer_ffn_type = "swiglu"`.
- Self-attention positional encoding is controlled by
  `transformer_positional_encoding`.
- Supported Transformer positional modes are `sinusoidal` and `rope`.
- The default Transformer positional mode is currently `rope`.
- RoPE is applied only to encoder/decoder self-attention; decoder cross-attention
  stays standard so alignment weights remain available for guided attention,
  metrics, and visualization.

## Configuration Files

- `hyperparams_base.py` - shared audio, text, paths, vocoder settings
- `hyperparams_rnn.py` - RNN Tacotron2 settings
- `hyperparams_transformer.py` - Transformer Tacotron settings
- `hyperparams_mamba.py` - Mamba Tacotron-style settings
- `hyperparams_mel2mag.py` - MelToMag settings
- `hyperparams_hifigan.py` - HiFi-GAN vocoder settings
- `hyperparams_diffusion_vocoder.py` - diffusion vocoder settings

## Data Preparation

Expected input layout:

```text
data/LJSpeech-1.1/metadata.csv
data/LJSpeech-1.1/wavs/*.wav
```

Prepare normalized mel and magnitude features:

```bash
python data_pipeline.py
```

Generated files:

```text
data/LJSpeech-1.1/features/*.mel.npy
data/LJSpeech-1.1/features/*.mag.npy
```

## Training

### RNN Tacotron2

```bash
python tts_rnn_train.py
tensorboard --logdir ./outputs/logs/rnn
```

### Transformer Tacotron

```bash
python tts_transformer_train.py
tensorboard --logdir ./outputs/logs/transformer
```

### Mamba3 Tacotron-Style Model

```bash
python tts_mamba_train.py
tensorboard --logdir ./outputs/logs/mamba
```

RNN, Transformer, and Mamba training scripts use the same external workflow:

```text
outputs/
  checkpoints/{rnn,transformer,mamba,mel2mag,vocoder,hifigan}/
  logs/{rnn,transformer,mamba}/
  samples/{rnn,transformer,mamba}/
  synthesis/{model}_{backend}/
```

All three acoustic trainers print aligned epoch summaries:

```text
[TRAIN epoch=N] loss=... mel=... gate=... attn=... sharp=... melE=... gaw=... lr=...
[VAL epoch=N]   loss=... mel=... gate=... attn=... sharp=... melE=... gaw=... acc=... ar_len=... ar_cov=... ar_back=...
```

### MelToMag

```bash
python tts_mel2mag_train.py
tensorboard --logdir ./logs/mel2mag
```

### HiFi-GAN

```bash
python tts_hifigan_train.py
```

### Diffusion Vocoder

Experimental DiffWave-like vocoder trained on the same normalized mel features
and trimmed + preemphasized waveform target used by the HiFi-GAN variant-A path.

```bash
python tts_diffusion_vocoder_train.py
tensorboard --logdir ./outputs/logs/diffusion_vocoder
```

Standalone synthesis from a saved mel artifact:

```bash
python tts_diffusion_vocoder_synthesis.py \
  --mel_path ./outputs/samples/hifigan/hifigan_mel_epoch_0245.npy \
  --checkpoint ./outputs/checkpoints/diffusion_vocoder/checkpoint_diffusion_vocoder_epoch_0100.pth.tar \
  --out_path ./outputs/synthesis/diffusion_vocoder/sample.wav
```

## Synthesis

The project currently has standalone synthesis scripts for RNN, Transformer,
and Mamba acoustic models. There is no `tts_unified_synthesis.py` entrypoint in
this repository.

All synthesis scripts support these waveform backends:

- `griffinlim` - requires a MelToMag checkpoint
- `hifigan` - requires a HiFi-GAN generator checkpoint
- `diffusion` - requires a diffusion vocoder checkpoint

By default, synthesis artifacts are saved under `outputs/synthesis/{model}_{backend}`,
for example `outputs/synthesis/rnn_hifigan`,
`outputs/synthesis/transformer_griffinlim`, or
`outputs/synthesis/mamba_diffusion`. Passing `--out_dir` overrides this default.

### RNN Tacotron2

```bash
python tts_rnn_synthesis.py \
  --backend griffinlim \
  --text "Hello world" \
  --tacotron_ckpt ./outputs/checkpoints/rnn/checkpoint_rnn_tacotron2_epoch_0100.pth.tar \
  --mel2mag_ckpt ./outputs/checkpoints/mel2mag/checkpoint_mel2mag_50000.pth.tar
```

### Transformer Tacotron

```bash
python tts_transformer_synthesis.py \
  --backend hifigan \
  --text "Hello world" \
  --transformer_ckpt ./outputs/checkpoints/transformer/checkpoint_transformer_tacotron2_epoch_0100.pth.tar \
  --hifigan_ckpt ./outputs/checkpoints/hifigan/checkpoint_hifigan_epoch_0120.pth.tar
```

### Mamba3 Tacotron-Style Model

```bash
python tts_mamba_synthesis.py \
  --backend griffinlim \
  --text "Hello world" \
  --mamba_ckpt ./outputs/checkpoints/mamba/checkpoint_mamba_tacotron2_epoch_0100.pth.tar \
  --mel2mag_ckpt ./outputs/checkpoints/mel2mag/checkpoint_mel2mag_50000.pth.tar
```

License-plate spelling example:

```bash
python tts_mamba_synthesis.py \
  --backend hifigan \
  --text "B374KH50" \
  --spell_plate \
  --mamba_ckpt ./outputs/checkpoints/mamba/checkpoint_mamba_tacotron2_epoch_0100.pth.tar \
  --hifigan_ckpt ./outputs/checkpoints/hifigan/checkpoint_hifigan_epoch_0120.pth.tar
```

Diffusion vocoder example:

```bash
python tts_mamba_synthesis.py \
  --backend diffusion \
  --text "Hello world" \
  --mamba_ckpt ./outputs/checkpoints/mamba/checkpoint_mamba_tacotron2_epoch_0100.pth.tar \
  --diffusion_ckpt ./outputs/checkpoints/diffusion_vocoder/checkpoint_diffusion_vocoder_epoch_0100.pth.tar \
  --diffusion_steps 50
```

## Notes

1. `tts_dataset.py` and `data_pipeline.py` import `hyperparams_base.py`.
2. RNN model/train/synthesis import `hyperparams_rnn.py`.
3. Transformer model/train/synthesis import `hyperparams_transformer.py`.
4. Mamba model/train/synthesis import `hyperparams_mamba.py`.
5. MelToMag model/train import `hyperparams_mel2mag.py`.
6. `tts_tacotron_losses.py` provides the shared Tacotron-style mel, gate, and guided-attention losses.
7. `tts_seed.py` provides shared reproducibility helpers used by all training scripts.

## Current Caveats

- No dependency lockfile is included yet. Install the Python stack used by the
  scripts: PyTorch, NumPy, pandas, librosa, SciPy, tqdm, matplotlib, soundfile,
  Unidecode, inflect, TensorBoard, and optionally `mamba-ssm`.
- Mamba synthesis requires a `mamba-ssm` build that provides Mamba3.
- HiFi-GAN training uses the repository's current experimental vocoder path.
