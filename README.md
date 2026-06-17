# TTS Project Bundle

Research-oriented text-to-speech project for LJSpeech-style data. The repository
contains several acoustic model experiments, feature preparation, vocoder
training, and standalone synthesis scripts.

## Contents

- RNN Tacotron2 acoustic model
- Transformer Tacotron acoustic model
- Mamba Tacotron-style acoustic model
- MelToMag model for Griffin-Lim synthesis
- Simple neural vocoder
- HiFi-GAN-style vocoder
- Shared dataset, text normalization, losses, and seeding utilities

The Mamba path is experimental. It requires `mamba-ssm` and is intended for
research comparison rather than as the most reliable synthesis path.

## Configuration Files

- `hyperparams_base.py` - shared audio, text, paths, vocoder settings
- `hyperparams_rnn.py` - RNN Tacotron2 settings
- `hyperparams_transformer.py` - Transformer Tacotron settings
- `hyperparams_mamba.py` - Mamba Tacotron-style settings
- `hyperparams_mel2mag.py` - MelToMag settings

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
tensorboard --logdir ./logs/rnn_tacotron
```

### Transformer Tacotron

```bash
python tts_transformer_train.py
tensorboard --logdir ./logs/transformer_tacotron
```

### Mamba Tacotron-Style Model

```bash
python tts_mamba_train.py
tensorboard --logdir ./logs/mamba_tacotron
```

### MelToMag

```bash
python tts_mel2mag_train.py
tensorboard --logdir ./logs/mel2mag
```

### Simple Vocoder

```bash
python tts_vocoder_train.py
```

### HiFi-GAN

```bash
python tts_hifigan_train.py
```

## Synthesis

The project currently has standalone synthesis scripts for RNN, Transformer,
and Mamba acoustic models. There is no `tts_unified_synthesis.py` entrypoint in
this repository.

All synthesis scripts support these waveform backends:

- `griffinlim` - requires a MelToMag checkpoint
- `simple_vocoder` - requires a simple vocoder checkpoint
- `hifigan` - requires a HiFi-GAN generator checkpoint

### RNN Tacotron2

```bash
python tts_rnn_synthesis.py \
  --backend griffinlim \
  --text "Hello world" \
  --tacotron_ckpt ./checkpoint/checkpoint_tacotron2_100000.pth.tar \
  --mel2mag_ckpt ./checkpoint/checkpoint_mel2mag_50000.pth.tar
```

### Transformer Tacotron

```bash
python tts_transformer_synthesis.py \
  --backend hifigan \
  --text "Hello world" \
  --transformer_ckpt ./checkpoint/checkpoint_transformer_tacotron2_epoch_0100.pth.tar \
  --hifigan_ckpt ./checkpoint_hifigan/checkpoint_hifigan_50000.pth.tar
```

### Mamba Tacotron-Style Model

```bash
python tts_mamba_synthesis.py \
  --backend griffinlim \
  --text "Hello world" \
  --mamba_ckpt ./checkpoint/mamba_tacotron/checkpoint_mamba_tacotron2_epoch_0100.pth.tar \
  --mel2mag_ckpt ./checkpoint/checkpoint_mel2mag_50000.pth.tar
```

License-plate spelling example:

```bash
python tts_mamba_synthesis.py \
  --backend hifigan \
  --text "B374KH50" \
  --spell_plate \
  --mamba_ckpt ./checkpoint/mamba_tacotron/checkpoint_mamba_tacotron2_epoch_0100.pth.tar \
  --hifigan_ckpt ./checkpoint_hifigan/checkpoint_hifigan_50000.pth.tar
```

## Notes

1. `tts_dataset.py` and `data_pipeline.py` import `hyperparams_base.py`.
2. RNN model/train/synthesis import `hyperparams_rnn.py`.
3. Transformer model/train/synthesis import `hyperparams_transformer.py`.
4. Mamba model/train/synthesis import `hyperparams_mamba.py`.
5. MelToMag model/train import `hyperparams_mel2mag.py`.
6. `tts_tacotron_losses.py` provides the shared Tacotron-style mel, gate, and guided-attention losses.
7. `tts_seed.py` provides reproducibility helpers for training scripts.

## Current Caveats

- No dependency lockfile is included yet. Install the Python stack used by the
  scripts: PyTorch, NumPy, pandas, librosa, SciPy, tqdm, matplotlib, soundfile,
  Unidecode, inflect, TensorBoard, and optionally `mamba-ssm`.
- Mamba synthesis quality is experimental and sensitive to train/inference
  mismatch, teacher forcing schedule, and attention formation.
- HiFi-GAN training uses the repository's current experimental vocoder path.
