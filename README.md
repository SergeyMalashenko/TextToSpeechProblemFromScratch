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
tensorboard --logdir ./outputs/logs/mel2mag
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

Replace `transformer_RUN` with the run directory printed during training.

```bash
python tts_transformer_synthesis.py \
  --backend hifigan \
  --text "Hello world" \
  --transformer_ckpt ./outputs/checkpoints/transformer/transformer_RUN/transformer_best.pth.tar \
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

## MelToMag residual model

New training uses 256 channels and eight residual temporal convolutions with
per-frame LayerNorm, dilation `[1, 2, 4, 8, 16, 1, 2, 4]`, and hidden dropout 0.1.
The output projection has no normalization or dropout. The default sigmoid bounds
normalized dB predictions to `[0, 1]`; `mel2mag_output_activation = "linear"`
enables a comparison. Length masks are applied between blocks.

The training loss combines valid-bin L1 with mel consistency and temporal/frequency
difference losses. Each auxiliary weight is configurable in `hyperparams_mel2mag.py`;
set it to zero to disable it. Linear-amplitude spectral convergence and input mel
noise are opt-in. Mel consistency uses the same librosa mel basis and amplitude/dB
conversion as feature preparation. Nonlinear conversions and reductions use FP32.

Validation ranks checkpoints by valid-bin L1, weighted by the number of valid bins,
and reports the fraction of predictions outside `[0, 1]`. It runs at checkpoint
intervals and at every epoch end. `mel2mag_best.pth.tar` and `mel2mag_last.pth.tar`
are retained independently of numbered checkpoint cleanup. Checkpoints include
model/training configuration and the batch position for resume. Resume preserves
data order but does not promise bitwise-identical dropout or augmentation RNG.
Every five epochs the first validation example is saved under
`outputs/samples/mel2mag`, with predicted/target magnitude arrays and Griffin-Lim
WAVs. Set `mel2mag_sample_every_epoch = 0` to disable listening samples.

The new architecture requires fresh training. All three acoustic synthesis scripts
automatically recognize old convolution/BatchNorm checkpoints for inference;
new checkpoints must include `model_config`. Legacy weights cannot resume training
with the residual architecture.

Griffin-Lim now uses `griffinlim_power = 1.0` so inversion does not sharpen the
predicted amplitude spectrum. Set it to `1.2` in `hyperparams_base.py` to compare
legacy reconstruction; existing feature files do not need regeneration.

Run regression checks with `python -m unittest discover -s tests -v` in an environment
with the project dependencies installed. Quality comparisons still require training
on the same split and listening to both true-mel and predicted-mel synthesis.

## Transformer teaching workflow

The Transformer architecture and straightforward training loop are unchanged.
Correctness checks are kept in the existing scripts:

- Mel L1 and guided-attention reductions use FP32 to avoid FP16 overflow under AMP.
- Gate metrics exclude padding and use the synthesis `gate_threshold`.
- `ar_text_coverage` is the fraction of token positions visited by argmax attention.
  `ar_final_position_ratio` separately reports how far the path reached;
  `ar_forward_skip_rate` reports jumps over positions. These are attention-path
  diagnostics, not pronunciation or intelligibility scores.
- Up to `autoregressive_val_samples` fixed validation examples are decoded with a
  local seed. Prenet's existing inference dropout is preserved. Validation mel
  losses are weighted by valid frames; other logged metrics are batch averages.
- The bucket sampler receives the epoch number, so its order changes each epoch.

Each training invocation creates matching run subdirectories under Transformer
logs, checkpoints, and samples. `config.json` and `split.json` are written beside
TensorBoard logs. `transformer_last.pth.tar` is saved every epoch;
`transformer_best.pth.tar` selects the lowest validation `mel_after_loss` in this
invocation, independently of the changing guided-attention weight. The final epoch
is always validated and saved. A resumed invocation gets a separate run directory;
resume is not promised to reproduce dropout RNG bit for bit.

New checkpoints include model configuration, checked against
`hyperparams_transformer.py` during resume and synthesis, allowing changes to
`gate_threshold` and `max_decoder_steps`. Weight keys prefixed with `module.` are
accepted for compatibility with DataParallel checkpoints. Older checkpoints without
configuration remain loadable with matching current hyperparameters. Synthesis
requires matching parameter keys by default; `--allow_partial` explicitly permits
missing/unexpected keys and emits a warning. `--sr` now resamples the waveform.
For example, use `--transformer_ckpt
./outputs/checkpoints/transformer/transformer_RUN/transformer_best.pth.tar` with the
actual run directory printed by training. Alternatively, `restore_transformer_epoch`
selects the matching numbered checkpoint in the checkpoint root or a single run
subdirectory; if multiple runs match, pass `--transformer_ckpt` explicitly.

There is no KV cache, profiler, or additional training framework. Run the small
regression examples with `python -m unittest discover -s tests -v` after installing
the project dependencies.
