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
  --mel2mag_ckpt ./outputs/checkpoints/mel2mag/checkpoint_mel2mag_epoch_0100.pth.tar
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
  --mel2mag_ckpt ./outputs/checkpoints/mel2mag/checkpoint_mel2mag_epoch_0100.pth.tar
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
and reports the fraction of predictions outside `[0, 1]`. It runs at every epoch end. Numbered checkpoints are saved every
`mel2mag_save_every_epoch` completed epochs (default 1), and always at the final
epoch, as `checkpoint_mel2mag_epoch_0001.pth.tar`. `mel2mag_best.pth.tar` and `mel2mag_last.pth.tar`
are retained independently of numbered checkpoint cleanup. Checkpoints include
model/training configuration and the batch position for resume. Resume preserves
data order but does not promise bitwise-identical dropout or augmentation RNG.
New checkpoints store the number of completed epochs; `global_step` is retained
only for optimizer/LR bookkeeping and per-step training logs. Validation logs use
epoch numbers. Set `restore_mel2mag_epoch` in `hyperparams_mel2mag.py` to select an
epoch for synthesis, or pass any checkpoint directly with `--mel2mag_ckpt`.
Older step-named files are still loadable by explicit path; old `restore_step2`
settings are accepted as a fallback. Checkpoint cleanup only removes epoch-named
files and leaves older step-named files untouched.
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

With scheduled sampling disabled, the Transformer architecture and teacher-forcing
forward path remain unchanged.
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

## Two-pass scheduled sampling for Transformer

This implements a mel-frame adaptation of the shared-decoder, stopped-gradient
variant of [Mihaylova and Martins (2019)](https://arxiv.org/abs/1906.07651).
It is an experimental training option; improved speech quality is not guaranteed.
No sequential autoregressive training loop or new third-party dependency is used.

### Forward path

`TransformerTacotron2.forward` still accepts the existing shifted `mel_input` and
`output_lengths`. New optional arguments default to ordinary teacher forcing:

```python
outputs = model(text, text_lengths, mel_input, output_lengths,
                teacher_forcing_ratio=0.5, return_sampling_stats=True)
```

The encoder runs once with its graph intact. When training with ratio below 1,
the same decoder first predicts `mel_before` under `torch.no_grad()`, without gate
projection, attention-weight collection or PostNet. Shifted, detached predictions
replace entire input frames using a `(B, T, 1)` Bernoulli mask. Input position `t`
receives prediction `t-1`; GO and padding positions retain their original values.
The second pass uses the same encoder memory and causal/text/mel padding masks.
PostNet and all existing losses, including guided attention, use the second pass.
The decoder has no BatchNorm; PostNet BatchNorm updates only once per training
forward. `no_grad()` never changes train/eval mode or parameter trainability.

Disabled SS and ratio 1 use one decoder pass. Ratio 0 replaces all eligible
frames, but is still an approximation: first-pass predictions were teacher-forced.
The existing `inference()` algorithm and model state-dict keys are unchanged.

### Configuration and schedules

`hyperparams_transformer.py` contains the default baseline:

```python
scheduled_sampling = {"enabled": False}
freeze = {name: False for name in (
    "text_embedding", "encoder", "decoder_prenet", "decoder",
    "mel_projection", "gate_projection", "postnet",
)}
lr_multipliers = {name: 1.0 for name in freeze}
```

The complete shared fine-tuning configuration is
[`configs/transformer_finetune.json`](configs/transformer_finetune.json). It uses
AdamW, base LR `3.75e-5`, five warmup epochs, and this schedule:

```json
{
  "enabled": true,
  "method": "two_pass",
  "detach_predictions": true,
  "sampling_granularity": "frame",
  "prediction_source": "mel_before_postnet",
  "validation_enabled": true,
  "scheduler": {
    "type": "piecewise_linear",
    "unit": "epoch",
    "points": [[0, 1.0], [5, 1.0], [15, 0.9], [35, 0.5], [60, 0.2]]
  }
}
```

Coordinates are zero-based and relative to the start of the new stage. With 60
training epochs the last coordinate is 59; use 61 epochs to train at coordinate 60.
Schedules support `unit: "epoch"` or `"step"` (completed training-batch count):

| Type | Additional scheduler fields | Behavior |
| --- | --- | --- |
| `constant` | `value` | Fixed ratio |
| `linear` | `start`, `end`, `duration` | Linear interpolation, then hold `end` |
| `exponential` | `start`, `end`, `decay_rate` | `end + (start-end) * decay_rate**position` |
| `inverse_sigmoid` | `start`, `end`, `k` | `end + (start-end)*(k+1)/(k+exp(position/k))` |
| `piecewise_linear` | `points` | Interpolate points; hold final value afterwards |

The inverse sigmoid is normalized to begin at `start`. By default, nonconstant
schedules use `start=1`, `end=0.2`; see the helper for other defaults.

Components are independent: `decoder_prenet` includes its input projection;
`decoder` means the positional encoding and decoder layers, excluding the prenet
and mel/gate heads. Freezing the encoder alone still permits gradients to flow
through it to a trainable embedding. Frozen parameters are excluded from AdamW;
frozen blocks run in eval mode (including frozen prenet's normally active dropout).
Frozen PostNet BatchNorm buffers do not update. LR groups cover every trainable
parameter once; epoch LR scheduling preserves each group's multiplier.

### Commands and ablations

Prepare the same data/features as for normal Transformer training. A fresh baseline
still runs with the existing command:

```bash
python tts_transformer_train.py
```

For controlled fine-tuning comparisons, start every experiment from the same
checkpoint and shared JSON configuration. Replace this example checkpoint path:

```bash
BASE_CKPT=./outputs/checkpoints/transformer/transformer_RUN/transformer_best.pth.tar

python tts_transformer_experiment.py --config configs/transformer_finetune.json \
  --experiment baseline --checkpoint "$BASE_CKPT" --weights-only

python tts_transformer_experiment.py --config configs/transformer_finetune.json \
  --experiment ss2 --checkpoint "$BASE_CKPT" --weights-only
```

| Preset | SS | Encoder | Embedding/encoder LR multipliers |
| --- | --- | --- | --- |
| `baseline` | Off | Trainable | 1 / 1 |
| `ss1` | On | Trainable | 1 / 1 |
| `ss2` | On | Trainable | 0.1 / 0.1 |
| `ss3` | On | Frozen | 1 / excluded |

Use `--experiment ss1` or `ss3` with the same command for the other ablations.
Presets reset freezing/multipliers; omit `--experiment` to use custom JSON values
verbatim. JSON is applied before importing the model, whose existing constants
are read at import time. Run each experiment in its own Python process.

`--weights-only` starts a new stage at epoch/step zero with a fresh optimizer and
relative TF schedule; it is the default when passing `--checkpoint`. Old
single-group AdamW checkpoints require this mode when moving to component groups.
No optimizer state is silently discarded in full-resume mode. Architecture and
model weight keys remain compatible with older checkpoints.

To continue the same stage with its optimizer, TF schedule, split, best score and
RNG states, use the same configuration/preset and its last checkpoint:

```bash
python tts_transformer_experiment.py --config configs/transformer_finetune.json \
  --experiment ss2 --checkpoint ./outputs/checkpoints/transformer/transformer_SS2_RUN/transformer_last.pth.tar \
  --resume --epochs 90
```

`--epochs` is the total length of this stage, including completed epochs on resume.
Changes to freeze settings, LR multipliers or TF schedule require a new stage.
Saved split indices and a metadata fingerprint are checked on resume. Exact
cross-hardware or cross-version numerical reproducibility is not guaranteed.

### Validation, logs and tests

Teacher-forced validation remains the checkpoint selection metric and is also
logged as `val/teacher_forced_loss`. Optional `val/scheduled_sampling_loss` uses
an independently seeded mask in eval mode, without changing the teacher-forced
metric or consuming its training RNG stream. Ratio 1 reuses the TF result.
Autoregressive validation and synthesis never use target mel or two-pass sampling.
Use the existing synthesis script and the same text/vocoder for listening comparisons.

New epoch logs include requested/effective TF ratios, replacement ratio,
`postnet_loss` (an alias of existing `mel_after_loss`), `guided_attention_loss`,
`lr_<component>` and `grad_norm_<component>` (zero for frozen components). Gradient norms
are measured after AMP unscaling and before clipping. Effective ratios count only
eligible frames, excluding GO and padding; their epoch aggregate is frame-weighted.
Existing attention metrics remain; validation also provides `attention_coverage`,
`attention_backward`, and `attention_ar_len` aliases for the autoregressive checks.
Training attention visualizations use second-pass alignments.

```bash
python -m unittest discover -s tests -v
```

Tests cover ratios 0/1, frame-wise masking, shifts, GO/padding, detached first-pass
predictions, shared decoder/memory, encoder/decoder gradients, frozen parameters
and buffers, optimizer coverage, LR multipliers, all five TF schedules, CPU AMP,
causal attention in both positional modes, separated validation, legacy weights,
unchanged inference, and save/resume of the actual trainer. A CUDA FP16 test runs
when a GPU is available. Quality and robustness still require real training and
listening; two-pass validation is not a substitute for autoregressive evaluation.
