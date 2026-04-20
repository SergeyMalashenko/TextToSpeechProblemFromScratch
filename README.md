# TTS Project Bundle

This archive contains a consolidated research-oriented TTS project with:

- RNN Tacotron2 acoustic model
- Transformer Tacotron acoustic model
- MelToMag model for Griffin-Lim synthesis
- Simple neural vocoder
- HiFi-GAN vocoder
- Standalone and unified synthesis tools
- Split hyperparameter configuration files

## Configuration files

- `hyperparams_base.py` — shared audio/text/path/vocoder settings
- `hyperparams_rnn.py` — RNN Tacotron2 settings
- `hyperparams_transformer.py` — Transformer Tacotron settings
- `hyperparams_mel2mag.py` — MelToMag settings

## Main training tools

### 1) Train RNN Tacotron2
```bash
python tts_rnn_train.py
```

Logs:
```bash
tensorboard --logdir ./logs/rnn_tacotron
```

### 2) Train Transformer Tacotron
```bash
python tts_transformer_train.py
```

Logs:
```bash
tensorboard --logdir ./logs/transformer_tacotron
```

### 3) Train MelToMag
```bash
python tts_mel2mag_train.py
```

Logs:
```bash
tensorboard --logdir ./logs/mel2mag
```

### 4) Train simple vocoder
```bash
python tts_vocoder_train.py
```

### 5) Train HiFi-GAN
```bash
python tts_hifigan_train.py
```

## Data preparation

```bash
python data_pipeline.py
```

Expected layout:
- `data/LJSpeech-1.1/metadata.csv`
- `data/LJSpeech-1.1/wavs/*.wav`

Generated features:
- `data/LJSpeech-1.1/features/*.mel.npy`
- `data/LJSpeech-1.1/features/*.mag.npy`

## Synthesis tools

### Legacy / RNN-focused synthesis
```bash
python tts_synthesis.py   --backend griffinlim   --text "Hello world"   --tacotron_ckpt ./checkpoint/checkpoint_tacotron2_100000.pth.tar   --mel2mag_ckpt ./checkpoint/checkpoint_mel2mag_50000.pth.tar
```

### Transformer-only synthesis
```bash
python tts_transformer_synthesis.py   --backend hifigan   --text "Hello world"   --transformer_ckpt ./checkpoint/checkpoint_transformer_tacotron2_100000.pth.tar   --hifigan_ckpt ./checkpoint_hifigan/checkpoint_hifigan_50000.pth.tar
```

### Unified synthesis
```bash
python tts_unified_synthesis.py   --acoustic_model transformer   --backend hifigan   --text "Hello world"   --transformer_ckpt ./checkpoint/checkpoint_transformer_tacotron2_100000.pth.tar   --hifigan_ckpt ./checkpoint_hifigan/checkpoint_hifigan_50000.pth.tar
```

RNN + Griffin-Lim:
```bash
python tts_unified_synthesis.py   --acoustic_model rnn   --backend griffinlim   --text "Hello world"   --rnn_ckpt ./checkpoint/checkpoint_tacotron2_100000.pth.tar   --mel2mag_ckpt ./checkpoint/checkpoint_mel2mag_50000.pth.tar
```

License-plate spelling example:
```bash
python tts_unified_synthesis.py   --acoustic_model transformer   --backend hifigan   --text "B374KH50"   --spell_plate   --transformer_ckpt ./checkpoint/checkpoint_transformer_tacotron2_100000.pth.tar   --hifigan_ckpt ./checkpoint_hifigan/checkpoint_hifigan_50000.pth.tar
```

## Notes

1. `tts_dataset.py` and `data_pipeline.py` import `hyperparams_base.py`.
2. RNN model/train/synthesis import `hyperparams_rnn.py`.
3. Transformer model/train/synthesis import `hyperparams_transformer.py`.
4. MelToMag model/train import `hyperparams_mel2mag.py`.

## Recommended Transformer experiment

If Transformer attention is too diffuse, try:
```python
transformer_encoder_layers = 3
transformer_decoder_layers = 3
transformer_ffn_dim = 512
transformer_dropout = 0.05
guided_attn_weight = 2.0
guided_attn_sigma = 0.2
```
