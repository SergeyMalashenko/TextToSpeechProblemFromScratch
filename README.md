
# Neural Text-to-Speech System (Tacotron2 + Neural Vocoders)

This repository implements a modular **Text-to-Speech (TTS)** pipeline built around **Tacotron2** and multiple waveform synthesis backends.

The system supports three synthesis pipelines:

1. Legacy pipeline  
Tacotron2 → Mel → MelToMag → Griffin-Lim → Waveform

2. Simple neural vocoder  
Tacotron2 → Mel → Simple Neural Vocoder → Waveform

3. HiFi-GAN vocoder  
Tacotron2 → Mel → HiFi-GAN → Waveform

The project is designed for experimentation and research on **neural speech synthesis pipelines**, especially in scenarios where different vocoders need to be compared or trained independently.

---

# Architecture Overview

Text → Tacotron2 → Mel Spectrogram → (one of vocoders) → Waveform

Paths:

1) Tacotron2 → Mel → MelToMagModel → Griffin-Lim → WAV  
2) Tacotron2 → Mel → Simple Neural Vocoder → WAV  
3) Tacotron2 → Mel → HiFi-GAN → WAV

---

# Repository Structure

project/

hyperparams.py

tts_dataset.py

Tacotron
- tts_rnn_model.py
- tts_rnn_train.py

Mel-to-Magnitude (renamed PostNet)
- tts_mel2mag_model.py
- tts_mel2mag_train.py

Simple Neural Vocoder
- tts_vocoder_model.py
- tts_vocoder_train.py

HiFi-GAN Vocoder
- tts_hifigan_model.py
- tts_hifigan_train.py

Synthesis
- tts_synthesis.py

Utilities
- data_pipeline.py

---

# Core Components

## Tacotron2

Tacotron2 converts **text sequences into mel-spectrograms**.

Files:
- tts_rnn_model.py
- tts_rnn_train.py

Architecture:
Embedding → Encoder → Attention → Decoder → PostNet

Outputs:
- mel_before
- mel_after
- gate
- attention alignment

Loss:
- Mel spectrogram L1 loss
- Gate BCE loss

---

## MelToMagModel (Renamed PostNet)

File:
tts_mel2mag_model.py

Purpose:
Convert mel spectrogram to **linear magnitude spectrogram**.

mel → magnitude spectrogram

Architecture:
- Conv1D layers
- BatchNorm
- Tanh activation
- Dropout

Used only in the **Griffin-Lim pipeline**.

Training:
tts_mel2mag_train.py

Loss:
L1 loss between predicted and target magnitude spectrograms.

---

## Simple Neural Vocoder

Files:
- tts_vocoder_model.py
- tts_vocoder_train.py

Pipeline:
mel → Conv1D stack → waveform

Architecture:
- Conv1D layers
- ReLU activations
- tanh output

Advantages:
- fast training
- minimal architecture
- useful for debugging

Limitations:
- lower audio quality than GAN-based vocoders

---

## HiFi-GAN Vocoder

Files:
- tts_hifigan_model.py
- tts_hifigan_train.py

HiFi-GAN is a **state-of-the-art neural vocoder**.

Input:
mel spectrogram

Output:
raw waveform

Architecture:
- Generator with upsampling layers
- Residual blocks
- Optional adversarial discriminators

Training losses:
- adversarial loss
- feature matching loss
- mel spectrogram reconstruction loss

Advantages:
- very high audio quality
- real-time inference

---

# Dataset

File:
tts_dataset.py

Provides dataset interfaces for different tasks.

Datasets:

TacotronDataset
returns:
- text
- mel spectrogram
- lengths

Mel2MagDataset
returns:
- mel
- linear spectrogram

VocoderDataset
returns:
- mel
- waveform

HiFiGANDataset
returns:
- mel
- waveform segments

Dataset logic includes:
- random audio segment sampling
- mel alignment
- waveform slicing

---

# Synthesis

File:
tts_synthesis.py

Supported backends:

griffinlim  
simple_vocoder  
hifigan

---

## Griffin-Lim Pipeline

Tacotron → mel → MelToMagModel → magnitude → Griffin-Lim → waveform

Run:

python tts_synthesis.py --backend griffinlim

---

## Simple Vocoder Pipeline

Tacotron → mel → SimpleVocoder → waveform

Run:

python tts_synthesis.py --backend simple_vocoder

---

## HiFi-GAN Pipeline

Tacotron → mel → HiFi-GAN → waveform

Run:

python tts_synthesis.py --backend hifigan

---

# Training

Train Tacotron

python tts_rnn_train.py

Train Mel2Mag

python tts_mel2mag_train.py

Train Simple Vocoder

python tts_vocoder_train.py

Train HiFi-GAN

python tts_hifigan_train.py

---

# Hyperparameters

File:
hyperparams.py

Contains configuration for:

- audio processing
- Tacotron architecture
- training parameters
- vocoder parameters
- HiFi-GAN parameters

Example:

n_mels = 80  
sr = 22050  
hop_length = 275  

hifigan_batch_size = 16  
hifigan_lr = 2e-4  

---

# Data Pipeline

File:
data_pipeline.py

Responsibilities:

- audio preprocessing
- mel spectrogram extraction
- dataset preparation
- metadata generation

---

# Extending the System

New vocoders can be easily integrated.

Examples:

WaveGlow  
ParallelWaveGAN  
DiffWave

Steps:
1) implement model  
2) add training script  
3) add dataset interface  
4) connect in tts_synthesis.py

---

# Recommended Workflow

1) Train Tacotron
python tts_rnn_train.py

2) Train a vocoder
python tts_vocoder_train.py
or
python tts_hifigan_train.py

3) Run synthesis
python tts_synthesis.py

---

# Requirements

python ≥ 3.9

Main dependencies:

torch  
numpy  
librosa  
scipy  
tqdm  
matplotlib

Install:

pip install torch librosa numpy scipy tqdm matplotlib

---

# Research Goals

This repository is intended for:

- speech synthesis research
- vocoder comparison
- TTS experimentation
- Tacotron architecture studies

---

# License

This project is intended for research and educational purposes.
