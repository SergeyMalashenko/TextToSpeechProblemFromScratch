#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent Mamba-based Tacotron-style model.

The model intentionally does not inherit from the existing RNN Tacotron2 class.
It keeps the same training I/O contract used by tts_rnn_train.py:

    forward(text, text_lengths, mel_input, output_lengths=None) -> {
        "mel_before": (B, T_mel, n_mels),
        "mel_after" : (B, T_mel, n_mels),
        "gate"      : (B, T_mel),
        "alignments": (B, T_mel, T_text),
    }

It requires mamba-ssm to be installed and fails explicitly otherwise.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import hyperparams_mamba as hp
except Exception:  # pragma: no cover
    hp = None


def hp_get(name: str, default):
    return getattr(hp, name, default) if hp is not None else default


from mamba_ssm import Mamba


class LengthMask:
    @staticmethod
    def make(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        if max_len is None:
            max_len = int(lengths.max().item())
        ids = torch.arange(max_len, device=lengths.device)
        return ids.unsqueeze(0) < lengths.unsqueeze(1)


class Prenet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tacotron-style prenet dropout is useful during training. During eval we
        # use standard deterministic behavior.
        for layer in self.layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sequence = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.sequence(x)
        x = self.dropout(x)
        return residual + x


class MambaStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device, dtype=torch.float32).unsqueeze(1)
        even_dims = torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
        angles = positions * torch.exp(-torch.log(x.new_tensor(10000.0).float()) * even_dims / self.d_model)

        encoding = torch.zeros(x.size(1), self.d_model, device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angles)
        if self.d_model > 1:
            encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].size(1)])
        return x + encoding.to(dtype=x.dtype).unsqueeze(0)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _forward_attention(
        energies: torch.Tensor,
        valid_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert content attention into a monotonic forward path.

        At every decoder step attention may stay on the current text position
        or advance by one position. It cannot jump backwards.
        """
        content_weights = torch.softmax(
            energies.masked_fill(~valid_memory[:, None, None, :], torch.finfo(energies.dtype).min),
            dim=-1,
        )

        B, H, T_mel, T_text = content_weights.shape
        previous = content_weights.new_zeros(B, H, T_text)
        previous[:, :, 0] = 1.0
        alignments = []

        for t in range(T_mel):
            stay_or_advance = previous + F.pad(previous[:, :, :-1], (1, 0))
            current = content_weights[:, :, t, :] * stay_or_advance
            current = current * valid_memory[:, None, :].to(current.dtype)
            current = current / current.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(current.dtype).tiny)
            alignments.append(current)
            previous = current

        return torch.stack(alignments, dim=2)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query:  (B, T_mel, D)
        # memory: (B, T_text, D)
        B, T_mel, D = query.shape
        T_text = memory.size(1)
        H = self.attn.num_heads
        head_dim = D // H

        q_weight, k_weight, v_weight = self.attn.in_proj_weight.chunk(3, dim=0)
        if self.attn.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.attn.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, q_weight, q_bias).view(B, T_mel, H, head_dim).transpose(1, 2)
        k = F.linear(memory, k_weight, k_bias).view(B, T_text, H, head_dim).transpose(1, 2)
        v = F.linear(memory, v_weight, v_bias).view(B, T_text, H, head_dim).transpose(1, 2)

        energies = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        valid_memory = LengthMask.make(memory_lengths, max_len=T_text)
        head_weights = self._forward_attention(energies, valid_memory)
        context = torch.matmul(head_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T_mel, D)
        context = self.attn.out_proj(context)

        x = self.norm(query + self.dropout(context))
        weights = head_weights.mean(dim=1)
        return x, weights


class Postnet(nn.Module):
    def __init__(self, n_mels: int, channels: int = 512, kernel_size: int = 5, n_layers: int = 5, dropout: float = 0.5) -> None:
        super().__init__()
        layers = []
        padding = (kernel_size - 1) // 2
        in_ch = n_mels
        for i in range(n_layers):
            out_ch = channels if i < n_layers - 1 else n_mels
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding))
            if i < n_layers - 1:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, T, n_mels)
        residual = self.net(mel.transpose(1, 2)).transpose(1, 2)
        return mel + residual


class MambaTacotron2(nn.Module):
    def __init__(
        self,
        n_symbols: Optional[int] = None,
        n_mels: Optional[int] = None,
        d_model: Optional[int] = None,
        encoder_layers: Optional[int] = None,
        decoder_layers: Optional[int] = None,
        d_state: Optional[int] = None,
        d_conv: Optional[int] = None,
        expand: Optional[int] = None,
        n_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        max_decoder_steps: Optional[int] = None,
        gate_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.n_symbols = int(n_symbols if n_symbols is not None else hp_get("n_symbols", hp_get("vocab_size", 128)))
        self.n_mels = int(n_mels if n_mels is not None else hp_get("n_mels", hp_get("num_mels", 80)))
        self.d_model = int(d_model if d_model is not None else hp_get("mamba_d_model", 256))
        self.max_decoder_steps = int(max_decoder_steps if max_decoder_steps is not None else hp_get("max_decoder_steps", 1000))
        self.gate_threshold = float(gate_threshold if gate_threshold is not None else hp_get("gate_threshold", 0.5))

        enc_layers = int(encoder_layers if encoder_layers is not None else hp_get("mamba_encoder_layers", 4))
        dec_layers = int(decoder_layers if decoder_layers is not None else hp_get("mamba_decoder_layers", 4))
        d_state = int(d_state if d_state is not None else hp_get("mamba_d_state", 16))
        d_conv = int(d_conv if d_conv is not None else hp_get("mamba_d_conv", 4))
        expand = int(expand if expand is not None else hp_get("mamba_expand", 2))
        n_heads = int(n_heads if n_heads is not None else hp_get("mamba_attention_heads", 4))
        dropout = float(dropout if dropout is not None else hp_get("mamba_dropout", 0.1))

        self.embedding = nn.Embedding(self.n_symbols, self.d_model, padding_idx=0)
        self.text_position = SinusoidalPositionEncoding(self.d_model)
        self.mel_position = SinusoidalPositionEncoding(self.d_model)
        self.encoder = MambaStack(
            d_model=self.d_model,
            n_layers=enc_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.prenet = Prenet(
            in_dim=self.n_mels,
            hidden_dim=int(hp_get("mamba_prenet_hidden", 256)),
            out_dim=self.d_model,
            dropout=float(hp_get("mamba_prenet_dropout", 0.5)),
        )
        self.decoder = MambaStack(
            d_model=self.d_model,
            n_layers=dec_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.cross_attn = CrossAttention(self.d_model, n_heads=n_heads, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.mel_proj = nn.Linear(self.d_model, self.n_mels)
        self.gate_proj = nn.Linear(self.d_model, 1)
        self.postnet = Postnet(
            n_mels=self.n_mels,
            channels=int(hp_get("postnet_channels", 512)),
            kernel_size=int(hp_get("postnet_kernel_size", 5)),
            n_layers=int(hp_get("postnet_layers", 5)),
            dropout=float(hp_get("postnet_dropout", 0.5)),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()
        nn.init.xavier_uniform_(self.mel_proj.weight)
        nn.init.zeros_(self.mel_proj.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -3.0)

    def encode(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(text)
        x = self.text_position(x)
        x = self.encoder(x)
        mask = LengthMask.make(text_lengths, max_len=x.size(1)).unsqueeze(-1).to(x.dtype)
        return x * mask

    def decode_teacher_forced(
        self,
        memory: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mel_before, gate, alignments = self.decode_sequence(
            memory=memory,
            text_lengths=text_lengths,
            mel_input=mel_input,
        )
        mel_after = self.postnet(mel_before)
        return {
            "mel_before": mel_before,
            "mel_after": mel_after,
            "gate": gate,
            "alignments": alignments,
        }

    def decode_sequence(
        self,
        memory: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a mel prefix without running the non-causal postnet."""
        x = self.prenet(mel_input)
        x = self.mel_position(x)
        x = self.decoder(x)
        x, alignments = self.cross_attn(x, memory, text_lengths)
        x = self.fusion(x)
        mel_before = self.mel_proj(x)
        gate = self.gate_proj(x).squeeze(-1)
        return mel_before, gate, alignments

    def build_scheduled_mel_input(
        self,
        memory: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_input: torch.Tensor,
        teacher_forcing_ratio: float,
        output_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build a mixed decoder input for Mamba scheduled teacher forcing.

        Mamba decoder processes the whole mel sequence in parallel, unlike the
        RNN Tacotron decoder. Therefore we cannot switch teacher forcing inside
        a recurrent decoder loop. Instead we use a two-pass approximation:

          1. no-grad teacher-forced pass predicts mel frames;
          2. predicted frames are shifted by one frame and mixed with the
             ground-truth mel_input according to teacher_forcing_ratio;
          3. the real training pass uses this mixed mel_input.

        teacher_forcing_ratio = 1.0 -> pure ground truth mel_input
        teacher_forcing_ratio = 0.0 -> pure shifted model prediction
        """
        ratio = float(teacher_forcing_ratio)
        ratio = max(0.0, min(1.0, ratio))

        if ratio >= 1.0:
            return mel_input

        with torch.no_grad():
            teacher_outputs = self.decode_teacher_forced(
                memory=memory,
                text_lengths=text_lengths,
                mel_input=mel_input,
            )
            predicted = teacher_outputs["mel_before"].detach()

        # The decoder input at time t should depend on the previous generated
        # frame. Frame 0 uses the standard all-zero GO frame.
        go = torch.zeros_like(mel_input[:, :1, :])
        predicted_prev = torch.cat([go, predicted[:, :-1, :]], dim=1)

        if ratio <= 0.0:
            mixed = predicted_prev
        else:
            keep_gt = torch.rand(
                mel_input.size(0),
                mel_input.size(1),
                1,
                device=mel_input.device,
                dtype=mel_input.dtype,
            ) < ratio
            mixed = torch.where(keep_gt, mel_input, predicted_prev)

        if output_lengths is not None:
            valid = LengthMask.make(output_lengths, max_len=mel_input.size(1)).unsqueeze(-1)
            mixed = torch.where(valid, mixed, mel_input)

        return mixed

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_input: torch.Tensor,
        output_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        memory = self.encode(text, text_lengths)

        if self.training and float(teacher_forcing_ratio) < 1.0:
            mel_input = self.build_scheduled_mel_input(
                memory=memory,
                text_lengths=text_lengths,
                mel_input=mel_input,
                teacher_forcing_ratio=teacher_forcing_ratio,
                output_lengths=output_lengths,
            )

        return self.decode_teacher_forced(memory, text_lengths, mel_input)

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if text_lengths is None:
            text_lengths = torch.full(
                (text.size(0),),
                fill_value=text.size(1),
                dtype=torch.long,
                device=text.device,
            )

        memory = self.encode(text, text_lengths)
        B = text.size(0)

        mel_before_frames = []
        gates = []
        aligns = []
        # Same decoder contract as RNN Tacotron2:
        # input[0] is GO, input[t] is the raw mel prediction from step t - 1.
        decoder_inputs = memory.new_zeros(B, 1, self.n_mels)

        for _ in range(self.max_decoder_steps):
            mel_sequence, gate_sequence, alignment_sequence = self.decode_sequence(
                memory=memory,
                text_lengths=text_lengths,
                mel_input=decoder_inputs,
            )
            next_mel_before = mel_sequence[:, -1:, :]
            next_gate = gate_sequence[:, -1:]
            next_align = alignment_sequence[:, -1:, :]

            mel_before_frames.append(next_mel_before)
            gates.append(next_gate)
            aligns.append(next_align)

            if torch.sigmoid(next_gate).max().item() > self.gate_threshold:
                break

            decoder_inputs = torch.cat([decoder_inputs, next_mel_before], dim=1)

        mel_before = torch.cat(mel_before_frames, dim=1)
        # Tacotron2 postnet sees the complete generated sequence once. Running it
        # on every prefix would make stored early frames inconsistent because
        # its convolutions use neighboring frames in both directions.
        mel_after = self.postnet(mel_before)
        gate = torch.cat(gates, dim=1)
        alignments = torch.cat(aligns, dim=1)
        return {
            "mel_before": mel_before,
            "mel_after": mel_after,
            "gate": gate,
            "alignments": alignments,
        }


if __name__ == "__main__":
    model = MambaTacotron2(n_symbols=64, n_mels=80, d_model=128, encoder_layers=2, decoder_layers=2)
    text = torch.randint(1, 64, (2, 12))
    text_lengths = torch.tensor([12, 9])
    mel_input = torch.randn(2, 50, 80)
    out = model(text=text, text_lengths=text_lengths, mel_input=mel_input)
    for k, v in out.items():
        print(k, tuple(v.shape))
