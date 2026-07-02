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

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


def reverse_padded_sequence(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Reverse only the valid prefix of each sequence.

    Padding remains on the right side, unlike torch.flip(x, dims=(1,)), which
    moves padded frames to the beginning and can leak padding through a
    recurrent/SSM encoder.
    """
    B, T, _ = x.shape
    ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    reverse_ids = (lengths.unsqueeze(1) - 1 - ids).clamp(min=0)
    gather_ids = torch.where(ids < lengths.unsqueeze(1), reverse_ids, ids)
    return x.gather(1, gather_ids.unsqueeze(-1).expand_as(x))


class Prenet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unlike the step-wise RNN decoder, Mamba recomputes the full generated
        # prefix at inference. Keep eval deterministic so historical decoder
        # states do not change randomly between generation steps.
        for layer in self.layers:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class RNNPrenet(nn.Module):
    """
    Tacotron-style prenet used by the step-wise RNN decoder.

    The original RNN baseline keeps prenet dropout enabled during inference, so
    this hybrid branch follows that behavior instead of the deterministic
    full-prefix Mamba prenet.
    """
    def __init__(self, in_dim: int, sizes: list[int], dropout: float = 0.5) -> None:
        super().__init__()
        layers = []
        current_dim = in_dim
        for out_dim in sizes:
            out_dim = int(out_dim)
            layers.append(nn.Linear(current_dim, out_dim))
            current_dim = out_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = float(dropout)

    @property
    def out_dim(self) -> int:
        return self.layers[-1].out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.dropout, training=True)
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

    def allocate_step_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.sequence, "allocate_inference_cache"):
            raise RuntimeError("mamba_ssm.Mamba does not expose allocate_inference_cache()")
        return self.sequence.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            dtype=dtype,
        )

    def step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(self.sequence, "step"):
            raise RuntimeError("mamba_ssm.Mamba does not expose step()")
        residual = x
        x = self.norm(x)
        conv_state, ssm_state = cache
        x, conv_state, ssm_state = self.sequence.step(x, conv_state, ssm_state)
        x = self.dropout(x)
        return residual + x, (conv_state, ssm_state)


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

    def allocate_step_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            block.allocate_step_cache(
                batch_size=batch_size,
                max_seqlen=max_seqlen,
                dtype=dtype,
            )
            for block in self.blocks
        ]

    def step(
        self,
        x: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        next_caches = []
        for block, cache in zip(self.blocks, caches):
            x, cache = block.step(x, cache)
            next_caches.append(cache)
        return self.final_norm(x), next_caches


class LocationLayer(nn.Module):
    def __init__(
        self,
        attention_dim: int,
        n_filters: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.location_conv = nn.Conv1d(
            in_channels=2,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.location_dense = nn.Linear(n_filters, attention_dim, bias=False)

    def forward(self, attention_weights_cat: torch.Tensor) -> torch.Tensor:
        processed = self.location_conv(attention_weights_cat)
        return self.location_dense(processed.transpose(1, 2))


class CrossAttention(nn.Module):
    """
    RNN Tacotron2-style location-sensitive attention.

    Mamba decoder states replace the RNN attention hidden state, while the
    previous and cumulative alignment maps are updated exactly as in the RNN
    decoder.
    """

    def __init__(
        self,
        d_model: int,
        attention_dim: int = 128,
        location_filters: int = 32,
        location_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_layer = nn.Linear(d_model, attention_dim, bias=False)
        self.memory_layer = nn.Linear(d_model, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_dim=attention_dim,
            n_filters=location_filters,
            kernel_size=location_kernel_size,
        )
        self.context_projection = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def get_alignment_energies(
        self,
        query: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
    ) -> torch.Tensor:
        processed_query = self.query_layer(query).unsqueeze(1)
        processed_attention = self.location_layer(attention_weights_cat)
        return self.v(
            torch.tanh(processed_query + processed_attention + processed_memory)
        ).squeeze(-1)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query:  (B, T_mel, D)
        # memory: (B, T_text, D)
        B, T_mel, _ = query.shape
        T_text = memory.size(1)
        processed_memory = self.memory_layer(memory)
        invalid_memory = ~LengthMask.make(memory_lengths, max_len=T_text)

        attention_weights = memory.new_zeros(B, T_text)
        attention_weights_cum = memory.new_zeros(B, T_text)
        contexts = []
        alignments = []

        for t in range(T_mel):
            attention_weights_cat = torch.stack(
                [attention_weights, attention_weights_cum],
                dim=1,
            )
            energies = self.get_alignment_energies(
                query=query[:, t, :],
                processed_memory=processed_memory,
                attention_weights_cat=attention_weights_cat,
            )
            energies = energies.masked_fill(invalid_memory, -float("inf"))
            attention_weights = F.softmax(energies, dim=1)
            attention_weights_cum = attention_weights_cum + attention_weights
            context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)

            contexts.append(context)
            alignments.append(attention_weights)

        context_sequence = torch.stack(contexts, dim=1)
        alignment_sequence = torch.stack(alignments, dim=1)
        x = self.context_projection(
            torch.cat([query, self.dropout(context_sequence)], dim=-1)
        )
        x = self.norm(x)
        return x, alignment_sequence

    def step(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        invalid_memory: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_weights_cum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_weights_cat = torch.stack(
            [attention_weights, attention_weights_cum],
            dim=1,
        )
        energies = self.get_alignment_energies(
            query=query,
            processed_memory=processed_memory,
            attention_weights_cat=attention_weights_cat,
        )
        energies = energies.masked_fill(invalid_memory, -float("inf"))
        attention_weights = F.softmax(energies, dim=1)
        attention_weights_cum = attention_weights_cum + attention_weights
        context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        x = self.context_projection(
            torch.cat([query, self.dropout(context)], dim=-1)
        )
        x = self.norm(x)
        return x, attention_weights, attention_weights_cum


class RNNLocationSensitiveAttention(nn.Module):
    """Location-sensitive attention used by the RNN decoder branch."""
    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        attention_dim: int,
        location_filters: int,
        location_kernel_size: int,
    ) -> None:
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(memory_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_dim=attention_dim,
            n_filters=location_filters,
            kernel_size=location_kernel_size,
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
        self,
        query: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
    ) -> torch.Tensor:
        processed_query = self.query_layer(query).unsqueeze(1)
        processed_attention = self.location_layer(attention_weights_cat)
        return self.v(
            torch.tanh(processed_query + processed_attention + processed_memory)
        ).squeeze(-1)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        energies = self.get_alignment_energies(
            query=query,
            processed_memory=processed_memory,
            attention_weights_cat=attention_weights_cat,
        )
        if mask is not None:
            energies = energies.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(energies, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return attention_context, attention_weights


@dataclass
class RNNDecoderStates:
    attention_hidden: torch.Tensor
    attention_cell: torch.Tensor
    decoder_hidden: torch.Tensor
    decoder_cell: torch.Tensor
    attention_weights: torch.Tensor
    attention_weights_cum: torch.Tensor
    attention_context: torch.Tensor


class RNNDecoder(nn.Module):
    """
    Tacotron-style autoregressive decoder for the hybrid Mamba experiment.

    The encoder memory still comes from the bidirectional Mamba encoder, but
    mel generation is step-wise and stateful like the working RNN baseline.
    """
    def __init__(
        self,
        n_mels: int,
        encoder_dim: int,
        attention_rnn_dim: int,
        decoder_rnn_dim: int,
        attention_dim: int,
        location_filters: int,
        location_kernel_size: int,
        prenet_dims: list[int],
        prenet_dropout: float,
        attention_dropout: float,
        decoder_dropout: float,
        max_decoder_steps: int,
        gate_threshold: float,
    ) -> None:
        super().__init__()
        self.n_mels = int(n_mels)
        self.encoder_dim = int(encoder_dim)
        self.attention_rnn_dim = int(attention_rnn_dim)
        self.decoder_rnn_dim = int(decoder_rnn_dim)
        self.attention_dropout = float(attention_dropout)
        self.decoder_dropout = float(decoder_dropout)
        self.max_decoder_steps = int(max_decoder_steps)
        self.gate_threshold = float(gate_threshold)

        self.prenet = RNNPrenet(
            in_dim=self.n_mels,
            sizes=prenet_dims,
            dropout=prenet_dropout,
        )
        self.attention_rnn = nn.LSTMCell(
            self.prenet.out_dim + self.encoder_dim,
            self.attention_rnn_dim,
        )
        self.attention_layer = RNNLocationSensitiveAttention(
            query_dim=self.attention_rnn_dim,
            memory_dim=self.encoder_dim,
            attention_dim=attention_dim,
            location_filters=location_filters,
            location_kernel_size=location_kernel_size,
        )
        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_dim,
            self.decoder_rnn_dim,
        )
        self.linear_projection = nn.Linear(
            self.decoder_rnn_dim + self.encoder_dim,
            self.n_mels,
        )
        self.gate_layer = nn.Linear(
            self.decoder_rnn_dim + self.encoder_dim,
            1,
        )

    def get_go_frame(self, memory: torch.Tensor) -> torch.Tensor:
        return memory.new_zeros(memory.size(0), self.n_mels)

    def initialize_decoder_states(
        self,
        memory: torch.Tensor,
        memory_lengths: torch.Tensor,
    ) -> Tuple[RNNDecoderStates, torch.Tensor, torch.Tensor]:
        B, T_enc, _ = memory.size()
        device = memory.device
        dtype = memory.dtype
        states = RNNDecoderStates(
            attention_hidden=torch.zeros(B, self.attention_rnn_dim, device=device, dtype=dtype),
            attention_cell=torch.zeros(B, self.attention_rnn_dim, device=device, dtype=dtype),
            decoder_hidden=torch.zeros(B, self.decoder_rnn_dim, device=device, dtype=dtype),
            decoder_cell=torch.zeros(B, self.decoder_rnn_dim, device=device, dtype=dtype),
            attention_weights=torch.zeros(B, T_enc, device=device, dtype=dtype),
            attention_weights_cum=torch.zeros(B, T_enc, device=device, dtype=dtype),
            attention_context=torch.zeros(B, self.encoder_dim, device=device, dtype=dtype),
        )
        processed_memory = self.attention_layer.memory_layer(memory)
        mask = ~LengthMask.make(memory_lengths, max_len=T_enc)
        return states, processed_memory, mask

    def decode(
        self,
        decoder_input: torch.Tensor,
        states: RNNDecoderStates,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNDecoderStates]:
        cell_input = torch.cat([decoder_input, states.attention_context], dim=-1)
        attention_hidden, attention_cell = self.attention_rnn(
            cell_input,
            (states.attention_hidden, states.attention_cell),
        )
        attention_hidden = F.dropout(
            attention_hidden,
            p=self.attention_dropout,
            training=self.training,
        )

        attention_weights_cat = torch.stack(
            [states.attention_weights, states.attention_weights_cum],
            dim=1,
        )
        attention_context, attention_weights = self.attention_layer(
            query=attention_hidden,
            memory=memory,
            processed_memory=processed_memory,
            attention_weights_cat=attention_weights_cat,
            mask=mask,
        )
        attention_weights_cum = states.attention_weights_cum + attention_weights

        decoder_input = torch.cat([attention_hidden, attention_context], dim=-1)
        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input,
            (states.decoder_hidden, states.decoder_cell),
        )
        decoder_hidden = F.dropout(
            decoder_hidden,
            p=self.decoder_dropout,
            training=self.training,
        )

        projection_input = torch.cat([decoder_hidden, attention_context], dim=-1)
        mel_output = self.linear_projection(projection_input)
        gate_output = self.gate_layer(projection_input)

        new_states = RNNDecoderStates(
            attention_hidden=attention_hidden,
            attention_cell=attention_cell,
            decoder_hidden=decoder_hidden,
            decoder_cell=decoder_cell,
            attention_weights=attention_weights,
            attention_weights_cum=attention_weights_cum,
            attention_context=attention_context,
        )
        return mel_output, gate_output, attention_weights, new_states

    def parse_decoder_outputs(
        self,
        mel_outputs: list[torch.Tensor],
        gate_outputs: list[torch.Tensor],
        alignments: list[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs = torch.stack(gate_outputs, dim=1).squeeze(-1)
        alignments = torch.stack(alignments, dim=1)
        return mel_outputs, gate_outputs, alignments

    def forward(
        self,
        memory: torch.Tensor,
        memory_lengths: torch.Tensor,
        mel_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, processed_memory, mask = self.initialize_decoder_states(memory, memory_lengths)
        decoder_inputs = self.prenet(mel_input).transpose(0, 1)

        mel_outputs = []
        gate_outputs = []
        alignments = []
        for decoder_input in decoder_inputs:
            mel_output, gate_output, alignment, states = self.decode(
                decoder_input=decoder_input,
                states=states,
                memory=memory,
                processed_memory=processed_memory,
                mask=mask,
            )
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(alignment)

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    @torch.no_grad()
    def inference(
        self,
        memory: torch.Tensor,
        memory_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, processed_memory, mask = self.initialize_decoder_states(memory, memory_lengths)
        decoder_input = self.get_go_frame(memory)

        mel_outputs = []
        gate_outputs = []
        alignments = []
        for _ in range(self.max_decoder_steps):
            prenet_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment, states = self.decode(
                decoder_input=prenet_input,
                states=states,
                memory=memory,
                processed_memory=processed_memory,
                mask=mask,
            )
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(alignment)

            if torch.sigmoid(gate_output).max().item() > self.gate_threshold:
                break

            decoder_input = mel_output

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)


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
        dropout = float(dropout if dropout is not None else hp_get("mamba_dropout", 0.1))
        self.decoder_type = str(hp_get("mamba_decoder_type", "mamba")).lower()
        if self.decoder_type not in {"mamba", "mamba_step", "rnn"}:
            raise ValueError(
                "mamba_decoder_type must be 'mamba', 'mamba_step', or 'rnn', "
                f"got {self.decoder_type!r}"
            )

        self.embedding = nn.Embedding(self.n_symbols, self.d_model, padding_idx=0)
        self.encoder_forward = MambaStack(
            d_model=self.d_model,
            n_layers=enc_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.encoder_backward = MambaStack(
            d_model=self.d_model,
            n_layers=enc_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.encoder_projection = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.prenet = Prenet(
            in_dim=self.n_mels,
            hidden_dim=int(hp_get("mamba_prenet_hidden", 256)),
            out_dim=self.d_model,
            dropout=float(hp_get("mamba_prenet_dropout", 0.5)),
        )
        self.mamba_decoder = MambaStack(
            d_model=self.d_model,
            n_layers=dec_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.cross_attn = CrossAttention(
            d_model=self.d_model,
            attention_dim=int(hp_get("mamba_attention_dim", 128)),
            location_filters=int(hp_get("mamba_attention_location_filters", 32)),
            location_kernel_size=int(hp_get("mamba_attention_location_kernel_size", 31)),
            dropout=dropout,
        )
        self.rnn_decoder = RNNDecoder(
            n_mels=self.n_mels,
            encoder_dim=self.d_model,
            attention_rnn_dim=int(hp_get("mamba_rnn_attention_dim", hp_get("attention_rnn_dim", 1024))),
            decoder_rnn_dim=int(hp_get("mamba_rnn_decoder_dim", hp_get("decoder_rnn_dim", 1024))),
            attention_dim=int(hp_get("mamba_attention_dim", 128)),
            location_filters=int(hp_get("mamba_attention_location_filters", 32)),
            location_kernel_size=int(hp_get("mamba_attention_location_kernel_size", 31)),
            prenet_dims=list(hp_get("mamba_rnn_prenet_dims", [256, 256])),
            prenet_dropout=float(hp_get("mamba_rnn_prenet_dropout", hp_get("mamba_prenet_dropout", 0.5))),
            attention_dropout=float(hp_get("mamba_rnn_attention_dropout", 0.1)),
            decoder_dropout=float(hp_get("mamba_rnn_decoder_dropout", 0.1)),
            max_decoder_steps=self.max_decoder_steps,
            gate_threshold=self.gate_threshold,
        )
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
        for module in self.encoder_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _remap_legacy_encoder_keys(state_dict):
        """
        Keep old Mamba checkpoints loadable after renaming encoder branches.

        Previous names:
          encoder.*     -> encoder_forward.*
          encoder_rev.* -> encoder_backward.*
          decoder.*     -> mamba_decoder.*
        """
        renamed_prefixes = {
            "encoder.": "encoder_forward.",
            "encoder_rev.": "encoder_backward.",
            "decoder.": "mamba_decoder.",
        }

        remapped = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in renamed_prefixes.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            remapped[new_key] = value

        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            remapped._metadata = OrderedDict()
            for key, value in metadata.items():
                new_key = key
                for old_prefix, new_prefix in renamed_prefixes.items():
                    metadata_prefix = old_prefix.rstrip(".")
                    if key == metadata_prefix or key.startswith(old_prefix):
                        new_key = new_prefix.rstrip(".") + key[len(metadata_prefix):]
                        break
                remapped._metadata[new_key] = value

        return remapped

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = self._remap_legacy_encoder_keys(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def encode(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(text)
        x_forward = self.encoder_forward(x)
        x_backward = self.encoder_backward(reverse_padded_sequence(x, text_lengths))
        x_backward = reverse_padded_sequence(x_backward, text_lengths)
        x = self.encoder_projection(torch.cat([x_forward, x_backward], dim=-1))
        mask = LengthMask.make(text_lengths, max_len=x.size(1)).unsqueeze(-1).to(x.dtype)
        return x * mask

    def decode_teacher_forced(
        self,
        memory: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.decoder_type == "rnn":
            mel_before, gate, alignments = self.rnn_decoder(
                memory=memory,
                memory_lengths=text_lengths,
                mel_input=mel_input,
            )
        else:
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
        x = self.mamba_decoder(x)
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

        if self.decoder_type in {"mamba", "mamba_step"} and self.training and float(teacher_forcing_ratio) < 1.0:
            mel_input = self.build_scheduled_mel_input(
                memory=memory,
                text_lengths=text_lengths,
                mel_input=mel_input,
                teacher_forcing_ratio=teacher_forcing_ratio,
                output_lengths=output_lengths,
            )

        return self.decode_teacher_forced(memory, text_lengths, mel_input)

    @torch.no_grad()
    def inference_mamba_step(
        self,
        memory: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = memory.size(0)
        T_text = memory.size(1)

        decoder_caches = self.mamba_decoder.allocate_step_cache(
            batch_size=B,
            max_seqlen=self.max_decoder_steps,
            dtype=memory.dtype,
        )
        processed_memory = self.cross_attn.memory_layer(memory)
        invalid_memory = ~LengthMask.make(text_lengths, max_len=T_text)
        attention_weights = memory.new_zeros(B, T_text)
        attention_weights_cum = memory.new_zeros(B, T_text)

        decoder_input = memory.new_zeros(B, 1, self.n_mels)
        mel_before_frames = []
        gates = []
        aligns = []

        for _ in range(self.max_decoder_steps):
            x = self.prenet(decoder_input)
            x, decoder_caches = self.mamba_decoder.step(x, decoder_caches)
            x, attention_weights, attention_weights_cum = self.cross_attn.step(
                query=x[:, 0, :],
                memory=memory,
                processed_memory=processed_memory,
                invalid_memory=invalid_memory,
                attention_weights=attention_weights,
                attention_weights_cum=attention_weights_cum,
            )
            x = self.fusion(x.unsqueeze(1))

            next_mel_before = self.mel_proj(x)
            next_gate = self.gate_proj(x).squeeze(-1)
            next_align = attention_weights.unsqueeze(1)

            mel_before_frames.append(next_mel_before)
            gates.append(next_gate)
            aligns.append(next_align)

            if torch.sigmoid(next_gate).max().item() > self.gate_threshold:
                break

            decoder_input = next_mel_before

        mel_before = torch.cat(mel_before_frames, dim=1)
        mel_after = self.postnet(mel_before)
        gate = torch.cat(gates, dim=1)
        alignments = torch.cat(aligns, dim=1)
        return {
            "mel_before": mel_before,
            "mel_after": mel_after,
            "gate": gate,
            "alignments": alignments,
        }

    @torch.no_grad()
    def decode_sequence_step_teacher_forced(
        self,
        memory: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode a known mel-input sequence through the stateful Mamba step path.

        This is a diagnostic counterpart to decode_sequence(). In eval mode,
        its outputs should be close to full-sequence decode_sequence() if the
        Mamba step cache path is equivalent to the normal Mamba forward path.
        """
        B, T_mel, _ = mel_input.shape
        T_text = memory.size(1)

        decoder_caches = self.mamba_decoder.allocate_step_cache(
            batch_size=B,
            max_seqlen=T_mel,
            dtype=memory.dtype,
        )
        processed_memory = self.cross_attn.memory_layer(memory)
        invalid_memory = ~LengthMask.make(text_lengths, max_len=T_text)
        attention_weights = memory.new_zeros(B, T_text)
        attention_weights_cum = memory.new_zeros(B, T_text)

        mel_outputs = []
        gate_outputs = []
        alignments = []
        for t in range(T_mel):
            x = self.prenet(mel_input[:, t:t + 1, :])
            x, decoder_caches = self.mamba_decoder.step(x, decoder_caches)
            x, attention_weights, attention_weights_cum = self.cross_attn.step(
                query=x[:, 0, :],
                memory=memory,
                processed_memory=processed_memory,
                invalid_memory=invalid_memory,
                attention_weights=attention_weights,
                attention_weights_cum=attention_weights_cum,
            )
            x = self.fusion(x.unsqueeze(1))
            mel_outputs.append(self.mel_proj(x))
            gate_outputs.append(self.gate_proj(x).squeeze(-1))
            alignments.append(attention_weights.unsqueeze(1))

        return (
            torch.cat(mel_outputs, dim=1),
            torch.cat(gate_outputs, dim=1),
            torch.cat(alignments, dim=1),
        )

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

        if self.decoder_type == "rnn":
            mel_before, gate, alignments = self.rnn_decoder.inference(
                memory=memory,
                memory_lengths=text_lengths,
            )
            mel_after = self.postnet(mel_before)
            return {
                "mel_before": mel_before,
                "mel_after": mel_after,
                "gate": gate,
                "alignments": alignments,
            }

        if self.decoder_type == "mamba_step":
            return self.inference_mamba_step(memory=memory, text_lengths=text_lengths)

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
