#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperparams_transformer as hp


def hp_get(name: str, default):
    return getattr(hp, name, default)


def get_n_mels() -> int:
    if hasattr(hp, "n_mels"):
        return int(hp.n_mels)
    if hasattr(hp, "num_mels"):
        return int(hp.num_mels)
    raise AttributeError("hyperparams must define hp.n_mels or hp.num_mels")


def get_n_symbols() -> int:
    for key in ["n_symbols", "num_symbols", "vocab_size"]:
        if hasattr(hp, key):
            return int(getattr(hp, key))

    try:
        from text.symbols import symbols
        return len(symbols)
    except Exception as exc:
        raise AttributeError(
            "hyperparams must define one of: n_symbols, num_symbols, vocab_size; "
            "or text.symbols must provide `symbols`"
        ) from exc


def get_outputs_per_step() -> int:
    if hasattr(hp, "outputs_per_step"):
        return int(hp.outputs_per_step)
    if hasattr(hp, "r"):
        return int(hp.r)
    return 1


def get_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) >= lengths.unsqueeze(1)


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)


N_MELS = get_n_mels()
N_SYMBOLS = get_n_symbols()
R = get_outputs_per_step()

SYMBOL_EMBED_DIM = hp_get("symbols_embedding_dim", 512)

TRANSFORMER_D_MODEL = hp_get("transformer_d_model", 256)
TRANSFORMER_NHEAD = hp_get("transformer_nhead", 4)
TRANSFORMER_ENCODER_LAYERS = hp_get("transformer_encoder_layers", 4)
TRANSFORMER_DECODER_LAYERS = hp_get("transformer_decoder_layers", 4)
TRANSFORMER_FFN_DIM = hp_get("transformer_ffn_dim", 1024)
TRANSFORMER_FFN_TYPE = hp_get("transformer_ffn_type", "swiglu")
TRANSFORMER_DROPOUT = hp_get("transformer_dropout", 0.1)

PRENET_DIMS = hp_get("prenet_dims", [256, 256])

POSTNET_EMBED_DIM = hp_get("postnet_embedding_dim", 512)
POSTNET_KERNEL_SIZE = hp_get("postnet_kernel_size", 5)
POSTNET_N_CONVS = hp_get("postnet_n_convolutions", 5)

MAX_DECODER_STEPS = hp_get("max_decoder_steps", 1000)
GATE_THRESHOLD = hp_get("gate_threshold", 0.5)
MAX_TEXT_POSITIONS = hp_get("max_text_positions", 2048)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding limit {self.pe.size(1)}"
            )
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: list[int]) -> None:
        super().__init__()
        layers = []
        current_dim = in_dim
        for out_dim in sizes:
            layers.append(nn.Linear(current_dim, out_dim))
            current_dim = out_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        convs = []
        convs.append(
            nn.Sequential(
                nn.Conv1d(
                    N_MELS,
                    POSTNET_EMBED_DIM,
                    kernel_size=POSTNET_KERNEL_SIZE,
                    padding=(POSTNET_KERNEL_SIZE - 1) // 2,
                ),
                nn.BatchNorm1d(POSTNET_EMBED_DIM),
            )
        )

        for _ in range(1, POSTNET_N_CONVS - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        POSTNET_EMBED_DIM,
                        POSTNET_EMBED_DIM,
                        kernel_size=POSTNET_KERNEL_SIZE,
                        padding=(POSTNET_KERNEL_SIZE - 1) // 2,
                    ),
                    nn.BatchNorm1d(POSTNET_EMBED_DIM),
                )
            )

        convs.append(
            nn.Sequential(
                nn.Conv1d(
                    POSTNET_EMBED_DIM,
                    N_MELS,
                    kernel_size=POSTNET_KERNEL_SIZE,
                    padding=(POSTNET_KERNEL_SIZE - 1) // 2,
                ),
                nn.BatchNorm1d(N_MELS),
            )
        )
        self.convolutions = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for i, conv in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, self.training)
            else:
                x = F.dropout(conv(x), 0.5, self.training)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        if TRANSFORMER_FFN_TYPE == "relu":
            self.net = nn.Sequential(
                nn.Linear(TRANSFORMER_D_MODEL, TRANSFORMER_FFN_DIM),
                nn.ReLU(),
                nn.Dropout(TRANSFORMER_DROPOUT),
                nn.Linear(TRANSFORMER_FFN_DIM, TRANSFORMER_D_MODEL),
            )
        elif TRANSFORMER_FFN_TYPE == "swiglu":
            self.net = None
            self.input_proj = nn.Linear(TRANSFORMER_D_MODEL, TRANSFORMER_FFN_DIM * 2)
            self.dropout = nn.Dropout(TRANSFORMER_DROPOUT)
            self.output_proj = nn.Linear(TRANSFORMER_FFN_DIM, TRANSFORMER_D_MODEL)
        else:
            raise ValueError(f"Unknown transformer_ffn_type: {TRANSFORMER_FFN_TYPE}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is not None:
            return self.net(x)

        value, gate = self.input_proj(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.dropout(x)
        return self.output_proj(x)


class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=TRANSFORMER_D_MODEL,
            num_heads=TRANSFORMER_NHEAD,
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True,
        )
        self.ffn = FeedForward()
        self.norm1 = nn.LayerNorm(TRANSFORMER_D_MODEL)
        self.norm2 = nn.LayerNorm(TRANSFORMER_D_MODEL)
        self.dropout = nn.Dropout(TRANSFORMER_DROPOUT)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        y, _ = self.self_attn(
            y,
            y,
            y,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(y)

        residual = x
        y = self.norm2(x)
        y = self.ffn(y)
        return residual + self.dropout(y)


class TransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(SYMBOL_EMBED_DIM, TRANSFORMER_D_MODEL)
        self.positional_encoding = PositionalEncoding(
            d_model=TRANSFORMER_D_MODEL,
            max_len=MAX_TEXT_POSITIONS,
            dropout=TRANSFORMER_DROPOUT,
        )

        self.layers = nn.ModuleList([TransformerEncoderLayerWithAttention() for _ in range(TRANSFORMER_ENCODER_LAYERS)])

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        src_key_padding_mask = get_mask_from_lengths(input_lengths, max_len=x.size(1))
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=None)
        return x


class TransformerDecoderLayerWithAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=TRANSFORMER_D_MODEL,
            num_heads=TRANSFORMER_NHEAD,
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=TRANSFORMER_D_MODEL,
            num_heads=TRANSFORMER_NHEAD,
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True,
        )

        self.ffn = FeedForward()

        self.norm1 = nn.LayerNorm(TRANSFORMER_D_MODEL)
        self.norm2 = nn.LayerNorm(TRANSFORMER_D_MODEL)
        self.norm3 = nn.LayerNorm(TRANSFORMER_D_MODEL)

        self.dropout = nn.Dropout(TRANSFORMER_DROPOUT)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor],
        memory_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = tgt
        x = self.norm1(tgt)
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = residual + self.dropout(x)

        residual = tgt
        x = self.norm2(tgt)
        x, cross_attn_weights = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = residual + self.dropout(x)

        residual = tgt
        x = self.norm3(tgt)
        x = self.ffn(x)
        tgt = residual + self.dropout(x)

        return tgt, cross_attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        if R != 1:
            raise ValueError("This Transformer Tacotron implementation currently supports outputs_per_step=1 only.")

        self.prenet = Prenet(N_MELS, PRENET_DIMS)
        self.prenet_proj = nn.Linear(PRENET_DIMS[-1], TRANSFORMER_D_MODEL)
        self.positional_encoding = PositionalEncoding(
            d_model=TRANSFORMER_D_MODEL,
            max_len=MAX_DECODER_STEPS + 8,
            dropout=TRANSFORMER_DROPOUT,
        )

        self.layers = nn.ModuleList([TransformerDecoderLayerWithAttention() for _ in range(TRANSFORMER_DECODER_LAYERS)])

        self.mel_projection = nn.Linear(TRANSFORMER_D_MODEL, N_MELS)
        self.gate_projection = nn.Linear(TRANSFORMER_D_MODEL, 1)

    def parse_decoder_inputs(self, decoder_inputs: torch.Tensor) -> torch.Tensor:
        if decoder_inputs.size(-1) != N_MELS:
            raise ValueError(f"Expected mel dim {N_MELS}, got {decoder_inputs.size(-1)}")
        return decoder_inputs

    def decode(self, decoder_inputs: torch.Tensor, memory: torch.Tensor, text_lengths: torch.Tensor,
               output_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.parse_decoder_inputs(decoder_inputs)
        x = self.prenet(x)
        x = self.prenet_proj(x)
        x = self.positional_encoding(x)

        tgt_mask = generate_square_subsequent_mask(x.size(1), device=x.device)

        tgt_key_padding_mask = None
        if output_lengths is not None:
            tgt_key_padding_mask = get_mask_from_lengths(output_lengths, max_len=x.size(1))

        memory_key_padding_mask = get_mask_from_lengths(text_lengths, max_len=memory.size(1))

        cross_attn_last = None
        for layer in self.layers:
            x, cross_attn_last = layer(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        mel_outputs = self.mel_projection(x)
        gate_outputs = self.gate_projection(x).squeeze(-1)
        alignments = cross_attn_last.mean(dim=1)

        return (
            mel_outputs,
            gate_outputs,
            alignments,
        )

    def forward(self, memory: torch.Tensor, mel_input: torch.Tensor, text_lengths: torch.Tensor,
                output_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decode(mel_input, memory, text_lengths, output_lengths)

    @torch.no_grad()
    def inference(self, memory: torch.Tensor, text_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = memory.size(0)
        device = memory.device
        decoder_inputs = memory.new_zeros(bsz, 1, N_MELS)

        generated, gates, alignment_history = [], [], []
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(MAX_DECODER_STEPS):
            mel_seq, gate_seq, align_seq = self.decode(
                decoder_inputs=decoder_inputs,
                memory=memory,
                text_lengths=text_lengths,
                output_lengths=None,
            )

            mel_last = mel_seq[:, -1:, :]
            gate_last = gate_seq[:, -1]
            align_last = align_seq[:, -1:, :]

            generated.append(mel_last)
            gates.append(gate_last.unsqueeze(1))
            alignment_history.append(align_last)

            finished = finished | (torch.sigmoid(gate_last) > GATE_THRESHOLD)
            decoder_inputs = torch.cat([decoder_inputs, mel_last], dim=1)

            if bool(finished.all()):
                break

        mel_outputs = torch.cat(generated, dim=1)
        gate_outputs = torch.cat(gates, dim=1)
        alignments = torch.cat(alignment_history, dim=1)

        return mel_outputs, gate_outputs, alignments


class TransformerTacotron2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        if TRANSFORMER_D_MODEL % TRANSFORMER_NHEAD != 0:
            raise ValueError(
                f"transformer_d_model ({TRANSFORMER_D_MODEL}) must be divisible by "
                f"transformer_nhead ({TRANSFORMER_NHEAD})."
            )

        self.embedding = nn.Embedding(N_SYMBOLS, SYMBOL_EMBED_DIM)
        std = (2.0 / (N_SYMBOLS + SYMBOL_EMBED_DIM)) ** 0.5
        val = (3.0 ** 0.5) * std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.postnet = Postnet()

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor, mel_input: torch.Tensor,
                output_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(text)
        memory = self.encoder(embedded, text_lengths)

        mel_before, gate, alignments = self.decoder(
            memory=memory,
            mel_input=mel_input,
            text_lengths=text_lengths,
            output_lengths=output_lengths,
        )
        mel_after = mel_before + self.postnet(mel_before)

        return {
            "mel_before": mel_before,
            "mel_after": mel_after,
            "gate": gate,
            "alignments": alignments,
        }

    @torch.no_grad()
    def inference(self, text: torch.Tensor, text_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(text)

        if text_lengths is None:
            text_lengths = torch.full(
                (text.size(0),),
                fill_value=text.size(1),
                dtype=torch.long,
                device=text.device,
            )
            memory = self.encoder.inference(embedded)
        else:
            memory = self.encoder(embedded, text_lengths)

        mel_before, gate, alignments = self.decoder.inference(memory, text_lengths)
        mel_after = mel_before + self.postnet(mel_before)

        return {
            "mel_before": mel_before,
            "mel_after": mel_after,
            "gate": gate,
            "alignments": alignments,
        }
