#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import hyperparams as hp


# =============================================================================
# Helpers
# =============================================================================

def get_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Args:
        lengths: (B,)
    Returns:
        mask: (B, max_len), True where padded positions are located
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) >= lengths.unsqueeze(1)


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


def hp_get(name: str, default):
    return getattr(hp, name, default)


# =============================================================================
# Hyperparameters
# =============================================================================

N_MELS = get_n_mels()
N_SYMBOLS = get_n_symbols()
R = get_outputs_per_step()

SYMBOL_EMBED_DIM = hp_get("symbols_embedding_dim", hp_get("embedding_size", 512))

ENC_CONV_CHANNELS = hp_get("encoder_embedding_dim", hp_get("embedding_size", 512))
ENC_CONV_KERNEL = hp_get("encoder_kernel_size", 5)
ENC_CONV_LAYERS = hp_get("encoder_n_convolutions", 3)
ENC_DROPOUT = hp_get("encoder_dropout", 0.5)

ATTENTION_RNN_DIM = hp_get("attention_rnn_dim", 1024)
DECODER_RNN_DIM = hp_get("decoder_rnn_dim", 1024)
ATTENTION_DIM = hp_get("attention_dim", 128)
ATTENTION_LOCATION_N_FILTERS = hp_get("attention_location_n_filters", 32)
ATTENTION_LOCATION_KERNEL_SIZE = hp_get("attention_location_kernel_size", 31)

PRENET_DIMS = hp_get("prenet_dims", [256, 256])

POSTNET_EMBED_DIM = hp_get("postnet_embedding_dim", 512)
POSTNET_KERNEL_SIZE = hp_get("postnet_kernel_size", 5)
POSTNET_N_CONVS = hp_get("postnet_n_convolutions", 5)

MAX_DECODER_STEPS = hp_get("max_decoder_steps", 1000)
GATE_THRESHOLD = hp_get("gate_threshold", 0.5)
P_ATTENTION_DROPOUT = hp_get("p_attention_dropout", 0.1)
P_DECODER_DROPOUT = hp_get("p_decoder_dropout", 0.1)


# =============================================================================
# Prenet
# =============================================================================

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


# =============================================================================
# Postnet
# =============================================================================

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
        x = x.transpose(1, 2)  # (B, n_mels, T)

        for i, conv in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, self.training)
            else:
                x = F.dropout(conv(x), 0.5, self.training)

        return x.transpose(1, 2)


# =============================================================================
# Encoder
# =============================================================================

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        convs = []
        for i in range(ENC_CONV_LAYERS):
            in_ch = SYMBOL_EMBED_DIM if i == 0 else ENC_CONV_CHANNELS
            convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        ENC_CONV_CHANNELS,
                        kernel_size=ENC_CONV_KERNEL,
                        padding=(ENC_CONV_KERNEL - 1) // 2,
                    ),
                    nn.BatchNorm1d(ENC_CONV_CHANNELS),
                )
            )
        self.convolutions = nn.ModuleList(convs)

        self.lstm = nn.LSTM(
            input_size=ENC_CONV_CHANNELS,
            hidden_size=ENC_CONV_CHANNELS // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, embed_dim, T_text)

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), p=ENC_DROPOUT, training=self.training)

        x = x.transpose(1, 2)  # (B, T_text, encoder_dim)

        packed = pack_padded_sequence(
            x,
            input_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        return outputs

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), p=ENC_DROPOUT, training=self.training)
        x = x.transpose(1, 2)
        outputs, _ = self.lstm(x)
        return outputs


# =============================================================================
# Attention
# =============================================================================

class LocationLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        padding = (ATTENTION_LOCATION_KERNEL_SIZE - 1) // 2
        self.location_conv = nn.Conv1d(
            in_channels=2,
            out_channels=ATTENTION_LOCATION_N_FILTERS,
            kernel_size=ATTENTION_LOCATION_KERNEL_SIZE,
            padding=padding,
            bias=False,
        )
        self.location_dense = nn.Linear(
            ATTENTION_LOCATION_N_FILTERS,
            ATTENTION_DIM,
            bias=False,
        )

    def forward(self, attention_weights_cat: torch.Tensor) -> torch.Tensor:
        processed = self.location_conv(attention_weights_cat)
        processed = processed.transpose(1, 2)
        processed = self.location_dense(processed)
        return processed


class Attention(nn.Module):
    def __init__(self, encoder_dim: int) -> None:
        super().__init__()

        self.query_layer = nn.Linear(ATTENTION_RNN_DIM, ATTENTION_DIM, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, ATTENTION_DIM, bias=False)
        self.v = nn.Linear(ATTENTION_DIM, 1, bias=False)
        self.location_layer = LocationLayer()
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
        self,
        query: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
    ) -> torch.Tensor:
        processed_query = self.query_layer(query).unsqueeze(1)
        processed_attention = self.location_layer(attention_weights_cat)

        energies = self.v(
            torch.tanh(processed_query + processed_attention + processed_memory)
        ).squeeze(-1)

        return energies

    def forward(
        self,
        attention_hidden_state: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alignment = self.get_alignment_energies(
            attention_hidden_state,
            processed_memory,
            attention_weights_cat,
        )

        if mask is not None:
            alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)

        return attention_context, attention_weights


# =============================================================================
# Decoder
# =============================================================================

@dataclass
class DecoderStates:
    attention_hidden: torch.Tensor
    attention_cell: torch.Tensor
    decoder_hidden: torch.Tensor
    decoder_cell: torch.Tensor
    attention_weights: torch.Tensor
    attention_weights_cum: torch.Tensor
    attention_context: torch.Tensor


class Decoder(nn.Module):
    def __init__(self, encoder_dim: int) -> None:
        super().__init__()

        self.n_mels = N_MELS
        self.r = R
        self.encoder_dim = encoder_dim

        prenet_out_dim = PRENET_DIMS[-1]

        self.prenet = Prenet(self.n_mels * self.r, PRENET_DIMS)

        self.attention_rnn = nn.LSTMCell(
            prenet_out_dim + encoder_dim,
            ATTENTION_RNN_DIM,
        )

        self.attention_layer = Attention(encoder_dim)

        self.decoder_rnn = nn.LSTMCell(
            ATTENTION_RNN_DIM + encoder_dim,
            DECODER_RNN_DIM,
        )

        self.linear_projection = nn.Linear(
            DECODER_RNN_DIM + encoder_dim,
            self.n_mels * self.r,
        )

        self.gate_layer = nn.Linear(
            DECODER_RNN_DIM + encoder_dim,
            1,
        )

        self.memory_layer = nn.Linear(encoder_dim, ATTENTION_DIM, bias=False)

    def get_go_frame(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.size(0)
        return memory.new_zeros(B, self.n_mels * self.r)

    def initialize_decoder_states(
        self,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[DecoderStates, torch.Tensor, Optional[torch.Tensor]]:
        B, T_enc, _ = memory.size()
        device = memory.device
        dtype = memory.dtype

        states = DecoderStates(
            attention_hidden=torch.zeros(B, ATTENTION_RNN_DIM, device=device, dtype=dtype),
            attention_cell=torch.zeros(B, ATTENTION_RNN_DIM, device=device, dtype=dtype),
            decoder_hidden=torch.zeros(B, DECODER_RNN_DIM, device=device, dtype=dtype),
            decoder_cell=torch.zeros(B, DECODER_RNN_DIM, device=device, dtype=dtype),
            attention_weights=torch.zeros(B, T_enc, device=device, dtype=dtype),
            attention_weights_cum=torch.zeros(B, T_enc, device=device, dtype=dtype),
            attention_context=torch.zeros(B, self.encoder_dim, device=device, dtype=dtype),
        )

        processed_memory = self.memory_layer(memory)
        return states, processed_memory, mask

    def parse_decoder_inputs(self, decoder_inputs: torch.Tensor) -> torch.Tensor:
        """
        decoder_inputs: (B, T_mel, n_mels)
        returns: (T_out, B, n_mels * r)
        """
        B, T, C = decoder_inputs.size()

        if C != self.n_mels:
            raise ValueError(f"Expected mel dim {self.n_mels}, got {C}")

        if self.r > 1:
            if T % self.r != 0:
                raise ValueError(f"Decoder input length {T} must be divisible by r={self.r}")
            decoder_inputs = decoder_inputs.view(B, T // self.r, C * self.r)

        return decoder_inputs.transpose(0, 1)

    def parse_decoder_outputs(
        self,
        mel_outputs: list[torch.Tensor],
        gate_outputs: list[torch.Tensor],
        alignments: list[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel_outputs = torch.stack(mel_outputs, dim=1)

        if self.r > 1:
            B, T_out, _ = mel_outputs.size()
            mel_outputs = mel_outputs.view(B, T_out * self.r, self.n_mels)

        gate_outputs = torch.stack(gate_outputs, dim=1).squeeze(-1)
        alignments = torch.stack(alignments, dim=1)

        return mel_outputs, gate_outputs, alignments

    def decode(
        self,
        decoder_input: torch.Tensor,
        states: DecoderStates,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, DecoderStates]:
        cell_input = torch.cat([decoder_input, states.attention_context], dim=-1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input,
            (states.attention_hidden, states.attention_cell),
        )
        attention_hidden = F.dropout(attention_hidden, P_ATTENTION_DROPOUT, self.training)

        attention_weights_cat = torch.stack(
            [states.attention_weights, states.attention_weights_cum],
            dim=1,
        )

        attention_context, attention_weights = self.attention_layer(
            attention_hidden,
            memory,
            processed_memory,
            attention_weights_cat,
            mask,
        )

        attention_weights_cum = states.attention_weights_cum + attention_weights

        decoder_input2 = torch.cat([attention_hidden, attention_context], dim=-1)
        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input2,
            (states.decoder_hidden, states.decoder_cell),
        )
        decoder_hidden = F.dropout(decoder_hidden, P_DECODER_DROPOUT, self.training)

        proj_input = torch.cat([decoder_hidden, attention_context], dim=-1)

        mel_output = self.linear_projection(proj_input)
        gate_output = self.gate_layer(proj_input)

        new_states = DecoderStates(
            attention_hidden=attention_hidden,
            attention_cell=attention_cell,
            decoder_hidden=decoder_hidden,
            decoder_cell=decoder_cell,
            attention_weights=attention_weights,
            attention_weights_cum=attention_weights_cum,
            attention_context=attention_context,
        )

        return mel_output, gate_output, attention_weights, new_states

    def forward(
        self,
        memory: torch.Tensor,
        decoder_inputs: torch.Tensor,
        memory_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = get_mask_from_lengths(memory_lengths, max_len=memory.size(1))

        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = self.prenet(decoder_inputs)

        states, processed_memory, mask = self.initialize_decoder_states(memory, mask)

        mel_outputs = []
        gate_outputs = []
        alignments = []

        for t in range(decoder_inputs.size(0)):
            decoder_input = decoder_inputs[t]
            mel_output, gate_output, alignment, states = self.decode(
                decoder_input,
                states,
                memory,
                processed_memory,
                mask,
            )
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(alignment)

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    @torch.no_grad()
    def inference(
        self,
        memory: torch.Tensor,
        memory_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = None
        if memory_lengths is not None:
            mask = get_mask_from_lengths(memory_lengths, max_len=memory.size(1))

        states, processed_memory, mask = self.initialize_decoder_states(memory, mask)

        decoder_input = self.get_go_frame(memory)

        mel_outputs = []
        gate_outputs = []
        alignments = []

        for _ in range(MAX_DECODER_STEPS):
            prenet_input = self.prenet(decoder_input)

            mel_output, gate_output, alignment, states = self.decode(
                prenet_input,
                states,
                memory,
                processed_memory,
                mask,
            )

            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(alignment)

            if torch.sigmoid(gate_output).max().item() > GATE_THRESHOLD:
                break

            decoder_input = mel_output

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)


# =============================================================================
# Tacotron2
# =============================================================================

class Tacotron2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.n_symbols = N_SYMBOLS
        self.n_mels = N_MELS
        self.encoder_dim = ENC_CONV_CHANNELS

        self.embedding = nn.Embedding(self.n_symbols, SYMBOL_EMBED_DIM)

        std = (2.0 / (self.n_symbols + SYMBOL_EMBED_DIM)) ** 0.5
        val = (3.0 ** 0.5) * std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder()
        self.decoder = Decoder(encoder_dim=self.encoder_dim)
        self.postnet = Postnet()

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(text)
        memory = self.encoder(embedded, text_lengths)

        mel_before, gate, alignments = self.decoder(memory, mel_input, text_lengths)
        mel_after = mel_before + self.postnet(mel_before)

        return {
            "mel_before": mel_before,
            "mel_after": mel_after,
            "gate": gate,
            "alignments": alignments,
        }

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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


# =============================================================================
# Loss
# =============================================================================

class Tacotron2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        gate_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mel_before = outputs["mel_before"]
        mel_after = outputs["mel_after"]
        gate_pred = outputs["gate"]

        if R > 1:
            if gate_target.size(1) % R != 0:
                raise ValueError("gate_target length must be divisible by outputs_per_step")
            gate_target = gate_target[:, R - 1 :: R]

        mel_loss = self.mse(mel_before, mel_target) + self.mse(mel_after, mel_target)
        gate_loss = self.bce(gate_pred, gate_target)
        total_loss = mel_loss + gate_loss

        return {
            "loss": total_loss,
            "mel_loss": mel_loss,
            "gate_loss": gate_loss,
        }


# =============================================================================
# Optional smoke test
# =============================================================================

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_model_smoke_test() -> None:
    device = _get_device()
    model = Tacotron2().to(device)

    B = 2
    T_text = 30
    T_mel = 100

    text = torch.randint(0, N_SYMBOLS, (B, T_text), device=device)
    text_lengths = torch.tensor([30, 24], dtype=torch.long, device=device)
    mel_input = torch.randn(B, T_mel, N_MELS, device=device)

    outputs = model(text=text, text_lengths=text_lengths, mel_input=mel_input)

    print("=" * 80)
    print("Tacotron2 smoke test")
    print("=" * 80)
    print(f"N_SYMBOLS          : {N_SYMBOLS}")
    print(f"N_MELS             : {N_MELS}")
    print(f"text.shape         : {tuple(text.shape)}")
    print(f"text_lengths       : {text_lengths.tolist()}")
    print(f"mel_input.shape    : {tuple(mel_input.shape)}")
    print(f"mel_before.shape   : {tuple(outputs['mel_before'].shape)}")
    print(f"mel_after.shape    : {tuple(outputs['mel_after'].shape)}")
    print(f"gate.shape         : {tuple(outputs['gate'].shape)}")
    print(f"alignments.shape   : {tuple(outputs['alignments'].shape)}")
    print("=" * 80)


if __name__ == "__main__":
    run_model_smoke_test()
