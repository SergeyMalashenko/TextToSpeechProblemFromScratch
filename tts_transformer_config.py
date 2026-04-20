#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import hyperparams as hp


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


@dataclass
class TransformerTacotronConfig:
    # Core data dimensions
    n_mels: int = field(default_factory=get_n_mels)
    n_symbols: int = field(default_factory=get_n_symbols)
    outputs_per_step: int = field(default_factory=get_outputs_per_step)

    # Text embedding
    symbols_embedding_dim: int = field(
        default_factory=lambda: int(hp_get("symbols_embedding_dim", hp_get("embedding_size", 512)))
    )

    # Transformer backbone
    transformer_d_model: int = field(default_factory=lambda: int(hp_get("transformer_d_model", 256)))
    transformer_nhead: int = field(default_factory=lambda: int(hp_get("transformer_nhead", 4)))
    transformer_encoder_layers: int = field(
        default_factory=lambda: int(hp_get("transformer_encoder_layers", 4))
    )
    transformer_decoder_layers: int = field(
        default_factory=lambda: int(hp_get("transformer_decoder_layers", 4))
    )
    transformer_ffn_dim: int = field(default_factory=lambda: int(hp_get("transformer_ffn_dim", 1024)))
    transformer_dropout: float = field(default_factory=lambda: float(hp_get("transformer_dropout", 0.1)))

    # Tacotron-like prenet/postnet
    prenet_dims: List[int] = field(default_factory=lambda: list(hp_get("prenet_dims", [256, 256])))
    postnet_embedding_dim: int = field(default_factory=lambda: int(hp_get("postnet_embedding_dim", 512)))
    postnet_kernel_size: int = field(default_factory=lambda: int(hp_get("postnet_kernel_size", 5)))
    postnet_n_convolutions: int = field(default_factory=lambda: int(hp_get("postnet_n_convolutions", 5)))

    # Decoding
    max_decoder_steps: int = field(default_factory=lambda: int(hp_get("max_decoder_steps", 1000)))
    gate_threshold: float = field(default_factory=lambda: float(hp_get("gate_threshold", 0.5)))

    # Positional encoding limits
    max_text_positions: int = field(default_factory=lambda: int(hp_get("max_text_positions", 2048)))

    def validate(self) -> None:
        if self.outputs_per_step != 1:
            raise ValueError(
                "This Transformer Tacotron configuration currently supports outputs_per_step=1 only."
            )

        if self.transformer_d_model % self.transformer_nhead != 0:
            raise ValueError(
                f"transformer_d_model ({self.transformer_d_model}) must be divisible by "
                f"transformer_nhead ({self.transformer_nhead})."
            )

        if len(self.prenet_dims) == 0:
            raise ValueError("prenet_dims must not be empty.")

        if self.postnet_n_convolutions < 2:
            raise ValueError("postnet_n_convolutions must be >= 2.")

        if self.max_decoder_steps <= 0:
            raise ValueError("max_decoder_steps must be positive.")

        if self.max_text_positions <= 0:
            raise ValueError("max_text_positions must be positive.")

    def pretty_print(self) -> None:
        print("=" * 80)
        print("TransformerTacotronConfig")
        print("=" * 80)
        for key, value in self.__dict__.items():
            print(f"{key:28s}: {value}")
        print("=" * 80)


def build_transformer_tacotron_config() -> TransformerTacotronConfig:
    cfg = TransformerTacotronConfig()
    cfg.validate()
    return cfg


if __name__ == "__main__":
    cfg = build_transformer_tacotron_config()
    cfg.pretty_print()
