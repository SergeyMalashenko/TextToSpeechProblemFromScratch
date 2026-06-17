#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

import random
import torch
import os

from typing import Optional

def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    set_python_hash_seed: bool = True,
) -> None:
    """
    Fixes the main sources of randomness for reproducible experiments.

    Notes:
        - Full bitwise reproducibility on CUDA is not always guaranteed for
          recurrent layers and some GPU kernels.
        - This function avoids torch.use_deterministic_algorithms(True), because
          it may break training if the model uses a non-deterministic operation.
    """
    print("Hello world!")
    random.seed   (seed)
    np.random.seed(seed)

    torch.manual_seed         (seed)
    torch.cuda.manual_seed    (seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.deterministic = bool(deterministic)
    #torch.backends.cudnn.benchmark = bool(benchmark)


def seed_worker(worker_id: int) -> None:
    """
    Initializes DataLoader worker RNGs using PyTorch worker seed.
    """
    worker_seed = torch.initial_seed() % 2**32

    random.seed(worker_seed)

    if np is not None:
        np.random.seed(worker_seed)


def make_torch_generator(seed: Optional[int] = None, device: str = "cpu") -> torch.Generator:
    """
    Creates a torch.Generator with an optional fixed seed.
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator


# =============================================================================
# Smoke tests
# =============================================================================

def _smoke_test_torch_reproducibility() -> None:
    set_seed(42)
    x1 = torch.randn(10)

    set_seed(42)
    x2 = torch.randn(10)

    assert torch.equal(x1, x2), "torch.randn is not reproducible after set_seed"


def _smoke_test_python_random_reproducibility() -> None:
    set_seed(42)
    x1 = [random.random() for _ in range(10)]

    set_seed(42)
    x2 = [random.random() for _ in range(10)]

    assert x1 == x2, "python random is not reproducible after set_seed"


def _smoke_test_numpy_reproducibility() -> None:
    if np is None:
        print("  SKIP: numpy is not installed")
        return

    set_seed(42)
    x1 = np.random.rand(10)

    set_seed(42)
    x2 = np.random.rand(10)

    assert np.allclose(x1, x2), "numpy random is not reproducible after set_seed"


def _smoke_test_make_torch_generator() -> None:
    g1 = make_torch_generator(123)
    g2 = make_torch_generator(123)

    x1 = torch.randn(10, generator=g1)
    x2 = torch.randn(10, generator=g2)

    assert torch.equal(x1, x2), "make_torch_generator does not create reproducible generators"


def _smoke_test_seed_worker_runs() -> None:
    seed_worker(0)


def run_smoke_tests() -> None:
    print("Running tts_seed.py smoke tests...")
    _smoke_test_torch_reproducibility()
    print("  OK: torch reproducibility")
    _smoke_test_python_random_reproducibility()
    print("  OK: python random reproducibility")
    _smoke_test_numpy_reproducibility()
    if np is not None:
        print("  OK: numpy reproducibility")
    _smoke_test_make_torch_generator()
    print("  OK: make_torch_generator")
    _smoke_test_seed_worker_runs()
    print("  OK: seed_worker")
    print("All tts_seed.py smoke tests passed.")


if __name__ == "__main__":
    run_smoke_tests()
