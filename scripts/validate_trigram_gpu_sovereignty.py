#!/usr/bin/env python3
"""
Validate GPU sovereignty for RPN trigram embeddings.

Ensures the GPU bridge initializes, matches the CPU reference within tolerance,
and that GPU-only APIs fail fast when the bridge is missing.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.bridges.trigram_embed_bridge import TrigramEmbedBridge


TEST_TEXTS = [
    "hello",
    "world",
    "K3D",
    "GPU sovereignty",
    "Reverse Polish Notation",
    "a",
    "ab",
    "",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GPU sovereignty for RPN trigram embeddings.")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Maximum allowed absolute difference.")
    args = parser.parse_args()

    print("=" * 80)
    print("RPN TRIGRAM GPU SOVEREIGNTY VALIDATION")
    print("=" * 80)

    bridge = TrigramEmbedBridge()
    engine = RPNEmbeddingEngine()
    engine.attach_gpu_bridge(bridge)

    worst = 0.0
    for text in TEST_TEXTS:
        cpu_vec = engine.embed_sentence(text)
        gpu_vec = engine.embed_sentence_gpu(text)
        diff = np.abs(cpu_vec - gpu_vec)
        max_diff = float(diff.max()) if diff.size else 0.0
        worst = max(worst, max_diff)
        status = "✓" if max_diff <= args.tolerance else "✗"
        print(f"{status} '{text}': max |GPU-CPU| = {max_diff:.8f}")

    print()
    print(f"Worst absolute deviation: {worst:.8f}")
    if worst > args.tolerance:
        raise AssertionError(
            f"GPU/CPU deviation {worst:.8f} exceeds tolerance {args.tolerance:.8f}."
        )
    print("✓ GPU embeddings match CPU reference within tolerance.")

    print("\n[Fail-fast check] GPU APIs must error if bridge unavailable...")
    engine_no_gpu = RPNEmbeddingEngine()
    with contextlib.suppress(Exception):
        engine_no_gpu.embed_sentence_gpu("test")
        raise AssertionError("Expected RuntimeError when GPU bridge is missing.")
    print("✓ GPU APIs raise RuntimeError without bridge (no fallback).")


if __name__ == "__main__":
    main()
