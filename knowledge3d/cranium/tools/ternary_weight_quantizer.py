"""
Ternary weight quantizer (post-training).

Quantizes float weights to {-1,0,+1} with configurable threshold, writes NPZ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


def quantize_to_ternary(weights: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Quantize float weights to {-1,0,+1}."""
    out = np.zeros_like(weights, dtype=np.int8)
    out[weights > threshold] = 1
    out[weights < -threshold] = -1
    return out


def quantize_file(src: Path, dst: Path, threshold: float = 0.05) -> Dict[str, float]:
    data = np.load(src)
    quantized = {}
    stats = {}
    for k, v in data.items():
        q = quantize_to_ternary(v, threshold=threshold)
        quantized[k] = q
        sparsity = float(np.mean(q == 0))
        stats[k] = sparsity
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **quantized)
    return {
        "source": str(src),
        "output": str(dst),
        "threshold": threshold,
        "avg_sparsity": float(np.mean(list(stats.values()))) if stats else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-training ternary weight quantizer")
    ap.add_argument("--input", required=True, help="Path to .npz weights")
    ap.add_argument("--output", required=True, help="Output .npz for ternary weights")
    ap.add_argument("--threshold", type=float, default=0.05, help="Threshold for quantization")
    args = ap.parse_args()

    result = quantize_file(Path(args.input), Path(args.output), threshold=args.threshold)
    print(f"Ternary quantization complete: {result}")


if __name__ == "__main__":
    main()
