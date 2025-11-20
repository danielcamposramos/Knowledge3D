#!/usr/bin/env python3
"""
Benchmark ternary video codec on synthetic procedural frames.

Usage:
    python scripts/benchmark_ternary_video.py
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from knowledge3d.cranium.codecs.procedural_video import ProceduralVideoGenerator
from knowledge3d.cranium.codecs.ternary_video_codec import TernaryVideoCodec


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0**2) / mse)


def compute_encoded_size(encoded: Dict) -> int:
    quantized = np.asarray(encoded.get("quantized"))
    seed = np.asarray(encoded.get("seed"))
    return quantized.size + seed.size * 4


def print_table(results: List[Dict]) -> None:
    headers = ["Case", "Size (KB)", "Compressed (KB)", "Ratio", "Encode (ms)", "Decode (ms)", "PSNR (dB)"]
    row_fmt = "{:<12} | {:>10} | {:>15} | {:>6} | {:>11} | {:>11} | {:>9}"
    print("Ternary Video Codec Benchmark Results")
    print("=" * 78)
    print(row_fmt.format(*headers))
    print("-" * 78)
    for r in results:
        print(
            row_fmt.format(
                r["name"],
                f"{r['original_kb']:.1f}",
                f"{r['compressed_kb']:.1f}",
                f"{r['ratio']:.1f}",
                f"{r['encode_ms']:.1f}",
                f"{r['decode_ms']:.1f}",
                f"{r['psnr_db']:.1f}",
            )
        )


def benchmark() -> List[Dict]:
    codec = TernaryVideoCodec(width=128, height=128)
    generator = ProceduralVideoGenerator(width=128, height=128)
    rng = np.random.default_rng(0)

    seeds = [rng.standard_normal(64).astype(np.float32) for _ in range(3)]
    names = ["pattern_a", "pattern_b", "pattern_c"]
    results: List[Dict] = []

    for name, seed in zip(names, seeds):
        frame = generator.generate_frame(seed, time_param=0.25)

        start = time.perf_counter()
        encoded = codec.encode(frame, seed=seed)
        encode_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        decoded = codec.decode(encoded)
        decode_ms = (time.perf_counter() - start) * 1000

        original_size = frame.size  # bytes approximated (uint8).
        compressed_size = compute_encoded_size(encoded)
        ratio = original_size / compressed_size if compressed_size else float("inf")
        psnr = compute_psnr(frame, decoded)

        results.append(
            {
                "name": name,
                "original_kb": original_size / 1024,
                "compressed_kb": compressed_size / 1024,
                "ratio": ratio,
                "encode_ms": encode_ms,
                "decode_ms": decode_ms,
                "psnr_db": psnr,
            }
        )
    return results


if __name__ == "__main__":
    results = benchmark()
    print_table(results)
