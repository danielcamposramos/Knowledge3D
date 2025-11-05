#!/usr/bin/env python3
"""Compare feature statistics between APOLLO patches and synthetic glyph renders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge
from scripts.train_atomic_character import (
    DeepSeekOCRModel,
    render_glyph_image,
    load_fonts_for_script,
)


def stats(name: str, samples: np.ndarray) -> None:
    samples = samples.reshape(samples.shape[0], -1)
    mean = float(samples.mean())
    std = float(samples.std())
    min_val = float(samples.min())
    max_val = float(samples.max())
    print(f"{name}:\n  mean={mean:.6f}  std={std:.6f}  min={min_val:.6f}  max={max_val:.6f}")


def collect_real_features(pdf_path: Path, page: int = 0) -> np.ndarray:
    bridge = PhaseGPDFIngestionBridge()
    result = bridge.ingest_pdf_page(pdf_path, page)
    detector = bridge.character_detector
    raw = getattr(detector, "debug_last_patch_raw_features", None)
    norm = getattr(detector, "debug_last_patch_features", None)
    if raw is None or norm is None:
        raise RuntimeError("Character detector did not expose debug features.")
    return raw, norm


def collect_synthetic_features(chars: List[str], fonts_per_char: int, image_size: int = 64) -> np.ndarray:
    model = DeepSeekOCRModel()
    features = []
    for ch in chars:
        script = "latin"
        fonts = load_fonts_for_script(script, fonts_per_char)
        for font_path in fonts:
            img = render_glyph_image(ch, str(font_path), size=image_size)
            if img is None:
                continue
            result = model.forward(img, cache_for_backward=True)
            cache = result.get("cache", {})
            feature_map = cache.get("conv3_out", result.get("feature_map"))
            if feature_map is None:
                continue
            vec = feature_map.mean(axis=(0, 1)).astype(np.float32)
            features.append(vec)
    if not features:
        raise RuntimeError("No synthetic features collected")
    return np.vstack(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare feature statistics for APOLLO vs synthetic glyphs")
    parser.add_argument("--pdf", type=str,
                        default="/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/APOLLO.PDF",
                        help="Path to APOLLO PDF")
    parser.add_argument("--fonts", type=int, default=5, help="Fonts per character for synthetic sampling")
    parser.add_argument("--chars", type=str,
                        default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                        help="Characters to sample for synthetic stats")
    parser.add_argument("--save", action="store_true", help="Save synthetic mean/std to disk")
    parser.add_argument("--output", type=str,
                        default="/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/feature_stats.npz",
                        help="Path to store synthetic stats when --save is used")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    print("Collecting real APOLLO features...")
    real_raw, real_norm = collect_real_features(pdf_path)
    print(f"Real samples: raw={real_raw.shape}, norm={real_norm.shape}")
    stats("Real raw", real_raw)
    stats("Real normalized", real_norm)

    subset_chars = list(dict.fromkeys(list(args.chars)))
    print("\nCollecting synthetic glyph features...")
    synthetic = collect_synthetic_features(subset_chars, args.fonts)
    print(f"Synthetic samples: {synthetic.shape}")
    stats("Synthetic", synthetic)

    real_raw_mean = float(real_raw.mean())
    synthetic_mean = float(synthetic.mean())
    real_raw_std = float(real_raw.std())
    synthetic_std = float(synthetic.std())
    print("\nMean difference (raw)", abs(real_raw_mean - synthetic_mean))
    print("Std difference (raw)", abs(real_raw_std - synthetic_std))

    if args.save:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mean_vec = synthetic.mean(axis=0)
        std_vec = synthetic.std(axis=0)
        std_vec = np.where(std_vec < 1e-6, 1.0, std_vec)
        np.savez(out_path, mean=mean_vec.astype(np.float32), std=std_vec.astype(np.float32))
        print(f"\nSaved synthetic feature stats to {out_path}")


if __name__ == "__main__":
    main()
