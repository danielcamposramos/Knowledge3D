#!/usr/bin/env python3
"""
Train all 62 base characters using the GPU-sovereign atomic pipeline.

Requires GPU sovereignty across spatial pooling, Matryoshka projection, and
RPN trigram embeddings (no CPU fallbacks).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.train_atomic_character import train_single_character


BASE_CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
CHECKPOINT_DIR = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars")


def check_embedding_file(char: str) -> Dict[str, float]:
    """Validate saved embedding file for NaN/Inf and return stats."""
    char_code = ord(char)
    path = CHECKPOINT_DIR / f"char_{char_code}_{char}_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found for '{char}' at {path}")

    data = np.load(path)
    embeddings = data["embeddings"]
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Non-finite values detected in embeddings for '{char}'.")
    return {
        "count": float(embeddings.shape[0]),
        "dim": float(embeddings.shape[1]),
        "min": float(embeddings.min(initial=0.0)),
        "max": float(embeddings.max(initial=0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all 62 base characters with GPU-sovereign pipeline.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate per character.")
    parser.add_argument("--epochs", type=int, default=1500, help="Epochs per character (default: 1500).")
    parser.add_argument(
        "--fonts",
        type=int,
        default=0,
        help="Font samples per character (0 uses all fonts declared in font_db.pkl).",
    )
    parser.add_argument("--fc-only", action="store_true", help="Train only the final FC layer.")
    args = parser.parse_args()

    print("=" * 80)
    print("GPU-SOVEREIGN ATOMIC CHARACTER TRAINING")
    print("=" * 80)
    print(f"Characters: {len(BASE_CHARACTERS)} (A-Z, a-z, 0-9)")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    font_desc = "all available fonts" if args.fonts == 0 else str(args.fonts)
    print(f"Fonts per character: {font_desc}")
    print(f"Mode: {'FC-only' if args.fc_only else 'Full CNN fine-tuning'}")
    print("=" * 80)

    results: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    start_time = time.time()

    for idx, char in enumerate(BASE_CHARACTERS, start=1):
        print(f"\n[{idx}/{len(BASE_CHARACTERS)}] Training character '{char}'")
        try:
            result = train_single_character(
                target_char=char,
                learning_rate=args.lr,
                n_epochs=args.epochs,
                n_fonts=args.fonts,
                fc_only=args.fc_only,
            )
            stats = check_embedding_file(char)
            status = "SUCCESS" if result["best_accuracy"] >= 0.85 else "LOW_ACCURACY"
            results.append(
                {
                    "char": char,
                    "accuracy": float(result["best_accuracy"]),
                    "embeddings": stats["count"],
                    "dimension": stats["dim"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "status": status,
                }
            )
            print(
                f"✓ '{char}' trained | accuracy={result['best_accuracy'] * 100:.2f}% "
                f"| embeddings={stats['count']:.0f} dim={stats['dim']:.0f} "
                f"| status={status}"
            )
        except Exception as exc:
            print(f"✗ Training failed for '{char}': {exc}")
            failures.append({"char": char, "error": str(exc)})

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    successes = len(results)
    print(f"Characters trained successfully: {successes}/{len(BASE_CHARACTERS)}")

    if results:
        accuracies = [item["accuracy"] for item in results]
        avg_accuracy = float(np.mean(accuracies) * 100.0)
        success_85 = sum(1 for item in results if item["status"] == "SUCCESS")
        print(f"Average accuracy: {avg_accuracy:.2f}%")
        print(f"≥85% accuracy: {success_85}/{successes}")
        print(f"Embedding dimension: {results[0]['dimension']:.0f}")
    else:
        print("Average accuracy: N/A (no successful trainings)")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f" - '{failure['char']}': {failure['error']}")

    print(f"\nElapsed time: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    main()
