#!/usr/bin/env python3
"""
Batch Math Symbol Training

Trains all registered math symbols (∑, ∫, ∂, ∇, α, β, etc.) and stores them
in the Math Galaxy for later symlink reference during PDF ingestion.

This script leverages the existing train_atomic_character.py infrastructure
but orchestrates batch training of all ~121 math symbols from math_symbols_registry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge3d.cranium.math_symbols_registry import (
    CALCULUS,
    GREEK_ALL,
    SET_THEORY,
    LOGIC,
    ARROWS,
    RELATIONS,
    OPERATORS,
    GEOMETRY,
)
from knowledge3d.cranium.math_galaxy import MathGalaxy
from scripts.train_atomic_character import train_single_character


# Priority categories (train these first - most common in advanced math)
PRIORITY_SYMBOLS = {
    "high": list("∑∫∂∇∆∏√∞±αβγδεθλμπσω"),  # Calculus + common Greek
    "medium": list("∈∉⊂⊃⊆⊇∪∩∅∀∃∧∨¬⇒⇔"),  # Set theory + logic
    "low": ARROWS + RELATIONS + OPERATORS + GEOMETRY,  # Extended symbols
}


def collect_all_math_symbols() -> Dict[str, List[str]]:
    """
    Collect all registered math symbols organized by priority.

    Returns:
        Dictionary mapping priority level to list of symbols
    """
    all_symbols = {
        "high": [],
        "medium": [],
        "low": [],
    }

    # Collect unique symbols from all categories
    seen = set()

    # High priority: Calculus + Greek
    for sym in PRIORITY_SYMBOLS["high"]:
        if sym not in seen:
            all_symbols["high"].append(sym)
            seen.add(sym)

    # Medium priority: Set theory + Logic
    for sym in PRIORITY_SYMBOLS["medium"]:
        if sym not in seen:
            all_symbols["medium"].append(sym)
            seen.add(sym)

    # Low priority: Remaining symbols from all categories
    for category in [SET_THEORY, LOGIC, ARROWS, RELATIONS, OPERATORS, GEOMETRY, CALCULUS, GREEK_ALL]:
        for sym in category:
            if sym not in seen:
                all_symbols["low"].append(sym)
                seen.add(sym)

    return all_symbols


def train_math_symbols(
    priority: str = "high",
    learning_rate: float = 0.5,
    epochs: int = 1500,
    n_fonts: int = 0,
    fc_only: bool = False,
    max_epochs: int = 3000,
    skip_existing: bool = True,
) -> Dict[str, object]:
    """
    Train all math symbols at specified priority level.

    Args:
        priority: Priority level ("high", "medium", "low", or "all")
        learning_rate: Initial learning rate (plateau scheduler adjusts)
        epochs: Target epochs per symbol
        n_fonts: Number of fonts to use (0 = all available)
        fc_only: Train only FC layer (freeze CNN)
        max_epochs: Maximum epochs if extension needed
        skip_existing: Skip symbols with existing checkpoints

    Returns:
        Training summary statistics
    """
    print("=" * 80)
    print("BATCH MATH SYMBOL TRAINING")
    print("=" * 80)

    # Initialize Math Galaxy for storage
    math_galaxy = MathGalaxy()
    print(f"Math Galaxy initialized: {math_galaxy.root}")
    print()

    # Collect symbols by priority
    symbol_groups = collect_all_math_symbols()

    if priority == "all":
        symbols_to_train = (
            symbol_groups["high"] +
            symbol_groups["medium"] +
            symbol_groups["low"]
        )
    else:
        symbols_to_train = symbol_groups.get(priority, [])

    print(f"Training priority: {priority}")
    print(f"Total symbols to train: {len(symbols_to_train)}")
    print()

    # Check existing checkpoints
    checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars")
    existing_symbols = []
    if skip_existing and checkpoint_dir.exists():
        for sym in symbols_to_train:
            char_code = ord(sym)
            weights_path = checkpoint_dir / f"char_{char_code}_{sym}_weights.npz"
            if weights_path.exists():
                existing_symbols.append(sym)

        if existing_symbols:
            print(f"Found {len(existing_symbols)} existing checkpoints (will skip)")
            print(f"Symbols to train: {len(symbols_to_train) - len(existing_symbols)}")
            print()

    # Training statistics
    results = {
        "total": len(symbols_to_train),
        "skipped": 0,
        "trained": 0,
        "failed": [],
        "accuracies": {},
    }

    # Train each symbol
    for idx, symbol in enumerate(symbols_to_train, 1):
        print()
        print("=" * 80)
        print(f"SYMBOL {idx}/{len(symbols_to_train)}: '{symbol}' (U+{ord(symbol):04X})")
        print("=" * 80)

        # Skip if checkpoint exists
        if skip_existing and symbol in existing_symbols:
            print(f"✓ Checkpoint exists for '{symbol}', skipping...")
            results["skipped"] += 1
            continue

        try:
            result = train_single_character(
                target_char=symbol,
                learning_rate=learning_rate,
                n_epochs=epochs,
                n_fonts=n_fonts,
                fc_only=fc_only,
                max_epochs=max_epochs,
                compressor=None,  # Math Galaxy has its own storage
                galaxy=None,
                use_procedural=True,  # Use GPU procedural rasterizer
            )

            # Store in Math Galaxy
            char_code = result["char_id"]
            embeddings = result["embeddings"]
            canonical_embedding = embeddings.mean(axis=0).astype(np.float32)

            math_galaxy.store_symbol(symbol, canonical_embedding)

            results["trained"] += 1
            results["accuracies"][symbol] = result["best_accuracy"]

            print(f"✓ Symbol '{symbol}' trained and stored in Math Galaxy")
            print(f"  Accuracy: {result['best_accuracy'] * 100:.2f}%")
            print(f"  Embeddings: {embeddings.shape}")

        except Exception as exc:
            print(f"✗ Failed to train symbol '{symbol}': {exc}")
            results["failed"].append((symbol, str(exc)))

    # Print summary
    print()
    print("=" * 80)
    print("BATCH TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total symbols: {results['total']}")
    print(f"Skipped (existing): {results['skipped']}")
    print(f"Trained: {results['trained']}")
    print(f"Failed: {len(results['failed'])}")

    if results["accuracies"]:
        accuracies = list(results["accuracies"].values())
        print()
        print(f"Accuracy statistics:")
        print(f"  Mean: {np.mean(accuracies) * 100:.2f}%")
        print(f"  Min: {np.min(accuracies) * 100:.2f}%")
        print(f"  Max: {np.max(accuracies) * 100:.2f}%")

    if results["failed"]:
        print()
        print("Failed symbols:")
        for sym, error in results["failed"]:
            print(f"  '{sym}' (U+{ord(sym):04X}): {error}")

    print()
    print(f"Math Galaxy storage: {math_galaxy.symbols_dir}")
    print("=" * 80)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch train all math symbols for Math Galaxy"
    )
    parser.add_argument(
        "--priority",
        type=str,
        default="high",
        choices=["high", "medium", "low", "all"],
        help="Priority level (default: high - calculus + common Greek)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        help="Initial learning rate (default: 0.5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="Target epochs per symbol (default: 1500)",
    )
    parser.add_argument(
        "--fonts",
        type=int,
        default=0,
        help="Number of fonts to use (0 = all available, default: 0)",
    )
    parser.add_argument(
        "--fc-only",
        action="store_true",
        help="Train only FC layer (freeze CNN)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=3000,
        help="Maximum epochs if extension needed (default: 3000)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Retrain symbols even if checkpoints exist",
    )

    args = parser.parse_args()

    results = train_math_symbols(
        priority=args.priority,
        learning_rate=args.lr,
        epochs=args.epochs,
        n_fonts=args.fonts,
        fc_only=args.fc_only,
        max_epochs=args.max_epochs,
        skip_existing=not args.no_skip_existing,
    )

    # Exit code: 0 if all succeeded, 1 if any failures
    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()
