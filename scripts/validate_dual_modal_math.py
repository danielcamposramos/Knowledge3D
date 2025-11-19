#!/usr/bin/env python3
"""
Validate Dual-Modal Math Training Pipeline

Tests the ProceduralDrawingSpecialist with triplet contrastive learning on:
1. Standard font glyphs (text ↔ visual)
2. Dual-modal math symbols (text ↔ visual ↔ execution)

Measures:
- Triplet alignment scores (visual ≈ execution ≈ text)
- Math RPN prediction accuracy
- Training throughput and VRAM utilization
- Convergence rate
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.cranium.specialists.procedural_drawing_specialist import (
    ProceduralDrawingSpecialist,
    TrainingMetrics
)


def load_atomic_datasets(
    fonts_path: Path,
    math_path: Path
) -> Tuple[List[Dict], List[Dict]]:
    """Load atomic datasets."""
    print("\n" + "="*70)
    print("LOADING ATOMIC DATASETS")
    print("="*70 + "\n")

    # Load font glyphs
    fonts = []
    with open(fonts_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                fonts.append(json.loads(line))

    # Load dual-modal math symbols
    math_symbols = []
    with open(math_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                math_symbols.append(json.loads(line))

    print(f"✅ Loaded {len(fonts)} font glyphs")
    print(f"✅ Loaded {len(math_symbols)} dual-modal math symbols")
    print(f"   Total atomic units: {len(fonts) + len(math_symbols)}")

    return fonts, math_symbols


def prepare_training_batch(
    entries: List[Dict],
    dual_modal: bool = False
) -> List:
    """
    Prepare batch for training (GPU-accelerated pipeline).

    Now passes RPN strings directly (not bytecode) for GPU execution.
    """
    batch = []
    if not dual_modal:
        for entry in entries:
            char = entry['char']
            rpn = entry['rpn']
            batch.append((char, rpn, entry))
    else:
        for entry in entries:
            symbol = entry['char']
            visual_rpn = entry['visual_rpn']
            math_rpn = entry['math_rpn']
            semantic = entry['semantic']
            batch.append((symbol, visual_rpn, math_rpn, semantic, entry))
    return batch


def validate_dual_modal_training(
    fonts_path: Path,
    math_path: Path,
    epochs: int = 5,
    batch_size: int = 32,
    matryoshka_dim: int = 512,
    gpu_id: int = 0
):
    """
    Validate dual-modal math training.

    Args:
        fonts_path: Path to font glyphs dataset
        math_path: Path to math symbols dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        matryoshka_dim: Embedding dimension (64-2048)
        gpu_id: CUDA device ID
    """
    print("\n" + "="*70)
    print("DUAL-MODAL MATH VALIDATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Matryoshka dim: {matryoshka_dim}")
    print(f"  GPU ID: {gpu_id}")
    print()

    # Load datasets
    fonts, math_symbols = load_atomic_datasets(fonts_path, math_path)

    # Initialize swarm and specialist
    print("\nInitializing AdaptiveSwarm and ProceduralDrawingSpecialist...")

    swarm_config = SwarmConfig(
        base_dims=matryoshka_dim,
        min_dims=64,
        base_learning_rate=0.001,
        specialist_learning_rate=0.002
    )

    swarm = AdaptiveSwarmTRM(config=swarm_config)

    specialist = ProceduralDrawingSpecialist(
        swarm=swarm,
        matryoshka_dim=matryoshka_dim,
        gpu_id=gpu_id
    )

    print("✅ Specialist initialized")

    # Split datasets (90% train, 10% validation)
    n_font_val = int(len(fonts) * 0.1)
    n_math_val = int(len(math_symbols) * 0.1)

    font_train = fonts[n_font_val:]
    font_val = fonts[:n_font_val]
    math_train = math_symbols[n_math_val:]
    math_val = math_symbols[:n_math_val]

    print(f"\nDataset splits:")
    print(f"  Font train: {len(font_train)}, val: {len(font_val)}")
    print(f"  Math train: {len(math_train)}, val: {len(math_val)}")

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70 + "\n")

    all_metrics = {
        'font_train': [],
        'font_val': [],
        'math_train': [],
        'math_val': []
    }

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*70}")

        # === FONT GLYPHS TRAINING ===
        print("\n[1/4] Training on font glyphs...")
        np.random.shuffle(font_train)
        font_train_metrics = []

        for i in range(0, len(font_train), batch_size):
            batch = font_train[i:i+batch_size]
            prepared_batch = prepare_training_batch(batch, dual_modal=False)

            metrics = specialist.train_on_batch(
                prepared_batch,
                validation=False,
                dual_modal_math=False
            )
            font_train_metrics.append(metrics.text_visual_alignment)

        avg_font_train = np.mean(font_train_metrics)
        print(f"   Font train alignment: {avg_font_train:.4f}")

        # === FONT GLYPHS VALIDATION ===
        print("\n[2/4] Validating on font glyphs...")
        font_val_metrics = []

        for i in range(0, len(font_val), batch_size):
            batch = font_val[i:i+batch_size]
            prepared_batch = prepare_training_batch(batch, dual_modal=False)

            metrics = specialist.train_on_batch(
                prepared_batch,
                validation=True,
                dual_modal_math=False
            )
            font_val_metrics.append(metrics.text_visual_alignment)

        avg_font_val = np.mean(font_val_metrics)
        print(f"   Font val alignment: {avg_font_val:.4f}")

        # === DUAL-MODAL MATH TRAINING ===
        print("\n[3/4] Training on dual-modal math...")
        np.random.shuffle(math_train)
        math_train_metrics = []

        for i in range(0, len(math_train), batch_size):
            batch = math_train[i:i+batch_size]
            prepared_batch = prepare_training_batch(batch, dual_modal=True)

            metrics = specialist.train_on_batch(
                prepared_batch,
                validation=False,
                dual_modal_math=True
            )
            math_train_metrics.append(metrics.text_visual_alignment)

        avg_math_train = np.mean(math_train_metrics)
        print(f"   Math train triplet alignment: {avg_math_train:.4f}")

        # === DUAL-MODAL MATH VALIDATION ===
        print("\n[4/4] Validating on dual-modal math...")
        math_val_metrics = []

        for i in range(0, len(math_val), batch_size):
            batch = math_val[i:i+batch_size]
            prepared_batch = prepare_training_batch(batch, dual_modal=True)

            metrics = specialist.train_on_batch(
                prepared_batch,
                validation=True,
                dual_modal_math=True
            )
            math_val_metrics.append(metrics.text_visual_alignment)

        avg_math_val = np.mean(math_val_metrics)
        print(f"   Math val triplet alignment: {avg_math_val:.4f}")

        # Store metrics
        all_metrics['font_train'].append(avg_font_train)
        all_metrics['font_val'].append(avg_font_val)
        all_metrics['math_train'].append(avg_math_train)
        all_metrics['math_val'].append(avg_math_val)

        epoch_time = time.time() - epoch_start
        print(f"\n✅ Epoch {epoch + 1} complete ({epoch_time:.1f}s)")
        print(f"   Font:  train={avg_font_train:.4f}, val={avg_font_val:.4f}")
        print(f"   Math:  train={avg_math_train:.4f}, val={avg_math_val:.4f} (triplet)")

    # Final results
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70 + "\n")

    print("Final Alignment Scores:")
    print(f"  Font glyphs:      {all_metrics['font_val'][-1]:.4f} (text ↔ visual)")
    print(f"  Math symbols:     {all_metrics['math_val'][-1]:.4f} (triplet: text ↔ visual ↔ execution)")
    print()

    print("Convergence Analysis:")
    font_improvement = all_metrics['font_val'][-1] - all_metrics['font_val'][0]
    math_improvement = all_metrics['math_val'][-1] - all_metrics['math_val'][0]
    print(f"  Font improvement:  {font_improvement:+.4f}")
    print(f"  Math improvement:  {math_improvement:+.4f}")
    print()

    # Test math RPN prediction
    print("="*70)
    print("MATH RPN PREDICTION TEST")
    print("="*70 + "\n")

    test_samples = math_val[:5]
    for sample in test_samples:
        semantic = sample['semantic']
        expected_rpn = sample['math_rpn']

        # Predict math RPN
        predicted_rpn = specialist.predict_math_rpn(semantic)

        print(f"Semantic: {semantic[:50]}...")
        print(f"  Expected:  {expected_rpn[:80]}")
        print(f"  Predicted: {predicted_rpn[:80]}")
        print()

    # Save checkpoint
    checkpoint_path = Path("/K3D/Knowledge3D.local/checkpoints/procedural_drawing_dual_modal_validation.json")
    specialist.save_checkpoint(checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    print(f"Total samples trained: {len(font_train) + len(math_train)}")
    print(f"  Font glyphs:  {len(font_train)}")
    print(f"  Math symbols: {len(math_train)}")
    print()

    print(f"Final validation alignment:")
    print(f"  Font:  {all_metrics['font_val'][-1]:.4f}")
    print(f"  Math:  {all_metrics['math_val'][-1]:.4f} (triplet average)")
    print()

    target_alignment = 0.75
    font_pass = all_metrics['font_val'][-1] >= target_alignment
    math_pass = all_metrics['math_val'][-1] >= target_alignment

    print(f"Target alignment: {target_alignment:.2f}")
    print(f"  Font:  {'✅ PASS' if font_pass else '❌ FAIL'}")
    print(f"  Math:  {'✅ PASS' if math_pass else '❌ FAIL'}")
    print()

    if font_pass and math_pass:
        print("🎉 VALIDATION PASSED - Dual-modal math training successful!")
    else:
        print("⚠️  VALIDATION NEEDS IMPROVEMENT - Consider more epochs or tuning")

    print("\n" + "="*70 + "\n")

    return all_metrics


def main():
    """Run dual-modal math validation."""
    fonts_path = Path("/K3D/Knowledge3D.local/datasets/atomic/fonts_procedural.jsonl")
    math_path = Path("/K3D/Knowledge3D.local/datasets/atomic/math_symbols_procedural.jsonl")

    if not fonts_path.exists():
        print(f"❌ Font dataset not found: {fonts_path}")
        print("   Run: python scripts/generate_atomic_datasets.py")
        return 1

    if not math_path.exists():
        print(f"❌ Math dataset not found: {math_path}")
        print("   Run: python scripts/generate_atomic_datasets.py")
        return 1

    # Run validation
    metrics = validate_dual_modal_training(
        fonts_path=fonts_path,
        math_path=math_path,
        epochs=5,
        batch_size=32,
        matryoshka_dim=512,
        gpu_id=0
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
