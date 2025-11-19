#!/usr/bin/env python3
"""
Limited test of atomic knowledge formation.

Tests with 50 font chars + 50 math symbols to validate approach before full training.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM
from knowledge3d.cranium.specialists.procedural_drawing_specialist import ProceduralDrawingSpecialist


def load_limited_datasets():
    """Load small subsets for testing."""
    # Load font glyphs
    font_path = Path("/K3D/Knowledge3D.local/datasets/atomic/fonts_procedural.jsonl")
    math_path = Path("/K3D/Knowledge3D.local/datasets/atomic/math_symbols_procedural.jsonl")

    font_data = []
    with open(font_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:  # Limit to 50
                break
            font_data.append(json.loads(line))

    math_data = []
    with open(math_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:  # Limit to 50
                break
            math_data.append(json.loads(line))

    print(f"✅ Loaded {len(font_data)} font glyphs, {len(math_data)} math symbols")
    return font_data, math_data


def test_atomic_formation():
    """Test atomic knowledge formation with limited dataset."""
    print("\n" + "=" * 70)
    print("ATOMIC KNOWLEDGE FORMATION - Limited Test")
    print("=" * 70)

    # Initialize swarm and specialist
    print("\n[1/5] Initializing AdaptiveSwarm...")
    from knowledge3d.cranium.adaptive_swarm import SwarmConfig
    config = SwarmConfig(base_dims=512, min_dims=64)
    swarm = AdaptiveSwarmTRM(config=config)

    print("\n[2/5] Initializing ProceduralDrawingSpecialist...")
    specialist = ProceduralDrawingSpecialist(
        swarm=swarm,
        matryoshka_dim=512,
        gpu_id=0
    )

    # Load limited datasets
    print("\n[3/5] Loading limited datasets...")
    font_data, math_data = load_limited_datasets()

    # Split into train/val
    font_train = font_data[:40]
    font_val = font_data[40:50]
    math_train = math_data[:40]
    math_val = math_data[40:50]

    print(f"  Font: {len(font_train)} train, {len(font_val)} val")
    print(f"  Math: {len(math_train)} train, {len(math_val)} val")

    # Train for 1 epoch
    print("\n[4/5] Training atomic formation (1 epoch)...")
    print("\n  [Font Glyphs Training]")

    # Convert font data to (char, rpn_program) tuples
    font_batch = [(item['char'], item['rpn']) for item in font_train]
    font_metrics = specialist.train_on_batch(
        font_batch,
        validation=False,
        dual_modal_math=False
    )
    print(f"    Alignment: {font_metrics.text_visual_alignment:.4f}")

    print("\n  [Font Glyphs Validation]")
    font_val_batch = [(item['char'], item['rpn']) for item in font_val]
    font_val_metrics = specialist.train_on_batch(
        font_val_batch,
        validation=True,
        dual_modal_math=False
    )
    print(f"    Alignment: {font_val_metrics.text_visual_alignment:.4f}")

    print("\n  [Math Symbols Training]")
    # Math data: (char, visual_rpn, math_rpn, semantic)
    math_batch = [
        (item['char'], item['visual_rpn'], item.get('math_rpn', ''), item['semantic'])
        for item in math_train
    ]
    math_metrics = specialist.train_on_batch(
        math_batch,
        validation=False,
        dual_modal_math=True
    )
    print(f"    Alignment: {math_metrics.text_visual_alignment:.4f}")

    print("\n  [Math Symbols Validation]")
    math_val_batch = [
        (item['char'], item['visual_rpn'], item.get('math_rpn', ''), item['semantic'])
        for item in math_val
    ]
    math_val_metrics = specialist.train_on_batch(
        math_val_batch,
        validation=True,
        dual_modal_math=True
    )
    print(f"    Alignment: {math_val_metrics.text_visual_alignment:.4f}")

    # Check atomic units cache
    print("\n[5/5] Checking atomic units cache...")
    print(f"  Accumulated: {len(specialist.atomic_units)} atomic units")

    # Sample a few
    sample_chars = list(specialist.atomic_units.keys())[:5]
    for char in sample_chars:
        unit = specialist.atomic_units[char]
        print(f"\n  '{char}':")
        print(f"    Embedding shape: {unit['embedding'].shape}")
        print(f"    Visual RPN: {unit['visual_rpn'][:50]}...")
        print(f"    Math RPN: {unit['math_rpn']}")

    # Commit to ProceduralGalaxy
    print("\n[OPTIONAL] Committing to ProceduralGalaxy...")
    committed = specialist.commit_atomic_units_to_galaxy()
    print(f"  ✅ Committed {committed} atomic units")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Font glyphs:  train={font_metrics.text_visual_alignment:.4f}, val={font_val_metrics.text_visual_alignment:.4f}")
    print(f"  Math symbols: train={math_metrics.text_visual_alignment:.4f}, val={math_val_metrics.text_visual_alignment:.4f}")
    print(f"  Atomic units: {len(specialist.atomic_units)} stored")
    print(f"  Status: {'✅ PASS' if len(specialist.atomic_units) == 80 else '❌ FAIL'}")

    return specialist


if __name__ == "__main__":
    try:
        specialist = test_atomic_formation()
        print("\n✅ Limited test completed successfully!\n")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
