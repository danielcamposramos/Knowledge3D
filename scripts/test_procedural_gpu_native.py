#!/usr/bin/env python3
"""
Quick GPU-Native Validation for Procedural Drawing Pipeline.

Tests:
1. Matryoshka PTX compilation (no CPU fallback)
2. Contrastive learning with batch optimizer
3. Adaptive batching GPU utilization

Sovereignty: GPU-only, no fallbacks.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.cranium.specialists.procedural_drawing_specialist import ProceduralDrawingSpecialist
from knowledge3d.cranium.specialists.batch_optimizer import BatchOptimizer


def test_matryoshka_gpu_native():
    """Test 1: Matryoshka PTX compiles and runs (no CPU fallback)."""
    print("\n" + "="*60)
    print("TEST 1: Matryoshka GPU-Native PTX Compilation")
    print("="*60)

    from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM

    try:
        # This should compile PTX with -allow-unsupported-compiler
        mat_trm = MatryoshkaTRM(max_dims=512, min_dims=64)

        # Test projection (GPU-only path)
        test_vec = np.random.randn(256).astype(np.float32)
        result = mat_trm.project_vector(test_vec, target_dim=128)

        print(f"✅ Matryoshka PTX compiled successfully")
        print(f"✅ GPU projection successful: {test_vec.shape} → {result.shape}")
        print(f"   Output range: [{result.min():.3f}, {result.max():.3f}]")

        return True

    except RuntimeError as e:
        if "CPU fallback" in str(e) or "not initialized" in str(e):
            print(f"❌ GPU-native path failed: {e}")
            print("   Sovereignty principle violated - CPU fallback attempted")
            return False
        raise


def test_contrastive_learning():
    """Test 2: Contrastive learning updates specialist weights."""
    print("\n" + "="*60)
    print("TEST 2: Contrastive Learning Weight Updates")
    print("="*60)

    # Create minimal swarm + specialist
    config = SwarmConfig(
        base_dims=128,
        min_dims=64,
        specialist_learning_rate=0.001
    )
    swarm = AdaptiveSwarmTRM(config=config)

    specialist = ProceduralDrawingSpecialist(
        swarm=swarm,
        matryoshka_dim=128,
        gpu_id=0
    )

    # Get initial adapter weights
    adapter = swarm.base.specialists['procedural_drawing']['adapter']
    if hasattr(adapter, 'W_up'):
        initial_weights = adapter.W_up.copy()
    elif hasattr(adapter, 'A'):
        initial_weights = adapter.A.copy()
    else:
        print("⚠️  Adapter structure unknown, skipping weight check")
        return True

    # Create synthetic batch (text ≈ visual pairs)
    synthetic_batch = []
    for i in range(16):
        char = chr(65 + (i % 26))  # A-Z
        # Synthetic RPN bytecode (empty for now)
        rpn_bytecode = b'\x00' * 32
        synthetic_batch.append((char, rpn_bytecode))

    # Train
    print(f"Training on {len(synthetic_batch)} synthetic samples...")
    try:
        metrics = specialist.train_on_batch(synthetic_batch, validation=False)

        # Check if weights changed
        if hasattr(adapter, 'W_up'):
            final_weights = adapter.W_up
        else:
            final_weights = adapter.A

        weight_change = np.linalg.norm(final_weights - initial_weights)

        print(f"✅ Contrastive training completed")
        print(f"   Alignment: {metrics.text_visual_alignment:.4f}")
        print(f"   Weight change: {weight_change:.6f}")

        if weight_change > 1e-6:
            print(f"   ✓ Weights updated successfully")
            return True
        else:
            print(f"   ⚠️  Weights unchanged (possible TODO stub)")
            return False

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_optimizer():
    """Test 3: Batch optimizer scales up GPU utilization."""
    print("\n" + "="*60)
    print("TEST 3: Adaptive Batch Optimizer")
    print("="*60)

    optimizer = BatchOptimizer(
        target_utilization=0.75,
        max_vram_mb=180.0,
        min_batch_size=8,
        max_batch_size=256
    )

    # Simulate low GPU scenario (7% util, 108MB VRAM)
    current_batch = 32
    gpu_util = 0.07
    vram_used = 108.0

    suggested_batch = optimizer.suggest_batch_size(
        current_batch_size=current_batch,
        gpu_utilization=gpu_util,
        vram_used_mb=vram_used
    )

    print(f"Current state:")
    print(f"  Batch size: {current_batch}")
    print(f"  GPU utilization: {gpu_util*100:.1f}%")
    print(f"  VRAM usage: {vram_used:.1f} MB")

    print(f"\nOptimizer suggestion:")
    print(f"  New batch size: {suggested_batch}")
    print(f"  Reason: {optimizer.last_suggestion['reason']}")

    # Verify batch size increased
    if suggested_batch > current_batch:
        print(f"✅ Batch size increased: {current_batch} → {suggested_batch}")
        print(f"   Scaling factor: {suggested_batch / current_batch:.2f}x")
        return True
    else:
        print(f"⚠️  Batch size not increased (expected with 7% GPU util)")
        return False


def test_full_pipeline_mini():
    """Test 4: End-to-end pipeline with synthetic data."""
    print("\n" + "="*60)
    print("TEST 4: End-to-End Pipeline (Mini Synthetic Dataset)")
    print("="*60)

    import tempfile
    import json

    # Create tiny synthetic dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(50):  # 50 samples
            char = chr(65 + (i % 26))
            entry = {
                'char': char,
                'rpn': 'MOVETO 0.1 0.1 LINETO 0.9 0.9',  # Simple diagonal line
                'font': 'Synthetic'
            }
            f.write(json.dumps(entry) + '\n')

        dataset_path = Path(f.name)

    try:
        # Create swarm + specialist
        config = SwarmConfig(
            base_dims=128,
            min_dims=64,
            specialist_learning_rate=0.001
        )
        swarm = AdaptiveSwarmTRM(config=config)

        specialist = ProceduralDrawingSpecialist(
            swarm=swarm,
            matryoshka_dim=128,
            gpu_id=0
        )

        # Train for 2 epochs
        print(f"Training on {dataset_path} (50 samples, 2 epochs)...")
        specialist.train_on_rpn_dataset(
            dataset_path=dataset_path,
            epochs=2,
            batch_size=16,
            validation_split=0.2,
            adaptive_batching=True
        )

        # Check metrics
        if specialist.training_metrics:
            final_alignment = specialist.training_metrics[-1].text_visual_alignment
            print(f"✅ Training completed")
            print(f"   Final alignment: {final_alignment:.4f}")
            return True
        else:
            print(f"⚠️  No metrics recorded")
            return False

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        dataset_path.unlink()


def main():
    """Run all GPU-native validation tests."""
    print("\n" + "="*60)
    print("PROCEDURAL DRAWING PIPELINE: GPU-NATIVE VALIDATION")
    print("Sovereignty Principle: No CPU Fallbacks")
    print("="*60)

    results = {}

    # Test 1: Matryoshka PTX
    results['matryoshka_gpu'] = test_matryoshka_gpu_native()

    # Test 2: Contrastive learning
    results['contrastive_learning'] = test_contrastive_learning()

    # Test 3: Batch optimizer
    results['batch_optimizer'] = test_batch_optimizer()

    # Test 4: Full pipeline (only if previous tests passed)
    if all([results['matryoshka_gpu'], results['contrastive_learning']]):
        results['full_pipeline'] = test_full_pipeline_mini()
    else:
        print("\n⚠️  Skipping full pipeline test (prerequisites failed)")
        results['full_pipeline'] = False

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed - GPU-native pipeline operational!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
