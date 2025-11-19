#!/usr/bin/env python3
"""
Train Adaptive Swarm TRM

Trains self-updating base model and specialist adapters.

Training Modes:
1. Base-only: Train base model on general reasoning
2. Specialist-only: Train specific specialist on domain data
3. Base-first: Train base, then all specialists
4. Joint: Train base and specialists in parallel

Usage:
    # Train base model
    python scripts/train_adaptive_swarm.py --mode base \
        --dataset /path/to/general_samples.jsonl \
        --epochs 3

    # Train specialist
    python scripts/train_adaptive_swarm.py --mode specialist \
        --specialist ocr \
        --dataset /path/to/ocr_samples.jsonl \
        --epochs 5

    # Train base-first (recommended for new swarms)
    python scripts/train_adaptive_swarm.py --mode base-first \
        --base-dataset /path/to/general.jsonl \
        --specialist-datasets ocr=/path/to/ocr.jsonl,math=/path/to/math.jsonl \
        --epochs 3

Expected Results:
    - Base model: General reasoning capability
    - Specialists: Domain expertise (OCR, Math, Code, etc.)
    - Self-updating: Automatic validation gating prevents forgetting
    - Memory efficient: 18× smaller than full specialists
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from knowledge3d.cranium.adaptive_swarm import (
    AdaptiveSwarmTRM,
    SwarmConfig,
    SwarmTrainingProtocol
)
from knowledge3d.cranium.moe_router import MoERouter, RoutingConfig


def load_jsonl_samples(path: Path, start_idx: int = 0, end_idx: Optional[int] = None) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if end_idx is not None and i >= end_idx:
                break

            samples.append(json.loads(line))

    return samples


def dummy_eval_fn(weights: np.ndarray, samples: List[Dict]) -> float:
    """
    Dummy evaluation function (placeholder).

    Real implementation would:
    1. Run TRM forward pass with weights
    2. Compute loss on samples
    3. Return performance metric

    Args:
        weights: Model weights
        samples: Validation samples

    Returns:
        Performance score (higher = better)
    """
    # Placeholder: Return random score in [0.2, 0.3]
    # Real implementation would compute actual performance
    return 0.25 + np.random.rand() * 0.05


def train_base_mode(swarm: AdaptiveSwarmTRM, args):
    """Train base model only."""
    print("\n" + "="*80)
    print("Training Mode: Base Model Only")
    print("="*80)

    # Load samples
    print(f"\nLoading samples from {args.dataset}...")
    samples = load_jsonl_samples(Path(args.dataset), args.start, args.end)
    print(f"✓ Loaded {len(samples)} samples")

    # Split train/validation
    split_idx = int(len(samples) * (1 - args.validation_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"[Dataset] Train: {len(train_samples)}, Validation: {len(val_samples)}")

    swarm.set_base_validation_samples(val_samples)

    # Train epochs
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")

        stats = swarm.train_base_epoch(train_samples, dummy_eval_fn, use_self_update=args.self_update)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Average loss: {stats['avg_loss']:.4f}")
        if 'update_accepted' in stats:
            print(f"  Update accepted: {stats['update_accepted']}")

    # Save checkpoint
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir) / f"base_epoch_{args.epochs}"
        swarm.save_checkpoint(checkpoint_path)


def train_specialist_mode(swarm: AdaptiveSwarmTRM, args):
    """Train specific specialist."""
    print("\n" + "="*80)
    print(f"Training Mode: Specialist '{args.specialist}'")
    print("="*80)

    # Check specialist exists
    if args.specialist not in swarm.base.specialists:
        print(f"\n❌ Specialist '{args.specialist}' not registered")
        print(f"Available specialists: {list(swarm.base.specialists.keys())}")
        print(f"\nRegister specialist first:")
        print(f"  python scripts/register_specialist.py --name {args.specialist} --dims 512")
        return False

    # Load samples
    print(f"\nLoading samples from {args.dataset}...")
    samples = load_jsonl_samples(Path(args.dataset), args.start, args.end)
    print(f"✓ Loaded {len(samples)} samples")

    # Split train/validation
    split_idx = int(len(samples) * (1 - args.validation_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"[Dataset] Train: {len(train_samples)}, Validation: {len(val_samples)}")

    swarm.set_specialist_validation_samples(args.specialist, val_samples)

    # Train epochs
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")

        stats = swarm.train_specialist_epoch(args.specialist, train_samples, dummy_eval_fn,
                                            use_self_update=args.self_update)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Average loss: {stats['avg_loss']:.4f}")
        if 'update_accepted' in stats:
            print(f"  Update accepted: {stats['update_accepted']}")
            print(f"  Baseline: {stats['baseline_performance']:.4f}")
            print(f"  Shadow: {stats['shadow_performance']:.4f}")

    # Save checkpoint
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir) / f"{args.specialist}_epoch_{args.epochs}"
        swarm.save_checkpoint(checkpoint_path)

    return True


def train_base_first_mode(swarm: AdaptiveSwarmTRM, args):
    """Train base first, then all specialists."""
    print("\n" + "="*80)
    print("Training Mode: Base-First (Recommended)")
    print("="*80)

    # Load base dataset
    print(f"\nLoading base dataset from {args.base_dataset}...")
    general_samples = load_jsonl_samples(Path(args.base_dataset), args.start, args.end)
    print(f"✓ Loaded {len(general_samples)} general samples")

    # Parse specialist datasets
    specialist_samples = {}

    if args.specialist_datasets:
        for spec_dataset in args.specialist_datasets.split(','):
            name, path = spec_dataset.split('=')
            print(f"\nLoading specialist '{name}' dataset from {path}...")
            samples = load_jsonl_samples(Path(path))
            specialist_samples[name] = samples
            print(f"✓ Loaded {len(samples)} samples for '{name}'")

    # Run base-first protocol
    stats = SwarmTrainingProtocol.train_base_first(
        swarm,
        general_samples,
        specialist_samples,
        dummy_eval_fn
    )

    # Summary
    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)

    print(f"\nBase Model:")
    print(f"  Average loss: {stats['base']['avg_loss']:.4f}")

    for name, spec_stats in stats['specialists'].items():
        print(f"\nSpecialist '{name}':")
        print(f"  Average loss: {spec_stats['avg_loss']:.4f}")
        if 'update_accepted' in spec_stats:
            print(f"  Final performance: {spec_stats['shadow_performance']:.4f}")

    # Save checkpoint
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir) / "base_first_complete"
        swarm.save_checkpoint(checkpoint_path)


def train_procedural_drawing_mode(swarm: AdaptiveSwarmTRM, args):
    """Train procedural drawing specialist on RPN font dataset."""
    from knowledge3d.cranium.specialists.procedural_drawing_specialist import ProceduralDrawingSpecialist

    print("\n" + "="*80)
    print("Training Mode: Procedural Drawing (Atomic Cognition)")
    print("="*80)

    # Initialize specialist
    matryoshka_dim = args.matryoshka_dim if hasattr(args, 'matryoshka_dim') else 512
    specialist = ProceduralDrawingSpecialist(
        swarm=swarm,
        matryoshka_dim=matryoshka_dim,
        gpu_id=0
    )

    print(f"\n[Config]")
    print(f"  Matryoshka dimension: {matryoshka_dim}")
    print(f"  Dataset: {args.rpn_dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size if hasattr(args, 'batch_size') else 32}")

    # Train on RPN dataset
    specialist.train_on_rpn_dataset(
        dataset_path=Path(args.rpn_dataset),
        epochs=args.epochs,
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
        validation_split=args.validation_split
    )

    # Save checkpoint
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir) / f"procedural_drawing_epoch_{args.epochs}"
        specialist.save_checkpoint(checkpoint_path)
        print(f"\n✓ Saved checkpoint to {checkpoint_path}")

    # Print final metrics
    if specialist.training_metrics:
        final_metrics = specialist.training_metrics[-1]
        print(f"\n[Final Metrics]")
        print(f"  Text-visual alignment: {final_metrics.text_visual_alignment:.3f}")
        print(f"  Reconstruction fidelity: {final_metrics.reconstruction_fidelity:.3f}")
        print(f"  Generation quality: {final_metrics.generation_quality:.3f}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Train Adaptive Swarm TRM')

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['base', 'specialist', 'base-first', 'joint', 'procedural_drawing'],
                       help='Training mode')

    # Model configuration
    parser.add_argument('--base-dims', type=int, default=2048,
                       help='Base model dimensions')
    parser.add_argument('--min-dims', type=int, default=64,
                       help='Minimum dimensions')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--validation-split', type=float, default=0.1,
                       help='Validation set percentage')
    parser.add_argument('--self-update', action='store_true',
                       help='Enable self-updating (shadow weights + validation)')

    # Dataset paths
    parser.add_argument('--dataset', type=str,
                       help='Dataset path (for base or specialist mode)')
    parser.add_argument('--base-dataset', type=str,
                       help='Base model dataset (for base-first mode)')
    parser.add_argument('--specialist-datasets', type=str,
                       help='Specialist datasets: name1=/path1,name2=/path2')

    # Dataset range
    parser.add_argument('--start', type=int, default=0,
                       help='Start sample index')
    parser.add_argument('--end', type=int, default=None,
                       help='End sample index')

    # Specialist selection (for specialist mode)
    parser.add_argument('--specialist', type=str,
                       help='Specialist name (for specialist mode)')

    # Checkpoint management
    parser.add_argument('--checkpoint-dir', type=str,
                       default='/K3D/Knowledge3D.local/checkpoints/adaptive_swarm',
                       help='Checkpoint directory')
    parser.add_argument('--load-checkpoint', type=str,
                       help='Load checkpoint from path')

    # Learning rates
    parser.add_argument('--base-lr', type=float, default=0.001,
                       help='Base model learning rate')
    parser.add_argument('--specialist-lr', type=float, default=0.002,
                       help='Specialist learning rate')

    # Procedural drawing mode arguments
    parser.add_argument('--rpn-dataset', type=str,
                       help='RPN dataset path (for procedural_drawing mode)')
    parser.add_argument('--matryoshka-dim', type=int, default=512,
                       help='Matryoshka dimension (64-2048)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')

    args = parser.parse_args()

    print("="*80)
    print("Adaptive Swarm TRM Training")
    print("="*80)
    print(f"\nMode: {args.mode}")
    print(f"Base dimensions: {args.base_dims}")
    print(f"Self-updating: {'Enabled' if args.self_update else 'Disabled'}")
    print()

    # Create swarm config
    config = SwarmConfig(
        base_dims=args.base_dims,
        min_dims=args.min_dims,
        base_learning_rate=args.base_lr,
        specialist_learning_rate=args.specialist_lr,
        validation_split=args.validation_split
    )

    # Initialize swarm
    swarm = AdaptiveSwarmTRM(config=config)

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"\nLoading checkpoint from {args.load_checkpoint}...")
        swarm.load_checkpoint(Path(args.load_checkpoint))

    # Route to appropriate training mode
    if args.mode == 'base':
        if not args.dataset:
            print("❌ --dataset required for base mode")
            return False

        train_base_mode(swarm, args)

    elif args.mode == 'specialist':
        if not args.dataset or not args.specialist:
            print("❌ --dataset and --specialist required for specialist mode")
            return False

        success = train_specialist_mode(swarm, args)
        if not success:
            return False

    elif args.mode == 'base-first':
        if not args.base_dataset:
            print("❌ --base-dataset required for base-first mode")
            return False

        train_base_first_mode(swarm, args)

    elif args.mode == 'joint':
        print("❌ Joint training mode not yet implemented")
        return False

    elif args.mode == 'procedural_drawing':
        if not args.rpn_dataset:
            print("❌ --rpn-dataset required for procedural_drawing mode")
            print("\nExample usage:")
            print("  python scripts/train_adaptive_swarm.py --mode procedural_drawing \\")
            print("    --rpn-dataset /K3D/Knowledge3D.local/datasets/font_rpn_168k.jsonl \\")
            print("    --epochs 10 --matryoshka-dim 512 --batch-size 32")
            return False

        success = train_procedural_drawing_mode(swarm, args)
        if not success:
            return False

    # Print system statistics
    print("\n" + "="*80)
    print("System Statistics")
    print("="*80)

    stats = swarm.get_system_stats()

    print(f"\nBase Model:")
    print(f"  Dimensions: {stats['base_model']['max_dims']}")
    print(f"  Parameters: {stats['base_model']['params']/1e6:.2f}M")
    print(f"  Memory: {stats['base_model']['memory_mb']:.1f} MB")
    print(f"  Training steps: {stats['base_training']['steps']}")

    if args.self_update:
        print(f"  Update acceptance rate: {stats['base_training']['acceptance_rate']*100:.1f}%")

    print(f"\nSpecialists: {stats['base_model']['max_dims']}")

    for spec in stats['specialists']:
        print(f"  '{spec['specialist_name']}':")
        print(f"    Dimensions: {spec['shape'][0]}")
        print(f"    Parameters: {spec['params']/1e3:.1f}K")
        print(f"    Memory: {spec['memory_mb']:.2f} MB")

        if args.self_update and 'acceptance_rate' in spec:
            print(f"    Acceptance rate: {spec['acceptance_rate']*100:.1f}%")

    print(f"\nTotal System:")
    print(f"  Parameters: {stats['total_params']/1e6:.2f}M")
    print(f"  Memory: {stats['total_memory_mb']:.1f} MB")

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)

    if args.mode == 'base':
        print("  ✓ Base model trained")
        print("  → Register specialists: python scripts/register_specialist.py")
        print("  → Train specialists: python scripts/train_adaptive_swarm.py --mode specialist")

    elif args.mode == 'specialist':
        print(f"  ✓ Specialist '{args.specialist}' trained")
        print("  → Train more specialists or run inference")
        print("  → Test routing: python scripts/test_moe_routing.py")

    elif args.mode == 'base-first':
        print("  ✓ Base and all specialists trained")
        print("  → Run inference with MoE routing")
        print("  → Continue training with self-updating mode")

    print()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
