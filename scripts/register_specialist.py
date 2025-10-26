#!/usr/bin/env python3
"""
Register Specialist in Adaptive Swarm

Registers a new specialist with the swarm system.

Each specialist:
- Has a unique name (e.g., 'ocr', 'math', 'code')
- Operates at specific dimension level (64 - 16384)
- Uses low-rank adapter (LoRA-style)
- Self-updates independently

Usage:
    # Register OCR specialist (medium complexity, 512 dims)
    python scripts/register_specialist.py --name ocr --dims 512 --rank 32

    # Register math specialist (high complexity, 1024 dims)
    python scripts/register_specialist.py --name math --dims 1024 --rank 64

    # Register code specialist (very high complexity, 2048 dims)
    python scripts/register_specialist.py --name code --dims 2048 --rank 128

    # Auto-select dimensions based on complexity
    python scripts/register_specialist.py --name ocr --complexity 0.5

Dimension Guidelines:
    64 dims:   Trivial tasks (single char OCR, basic arithmetic)
    128 dims:  Simple tasks (word recognition, simple math)
    256 dims:  Medium-low (sentence parsing, basic reasoning)
    512 dims:  Medium (paragraph understanding, moderate reasoning)
    1024 dims: Complex (multi-paragraph, multi-hop reasoning)
    2048 dims: Very complex (document analysis, deep reasoning)
    4096 dims: Maximum (corpus analysis, meta-reasoning)

Rank Guidelines:
    Rank = dims // 32 (default)
    - Higher rank: More capacity, more memory
    - Lower rank: Less capacity, less memory
    - Typical: 16-128
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.cranium.matryoshka_trm import DimensionSelector
from knowledge3d.cranium.trm_adapters import AdapterConfig


def main():
    parser = argparse.ArgumentParser(description='Register Specialist')

    # Specialist identification
    parser.add_argument('--name', type=str, required=True,
                       help='Specialist name (e.g., ocr, math, code)')

    # Dimension selection (mutually exclusive)
    dim_group = parser.add_mutually_exclusive_group()
    dim_group.add_argument('--dims', type=int,
                          help='Explicit dimension level (64-16384)')
    dim_group.add_argument('--complexity', type=float,
                          help='Task complexity [0-1] for auto-dimension selection')

    # Adapter configuration
    parser.add_argument('--rank', type=int, default=None,
                       help='Adapter rank (default: dims//32)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Adapter alpha (scaling factor)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Adapter learning rate')

    # Swarm configuration
    parser.add_argument('--base-dims', type=int, default=2048,
                       help='Base model dimensions')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='/K3D/Knowledge3D.local/checkpoints/adaptive_swarm',
                       help='Checkpoint directory')
    parser.add_argument('--load-checkpoint', type=str,
                       help='Load existing swarm from checkpoint')

    # Description (for documentation)
    parser.add_argument('--description', type=str,
                       help='Specialist description')

    args = parser.parse_args()

    print("="*80)
    print("Register Specialist in Adaptive Swarm")
    print("="*80)

    # Initialize or load swarm
    if args.load_checkpoint:
        print(f"\nLoading swarm from {args.load_checkpoint}...")
        swarm = AdaptiveSwarmTRM()
        swarm.load_checkpoint(Path(args.load_checkpoint))
    else:
        print(f"\nInitializing new swarm (base dims: {args.base_dims})...")
        config = SwarmConfig(base_dims=args.base_dims)
        swarm = AdaptiveSwarmTRM(config=config)

    # Determine dimensions
    if args.dims is not None:
        required_dims = args.dims
        print(f"\nDimension: {required_dims} (explicit)")
    elif args.complexity is not None:
        required_dims = DimensionSelector.select_dim(args.complexity)
        print(f"\nComplexity: {args.complexity:.2f}")
        print(f"Auto-selected dimension: {required_dims}")
    else:
        # Default: Use half of base dims
        required_dims = args.base_dims // 2
        print(f"\nDimension: {required_dims} (default: base_dims // 2)")

    # Validate dimensions
    if required_dims > swarm.config.base_dims:
        print(f"\n⚠ Warning: Requested dims ({required_dims}) > base dims ({swarm.config.base_dims})")
        print(f"   Expanding base to {required_dims}...")
        swarm.expand_capacity(required_dims)

    # Check if specialist already exists
    if args.name in swarm.base.specialists:
        print(f"\n⚠ Specialist '{args.name}' already registered")

        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Registration cancelled")
            return False

        # Remove existing
        swarm.base.remove_specialist(args.name)

    # Create adapter config
    adapter_config = AdapterConfig(
        rank=args.rank or (required_dims // 32),
        alpha=args.alpha,
        learning_rate=args.learning_rate
    )

    # Register specialist
    print(f"\nRegistering specialist '{args.name}'...")
    swarm.register_specialist(
        name=args.name,
        required_dims=required_dims,
        rank=adapter_config.rank,
        adapter_config=adapter_config
    )

    # Print specialist info
    specialist = swarm.base.specialists[args.name]

    print("\n" + "="*80)
    print(f"Specialist '{args.name}' Registered")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Dimensions: {specialist['dims']}")
    print(f"  Rank: {specialist['rank']}")
    print(f"  Parameters: {specialist['params']/1e3:.1f}K")
    print(f"  Memory: {specialist['memory_mb']:.2f} MB")

    # Estimate performance
    speedup = DimensionSelector.estimate_speedup(swarm.config.base_dims, specialist['dims'])
    memory_savings = DimensionSelector.estimate_memory_savings(swarm.config.base_dims, specialist['dims'])

    print(f"\nPerformance:")
    print(f"  Speedup vs full base: {speedup:.0f}×")
    print(f"  Memory savings: {memory_savings:.1f} MB")

    # Save metadata
    metadata_path = Path(args.checkpoint_dir) / 'specialists' / f'{args.name}.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'name': args.name,
        'description': args.description or f"Specialist for {args.name} tasks",
        'dims': specialist['dims'],
        'rank': specialist['rank'],
        'params': specialist['params'],
        'memory_mb': specialist['memory_mb'],
        'config': {
            'alpha': adapter_config.alpha,
            'learning_rate': adapter_config.learning_rate
        },
        'registered': True
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to {metadata_path}")

    # Save swarm checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / 'current'
    swarm.save_checkpoint(checkpoint_path)

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)

    print(f"\n1. Prepare training data for '{args.name}'")
    print(f"   Format: JSONL with samples specific to {args.name} domain")

    print(f"\n2. Train specialist:")
    print(f"   python scripts/train_adaptive_swarm.py \\")
    print(f"       --mode specialist \\")
    print(f"       --specialist {args.name} \\")
    print(f"       --dataset /path/to/{args.name}_samples.jsonl \\")
    print(f"       --epochs 5 \\")
    print(f"       --self-update")

    print(f"\n3. Test specialist:")
    print(f"   python scripts/test_specialist.py --specialist {args.name}")

    print()

    # Show all specialists
    all_specialists = list(swarm.base.specialists.keys())
    if len(all_specialists) > 1:
        print(f"Registered specialists: {', '.join(all_specialists)}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
