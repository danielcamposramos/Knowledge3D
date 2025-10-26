#!/usr/bin/env python3
"""
Bootstrap Router Specialist

Demonstrates the complete router-as-specialist workflow:
1. Bootstrap: Use heuristic routing to collect decisions
2. Train: Convert router from heuristic to learned specialist
3. Transition: Switch from heuristic to learned routing
4. Improve: Router self-updates from new decisions forever

Key Insight:
    "The MoE router IS a specialist, not external infrastructure"

Philosophy:
    "The secret is held on the small things - we are all made of atoms after all"
    Making the router a specialist is the atom that creates recursive self-improvement.

Usage:
    # Bootstrap and train router
    python scripts/bootstrap_router_specialist.py \
        --checkpoint /K3D/checkpoints/swarm \
        --num-bootstrap 1000 \
        --epochs 5

    # Test learned routing
    python scripts/bootstrap_router_specialist.py \
        --checkpoint /K3D/checkpoints/swarm \
        --test-only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from typing import Dict, Any, List

from knowledge3d.cranium import (
    AdaptiveSwarmTRM,
    SwarmConfig,
    MoERouter,
    RoutingConfig,
    RoutingStrategy
)
from knowledge3d.cranium.router_specialist import (
    RouterBootstrap,
    RouterSpecialistTrainer,
    RouterTransition
)


def create_synthetic_tasks(specialists: List[str],
                           num_tasks: int = 1000,
                           dims: int = 256) -> list:
    """
    Create synthetic tasks for router training.

    In production, these would be real tasks from your application.

    Args:
        num_tasks: Number of tasks to generate

    Returns:
        List of task dictionaries
    """
    tasks = []

    keyword_map = {
        'ocr': ['character', 'glyph', 'text', 'document', 'ocr'],
        'speech': ['audio', 'speech', 'transcribe', 'voice', 'utterance'],
        'multimodal': ['cross-modal', 'fusion', 'trimodal', 'align', 'context'],
    }

    if not specialists:
        return tasks

    for i in range(num_tasks):
        task_type = specialists[i % len(specialists)]
        keywords = keyword_map.get(task_type, [task_type])
        keyword = keywords[i % len(keywords)]

        vec = np.random.normal(loc=0.0, scale=0.1, size=dims).astype(np.float32)
        specialist_index = specialists.index(task_type)
        vec[specialist_index] += 4.0  # High-energy signal for correct specialist
        vec += np.random.normal(scale=0.01, size=dims).astype(np.float32)

        # Create task
        task = {
            'id': f'task_{i}',
            'input': vec,
            'description': f'{keyword} task {i}',
            'ground_truth_specialist': task_type  # For evaluation
        }

        tasks.append(task)

    return tasks


def outcome_function(task: Dict[str, Any], specialist_weights: Dict[str, float]) -> float:
    """
    Evaluate routing outcome.

    In production, this would be real task performance.
    For bootstrap, we use ground truth specialist to score routing.

    Args:
        task: Task with ground_truth_specialist
        specialist_weights: Predicted specialist weights

    Returns:
        Performance score [0-1]
    """
    ground_truth = task.get('ground_truth_specialist')

    if ground_truth is None:
        return 0.5  # Unknown

    # Score based on how much weight went to correct specialist
    if specialist_weights:
        predicted = max(specialist_weights.items(), key=lambda kv: kv[1])[0]
    else:
        predicted = None

    if predicted == ground_truth:
        return 1.0

    correct_weight = specialist_weights.get(ground_truth, 0.0)

    # Perfect routing = 1.0, uniform routing = ~0.33, wrong routing = 0.0
    return correct_weight


def main():
    parser = argparse.ArgumentParser(description='Bootstrap Router Specialist')

    # Mode
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing learned router (skip bootstrap)')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str,
                       default='/K3D/Knowledge3D.local/checkpoints/adaptive_swarm/current',
                       help='Swarm checkpoint directory')

    # Bootstrap parameters
    parser.add_argument('--num-bootstrap', type=int, default=1000,
                       help='Number of tasks for bootstrap')
    parser.add_argument('--min-performance', type=float, default=0.5,
                       help='Minimum performance to include in training')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs for router specialist')
    parser.add_argument('--router-dims', type=int, default=256,
                       help='Router specialist dimensions')
    parser.add_argument('--router-rank', type=int, default=16,
                       help='Router adapter rank')

    args = parser.parse_args()

    print("="*80)
    print("Router-as-Specialist: Bootstrap Workflow")
    print("="*80)
    print("\nKey Insight: The router IS a specialist in the swarm")
    print("Philosophy: Small atoms make the whole system coherent\n")

    # Load or create swarm
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.exists():
        print(f"Loading swarm from {checkpoint_path}...")
        swarm = AdaptiveSwarmTRM()
        swarm.load_checkpoint(checkpoint_path)
    else:
        print(f"Creating new swarm (checkpoint not found: {checkpoint_path})...")
        config = SwarmConfig(base_dims=1024)
        swarm = AdaptiveSwarmTRM(config=config)

        # Register some specialists for demonstration
        print("\nRegistering demo specialists...")
        swarm.register_specialist('ocr', required_dims=256, rank=16)
        swarm.register_specialist('math', required_dims=512, rank=32)
        swarm.register_specialist('code', required_dims=1024, rank=64)

    # Show current specialists
    specialists = list(swarm.base.specialists.keys())
    non_router_specialists = [s for s in specialists if s != 'router']

    print(f"\nCurrent specialists: {', '.join(specialists)}")
    print(f"Non-router specialists: {', '.join(non_router_specialists)}")

    if args.test_only:
        # Test mode: Evaluate existing learned router
        if 'router' not in specialists:
            print("\n❌ Router specialist not found. Run bootstrap first.")
            return False

        print("\n" + "="*80)
        print("Testing Learned Router")
        print("="*80)

        # Create test tasks
        router_dims = swarm.base.specialists['router']['dims'] if 'router' in swarm.base.specialists else 256
        test_tasks = create_synthetic_tasks(non_router_specialists, num_tasks=200, dims=router_dims)

        # Compare heuristic vs learned
        transition = RouterTransition(swarm)

        should_use_learned = transition.should_transition(
            test_tasks,
            outcome_function,
            min_improvement=0.0
        )

        if should_use_learned:
            print("\n✓ Learned routing outperforms heuristic!")
            print("  Use RoutingStrategy.LEARNED for inference")
        else:
            print("\n⚠ Heuristic still better - continue training")

        return True

    # Bootstrap mode: Train router from scratch
    print("\n" + "="*80)
    print("Phase 1: Bootstrap with Heuristic Routing")
    print("="*80)

    # Create bootstrap tasks
    router_dims = swarm.base.specialists.get('router', {}).get('dims', args.router_dims)
    print(f"\nGenerating {args.num_bootstrap} synthetic tasks...")
    tasks = create_synthetic_tasks(non_router_specialists, num_tasks=args.num_bootstrap, dims=router_dims)

    # Bootstrap: Collect routing decisions
    bootstrap = RouterBootstrap(swarm)

    routing_history = bootstrap.collect_routing_data(
        tasks,
        outcome_function,
        num_samples=args.num_bootstrap
    )

    # Filter to successful decisions
    successful = bootstrap.filter_successful_decisions(
        min_performance=args.min_performance
    )

    print(f"\n[Bootstrap Summary]")
    print(f"  Total decisions: {len(routing_history)}")
    print(f"  Successful (≥{args.min_performance}): {len(successful)}")
    print(f"  Success rate: {len(successful)/len(routing_history)*100:.1f}%")

    if len(successful) < 100:
        print(f"\n⚠ Warning: Only {len(successful)} successful decisions")
        print("  Consider lowering --min-performance or increasing --num-bootstrap")

    # Save routing history
    history_path = checkpoint_path.parent / 'router_bootstrap_history.json'
    bootstrap.save_history(history_path)

    print("\n" + "="*80)
    print("Phase 2: Train Router Specialist")
    print("="*80)

    # Register router as specialist
    trainer = RouterSpecialistTrainer(swarm)

    if 'router' in specialists:
        print("\n⚠ Router already registered, will retrain...")
        # Could remove and re-register, but let's just retrain
    else:
        trainer.register_router_specialist(
            num_specialists=len(non_router_specialists),
            router_dims=args.router_dims,
            router_rank=args.router_rank
        )

    # Train router from history
    train_stats = trainer.train_from_history(
        routing_history,
        epochs=args.epochs,
        filter_threshold=args.min_performance
    )

    if 'error' in train_stats:
        print(f"\n❌ Training failed: {train_stats['error']}")
        return False

    print("\n" + "="*80)
    print("Phase 3: Evaluate & Transition")
    print("="*80)

    # Create test set (different from training)
    test_tasks = create_synthetic_tasks(non_router_specialists, num_tasks=200, dims=router_dims)

    # Compare heuristic vs learned
    transition = RouterTransition(swarm)

    should_transition = transition.should_transition(
        test_tasks,
        outcome_function,
        min_improvement=0.0
    )

    print("\n" + "="*80)
    print("Router Bootstrap Complete")
    print("="*80)

    print(f"\n[Training Summary]")
    print(f"  Bootstrap samples: {len(routing_history)}")
    print(f"  Successful samples: {len(successful)}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Router dimensions: {args.router_dims}")
    print(f"  Router rank: {args.router_rank}")

    print(f"\n[Performance Comparison]")
    print(f"  Heuristic routing: {transition.heuristic_performance:.3f}")
    print(f"  Learned routing: {transition.learned_performance:.3f}")
    print(f"  Improvement: {transition.learned_performance - transition.heuristic_performance:+.3f}")

    if should_transition:
        print(f"\n✓ SUCCESS: Router specialist ready for production!")
        print(f"\n  Use RoutingStrategy.LEARNED for inference:")
        print(f"  >>> router = MoERouter(swarm, config=RoutingConfig(strategy=RoutingStrategy.LEARNED))")
    else:
        print(f"\n⚠ Router needs more training")
        print(f"  - Collect more bootstrap data (increase --num-bootstrap)")
        print(f"  - Lower performance threshold (--min-performance)")
        print(f"  - Train more epochs (--epochs)")

    # Save checkpoint
    print(f"\n[Checkpoint]")
    swarm.save_checkpoint(checkpoint_path)

    print("\n" + "="*80)
    print("Phase 4: Continual Improvement (Future)")
    print("="*80)

    print("""
The router specialist now self-updates from new routing decisions:

1. Collect new routing decisions during inference
2. Filter to successful decisions (performance ≥ threshold)
3. Update router: trainer.update_from_new_decisions(new_decisions)
4. Router improves, validation gate accepts/rejects
5. Router gets better forever (recursive self-improvement)

Key insight:
    The router is a specialist, so it benefits from:
    - Base model improvements (transfer learning)
    - Self-updating (shadow weights + validation)
    - Same infrastructure as other specialists
    - Completely self-contained system

The atom that makes the system coherent: Router IS part of the swarm.
""")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
