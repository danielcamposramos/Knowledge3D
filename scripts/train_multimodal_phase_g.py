#!/usr/bin/env python3
"""
Phase G.1: Multi-Modal Parallel Training

Trains TRM on RLWHF dataset with simultaneous:
1. OCR: Visual features → Character embeddings
2. Text: Semantic reasoning → Answer quality
3. Alignment: Visual ↔ Semantic cross-modal learning

Usage:
    # Train on samples 8042-10000 (complete to 10K)
    python scripts/train_multimodal_phase_g.py --start 8042 --end 10000

    # Continue with self-updating mode (10K+)
    python scripts/train_multimodal_phase_g.py --self-update --start 10000

Expected Results:
    - Multi-modal learning improves both OCR and text reasoning
    - Cross-modal alignment enables grounded understanding
    - Self-updating maintains performance without forgetting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from datetime import datetime

from knowledge3d.training.multimodal.multimodal_trainer import (
    MultiModalTRMTrainer,
    TrainingConfig
)
from knowledge3d.training.multimodal.self_updating_trm import (
    SelfUpdatingTRM,
    UpdateConfig,
    UpdateStrategy
)


def main():
    parser = argparse.ArgumentParser(description='Phase G.1: Multi-Modal Training')
    parser.add_argument('--start', type=int, default=8042,
                       help='Start sample index')
    parser.add_argument('--end', type=int, default=10000,
                       help='End sample index')
    parser.add_argument('--dataset', type=str,
                       default='/K3D/Knowledge3D.local/datasets/rlwhf/teacher_evaluations.jsonl',
                       help='RLWHF dataset path')
    parser.add_argument('--self-update', action='store_true',
                       help='Enable self-updating mode (after 10K)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                       help='Validation set percentage')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='/K3D/Knowledge3D.local/checkpoints/multimodal',
                       help='Checkpoint directory')
    parser.add_argument('--ocr-weight', type=float, default=1.0,
                       help='OCR loss weight')
    parser.add_argument('--text-weight', type=float, default=1.0,
                       help='Text loss weight')
    parser.add_argument('--alignment-weight', type=float, default=0.1,
                       help='Cross-modal alignment weight')

    args = parser.parse_args()

    print("=" * 80)
    print("Phase G.1: Multi-Modal TRM Training")
    print("=" * 80)
    print()
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.start} → {args.end}")
    print(f"Self-updating: {'Enabled' if args.self_update else 'Disabled'}")
    print()

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False

    # Count available samples
    with open(dataset_path, 'r') as f:
        total_samples = sum(1 for _ in f)

    print(f"✓ Dataset loaded: {total_samples} total samples")

    if args.end > total_samples:
        print(f"⚠ Requested end ({args.end}) > available ({total_samples}), "
              f"adjusting to {total_samples}")
        args.end = total_samples

    # Configure training
    config = TrainingConfig(
        ocr_weight=args.ocr_weight,
        text_weight=args.text_weight,
        alignment_weight=args.alignment_weight,
        validation_split=args.validation_split
    )

    # Initialize trainer
    trainer = MultiModalTRMTrainer(config=config)

    # Load dataset range
    print(f"\nLoading samples {args.start}-{args.end}...")
    samples = trainer.load_rlwhf_dataset(
        dataset_path,
        start_idx=args.start,
        end_idx=args.end
    )
    print(f"✓ Loaded {len(samples)} samples")

    # Split train/validation
    train_samples, val_samples = trainer.split_train_validation(samples)

    # Standard training mode
    if not args.self_update:
        print("\n" + "=" * 80)
        print("Mode: Standard Multi-Modal Training")
        print("=" * 80)

        # Train one epoch
        avg_loss = trainer.train_epoch(train_samples)

        # Validate
        val_loss = trainer.evaluate(val_samples)

        # Summary
        print("\n" + "=" * 80)
        print("Training Complete")
        print("=" * 80)
        print(f"  Training loss: {avg_loss:.4f}")
        print(f"  Validation loss: {val_loss:.4f}")
        print(f"  Total steps: {trainer.step}")

        # Save checkpoint
        checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_{args.end}.json"
        metrics = {
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'samples_trained': len(train_samples),
            'timestamp': datetime.now().isoformat()
        }
        trainer.save_checkpoint(checkpoint_path, metrics)

    # Self-updating mode
    else:
        print("\n" + "=" * 80)
        print("Mode: Self-Updating (Continual Learning)")
        print("=" * 80)

        # Initialize self-updater
        update_config = UpdateConfig(
            strategy=UpdateStrategy.BLEND,
            blend_alpha=0.1,
            min_improvement=0.001
        )

        updater = SelfUpdatingTRM(config=update_config)
        updater.set_validation_set(val_samples)

        # Evaluate baseline
        baseline_perf = updater.evaluate_performance(
            updater.weight_manager.W_primary,
            val_samples
        )
        print(f"\nBaseline performance: {baseline_perf:.4f}")

        # Process training samples incrementally
        print(f"\nProcessing {len(train_samples)} samples with self-updating...")

        for i, sample in enumerate(train_samples):
            # Train on this sample
            losses = trainer.training_step(sample)

            # Every 100 samples, propose weight update
            if (i + 1) % 100 == 0:
                print(f"\n--- Update checkpoint at sample {i+1} ---")

                # Compute synthetic gradient (placeholder)
                # In real implementation, this would be actual gradients
                gradient = np.random.randn(*updater.weight_manager.weight_shape) * 0.01

                # Propose update
                updater.propose_update(gradient, learning_rate=0.001)

                # Validate and commit
                success, baseline, shadow = updater.validate_and_commit()

                if success:
                    print(f"  Performance improved: {baseline:.4f} → {shadow:.4f}")
                else:
                    print(f"  Update rejected, keeping baseline: {baseline:.4f}")

        # Final summary
        stats = updater.get_update_stats()

        print("\n" + "=" * 80)
        print("Self-Updating Complete")
        print("=" * 80)
        print(f"  Total updates proposed: {stats['total_updates']}")
        print(f"  Accepted: {stats['accepted']} ({stats['acceptance_rate']*100:.1f}%)")
        print(f"  Rejected: {stats['rejected']}")
        print(f"  Final performance: {stats['current_performance']:.4f}")

        # Save self-updating checkpoint
        checkpoint_dir = Path(args.checkpoint_dir) / f"self_update_{args.end}"
        updater.save_checkpoint(checkpoint_dir)

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)

    if not args.self_update and args.end >= 10000:
        print("  ✓ Reached 10K milestone!")
        print("  → Train GalacticTemplateBank Layer 3 with learned embeddings")
        print("  → Validate on Apollo ground truth (target: 90%+ detection)")
        print("  → Enable self-updating mode for continual learning")
    elif args.self_update:
        print("  ✓ Self-updating active - model will continue improving")
        print("  → Monitor acceptance rate (target: >20%)")
        print("  → Check validation performance trend")
        print("  → Deploy to Phase F.2 character detection")
    else:
        print(f"  → Continue training to 10K ({10000 - args.end} samples remaining)")
        print(f"  → Run: python scripts/train_multimodal_phase_g.py --start {args.end} --end 10000")

    print()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
