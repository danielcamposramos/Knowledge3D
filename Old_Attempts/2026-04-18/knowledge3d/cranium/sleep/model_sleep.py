"""
Model Sleep Cycle (Sleep 1): Shadow Weights Validation

Updates: MODELS = LOGIC (how to process information)

Process:
1. For each trained specialist:
   - Evaluate baseline performance (primary weights)
   - Evaluate shadow performance (shadow weights)
   - If shadow > baseline + min_improvement: COMMIT
   - Else: REJECT
2. Update specialist checkpoints with accepted weights
3. Log acceptance/rejection rates

Result: Safely improved specialist models without catastrophic forgetting
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM


class ModelSleepCycle:
    """
    Model sleep cycle for safe self-updating via shadow weights.

    Validates and commits specialist improvements learned during training.
    """

    def __init__(self, matryoshka_system: MatryoshkaTRM, checkpoint_dir: Path):
        """
        Initialize model sleep cycle.

        Args:
            matryoshka_system: Matryoshka TRM with specialists
            checkpoint_dir: Directory to save updated checkpoints
        """
        self.system = matryoshka_system
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "total_specialists": 0,
            "weights_accepted": 0,
            "weights_rejected": 0,
            "acceptance_rate": 0.0,
            "average_improvement": 0.0,
            "specialists_updated": []
        }

    def validate_specialist(self, specialist_name: str) -> Dict[str, Any]:
        """
        Validate single specialist's shadow weights.

        Args:
            specialist_name: Name of specialist

        Returns:
            Validation result
        """
        if specialist_name not in self.system.specialists:
            raise ValueError(f"Unknown specialist: {specialist_name}")

        specialist = self.system.specialists[specialist_name]
        adapter = specialist['adapter']

        # Get validation samples
        validation_samples = adapter.validation_samples
        if not validation_samples:
            print(f"  ⚠️  {specialist_name}: No validation samples, skipping")
            return {
                "specialist": specialist_name,
                "validated": False,
                "reason": "no_validation_samples"
            }

        # Fork primary → shadow (if not already done)
        adapter.fork_to_shadow()

        # Prepare evaluation function
        def eval_fn(W, samples):
            """Evaluate model on validation samples."""
            total_loss = 0.0

            for sample in samples:
                input_data = np.array(sample['input'], dtype=np.float32)
                target = np.array(sample['target'], dtype=np.float32)

                # Forward pass
                output = W @ input_data

                # Loss (MSE)
                loss = np.mean((output - target) ** 2)
                total_loss += loss

            return total_loss / len(samples) if samples else 0.0

        # Get base weights
        dims = specialist['dims']
        W_base = self.system.get_base_at_dim(dims)

        # Evaluate baseline (primary weights)
        W_baseline = W_base + adapter.get_delta()
        baseline_loss = eval_fn(W_baseline, validation_samples)

        # Evaluate shadow (shadow weights)
        W_shadow = W_base + adapter.get_delta_shadow()
        shadow_loss = eval_fn(W_shadow, validation_samples)

        # Compute improvement
        improvement = baseline_loss - shadow_loss  # Positive = better
        improvement_percent = (improvement / baseline_loss * 100) if baseline_loss > 0 else 0.0

        # Decision: commit or reject
        min_improvement = adapter.config.min_improvement
        should_commit = improvement >= min_improvement

        result = {
            "specialist": specialist_name,
            "validated": True,
            "baseline_loss": float(baseline_loss),
            "shadow_loss": float(shadow_loss),
            "improvement": float(improvement),
            "improvement_percent": float(improvement_percent),
            "should_commit": should_commit,
            "min_improvement_threshold": float(min_improvement)
        }

        if should_commit:
            # Commit shadow → primary
            adapter.commit_shadow_to_primary()
            result["action"] = "COMMITTED"
            print(f"  ✅ {specialist_name}: COMMITTED (improvement: {improvement_percent:+.2f}%)")
        else:
            result["action"] = "REJECTED"
            print(f"  ❌ {specialist_name}: REJECTED (improvement: {improvement_percent:+.2f}% < threshold)")

        return result

    def run(self) -> Dict[str, Any]:
        """
        Run model sleep cycle on all specialists.

        Returns:
            Sleep metrics
        """
        print("\n" + "="*80)
        print("MODEL SLEEP CYCLE - Shadow Weights Validation")
        print("="*80)
        print(f"Specialists: {len(self.system.specialists)}")
        print()

        start_time = time.time()

        specialist_results = []
        total_improvement = 0.0
        validated_count = 0

        for specialist_name in self.system.specialists.keys():
            print(f"Validating: {specialist_name}")

            result = self.validate_specialist(specialist_name)
            specialist_results.append(result)

            if result.get("validated"):
                validated_count += 1

                if result.get("should_commit"):
                    self.metrics["weights_accepted"] += 1
                    total_improvement += result.get("improvement_percent", 0.0)

                    # Save updated checkpoint
                    self.save_specialist_checkpoint(specialist_name)

                    self.metrics["specialists_updated"].append({
                        "name": specialist_name,
                        "improvement_percent": result.get("improvement_percent"),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    self.metrics["weights_rejected"] += 1

        # Compute metrics
        self.metrics["total_specialists"] = len(self.system.specialists)

        if validated_count > 0:
            self.metrics["acceptance_rate"] = (
                self.metrics["weights_accepted"] / validated_count * 100
            )
        else:
            self.metrics["acceptance_rate"] = 0.0

        if self.metrics["weights_accepted"] > 0:
            self.metrics["average_improvement"] = (
                total_improvement / self.metrics["weights_accepted"]
            )
        else:
            self.metrics["average_improvement"] = 0.0

        elapsed = time.time() - start_time

        # Summary
        print("\n" + "─"*80)
        print("MODEL SLEEP SUMMARY")
        print("─"*80)
        print(f"Total specialists: {self.metrics['total_specialists']}")
        print(f"Weights accepted: {self.metrics['weights_accepted']}")
        print(f"Weights rejected: {self.metrics['weights_rejected']}")
        print(f"Acceptance rate: {self.metrics['acceptance_rate']:.1f}%")
        print(f"Average improvement: {self.metrics['average_improvement']:+.2f}%")
        print(f"Time: {elapsed:.1f}s")
        print("="*80 + "\n")

        return {
            "cycle_type": "model_sleep",
            "metrics": self.metrics,
            "specialist_results": specialist_results,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }

    def save_specialist_checkpoint(self, specialist_name: str):
        """Save updated specialist checkpoint."""
        if specialist_name not in self.system.specialists:
            return

        specialist = self.system.specialists[specialist_name]
        adapter = specialist['adapter']

        # Save to checkpoint directory
        specialist_dir = self.checkpoint_dir / f"{specialist_name}_updated"
        specialist_dir.mkdir(parents=True, exist_ok=True)

        adapter.save_checkpoint(specialist_dir)

        print(f"    Saved checkpoint: {specialist_dir}")


__all__ = ['ModelSleepCycle']
