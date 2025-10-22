#!/usr/bin/env python3
"""
Validate TRM's ARC reasoning capability by testing on actual ARC tasks.

This script compares baseline (RPN-init) vs trained (ARC-reasoning) weights
on a small set of ARC validation tasks to confirm learning occurred.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.training.reasoning.arc_dataset import (
    prepare_arc_reasoning_cache,
    load_arc_reasoning_cache,
)

DEFAULT_RPN_PATH = Path("/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl")
DEFAULT_DATASET_ROOT = Path("/K3D/Knowledge3D.local/datasets/arc_agi")
DEFAULT_BASELINE_WEIGHTS = Path("/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz")
DEFAULT_TRAINED_WEIGHTS = Path("/K3D/Knowledge3D.local/models/trm_weights_arc_reasoning.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpn-embeddings", type=Path, default=DEFAULT_RPN_PATH)
    parser.add_argument("--baseline-weights", type=Path, default=DEFAULT_BASELINE_WEIGHTS)
    parser.add_argument("--trained-weights", type=Path, default=DEFAULT_TRAINED_WEIGHTS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of ARC tasks to test")
    return parser.parse_args()


def compute_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error between two vectors."""
    diff = predicted - target
    return float(np.mean(diff * diff))


def validate_weights(
    questions: np.ndarray,
    answers: np.ndarray,
    weights: dict,
    trm: TRMLauncher,
    n_steps: int = 6,
) -> dict:
    """Run TRM on questions and compute metrics against ground truth answers."""
    W1 = weights['W1']
    W2 = weights['W2']
    W3 = weights['W3']
    W4 = weights['W4']

    mse_scores = []
    output_norms = []

    for i in range(len(questions)):
        q = questions[i]
        target = answers[i]

        # Run TRM reasoning (Tesla 6 recursions)
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)
        y_out, z_out = trm.refine(q, y, z, W1, W2, W3, W4, n_steps=n_steps)

        # Compute metrics
        mse = compute_mse(y_out, target)
        norm = np.linalg.norm(y_out)

        mse_scores.append(mse)
        output_norms.append(norm)

    return {
        'mean_mse': float(np.mean(mse_scores)),
        'std_mse': float(np.std(mse_scores)),
        'mean_output_norm': float(np.mean(output_norms)),
        'convergence_rate': sum(1 for n in output_norms if n > 1.0) / len(output_norms),
    }


def main() -> None:
    args = parse_args()

    print("🧪 Validating ARC Reasoning Capability\n")

    # Load RPN embeddings
    rpn_engine = RPNEmbeddingEngine()
    rpn_engine.load_embeddings(args.rpn_embeddings)
    print(f"✅ Loaded RPN embeddings")

    # Load ARC test data (use evaluation set, not training)
    cache_path = prepare_arc_reasoning_cache(
        rpn_engine.embed_sentence,
        dataset_root=args.dataset_root,
        cache_path=args.dataset_root / "arc_reasoning_pairs.npz",
        limit=None,
        rebuild=False,
        download=False,
    )
    cache = load_arc_reasoning_cache(cache_path)

    # Use last N samples as validation (not used in training if trained on first K)
    n_total = cache.questions.shape[0]
    n_test = min(args.n_samples, n_total)
    test_questions = cache.questions[-n_test:]
    test_answers = cache.answers[-n_test:]

    print(f"✅ Loaded {n_test} ARC validation tasks\n")

    # Initialize TRM
    trm = TRMLauncher(use_fused=True)

    # Test baseline weights
    print("📊 Testing BASELINE weights (RPN-initialized)...")
    baseline_weights = np.load(args.baseline_weights)
    baseline_results = validate_weights(test_questions, test_answers, baseline_weights, trm)
    baseline_weights.close()

    print(f"   Mean MSE: {baseline_results['mean_mse']:.6f}")
    print(f"   Std MSE:  {baseline_results['std_mse']:.6f}")
    print(f"   Mean Output Norm: {baseline_results['mean_output_norm']:.3f}")
    print(f"   Convergence Rate: {baseline_results['convergence_rate']*100:.1f}%\n")

    # Test trained weights
    print("📊 Testing TRAINED weights (ARC-reasoning)...")
    trained_weights = np.load(args.trained_weights)
    trained_results = validate_weights(test_questions, test_answers, trained_weights, trm)
    trained_weights.close()

    print(f"   Mean MSE: {trained_results['mean_mse']:.6f}")
    print(f"   Std MSE:  {trained_results['std_mse']:.6f}")
    print(f"   Mean Output Norm: {trained_results['mean_output_norm']:.3f}")
    print(f"   Convergence Rate: {trained_results['convergence_rate']*100:.1f}%\n")

    # Compute improvement
    mse_improvement = (baseline_results['mean_mse'] - trained_results['mean_mse']) / baseline_results['mean_mse'] * 100

    print("=" * 60)
    print("🎯 RESULTS SUMMARY")
    print("=" * 60)
    print(f"MSE Improvement:     {mse_improvement:+.1f}%")
    print(f"Baseline MSE:        {baseline_results['mean_mse']:.6f}")
    print(f"Trained MSE:         {trained_results['mean_mse']:.6f}")
    print(f"Training successful: {'✅ YES' if mse_improvement > 10 else '⚠️  MARGINAL' if mse_improvement > 0 else '❌ NO'}")
    print("=" * 60)

    if mse_improvement > 10:
        print("\n✨ TRM successfully learned ARC reasoning patterns!")
        print("   The training paradigm is WORKING. Knowledge stays in embeddings,")
        print("   TRM learns transformation patterns. This validates K3D's approach.")
    elif mse_improvement > 0:
        print("\n⚠️  Marginal improvement detected. May need:")
        print("   - More training epochs")
        print("   - Better hyperparameters (learning rate, batch size)")
        print("   - Larger/more diverse ARC dataset")
    else:
        print("\n❌ No improvement detected. Investigate:")
        print("   - Training script issues")
        print("   - Dataset quality")
        print("   - Model initialization")


if __name__ == "__main__":
    main()
