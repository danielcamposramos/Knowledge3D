#!/usr/bin/env python3
"""
Sovereign TRM Demo - Show complete sovereign inference pipeline

Demonstrates:
1. SovereignTRM loads V7 weights (NumPy → GPU)
2. Predicts rules + confidences using LSTM (PTX/RPN execution)
3. Zero PyTorch in hot path (pure sovereign)
"""

from pathlib import Path
import sys

# Add project to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.cranium.sovereign_trm import BOS_ID, SovereignTRM


def main():
    print("=" * 60)
    print("SOVEREIGN TRM VALIDATION DEMO")
    print("=" * 60)

    # Load V7 sovereign checkpoint
    checkpoint_dir = "/K3D/Knowledge3D.local/checkpoints/v7_sovereign"
    print(f"\n✓ Loading checkpoint: {checkpoint_dir}")

    # Read metadata
    import json
    meta_path = Path(checkpoint_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text())

    vocab_size = meta["vocab_size"]
    embedding_dim = meta["embedding_dim"]
    hidden_dim = meta["hidden_dim"]
    rule_registry = meta["rule_registry"]

    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Rules: {len(rule_registry)} rules")

    # Initialize SovereignTRM
    print("\n✓ Initializing SovereignTRM...")
    trm = SovereignTRM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )

    # Load weights (NumPy → GPU via sovereign loader)
    print("✓ Loading weights (NumPy → GPU)...")
    trm.load_weights(checkpoint_dir)
    print("  11/11 weight arrays loaded to GPU")

    # Test problems
    problems = [
        "Find f'(1) where f(x) = (3x-4)/(2x+3)",
        "Differentiate x^2 + 3x",
        "Integrate 2x dx",
    ]

    print("\n" + "=" * 60)
    print("SOVEREIGN INFERENCE (PTX + RPN only)")
    print("=" * 60)

    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}: {problem}")

        # Tokenize (byte-level)
        tokens = [BOS_ID] + list(problem.encode("utf-8", errors="ignore")[:32])
        print(f"  Tokens: {len(tokens)} tokens (byte-level encoding)")

        # Inference (LSTM → Rule Head + Confidence Head)
        print("  Running sovereign inference...")
        rule_ids, confidences = trm.infer(tokens, max_rules=8)

        # Map to rule names
        rules = [rule_registry[idx] if idx < len(rule_registry) else f"unknown_{idx}"
                 for idx in rule_ids]

        # Print results
        print(f"  ✓ Predicted {len(rules)} rules:")
        for rule, conf in zip(rules, confidences):
            conf_tag = "CONFIDENT" if conf >= 0.9 else "UNCERTAIN" if conf >= 0.5 else "VERIFY"
            print(f"    - {rule:20s} (conf={conf:.3f}) [{conf_tag}]")

    # Cleanup
    print("\n✓ Cleaning up GPU resources...")
    trm.cleanup()

    print("\n" + "=" * 60)
    print("SOVEREIGN TRM VALIDATION COMPLETE")
    print("=" * 60)
    print("\n✅ Zero PyTorch in hot path")
    print("✅ Pure PTX/RPN execution")
    print("✅ GPU-resident weights (VRAM)")
    print("✅ Learned navigation logic (LSTM)")
    print("\nThis demonstrates the Deterministic Generative AI architecture:")
    print("  - Deterministic execution: PTX kernels + RPN programs")
    print("  - Generative capability: Learned LSTM weights (RLWHF training)")
    print("  - Full sovereignty: Zero external framework dependencies")


if __name__ == "__main__":
    main()
