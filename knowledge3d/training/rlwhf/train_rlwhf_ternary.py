#!/usr/bin/env python3
"""
RLWHF trainer with ternary gradient descent + ternary attention masks.

Soviet Setun heritage: {-1, 0, +1} logic for gradients and attention
Tesla 3-6-9 resonance: 18 batch size, 6 refinement steps, 69 stack depth
"""

from __future__ import annotations

import argparse
import numpy as np

from knowledge3d.training.rlwhf.train_rlwhf import RLWHFTrainer

try:
    from knowledge3d.cranium.sovereign.trm_ternary_launcher import TRMTernaryLauncher
    from knowledge3d.cranium.tools.ternary_attention import TernaryAttention
    TERNARY_ATTENTION_AVAILABLE = True
except ImportError:
    TERNARY_ATTENTION_AVAILABLE = False
    print("⚠️  Ternary attention not available, falling back to gradients-only mode")


def ternary_sign(arr: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Quantize gradients to {-1,0,+1} with dead-zone threshold."""
    out = np.zeros_like(arr, dtype=np.float32)
    out[arr > threshold] = 1.0
    out[arr < -threshold] = -1.0
    return out


class RLWHFTernaryTrainer(RLWHFTrainer):
    """RLWHF trainer with ternary gradients + ternary attention masks.

    Features:
    - Ternary gradient descent: {-1, 0, +1} updates (Soviet Setun)
    - Ternary attention masks: Sparse refinement paths (2× speedup potential)
    - Tesla resonance: 18 batch, 6 steps, 69 stack
    """

    def __init__(
        self,
        *args,
        grad_threshold: float = 1e-3,
        use_ternary_attention: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.grad_threshold = grad_threshold
        self.use_ternary_attention = use_ternary_attention and TERNARY_ATTENTION_AVAILABLE

        # Replace TRM with ternary-capable version
        if self.use_ternary_attention:
            self.trm = TRMTernaryLauncher(use_fused=True)
            self.attn = TernaryAttention(adaptive_thresholds=True)
            print("✨ Ternary attention enabled (Soviet Setun + Tesla 3-6-9)")
        else:
            print("⚠️  Ternary gradients only (attention masks disabled)")

        # Track statistics
        self.ternary_stats = {
            'grad_sparsity': [],
            'attn_sparsity': [],
        }

    def train_step_weighted(
        self,
        q: np.ndarray,
        target: np.ndarray,
        reward_weight: float
    ):
        """Training step with ternary gradients + optional ternary attention."""
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)

        # Compute ternary attention mask if enabled
        ternary_mask = None
        attn_sparsity = 0.0

        if self.use_ternary_attention:
            Q = q.reshape(1, 1, 512)
            ternary_mask = self.attn.compute_masks(Q, Q)
            stats = self.attn.get_sparsity_stats(ternary_mask, seq_len=1)
            attn_sparsity = stats['repel_fraction']

        # Forward pass with optional ternary attention
        if self.use_ternary_attention and ternary_mask is not None:
            y_pred, z_pred = self.trm.refine(
                q, y, z,
                self.W1, self.W2, self.W3, self.W4,
                n_steps=6,
                eps=1e-4,
                ternary_mask=ternary_mask,
            )
        else:
            y_pred, z_pred = self.trm.refine(
                q, y, z,
                self.W1, self.W2, self.W3, self.W4,
                n_steps=6,
                eps=1e-4
            )

        # Loss and gradients
        diff = y_pred - target
        loss = np.mean(diff ** 2)
        effective_weight = reward_weight ** self.reward_scale
        grad_output = 2.0 * diff / len(diff) * effective_weight
        epsilon = 1e-4
        grad_W2 = np.outer(grad_output, z_pred) * epsilon
        grad_W4 = np.outer(grad_output, z_pred) * epsilon

        # Ternary gradient quantization (Soviet Setun)
        g2 = ternary_sign(grad_W2, threshold=self.grad_threshold)
        g4 = ternary_sign(grad_W4, threshold=self.grad_threshold)
        grad_sparsity = float((np.abs(g2) < 1e-6).mean())

        # Momentum updates using ternary gradients
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * g2
        self.v_W4 = self.momentum * self.v_W4 - self.learning_rate * g4
        self.W2 += self.v_W2
        self.W4 += self.v_W4

        # Track ternary statistics
        self.ternary_stats['grad_sparsity'].append(grad_sparsity)
        self.ternary_stats['attn_sparsity'].append(attn_sparsity)

        return float(loss), float(loss * effective_weight), grad_sparsity, attn_sparsity

    def get_ternary_stats_summary(self) -> dict:
        """Get summary of ternary statistics."""
        return {
            'mean_grad_sparsity': np.mean(self.ternary_stats['grad_sparsity']),
            'mean_attn_sparsity': np.mean(self.ternary_stats['attn_sparsity']),
            'std_grad_sparsity': np.std(self.ternary_stats['grad_sparsity']),
            'std_attn_sparsity': np.std(self.ternary_stats['attn_sparsity']),
        }


def main():
    ap = argparse.ArgumentParser(
        description="RLWHF ternary trainer (Soviet Setun + Tesla 3-6-9)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with ternary gradients + attention (Tesla 18 batch)
  python -m knowledge3d.training.rlwhf.train_rlwhf_ternary \\
    --dataset data.npz --batch-size 18 --epochs 10

  # Train with ternary gradients only (no attention masks)
  python -m knowledge3d.training.rlwhf.train_rlwhf_ternary \\
    --dataset data.npz --no-ternary-attention
        """
    )
    ap.add_argument("--dataset", required=True, help="Path to RLWHF dataset (.npz)")
    ap.add_argument("--epochs", type=int, default=1, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=18, help="Batch size (Tesla: 18)")
    ap.add_argument("--grad-threshold", type=float, default=1e-3,
                    help="Threshold for ternary gradient quantization")
    ap.add_argument("--no-ternary-attention", action="store_true",
                    help="Disable ternary attention masks (gradients only)")
    ap.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate")
    ap.add_argument("--momentum", type=float, default=0.9, help="Momentum coefficient")
    ap.add_argument("--reward-scale", type=float, default=2.0, help="Reward scaling exponent")
    args = ap.parse_args()

    print("🎯 RLWHF Ternary Trainer")
    print(f"   Soviet Setun (1958) + Tesla 3-6-9 + K3D Cranium\n")

    trainer = RLWHFTernaryTrainer(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        reward_scale=args.reward_scale,
        grad_threshold=args.grad_threshold,
        use_ternary_attention=not args.no_ternary_attention,
    )

    print(f"📊 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size} (Tesla resonance: {args.batch_size}/3={args.batch_size//3})")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Momentum: {args.momentum}")
    print(f"   Gradient threshold: {args.grad_threshold}")
    print(f"   Ternary attention: {'enabled' if not args.no_ternary_attention else 'disabled'}")
    print()

    trainer.train(dataset_path=args.dataset, epochs=args.epochs, batch_size=args.batch_size)

    # Print ternary statistics summary
    stats = trainer.get_ternary_stats_summary()
    print(f"\n📈 Ternary Statistics:")
    print(f"   Mean gradient sparsity: {stats['mean_grad_sparsity']*100:.1f}% ± {stats['std_grad_sparsity']*100:.1f}%")
    print(f"   Mean attention sparsity: {stats['mean_attn_sparsity']*100:.1f}% ± {stats['std_attn_sparsity']*100:.1f}%")
    print(f"\n💡 Expected speedup from attention: {1.0 + stats['mean_attn_sparsity']*2.0:.2f}× (when kernel-level skip implemented)")


if __name__ == "__main__":
    main()
