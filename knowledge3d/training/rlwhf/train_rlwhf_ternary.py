#!/usr/bin/env python3
"""
RLWHF trainer with ternary gradient descent (sign + sparsity).
"""

from __future__ import annotations

import argparse
import numpy as np

from knowledge3d.training.rlwhf.train_rlwhf import RLWHFTrainer


def ternary_sign(arr: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Quantize gradients to {-1,0,+1} with dead-zone threshold."""
    out = np.zeros_like(arr, dtype=np.float32)
    out[arr > threshold] = 1.0
    out[arr < -threshold] = -1.0
    return out


class RLWHFTernaryTrainer(RLWHFTrainer):
    def __init__(self, *args, grad_threshold: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_threshold = grad_threshold

    def train_step_weighted(
        self,
        q: np.ndarray,
        target: np.ndarray,
        reward_weight: float
    ):
        # Use parent for forward + loss
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)
        y_pred, z_pred = self.trm.refine(
            q, y, z,
            self.W1, self.W2, self.W3, self.W4,
            n_steps=6,
            eps=1e-4
        )
        diff = y_pred - target
        loss = np.mean(diff ** 2)
        effective_weight = reward_weight ** self.reward_scale
        grad_output = 2.0 * diff / len(diff) * effective_weight
        epsilon = 1e-4
        grad_W2 = np.outer(grad_output, z_pred) * epsilon
        grad_W4 = np.outer(grad_output, z_pred) * epsilon

        # Ternary quantization
        g2 = ternary_sign(grad_W2, threshold=self.grad_threshold)
        g4 = ternary_sign(grad_W4, threshold=self.grad_threshold)
        sparsity = float((np.abs(g2) < 1e-6).mean())

        # Momentum updates using ternary gradients
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * g2
        self.v_W4 = self.momentum * self.v_W4 - self.learning_rate * g4
        self.W2 += self.v_W2
        self.W4 += self.v_W4
        return float(loss), float(loss * effective_weight), sparsity


def main():
    ap = argparse.ArgumentParser(description="RLWHF ternary trainer")
    ap.add_argument("--dataset", required=True, help="Path to RLWHF dataset (.npz)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-threshold", type=float, default=1e-3)
    args = ap.parse_args()

    trainer = RLWHFTernaryTrainer(learning_rate=0.0005, momentum=0.9, reward_scale=2.0, grad_threshold=args.grad_threshold)
    trainer.train(dataset_path=args.dataset, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
