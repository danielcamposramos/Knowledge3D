#!/usr/bin/env python3
"""
Train TRM using RLWHF (Reinforcement Learning with Honesty and Feedback).

Training Strategy:
- Load teacher-evaluated Q&A pairs with ratings (-2 to +2)
- Weight gradient updates by reward (higher reward = stronger update)
- Use pre-computed embeddings from student attempts
- 6 recursions (Tesla alignment)
- Gradient descent with momentum

Usage:
    PYTHONPATH=. python knowledge3d/training/rlwhf/train_rlwhf.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import numpy as np
from typing import Dict, Tuple

from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher


class RLWHFTrainer:
    def __init__(self, learning_rate=0.0005, momentum=0.9, reward_scale=2.0):
        """
        Initialize RLWHF trainer.

        Args:
            learning_rate: Base learning rate for gradient updates
            momentum: Momentum coefficient (0.9 = 90% previous gradient)
            reward_scale: Multiplier for reward weighting (amplifies good/bad distinction)
        """
        print("=" * 80)
        print("RLWHF TRAINING - Reward-Weighted TRM Training")
        print("=" * 80)
        print()

        # Load TRM launcher
        print("[1/3] Initializing TRM launcher...")
        self.trm = TRMLauncher(use_fused=True)
        print("✓ TRM launcher ready (fused kernel)")
        print()

        # Load initialized weights
        print("[2/3] Loading RPN-initialized weights...")
        weights_path = Path('/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz')
        weights = np.load(weights_path)
        self.W1 = weights['W1'].copy()
        self.W2 = weights['W2'].copy()
        self.W3 = weights['W3'].copy()
        self.W4 = weights['W4'].copy()
        total_params = (self.W1.size + self.W2.size + self.W3.size + self.W4.size) / 1e6
        print(f"✓ Loaded weights: {total_params:.2f}M params")
        print()

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.reward_scale = reward_scale

        # Initialize momentum buffers
        self.v_W1 = np.zeros_like(self.W1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_W3 = np.zeros_like(self.W3)
        self.v_W4 = np.zeros_like(self.W4)

        # Metrics
        self.training_history = []

        print("[3/3] Hyperparameters:")
        print(f"  Learning rate:  {learning_rate}")
        print(f"  Momentum:       {momentum}")
        print(f"  Reward scale:   {reward_scale}")
        print(f"  Recursions:     6 (Tesla alignment)")
        print()

    def load_dataset(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Load RLWHF training dataset."""
        print(f"Loading dataset: {dataset_path}")
        dataset = np.load(dataset_path, allow_pickle=True)

        # Extract arrays
        data = {
            'answer_embeddings': dataset['answer_embeddings'],
            'latent_embeddings': dataset['latent_embeddings'],
            'ratings': dataset['ratings'],
            'reward_weights': dataset['reward_weights'],
            'questions': dataset['questions'],
            'answers': dataset['answers'],
        }

        print(f"✓ Loaded dataset:")
        print(f"  Samples: {len(data['ratings'])}")
        print(f"  Answer embeddings: {data['answer_embeddings'].shape}")
        print(f"  Latent embeddings: {data['latent_embeddings'].shape}")
        print(f"  Reward range: [{data['reward_weights'].min():.2f}, {data['reward_weights'].max():.2f}]")
        print()

        return data

    def train_step_weighted(
        self,
        q: np.ndarray,
        target: np.ndarray,
        reward_weight: float
    ) -> Tuple[float, float]:
        """
        Single reward-weighted training step.

        Args:
            q: Question embedding (512,) - not used directly, we use pre-computed latent
            target: Target answer embedding (512,)
            reward_weight: Reward weight [0.0, 1.0] where 1.0 = highest reward

        Returns:
            (loss, effective_loss)
            loss: Unweighted MSE
            effective_loss: Reward-weighted MSE (used for gradient)
        """
        # Forward pass (start with zero latent, let TRM reason from scratch)
        y = np.zeros(512, dtype=np.float32)
        z = np.zeros(512, dtype=np.float32)

        y_pred, z_pred = self.trm.refine(
            q, y, z,
            self.W1, self.W2, self.W3, self.W4,
            n_steps=6,  # Tesla alignment
            eps=1e-4
        )

        # Compute loss (MSE)
        diff = y_pred - target
        loss = np.mean(diff ** 2)

        # Apply reward weighting
        # High reward (good answer) → scale up gradient (learn more from this example)
        # Low reward (bad answer) → scale down gradient (learn less from this example)
        # Amplify the distinction with reward_scale
        effective_weight = reward_weight ** self.reward_scale
        effective_loss = loss * effective_weight

        # Compute gradients
        grad_output = 2.0 * diff / len(diff) * effective_weight  # Weighted gradient

        # Approximate gradients using finite differences
        epsilon = 1e-4

        # Gradient W2 (output projection)
        grad_W2 = np.outer(grad_output, z_pred) * epsilon

        # Gradient W4 (answer refinement)
        grad_W4 = np.outer(grad_output, z_pred) * epsilon

        # Clip gradients for stability
        grad_W2 = np.clip(grad_W2, -1.0, 1.0)
        grad_W4 = np.clip(grad_W4, -1.0, 1.0)

        # Update with momentum
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * grad_W2
        self.v_W4 = self.momentum * self.v_W4 - self.learning_rate * grad_W4

        self.W2 += self.v_W2
        self.W4 += self.v_W4

        return float(loss), float(effective_loss)

    def train(
        self,
        dataset_path: str,
        epochs: int = 5,
        batch_size: int = 32,
        validation_split: float = 0.1
    ):
        """
        Train TRM using RLWHF.

        Args:
            dataset_path: Path to RLWHF training dataset (.npz)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            validation_split: Fraction of data to use for validation
        """
        print("=" * 80)
        print("STARTING RLWHF TRAINING")
        print("=" * 80)
        print()

        # Load dataset
        data = self.load_dataset(dataset_path)
        n_samples = len(data['ratings'])

        # Split train/validation
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        print(f"Dataset split:")
        print(f"  Training:   {n_train} samples ({(1-validation_split)*100:.0f}%)")
        print(f"  Validation: {n_val} samples ({validation_split*100:.0f}%)")
        print()

        print(f"Training configuration:")
        print(f"  Epochs:      {epochs}")
        print(f"  Batch size:  {batch_size}")
        print(f"  Total steps: {(n_train * epochs) // batch_size:,}")
        print()

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(epochs):
            epoch_start = time.time()
            train_losses = []
            train_effective_losses = []

            print(f"{'='*80}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*80}")

            # Shuffle training data
            np.random.shuffle(train_indices)

            # Mini-batch training
            step = 0
            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i+batch_size]
                batch_losses = []
                batch_effective_losses = []

                for idx in batch_indices:
                    # Get question (use answer embedding as input for now)
                    # In future: could use latent_embeddings or question embeddings
                    q = data['answer_embeddings'][idx].astype(np.float32)
                    target = data['answer_embeddings'][idx].astype(np.float32)
                    reward_weight = float(data['reward_weights'][idx])

                    loss, effective_loss = self.train_step_weighted(q, target, reward_weight)
                    batch_losses.append(loss)
                    batch_effective_losses.append(effective_loss)

                avg_loss = np.mean(batch_losses)
                avg_effective_loss = np.mean(batch_effective_losses)
                train_losses.append(avg_loss)
                train_effective_losses.append(avg_effective_loss)

                # Progress update
                step += 1
                total_steps = len(train_indices) // batch_size
                if step % 10 == 0 or step == total_steps:
                    print(f"  Step {step}/{total_steps}: "
                          f"Loss={avg_loss:.6f}, "
                          f"Effective={avg_effective_loss:.6f}",
                          end='\r')

            print()  # New line after progress

            # Validation
            print("  Running validation...")
            val_losses = []
            for idx in val_indices:
                q = data['answer_embeddings'][idx].astype(np.float32)
                target = data['answer_embeddings'][idx].astype(np.float32)

                # Forward pass (no gradient update)
                y = np.zeros(512, dtype=np.float32)
                z = np.zeros(512, dtype=np.float32)
                y_pred, _ = self.trm.refine(q, y, z, self.W1, self.W2, self.W3, self.W4, n_steps=6)

                diff = y_pred - target
                loss = np.mean(diff ** 2)
                val_losses.append(loss)

            # Epoch summary
            epoch_loss = np.mean(train_losses)
            epoch_effective_loss = np.mean(train_effective_losses)
            val_loss = np.mean(val_losses)
            epoch_time = time.time() - epoch_start

            print()
            print(f"  Epoch {epoch+1} Summary:")
            print(f"    Train Loss:      {epoch_loss:.6f}")
            print(f"    Train Effective: {epoch_effective_loss:.6f}")
            print(f"    Val Loss:        {val_loss:.6f}")
            print(f"    Time:            {epoch_time:.1f}s ({n_train/epoch_time:.1f} samples/sec)")

            # Save metrics
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(epoch_loss),
                'train_effective_loss': float(epoch_effective_loss),
                'val_loss': float(val_loss),
                'time': epoch_time
            })

            # Save checkpoint if best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"    ✓ New best validation loss: {val_loss:.6f}")
                self.save_checkpoint(epoch + 1, is_best=True)
            else:
                self.save_checkpoint(epoch + 1, is_best=False)

            print()

        print("=" * 80)
        print("RLWHF TRAINING COMPLETE")
        print("=" * 80)
        print()

        # Save final model
        self.save_final_model()

        # Save training history
        self.save_training_history()

        print(f"✓ Training complete!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path('/K3D/Knowledge3D.local/models/checkpoints/rlwhf')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'trm_rlwhf_epoch_{epoch}.npz'

        np.savez_compressed(
            checkpoint_path,
            W1=self.W1,
            W2=self.W2,
            W3=self.W3,
            W4=self.W4,
            epoch=epoch
        )

        # Also save as "best" if this is the best model so far
        if is_best:
            best_path = checkpoint_dir / 'trm_rlwhf_best.npz'
            np.savez_compressed(
                best_path,
                W1=self.W1,
                W2=self.W2,
                W3=self.W3,
                W4=self.W4,
                epoch=epoch
            )

    def save_final_model(self):
        """Save final trained model."""
        model_dir = Path('/K3D/Knowledge3D.local/models')
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / 'trm_weights_rlwhf_trained.npz'

        np.savez_compressed(
            model_path,
            W1=self.W1,
            W2=self.W2,
            W3=self.W3,
            W4=self.W4,
        )

        print(f"✓ Final model saved: {model_path}")

    def save_training_history(self):
        """Save training history as JSON."""
        import json

        history_dir = Path('/K3D/Knowledge3D.local/models/training_history')
        history_dir.mkdir(parents=True, exist_ok=True)

        history_path = history_dir / 'rlwhf_training_history.json'

        with open(history_path, 'w') as f:
            json.dump({
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'momentum': self.momentum,
                    'reward_scale': self.reward_scale,
                },
                'history': self.training_history,
            }, f, indent=2)

        print(f"✓ Training history saved: {history_path}")


def main():
    # Configuration
    dataset_path = "/K3D/Knowledge3D.local/datasets/rlwhf/rlwhf_training_dataset.npz"

    # Initialize trainer
    trainer = RLWHFTrainer(
        learning_rate=0.0005,  # Slightly lower than standard training
        momentum=0.9,
        reward_scale=2.0,      # Amplify reward signal
    )

    # Train
    trainer.train(
        dataset_path=dataset_path,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
    )

    print()
    print("=" * 80)
    print("Next steps:")
    print("  1. Validate trained model on ARC-AGI or test questions")
    print("  2. Compare performance against untrained baseline")
    print("  3. Iterate with more training data if needed")
    print("=" * 80)


if __name__ == "__main__":
    main()
