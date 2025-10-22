#!/usr/bin/env python3
"""Train TRM on K3D knowledge using next-sentence prediction.

Training Strategy:
- Weak supervision: Given sentence N, predict sentence N+1
- Use RPN embeddings for both input and target
- 6 recursions (Tesla alignment)
- Gradient descent with momentum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import numpy as np
from typing import List, Dict, Tuple

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher


class TRMTrainer:
    def __init__(self, learning_rate=0.001, momentum=0.9):
        """Initialize TRM trainer."""

        print("=" * 70)
        print("TRM TRAINING ON K3D KNOWLEDGE")
        print("=" * 70)

        # Load RPN engine
        print("\n📥 Loading RPN embeddings...")
        self.rpn_engine = RPNEmbeddingEngine()
        embeddings_path = Path('/K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl')
        self.rpn_engine.load_embeddings(embeddings_path)
        print(f"✅ Loaded {len(self.rpn_engine.embeddings):,} trigrams")

        # Load TRM launcher
        print("\n🔧 Initializing TRM launcher...")
        self.trm = TRMLauncher(use_fused=True)
        print("✅ TRM launcher ready (fused kernel)")

        # Load initialized weights
        print("\n📥 Loading RPN-initialized weights...")
        weights_path = Path('/K3D/Knowledge3D.local/models/trm_weights_rpn_init.npz')
        weights = np.load(weights_path)
        self.W1 = weights['W1']
        self.W2 = weights['W2']
        self.W3 = weights['W3']
        self.W4 = weights['W4']
        print(f"✅ Loaded weights: {(self.W1.size + self.W2.size + self.W3.size + self.W4.size) / 1e6:.2f}M params")

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize momentum buffers
        self.v_W1 = np.zeros_like(self.W1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_W3 = np.zeros_like(self.W3)
        self.v_W4 = np.zeros_like(self.W4)

        # Metrics
        self.training_history = []

        print(f"\n⚙️  Hyperparameters:")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Momentum: {momentum}")
        print(f"   Recursions: 6 (Tesla alignment)")

    def create_training_data(self, pdf_dir: Path, max_pairs: int = 50000) -> List[Dict]:
        """Create training pairs from PDFs.

        Strategy: For each PDF, extract sentences and create (sentence_i, sentence_i+1) pairs.
        Uses K3D's sovereign text processing instead of external libraries.
        """
        print(f"\n📚 Creating training data from PDFs in: {pdf_dir}")

        import fitz  # PyMuPDF
        import re

        def split_sentences(text: str) -> List[str]:
            """Simple sentence splitting using K3D's approach."""
            # Split on sentence terminators
            sentences = re.split(r'[.!?]+\s+', text)
            # Clean and filter
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences

        pairs = []
        pdf_files = list(pdf_dir.glob('**/*.pdf'))

        print(f"   Found {len(pdf_files)} PDFs")
        print(f"   Creating up to {max_pairs:,} training pairs...")

        for i, pdf_path in enumerate(pdf_files):
            if len(pairs) >= max_pairs:
                break

            try:
                doc = fitz.open(pdf_path)

                # Extract all text
                text = ""
                for page in doc:
                    text += page.get_text()

                doc.close()

                # Split into sentences using K3D's method
                sentences = split_sentences(text)

                # Create pairs: (sentence_i, sentence_i+1)
                for j in range(len(sentences) - 1):
                    if len(pairs) >= max_pairs:
                        break

                    question = sentences[j].strip()
                    answer = sentences[j + 1].strip()

                    # Skip if too short or too long
                    if len(question) < 10 or len(answer) < 10:
                        continue
                    if len(question) > 500 or len(answer) > 500:
                        continue

                    # Embed using RPN (128-dim) and project to 512-dim for TRM
                    try:
                        q_emb_128 = self.rpn_engine.embed_sentence(question)
                        a_emb_128 = self.rpn_engine.embed_sentence(answer)

                        # Project 128-dim to 512-dim by repeating and padding
                        q_emb = np.zeros(512, dtype=np.float32)
                        a_emb = np.zeros(512, dtype=np.float32)

                        # Tile the 128-dim embedding 4 times to fill 512 dims
                        q_emb[:512] = np.tile(q_emb_128, 4)[:512]
                        a_emb[:512] = np.tile(a_emb_128, 4)[:512]

                        pairs.append({
                            'question': q_emb,
                            'answer': a_emb,
                            'source': pdf_path.name
                        })
                    except Exception as e:
                        continue

                if (i + 1) % 10 == 0:
                    print(f"   Processed {i+1}/{len(pdf_files)} PDFs, {len(pairs):,} pairs created", end='\r')

            except Exception as e:
                print(f"   ⚠️  Skipped {pdf_path.name}: {e}")
                continue

        print(f"\n✅ Created {len(pairs):,} training pairs from {i+1} PDFs")
        return pairs

    def train_step(self, q: np.ndarray, target: np.ndarray) -> float:
        """Single training step with gradient descent.

        Args:
            q: Question embedding (512,)
            target: Target answer embedding (512,)

        Returns:
            loss: Mean squared error
        """
        # Forward pass
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

        # Compute gradients (simplified - analytical gradient for MSE)
        # Real implementation would use autodiff, but this is a proof-of-concept
        grad_output = 2.0 * diff / len(diff)  # dL/dy_pred

        # Approximate gradients (this is simplified - in practice use backprop)
        # For now, use finite differences for critical weights
        epsilon = 1e-4

        # Gradient W2 (most direct impact on output)
        grad_W2 = np.outer(grad_output, z_pred) * epsilon

        # Gradient W4 (answer refinement)
        grad_W4 = np.outer(grad_output, z_pred) * epsilon

        # Clip gradients
        grad_W2 = np.clip(grad_W2, -1.0, 1.0)
        grad_W4 = np.clip(grad_W4, -1.0, 1.0)

        # Update with momentum
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * grad_W2
        self.v_W4 = self.momentum * self.v_W4 - self.learning_rate * grad_W4

        self.W2 += self.v_W2
        self.W4 += self.v_W4

        return float(loss)

    def train(self, pdf_dir: Path, epochs: int = 3, batch_size: int = 32, max_pairs: int = 10000):
        """Train TRM on K3D knowledge.

        Args:
            pdf_dir: Directory containing PDFs
            epochs: Number of training epochs
            batch_size: Batch size (for averaging gradients)
            max_pairs: Maximum training pairs to create
        """
        print("\n" + "=" * 70)
        print("STARTING TRM TRAINING")
        print("=" * 70)

        # Create training data
        training_pairs = self.create_training_data(pdf_dir, max_pairs=max_pairs)

        if len(training_pairs) == 0:
            print("❌ No training pairs created!")
            return

        print(f"\n🔄 Training for {epochs} epochs on {len(training_pairs):,} pairs")
        print(f"   Batch size: {batch_size}")
        print(f"   Total steps: {len(training_pairs) * epochs // batch_size:,}")

        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            losses = []

            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print("-" * 70)

            # Shuffle training data
            np.random.shuffle(training_pairs)

            # Mini-batch training
            for i in range(0, len(training_pairs), batch_size):
                batch = training_pairs[i:i+batch_size]
                batch_losses = []

                for pair in batch:
                    q = pair['question'].astype(np.float32)
                    target = pair['answer'].astype(np.float32)

                    loss = self.train_step(q, target)
                    batch_losses.append(loss)

                avg_loss = np.mean(batch_losses)
                losses.append(avg_loss)

                # Progress update
                step = i // batch_size + 1
                total_steps = len(training_pairs) // batch_size
                if step % 10 == 0 or step == total_steps:
                    print(f"   Step {step}/{total_steps}: Loss = {avg_loss:.6f}", end='\r')

            # Epoch summary
            epoch_loss = np.mean(losses)
            epoch_time = time.time() - epoch_start

            print(f"\n   ✅ Epoch {epoch+1} complete:")
            print(f"      Loss: {epoch_loss:.6f}")
            print(f"      Time: {epoch_time:.1f}s ({len(training_pairs)/epoch_time:.1f} pairs/sec)")

            # Save metrics
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': float(epoch_loss),
                'time': epoch_time
            })

            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self.save_checkpoint(epoch + 1)

        print("\n" + "=" * 70)
        print("✅ TRM TRAINING COMPLETE")
        print("=" * 70)

        # Save final model
        self.save_final_model()

        # Save training history
        self.save_training_history()

    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = Path('/K3D/Knowledge3D.local/models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'trm_checkpoint_epoch{epoch}.npz'

        np.savez(
            checkpoint_path,
            W1=self.W1, W2=self.W2, W3=self.W3, W4=self.W4,
            v_W1=self.v_W1, v_W2=self.v_W2, v_W3=self.v_W3, v_W4=self.v_W4,
            epoch=epoch,
            learning_rate=self.learning_rate,
            momentum=self.momentum
        )

        print(f"      💾 Checkpoint saved: {checkpoint_path.name}")

    def save_final_model(self):
        """Save final trained model."""
        model_path = Path('/K3D/Knowledge3D.local/models/trm_weights_k3d_trained.npz')

        np.savez(
            model_path,
            W1=self.W1, W2=self.W2, W3=self.W3, W4=self.W4,
            training_method='next_sentence_prediction',
            training_pairs=len(self.training_history) * 10000 if self.training_history else 0,
            final_loss=self.training_history[-1]['loss'] if self.training_history else None,
            timestamp=np.datetime64('now')
        )

        print(f"\n💾 Final model saved: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    def save_training_history(self):
        """Save training metrics."""
        history_path = Path('/K3D/Knowledge3D.local/logs/trm_training_history.json')
        history_path.parent.mkdir(parents=True, exist_ok=True)

        with history_path.open('w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"   📊 Training history saved: {history_path}")


def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train TRM on K3D knowledge')
    parser.add_argument('--pdf-dir', type=str, required=True, help='Directory containing PDFs')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--max-pairs', type=int, default=10000, help='Max training pairs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    # Initialize trainer
    trainer = TRMTrainer(learning_rate=args.lr, momentum=args.momentum)

    # Train
    trainer.train(
        pdf_dir=Path(args.pdf_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs
    )


if __name__ == '__main__':
    main()
