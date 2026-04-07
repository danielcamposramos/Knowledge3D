#!/usr/bin/env python3
"""Test parallel LoRA training to verify GPU utilization."""

import numpy as np
import time
from knowledge3d.cranium.sovereign.lora_gpu_trainer import LoRAGPUEngine

def test_parallel_training():
    """Test batched training with 15 samples."""
    print("="*70)
    print("Testing Parallel LoRA Training")
    print("="*70)

    # Setup
    dims = 128
    rank = 16
    n_samples = 150  # Multiple batches
    batch_size = 15  # Process 15 at a time

    # Create synthetic data
    print(f"\nCreating {n_samples} samples (dims={dims}, rank={rank})")
    base_matrix = np.random.randn(dims, dims).astype(np.float32) * 0.01
    A = np.random.randn(dims, rank).astype(np.float32) * 0.01
    B = np.random.randn(rank, dims).astype(np.float32) * 0.01
    inputs = np.random.randn(n_samples, dims).astype(np.float32)
    targets = np.random.randn(n_samples, dims).astype(np.float32)

    # Initialize engine
    print("\nInitializing LoRA GPU engine...")
    engine = LoRAGPUEngine()

    # Allocate buffers
    print(f"Allocating GPU buffers (batch_size={batch_size})...")
    buffers = engine.allocate_buffers(
        base_matrix, A, B, inputs, targets,
        max_batch=batch_size
    )

    # Training parameters
    alpha = 0.5
    learning_rate = 0.001
    epochs = 10

    print(f"\nTraining for {epochs} epochs...")
    print(f"- Samples: {n_samples}")
    print(f"- Batch size: {batch_size}")
    print(f"- Batches per epoch: {(n_samples + batch_size - 1) // batch_size}")
    print()

    # Train
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = np.arange(batch_start, batch_end, dtype=np.int32)

            loss = engine.train_batch(
                buffers=buffers,
                batch_indices=batch_indices,
                dims=dims,
                rank=rank,
                alpha=alpha,
                learning_rate=learning_rate,
            )

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - start_time
        samples_per_sec = (epoch * n_samples) / elapsed

        print(f"Epoch {epoch:2d}/{epochs}: loss={avg_loss:.6f}  "
              f"({samples_per_sec:.1f} samples/sec)")

    # Final stats
    total_time = time.time() - start_time
    total_samples = epochs * n_samples
    throughput = total_samples / total_time

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total time:      {total_time:.2f} seconds")
    print(f"Total samples:   {total_samples}")
    print(f"Throughput:      {throughput:.1f} samples/sec")
    print(f"Time per sample: {1000 * total_time / total_samples:.2f} ms")
    print()
    print("✅ Training completed successfully!")
    print()
    print("NOTE: Check GPU utilization with: nvidia-smi dmon -c 10 -s ucm")
    print("      Expected: 80-95% GPU utilization during training")


if __name__ == "__main__":
    test_parallel_training()
