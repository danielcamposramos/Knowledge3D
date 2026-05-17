#!/usr/bin/env python3
"""Quick training test with random data"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer

print("Initializing model...")
model = DeepSeekOCRModel()
trainer = GPUCNNTrainer(model, learning_rate=0.09, momentum=0.9, num_classes=10)

# Generate random training data
np.random.seed(42)
n_samples = 100
images = []
labels = []

for i in range(n_samples):
    img = np.random.rand(64, 64, 3).astype(np.float32)  # [0, 1] range
    label = i % 10
    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels, dtype=np.int32)

print(f"Generated {n_samples} random samples")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Train for 3 epochs
n_epochs = 3
batch_size = 48

print(f"\nTraining for {n_epochs} epochs...")
print("=" * 60)

for epoch in range(1, n_epochs + 1):
    print(f"\nEpoch {epoch}/{n_epochs}")

    # Shuffle data
    indices = np.random.permutation(n_samples)
    images_shuffled = images[indices]
    labels_shuffled = labels[indices]

    epoch_losses = []
    epoch_accs = []

    # Train in batches
    n_batches = n_samples // batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size

        batch_imgs = images_shuffled[start:end]
        batch_labels = labels_shuffled[start:end]

        loss, acc = trainer.train_batch(batch_imgs, batch_labels)
        epoch_losses.append(loss)
        epoch_accs.append(acc)

        if batch_idx % 1 == 0:
            print(f"  Batch {batch_idx+1}/{n_batches}: Loss={loss:.4f}, Acc={acc:.2f}%")

    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accs)
    print(f"\n  Epoch {epoch} Summary: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")

print("\n" + "=" * 60)
print("Training complete!")
