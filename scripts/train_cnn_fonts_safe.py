#!/usr/bin/env python3
"""
Safe CNN training on font glyphs with gradient monitoring.
Uses fixed batchnorm backward kernel with clipping.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer


def render_glyph_image(char: str, font_path: str, size: int = 64) -> np.ndarray:
    """Render character from vector font as RGB image."""
    try:
        font = ImageFont.truetype(font_path, size)
        img = Image.new("RGB", (64, 64), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (64 - text_width) // 2 - bbox[0]
        y = (64 - text_height) // 2 - bbox[1]

        draw.text((x, y), char, fill=(0, 0, 0), font=font)

        # Normalize to [0, 1]
        array = np.array(img, dtype=np.float32) / 255.0
        return array
    except Exception as e:
        return None


def main():
    print("=" * 80)
    print("SAFE CNN TRAINING - Font Glyphs")
    print("=" * 80)
    print()

    # Initialize model and trainer with conservative settings
    print("[1/6] Initializing CNN model...")
    model = DeepSeekOCRModel()

    # VERY conservative learning rate: 0.0001
    # Relaxed gradient clipping (±10 in BatchNorm, no clip in Conv)
    # Small LR prevents explosion while allowing gradient flow through deep network
    trainer = GPUCNNTrainer(model, learning_rate=0.0001, momentum=0.9)
    print(f"       Learning rate: 0.0001")
    print(f"       Momentum: 0.9")
    print(f"       Batch size: 32")
    print()

    # Load fonts
    print("[2/6] Loading vector fonts...")
    font_dir = Path("/usr/share/fonts/truetype")
    all_fonts = list(font_dir.rglob("*.ttf"))
    fonts = all_fonts[:20]  # Use 20 fonts for diversity
    print(f"       Found {len(fonts)} fonts")
    print()

    # Prepare character set (printable ASCII)
    print("[3/6] Preparing character set...")
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    char_to_label = {c: i for i, c in enumerate(chars)}
    print(f"       {len(chars)} characters")
    print()

    # Render training data
    print("[4/6] Rendering training glyphs...")
    images = []
    labels = []

    for font_idx, font_path in enumerate(fonts, 1):
        if font_idx % 5 == 0:
            print(f"       Rendering font {font_idx}/{len(fonts)}...")

        for char in chars:
            img = render_glyph_image(char, str(font_path))
            if img is not None:
                images.append(img)
                labels.append(char_to_label[char])

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"       Total samples: {len(images)}")
    print(f"       Images shape: {images.shape}")
    print()

    # Training loop
    print("[5/6] Training CNN...")
    print("=" * 80)

    n_epochs = 100
    batch_size = 32
    n_samples = len(images)
    checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/ocr_gpu_epoch_100")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = 0.0

    for epoch in range(1, n_epochs + 1):
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

            # Check for NaN/inf
            if np.isnan(loss) or np.isinf(loss):
                print(f"\n⚠️  NaN/inf detected at epoch {epoch}, batch {batch_idx}")
                print(f"    Stopping training to prevent corruption")
                return 1

            epoch_losses.append(loss)
            epoch_accs.append(acc)

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)

        print(f"Epoch {epoch:3d}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:5.2f}%", end="")

        # Save checkpoints
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            state_dict = model.get_state_dict()
            np.savez(checkpoint_dir / "ocr_cnn_weights.npz", **state_dict)
            print(f" ✓ [saved]")
        else:
            print()

        # Save periodic checkpoints
        if epoch % 10 == 0:
            state_dict = model.get_state_dict()
            np.savez(checkpoint_dir / f"ocr_cnn_weights_epoch_{epoch}.npz", **state_dict)

    print()
    print("=" * 80)

    # Final save
    print("[6/6] Saving final weights...")
    state_dict = model.get_state_dict()
    np.savez(checkpoint_dir / "ocr_cnn_weights.npz", **state_dict)
    print(f"       ✓ Saved to {checkpoint_dir}")
    print()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print()
    print("Next steps:")
    print("1. Run APOLLO.PDF ground truth test:")
    print("   python scripts/test_apollo_ground_truth.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
