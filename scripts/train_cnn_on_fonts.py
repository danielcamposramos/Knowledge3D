#!/usr/bin/env python3
"""
Train DeepSeek CNN on font glyphs - GPU ONLY.

Trains CNN weights to extract features from character images.
Uses the 123K glyph samples from font_db.pkl.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
import random
import ctypes

# GPU imports
from knowledge3d.cranium.sovereign.loader import (
    load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh, gpu_free
)


def render_glyph_image(char: str, font_path: str, size: int = 32) -> np.ndarray:
    """Render character from vector font as RGB image."""
    try:
        font = ImageFont.truetype(font_path, size)
    except:
        return None

    canvas_size = (64, 64)
    canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
    except:
        return None

    if char_w <= 0 or char_h <= 0:
        return None

    x = (canvas_size[0] - char_w) // 2 - bbox[0]
    y = (canvas_size[1] - char_h) // 2 - bbox[1]

    draw.text((x, y), char, font=font, fill=(0, 0, 0))

    return np.array(canvas, dtype=np.uint8)


def main():
    print("=" * 80)
    print("DEEPSEEK CNN TRAINING - GPU ONLY")
    print("=" * 80)

    # Load font database
    font_db_path = Path("/K3D/Knowledge3D.local/font_db.pkl")
    print(f"\n[1/6] Loading glyph database...")
    with open(font_db_path, "rb") as f:
        font_db = pickle.load(f)

    n_fonts = len(font_db)
    n_glyphs = sum(len(font_data["glyphs"]) for font_data in font_db.values())
    print(f"       Found: {n_fonts} fonts, {n_glyphs} glyphs")

    # Build character set and training samples
    print(f"\n[2/6] Preparing training data...")

    chars = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
    )
    char_to_id = {c: i for i, c in enumerate(chars)}

    training_samples = []

    for font_name, font_data in font_db.items():
        font_path = font_data.get("font_path")
        if not font_path or not Path(font_path).exists():
            continue

        if font_data.get("is_symbol_font", False):
            continue

        for char in chars:
            if char not in font_data["glyphs"]:
                continue
            if char not in char_to_id:
                continue

            training_samples.append({
                "char": char,
                "label": char_to_id[char],
                "font_path": font_path
            })

    print(f"       Training samples: {len(training_samples)}")

    # Limit to reasonable number for initial training
    max_samples = 50000
    if len(training_samples) > max_samples:
        random.shuffle(training_samples)
        training_samples = training_samples[:max_samples]
        print(f"       Using subset: {max_samples} samples")

    # Render all training images
    print(f"\n[3/6] Rendering {len(training_samples)} glyph images...")
    images = []
    labels = []

    for idx, sample in enumerate(training_samples):
        if idx % 5000 == 0:
            print(f"       Rendered: {idx}/{len(training_samples)}")

        img = render_glyph_image(sample["char"], sample["font_path"], size=32)
        if img is None:
            continue

        images.append(img)
        labels.append(sample["label"])

    if len(images) == 0:
        print("\n❌ No images rendered!")
        return 1

    images = np.stack(images, axis=0)  # [N, H, W, 3]
    labels = np.array(labels, dtype=np.int32)  # [N]

    print(f"       Dataset ready: {len(images)} images")
    print(f"       Image shape: {images.shape}")

    # Initialize DeepSeek CNN
    print(f"\n[4/6] Initializing DeepSeek CNN model...")
    from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
    from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer

    checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/ocr_gpu_epoch_100")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create CNN model
    cnn_model = DeepSeekOCRModel(num_glyphs=len(chars), input_channels=3)
    print(f"       ✓ GPU model initialized")

    # Create GPU trainer with backpropagation
    print(f"\n[5/6] Initializing GPU trainer with backpropagation...")
    trainer = GPUCNNTrainer(
        model=cnn_model,
        num_classes=len(chars),
        learning_rate=0.01,
        momentum=0.9
    )
    print(f"       ✓ GPU trainer initialized (SGD with momentum)")

    # GPU training loop with backpropagation
    print(f"\n[6/6] Training CNN on character images (GPU backprop)...")

    n_epochs = 5  # Reduced for faster training
    batch_size = 32  # Smaller batch for GPU memory
    n_batches = len(images) // batch_size

    for epoch in range(1, n_epochs + 1):
        print(f"\n  Epoch {epoch}/{n_epochs}")

        # Shuffle data
        indices = np.arange(len(images))
        np.random.shuffle(indices)

        epoch_loss = 0.0
        epoch_acc = 0.0
        n_samples = 0

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(images))
            batch_indices = indices[batch_start:batch_end]

            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]

            # Prepare batch
            batch_imgs_normalized = []
            for img in batch_images:
                # Convert to float32 and normalize to [0, 1]
                img_float = img.astype(np.float32) / 255.0
                batch_imgs_normalized.append(img_float)

            # Train batch (forward + backward + update)
            loss, acc = trainer.train_batch(batch_imgs_normalized, batch_labels.tolist())

            epoch_loss += loss * len(batch_imgs_normalized)
            epoch_acc += acc * len(batch_imgs_normalized)
            n_samples += len(batch_imgs_normalized)

            if batch_idx % 20 == 0:
                print(f"    Batch {batch_idx}/{n_batches} - Loss: {loss:.4f}, Acc: {acc:.2%}")

        avg_loss = epoch_loss / n_samples
        avg_acc = epoch_acc / n_samples
        print(f"  Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2%}")

    # Save trained CNN state
    print(f"\n[7/7] Saving trained CNN weights...")

    # Get model state (updated weights)
    state_dict = cnn_model.get_state_dict()

    # Save as NPZ
    weights_path = checkpoint_dir / "ocr_cnn_weights.npz"
    np.savez(weights_path, **state_dict)

    print(f"       ✓ Weights saved: {weights_path}")
    print(f"       Parameters: {list(state_dict.keys())}")

    print("\n" + "=" * 80)
    print("GPU BACKPROPAGATION TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("✓ CNN trained with full GPU backpropagation")
    print("✓ Weights updated via SGD with momentum")
    print("✓ Classification head trained on character recognition")
    print()
    print("Next: Test APOLLO.PDF with trained CNN")
    print("  python scripts/test_apollo_ground_truth.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
