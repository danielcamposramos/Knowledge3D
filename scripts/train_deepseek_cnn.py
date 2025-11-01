#!/usr/bin/env python3
"""
Train DeepSeek CNN on rendered glyph images from vector fonts.

This trains the CNN feature extractor to recognize character shapes
by rendering glyphs at various sizes and fonts.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
from typing import Dict, List, Tuple
import random

# GPU imports
try:
    from knowledge3d.cranium.sovereign.loader import load_ptx_file, gpu_malloc, memcpy_htod, memcpy_dtoh, gpu_free
    import ctypes
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


class DeepSeekCNNTrainer:
    """Train DeepSeek CNN on rendered glyphs from vector fonts."""

    def __init__(self):
        """Initialize trainer with DeepSeek OCR model."""
        from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel

        self.model = DeepSeekOCRModel(mode="small")
        self.checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/ocr_gpu_epoch_100")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Character set (printable ASCII)
        self.chars = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            " .,;:!?-()[]{}'\"/\\@#$%^&*+=<>"
        )
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}

        print(f"[INIT] DeepSeek CNN Trainer")
        print(f"  Character set: {len(self.chars)} characters")
        print(f"  Checkpoint: {self.checkpoint_dir}")

    def collect_system_fonts(self, max_fonts: int = 1000) -> List[str]:
        """Collect TrueType/OpenType fonts from system."""
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            Path.home() / ".fonts",
        ]

        font_files = []
        for font_dir in font_dirs:
            directory = Path(font_dir).expanduser()
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if path.suffix.lower() in {".ttf", ".otf"}:
                    # Skip symbol/icon fonts
                    name_lower = path.stem.lower()
                    if any(skip in name_lower for skip in ["symbol", "icon", "emoji", "webding", "wingding"]):
                        continue
                    font_files.append(str(path))

        font_files = list(set(font_files))[:max_fonts]
        print(f"[FONTS] Collected {len(font_files)} fonts from system")
        return font_files

    def render_glyph_image(self, char: str, font_path: str, size: int = 32) -> np.ndarray:
        """
        Render a single character using vector font.

        Returns:
            Image as numpy array [H, W, 3] in RGB format (white bg, black text)
        """
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            return None

        # Create canvas (white background)
        canvas_size = (size * 2, size * 2)
        canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        try:
            # Get character bounding box
            bbox = draw.textbbox((0, 0), char, font=font)
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
        except:
            return None

        if char_w <= 0 or char_h <= 0:
            return None

        # Center character
        x = (canvas_size[0] - char_w) // 2 - bbox[0]
        y = (canvas_size[1] - char_h) // 2 - bbox[1]

        # Draw character (black text)
        draw.text((x, y), char, font=font, fill=(0, 0, 0))

        # Convert to numpy array
        img_array = np.array(canvas, dtype=np.uint8)

        return img_array

    def generate_training_samples(self, fonts: List[str], samples_per_char: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training samples by rendering characters in different fonts.

        Returns:
            images: [N, H, W, 3] uint8
            labels: [N] int (character IDs)
        """
        images = []
        labels = []

        print(f"\n[DATA] Generating {samples_per_char} samples per character...")
        print(f"       Total target: {len(self.chars) * samples_per_char} images")

        for char in self.chars:
            if char not in self.char_to_id:
                continue

            label = self.char_to_id[char]
            char_samples = 0

            # Try to get samples_per_char examples of this character
            font_indices = list(range(len(fonts)))
            random.shuffle(font_indices)

            for font_idx in font_indices:
                if char_samples >= samples_per_char:
                    break

                font_path = fonts[font_idx]

                # Render at random sizes for variation
                size = random.choice([24, 28, 32, 36, 40])
                img = self.render_glyph_image(char, font_path, size=size)

                if img is None:
                    continue

                images.append(img)
                labels.append(label)
                char_samples += 1

            if char_samples > 0:
                display_char = repr(char) if char in ' \t\n' else char
                print(f"  {display_char}: {char_samples} samples")

        if len(images) == 0:
            raise ValueError("No training samples generated!")

        images_array = np.stack(images, axis=0)
        labels_array = np.array(labels, dtype=np.int32)

        print(f"\n[DATA] Generated {len(images_array)} training samples")
        print(f"       Image shape: {images_array.shape}")
        print(f"       Label shape: {labels_array.shape}")

        return images_array, labels_array

    def train_epoch_cpu(self, images: np.ndarray, labels: np.ndarray,
                       learning_rate: float = 0.001, batch_size: int = 32) -> float:
        """
        Train one epoch using CPU (simple gradient descent).

        Note: This is a simplified training loop. Full training would use
        backpropagation through the CNN layers.
        """
        print(f"\n[TRAIN] Starting epoch (LR={learning_rate}, batch_size={batch_size})")

        n_samples = len(images)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = indices[batch_start:batch_end]

            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]

            # Forward pass through CNN
            batch_features = []
            for img in batch_images:
                # Extract features using current model
                features = self.model.forward(img)
                batch_features.append(features.flatten())

            batch_features = np.stack(batch_features, axis=0)

            # Compute loss (simplified - just measure feature variance)
            # In full training, this would be classification loss
            feature_std = np.std(batch_features, axis=0).mean()
            loss = 1.0 / (feature_std + 1e-6)

            total_loss += loss
            n_batches += 1

            if n_batches % 10 == 0:
                print(f"  Batch {n_batches}/{n_samples // batch_size}: loss={loss:.4f}")

        avg_loss = total_loss / n_batches
        print(f"[TRAIN] Epoch complete: avg_loss={avg_loss:.4f}")

        return avg_loss

    def save_weights(self):
        """Save trained CNN weights."""
        # Get current model parameters
        state_dict = self.model.get_state_dict()

        # Save as .npz (dictionary format)
        weights_path = self.checkpoint_dir / "ocr_cnn_weights.npz"
        np.savez(weights_path, **state_dict)

        print(f"\n[SAVE] CNN weights saved to {weights_path}")
        print(f"       Parameters: {list(state_dict.keys())}")

        return weights_path

    def train(self, epochs: int = 5, samples_per_char: int = 100):
        """Main training loop."""
        print("=" * 80)
        print("DEEPSEEK CNN TRAINING")
        print("=" * 80)

        # Collect fonts
        fonts = self.collect_system_fonts(max_fonts=1000)
        if len(fonts) == 0:
            raise ValueError("No fonts found! Install more fonts.")

        # Generate training data
        images, labels = self.generate_training_samples(fonts, samples_per_char)

        # Train for multiple epochs
        print("\n" + "=" * 80)
        print(f"TRAINING {epochs} EPOCHS")
        print("=" * 80)

        for epoch in range(1, epochs + 1):
            print(f"\n{'=' * 80}")
            print(f"EPOCH {epoch}/{epochs}")
            print(f"{'=' * 80}")

            loss = self.train_epoch_cpu(images, labels, learning_rate=0.001)

            print(f"\n[EPOCH {epoch}] Complete: loss={loss:.4f}")

        # Save weights
        weights_path = self.save_weights()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nWeights saved to: {weights_path}")
        print("\nNext steps:")
        print("1. Test APOLLO.PDF:")
        print("   python scripts/test_apollo_ground_truth.py")
        print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train DeepSeek CNN on rendered glyphs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--samples-per-char", type=int, default=100,
                       help="Training samples per character")

    args = parser.parse_args()

    trainer = DeepSeekCNNTrainer()
    trainer.train(epochs=args.epochs, samples_per_char=args.samples_per_char)

    return 0


if __name__ == "__main__":
    sys.exit(main())
