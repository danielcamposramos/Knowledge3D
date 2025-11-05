# Codex Mission: Contextual OCR Training - Fix Distribution Mismatch

**Date**: 2025-11-03
**From**: Claude + Daniel (architect)
**To**: Codex
**Priority**: CRITICAL - APOLLO OCR still at F1=0%

---

## Executive Summary

**Problem Identified by Daniel**: "We never trained on words using PDFs that have images and objects"

**Root Cause Analysis**:
1. ❌ Trained on **isolated synthetic glyphs** (white background, 20 fonts)
2. ❌ Tested on **real PDF context** (surrounding text, noise, compression)
3. ❌ **Distribution mismatch**: Synthetic μ=0.85, σ=0.20 → Real μ=0, σ=0.70
4. ❌ **Font coverage**: Used 20 fonts, but **1,999 available** in font_db.pkl
5. ❌ **CNN not learning**: 100 epochs → 0.01% accuracy (gradient flow broken)

**Your Mission**: Train on contextual data using ALL fonts, starting from existing weights

---

## Critical Context: What Daniel Saw

### Issue #1: Limited Font Coverage
```
Font database: 1,999 fonts, 123,938 glyphs
Your training:  20 fonts,   1,240 samples  ← 0.5% of available data!
```

### Issue #2: No Contextual Training
```
Current approach:  Isolated glyphs on white background
Real-world data:   Text in paragraphs, surrounded by other text, images, objects
Result:            Model sees out-of-distribution features
```

### Issue #3: CNN Not Learning
```
Epoch 100/100 | Loss: 4.1272 | Acc: 0.02%  ← Same loss for 100 epochs!
```
This indicates:
- Gradient flow is broken, OR
- Learning rate too low (0.001), OR
- Weights not updating properly

---

## Your Tasks: Build on What Works

### ✅ What's Already Trained (DON'T RETRAIN FROM SCRATCH)

1. **Atomic Character Classifiers** - `/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/`
   - 62 characters (A-Z, a-z, 0-9)
   - 87.77% average accuracy on synthetic data
   - FC heads: `char_*_weights.npz` files
   - **Status**: Good on synthetic, needs adaptation to real PDFs

2. **CNN Base Weights** - `/K3D/Knowledge3D.local/checkpoints/phase_g/ocr_gpu_epoch_100/`
   - DeepSeek OCR model
   - Conv layers: `ocr_cnn_weights_epoch_100.npz`
   - **Status**: Trained but stuck (needs gradient flow fix)

---

## Task 1: Fine-Tune Atomic Classifiers on Real PDF Context ⚡ PRIORITY

**Objective**: Adapt existing 62 atomic classifiers to real PDF features

### Step 1.1: Extract Real PDF Training Patches

Use the **6 scanned PDFs with OCR layers** (ground truth available):

```python
#!/usr/bin/env python3
"""Extract character patches from scanned PDFs with OCR ground truth."""

import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel

scanned_pdfs = [
    "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Calculus/Advanced_Calculus.pdf",
    "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Numerology/The Greek Qabalah_ Alphabetic Mysticism and Numerology in the Ancient World-State University of New York Press (2003).pdf",
    "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Context/ssoar-1987-iversen-introduction_to_contextual_analysis.pdf",
    "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/19990053708.pdf",
    "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/19730016146.pdf",
    "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/RPG Design/LA1-Cover.pdf",
]

# For each PDF:
# 1. Extract OCR text layer (ground truth)
# 2. Render page as image
# 3. Match bounding boxes from OCR layer to image regions
# 4. Crop character patches (with surrounding context!)
# 5. Store: (patch_image, character_label, context_features)

# Example for one PDF:
doc = fitz.open(scanned_pdfs[0])
page = doc[0]

# Get OCR text with bounding boxes
text_dict = page.get_text("dict")

# Render page as image
pix = page.get_pixmap(dpi=200)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

# For each character in OCR layer:
for block in text_dict["blocks"]:
    if "lines" not in block:
        continue
    for line in block["lines"]:
        for span in line["spans"]:
            text = span["text"]
            bbox = span["bbox"]  # (x0, y0, x1, y1)

            # Crop patch WITH CONTEXT (expand bbox by 20%)
            x0, y0, x1, y1 = bbox
            w, h = x1 - x0, y1 - y0
            x0_ctx = max(0, x0 - w * 0.2)
            x1_ctx = min(img.shape[1], x1 + w * 0.2)
            y0_ctx = max(0, y0 - h * 0.2)
            y1_ctx = min(img.shape[0], y1 + h * 0.2)

            patch = img[int(y0_ctx):int(y1_ctx), int(x0_ctx):int(x1_ctx)]

            # Extract features using existing CNN
            model = DeepSeekOCRModel()
            model.load_weights("path/to/ocr_cnn_weights_epoch_100.npz")
            result = model.forward(patch, cache_for_backward=True)
            features = result["feature_map"].mean(axis=(0, 1))

            # Store for training
            yield {
                "char": text[0] if text else None,
                "features": features,  # Real PDF features!
                "label": ord(text[0]) if text else None,
            }
```

**Expected output**: 10,000-50,000 real PDF character patches with ground truth

### Step 1.2: Fine-Tune FC Classifiers on Real Features

**CRITICAL**: Don't retrain from scratch! Fine-tune existing classifiers.

```python
#!/usr/bin/env python3
"""Fine-tune atomic classifiers on real PDF features."""

from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer
from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel

# Load existing CNN weights
model = DeepSeekOCRModel()
model.load_weights("/K3D/Knowledge3D.local/checkpoints/phase_g/ocr_gpu_epoch_100/ocr_cnn_weights_epoch_100.npz")

# For each character (A-Z, a-z, 0-9):
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
    char_id = ord(char)

    # Load existing FC weights
    weight_path = f"/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/char_{char_id}_{char}_weights.npz"
    weights = np.load(weight_path)
    fc_weight = weights["fc_weight"]
    fc_bias = weights["fc_bias"]

    # Create trainer with EXISTING weights
    trainer = GPUCNNTrainer(
        model=model,
        num_classes=2,  # Binary: is_char / not_char
        fc_only=True,   # Freeze CNN, train FC only
        learning_rate=0.001,
        momentum=0.9
    )

    # Initialize with existing weights (transfer learning)
    trainer.fc_weight = fc_weight
    trainer.fc_bias = fc_bias

    # Load real PDF patches for this character
    positive_samples = [p for p in pdf_patches if p["char"] == char]
    negative_samples = [p for p in pdf_patches if p["char"] != char and p["char"] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"]

    # Fine-tune on real data (10-50 epochs, not 3000!)
    for epoch in range(10):
        # Mix positive and negative samples
        batch = positive_samples[:32] + negative_samples[:32]
        labels = [1] * len(positive_samples[:32]) + [0] * len(negative_samples[:32])

        # Train batch
        loss, acc = trainer.train_batch(
            [sample["patch"] for sample in batch],
            labels
        )

        print(f"Char '{char}' Epoch {epoch}/10 - Loss: {loss:.4f}, Acc: {acc:.2%}")

    # Save fine-tuned weights (overwrite existing)
    np.savez(
        weight_path,
        fc_weight=trainer.fc_weight,
        fc_bias=trainer.fc_bias
    )

    print(f"✓ Fine-tuned '{char}' on {len(positive_samples)} real PDF samples")
```

**Expected outcome**: Atomic classifiers adapt to real PDF feature distribution

---

## Task 2: Retrain CNN on Contextual Synthetic Data (Long-term Fix)

**Objective**: Train CNN to extract features from text in context, not isolated glyphs

### Step 2.1: Render Contextual Training Data with ALL Fonts

**Use ALL 1,999 fonts**, not just 20!

```python
#!/usr/bin/env python3
"""Render contextual text using ALL fonts from font_db.pkl."""

import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load FULL font database
with open("/K3D/Knowledge3D.local/font_db.pkl", "rb") as f:
    font_db = pickle.load(f)

print(f"Total fonts: {len(font_db)}")  # Should be 1,999

# Render words and sentences, not isolated glyphs
words = [
    "APOLLO", "Teacher", "Resource", "ICASE", "NASA", "the", "and", "for",
    "Hello", "World", "Python", "Code", "Test", "Data", "Model", "Train",
    # Add 1000+ common words
]

training_samples = []

for font_name, font_data in font_db.items():
    font_path = font_data.get("font_path")
    if not font_path or not Path(font_path).exists():
        continue

    if font_data.get("is_symbol_font", False):
        continue

    # Render multiple words per font
    for word in words[:100]:  # 100 words per font = 199,900 samples
        # Render word as sentence context
        sentence = f"The quick brown fox jumps over {word} and runs away."

        # Render sentence
        img = render_sentence(sentence, font_path, size=16)

        # Extract individual character patches from sentence
        # (with surrounding context!)
        for char_idx, char in enumerate(word):
            # Get bounding box of this character in sentence
            bbox = get_char_bbox(sentence, char_idx + word_start_idx, font_path)

            # Crop patch WITH CONTEXT (include surrounding chars)
            patch = crop_with_context(img, bbox, context_ratio=0.3)

            training_samples.append({
                "image": patch,
                "label": ord(char),
                "context": "in_sentence"  # Flag: not isolated
            })

print(f"Total contextual samples: {len(training_samples)}")
# Expected: ~200,000 samples (1,999 fonts × 100 words)
```

### Step 2.2: Add PDF-Style Augmentation

Make synthetic data look like real PDFs:

```python
def augment_pdf_style(img: np.ndarray) -> np.ndarray:
    """Apply PDF-like degradation to synthetic image."""

    # 1. Add Gaussian noise (scanner noise)
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img_noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

    # 2. JPEG compression artifacts
    from io import BytesIO
    from PIL import Image
    pil_img = Image.fromarray(img_noisy)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=60)  # Low quality = compression
    buffer.seek(0)
    img_compressed = np.array(Image.open(buffer))

    # 3. Slight blur (PDF rendering artifacts)
    from scipy.ndimage import gaussian_filter
    img_blurred = gaussian_filter(img_compressed, sigma=0.5)

    # 4. Background texture (paper texture)
    texture = np.random.randint(245, 255, img.shape, dtype=np.uint8)
    img_final = (img_blurred * 0.9 + texture * 0.1).astype(np.uint8)

    return img_final
```

### Step 2.3: Train CNN with Gradient Flow Fix

**Fix the gradient flow issue** (current training stuck at 4.1272 loss):

```python
from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer
from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel

# Load existing weights as starting point
model = DeepSeekOCRModel(num_glyphs=62, input_channels=3)
model.load_weights("/K3D/Knowledge3D.local/checkpoints/phase_g/ocr_gpu_epoch_100/ocr_cnn_weights_epoch_100.npz")

# Create trainer with HIGHER learning rate
trainer = GPUCNNTrainer(
    model=model,
    num_classes=62,
    learning_rate=0.01,  # Was 0.001 - increase 10x
    momentum=0.9
)

# Check gradient flow
print("Testing gradient flow...")
test_batch = training_samples[:32]
test_labels = [sample["label"] for sample in test_batch]

loss_before = trainer.compute_loss(test_batch, test_labels)
trainer.train_batch(test_batch, test_labels)
loss_after = trainer.compute_loss(test_batch, test_labels)

if abs(loss_before - loss_after) < 0.001:
    raise RuntimeError("Gradient flow broken! Loss not changing.")
else:
    print(f"✓ Gradient flow OK: Δloss = {abs(loss_before - loss_after):.6f}")

# Train on contextual + augmented data
n_epochs = 50  # Reduced from 100 (should converge faster with better data)
batch_size = 64

for epoch in range(1, n_epochs + 1):
    # Shuffle and augment
    np.random.shuffle(training_samples)

    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx in range(0, len(training_samples), batch_size):
        batch = training_samples[batch_idx:batch_idx + batch_size]

        # Augment on-the-fly
        batch_images = [augment_pdf_style(s["image"]) for s in batch]
        batch_labels = [s["label"] for s in batch]

        # Train
        loss, acc = trainer.train_batch(batch_images, batch_labels)

        epoch_loss += loss
        epoch_acc += acc

    avg_loss = epoch_loss / (len(training_samples) // batch_size)
    avg_acc = epoch_acc / (len(training_samples) // batch_size)

    print(f"Epoch {epoch}/{n_epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2%}")

    # Save every 10 epochs
    if epoch % 10 == 0:
        model.save_weights(f"ocr_cnn_contextual_epoch_{epoch}.npz")

# Save final weights
model.save_weights("ocr_cnn_contextual_final.npz")
```

---

## Task 3: Validation Protocol

### Test 1: Feature Distribution Match
```python
# Compare old vs new feature distributions
old_features = extract_features_with_old_cnn(apollo_patches)
new_features = extract_features_with_new_cnn(apollo_patches)

print(f"Old: μ={old_features.mean():.3f}, σ={old_features.std():.3f}")
print(f"New: μ={new_features.mean():.3f}, σ={new_features.std():.3f}")
print(f"Should be closer to real PDF: μ≈0, σ≈0.70")
```

### Test 2: APOLLO Ground Truth
```bash
python scripts/test_apollo_ground_truth.py
```
**Success criteria**: F1 > 30% (up from 0%)

### Test 3: Atomic Classifier Accuracy on Real Data
```python
# Test fine-tuned classifiers on held-out real PDF patches
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
    test_patches = get_real_pdf_patches(char, n=100)
    predictions = classifier.predict(test_patches)
    accuracy = (predictions == char).mean()
    print(f"Char '{char}': {accuracy:.1%}")

# Expected: >70% average (up from ~30% on synthetic)
```

---

## Critical Success Factors

### ✅ DO:
1. **Use ALL 1,999 fonts** from font_db.pkl (not 20!)
2. **Load existing weights** as starting point (transfer learning, not from scratch)
3. **Render text in context** (words/sentences, not isolated glyphs)
4. **Add PDF-style augmentation** (noise, compression, blur, background)
5. **Fix gradient flow** (test that loss actually decreases)
6. **Fine-tune on real PDFs** (use the 6 scanned PDFs with OCR ground truth)

### ❌ DON'T:
1. ❌ Train from random initialization (wastes existing work)
2. ❌ Use only 20 fonts (0.5% of available data)
3. ❌ Train on isolated white-background glyphs (distribution mismatch)
4. ❌ Ignore gradient flow issues (loss stuck = not learning)
5. ❌ Skip validation on real data (synthetic accuracy doesn't transfer)

---

## Timeline & Priorities

### Phase 1: Quick Win (Hours) ⚡
1. Fine-tune existing atomic classifiers on real PDF patches
2. Re-test APOLLO → expect F1 > 30%

### Phase 2: Full Solution (1-2 Days)
1. Retrain CNN on contextual data with ALL fonts
2. Add PDF-style augmentation
3. Fix gradient flow issue
4. Re-test APOLLO → expect F1 > 70%

### Phase 3: Production Ready (3-5 Days)
1. Train on all 6 scanned PDFs
2. Per-character threshold calibration
3. Ensemble methods
4. Re-test APOLLO → expect F1 > 90%

---

## Resources & References

### Data Sources
- **Font database**: `/K3D/Knowledge3D.local/font_db.pkl` (1,999 fonts, 123,938 glyphs)
- **Scanned PDFs with OCR**: 6 PDFs listed above (~1,000 pages total)
- **Existing weights**: `/K3D/Knowledge3D.local/checkpoints/phase_g/`

### Code Examples
- **PDF patch extraction**: See Step 1.1 above
- **Contextual rendering**: See Step 2.1 above
- **PDF augmentation**: See Step 2.2 above
- **Gradient flow check**: See Step 2.3 above

### Validation
- **APOLLO test**: `python scripts/test_apollo_ground_truth.py`
- **Feature stats**: `python scripts/debug_feature_stats.py`
- **Character IDs**: `python scripts/debug_char_ids.py`

---

## Daniel's Insight: The Missing Piece

**Quote**: "We never trained on words using PDFs that have images and objects"

**Translation**:
- Current training: Synthetic glyphs on clean white background
- Real-world use: Text surrounded by other text, images, graphics, noise
- **Solution**: Train on contextual data that matches real-world usage

This is the key to bridging the synthetic→real distribution gap. Contextual training + PDF augmentation + ALL fonts = working APOLLO OCR.

---

## Questions & Clarifications

1. **Should I fine-tune atomic classifiers first** (quick win) **or retrain CNN** (full solution)?
   - **Answer**: Do both in parallel! Fine-tune for quick validation, retrain for long-term fix.

2. **How many real PDF patches do I need** for fine-tuning?
   - **Answer**: 100-200 per character minimum. More is better. The 6 PDFs should provide 10,000+ total.

3. **Should I retrain from scratch or continue** from epoch 100?
   - **Answer**: Continue from existing weights! They're stuck but not broken. Higher LR + better data will unstick them.

4. **What if APOLLO is still failing** after fine-tuning?
   - **Answer**: Check feature statistics (debug_feature_stats.py). If still mismatched, add more augmentation.

---

**Ready to proceed? Start with Task 1 (fine-tuning) for quick validation, then Task 2 (contextual retraining) for the full fix. Good luck! 🚀**
