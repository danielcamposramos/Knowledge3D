# Codex Mission: Extend Proven Training Script for Contextual OCR

**Date**: 2025-11-03
**From**: Claude + Daniel (architect)
**To**: Codex
**Priority**: CRITICAL - Leverage what works, fix what doesn't

---

## Daniel's Key Insight

> **"I am afraid Codex will use the old paradigm to craft that training script, can you leverage what worked for the characters? Since we have a solid base, we do not need 3000 epochs, my guess is half - 1500 epochs each character"**

**Translation**: Don't reinvent the wheel. **Extend** `train_atomic_character.py` which achieved **87.77% accuracy** on synthetic data.

---

## Executive Summary

**What Works** (KEEP THIS):
- ✅ `scripts/train_atomic_character.py` - 87.77% accuracy on synthetic data
- ✅ GPU-sovereign training pipeline
- ✅ Binary FC classifiers per character
- ✅ RPN embedding integration
- ✅ Augmentation pipeline

**What Needs Fixing** (CHANGE THIS):
- ❌ **Font coverage**: Uses ~20 fonts → Need ALL 1,999 fonts
- ❌ **Context**: Isolated glyphs → Need words/sentences
- ❌ **Augmentation**: Basic transforms → Need PDF-style degradation
- ❌ **Epochs**: 3000 → 1500 (better data = faster convergence)

---

## Your Mission: Three-Step Enhancement

### Step 1: Add Contextual Rendering Function 🎯

**Location**: `scripts/train_atomic_character.py`, after `render_glyph_image()` (line ~246)

**Add this new function**:

```python
def render_contextual_glyph(char: str, font_path: str, size: int = 64, context: bool = True) -> Optional[np.ndarray]:
    """
    Render character in contextual setting (surrounded by other text).

    Args:
        char: Target character to render
        font_path: Path to font file
        size: Font size
        context: If True, render in sentence context; if False, isolated (fallback)

    Returns:
        RGB image array [64, 64, 3] normalized to [0, 1]
    """
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None

    if not context:
        # Fallback to isolated rendering (original behavior)
        return render_glyph_image(char, font_path, size)

    # Render character in sentence context
    # Build contextual sentence with target char in the middle
    prefix_chars = "The quick "
    suffix_chars = " jumps over"
    sentence = f"{prefix_chars}{char}{suffix_chars}"

    # Create larger canvas for full sentence
    canvas_width = 256
    canvas_height = 64
    bg_color = (255, 255, 255)
    img = Image.new("RGB", (canvas_width, canvas_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Render full sentence
    try:
        draw.text((10, 10), sentence, fill=(0, 0, 0), font=font)
    except Exception:
        return render_glyph_image(char, font_path, size)  # Fallback

    # Find bounding box of target character
    # Measure prefix to find char position
    try:
        prefix_bbox = draw.textbbox((10, 10), prefix_chars, font=font)
        prefix_width = prefix_bbox[2] - prefix_bbox[0]

        # Get target char bbox (with context!)
        char_bbox = draw.textbbox((10 + prefix_width, 10), char, font=font)
        char_x0 = char_bbox[0]
        char_y0 = char_bbox[1]
        char_x1 = char_bbox[2]
        char_y1 = char_bbox[3]

        # Expand bbox to include surrounding context (±30% on each side)
        char_w = char_x1 - char_x0
        char_h = char_y1 - char_y0
        context_x0 = max(0, char_x0 - int(char_w * 0.3))
        context_x1 = min(canvas_width, char_x1 + int(char_w * 0.3))
        context_y0 = max(0, char_y0 - int(char_h * 0.3))
        context_y1 = min(canvas_height, char_y1 + int(char_h * 0.3))

    except Exception:
        # If bbox extraction fails, fallback to isolated
        return render_glyph_image(char, font_path, size)

    # Crop patch WITH CONTEXT
    img_array = np.array(img, dtype=np.uint8)
    patch = img_array[context_y0:context_y1, context_x0:context_x1]

    # Resize to 64x64
    patch_pil = Image.fromarray(patch)
    patch_resized = patch_pil.resize((64, 64), Image.Resampling.LANCZOS)

    # Convert to float32 [0, 1]
    array = np.array(patch_resized, dtype=np.float32) / 255.0

    return array
```

**Why this works**:
- Renders character **in sentence context** (surrounded by other text)
- Extracts patch WITH surrounding characters (±30% expansion)
- Matches real-world PDF usage (text never appears isolated)
- Falls back to isolated rendering if context fails

---

### Step 2: Add PDF-Style Augmentation 📄

**Location**: `scripts/train_atomic_character.py`, modify `augment_character_patch()` (line ~249)

**Add PDF degradation** to the existing augmentation function:

```python
def augment_pdf_style(img: np.ndarray) -> np.ndarray:
    """
    Apply PDF-like degradation to match real scanned documents.

    Input: uint8 image [H, W, 3]
    Output: uint8 image [H, W, 3] with PDF-style artifacts
    """
    # 1. Add scanner noise
    noise = np.random.normal(0, 3, img.shape).astype(np.float32)
    img_noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 2. JPEG compression artifacts (50% chance)
    if random.random() < 0.5:
        from io import BytesIO
        pil_img = Image.fromarray(img_noisy)
        buffer = BytesIO()
        quality = random.randint(50, 80)  # Low quality = compression
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img_noisy = np.array(Image.open(buffer))

    # 3. Slight blur (PDF rendering artifacts)
    if random.random() < 0.3 and cv2 is not None:
        kernel_size = random.choice([3, 5])
        img_noisy = cv2.GaussianBlur(img_noisy, (kernel_size, kernel_size), 0.5)

    # 4. Background texture (paper texture) - very subtle
    if random.random() < 0.2:
        texture = np.random.randint(248, 255, img.shape, dtype=np.uint8)
        img_noisy = (img_noisy.astype(np.float32) * 0.95 +
                     texture.astype(np.float32) * 0.05).astype(np.uint8)

    return img_noisy


# Then MODIFY augment_character_patch() to call augment_pdf_style():
def augment_character_patch(img: np.ndarray, pdf_augment: bool = True) -> List[np.ndarray]:
    """Augment a glyph image to match patch-level inference distribution."""
    base_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Apply PDF-style augmentation FIRST (before geometric transforms)
    if pdf_augment and random.random() < 0.5:
        base_uint8 = augment_pdf_style(base_uint8)

    augmented: List[np.ndarray] = [base_uint8]

    # ... rest of existing augmentation code stays the same ...
    # (crops, scaling, rotations, etc.)
```

**Why this works**:
- Adds noise, compression, blur, texture to match real PDFs
- Applied probabilistically (50% chance) to maintain variety
- Works WITH existing geometric augmentations (crops, scales)
- Bridges synthetic→real distribution gap

---

### Step 3: Modify Font Loading to Use ALL Fonts 🎨

**Location**: `scripts/train_atomic_character.py`, modify `train_single_character()` (line ~458)

**Change this**:
```python
# OLD (line ~474):
fonts = load_fonts_for_script(script, n_fonts)  # Uses DEFAULT_FONTS_PER_SCRIPT (~20)
```

**To this**:
```python
# NEW - Use ALL fonts from font_db.pkl:
print("[1/6] Loading ALL fonts from font database...")
import pickle
font_db_path = Path("/K3D/Knowledge3D.local/font_db.pkl")
if font_db_path.exists():
    with open(font_db_path, "rb") as f:
        font_db = pickle.load(f)

    # Filter to non-symbol fonts
    all_fonts = []
    for font_name, font_data in font_db.items():
        font_path = font_data.get("font_path")
        if not font_path or not Path(font_path).exists():
            continue
        if font_data.get("is_symbol_font", False):
            continue
        # Check if font has target character
        if target_char in font_data.get("glyphs", []):
            all_fonts.append(Path(font_path))

    print(f"       Using {len(all_fonts)} fonts that support '{target_char}'")
    fonts = all_fonts
else:
    # Fallback to script-specific fonts
    fonts = load_fonts_for_script(script, n_fonts)
    print(f"       Using {len(fonts)} fonts for script '{script}'")
```

**Why this works**:
- Uses **ALL 1,999 fonts** that support the target character
- Filters out symbol fonts (Wingdings, etc.)
- Only uses fonts that actually have the character (avoids missing glyphs)
- 100x more training data variety

---

### Step 4: Update Training Parameters 🎛️

**Location**: `scripts/train_all_atomic_characters.py` (batch training script)

**Change this**:
```python
# OLD:
n_epochs=3000
```

**To this**:
```python
# NEW (Daniel's recommendation):
n_epochs=1500  # Half of 3000 - better data converges faster
```

**Also update** the individual character training call to use contextual rendering:

```python
# In _build_dataset() function, line ~360:
# OLD:
glyph = render_glyph_image(target_char, str(font_path))

# NEW:
glyph = render_contextual_glyph(target_char, str(font_path), context=True)
```

---

## Summary of Changes

### Minimal Code Changes (Leverage Existing)

| Component | Status | Change Required |
|-----------|--------|-----------------|
| Training loop | ✅ Keep | None (works great!) |
| GPU trainer | ✅ Keep | None (87.77% accuracy) |
| Augmentation | 🔧 Extend | Add `augment_pdf_style()` |
| Rendering | 🔧 Extend | Add `render_contextual_glyph()` |
| Font loading | 🔧 Modify | Use ALL 1,999 fonts |
| Epochs | 🔧 Modify | 3000 → 1500 |

**Total lines of code to add**: ~100 lines
**Existing code preserved**: ~600 lines (no rewrite!)

---

## Testing Protocol

### Test 1: Verify Font Coverage
```bash
# Should see "Using 1500-1999 fonts that support 'A'" (not 20!)
python scripts/train_atomic_character.py --char A --epochs 1
```

### Test 2: Verify Contextual Rendering
```python
# Quick visual test
from scripts.train_atomic_character import render_contextual_glyph
import matplotlib.pyplot as plt

font_path = "/path/to/some/font.ttf"
isolated = render_glyph_image('A', font_path)
contextual = render_contextual_glyph('A', font_path, context=True)

plt.subplot(1, 2, 1)
plt.imshow(isolated)
plt.title("Isolated (OLD)")

plt.subplot(1, 2, 2)
plt.imshow(contextual)
plt.title("Contextual (NEW)")

plt.show()
# Should see 'A' surrounded by "The quick" and "jumps" in second image
```

### Test 3: Verify PDF Augmentation
```python
# Check augmented samples have noise/compression
from scripts.train_atomic_character import augment_pdf_style
import numpy as np

test_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
augmented = augment_pdf_style(test_img)

print(f"Original range: {test_img.min()}-{test_img.max()}")
print(f"Augmented range: {augmented.min()}-{augmented.max()}")
print(f"Mean difference: {np.abs(test_img.astype(float) - augmented.astype(float)).mean():.2f}")
# Should see some pixel differences from noise/compression
```

### Test 4: Full Training Run (Single Character)
```bash
# Train 'A' with new pipeline (1500 epochs)
python scripts/train_atomic_character.py --char A --epochs 1500 --fonts 1999

# Expected output:
# "Using 1500-1999 fonts that support 'A'"
# "Best accuracy: >85%" (should match or exceed 87.77%)
# Training time: ~same as before (more data, but fewer epochs)
```

### Test 5: APOLLO Validation
```bash
# After training all 62 chars with new method:
python scripts/test_apollo_ground_truth.py

# Expected: F1 > 30% (up from 0%)
# If F1 still 0%, debug feature distribution (use debug_feature_stats.py)
```

---

## Implementation Checklist

### Phase 1: Code Changes (1-2 hours)
- [ ] Add `render_contextual_glyph()` function
- [ ] Add `augment_pdf_style()` function
- [ ] Modify `augment_character_patch()` to call PDF augmentation
- [ ] Modify font loading to use ALL fonts from font_db.pkl
- [ ] Update epochs: 3000 → 1500
- [ ] Verify code compiles (no syntax errors)

### Phase 2: Single Character Test (30 min)
- [ ] Train character 'A' with new pipeline
- [ ] Verify: Using 1500+ fonts
- [ ] Verify: Accuracy >85%
- [ ] Verify: Contextual rendering working (visual check)
- [ ] Verify: PDF augmentation working (visual check)

### Phase 3: Full Training (1-2 days)
- [ ] Train all 62 characters (A-Z, a-z, 0-9)
- [ ] Each character: 1500 epochs
- [ ] Save checkpoints: `/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars_contextual/`
- [ ] Log results: `/tmp/contextual_training.log`

### Phase 4: Validation (1 hour)
- [ ] Test APOLLO.PDF (expect F1 > 30%)
- [ ] Compare feature distributions (should be closer to real PDFs)
- [ ] If F1 still low, analyze failures and iterate

---

## Critical DO/DON'T

### ✅ DO:
1. **Extend existing script** (`train_atomic_character.py`)
2. **Keep GPU sovereignty** (no CPU fallbacks)
3. **Keep RPN integration** (trigram embeddings)
4. **Use ALL 1,999 fonts** from font_db.pkl
5. **Add contextual rendering** (characters in sentences)
6. **Add PDF augmentation** (noise, compression, blur)
7. **Reduce epochs to 1500** (better data = faster convergence)

### ❌ DON'T:
1. ❌ Create new training script from scratch
2. ❌ Remove GPU sovereignty (Daniel: "We fix what is not GPU, we do not fallback")
3. ❌ Remove RPN embeddings
4. ❌ Use only 20 fonts
5. ❌ Keep isolated glyph rendering
6. ❌ Skip PDF augmentation
7. ❌ Use 3000 epochs (overkill with better data)

---

## Expected Outcomes

### Immediate (After Phase 2)
- ✅ Single character training works with contextual rendering
- ✅ Using 1,500-1,999 fonts (100x more than before)
- ✅ Accuracy ≥85% maintained (same as synthetic)

### Short-term (After Phase 3)
- ✅ All 62 characters trained on contextual data
- ✅ Feature distribution closer to real PDFs
- ✅ APOLLO F1: 0% → 30-50%

### Long-term (After Phase 4 + Iteration)
- ✅ APOLLO F1: 50% → 70%+
- ✅ Production-ready OCR
- ✅ Generalizes to other PDFs

---

## Daniel's Wisdom

**Quote**: "I am afraid Codex will use the old paradigm to craft that training script, can you leverage what worked for the characters?"

**Translation**:
- Don't reinvent: **Extend** `train_atomic_character.py`
- Don't overtrain: **1500 epochs** (half of 3000)
- Don't waste data: Use **ALL 1,999 fonts**
- Don't ignore context: Render **characters in sentences**
- Don't skip reality: Add **PDF-style augmentation**

**The proven pipeline + better data = working APOLLO OCR**

---

**Ready to proceed? Make the minimal changes above, test on 'A', then train all 62 characters. Good luck! 🚀**
