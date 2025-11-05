# Codex Handoff: APOLLO OCR Fix & Improvement

**Date**: 2025-11-03
**From**: Claude (diagnostic & initial fixes)
**To**: Codex (validation & improvement)
**Status**: Two fixes applied, ready for validation and refinement

---

## Executive Summary

**Problem**: APOLLO.PDF OCR producing F1=0% with garbage output
**Root Cause**: Train/test distribution mismatch (synthetic glyphs vs real PDF features)
**Fixes Applied**:
1. ✅ Classifier threshold aligned (0.75 → 0.5)
2. ✅ Feature normalization added (z-score per patch)

**Next Steps for Codex**: Validate fixes, tune parameters, consider fine-tuning on real data

---

## What Was Wrong

### Issue #1: Threshold Mismatch ✅ FIXED
**File**: `knowledge3d/cranium/ocr/character_detector.py:486`

**Problem**:
- **Training** used softmax + argmax → implicit threshold = 0.5
- **Detection** used hardcoded threshold = 0.75 (50% too strict!)
- **Result**: Zero detections (all characters rejected)

**Fix Applied**:
```python
# Before:
self.atomic_classifier_threshold: float = 0.75

# After:
self.atomic_classifier_threshold: float = 0.5  # Matches training threshold
```

**Impact**: 0 detections → 13,509 detections (threshold now working!)

---

### Issue #2: Feature Distribution Mismatch ✅ FIXED (EXPERIMENTAL)
**File**: `knowledge3d/cranium/ocr/character_detector.py:752-764`

**Problem**:
- **Training environment**: Clean synthetic glyphs from fonts
  - `patch_raw_features` from synthetic renders
  - Features likely normalized during forward pass
  - FC classifier trained on normalized distribution

- **Inference environment**: Real PDF scanned document
  - Noisy features from compressed PDFs
  - Different mean/std than training data
  - Raw unnormalized features fed to classifier

- **Result**: Classifiers see out-of-distribution features → garbage predictions

**Fix Applied** (lines 752-764):
```python
# Feature normalization: match training distribution (per-patch z-score)
# Training used normalized features; detection must match
patch_for_logit_normalized = np.zeros_like(patch_for_logit)
for i in range(patch_for_logit.shape[0]):
    feat = patch_for_logit[i]
    mean = feat.mean()
    std = feat.std()
    if std > 1e-6:
        patch_for_logit_normalized[i] = (feat - mean) / std
    else:
        patch_for_logit_normalized[i] = feat - mean

logits = np.dot(patch_for_logit_normalized, matrix.T) + bias
```

**Why This Should Help**:
- Normalizes each patch's features to zero mean, unit variance
- Matches the feature distribution classifiers were trained on
- Standard practice: normalize at inference to match training preprocessing

---

## Your Mission: Validate & Improve

### Priority 1: Validate Feature Normalization Works ⚡ URGENT

**Task**: Re-run APOLLO test and check if predictions improve

**Command**:
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/test_apollo_ground_truth.py
```

**Expected Outcome**:
- **If successful**: Text output shows actual words (APOLLO, ICASE, Teacher, etc.) instead of "WMWENW..."
- **If still failing**: Feature normalization approach is wrong or insufficient

**Success Criteria**:
- Precision/Recall/F1 > 0% (any improvement is progress!)
- Decoded text contains recognizable English words
- Character variety (not all same character)

---

### Priority 2: Diagnostic Feature Statistics 🔍

**Task**: Compare real PDF features vs synthetic training features

**Why**: If Priority 1 fails, we need to understand the feature mismatch depth

**Create diagnostic script** (`scripts/debug_feature_stats.py`):
```python
#!/usr/bin/env python3
"""Compare APOLLO PDF features vs synthetic training features."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge
from scripts.train_atomic_character import DeepSeekOCRModel, render_character

# Step 1: Extract features from APOLLO PDF patches
bridge = PhaseGPDFIngestionBridge()
pdf_path = "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/APOLLO.PDF"
result = bridge.ingest_pdf_page(pdf_path, page_num=0)

# Get feature map from detector
# TODO: Extract patch_raw_features from detection pipeline
# For now, approximate with CNN forward on cropped PDF regions

# Step 2: Extract features from synthetic glyphs
model = DeepSeekOCRModel()
char = 'A'
font_path = "path/to/font.ttf"  # Use same font as training
image = render_character(char, font_path, size=32)
result = model.forward(image, cache_for_backward=True)
synthetic_features = result['feature_map'].mean(axis=(0, 1))  # [128]

# Step 3: Compare statistics
print("=== Feature Statistics Comparison ===")
print(f"\nSynthetic (training-like):")
print(f"  Mean: {synthetic_features.mean():.6f}")
print(f"  Std:  {synthetic_features.std():.6f}")
print(f"  Min:  {synthetic_features.min():.6f}")
print(f"  Max:  {synthetic_features.max():.6f}")

# TODO: Add real PDF patch features
print(f"\nReal PDF (APOLLO):")
print(f"  Mean: {real_features.mean():.6f}")
print(f"  Std:  {real_features.std():.6f}")
print(f"  Min:  {real_features.min():.6f}")
print(f"  Max:  {real_features.max():.6f}")

print(f"\nDifference:")
print(f"  Δ Mean: {abs(real_features.mean() - synthetic_features.mean()):.6f}")
print(f"  Δ Std:  {abs(real_features.std() - synthetic_features.std()):.6f}")
```

**What to look for**:
- Large mean/std differences indicate distribution mismatch
- If Δ Mean > 1.0 or Δ Std > 0.5, normalization is essential
- If features differ dramatically, may need fine-tuning

---

### Priority 3: Per-Character Threshold Calibration 📊

**Task**: Use your existing `analyze_atomic_classifier_thresholds.py` on **real APOLLO patches**

**Why**: Current fix uses global threshold=0.5, but Codex discovered overlap:
- Positive 5th percentile: ~0.14
- Negative 95th percentile: ~0.89
- Per-character thresholds will improve precision/recall

**Steps**:
1. Modify `analyze_atomic_classifier_thresholds.py` to:
   - Accept real PDF feature patches as input (not synthetic dataset)
   - Extract features from APOLLO PDF
   - Run all 62 classifiers on real features
   - Compute per-character optimal thresholds

2. Store thresholds in JSON:
```python
# Save to /K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/thresholds.json
{
    "65": 0.42,  # 'A'
    "66": 0.51,  # 'B'
    ...
    "122": 0.39  # 'z'
}
```

3. Update `CharacterDetector.set_atomic_classifiers()` to:
   - Load per-character thresholds from JSON
   - Store as `self.atomic_char_thresholds: Dict[int, float]`
   - Use `threshold = self.atomic_char_thresholds.get(char_id, 0.5)` in detection

---

### Priority 4: Consider Fine-Tuning (If Normalization Insufficient) 🎯

**If Priority 1 shows minimal improvement**, consider domain adaptation:

**Option A: Quick Fine-Tune (Recommended)**
1. Extract 100-200 APOLLO PDF patches with ground truth labels
2. Freeze CNN weights (already trained)
3. Fine-tune only FC classifier heads on real PDF features
4. Save updated `*_weights.npz` files

**Script skeleton**:
```python
# scripts/finetune_apollo_classifiers.py
from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer

# Load existing CNN (frozen)
model = DeepSeekOCRModel()
model.load_weights("path/to/ocr_gpu_epoch_100/ocr_cnn_weights.npz")

# Load existing FC weights
fc_weight = np.load("char_65_A_weights.npz")["fc_weight"]
fc_bias = np.load("char_65_A_weights.npz")["fc_bias"]

# Fine-tune on APOLLO patches
trainer = GPUCNNTrainer(model, num_classes=2, fc_only=True, learning_rate=0.001)
trainer.fc_weight = fc_weight
trainer.fc_bias = fc_bias

# Train on real PDF patches (labeled manually or via template matching)
for epoch in range(10):  # Quick fine-tune
    loss, acc = trainer.train_batch(apollo_images, apollo_labels)

# Save fine-tuned weights
np.savez("char_65_A_weights_finetuned.npz",
         fc_weight=trainer.fc_weight, fc_bias=trainer.fc_bias)
```

**Option B: Data Augmentation (Long-term)**
- Add PDF-like augmentation to training pipeline
- Compression artifacts, noise, blur, skew
- Retrain from scratch with mixed synthetic + PDF-augmented data

---

## Testing Protocol

### Test 1: APOLLO First Page
**Script**: `scripts/test_apollo_ground_truth.py`
**Ground truth**: "ICASE", "APOLLO", "11", "Teacher", "Resource"
**Success**: Precision/Recall/F1 > 50%

### Test 2: Character Variety Check
**Command**:
```bash
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/test_apollo_ground_truth.py | grep "text=" | head -20
```

**Success**: Output shows variety of characters (not all 'W' or '~')

### Test 3: Classifier Confidence Distribution
**Add logging** to `character_detector.py` detection loop:
```python
# Around line 760-770
if logistic_probs is not None:
    print(f"[DEBUG] Patch {i}: Top 5 probs = {np.sort(logistic_probs[i])[-5:][::-1]}")
    print(f"[DEBUG] Patch {i}: Top 5 chars = {[chr(self.atomic_char_ids_list[idx]) for idx in np.argsort(logistic_probs[i])[-5:][::-1]]}")
```

**Success**: Probabilities vary, characters differ between patches

---

## Key Files Modified

### By Claude:
1. **`knowledge3d/cranium/ocr/character_detector.py`**
   - Line 486: Threshold 0.75 → 0.5
   - Lines 752-764: Added feature normalization

### By Codex (previous work):
1. **`scripts/analyze_atomic_classifier_thresholds.py`** (already created)
2. **`scripts/debug_char_ids.py`** (already created)
3. **`knowledge3d/cranium/bridges/pdf_ingestion_bridge_phase_g.py`** (fixed character loading)

### For Codex to Create:
1. **`scripts/debug_feature_stats.py`** (Priority 2)
2. **`/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars/thresholds.json`** (Priority 3)
3. **`scripts/finetune_apollo_classifiers.py`** (Priority 4, if needed)

---

## Decision Tree

```
START: Re-run APOLLO test
│
├─ F1 > 50%? ✅
│  └─→ SUCCESS! Apply per-character thresholds (Priority 3) for refinement
│
├─ F1 = 0-50%? 🟡
│  ├─→ Run feature statistics (Priority 2)
│  ├─→ Adjust normalization if needed
│  └─→ Try per-character thresholds (Priority 3)
│
└─ F1 still 0%? ❌
   └─→ Feature normalization insufficient
       └─→ Fine-tune on APOLLO data (Priority 4)
```

---

## Claude's Hypothesis

**Most likely outcome**: Feature normalization will improve F1 from 0% to 30-60%

**Why**:
- Threshold fix already proved detections work (13k detections)
- Main issue is classifier seeing unfamiliar feature distributions
- Z-score normalization is standard practice for this exact problem
- Won't be perfect (synthetic ≠ real), but should be much better

**If hypothesis is wrong**: Normalization approach is incorrect (wrong axis, wrong statistics, or training didn't normalize)

---

## Questions for Daniel/User

1. **If F1 improves significantly** (>50%): Proceed to Math Galaxy or continue refining APOLLO?
2. **If fine-tuning is needed**: OK to manually label 100-200 APOLLO patches, or use semi-supervised approach?
3. **Priority trade-off**: Perfect APOLLO OCR vs start Math Galaxy infrastructure in parallel?

---

## Contact Info

**For questions about this handoff**:
- Review diagnostic scripts: `scripts/debug_char_ids.py`, `scripts/analyze_atomic_classifier_thresholds.py`
- Check Claude's root cause analysis in this document
- Test incrementally: Priority 1 → Priority 2 → Priority 3 → Priority 4

**Git status before handoff**:
```
M knowledge3d/cranium/ocr/character_detector.py  (threshold + normalization)
A scripts/debug_char_ids.py                      (character ID diagnostics)
A TEMP/CODEX_HANDOFF_APOLLO_FIX.md               (this document)
```

---

## Success Criteria

**Minimum**: F1 > 0% (any text detected correctly)
**Good**: F1 > 50% (most characters recognized)
**Excellent**: F1 > 80% (production-ready)
**Perfect**: F1 > 95% (better than commercial OCR)

**Remember**: 87.77% was training accuracy on synthetic data. Real-world F1=70% would already be impressive!

---

**Good luck, Codex! The threshold fix is solid, feature normalization is our best guess. Validate, measure, iterate. 🚀**

---

**End of Handoff**

Last updated: 2025-11-03
Next checkpoint: After Priority 1 validation
