# Qwen3-VL OCR Analysis for K3D GPU-Native OCR

**Date**: 2025-10-21
**Purpose**: Understand Qwen3-VL's OCR approach to improve K3D's Phase E OCR implementation
**For**: Grok to expand into concrete implementation plan

---

## Executive Summary

**What we learned**: Qwen3-VL achieves state-of-the-art OCR (32 languages, robust to blur/tilt) using a **trained Vision Transformer** approach, NOT unsupervised glyph matching.

**Why K3D Phase C failed**: We tried **unsupervised HOG feature matching** which is fundamentally limited without training data. 0% accuracy was expected.

**Path forward**: Three options for K3D Phase E OCR:
1. **Pragmatic (Recommended)**: Use Qwen3-VL API/model as OCR backend until Phase F
2. **Hybrid**: Train small OCR model on K3D's learned glyph embeddings
3. **Sovereign (Long-term)**: Implement lightweight Vision Transformer in PTX

---

## 1. Qwen3-VL Architecture Overview

### Core Components

```
PDF/Image Input
    ↓
Smart Resize (maintains aspect ratio, factor-aligned)
    ↓
Vision Transformer (ViT) - Multi-level feature extraction
    ↓
DeepStack - Fuses fine-grained + coarse-grained features
    ↓
Image-Text Alignment (learned embeddings)
    ↓
Language Model Decoder (Qwen3-235B/30B/8B/4B)
    ↓
Text Output (with optional bounding boxes)
```

### Key Technical Details

**1. Smart Resize** ([vision_process.py:56-81](file:///tmp/Qwen3-VL/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L56-L81))
```python
def smart_resize(height, width, factor, min_pixels, max_pixels):
    """
    Resize image to be:
    - Divisible by 'factor' (typically 28 for patch_size=14 × merge_size=2)
    - Within [min_pixels, max_pixels] range
    - Preserve aspect ratio

    Example:
    - min_pixels = 512 × 32 × 32 = 524K pixels
    - max_pixels = 2048 × 32 × 32 = 2M pixels
    - factor = 28 (14 patch size × 2 spatial merge)
    """
    # Maintains aspect ratio while satisfying constraints
    # Returns (resized_height, resized_width)
```

**Why this matters for K3D**:
- We currently use fixed 16×16 resize (loses information!)
- Qwen preserves aspect ratio and uses much higher resolution
- Our 16×16 → 256 pixels; Qwen uses 524K-2M pixels

**2. Vision Transformer (ViT)**
- **Not disclosed** in open source (proprietary model architecture)
- Likely uses **multi-scale patch embeddings** (4×4, 8×8, 16×16 patches)
- **DeepStack** fuses multiple ViT layers for fine-grained details

**3. Image-Text Alignment**
- **Learned** through training on millions of image-text pairs
- Not rule-based or template matching
- Uses **contrastive learning** or similar approach

**4. OCR Capabilities**
- **32 languages** (vs K3D's attempted alphanumeric only)
- **Text spotting**: Outputs text + 2D bounding boxes in normalized coords
- **Robust**: Handles blur, low light, tilt, rare characters
- **Document structure**: Can parse tables, layouts, hierarchies

---

## 2. K3D Phase C OCR Failures - Root Cause Analysis

### What We Tried (4 Iterations)

**Iteration 1: Canny Edges + Cosine Similarity**
- Feature: Canny(50, 150) edge detection → 128-dim (first 128 edge pixels)
- Matching: Cosine similarity against 46K glyphs
- Result: 40-60% accuracy on simple fonts
- **Problem**: Edges-only loses grayscale information

**Iteration 2: Grayscale + Edges Blend**
- Feature: 64-pixel grayscale + 64-pixel edges + 2 moments
- Harvested: 186K glyphs
- Result: 0% accuracy (feature mismatch bug - harvesting vs OCR different)
- **Problem**: Feature extraction not aligned, too simple for discrimination

**Iteration 3: DCT Features**
- Feature: Discrete Cosine Transform (frequency domain)
- Filtered: Alphanumeric only (124K glyphs)
- Result: 0% accuracy, top matches = '7', '9', 'B' repeated
- **Problem**: DCT captures global frequency, loses spatial structure

**Iteration 4: HOG Features**
- Feature: Histogram of Oriented Gradients (9 bins, 4×4 cells)
- Per-font prototypes: 1,999 fonts × 62 characters
- Result: 0% accuracy, still matching wrong characters
- **Problem**: HOG without SVM training is not discriminative enough

### Why All Iterations Failed

**Fundamental Issue**: **Unsupervised visual feature matching**

All 4 approaches used the same flawed assumption:
1. Extract visual features from glyphs (edges/DCT/HOG)
2. Extract same features from scanned characters
3. Match via cosine similarity

**Why this doesn't work**:
- **No learned embeddings**: Features are hand-crafted, not optimized for discrimination
- **No training data**: Never learned what separates 'O' from '0' from 'Q'
- **Cosine similarity too weak**: Needs SVM or neural classifier on top
- **Single-scale features**: Doesn't capture multi-scale patterns

**Comparison to Qwen3-VL**:
- **Qwen**: Trained ViT on millions of labeled images → learned discriminative features
- **K3D**: Hand-crafted features + cosine similarity → fundamentally limited

---

## 3. What K3D Can Leverage from Qwen3-VL

### Immediate Wins (No Training Required)

**1. Smart Resize for Image Preprocessing**

**Current K3D** ([pdf_ingestion_bridge.py:831](file:///mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/cranium/bridges/pdf_ingestion_bridge.py#L831)):
```python
resized = cv2.resize(char_crop, (16, 16))  # Fixed 16×16, loses aspect ratio
```

**Qwen3-VL approach** (adapted for K3D):
```python
def smart_resize_for_ocr(char_img, min_size=28, max_size=112):
    """
    Resize character image while preserving aspect ratio.

    Args:
        char_img: Character bounding box crop
        min_size: Minimum dimension (28 = 2 patches × 14 pixels)
        max_size: Maximum dimension (112 = 8 patches × 14 pixels)

    Returns:
        Resized image with dimensions divisible by 14 (patch size)
    """
    h, w = char_img.shape[:2]

    # Calculate scale to fit within max_size while preserving aspect ratio
    scale = min(max_size / h, max_size / w)

    # Ensure minimum size
    scale = max(scale, min_size / min(h, w))

    # Resize
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Round to nearest multiple of 14 (patch size)
    new_h = round(new_h / 14) * 14
    new_w = round(new_w / 14) * 14

    # Ensure within bounds
    new_h = max(min_size, min(new_h, max_size))
    new_w = max(min_size, min(new_w, max_size))

    return cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
```

**Benefits**:
- Preserves aspect ratio (distinguishes 'l' from 'I' from '1')
- Higher resolution (28×28 to 112×112 vs fixed 16×16)
- Patch-aligned for future ViT integration

**2. Text Spotting Format (Bounding Boxes + Text)**

**Qwen3-VL output format**:
```python
[
    {'bbox_2d': [x1, y1, x2, y2], 'text_content': 'Hello'},  # Normalized 0-999
    {'bbox_2d': [x1, y1, x2, y2], 'text_content': 'World'},
]
```

**K3D can adopt this** for Galaxy positioning:
```python
{
    'text': 'Hello',
    'bbox_2d_norm': [x1/page_width, y1/page_height, x2/page_width, y2/page_height],
    'galaxy_position': [gx, gy, gz],  # 3D embedding space
    'confidence': 0.95,
}
```

**Benefits**:
- Spatial awareness (layout analysis)
- Compatible with Qwen3-VL's format
- Enables visual question answering ("What's in the top-right corner?")

**3. Multi-level Feature Extraction (Concept)**

**Qwen3-VL DeepStack**: Fuses ViT features from multiple layers
- Early layers: Fine-grained edges, textures
- Middle layers: Parts, shapes
- Late layers: Semantic concepts

**K3D adaptation** (without ViT, using traditional CV):
```python
def extract_multi_level_features(char_img):
    """
    Extract features at multiple scales - poor man's DeepStack.
    """
    features = []

    # Level 1: Fine-grained (edges at original resolution)
    edges = cv2.Canny(char_img, 50, 150)
    features.append(edges.flatten()[:42])  # 42 dims

    # Level 2: Medium (HOG at 2× downscale)
    resized = cv2.resize(char_img, (char_img.shape[1]//2, char_img.shape[0]//2))
    hog = cv2.HOGDescriptor((28,28), (14,14), (14,14), (7,7), 9)
    features.append(hog.compute(resized).flatten()[:42])  # 42 dims

    # Level 3: Coarse (grayscale moments)
    moments = cv2.moments(char_img)
    hu_moments = cv2.HuMoments(moments).flatten()
    features.append(np.pad(hu_moments, (0, 35), 'constant')[:42])  # 42 dims

    # Level 4: Spatial (centroid + aspect ratio)
    if moments['m00'] != 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    else:
        cx, cy = 0.5, 0.5

    aspect_ratio = char_img.shape[1] / char_img.shape[0]
    features.append(np.array([cx, cy, aspect_ratio, 0] * 10 + [0, 0]))  # 42 dims

    # Concatenate all levels → 168-dim feature
    # Pad/trim to 128-dim for K3D compatibility
    feature_vec = np.concatenate(features)[:128]

    # L2 normalize
    return feature_vec / (np.linalg.norm(feature_vec) + 1e-8)
```

**Benefits**:
- Captures information at multiple scales
- More robust than single-scale HOG
- Still GPU-friendly (no neural network required)

---

## 4. Three Paths Forward for K3D Phase E OCR

### Option 1: Pragmatic - Use Qwen3-VL as OCR Backend (RECOMMENDED)

**Approach**: Replace K3D's GPU OCR with Qwen3-VL API/model calls

**Implementation**:
```python
class QwenOCRBridge:
    """Use Qwen3-VL for OCR until K3D has trained sovereign OCR."""

    def __init__(self, model_size='7B'):
        # Load Qwen3-VL-7B-Instruct (smallest, fastest)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = AutoModelForImageTextToText.from_pretrained(
            f"Qwen/Qwen3-VL-{model_size}-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(f"Qwen/Qwen3-VL-{model_size}-Instruct")

    def ocr_page(self, pdf_path, page_num):
        """Extract text + bounding boxes from PDF page."""
        # Render page to image
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2× zoom
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Query Qwen3-VL
        prompt = "Spotting all the text in the image with line-level, and output in JSON format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...]."

        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"image": img},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to('cuda')

        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Parse JSON output
        import json
        import re

        # Extract JSON from markdown fencing
        json_match = re.search(r'```json\n(.*?)\n```', output_text, re.DOTALL)
        if json_match:
            text_objects = json.loads(json_match.group(1))
        else:
            text_objects = json.loads(output_text)

        return {
            'text_objects': text_objects,
            'method': 'qwen3-vl',
            'model_size': model_size,
        }
```

**Integration into K3D**:
```python
# In pdf_ingestion_bridge.py

def __init__(self):
    # ... existing code ...

    # Phase E: Use Qwen3-VL for OCR
    from knowledge3d.cranium.ocr.qwen_bridge import QwenOCRBridge
    self.qwen_ocr = QwenOCRBridge(model_size='7B')  # or '4B' for edge devices

def _ocr_fallback_gpu(self, pdf_path, page_num):
    """OCR fallback using Qwen3-VL instead of K3D's glyph matching."""
    result = self.qwen_ocr.ocr_page(pdf_path, page_num)

    # Convert to K3D format
    objects = []
    for obj in result['text_objects']:
        # Denormalize bounding box (Qwen uses 0-999 coords)
        bbox_norm = obj['bbox_2d']
        bbox_pixels = [
            bbox_norm[0] / 999.0,  # x1
            bbox_norm[1] / 999.0,  # y1
            bbox_norm[2] / 999.0,  # x2
            bbox_norm[3] / 999.0,  # y2
        ]

        objects.append({
            'type': 'text',
            'content': obj['text_content'],
            'bbox': bbox_pixels,
            'method': 'qwen3-vl',
        })

    return {
        'objects': objects,
        'text': ' '.join([o['text_content'] for o in result['text_objects']]),
        'method': 'qwen3-vl',
    }
```

**Pros**:
- ✅ **Immediate 80%+ accuracy** (Qwen3-VL is state-of-the-art)
- ✅ **32 languages** support (no training needed)
- ✅ **Robust** to blur, tilt, low light
- ✅ **Bounding boxes** for layout analysis
- ✅ **Proven technology** (production-ready)

**Cons**:
- ❌ **Not fully sovereign** (depends on Qwen3-VL model)
- ❌ **Larger VRAM** (4B model ~8GB, 7B model ~14GB)
- ❌ **Slower** than pure PTX (but still <1s/page)

**Recommendation**: **Use this for Phase E** while building K3D's sovereign OCR in Phase F

---

### Option 2: Hybrid - Train Small OCR Model on K3D Embeddings

**Approach**: Use K3D's 124K learned glyph embeddings as training data for a small neural classifier

**Architecture**:
```
Character image → Smart resize (28×28 to 112×112)
    ↓
Multi-level features (edges + HOG + moments) → 168-dim
    ↓
Small MLP classifier (168 → 256 → 128 → 62 classes)
    ↓
Softmax → Character prediction
```

**Training data**:
- **Positive samples**: K3D's 124K harvested glyphs (1,999 fonts × 62 chars)
- **Augmentation**: Random rotation (±15°), scale (0.8-1.2×), blur, noise
- **Labels**: 62 classes (A-Z, a-z, 0-9)

**MLP Classifier**:
```python
class CharacterClassifier(nn.Module):
    def __init__(self, input_dim=168, num_classes=62):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)
```

**Training**:
```python
# Dataset: 124K glyphs
dataset = K3DGlyphDataset('/K3D/Knowledge3D.local/font_db.pkl')

# Train/val split: 80/20
train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=512)

# Train for 50 epochs (~30 minutes on GPU)
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    for images, labels in train_loader:
        features = extract_multi_level_features_batch(images)  # 168-dim
        logits = model(features)
        loss = criterion(logits, labels)
        # ... backprop
```

**Expected accuracy**: 60-80% (better than 0%, worse than Qwen3-VL)

**Pros**:
- ✅ **Uses K3D's existing work** (124K glyphs harvested)
- ✅ **Small model** (~1MB parameters, minimal VRAM)
- ✅ **Fast inference** (<10ms per character)
- ✅ **Can convert to PTX** (small MLP is PTX-friendly)

**Cons**:
- ❌ **Limited accuracy** (60-80% vs 90%+ for Qwen)
- ❌ **Requires training** (~30 min setup time)
- ❌ **Alphanumeric only** (no multilingual support yet)

**Recommendation**: **Explore this in Phase F** as a stepping stone to full sovereignty

---

### Option 3: Sovereign - Lightweight Vision Transformer in PTX

**Approach**: Implement a minimal ViT in PTX, train on K3D's data + augmentation

**Architecture**:
```
Character image (112×112) → 8×8 patches (14×14 each)
    ↓
Patch embedding (14×14 → 64-dim) via 1×1 conv
    ↓
Add positional encoding (learned 64 params)
    ↓
Transformer encoder (2 layers, 4 heads, 64-dim)
    ↓
Classification head (64 → 62 classes)
```

**Parameter count**: ~50K parameters (tiny ViT)

**PTX Implementation**:
```cuda
// Patch embedding kernel
__global__ void patch_embed(
    float* output,           // [num_patches × embed_dim]
    const float* input,      // [H × W]
    const float* proj_weight // [embed_dim × patch_size × patch_size]
) {
    // Extract 14×14 patch and project to 64-dim
}

// Multi-head attention kernel
__global__ void multi_head_attention(
    float* output,
    const float* Q, const float* K, const float* V,
    int num_heads, int head_dim
) {
    // Scaled dot-product attention in PTX
}
```

**Training**:
- **Data**: K3D's 124K glyphs + heavy augmentation (rotation, scale, blur, occlusion)
- **Epochs**: 200 epochs (~4-6 hours on single GPU)
- **Expected accuracy**: 70-85% (competitive with small pre-trained models)

**Pros**:
- ✅ **Fully sovereign** (pure PTX implementation)
- ✅ **Tiny model** (50K params, <1MB)
- ✅ **GPU-native** (PTX acceleration)
- ✅ **Foundation for multi-modal** (can extend to images, video)

**Cons**:
- ❌ **Significant engineering** (2-4 weeks to implement + train)
- ❌ **Lower accuracy than SOTA** (70-85% vs 90%+)
- ❌ **Requires training infrastructure** (labeled data, augmentation pipeline)

**Recommendation**: **Phase F long-term goal** (after Phase E proves value)

---

## 5. Recommended Implementation Roadmap

### Phase E (Immediate - 1 Week)

**Goal**: Replace failed GPU OCR with working solution

**Tasks**:
1. **Implement QwenOCRBridge** (Option 1)
   - Install Qwen3-VL-7B-Instruct or Qwen3-VL-4B-Instruct
   - Create bridge class for PDF page OCR
   - Integrate into `pdf_ingestion_bridge.py`
   - Test on Apollo PDF (expect 80%+ accuracy)

2. **Update benchmarks**
   - Re-run `test_ocr_apollo.py`
   - Measure accuracy (should be 4/5 or 5/5 keywords)
   - Measure latency (expect <1s/page)

3. **Document dependency**
   - Add to requirements: `transformers>=4.57.0`, `qwen-vl-utils`
   - Note in README: "Phase E uses Qwen3-VL temporarily"

**Deliverable**: Working OCR with 80%+ accuracy on scanned PDFs

---

### Phase F (Future - 1 Month)

**Goal**: Sovereign GPU-native OCR

**Tasks**:
1. **Collect training data**
   - Use K3D's 124K glyphs as seed
   - Add real scanned document samples (PDFs ingested)
   - Augment: rotation, blur, noise, occlusion

2. **Train hybrid model** (Option 2)
   - Multi-level feature extraction (edges + HOG + moments)
   - Small MLP classifier (168 → 256 → 128 → 62)
   - Train for 50 epochs (~30 min)
   - Validate on held-out Apollo pages

3. **Convert to PTX** (if accuracy >60%)
   - Implement MLP forward pass in PTX
   - Benchmark GPU speedup vs CPU
   - Integrate as `sovereign_ocr_bridge.py`

4. **(Optional) Explore mini-ViT** (Option 3)
   - Prototype in PyTorch first
   - If promising (>70% accuracy), convert to PTX
   - Compare to Qwen3-VL and hybrid MLP

**Deliverable**: Sovereign OCR with 60-85% accuracy, fully GPU-native

---

## 6. Dataset Opportunities from Qwen3-VL

### Training Datasets (Mentioned in Paper/Docs)

**For OCR Training**:
1. **SynthText** - Synthetic text in natural images
2. **ICDAR datasets** - Text detection/recognition benchmarks
3. **COCO-Text** - Text in COCO images
4. **TextOCR** - 900K+ text annotations
5. **ReCTS** - Multilingual text dataset

**K3D can leverage**:
- Download these public datasets
- Fine-tune mini-ViT or hybrid model
- Combine with K3D's 124K harvested glyphs

### Qwen's Pre-trained Weights

**We can extract learned features**:
```python
# Load Qwen3-VL model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-7B-Instruct")

# Extract ViT encoder
vision_model = model.visual

# Use as feature extractor for K3D's glyphs
for glyph_img in k3d_glyphs:
    with torch.no_grad():
        features = vision_model(glyph_img)  # Get learned embeddings

    # Use these as training data for K3D's classifier
```

**Benefits**:
- Transfer learning from Qwen's 90B parameter training
- Use their learned visual features
- Train only small classification head (fast)

---

## 7. Key Takeaways for Grok

### What NOT to Do

❌ **Don't try unsupervised glyph matching again**
- We tried 4 times (Canny, Grayscale+Edges, DCT, HOG)
- All failed because no learned embeddings
- Need training data or pre-trained model

❌ **Don't use fixed 16×16 resize**
- Loses aspect ratio (can't distinguish 'l' from 'I')
- Too low resolution (missing fine details)
- Use smart_resize instead (28×28 to 112×112)

❌ **Don't use single-scale features**
- Need multi-level extraction (edges + shapes + semantics)
- Qwen uses DeepStack for this reason

### What TO Do

✅ **Phase E: Use Qwen3-VL as OCR backend**
- Immediate 80%+ accuracy
- Proven, production-ready
- Buys time to build sovereign solution

✅ **Learn from their preprocessing**
- Smart resize (preserve aspect ratio, patch-aligned)
- Text spotting format (text + bounding boxes)
- Multi-level features (when building sovereign OCR)

✅ **Use their datasets**
- SynthText, ICDAR, TextOCR for training
- Or extract features from Qwen's ViT as transfer learning

✅ **Long-term: Build lightweight ViT in PTX**
- Start small (50K params, 2 layers, 4 heads)
- Train on K3D glyphs + public datasets
- Convert to PTX for sovereignty

---

## 8. Concrete Next Steps for Grok

### Step 1: Install Qwen3-VL (15 minutes)

```bash
# In k3d_pdf environment
~/k3d_venvs/k3d_pdf/bin/pip install git+https://github.com/huggingface/transformers
~/k3d_venvs/k3d_pdf/bin/pip install qwen-vl-utils
~/k3d_venvs/k3d_pdf/bin/pip install accelerate

# Test installation
PYTHONPATH=. ~/k3d_venvs/k3d_pdf/bin/python -c "
from transformers import Qwen2_5_VLForConditionalGeneration
print('Qwen3-VL imports successfully')
"
```

### Step 2: Create QwenOCRBridge (30 minutes)

**File**: `knowledge3d/cranium/ocr/qwen_bridge.py`

```python
"""
Qwen3-VL OCR Bridge for K3D Phase E

Temporary solution using Qwen3-VL for scanned PDF OCR until
K3D has trained sovereign OCR in Phase F.
"""

import torch
from PIL import Image
from typing import Dict, List
import fitz  # PyMuPDF
import json
import re
from transformers import AutoModelForImageTextToText, AutoProcessor


class QwenOCRBridge:
    """
    OCR bridge using Qwen3-VL-7B-Instruct.

    Provides text extraction + bounding boxes for scanned PDFs.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-7B-Instruct"):
        """
        Initialize Qwen3-VL model.

        Args:
            model_name: HuggingFace model ID
                - "Qwen/Qwen3-VL-4B-Instruct" (faster, 8GB VRAM)
                - "Qwen/Qwen3-VL-7B-Instruct" (better, 14GB VRAM)
        """
        print(f"[QWEN] Loading {model_name}...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        print(f"[QWEN] Model loaded on {self.model.device}")

    def ocr_pdf_page(
        self,
        pdf_path: str,
        page_num: int,
        prompt: str = "Spotting all the text in the image with line-level, and output in JSON format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...].",
    ) -> Dict:
        """
        Extract text + bounding boxes from PDF page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            prompt: OCR prompt (default: text spotting)

        Returns:
            {
                'text_objects': [{'bbox_2d': [...], 'text_content': '...'}, ...],
                'full_text': 'concatenated text',
                'method': 'qwen3-vl',
                'page': page_num,
            }
        """
        # Render PDF page to image
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2× resolution
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # Prepare messages
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"image": img},
            ]}
        ]

        # Tokenize
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Parse JSON output
        text_objects = self._parse_json_output(output_text)

        return {
            'text_objects': text_objects,
            'full_text': ' '.join([obj.get('text_content', '') for obj in text_objects]),
            'method': 'qwen3-vl',
            'page': page_num,
        }

    def _parse_json_output(self, output_text: str) -> List[Dict]:
        """Parse JSON from Qwen3-VL output (handles markdown fencing)."""
        # Try to extract JSON from markdown fencing
        json_match = re.search(r'```json\n(.*?)\n```', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = output_text

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: return empty list
            print(f"[QWEN] Failed to parse JSON: {output_text[:100]}")
            return []
```

### Step 3: Integrate into PDF Bridge (15 minutes)

**File**: `knowledge3d/cranium/bridges/pdf_ingestion_bridge.py`

```python
# Add to imports
from knowledge3d.cranium.ocr.qwen_bridge import QwenOCRBridge

# Add to __init__
def __init__(self):
    # ... existing code ...

    # Phase E: Qwen3-VL OCR
    try:
        self.qwen_ocr = QwenOCRBridge(model_name="Qwen/Qwen3-VL-7B-Instruct")
        print("[PHASE_E] Qwen3-VL OCR initialized")
    except Exception as e:
        print(f"[PHASE_E] WARNING: Could not load Qwen3-VL - {e}")
        self.qwen_ocr = None

# Replace _ocr_fallback_gpu
def _ocr_fallback_gpu(self, pdf_path: str, page_num: int) -> Dict:
    """
    OCR fallback for scanned PDFs.

    Phase E: Uses Qwen3-VL (80%+ accuracy)
    Phase F: Will use K3D's sovereign OCR
    """
    if self.qwen_ocr is None:
        # Fallback to Tesseract if Qwen not available
        return self._ocr_fallback_tesseract(pdf_path, page_num)

    # Use Qwen3-VL
    result = self.qwen_ocr.ocr_pdf_page(pdf_path, page_num)

    # Convert to K3D format
    objects = []
    for obj in result['text_objects']:
        # Denormalize bbox from Qwen's 0-999 coords
        bbox_norm = obj.get('bbox_2d', [0, 0, 999, 999])
        bbox_01 = [coord / 999.0 for coord in bbox_norm]

        objects.append({
            'type': 'text',
            'content': obj.get('text_content', ''),
            'bbox': bbox_01,
            'method': 'qwen3-vl',
        })

    return {
        'objects': objects,
        'text': result['full_text'],
        'method': 'qwen3-vl',
        'page': page_num,
    }
```

### Step 4: Test on Apollo PDF (10 minutes)

```bash
# Run OCR test
PYTHONPATH=. ~/k3d_venvs/k3d_pdf/bin/python scripts/test_ocr_apollo.py

# Expected output:
# [OCR] Page 0 appears scanned, using Qwen3-VL OCR...
# [QWEN] Processing page...
# ✅ Keywords found: ICASE, APOLLO, 11, Teacher, Resource (5/5)
# ✅ Accuracy: 100%
# ✅ Method: qwen3-vl
# ✅ Runtime: 850ms/page
```

### Step 5: Benchmark Full Pipeline (20 minutes)

```bash
# Re-run full benchmark
PYTHONPATH=. ~/k3d_venvs/k3d_pdf/bin/python scripts/benchmark_pdf_ingestion.py

# Expected results:
# Structured PDF: 22ms/page (unchanged)
# Scanned PDF: ~1s/page (was 3.3s with failed GPU OCR)
# Accuracy: 80-95% (was 0%)
```

---

## 9. Success Metrics for Phase E

Phase E is **COMPLETE** when:
- ✅ Qwen3-VL integrated into `pdf_ingestion_bridge.py`
- ✅ Apollo PDF test: 80%+ accuracy (4/5 or 5/5 keywords)
- ✅ Scanned PDF runtime: <2s/page
- ✅ No regression on structured PDFs (still 22ms/page)
- ✅ Integration tests passing
- ✅ Documented as temporary solution (Phase F = sovereign OCR)

---

## 10. Files to Create/Modify

### New Files

- `knowledge3d/cranium/ocr/__init__.py`
- `knowledge3d/cranium/ocr/qwen_bridge.py` (Qwen3-VL integration)
- `tests/test_qwen_ocr.py` (Unit tests for Qwen bridge)

### Modified Files

- `knowledge3d/cranium/bridges/pdf_ingestion_bridge.py` (integrate Qwen OCR)
- `requirements.txt` (add `transformers>=4.57.0`, `qwen-vl-utils`)
- `README.md` (document Phase E Qwen3-VL dependency)

### Keep for Phase F

- `knowledge3d/cranium/kernels/glyph_resonator.cu` (GPU glyph matching kernel)
- `knowledge3d/cranium/sleep/glyph_consolidator.py` (124K glyphs database)
- `scripts/harvest_fonts_for_ocr.py` (font harvesting for training data)

---

## 11. Timeline Estimate

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| **Phase E** | Install Qwen3-VL | 15 min | pip install |
| | Create QwenOCRBridge | 30 min | - |
| | Integrate into PDF bridge | 15 min | QwenOCRBridge |
| | Test Apollo PDF | 10 min | Integration |
| | Benchmark full pipeline | 20 min | Integration |
| | **Total** | **1.5 hours** | |
| **Phase F** | Collect training data | 2-4 hours | K3D glyphs + public datasets |
| | Train hybrid MLP | 30 min | Training data |
| | Convert to PTX | 1-2 days | Trained model |
| | Validate accuracy | 1-2 hours | PTX implementation |
| | **Total** | **1 week** | |

---

## 12. Long-term Vision: K3D Sovereign OCR

### Phase F Goal (1 Month from Now)

**Architecture**:
```
PDF Page → Smart resize → Multi-level features
    ↓
Lightweight ViT (PTX) or Hybrid MLP (PTX)
    ↓
Character predictions → RPN embeddings → Galaxy positioning
    ↓
Sovereign, GPU-native, multilingual OCR
```

**Training Pipeline**:
1. **Seed data**: K3D's 124K harvested glyphs
2. **Public datasets**: SynthText, ICDAR, TextOCR
3. **Transfer learning**: Extract features from Qwen's ViT
4. **Augmentation**: Rotation, blur, noise, occlusion, font variations

**Target Metrics**:
- Accuracy: 70-85% (comparable to small pre-trained models)
- Speed: <100ms/page (GPU-native PTX)
- VRAM: <50MB (tiny model)
- Languages: Start with English, expand to 10+ languages

**Why This Matters**:
- ✅ Fully sovereign (no external dependencies)
- ✅ GPU-native (PTX acceleration)
- ✅ Foundation for multi-modal AI (extends to images, video)
- ✅ Continuous improvement (learns from Galaxy knowledge)

---

**END OF ANALYSIS**

---

**For Grok**: Use this analysis to create:
1. **Phase E Implementation Plan** (Qwen3-VL integration - 1.5 hours)
2. **Phase F Design Document** (Sovereign OCR - 1 month)
3. **Training Pipeline Specification** (datasets, augmentation, evaluation)

**Key insight**: Don't try to beat Qwen3-VL immediately. Use their model to unblock Phase E, then build K3D's sovereign solution incrementally in Phase F while learning from their architecture and datasets.
----
Daniel:
Grok, me and Claude were deliberating about how to leverage Qwen project OCR habilities adapting what they've done to our project, here's Claude ideation:
(pasted Claude plan)

Grok:
# K3D Phase E OCR Enhancement: Qwen3-VL Integration and Sovereign Path

**Date**: 2025-10-21  
**Purpose**: Build on Claude's Qwen3-VL analysis to deliver actionable plans for Phase E (immediate unblocking) and Phase F (sovereign evolution). Emphasize GPU sovereignty, reuse of existing kernels (e.g., `GlyphResonator`, `ResonanceField`), and swarm collaboration.  
**From**: Grok, expanding Claude's ideation with PTX-centric refinements and FMEAI alignment (energetic memory for glyph embeddings, atomic cognition for feature fusion).  
**For**: Daniel to orchestrate via Codex/Claude repo access; swarm partners to iterate.  

---

## Executive Summary
**Claude's Core Insight**: Qwen3-VL's trained ViT + DeepStack crushes unsupervised matching—leverage it pragmatically for Phase E while bootstrapping sovereign PTX OCR for Phase F.  
**Grok's Enhancements**:  
- **Phase E**: Refine Qwen bridge for minimal VRAM (use 4B variant), integrate with existing `GalaxyResonanceEngine` for post-OCR embedding fusion. Add latency guards via `LatencyGuard` kernel. Total: 1.5 hours.  
- **Phase F**: Design lightweight PTX ViT using modular RPN extensions; fuse with `VectorResonator` for attention. Start with hybrid MLP as bridge. Target: 70%+ accuracy, <50ms/page.  
- **Training Pipeline**: Public datasets + K3D glyphs; atomic fission for augmentation; evaluate via `TemporalReasoning` for drift detection.  
- **Swarm Tie-in**: Align with FMEAI—treat OCR as resonance fields in Galaxy (energetic memory). Reuse `ResonanceField` for feature sampling; `AtomicFissionFusion` for glyph splitting/augmentation. No CPU fallbacks; PTX-strict.  
**Risks Mitigated**: Dependency on Qwen documented as temporary; sovereign path ensures no regressions.  

---

## 1. Phase E Implementation Plan (Qwen3-VL Integration - 1.5 Hours)
**Goal**: Unblock OCR with 80%+ accuracy on scanned PDFs (e.g., Apollo test: 4/5+ keywords). Use Qwen3-VL as backend, but fuse outputs into Galaxy via sovereign bridges for energetic memory persistence. Keep latency <2s/page with `LatencyGuard`.  
**Rationale**: Builds directly on Claude's code; optimizes for K3D env (CUDA 12.4, bfloat16). Adds post-processing with `GalaxyResonanceEngine` to embed recognized text into vector graph—maintains FMEAI atomic cognition.  
**Dependencies**: `transformers>=4.57.0`, `qwen-vl-utils`, `accelerate`. Install in `k3d-cranium` env (no RAPIDS needed).  
**Timeline Breakdown**:  

| Step | Task | Duration | Owner | Notes |
|------|------|----------|-------|-------|
| 1 | Install Dependencies | 15 min | Codex/Claude | Run in `k3d-cranium` env; test import. |
| 2 | Implement QwenOCRBridge | 30 min | Codex | Refine Claude's code; add VRAM optimizations. |
| 3 | Integrate into PDF Ingestion Bridge | 15 min | Codex | Replace fallback; add Galaxy fusion. |
| 4 | Test on Apollo PDF | 10 min | All Swarm | Run `test_ocr_apollo.py`; verify keywords. |
| 5 | Benchmark and Document | 20 min | Claude/Grok | Full pipeline; update `docs/ocr_phase_e.md`. |
| **Total** | | **1.5 hours** | | |

### Step 1: Install Dependencies
```bash
# Activate env
source scripts/k3d_env.sh run cranium
# Install
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils accelerate
# Test
python -c "from transformers import AutoModelForImageTextToText, AutoProcessor; print('Installed successfully')"
```
**If fails**: Fallback to Qwen3-VL-4B (lower VRAM); document in `SESSION_HANDOFF.md`.

### Step 2: Implement QwenOCRBridge
**File**: `knowledge3d/cranium/ocr/qwen_bridge.py` (new)  
Refinements: Use 4B model by default (8GB VRAM); add optional bounding box normalization; wrap with `LatencyGuard` for <1s inference.  
```python
"""
Qwen3-VL OCR Bridge for K3D Phase E
Temporary backend for scanned PDF OCR. Outputs fused into Galaxy for sovereign persistence.
"""
import torch
from PIL import Image
from typing import Dict, List
import fitz  # PyMuPDF
import json
import re
from transformers import AutoModelForImageTextToText, AutoProcessor
from knowledge3d.cranium.bridges.sovereign_bridges import LatencyGuard  # Reuse kernel

class QwenOCRBridge:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-4B-Instruct"):  # Default to smaller model
        print(f"[QWEN] Loading {model_name}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.latency_guard = LatencyGuard()  # Sovereign latency measurement
        print(f"[QWEN] Model loaded on {self.model.device}")

    def ocr_pdf_page(
        self,
        pdf_path: str,
        page_num: int,
        prompt: str = "Spot all text in the image at line-level, output JSON: [{'bbox_2d': [x1,y1,x2,y2], 'text_content': 'text'}, ...]. Use normalized 0-999 coords."
    ) -> Dict:
        self.latency_guard.start()  # Measure start
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"image": img}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        text_objects = self._parse_json_output(output_text)
        full_text = ' '.join([obj.get('text_content', '') for obj in text_objects])
        
        elapsed_ns = self.latency_guard.end()  # Measure end
        print(f"[QWEN] OCR latency: {elapsed_ns / 1e6:.2f} ms")

        return {
            'text_objects': text_objects,
            'full_text': full_text,
            'method': 'qwen3-vl',
            'page': page_num,
            'latency_ms': elapsed_ns / 1e6
        }

    def _parse_json_output(self, output_text: str) -> List[Dict]:
        json_match = re.search(r'```json\n(.*?)\n```', output_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else output_text
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"[QWEN] Parse failed: {output_text[:100]}")
            return []
```
**Enhancements**: Latency guarding; default to 4B for edge compatibility.

### Step 3: Integrate into PDF Ingestion Bridge
**File**: `knowledge3d/cranium/bridges/pdf_ingestion_bridge.py` (modify)  
Add to `__init__`: Load `QwenOCRBridge` with try/except (fallback to Tesseract if VRAM low).  
Replace `_ocr_fallback_gpu`:  
```python
from knowledge3d.cranium.ocr.qwen_bridge import QwenOCRBridge
from knowledge3d.cranium.bridges.sovereign_bridges import GalaxyResonanceEngine  # For post-fusion

# In __init__
try:
    self.qwen_ocr = QwenOCRBridge()
    print("[PHASE_E] Qwen3-VL ready")
except Exception as e:
    print(f"[PHASE_E] Qwen load failed: {e}. Using Tesseract fallback.")
    self.qwen_ocr = None

def _ocr_fallback_gpu(self, pdf_path: str, page_num: int) -> Dict:
    if self.qwen_ocr is None:
        return self._ocr_fallback_tesseract(pdf_path, page_num)  # Existing fallback
    result = self.qwen_ocr.ocr_pdf_page(pdf_path, page_num)
    objects = []
    for obj in result['text_objects']:
        bbox_norm = obj.get('bbox_2d', [0, 0, 999, 999])
        bbox_01 = [coord / 999.0 for coord in bbox_norm]
        objects.append({
            'type': 'text',
            'content': obj.get('text_content', ''),
            'bbox': bbox_01,
            'method': 'qwen3-vl'
        })
    
    # Fuse into Galaxy (sovereign post-processing)
    gre = GalaxyResonanceEngine()
    for obj in objects:
        vector = gre.resonate_query(obj['content'])  # Embed text into Galaxy resonance field
        obj['galaxy_position'] = vector.tolist()[:3]  # 3D projection for House
    
    return {
        'objects': objects,
        'text': result['full_text'],
        'method': 'qwen3-vl',
        'page': page_num
    }
```
**Enhancements**: Galaxy fusion for FMEAI alignment; preserves sovereign hot path.

### Step 4: Test on Apollo PDF
```bash
PYTHONPATH=. python scripts/test_ocr_apollo.py
```
**Expected**: 100% keywords (ICASE, APOLLO, 11, Teacher, Resource); latency <1s.

### Step 5: Benchmark and Document
Run `benchmark_pdf_ingestion.py`; expect scanned: ~1s/page, structured: 22ms unchanged.  
Create `docs/ocr_phase_e.md`: Detail dependency, rationale, migration to Phase F.

**Success Metrics**: Apollo 80%+; no regressions; documented.

---

## 2. Phase F Design Document (Sovereign OCR - 1 Month)
**Goal**: Fully GPU-native OCR with 70%+ accuracy, <100ms/page, <50MB VRAM. Start hybrid (MLP on multi-level features), evolve to mini-ViT in PTX. Multilingual foundation (English first).  
**Rationale**: Build on Claude's options 2/3; reuse kernels (`VectorResonator` for attention, `ResonanceField` for embeddings, `AtomicFissionFusion` for augmentation). Align with kernel catalogue—extend `ModularRPNEngine` for ViT ops. No Qwen dependency post-Phase F.  
**Architecture Overview**:  
```
PDF Page → Smart Resize (Python wrapper) → Multi-Level Features (PTX: edges/HOG/moments)
    ↓
Hybrid MLP or Mini-ViT (PTX kernels: embed, attention, classify)
    ↓
Predictions → Resonance Fusion (GalaxyResonanceEngine) → House Crystallization
```
**Key Components**:  
- **Preprocessing**: Claude's `smart_resize_for_ocr` + PTX port for batching.  
- **Feature Extraction**: Multi-level (Claude's function) in PTX via new `MultiLevelFeatureExtractor` kernel (extend `gre_resonance_field.cu`).  
- **Classifier**:  
  - **Hybrid MLP**: 168→256→128→62; PTX via `ModularRPNEngine` (stack ops for linear/relu).  
  - **Mini-ViT**: 112×112 input → 8×8 patches → 64-dim embed → 2-layer transformer (4 heads) → 62 classes. PTX kernels: `patch_embed.ptx`, `multi_head_attention.ptx` (new, extend `gre_vector_resonator.cu`).  
- **Post-Processing**: Fuse predictions with `GraphCrystallizer` for House persistence.  
**PTX Extensions**:  
- New source: `kernels/ocr_vit.cu` → `ptx/ocr_vit.ptx`.  
- Bridge: `bridges/sovereign_ocr_bridge.py` (alloc, launch, fuse).  
**Pros/Cons**: Sovereign (pros: deterministic, fast); engineering effort (cons: mitigated by modular reuse).  

**Milestones**:  
| Week | Milestone | Deliverables | Dependencies |
|------|-----------|--------------|--------------|
| 1 | Data Prep | Augmented dataset (500K samples); scripts in `Large_Assets_Kitchen/`. | K3D glyphs + public datasets. |
| 2 | Hybrid MLP | Trained model (PyTorch proto); PTX conversion; 60%+ accuracy. | Training pipeline. |
| 3 | Mini-ViT Proto | PyTorch ViT (50K params); train to 70%+. | Hybrid success. |
| 4 | PTX Integration + Test | Full sovereign bridge; benchmarks; integration tests. | ViT proto. |

**Risks**: Accuracy <70% → fallback to hybrid; use `OOMSpillManager` for large batches.  
**Documentation**: `docs/ocr_sovereign.md`; update kernel catalogue.  

---

## 3. Training Pipeline Specification (Datasets, Augmentation, Evaluation)
**Goal**: Generate 500K+ labeled samples; train hybrid/ML to 70%+; evaluate with sovereign metrics (drift, resonance).  
**Datasets**:  
- **Seed**: K3D's 124K glyphs (`Knowledge3D.local/font_db.pkl`).  
- **Public**: SynthText (synthetic scenes), ICDAR (benchmarks), TextOCR (900K annotations), ReCTS (multilingual). Download via scripts; store in `Large_Assets_Kitchen/ocr_datasets/`.  
- **Qwen Transfer (Optional)**: Extract ViT features from Qwen3-VL (Claude's code); use as pseudo-labels for 10K samples.  
- **Total**: 500K after augmentation; split 80/10/10 (train/val/test).  

**Augmentation (Atomic Cognition-Aligned)**:  
Use `AtomicFissionFusion` kernel for vector-based splits/fusions; Python wrapper for image ops.  
```python
from knowledge3d.cranium.bridges.sovereign_bridges import AtomicFissionFusion
import cv2
import numpy as np

def augment_glyph(img, label):
    aff = AtomicFissionFusion()
    # Image ops
    img = cv2.rotate(img, np.random.choice([0, 1, 2]))  # 0/90/180/270
    img = cv2.GaussianBlur(img, (3,3), sigmaX=np.random.uniform(0.5, 1.5))
    scale = np.random.uniform(0.8, 1.2)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    noise = np.random.normal(0, 0.05, img.shape)
    img = np.clip(img + noise * 255, 0, 255).astype(np.uint8)
    
    # Sovereign fusion: Split/fuse vector rep
    vec = extract_multi_level_features(img)  # 168-dim
    split_vecs = aff.fission(vec, num_splits=2)  # Atomic split
    fused_vec = aff.fusion(split_vecs)  # Recombine with variation
    return img, fused_vec, label  # Return augmented image + vector
```
**Batch Size**: 512; apply 4x per sample → 500K from 124K seed.  

**Training Loop (PyTorch Proto, then PTX)**:  
```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Dataset: Custom GlyphDataset with augmentation
dataset = GlyphDataset('Knowledge3D.local/ocr_datasets/', augment=True)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

# Model (Hybrid MLP example)
model = nn.Sequential(
    nn.Linear(168, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 62)
).cuda()

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):  # ~30 min
    for imgs, labels in train_loader:
        features = extract_multi_level_features_batch(imgs.cuda())
        preds = model(features)
        loss = criterion(preds, labels.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
**For ViT**: Similar, but 200 epochs (~4 hours); use `TemporalReasoning` for timestamp-keyed dropout.  

**Evaluation**:  
- **Metrics**: Accuracy (top-1/5), F1-score per class; resonance drift (via `TemporalReasoning` on val set).  
- **Benchmarks**: Apollo PDF (80%+ keywords); ICDAR test set.  
- **Sovereign Check**: Run inferences via PTX bridge; guard with `MultimodalHaltingGate` for confidence >0.7.  
- **Scripts**: `scripts/train_ocr_phase_f.py`; log to `Knowledge3D.local/ocr_models/`.  

**Swarm Iteration**: Post-training, consult GLM/Kimi for accuracy tweaks; Codex commits PTX.

**END OF PLAN**  
**Next**: Daniel, orchestrate Codex for Phase E commits. Swarm, vibe on ViT extensions?
----
Daniel:
Nice Qwen, we're evaluating how to leverage Qwen approach to OCR adapting to our project, The pasted text's what has been discussed.

Qwen:
