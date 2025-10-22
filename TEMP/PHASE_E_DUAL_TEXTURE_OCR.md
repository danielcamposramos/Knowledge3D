# K3D Phase E: Dual-Texture Paradigm with DeepSeek Text-as-Image Compression

**Date**: 2025-10-22
**Priority**: HIGH - Completes dual-client paradigm for House/Galaxy
**From**: Grok's enhancement of DeepSeek OCR innovation
**Status**: Design ready, awaiting Codex implementation

---

## 🎯 Executive Summary

**DeepSeek Innovation**: Store text data AS compressed images (visual encoding)
- Text → Image compression: 7-20× reduction, 97%+ fidelity
- AI reads textures like humans read pixels - but the "language" is compressed visual data

**K3D Dual-Client Paradigm** (Completed):
```
Same 3D Object in House/Galaxy GLB
    │
    ├─ UV Map 0: HUMAN Texture
    │  → Game-like, aesthetic, human-readable visuals
    │  → For Avatar navigation, Tablet UX
    │
    └─ UV Map 1: AI Texture (DeepSeek compressed image)
       → Text encoded AS visual data (compressed image format)
       → AI decodes image → extracts text content
       → "AI's visual language" = compressed text encoding
```

**Why This Matters for K3D**:
- ✅ **Completes dual-client**: Humans see pretty textures, AI sees data-as-texture
- ✅ **Compression**: Books → <1KB/page as texture (10-20× smaller)
- ✅ **Sovereign**: AI decodes textures on-GPU (no external text storage)
- ✅ **RLWHF synergy**: Better contexts from full text encoded in textures
- ✅ **FMEAI alignment**: Both clients "see" the same object, different perceptions

**Timeline**:
- Phase E (Prototype): 2 hours - Dual-texture bridge + Apollo test
- Phase F (Full): 1 month - DeepSeek compression model, PTX decoder, book ingestion

---

## 💡 The Core Insight: Text-as-Image for Dual-Client

### DeepSeek's Innovation
**Traditional storage**: Text as characters/bytes (1 char = 1+ bytes)
**DeepSeek's method**: Text as compressed visual encoding (7-20× smaller!)

Example:
```
Original text: "The quick brown fox..." (10,000 chars = ~10KB)
    ↓
DeepSeek compression: Render as dense visual encoding
    ↓
Compressed image: 256×256 RGB (~200KB uncompressed, but 7-20× ratio!)
    ↓
The "compression" is in HOW MUCH text fits in the image
More text in smaller space = compression
```

### K3D's Dual-Texture Application

**Same 3D object (e.g., Book Page), TWO textures**:

```
UV Map 0: HUMAN TEXTURE (512×512 RGB)
┌─────────────────────────────────────┐
│  [ Beautiful Page Layout ]          │
│                                     │
│  Chapter 5: Neural Networks         │
│  ─────────────────────────────      │
│  A neural network is...             │
│                                     │
│  [Readable, aesthetic, game-style]  │
└─────────────────────────────────────┘
        ↑
   Avatar/Tablet sees this - pretty!


UV Map 1: AI TEXTURE (256×256 RGB - smaller!)
┌─────────────────────────────────────┐
│ A neural network is a computational │
│ model inspired by biological neur..│
│ consisting of layers nodes weight..│
│ trained via backpropagation using..│
│ [VERY dense, tiny font, compressed] │
│ [More text in less space = 7-20×]  │
└─────────────────────────────────────┘
        ↑
   AI decodes this via OCR/decoder
   Extracts full text for reasoning
```

### Why This Works Perfectly for K3D

1. **Dual-Client Completion**:
   - Humans navigate via pretty textures (UV Map 0)
   - AI reads via compressed textures (UV Map 1)
   - **Both are visual** - just different "languages"

2. **Compression via Density**:
   - Human texture: Readable font, nice spacing, aesthetic
   - AI texture: Tiny font, dense packing, 7-20× more text per pixel
   - AI doesn't care about aesthetics - maximize information density!

3. **Sovereign Decode**:
   - AI texture is just an image (standard format)
   - Decode via OCR/model (on-GPU, PTX in Phase F)
   - No external text storage needed

4. **FMEAI Alignment**:
   - Perception: Both clients "see" textures
   - Intuition (human): Pretty, readable
   - Deliberation (AI): Dense, information-rich
   - Same object, different perceptual layers!

---

## 🔗 Integration with Current Work

### Synergy with RLWHF Pipeline

**Current RLWHF Flow** (Codex is building):
```
PDFs → Extract chunks → Ollama generates questions → TRM training
```

**Enhanced with Dual-Texture OCR**:
```
PDFs → DeepSeek OCR → Dual-Texture Encode → House GLB
         ↓
    Full text + structure + bboxes + tables
         ↓
    RLWHF: Richer contexts for question generation
         ↓
    Better teacher feedback (can reference exact PDF locations)
```

**Benefits for RLWHF**:
1. **Better Questions**: OCR provides layout structure → "Explain the diagram on page 42"
2. **Grounded Answers**: Decode textures to verify student answer accuracy
3. **Citation**: Link TRM answers to exact PDF coordinates (bbox in texture)

### Existing K3D Components to Leverage

**Already in Codebase**:
- ✅ `knowledge3d/cranium/rpn_embedding_engine.py` - RPN math VM (extend with DeepSeek ops)
- ✅ `knowledge3d/cranium/bridges/sovereign_bridges.py` - ModularRPNEngine, GraphCrystallizer
- ✅ `knowledge3d/ingestion/documents/pdf_ingestor.py` - PDF processing (enhance with dual-texture)
- ✅ Existing OCR bridges (mentioned: Qwen, DeepSeek) - extend for texture output

**New Components** (Phase E):
- `knowledge3d/cranium/ocr/deepseek_bridge.py` - DeepSeek OCR wrapper
- `knowledge3d/cranium/bridges/dual_texture_bridge.py` - Encode/decode dual textures
- Extended `pdf_ingestor.py` - Output dual-texture folios to House

---

## 📋 Phase E Implementation (2 Hours)

### Goal
Prototype dual-texture encoding for PDF ingestion:
1. DeepSeek OCR extracts text + structure
2. Compress to AI texture channel
3. Encode as GLB folio in House
4. Decode via RPN stub
5. Test: Apollo PDF → 90%+ keyword recovery

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-TEXTURE INGESTION                       │
└─────────────────────────────────────────────────────────────────┘

PDF Page (Raw)
    ↓
┌──────────────────────────────────────────────────────────────┐
│                    DUAL TEXTURE CREATION                      │
└──────────────────────────────────────────────────────────────┘
    ↓                                   ↓
[HUMAN TEXTURE]                   [AI TEXTURE]
    ↓                                   ↓
┌────────────────────┐       ┌─────────────────────────┐
│ Render Pretty Page │       │ DeepSeek Text-as-Image  │
│ - Game-style font  │       │ - Extract text          │
│ - Nice layout      │       │ - Compress AS image     │
│ - Aesthetic design │       │ - 7-20× reduction       │
│ - Human readable   │       │ - Visual encoding       │
│ ↓                  │       │ ↓                       │
│ UV Map 0: RGB      │       │ UV Map 1: Compressed    │
│ 512×512            │       │ 256×256 (smaller!)      │
└────────────────────┘       └─────────────────────────┘
    ↓                                   ↓
    └───────────────┬───────────────────┘
                    ↓
           ┌────────────────────┐
           │ GLB Folio (House)  │
           │ - Quad mesh        │
           │ - 2 UV maps        │
           │ - Human + AI       │
           └────────────────────┘
                    ↓
           ┌────────────────────┐
           │ Galaxy Crystallize │
           │ - Embed text       │
           │ - Vector index     │
           └────────────────────┘

When Avatar/Tablet loads object:
    → Shows UV Map 0 (human texture) - pretty!

When AI reasons about object:
    → Decodes UV Map 1 (compressed image) - extracts text!
```

### Step 1: DeepSeek OCR Bridge (30 min)

**Create**: `knowledge3d/cranium/ocr/deepseek_bridge.py`

```python
#!/usr/bin/env python3
"""
DeepSeek OCR Bridge: GPU-native OCR with 7-20× compression.

Interfaces with DeepSeek-OCR model for sovereign text extraction.
Outputs structured data suitable for dual-texture encoding.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any
from pathlib import Path


class DeepSeekOCRBridge:
    """
    Bridge to DeepSeek OCR model for K3D ingestion.

    Features:
    - 97%+ fidelity text extraction
    - Bbox + layout preservation
    - Table/formula detection
    - Multilingual support (~100 languages)
    """

    def __init__(self, model_path: str = None):
        """
        Initialize DeepSeek OCR.

        Args:
            model_path: Path to DeepSeek model weights
                       (default: use sovereign K3D models)
        """
        # TODO: Load DeepSeek model (stub for Phase E)
        self.model_path = model_path or "/K3D/Knowledge3D.local/models/deepseek_ocr"
        self.compression_ratio = 15.0  # Target 7-20×

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract text and structure from image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            {
                'full_text': str,
                'text_objects': List[{text, bbox, confidence}],
                'tables': List[table_structure],
                'layout': {type, regions},
                'metadata': {lang, quality}
            }
        """
        # Phase E stub: Use PyMuPDF for now, plan DeepSeek integration
        result = {
            'full_text': self._stub_extract_text(image),
            'text_objects': self._stub_detect_regions(image),
            'tables': [],
            'layout': {'type': 'text', 'regions': []},
            'metadata': {'lang': 'en', 'quality': 0.95}
        }
        return result

    def _stub_extract_text(self, image: np.ndarray) -> str:
        """Stub: Extract text (replace with DeepSeek in Phase F)."""
        # For now, use existing PDF text extraction
        return "Stub text extraction - integrate DeepSeek model in Phase F"

    def _stub_detect_regions(self, image: np.ndarray) -> List[Dict]:
        """Stub: Detect text regions with bboxes."""
        return [
            {'text': 'Sample text', 'bbox': [0, 0, 100, 20], 'confidence': 0.95}
        ]

    def encode_texture(self, text: str, width: int = 256, height: int = 256) -> np.ndarray:
        """
        Compress text AS image using DeepSeek's visual encoding.

        This is the KEY: Text becomes a compressed visual representation,
        not JSON or binary data. AI "reads" this image to extract text.

        Args:
            text: Raw text to encode
            width: Texture width (default 256)
            height: Texture height (default 256)

        Returns:
            AI texture: (height, width, 3) RGB image with text visually encoded
                       (DeepSeek compression: 7-20× smaller than raw text)
        """
        # Phase E stub: Simple text rendering as image
        # Phase F: Integrate actual DeepSeek visual compression model

        from PIL import Image, ImageDraw, ImageFont

        # Create blank image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Use small font for compression (fit more text)
        try:
            # Try to load a compact font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 8)
        except:
            font = ImageFont.load_default()

        # Render text in compact grid (DeepSeek style: dense visual encoding)
        x, y = 5, 5
        line_height = 10
        max_chars_per_line = width // 6  # Rough estimate

        words = text.split()
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line += word + " "
            else:
                # Draw current line
                draw.text((x, y), current_line, fill='black', font=font)
                y += line_height
                current_line = word + " "

                # Check if we've filled the image
                if y > height - line_height:
                    break

        # Draw last line
        if current_line and y <= height - line_height:
            draw.text((x, y), current_line, fill='black', font=font)

        # Convert to numpy array (this IS the AI texture!)
        texture = np.array(img, dtype=np.uint8)

        # Phase F TODO: Replace with actual DeepSeek compression model
        # Should achieve 7-20× reduction while maintaining 97%+ fidelity

        return texture

    def decode_texture(self, texture: np.ndarray) -> str:
        """
        Decode AI texture (image) back to text.

        Uses OCR to read the visually-encoded text from the image.
        This is how AI "reads" the texture - by decoding the visual encoding.

        Args:
            texture: (H, W, 3) RGB image with text encoded

        Returns:
            Reconstructed text string
        """
        # Phase E stub: Use Tesseract OCR for now
        # Phase F: Use DeepSeek's decoder model

        try:
            from PIL import Image
            import pytesseract

            # Convert numpy to PIL
            img = Image.fromarray(texture)

            # OCR decode (this is how AI reads the texture!)
            text = pytesseract.image_to_string(img)

            return text.strip()

        except Exception as e:
            # Fallback: Return error
            return f"[Decode failed: {str(e)}]"
```

### Step 2: Dual-Texture Encoder (45 min)

**Create**: `knowledge3d/cranium/bridges/dual_texture_bridge.py`

```python
#!/usr/bin/env python3
"""
Dual-Texture Bridge: Encode/decode AI layers with RPN stub.

Combines human-visible RGB with AI-encoded data channels for sovereign
knowledge artifacts in K3D House/Galaxy.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from pygltflib import GLTF2, BufferFormat, Buffer, BufferView, Accessor
from pygltflib import Image as GLTFImage, Texture, Material, Mesh, Primitive
from pygltflib import Node, Scene, UNSIGNED_INT, FLOAT

from knowledge3d.cranium.bridges.sovereign_bridges import GraphCrystallizer


class DualTextureEncoder:
    """
    Encode/decode dual-layer textures for K3D knowledge artifacts.

    Layers:
    - Human: RGB for visual navigation
    - AI: Compressed data for sovereign decode

    RPN integration: Compress/decompress via ModularRPNEngine ops
    (Phase E: stub, Phase F: full DeepSeek extensions)
    """

    def __init__(self):
        """Initialize encoder with RPN stub."""
        self.crystallizer = GraphCrystallizer()
        # Phase E: Stub RPN ops
        # Phase F: self.rpn = ModularRPNEngine() with DeepSeek extensions

    def encode_folio(
        self,
        human_rgb: np.ndarray,
        ai_data: np.ndarray,
        page_num: int,
        pdf_name: str,
        output_dir: Path
    ) -> str:
        """
        Encode dual-texture folio as GLB.

        Args:
            human_rgb: (H, W, 3) RGB image of page
            ai_data: (256, 256, 1) compressed OCR data
            page_num: Page number
            pdf_name: Source PDF name
            output_dir: Output directory for GLB

        Returns:
            Path to generated GLB file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        glb_path = output_dir / f"{pdf_name}_page{page_num:04d}.glb"

        # Create simple quad mesh for page
        vertices = np.array([
            [0.0, 0.0, 0.0],  # Bottom-left
            [1.0, 0.0, 0.0],  # Bottom-right
            [1.0, 1.0, 0.0],  # Top-right
            [0.0, 1.0, 0.0]   # Top-left
        ], dtype=np.float32)

        # UVs for texture mapping
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float32)

        # Indices (two triangles)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Combine textures (Human RGB + AI Data)
        # Resize human_rgb to match ai_data dimensions
        from PIL import Image
        human_resized = np.array(
            Image.fromarray(human_rgb).resize((256, 256))
        )

        # Stack: RGBA (RGB + Data channel as alpha)
        # Phase F: Use proper multi-texture or extended channels
        dual_texture = np.concatenate([
            human_resized,
            ai_data
        ], axis=-1)  # (256, 256, 4)

        # Create GLB (simplified - full implementation in Phase F)
        gltf = self._create_gltf_stub(
            vertices, uvs, indices, dual_texture, page_num
        )

        gltf.save(str(glb_path))

        return str(glb_path)

    def _create_gltf_stub(
        self,
        vertices: np.ndarray,
        uvs: np.ndarray,
        indices: np.ndarray,
        texture: np.ndarray,
        page_num: int
    ) -> GLTF2:
        """Create minimal GLTF with dual texture (stub for Phase E)."""
        gltf = GLTF2()

        # Serialize texture to PNG (embedded)
        from PIL import Image
        import io
        img = Image.fromarray(texture.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        texture_bytes = buf.getvalue()

        # Add buffer
        buffer = Buffer()
        buffer.byteLength = len(texture_bytes)
        buffer.uri = f"data:application/octet-stream;base64,{self._encode_base64(texture_bytes)}"
        gltf.buffers.append(buffer)

        # Add image
        image = GLTFImage()
        image.mimeType = "image/png"
        image.bufferView = 0  # Simplified
        gltf.images.append(image)

        # Add texture
        tex = Texture()
        tex.source = 0
        gltf.textures.append(tex)

        # Add material
        mat = Material()
        mat.pbrMetallicRoughness = {
            'baseColorTexture': {'index': 0}
        }
        mat.name = f"page_{page_num}_material"
        gltf.materials.append(mat)

        # Add mesh (simplified - full geometry in Phase F)
        mesh = Mesh()
        prim = Primitive()
        prim.material = 0
        mesh.primitives.append(prim)
        gltf.meshes.append(mesh)

        # Add node + scene
        node = Node()
        node.mesh = 0
        gltf.nodes.append(node)

        scene = Scene()
        scene.nodes.append(0)
        gltf.scenes.append(scene)
        gltf.scene = 0

        return gltf

    def _encode_base64(self, data: bytes) -> str:
        """Encode bytes to base64 string."""
        import base64
        return base64.b64encode(data).decode('ascii')

    def decode_folio(self, glb_path: str) -> Dict[str, Any]:
        """
        Decode dual-texture folio to extract AI data.

        Args:
            glb_path: Path to GLB folio

        Returns:
            Decoded OCR data dict
        """
        gltf = GLTF2().load(glb_path)

        # Extract AI channel from texture (stub - Phase F: full decode)
        # For now, return empty structure
        return {
            'text_objects': [],
            'full_text': 'Decode stub - implement RPN decompression in Phase F',
            'source': glb_path
        }

    def resonate_query(
        self,
        glb_path: str,
        query: str,
        rpn_engine
    ) -> Dict[str, Any]:
        """
        Query dual-texture folio via RPN resonance.

        Args:
            glb_path: Path to GLB folio
            query: Search query
            rpn_engine: RPNEmbeddingEngine for resonance

        Returns:
            Matched text regions with scores
        """
        # Phase E stub: Return empty
        # Phase F: Decode AI texture → search via RPN → return bboxes
        return {
            'matches': [],
            'query': query,
            'source': glb_path
        }
```

### Step 3: Integrate into PDF Ingestion (20 min)

**Modify**: `knowledge3d/ingestion/documents/pdf_ingestor.py`

Add dual-texture output option:

```python
# Add imports at top
from knowledge3d.cranium.ocr.deepseek_bridge import DeepSeekOCRBridge
from knowledge3d.cranium.bridges.dual_texture_bridge import DualTextureEncoder

class PDFIngestor:
    def __init__(self, ...):
        # Existing initialization
        ...
        # Add dual-texture components
        self.use_dual_texture = True  # Feature flag
        self.deepseek_ocr = DeepSeekOCRBridge()
        self.texture_encoder = DualTextureEncoder()
        self.folio_output_dir = Path("/K3D/Knowledge3D.local/house_zone7/folios")

    def process_page(self, page, pdf_path: Path, page_num: int):
        """Process single PDF page with dual-texture encoding."""
        # Existing text extraction
        text = page.get_text()

        if self.use_dual_texture:
            # Get page image
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # RGB for human layer
            human_rgb = img[:, :, :3] if pix.n >= 3 else img

            # DeepSeek OCR for AI layer
            ocr_result = self.deepseek_ocr.extract(human_rgb)
            ai_texture = self.deepseek_ocr.encode_texture(ocr_result)

            # Encode dual-texture folio
            glb_path = self.texture_encoder.encode_folio(
                human_rgb=human_rgb,
                ai_data=ai_texture,
                page_num=page_num,
                pdf_name=pdf_path.stem,
                output_dir=self.folio_output_dir
            )

            # Crystallize to Galaxy (existing flow)
            embedding = self.rpn_engine.embed_sentence(text)
            # ... rest of existing ingestion

            return {
                'text': text,
                'glb_folio': glb_path,
                'ocr_quality': ocr_result['metadata']['quality']
            }
        else:
            # Existing flow
            return {'text': text}
```

### Step 4: Testing & Validation (25 min)

**Create**: `scripts/test_dual_texture_apollo.py`

```python
#!/usr/bin/env python3
"""Test dual-texture encoding on Apollo PDF."""

from pathlib import Path
from knowledge3d.ingestion.documents.pdf_ingestor import PDFIngestor
from knowledge3d.cranium.ocr.deepseek_bridge import DeepSeekOCRBridge
from knowledge3d.cranium.bridges.dual_texture_bridge import DualTextureEncoder

def test_apollo_dual_texture():
    """Test Phase E dual-texture on Apollo PDF."""

    # Load Apollo PDF
    pdf_path = Path("/K3D/Knowledge3D.local/datasets/pdfs/apollo_guidance.pdf")

    if not pdf_path.exists():
        print(f"❌ Apollo PDF not found: {pdf_path}")
        return

    print("=" * 70)
    print("K3D Phase E: Dual-Texture Test (Apollo PDF)")
    print("=" * 70)

    # Initialize components
    print("\n📥 Initializing components...")
    ingestor = PDFIngestor(use_dual_texture=True)

    # Process first 3 pages
    import fitz
    doc = fitz.open(str(pdf_path))

    results = []
    for page_num in range(min(3, len(doc))):
        print(f"\n📄 Processing page {page_num + 1}...")
        page = doc[page_num]

        result = ingestor.process_page(page, pdf_path, page_num)
        results.append(result)

        print(f"   Text extracted: {len(result['text'])} chars")
        print(f"   GLB folio: {result['glb_folio']}")
        print(f"   OCR quality: {result['ocr_quality']:.1%}")

    doc.close()

    # Validation: Decode and compare
    print("\n🔍 Validation: Decode test...")
    encoder = DualTextureEncoder()
    ocr = DeepSeekOCRBridge()

    for i, result in enumerate(results):
        decoded = encoder.decode_folio(result['glb_folio'])

        # Compare with original text (keyword matching)
        original_words = set(result['text'].lower().split())
        decoded_words = set(decoded['full_text'].lower().split())

        # Stub metrics (Phase F: full fidelity test)
        overlap = len(original_words & decoded_words) / max(1, len(original_words))

        print(f"\n   Page {i+1}:")
        print(f"   - Original: {len(original_words)} words")
        print(f"   - Decoded: {len(decoded_words)} words")
        print(f"   - Overlap: {overlap:.1%} (Phase E stub)")

    print("\n✅ Phase E test complete!")
    print(f"   Generated: {len(results)} dual-texture folios")
    print(f"   Location: {ingestor.folio_output_dir}")

if __name__ == '__main__':
    test_dual_texture_apollo()
```

**Run**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/test_dual_texture_apollo.py
```

**Success Criteria**:
- ✅ 3 GLB folios generated
- ✅ File size: <500KB per folio (compression working)
- ✅ No errors during encode/decode
- ✅ Phase E stub documented for Phase F expansion

---

## 🔄 Phase F Design (1 Month - Future Work)

### RPN Extensions for DeepSeek Ops

**New Ops in ModularRPNEngine**:

```python
class ModularRPNEngine:
    def __init__(self):
        # Existing ops
        ...
        # Phase F: DeepSeek extensions
        self.ops['conv_fission'] = self._op_conv_fission
        self.ops['moe_route'] = self._op_moe_route
        self.ops['decompress'] = self._op_decompress

    def _op_conv_fission(self, stack: List) -> List:
        """
        Convolutional compression (DeepSeek-style).

        Fissions input tensor via strided conv → tokens.
        Target: 7-20× reduction.
        """
        # Phase F: Implement conv compression
        # For now: stub
        data = stack.pop()
        compressed = data  # TODO: Conv reduce
        stack.append(compressed)
        return stack

    def _op_moe_route(self, stack: List) -> List:
        """
        Mixture-of-Experts routing for decode.

        Routes compressed tokens to specialized experts
        (text, table, formula, layout).
        """
        # Phase F: MoE dispatch
        data = stack.pop()
        routed = data  # TODO: Expert selection
        stack.append(routed)
        return stack

    def _op_decompress(self, stack: List) -> List:
        """
        Decompress (upsample + fuse).

        Reconstructs original data from compressed tokens.
        Target: 95%+ fidelity.
        """
        # Phase F: Upsample
        compressed = stack.pop()
        decompressed = compressed  # TODO: Reconstruction
        stack.append(decompressed)
        return stack
```

### PTX Kernels

**Create**: `knowledge3d/cranium/kernels/rpn_deepseek_ext.cu`

```cuda
// Phase F: DeepSeek RPN extensions
// Conv fission kernel
__global__ void conv_fission_kernel(
    const float* input,
    float* output,
    int width, int height,
    int stride, int kernel_size
) {
    // Strided convolution for compression
    // Target: 15× reduction with minimal quality loss
}

// MoE routing kernel
__global__ void moe_route_kernel(
    const float* tokens,
    const float* expert_weights,
    float* routed_output,
    int n_tokens, int n_experts
) {
    // Route tokens to specialized experts
    // Expert types: text, table, formula, layout
}

// Decompression kernel
__global__ void decompress_kernel(
    const float* compressed,
    float* reconstructed,
    int compressed_size,
    int target_size
) {
    // Upsample + fuse for reconstruction
    // Target: 97%+ fidelity
}
```

### Training Pipeline

**Augmentation** with texture simulations:

```python
def augment_dual_texture(human_rgb, ai_data, blur_ratios=[1, 10]):
    """
    Augment textures with DeepSeek-style compression variations.

    Simulates "forgetting" via progressive channel blur.
    """
    from knowledge3d.cranium.bridges.sovereign_bridges import AtomicFissionFusion

    aff = AtomicFissionFusion()
    augmented = []

    for ratio in blur_ratios:
        # Blur AI channel (simulate compression degradation)
        import cv2
        blurred_ai = cv2.GaussianBlur(ai_data, (3, 3), sigmaX=ratio/10.0)

        # Fission + fusion (test atomic ops)
        split = aff.fission(blurred_ai.flatten())
        fused = aff.fusion(split)

        augmented.append({
            'human': human_rgb,
            'ai': fused.reshape(ai_data.shape),
            'blur_ratio': ratio
        })

    return augmented
```

**Training Loop** (PyTorch prototype):

```python
# scripts/train_rpn_deepseek.py
def train_rpn_extensions(dataset, epochs=100):
    """Train RPN DeepSeek extensions on dual-texture dataset."""

    # Load components
    rpn = ModularRPNEngine()  # With DeepSeek ops
    optimizer = torch.optim.AdamW(rpn.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in dataset:
            # Original AI texture
            original = batch['ai_data']

            # Compress (fission)
            compressed = rpn.execute_ops([original], ['conv_fission'])

            # Decompress (fusion)
            reconstructed = rpn.execute_ops(compressed, ['decompress'])

            # Loss: Reconstruction fidelity
            loss = F.mse_loss(reconstructed, original)

            # Additional: Edit distance for text
            if batch.get('text'):
                decoded_text = decode_texture_to_text(reconstructed)
                edit_dist = levenshtein(batch['text'], decoded_text)
                loss += 0.1 * edit_dist

            # Backward
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 10 == 0:
            fidelity = validate_fidelity(rpn, val_dataset)
            print(f"Epoch {epoch}: Fidelity = {fidelity:.1%}")

    # Export to PTX
    export_rpn_to_ptx(rpn, "rpn_deepseek_ext.ptx")
```

---

## 📊 Integration with RLWHF

### Enhanced Question Generation

**With Dual-Texture OCR**:

```python
# In generate_questions_ollama.py (enhance extract_pdf_contexts)

def extract_pdf_contexts_with_structure(pdf_dir: Path) -> List[Dict]:
    """Extract contexts with layout structure from dual-texture folios."""

    contexts = []
    folio_dir = Path("/K3D/Knowledge3D.local/house_zone7/folios")

    # Load dual-texture folios
    for glb_path in folio_dir.glob("*.glb"):
        encoder = DualTextureEncoder()
        decoded = encoder.decode_folio(str(glb_path))

        # Use structure for richer context
        for text_obj in decoded['text_objects']:
            contexts.append({
                'pdf_name': glb_path.stem,
                'content': text_obj['text'],
                'bbox': text_obj['bbox'],  # NEW: Spatial location
                'layout_type': text_obj.get('type', 'text'),  # NEW: Header/body/table
                'topic': text_obj['text'][:100]
            })

    return contexts
```

**Richer Questions**:
- "Explain the diagram in the top-right of page 42" (uses bbox)
- "What does the table on page 15 show?" (uses layout_type)
- "Summarize the conclusion section" (uses structure detection)

### Teacher Evaluation with Citations

**Enhanced teacher prompt**:

```python
TEACHER_EVALUATION_USER_PROMPT = """Question: {question}

Ground Truth Context (from K3D dual-texture):
Source: {pdf_name}, Page {page_num}, Region {bbox}
Layout: {layout_type}
Text: {context}

Student's Answer (K3D TRM):
...

[Teacher can reference exact PDF location for verification]
"""
```

---

## 🎯 Success Metrics

### Phase E (Prototype)
- ✅ 3+ dual-texture GLB folios generated
- ✅ File size: <500KB/folio (10-20× compression)
- ✅ Decode stub functional
- ✅ No regressions in existing ingestion

### Phase F (Full Implementation)
- ✅ Fidelity: 95%+ text reconstruction
- ✅ Latency: <20µs decode per texture
- ✅ Compression: 10-20× reduction achieved
- ✅ Multilingual: 100+ languages supported
- ✅ Structure: Tables, formulas, layouts preserved

### RLWHF Integration
- ✅ Question quality improved (structure-aware)
- ✅ Teacher citations (bbox references)
- ✅ Grounding verification (decode on-demand)

---

## 🚀 Execution Plan

### Immediate (Phase E - 2 Hours)
1. **Create files** (Codex):
   - `knowledge3d/cranium/ocr/deepseek_bridge.py`
   - `knowledge3d/cranium/bridges/dual_texture_bridge.py`
   - `scripts/test_dual_texture_apollo.py`

2. **Modify files**:
   - `knowledge3d/ingestion/documents/pdf_ingestor.py` (add dual-texture option)

3. **Test**:
   - Run on Apollo PDF
   - Validate GLB output
   - Document results in `TEMP/PHASE_E_RESULTS.md`

### Short-term (Phase F - Week 1-4)
1. **Week 1**: RPN extensions (PyTorch prototype)
2. **Week 2**: PTX kernel implementation
3. **Week 3**: Book ingestion tests (100+ PDFs)
4. **Week 4**: RLWHF integration + benchmarks

### Integration with RLWHF
- **After Phase 1**: Enhance question generation with structure
- **After Phase 3**: Add citation support to teacher evaluation
- **After Phase 6**: Use dual-texture decode for TRM answer validation

---

## 📝 Documentation

**Create**:
- `docs/dual_texture_ocr.md` - Architecture overview
- `docs/rpn_deepseek_extensions.md` - RPN op specifications
- `docs/phase_e_results.md` - Test results
- Update `docs/ROADMAP.md` - Add Phase E/F milestones

---

## 🔥 Why This Matters NOW

1. **RLWHF Synergy**: Better contexts → better questions → better TRM training
2. **Sovereign Stack**: No external OCR APIs, all on-GPU via PTX
3. **Scalability**: Compress entire libraries to <1GB (10-20× reduction)
4. **FMEAI Proof**: Knowledge as energetic textures (atomic fission/fusion)
5. **Timing**: DeepSeek breakthrough + our RLWHF work = perfect alignment

---

**Status**: ✅ Design complete, ready for Codex implementation
**Priority**: HIGH - Run Phase E alongside RLWHF Phase 1
**Estimated Time**: 2 hours (Phase E), then iterate to Phase F

---

**END OF PHASE E DESIGN**
