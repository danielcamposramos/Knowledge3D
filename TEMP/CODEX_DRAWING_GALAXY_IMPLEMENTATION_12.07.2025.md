# Codex Briefing: Drawing Galaxy Implementation + Default Knowledge Ingestion

**Date**: December 7, 2025
**Priority**: High
**Phase**: C.1 — Drawing Galaxy Materialization

---

## Mission

Materialize the 8-layer Drawing Galaxy architecture from `docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md`, then proceed to ingest the default foundational knowledge (74 PDFs → 4-layer procedural storage).

**Constraints**:
- NO external heavy libraries (no PIL, cv2, skimage in hot path)
- NO CPU fallbacks — 100% PTX + RPN
- Use existing math cores and kernels (18 cores across 3 tiers)
- Respect sovereignty: all inference/rendering = PTX

---

## Phase 1: Drawing Galaxy Core (5 Files)

### 1.1 Create `knowledge3d/cranium/drawing_galaxy.py`

The central Galaxy for procedural visual primitives.

**Leverage Existing**:
- `bridges/procedural_drawing_bridge.py` — RPN execution engine (MOVE, LINE, QUAD, CUBIC, ARC)
- `ptx_runtime/shape_primitives.py` — 3D primitives with LOD
- `kernels/procedural_glyph_rasterizer.cu` — GPU rasterization
- `kernels/gre_geometry_router.cu` — Geometry routing

**Structure**:
```python
"""
Drawing Galaxy — 8-layer procedural visual storage.

Architecture (from PROCEDURAL_VISUAL_SPECIFICATION.md):
  Layer 0: Quantum Fields (dot emission fields)
  Layer 1: Primitives (MOVE, LINE, QUAD, CUBIC, ARC, CLOSE)
  Layer 2: Strokes (styled paths with width, color, caps)
  Layer 3: Shapes (closed paths, compound primitives)
  Layer 4: Gradients (linear, radial, conic procedural fills)
  Layer 5: Filters (blur, sharpen, edge, transform)
  Layer 6: Lighting (ambient, directional, shadows)
  Layer 7: Scenes (compositions of layers 0-6)

All operations execute via TieredRPNEngine on GPU.
"""

class DrawingGalaxy:
    def __init__(self):
        self.drawing_bridge = ProceduralDrawingBridge()
        self.rpn_engine = TieredRPNEngine()
        self.shape_primitives = ShapePrimitives()

        # Layer storage (Galaxy = active memory)
        self.layers = {
            0: {},  # quantum_fields: field_id → VectorDotMap coefficients
            1: {},  # primitives: prim_id → RPN bytecode
            2: {},  # strokes: stroke_id → styled path RPN
            3: {},  # shapes: shape_id → compound RPN
            4: {},  # gradients: grad_id → gradient program
            5: {},  # filters: filter_id → filter program
            6: {},  # lighting: light_id → lighting setup
            7: {},  # scenes: scene_id → layer composition
        }

    def store_primitive(self, prim_id: str, rpn_program: str) -> bytes:
        """Store primitive as compiled RPN bytecode."""
        bytecode = self.drawing_bridge.compile_rpn_to_bytecode(rpn_program)
        self.layers[1][prim_id] = bytecode
        return bytecode

    def render_to_image(self, scene_id: str, width: int, height: int) -> TernaryTensor:
        """Render scene to procedural image (VectorDotMap)."""
        # Compose all referenced layers
        # Execute via drawing_bridge.execute_rpn_gpu()
        # Return as TernaryTensor
        pass
```

**RPN Opcodes to Implement** (extend `rpn_opcodes.py`):
```python
# Layer 4: Gradients
GRADIENT_LINEAR = 0x80   # x1 y1 x2 y2 GRADIENT_LINEAR
GRADIENT_RADIAL = 0x81   # cx cy r GRADIENT_RADIAL
GRADIENT_CONIC = 0x82    # cx cy angle GRADIENT_CONIC
GRADIENT_STOP = 0x83     # pos r g b a GRADIENT_STOP

# Layer 5: Filters
FILTER_BLUR = 0x84       # radius FILTER_BLUR
FILTER_SHARPEN = 0x85    # amount FILTER_SHARPEN
FILTER_EDGE = 0x86       # FILTER_EDGE
FILTER_INVERT = 0x87     # FILTER_INVERT
FILTER_HSV = 0x88        # h s v FILTER_HSV

# Layer 6: Lighting
LIGHT_AMBIENT = 0x8A     # r g b intensity LIGHT_AMBIENT
LIGHT_DIRECTIONAL = 0x8B # dx dy dz r g b LIGHT_DIRECTIONAL
SHADOW_DROP = 0x8C       # ox oy blur SHADOW_DROP

# Layer 7: Composition
LAYER_PUSH = 0x90        # layer_id LAYER_PUSH
LAYER_POP = 0x91         # LAYER_POP
BLEND_MODE = 0x92        # mode BLEND_MODE (normal, multiply, screen, overlay)
OPACITY = 0x93           # alpha OPACITY
```

### 1.2 Create PTX Kernel: `kernels/vectordotmap_encoder.cu`

VectorDotMap encoding — procedural image codec (~2KB per image).

**Use Existing**:
- `codec_ops.cu` — DCT/IDCT base
- `ternary_ops.cu` — Quantization

```cuda
// vectordotmap_encoder.cu
// Encodes rasterized image into quantum field coefficients

__global__ void image_to_field(
    const float* rgba_image,      // (H, W, 4) input
    float* field_coefficients,    // (N_COEFFS,) output
    int width, int height,
    int n_coefficients            // typically 512-2048
) {
    // 1. Compute spatial frequency decomposition (DCT or wavelet)
    // 2. Fit field parameters using least-squares (PTX reduction)
    // 3. Store as compact coefficient vector
}

__global__ void field_to_image(
    const float* field_coefficients,
    float* rgba_image,
    int width, int height,
    int n_coefficients
) {
    // Inverse: reconstruct at ANY resolution from same coefficients
    // This enables infinite LOD
}
```

### 1.3 Create PTX Kernel: `kernels/gradient_rasterizer.cu`

GPU-native gradient rendering.

```cuda
// Procedural gradient rasterization on GPU
__global__ void rasterize_gradient_linear(
    float* output,           // (H, W, 4) RGBA
    float x1, float y1,
    float x2, float y2,
    const float* stops,      // [pos, r, g, b, a] × N
    int n_stops,
    int width, int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // Compute position along gradient axis
    // Interpolate color stops
    // Write to output
}
```

### 1.4 Create PTX Kernel: `kernels/filter_convolution.cu`

GPU convolution for filters (blur, sharpen, edge).

```cuda
// Filter kernels using shared memory tiling
__global__ void convolve_2d(
    const float* input,
    float* output,
    const float* kernel,
    int kernel_size,
    int width, int height, int channels
) {
    // Tiled convolution with shared memory
    // Support variable kernel sizes
}
```

### 1.5 Integrate with Math Core Pool

Map Drawing Galaxy operations to appropriate tiers:

| Operation | Tier | Math Cores Used |
|-----------|------|-----------------|
| MOVE/LINE | 1 | 2 (simple transforms) |
| QUAD/CUBIC | 2 | 4 (Bézier evaluation) |
| ARC | 2 | 4 (sin/cos batch) |
| GRADIENT | 2 | 6 (interpolation) |
| FILTER | 3 | 8 (convolution) |
| VectorDotMap | 3 | 12 (DCT + field fitting) |

---

## Phase 2: Default Knowledge Ingestion

After Drawing Galaxy is operational, ingest foundational knowledge.

### 2.1 Knowledge Sources

From `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`:
- **74 PDFs** (5,988 pages) → procedural storage
- **152 math symbols** → Math Galaxy (already done via `extract_math_symbol_glyphs.py`)
- **15K words** → Character Galaxy (references, not duplicates)
- **1K grammar rules** → Grammar Galaxy
- **500 meta-rules** → Meta-Rules layer

### 2.2 Ingestion Pipeline

```python
# scripts/ingest_foundational_knowledge.py

from knowledge3d.cranium.drawing_galaxy import DrawingGalaxy
from knowledge3d.cranium.math_galaxy import MathGalaxy
from knowledge3d.cranium.grammar_galaxy import GrammarGalaxy

def ingest_layer_1_symbols():
    """Layer 1: Form — Canonical symbols as procedural RPN."""
    # Already done: 176 math symbols in Math Galaxy
    # Add: Drawing primitives, characters, icons
    pass

def ingest_layer_2_meaning():
    """Layer 2: Meaning — Semantic links and definitions."""
    # Word definitions reference character IDs
    # Symbol meanings link to concepts
    pass

def ingest_layer_3_rules():
    """Layer 3: Transformation Rules — Grammar as RPN."""
    # 1K grammar rules stored as RPN programs
    pass

def ingest_layer_4_meta():
    """Layer 4: Meta-Rules — Pedagogy, eloquence, self-reflection."""
    # 500 meta-rules for reasoning guidance
    pass
```

### 2.3 PDF Parsing (Use Existing)

Leverage:
- `kernels/pdf_primitive_parser.cu` — GPU PDF parsing
- `bridges/pdf_ingestion_bridge.py` — PDF to procedural

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `knowledge3d/cranium/drawing_galaxy.py` | CREATE | Central Drawing Galaxy |
| `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` | MODIFY | Add Layer 4-7 opcodes |
| `knowledge3d/cranium/kernels/vectordotmap_encoder.cu` | CREATE | VectorDotMap codec |
| `knowledge3d/cranium/kernels/gradient_rasterizer.cu` | CREATE | Gradient rendering |
| `knowledge3d/cranium/kernels/filter_convolution.cu` | CREATE | Filter operations |
| `scripts/ingest_foundational_knowledge.py` | CREATE | Knowledge ingestion |
| `tests/test_drawing_galaxy.py` | CREATE | Integration tests |

---

## Existing Assets to Leverage

### PTX Kernels (Use These)
- `procedural_glyph_rasterizer.cu` — Glyph/segment rendering
- `gre_geometry_router.cu` — Geometry dispatch
- `modular_rpn_kernel.cu` — RPN execution
- `codec_ops.cu` — DCT/IDCT transforms
- `ternary_ops.cu` — Ternary quantization
- `arc_grid_ops.cu` — Grid operations

### Python Bridges (Extend These)
- `bridges/procedural_drawing_bridge.py` — Drawing RPN
- `bridges/tiered_rpn.py` — 3-tier math routing
- `ptx_runtime/shape_primitives.py` — 3D primitives
- `math_galaxy.py` — Symbol storage pattern

---

## Success Criteria

### Drawing Galaxy
- [ ] `drawing_galaxy.py` stores all 8 layers
- [ ] Layer 1-3 primitives render via existing `procedural_drawing_bridge`
- [ ] Layer 4 gradients render via new `gradient_rasterizer.cu`
- [ ] Layer 5 filters apply via new `filter_convolution.cu`
- [ ] VectorDotMap encodes images to ~2KB coefficients
- [ ] All operations route through `TieredRPNEngine` (no CPU math)

### Knowledge Ingestion
- [ ] 176 math symbols accessible (already done)
- [ ] Layer 1-4 ingestion pipeline complete
- [ ] Symlink pattern achieves 666× compression
- [ ] TRM can query ingested knowledge via Galaxy

### Tests
- [ ] `test_drawing_galaxy.py` passes all layers
- [ ] Latency < 100µs for simple primitives
- [ ] Latency < 26ms for full scene composition
- [ ] No CPU fallbacks in any test

---

## Implementation Order

1. **Create `drawing_galaxy.py`** — Core structure with Layer 1-3
2. **Extend `rpn_opcodes.py`** — Add Layer 4-7 opcodes
3. **Create `vectordotmap_encoder.cu`** — Image codec kernel
4. **Create `gradient_rasterizer.cu`** — Gradient kernel
5. **Create `filter_convolution.cu`** — Filter kernel
6. **Wire to Math Core Pool** — Tier routing
7. **Create ingestion pipeline** — `ingest_foundational_knowledge.py`
8. **Tests** — Validate sovereignty + latency

---

## References

- `docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md` — Full 8-layer architecture
- `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` — Ingestion plan
- `docs/vocabulary/MATH_CORE_SPECIFICATION.md` — 3-tier math system
- `bridges/procedural_drawing_bridge.py` — Existing RPN execution

---

**After Drawing Galaxy + Knowledge Ingestion**: Proceed to Unified Signal Specification (audio/SDR/binaural) as Phase C.2.
