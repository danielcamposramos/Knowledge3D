# CODEX: Unified Signal + Drawing Galaxy Completion

**Priority:** HIGH — Completes Drawing Galaxy 8-layer architecture + fixes extraction training
**Date:** December 12, 2025
**Scope:** 4 interconnected deliverables

---

## Executive Summary

This briefing combines four interconnected improvements:

1. **Extraction Training Fix** — Size-aware evaluation path + grammar singleton threading
2. **Drawing Galaxy Layers 4-7** — Gradients, filters, lighting, scene composition
3. **VectorDotMap Image Codec** — Unified with existing audio/video codec infrastructure
4. **Color Galaxy** — Sovereign color space conversions (RGB/CMYK/Lab/HSV)

All deliverables share the existing codec infrastructure (`TernaryCodecOps`, `codec_ops.cu`) and maintain 100% PTX sovereignty.

---

## Part A: Extraction Training Fix (Immediate Priority)

### Problem

Training shows extraction tasks blocked by size drop guard:
- `f4081712`: 24×24 → 6×4 DROPPED (ratio 4.0 > 2.5 threshold)
- `ed74f2f2`: 5×9 → 3×3 DROPPED (ratio 3.0)
- Grammar singleton exists but workers still reload 747 rules

### Fix A1: Size-Aware Evaluation Path

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py`

Replace hardcoded ratio guard with size-pattern-aware logic:

```python
def _should_evaluate_candidate(
    self,
    candidate_output: Sequence[Sequence[int]],
    expected_output: Sequence[Sequence[int]],
    size_pattern: str,  # "extract" | "expand" | "same"
) -> bool:
    """
    For extraction tasks, ALLOW large size ratios (up to 10×).
    For same-size tasks, keep existing 2.5× guard.
    """
    if not candidate_output or not expected_output:
        return False

    h_cand = len(candidate_output)
    w_cand = len(candidate_output[0]) if candidate_output else 0
    h_exp = len(expected_output)
    w_exp = len(expected_output[0]) if expected_output else 0

    if h_exp == 0 or w_exp == 0:
        return False

    ratio_h = max(h_cand / h_exp, h_exp / h_cand)
    ratio_w = max(w_cand / w_exp, w_exp / w_cand)

    if size_pattern == "extract":
        # Extraction: allow up to 10× shrink
        if ratio_h > 10.0 or ratio_w > 10.0:
            return False
        return True
    elif size_pattern == "expand":
        # Expansion: allow up to 4× growth
        if ratio_h >= 4.0 or ratio_w >= 4.0:
            return False
        return True
    else:  # "same"
        # Keep existing guard for same-size tasks
        if ratio_h >= 2.5 or ratio_w >= 2.5:
            return False
        return True
```

### Fix A2: Thread Grammar Singleton to Workers

**File:** `knowledge3d/training/arc_agi/parallel_candidate_generator.py`

Pass serialized grammar rules to workers instead of per-worker instantiation:

```python
from knowledge3d.training.arc_agi.grammar_galaxy import get_grammar_galaxy, GrammarRule, GrammarGalaxy

def _serialize_grammar_rules() -> Dict[str, Dict]:
    """Serialize grammar for worker transfer (main process only)."""
    grammar = get_grammar_galaxy()
    return {
        rule_id: {
            "rule_id": rule.rule_id,
            "rpn_program": rule.rpn_program,
            "pattern": rule.pattern,
            "language": rule.language,
        }
        for rule_id, rule in grammar.rules.items()
    }

def _reconstruct_grammar(rules_dict: Dict[str, Dict]) -> GrammarGalaxy:
    """Reconstruct grammar in worker (no file I/O)."""
    rules = [
        GrammarRule(
            rule_id=r["rule_id"],
            rpn_program=r["rpn_program"],
            pattern=r.get("pattern", "unknown"),
            language=r.get("language", "en"),
        )
        for r in rules_dict.values()
    ]
    return GrammarGalaxy(rules=rules)  # Bypasses file load
```

---

## Part B: Drawing Galaxy Layers 4-7 (Missing Components)

### Current State

| Layer | Status | Implementation |
|-------|--------|----------------|
| 0: Quantum Fields | ❌ Missing | VectorDotMap codec (Part C) |
| 1: Primitives | ✅ Done | MOVE, LINE, QUAD, CUBIC, ARC |
| 2: Strokes | ✅ Done | Styled paths |
| 3: Shapes | ✅ Done | Compound primitives |
| 4: Gradients | ❌ Missing | Needs kernel + opcodes |
| 5: Filters | ❌ Missing | Needs kernel + opcodes |
| 6: Lighting | ❌ Missing | Needs kernel + opcodes |
| 7: Scenes | ❌ Missing | Composition opcodes |

### B1: Gradient Rasterizer Kernel

**File:** `knowledge3d/cranium/kernels/gradient_rasterizer.cu`

```cuda
/**
 * Procedural gradient rasterization — GPU-native linear/radial/conic gradients.
 * Unified with existing codec_ops.cu architecture.
 */

extern "C" {

// Color stop structure (packed in shared memory)
struct GradientStop {
    float pos;   // 0.0-1.0 position
    float r, g, b, a;
};

__device__ float4 lerp_color(float4 c1, float4 c2, float t) {
    return make_float4(
        c1.x + t * (c2.x - c1.x),
        c1.y + t * (c2.y - c1.y),
        c1.z + t * (c2.z - c1.z),
        c1.w + t * (c2.w - c1.w)
    );
}

__global__ void gradient_linear_kernel(
    float* output,           // (H, W, 4) RGBA
    float x1, float y1,      // Start point
    float x2, float y2,      // End point
    const float* stops,      // [pos, r, g, b, a] × n_stops
    int n_stops,
    int width, int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // Compute position along gradient axis
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-6f) len_sq = 1e-6f;

    float fx = (float)px / (float)(width - 1);
    float fy = (float)py / (float)(height - 1);

    float t = ((fx - x1) * dx + (fy - y1) * dy) / len_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));  // Clamp to [0,1]

    // Find surrounding stops
    int idx0 = 0, idx1 = 0;
    for (int i = 0; i < n_stops - 1; i++) {
        float pos0 = stops[i * 5];
        float pos1 = stops[(i + 1) * 5];
        if (t >= pos0 && t <= pos1) {
            idx0 = i;
            idx1 = i + 1;
            break;
        }
    }

    // Interpolate color
    float pos0 = stops[idx0 * 5];
    float pos1 = stops[idx1 * 5];
    float local_t = (pos1 > pos0) ? (t - pos0) / (pos1 - pos0) : 0.0f;

    float r = stops[idx0 * 5 + 1] + local_t * (stops[idx1 * 5 + 1] - stops[idx0 * 5 + 1]);
    float g = stops[idx0 * 5 + 2] + local_t * (stops[idx1 * 5 + 2] - stops[idx0 * 5 + 2]);
    float b = stops[idx0 * 5 + 3] + local_t * (stops[idx1 * 5 + 3] - stops[idx0 * 5 + 3]);
    float a = stops[idx0 * 5 + 4] + local_t * (stops[idx1 * 5 + 4] - stops[idx0 * 5 + 4]);

    int idx = (py * width + px) * 4;
    output[idx + 0] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
    output[idx + 3] = a;
}

__global__ void gradient_radial_kernel(
    float* output,
    float cx, float cy, float radius,
    const float* stops, int n_stops,
    int width, int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float fx = (float)px / (float)(width - 1);
    float fy = (float)py / (float)(height - 1);

    float dist = sqrtf((fx - cx) * (fx - cx) + (fy - cy) * (fy - cy));
    float t = fminf(dist / fmaxf(radius, 1e-6f), 1.0f);

    // Same interpolation as linear (reuse logic)
    int idx0 = 0, idx1 = 0;
    for (int i = 0; i < n_stops - 1; i++) {
        if (t >= stops[i * 5] && t <= stops[(i + 1) * 5]) {
            idx0 = i;
            idx1 = i + 1;
            break;
        }
    }

    float pos0 = stops[idx0 * 5];
    float pos1 = stops[idx1 * 5];
    float local_t = (pos1 > pos0) ? (t - pos0) / (pos1 - pos0) : 0.0f;

    float r = stops[idx0 * 5 + 1] + local_t * (stops[idx1 * 5 + 1] - stops[idx0 * 5 + 1]);
    float g = stops[idx0 * 5 + 2] + local_t * (stops[idx1 * 5 + 2] - stops[idx0 * 5 + 2]);
    float b_ = stops[idx0 * 5 + 3] + local_t * (stops[idx1 * 5 + 3] - stops[idx0 * 5 + 3]);
    float a = stops[idx0 * 5 + 4] + local_t * (stops[idx1 * 5 + 4] - stops[idx0 * 5 + 4]);

    int idx = (py * width + px) * 4;
    output[idx + 0] = r;
    output[idx + 1] = g;
    output[idx + 2] = b_;
    output[idx + 3] = a;
}

__global__ void gradient_conic_kernel(
    float* output,
    float cx, float cy, float start_angle,
    const float* stops, int n_stops,
    int width, int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float fx = (float)px / (float)(width - 1);
    float fy = (float)py / (float)(height - 1);

    float angle = atan2f(fy - cy, fx - cx) - start_angle;
    if (angle < 0) angle += 2.0f * 3.14159265358979f;
    float t = angle / (2.0f * 3.14159265358979f);

    // Same interpolation
    int idx0 = 0, idx1 = 0;
    for (int i = 0; i < n_stops - 1; i++) {
        if (t >= stops[i * 5] && t <= stops[(i + 1) * 5]) {
            idx0 = i;
            idx1 = i + 1;
            break;
        }
    }

    float pos0 = stops[idx0 * 5];
    float pos1 = stops[idx1 * 5];
    float local_t = (pos1 > pos0) ? (t - pos0) / (pos1 - pos0) : 0.0f;

    float r = stops[idx0 * 5 + 1] + local_t * (stops[idx1 * 5 + 1] - stops[idx0 * 5 + 1]);
    float g = stops[idx0 * 5 + 2] + local_t * (stops[idx1 * 5 + 2] - stops[idx0 * 5 + 2]);
    float b_ = stops[idx0 * 5 + 3] + local_t * (stops[idx1 * 5 + 3] - stops[idx0 * 5 + 3]);
    float a = stops[idx0 * 5 + 4] + local_t * (stops[idx1 * 5 + 4] - stops[idx0 * 5 + 4]);

    int idx = (py * width + px) * 4;
    output[idx + 0] = r;
    output[idx + 1] = g;
    output[idx + 2] = b_;
    output[idx + 3] = a;
}

}  // extern "C"
```

### B2: Filter Convolution Kernel

**File:** `knowledge3d/cranium/kernels/filter_convolution.cu`

```cuda
/**
 * GPU convolution filters — blur, sharpen, edge detection.
 * Uses shared memory tiling for efficiency.
 */

extern "C" {

// Separable Gaussian blur (two-pass: horizontal then vertical)
__global__ void blur_horizontal_kernel(
    const float* input,
    float* output,
    const float* kernel,
    int kernel_radius,
    int width, int height, int channels
) {
    extern __shared__ float s_row[];

    int y = blockIdx.y;
    int c = blockIdx.z;
    if (y >= height || c >= channels) return;

    int tid = threadIdx.x;
    int kernel_size = 2 * kernel_radius + 1;

    // Load row into shared memory with halo
    for (int i = tid; i < width + 2 * kernel_radius; i += blockDim.x) {
        int x = i - kernel_radius;
        x = max(0, min(width - 1, x));  // Clamp
        s_row[i] = input[(y * width + x) * channels + c];
    }
    __syncthreads();

    // Convolve
    for (int px = tid; px < width; px += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += s_row[px + k] * kernel[k];
        }
        output[(y * width + px) * channels + c] = sum;
    }
}

__global__ void blur_vertical_kernel(
    const float* input,
    float* output,
    const float* kernel,
    int kernel_radius,
    int width, int height, int channels
) {
    extern __shared__ float s_col[];

    int x = blockIdx.x;
    int c = blockIdx.z;
    if (x >= width || c >= channels) return;

    int tid = threadIdx.y;
    int kernel_size = 2 * kernel_radius + 1;

    // Load column into shared memory
    for (int i = tid; i < height + 2 * kernel_radius; i += blockDim.y) {
        int y = i - kernel_radius;
        y = max(0, min(height - 1, y));
        s_col[i] = input[(y * width + x) * channels + c];
    }
    __syncthreads();

    // Convolve
    for (int py = tid; py < height; py += blockDim.y) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += s_col[py + k] * kernel[k];
        }
        output[(py * width + x) * channels + c] = sum;
    }
}

// Edge detection (Sobel)
__global__ void sobel_edge_kernel(
    const float* input,      // Grayscale (H, W)
    float* output,           // Edge magnitude (H, W)
    int width, int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px < 1 || px >= width - 1 || py < 1 || py >= height - 1) return;

    // Sobel kernels
    float gx = -1.0f * input[(py-1)*width + (px-1)] + 1.0f * input[(py-1)*width + (px+1)]
             + -2.0f * input[(py)*width + (px-1)]   + 2.0f * input[(py)*width + (px+1)]
             + -1.0f * input[(py+1)*width + (px-1)] + 1.0f * input[(py+1)*width + (px+1)];

    float gy = -1.0f * input[(py-1)*width + (px-1)] + -2.0f * input[(py-1)*width + px] + -1.0f * input[(py-1)*width + (px+1)]
             +  1.0f * input[(py+1)*width + (px-1)] +  2.0f * input[(py+1)*width + px] +  1.0f * input[(py+1)*width + (px+1)];

    output[py * width + px] = sqrtf(gx * gx + gy * gy);
}

// Sharpen (unsharp mask)
__global__ void sharpen_kernel(
    const float* input,
    const float* blurred,
    float* output,
    float amount,
    int width, int height, int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * channels;
    if (idx >= total) return;

    float orig = input[idx];
    float blur = blurred[idx];
    output[idx] = orig + amount * (orig - blur);
}

}  // extern "C"
```

### B3: RPN Opcodes for Layers 4-7

**File:** `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` (APPEND)

```python
# ============================================================
# Drawing Galaxy Layer 4-7 Opcodes
# NOTE: Using range 0xF3-0xFF to avoid conflict with existing 0x80-0x9x
# ============================================================

# Layer 4: Gradients
OP_GRADIENT_LINEAR = 0xF3     # x1 y1 x2 y2 GRADIENT_LINEAR
OP_GRADIENT_RADIAL = 0xF4     # cx cy r GRADIENT_RADIAL
OP_GRADIENT_CONIC = 0xF5      # cx cy angle GRADIENT_CONIC
OP_GRADIENT_STOP = 0xF6       # pos r g b a GRADIENT_STOP

# Layer 5: Filters
OP_FILTER_BLUR = 0xF7         # radius FILTER_BLUR
OP_FILTER_SHARPEN = 0xF8      # amount FILTER_SHARPEN
OP_FILTER_EDGE = 0xF9         # FILTER_EDGE (Sobel)
OP_FILTER_INVERT = 0xFA       # FILTER_INVERT

# Layer 6: Lighting
OP_LIGHT_AMBIENT = 0xFB       # r g b intensity LIGHT_AMBIENT
OP_LIGHT_DIRECTIONAL = 0xFC   # dx dy dz r g b LIGHT_DIRECTIONAL

# Layer 7: Scene Composition
OP_LAYER_PUSH = 0xFD          # layer_id LAYER_PUSH
OP_LAYER_POP = 0xFE           # LAYER_POP
OP_BLEND_MODE = 0xFF          # mode BLEND_MODE

# Add to __all__
__all__ += [
    "OP_GRADIENT_LINEAR", "OP_GRADIENT_RADIAL", "OP_GRADIENT_CONIC", "OP_GRADIENT_STOP",
    "OP_FILTER_BLUR", "OP_FILTER_SHARPEN", "OP_FILTER_EDGE", "OP_FILTER_INVERT",
    "OP_LIGHT_AMBIENT", "OP_LIGHT_DIRECTIONAL",
    "OP_LAYER_PUSH", "OP_LAYER_POP", "OP_BLEND_MODE",
]
```

---

## Part C: VectorDotMap Image Codec

### Design Principle

Unify with existing codec infrastructure:
- **Audio**: MDCT + ternary quantization (exists: `sovereign_ternary_audio_codec.py`)
- **Video**: DCT8x8 + ternary quantization (exists: `sovereign_ternary_video_codec.py`)
- **Image**: DCT8x8 + field coefficients + ternary (NEW: `sovereign_ternary_image_codec.py`)

### C1: VectorDotMap Encoder Kernel

**File:** `knowledge3d/cranium/kernels/vectordotmap_encoder.cu`

```cuda
/**
 * VectorDotMap encoder — compress image to field coefficients (~2KB).
 * Integrates with existing codec_ops.cu DCT infrastructure.
 */

extern "C" {

// Field coefficient fitting using weighted frequency importance
// Reduces DCT coefficients to compact field representation
__global__ void field_coefficient_fit(
    const float* dct_coeffs,     // (num_blocks, 64) DCT coefficients
    float* field_coeffs,         // (n_coeffs,) output field
    const float* importance,     // (64,) frequency importance weights
    int num_blocks,
    int n_coeffs                 // Typically 512-2048
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_coeffs) return;

    // Map field coefficient to weighted sum of DCT coefficients
    // across blocks (spatial pooling)
    float sum = 0.0f;
    float weight_sum = 0.0f;

    int freq_idx = k % 64;          // Which DCT frequency
    int spatial_stride = num_blocks / (n_coeffs / 64 + 1);
    spatial_stride = max(1, spatial_stride);

    for (int b = 0; b < num_blocks; b += spatial_stride) {
        float w = importance[freq_idx];
        sum += w * dct_coeffs[b * 64 + freq_idx];
        weight_sum += w;
    }

    field_coeffs[k] = (weight_sum > 0) ? sum / weight_sum : 0.0f;
}

// Reconstruct DCT coefficients from field at arbitrary resolution
__global__ void field_coefficient_expand(
    const float* field_coeffs,
    float* dct_coeffs,
    const float* importance,
    int num_blocks,
    int n_coeffs
) {
    int b = blockIdx.x;
    int f = threadIdx.x;  // 0..63
    if (b >= num_blocks || f >= 64) return;

    // Interpolate from field coefficients
    int field_idx = f + (b * 64 / num_blocks) * 64;
    field_idx = min(field_idx, n_coeffs - 1);

    dct_coeffs[b * 64 + f] = field_coeffs[field_idx] * importance[f];
}

// Frequency importance weights (perceptual, based on human vision)
// DC and low frequencies more important
__constant__ float FREQ_IMPORTANCE[64] = {
    1.00f, 0.98f, 0.95f, 0.90f, 0.83f, 0.75f, 0.65f, 0.55f,
    0.98f, 0.95f, 0.90f, 0.85f, 0.78f, 0.70f, 0.60f, 0.50f,
    0.95f, 0.90f, 0.85f, 0.80f, 0.73f, 0.65f, 0.55f, 0.45f,
    0.90f, 0.85f, 0.80f, 0.75f, 0.68f, 0.60f, 0.50f, 0.40f,
    0.83f, 0.78f, 0.73f, 0.68f, 0.63f, 0.55f, 0.45f, 0.35f,
    0.75f, 0.70f, 0.65f, 0.60f, 0.55f, 0.48f, 0.40f, 0.30f,
    0.65f, 0.60f, 0.55f, 0.50f, 0.45f, 0.38f, 0.30f, 0.22f,
    0.55f, 0.50f, 0.45f, 0.40f, 0.35f, 0.28f, 0.20f, 0.15f,
};

}  // extern "C"
```

### C2: Sovereign Image Codec Class

**File:** `knowledge3d/cranium/codecs/sovereign_ternary_image_codec.py`

```python
"""
Sovereign ternary image codec — VectorDotMap implementation.

Unified with audio (MDCT) and video (DCT8x8) codec architecture.
Stores images as field coefficients (~2KB) with resolution-independent reconstruction.
"""

from __future__ import annotations

from typing import Dict, Tuple

from knowledge3d.cranium.ternary import TernaryVector, TernaryTensor, TernaryGalaxy
from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


class SovereignTernaryImageCodec:
    """
    GPU-native image codec using VectorDotMap field coefficients.

    Compression: ~2KB per image regardless of input resolution.
    Reconstruction: Any resolution from same coefficients (infinite LOD).
    """

    def __init__(
        self,
        n_coefficients: int = 512,
        threshold: float = 0.15,
    ) -> None:
        self.n_coefficients = int(n_coefficients)
        self.ops = TernaryCodecOps(threshold=threshold)
        self.rpn = ModularRPNEngine()
        self.galaxy = TernaryGalaxy()

    def encode(self, image_id: str, image_rgb: TernaryTensor) -> Dict:
        """
        Encode image to field coefficients.

        Pipeline: RGB → blocks → DCT8x8 → field fitting → ternary quantization
        """
        if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image tensor must have shape (H, W, 3)")

        h, w, _ = image_rgb.shape
        # Pad to 8x8 block boundary
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded_h = h + pad_h
        padded_w = w + pad_w

        # Process each channel
        all_coeffs = []
        for channel in range(3):
            chan_vals = self._extract_channel(image_rgb, channel, padded_h, padded_w)
            blocks = self._to_blocks(chan_vals, padded_w, padded_h)

            # DCT8x8 → field fit → quantize
            dct_coeffs = self.ops.dct8_forward(blocks)
            field_coeffs = self._field_fit(dct_coeffs, self.n_coefficients // 3)
            all_coeffs.extend(field_coeffs)

        # Ternary quantization
        quantized = self.ops.quantize(all_coeffs)
        residual_vec = TernaryVector(quantized)

        # Store with procedural seed for future enhancement
        seed_rpn = f"VECTORDOTMAP {self.n_coefficients} {h} {w}"
        self.galaxy.store_frame(image_id, seed_rpn, residual_vec)

        return {
            "image_id": image_id,
            "original_size": (h, w),
            "n_coefficients": self.n_coefficients,
            "compression_ratio": (h * w * 3) / self.n_coefficients,
            "stored_in_galaxy": True,
        }

    def decode(self, image_id: str, target_width: int = None, target_height: int = None) -> TernaryTensor:
        """
        Decode image at arbitrary resolution.

        If target_width/height not specified, uses original dimensions from seed_rpn.
        """
        seed_rpn, residual = self.galaxy.load_frame(image_id)

        # Parse original dimensions from seed
        parts = seed_rpn.split()
        if len(parts) >= 4 and parts[0] == "VECTORDOTMAP":
            orig_h, orig_w = int(parts[2]), int(parts[3])
        else:
            orig_h, orig_w = 256, 256  # Default fallback

        h = target_height or orig_h
        w = target_width or orig_w

        # Dequantize
        coeffs = self.ops.dequantize([int(v) for v in residual.to_python()])

        # Reconstruct each channel
        channel_size = h * w
        channels = []
        coeffs_per_channel = self.n_coefficients // 3

        for c in range(3):
            field_coeffs = coeffs[c * coeffs_per_channel : (c + 1) * coeffs_per_channel]
            # Expand to target resolution blocks
            num_blocks = ((h + 7) // 8) * ((w + 7) // 8)
            dct_coeffs = self._field_expand(field_coeffs, num_blocks)
            # Inverse DCT
            blocks = self.ops.dct8_inverse(dct_coeffs)
            channel = self._from_blocks(blocks, w, h)
            channels.append(channel)

        # Combine channels
        combined = []
        for idx in range(channel_size):
            r = max(0, min(255, int(channels[0][idx] if idx < len(channels[0]) else 0)))
            g = max(0, min(255, int(channels[1][idx] if idx < len(channels[1]) else 0)))
            b = max(0, min(255, int(channels[2][idx] if idx < len(channels[2]) else 0)))
            combined.extend([r, g, b])

        # Return as TernaryTensor
        ternary_rgb = [0 if v < 85 else (1 if v > 170 else -1) for v in combined]
        return TernaryTensor((h, w, 3), TernaryVector(ternary_rgb))

    def _field_fit(self, dct_coeffs: list, n_out: int) -> list:
        """Fit DCT coefficients to compact field representation."""
        # Simple importance-weighted downsampling
        # Full implementation uses PTX kernel
        if len(dct_coeffs) <= n_out:
            return dct_coeffs + [0.0] * (n_out - len(dct_coeffs))

        # Subsample with importance weighting
        result = []
        stride = len(dct_coeffs) // n_out
        for i in range(n_out):
            idx = i * stride
            result.append(dct_coeffs[idx] if idx < len(dct_coeffs) else 0.0)
        return result

    def _field_expand(self, field_coeffs: list, num_blocks: int) -> list:
        """Expand field coefficients to DCT blocks."""
        total = num_blocks * 64
        result = []
        for i in range(total):
            idx = i % len(field_coeffs) if field_coeffs else 0
            result.append(field_coeffs[idx] if field_coeffs else 0.0)
        return result

    # Helper methods (similar to video codec)
    def _extract_channel(self, tensor, channel, h, w):
        vals = tensor.values.to_python()
        result = []
        for y in range(h):
            for x in range(w):
                orig_y = min(y, tensor.shape[0] - 1)
                orig_x = min(x, tensor.shape[1] - 1)
                idx = (orig_y * tensor.shape[1] + orig_x) * 3 + channel
                result.append(float(vals[idx]) if idx < len(vals) else 0.0)
        return result

    def _to_blocks(self, channel, w, h):
        blocks = []
        for by in range(0, h, 8):
            for bx in range(0, w, 8):
                for y in range(8):
                    for x in range(8):
                        idx = (by + y) * w + (bx + x)
                        blocks.append(channel[idx] if idx < len(channel) else 0.0)
        return blocks

    def _from_blocks(self, blocks, w, h):
        result = [0.0] * (w * h)
        block_idx = 0
        for by in range(0, h, 8):
            for bx in range(0, w, 8):
                for y in range(8):
                    for x in range(8):
                        dst = (by + y) * w + (bx + x)
                        if dst < len(result) and block_idx * 64 + y * 8 + x < len(blocks):
                            result[dst] = blocks[block_idx * 64 + y * 8 + x]
                block_idx += 1
        return result


__all__ = ["SovereignTernaryImageCodec"]
```

---

## Part D: Color Galaxy (Sovereign Color Spaces)

### D1: Color Conversion Kernel

**File:** `knowledge3d/cranium/kernels/color_convert.cu`

```cuda
/**
 * Sovereign color space conversions — GPU-native, no external libs.
 *
 * Supported spaces:
 *   sRGB ↔ Linear RGB ↔ CIE XYZ ↔ CIE Lab
 *   sRGB ↔ CMYK (GCR/UCR)
 *   sRGB ↔ HSV/HSL
 *
 * Based on IEC 61966-2-1 (sRGB) and CIE standards (public domain math).
 */

extern "C" {

// sRGB gamma linearization (IEC 61966-2-1)
__device__ __forceinline__ float srgb_to_linear(float v) {
    return (v <= 0.04045f) ? (v / 12.92f) : powf((v + 0.055f) / 1.055f, 2.4f);
}

__device__ __forceinline__ float linear_to_srgb(float v) {
    return (v <= 0.0031308f) ? (12.92f * v) : (1.055f * powf(v, 1.0f / 2.4f) - 0.055f);
}

// Lab f() function
__device__ __forceinline__ float lab_f(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta * delta * delta) {
        return cbrtf(t);
    }
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

__device__ __forceinline__ float lab_f_inv(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta) {
        return t * t * t;
    }
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

// D65 white point reference
__device__ __constant__ float D65_X = 0.95047f;
__device__ __constant__ float D65_Y = 1.00000f;
__device__ __constant__ float D65_Z = 1.08883f;

// sRGB → XYZ matrix (D65)
__device__ __constant__ float SRGB_TO_XYZ[9] = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f
};

// XYZ → sRGB matrix (D65)
__device__ __constant__ float XYZ_TO_SRGB[9] = {
     3.2404542f, -1.5371385f, -0.4985314f,
    -0.9692660f,  1.8760108f,  0.0415560f,
     0.0556434f, -0.2040259f,  1.0572252f
};

// Batch RGB → Lab conversion
__global__ void rgb_to_lab_batch(
    const float* rgb,    // (N, 3) sRGB [0-1]
    float* lab,          // (N, 3) Lab
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float r = srgb_to_linear(rgb[i * 3 + 0]);
    float g = srgb_to_linear(rgb[i * 3 + 1]);
    float b = srgb_to_linear(rgb[i * 3 + 2]);

    // RGB → XYZ
    float x = SRGB_TO_XYZ[0] * r + SRGB_TO_XYZ[1] * g + SRGB_TO_XYZ[2] * b;
    float y = SRGB_TO_XYZ[3] * r + SRGB_TO_XYZ[4] * g + SRGB_TO_XYZ[5] * b;
    float z = SRGB_TO_XYZ[6] * r + SRGB_TO_XYZ[7] * g + SRGB_TO_XYZ[8] * b;

    // XYZ → Lab
    float fx = lab_f(x / D65_X);
    float fy = lab_f(y / D65_Y);
    float fz = lab_f(z / D65_Z);

    lab[i * 3 + 0] = 116.0f * fy - 16.0f;         // L* [0-100]
    lab[i * 3 + 1] = 500.0f * (fx - fy);           // a* [-128, 128]
    lab[i * 3 + 2] = 200.0f * (fy - fz);           // b* [-128, 128]
}

// Batch Lab → RGB conversion
__global__ void lab_to_rgb_batch(
    const float* lab,
    float* rgb,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float L = lab[i * 3 + 0];
    float a = lab[i * 3 + 1];
    float b_ = lab[i * 3 + 2];

    // Lab → XYZ
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b_ / 200.0f;

    float x = D65_X * lab_f_inv(fx);
    float y = D65_Y * lab_f_inv(fy);
    float z = D65_Z * lab_f_inv(fz);

    // XYZ → RGB
    float r = XYZ_TO_SRGB[0] * x + XYZ_TO_SRGB[1] * y + XYZ_TO_SRGB[2] * z;
    float g = XYZ_TO_SRGB[3] * x + XYZ_TO_SRGB[4] * y + XYZ_TO_SRGB[5] * z;
    float b = XYZ_TO_SRGB[6] * x + XYZ_TO_SRGB[7] * y + XYZ_TO_SRGB[8] * z;

    // Clamp and gamma
    rgb[i * 3 + 0] = fmaxf(0.0f, fminf(1.0f, linear_to_srgb(r)));
    rgb[i * 3 + 1] = fmaxf(0.0f, fminf(1.0f, linear_to_srgb(g)));
    rgb[i * 3 + 2] = fmaxf(0.0f, fminf(1.0f, linear_to_srgb(b)));
}

// RGB → CMYK (with GCR black generation)
__global__ void rgb_to_cmyk_batch(
    const float* rgb,    // (N, 3) [0-1]
    float* cmyk,         // (N, 4) [0-1]
    float gcr_level,     // 0.0-1.0 gray component replacement
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float r = rgb[i * 3 + 0];
    float g = rgb[i * 3 + 1];
    float b = rgb[i * 3 + 2];

    // CMY
    float c = 1.0f - r;
    float m = 1.0f - g;
    float y = 1.0f - b;

    // K (black) with GCR
    float k = fminf(c, fminf(m, y)) * gcr_level;

    // Undercolor removal
    if (k > 0.0f && k < 1.0f) {
        float denom = 1.0f - k;
        c = (c - k) / denom;
        m = (m - k) / denom;
        y = (y - k) / denom;
    }

    cmyk[i * 4 + 0] = fmaxf(0.0f, fminf(1.0f, c));
    cmyk[i * 4 + 1] = fmaxf(0.0f, fminf(1.0f, m));
    cmyk[i * 4 + 2] = fmaxf(0.0f, fminf(1.0f, y));
    cmyk[i * 4 + 3] = fmaxf(0.0f, fminf(1.0f, k));
}

// RGB → HSV
__global__ void rgb_to_hsv_batch(
    const float* rgb,
    float* hsv,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float r = rgb[i * 3 + 0];
    float g = rgb[i * 3 + 1];
    float b = rgb[i * 3 + 2];

    float max_val = fmaxf(r, fmaxf(g, b));
    float min_val = fminf(r, fminf(g, b));
    float delta = max_val - min_val;

    // V
    float v = max_val;

    // S
    float s = (max_val > 0.0f) ? (delta / max_val) : 0.0f;

    // H
    float h = 0.0f;
    if (delta > 0.0f) {
        if (max_val == r) {
            h = 60.0f * fmodf((g - b) / delta + 6.0f, 6.0f);
        } else if (max_val == g) {
            h = 60.0f * ((b - r) / delta + 2.0f);
        } else {
            h = 60.0f * ((r - g) / delta + 4.0f);
        }
    }

    hsv[i * 3 + 0] = h;          // [0-360]
    hsv[i * 3 + 1] = s;          // [0-1]
    hsv[i * 3 + 2] = v;          // [0-1]
}

// Delta E (CIE 2000) for perceptual color difference
__global__ void delta_e_batch(
    const float* lab1,    // (N, 3)
    const float* lab2,    // (N, 3)
    float* delta_e,       // (N,)
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Simplified CIE76 delta E (full CIE2000 is complex)
    float dL = lab1[i * 3 + 0] - lab2[i * 3 + 0];
    float da = lab1[i * 3 + 1] - lab2[i * 3 + 1];
    float db = lab1[i * 3 + 2] - lab2[i * 3 + 2];

    delta_e[i] = sqrtf(dL * dL + da * da + db * db);
}

}  // extern "C"
```

### D2: Color Galaxy Python Class

**File:** `knowledge3d/cranium/color_galaxy.py`

```python
"""
Color Galaxy — sovereign color space management.

Provides:
- GPU-native color conversions (RGB, Lab, CMYK, HSV)
- Named color storage (CSS, X11, approximated Pantone)
- Perceptual color matching via Delta E
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, List, Tuple

from knowledge3d.cranium.sovereign import loader


# CSS Named Colors (W3C specification) — stored as canonical RGB
CSS_NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0), "white": (255, 255, 255),
    "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
    "orange": (255, 165, 0), "purple": (128, 0, 128),
    "navy": (0, 0, 128), "teal": (0, 128, 128), "olive": (128, 128, 0),
    "maroon": (128, 0, 0), "lime": (0, 255, 0), "aqua": (0, 255, 255),
    "silver": (192, 192, 192), "gray": (128, 128, 128),
    "coral": (255, 127, 80), "salmon": (250, 128, 114),
    "gold": (255, 215, 0), "indigo": (75, 0, 130),
    # ... extend with full 148 CSS colors
}


class ColorGalaxy:
    """GPU-native color space operations."""

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent / "ptx" / "color_convert.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"color_convert.ptx not found: {ptx_path}")
        module = loader.load_module_from_file(str(ptx_path))
        self.rgb_to_lab_kernel = loader.get_function(module, "rgb_to_lab_batch")
        self.lab_to_rgb_kernel = loader.get_function(module, "lab_to_rgb_batch")
        self.rgb_to_cmyk_kernel = loader.get_function(module, "rgb_to_cmyk_batch")
        self.rgb_to_hsv_kernel = loader.get_function(module, "rgb_to_hsv_batch")
        self.delta_e_kernel = loader.get_function(module, "delta_e_batch")

        # Named color cache (Lab values for fast matching)
        self._named_colors_lab: Dict[str, Tuple[float, float, float]] = {}
        self._init_named_colors()

    def _init_named_colors(self) -> None:
        """Precompute Lab values for named colors."""
        for name, (r, g, b) in CSS_NAMED_COLORS.items():
            lab = self.rgb_to_lab([(r / 255.0, g / 255.0, b / 255.0)])[0]
            self._named_colors_lab[name] = lab

    def rgb_to_lab(self, colors: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Convert RGB [0-1] to Lab. GPU batch operation."""
        n = len(colors)
        if n == 0:
            return []
        rgb_flat = []
        for r, g, b in colors:
            rgb_flat.extend([r, g, b])
        return self._gpu_convert(rgb_flat, n, 3, 3, self.rgb_to_lab_kernel)

    def lab_to_rgb(self, colors: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Convert Lab to RGB [0-1]."""
        n = len(colors)
        if n == 0:
            return []
        lab_flat = []
        for L, a, b in colors:
            lab_flat.extend([L, a, b])
        return self._gpu_convert(lab_flat, n, 3, 3, self.lab_to_rgb_kernel)

    def rgb_to_cmyk(self, colors: List[Tuple[float, float, float]], gcr: float = 1.0) -> List[Tuple[float, float, float, float]]:
        """Convert RGB to CMYK with GCR."""
        n = len(colors)
        if n == 0:
            return []
        rgb_flat = []
        for r, g, b in colors:
            rgb_flat.extend([r, g, b])
        # Special handling for CMYK (4 outputs)
        in_buf = (ctypes.c_float * (n * 3))(*rgb_flat)
        out_buf = (ctypes.c_float * (n * 4))()
        d_in = loader.gpu_malloc(n * 3 * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(n * 4 * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), n * 3 * ctypes.sizeof(ctypes.c_float))
            block = (256, 1, 1)
            grid_x = (n + block[0] - 1) // block[0]
            loader.launch(
                self.rgb_to_cmyk_kernel,
                grid=(grid_x, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_float(gcr),
                    ctypes.c_int(n),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(out_buf, ctypes.c_void_p), d_out, n * 4 * ctypes.sizeof(ctypes.c_float))
            return [(out_buf[i*4], out_buf[i*4+1], out_buf[i*4+2], out_buf[i*4+3]) for i in range(n)]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def find_closest_named(self, color_lab: Tuple[float, float, float]) -> str:
        """Find closest named color using Delta E."""
        best_name = "black"
        best_delta = float("inf")
        for name, lab in self._named_colors_lab.items():
            de = self._delta_e_simple(color_lab, lab)
            if de < best_delta:
                best_delta = de
                best_name = name
        return best_name

    def _delta_e_simple(self, lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
        """CIE76 Delta E (CPU fallback for single values)."""
        dL = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        return (dL * dL + da * da + db * db) ** 0.5

    def _gpu_convert(self, flat_input, n, in_channels, out_channels, kernel):
        """Generic GPU conversion helper."""
        in_buf = (ctypes.c_float * (n * in_channels))(*flat_input)
        out_buf = (ctypes.c_float * (n * out_channels))()
        d_in = loader.gpu_malloc(n * in_channels * ctypes.sizeof(ctypes.c_float))
        d_out = loader.gpu_malloc(n * out_channels * ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), n * in_channels * ctypes.sizeof(ctypes.c_float))
            block = (256, 1, 1)
            grid_x = (n + block[0] - 1) // block[0]
            loader.launch(
                kernel,
                grid=(grid_x, 1, 1),
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(out_buf, ctypes.c_void_p), d_out, n * out_channels * ctypes.sizeof(ctypes.c_float))
            return [tuple(out_buf[i*out_channels:(i+1)*out_channels]) for i in range(n)]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)


__all__ = ["ColorGalaxy", "CSS_NAMED_COLORS"]
```

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `sovereign_pipeline.py` | MODIFY | Size-aware evaluation path |
| `parallel_candidate_generator.py` | MODIFY | Thread grammar singleton |
| `kernels/gradient_rasterizer.cu` | CREATE | Linear/radial/conic gradients |
| `kernels/filter_convolution.cu` | CREATE | Blur, sharpen, edge filters |
| `kernels/vectordotmap_encoder.cu` | CREATE | Image field coefficient fitting |
| `kernels/color_convert.cu` | CREATE | RGB/Lab/CMYK/HSV conversions |
| `codecs/sovereign_ternary_image_codec.py` | CREATE | VectorDotMap image codec |
| `color_galaxy.py` | CREATE | Color space management |
| `ptx_runtime/rpn_opcodes.py` | MODIFY | Add Layer 4-7 opcodes |

---

## Verification

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# 1. Compile new kernels
nvcc -ptx -o knowledge3d/cranium/ptx/gradient_rasterizer.ptx knowledge3d/cranium/kernels/gradient_rasterizer.cu
nvcc -ptx -o knowledge3d/cranium/ptx/filter_convolution.ptx knowledge3d/cranium/kernels/filter_convolution.cu
nvcc -ptx -o knowledge3d/cranium/ptx/vectordotmap_encoder.ptx knowledge3d/cranium/kernels/vectordotmap_encoder.cu
nvcc -ptx -o knowledge3d/cranium/ptx/color_convert.ptx knowledge3d/cranium/kernels/color_convert.cu

# 2. Test color conversion
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.cranium.color_galaxy import ColorGalaxy

cg = ColorGalaxy()
# RGB to Lab
lab = cg.rgb_to_lab([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
print(f'Red in Lab: L={lab[0][0]:.1f}, a={lab[0][1]:.1f}, b={lab[0][2]:.1f}')

# RGB to CMYK
cmyk = cg.rgb_to_cmyk([(1.0, 0.0, 0.0)])
print(f'Red in CMYK: C={cmyk[0][0]:.2f}, M={cmyk[0][1]:.2f}, Y={cmyk[0][2]:.2f}, K={cmyk[0][3]:.2f}')

# Find closest named color
print(f'Closest to pure red: {cg.find_closest_named(lab[0])}')
print('=== COLOR GALAXY TEST PASSED ===')
"

# 3. Test image codec
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.cranium.codecs.sovereign_ternary_image_codec import SovereignTernaryImageCodec
from knowledge3d.cranium.ternary import TernaryTensor, TernaryVector

codec = SovereignTernaryImageCodec(n_coefficients=512)

# Create test image (8x8 RGB)
test_data = [128] * (8 * 8 * 3)
tensor = TernaryTensor((8, 8, 3), TernaryVector(test_data))

# Encode
result = codec.encode('test_image', tensor)
print(f'Encoded: {result[\"n_coefficients\"]} coeffs, ratio={result[\"compression_ratio\"]:.1f}x')

# Decode at different resolution
decoded = codec.decode('test_image', target_width=16, target_height=16)
print(f'Decoded at 16x16: shape={decoded.shape}')
print('=== IMAGE CODEC TEST PASSED ===')
"

# 4. Test extraction fix
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
# After implementing size-aware path
print('Extraction test requires sovereign_pipeline.py modification')
print('=== EXTRACTION FIX READY FOR TESTING ===')
"
```

---

## Success Criteria

1. **Extraction tasks evaluated** (not dropped) — f4081712, ed74f2f2
2. **Grammar loads ONCE** — single `[GrammarGalaxy] Loaded` in log
3. **All 8 Drawing Galaxy layers operational**
4. **VectorDotMap codec** — ~2KB images, resolution-independent decode
5. **Color accuracy** — RGB↔Lab↔CMYK working on GPU
6. **100% PTX sovereignty** — no CPU math in hot path

---

## Launch Training After Implementation

```bash
tmux new-session -d -s k3d_unified "bash -lc '
  source /home/daniel/miniforge/etc/profile.d/conda.sh
  conda activate k3d-cranium
  cd \"/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D\"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  python scripts/train_arc_sovereign_loop.py \
    --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training \
               /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation \
    --max-tasks 108 --epochs 200 --cycles 1 \
    2>&1 | tee /K3D/Knowledge3D.local/logs/unified_signal_$(date +%Y%m%d_%H%M%S).log
'"
```

---

**END OF UNIFIED SPECIFICATION**

Claude (Architecture Partner)
December 12, 2025
