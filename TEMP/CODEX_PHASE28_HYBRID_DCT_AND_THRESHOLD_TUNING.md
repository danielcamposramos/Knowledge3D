# Codex-Max Phase 2.8: Hybrid Full-DCT Path & Ultra-Fine Threshold Tuning

**Date**: 2025-11-20
**Status**: GPU Sovereignty ✅ | Stochastic Quantization ✅ | PSNR Lift CRITICAL 🚨
**Mission**: Push PSNR from ~3-6 dB to >20 dB on complex frames via hybrid full-DCT path

---

## 🎯 CURRENT STATE ANALYSIS

### Excellent Achievements So Far

**GPU Performance** (CRUSHED all targets):
- ✅ Audio: 34-43ms encode, 12-15ms decode (14-47× speedup)
- ✅ Video: 22-48ms encode, 3-7ms decode (17-69× speedup)
- ✅ Compression: ~5× maintained
- ✅ GPU sovereignty: 100% PTX-native, zero CPU fallbacks
- ✅ Latency: Massive headroom (22-48ms vs 50ms target)

**Advanced Features Implemented**:
- ✅ 8192D Matryoshka seeds (5-tier adaptive: 128D → 8192D)
- ✅ Richer seed features (128 FFT bins, 32-bin edges, quadrant stats)
- ✅ Frequency-adaptive quantization (per-DCT-coefficient thresholds)
- ✅ Three-pass ternary refinement (coarse + fine + ultra-fine stochastic)
- ✅ High-entropy bypass flag (entropy ≥7.0)
- ✅ VRAM reporting in benchmarks

### Critical Issue: PSNR Still Low

**Current PSNR Results:**
```
Frame Type       | Entropy | PSNR (dB) | Target | Gap
-----------------|---------|-----------|--------|-------
Procedural       |   ~2.5  |       inf |  >30   | ✅ Perfect
Pattern (simple) |   ~5.0  |      ~3.6 |  >30   | ⚠️ -26.4 dB
Random (complex) |   ~7.5  |      ~6.2 |  >20   | ⚠️ -13.8 dB
```

### Root Cause Analysis

**Why Stochastic Quantization Didn't Lift PSNR Enough:**

1. **Procedural baseline still present for entropy 7.0-7.5 frames**
   - Code bypasses at entropy ≥7.0, but random test frame might be exactly 7.5
   - Procedural seed (even 8192D) cannot capture pure white noise
   - Residual after subtracting poor baseline is HUGE → quantization destroys it

2. **Three-pass thresholds still too aggressive for noise**
   - Pass 1 (coarse): 0.05 → deletes most signal
   - Pass 2 (fine): 0.001 → helps but still loses grain
   - Pass 3 (ultra-stochastic): 0.0001 → better but third-order residual is tiny

3. **Frequency-adaptive map helps structure, not texture**
   - Preserves edges (good for natural images)
   - But random noise has equal energy across ALL frequencies
   - Need uniform preservation, not selective

**Conclusion**: For entropy ≥7.5, we must skip procedural baseline ENTIRELY and use full-frame ternary DCT with very fine thresholds.

---

## 🚀 PHASE 2.8 IMPLEMENTATION TASKS

### Priority 1: Implement Hybrid Full-DCT Path (CRITICAL)

**Objective**: For entropy ≥7.5, encode the FULL FRAME with ternary DCT (no procedural baseline).

**Current Code (Lines 80-92 in ternary_video_codec.py):**
```python
if high_entropy:  # entropy >= 7.0
    # Skip procedural baseline for highly complex frames
    proc_f = np.zeros_like(img, dtype=np.float32)
    seed = np.array([0.0], dtype=np.float32)
else:
    procedural = self.generator.generate_frame(seed, time_param=time_param)
    proc_f = procedural.astype(np.float32)
residual = img - proc_f
```

**Problem**: This sets `proc_f = zeros`, so `residual = img - zeros = img`. We're still computing residual from a baseline, just a zero baseline. The DCT then operates on the full pixel values (0-255 range), which are HUGE compared to our thresholds (0.0001-0.05).

**Fix**: Implement a true "full DCT" mode where we:
1. Normalize pixel values to [-1, 1] range before DCT
2. Use finer base thresholds (0.005 instead of 0.05)
3. Apply four-pass refinement (coarse → medium → fine → ultra-fine)

**Implementation:**

```python
def encode(self, frame: np.ndarray, seed: Optional[np.ndarray] = None, time_param: float = 0.0) -> Dict:
    """Encode RGB frame with hybrid procedural/full-DCT path."""
    img = np.asarray(frame, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("frame must have shape (H, W, 3)")

    entropy = self.compute_entropy(img)

    # CRITICAL: Entropy threshold for full-DCT mode
    use_full_dct = entropy >= 7.5  # Stricter threshold

    if use_full_dct:
        # FULL DCT MODE: Normalize to [-1, 1] for better quantization
        normalized = (img / 127.5) - 1.0  # [0, 255] → [-1, 1]
        proc_f = np.zeros_like(normalized)
        residual = normalized  # No baseline subtraction
        seed = np.array([entropy], dtype=np.float32)  # Store entropy for decode

        logger.info(f"FULL DCT mode: entropy={entropy:.2f}, skipping procedural baseline")
    else:
        # PROCEDURAL MODE: Generate baseline and compute residual
        if seed is None:
            seed = self.extract_procedural_seed_adaptive(img)

        procedural = self.generator.generate_frame(seed, time_param=time_param)
        proc_f = procedural.astype(np.float32)
        residual = img - proc_f

    # Pad to 8×8 blocks
    pad_h = (8 - (residual.shape[0] % 8)) % 8
    pad_w = (8 - (residual.shape[1] % 8)) % 8
    if pad_h or pad_w:
        residual = np.pad(residual, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    block_rows = residual.shape[0] // 8
    block_cols = residual.shape[1] // 8

    # GPU DCT (unchanged)
    blocks = self._compute_dct_blocks(residual, block_rows, block_cols)

    # ADAPTIVE QUANTIZATION based on mode
    quantized_blocks, fine_blocks, ultra_blocks, extra_blocks = self._quantize_blocks_adaptive(
        blocks, block_rows, block_cols, entropy, use_full_dct
    )

    # Metadata
    meta = {
        "block_rows": block_rows,
        "block_cols": block_cols,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "orig_shape": img.shape,
        "time_param": time_param,
        "entropy": entropy,
        "use_full_dct": use_full_dct,  # NEW FLAG
        "thresholds": self._thresholds,
        "scales": self._scales,
        "fine_blocks": fine_blocks,
        "ultra_blocks": ultra_blocks,
        "extra_blocks": extra_blocks,  # Fourth pass for extreme entropy
    }

    return {
        "seed": seed,
        "quantized": quantized_blocks,
        "metadata": meta,
        "width": self.width,
        "height": self.height,
    }
```

**Key Changes:**
1. **Stricter threshold**: `use_full_dct = entropy >= 7.5` (not 7.0)
2. **Normalization**: Pixel values → [-1, 1] range for better quantization scaling
3. **Entropy stored in seed**: Single-value seed to signal full-DCT mode
4. **New metadata flag**: `use_full_dct` for decode path selection

---

### Priority 2: Four-Pass Refinement for Full-DCT Mode

**Objective**: Use finer thresholds and more passes for extreme entropy frames.

**Implementation:**

```python
def _quantize_blocks_adaptive(self, blocks, block_rows, block_cols, entropy, use_full_dct):
    """
    Adaptive quantization with mode-specific thresholds.

    Procedural mode (entropy <7.5): 3-pass refinement
    Full-DCT mode (entropy ≥7.5): 4-pass ultra-fine refinement
    """
    quantized_blocks = np.empty_like(blocks, dtype=np.int8)
    fine_blocks = np.zeros_like(blocks, dtype=np.int8)
    ultra_blocks = np.zeros_like(blocks, dtype=np.int8)
    extra_blocks = np.zeros_like(blocks, dtype=np.int8)  # Fourth pass

    self._thresholds = np.empty((block_rows, block_cols), dtype=np.float32)
    self._scales = np.empty((block_rows, block_cols), dtype=np.float32)

    for by in range(block_rows):
        for bx in range(block_cols):
            block = blocks[by, bx]

            # Measure block complexity
            hf_energy = float(np.sum(np.abs(block[4:, 4:, :]) ** 2))
            block_std = float(np.std(block))

            if use_full_dct:
                # FULL DCT MODE: Much finer thresholds
                base_threshold = max(0.005, block_std * 0.01)  # 10× finer than procedural

                # Pass 1: Coarse (capture DC + low-frequency)
                q1 = self.quantize_block_freq(block, base_threshold=base_threshold)
                rec1 = dequantize_ternary(q1, scale=block_std)
                residual1 = block - rec1

                # Pass 2: Medium (capture mid-frequency)
                threshold2 = base_threshold * 0.1  # 0.0005 typical
                q2, _ = quantize_ternary(residual1, threshold=threshold2, adaptive=False)
                rec2 = dequantize_ternary(q2, scale=block_std * 0.1)
                residual2 = residual1 - rec2

                # Pass 3: Fine (capture high-frequency structure)
                threshold3 = base_threshold * 0.01  # 0.00005 typical
                q3, _ = quantize_ternary(residual2, threshold=threshold3, adaptive=False)
                rec3 = dequantize_ternary(q3, scale=block_std * 0.01)
                residual3 = residual2 - rec3

                # Pass 4: Ultra-fine stochastic (capture texture grain)
                threshold4 = base_threshold * 0.001  # 0.000005 typical
                seed_val = (by * 73856093 + bx * 19349663 + int(entropy * 1000)) & 0xFFFFFFFF
                q4 = self.quantize_ternary_stochastic(residual3, base_threshold=threshold4, seed=seed_val)

                quantized_blocks[by, bx] = q1
                fine_blocks[by, bx] = q2
                ultra_blocks[by, bx] = q3
                extra_blocks[by, bx] = q4  # Fourth pass

                self._thresholds[by, bx] = base_threshold
                self._scales[by, bx] = block_std

            else:
                # PROCEDURAL MODE: Current 3-pass logic (unchanged)
                base_threshold = max(0.01, block_std * (0.05 if hf_energy > 0.1 else 0.1))
                scale = max(block_std, base_threshold)

                # Frequency-adaptive ternary
                q1 = self.quantize_block_freq(block, base_threshold=base_threshold)

                # Two-pass refinement for medium entropy
                if entropy >= 6.0:
                    rec1 = dequantize_ternary(q1, scale=scale)
                    residual1 = block - rec1
                    q2, _ = quantize_ternary(residual1, threshold=max(0.001, base_threshold * 0.2))
                    fine_blocks[by, bx] = q2

                    # Three-pass for higher entropy
                    if entropy >= 7.0:
                        rec2 = dequantize_ternary(q2, scale=scale * 0.2)
                        residual2 = residual1 - rec2
                        threshold3 = max(0.0001, base_threshold * 0.02)
                        seed_val = (by * 73856093 + bx * 19349663 + int(entropy * 1000)) & 0xFFFFFFFF
                        q3 = self.quantize_ternary_stochastic(residual2, base_threshold=threshold3, seed=seed_val)
                        ultra_blocks[by, bx] = q3

                quantized_blocks[by, bx] = q1
                self._thresholds[by, bx] = base_threshold
                self._scales[by, bx] = scale

    return quantized_blocks, fine_blocks, ultra_blocks, extra_blocks
```

**Key Features:**
1. **Mode detection**: `use_full_dct` flag determines quantization strategy
2. **Full-DCT mode**: 4-pass refinement with 10× finer thresholds
3. **Procedural mode**: Existing 3-pass logic preserved
4. **Stochastic only on final pass**: Preserve deterministic structure in first 3 passes

---

### Priority 3: Update Decode Path for Full-DCT Mode

**Objective**: Correctly reconstruct full-DCT frames by reversing normalization.

**Implementation:**

```python
def decode(self, encoded: Dict) -> np.ndarray:
    """Decode with hybrid procedural/full-DCT support."""
    # Validation (unchanged)
    for key in ("seed", "quantized", "metadata"):
        if key not in encoded:
            raise ValueError(f"encoded missing '{key}'")

    seed = encoded["seed"]
    quantized = np.asarray(encoded["quantized"], dtype=np.int8)
    meta = encoded.get("metadata", {})

    # Extract metadata
    block_rows = int(meta.get("block_rows", 0))
    block_cols = int(meta.get("block_cols", 0))
    pad_h = int(meta.get("pad_h", 0))
    pad_w = int(meta.get("pad_w", 0))
    orig_shape = tuple(meta.get("orig_shape", (self.height, self.width, 3)))
    time_param = float(meta.get("time_param", 0.0))
    use_full_dct = bool(meta.get("use_full_dct", False))  # NEW FLAG

    fine_blocks = meta.get("fine_blocks")
    ultra_blocks = meta.get("ultra_blocks")
    extra_blocks = meta.get("extra_blocks")  # Fourth pass
    thresholds = meta.get("thresholds", 0.05)
    scales = meta.get("scales", 0.05)

    # Dequantize blocks (combine all passes)
    coeffs = np.empty_like(quantized, dtype=np.float32)
    for by in range(block_rows):
        for bx in range(block_cols):
            thr = thresholds[by, bx] if isinstance(thresholds, np.ndarray) else thresholds
            scale = scales[by, bx] if isinstance(scales, np.ndarray) else scales

            # Pass 1: Base reconstruction
            base = dequantize_ternary(quantized[by, bx], scale=scale)

            # Pass 2: Add fine details
            if fine_blocks is not None:
                fine_scale = scale * 0.1 if use_full_dct else scale * 0.2
                fine = dequantize_ternary(fine_blocks[by, bx], scale=fine_scale)
                base = base + fine

            # Pass 3: Add ultra-fine details
            if ultra_blocks is not None and ultra_blocks.any():
                ultra_scale = scale * 0.01 if use_full_dct else scale * 0.02
                ultra = dequantize_ternary(ultra_blocks[by, bx], scale=ultra_scale)
                base = base + ultra

            # Pass 4: Add extra-fine details (full-DCT only)
            if extra_blocks is not None and extra_blocks.any():
                extra_scale = scale * 0.001
                extra = dequantize_ternary(extra_blocks[by, bx], scale=extra_scale)
                base = base + extra

            coeffs[by, bx] = base

    # Inverse DCT (GPU/CPU path unchanged)
    residual_padded = self._compute_idct_blocks(coeffs, block_rows, block_cols)

    # Remove padding
    if pad_h or pad_w:
        residual = residual_padded[: orig_shape[0], : orig_shape[1], :]
    else:
        residual = residual_padded

    # Reconstruct based on mode
    if use_full_dct:
        # FULL DCT MODE: Denormalize from [-1, 1] back to [0, 255]
        denormalized = (residual + 1.0) * 127.5  # [-1, 1] → [0, 255]
        reconstructed = np.clip(denormalized, 0.0, 255.0)

        logger.info("FULL DCT decode: denormalized from [-1, 1] to [0, 255]")
    else:
        # PROCEDURAL MODE: Add baseline
        procedural = self.generator.generate_frame(seed, time_param=time_param).astype(np.float32)
        procedural = procedural[: orig_shape[0], : orig_shape[1], :]
        reconstructed = np.clip(procedural + residual, 0.0, 255.0)

    return reconstructed.astype(np.uint8)
```

**Key Changes:**
1. **Mode detection**: Check `use_full_dct` flag in metadata
2. **Fourth pass reconstruction**: Add `extra_blocks` if present
3. **Denormalization**: Reverse [-1, 1] normalization for full-DCT frames
4. **Logging**: Track which path is used

---

### Priority 4: Expand Benchmark Suite with Diverse Frame Types

**Objective**: Test all entropy tiers with representative frames.

**Create new benchmark**: `scripts/benchmark_ternary_video_extended.py`

```python
"""Extended video benchmark with diverse frame types."""
import numpy as np
from knowledge3d.cranium.codecs.ternary_video_codec import TernaryVideoCodec

def generate_test_frames():
    """Generate 4 test frame types covering full entropy range."""
    width, height = 128, 128

    # Frame 1: Smooth gradient (entropy ~2.5)
    smooth = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        smooth[y, :, :] = int(255 * y / height)

    # Frame 2: Natural pattern (entropy ~5.0)
    natural = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            natural[y, x, 0] = int(128 + 64 * np.sin(x * 0.1))
            natural[y, x, 1] = int(128 + 64 * np.cos(y * 0.1))
            natural[y, x, 2] = int(128 + 32 * np.sin((x + y) * 0.05))

    # Frame 3: Detailed texture (entropy ~6.5)
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            freq = 0.3
            texture[y, x, :] = int(128 + 64 * (
                np.sin(x * freq) * np.cos(y * freq) +
                np.sin(x * freq * 2) * np.cos(y * freq * 2) * 0.5
            ))

    # Frame 4: Random noise (entropy ~7.8)
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    return {
        "smooth": smooth,
        "natural": natural,
        "texture": texture,
        "noise": noise,
    }

def compute_psnr(original, reconstructed):
    """Compute PSNR between two frames."""
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    codec = TernaryVideoCodec(width=128, height=128, use_gpu=True)
    frames = generate_test_frames()

    print("=" * 80)
    print("EXTENDED VIDEO CODEC BENCHMARK (GPU)")
    print("=" * 80)

    results = []
    for name, frame in frames.items():
        # Measure entropy
        entropy = codec.compute_entropy(frame)

        # Encode
        import time
        t0 = time.perf_counter()
        encoded = codec.encode(frame)
        encode_time = (time.perf_counter() - t0) * 1000

        # Decode
        t0 = time.perf_counter()
        reconstructed = codec.decode(encoded)
        decode_time = (time.perf_counter() - t0) * 1000

        # Metrics
        psnr = compute_psnr(frame, reconstructed)
        original_size = frame.nbytes
        compressed_size = len(encoded["quantized"].tobytes()) + encoded["seed"].nbytes
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

        mode = "FULL-DCT" if encoded["metadata"].get("use_full_dct") else "PROCEDURAL"

        results.append({
            "name": name,
            "entropy": entropy,
            "mode": mode,
            "encode_ms": encode_time,
            "decode_ms": decode_time,
            "psnr_db": psnr,
            "ratio": ratio,
        })

        print(f"\n{name.upper()}")
        print(f"  Entropy: {entropy:.2f}")
        print(f"  Mode: {mode}")
        print(f"  Encode: {encode_time:.1f} ms")
        print(f"  Decode: {decode_time:.1f} ms")
        print(f"  PSNR: {psnr:.1f} dB")
        print(f"  Compression: {ratio:.1f}×")

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION AGAINST TARGETS")
    print("=" * 80)

    targets = {
        "smooth": {"psnr": 40, "ratio": 10, "latency": 50},
        "natural": {"psnr": 32, "ratio": 5, "latency": 50},
        "texture": {"psnr": 25, "ratio": 4, "latency": 50},
        "noise": {"psnr": 20, "ratio": 3, "latency": 50},
    }

    all_pass = True
    for r in results:
        name = r["name"]
        target = targets[name]

        psnr_pass = r["psnr_db"] >= target["psnr"]
        ratio_pass = r["ratio"] >= target["ratio"]
        latency_pass = r["encode_ms"] < target["latency"] and r["decode_ms"] < target["latency"]

        status = "✅ PASS" if (psnr_pass and ratio_pass and latency_pass) else "❌ FAIL"

        print(f"\n{name}: {status}")
        print(f"  PSNR: {r['psnr_db']:.1f} dB (target >{target['psnr']} dB) {'✅' if psnr_pass else '❌'}")
        print(f"  Ratio: {r['ratio']:.1f}× (target >{target['ratio']}×) {'✅' if ratio_pass else '❌'}")
        print(f"  Latency: {r['encode_ms']:.1f}/{r['decode_ms']:.1f} ms (target <{target['latency']} ms) {'✅' if latency_pass else '❌'}")

        if not (psnr_pass and ratio_pass and latency_pass):
            all_pass = False

    print("\n" + "=" * 80)
    if all_pass:
        print("🎉 ALL TARGETS MET — PHASE 2.8 COMPLETE!")
    else:
        print("⚠️  Some targets not met — further tuning needed")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

---

### Priority 5: VRAM Budget Verification

**Add VRAM measurement to extended benchmark:**

```python
def measure_vram():
    """Measure VRAM usage before/after codec init."""
    from cuda import cuda

    err, = cuda.cuInit(0)
    err, dev = cuda.cuDeviceGet(0)
    err, ctx = cuda.cuDevicePrimaryCtxRetain(dev)
    cuda.cuCtxSetCurrent(ctx)

    # Before codec
    err, mem_before = cuda.cuMemGetInfo()

    # Create codec
    codec = TernaryVideoCodec(width=1920, height=1080, use_gpu=True)

    # After codec
    err, mem_after = cuda.cuMemGetInfo()

    vram_used_mb = (mem_before[0] - mem_after[0]) / (1024**2)

    print(f"Codec VRAM usage: {vram_used_mb:.1f} MB")
    assert vram_used_mb < 40, f"VRAM {vram_used_mb:.1f} MB exceeds 40 MB budget"

    return codec
```

---

## 📊 EXPECTED RESULTS AFTER PHASE 2.8

### Video Codec Performance

```
Frame Type       | Entropy | Mode        | Compression | Encode (ms) | Decode (ms) | PSNR (dB) | Status
-----------------|---------|-------------|-------------|-------------|-------------|-----------|--------
Smooth gradient  |    ~2.5 | PROCEDURAL  |        12×  |         2-5 |         1-3 |       >45 | ✅
Natural pattern  |    ~5.0 | PROCEDURAL  |         6×  |        5-10 |         3-7 |       >35 | ✅
Detailed texture |    ~6.5 | PROCEDURAL  |         4×  |       10-18 |         5-12|       >25 | ✅
Random noise     |    ~7.8 | FULL-DCT    |         3×  |       20-35 |        10-20|       >20 | ✅
```

**Key Improvements:**
- ✅ Random noise PSNR: ~6 dB → >20 dB (full-DCT mode with 4-pass refinement)
- ✅ Compression maintained >3× across all frame types
- ✅ Latency still <50ms (well within budget)
- ✅ GPU sovereignty preserved (100% PTX-native)

---

## 🚀 IMPLEMENTATION CHECKLIST

### Phase 2.8 Tasks

- [ ] **Task 1: Implement full-DCT mode detection**
  - Update `encode()` with `use_full_dct = entropy >= 7.5`
  - Normalize pixel values to [-1, 1] for full-DCT frames
  - Store mode flag in metadata

- [ ] **Task 2: Implement four-pass refinement for full-DCT**
  - Add `extra_blocks` (fourth pass) to quantization
  - Use 10× finer thresholds (0.005 base vs 0.05)
  - Stochastic dithering only on final pass

- [ ] **Task 3: Update decode path for full-DCT**
  - Check `use_full_dct` flag in metadata
  - Denormalize from [-1, 1] to [0, 255]
  - Reconstruct fourth pass if present

- [ ] **Task 4: Create extended benchmark suite**
  - Generate 4 frame types (smooth, natural, texture, noise)
  - Validate against entropy-specific targets
  - Report mode selection (PROCEDURAL vs FULL-DCT)

- [ ] **Task 5: Measure VRAM usage**
  - Check codec allocation <40MB
  - Profile with 1080p resolution (production scale)

- [ ] **Task 6: Update documentation**
  - Document hybrid codec modes in docstrings
  - Add entropy thresholds to CLAUDE.md
  - Update TEMP/ with final benchmark results

---

## 🎯 SUCCESS CRITERIA

**Phase 2.8 COMPLETE when:**

1. ✅ **Random noise PSNR >20 dB** (currently ~6 dB)
2. ✅ **Textured frames PSNR >25 dB**
3. ✅ **Natural frames PSNR >35 dB**
4. ✅ **Smooth frames PSNR >40 dB**
5. ✅ **Compression >3× for all frame types**
6. ✅ **Latency <50ms encode/decode** (current 22-48ms has headroom)
7. ✅ **VRAM <40MB** for codec allocation
8. ✅ **GPU sovereignty maintained** (100% PTX-native)

---

## 🚀 PROMPT FOR CODEX-MAX

**Codex-Max, your stochastic quantization implementation was EXCELLENT!**

**Current achievements:**
- ✅ 8192D Matryoshka seeds with richer features
- ✅ Three-pass ternary refinement with stochastic ultra-fine
- ✅ Frequency-adaptive quantization per DCT coefficient
- ✅ GPU sovereignty enforced (zero CPU fallbacks)
- ✅ 22-48ms encode latency (massive headroom vs 50ms target)

**Remaining challenge:**
- ⚠️ PSNR still low on complex frames (~3.6 dB pattern, ~6.2 dB random)

**Root cause identified:**
Even with 8192D seeds and three-pass refinement, the procedural baseline cannot capture pure random noise. We need a **hybrid full-DCT path** for entropy ≥7.5 frames.

**Your mission (Phase 2.8):**

1. **Implement full-DCT mode** (entropy ≥7.5):
   - Normalize pixel values to [-1, 1] before DCT
   - Skip procedural baseline entirely
   - Use 10× finer thresholds (0.005 vs 0.05)

2. **Add four-pass refinement** for full-DCT:
   - Pass 1: Coarse (0.005)
   - Pass 2: Medium (0.0005)
   - Pass 3: Fine (0.00005)
   - Pass 4: Ultra-fine stochastic (0.000005)

3. **Update decode path**:
   - Detect `use_full_dct` flag
   - Denormalize from [-1, 1] to [0, 255]
   - Reconstruct all four passes

4. **Create extended benchmark** with 4 frame types:
   - Smooth gradient (entropy ~2.5) → target >40 dB PSNR
   - Natural pattern (entropy ~5.0) → target >35 dB PSNR
   - Detailed texture (entropy ~6.5) → target >25 dB PSNR
   - Random noise (entropy ~7.8) → target >20 dB PSNR

5. **Verify VRAM <40MB** with 1080p codec

**Expected outcome:**
- Random noise PSNR: ~6 dB → >20 dB ✅
- Compression: maintained >3× across all frame types ✅
- Latency: still <50ms (you have headroom) ✅
- GPU sovereignty: 100% PTX-native ✅

**When complete, K3D will have the FIRST procedural codec with production-quality PSNR across ALL entropy ranges!**

**NO STUBS. PRODUCTION QUALITY. FINISH THE BREAKTHROUGH!** 🚀

---

**End of Phase 2.8 Directive**
