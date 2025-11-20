# Codex-Max Phase 2.7: PSNR Quality Optimization

**Date**: 2025-11-20
**Status**: GPU Acceleration COMPLETE ✅ — Now Optimizing PSNR Quality
**Mission**: Push PSNR >20 dB on complex frames while maintaining 5× compression and <10ms latency

---

## 🎉 PHASE 2 ACHIEVEMENTS (RECAP)

**Transformative GPU Acceleration Results:**

### Audio Codec
- **Encode**: 34-43ms (was 600-716ms) → **14-17× speedup** ✅
- **Decode**: 13-15ms (was 415-708ms) → **28-47× speedup** ✅
- **Target <100ms** → **CRUSHED by 2-7×** ✅
- **PSNR**: 23-90 dB (excellent) ✅
- **Compression**: 5.2-8.9× maintained ✅

### Video Codec
- **Encode**: 2-8ms (was 138-177ms) → **17-69× speedup** ✅
- **Decode**: 2-5ms (was 142-177ms) → **28-71× speedup** ✅
- **Target <50ms** → **CRUSHED by 5-25×** ✅
- **Compression**: 4.6-4.9× (exceeds >3× target) ✅
- **GPU Sovereignty**: 100% PTX-native, zero CPU fallbacks ✅

### Matryoshka Seed Breakthrough
- **Upgraded from 256D to 8192D** leveraging K3D's 16K capacity
- **5-tier adaptive system**: 128D → 512D → 2048D → 4096D → 8192D
- **Entropy-based selection**: Frame complexity determines seed dimension
- **Two-pass ternary quantization**: Coarse + fine refinement for high-entropy blocks
- **Richer seed features**: 128 FFT bins, 32-bin edge histograms, quadrant statistics

---

## 🚨 REMAINING CHALLENGE: PSNR QUALITY

**Current State:**
```
Frame Type       | Compression | PSNR (dB) | Target | Status
-----------------|-------------|-----------|--------|--------
Procedural       |        4.9× |       inf |  >30   | ✅ Perfect
Pattern (simple) |        ~5×  |       ~3  |  >30   | ⚠️ Too low
Random (complex) |        ~5×  |       ~6  |  >30   | ⚠️ Too low
```

**Analysis:**
- ✅ **Compression maintained**: ~5× even with 8192D seeds
- ✅ **Latency exceptional**: Sub-10ms encode/decode
- ⚠️ **PSNR on complex frames**: Still below 30 dB target
- ✅ **Groundwork complete**: Adaptive seeds + two-pass ternary in place

**Root Cause:**
Even with 8192D seeds and two-pass ternary, high-frequency random content suffers from:
1. **Aggressive quantization thresholds** (0.001 still too coarse for noise)
2. **Two-pass refinement limited** (needs third pass for ultra-fine details)
3. **Procedural baseline mismatch** (8192D seed helps but not enough for pure noise)

---

## 🎯 PHASE 2.7 OPTIMIZATION STRATEGY

### Goal: Lift PSNR to >20 dB (minimum) or >30 dB (ideal) on complex frames

**Constraints:**
- ✅ Maintain compression >3× (currently ~5×, headroom available)
- ✅ Keep latency <50ms for video (currently 2-18ms, massive headroom)
- ✅ Preserve GPU sovereignty (100% PTX-native)
- ✅ Stay within VRAM budget (<40MB for codecs)

---

## 📋 IMPLEMENTATION TASKS

### Priority 1: Three-Pass Ternary Refinement (CRITICAL)

**Current (Two-Pass):**
```python
# Pass 1: Coarse quantization
q_block, _ = quantize_ternary(blocks[by, bx], threshold=0.05)

# Pass 2: Fine quantization (only for high-entropy)
if high_entropy:
    coarse_rec = dequantize_ternary(q_block, scale=scale)
    residual = blocks[by, bx] - coarse_rec
    q_fine, _ = quantize_ternary(residual, threshold=0.001)
```

**Upgrade to Three-Pass:**
```python
def encode_three_pass_ternary(self, block: np.ndarray, entropy: float) -> dict:
    """
    Three-pass ternary quantization for high-entropy blocks.

    Pass 1: Coarse (threshold=0.05) - capture low-frequency
    Pass 2: Medium (threshold=0.001) - capture mid-frequency
    Pass 3: Ultra-fine (threshold=0.0001) - capture high-frequency noise
    """
    # Pass 1: Coarse
    q1, meta1 = quantize_ternary(block, threshold=0.05, adaptive=True)
    rec1 = dequantize_ternary(q1, metadata=meta1)
    residual1 = block - rec1

    # Pass 2: Medium (for entropy >=6.0)
    if entropy >= 6.0:
        q2, meta2 = quantize_ternary(residual1, threshold=0.001, adaptive=True)
        rec2 = dequantize_ternary(q2, metadata=meta2)
        residual2 = residual1 - rec2

        # Pass 3: Ultra-fine (for entropy >=7.0)
        if entropy >= 7.0:
            q3, meta3 = quantize_ternary(residual2, threshold=0.0001, adaptive=True)

            return {
                'coarse': q1,
                'medium': q2,
                'fine': q3,
                'metadata': [meta1, meta2, meta3],
                'passes': 3
            }

        return {
            'coarse': q1,
            'medium': q2,
            'metadata': [meta1, meta2],
            'passes': 2
        }

    return {
        'coarse': q1,
        'metadata': [meta1],
        'passes': 1
    }
```

**Expected Improvement:**
- Simple frames: 1-pass (fast, efficient) ✅
- Medium frames: 2-pass (current quality) ✅
- Complex frames: 3-pass (lift PSNR from ~6 dB to >20 dB) 🎯
- Compression: Slight reduction for complex frames (5× → 4×, acceptable trade-off)

---

### Priority 2: Adaptive Quantization Thresholds per DCT Frequency Band

**Current:** Same threshold for entire 8×8 DCT block

**Upgrade:** Frequency-adaptive thresholds (preserve high-frequency better)

```python
def get_frequency_adaptive_threshold(self, block: np.ndarray, base_threshold: float = 0.05) -> np.ndarray:
    """
    Return per-coefficient thresholds based on DCT frequency position.

    Low-frequency (top-left):   coarse threshold (base_threshold)
    Mid-frequency (diagonal):   medium threshold (base_threshold * 0.2)
    High-frequency (bottom-right): fine threshold (base_threshold * 0.02)
    """
    threshold_map = np.ones((8, 8), dtype=np.float32) * base_threshold

    for by in range(8):
        for bx in range(8):
            freq_index = by + bx  # Manhattan distance from DC component

            if freq_index <= 2:  # Low-frequency
                threshold_map[by, bx] = base_threshold
            elif freq_index <= 6:  # Mid-frequency
                threshold_map[by, bx] = base_threshold * 0.2
            else:  # High-frequency (freq_index >= 7)
                threshold_map[by, bx] = base_threshold * 0.02

    return threshold_map

def quantize_ternary_adaptive_freq(self, block: np.ndarray, base_threshold: float = 0.05):
    """
    Quantize DCT block with per-coefficient thresholds.
    """
    threshold_map = self.get_frequency_adaptive_threshold(block, base_threshold)

    quantized = np.zeros_like(block, dtype=np.int8)
    for by in range(8):
        for bx in range(8):
            val = block[by, bx]
            thresh = threshold_map[by, bx]

            if val > thresh:
                quantized[by, bx] = 1
            elif val < -thresh:
                quantized[by, bx] = -1
            else:
                quantized[by, bx] = 0

    return quantized
```

**Expected Improvement:**
- Preserve high-frequency details (bottom-right of DCT block)
- Maintain compression efficiency on low-frequency (top-left)
- Lift PSNR on textured/noisy frames

---

### Priority 3: Hybrid Codec Path for Extreme Entropy (≥7.5)

**Concept:** For frames with entropy ≥7.5 (pure noise, random patterns), skip procedural baseline entirely and use full ternary DCT with higher precision.

```python
def encode_hybrid(self, frame: np.ndarray) -> dict:
    """
    Hybrid encoding:
    - Low/medium entropy (<7.5): Procedural + ternary residual (current path)
    - High entropy (≥7.5): Full ternary DCT with 3-pass refinement
    """
    entropy = self.compute_entropy(frame)

    if entropy < 7.5:
        # Standard path: procedural + residual
        seed = self.extract_procedural_seed_adaptive(frame)
        baseline = self.generate_procedural_baseline(seed)
        residual = frame - baseline

        # 8×8 DCT + ternary quantization
        dct_blocks = self.dct_8x8_forward(residual)
        quantized = self.quantize_blocks_adaptive(dct_blocks, entropy)

        return {
            'mode': 'procedural',
            'seed': seed,
            'residual': quantized,
            'entropy': entropy
        }
    else:
        # Extreme entropy path: full DCT, no procedural baseline
        dct_blocks = self.dct_8x8_forward(frame)  # Full frame, no residual

        # Three-pass ternary with ultra-fine thresholds
        quantized = self.encode_three_pass_ternary_all_blocks(
            dct_blocks,
            base_threshold=0.01  # Finer than standard 0.05
        )

        return {
            'mode': 'full_dct',
            'blocks': quantized,
            'entropy': entropy
        }
```

**Expected Improvement:**
- Random frames: PSNR from ~6 dB to >25 dB (target achieved)
- Compression: 5× → 3-4× on extreme entropy (acceptable)
- Latency: Still <20ms (DCT already GPU-accelerated)

---

### Priority 4: Perceptual Quantization Tuning

**Current:** Fixed threshold based on coefficient magnitude

**Upgrade:** Consider human visual perception (preserve edges/contrast)

```python
def compute_perceptual_threshold(self, block: np.ndarray, base_threshold: float = 0.05) -> float:
    """
    Adjust threshold based on perceptual importance.

    High edge energy → lower threshold (preserve edges)
    Low contrast → higher threshold (aggressive compression)
    """
    # Measure edge energy (high-frequency content)
    hf_energy = np.sum(np.abs(block[4:, 4:])**2)  # Bottom-right quadrant

    # Measure local contrast (std dev)
    local_contrast = np.std(block)

    if hf_energy > 0.5 or local_contrast > 0.3:
        # High detail → preserve with fine threshold
        return base_threshold * 0.1
    elif hf_energy < 0.1 and local_contrast < 0.1:
        # Low detail → aggressive compression
        return base_threshold * 2.0
    else:
        # Medium detail → standard threshold
        return base_threshold
```

**Expected Improvement:**
- Better perceptual quality (preserve edges, smooth backgrounds)
- Optimal compression/quality trade-off

---

## 🔬 VALIDATION STRATEGY

### Test Suite Requirements

**1. Benchmark Suite Expansion:**
```bash
# Add diverse frame types to benchmark
python scripts/benchmark_ternary_video.py --gpu --test-suite extended

# Extended test suite:
# - Smooth gradient (entropy ~2.0) → expect 1-pass, high PSNR
# - Natural image (entropy ~5.5) → expect 2-pass, good PSNR
# - Detailed texture (entropy ~6.8) → expect 3-pass, medium PSNR
# - Random noise (entropy ~7.8) → expect full DCT, target >20 dB PSNR
```

**2. PSNR Targets:**
```
Frame Entropy    | Mode        | Target PSNR | Target Compression
-----------------|-------------|-------------|-------------------
<4.0 (smooth)    | 1-pass proc |     >40 dB  |              >10×
4.0-6.0 (natural)| 2-pass proc |     >35 dB  |               >5×
6.0-7.5 (texture)| 3-pass proc |     >25 dB  |               >4×
≥7.5 (noise)     | full DCT    |     >20 dB  |               >3×
```

**3. Latency Verification:**
```python
# Ensure 3-pass doesn't break latency budget
assert encode_time < 50.0, f"Video encode {encode_time:.1f}ms exceeds 50ms target"
assert decode_time < 50.0, f"Video decode {decode_time:.1f}ms exceeds 50ms target"
```

**4. VRAM Budget Check:**
```python
from cuda import cuda

# Before codec init
err, mem_before = cuda.cuMemGetInfo()

# After codec init
codec = TernaryVideoCodec(use_gpu=True)
err, mem_after = cuda.cuMemGetInfo()

vram_used = (mem_before[0] - mem_after[0]) / (1024**2)
print(f"Codec VRAM: {vram_used:.1f} MB")
assert vram_used < 40, f"VRAM {vram_used:.1f}MB exceeds 40MB budget"
```

---

## 📊 EXPECTED FINAL RESULTS

### Video Codec (After Phase 2.7 Optimizations)

```
Frame Type       | Compression | Encode (ms) | Decode (ms) | PSNR (dB) | Status
-----------------|-------------|-------------|-------------|-----------|--------
Smooth gradient  |        12×  |         2-5 |         1-3 |       >45 | ✅ Excellent
Natural image    |         6×  |        5-10 |         3-7 |       >35 | ✅ Excellent
Detailed texture |         4×  |       10-18 |         5-12|       >25 | ✅ Good
Random noise     |         3×  |       15-25 |         8-15|       >20 | ✅ Acceptable
```

**All targets met:**
- ✅ Compression >3× for all frame types
- ✅ Latency <50ms encode/decode (still massive headroom)
- ✅ PSNR >20 dB minimum, >35 dB for natural images
- ✅ GPU sovereignty maintained

### Audio Codec (No Changes Needed)

**Current performance already exceeds all targets:**
- ✅ Encode: 34-43ms (<100ms target, 2-3× better)
- ✅ Decode: 13-15ms (<100ms target, 6-7× better)
- ✅ PSNR: 23-90 dB (excellent)
- ✅ Compression: 5.2-8.9× (exceeds >5× target)

**Action:** Audio codec Phase 2 COMPLETE — no further optimization needed.

---

## 🚀 IMPLEMENTATION CHECKLIST

### Phase 2.7 Tasks (Priority Order)

- [ ] **Task 1: Implement three-pass ternary refinement**
  - Add `encode_three_pass_ternary()` method
  - Entropy-based pass selection (1/2/3 passes)
  - Update `encode()` to use three-pass for entropy ≥6.0

- [ ] **Task 2: Add frequency-adaptive quantization thresholds**
  - Implement `get_frequency_adaptive_threshold()`
  - Replace uniform thresholds with per-coefficient maps
  - Test on textured frames (expect PSNR lift)

- [ ] **Task 3: Implement hybrid codec path**
  - Add `encode_hybrid()` method with entropy ≥7.5 detection
  - Full DCT path (no procedural baseline)
  - Update `decode()` to handle both modes

- [ ] **Task 4: Expand benchmark suite**
  - Add smooth gradient test case
  - Add natural image test case
  - Add detailed texture test case
  - Add random noise test case
  - Target PSNR validation for each

- [ ] **Task 5: VRAM budget verification**
  - Measure codec VRAM before/after init
  - Verify <40MB target
  - Profile for memory leaks during encode/decode

- [ ] **Task 6: Update documentation**
  - Document three-pass ternary in codec docstrings
  - Update benchmark results in TEMP/
  - Add PSNR optimization notes to CLAUDE.md

---

## 🎯 SUCCESS CRITERIA FOR PHASE 2.7

**Phase 2.7 COMPLETE when:**

1. ✅ **Video PSNR targets met:**
   - Smooth frames: >40 dB
   - Natural images: >35 dB
   - Detailed textures: >25 dB
   - Random noise: >20 dB

2. ✅ **Compression maintained:** >3× for all frame types

3. ✅ **Latency budget preserved:** <50ms encode/decode

4. ✅ **VRAM budget verified:** Codec allocation <40MB

5. ✅ **GPU sovereignty:** 100% PTX-native (no CPU fallbacks)

6. ✅ **Benchmarks passing:** Extended test suite validates all targets

---

## 🚀 NEXT PROMPT FOR CODEX-MAX

**Codex-Max, Phase 2 was a TRANSFORMATIVE SUCCESS!**

**Your achievements so far:**
- ✅ **14-71× GPU speedups** (crushing all latency targets by 2-25×)
- ✅ **8192D Matryoshka seeds** leveraging K3D's 16K capacity
- ✅ **Two-pass ternary quantization** for high-entropy blocks
- ✅ **100% GPU sovereignty** (zero CPU fallbacks)
- ✅ **5× compression maintained** with excellent latency

**Final optimization mission (Phase 2.7):**

Push PSNR quality from ~3-6 dB to >20 dB (minimum) or >30 dB (ideal) on complex frames while maintaining your exceptional compression and latency performance.

**Your tasks:**

1. **Implement three-pass ternary refinement** for entropy ≥6.0 frames
   - Pass 1: Coarse (threshold=0.05)
   - Pass 2: Medium (threshold=0.001)
   - Pass 3: Ultra-fine (threshold=0.0001) for entropy ≥7.0

2. **Add frequency-adaptive quantization thresholds**
   - Fine thresholds for high-frequency DCT coefficients
   - Coarse thresholds for low-frequency (DC/low)

3. **Implement hybrid codec path** for extreme entropy (≥7.5)
   - Skip procedural baseline
   - Use full ternary DCT with 3-pass refinement

4. **Expand benchmark suite** with diverse frame types
   - Smooth, natural, texture, random noise
   - Validate PSNR targets for each

5. **Verify VRAM budget** (<40MB for codec allocation)

**Constraints:**
- ✅ Maintain compression >3× (you have headroom: currently ~5×)
- ✅ Keep latency <50ms (you have MASSIVE headroom: currently 2-18ms)
- ✅ Preserve GPU sovereignty (100% PTX-native)

**When complete, K3D will have:**
- Fastest procedural codec in existence (14-71× GPU speedups) ✅
- Production-quality PSNR (>20 dB on all frames) 🎯
- Sovereign, GPU-native, zero dependencies ✅

**NO STUBS. PRODUCTION QUALITY. FINISH STRONG!** 🚀

---

**End of Phase 2.7 Directive**
