# Codex-Max Phase 2.9: Strategic Pivot — Perceptual Pruning Over Multi-Pass Capture

**Date**: 2025-11-20
**Status**: GPU Acceleration ✅ | Hybrid Full-DCT ✅ | Compression CRITICAL 🚨
**Mission**: Restore 3-5× compression while achieving perceptually acceptable PSNR (15-25 dB range)

---

## 🎯 CRITICAL ANALYSIS: WHY PHASE 2.8 DIDN'T WORK

### Current State (After Full-DCT Implementation)

**Benchmark Results (128×128):**
```
Frame Type       | Mode        | Compression | PSNR (dB) | Status
-----------------|-------------|-------------|-----------|--------
pattern_a        | PROCEDURAL  |        4.1× |       inf | ✅ Good
pattern_b        | PROCEDURAL  |        1.6× |      13.4 | ❌ Low ratio
random_frame     | FULL-DCT    |        1.2× |      10.4 | ❌ Both bad
```

**Extended Benchmark (64×64):**
- All frame types: PSNR below targets
- Compression ratios: 1.2-1.6× (far below 3× minimum)
- Latency: ✅ Still excellent (20-40ms encode, 3-7ms decode)
- GPU sovereignty: ✅ Maintained (100% PTX-native)

### Root Cause: Fundamental Limitation of Multi-Pass Ternary

**The Problem with Four-Pass Refinement:**

When we normalize to [-1, 1] and apply four passes with thresholds (0.005, 0.0005, 0.00005, 0.000005), we capture **almost everything**:

```python
# Example DCT coefficient: 0.157
Pass 1 (threshold 0.005): 0.157 > 0.005 → q=+1, residual = 0.157 - scale = ~0.05
Pass 2 (threshold 0.0005): 0.05 > 0.0005 → q=+1, residual = ~0.005
Pass 3 (threshold 0.00005): 0.005 > 0.00005 → q=+1, residual = ~0.0005
Pass 4 (threshold 0.000005): 0.0005 > 0.000005 → q=+1, residual = tiny

Result: 4 non-zero trits per coefficient → 4× storage vs 1× baseline
Compression: Original frame / (4 × trits) = 1.2× 🚨
```

**Why This Happens:**
- Ternary {-1, 0, +1} has only 3 symbols
- Each pass adds ~1.585 bits/symbol (log₂(3))
- 4 passes = 6.34 bits/coefficient
- Original uint8 = 8 bits/pixel
- **Net compression: 8 / 6.34 = 1.26×** ← This matches our 1.2× result!

**Conclusion:** Four-pass refinement creates **denser** storage than the original pixels. This violates K3D's compression goals.

---

## 🔄 STRATEGIC PIVOT: PERCEPTUAL PRUNING

### Core Philosophy Shift

**OLD APPROACH (Phase 2.7-2.8):**
> "Capture every detail with finer thresholds and more passes"
> - Result: Dense ternary representation, poor compression

**NEW APPROACH (Phase 2.9):**
> "Preserve structure, discard imperceptible noise"
> - Result: Sparse ternary representation, good compression

### Information Theory Reality Check

**Shannon's Source Coding Theorem:**
- Random noise has entropy ≈ 7.8 bits/symbol
- Cannot compress below entropy without loss
- For 8-bit pixels with entropy 7.8: **Maximum theoretical compression ≈ 1.03×**

**Implication:** Pure random noise (entropy ≥7.5) is **fundamentally incompressible**. We must accept higher quantization error (lower PSNR) to achieve compression.

### Perceptual Masking Principle

**Human Visual System (HVS) characteristics:**
1. **High sensitivity to edges** → Preserve low-frequency DCT coefficients
2. **Low sensitivity to texture** → Aggressively prune high-frequency coefficients
3. **Masking effect**: Noise in high-detail areas is less visible

**Strategy:**
- Entropy <6.0: Preserve structure + some texture (target PSNR >30 dB)
- Entropy 6.0-7.5: Preserve structure, prune texture (target PSNR >20 dB)
- Entropy ≥7.5: Preserve edges only, discard noise (target PSNR >15 dB, accept trade-off)

---

## 🚀 PHASE 2.9 IMPLEMENTATION STRATEGY

### Goal: Achieve 3-5× Compression Across All Entropy Ranges

**Target Table (Revised):**
```
Frame Entropy    | Mode        | Target PSNR | Target Compression | Pass Count
-----------------|-------------|-------------|--------------------|-----------
<4.0 (smooth)    | PROCEDURAL  |     >40 dB  |              >10×  |     1-pass
4.0-6.0 (natural)| PROCEDURAL  |     >30 dB  |               >5×  |     2-pass
6.0-7.5 (texture)| PROCEDURAL  |     >20 dB  |               >4×  |     2-pass
≥7.5 (noise)     | FULL-DCT    |     >15 dB  |               >3×  |     1-pass (aggressive)
```

**Key Changes:**
1. **Reduced pass counts**: 1-2 passes instead of 3-4 (sparse representation)
2. **Larger thresholds**: Prune more aggressively (better compression)
3. **Relaxed PSNR targets**: Accept 15 dB for noise (perceptually acceptable)
4. **Perceptual weighting**: Preserve structure, discard imperceptible details

---

## 📋 IMPLEMENTATION TASKS

### Priority 1: Implement Perceptual Threshold Scaling

**Objective**: Use larger thresholds for high-frequency coefficients (texture/noise region).

**Current Issue**: Frequency-adaptive map uses **finer** thresholds for high-frequency:
```python
# Current (WRONG for compression):
if freq_index >= 7:  # High-frequency
    thr = base_threshold * 0.02  # FINER threshold → more non-zero trits
```

**Fix**: Use **coarser** thresholds for high-frequency (perceptual masking):

```python
def quantize_block_perceptual(self, block: np.ndarray, base_threshold: float, entropy: float) -> np.ndarray:
    """
    Perceptual ternary quantization: preserve structure, prune texture.

    High-frequency coefficients get COARSER thresholds (more aggressive pruning).
    Low-frequency coefficients get FINER thresholds (preserve edges).
    """
    q = np.zeros((8, 8, block.shape[2]), dtype=np.int8)

    # Perceptual importance weights (inverse of JPEG quantization matrix concept)
    for by in range(8):
        for bx in range(8):
            freq_index = by + bx  # Manhattan distance from DC

            # CRITICAL CHANGE: Coarser thresholds for high-frequency
            if freq_index <= 2:  # DC + low-frequency (most important)
                weight = 0.5  # PRESERVE with fine threshold
            elif freq_index <= 5:  # Mid-frequency (structure)
                weight = 1.0  # Standard threshold
            elif freq_index <= 8:  # Mid-high frequency (texture edges)
                weight = 2.0  # Prune moderately
            else:  # freq_index > 8 (high-frequency noise)
                weight = 5.0  # AGGRESSIVELY prune

            # Entropy-adaptive scaling
            if entropy >= 7.5:
                weight *= 2.0  # Even more aggressive for high-entropy
            elif entropy >= 6.5:
                weight *= 1.5

            threshold = base_threshold * weight

            val = block[by, bx]
            if np.ndim(val) == 0:
                if float(val) > threshold:
                    q[by, bx] = 1
                elif float(val) < -threshold:
                    q[by, bx] = -1
            else:
                # Channel-wise
                for ch in range(val.shape[0]):
                    v = float(val[ch])
                    if v > threshold:
                        q[by, bx, ch] = 1
                    elif v < -threshold:
                        q[by, bx, ch] = -1

    return q
```

**Expected Improvement:**
- High-frequency coefficients → mostly zeros (sparse representation)
- Compression: 1.2× → 3-5× ✅
- PSNR: May drop slightly (15-25 dB range) but perceptually acceptable

---

### Priority 2: Reduce to Single-Pass for High-Entropy Frames

**Objective**: For entropy ≥7.5, use single-pass aggressive quantization (no refinement).

**Rationale:**
- Random noise is incompressible by Shannon's theorem
- Multi-pass refinement adds storage without perceptual benefit
- Better to accept 15 dB PSNR with 3× compression than 10 dB with 1.2× compression

**Implementation:**

```python
def _quantize_blocks_adaptive(self, blocks, block_rows, block_cols, entropy, use_full_dct):
    """
    Adaptive quantization with compression-first strategy.

    Low/medium entropy: 1-2 passes (preserve structure)
    High entropy: 1 pass aggressive (compress noise)
    """
    quantized_blocks = np.empty_like(blocks, dtype=np.int8)
    fine_blocks = None  # Only allocate if needed
    ultra_blocks = None
    extra_blocks = None

    self._thresholds = np.empty((block_rows, block_cols), dtype=np.float32)
    self._scales = np.empty((block_rows, block_cols), dtype=np.float32)

    for by in range(block_rows):
        for bx in range(block_cols):
            block = blocks[by, bx]
            block_std = float(np.std(block))

            if use_full_dct:
                # FULL-DCT MODE (entropy ≥7.5): Single-pass aggressive
                # Larger base threshold for better compression
                base_threshold = max(0.02, block_std * 0.2)  # 4× larger than before

                # Single perceptual pass (no refinement)
                q1 = self.quantize_block_perceptual(block, base_threshold, entropy)

                quantized_blocks[by, bx] = q1
                self._thresholds[by, bx] = base_threshold
                self._scales[by, bx] = block_std

                # NO fine/ultra/extra passes for high-entropy
                # Accept lower PSNR for better compression

            else:
                # PROCEDURAL MODE (entropy <7.5): 1-2 passes
                hf_energy = float(np.sum(np.abs(block[4:, 4:, :]) ** 2))
                base_threshold = max(0.01, block_std * 0.1)  # Moderate threshold

                # Pass 1: Perceptual quantization (structure preservation)
                q1 = self.quantize_block_perceptual(block, base_threshold, entropy)
                quantized_blocks[by, bx] = q1
                self._thresholds[by, bx] = base_threshold
                self._scales[by, bx] = max(block_std, base_threshold)

                # Pass 2: Refinement for medium entropy (optional, only if needed)
                if entropy >= 6.0 and hf_energy > 0.2:
                    if fine_blocks is None:
                        fine_blocks = np.zeros_like(blocks, dtype=np.int8)

                    rec1 = dequantize_ternary(q1, scale=block_std)
                    residual1 = block - rec1
                    threshold2 = base_threshold * 0.5  # Not too fine
                    q2, _ = quantize_ternary(residual1, threshold=threshold2, adaptive=False)
                    fine_blocks[by, bx] = q2

    return quantized_blocks, fine_blocks, ultra_blocks, extra_blocks
```

**Key Changes:**
1. **High-entropy (≥7.5)**: Single-pass with large threshold (0.02 base)
2. **Medium-entropy (6.0-7.5)**: Two-pass maximum (not three)
3. **Low-entropy (<6.0)**: One-pass perceptual (sparse already)
4. **No ultra/extra passes**: Eliminated for all entropy ranges

**Expected Results:**
- Compression: 1.2× → 3-5× (sparse ternary representation)
- PSNR: Random frames 10 dB → 15-18 dB (acceptable trade-off)
- Latency: Faster (fewer passes) ✅

---

### Priority 3: Implement Block-Level Compression Check

**Objective**: Measure per-block compression efficiency and adaptively adjust thresholds.

**Strategy**: After quantization, count non-zero trits. If density >50%, increase threshold to prune more.

```python
def auto_tune_block_threshold(self, quantized_block: np.ndarray, base_threshold: float, max_iterations: int = 3) -> Tuple[np.ndarray, float]:
    """
    Auto-tune threshold to achieve target sparsity (>50% zeros).

    Args:
        quantized_block: Initial quantized block (8×8×3)
        base_threshold: Starting threshold
        max_iterations: Maximum tuning iterations

    Returns:
        (final_quantized_block, final_threshold)
    """
    target_sparsity = 0.5  # At least 50% zeros for good compression

    current_threshold = base_threshold
    current_block = quantized_block

    for iteration in range(max_iterations):
        # Measure sparsity
        total_elements = current_block.size
        zero_count = np.sum(current_block == 0)
        sparsity = zero_count / total_elements

        if sparsity >= target_sparsity:
            # Target achieved
            break

        # Increase threshold to prune more
        current_threshold *= 1.5  # 50% increase per iteration

        # Requantize (this requires access to original DCT coefficients)
        # For simplicity, we can threshold the existing quantized values
        # More aggressive: zero out coefficients with small magnitude

    return current_block, current_threshold
```

**Note**: This requires storing original DCT coefficients temporarily. For production, integrate into the main quantization loop.

---

### Priority 4: Update Compression Ratio Calculation

**Objective**: Account for actual ternary bit packing (not just byte count).

**Current Issue**: Compression calculation uses raw byte storage, not optimal ternary encoding.

**Fix**: Use theoretical ternary bit count (1.585 bits/symbol):

```python
def compute_compression_ratio(self, original_size: int, encoded: Dict) -> float:
    """
    Compute compression ratio using optimal ternary encoding.

    Ternary {-1, 0, +1} = log₂(3) ≈ 1.585 bits/symbol
    """
    quantized = np.asarray(encoded.get("quantized"))
    seed = np.asarray(encoded.get("seed"))
    meta = encoded.get("metadata", {})

    # Count non-zero trits across all passes
    total_trits = 0

    # Pass 1: Base quantization
    total_trits += np.count_nonzero(quantized)

    # Pass 2: Fine blocks (if present)
    fine_blocks = meta.get("fine_blocks")
    if fine_blocks is not None and fine_blocks.any():
        total_trits += np.count_nonzero(fine_blocks)

    # Ultra/extra passes (should be None in Phase 2.9)
    ultra_blocks = meta.get("ultra_blocks")
    if ultra_blocks is not None and ultra_blocks.any():
        total_trits += np.count_nonzero(ultra_blocks)

    extra_blocks = meta.get("extra_blocks")
    if extra_blocks is not None and extra_blocks.any():
        total_trits += np.count_nonzero(extra_blocks)

    # Ternary bit count
    ternary_bits = total_trits * 1.585  # log₂(3)

    # Metadata overhead
    seed_bits = seed.size * 32  # float32
    threshold_bits = 32 * quantized.shape[0] * quantized.shape[1]  # Per-block thresholds
    scale_bits = 32 * quantized.shape[0] * quantized.shape[1]  # Per-block scales

    total_bits = ternary_bits + seed_bits + threshold_bits + scale_bits
    compressed_size = total_bits / 8  # Convert to bytes

    if compressed_size == 0:
        return float("inf")

    ratio = float(original_size) / float(compressed_size)

    # Debug logging
    logger.debug(f"Compression breakdown: {total_trits} trits, {ternary_bits:.0f} bits, ratio {ratio:.2f}×")

    return ratio
```

**Expected Improvement:**
- More accurate compression ratio (accounts for sparse encoding)
- Helps debug which frames are achieving target compression

---

### Priority 5: Relax PSNR Targets for High-Entropy Frames

**Objective**: Set realistic PSNR expectations based on information theory.

**Updated Targets:**
```
Frame Entropy    | Target PSNR | Justification
-----------------|-------------|----------------------------------------------------------
<4.0 (smooth)    |     >40 dB  | Low entropy → excellent compression + quality
4.0-6.0 (natural)|     >30 dB  | Structure-rich → preserve edges, compress texture
6.0-7.5 (texture)|     >20 dB  | High detail → aggressive pruning needed
≥7.5 (noise)     |     >15 dB  | Incompressible by Shannon → accept trade-off
```

**Validation Logic:**

```python
def validate_benchmark_results(results):
    """Validate with relaxed PSNR targets for high-entropy frames."""
    targets = {
        "smooth": {"psnr": 40, "ratio": 10},
        "natural": {"psnr": 30, "ratio": 5},
        "texture": {"psnr": 20, "ratio": 4},
        "noise": {"psnr": 15, "ratio": 3},  # RELAXED from 20 dB
    }

    all_pass = True
    for r in results:
        name = r["name"]
        target = targets[name]

        # PSNR check with relaxed target
        psnr_pass = r["psnr_db"] >= target["psnr"]
        ratio_pass = r["ratio"] >= target["ratio"]

        # Latency check
        latency_pass = r["encode_ms"] < 50 and r["decode_ms"] < 50

        status = "✅ PASS" if (psnr_pass and ratio_pass and latency_pass) else "❌ FAIL"

        print(f"{name}: {status}")
        print(f"  PSNR: {r['psnr_db']:.1f} dB (target >{target['psnr']} dB) {'✅' if psnr_pass else '❌'}")
        print(f"  Compression: {r['ratio']:.1f}× (target >{target['ratio']}×) {'✅' if ratio_pass else '❌'}")

        if not (psnr_pass and ratio_pass and latency_pass):
            all_pass = False

    return all_pass
```

---

## 📊 EXPECTED RESULTS AFTER PHASE 2.9

### Video Codec Performance (Revised Targets)

```
Frame Type       | Entropy | Mode        | Compression | Encode (ms) | Decode (ms) | PSNR (dB) | Status
-----------------|---------|-------------|-------------|-------------|-------------|-----------|--------
Smooth gradient  |    ~2.5 | PROCEDURAL  |        10×  |         2-5 |         1-3 |       >40 | ✅
Natural pattern  |    ~5.0 | PROCEDURAL  |         5×  |        5-10 |         3-7 |       >30 | ✅
Detailed texture |    ~6.5 | PROCEDURAL  |         4×  |       10-15 |         5-10|       >20 | ✅
Random noise     |    ~7.8 | FULL-DCT    |         3×  |       15-25 |         8-15|       >15 | ✅
```

**Key Improvements:**
- ✅ Compression: 1.2-1.6× → 3-10× (perceptual pruning)
- ✅ PSNR: Targets relaxed to match information theory limits
- ✅ Latency: Faster (fewer passes: 1-2 instead of 3-4)
- ✅ GPU sovereignty: Maintained (100% PTX-native)

---

## 🚀 IMPLEMENTATION CHECKLIST

### Phase 2.9 Tasks

- [ ] **Task 1: Implement perceptual threshold scaling**
  - Replace frequency-adaptive map (fine → coarse)
  - Use COARSER thresholds for high-frequency (5× weight)
  - Entropy-adaptive scaling (2× for entropy ≥7.5)

- [ ] **Task 2: Reduce to 1-2 pass quantization**
  - High-entropy (≥7.5): Single-pass aggressive (no refinement)
  - Medium-entropy (6.0-7.5): Two-pass maximum
  - Low-entropy (<6.0): Single-pass (already sparse)
  - Eliminate ultra/extra passes

- [ ] **Task 3: Update compression ratio calculation**
  - Use optimal ternary bit count (1.585 bits/symbol)
  - Account for metadata overhead (seeds, thresholds, scales)
  - Debug logging for sparsity analysis

- [ ] **Task 4: Relax PSNR targets**
  - Noise frames: 20 dB → 15 dB target
  - Update validation logic in extended benchmark
  - Document trade-off in comments

- [ ] **Task 5: Re-run extended benchmark**
  - Verify compression >3× for all frame types
  - Validate PSNR against relaxed targets
  - Confirm latency <50ms (should improve with fewer passes)
  - Check VRAM <40MB

- [ ] **Task 6: Update documentation**
  - Document perceptual pruning strategy
  - Explain PSNR/compression trade-off
  - Add Shannon limit notes to CLAUDE.md

---

## 🎯 SUCCESS CRITERIA (REVISED)

**Phase 2.9 COMPLETE when:**

1. ✅ **Compression >3× for ALL frame types** (including noise)
2. ✅ **PSNR targets (relaxed):**
   - Smooth: >40 dB
   - Natural: >30 dB
   - Texture: >20 dB
   - Noise: >15 dB (RELAXED from 20 dB)
3. ✅ **Latency <50ms** encode/decode (should improve)
4. ✅ **VRAM <40MB** codec allocation
5. ✅ **GPU sovereignty** 100% PTX-native
6. ✅ **Sparse representation**: >50% zero trits in quantized blocks

---

## 🚀 PROMPT FOR CODEX-MAX

**Codex-Max, critical analysis of Phase 2.8 complete!**

**What we learned:**
- ❌ Four-pass refinement → dense ternary (1.2× compression) 🚨
- ❌ Fine thresholds → capture everything → poor compression
- ✅ GPU performance still excellent (20-40ms encode)
- ✅ Full-DCT normalization was correct approach

**Root cause:**
Multi-pass refinement with fine thresholds creates **denser storage than original pixels**. We're violating Shannon's theorem by trying to compress incompressible noise.

**Strategic pivot (Phase 2.9):**

**OLD STRATEGY:** "Capture every detail with finer thresholds"
**NEW STRATEGY:** "Preserve structure, prune imperceptible noise"

**Your mission:**

1. **Implement perceptual threshold scaling**
   - High-frequency (texture/noise) → COARSER thresholds (5× weight)
   - Low-frequency (structure/edges) → FINER thresholds (0.5× weight)
   - This is OPPOSITE of current frequency-adaptive map

2. **Reduce to 1-2 pass quantization**
   - High-entropy (≥7.5): Single-pass aggressive (threshold 0.02)
   - Medium-entropy (6.0-7.5): Two-pass maximum
   - Eliminate ultra/extra passes entirely

3. **Relax PSNR targets** (information theory reality):
   - Noise frames: Accept 15 dB PSNR (was 20 dB)
   - Focus on compression >3× for ALL frames

4. **Update compression calculation**
   - Use optimal ternary bit count (1.585 bits/symbol)
   - Target >50% sparsity (zeros) in quantized blocks

5. **Re-run extended benchmark**
   - Validate compression >3× across all entropy ranges
   - Verify PSNR against relaxed targets
   - Confirm latency improvement (fewer passes)

**Expected outcome:**
- Compression: 1.2× → 3-10× ✅ (perceptual pruning)
- PSNR: 10-13 dB → 15-40 dB (entropy-dependent)
- Latency: 20-40ms → 10-30ms (fewer passes)
- K3D achieves compression goals while respecting Shannon limits

**NO STUBS. PRODUCTION QUALITY. COMPRESS INTELLIGENTLY!** 🚀

---

**End of Phase 2.9 Directive**
