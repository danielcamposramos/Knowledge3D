# Claude → Codex: PTX Python Bottleneck Fix (Week 21.5)

**Date:** February 9, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** CRITICAL
**Scope:** Fix Python bottlenecks in PTX functions

---

## Problem Identified

Week 21.4 implementation has **routing working** but **GPU utilization 0%** due to Python bottlenecks.

### Evidence:
- `ptx_full_used_rate: 1.0` ✅ (routing correct)
- GPU usage: **0%** ❌ (functions mostly Python)
- Runtime: **still hours** ❌ (Python loops dominate)
- Tests passing: **15/15** ✅ (routing logic correct)

### Root Cause:
PTX functions call GPU kernels, but **feature extraction and filtering happen in Python loops** that dominate runtime.

---

## Detailed Analysis

### Function: `discover_patterns_ptx` (arc_ops.py:243-257)

**Current Implementation:**
```python
# ❌ BOTTLENECK: Python loop extracting features
for pattern in patterns:
    conf = self._pattern_confidence(pattern)        # Python method call
    src = self._pattern_source(pattern)             # Python method call
    family = self._pattern_family(pattern)          # Python method call
    confidence.append(conf)                         # Python list
    source_prior.append(self._source_prior(src))    # Python method call
    family_match.append(...)                        # Python computation
    novelty.append(...)                             # Python computation

# ✅ Tiny GPU call (10% of runtime)
self._discovery_kernel(...)  # GPU scoring
```

**Problem:** The loop at lines 243-257 runs **entirely on CPU** in Python. Even if you have 1000 patterns, each one is processed serially in Python.

**Why GPU is idle:** GPU kernel only runs for ~1-2ms at the end, after Python spent seconds/minutes extracting features.

---

### Function: `apply_validity_gates_relaxed_ptx` (arc_ops.py:346-360, 391-404)

**Current Implementation:**
```python
# ❌ BOTTLENECK #1: Python loop extracting validity bits
for item in ranked_candidates:
    grid = self._to_grid(item.get("candidate"))   # Python grid conversion
    family_ok, shape_ok, palette_ok, object_ok = \
        self._candidate_validity_bits(grid, validity_profile)  # Python analysis
    fam_v.append(...)
    # ...

# ✅ GPU call (10% of runtime)
self._validity_kernel(...)

# ❌ BOTTLENECK #2: Python loop filtering results
for idx, item in enumerate(ranked_candidates):
    hard_block = strictness_key == "strict" and fam_v[idx] < 0.5  # Python
    keep = (float(score[idx]) >= float(threshold)) and (not hard_block)  # Python
    if keep:
        filtered.append(item)  # Python list operations
```

**Problem:** Two massive Python loops (lines 346-360, 391-404) sandwich a tiny GPU call. GPU kernel is idle 95% of the time.

---

### Function: `check_oracle_fuzzy_ptx` (likely similar pattern)

**Expected Problem:** Probably has Python loop comparing grids pixel-by-pixel on CPU instead of using GPU parallelism.

---

## Solution Strategy

### Option 1: Move Feature Extraction to GPU (Recommended)

**For `discover_patterns_ptx`:**

1. **Batch pattern data on GPU first:**
   ```python
   # Upload all pattern metadata to GPU as structured arrays
   pattern_ids_gpu = cp.asarray([p.get("id") for p in patterns])
   pattern_sources_gpu = cp.asarray([self._pattern_source(p) for p in patterns])
   # ... all metadata as GPU arrays
   ```

2. **Create GPU kernel for feature extraction:**
   ```cuda
   __global__ void extract_pattern_features(
       const int* pattern_sources,
       const int* pattern_families,
       const float* query_counts,
       float* out_confidence,
       float* out_source_prior,
       float* out_family_match,
       float* out_novelty,
       int n
   ) {
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       if (idx >= n) return;

       // Extract features in parallel on GPU
       out_confidence[idx] = compute_confidence(pattern_sources[idx]);
       out_source_prior[idx] = lookup_source_prior(pattern_sources[idx]);
       // ...
   }
   ```

3. **Call GPU kernel for extraction + scoring:**
   ```python
   # Feature extraction on GPU
   extract_features_kernel(...)
   cp.cuda.runtime.deviceSynchronize()

   # Scoring on GPU (already exists)
   self._discovery_kernel(...)
   ```

---

**For `apply_validity_gates_relaxed_ptx`:**

1. **Upload all candidate grids to GPU:**
   ```python
   # Flatten all grids into one big GPU array
   all_grids_gpu = cp.asarray(np.array([self._to_grid(c.get("candidate"))
                                          for c in ranked_candidates]).flatten())
   ```

2. **Create GPU kernel for validity checking:**
   ```cuda
   __global__ void check_validity_bits(
       const int* grids,           // All candidate grids flattened
       const int* validity_profile, // Expected family, shape, palette, objects
       float* out_family,
       float* out_shape,
       float* out_palette,
       float* out_object,
       int n_candidates,
       int grid_size
   ) {
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       if (idx >= n_candidates) return;

       // Each thread checks one candidate grid in parallel
       int offset = idx * grid_size;
       out_family[idx] = check_family(&grids[offset], validity_profile);
       out_shape[idx] = check_shape(&grids[offset], validity_profile);
       // ...
   }
   ```

3. **Create GPU kernel for filtering:**
   ```cuda
   __global__ void filter_by_threshold(
       const float* scores,
       const float* family_bits,
       const float threshold,
       const bool strict_mode,
       int* out_keep_mask,
       int n
   ) {
       int idx = blockDim.x * blockIdx.x + threadIdx.x;
       if (idx >= n) return;

       bool hard_block = strict_mode && family_bits[idx] < 0.5;
       bool keep = (scores[idx] >= threshold) && !hard_block;
       out_keep_mask[idx] = keep ? 1 : 0;
   }
   ```

4. **Filter on GPU using mask:**
   ```python
   # Generate keep mask on GPU
   keep_mask_gpu = cp.zeros(n, dtype=cp.int32)
   filter_kernel(..., keep_mask_gpu, ...)

   # Use GPU boolean indexing (fast)
   filtered_indices = cp.where(keep_mask_gpu)[0]
   filtered = [ranked_candidates[int(i)] for i in cp.asnumpy(filtered_indices)]
   ```

---

### Option 2: Hybrid Optimization (Faster to Implement)

If moving everything to GPU is too complex, **minimize Python loops**:

1. **Vectorize Python loops using NumPy:**
   ```python
   # Instead of:
   for pattern in patterns:
       confidence.append(self._pattern_confidence(pattern))

   # Do:
   confidence = np.array([self._pattern_confidence(p) for p in patterns])  # List comprehension is faster
   ```

2. **Use CuPy for array operations:**
   ```python
   # Instead of:
   for idx, item in enumerate(ranked_candidates):
       keep = (float(score[idx]) >= float(threshold))
       if keep:
           filtered.append(item)

   # Do:
   keep_mask = cp.asnumpy(score_gpu >= threshold)  # Boolean indexing on GPU
   filtered = [item for item, keep in zip(ranked_candidates, keep_mask) if keep]
   ```

---

## Implementation Priority

### Phase 1: Quick Win (Option 2 - 1 hour)
- Replace Python loops with NumPy vectorization where possible
- Use CuPy boolean indexing for filtering
- **Expected impact:** 2-3× speedup, GPU usage 10-20%

### Phase 2: Full GPU (Option 1 - 4-6 hours)
- Move feature extraction to GPU kernels
- Move validity checking to GPU kernels
- Move filtering to GPU kernels
- **Expected impact:** 50-100× speedup, GPU usage 60-90%

---

## Success Criteria

**After Phase 1 (Hybrid):**
- Runtime: 2 hours → **40-60 minutes** (2-3× speedup)
- GPU usage: 0% → **10-20%**
- Tests: 15/15 still passing

**After Phase 2 (Full GPU):**
- Runtime: 2 hours → **5-10 minutes** (10-20× speedup)
- GPU usage: 0% → **60-90%**
- Tests: 15/15 still passing
- Oracle unlock: `oracle_at_all: 0.0 → 0.10+` (faster iteration = more experiments)

---

## Testing Strategy

1. **Micro-benchmark individual functions:**
   ```python
   import time
   start = time.time()
   result = ops.discover_patterns_ptx(train_examples=examples, patterns=patterns, top_k=64)
   print(f"discover_patterns_ptx: {time.time() - start:.3f}s")
   ```

2. **GPU utilization monitoring:**
   ```bash
   # In separate terminal while benchmark runs:
   watch -n 0.5 nvidia-smi
   # Should see GPU usage spike to 60-90% during PTX functions
   ```

3. **Full benchmark with 20 tasks:**
   ```bash
   python scripts/run_all_benchmarks.py \
     --max-arc-tasks 20 \
     --arc-enable-full-ptx \
     --output-dir ../Knowledge3D.local/results/week21_5_gpu_optimized
   ```

---

## Expected Timeline

- **Phase 1 (Hybrid):** 1-2 hours
- **Phase 2 (Full GPU):** 4-6 hours
- **Testing + Validation:** 2 hours
- **Total:** 1 day

---

## Communication

**After Phase 1 completion:**
- Report: runtime improvement, GPU usage %
- Micro-benchmark times for each PTX function

**After Phase 2 completion:**
- Full benchmark results (100 ARC tasks)
- GPU utilization graphs
- Oracle unlock metrics

---

**Priority:** Start with Phase 1 (hybrid) to validate approach, then proceed to Phase 2 (full GPU) once confirmed working.

**Key Principle:** The goal is **GPU utilization 60-90%** during PTX functions, not just "PTX functions being called."

---

**Claude (Architecture Partner)**
February 9, 2026
