# Claude → Codex: Complete GPU Kernel Inventory & Missing Pieces (Week 21.6)

**Date:** February 9, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** HIGH
**Scope:** Identify missing GPU kernels, complete sovereignty

---

## Executive Summary

**Progress Confirmed:**
- ✅ PTX routing working (`ptx_full_used: true` in GPU env)
- ✅ Some GPU kernels operational (scoring, argmax)
- ✅ Tests passing (15/15)

**Remaining Bottlenecks:**
- ❌ Feature extraction still Python loops (discover_patterns_ptx lines 243-257)
- ❌ Grid operations still Python (validity checking, oracle comparison)
- ❌ Grammar reload logs appearing (architecture leak, not PTX-specific)

**User's Insight:** "We developed part on python, despite having already several kernels (meaning we only need some more)"

**This Directive:** Complete kernel inventory → identify gaps → implement missing kernels

---

## Part 1: GPU Kernels We HAVE (Working)

### ✅ Existing Kernels in `knowledge3d/cranium/ptx/arc_ops.py`:

#### 1. **Weighted Score Kernel** (lines 44-72)
```cuda
__global__ void weighted_score_kernel(
    const float* source_precision,
    const float* quality_prior,
    const float* train_similarity,
    const float* novelty,
    const float* grammar_confidence,
    const float* cross_modal,
    const float* compositional,
    const float* reuse,
    const float* family_bonus,
    float* out_score,
    const int n
)
```
**Status:** ✅ Working (called in `rank_candidates_ternary`)
**Usage:** Final candidate scoring

---

#### 2. **Argmax Kernel** (lines 74-89)
```cuda
__global__ void argmax_kernel(
    const float* scores,
    const int n,
    int* out_idx
)
```
**Status:** ✅ Working (called in `rank_candidates_ternary`)
**Usage:** Find top-ranked candidate

---

#### 3. **Discovery Score Kernel** (lines 91-109)
```cuda
__global__ void discovery_score_kernel(
    const float* confidence,
    const float* source_prior,
    const float* family_match,
    const float* novelty,
    float* out_score,
    const int n
)
```
**Status:** ✅ Working (called in `discover_patterns_ptx` line 267)
**Usage:** Pattern discovery scoring

---

#### 4. **Validity Score Kernel** (lines 111-133)
```cuda
__global__ void validity_score_kernel(
    const float* family,
    const float* shape,
    const float* palette,
    const float* object,
    const float w_family,
    const float w_shape,
    const float w_palette,
    const float w_object,
    float* out_score,
    const int n
)
```
**Status:** ✅ Working (called in `apply_validity_gates_relaxed_ptx` line 370)
**Usage:** Validity gate scoring

---

## Part 2: Operations Still in PYTHON (Need GPU Kernels)

### ❌ Missing Kernel #1: Pattern Feature Extraction

**Current Python Code** (`discover_patterns_ptx` lines 243-257):
```python
# ❌ CPU-BOUND LOOP
for pattern in patterns:
    conf = self._pattern_confidence(pattern)        # Python method
    src = self._pattern_source(pattern)             # Python method
    family = self._pattern_family(pattern)          # Python method
    confidence.append(conf)
    source_prior.append(self._source_prior(src))    # Python method
    family_match.append(1.0 if self._families_compatible(...))
    novelty.append(1.0 / max(1.0, float(...)))
```

**What This Does:**
- Extracts `confidence`, `source_prior`, `family_match`, `novelty` from pattern objects
- Each pattern processed serially on CPU

**Why It's Slow:**
- 686 patterns × 4 features × Python method calls = thousands of CPU operations
- GPU sits idle during entire loop

**Needed GPU Kernel:**
```cuda
__global__ void extract_pattern_features(
    // Input: pattern metadata arrays (pre-uploaded)
    const int* pattern_sources,      // Pattern source IDs
    const int* pattern_families,     // Pattern family IDs
    const float* pattern_confidences,// Raw confidence values
    const int* query_counts,         // Novelty lookup table
    const int expected_family,       // For family matching
    // Output: extracted features
    float* out_confidence,
    float* out_source_prior,
    float* out_family_match,
    float* out_novelty,
    int n
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // Extract features in parallel
    out_confidence[idx] = pattern_confidences[idx];
    out_source_prior[idx] = source_prior_lookup(pattern_sources[idx]);
    out_family_match[idx] = (pattern_families[idx] == expected_family) ? 1.0f : 0.0f;
    out_novelty[idx] = 1.0f / max(1.0f, (float)query_counts[idx]);
}
```

**Implementation Steps:**
1. Add kernel definition to `arc_ops.py` (after line 109)
2. Pre-process patterns to extract metadata arrays:
   ```python
   pattern_sources = np.array([self._pattern_source(p) for p in patterns], dtype=np.int32)
   pattern_families = np.array([self._pattern_family(p) for p in patterns], dtype=np.int32)
   # Upload to GPU once
   ```
3. Call kernel instead of Python loop
4. **Expected speedup:** 10-50× (thousands of Python calls → one GPU kernel)

---

### ❌ Missing Kernel #2: Grid Validity Checking

**Current Python Code** (`apply_validity_gates_relaxed_ptx` lines 346-360):
```python
# ❌ CPU-BOUND LOOP
for item in ranked_candidates:
    grid = self._to_grid(item.get("candidate"))   # Python grid conversion
    family_ok, shape_ok, palette_ok, object_ok = \
        self._candidate_validity_bits(grid, validity_profile)  # Python analysis
    fam_v.append(1.0 if family_ok else 0.0)
    # ...
```

**What This Does:**
- Converts candidate to grid (potentially complex Python logic)
- Checks family/shape/palette/object validity (grid analysis)
- Each candidate processed serially on CPU

**Why It's Slow:**
- For each candidate: grid conversion + 4 validity checks in Python
- 100-500 candidates × grid operations = major bottleneck

**Needed GPU Kernel:**
```cuda
__global__ void check_grid_validity(
    const int* candidate_grids,      // All grids flattened (n_candidates × height × width)
    const int* validity_profile,     // Expected family, shape range, palette, objects
    const int grid_height,
    const int grid_width,
    // Output: validity bits per candidate
    float* out_family,
    float* out_shape,
    float* out_palette,
    float* out_object,
    int n_candidates
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_candidates) return;

    // Each thread checks one candidate grid
    int grid_offset = idx * grid_height * grid_width;
    const int* grid = &candidate_grids[grid_offset];

    // Check family (e.g., pattern type)
    out_family[idx] = check_family_match(grid, validity_profile, grid_height, grid_width);

    // Check shape (height/width constraints)
    out_shape[idx] = check_shape_match(grid, validity_profile, grid_height, grid_width);

    // Check palette (color constraints)
    out_palette[idx] = check_palette_match(grid, validity_profile, grid_height, grid_width);

    // Check objects (object count/types)
    out_object[idx] = check_object_match(grid, validity_profile, grid_height, grid_width);
}
```

**Helper Device Functions Needed:**
```cuda
__device__ float check_family_match(const int* grid, const int* profile, int h, int w) {
    // Analyze grid pattern family (e.g., rotation, reflection, etc.)
    // Return 1.0 if matches expected family, else 0.0
}

__device__ float check_shape_match(const int* grid, const int* profile, int h, int w) {
    // Check if grid dimensions match expected range
    int expected_h = profile[0];
    int expected_w = profile[1];
    return ((h == expected_h) && (w == expected_w)) ? 1.0f : 0.0f;
}

__device__ float check_palette_match(const int* grid, const int* profile, int h, int w) {
    // Count unique colors, check against expected palette
    // (Can use shared memory for color histogram)
}

__device__ float check_object_match(const int* grid, const int* profile, int h, int w) {
    // Count objects (connected components), check against expected count
    // (More complex, may need simplified heuristic on GPU)
}
```

**Implementation Steps:**
1. Add kernel + device functions to `arc_ops.py`
2. Pre-flatten all candidate grids into single GPU array
3. Upload validity profile to GPU
4. Call kernel once for all candidates
5. **Expected speedup:** 20-100× (serial Python grid ops → parallel GPU)

---

### ❌ Missing Kernel #3: Grid Comparison (Oracle Checking)

**Current Python Code** (`check_oracle_fuzzy_ptx`, likely `_check_oracle_fuzzy_cpu`):
```python
# ❌ Probably has Python loop like:
for idx, candidate in enumerate(ranked_candidates):
    candidate_grid = self._to_grid(candidate.get("candidate"))
    # Compare candidate_grid to expected_grid pixel-by-pixel
    exact_match = (candidate_grid == expected_grid).all()
    fuzzy_score = compute_similarity(candidate_grid, expected_grid)
```

**What This Does:**
- Compares each candidate grid to ground truth grid
- Computes exact match (oracle) and fuzzy match (partial credit)

**Why It's Slow:**
- Grid comparison in Python (NumPy `.all()` etc.)
- Potentially hundreds of grid comparisons

**Needed GPU Kernel:**
```cuda
__global__ void compare_grids(
    const int* candidate_grids,      // All candidate grids (n_candidates × h × w)
    const int* expected_grid,        // Ground truth grid (h × w)
    const int grid_height,
    const int grid_width,
    // Output: match scores per candidate
    int* out_exact_match,            // 1 if exact match, else 0
    float* out_fuzzy_score,          // Similarity score [0.0, 1.0]
    int n_candidates
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_candidates) return;

    int grid_size = grid_height * grid_width;
    int grid_offset = idx * grid_size;
    const int* candidate = &candidate_grids[grid_offset];

    // Check exact match
    int exact = 1;
    int matching_pixels = 0;
    for (int i = 0; i < grid_size; ++i) {
        if (candidate[i] == expected_grid[i]) {
            matching_pixels++;
        } else {
            exact = 0;
        }
    }
    out_exact_match[idx] = exact;

    // Compute fuzzy score (% pixels matching)
    out_fuzzy_score[idx] = (float)matching_pixels / (float)grid_size;
}
```

**Implementation Steps:**
1. Add kernel to `arc_ops.py`
2. Flatten all candidate grids + expected grid
3. Upload to GPU
4. Call kernel once
5. **Expected speedup:** 10-30× (serial Python comparisons → parallel GPU)

---

### ❌ Missing Kernel #4: Grid Filtering by Threshold

**Current Python Code** (`apply_validity_gates_relaxed_ptx` lines 391-404):
```python
# ❌ CPU-BOUND LOOP
filtered: list[dict[str, Any]] = []
for idx, item in enumerate(ranked_candidates):
    hard_block = strictness_key == "strict" and fam_v[idx] < 0.5
    keep = (float(score[idx]) >= float(threshold)) and (not hard_block)
    if keep:
        filtered.append(item)
    else:
        # Count reject reasons
```

**What This Does:**
- Filters candidates based on threshold and strictness
- Builds filtered list + counts reject reasons

**Why It's Slow:**
- Python loop over all candidates
- List append operations

**Needed GPU Kernel:**
```cuda
__global__ void filter_by_threshold(
    const float* scores,
    const float* family_bits,
    const float threshold,
    const int strict_mode,         // 1 if strict, else 0
    // Output: keep mask
    int* out_keep_mask,
    int n
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    int hard_block = strict_mode && (family_bits[idx] < 0.5f);
    int keep = (scores[idx] >= threshold) && !hard_block;
    out_keep_mask[idx] = keep;
}
```

**Then use CuPy indexing to filter:**
```python
# Instead of Python loop
keep_mask_gpu = cp.zeros(n, dtype=cp.int32)
filter_kernel(..., keep_mask_gpu, ...)
cp.cuda.runtime.deviceSynchronize()

# GPU boolean indexing (fast!)
keep_indices = cp.where(keep_mask_gpu)[0]
filtered = [ranked_candidates[int(i)] for i in cp.asnumpy(keep_indices)]
```

**Expected speedup:** 5-10× (Python loop + list ops → GPU mask + indexing)

---

## Part 3: Architecture Leak (Separate Issue)

### ❌ Grammar Reload Logs (Not PTX-Related)

**Observation:** "still seeing repeated Grammar reload logs from legacy ARC path"

**Root Cause:** Benchmarks calling `ensure_default_galaxies_loaded()` per task instead of once at init

**Location:** Likely in `benchmarks/arc_agi_2.py` or adapter calling Knowledgeverse methods that reload

**Solution:**
1. Ensure Knowledgeverse loads all galaxies **once** at initialization
2. Benchmarks should **never** call galaxy loading (sovereignty violation)
3. Add assertion to prevent reloading:
   ```python
   if self._galaxies_loaded:
       raise RuntimeError("Galaxy reload attempted during benchmark - sovereignty violation!")
   ```

**This is separate from PTX work** but important for full sovereignty.

---

## Part 4: Implementation Plan

### Phase 1: Feature Extraction Kernel (2-3 hours)
**Goal:** Eliminate Python loop in `discover_patterns_ptx`

**Steps:**
1. Add `extract_pattern_features` kernel to `arc_ops.py`
2. Modify `discover_patterns_ptx`:
   - Pre-extract pattern metadata into NumPy arrays
   - Upload to GPU
   - Call kernel instead of Python loop
3. Test with micro-benchmark
4. Validate with 20-task run

**Expected Result:**
- `discover_patterns_ptx` runtime: 5-10s → **0.5-1s** (10× speedup)
- GPU usage during discovery: **40-60%**

---

### Phase 2: Grid Validity Kernel (3-4 hours)
**Goal:** Eliminate Python loop in `apply_validity_gates_relaxed_ptx`

**Steps:**
1. Add `check_grid_validity` kernel + device functions
2. Modify `apply_validity_gates_relaxed_ptx`:
   - Flatten all candidate grids into one GPU array
   - Upload validity profile to GPU
   - Call kernel instead of Python loop
3. Test with micro-benchmark
4. Validate with 20-task run

**Expected Result:**
- `apply_validity_gates_relaxed_ptx` runtime: 3-5s → **0.3-0.5s** (10× speedup)
- GPU usage during validity: **60-80%**

---

### Phase 3: Grid Comparison Kernel (2-3 hours)
**Goal:** Eliminate Python comparisons in `check_oracle_fuzzy_ptx`

**Steps:**
1. Add `compare_grids` kernel
2. Modify `check_oracle_fuzzy_ptx`:
   - Flatten candidate grids + expected grid
   - Upload to GPU
   - Call kernel instead of CPU comparison
3. Test with micro-benchmark
4. Validate with 20-task run

**Expected Result:**
- `check_oracle_fuzzy_ptx` runtime: 1-2s → **0.1-0.2s** (10× speedup)
- GPU usage during oracle: **50-70%**

---

### Phase 4: Filtering Kernel (1-2 hours)
**Goal:** Eliminate Python loop for candidate filtering

**Steps:**
1. Add `filter_by_threshold` kernel
2. Use CuPy boolean indexing
3. Test with micro-benchmark

**Expected Result:**
- Filtering runtime: 0.5-1s → **0.05-0.1s** (10× speedup)
- GPU usage during filtering: **30-50%**

---

### Phase 5: Full Integration Test (2 hours)
**Goal:** Validate all kernels working together

**Steps:**
1. Run 100-task ARC validation in GPU env:
   ```bash
   conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
     python scripts/run_all_benchmarks.py \
       --max-arc-tasks 100 \
       --arc-enable-full-ptx \
       --output-dir ../Knowledge3D.local/results/week21_6_full_gpu
   ```
2. Monitor GPU utilization:
   ```bash
   watch -n 0.5 nvidia-smi
   # Should see 60-90% GPU usage consistently
   ```
3. Validate metrics:
   - Runtime: 2 hours → **5-10 minutes**
   - GPU usage: 0% → **60-90%**
   - Oracle unlock: 0.0 → **0.10-0.30+** (faster iteration)

---

## Part 5: Testing Strategy

### Micro-Benchmarks (Test Each Kernel Individually)

**Pattern Feature Extraction:**
```python
import time
import numpy as np
from knowledge3d.cranium.ptx.arc_ops import ARCPTXOps

ops = ARCPTXOps()
# Create 686 mock patterns
patterns = [{"source": "contrastive_anti", "confidence": 0.8, "family": "rotation"}
            for _ in range(686)]
train_examples = [...]  # Mock train data

start = time.time()
result = ops.discover_patterns_ptx(train_examples=train_examples, patterns=patterns, top_k=64)
elapsed = time.time() - start
print(f"discover_patterns_ptx: {elapsed:.3f}s (target: <1s)")
```

**Grid Validity Checking:**
```python
# Create 100 mock candidates
candidates = [{"candidate": np.random.randint(0, 10, (30, 30)).tolist()} for _ in range(100)]
validity_profile = {...}

start = time.time()
filtered, report = ops.apply_validity_gates_relaxed_ptx(
    ranked_candidates=candidates,
    validity_profile=validity_profile,
    strictness="medium"
)
elapsed = time.time() - start
print(f"validity_gates: {elapsed:.3f}s (target: <0.5s)")
```

**Grid Comparison:**
```python
expected_grid = np.random.randint(0, 10, (30, 30)).tolist()

start = time.time()
oracle_metrics = ops.check_oracle_fuzzy_ptx(
    ranked_candidates=candidates,
    expected_grid=expected_grid,
    fuzzy_threshold=0.95
)
elapsed = time.time() - start
print(f"oracle_check: {elapsed:.3f}s (target: <0.2s)")
```

---

### GPU Utilization Monitoring

**Real-time monitoring during benchmark:**
```bash
# Terminal 1: Run benchmark
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  python scripts/run_all_benchmarks.py --max-arc-tasks 20 --arc-enable-full-ptx

# Terminal 2: Monitor GPU
watch -n 0.5 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv'
```

**Expected output during PTX functions:**
```
utilization.gpu [%], utilization.memory [%], memory.used [MiB]
75 %, 60 %, 2048 MiB
82 %, 65 %, 2100 MiB
68 %, 58 %, 2000 MiB
```

---

## Part 6: Success Criteria

### After All Phases Complete:

| Metric | Before (Week 21.4) | After (Week 21.6) | Target |
|--------|-------------------|-------------------|---------|
| **Runtime (100 tasks)** | ~2 hours | **5-10 minutes** | <15 min |
| **GPU Usage (avg)** | 0-5% | **60-90%** | >50% |
| **discover_patterns_ptx** | 5-10s | **0.5-1s** | <1s |
| **validity_gates** | 3-5s | **0.3-0.5s** | <1s |
| **oracle_check** | 1-2s | **0.1-0.2s** | <0.5s |
| **filtering** | 0.5-1s | **0.05-0.1s** | <0.2s |

### Quality Metrics (Secondary):
- Oracle unlock: 0.0 → **0.10-0.30+** (faster iteration enables experiments)
- ARC accuracy: 0.28 → **0.30-0.35+** (more candidates evaluated due to speed)

---

## Part 7: Timeline

**Total Estimated Time: 1.5 days**

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Feature extraction kernel | 2-3 hrs | 3 hrs |
| 2 | Grid validity kernel | 3-4 hrs | 7 hrs |
| 3 | Grid comparison kernel | 2-3 hrs | 10 hrs |
| 4 | Filtering kernel | 1-2 hrs | 12 hrs |
| 5 | Integration testing | 2 hrs | 14 hrs |

**Deliverables:**
- 4 new GPU kernels in `arc_ops.py`
- Micro-benchmark results for each kernel
- Full 100-task validation with GPU utilization graphs
- Updated progress report with speedup metrics

---

## Part 8: Code Template for Codex

### Template: Adding New Kernel

**Step 1: Define kernel source (after line 133 in arc_ops.py):**
```python
_FEATURE_EXTRACTION_KERNEL = r"""
extern "C" __global__
void extract_pattern_features(
    const int* pattern_sources,
    const int* pattern_families,
    const float* pattern_confidences,
    const int* query_counts,
    const int expected_family,
    float* out_confidence,
    float* out_source_prior,
    float* out_family_match,
    float* out_novelty,
    const int n
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // Feature extraction logic here
    out_confidence[idx] = pattern_confidences[idx];
    // ... etc
}
"""
```

**Step 2: Add kernel instance variable (line ~140):**
```python
def __init__(self) -> None:
    # ... existing kernels
    self._feature_extraction_kernel: cp.RawKernel | None = None
```

**Step 3: JIT compile in _ensure_kernels (line ~145):**
```python
def _ensure_kernels(self) -> None:
    # ... existing kernels
    if self._feature_extraction_kernel is None:
        self._feature_extraction_kernel = cp.RawKernel(
            self._FEATURE_EXTRACTION_KERNEL,
            "extract_pattern_features"
        )
```

**Step 4: Call kernel in function:**
```python
def discover_patterns_ptx(self, ...):
    # Upload data to GPU
    sources_gpu = cp.asarray(pattern_sources, dtype=cp.int32)
    # ...

    # Call kernel
    threads = 128
    blocks = (n + threads - 1) // threads
    self._feature_extraction_kernel(
        (blocks,), (threads,),
        (sources_gpu, families_gpu, ..., out_gpu, np.int32(n))
    )
    cp.cuda.runtime.deviceSynchronize()

    # Download results
    features = cp.asnumpy(out_gpu)
```

---

## Summary for Codex

**What You've Done Well:**
- ✅ PTX routing working perfectly
- ✅ 4 scoring kernels operational
- ✅ Tests passing

**What's Missing:**
- ❌ 4 feature extraction/comparison/filtering kernels
- ❌ GPU sitting idle during Python loops

**What to Do Next:**
1. Implement 4 missing kernels (see templates above)
2. Replace Python loops with GPU kernel calls
3. Micro-benchmark each kernel (target: 10× speedup each)
4. Run full 100-task validation
5. Achieve **60-90% GPU usage** consistently

**Key Principle:** Every operation on candidate arrays should happen on GPU, not CPU.

**Expected Outcome:** 2 hours → 5-10 minutes (10-20× total speedup)

---

**Claude (Architecture Partner)**
February 9, 2026
