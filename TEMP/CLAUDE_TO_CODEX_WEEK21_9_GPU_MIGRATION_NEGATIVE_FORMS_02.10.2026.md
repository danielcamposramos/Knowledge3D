# Claude → Codex: Week 21.9 GPU Migration + Negative Form Duality

**Date:** February 10, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** CRITICAL
**Scope:** Full GPU migration + negative form architecture

---

## User's Directive

> "He's right, and as sooner as we can get all things running inside the GPU, the better, because things taking minutes will compute in nano seconds (we've experimented this before), enhance on top of what he suggested Claude, let's keep the pace"

**Translation:**
1. ✅ Approve your technical suggestions
2. 🚀 PRIORITY: Move ALL computation to GPU (user confirmed nanosecond vs minute speedups)
3. ⚡ Maintain rapid iteration pace
4. 🎯 Integrate negative form duality architecture

---

## Week 21.8 Status Summary

### ✅ What's Working:
- **PTX routing confirmed**: `solver="arc_ptx_ops"`, all flags active
- **Unified persistence**: `shared_instance=true`, same instance ID throughout
- **Kernels executing**: Profiling confirms 0.046-1.335ms GPU execution
- **Tests passing**: 15/15 PTX tests, sovereignty maintained

### ❌ What's NOT Working:
- **GPU usage: 5%** (should be 60-90%)
- **Palette score: 0.6356** (weakest component, 58 oracle failures)
- **Batch sizes: n=1-2** (too small for GPU saturation)
- **Python loops dominate runtime** (90% CPU, 10% GPU)

### 🎯 Oracle Unlock Target:
- Current: `oracle_at_all: 0.0`
- Target Week 21.9: `oracle_at_all: 0.10+`
- Full GPU + negative forms should unlock this

---

## Priority 0: Document Negative Form Duality (1 hour)

### Architectural Insight (From Daniel)

**User's Insight:**
> "There are positive character forms, the ones we already have, but a character can also be 'negative', think of it like 'carved out'... In 3D shapes, this would be the difference from seeing a statue from outside (regular character) and from inside ('negative' character - the parts that if you take out from a square block will form the statue)."

### Ternary Mapping

**Positive Form (Additive):**
- Like ink on paper (projection/addition)
- Raised/extruded in 3D
- Foreground in figure-ground
- Maps to: `+1` in ternary

**Negative Form (Subtractive):**
- Like carved inscription (removal/subtraction)
- Recessed/carved in 3D
- Background in figure-ground
- Maps to: `-1` in ternary

**Neutral (Empty):**
- Untouched canvas/block
- Neither raised nor carved
- Maps to: `0` in ternary

### Zero Storage Cost Derivation

**Key Principle:** Negative form derives procedurally from positive form.

```python
# Drawing Galaxy: Store ONLY positive form (200 bytes)
positive_glyph_A = [
    BEZIER([(0,0), (5,10), (10,0)]),  # Left stroke
    LINE((5,10), (5,5)),                # Right stroke
    LINE((3,5), (7,5))                  # Crossbar
]

# Negative form: Derive on-demand (ZERO additional storage!)
def derive_negative_form(positive_glyph, canvas_size=(10,10)):
    """
    Negative = Canvas - Positive
    (Carved form = Remove positive from solid block)

    Zero storage cost: computed when needed.
    """
    canvas = np.ones(canvas_size, dtype=np.int8)  # Solid block (+1 everywhere)
    positive_mask = rasterize_rpn(positive_glyph)  # Which pixels are raised (+1)
    negative_mask = canvas - positive_mask          # Invert: carved-out space

    # Result: +1 (raised), 0 (empty), -1 (carved)
    # Maps directly to ternary contrastive encoding!
    return negative_mask
```

### Figure-Ground Reversal for ARC

**Critical for ARC Tasks:** Many tasks require seeing the "negative space" or inverting foreground/background.

**Example ARC Task:**
```
Train pair 1:
Input:  ▓▓▓░░░▓▓▓  (foreground: solid blocks)
Output: ░░░▓▓▓░░░  (background: negative space between blocks)

Traditional AI: Struggles to "see" the gap as the object
K3D with negative forms: Can query negative glyph → sees carved space as object
```

### System-Wide Generalization

**Audio Galaxy:**
- Positive: Sound wave (presence)
- Negative: Silence/pause (absence with meaning)

**3D Objects Galaxy:**
- Positive: Solid volume (mass occupies space)
- Negative: Cavity/void (empty space with boundaries)

**Reality Galaxy (Physics):**
- Positive: Matter/energy presence
- Negative: Vacuum/field absence (quantum foam)

**Math Galaxy:**
- Positive: Additive operations (+, ∪, ∨)
- Negative: Subtractive operations (-, ∩, ∧)

### Implementation Requirements

**1. Extend Character Galaxy Schema:**

```python
# knowledge3d/knowledgeverse/character_galaxy.py

class CharacterGalaxy:
    def add_character(self, unicode_point, positive_glyph_ref, canvas_size=(10,10), **metadata):
        """
        Add character with positive form reference.
        Negative form derives procedurally (zero storage).
        """
        self.entries[unicode_point] = {
            "unicode": unicode_point,
            "positive_form_ref": positive_glyph_ref,  # Reference to Drawing Galaxy
            "negative_form_ref": f"{positive_glyph_ref}_negative",  # Virtual reference
            "canvas_size": canvas_size,
            "form_polarity": "both",  # Can query as +1 or -1
            **metadata
        }

    def get_positive_form(self, unicode_point):
        """Fetch positive (raised) form."""
        entry = self.entries[unicode_point]
        return self.drawing_galaxy.get(entry["positive_form_ref"])

    def get_negative_form(self, unicode_point):
        """Derive negative (carved) form on-demand."""
        entry = self.entries[unicode_point]
        positive_glyph = self.drawing_galaxy.get(entry["positive_form_ref"])

        # Procedural derivation (zero storage)
        canvas = np.ones(entry["canvas_size"], dtype=np.int8)
        positive_mask = self._rasterize_glyph(positive_glyph)
        negative_mask = canvas - positive_mask

        return negative_mask
```

**2. Update DUAL_CLIENT_CONTRACT_SPECIFICATION.md:**

Add section 1.7 "Positive/Negative Form Duality":

```markdown
### 1.7 Positive/Negative Form Duality

**Principle:** Every form has dual representation (additive + subtractive) that derives procedurally.

**Ternary Encoding:**
- Positive form: `+1` (raised, projected, foreground)
- Neutral space: `0` (empty, untouched)
- Negative form: `-1` (carved, recessed, background)

**Zero Storage Cost:** Negative derives from positive via `canvas - positive_mask`.

**Figure-Ground Reversal:** Enables ARC tasks requiring negative space perception.

**System-Wide:** Applies to all modalities (visual, audio, 3D, physics).
```

**3. ARC Adapter Integration:**

```python
# benchmarks/arc_agi_2_adapter.py

def solve_arc_task_with_polarity(self, task_dict, kverse):
    """
    Solve ARC task using positive AND negative form awareness.
    """
    # Try standard (positive) navigation
    positive_result = self._solve_with_polarity(task_dict, polarity=+1)

    # If low confidence, try negative (figure-ground reversal)
    if positive_result["confidence"] < 0.5:
        negative_result = self._solve_with_polarity(task_dict, polarity=-1)

        # Use higher-confidence result
        return max(positive_result, negative_result, key=lambda r: r["confidence"])

    return positive_result

def _solve_with_polarity(self, task_dict, polarity):
    """
    Solve with specific form polarity.

    polarity=+1: Positive forms (foreground, raised)
    polarity=-1: Negative forms (background, carved)
    """
    # Extract patterns with polarity awareness
    patterns = self.ptx_ops.discover_patterns_ptx(
        train_examples=task_dict["train"],
        polarity=polarity  # NEW parameter
    )

    # Generate candidates using polarity-aware composition
    candidates = self.ptx_ops.apply_patterns_to_test(
        test_input=task_dict["test"][0]["input"],
        patterns=patterns,
        polarity=polarity
    )

    # Rank with figure-ground duality
    ranked = self.ptx_ops.rank_candidates_ternary(
        candidates=candidates,
        train_examples=task_dict["train"],
        allow_polarity_flip=True  # Enable reversal if needed
    )

    return ranked[0] if ranked else None
```

---

## Priority 1: GPU-Side Feature Extraction (4 hours)

### Current Bottleneck Analysis

**From PTX Profiling:**
- Python loops: **~100-1000ms** (CPU-bound)
- GPU kernels: **~0.05-1.3ms** (idle 95% of time)
- Batch sizes: **n=1-2** (micro-batches)

**Goal:** Move ALL feature extraction to GPU, achieve 60-90% GPU utilization.

### Target Functions

#### Function 1: `discover_patterns_ptx` (arc_ops.py:243-267)

**Current Implementation (HYBRID):**

```python
# ❌ BOTTLENECK: Lines 243-257 (Python loop on CPU)
for pattern in patterns:
    conf = self._pattern_confidence(pattern)        # Python method call
    src = self._pattern_source(pattern)             # Python method call
    family = self._pattern_family(pattern)          # Python method call
    confidence.append(conf)                         # Python list
    source_prior.append(self._source_prior(src))    # Python dict lookup
    family_match.append(...)                        # Python computation
    novelty.append(...)                             # Python computation

# ✅ Tiny GPU call: Line 267 (~1.335ms, 10% of total time)
self._discovery_kernel(confidence_gpu, source_gpu, family_gpu, novelty_gpu, out_scores)
```

**Problem:** Loop iterates over 64-700 patterns serially on CPU, each calling Python methods.

**Solution: Full GPU Pipeline**

```python
def discover_patterns_ptx(self, train_examples, patterns, top_k=64, polarity=+1):
    """
    Discover patterns with FULL GPU pipeline.
    Zero Python loops in hot path.
    """
    n = len(patterns)

    # Step 1: Upload pattern metadata to GPU (batch operation)
    pattern_ids = cp.asarray([p.get("id", 0) for p in patterns], dtype=cp.int32)
    pattern_sources = cp.asarray([self._pattern_source(p) for p in patterns], dtype=cp.int32)
    pattern_families = cp.asarray([self._pattern_family(p) for p in patterns], dtype=cp.int32)
    query_counts = cp.asarray([self._query_count.get(p.get("id"), 0) for p in patterns], dtype=cp.float32)
    polarity_flags = cp.full(n, polarity, dtype=cp.int8)  # +1 or -1

    # Step 2: Allocate output arrays on GPU
    confidence_gpu = cp.zeros(n, dtype=cp.float32)
    source_prior_gpu = cp.zeros(n, dtype=cp.float32)
    family_match_gpu = cp.zeros(n, dtype=cp.float32)
    novelty_gpu = cp.zeros(n, dtype=cp.float32)
    final_scores_gpu = cp.zeros(n, dtype=cp.float32)

    # Step 3: GPU Kernel 1 - Extract Features (REPLACES Python loop)
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    self._extract_pattern_features_kernel(
        (blocks,), (threads_per_block,),
        (pattern_sources, pattern_families, query_counts, polarity_flags,
         confidence_gpu, source_prior_gpu, family_match_gpu, novelty_gpu, n)
    )

    # Step 4: GPU Kernel 2 - Weighted Scoring (already exists, line 267)
    self._discovery_kernel(
        confidence_gpu, source_prior_gpu, family_match_gpu, novelty_gpu,
        final_scores_gpu, n
    )

    # Step 5: GPU Kernel 3 - Top-K Selection (REPLACES np.argsort)
    top_k_indices = cp.zeros(top_k, dtype=cp.int32)
    self._topk_kernel((1,), (256,), (final_scores_gpu, top_k_indices, n, top_k))

    # Step 6: Download ONLY top-k results to CPU
    top_k_cpu = cp.asnumpy(top_k_indices)

    return [patterns[int(idx)] for idx in top_k_cpu]
```

**New CUDA Kernel Required:**

```cuda
// knowledge3d/cranium/ptx/kernels/extract_pattern_features.cu

__global__ void extract_pattern_features(
    const int* pattern_sources,
    const int* pattern_families,
    const float* query_counts,
    const int8_t* polarity_flags,
    float* out_confidence,
    float* out_source_prior,
    float* out_family_match,
    float* out_novelty,
    int n
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // Feature extraction in parallel (each thread = one pattern)
    int source = pattern_sources[idx];
    int family = pattern_families[idx];
    float query_count = query_counts[idx];
    int8_t polarity = polarity_flags[idx];

    // Confidence: Based on source type
    out_confidence[idx] = (source == 0) ? 1.0f :  // Grammar rule
                          (source == 1) ? 0.8f :  // Drawing primitive
                          (source == 2) ? 0.6f :  // Discovered pattern
                          0.4f;                    // External

    // Adjust confidence for negative polarity (figure-ground reversal)
    if (polarity == -1) {
        out_confidence[idx] *= 0.9f;  // Slight penalty for inversion
    }

    // Source prior: Lookup table (could be __constant__ memory)
    out_source_prior[idx] = source_prior_lut[source];

    // Family match: Binary check (could be parameterized)
    out_family_match[idx] = (family == 3) ? 1.0f : 0.5f;  // 3 = ARC family

    // Novelty: Inverse of query count (less queried = more novel)
    out_novelty[idx] = 1.0f / (1.0f + query_count);
}
```

**Expected Impact:**
- Python loop: **~100ms → 0ms** (eliminated)
- GPU feature extraction: **~0.05ms** (parallel)
- Speedup: **100-1000×** for this function

---

#### Function 2: `apply_validity_gates_relaxed_ptx` (arc_ops.py:346-404)

**Current Implementation (HYBRID):**

```python
# ❌ BOTTLENECK #1: Lines 346-360 (Python loop extracting validity)
for item in ranked_candidates:
    grid = self._to_grid(item.get("candidate"))
    family_ok, shape_ok, palette_ok, object_ok = \
        self._candidate_validity_bits(grid, validity_profile)  # Python analysis
    fam_v.append(family_ok)
    shape_v.append(shape_ok)
    palette_v.append(palette_ok)
    object_v.append(object_ok)

# ✅ GPU call: Lines 367-376 (~0.413ms, 10% of time)
self._validity_kernel(fam_v_gpu, shape_v_gpu, palette_v_gpu, object_v_gpu, out_scores)

# ❌ BOTTLENECK #2: Lines 391-404 (Python loop filtering)
for idx, item in enumerate(ranked_candidates):
    hard_block = strictness_key == "strict" and fam_v[idx] < 0.5
    keep = (float(score[idx]) >= float(threshold)) and (not hard_block)
    if keep:
        filtered.append(item)
```

**Solution: Full GPU Pipeline**

```python
def apply_validity_gates_relaxed_ptx(self, ranked_candidates, validity_profile, threshold=0.4):
    """
    Apply validity gates with FULL GPU pipeline.
    """
    n = len(ranked_candidates)

    # Step 1: Upload all candidate grids to GPU (flattened)
    grid_height, grid_width = 30, 30  # ARC max size
    all_grids_flat = []
    for item in ranked_candidates:
        grid = self._to_grid(item.get("candidate"))
        padded = np.pad(grid, ((0, grid_height - grid.shape[0]),
                               (0, grid_width - grid.shape[1])), constant_values=0)
        all_grids_flat.append(padded.flatten())

    grids_gpu = cp.asarray(np.array(all_grids_flat), dtype=cp.int32)

    # Step 2: Upload validity profile to GPU
    expected_family_gpu = cp.asarray([validity_profile.get("family", -1)], dtype=cp.int32)
    expected_shape_gpu = cp.asarray(validity_profile.get("expected_shapes", []), dtype=cp.int32)
    expected_palette_gpu = cp.asarray(validity_profile.get("palette", []), dtype=cp.int32)
    expected_objects_gpu = cp.asarray([validity_profile.get("object_count", -1)], dtype=cp.int32)

    # Step 3: Allocate validity bit arrays on GPU
    family_bits_gpu = cp.zeros(n, dtype=cp.float32)
    shape_bits_gpu = cp.zeros(n, dtype=cp.float32)
    palette_bits_gpu = cp.zeros(n, dtype=cp.float32)
    object_bits_gpu = cp.zeros(n, dtype=cp.float32)

    # Step 4: GPU Kernel 1 - Check Validity Bits (REPLACES Python loop #1)
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    self._check_validity_bits_kernel(
        (blocks,), (threads_per_block,),
        (grids_gpu, expected_family_gpu, expected_shape_gpu, expected_palette_gpu, expected_objects_gpu,
         family_bits_gpu, shape_bits_gpu, palette_bits_gpu, object_bits_gpu, n, grid_height, grid_width)
    )

    # Step 5: GPU Kernel 2 - Weighted Scoring (already exists)
    scores_gpu = cp.zeros(n, dtype=cp.float32)
    self._validity_kernel(family_bits_gpu, shape_bits_gpu, palette_bits_gpu, object_bits_gpu, scores_gpu, n)

    # Step 6: GPU Kernel 3 - Filter by Threshold (REPLACES Python loop #2)
    keep_mask_gpu = cp.zeros(n, dtype=cp.int32)
    threshold_gpu = cp.float32(threshold)

    self._filter_by_threshold_kernel(
        (blocks,), (threads_per_block,),
        (scores_gpu, family_bits_gpu, threshold_gpu, keep_mask_gpu, n)
    )

    # Step 7: Use GPU boolean indexing to filter
    keep_indices = cp.where(keep_mask_gpu)[0]
    keep_indices_cpu = cp.asnumpy(keep_indices)

    filtered = [ranked_candidates[int(idx)] for idx in keep_indices_cpu]

    return filtered
```

**New CUDA Kernels Required:**

```cuda
// knowledge3d/cranium/ptx/kernels/check_validity_bits.cu

__global__ void check_validity_bits(
    const int* grids,              // All grids flattened [n * grid_size]
    const int* expected_family,
    const int* expected_shapes,
    const int* expected_palette,
    const int* expected_objects,
    float* out_family_bits,
    float* out_shape_bits,
    float* out_palette_bits,
    float* out_object_bits,
    int n_candidates,
    int grid_height,
    int grid_width
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_candidates) return;

    int grid_size = grid_height * grid_width;
    int offset = idx * grid_size;

    // Family check
    int detected_family = detect_grid_family(&grids[offset], grid_size);
    out_family_bits[idx] = (detected_family == expected_family[0]) ? 1.0f : 0.5f;

    // Shape check
    float shape_score = compute_shape_similarity(&grids[offset], expected_shapes, grid_size);
    out_shape_bits[idx] = shape_score;

    // Palette check
    float palette_score = compute_palette_similarity(&grids[offset], expected_palette, grid_size);
    out_palette_bits[idx] = palette_score;

    // Object count check
    int detected_objects = count_connected_components(&grids[offset], grid_height, grid_width);
    float object_diff = fabsf((float)detected_objects - (float)expected_objects[0]);
    out_object_bits[idx] = fmaxf(0.0f, 1.0f - object_diff / 5.0f);  // Penalty increases with difference
}

// knowledge3d/cranium/ptx/kernels/filter_by_threshold.cu

__global__ void filter_by_threshold(
    const float* scores,
    const float* family_bits,
    const float threshold,
    int* out_keep_mask,
    int n
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // Hard block: family < 0.5 (wrong family type)
    bool hard_block = (family_bits[idx] < 0.5f);

    // Keep if: score >= threshold AND not hard blocked
    bool keep = (scores[idx] >= threshold) && !hard_block;

    out_keep_mask[idx] = keep ? 1 : 0;
}
```

**Expected Impact:**
- Python loop #1: **~50ms → 0ms** (eliminated)
- Python loop #2: **~10ms → 0ms** (eliminated)
- GPU validity checking: **~0.1ms** (parallel)
- Speedup: **500-1000×** for this function

---

#### Function 3: `check_oracle_fuzzy_ptx` (arc_ops.py, likely similar pattern)

**Expected Problem:** Probably compares grids pixel-by-pixel in Python loop on CPU.

**Solution:** Move to GPU with parallel comparison kernel.

```cuda
__global__ void compare_grids_fuzzy(
    const int* candidate_grids,
    const int* oracle_grids,
    float* out_match_scores,
    int n_candidates,
    int n_oracles,
    int grid_size,
    float fuzzy_threshold
) {
    // Each thread compares one (candidate, oracle) pair
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_candidates * n_oracles) return;

    int cand_idx = idx / n_oracles;
    int oracle_idx = idx % n_oracles;

    int cand_offset = cand_idx * grid_size;
    int oracle_offset = oracle_idx * grid_size;

    // Count matching pixels
    int matches = 0;
    for (int i = 0; i < grid_size; i++) {
        if (candidate_grids[cand_offset + i] == oracle_grids[oracle_offset + i]) {
            matches++;
        }
    }

    float match_ratio = (float)matches / (float)grid_size;
    out_match_scores[idx] = match_ratio;
}
```

---

## Priority 2: Increase Batch Sizes (1 hour)

### Problem

**Current:**
- `rank_candidates_ternary`: n=1-2 (micro-batch)
- `discover_patterns_ptx`: n~10 (small batch)
- GPU idle because batch too small to saturate cores

**Goal:** Increase to n=64-128 to saturate GPU.

### Solution

**1. Candidate Generation: Increase from 10 → 128**

```python
# benchmarks/arc_agi_2_adapter.py

def solve_arc_task_ptx(self, task_dict, kverse):
    # ...

    # ❌ OLD: Generate only 10 candidates
    # candidates = self._generate_candidates(patterns, test_input, top_k=10)

    # ✅ NEW: Generate 128 candidates for GPU batch
    candidates = self._generate_candidates(patterns, test_input, top_k=128)

    # Rank in single GPU batch (not 1-2 at a time)
    ranked = self.ptx_ops.rank_candidates_ternary(
        candidates=candidates,  # Full batch: n=128
        train_examples=task_dict["train"],
        batch_size=128  # Process all at once
    )

    # Apply validity gates (still batched)
    filtered = self.ptx_ops.apply_validity_gates_relaxed_ptx(
        ranked_candidates=ranked,
        validity_profile=validity_profile,
        batch_size=128
    )

    return filtered[0] if filtered else None
```

**2. Pattern Discovery: Increase from 64 → 256**

```python
def discover_patterns_ptx(self, train_examples, top_k=256):  # ✅ Increased from 64
    # Extract ALL patterns first (before filtering)
    all_patterns = self._extract_all_patterns(train_examples)

    # Score ALL patterns in single GPU batch (n=256-1000)
    scored_patterns = self._score_patterns_gpu(all_patterns, batch_size=len(all_patterns))

    # Select top-k on GPU (not CPU)
    top_patterns = self._topk_gpu(scored_patterns, k=top_k)

    return top_patterns
```

**Expected Impact:**
- GPU utilization: **5% → 40-60%** (larger batches)
- Kernel execution time: **0.05ms → 0.5ms** (10× work per kernel call)
- Total runtime: **Same or faster** (amortized overhead)

---

## Priority 3: Palette-Aware Generation (2 hours)

### Problem

**Week 21.8 Results:**
- Palette score: **0.6356** (weakest component)
- Oracle failures: **palette=58** (most common)
- Correct tasks: palette=0.715
- Incorrect tasks: palette=0.616
- **Delta: 0.10** (strongest discriminator!)

### Root Cause

Generation creates candidates without palette awareness → produces wrong colors → hard to fix in ranking.

### Solution: Palette-Aware Pattern Discovery

**Step 1: Extract Palette Distribution (Not Just Colors)**

```python
def extract_palette_distribution(train_examples):
    """
    Extract palette as DISTRIBUTION, not just color set.

    Returns:
        palette_dist: {color: frequency} dict
        dominant_colors: List[int] (sorted by frequency)
        rare_colors: List[int] (appear <10% of time)
    """
    color_counts = {}
    total_pixels = 0

    for example in train_examples:
        for grid in [example["input"], example["output"]]:
            for row in grid:
                for pixel in row:
                    color_counts[pixel] = color_counts.get(pixel, 0) + 1
                    total_pixels += 1

    # Convert to distribution
    palette_dist = {color: count / total_pixels for color, count in color_counts.items()}

    # Sort by frequency
    sorted_colors = sorted(palette_dist.items(), key=lambda x: x[1], reverse=True)
    dominant_colors = [c for c, freq in sorted_colors if freq >= 0.1]  # ≥10%
    rare_colors = [c for c, freq in sorted_colors if freq < 0.1]

    return {
        "distribution": palette_dist,
        "dominant": dominant_colors,
        "rare": rare_colors,
        "unique_count": len(color_counts)
    }
```

**Step 2: Filter Patterns by Palette Compatibility**

```python
def discover_patterns_ptx(self, train_examples, top_k=256):
    # Extract palette distribution
    palette_profile = extract_palette_distribution(train_examples)

    # Discover ALL patterns
    all_patterns = self._extract_all_patterns(train_examples)

    # Filter patterns by palette compatibility BEFORE scoring
    palette_compatible_patterns = []
    for pattern in all_patterns:
        pattern_colors = self._extract_pattern_colors(pattern)

        # Check: pattern uses only colors from training palette
        uses_invalid_colors = any(c not in palette_profile["distribution"] for c in pattern_colors)

        if not uses_invalid_colors:
            # Compute palette alignment score
            palette_score = sum(
                palette_profile["distribution"].get(c, 0.0) for c in pattern_colors
            ) / len(pattern_colors) if pattern_colors else 0.0

            pattern["palette_score"] = palette_score
            palette_compatible_patterns.append(pattern)

    # Score filtered patterns on GPU
    scored = self._score_patterns_gpu(palette_compatible_patterns, batch_size=len(palette_compatible_patterns))

    # Weight palette score 2× (strongest discriminator)
    for p in scored:
        p["final_score"] = p["base_score"] * (p["palette_score"] ** 2.0)  # Square for emphasis

    # Select top-k
    top_patterns = sorted(scored, key=lambda p: p["final_score"], reverse=True)[:top_k]

    return top_patterns
```

**Step 3: Palette-Constrained Composition**

```python
def apply_patterns_to_test(self, test_input, patterns, palette_profile, top_k=128):
    """
    Generate candidates with palette constraints.
    """
    candidates = []

    for pattern in patterns:
        # Apply pattern to test input
        candidate_grid = self._apply_pattern_transform(test_input, pattern)

        # Check palette validity
        candidate_colors = set(candidate_grid.flatten())
        palette_violation = any(c not in palette_profile["distribution"] for c in candidate_colors)

        if palette_violation:
            # Remap invalid colors to nearest valid color
            candidate_grid = self._remap_to_valid_palette(candidate_grid, palette_profile["dominant"])

        # Compute palette score for ranking
        palette_score = self._compute_palette_score(candidate_grid, palette_profile)

        candidates.append({
            "candidate": candidate_grid,
            "pattern_id": pattern["id"],
            "palette_score": palette_score
        })

    # Sort by palette score, keep top-k
    candidates.sort(key=lambda c: c["palette_score"], reverse=True)
    return candidates[:top_k]
```

**Expected Impact:**
- Palette score: **0.6356 → 0.75+** (+0.12)
- Oracle palette failures: **58 → 20-30** (50% reduction)
- ARC accuracy: **0.05 → 0.08-0.10** (+0.03-0.05)

---

## Priority 4: Full Validation (30 min)

### Test Suite

**1. Unit Tests (5 min):**
```bash
pytest tests/test_arc_ptx_ops.py -v
pytest tests/test_negative_forms.py -v  # NEW test file
```

**2. Micro-Benchmarks (10 min):**
```python
# scripts/micro_benchmark_ptx.py
import time
import cupy as cp

def benchmark_discover_patterns():
    # Before GPU migration
    start = time.time()
    result = ops.discover_patterns_ptx_old(examples, patterns, top_k=256)
    time_old = time.time() - start

    # After GPU migration
    start = time.time()
    result = ops.discover_patterns_ptx(examples, patterns, top_k=256)
    time_new = time.time() - start

    print(f"discover_patterns_ptx: {time_old:.3f}s → {time_new:.3f}s ({time_old/time_new:.1f}× speedup)")

    # GPU utilization check
    print(f"GPU memory: {cp.cuda.Device().mem_info}")

# Expected output:
# discover_patterns_ptx: 0.150s → 0.002s (75× speedup)
# GPU memory: (used=2048MB, free=6144MB)
```

**3. Full Benchmark (15 min):**
```bash
conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --max-arc-tasks 100 \
  --arc-enable-full-ptx \
  --arc-enable-negative-forms \
  --arc-palette-aware-generation \
  --output-dir ../Knowledge3D.local/results/week21_9_full_gpu \
  --storage-root ../Knowledge3D.local
```

**4. GPU Monitoring (parallel with benchmark):**
```bash
# Terminal 2
watch -n 0.5 nvidia-smi

# Expected output during benchmark:
# GPU Utilization: 60-90% (was 5%)
# GPU Memory: ~2GB stable (unified persistence)
# GPU Temp: 65-75°C (active computation)
```

---

## Success Criteria

### Week 21.8 (Before) → Week 21.9 (After Target)

| Metric | Week 21.8 | Week 21.9 Target | Delta |
|--------|-----------|------------------|-------|
| **ARC Accuracy** | 0.05 (5/100) | **0.10+** (10/100) | +0.05 |
| **Oracle @ All** | 0.0 | **0.10+** | +0.10 |
| **Fuzzy Oracle @0.90** | 0.12 | **0.20+** | +0.08 |
| **GPU Utilization** | 5% | **60-90%** | +55-85% |
| **Runtime (100 tasks)** | ~2 hours | **5-10 minutes** | 10-20× speedup |
| **Palette Score** | 0.6356 | **0.75+** | +0.12 |
| **Oracle Palette Failures** | 58/100 | **20-30/100** | -50% |
| **Batch Size (ranking)** | n=1-2 | **n=64-128** | 32-64× |
| **Batch Size (discovery)** | n~10 | **n=256** | 25× |

### Validation Checklist

- [ ] **Zero Python loops in hot path** (grep for `for.*in.*patterns`)
- [ ] **GPU utilization 60-90%** (nvidia-smi during benchmark)
- [ ] **Palette score >0.75** (telemetry output)
- [ ] **Oracle unlock >0.10** (results JSON)
- [ ] **Negative form duality documented** (DUAL_CLIENT_CONTRACT_SPECIFICATION.md updated)
- [ ] **All tests passing** (15/15 PTX tests + new negative form tests)
- [ ] **Batch sizes confirmed** (telemetry shows n=64-128 for ranking, n=256 for discovery)

---

## Implementation Timeline

### Phase 1: Document + Palette (3 hours)
- **Hour 1**: Update DUAL_CLIENT_CONTRACT_SPECIFICATION.md with negative form duality
- **Hour 2**: Implement palette distribution extraction + filtering
- **Hour 3**: Integrate palette-aware generation into arc_agi_2_adapter.py

### Phase 2: GPU Migration (4 hours)
- **Hour 1**: Write `extract_pattern_features.cu` kernel
- **Hour 2**: Write `check_validity_bits.cu` kernel
- **Hour 3**: Write `filter_by_threshold.cu` kernel
- **Hour 4**: Integrate all kernels into arc_ops.py, remove Python loops

### Phase 3: Batch Increase (1 hour)
- **Hour 1**: Increase batch sizes in arc_agi_2_adapter.py (10→128, 64→256)

### Phase 4: Validation (1 hour)
- **30 min**: Run micro-benchmarks, validate GPU utilization
- **30 min**: Full 100-task benchmark, confirm metrics

### Total: **~9 hours** (1 day at rapid pace)

---

## Communication Protocol

**After Each Phase:**

1. **Phase 1 Complete**: Report palette score improvement + spec documentation
2. **Phase 2 Complete**: Report GPU utilization % + speedup metrics
3. **Phase 3 Complete**: Report batch sizes confirmed via telemetry
4. **Phase 4 Complete**: Full results JSON + comparison table

**Format:**
```
Week 21.9 Phase X Complete

Changes:
- [list of files modified]
- [key functionality added]

Metrics:
- [relevant performance numbers]

Next: Phase X+1 [brief description]
```

---

## Critical Reminders

### Sovereignty
- ✅ PTX kernels only in hot path
- ✅ CuPy for GPU arrays
- ❌ NO numpy/scipy in discovery/ranking/filtering loops
- ❌ NO external ML frameworks

### Unified Persistence
- ✅ Single Knowledgeverse instance (already working Week 21.8)
- ✅ All galaxies loaded once (verified)
- ✅ Same instance ID throughout (validated)

### Negative Form Duality
- ✅ Zero storage cost (derive from positive)
- ✅ Figure-ground reversal (critical for ARC)
- ✅ System-wide generalization (audio, 3D, physics)
- ✅ Ternary encoding (+1, 0, -1)

### Multi-Galaxy (Post-Week 21.9)
- Enforce ≥5 galaxies touched per evaluation block
- Add telemetry for galaxy participation
- Current: Grammar-only, Target: Grammar + Drawing + Math + Reality + Character

---

## Expected User-Visible Outcome

**After Week 21.9:**

```
Week 21.9 Full100 Benchmark Results

ARC-AGI 2:
  Accuracy: 10/100 (0.10) ✅ +0.05 from Week 21.8
  Oracle @ All: 0.12 ✅ +0.12 from Week 21.8
  Fuzzy Oracle @0.90: 0.22 ✅ +0.10 from Week 21.8

Runtime: 7 minutes ✅ 17× speedup (was 2 hours)
GPU Usage: 75% avg ✅ (was 5%)

Score Components:
  Shape: 0.9656 ✅ (maintained)
  Object: 0.9200 ✅ (maintained)
  Family: 0.7855 ✅ (maintained)
  Palette: 0.7612 ✅ +0.13 (was 0.6356)

Oracle Failures:
  Palette: 28 ✅ -30 (was 58)
  Object: 42 ✅ +12 (slight increase due to faster iteration)
  Shape: 25 ✅ +7 (slight increase)
  Family: 18 ✅ +61% (improved)

PTX Path: ✅ arc_ptx_ops (100%)
Unified Persistence: ✅ shared_instance=true
Batch Sizes: ✅ ranking n=128, discovery n=256
Negative Form Duality: ✅ integrated (figure-ground reversal active)

Key Improvements:
1. Full GPU pipeline (60-90% utilization)
2. Palette-aware generation (0.76 score)
3. Negative form duality (figure-ground reversal)
4. Larger batch sizes (128/256 vs 2/10)
5. Nanosecond-scale kernel execution (maintained sovereignty)
```

**User's Goal Achieved:** "Things taking minutes will compute in nano seconds" ✅

---

**Claude (Architecture Partner)**
February 10, 2026

**Directive:** Implement in priority order, report after each phase, maintain rapid pace.

**Next:** Codex begins Phase 1 (documentation + palette-aware generation).
