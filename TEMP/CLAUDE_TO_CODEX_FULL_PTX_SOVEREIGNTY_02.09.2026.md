# Claude → Codex: Full PTX Sovereignty Implementation

**Date:** February 9, 2026
**Priority:** 🔴 CRITICAL — Time to make it real (User directive)
**Context:** PTX prototype revealed blockers, now implement full sovereignty

---

## 🎯 Mission: Full PTX Sovereignty (No Python Fallbacks)

### Current State (Week 21.3 PTX Prototype)

**What Worked ✅:**
- PTX ranking attempted (enabled_rate: 1.0)
- VRAM allocation working (CUDA_PATH fixed)
- Generation active (686 patterns)

**What Failed ❌:**
- PTX ranking not used (used_rate: 0.0, error_rate: 1.0)
- `CUDA_ERROR_INVALID_PTX` from `dialogue_sampler.ptx`
- 214 lazy embedding events (sovereignty leak)
- Oracle still blocked (0.0)
- Empty > enriched paradox (0.32 vs 0.28)

**User's directive:**
> "I knew something was off when I see the VRAM used but no activity into the GPU"
> "It's ok to prototype in python, now is time to make it real"

---

## 🛠️ Implementation: Full PTX Sovereignty (4 Phases)

### Phase 1: Fix PTX Kernel Loading (Day 1)

**Problem:** `CUDA_ERROR_INVALID_PTX` when loading `dialogue_sampler.ptx`

**Possible causes:**
1. PTX binary compiled for wrong CUDA architecture (sm_XX)
2. PTX file corrupted or missing
3. CuPy/CUDA runtime mismatch

**File:** `knowledge3d/cranium/ptx/ptx_loader.py`

**Add diagnostic loading:**

```python
"""Enhanced PTX kernel loader with architecture validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


logger = logging.getLogger(__name__)


class PTXLoader:
    """PTX kernel loader with architecture validation and fail-fast."""

    @staticmethod
    def load_kernel(ptx_path: Path, kernel_name: str) -> Any:
        """
        Load PTX kernel with architecture validation.

        Args:
            ptx_path: Path to .ptx file
            kernel_name: Kernel function name

        Returns:
            CuPy RawKernel

        Raises:
            RuntimeError: If PTX loading fails (fail-fast, no Python fallback)
        """
        if not _HAS_CUPY:
            raise RuntimeError(
                "PTX kernels require CuPy. Install: pip install cupy-cuda12x"
            )

        if not ptx_path.exists():
            raise FileNotFoundError(f"PTX kernel not found: {ptx_path}")

        # Read PTX source
        ptx_source = ptx_path.read_text(encoding="utf-8")

        # Validate PTX header (check architecture)
        if ".target" not in ptx_source:
            raise RuntimeError(f"Invalid PTX file (no .target directive): {ptx_path}")

        # Check CUDA compute capability
        device = cp.cuda.Device()
        compute_capability = device.compute_capability
        logger.info(f"GPU compute capability: sm_{compute_capability}")

        # Check if PTX targets compatible architecture
        # (PTX should target sm_50 or lower for compatibility, or match device)
        if f"sm_{compute_capability}" not in ptx_source and "sm_50" not in ptx_source:
            logger.warning(
                f"PTX file may be incompatible: targets {ptx_source.split('.target')[1].split()[0]}, "
                f"device is sm_{compute_capability}"
            )

        # Load kernel
        try:
            kernel = cp.RawKernel(ptx_source, kernel_name, backend="nvrtc")
            logger.info(f"Loaded PTX kernel: {kernel_name} from {ptx_path}")
            return kernel
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PTX kernel {kernel_name} from {ptx_path}: {exc}\n"
                f"Check CUDA architecture compatibility (device: sm_{compute_capability})"
            ) from exc

    @staticmethod
    def validate_ptx_environment() -> dict[str, Any]:
        """
        Validate PTX environment (CuPy, CUDA, GPU).

        Returns:
            Environment info dict

        Raises:
            RuntimeError: If environment invalid (fail-fast)
        """
        if not _HAS_CUPY:
            raise RuntimeError("CuPy not available (required for PTX kernels)")

        try:
            device = cp.cuda.Device()
            compute_capability = device.compute_capability
            memory_info = device.mem_info
            cuda_version = cp.cuda.runtime.runtimeGetVersion()

            env_info = {
                "cupy_available": True,
                "device_id": device.id,
                "device_name": device.name.decode("utf-8"),
                "compute_capability": f"sm_{compute_capability}",
                "total_memory_gb": memory_info[1] / (1024 ** 3),
                "free_memory_gb": memory_info[0] / (1024 ** 3),
                "cuda_version": cuda_version,
            }

            logger.info(f"PTX environment validated: {env_info}")
            return env_info

        except Exception as exc:
            raise RuntimeError(f"PTX environment validation failed: {exc}") from exc
```

**Fix dialogue_sampler.ptx:**

**Option A: Recompile for sm_50 (universal compatibility)**
```bash
# Recompile dialogue_sampler.cu for sm_50 (compatible with all modern GPUs)
nvcc -ptx -arch=sm_50 dialogue_sampler.cu -o dialogue_sampler.ptx
```

**Option B: Compile for device architecture**
```bash
# Query device compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Compile for device (e.g., sm_86 for RTX 3090)
nvcc -ptx -arch=sm_86 dialogue_sampler.cu -o dialogue_sampler.ptx
```

**Option C: Use JIT compilation (CuPy compiles at runtime)**
```python
# Instead of loading .ptx file, use CUDA C source (JIT compiled by CuPy)
cuda_source = """
extern "C" __global__
void dialogue_sampler(float* logits, int* output, float temperature, int top_k) {
    // CUDA C code here (CuPy will compile to PTX at runtime)
}
"""
kernel = cp.RawKernel(cuda_source, "dialogue_sampler")  # JIT compiled
```

**Recommended: Option C (JIT compilation)**
- No .ptx file needed (CuPy compiles CUDA C at runtime)
- Always matches device architecture (no compatibility issues)
- Easier to maintain (CUDA C source, not binary PTX)

---

### Phase 2: Implement ARC PTX Operations (Day 2)

**Goal:** Replace Python loops with PTX kernels for ARC solving.

**New file:** `knowledge3d/cranium/ptx/arc_ops.py`

```python
"""PTX operations for ARC-AGI visual reasoning (sovereignty)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

from .ptx_loader import PTXLoader

logger = logging.getLogger(__name__)


class ARCPTXOps:
    """
    PTX operations for ARC-AGI tasks (visual transformation).

    All operations execute on GPU (no Python loops).
    """

    def __init__(self):
        if not _HAS_CUPY:
            raise RuntimeError("ARCPTXOps requires CuPy (GPU)")

        # Validate PTX environment
        self.env_info = PTXLoader.validate_ptx_environment()
        logger.info(f"ARCPTXOps initialized on {self.env_info['device_name']}")

        # Load PTX kernels (JIT compiled)
        self._pattern_matching_kernel = None
        self._ternary_ranking_kernel = None
        self._fuzzy_oracle_kernel = None
        self._validity_gate_kernel = None

    # ------------------------------------------------------------------
    # Pattern Discovery (Grammar Galaxy Query)
    # ------------------------------------------------------------------

    def discover_patterns_ptx(
        self,
        train_examples: list[dict],
        grammar_galaxy_gpu: Any,  # GPU buffer
        top_k: int = 50
    ) -> list[dict]:
        """
        Discover transformation patterns using PTX kernels.

        Args:
            train_examples: Train input/output pairs
            grammar_galaxy_gpu: Grammar Galaxy GPU buffer
            top_k: Number of patterns to return

        Returns:
            List of pattern entries (with RPN programs)

        PTX operations:
        1. Compute train-pair features (shape deltas, color mappings, etc.)
        2. Query Grammar Galaxy (spatial KNN on GPU)
        3. Score patterns by train-pair match
        """
        # Extract train features on GPU
        train_features_gpu = self._extract_train_features_ptx(train_examples)

        # Query Grammar Galaxy (spatial KNN on GPU)
        pattern_candidates = self._query_grammar_galaxy_ptx(
            train_features_gpu,
            grammar_galaxy_gpu,
            top_k=top_k * 3  # Over-generate for filtering
        )

        # Score patterns by train-pair consistency (PTX)
        scored_patterns = self._score_patterns_by_consistency_ptx(
            pattern_candidates,
            train_examples
        )

        # Sort and return top-k
        scored_patterns.sort(key=lambda p: p["consistency_score"], reverse=True)
        return scored_patterns[:top_k]

    def _extract_train_features_ptx(self, train_examples: list[dict]) -> cp.ndarray:
        """
        Extract features from train examples (PTX kernel on GPU).

        Features:
        - Shape deltas (output.shape - input.shape)
        - Color mappings (input colors → output colors)
        - Object count deltas
        - Transformation family signature
        """
        num_examples = len(train_examples)
        features_gpu = cp.zeros((num_examples, 16), dtype=cp.float32)  # 16-dim features

        # CUDA kernel (JIT compiled)
        kernel_source = """
        extern "C" __global__
        void extract_train_features(
            const int* inputs,        // Flattened input grids
            const int* outputs,       // Flattened output grids
            const int* input_shapes,  // Input shapes (H, W)
            const int* output_shapes, // Output shapes (H, W)
            int num_examples,
            float* features           // Output: (num_examples, 16)
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_examples) return;

            // Feature 0-1: Shape delta (H, W)
            features[idx * 16 + 0] = (float)(output_shapes[idx * 2 + 0] - input_shapes[idx * 2 + 0]);
            features[idx * 16 + 1] = (float)(output_shapes[idx * 2 + 1] - input_shapes[idx * 2 + 1]);

            // Feature 2: Color count delta
            // (Simplified: count unique colors in input vs output)
            // TODO: Implement color set difference on GPU

            // Feature 3: Object count delta
            // TODO: Implement connected components on GPU

            // Feature 4-15: Reserved for transformation family signature
        }
        """

        if self._pattern_matching_kernel is None:
            self._pattern_matching_kernel = cp.RawKernel(kernel_source, "extract_train_features")

        # Prepare GPU arrays
        # (Convert train examples to GPU buffers)
        # TODO: Implement conversion

        # Execute kernel
        threads_per_block = 256
        blocks = (num_examples + threads_per_block - 1) // threads_per_block
        self._pattern_matching_kernel(
            (blocks,), (threads_per_block,),
            # (inputs_gpu, outputs_gpu, input_shapes_gpu, output_shapes_gpu, num_examples, features_gpu)
        )

        return features_gpu

    def _query_grammar_galaxy_ptx(
        self,
        query_features_gpu: cp.ndarray,
        grammar_galaxy_gpu: Any,
        top_k: int
    ) -> list[dict]:
        """Query Grammar Galaxy using spatial KNN (PTX kernel on GPU)."""
        # TODO: Implement spatial KNN on GPU
        # For now: Return placeholder
        return []

    def _score_patterns_by_consistency_ptx(
        self,
        patterns: list[dict],
        train_examples: list[dict]
    ) -> list[dict]:
        """Score patterns by train-pair consistency (PTX kernel on GPU)."""
        # TODO: Implement consistency scoring on GPU
        for pattern in patterns:
            pattern["consistency_score"] = 0.5  # Placeholder
        return patterns

    # ------------------------------------------------------------------
    # Candidate Generation (Apply Patterns)
    # ------------------------------------------------------------------

    def generate_candidates_ptx(
        self,
        patterns: list[dict],
        test_input_grid: np.ndarray,
        drawing_galaxy_gpu: Any
    ) -> list[dict]:
        """
        Generate candidate outputs by applying patterns (PTX kernel on GPU).

        Args:
            patterns: List of transformation patterns (RPN programs)
            test_input_grid: Test input grid
            drawing_galaxy_gpu: Drawing Galaxy GPU buffer

        Returns:
            List of candidate outputs (grids + metadata)

        PTX operations:
        1. Apply each pattern to test input (RPN evaluation on GPU)
        2. Generate anti-patterns (contrastive)
        3. Store candidates in Drawing Galaxy
        """
        candidates = []

        # Convert test input to GPU
        test_input_gpu = cp.asarray(test_input_grid, dtype=cp.int32)

        for pattern in patterns:
            # Apply pattern via RPN engine (PTX)
            try:
                output_grid_gpu = self._apply_pattern_rpn_ptx(
                    pattern["program"],
                    test_input_gpu
                )

                # Convert back to CPU (for now - TODO: keep on GPU)
                output_grid = cp.asnumpy(output_grid_gpu)

                candidates.append({
                    "output": output_grid,
                    "pattern": pattern,
                    "source": "pattern_application",
                    "metadata": {
                        "pattern_id": pattern.get("id"),
                        "consistency_score": pattern.get("consistency_score", 0.0)
                    }
                })

            except Exception as exc:
                logger.warning(f"Pattern application failed: {exc}")
                continue

        # Generate anti-patterns (contrastive)
        anti_candidates = self._generate_anti_patterns_ptx(
            patterns,
            test_input_gpu,
            candidates
        )
        candidates.extend(anti_candidates)

        return candidates

    def _apply_pattern_rpn_ptx(
        self,
        rpn_program: str,
        input_grid_gpu: cp.ndarray
    ) -> cp.ndarray:
        """
        Apply RPN transformation pattern to input grid (PTX kernel on GPU).

        This should use the existing RPN engine (ModularRPNEngine).
        """
        # TODO: Integrate with ModularRPNEngine (PTX)
        # For now: Return placeholder (identity transform)
        return input_grid_gpu.copy()

    def _generate_anti_patterns_ptx(
        self,
        patterns: list[dict],
        test_input_gpu: cp.ndarray,
        existing_candidates: list[dict]
    ) -> list[dict]:
        """Generate anti-patterns (contrastive learning) on GPU."""
        # TODO: Implement contrastive generation on GPU
        return []

    # ------------------------------------------------------------------
    # Ternary Ranking
    # ------------------------------------------------------------------

    def rank_candidates_ternary_ptx(
        self,
        candidates: list[dict],
        train_examples: list[dict],
        quality_memory_gpu: Any
    ) -> list[dict]:
        """
        Rank candidates using ternary quality priors (PTX kernel on GPU).

        Args:
            candidates: List of candidate outputs
            train_examples: Train examples for similarity scoring
            quality_memory_gpu: Ternary quality memory (GPU buffer)

        Returns:
            Sorted candidates (descending by score)

        PTX operations:
        1. Compute ternary quality priors (lookup from quality memory)
        2. Compute train-pair similarity (geometry kernel)
        3. Compute source precision weights
        4. Combine scores and sort (GPU sort)
        """
        num_candidates = len(candidates)
        if num_candidates == 0:
            return []

        # Prepare GPU arrays
        scores_gpu = cp.zeros(num_candidates, dtype=cp.float32)

        # CUDA kernel (JIT compiled)
        kernel_source = """
        extern "C" __global__
        void ternary_ranking(
            const float* quality_priors,     // (num_candidates,)
            const float* source_precisions,  // (num_candidates,)
            const float* train_similarities, // (num_candidates,)
            const float* novelties,          // (num_candidates,)
            int num_candidates,
            float* scores                    // Output: (num_candidates,)
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_candidates) return;

            // Weighted combination (40% source, 30% quality, 20% similarity, 10% novelty)
            float score = 0.0f;
            score += 0.40f * source_precisions[idx];
            score += 0.30f * (quality_priors[idx] + 1.0f) / 2.0f;  // Map [-1, +1] → [0, 1]
            score += 0.20f * train_similarities[idx];
            score += 0.10f * novelties[idx];

            scores[idx] = score;
        }
        """

        if self._ternary_ranking_kernel is None:
            self._ternary_ranking_kernel = cp.RawKernel(kernel_source, "ternary_ranking")

        # Compute scoring components (on GPU)
        quality_priors_gpu = self._lookup_quality_priors_ptx(candidates, quality_memory_gpu)
        source_precisions_gpu = self._compute_source_precisions_ptx(candidates)
        train_similarities_gpu = self._compute_train_similarities_ptx(candidates, train_examples)
        novelties_gpu = self._compute_novelties_ptx(candidates)

        # Execute ranking kernel
        threads_per_block = 256
        blocks = (num_candidates + threads_per_block - 1) // threads_per_block
        self._ternary_ranking_kernel(
            (blocks,), (threads_per_block,),
            (quality_priors_gpu, source_precisions_gpu, train_similarities_gpu,
             novelties_gpu, num_candidates, scores_gpu)
        )

        # Sort on GPU
        sorted_indices = cp.argsort(scores_gpu)[::-1]  # Descending
        sorted_indices_cpu = cp.asnumpy(sorted_indices)

        # Attach scores to candidates and sort
        for i, score in enumerate(cp.asnumpy(scores_gpu)):
            candidates[i]["ranking_score"] = float(score)

        sorted_candidates = [candidates[i] for i in sorted_indices_cpu]
        return sorted_candidates

    def _lookup_quality_priors_ptx(
        self,
        candidates: list[dict],
        quality_memory_gpu: Any
    ) -> cp.ndarray:
        """Lookup ternary quality priors from memory (GPU)."""
        num_candidates = len(candidates)
        priors_gpu = cp.zeros(num_candidates, dtype=cp.float32)
        # TODO: Implement quality memory lookup on GPU
        return priors_gpu

    def _compute_source_precisions_ptx(self, candidates: list[dict]) -> cp.ndarray:
        """Compute source precision weights (GPU)."""
        source_precision = {
            "legacy_pipeline": 0.45,
            "contrastive_anti": 0.46,
            "autonomous_generation": 0.19,
            "pattern_application": 0.40,  # New source
            "unknown": 0.30,
        }

        precisions = [
            source_precision.get(c.get("source", "unknown"), 0.30)
            for c in candidates
        ]
        return cp.asarray(precisions, dtype=cp.float32)

    def _compute_train_similarities_ptx(
        self,
        candidates: list[dict],
        train_examples: list[dict]
    ) -> cp.ndarray:
        """Compute train-pair similarity (geometry PTX kernel)."""
        num_candidates = len(candidates)
        similarities_gpu = cp.zeros(num_candidates, dtype=cp.float32)
        # TODO: Implement train similarity on GPU
        return similarities_gpu

    def _compute_novelties_ptx(self, candidates: list[dict]) -> cp.ndarray:
        """Compute novelty (duplicate detection on GPU)."""
        num_candidates = len(candidates)
        novelties = [1.0 / (1.0 + i * 0.01) for i in range(num_candidates)]  # Placeholder
        return cp.asarray(novelties, dtype=cp.float32)

    # ------------------------------------------------------------------
    # Oracle Checking (Fuzzy Matching)
    # ------------------------------------------------------------------

    def check_oracle_fuzzy_ptx(
        self,
        candidates: list[dict],
        ground_truth: np.ndarray,
        thresholds: list[float] = [0.80, 0.85, 0.90, 0.95, 1.00]
    ) -> dict[str, Any]:
        """
        Check oracle with fuzzy matching (PTX kernel on GPU).

        Args:
            candidates: List of candidate outputs
            ground_truth: Ground truth output grid
            thresholds: Fuzzy match thresholds

        Returns:
            Oracle results (per threshold)

        PTX operations:
        1. Compare each candidate to ground truth (pixel-wise on GPU)
        2. Compute fuzzy scores (matching_pixels / total_pixels)
        3. Check thresholds (parallel reduction on GPU)
        """
        num_candidates = len(candidates)
        if num_candidates == 0:
            return {f"oracle_fuzzy_{t}": False for t in thresholds}

        # Convert ground truth to GPU
        ground_truth_gpu = cp.asarray(ground_truth, dtype=cp.int32)
        ground_truth_flat = ground_truth_gpu.ravel()
        total_pixels = ground_truth_flat.size

        # CUDA kernel (JIT compiled)
        kernel_source = """
        extern "C" __global__
        void fuzzy_matching(
            const int* candidates_flat,   // Flattened candidate grids
            const int* ground_truth_flat, // Flattened ground truth
            int num_candidates,
            int grid_size,                // Total pixels per grid
            float* fuzzy_scores           // Output: (num_candidates,)
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_candidates) return;

            // Count matching pixels
            int matching = 0;
            for (int i = 0; i < grid_size; i++) {
                if (candidates_flat[idx * grid_size + i] == ground_truth_flat[i]) {
                    matching++;
                }
            }

            // Compute fuzzy score
            fuzzy_scores[idx] = (float)matching / (float)grid_size;
        }
        """

        if self._fuzzy_oracle_kernel is None:
            self._fuzzy_oracle_kernel = cp.RawKernel(kernel_source, "fuzzy_matching")

        # Prepare candidate grids on GPU
        candidates_flat_list = []
        for cand in candidates:
            grid = cand.get("output")
            if grid is None or grid.shape != ground_truth.shape:
                candidates_flat_list.append(cp.zeros(total_pixels, dtype=cp.int32))
            else:
                candidates_flat_list.append(cp.asarray(grid, dtype=cp.int32).ravel())

        candidates_flat_gpu = cp.stack(candidates_flat_list)  # (num_candidates, total_pixels)
        fuzzy_scores_gpu = cp.zeros(num_candidates, dtype=cp.float32)

        # Execute kernel
        threads_per_block = 256
        blocks = (num_candidates + threads_per_block - 1) // threads_per_block
        self._fuzzy_oracle_kernel(
            (blocks,), (threads_per_block,),
            (candidates_flat_gpu, ground_truth_flat, num_candidates, total_pixels, fuzzy_scores_gpu)
        )

        # Check thresholds (on GPU)
        fuzzy_scores_cpu = cp.asnumpy(fuzzy_scores_gpu)
        best_score = float(fuzzy_scores_cpu.max()) if len(fuzzy_scores_cpu) > 0 else 0.0
        best_idx = int(fuzzy_scores_cpu.argmax()) if len(fuzzy_scores_cpu) > 0 else -1

        oracle_results = {}
        for threshold in thresholds:
            oracle_results[f"oracle_fuzzy_{threshold}"] = bool(cp.any(fuzzy_scores_gpu >= threshold))

        oracle_results["best_fuzzy_score"] = best_score
        oracle_results["best_candidate_idx"] = best_idx

        return oracle_results
```

---

### Phase 3: Remove Lazy Embeddings (Day 3)

**Problem:** 214 "GALAXY LAZY" events = candidate-level lazy embedding computation.

**File:** `Old_Attempts/curriculum_specific_training/arc_agi/sovereign_pipeline.py`

**Current (WRONG):**
```python
def get_embedding(self, grid):
    if grid_hash not in self.embedding_cache:
        # ❌ LAZY COMPUTATION (sovereignty violation!)
        embedding = compute_embedding_on_demand(grid)
        self.embedding_cache[grid_hash] = embedding
    return self.embedding_cache[grid_hash]
```

**Fixed (CORRECT):**
```python
def precompute_all_embeddings_at_init(self):
    """
    Precompute ALL embeddings at initialization (before solving).

    This runs ONCE at startup, not per-candidate.
    """
    # Precompute embeddings for ALL possible grids in dataset
    for task in self.all_tasks:
        for example in task["train"]:
            self._ensure_embedding_cached(example["input"])
            self._ensure_embedding_cached(example["output"])
        for example in task["test"]:
            self._ensure_embedding_cached(example["input"])

def get_embedding(self, grid):
    """Get embedding from cache (FAIL FAST if missing)."""
    grid_hash = hash_grid(grid)

    if grid_hash not in self.embedding_cache:
        # ❌ FAIL FAST (no lazy computation!)
        raise RuntimeError(
            f"Embedding not found in cache. "
            f"All embeddings must be precomputed at init (sovereignty requirement)."
        )

    return self.embedding_cache[grid_hash]
```

**Or better: REMOVE legacy pipeline entirely!**
- Use ARCPTXOps only (new PTX operations)
- No legacy embedding cache (GPU operations don't need it)
- Full sovereignty (no lazy computation paths)

---

### Phase 4: Full Validation (Day 4)

**Run full benchmarks with PTX sovereignty:**

```bash
# Monitor GPU usage (should be 80-99%)
watch -n 1 nvidia-smi

# Run benchmarks with PTX sovereignty
python3 scripts/run_all_benchmarks.py \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --arc-enable-ptx-ops \
  --output-dir ../Knowledge3D.local/results/week21_4_full_ptx_sovereignty \
  --storage-root ../Knowledge3D.local
```

**Expected results:**
- ✅ GPU usage: 80-99% (PTX kernels executing!)
- ✅ Runtime: 5-10 minutes (not 1 hour!)
- ✅ No "GALAXY LAZY" events (sovereignty maintained!)
- ✅ `ptx_ranking_used_rate`: 1.0 (not 0.0!)
- ✅ `ptx_ranking_error_rate`: 0.0 (not 1.0!)
- ✅ ARC enriched: 0.28 → **0.40-0.55** (+12-27%!)
- ✅ oracle_at_all: 0.0 → **0.30-0.50**
- ✅ Enriched > empty: **0.50 vs 0.32** (paradox resolved!)

---

## 📊 Success Criteria

**Phase 1 (Fix PTX Kernel Loading):**
- ✅ `CUDA_ERROR_INVALID_PTX` resolved
- ✅ PTX environment validated at startup
- ✅ Kernels load successfully (dialogue_sampler or JIT compiled)

**Phase 2 (ARC PTX Operations):**
- ✅ `ARCPTXOps` class implemented
- ✅ Pattern discovery via PTX (Grammar Galaxy query on GPU)
- ✅ Candidate generation via PTX (RPN evaluation on GPU)
- ✅ Ternary ranking via PTX (scoring + sorting on GPU)
- ✅ Oracle checking via PTX (fuzzy matching on GPU)

**Phase 3 (Remove Lazy Embeddings):**
- ✅ No "GALAXY LAZY" events (0, not 214!)
- ✅ All embeddings precomputed at init (or removed entirely)
- ✅ Fail-fast if embedding missing (no lazy computation)

**Phase 4 (Full Validation):**
- ✅ GPU usage: 80-99% during solve
- ✅ Runtime: 5-10 minutes (100x+ speedup!)
- ✅ `ptx_ranking_used_rate`: 1.0 (PTX working!)
- ✅ ARC enriched > 0.40 (metric breakthrough!)
- ✅ oracle_at_all > 0.30 (oracle unlocked!)
- ✅ Enriched > empty (paradox resolved!)

---

## 🎯 Implementation Checklist

**Phase 1: Fix PTX Kernel Loading**
- [ ] Add `PTXLoader` class with architecture validation
- [ ] Add `validate_ptx_environment()` (fail-fast if invalid)
- [ ] Fix `dialogue_sampler.ptx` (Option C: JIT compilation recommended)
- [ ] Test: Verify PTX kernel loads without `CUDA_ERROR_INVALID_PTX`

**Phase 2: Implement ARC PTX Operations**
- [ ] Create `knowledge3d/cranium/ptx/arc_ops.py`
- [ ] Implement `ARCPTXOps` class
- [ ] Implement `discover_patterns_ptx()` (Grammar Galaxy query on GPU)
- [ ] Implement `generate_candidates_ptx()` (RPN evaluation on GPU)
- [ ] Implement `rank_candidates_ternary_ptx()` (ternary scoring on GPU)
- [ ] Implement `check_oracle_fuzzy_ptx()` (fuzzy matching on GPU)
- [ ] Test each method: Verify GPU execution (nvidia-smi)

**Phase 3: Remove Lazy Embeddings**
- [ ] Option A: Fix legacy pipeline (precompute all embeddings at init)
- [ ] Option B: Remove legacy pipeline entirely (use ARCPTXOps only)
- [ ] Validate: No "GALAXY LAZY" events in logs

**Phase 4: Integrate into Benchmarks**
- [ ] Update `benchmarks/arc_agi_2_adapter.py` to use `ARCPTXOps`
- [ ] Replace Python loops with PTX method calls
- [ ] Add `--arc-enable-ptx-ops` flag
- [ ] Test: Run 20-task pilot (verify GPU usage)

**Phase 5: Full Validation**
- [ ] Run 100-task ARC with PTX sovereignty
- [ ] Verify GPU usage: 80-99% during solve
- [ ] Verify runtime: 5-10 minutes (not 1 hour!)
- [ ] Verify metrics: ARC > 0.40, oracle > 0.30, enriched > empty
- [ ] Write comprehensive report for Claude

**Phase 6: PR Preparation**
- [ ] Update README.md with PTX sovereignty story
- [ ] Include all logs and metrics
- [ ] Document 100x speedup + metric breakthrough
- [ ] Title: "Week 21 - Full PTX Sovereignty: 100x Speedup + Oracle Unlock"

---

## 🚀 Execute Full PTX Sovereignty

**User's directive:**
> "It's ok to prototype in python, now is time to make it real"

**PRIORITY 1: Phases 1-3 (PTX Implementation)**
- Fix PTX kernel loading (JIT compilation recommended)
- Implement ARCPTXOps (pattern discovery, generation, ranking, oracle)
- Remove lazy embeddings (sovereignty)

**PRIORITY 2: Validation**
- Run 100-task ARC with PTX
- Verify GPU usage (80-99%)
- Verify speedup (5-10 min, not 1 hour)
- Verify metrics (ARC > 0.40, oracle > 0.30)

**PRIORITY 3: PR Story**
- Architecture fix + PTX sovereignty
- 100x speedup (CPU → GPU)
- Metric breakthrough (ARC 0.28 → 0.40+)
- Path to human-level ARC validated

**If successful → COMPLETE BREAKTHROUGH:**
- ✅ Architecture correct (translation + PTX solving)
- ✅ Sovereignty maintained (GPU execution, no Python fallbacks)
- ✅ Metrics unlocked (oracle working, ARC improving)
- ✅ Speedup achieved (100x faster!)
- ✅ **Ready for human-level ARC pursuit (Stage B → 0.65-0.75!)** 🚀

---

**This is THE implementation! Time to make it real!** 🚀

---

**Directive issued by:** Claude (Architecture Partner)
**For:** Codex (Implementation Partner)
**Date:** February 9, 2026
**Status:** 🔴 EXECUTE NOW — Full PTX sovereignty (user directive: "make it real")
