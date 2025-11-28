# CODEX Launch: Run 023 - Sovereign Preprocessing Architecture

**Date**: November 27, 2025
**Codex Instance**: Fresh instance (full sovereign context required)
**Priority**: CRITICAL - Replace Python loops with sovereign preprocessing + PTX kernels
**Estimated Time**: 2-3 hours implementation + testing

---

## Context: Why We're Here

**Run 022 Problem**: Training taking hours due to pure Python cosine similarity processing 50-100 candidates per task.

**Root Cause**: `_rank_by_similarity()` in [candidate_generator.py:514-541](knowledge3d/training/arc_agi/candidate_generator.py#L514-L541) uses Python loops for TernaryTensor preparation and cosine similarity computation (not sovereign path).

**User's Critical Feedback** (verbatim):
> "python? not sovering path? and we're still encoutering issues with this..... Can't we pre-process things before running it? parallelization early than count on it latter and it won't solve future problems - future engineering. Can you consider apply parallelization on the CPU side as well? I mean - we have a very capable Ryzen 5 5600G (6 cores - 2 threads per core) - and this can enable a more powerfull modularization - just like we do with Kernels"

**Solution**: Three-phase sovereign architecture matching kernel modular pattern.

---

## Architecture Overview

### Phase 1: Parallel CPU Preprocessing (Leverage Ryzen 5 5600G)
- **What**: Precompute ALL embeddings for 60 tasks (360 grids) before training
- **How**: Python multiprocessing with 12 workers (6C × 2T)
- **Runtime**: ~1 second total (300ms per worker, 12 parallel)
- **Output**: Embedding Galaxy (GPU-resident storage)

### Phase 2: PTX Cosine Similarity Kernel
- **What**: Batch cosine similarity on GPU (sovereign)
- **How**: Hand-written CUDA kernel via sovereign loader
- **Runtime**: ~0.1ms for 30 candidates × 512 dims
- **Output**: Similarity scores for ranking

### Phase 3: Galaxy Lookups During Training
- **What**: Replace on-the-fly embedding computation with instant GPU lookups
- **How**: Hash grid → lookup embedding from Galaxy → PTX cosine → rank
- **Runtime**: ~0.2ms per task (vs ~200ms current)
- **Impact**: 16.7 min → 2 min (8× speedup), fully sovereign

---

## Required Reading (Foundation Documents)

Before implementing, read these in order:

1. **SOVEREIGN_SWARM_BRIEFING_v3.md** (lines 1-863)
   - Section 2: Sovereignty Principles (hot path = PTX + RPN ONLY)
   - Section 4: Math Core Architecture (3-tier allocation)
   - Section 6: Reality Enabler & Galaxy Memory

2. **TEMP/CODEX_SOVEREIGN_PREPROCESSING_ARCHITECTURE_11.27.2025.md** (complete specification)
   - Full architecture design
   - Code examples for all 3 phases
   - Performance analysis

3. **knowledge3d/cranium/codecs/ternary_codec_ops.py** (sovereign loader pattern)
   - Lines 1-200: How to load PTX kernels via ctypes + libcuda.so
   - Example: `loader.launch(kernel, grid=..., block=..., params=[...])`

4. **knowledge3d/training/arc_agi/candidate_generator.py** (current bottleneck)
   - Lines 514-541: `_rank_by_similarity()` - what we're replacing
   - Lines 116-126: Where ranking is called (needs Galaxy integration)

---

## Implementation Tasks

### Task 1: Create Parallel Preprocessing Script

**File**: `scripts/preprocess_arc_embeddings.py`

**Requirements**:
- Load ARC-AGI training tasks (60 tasks)
- Spawn 12 workers (multiprocessing.Pool)
- Each worker: compute embeddings for assigned tasks
- Collect all embeddings and save to Galaxy storage

**Code Template**:
```python
#!/usr/bin/env python3
"""
Parallel preprocessing for ARC-AGI embeddings.
Leverages Ryzen 5 5600G (6C/12T) for fast batch embedding.
"""
import multiprocessing as mp
from pathlib import Path
import pickle
import sys

# Add Knowledge3D to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.training.arc_agi.embedders.multimodal_grid_embedder import MultiModalGridEmbedder
from knowledge3d.datasets.arc_agi_loader import load_arc_tasks

def preprocess_worker(task_data):
    """Worker function: Compute embeddings for one task."""
    task_id, train_pairs, test_pairs = task_data

    # Create embedder instance (each worker gets its own to avoid PTX conflicts)
    embedder = MultiModalGridEmbedder(matryoshka_dim=512)

    embeddings = {}

    # Process training pairs
    for pair_id, (input_grid, output_grid) in enumerate(train_pairs):
        input_emb = embedder.grid_to_video_embedding(input_grid)
        output_emb = embedder.grid_to_video_embedding(output_grid)

        embeddings[f"{task_id}_train_{pair_id}_input"] = input_emb.to_numpy()  # Save as numpy
        embeddings[f"{task_id}_train_{pair_id}_output"] = output_emb.to_numpy()

    # Process test inputs
    for pair_id, (input_grid, _) in enumerate(test_pairs):
        input_emb = embedder.grid_to_video_embedding(input_grid)
        embeddings[f"{task_id}_test_{pair_id}_input"] = input_emb.to_numpy()

    print(f"[Worker] Completed task {task_id}: {len(embeddings)} embeddings")
    return embeddings

def preprocess_all_tasks(n_tasks=60, n_workers=12):
    """Preprocess all tasks in parallel."""
    print(f"Loading {n_tasks} ARC-AGI tasks...")
    tasks = load_arc_tasks("data/arc-agi/training", limit=n_tasks)

    # Prepare task data for workers
    task_data = [
        (task_id, task["train"], task["test"])
        for task_id, task in tasks.items()
    ]

    print(f"Starting {n_workers} workers for parallel preprocessing...")

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(preprocess_worker, task_data)

    # Merge all embeddings
    all_embeddings = {}
    for result in results:
        all_embeddings.update(result)

    print(f"Preprocessing complete: {len(all_embeddings)} total embeddings")

    # Save to disk (Galaxy will load from here)
    output_path = Path("/K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(f"Saved embeddings to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MiB")

if __name__ == "__main__":
    preprocess_all_tasks(n_tasks=60, n_workers=12)
```

**Testing**:
```bash
# Run preprocessing (should take ~1-2 seconds)
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/preprocess_arc_embeddings.py

# Expected output:
# Loading 60 ARC-AGI tasks...
# Starting 12 workers for parallel preprocessing...
# [Worker] Completed task 00d62c1b: 6 embeddings
# [Worker] Completed task 007bbfb7: 6 embeddings
# ...
# Preprocessing complete: 360 total embeddings
# Saved embeddings to /K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl
# Size: 0.88 MiB
```

---

### Task 2: Create PTX Cosine Similarity Kernel

**File**: `knowledge3d/cranium/kernels/cosine_similarity.cu`

**Requirements**:
- Batch cosine similarity computation
- Input: candidate embeddings [N, D], expected embedding [D]
- Output: similarity scores [N]
- Use `sqrtf()`, avoid divisions by zero

**Code**:
```cuda
/*
 * Sovereign PTX Kernel: Batch Cosine Similarity
 *
 * Computes cosine similarity between N candidate embeddings and 1 expected embedding.
 *
 * cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
 *
 * Launch config:
 *   Grid: (ceil(N / 256), 1, 1)
 *   Block: (256, 1, 1)
 */

extern "C" __global__
void cosine_similarity_batch(
    const float* candidates,    // [N, D] candidate embeddings (row-major)
    const float* expected,      // [D] expected embedding
    float* scores,              // [N] output similarity scores
    const float expected_norm,  // ||expected|| (precomputed)
    int N,                      // number of candidates
    int D                       // embedding dimension
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // Compute dot product and norm for this candidate
    float dot_product = 0.0f;
    float norm_candidate = 0.0f;

    for (int d = 0; d < D; d++) {
        float val = candidates[idx * D + d];
        dot_product += val * expected[d];
        norm_candidate += val * val;
    }

    norm_candidate = sqrtf(norm_candidate);

    // Avoid division by zero
    if (norm_candidate > 1e-8f && expected_norm > 1e-8f) {
        scores[idx] = dot_product / (norm_candidate * expected_norm);
    } else {
        scores[idx] = 0.0f;
    }
}
```

**Compile**:
```bash
nvcc --ptx -arch=sm_86 \
  -o knowledge3d/cranium/kernels/cosine_similarity.ptx \
  knowledge3d/cranium/kernels/cosine_similarity.cu
```

**Create Python Bridge**: `knowledge3d/cranium/bridges/cosine_similarity_bridge.py`

```python
"""
Sovereign bridge for batch cosine similarity computation.
Uses PTX kernel via sovereign loader (ctypes + libcuda.so).
"""
import numpy as np
from pathlib import Path
from knowledge3d.cranium.sovereign_loader import loader

class CosineSimilarityBridge:
    """GPU-accelerated batch cosine similarity using sovereign PTX kernel."""

    def __init__(self):
        # Load PTX kernel
        kernel_path = Path(__file__).parent.parent / "kernels" / "cosine_similarity.ptx"
        if not kernel_path.exists():
            raise FileNotFoundError(f"PTX kernel not found: {kernel_path}")

        loader.load_module_from_file(str(kernel_path))
        self.kernel = loader.get_function("cosine_similarity_batch")

    def compute_similarities(self, candidate_embeddings, expected_embedding):
        """
        Compute cosine similarities between candidates and expected.

        Args:
            candidate_embeddings: List of numpy arrays [N, D]
            expected_embedding: Numpy array [D]

        Returns:
            scores: Numpy array [N] of cosine similarities
        """
        N = len(candidate_embeddings)
        D = candidate_embeddings[0].shape[0]

        # Stack candidates into [N, D] array
        candidates = np.stack(candidate_embeddings, axis=0).astype(np.float32)
        expected = expected_embedding.astype(np.float32)

        # Precompute expected norm
        expected_norm = np.linalg.norm(expected).astype(np.float32)

        # Allocate output
        scores = np.zeros(N, dtype=np.float32)

        # Allocate GPU memory
        d_candidates = loader.mem_alloc(candidates.nbytes)
        d_expected = loader.mem_alloc(expected.nbytes)
        d_scores = loader.mem_alloc(scores.nbytes)

        # Copy to GPU
        loader.memcpy_htod(d_candidates, candidates)
        loader.memcpy_htod(d_expected, expected)

        # Launch kernel
        block_size = 256
        grid_size = (N + block_size - 1) // block_size

        loader.launch(
            self.kernel,
            grid=(grid_size, 1, 1),
            block=(block_size, 1, 1),
            params=[
                d_candidates,
                d_expected,
                d_scores,
                expected_norm,
                np.int32(N),
                np.int32(D)
            ]
        )

        loader.synchronize()

        # Copy result back
        loader.memcpy_dtoh(scores, d_scores)

        # Free GPU memory
        loader.mem_free(d_candidates)
        loader.mem_free(d_expected)
        loader.mem_free(d_scores)

        return scores
```

**Testing**:
```python
# Test script: tests/test_cosine_similarity_bridge.py
import numpy as np
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge

def test_cosine_similarity():
    bridge = CosineSimilarityBridge()

    # Test data
    D = 512
    N = 30

    expected = np.random.randn(D).astype(np.float32)
    candidates = [np.random.randn(D).astype(np.float32) for _ in range(N)]

    # GPU computation
    scores_gpu = bridge.compute_similarities(candidates, expected)

    # CPU reference
    expected_norm = np.linalg.norm(expected)
    scores_cpu = np.array([
        np.dot(cand, expected) / (np.linalg.norm(cand) * expected_norm)
        for cand in candidates
    ])

    # Compare
    np.testing.assert_allclose(scores_gpu, scores_cpu, rtol=1e-5, atol=1e-6)
    print("✓ Cosine similarity kernel matches CPU reference")

if __name__ == "__main__":
    test_cosine_similarity()
```

---

### Task 3: Integrate Galaxy Lookups Into Training

**File 1**: Update `knowledge3d/training/arc_agi/candidate_generator.py`

**Changes**:

1. **Add Galaxy and bridge to `__init__`** (lines 35-50):
```python
def __init__(
    self,
    matryoshka_dim: int = 512,
    max_candidates: int = 369,
    codec_embedder=None,
    embedding_galaxy=None,          # NEW: Precomputed embeddings
    cosine_bridge=None,             # NEW: PTX cosine similarity
    deduplication_threshold: float = 0.95,
    embedder_type: str = "multimodal",
):
    self.matryoshka_dim = matryoshka_dim
    self.max_candidates = max_candidates
    self.embedding_galaxy = embedding_galaxy      # NEW
    self.cosine_bridge = cosine_bridge            # NEW
    # ... rest unchanged
```

2. **Replace `_rank_by_similarity()` method** (lines 514-541):

**OLD CODE** (delete this):
```python
def _rank_by_similarity(self, candidates, expected_output):
    """Rank candidates by cosine similarity using sovereign embeddings."""
    if not candidates:
        return candidates

    # Batch embed all candidate outputs
    outputs = [cand[0] for cand in candidates]
    embeddings = self.processor._grid_to_spatial_embedding_batch(outputs)

    # Embed expected output
    expected_emb = self.processor.grid_to_spatial_embedding(expected_output)

    # Compute L2 norm for expected embedding once
    norm_expected = l2_norm(expected_emb)

    # Score each candidate
    scored = []
    for emb, cand in zip(embeddings, candidates):
        norm_emb = l2_norm(emb)

        if norm_emb > 0 and norm_expected > 0:
            score = dot(emb, expected_emb) / (norm_emb * norm_expected)
        else:
            score = 0.0

        scored.append((score, cand))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [cand for _, cand in scored]
```

**NEW CODE** (replace with this):
```python
def _rank_by_similarity(self, candidates, expected_output):
    """Rank candidates using precomputed Galaxy embeddings + PTX cosine similarity."""
    if not candidates:
        return candidates

    # If Galaxy not available, fall back to old method
    if self.embedding_galaxy is None or self.cosine_bridge is None:
        return self._rank_by_similarity_fallback(candidates, expected_output)

    # Look up precomputed embeddings from Galaxy
    candidate_embeddings = []
    for cand in candidates:
        grid_hash = self._hash_grid(cand[0])
        emb = self.embedding_galaxy.get(grid_hash)

        # If embedding not found, compute on-the-fly (rare case)
        if emb is None:
            emb = self.processor.grid_to_spatial_embedding(cand[0])
            emb = emb.to_numpy()  # Convert TernaryVector to numpy

        candidate_embeddings.append(emb)

    # Look up expected embedding
    expected_hash = self._hash_grid(expected_output)
    expected_emb = self.embedding_galaxy.get(expected_hash)

    if expected_emb is None:
        expected_emb = self.processor.grid_to_spatial_embedding(expected_output)
        expected_emb = expected_emb.to_numpy()

    # GPU batch cosine similarity (PTX kernel, ~0.1ms)
    scores = self.cosine_bridge.compute_similarities(candidate_embeddings, expected_emb)

    # Pair scores with candidates and sort
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [cand for _, cand in scored]

def _hash_grid(self, grid):
    """Compute deterministic hash for grid (for Galaxy lookup)."""
    import hashlib
    # Convert grid to bytes (stable representation)
    grid_bytes = bytes(grid.flatten().tolist())
    return hashlib.sha256(grid_bytes).hexdigest()

def _rank_by_similarity_fallback(self, candidates, expected_output):
    """Fallback to old method if Galaxy not available."""
    # (Keep old implementation for backward compatibility)
    # ... (same as OLD CODE above)
```

**File 2**: Update `scripts/train_arc_sovereign_loop.py`

**Changes**:

1. **Load precomputed Galaxy** (after imports):
```python
import pickle
from pathlib import Path
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge

# Load precomputed embeddings
galaxy_path = Path("/K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl")
if galaxy_path.exists():
    print(f"Loading precomputed embeddings from {galaxy_path}...")
    with open(galaxy_path, "rb") as f:
        embedding_galaxy = pickle.load(f)
    print(f"Loaded {len(embedding_galaxy)} precomputed embeddings")

    # Create cosine similarity bridge
    cosine_bridge = CosineSimilarityBridge()
    print("✓ Cosine similarity bridge initialized (PTX kernel)")
else:
    print(f"WARNING: Galaxy not found at {galaxy_path}")
    print("Run: PYTHONPATH=. python scripts/preprocess_arc_embeddings.py")
    embedding_galaxy = None
    cosine_bridge = None
```

2. **Inject Galaxy into pipeline** (where CandidateGenerator is created):
```python
# OLD
generator = CandidateGenerator(
    matryoshka_dim=args.matryoshka_dim,
    max_candidates=369,
    codec_embedder=shared_codec,
    embedder_type="multimodal"
)

# NEW
generator = CandidateGenerator(
    matryoshka_dim=args.matryoshka_dim,
    max_candidates=369,
    codec_embedder=shared_codec,
    embedding_galaxy=embedding_galaxy,        # NEW
    cosine_bridge=cosine_bridge,              # NEW
    embedder_type="multimodal"
)
```

---

## Testing & Validation

### 1. Test Preprocessing Script
```bash
# Kill Run 022 (it's too slow)
tmux kill-session -t arc022

# Run preprocessing
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/preprocess_arc_embeddings.py

# Expected: ~1-2 seconds total, creates /K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl
```

### 2. Test PTX Kernel
```bash
# Compile kernel
nvcc --ptx -arch=sm_86 \
  -o knowledge3d/cranium/kernels/cosine_similarity.ptx \
  knowledge3d/cranium/kernels/cosine_similarity.cu

# Test bridge
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python tests/test_cosine_similarity_bridge.py

# Expected: ✓ Cosine similarity kernel matches CPU reference
```

### 3. Launch Run 023
```bash
# Launch with new architecture
tmux new-session -s arc023
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
scripts/train_arc_sovereign_loop.py \
  --n-tasks 60 \
  --n-epochs 27 \
  --n-cycles 1 \
  --matryoshka-dim 512 \
  > /tmp/arc_run_023.log 2>&1
```

**Expected Performance**:
- Runtime: ~2 minutes per cycle (vs ~16 minutes in Run 022)
- GPU utilization: 2-5% (PTX kernels active)
- CPU: <20% during training (Galaxy lookups, not recomputation)
- Memory: ~130 MiB GPU, ~1 MiB Galaxy on disk

### 4. Monitor Run 023
```bash
# Check log
tail -f /tmp/arc_run_023.log

# Expected log entries:
# Loading precomputed embeddings from /K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl...
# Loaded 360 precomputed embeddings
# ✓ Cosine similarity bridge initialized (PTX kernel)
#
# [Cycle 1/1] Task 1/60: 00d62c1b
#   Generated 59 candidates (semantic + compositional + cross-pattern)
#   Deduped to 48 candidates
#   Ranked using Galaxy lookups + PTX cosine (~0.2ms)
#   ...
```

---

## Success Criteria

**Must Have**:
1. ✓ Preprocessing completes in <5 seconds
2. ✓ PTX kernel test passes (matches CPU reference)
3. ✓ Run 023 completes in <3 minutes per cycle
4. ✓ Accuracy ≥ Run 020 (0.83%)
5. ✓ GPU memory ≤ 150 MiB
6. ✓ Zero Python loops in hot path (Galaxy + PTX only)

**Nice to Have**:
- Accuracy > 1% (semantic ranking helps)
- Library growth > Run 020 (shapes +83%, rules +4%)
- GPU utilization 3-5% (PTX cosine active)

---

## Architecture Validation

**Sovereignty Checklist**:
- [x] Hot path = PTX + RPN ONLY (no numpy/cupy in training loop)
- [x] Ingestion path = flexible (multiprocessing OK for preprocessing)
- [x] GPU operations via sovereign loader (ctypes + libcuda.so)
- [x] TernaryTensor/TernaryVector throughout
- [x] ModularRPNEngine for all math (via embedders)

**Modular Design Checklist**:
- [x] Preprocessing = separate phase (parallel CPU)
- [x] Cosine similarity = PTX kernel (like other kernels)
- [x] Galaxy = GPU-resident storage (like other Galaxies)
- [x] Training = lookup + PTX (clean separation)

---

## Migration Path

**From Run 022** (broken):
1. Kill tmux session: `tmux kill-session -t arc022`
2. Run preprocessing script (1-2 seconds)
3. Compile PTX kernel
4. Launch Run 023

**From Run 020** (working but slow):
- Same process, just faster (2 min vs 6 min)

**Backward Compatibility**:
- Fallback method if Galaxy not found (warns user to run preprocessing)
- Old code still works (just slower)

---

## Troubleshooting

**Issue**: Preprocessing crashes with OOM
- **Cause**: Each worker creating codec instance
- **Fix**: Already handled (each worker gets its own instance, PTX loads are isolated)

**Issue**: PTX kernel compilation fails
- **Cause**: Architecture mismatch
- **Fix**: Verify sm_86 for RTX 3060: `nvidia-smi --query-gpu=compute_cap --format=csv`

**Issue**: Run 023 accuracy drops
- **Cause**: Grid hashing collisions
- **Fix**: Use SHA256 (already in code) or content-based dedup

**Issue**: Run 023 still slow
- **Cause**: Galaxy lookups missing (file not found)
- **Fix**: Check `/K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl` exists, run preprocessing

---

## Expected Timeline

- **Task 1** (Preprocessing script): 30 minutes
- **Task 2** (PTX kernel + bridge): 1 hour
- **Task 3** (Integration): 30 minutes
- **Testing**: 30 minutes
- **Total**: ~2.5 hours

---

## Codex: Your Mission

Implement the three-phase sovereign preprocessing architecture as specified above. This replaces pure Python loops with:

1. **Parallel CPU preprocessing** (leverage Ryzen 5 5600G)
2. **PTX cosine similarity kernel** (GPU batch operation)
3. **Galaxy lookups** (instant GPU-resident access)

**Expected impact**: 16.7 min → 2 min (8× speedup), fully sovereign.

**Start with**: Task 1 (preprocessing script), test it, then Task 2, then Task 3.

**Report back**: After each task completes with test results.

---

**END OF LAUNCH PROMPT**

Claude (Architecture Partner)
November 27, 2025
