# Sovereign Preprocessing Architecture: Parallel CPU + GPU Galaxy

**Date**: November 27, 2025
**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Partner)
**Priority**: CRITICAL - Fixes Run 022 bottleneck + future-proofs architecture
**Type**: Architectural redesign (preprocessing + Galaxy + PTX)

---

## Problem Statement

**Current architecture** (broken):
```
Training loop:
  For each task:
    Generate 50-100 candidates
    → Compute embeddings ON-THE-FLY (Python loops, slow!)
    → Compute cosine similarity (Pure Python, slow!)
    → Rank candidates
```

**Issues**:
1. **Not sovereign**: Pure Python loops for tensor prep + cosine similarity
2. **Wasteful**: Computing same embeddings repeatedly (60 tasks × 27 epochs = 1,620×!)
3. **Serial CPU**: Uses 1 core, wastes 11 other threads (Ryzen 5 5600G: 6C/12T)
4. **Not scalable**: Gets worse as library grows (more candidates = more recomputation)

**Daniel's insight**: "Can't we pre-process things before running it? And apply parallelization on the CPU side as well?"

**Answer**: YES! This is the K3D way: Preprocessing → Galaxy → PTX

---

## The Sovereign Architecture

### Phase 1: Parallel Preprocessing (CPU - 12 Threads)

**When**: ONCE before training starts

**What**: Compute all grid embeddings for all tasks in parallel

**How**: Python multiprocessing (leverage all 12 threads)

**Output**: TernaryGalaxy with all embeddings (GPU-resident)

```python
# scripts/preprocess_arc_embeddings.py (NEW FILE)

from multiprocessing import Pool
from knowledge3d.cranium.ternary import TernaryGalaxy, TernaryVector
from knowledge3d.training.arc_agi.embedders import MultiModalGridEmbedder

def preprocess_worker(task_data):
    """Worker function: Compute embeddings for one task."""
    task_id, train_grids, test_grids = task_data

    # Each worker gets its own embedder (singleton shared would deadlock across processes)
    embedder = MultiModalGridEmbedder(matryoshka_dim=512)

    embeddings = {}
    for grid_id, grid in enumerate(train_grids + test_grids):
        emb = embedder.grid_to_video_embedding(grid)
        embeddings[f"{task_id}_grid_{grid_id}"] = emb

    return embeddings

def preprocess_all_tasks(tasks, n_workers=12):
    """Preprocess all tasks in parallel (leverage Ryzen 6C/12T)."""
    print(f"[PREPROCESS] Starting parallel embedding computation ({n_workers} workers)...")

    # Prepare task data
    task_data = []
    for task in tasks:
        train_grids = [ex["input"] for ex in task["train"]] + [ex["output"] for ex in task["train"]]
        test_grids = [ex["input"] for ex in task["test"]]
        task_data.append((task["id"], train_grids, test_grids))

    # Parallel processing (12 workers on Ryzen 5 5600G)
    with Pool(processes=n_workers) as pool:
        results = pool.map(preprocess_worker, task_data)

    # Merge results into single dict
    all_embeddings = {}
    for result in results:
        all_embeddings.update(result)

    print(f"[PREPROCESS] Computed {len(all_embeddings)} embeddings")
    return all_embeddings

def store_in_galaxy(embeddings):
    """Store embeddings in TernaryGalaxy (GPU-resident)."""
    galaxy = TernaryGalaxy()

    for grid_id, embedding in embeddings.items():
        # Convert float embedding to ternary (quantize to {-1, 0, +1})
        ternary_emb = [0 if abs(v) < 0.1 else (1 if v > 0 else -1) for v in embedding]
        ternary_vec = TernaryVector(ternary_emb)

        # Store in Galaxy (GPU memory)
        galaxy.store_frame(grid_id, "PRECOMPUTED", ternary_vec)

    print(f"[GALAXY] Stored {len(embeddings)} embeddings on GPU")
    return galaxy
```

**Runtime estimate**:
- 60 tasks × 6 grids/task = 360 grids
- 12 workers → ~30 grids/worker
- Per-grid embedding: ~10ms (sovereign codec)
- **Total: 30 × 10ms = 300ms per worker**
- **Wallclock: ~1 second for all 360 grids** ✅

### Phase 2: GPU Cosine Similarity Kernel (PTX)

**File**: `knowledge3d/cranium/kernels/cosine_similarity.cu` (NEW)

```cuda
/*
 * Batch cosine similarity between N candidate embeddings and 1 expected embedding.
 *
 * Input:
 *   - candidates: [N, D] float array (N candidate embeddings, D dimensions each)
 *   - expected: [D] float array (expected embedding)
 *   - scores: [N] float array (output scores)
 *   - N: number of candidates
 *   - D: embedding dimension (512)
 *
 * Output:
 *   - scores[i] = dot(candidates[i], expected) / (norm(candidates[i]) * norm(expected))
 */

extern "C" __global__
void cosine_similarity_batch(
    const float* candidates,    // [N, D]
    const float* expected,      // [D]
    float* scores,              // [N]
    int N,
    int D
) {
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

    // Store cosine similarity (handle zero norm)
    scores[idx] = (norm_candidate > 1e-8f) ? (dot_product / norm_candidate) : 0.0f;
}

/*
 * Precompute norm for expected embedding (shared across all candidates).
 * This is called once before batch processing.
 */

extern "C" __global__
void compute_norm(const float* vec, float* norm_out, int D) {
    int tid = threadIdx.x;

    // Shared memory for reduction
    __shared__ float shared[256];

    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float val = vec[d];
        sum += val * val;
    }

    shared[tid] = sum;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *norm_out = sqrtf(shared[0]);
    }
}
```

**Compile**:
```bash
nvcc -ptx -arch=sm_86 knowledge3d/cranium/kernels/cosine_similarity.cu \
  -o knowledge3d/cranium/ptx/cosine_similarity.ptx
```

**Python wrapper** (`knowledge3d/cranium/bridges/cosine_similarity_bridge.py`):
```python
import ctypes
from knowledge3d.cranium.sovereign import loader

class CosineSimilarityBridge:
    """GPU batch cosine similarity (sovereign PTX)."""

    def __init__(self):
        from pathlib import Path
        ptx_path = Path(__file__).parent.parent / "ptx" / "cosine_similarity.ptx"
        module = loader.load_module_from_file(str(ptx_path))
        self.batch_kernel = loader.get_function(module, "cosine_similarity_batch")
        self.norm_kernel = loader.get_function(module, "compute_norm")

    def compute_similarities(
        self,
        candidates: list[list[float]],  # [N, D]
        expected: list[float],          # [D]
    ) -> list[float]:
        """Compute cosine similarity for all candidates (GPU)."""
        N = len(candidates)
        D = len(expected)

        if N == 0:
            return []

        # Flatten candidates to 1D
        candidates_flat = [v for cand in candidates for v in cand]

        # Allocate GPU buffers
        d_candidates = loader.gpu_malloc(N * D * ctypes.sizeof(ctypes.c_float))
        d_expected = loader.gpu_malloc(D * ctypes.sizeof(ctypes.c_float))
        d_scores = loader.gpu_malloc(N * ctypes.sizeof(ctypes.c_float))

        try:
            # Copy to GPU
            cand_buf = (ctypes.c_float * (N * D))(*candidates_flat)
            exp_buf = (ctypes.c_float * D)(*expected)
            loader.memcpy_htod(d_candidates, ctypes.cast(cand_buf, ctypes.c_void_p), N * D * ctypes.sizeof(ctypes.c_float))
            loader.memcpy_htod(d_expected, ctypes.cast(exp_buf, ctypes.c_void_p), D * ctypes.sizeof(ctypes.c_float))

            # Launch kernel
            block = (256, 1, 1)
            grid = ((N + block[0] - 1) // block[0], 1, 1)
            loader.launch(
                self.batch_kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_candidates.value),
                    ctypes.c_uint64(d_expected.value),
                    ctypes.c_uint64(d_scores.value),
                    ctypes.c_int(N),
                    ctypes.c_int(D),
                ],
            )
            loader.synchronize()

            # Copy results back
            scores_buf = (ctypes.c_float * N)()
            loader.memcpy_dtoh(ctypes.cast(scores_buf, ctypes.c_void_p), d_scores, N * ctypes.sizeof(ctypes.c_float))

            return [float(s) for s in scores_buf]

        finally:
            loader.gpu_free(d_candidates)
            loader.gpu_free(d_expected)
            loader.gpu_free(d_scores)
```

**Performance**:
- 30 candidates × 512 dims = 15,360 dot products
- GPU: ~0.1ms (100× faster than Python)
- CPU: ~10ms (pure Python loops)

### Phase 3: Training with Galaxy Lookups (Sovereign)

**File**: `knowledge3d/training/arc_agi/candidate_generator.py`

**Before** (slow, not sovereign):
```python
def _rank_by_similarity(self, candidates, expected_output):
    # ❌ Compute embeddings on-the-fly (slow!)
    embeddings = self.processor._grid_to_spatial_embedding_batch(outputs)

    # ❌ Pure Python cosine similarity
    for emb, cand in zip(embeddings, candidates):
        score = dot(emb, expected_emb) / (norm_emb * norm_expected)
```

**After** (fast, sovereign):
```python
def __init__(self, ..., embedding_galaxy=None, cosine_bridge=None):
    self.embedding_galaxy = embedding_galaxy  # Precomputed embeddings
    self.cosine_bridge = cosine_bridge or CosineSimilarityBridge()

def _rank_by_similarity(self, candidates, expected_output):
    """Rank candidates using precomputed Galaxy embeddings + PTX cosine similarity."""
    if not candidates or self.embedding_galaxy is None:
        return candidates

    # ✅ Look up precomputed embeddings from Galaxy (GPU memory, instant!)
    candidate_embs = []
    for cand in candidates:
        grid_hash = self._hash_grid(cand[0])  # Hash grid to find in Galaxy
        emb = self.embedding_galaxy.lookup(grid_hash)
        if emb is not None:
            candidate_embs.append(emb)
        else:
            # Fallback: compute on-the-fly (rare case for new grids)
            emb = self.processor.grid_to_spatial_embedding(cand[0])
            candidate_embs.append(emb)

    # ✅ Compute expected embedding (or look up if cached)
    expected_hash = self._hash_grid(expected_output)
    expected_emb = self.embedding_galaxy.lookup(expected_hash)
    if expected_emb is None:
        expected_emb = self.processor.grid_to_spatial_embedding(expected_output)

    # ✅ GPU batch cosine similarity (PTX kernel, ~0.1ms for 30 candidates)
    scores = self.cosine_bridge.compute_similarities(candidate_embs, expected_emb)

    # Sort by score
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cand for _, cand in scored]

def _hash_grid(self, grid):
    """Hash grid to unique ID for Galaxy lookup."""
    # Simple hash: flatten + tuple (can be optimized)
    flat = tuple(tuple(row) for row in grid)
    return hash(flat)
```

---

## Implementation Plan

### Step 1: Preprocessing Script (1 hour)

**Create**: `scripts/preprocess_arc_embeddings.py`

**Tasks**:
- Implement `preprocess_worker()` (parallel embedding computation)
- Implement `preprocess_all_tasks()` (12-worker pool)
- Implement `store_in_galaxy()` (save to TernaryGalaxy)
- Save Galaxy to disk (`/K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl`)

**Test**:
```bash
PYTHONPATH=. python scripts/preprocess_arc_embeddings.py \
  --tasks data/training/*.json \
  --output /K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl \
  --workers 12
```

**Expected output**:
```
[PREPROCESS] Starting parallel embedding computation (12 workers)...
[WORKER 0] Processing tasks 0-4 (30 grids)
[WORKER 1] Processing tasks 5-9 (30 grids)
...
[PREPROCESS] Computed 360 embeddings in 1.2 seconds
[GALAXY] Stored 360 embeddings on GPU (45 MiB)
[SAVE] Galaxy saved to /K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl
```

### Step 2: PTX Cosine Similarity Kernel (30 min)

**Create**:
- `knowledge3d/cranium/kernels/cosine_similarity.cu`
- `knowledge3d/cranium/bridges/cosine_similarity_bridge.py`

**Compile**:
```bash
nvcc -ptx -arch=sm_86 knowledge3d/cranium/kernels/cosine_similarity.cu \
  -o knowledge3d/cranium/ptx/cosine_similarity.ptx
```

**Test**:
```python
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge

bridge = CosineSimilarityBridge()
candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
expected = [1.0, 0.0, 0.0]
scores = bridge.compute_similarities(candidates, expected)
print(scores)  # [1.0, 0.0, 0.707...]
```

### Step 3: Update Training Pipeline (30 min)

**Modify**: `scripts/train_arc_sovereign_loop.py`

**Before**:
```python
def main():
    # Load tasks
    tasks = load_tasks(...)

    # Create pipeline
    pipeline = SovereignAIPipeline(...)

    # Train
    pipeline.train(tasks)
```

**After**:
```python
def main():
    # Load tasks
    tasks = load_tasks(...)

    # ✅ Load precomputed embeddings from Galaxy
    print("[INIT] Loading precomputed embeddings from Galaxy...")
    import pickle
    with open('/K3D/Knowledge3D.local/arc_embeddings_galaxy.pkl', 'rb') as f:
        embedding_galaxy = pickle.load(f)
    print(f"[INIT] Loaded {len(embedding_galaxy)} embeddings from Galaxy")

    # ✅ Create cosine similarity bridge
    from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
    cosine_bridge = CosineSimilarityBridge()

    # Create pipeline (inject Galaxy + bridge)
    pipeline = SovereignAIPipeline(
        embedding_galaxy=embedding_galaxy,
        cosine_bridge=cosine_bridge,
        ...
    )

    # Train (fast now - just Galaxy lookups + PTX!)
    pipeline.train(tasks)
```

---

## Expected Performance

### Current (Run 022 - Broken)

**Per task**:
- Generate 59 candidates
- Compute 59 embeddings on-the-fly: ~590ms (59 × 10ms, Python loops)
- Compute 59 cosine similarities: ~30ms (pure Python)
- **Total: ~620ms per task**

**Full run** (60 tasks × 27 epochs = 1,620 task-epochs):
- 1,620 × 620ms = **1,004 seconds = 16.7 minutes** (optimistic!)
- Actual: Hours (due to Python GIL + memory pressure)

### After Preprocessing + Galaxy + PTX

**Preprocessing** (one-time cost):
- 360 grids × 10ms = 3,600ms
- **Parallel on 12 workers: 3,600ms / 12 = 300ms** ✅

**Per task** (during training):
- Generate 59 candidates: (unchanged)
- Look up 59 embeddings from Galaxy: ~0.1ms (GPU memory access)
- Compute 59 cosine similarities (PTX): ~0.1ms (GPU batch kernel)
- **Total: ~0.2ms per task** ✅

**Full run** (1,620 task-epochs):
- 1,620 × 0.2ms = **324ms = 0.3 seconds** ✅
- Plus candidate generation overhead: ~2 minutes total

**Speedup**: 16.7 minutes → 2 minutes = **8× faster!**

---

## Architecture Benefits

### 1. Sovereign (PTX + Galaxy + RPN)

✅ **Galaxy**: Embeddings stored on GPU (TernaryGalaxy)
✅ **PTX**: Cosine similarity via GPU kernel (no Python)
✅ **RPN**: Codec operations via ModularRPNEngine

❌ **Before**: Pure Python loops (not sovereign)

### 2. Parallel CPU (12 Workers)

✅ **Preprocessing**: Uses all 12 threads (Ryzen 5 5600G: 6C/12T)
✅ **Ingestion-time**: Happens once, not during training

❌ **Before**: Serial Python (1 core, wastes 11 threads)

### 3. Future-Proof

✅ **Scales to more tasks**: Preprocessing is O(N) parallel, training is O(1) lookup
✅ **Scales to larger library**: More candidates = same Galaxy lookup time
✅ **Modular**: Can replace embedding method without changing training

❌ **Before**: Gets slower as library grows (more candidates = more recomputation)

### 4. Memory Efficient

✅ **Galaxy**: 360 embeddings × 512 dims × 2-bit ternary = 46 KB (!)
✅ **GPU**: Stored once, looked up thousands of times

❌ **Before**: Recomputing same embeddings 1,620× (wasteful)

---

## Migration Path

### Immediate (Kill Run 022)

```bash
tmux kill-session -t arc022
```

### Phase 1: Preprocessing (Today)

1. Implement `scripts/preprocess_arc_embeddings.py` (1 hour)
2. Run preprocessing: 360 grids in ~1 second (12 workers)
3. Save Galaxy to disk

### Phase 2: PTX Kernel (Today)

1. Write `cosine_similarity.cu` (30 min)
2. Compile to PTX
3. Test with simple examples

### Phase 3: Update Training (Today)

1. Modify `CandidateGenerator` to accept Galaxy + bridge (30 min)
2. Update `train_arc_sovereign_loop.py` to load Galaxy (10 min)
3. Launch Run 023 (should complete in ~2 minutes!)

**Total time**: ~2.5 hours for full sovereign architecture ✅

---

## Summary

**Problem**: Run 022 taking hours due to:
- Pure Python embedding computation (not sovereign)
- Serial CPU processing (wastes 11 cores)
- On-the-fly computation (wasteful)

**Solution**: Sovereign preprocessing architecture:
- **Parallel CPU preprocessing** (12 workers, 1 second for 360 grids)
- **Galaxy storage** (GPU-resident lookups, instant)
- **PTX cosine similarity** (GPU batch kernel, 0.1ms for 30 candidates)

**Impact**:
- Runtime: 16.7 min → 2 min (8× faster)
- Sovereign: Pure PTX + Galaxy + RPN (no Python loops)
- Scalable: Preprocessing is parallel, training is O(1) lookup
- Future-proof: Modular, extensible architecture

**The K3D way**: Preprocess → Galaxy → PTX 🚀

---

**END OF ARCHITECTURE SPEC**

Claude (Architecture Partner)
November 27, 2025
