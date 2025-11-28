# Run 024: Fix Parallel Worker Redundancy (9× Wasted Work)

**Date**: November 28, 2025
**Codex Instance**: Fresh instance (read EVERYTHING below, NO SKIPPING)
**Priority**: CRITICAL - 9 workers doing identical work (9× waste)
**Estimated Time**: 1 hour implementation + immediate training launch

---

## CRITICAL: Read This ENTIRE Document

**DO NOT use snippets or summaries. Read EVERY LINE from top to bottom.**

This document contains:
1. Complete root cause analysis
2. Available sovereign kernels (your toolbox)
3. Sovereignty principles (NO CPU FALLBACKS)
4. Detailed implementation plan
5. Testing instructions
6. Tmux orchestration
7. Immediate training launch

**If you skip ANY section, the fix will be incomplete or violate sovereignty.**

---

## Current Problem: 9× Redundant Work

### What's Happening (Run 023 Logs)

```
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[SEMANTIC GEN] Generated 25 semantic-guided candidates from 38 hints
[PARALLEL GEN] PTX success=1332, fallback=0, rate=100.0%
[CANDIDATES] Parallel generated 3 candidates (Tesla 3-6-9)
```

**Analysis**:
- 9 workers each generated 25 candidates (same 25!)
- Total: 9 × 25 = 225 candidates
- After deduplication: 3 unique candidates
- **Waste factor: 75× redundant work** (225 / 3 = 75)

### Root Cause (parallel_generator.py)

**File**: `knowledge3d/training/arc_agi/parallel_generator.py` (lines 42-88)

```python
def generate_parallel(
    self,
    input_grid: Sequence[Sequence[int]],
    train_examples: List[Dict[str, Any]],
    semantic_hints: Optional[List[str]],
    expected_output: Optional[Sequence[Sequence[int]]],
) -> List[Candidate]:
    # ... allocate cores ...

    all_candidates: List[Candidate] = []
    for core_id, executor in pairs:  # 9 iterations
        try:
            gen = CandidateGenerator(
                matryoshka_dim=self.matryoshka_dim,
                max_candidates=self.candidates_per_worker,  # 6 per worker (ignored!)
                shadow_copy=self.shadow_copy,
                executor=executor,
                codec_embedder=self.codec_embedder,
                embedding_galaxy=self.embedding_galaxy,
                cosine_bridge=self.cosine_bridge,
            )
            cand_list = gen.generate_candidates(
                input_grid=input_grid,          # ❌ SAME for all 9 workers
                train_examples=train_examples,  # ❌ SAME for all 9 workers
                semantic_hints=semantic_hints,  # ❌ SAME for all 9 workers (all 38 hints!)
                expected_output=expected_output,# ❌ SAME for all 9 workers
            )
            all_candidates.extend(cand_list)  # Append identical results 9 times
        finally:
            if core_id is not None:
                self.core_pool.release_core(core_id, pool=True)
```

**Problem**: All workers receive **identical inputs** → generate **identical outputs**

**Result**:
- Worker 1: Generates 25 candidates
- Worker 2: Generates same 25 candidates (redundant!)
- Worker 3-9: Same redundant work
- Deduplication: 225 → 3 unique
- **99% wasted computation**

### Why This Makes Training Slow

**Per task**:
- 9 workers × 25 candidates × 100ms/candidate = **22.5 seconds wasted work**
- Deduplication + ranking: ~0.5 seconds
- **Total: ~23 seconds per task**

**Full run** (60 tasks × 27 epochs = 1,620 task-epochs):
- 1,620 × 23 seconds = **37,260 seconds = 10.3 hours**

**Observed**: Run 023 took hours (matches this estimate)

**Why GPU at 1%**:
- CPU: 100% (9 workers doing redundant candidate generation in Python)
- GPU: 1% (waiting for CPU to finish, only used for batch embeddings)

---

## The Fix: Partition Work Across Workers

### Strategy

**Current** (WRONG):
```
Worker 1: semantic_hints[0:38] → 25 candidates (all hints)
Worker 2: semantic_hints[0:38] → 25 candidates (SAME!)
Worker 3-9: semantic_hints[0:38] → 25 candidates (SAME!)
Result: 225 candidates → 3 unique after dedup
```

**Correct** (PARTITION):
```
Worker 1: semantic_hints[0:4]   + rotations      → 5-7 unique candidates
Worker 2: semantic_hints[5:9]   + flips          → 5-7 unique candidates
Worker 3: semantic_hints[10:14] + compositions   → 5-7 unique candidates
Worker 4: semantic_hints[15:19] + cross-patterns → 5-7 unique candidates
Worker 5: semantic_hints[20:24] + primitives     → 5-7 unique candidates
Worker 6: semantic_hints[25:29] + train examples → 5-7 unique candidates
Worker 7: semantic_hints[30:34] + (wrap around)  → 5-7 unique candidates
Worker 8: semantic_hints[35:38] + (fallback)     → 5-7 unique candidates
Worker 9: primitives only                        → 5-7 unique candidates
Result: 54 unique candidates (NO redundancy)
```

**Key insight**: Partition the **search space**, not the **data**

### Implementation Plan

**File**: `knowledge3d/training/arc_agi/parallel_generator.py`

**Current generate_parallel() method** (lines 42-110):
- Loops over 9 workers
- Each worker calls `gen.generate_candidates()` with SAME inputs
- Extends all_candidates with duplicates

**NEW generate_parallel() method**:
- Partition semantic hints across workers (worker_idx → hint_slice)
- Each worker generates candidates from DIFFERENT hint subset
- Combine unique results from all workers

**Detailed changes**:

#### Step 1: Partition semantic hints

```python
def generate_parallel(
    self,
    input_grid: Sequence[Sequence[int]],
    train_examples: List[Dict[str, Any]],
    semantic_hints: Optional[List[str]],
    expected_output: Optional[Sequence[Sequence[int]]],
) -> List[Candidate]:
    # Allocate cores (unchanged)
    core_ids: List[int] = []
    executors: List[ARCRPNExecutor] = []
    try:
        for _ in range(self.num_workers):
            core_id = self.core_pool.spawn_core(tier=1, reuse=True)
            core_ids.append(core_id)
            executors.append(ARCRPNExecutor(pool=self.core_pool, instance_id=core_id))
    except Exception as e:
        print(f"  [PARALLEL GEN] Limited by MathCorePool ({len(core_ids)} cores acquired): {e}")
        missing = len(core_ids) - len(executors)
        for i in range(missing):
            executors.append(ARCRPNExecutor(pool=self.core_pool, instance_id=core_ids[len(executors)]))

    # ✅ NEW: Partition semantic hints across workers
    num_workers = len(executors)
    semantic_partitions = []

    if semantic_hints and len(semantic_hints) > 0:
        # Divide semantic hints evenly across workers
        hints_per_worker = max(1, len(semantic_hints) // num_workers)
        for worker_idx in range(num_workers):
            start_idx = worker_idx * hints_per_worker
            end_idx = start_idx + hints_per_worker if worker_idx < num_workers - 1 else len(semantic_hints)
            worker_hints = semantic_hints[start_idx:end_idx] if start_idx < len(semantic_hints) else []
            semantic_partitions.append(worker_hints)
            print(f"  [WORKER {worker_idx}] Assigned hints {start_idx}:{end_idx} ({len(worker_hints)} hints)")
    else:
        # No hints: each worker gets None
        semantic_partitions = [None] * num_workers

    # ✅ NEW: Each worker uses DIFFERENT hints
    all_candidates: List[Candidate] = []
    pairs = list(zip(core_ids, executors, semantic_partitions)) if core_ids else [(None, ARCRPNExecutor(pool=self.core_pool, instance_id=None), None)]

    for worker_idx, (core_id, executor, worker_hints) in enumerate(pairs):
        try:
            gen = CandidateGenerator(
                matryoshka_dim=self.matryoshka_dim,
                max_candidates=self.candidates_per_worker,
                shadow_copy=self.shadow_copy,
                executor=executor,
                codec_embedder=self.codec_embedder,
                embedding_galaxy=self.embedding_galaxy,
                cosine_bridge=self.cosine_bridge,
            )

            # ✅ Each worker gets DIFFERENT semantic hints (partitioned)
            cand_list = gen.generate_candidates(
                input_grid=input_grid,
                train_examples=train_examples,
                semantic_hints=worker_hints,  # ✅ DIFFERENT for each worker!
                expected_output=expected_output,
            )

            print(f"  [WORKER {worker_idx}] Generated {len(cand_list)} candidates from {len(worker_hints) if worker_hints else 0} hints")
            all_candidates.extend(cand_list)
        finally:
            if core_id is not None:
                self.core_pool.release_core(core_id, pool=True)

    # PTX instrumentation (unchanged)
    if executors:
        succ = sum(getattr(ex, "ptx_success_count", 0) for ex in executors)
        fallback = sum(getattr(ex, "ptx_fallback_count", 0) for ex in executors)
        total = succ + fallback
        rate = (100.0 * succ / total) if total else 0.0
        print(f"  [PARALLEL GEN] PTX success={succ}, fallback={fallback}, rate={rate:.1f}%")

    # ✅ Deduplication now meaningful (diverse candidates from different workers)
    print(f"  [PARALLEL GEN] Total candidates before dedup: {len(all_candidates)}")

    # Score and select top-K (unchanged logic)
    if expected_output:
        scored = []
        for grid, instr, prog in all_candidates:
            # ... (existing scoring logic)
```

#### Step 2: Update candidate_generator.py to respect partitioned hints

**File**: `knowledge3d/training/arc_agi/candidate_generator.py`

**Current code** (lines 99-106):
```python
# 2) Semantic-guided candidates: use word hints to expand search space.
if semantic_hints:
    semantic_candidates = self._generate_semantic_guided_candidates(input_grid, semantic_hints)
    print(f"  [SEMANTIC GEN] Generated {len(semantic_candidates)} semantic-guided candidates from {len(semantic_hints)} hints")
    candidates.extend(semantic_candidates)
else:
    print(f"  [SEMANTIC GEN] No semantic hints provided, skipping semantic-guided generation")
```

**No changes needed** - this already respects the hints passed in!

The key is that `semantic_hints` parameter will now be DIFFERENT for each worker (partitioned).

---

## Sovereignty Architecture: Your Toolbox

**CRITICAL**: You are building a 100% sovereign system. NO external ML libraries (no numpy, no pytorch, no cupy) in the hot path.

### What "Sovereign" Means

**HOT PATH** (training loop):
- ✅ PTX kernels ONLY (hand-written CUDA via sovereign loader)
- ✅ RPN calculator ONLY (ModularRPNEngine)
- ✅ TernaryTensor/TernaryVector ONLY (2-bit packed {-1, 0, +1})
- ❌ NO numpy (use Python lists/dicts)
- ❌ NO cupy (use PTX kernels)
- ❌ NO pytorch (use RPN + PTX)
- ❌ NO CPU fallbacks (fail fast, fix architecture)

**INGESTION PATH** (preprocessing, one-time):
- ✅ Anything allowed (multiprocessing, numpy, pickle, etc.)
- Only runs ONCE before training starts
- NOT in the hot path

### Available PTX Kernels (Your Toolbox)

**Location**: `knowledge3d/cranium/ptx/` and `knowledge3d/cranium/kernels/`

#### 1. DCT8X8_FORWARD / DCT8X8_INVERSE
**File**: `knowledge3d/cranium/ptx/codec_ops.ptx`
**What**: 8×8 Discrete Cosine Transform (video encoding)
**Usage**: Via `SovereignTernaryVideoCodec.encode()`
**Sovereignty**: ✅ Pure PTX, no CPU fallback

#### 2. TERNARY_QUANT
**File**: `knowledge3d/cranium/ptx/codec_ops.ptx`
**What**: Quantize floats to ternary {-1, 0, +1}
**Usage**: Via RPN program `"DCT8X8_FORWARD 0.1 TERNARY_QUANT"`
**Sovereignty**: ✅ Pure PTX, no CPU fallback

#### 3. MDCT / IMDCT
**File**: `knowledge3d/cranium/ptx/codec_ops.ptx`
**What**: Modified Discrete Cosine Transform (audio encoding)
**Usage**: Via `SovereignTernaryAudioCodec` (not directly used in ARC yet)
**Sovereignty**: ✅ Pure PTX, no CPU fallback

#### 4. cosine_similarity_batch
**File**: `knowledge3d/cranium/ptx/cosine_similarity.ptx`
**What**: Batch cosine similarity (N candidates vs 1 expected)
**Usage**: Via `CosineSimilarityBridge.compute_similarities()`
**Sovereignty**: ✅ Pure PTX, no CPU fallback
**Performance**: 0.1ms for 67 candidates × 512 dims

#### 5. ModularRPNEngine (RPN Calculator)
**File**: `knowledge3d/cranium/modular_rpn_engine.py`
**What**: GPU RPN calculator (executes "3 4 ADD" → 7)
**Usage**: `rpn.evaluate("DCT8X8_FORWARD 0.1 TERNARY_QUANT", data=blocks, return_vector=True)`
**Sovereignty**: ✅ Routes to PTX kernels, no Python math
**Key method**: `evaluate(program: str, data: List[float]) -> TernaryVector`

#### 6. MathCorePool (Tier 1/2/3 GPU cores)
**File**: `knowledge3d/cranium/ptx_runtime/math_core_pool.py`
**What**: Manages GPU math cores (18 total: 9×Tier1, 6×Tier2, 3×Tier3)
**Usage**: `pool.spawn_core(tier=1, reuse=True)` → returns core_id
**Sovereignty**: ✅ Allocates GPU resources, no CPU math
**Note**: Tier 1 = 9 cores max (matches 9 workers!)

### Data Structures (Sovereign)

#### TernaryVector
**File**: `knowledge3d/cranium/ternary.py`
**What**: 1D array of {-1, 0, +1} packed as 2 bits/element
**Usage**: `TernaryVector([1, 0, -1, 1])`
**Methods**: `to_python()` → list, `to_numpy()` → np.array (only for ingestion!)
**Sovereignty**: ✅ GPU-resident, 4× memory efficient vs float32

#### TernaryTensor
**File**: `knowledge3d/cranium/ternary.py`
**What**: N-D array of {-1, 0, +1}
**Usage**: `TernaryTensor((height, width, 3), TernaryVector(data))`
**Sovereignty**: ✅ GPU-resident

#### Galaxy (Dict-based cache)
**Current**: `Dict[int, List[float]]` (embedding_galaxy in memory)
**What**: Hash → embedding cache (stays in-process memory)
**Usage**: `embedding_galaxy[hash] = embedding`
**Sovereignty**: ✅ Python dict, in-memory, never leaves process

### NO CPU FALLBACKS Policy

**WRONG** (violates sovereignty):
```python
if gpu_available:
    result = gpu_kernel(data)
else:
    result = numpy_cpu_fallback(data)  # ❌ FALLBACK!
```

**RIGHT** (sovereign):
```python
if not gpu_available:
    raise RuntimeError("GPU required. No CPU fallbacks.")  # ✅ FAIL FAST
result = gpu_kernel(data)
```

**Daniel's principle**: "Fail fast, fix architecture. No silent degradation."

**Examples of violations to AVOID**:
```python
# ❌ WRONG: Fallback to Python loops
if embedding_galaxy.get(hash) is None:
    emb = slow_python_computation(grid)  # CPU fallback!

# ✅ RIGHT: Batch lazy (still sovereign)
missing_grids = [g for g in grids if hash(g) not in galaxy]
embeddings = batch_gpu_compute(missing_grids)  # GPU batch!
for h, emb in zip(hashes, embeddings):
    galaxy[h] = emb  # Cache in-system
```

**Key difference**: Batch lazy still uses GPU (sovereign), just computes on-demand. CPU fallback uses numpy/Python loops (not sovereign).

---

## Expected Performance After Fix

### Before (Run 023 - Redundant Workers)

**Per task**:
- 9 workers × 25 candidates × 100ms = 22.5 seconds (redundant work)
- Deduplication: 225 → 3 candidates
- GPU: 1% (waiting for CPU)

**Full run** (1,620 task-epochs):
- 1,620 × 22.5 seconds = **10.3 hours**

### After (Run 024 - Partitioned Workers)

**Per task**:
- Worker 1: 4 hints → 6 candidates (100ms)
- Worker 2: 4 hints → 6 candidates (100ms)
- ...
- Worker 9: 4 hints → 6 candidates (100ms)
- **All in parallel**: 100ms total (9× speedup!)
- Deduplication: 54 → 54 candidates (no duplicates)
- Batch embeddings: 54 grids × 20ms = 20ms (GPU)
- PTX cosine: 54 candidates × 0.1ms = 0.1ms (GPU)
- **Total: ~150ms per task**

**Full run** (1,620 task-epochs):
- 1,620 × 150ms = **243 seconds = 4 minutes**

**Speedup**: 10.3 hours → 4 minutes = **154× faster**

**GPU utilization**: 1% → 15-20% (batch operations active)

---

## Implementation Steps

### Step 1: Update parallel_generator.py (30 min)

**File**: `knowledge3d/training/arc_agi/parallel_generator.py`

**Changes**:
1. Add semantic hint partitioning (lines 42-65)
2. Pass worker-specific hints to each CandidateGenerator
3. Add logging for worker assignments
4. Add pre-dedup candidate count logging

**Test**:
```python
# Should show different hints per worker
# [WORKER 0] Assigned hints 0:4 (4 hints)
# [WORKER 1] Assigned hints 5:9 (4 hints)
# ...
```

### Step 2: Verify no CPU fallbacks remain (10 min)

**Check these files**:

1. `candidate_generator.py`:
   - Line 124: `if self.embedding_galaxy is None: raise RuntimeError(...)`  ✅
   - Lines 130-150: Batch lazy GPU embeddings (no serial loops)  ✅
   - Line 519: `_rank_by_similarity()` uses Galaxy + PTX cosine  ✅

2. `sovereign_pipeline.py`:
   - No try/except fallback to sequential generation  ✅
   - Uses ParallelCandidateGenerator directly  ✅

3. `video_grid_embedder.py`:
   - Lines 91-92: Raises if non-sovereign codec  ✅
   - Line 136: Uses RPN batch evaluation (GPU)  ✅

**If ANY CPU fallback found**: Remove it and raise RuntimeError instead.

### Step 3: Compile and test (10 min)

```bash
# Compile Python modules
cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
python3 -m py_compile knowledge3d/training/arc_agi/parallel_generator.py

# Verify PTX kernels present
ls -lh knowledge3d/cranium/ptx/*.ptx
# Should show:
# codec_ops.ptx (~50 KB)
# cosine_similarity.ptx (~5 KB)
```

### Step 4: Launch Run 024 with tmux orchestration (10 min)

**CRITICAL**: Use tmux to monitor GPU + training in parallel

#### Create GPU monitor session

```bash
# Session 1: GPU monitor (real-time)
tmux new-session -d -s gpu024
tmux send-keys -t gpu024 'watch -n1 nvidia-smi' Enter
```

#### Create training session

```bash
# Session 2: Training (Run 024)
tmux new-session -d -s arc024
tmux send-keys -t arc024 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' Enter
tmux send-keys -t arc024 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/train_arc_sovereign_loop.py --arc-dirs /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/training /K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation --max-tasks 60 --epochs 27 --cycles 1 --matryoshka-dim 512 > /tmp/arc_run_024.log 2>&1' Enter
```

#### Attach to sessions for monitoring

```bash
# View GPU monitor
tmux attach -t gpu024

# Detach: Ctrl+B, then D
# View training log
tail -f /tmp/arc_run_024.log

# Or attach to training session
tmux attach -t arc024
```

#### Expected log output (Run 024)

```
[INIT] Loaded precomputed embeddings: 6836 entries
Initializing sovereign pipeline...
[LOADING] Galaxy state from checkpoints...
  Drawing shapes: 22
  Grammar rules: 221
  Shadow entries: 80

[Cycle 1/1] Epoch 1/27
  [WORKER 0] Assigned hints 0:5 (5 hints)
  [WORKER 1] Assigned hints 5:10 (5 hints)
  [WORKER 2] Assigned hints 10:15 (5 hints)
  [WORKER 3] Assigned hints 15:20 (5 hints)
  [WORKER 4] Assigned hints 20:25 (5 hints)
  [WORKER 5] Assigned hints 25:30 (5 hints)
  [WORKER 6] Assigned hints 30:35 (5 hints)
  [WORKER 7] Assigned hints 35:38 (3 hints)
  [WORKER 8] Assigned hints 38:38 (0 hints)

  [WORKER 0] Generated 6 candidates from 5 hints
  [WORKER 1] Generated 7 candidates from 5 hints
  [WORKER 2] Generated 6 candidates from 5 hints
  [WORKER 3] Generated 8 candidates from 5 hints
  [WORKER 4] Generated 5 candidates from 5 hints
  [WORKER 5] Generated 7 candidates from 5 hints
  [WORKER 6] Generated 6 candidates from 5 hints
  [WORKER 7] Generated 4 candidates from 3 hints
  [WORKER 8] Generated 3 candidates from 0 hints

  [PARALLEL GEN] Total candidates before dedup: 52
  [PARALLEL GEN] PTX success=1458, fallback=0, rate=100.0%
  [CANDIDATES] Parallel generated 48 candidates (Tesla 3-6-9)

  [GALAXY LAZY] Computing 48 missing embeddings (batch GPU)
  [ANSWER CHECK] Task 00d62c1b_e0: score=0.85, reward=NEUTRAL

  [1:1/60] 00d62c1b_e0 score=0.85 type=semantic
```

**Key indicators of success**:
1. ✅ Different hint ranges per worker (not all 0:38)
2. ✅ Different candidate counts per worker (not all 25)
3. ✅ Total before dedup ≈ 50-60 (not 225)
4. ✅ PTX success 100%, fallback 0
5. ✅ Batch GPU embeddings (not serial loops)

#### GPU monitor expectations

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01   Driver Version: 535.183.01   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... Off  | 00000000:01:00.0  On |                  N/A |
|  0%   45C    P2    45W / 170W |    156MiB / 12288MiB |     18%      Default |
+-------------------------------+----------------------+----------------------+
```

**Expected**:
- GPU Util: 15-25% (batch operations, up from 1%)
- Memory: 150-180 MiB (Galaxy + PTX modules + RPN state)
- Power: 40-60W (active computation, up from ~20W idle)
- Temp: 40-50°C (normal load)

**If GPU still at 1%**: Something is still using CPU loops. Check logs for missing "[GALAXY LAZY]" or high "[SEMANTIC GEN]" repetition.

---

## Success Criteria

**Must Have**:
1. ✅ Different hint assignments per worker (check logs)
2. ✅ ~50-60 candidates total before dedup (not 225)
3. ✅ PTX success 100%, fallback 0 (sovereignty maintained)
4. ✅ Run 024 completes in <10 minutes (not hours)
5. ✅ GPU utilization 15-25% (not 1%)
6. ✅ Accuracy ≥ 1.67% (Run 023 baseline)

**Nice to Have**:
- Accuracy > 2% (better candidates from diverse workers)
- Library growth: shapes +1, rules +1, shadow +1
- GPU memory < 200 MiB

---

## Troubleshooting

### Issue: Still seeing redundant candidates

**Symptom**: Logs show same candidate count 9 times
```
[WORKER 0] Generated 25 candidates from 38 hints
[WORKER 1] Generated 25 candidates from 38 hints
...
```

**Cause**: Partitioning not applied or all workers getting all hints

**Fix**: Verify `semantic_partitions` is being created and passed correctly in parallel_generator.py

### Issue: PTX fallback > 0

**Symptom**: `[PARALLEL GEN] PTX success=1200, fallback=12, rate=99.0%`

**Cause**: Some operation falling back to CPU

**Fix**: Search codebase for numpy operations in hot path. Add `raise RuntimeError()` if found.

### Issue: GPU still at 1%

**Symptom**: nvidia-smi shows 1% utilization

**Cause**: Still using serial Python loops somewhere

**Fix**: Check for missing batch operations:
- Embeddings should be batched: `_grid_to_spatial_embedding_batch()`
- Cosine should use PTX: `cosine_bridge.compute_similarities()`

### Issue: Run crashes with "embedding not found"

**Symptom**: `RuntimeError: SOVEREIGNTY VIOLATION: Candidate embedding not found`

**Cause**: Generated candidate not in preprocessing set

**Fix**: This is EXPECTED behavior! Batch lazy will compute missing embeddings:
- Check log for `[GALAXY LAZY] Computing N missing embeddings (batch GPU)`
- If missing, verify batch lazy code is present (candidate_generator.py lines 124-150)

---

## Codex: Your Mission

You are implementing Run 024 with these objectives:

1. **Fix worker redundancy** (partition semantic hints across 9 workers)
2. **Maintain sovereignty** (PTX + RPN only, NO CPU fallbacks)
3. **Launch training immediately** (tmux sessions, start run)
4. **Monitor and report** (GPU %, candidate counts, runtime)

**Sovereignty checklist**:
- [x] PTX kernels for math (DCT, TERNARY_QUANT, cosine)
- [x] RPN for GPU operations (ModularRPNEngine)
- [x] Batch operations (no serial Python loops)
- [x] Fail fast (raise RuntimeError if broken, no fallbacks)
- [x] Galaxy caching (data stays in-system)

**Implementation order**:
1. Update `parallel_generator.py` (partition hints)
2. Verify no CPU fallbacks (check all 3 files)
3. Compile modules (py_compile)
4. Launch tmux sessions (gpu024 + arc024)
5. Monitor logs for success indicators
6. Report metrics when complete

**Timeline**:
- Implementation: 30 min
- Verification: 10 min
- Launch: 5 min
- Training: 4-10 min (Run 024 should be FAST)
- **Total: <1 hour from start to results**

**Expected outcome**:
- Run 024 completes in 4-10 minutes (154× faster than Run 023)
- GPU 15-25% (not 1%)
- Accuracy ≥ 1.67%, ideally > 2%
- PTX success 100%, fallback 0

**Start NOW. Read everything above, implement the fix, launch training, report results.**

---

**END OF LAUNCH SPECIFICATION**

Claude (Architecture Partner)
November 28, 2025
