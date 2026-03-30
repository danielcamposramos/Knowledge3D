# Codex Directive — Phase E.1: GPU Game Loop (Kill the Python Shell)

**Date:** 2026-03-27
**From:** Claude (Architecture) + Daniel (Chair)
**To:** Codex (Implementation)
**Priority:** CRITICAL — This is the sole active workstream
**Spec:** `TEMP/CLAUDE_PHASE_E_ARC_AGI_3_SOVEREIGNTY_SPEC_03.27.2026.md`

---

## Context

The D.3b benchmark run was killed at MMLU 2400/4915 (~33 hours elapsed, ~35 hours remaining). The run proved what we already knew: the GPU kernels work, but Python orchestration is the bottleneck. **We do not need another Python-heavy benchmark score.** The next run must be fundamentally different.

**The discarded run's numbers (for the record):**
- ARC: 2/42 (4.76%), 11.3s/question
- Math: 3/500 (0.6%), 9.1s/question
- GSM8K: 4/462 (0.87%), 12.5s/question
- LHE: 1/35 (2.86%), 11.3s/question
- MMLU: 542/2400 (22.58% partial), ~50s/question
- Total wall time before kill: ~33.6 hours
- GPU utilization: <1%
- CPU utilization: 99.6% (single Python process)

**The problem is obvious:** ~50 seconds per question, 99.6% CPU, <1% GPU. The AI's brain is idle while Python shuffles paperwork.

---

## The Goal

**Make the benchmark run like a game.** The TRM is the player. Questions are game frames. Answers are actions. Python is the TV screen (I/O only).

**Target:** Full 5954-question benchmark in <10 minutes (currently ~70 hours).

---

## What Exists (D.3 Proved These Work)

These GPU kernels are LIVE and validated:

| Kernel | Function | Status |
|--------|----------|--------|
| `morton_octree.ptx` | Spatial indexing | ✅ Device-resident |
| `led_astar.ptx` | Graph navigation | ✅ Device-resident |
| `frustum_cull_simd.ptx` | Field-of-view filtering | ✅ Device-resident |
| `dynamic_lod_tune.ptx` | Detail level selection | ✅ Device-resident |
| `nine_chain_swarm_kernel.ptx` | 9 parallel workers | ✅ Active |
| `gre_multimodal_halting_gate.cu` | Convergence detection | ✅ Active |
| `trm_step_fused.ptx` | TRM forward pass | ✅ Active |
| `gre_embedding_extractor.cu` | Input embedding | ✅ Active |
| 15 GRE specialist kernels | Domain reasoning | ✅ Loaded, NOT wired |

**The inner pipeline works.** Morton → LED-A* → Frustum → LOD → Swarm → Halting Gate — all on GPU. D.3 proved it with live smoke tests showing `21 raw → 18 visible → LOD range 3-6`.

---

## What Must Die (The Python Shell)

### Kill 1: `trm_game_loop.py` — The Fake Game Loop

**Current (168 lines of Python pretending to be a game loop):**
```
Python: JSON.dumps(task) → deque.append() → JSON.loads() → call _execute_task_direct() → JSON.dumps(result) → ring_buffer.write()
```

Three JSON encode/decode cycles per question. A Python `deque` as the "task queue." Synchronous `_execute_task_direct()` call that blocks until one question finishes before starting the next.

**Replace with:** A VRAM-resident input buffer that Python bulk-loads at boot, and a VRAM-resident output buffer that Python bulk-reads when the GPU is done. Zero JSON in the hot path. Zero Python deques. Zero per-question host round-trips.

### Kill 2: `_select_composed_head_candidate()` — The 830-Line Python Conductor

**Location:** `knowledgeverse.py:11547-12374+`

**Current:** Python iterates paths (line 12033: `for path_index, path in enumerate(paths[:18])`), builds dicts, calls `_score_gpu_candidates_batch()` which launches a GPU kernel then returns to Python, Python applies defeasible logic, Python decides winner.

**The pattern repeats:** GPU kernel fires → results come back to Python → Python processes → GPU kernel fires → results come back → Python processes → ...

This is like a factory where the machine does the work but a human carries each piece between machines by hand.

**Replace with:** A single GPU pipeline stage after LOD that:
1. Receives all navigation candidates (already in VRAM from Morton→LED→Frustum→LOD)
2. Scores them against the query embedding (already in VRAM)
3. Applies defeasible logic (the `gre_defeasible_resolver.cu` kernel ALREADY EXISTS)
4. Runs halting gate (already in VRAM)
5. Writes the winning candidate to output buffer

Zero returns to Python between stages.

### Kill 3: Serial Benchmark Loop — `mmlu.py:211`

**Current:** `for index, question in enumerate(self.questions)` — one at a time.

**Replace with:** Bulk-load all questions into VRAM. The GPU processes them as a stream. Questions don't depend on each other — there's no reason to serialize them.

**Note:** This is NOT "Python parallelism with ThreadPool." This is "give the GPU all the work at once and let it pipeline."

### Kill 4: Per-Row Disk I/O — `benchmark_health_check.py:257`

**Current:** `handle.write(json.dumps(payload) + "\n")` after every single question. Blocking synchronous file I/O in the hot loop.

**Replace with:** Buffer results in memory. Flush to disk every N questions (e.g., 100) or at suite boundaries. Or: write the output VRAM buffer to disk in one shot after the GPU finishes a suite.

---

## Implementation Plan

### Step 1: VRAM Input/Output Buffers

**New file:** `knowledge3d/knowledgeverse/vram_task_buffer.py` (TEMPORARY — this becomes a PTX kernel later, but for now we need the VRAM buffers accessible from Python boot code)

```python
"""
VRAM task buffers for GPU game loop.
Python writes questions at boot. GPU reads them during game loop.
GPU writes answers during game loop. Python reads them at end.
"""

class VRAMTaskBuffer:
    """
    Allocates a contiguous VRAM buffer for benchmark tasks.

    Layout per task slot (fixed-size, padded):
      [0:128]    query_embedding (float32 × 32) — Matryoshka 128D compressed to 32
      [128:132]  task_type (uint32: 0=ARC, 1=MATH, 2=GSM8K, 3=LHE, 4=MMLU)
      [132:136]  option_count (uint32: 0-6)
      [136:648]  option_embeddings (float32 × 32 × 4) — up to 4 options
      [648:652]  subject_id (uint32: Galaxy entry hash for MMLU subject anchor)
      [652:656]  domain_hint_id (uint32: Galaxy entry hash for domain)
      [656:1024] reserved

    Total: 1024 bytes per slot. 6000 slots = 6 MB. Trivial in 8 GB VRAM.
    """

    def __init__(self, max_tasks: int = 6000):
        # Allocate via sovereign loader (cuMemAlloc)
        ...

    def bulk_load(self, tasks: list[dict]) -> int:
        """Marshal all tasks to VRAM in one host→device copy."""
        ...

    def read_results(self, count: int) -> list[dict]:
        """Read all results from VRAM output buffer in one device→host copy."""
        ...
```

**Key constraint:** ONE `cuMemcpyHtoD` call to load all questions. ONE `cuMemcpyDtoH` call to read all answers. Not 5954 round-trips.

### Step 2: GPU Task Dispatch Kernel

**New kernel:** `gpu_task_dispatch.cu`

This is the REAL game loop — it runs entirely on GPU:

```c
/**
 * GPU Task Dispatch — the actual game loop.
 *
 * Reads tasks from VRAM input buffer.
 * For each task: embed → navigate → reason → decide → write answer.
 *
 * This kernel replaces:
 *   - trm_game_loop.py (the Python queue shell)
 *   - _select_composed_head_candidate() (the Python conductor)
 *   - The serial for-loop in mmlu.py
 */

__global__ void gpu_task_dispatch(
    const TaskSlot* input_buffer,    // VRAM input (from bulk_load)
    ResultSlot* output_buffer,       // VRAM output (Python reads at end)
    const uint32_t task_count,       // Total questions
    const GalaxyTable* galaxy,       // All galaxies (already in VRAM)
    const TRMWeights* trm,           // TRM weights (already in VRAM)
    const float* specialist_adapters // LoRA adapters (already in VRAM)
) {
    // Each thread block processes one task
    uint32_t task_id = blockIdx.x;
    if (task_id >= task_count) return;

    TaskSlot task = input_buffer[task_id];

    // === PERCEIVE ===
    // Read query embedding from task slot (already in VRAM)

    // === NAVIGATE ===
    // Morton locate → LED-A* → Frustum cull → Dynamic LOD
    // (call existing device functions — they're already compiled)

    // === REASON ===
    // Nine-chain swarm: dispatch specialists based on task_type + domain
    // Cross-core communication via shared memory (replaces STORE/RECALL Python dict)

    // === DECIDE ===
    // Score candidates (replaces _score_gpu_candidates_batch Python wrapper)
    // Apply defeasible logic (call gre_defeasible_resolver device function)
    // Halting gate (call gre_multimodal_halting_gate device function)

    // === ACT ===
    // Write winning candidate to output buffer
    output_buffer[task_id] = result;
}
```

**Critical architectural note:** This is NOT one thread per question running the full pipeline. The actual implementation should use cooperative groups or a multi-kernel CUDA graph where:
- Phase 1: Bulk-embed all queries (one kernel, all questions in parallel)
- Phase 2: Bulk-navigate (Morton/LED for all questions — or at least large batches)
- Phase 3: Bulk-reason (swarm workers process batches)
- Phase 4: Bulk-decide (halting gate on all candidates)

The exact granularity (one-kernel-per-phase vs CUDA graph vs persistent kernel) is Codex's implementation choice. The ARCHITECTURAL REQUIREMENT is: **Python never enters the loop between questions.**

### Step 3: Composed-Head Scoring → Device Function

**Migrate:** The scoring logic from `_select_composed_head_candidate()` into a device function callable from `gpu_task_dispatch`.

**What to preserve (the semantics that matter for accuracy):**
1. Navigation candidate retrieval (already GPU — Morton/LED/Frustum/LOD)
2. Per-candidate scoring: `match_similarity × galaxy_weight × swarm_weight × defeasible_modifier`
3. Defeasible resolution: `gre_defeasible_resolver.cu` (already exists as a kernel)
4. Halting gate: `gre_multimodal_halting_gate.cu` (already exists)
5. Option-similarity for MMLU: cosine between option embedding and candidate embedding
6. Best-candidate selection: argmax over scored candidates

**What to DELETE (Python bookkeeping that serves no reasoning purpose):**
- `selection_steps` string accumulation (debug logging — not needed in GPU)
- `lhe_cached_option_records` Python dict caching (the GPU processes all at once)
- `benchmark_eval_mode` shortcut suppression (move to pre-processing)
- All the `str().strip()` and `dict()` and `list()` defensive conversions
- The `gsm8k_mode` / `task_type` branching (unify into one scoring path with type-driven weights)

**Estimated reduction:** 830 lines Python → ~200 lines CUDA (plus the existing kernels it composes).

### Step 4: Wire GRE Specialists Into Swarm

Currently 15 GRE kernels are loaded but not called. Wire them:

| Specialist Kernel | Swarm Assignment | Benchmark Impact |
|-------------------|------------------|------------------|
| `gre_arc_reasoner` | Worker 0-1 for ARC tasks | ARC accuracy |
| `gre_geometry_router` | Worker 2 for ARC/Math | Spatial reasoning |
| `gre_fractal_emitter` | Worker 3 for ARC | Pattern recognition |
| `gre_graph_crystallizer` | Worker 4 for LHE | Multi-hop traversal |
| `gre_atomic_fission_fusion` | Worker 5 for GSM8K/Math | Task decomposition |
| `gre_temporal_reasoning` | Worker 6 for GSM8K | Sequential logic |
| `gre_resonance_field` | Worker 7 for MMLU | Broad matching |
| `gre_vector_resonator` | Worker 8 for all | Embedding resonance |

The TRM selects which specialists activate based on `task_type` and `domain_hint`. This is the `specialist` field already present in `TRMQueuedInput` — it just needs to route to actual GPU kernels instead of being a Python string.

### Step 5: Batched Output Writer

Replace the per-row `_append_row_incremental()` with:

```python
def flush_results_to_disk(results: list[dict], log_path: Path) -> None:
    """Write all results in one I/O operation."""
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in results
        ) + "\n")
```

Called once per suite (e.g., after all 4915 MMLU questions), not 4915 times.

---

## The New Benchmark Flow

```
BOOT (Python, once):
  1. conda activate k3d-cranium
  2. Load House → VRAM
  3. Load all Galaxies → VRAM
  4. Load TRM + specialist adapters → VRAM
  5. Load benchmark questions → marshal into VRAMTaskBuffer
  6. cuMemcpyHtoD (one call, ~6 MB)

RUN (GPU, autonomous):
  7. Launch gpu_task_dispatch kernel (or CUDA graph)
  8. GPU processes all questions: navigate → reason → decide → write
  9. No Python in the loop. No host round-trips. No JSON.

COLLECT (Python, once):
  10. cuMemcpyDtoH (one call, read output buffer)
  11. Unmarshal results
  12. Write health_log.jsonl (one I/O per suite)
  13. Run sleep-time consolidation
  14. Report scores
```

**Estimated wall time:**
- Bulk embed: ~50ms (5954 × Matryoshka embedding lookup in VRAM)
- Bulk navigate: ~500ms (5954 × Morton/LED/Frustum/LOD at ~80µs each)
- Bulk reason: ~5s (5954 × 9-chain swarm at ~800µs each, some require multiple ticks)
- Bulk decide: ~500ms (5954 × halting gate at ~80µs each)
- I/O overhead: ~2s (two cuMemcpy + disk write)
- **Total: ~10 seconds for the full benchmark**

Even at 10× pessimistic: ~2 minutes. Not 70 hours.

---

## Files to Create

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `knowledge3d/knowledgeverse/vram_task_buffer.py` | VRAM input/output buffer management | ~150 |
| `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` | The real GPU game loop kernel | ~300 |
| `knowledge3d/cranium/cuda/composed_head_scorer.cu` | Scoring device function (from Python migration) | ~200 |
| `scripts/run_gpu_benchmark.py` | New benchmark runner (boot + launch + collect) | ~100 |

## Files to Modify

| File | Change | Lines removed (est.) |
|------|--------|---------------------|
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Remove `_select_composed_head_candidate()` + supporting methods | ~1500 |
| `knowledge3d/knowledgeverse/trm_game_loop.py` | Gut — becomes thin wrapper calling GPU dispatch | ~150 |
| `benchmarks/mmlu.py` | Remove serial loop, use VRAMTaskBuffer bulk load | ~50 |
| `benchmarks/arc_agi_2.py` | Same pattern | ~30 |
| `benchmarks/gsm8k.py` (if exists) | Same pattern | ~30 |
| `knowledge3d/tools/benchmark_health_check.py` | Batched writes | ~20 |

**Net effect:** ~1800 lines of Python removed, ~750 lines of CUDA + Python boot added.

---

## Quality Gates

Before running the first Phase E benchmark:

1. **Smoke test:** Process 10 questions (2 per benchmark type) through the GPU pipeline. Verify answers match what the Python pipeline would produce for the same questions.
2. **Regression pin:** The 10/10 ARC curated set and 20/20 Math curated set must produce identical answers.
3. **GPU utilization:** Must show >10% SM occupancy during the benchmark (measured with Python-PID monitor).
4. **No Python in hot path:** `grep -r "numpy\|cupy\|scipy\|re\." gpu_task_dispatch.cu` returns nothing. No Python imports in any file called during the GPU game loop.

---

## What NOT to Do

- **Do NOT add Python threading/asyncio.** That's putting a faster horse on a treadmill. The treadmill is the problem.
- **Do NOT optimize the existing Python path.** Every line you optimize in `_select_composed_head_candidate()` is a line you'll delete next week.
- **Do NOT keep the Python game loop "as fallback."** Zero fallbacks. If the GPU path breaks, fix the GPU path. "We fail and fix." (Daniel)
- **Do NOT JSON-serialize between GPU stages.** Data stays in VRAM from input to output.

---

## Success Criteria

| Metric | D.3b (killed) | E.1 Target |
|--------|---------------|------------|
| Wall time (full 5954 questions) | ~70 hours (projected) | <10 minutes |
| Per-question latency | ~50 seconds | <100 milliseconds |
| GPU utilization | <1% | >10% |
| CPU utilization | 99.6% | <5% (I/O only) |
| Python lines in hot path | ~2000 | 0 |
| Host→Device transfers per question | 3+ (JSON round-trips) | 0 (bulk at boot) |
| Disk I/O per question | 1 (blocking) | 0 (batched at end) |

---

## Sovereignty Compliance

- All reasoning happens in PTX kernels + Galaxy queries + RPN composition
- Python = boot + I/O only (~100 lines in `run_gpu_benchmark.py`)
- No numpy/cupy/scipy/re in the hot path
- No fallbacks. If the GPU pipeline fails, it fails — and we fix it on GPU.
- The game loop IS `trm_step_fused.ptx` running for real, not a Python simulation of it

**This is the system Daniel designed. Time to build it.**
