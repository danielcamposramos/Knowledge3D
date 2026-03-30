# Claude — Phase E.37: Hardware-Adaptive CPU Parallelization

**Date:** 2026-03-30
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** HIGH — 48.7 min for 100 questions is too slow

---

## Daniel's Direction

> "distribute the load to the 12 cores we have (and as many as the running
> hardware can provide — detect and maximize with a one time config run and a
> simple check before launching — so when we migrate it can adapt)"
>
> "where can we do async and/or parallel (can we do it also inside each process?)"

---

## Current Problem

The full benchmark (`run_full_benchmark.py`) runs suites SEQUENTIALLY:
MMLU → GSM8K → LHE → ARC2 → ARC3 Local. Total: 2923s (48.7 min) for 100 questions.

The RTX 3070 reports 100% GPU during hot path — but the 12 CPU cores are idle.
The bottleneck is the Python orchestration between GPU calls: parsing, routing,
logging, result extraction. This is all single-threaded.

---

## Architecture: Two Levels of Parallelism

### Level 1: Inter-Suite Parallelism (Across Cores)

Run benchmark suites in parallel processes. Each process gets its own
Knowledgeverse instance sharing the same cached GPU buffers (read-only VRAM).

**Critical constraint:** GPU memory is shared. Multiple processes submitting GPU
work simultaneously is fine (CUDA handles scheduling), but each process must NOT
allocate its own copy of the 278K-entry GPU buffer. The flat buffer and CSR graph
are read-only after init — share them via the existing cache files.

```
Process 0: MMLU (20 questions)     → CPU core 0 + GPU work queue
Process 1: GSM8K (20 questions)    → CPU core 1 + GPU work queue
Process 2: LHE (20 questions)      → CPU core 2 + GPU work queue
Process 3: ARC2 (20 questions)     → CPU core 3 + GPU work queue
Process 4: ARC3 Local (20 tasks)   → CPU core 4 + GPU work queue
```

### Level 2: Intra-Suite Parallelism (Within Each Process)

Within each benchmark suite, questions are independent. Use async I/O and
thread pools for the CPU-bound parsing/routing work:

- Parse question text → extract quantities → build query embedding (CPU)
- Submit GPU work → wait for result (GPU — async via CUDA streams)
- Compare answer → log result (CPU)

The CPU work between GPU calls can overlap with GPU execution of the
previous question.

---

## Hardware Detection (One-Time Config)

### Step 1: Detect and Cache Hardware Profile

Create a one-time hardware detection that runs ONCE and caches results:

```python
# knowledge3d/cranium/hardware_profile.py
# Runs once, caches to /K3D/Knowledge3D.local/hardware_profile.json

{
    "cpu_cores_physical": 6,        # os.cpu_count() // 2 or psutil
    "cpu_cores_logical": 12,        # os.cpu_count()
    "gpu_name": "NVIDIA GeForce RTX 3070",
    "gpu_vram_total_mb": 8192,
    "gpu_vram_free_mb": 5565,       # at detection time
    "gpu_sm_count": 46,
    "optimal_worker_count": 5,      # min(suites, cores_physical - 1)
    "optimal_gpu_streams": 4,       # heuristic based on SM count
    "detected_at": "2026-03-30T15:30:00",
    "hostname": "daniel-workstation"
}
```

**Detection logic:**
- `cpu_cores_physical`: Leave 1 physical core free for OS/display
- `optimal_worker_count`: `min(num_suites, physical_cores - 1)`
- `optimal_gpu_streams`: `min(4, sm_count // 12)` — don't over-subscribe SMs
- Cache as JSON — re-detect only if hostname changes or file is >24h old

### Step 2: Simple Check Before Launch

Before running benchmarks, read cached profile. If file exists and is fresh,
use it. If stale or missing, re-detect (takes <1s).

```python
profile = load_or_detect_hardware_profile()
worker_count = profile["optimal_worker_count"]
```

This adapts automatically when migrated to different hardware.

---

## Implementation Plan

### File: `knowledge3d/cranium/hardware_profile.py`

~50 lines. Detect CPU cores (os.cpu_count), GPU info (via existing
sovereign_loader or cupy), cache as JSON.

### File: `scripts/run_full_benchmark.py` Changes

Replace sequential suite loop with `multiprocessing.Pool`:

```python
# Current (sequential):
for suite_name, suite_count in suite_order:
    result = _run_native_suite(...)

# Target (parallel):
from multiprocessing import Pool
profile = load_or_detect_hardware_profile()
with Pool(processes=profile["optimal_worker_count"]) as pool:
    results = pool.starmap(_run_native_suite_worker, suite_args)
```

Each worker:
1. Creates its own Knowledgeverse (uses shared cached GPU buffers)
2. Runs its suite
3. Streams JSONL to its own log file
4. Returns result dict

Main process:
1. Collects all results
2. Writes summary.json
3. Runs post-benchmark consolidation

### Intra-Process Overlap

Within each suite worker, overlap CPU prep with GPU execution:

```python
# Prepare next question while GPU processes current one
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    # Thread 1: GPU work (current question)
    # Thread 2: CPU prep (next question's query text + embedding)
```

This is safe because:
- GPU work is submitted via CUDA driver (thread-safe)
- CPU prep is pure Python (no shared mutable state)
- Each suite's questions are independent

---

## What NOT to Do

- Do NOT use multiprocessing for GPU kernel launches (CUDA contexts don't fork well)
  — use the existing Knowledgeverse init per process which creates its own CUDA context
- Do NOT add process-level parallelism WITHIN a single suite's questions
  (GPU saturation from one process is already at 100%)
- Do NOT hardcode core count — always read from hardware profile
- Do NOT use asyncio for GPU work — CUDA is synchronous per stream, use threads

---

## Success Criteria

- [ ] Hardware profile detected and cached at `/K3D/Knowledge3D.local/hardware_profile.json`
- [ ] Profile re-read on subsequent runs (< 1ms check)
- [ ] 5 suites run in parallel (one per physical core)
- [ ] Total benchmark time reduced (target: < 15 min for same 100 questions)
- [ ] Each suite's JSONL streaming still works (separate files)
- [ ] summary.json still aggregates all results
- [ ] Works correctly on machines with different core counts

---

## Spec Grounding

- **SGI Spec §2.1**: "Zero external dependencies" — use os.cpu_count(), not psutil
- **Three Brain System §3.3**: Cranium atomic operations are thread-safe per CUDA context
- **Knowledgeverse Spec**: GPU flat buffer is read-only after init — safe to share
