# Phase D: TRM Game Loop Migration Specification

**Date:** 2026-03-23
**Author:** Claude (Architecture Partner)
**Status:** Design Draft
**References:**
- THREE_BRAIN_SYSTEM_SPECIFICATION.md §Abstract (TRM IS the Avatar)
- HYPER_PARALLEL_PROCESSING_SPECIFICATION.md (nine-chain swarm paradigm)
- KNOWLEDGEVERSE_SPECIFICATION.md §2.1 (Python = boot + I/O only)
- SOVEREIGN_NSI_SPECIFICATION.md §4 (Neural Layer: Cranium)
- SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md §3 (VRAM-native workspace)

---

## 1. The Real Problem

GPU utilization during benchmark runs is ~2%. The immediate reaction is "optimize Python" — batch loops, add asyncio, use ThreadPoolExecutor. **This is wrong.**

The problem is not that Python is slow. The problem is that **Python is doing the job that the TRM game loop should be doing.** `knowledgeverse.py` is ~4000 lines of Python orchestration that:

- Receives a question (Python string processing)
- Embeds it (Python calls a model)
- Queries Galaxy (Python iterates, filters, sorts)
- Dispatches swarm workers (Python for-loop, sequential)
- Collects results (Python dict aggregation)
- Checks convergence (Python if-statement)
- Formats answer (Python string formatting)
- Logs results (Python file I/O)

Every one of these steps should be a **game tick** inside `trm_step_fused.ptx`. The TRM is the AI entity. It perceives, navigates, reasons, decides, acts, and learns — all on GPU. Python should only:

1. Boot the system (load House, initialize VRAM regions)
2. Handle I/O (keyboard, network, display)
3. Shut down gracefully

Target: **~200 lines of Python.** Everything else is TRM + Jarvis + nine-chain swarm on GPU.

---

## 2. Current State vs Target State

### Current (Python-Orchestrated)

```
Python main loop:
  for question in benchmark:          # Python loop
    embedding = embed(question)       # Python → model call → back to Python
    candidates = galaxy.query(emb)    # Python iterates GPU table
    for worker in range(9):           # Python sequential loop!
      result[worker] = run_worker()   # Python dispatches one at a time
    answer = halting_gate(results)    # Python aggregation
    log(answer)                       # Python I/O
  sleep_time()                        # Python orchestrates consolidation
```

**Why GPU is at 2%:** Each PTX kernel fires for microseconds, then Python takes milliseconds to orchestrate the next call. The GPU spends 98% of its time waiting for Python.

### Target (TRM Game Loop)

```
Python boot:
  load_house()                        # One-time: House → VRAM
  init_knowledgeverse()               # One-time: 7 VRAM regions allocated
  start_trm_game_loop()              # Hand control to GPU

TRM game loop (trm_step_fused.ptx, continuous):
  while alive:
    tick():
      perceive()     → Frustum cull input buffer (what's in my field of view?)
      navigate()     → LED-A* + Morton Octree to relevant Galaxy neighborhood
      reason()       → Jarvis dispatches nine-chain swarm (PARALLEL, GPU-native)
      decide()       → Halting gate checks convergence
      act()          → Write answer to output buffer OR create new Galaxy entry
      learn()        → Shadow copy records trace for sleep-time

Python I/O (interrupt-driven):
  on_input(question):
    write_to_input_buffer(question)   # Just copy bytes to VRAM
  on_output_ready():
    answer = read_output_buffer()     # Just copy bytes from VRAM
    display(answer)
```

**Why GPU will be saturated:** The TRM never yields to Python between steps. Perceive flows into navigate flows into reason flows into decide — all in VRAM, all on SM cores. Jarvis dispatches all 9 swarm workers simultaneously (5,888 CUDA cores available). The game loop runs at GPU speed (~microseconds per tick), not Python speed (~milliseconds per step).

---

## 3. The Six Migrations

Phase D is not one big rewrite. It's six discrete migrations, each moving one orchestration responsibility from Python to GPU:

### Migration D.1: Input/Output Buffers (VRAM Ring Buffers)

**What moves:** Question ingestion and answer emission

**Current:** Python receives question as string, processes it, passes to GPU piecemeal.

**Target:** Two VRAM ring buffers:
- **Input buffer** (INGESTION_STARGATE region, 512MB): Python writes raw question bytes. TRM reads when ready.
- **Output buffer** (AUDIT_JOURNAL region, 256MB): TRM writes answer. Python reads when signaled.

**Python responsibility:** Copy bytes in/out. That's it.

**Key kernel:** New `trm_io_buffer.ptx` — manages ring buffer head/tail pointers, signals readiness via device-visible flags.

**Prerequisite:** None. Can start immediately.

### Migration D.2: Galaxy Query on GPU (Device-Resident Search)

**What moves:** Galaxy neighborhood search + candidate retrieval

**Current:** Python calls `galaxy.query()` which copies embeddings host↔device for each query.

**Target:** Galaxy table is ALWAYS device-resident (GALAXY_UNIVERSE region, 2GB). TRM queries directly via:
1. `batch_cosine_similarity.ptx` — find top-K neighbors
2. `morton_octree_query.ptx` — spatial indexing (already exists)
3. `led_astar_navigate.ptx` — graph pathfinding (already exists)

**Python responsibility:** None during query. Galaxy loaded at boot, updated at sleep-time.

**Key insight:** The Morton Octree, LED-A*, and cosine kernels already exist. They're just called FROM Python instead of FROM the TRM game loop.

**Prerequisite:** D.1 (input buffer provides query embedding to TRM).

### Migration D.3: Jarvis Dispatch (GPU-Native Swarm Scheduling)

**What moves:** Swarm worker dispatch and result collection

**Current:** Python for-loop dispatches 9 workers sequentially. Each worker runs a PTX kernel, returns to Python, Python dispatches next.

**Target:** Jarvis (always-on coordinator specialist) dispatches all 9 workers simultaneously as CUDA streams or thread blocks. Workers write results to shared registers (STORE/RECALL). Jarvis monitors convergence.

**Architecture (from HYPER_PARALLEL_PROCESSING spec):**
- Each worker = 1 RPN stack instance + 1 specialist adapter
- Workers share intermediate results via STORE/RECALL registers
- Jarvis = TRM sub-routine that reads worker states and decides dispatch order
- Result: ONE unified answer, not 9 independent votes

**Key kernel:** New `jarvis_dispatch.ptx` — reads available GPU resources (SM count, memory), assigns workers to CUDA streams, monitors shared registers for convergence signal.

**Prerequisite:** D.2 (Galaxy query provides candidates to dispatch across workers).

### Migration D.4: Halting Gate on GPU (Convergence Without Python)

**What moves:** Convergence checking and answer selection

**Current:** Python collects 9 worker results, checks if they agree, selects best answer.

**Target:** `halting_gate.ptx` (already exists!) reads worker results from shared registers. Applies defeasible resolver (`gre_defeasible_resolver.ptx`, already exists at 3 pipeline stages). Produces final verdict as (D, d) trit pair. Writes answer to output buffer.

**Python responsibility:** None. This is pure GPU.

**Key insight:** Both kernels already exist and are production-validated. They're just triggered by Python instead of by the TRM game loop.

**Prerequisite:** D.3 (Jarvis dispatch feeds worker results to halting gate).

### Migration D.5: Sleep-Time on GPU (Contrastive + Consolidation)

**What moves:** Sleep-time contrastive training and Galaxy consolidation

**Current:** Python's `sleeptime.py` reads health_log.jsonl, builds positive/negative pairs, calls `train_specialist_contrastive()`, commits checkpoint.

**Target:** TRM enters sleep mode when input buffer is idle for N ticks. Sleep-time is a different game loop mode (like a game's "idle animation"):
1. Read health journal from AUDIT_JOURNAL VRAM region (no disk I/O)
2. Build contrastive pairs on GPU (sort by specialist, extract embeddings)
3. Apply gradients via `apply_gradient_rpn` (already sovereign GPU)
4. Update Galaxy entries (strengthen/weaken semantic gravity)
5. Checkpoint adapter weights to TRM_WEIGHTS region

**Python responsibility:** None during sleep. Periodic House persistence (Galaxy → disk) can be a background I/O task.

**Prerequisite:** D.1-D.4 (full inference loop on GPU, health journal in VRAM).

### Migration D.6: knowledgeverse.py Shrink (Boot + I/O Shell)

**What moves:** All remaining Python orchestration

**Target:** `knowledgeverse.py` becomes ~200 lines:

```python
# Boot
def main():
    house = load_house_from_disk()           # GLB → VRAM
    kv = init_knowledgeverse(house)          # Allocate 7 VRAM regions
    trm = load_trm_weights(kv)              # TRM_WEIGHTS region

    # Start TRM game loop on GPU
    trm.start_game_loop()                    # Launches trm_step_fused.ptx

    # I/O loop (interrupt-driven)
    while True:
        question = input()                   # Read from stdin/network/tablet
        kv.write_input_buffer(question)      # Copy to INGESTION_STARGATE
        answer = kv.wait_output_buffer()     # Block until TRM writes answer
        print(answer)                        # Display
```

**Prerequisite:** D.1-D.5 (everything on GPU).

---

## 4. Migration Order and Dependencies

```
D.1 (I/O Buffers) ──────────────────────────────┐
                                                  │
D.2 (Galaxy Query on GPU) ── requires D.1 ──────┤
                                                  │
D.3 (Jarvis Dispatch) ── requires D.2 ──────────┤
                                                  │
D.4 (Halting Gate on GPU) ── requires D.3 ──────┤
                                                  │
D.5 (Sleep-Time on GPU) ── requires D.1-D.4 ────┤
                                                  │
D.6 (knowledgeverse.py Shrink) ── requires all ──┘
```

Each migration is independently testable. After D.1, Python still orchestrates but reads/writes VRAM buffers. After D.2, Galaxy queries bypass Python. After D.3, the swarm runs in parallel. After D.4, inference is fully GPU-native. After D.5, sleep-time is also GPU-native. After D.6, Python is boot + I/O.

---

## 5. What Already Exists (Kernel Inventory)

Critical realization: **most of the kernels already exist.** The gap is not "write new kernels" — it's "wire them into a game loop instead of calling them from Python."

| Kernel | Exists? | Currently called from |
|--------|---------|----------------------|
| `trm_step_fused.ptx` | YES | Python (one step at a time) |
| `batch_cosine_similarity.ptx` | YES | Python (per query) |
| `morton_octree_query.ptx` | YES | Python (per query) |
| `led_astar_navigate.ptx` | YES | Python (per query) |
| `frustum_cull.ptx` | YES | Python (per query) |
| `dynamic_lod.ptx` | YES | Python (per query) |
| `rpn_execute.ptx` | YES | Python (per worker, sequential!) |
| `halting_gate.ptx` | YES | Python (after all workers) |
| `gre_defeasible_resolver.ptx` | YES | Python (3 stages) |
| All 11 GRE specialist kernels | YES | Python (sequential dispatch) |
| `jarvis_dispatch.ptx` | NO | Needs writing |
| `trm_io_buffer.ptx` | NO | Needs writing |
| `trm_sleep_mode.ptx` | NO | Needs writing |

**Only 3 new kernels needed.** The rest is wiring.

---

## 6. GPU Saturation Impact

Current: ~2% GPU utilization (Python bottleneck)

After D.3 (Jarvis parallel dispatch): **Expected 30-50% utilization**
- 9 workers × 11 specialist kernels × concurrent CUDA streams
- No Python between kernel launches
- Shared register communication (no host roundtrip)

After D.4 (full inference on GPU): **Expected 50-70% utilization**
- Entire perceive→navigate→reason→decide pipeline in VRAM
- TRM game loop runs at GPU clock speed
- Only I/O requires host interaction

After D.5 (sleep-time on GPU): **Expected 70-90% during sleep**
- Contrastive gradient computation on device-resident arrays
- Galaxy consolidation without host↔device copies
- Continuous learning at full GPU throughput

---

## 7. Relationship to Current Work

**This spec does NOT block current work.** The contrastive training fix (loader.py argtypes + remove CPU fallback) is still needed NOW to make sleep-time work in the current Python-orchestrated system. Phase D migration is the long-term architectural plan.

**Current priority order:**
1. Remove CPU fallback from `adaptive_swarm.py` (sovereignty fix, immediate)
2. Verify GPU contrastive path works standalone (validation, immediate)
3. Run warm benchmark with sovereign contrastive training (proof, today)
4. Begin D.1 (I/O buffers) — smallest migration, unblocks everything else
5. D.2-D.6 as progressive migration over coming sessions

**Each migration immediately improves performance** — no need to wait for all 6 to see benefits. D.3 alone (Jarvis parallel dispatch) would jump GPU utilization from 2% to 30-50%.

---

## 8. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| GPU utilization > 50% during inference | `nvidia-smi` dmon during benchmark |
| GPU utilization > 70% during sleep-time | `nvidia-smi` dmon during consolidation |
| `knowledgeverse.py` < 300 lines | `wc -l` |
| Zero Python in inference hot path | `grep` for forbidden calls |
| TRM runs as continuous game loop | Process stays alive between questions |
| Jarvis dispatches 9 workers in parallel | CUDA stream count = 9 |
| Sleep-time runs without host copies | No `memcpy_htod`/`memcpy_dtoh` during sleep |
| All benchmarks pass at >= current score | Non-regression |

---

## 9. Daniel's Vision Alignment

From Three Brain System Specification:
> "TRM IS the Avatar — the TRM (~7M params) is NOT a function Python calls. It IS the AI entity that lives in the House and thinks inside the Galaxy. Runs as a game loop via `trm_step_fused.ptx`."

From Hyper-Parallel Processing Specification:
> "Specialists communicate during execution (register sharing), are orchestrated by one entity (TRM), and converge to one answer (halting gate)."

From Knowledgeverse Specification §2.1:
> "Python = Boot + I/O only (~200 lines target)."

From Daniel (2026-03-23):
> "This was supposed to be a single mind with internal swarms, not external python orchestration, the TRM itself with Jarvis is this layer."

Phase D makes these specifications reality. The architecture is designed. The kernels mostly exist. The migration is wiring, not invention.
