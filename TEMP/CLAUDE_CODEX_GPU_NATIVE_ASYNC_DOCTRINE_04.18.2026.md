# GPU-Native Async/Parallel Doctrine
**Date:** April 18, 2026
**Author:** Claude (Architecture Partner)
**Status:** Active Doctrine — binding on all K3D contributors and agents

---

## The Ruling

> "Everywhere we can async and parallel is better, after all, this is what a GPU is best at and that we need for a living game engine. And these are Python concepts to apply to our architecture, not to introduce Python!!"
> — Daniel Campos Ramos, April 18, 2026

This document freezes that ruling as enforceable doctrine. Every section below is citable in three sentences or fewer.

---

## 1. The Concept/Implementation Split

Async, parallel, pipelining, fan-out/fan-in, scheduling, and event loops are **algorithmic design patterns** from computer science. They have a Python surface form (`asyncio`, `threading`, `concurrent.futures`) and a GPU surface form (`cp.async`, cooperative groups, `cuda::pipeline`, persistent kernels). K3D uses the GPU surface form. The concepts are borrowed from CS culture; the implementation is pure GPU.

This is the same shape as the no-numpy ruling: "no bulk libraries — we code our own kernels for freedom." Here: **no Python concurrency runtime — we use GPU concurrency primitives for the same reason.** Freedom means the hot path cannot be interrupted by the OS scheduler, the GIL, or a Python event loop.

### Worked Example 1 — Producer-Consumer Pipeline

| Dimension | Wrong (Python) | Right (GPU) |
|---|---|---|
| Pattern | `asyncio.Queue` + `asyncio.gather` | `cuda::pipeline<thread_scope_block>` + `cp.async.ca` |
| Latency | Microseconds minimum; GIL + epoll overhead | Sub-microsecond; hardware DMA engine |
| Sovereignty | Python event loop must be running | Runs inside kernel; host is irrelevant |
| K3D instance | Galaxy star prefetch while Nine-Chain Swarm computes | `pipeline.producer_acquire()` → `cuda::memcpy_async(sdata, galaxy_ptr, ...)` → `pipeline.producer_commit()` → compute → `pipeline.consumer_wait()` → `pipeline.consumer_release()` |

**PTX surface:** `cp.async.ca.shared.global [smem_ptr], [global_ptr], 16;`

### Worked Example 2 — Fan-Out / Fan-In

| Dimension | Wrong (Python) | Right (GPU) |
|---|---|---|
| Pattern | `multiprocessing.Pool(9).map(worker, inputs)` | `cudaLaunchCooperativeKernel` + `cooperative_groups::this_grid().sync()` |
| Latency | ~50 ms fork/exec + pickle serialization | <1 µs warp launch; zero IPC |
| Sovereignty | Nine workers = nine processes; results cross PCIe | Nine blocks share Galaxy VRAM L2 coherently |
| K3D instance | Nine-Chain Swarm parallel cognitive workers | `blockIdx.x ∈ [0..8]` each runs a specialist RPN chain; `grid.sync()` before Halting Gate fan-in |

**C++ surface:** `cg::grid_group g = cg::this_grid(); g.sync();`

### Worked Example 3 — Scheduler

| Dimension | Wrong (Python) | Right (GPU) |
|---|---|---|
| Pattern | `APScheduler` / `asyncio.call_at` / `sched.scheduler` | SM hardware warp scheduler + `cudaStreamCreateWithPriority` |
| Latency | OS timer resolution ±1–5 ms | Hardware warp dispatch < 20 ns |
| Sovereignty | Python process must remain alive and scheduled | SM schedules warps; host OS irrelevant |
| K3D instance | TRM tick cadence at 60 Hz | High-priority stream for `trm_step_fused.ptx`; persistent kernel naturally yields via `__nanosleep(16_666_666)` between ticks |

**C++ surface:** `cudaStreamCreateWithPriority(&hot_stream, cudaStreamNonBlocking, -1);`

### Worked Example 4 — Event Loop

| Dimension | Wrong (Python) | Right (GPU) |
|---|---|---|
| Pattern | `asyncio` event loop with `loop.run_forever()` | Persistent kernel polling global-memory mailbox |
| Latency | Python event loop overhead + GIL | Warp polls L2-cached mailbox every ~100 ns |
| Sovereignty | asyncio must be running on host CPU | Kernel launched once; host CPU can sleep |
| K3D instance | TRM daemon awaiting next query | `while (atomicAdd(&mailbox[TICK_TRIGGER], 0) != 1) { __nanosleep(100); }` — then `trm_step_fused()` inline |

**PTX surface:** `atom.global.add.u32 %r, [mailbox], 0;` (read-only poll via zero-add)

---

## 2. GPU-Native Async/Parallel Primitive Inventory

All primitives validated for **sm_86 (RTX 3070), CUDA 12.x**. Build flag: `nvcc -arch=sm_86 -std=c++17`.

| # | Primitive | Header / PTX Instruction | Scope | K3D Usage Example | Hot-Path Rating |
|---|---|---|---|---|---|
| 1 | `cp.async.ca` | PTX: `cp.async.ca.shared.global` | Block (async) | Transfer Yard bank-parallel Galaxy VRAM → SMEM copy, keeping L1 warm for RPN compute | **5/5** — zero-latency async; hides memory behind compute |
| 2 | `cp.async.cg` | PTX: `cp.async.cg.shared.global` | Block (async) | Nine-Chain Swarm weight prefetch bypassing L1 when weights are write-once per tick | **5/5** — non-blocking; correct cache policy for streaming weights |
| 3 | `cuda::pipeline` | `<cuda/pipeline>`, `producer_acquire / commit / consumer_wait / release` | Block | TRM step pipeline: fetch Galaxy stars while previous step's RPN executes | **5/5** — canonical overlap of fetch and compute; no host involvement |
| 4 | `cuda::barrier` | `<cuda/barrier>`, `barrier.arrive_and_wait()` | Block | Nine-Chain intra-block sync at end of each reasoning phase | **4/5** — fast (nanosecond range); block-scope only |
| 5 | `__syncthreads` | PTX: `bar.sync 0` | Block | Transfer Yard SMEM consistency fence after async copy; before RPN read | **4/5** — classic; cheap when warp occupancy is high |
| 6 | `__syncwarp` | PTX: `bar.warp.sync 0xFFFFFFFF` | Warp | RPN lane convergence after divergent conditional path in specialist kernel | **5/5** — sub-cycle; mandatory after any warp-divergent branch |
| 7 | `__shfl_xor_sync` | PTX: `shfl.sync.bfly` | Warp | Nine-Chain Swarm parallel reduction of attention scores across warp lanes | **5/5** — butterfly reduction; no shared memory required |
| 8 | `__reduce_add_sync` | PTX: `redux.sync.add.u32` (sm_80+) | Warp | TRM accumulator: sum partial results before single global atomic commit | **5/5** — one instruction; replaces 5-step butterfly for simple sums |
| 9 | `__ballot_sync` | PTX: `vote.sync.ballot.b32` | Warp | Halting Gate: which of 9 workers have converged; `__popc(mask)` gives count | **5/5** — warp-level vote in one instruction |
| 10 | `__any_sync / __all_sync` | PTX: `vote.sync.any / vote.sync.all` | Warp | Early-exit check: if all workers agree, skip remaining budget iterations | **5/5** — one instruction; correct idiom for unanimous convergence |
| 11 | `cooperative_groups::this_block_tile<N>` | `<cooperative_groups.h>` | Block sub-tile | Transfer Yard: 8-thread tile executes bank-parallel row copy without aliasing | **4/5** — clean sub-block parallelism; N must be power-of-2 ≤ warp size |
| 12 | `cudaLaunchCooperativeKernel` | `cuda_runtime_api.h` | Grid | Nine-Chain Swarm fan-out: launch 9 blocks co-resident on GPU; guaranteed grid sync availability | **3/5** — launch overhead amortized over many ticks; required for `grid.sync()` |
| 13 | `cooperative_groups::this_grid().sync()` | `<cooperative_groups.h>`, `grid_group::sync()` | Grid | Halting Gate fan-in: barrier after all 9 workers complete before emitting answer | **3/5** — expensive if called mid-tick; acceptable once per frame boundary |
| 14 | `atomicAdd` (global queue) | PTX: `atom.global.add.u32` | Device | Cross-core message passing: worker pushes result index to tail of global output queue | **4/5** — lock-free; contention low when N workers ≤ 9 |
| 15 | Persistent kernel + `__nanosleep` | `__nanosleep(ns)` (sm_86+) | SM | TRM daemon loop: poll mailbox, yield SM time slice between ticks, never exit | **5/5** — zero re-launch overhead; deterministic tick cadence at 60 Hz |

**Note on `cudaStream_t`:** Streams are a host-side API primitive. They are permitted in the **ingestion path** for overlapping data transfers with host processing. In the hot path, stream management is handled at kernel launch time only; do not call `cudaStreamSynchronize` or `cudaDeviceSynchronize` during inference.

---

## 3. Banned Python Concurrency Primitives

### Banned in Hot Path — All Sovereign Modules

The following are **unconditionally banned** in any file under `knowledge3d/cranium/`, `knowledge3d/knowledgeverse/`, `knowledge3d/daemon/` (except the module edge described below), and any PTX bridge:

| Python Construct | Ban Reason | Sovereign Replacement |
|---|---|---|
| `import asyncio` / `async def` / `await` | Python event loop = OS scheduler dependency; GIL contention | Persistent kernel + global mailbox poll |
| `asyncio.gather` | Fan-in via Python coroutine; host CPU required | `cooperative_groups::this_grid().sync()` + global result array |
| `asyncio.Queue` / `asyncio.sleep` | Async queue and yield via Python runtime | `atomicAdd` global tail queue + `__nanosleep` |
| `threading.Thread` | OS-scheduled thread; preemptible; not deterministic | Cooperative groups tiled partition |
| `threading.Lock` / `threading.RLock` | CPU mutex; introduces OS scheduler into hot path | `atomicCAS` on global flag; or SMEM bank isolation |
| `threading.Event` / `threading.Barrier` | CPU synchronization primitive | `cuda::barrier` (device side); `__syncthreads` |
| `multiprocessing.Pool` / `multiprocessing.Process` | Fork/exec overhead; PCIe round-trip for data | `cudaLaunchCooperativeKernel` with 9 blocks |
| `concurrent.futures.ThreadPoolExecutor` | Thread pool on CPU; data must leave GPU to be scheduled | Warp-parallel kernel; stream-based launch |
| `concurrent.futures.ProcessPoolExecutor` | Same as Pool plus pickle serialization | Cooperative kernel launch |
| `queue.Queue` | CPU-side FIFO; requires Python GIL | `atomicAdd` global tail index on GPU |
| `time.sleep()` | Sleeps the OS thread; not the SM | `__nanosleep(ns)` inside kernel; stream sync on host |
| `sched.scheduler` / `APScheduler` | Python-level scheduler | SM hardware warp scheduler + stream priorities |

### Permitted Exception: Module Edge (Boot + I/O Only)

The **module-edge boundary** — defined as the outermost Python entry point that starts the daemon, handles keyboard/network/display, and launches the initial CUDA kernel — may use:

- `asyncio` for **network I/O** (WebSocket, HTTP): ingestion path and Door (network interface) only.
- `threading.Thread` for a **single watchdog thread** that monitors GPU health (not reasoning).
- `concurrent.futures` in the **ingestion path** for batch file reads and embedding calls.

The boundary is: **before `enqueue_task` / before any kernel launch**. Once execution enters the sovereign tick — from `enqueue_task` through `wait_output_buffer` — zero Python concurrency primitives may fire. If `nvidia-smi utilization.gpu = 0` during inference, the boundary has been violated.

---

## 4. The Living Game Engine Doctrine

A "living game engine" is always-on. It does not start and stop per query. The TRM entity lives in the House and thinks in the Galaxy at 30–60 Hz, whether or not a human is asking a question.

**Five properties that make it live:**

**4.1 Always-on persistent kernel(s).** The TRM daemon kernel is launched once at boot and never exits. It polls the global mailbox for incoming queries. There is no `cudaDeviceSynchronize` in the steady state. The kernel is the process.

**4.2 Sub-tick async overlap.** While the current tick's Nine-Chain Swarm is reasoning, the next tick's inputs are being prefetched into shared memory via `cp.async` + `cuda::pipeline`. The 16.67 ms (60 Hz) frame budget contains both compute and fetch in parallel, not sequentially.

**4.3 N cores concurrent, isolated.** Each of the nine swarm workers operates on its own RPN stack and specialist adapter. They do not share state mid-reasoning. Cross-core communication uses the `STORE/RECALL` register protocol — atomic writes to pre-agreed global addresses. No Python coordinates them.

**4.4 Async Galaxy loads do not stall compute.** Galaxy star prefetch is a `cp.async` background DMA operation. The SM executes RPN instructions while the DMA engine copies the next star. If a star is not yet in SMEM when needed, the `pipeline.consumer_wait()` barrier stalls only the threads that need that data — not the entire SM, and never the host CPU.

**4.5 Cross-core work via lock-free atomic queues.** When one worker produces a result that another worker needs, it writes the result index to a global tail via `atomicAdd(&tail, 1)`. The consuming worker reads from `atomicAdd(&head, 0)` (poll). No Python queue, no mutex, no OS involvement.

---

## 5. Anti-Patterns to Grep For

CI must run these checks on every commit. Exclude `Old_Attempts/` and `tests/` from the sovereignty checks (tests may use Python async for fixture setup). Exclude `knowledge3d/ingestion/` from checks 1–4 (ingestion path is permitted).

```bash
# Check 1: asyncio import in sovereign hot-path modules
grep -rn "import asyncio" \
  knowledge3d/cranium/ \
  knowledge3d/knowledgeverse/ \
  knowledge3d/daemon/ \
  --exclude-dir=Old_Attempts \
  | grep -v "# ingestion-edge-ok"
# MUST return zero lines

# Check 2: threading in sovereign modules (excluding single watchdog exception)
grep -rn "threading\.\(Thread\|Lock\|RLock\|Event\|Barrier\)" \
  knowledge3d/cranium/ \
  knowledge3d/knowledgeverse/ \
  --exclude-dir=Old_Attempts \
  | grep -v "# watchdog-ok"
# MUST return zero lines

# Check 3: multiprocessing in sovereign modules
grep -rn "multiprocessing\." \
  knowledge3d/cranium/ \
  knowledge3d/knowledgeverse/ \
  knowledge3d/daemon/ \
  --exclude-dir=Old_Attempts
# MUST return zero lines

# Check 4: concurrent.futures in hot path
grep -rn "concurrent\.futures" \
  knowledge3d/cranium/ \
  knowledge3d/knowledgeverse/ \
  knowledge3d/daemon/ \
  --exclude-dir=Old_Attempts
# MUST return zero lines

# Check 5: await keyword in hot-path modules (outside bridge edge)
grep -rn "^\s*await " \
  knowledge3d/cranium/ \
  knowledge3d/knowledgeverse/ \
  --exclude-dir=Old_Attempts
# MUST return zero lines

# Check 6: time.sleep in hot path (kernel should yield, not host thread)
grep -rn "time\.sleep" \
  knowledge3d/cranium/ \
  knowledge3d/knowledgeverse/ \
  knowledge3d/daemon/ \
  --exclude-dir=Old_Attempts \
  | grep -v "# boot-ok"
# MUST return zero lines

# Check 7: queue.Queue in sovereign modules
grep -rn "queue\.Queue\|from queue import" \
  knowledge3d/cranium/ \
  knowledge3d/knowledgeverse/ \
  --exclude-dir=Old_Attempts
# MUST return zero lines
```

**Canary signal (not grep-able, but auditable):** Run `nvidia-smi dmon -s u -d 1` during a benchmark query. If `utilization.gpu < 50%` for more than 2 consecutive seconds while a query is in flight, Python is in the hot path. The fix is not to optimize the Python — the fix is to eliminate it.

---

## 6. How This Composes with Earlier Rulings

This doctrine is the concurrency dimension of a family of sovereign rulings. They share the same root cause and the same fix:

**No bulk libraries → No concurrency runtime (same reason).**
The feedback `feedback_no_numpy_no_bulk_libraries_sovereign_only.md` was requested fourteen times because NumPy kept reappearing under new names. The present ruling is the same request applied to Python concurrency: `asyncio` keeps reappearing as "just an I/O helper" or "just for the daemon tick." It is not. The moment `asyncio` enters `knowledgeverse.py` or `trm_game_loop.py`, the sovereign tick re-enters the Python event loop and loses the GPU's determinism guarantee.

**TRM solves everything → TRM must have the concurrency tools to do so.**
`feedback_no_python_orchestration_trm_solves.md`: "The TRM IS the reasoner." A TRM that reasons through Python-orchestrated fan-out is not a TRM — it is a Python script with PTX decorations. The nine-chain swarm, the Halting Gate vote, and the budget iterations must all execute as GPU-native parallel operations so the TRM can run them without Python's involvement.

**Python dispatch is not a line item → Async drift is the same failure.**
`feedback_python_dispatch_is_not_a_line_item.md`: "The ring can appear sovereign (edges call PTX) while the middle is pure Python." `asyncio` and `threading` are exactly that middle. A kernel that launches nine workers and then `asyncio.gather`s their completion is Python dispatch with a PTX costume. The symptom is identical: `nvidia-smi utilization.gpu = 0` during inference, one CPU core pinned.

**Transfer Yard is the addressable matrix → Its async is GPU-native.**
`feedback_transfer_yard_is_the_addressable_matrix.md` established that bank-parallel copies are 15–51% faster than LIFO. The async mechanism that makes those parallel copies possible is `cp.async.ca` + `this_block_tile<8>()`, not `asyncio.to_thread`. The ruling today is the generalization of that specific Transfer Yard design decision to the entire engine.

---

## 7. Acceptance Gates

CI passes only when all of the following are green. Scope: `knowledge3d/cranium/`, `knowledge3d/knowledgeverse/`, `knowledge3d/daemon/` — excluding `Old_Attempts/` and `tests/`.

| Gate | Grep Pattern | Expected Result |
|---|---|---|
| G1 | `grep -rn "import asyncio" knowledge3d/cranium/ knowledge3d/knowledgeverse/ knowledge3d/daemon/ --exclude-dir=Old_Attempts \| grep -v "# ingestion-edge-ok"` | 0 lines |
| G2 | `grep -rn "threading\.Thread" knowledge3d/cranium/ knowledge3d/knowledgeverse/ --exclude-dir=Old_Attempts \| grep -v "# watchdog-ok"` | 0 lines |
| G3 | `grep -rn "threading\.\(Lock\|RLock\|Event\|Barrier\)" knowledge3d/cranium/ knowledge3d/knowledgeverse/ --exclude-dir=Old_Attempts` | 0 lines |
| G4 | `grep -rn "multiprocessing\." knowledge3d/cranium/ knowledge3d/knowledgeverse/ knowledge3d/daemon/ --exclude-dir=Old_Attempts` | 0 lines |
| G5 | `grep -rn "concurrent\.futures" knowledge3d/cranium/ knowledge3d/knowledgeverse/ knowledge3d/daemon/ --exclude-dir=Old_Attempts` | 0 lines |
| G6 | `grep -rn "^\s*await " knowledge3d/cranium/ knowledge3d/knowledgeverse/ --exclude-dir=Old_Attempts` | 0 lines |
| G7 | `grep -rn "time\.sleep" knowledge3d/cranium/ knowledge3d/knowledgeverse/ knowledge3d/daemon/ --exclude-dir=Old_Attempts \| grep -v "# boot-ok"` | 0 lines |
| G8 | `grep -rn "queue\.Queue\|from queue import" knowledge3d/cranium/ knowledge3d/knowledgeverse/ --exclude-dir=Old_Attempts` | 0 lines |

**Current baseline (April 18, 2026) — known violations to fix before gates can pass:**

The grep audit of the live codebase found the following violations that must be remediated:

- `knowledge3d/knowledgeverse/trm_game_loop.py:53` — `threading.RLock` (G3 fail)
- `knowledge3d/knowledgeverse/trm_game_loop.py:216` — `time.sleep(0.005)` (G7 fail)
- `knowledge3d/knowledgeverse/sovereign_hot_path.py:5` — `import concurrent.futures` (G5 fail)
- `knowledge3d/knowledgeverse/sovereign_hot_path.py:2607,2755` — `ProcessPoolExecutor` + `ThreadPoolExecutor` (G5 fail)
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py:216–218` — `threading.Event` + `threading.Thread` + `threading.RLock` (G2/G3 fail)
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py:1355` — `time.sleep(0.002)` (G7 fail)
- `knowledge3d/cranium/bridges/n_chain_swarm_bridge.py:269,272` — `time.sleep(0.0005)` (G7 fail)
- `knowledge3d/knowledgeverse/knowledgeverse.py:488–491` — `threading.Event` + `threading.Thread` + `threading.Lock` × 2 (G2/G3 fail)
- `knowledge3d/cranium/sovereign_matryoshka_embedder.py:38,74` — `threading.RLock` + `threading.Lock` (G3 fail)
- `knowledge3d/cranium/ptx_runtime/micro_specialist_pool.py:71` — `threading.Lock` (G3 fail)
- `knowledge3d/cranium/ptx_runtime/nvrtc_ptx_loader.py:21` — `threading.RLock` (G3 fail)
- `knowledge3d/cranium/sleep/scheduler.py:31,48,62` — `threading.Thread` + `time.sleep(30)` (G2/G7 fail)

These are not line items to fix one at a time. They are the Python dispatch problem described in `feedback_python_dispatch_is_not_a_line_item.md`. Remediating them means migrating the tick loop to the persistent-kernel pattern; the threading locks dissolve when there is no Python thread to protect.

---

## 8. Codex Handoff Checklist

Before marking any TRM tick / swarm / daemon work as complete, Codex must verify all 10 items:

- [ ] **C1. Primitive inventory consulted.** The implementation uses at minimum one primitive from Section 2 for each async/parallel operation. No Python concurrency construct is introduced as a "temporary" measure.

- [ ] **C2. Acceptance gates wired.** CI runs all 8 grep checks from Section 7. Gates G1–G8 pass (or new violations are explicitly tracked with a remediation ticket, not silently accepted).

- [ ] **C3. Python concurrency audit done.** Before submitting any spec or PR touching `knowledge3d/cranium/` or `knowledge3d/knowledgeverse/`, run the 8 grep checks manually and report the count. If count > 0, fix first.

- [ ] **C4. Persistent-kernel prototype for TRM daemon.** `knowledge3d/cranium/bridges/trm_step_fused_bridge.py` threading tick replaced by a persistent-kernel launch in `trm_step_fused.ptx` that polls a global mailbox. The `threading.Thread` tick driver in `knowledge3d/daemon/tick_driver.py` is the boot-only wrapper that launches this kernel — it does not orchestrate ticks.

- [ ] **C5. Global mailbox ABI defined.** A header (`.cuh` or equivalent) defines the mailbox layout: `TICK_TRIGGER` offset, `SHUTDOWN` sentinel value, `OUTPUT_READY` flag, memory ordering (`__threadfence_system` after write). Codex writes this; Claude reviews for sovereignty.

- [ ] **C6. Nine-Chain Swarm fan-out uses `cudaLaunchCooperativeKernel`.** The 9 workers are 9 blocks in a single cooperative kernel launch. `this_grid().sync()` is the Halting Gate barrier. No Python iterates over workers.

- [ ] **C7. Transfer Yard async copy uses `cp.async.ca` + `this_block_tile<8>()`.** The bank-parallel copy path established in `feedback_transfer_yard_is_the_addressable_matrix.md` is implemented via `cp.async.ca.shared.global` + tiled partition, not via `asyncio.to_thread` or `threading.Thread`.

- [ ] **C8. Ingestion-path exceptions annotated.** Any `asyncio`, `threading`, or `concurrent.futures` usage in the ingestion path carries an inline comment: `# ingestion-edge-ok` or `# watchdog-ok` so the grep gates do not false-positive.

- [ ] **C9. `nvidia-smi` canary check.** During the first full benchmark run after migration, capture `nvidia-smi dmon -s u -d 1` output. Attach to the PR as evidence. `utilization.gpu` must be ≥ 50% for the duration of a query.

- [ ] **C10. No fallbacks added.** No comment like "fallback to Python asyncio if GPU not available." K3D fails and fixes. If the GPU is not available, the system does not start. Sovereignty is not an optional feature.

---

## Summary

Daniel's ruling establishes that async and parallel are **concepts** — they belong in the architecture. Their **implementation** in K3D is GPU-native: `cp.async`, `cuda::pipeline`, cooperative groups, persistent kernels, `atomicAdd` queues, warp shuffles. Python's `asyncio`, `threading`, `multiprocessing`, and `concurrent.futures` are the wrong surface form for the right concept. They introduce OS scheduler dependency, GIL contention, PCIe round-trips, and Python orchestration — the exact problems that sovereignty exists to eliminate.

The living game engine runs at 30–60 Hz because the GPU hardware scheduler, not Python, decides when each warp runs. The TRM tick is deterministic because a persistent kernel on the SM, not an event loop on the host, drives it. The nine workers fan out in parallel because they are nine blocks in one cooperative kernel launch, not nine coroutines in `asyncio.gather`.

**One AI. One sovereign tick. Zero Python concurrency. GPU-native everywhere.**
