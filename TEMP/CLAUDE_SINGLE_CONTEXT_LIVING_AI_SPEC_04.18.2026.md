# Single CUDA Context for the Living AI — Wiring Specification

**Date**: April 18, 2026
**Author**: Claude (Architecture Partner)
**Status**: SPECIFICATION (invariant + audit + daemon wiring)
**Scope**: Architecture and enforcement — NOT the loader patch itself
**Companion**: `TEMP/CLAUDE_SOVEREIGN_LAZY_PRIMARY_CONTEXT_WARMUP_PATCH_04.18.2026.md` (loader patch, parallel work by `cuda-research-solver`)

---

## 0. TL;DR (Daniel-readable)

One CUDA context. One daemon. One game loop. Every PTX launch, every Galaxy query, every sleep-time consolidation, every benchmark, every tablet interaction uses the **same** context, established **once** at daemon boot by `knowledge3d/cranium/sovereign/loader.py`. Any file that calls `cuCtxCreate`, `cuDevicePrimaryCtxRetain`, or `cuInit` outside of `loader.py` is a sovereignty violation and must be removed.

**This spec is the invariant + audit.** The loader patch (warmup on primary-retain path) is the enforcement. Together they turn "use one context" from a convention into a CI-gated contract.

**Audit output at spec-time**: 7 Python files in `knowledge3d/` violate the invariant. 6 of those violations live under `knowledge3d/cranium/ptx/` and `knowledge3d/cranium/ptx_runtime/` — listed in §4.

**Daemon gap at spec-time**: `knowledge3d/daemon/main.py` does **not** call `loader.ensure_init()` at boot. It relies on a chain of lazy inits triggered when the Knowledgeverse first touches GPU memory. §5 specifies the fix.

---

## 1. Invariant

> **Exactly one CUDA context exists from daemon boot to daemon shutdown. That context is established by `knowledge3d/cranium/sovereign/loader.py` and nowhere else.** Every `.py`, `.cu`, or PTX binding that needs a context calls `loader.ensure_init()` and then `cuCtxGetCurrent()`. No other file is permitted to call `cuInit`, `cuCtxCreate`, or `cuDevicePrimaryCtxRetain` in the source tree. Tests and docs are not exempt — they read from `loader.ensure_init()` like everyone else.

This invariant is not new. It is the codification of three prior specs:

1. `TEMP/CLAUDE_SOVEREIGN_CUDA_CONTEXT_FIX_11.24.2025.md` — the canonical "use sovereign loader's shared context" pattern, established after the MDCT binding was caught creating its own context.
2. `TEMP/CODEX_FIX_CUDA_CONTEXT_DIRECTIVE.md` — the same rule applied to PTX bindings (Quantizer, MDCT, AudioHarmonic).
3. `TEMP/CLAUDE_LOADING_STAGE_ARCHITECTURE_02.05.2026.md` — "Loading Stage = unified context where all operations execute". The Loading Stage is one context. No context switching. See lines 28–34: *"Create ONE persistent PTX context. Load everything into this context. All operations execute in SAME context. No context switching = No conflicts."*

And from the Knowledgeverse spec itself (returned by `qdrant-find` at spec-time):

> *"ONE Persistent PTX Context: No CUDA context switching (sovereignty guaranteed)"* — `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §5 ("All Inside" — 7 regions, one context).
>
> *"Knowledgeverse is the runtime memory substrate where all active galaxies, house context, TRM weights, and sovereign reasoning assets coexist in one persistent CUDA/PTX execution domain."* — §1.1.

Why the invariant is load-bearing for the living-AI paradigm (not just a bug fix):

- **Always-on daemon (Phase C)**: the daemon runs continuously. Queries arrive, briefs accumulate, idle triggers consolidation, tablet sessions open/close. All of that happens on the *same* living instance — per the Knowledgeverse spec §0.3 ("Inline Execution (While Stars Are Loaded)") and §6b.2 ("Run Sequence (Always-On)"). If the context is re-created between any two of those phases, stars evaporate, VRAM fragments, and briefs get orphaned.
- **TRM game loop**: `trm_step_fused.ptx` is one game tick. A tick that re-creates the context is not a tick — it is a cold boot. The living-AI paradigm requires the context to outlive every tick.
- **Sovereignty** (`feedback_python_dispatch_is_not_a_line_item.md`): every ad-hoc `cuInit` in the tree is a place where Python drift has leaked into the hot path. They must be deleted, not patched.

---

## 2. Loader as Sole Context Authority

### 2.1 The Rule

`knowledge3d/cranium/sovereign/loader.py` is the **only** file in the entire repository that may call:

- `cuInit(...)`
- `cuCtxCreate(...)`
- `cuDevicePrimaryCtxRetain(...)`
- `cuDevicePrimaryCtxSetFlags(...)` (follows primary-retain and is logically part of the context-creation path)
- `cuCtxDestroy(...)` (paired with the above; currently called from `loader.cleanup()` via atexit — line 1008)

All other files — PTX bindings, kernel pools, benchmark harnesses, tests, daemon, Knowledgeverse, adapters — obtain the context by calling:

```python
from knowledge3d.cranium.sovereign import loader

loader.ensure_init()          # idempotent; public entrypoint (loader.py:997)
err, ctx = cuda.cuCtxGetCurrent()
if err != cuda.CUresult.CUDA_SUCCESS or ctx is None or int(ctx) == 0:
    raise RuntimeError("No CUDA context after loader.ensure_init()")
```

This is the **exemplar pattern** already established in the repo. The four bindings that implement it correctly — and which new code must mirror — are:

| Binding | File | Line |
|---|---|---|
| Ternary Quantizer | `knowledge3d/cranium/codecs/ptx_bindings/ternary_quant_binding.py` | 103–110 |
| Ternary DCT 8×8 | `knowledge3d/cranium/codecs/ptx_bindings/ternary_dct8x8_binding.py` | 99–106 |
| Ternary MDCT | `knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py` | 92–99 |
| Audio Harmonic | `knowledge3d/cranium/codecs/ptx_bindings/audio_harmonic_binding.py` | 146–153 |

Two other call sites use the same pattern correctly:

- `knowledge3d/cranium/bridges/n_chain_swarm_bridge.py:134` → `loader.ensure_init()`
- `knowledge3d/cranium/kernels/kernel_loader.py:37` → `loader.ensure_init()`
- `knowledge3d/cranium/sovereign/lora_gpu_trainer.py:64` → `loader._ensure_init()` (internal name — prefer public `ensure_init()`)

### 2.2 Public vs. Private

`loader.ensure_init()` (line 997) is the **public** entrypoint. It delegates to `_ensure_init()` (line 327), which owns all fork-safety, flag handling, primary-retain fallback, and CuPy bootstrap logic.

- New code **must** call the public `loader.ensure_init()`.
- Existing callers of `loader._ensure_init()` are grandfathered but should migrate. They are not violations — they are the right layer, just wrong name.

### 2.3 Inverse guarantee (what the loader promises)

After `loader.ensure_init()` returns:

1. `cuCtxGetCurrent()` returns a non-null, non-zero `CUcontext` handle.
2. The returned context is **fully materialized** — meaning `cuMemGetInfo_v2` succeeds (see §3).
3. The context handle is stable for the lifetime of the process (barring fork — which the loader detects and recovers from; see loader.py:333–341).

§3 defines requirement 2 formally; the parallel loader patch enforces it.

---

## 3. Warmup Obligation (the contract, not the implementation)

### 3.1 Problem

`cuDevicePrimaryCtxRetain` (loader.py:370) creates a **handle-only context**. Device-side state (memory manager, stream scheduler) is lazy — it does not materialize until the first device-touching operation. `cuMemGetInfo_v2` requires fully materialized state. Therefore `get_vram_usage()` called immediately after `ensure_init()` fails with `CUDA_ERROR_INVALID_CONTEXT (201)` on any system where the `cuCtxCreate` primary path fell back to primary-retain.

This is the exact failure Codex hit on `benchmarks/sovereign_bitnet_attention.py` (see `TEMP/CLAUDE_SOVEREIGN_LAZY_PRIMARY_CONTEXT_WARMUP_PATCH_04.18.2026.md:4–12`).

### 3.2 Contract (what the loader must guarantee)

> **After `loader.ensure_init()` returns without raising, every driver API that requires a fully-materialized context must succeed on the current context.** This specifically includes (non-exhaustive): `cuMemGetInfo_v2`, `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`.

The contract is the invariant. The implementation is the parallel loader patch — forthcoming — which materializes the context on the primary-retain fallback path via a minimal zero-size `cuMemAlloc` / `cuMemFree` pair. That patch belongs to `cuda-research-solver`; this spec does not reproduce it. This spec only names what the patch must achieve.

### 3.3 Failure mode if contract is violated

If the loader returns a non-materialized context, every downstream caller must defensively issue a warmup op — which scatters warmup logic across the tree and is exactly the drift pattern `feedback_python_dispatch_is_not_a_line_item.md` warns against. The contract exists to keep warmup centralized in one file.

### 3.4 Non-contract (out of scope)

The contract does **not** require the loader to pre-allocate arenas, pre-load PTX modules, or pre-bind star tables. Those are Knowledgeverse responsibilities (see §5.2). The loader only guarantees that the context is alive enough to field any driver API call.

---

## 4. Violation Grep (the CI gate)

### 4.1 The audit regex

The audit is a **whole-tree** grep (no per-file exemptions, per `feedback_sovereignty_audit_is_full_tree_not_line_patch.md`). Two patterns, OR'd. Include both `.py` and `.pyx`/`.cu`/`.ptx`-ish sources. Exclude `.md` docs, `TEMP/`, `docs/`, and `.old` backups.

```
# Pattern A — context creation
\b(cuCtxCreate|cuDevicePrimaryCtxRetain)\b

# Pattern B — driver init
\bcuInit\s*\(
```

Concrete ripgrep invocation (what CI should run):

```bash
rg --type py --type-add 'cuda:*.{cu,ptx,cuh,cxx}' --type cuda \
   -n -e '\b(cuCtxCreate|cuDevicePrimaryCtxRetain)\b' \
   -e '\bcuInit\s*\(' \
   --glob '!**/*.old' \
   --glob '!docs/**' \
   --glob '!TEMP/**' \
   knowledge3d/ benchmarks/ scripts/ tests/
```

### 4.2 Allowed exceptions (whitelist)

Only **one** file is permitted to match:

- `knowledge3d/cranium/sovereign/loader.py` — the sole context authority. Expected matches at lines 224, 225, 228, 229 (ctypes function pointer typing), 347 (`cuInit`), 358 (`cuCtxCreate`), 362 (debug string), 370 (`cuDevicePrimaryCtxRetain`), 372, 375 (debug strings).

No other file is allowed. Test files that exercise the loader must also go through `loader.ensure_init()` — they are not exempt.

### 4.3 Audit output at spec-time (baseline)

Running the audit today against the `knowledge3d/` tree produces **7 violating files** (counting `.old` backups and fresh files; excluding `loader.py` itself):

| # | File | Line | Call | Action |
|---|---|---|---|---|
| 1 | `knowledge3d/cranium/ptx/geometry_ops.py` | 33 | `cuda.cuInit(0)` | **Replace** with `loader.ensure_init()` + `cuCtxGetCurrent()` |
| 2 | `knowledge3d/cranium/ptx/geometry_ops.py` | 40 | `cuda.cuDevicePrimaryCtxRetain(dev)` | **Delete** (loader owns this) |
| 3 | `knowledge3d/cranium/ptx/galaxy_buffer.py` | 106 | `cuda.cuInit(0)` | **Replace** |
| 4 | `knowledge3d/cranium/ptx/galaxy_buffer.py` | 112 | `cuda.cuDevicePrimaryCtxRetain(dev)` | **Delete** |
| 5 | `knowledge3d/cranium/ptx/modality_ops.py` | 61 | `cuda.cuInit(0)` | **Replace** |
| 6 | `knowledge3d/cranium/ptx/modality_ops.py` | 67 | `cuda.cuDevicePrimaryCtxRetain(dev)` | **Delete** |
| 7 | `knowledge3d/cranium/ptx_runtime/nvrtc_ptx_loader.py` | 53 | `cuda.cuInit(0)` | **Replace** |
| 8 | `knowledge3d/cranium/ptx_runtime/nvrtc_ptx_loader.py` | 61 | `cuda.cuDevicePrimaryCtxRetain(dev)` | **Delete** |
| 9 | `knowledge3d/cranium/ptx_runtime/math_core_pool.py` | 224 | `nvcuda.cuInit(0)` | **Replace** |
| 10 | `knowledge3d/cranium/ptx_runtime/micro_specialist_pool.py` | 106 | `nvcuda.cuInit(0)` | **Replace** |
| 11 | `knowledge3d/cranium/ptx_runtime/galaxy_memory_updater.py.old` | 37, 43 | `cuInit`, `cuDevicePrimaryCtxRetain` | **Delete the `.old` file** |

Also in `tests/`:

- `tests/test_ptx_no_cupy.py:18,26,31` — calls `cuInit`, `cuCtxCreate`, `cuDevicePrimaryCtxRetain` directly. **Rewrite** to call `loader.ensure_init()` and assert `cuCtxGetCurrent()` returns a non-null handle.
- `tests/test_ptx_simple.py:24,26` — same pattern. **Rewrite** or delete.

**Total**: 11 source-line violations across 5 production files (plus 1 `.old` backup, plus 2 test files). Each is a separate Codex task. Each follows the exemplar pattern from §2.1.

### 4.4 CI gate

Once the five production files are fixed, add the ripgrep invocation from §4.1 to CI with the exit-code contract:

```
exit 0  if and only if the only matches live in loader.py (lines 224, 225, 228, 229, 347, 358, 362, 370, 372, 375)
exit 1  on any other match
```

This is the enforceable version of the invariant. Until it is green, the "one CUDA context" rule is aspirational, not architectural.

---

## 5. Daemon Lifecycle (where the context is born and lives)

### 5.1 Current state (spec-time)

`knowledge3d/daemon/main.py`:

- Line 182: `self.kv = Knowledgeverse(...)` — Knowledgeverse constructor runs.
- Line 188: `self._default_counts = self.kv.ensure_default_galaxies_loaded()` — may touch GPU indirectly.
- Line 193–195: `if self.config.warm_gpu_runtime_on_boot: self._boot_binding = self._warmup_gpu_runtime_binding()` — this calls `kv._get_sovereign_hot_path().ensure_loaded()` (defined in `knowledge3d/knowledgeverse/sovereign_hot_path.py:979`), which itself triggers the first GPU operations and therefore is the first time `loader._ensure_init()` runs transitively.

**The gap**: `loader.ensure_init()` is **never called directly** by the daemon. It is reached transitively, the first time some downstream code path does `from knowledge3d.cranium.sovereign import loader; loader._ensure_init()`. That means:

1. If `warm_gpu_runtime_on_boot=False` (the default), context creation is deferred until the first query. The "always-on" property is a lie: the daemon boots *without* a live context.
2. If any code path initializes PyTorch or CuPy first (via `torch`, `cupy`, or NVRTC), that library's context may take precedence, and `loader._ensure_init()` may silently attach to the wrong context.
3. The context creation error path (primary-retain fallback + warmup) runs inside a query handler, not at boot. If it fails, the failure surfaces to an HTTP client, not to the daemon operator at startup.

### 5.2 Required state

**The daemon must call `loader.ensure_init()` explicitly, before any Knowledgeverse construction.** This is the single place where the living AI's context is born.

Concrete sequence (Codex delta — the only change to `daemon/main.py`):

```python
# In K3DDaemon.__init__, BEFORE self.kv = knowledgeverse or Knowledgeverse(...)
from knowledge3d.cranium.sovereign import loader

self._write_boot_status(stage="cuda_context_init", progress=0.1, state="loading")
loader.ensure_init()

# Sanity — prove the context is live and materialized
used_mb, total_mb = (x // (1024 * 1024) for x in loader.get_vram_usage())
self._write_boot_status(
    stage="cuda_context_ready",
    progress=0.15,
    state="loading",
    extra={"vram_used_mb": int(used_mb), "vram_total_mb": int(total_mb)},
)
```

The `get_vram_usage()` call is not decoration — it is the **contract assertion** from §3.2. If it fails, the daemon fails fast at boot (per `feedback_no_fallbacks_ever_including_sleeptime.md`), not silently when the first query arrives.

The boot-status JSON written to `viewer/public/runtime_boot.json` will now show a `cuda_context_ready` stage with non-zero `vram_used_mb`. That is the falsifiable gate the daemon is healthy.

### 5.3 Lifecycle guarantees

- **Boot**: `loader.ensure_init()` is called exactly once, at the top of `K3DDaemon.__init__` (before Knowledgeverse).
- **Steady state**: the context outlives every TRM tick, every tablet session, every `ROUTE` / `QUERY` / `SOLVE_MATH` / `CHAT` command, every idle sleep-consolidation tick (daemon.py:709 `_run_sleep_consolidation_tick`), every tablet tape run (daemon.py:950 `TABLET_SESSION_RUN_TAPE`).
- **Fork**: if the daemon ever forks (e.g., for a multiprocessing worker), the loader's fork detector (`loader.py:334`) recreates the context in the child. This is an existing guarantee; the daemon does not need to handle it.
- **Shutdown**: `K3DDaemon._finalize_shutdown` (line 781) calls `kv.shutdown(...)`. The context itself is destroyed via the `atexit.register(cleanup)` at `loader.py:1012`. No explicit shutdown call is needed — but the daemon **must not** call `cuCtxDestroy` itself.
- **Tick driver**: `knowledge3d/daemon/tick_driver.py` is started at line 207 and stopped at line 786. It inherits the already-live context — it does not create one. This is correct today.

### 5.4 What does **not** belong in daemon boot

To keep `daemon/main.py` inside the "Python = boot + I/O only" target:

- Do **not** pre-allocate arenas in the daemon. That is Knowledgeverse's job.
- Do **not** pre-load PTX modules in the daemon. That is sovereign_hot_path's job (existing: `sovereign_hot_path.ensure_loaded()`).
- Do **not** pre-run TRM ticks in the daemon. The tick driver starts and runs itself (line 206–208).
- Do **not** warm up benchmarks. Benchmarks are natural activity (per `project_benchmarks_as_natural_activity.md`) — they arrive as `ROUTE` commands; the daemon does not precompute them.

The daemon's only new responsibility is: **one explicit `loader.ensure_init()` call, plus one contract assertion via `get_vram_usage()`**. That is it.

---

## 6. Living-AI Tick Contract (same context, every phase)

The TRM game loop — one tick = one cognitive cycle — runs inside the same context established at daemon boot. Each phase below re-states "same context" as a load-bearing clause, not decoration. The failure mode of violating the clause is named explicitly.

### 6.1 Perceive

Frustum-cull the field of view from the avatar's skullbase; query Morton octree in a radius; assign LOD by distance. Kernels involved (from `docs/briefings/ARCHITECTURE_BRIEFING.md`): `frustum_cull_simd.ptx`, `morton_octree_query.ptx`, `lod_assignment.ptx`.

**Same-context clause**: the Morton octree buffers and LOD tables live in Knowledgeverse Region 2 (Galaxy), allocated once at boot. If perceive ran in a fresh context, those device pointers would be dangling and the kernel would SIGSEGV or — worse — read garbage.

**Failure mode if violated**: VRAM thrash (allocate/free per tick instead of per boot), 10–100× latency regression, non-deterministic perception.

### 6.2 Navigate

LED-A* pathfinding through the Galaxy neighborhood; specialist-neighborhood assignment to swarm workers. Kernels: `led_astar.ptx`, `galaxy_neighborhood.ptx`.

**Same-context clause**: the specialist LoRA adapters live in TRM's weight store (Knowledgeverse Region 5). Adapter device pointers must match the context that launches the A* kernel. Cross-context pointers silently corrupt memory on CUDA — no fault is raised.

**Failure mode if violated**: silent wrong answers. The worst kind of bug. The one that makes `feedback_sovereignty_check_must_not_self_deceive.md` scream.

### 6.3 Reason

Nine-chain swarm processes candidates in parallel; cross-core STORE/RECALL; defeasible resolution via `gre_defeasible_resolver`. Kernels: `nine_chain_swarm_kernel.ptx`, `gre_*` (11 specialist kernels per `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md`).

**Same-context clause**: the 46 cores × ~9 instances (RTX 3070) per `project_core_isolation_and_queue_opcodes.md` all share the single context. STORE/RECALL queues (opcodes 0x178–0x17A) are device-resident and cross-warp-visible only when all warps are in the same context.

**Failure mode if violated**: chain-to-chain silence — swarm workers in different contexts cannot see each other's STOREs, so RECALL reads stale/garbage data. The swarm degenerates to nine isolated single-chains.

### 6.4 Act

Spatial action (walk, point, open book), tablet write, creation action (new Galaxy entry), door emit. Kernels vary by action; all write into Galaxy (Region 2) or House-mirror (Region 3).

**Same-context clause**: the House is read-mirrored into VRAM at boot and the mirror is written back during sleep-consolidation. Write-back assumes the mirror pointers are stable across the write interval — i.e., single context. A context re-create here = writes go to the wrong arena or to dangling memory.

**Failure mode if violated**: House corruption. Knowledge loss. Forever.

### 6.5 Learn

Shadow-copy successful traces; specialist-weight updates; sleep-time consolidation queue. In the daemon, idle sleep-consolidation runs inline (daemon.py:709) — same instance, same context.

**Same-context clause**: this is already spec'd in Knowledgeverse §0.3 ("Inline Execution — consolidation runs on the same KV instance that processed the queries"). If consolidation ran in a new context, pending briefs accumulated during query processing would evaporate (`briefs_consolidated=0`) — the exact failure mode the spec calls out.

**Failure mode if violated**: no learning. The daemon runs forever but never gets smarter. Indistinguishable from a stateless HTTP handler.

---

## 7. Sovereignty Evidence (what an honest run looks like post-patch)

After the loader warmup patch lands and the daemon gains its explicit `loader.ensure_init()` call, three falsifiable gates define a sovereign boot:

### Gate A — Loader init is silent and successful

```
$ python -m knowledge3d.daemon.main --mode stdio
{"status":"ok","message":"k3d_daemon_started","mode":"stdio","timestamp":"2026-04-18T..."}
```

No `[loader]` warning lines, no `cuCtxCreate failed`, no primary-retain debug chatter unless `K3D_RPN_DEBUG=1` is set. Silence = success.

### Gate B — VRAM is live at boot

`viewer/public/runtime_boot.json` (written by daemon.py:234) shows, before the first command arrives:

```json
{
  "stage": "cuda_context_ready",
  "vram_used_mb": 132,      // or more — must be > 0
  "vram_total_mb": 12288
}
```

If `vram_used_mb == 0`, the context is not materialized — the loader patch failed or the daemon did not call `ensure_init()`.

### Gate C — Context handle is stable across ticks

Add one diagnostic command to the daemon (follow-up, not part of this spec):

```
{"command":"CTX_HANDLE"} → {"status":"ok","ctx_handle":"0x7f8b4c000000"}
```

Called at boot and again after 100 TRM ticks, both responses must return the identical `ctx_handle`. Drift = violation.

### What a sham run looks like (for contrast — so Codex can't fake it)

Per `feedback_sovereignty_check_must_not_self_deceive.md` and the 2026-04-18 incident (commit `8f69675d`, discredited): a sham run has

- `vram_used_mb == 0` but reports success,
- benchmark artifacts committed before any PTX launch actually happened,
- `numpy` imports at module scope in a file named `sovereign_*`,
- a sovereignty gate that greps only the lines between its own `# gate start` / `# gate end` markers.

None of those appear here. All three gates above are *externally* observable — they do not depend on the daemon's self-report.

---

## 8. Codex Handoff Diff (the delta, not a new spec)

The existing runbook `TEMP/CODEX_RUNBOOK_SOVEREIGN_BITNET_BENCHMARK_04.18.2026.md` (v2) remains authoritative for the benchmark side. This spec adds **two** tasks to Codex's queue. Neither creates new specs. Both are small, mechanical, and green/red gated.

### Task 1: Daemon — explicit loader.ensure_init at boot

**File**: `knowledge3d/daemon/main.py`
**Location**: `K3DDaemon.__init__`, immediately after `self._write_boot_status(stage="daemon_boot", ...)` at line 173.
**Change**: Insert the 8-line block from §5.2 above (`from knowledge3d.cranium.sovereign import loader; loader.ensure_init(); ...`).
**Verify**: daemon boot JSON includes `cuda_context_ready` stage with `vram_used_mb > 0`.
**Cost**: ~10 lines added.

### Task 2: Re-run the BitNet benchmark

**File**: none edited — re-run existing `benchmarks/sovereign_bitnet_attention.py --quick`.
**Precondition**: loader patch from `TEMP/CLAUDE_SOVEREIGN_LAZY_PRIMARY_CONTEXT_WARMUP_PATCH_04.18.2026.md` is merged.
**Verify**:
  - `get_vram_usage()` succeeds without `CUDA_ERROR_INVALID_CONTEXT`.
  - Benchmark output includes non-zero `vram_peak_mb`.
  - PTX launches are visible via `nvidia-smi dmon -s u` (nonzero SM utilization).
**Commit**: artifacts per the existing runbook.

### Task 3 (separate commit, separate PR): Violation cleanup

**Files**: the 5 production files + 2 test files listed in §4.3.
**Change**: For each `cuInit` call — replace with `loader.ensure_init()`. For each `cuDevicePrimaryCtxRetain` call — delete (the loader already did it); retrieve the context via `cuCtxGetCurrent()`. Delete `knowledge3d/cranium/ptx_runtime/galaxy_memory_updater.py.old`.
**Verify**: the §4.1 audit regex returns zero hits outside `loader.py`. Wire the audit into CI (exit-code contract from §4.4).
**Cost**: ~5 line deltas per file × 5 files = ~25 lines changed; ~40 lines deleted; 1 file removed.

All three tasks can land in separate commits. None of them require a new opcode, a new galaxy, a new bridge, or a new kernel. This spec explicitly does not grow Python.

---

## 9. What This Spec Does NOT Do

Explicit non-goals, so no one (future Claude included) re-interprets this as a wedge for scope creep:

1. **Does not grow Python.** Task 1 adds ~10 lines to `daemon/main.py`. Task 3 removes more lines than it adds. Net Python footprint: smaller. Aligned with the ~200-line boot+I/O target.
2. **Does not add orchestration.** No new router, no new dispatcher, no new scheduler. The existing tick driver, idle-triggered sleep tick, and tablet command handlers are unchanged.
3. **Does not add opcodes.** No new RPN opcodes. No touching the 0x1A0–0x1CF range (locked per `project_new_opcodes_0x1A0_range.md`). The opcode range reservation protocol (`feedback_opcode_range_reservation_protocol.md`) is unaffected.
4. **Does not reorganize galaxies.** No Galaxy Universe structural changes. House ↔ Galaxy symlinks (per `project_embodiment_gaps_identified.md`) are unaffected.
5. **Does not change the loader's policy.** The `cuCtxCreate` → `cuDevicePrimaryCtxRetain` → CuPy-bootstrap fallback chain stays as-is. Only the warmup patch (handled by `cuda-research-solver` in parallel) changes loader behavior.
6. **Does not add a benchmark.** BitNet benchmark re-run is a *verification* step, not a new artifact.

What this spec **does** do: makes the existing single-context rule — already stated in three prior specs and the Knowledgeverse spec itself — *enforceable* via a CI-gated grep, and *observable* via a boot-status JSON field. It converts an aspirational architectural principle into a contract that fails loudly when violated.

---

## 10. References (file:line, verified at spec-time)

**Canonical specs (prior art)**:
- `TEMP/CLAUDE_SOVEREIGN_CUDA_CONTEXT_FIX_11.24.2025.md` — sovereign loader pattern (full file, 200 lines).
- `TEMP/CODEX_FIX_CUDA_CONTEXT_DIRECTIVE.md` — PTX bindings application.
- `TEMP/CLAUDE_LOADING_STAGE_ARCHITECTURE_02.05.2026.md:28–34` — "ONE persistent PTX context".
- `TEMP/ARC_KNOWLEDGEVERSE_INTEGRATION_SPECIFICATION_02.07.2026.md:14` — "Legacy components initialize their own sovereign loaders/contexts, conflicting with Knowledgeverse's ONE unified context".
- `TEMP/CLAUDE_SOVEREIGN_LAZY_PRIMARY_CONTEXT_WARMUP_PATCH_04.18.2026.md` — parallel loader patch.

**Exemplar bindings (correct pattern)**:
- `knowledge3d/cranium/codecs/ptx_bindings/ternary_quant_binding.py:103–110`
- `knowledge3d/cranium/codecs/ptx_bindings/ternary_dct8x8_binding.py:99–106`
- `knowledge3d/cranium/codecs/ptx_bindings/ternary_mdct_binding.py:92–99`
- `knowledge3d/cranium/codecs/ptx_bindings/audio_harmonic_binding.py:146–153`
- `knowledge3d/cranium/bridges/n_chain_swarm_bridge.py:134`
- `knowledge3d/cranium/kernels/kernel_loader.py:37`

**Loader internals**:
- `knowledge3d/cranium/sovereign/loader.py:327` — `_ensure_init` (private).
- `knowledge3d/cranium/sovereign/loader.py:347` — `cuInit`.
- `knowledge3d/cranium/sovereign/loader.py:358` — `cuCtxCreate`.
- `knowledge3d/cranium/sovereign/loader.py:370` — `cuDevicePrimaryCtxRetain`.
- `knowledge3d/cranium/sovereign/loader.py:997` — `ensure_init` (public).
- `knowledge3d/cranium/sovereign/loader.py:1012` — `atexit.register(cleanup)`.

**Daemon**:
- `knowledge3d/daemon/main.py:173` — first `_write_boot_status` (insertion point for Task 1).
- `knowledge3d/daemon/main.py:182` — Knowledgeverse construction.
- `knowledge3d/daemon/main.py:193–195` — conditional GPU warmup (today's transitive `ensure_init`).
- `knowledge3d/daemon/main.py:206–208` — tick driver start.
- `knowledge3d/daemon/main.py:709` — idle sleep-consolidation tick.
- `knowledge3d/daemon/main.py:781` — shutdown finalization.

**Knowledgeverse**:
- `knowledge3d/knowledgeverse/knowledgeverse.py:36` — imports `loader` symbols.
- `knowledge3d/knowledgeverse/sovereign_hot_path.py:979` — `ensure_loaded` (transitive trigger point today).

**Audit violations** (see §4.3 for full list).

**Memory files cited (implicit contract)**:
- `feedback_sovereignty_audit_is_full_tree_not_line_patch.md` — audits are whole-tree, not line-patches.
- `feedback_python_dispatch_is_not_a_line_item.md` — Python drift is the partner's responsibility to catch.
- `feedback_sovereignty_check_must_not_self_deceive.md` — don't author sham gates.
- `feedback_codex_cannot_silent_fix_to_unblock.md` — don't manufacture passes.
- `feedback_no_fallbacks_ever_including_sleeptime.md` — fail fast, don't patch around.
- `project_benchmarks_as_natural_activity.md` — benchmarks aren't modes.
- `project_embodiment_gaps_identified.md` — 6 embodiment gaps; this spec closes the context-identity gap.

---

## 11. Sign-off criteria

This spec is done when all of the following are true:

- [ ] `cuda-research-solver` lands the loader warmup patch (parallel work; contract defined in §3).
- [ ] Codex executes Task 1 (daemon `loader.ensure_init()` at boot, §8).
- [ ] Codex executes Task 2 (BitNet re-run, §8).
- [ ] Codex executes Task 3 (violation cleanup, §8) — production files and `.old` removal. Test rewrites can follow in a separate PR.
- [ ] CI gate (§4.4) is wired and green.
- [ ] Daemon boot emits `cuda_context_ready` stage with `vram_used_mb > 0` (Gate B, §7).

Only then is the "one context, always-on, living AI" claim architecturally enforceable rather than merely repeated.
