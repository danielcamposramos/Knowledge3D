# Kernel Audit — Phase Wiring for Persistent TRM Tick

**Date:** 2026-04-18
**Auditor:** Agent 1 (research-only, no source edits)
**Spec audited:** `TEMP/CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md`
**Repo root:** `/K3D/GitHub/Knowledge3D`

---

## 1. Phase → Kernel Wiring Table

| Phase | Kernel (file:lines) | Entry symbol | Input descriptor | Output descriptor | Status | Notes |
|-------|---------------------|--------------|------------------|-------------------|--------|-------|
| PERCEIVE | `cranium/ptx/morton_octree.cu:78–117` | `compute_morton_codes` | `float* positions[N,3]`, bbox floats | `uint32_t* morton_codes[N]` | **PRESENT** | Standalone `.cu` + `.ptx`. **NOT wired** into persistent tick. `trm_step_fused.cu` has its own inline Morton (`trm_encode_position_morton`). |
| PERCEIVE | `cranium/ptx/morton_octree.cu:172–208` | `octree_query_morton` | sorted Morton array, query Morton + radius | `uint32_t* result_buffer`, `result_count` | **PRESENT** | Single-threaded MVP. Not in tick. |
| PERCEIVE | `cranium/ptx/frustum_cull_simd.ptx:15` | `warp_frustum_cull_simd` | `u64 positions_ptr`, `u64 candidates_ptr`, `u32 node_count`, `u64 visible_flags_ptr` | `visible_flags` bitmask | **PRESENT (PTX-only)** | No `.cu` source. Targets `sm_80`. **Not wired** into tick. `trm_perceiving_phase` in `trm_step_fused.cu` has an inline frustum + dot-product loop. |
| NAVIGATE | `cranium/ptx/led_astar.cu:164–288` | `led_astar_navigate` | `DependencyKernel* kernel`, `start`, `goal`, `alpha`, `beta`, `path[]`, `pathLength`, `maxPathLength` | path array + length | **PRESENT** | `.cu` + `.ptx`. **Not wired** into tick. `trm_navigating_phase` uses its own inline graph walk. |
| NAVIGATE | `cranium/kernels/cosine_similarity.cu:3–19` | `cosine_similarity_batch` | `float* candidates[N,D]`, `float* expected[D]`, `int N, D` | `float* scores[N]` | **PRESENT** | `.cu` + `.ptx` (`cranium/ptx/cosine_similarity.ptx`). Not called in tick. |
| REASON (swarm) | `cranium/kernels/nine_chain_swarm_kernel.cu:111–176` | `nine_chain_swarm_kernel` | `float* input_embedding`, `float* chain_states`, `float* output_embedding`, `float* resonance_scores`, `int num_iterations` | chain states + resonance scores | **PRESENT** | `.cu` + `.ptx`. The *global* kernel is **not wired into tick**. `trm_step_fused.cu` calls `nine_chain_swarm_device` (inline `__device__` from `cuda/device_functions.cuh:705`). |
| REASON (conflict) | `cranium/kernels/gre_defeasible_resolver.cu:67–152` | `gre_defeasible_resolver` | `float* conclusions[W,C]`, `int8_t* rule_strengths`, `uint32_t* superiority`, `float* verdicts`, `uint32_t* proof_tags`, `num_workers`, `num_candidates`, `max_superiors` | `verdicts[]`, `proof_tags[]` | **PRESENT** | `.cu` + `.ptx`. Loaded via `sovereign_bridges.py:1298–1299`. **Not called on hot path** — only in Python bridge dispatch. |
| MULTI-HOP | `cranium/kernels/gre_graph_crystallizer.cu:13–73` | `gre_graph_crystallizer` | `float* node_features[N,D]`, `int* adjacency`, `int* neighbor_counts`, `float* refined_features`, `node_count`, `feature_dim`, `max_neighbors`, `self_weight`, `neighbor_weight` | `float* refined_features[N,D]` | **PRESENT** | `.cu` + `.ptx`. Multi-hop controlled by **Python bridge** (kernel launched once per hop, buffers swapped by Python). Source comment line 7 explicitly states this. Not in tick. |
| RPN dispatch | `cranium/kernels/modular_rpn_kernel.cu:982` | `modular_rpn_geometric_kernel` | `uint32_t instance_id`, `uint16_t* op_codes`, `float* scalars`, `float* vectors`, `InstanceState* states`, `uint32_t token_count` | `InstanceState` mutations | **PRESENT** | `.cu` + `.ptx`. Physics opcodes `0x150–0x162` (19 cases) present. Gap at `0x163–0x17F`. Python dispatches it today. |
| PHYSICS | `cranium/kernels/physics_integrate.cu:23–47` | `physics_integrate` | `PhysicsBodySOA* bodies`, `PhysicsPredictedSOA predicted`, `float dt`, `float gravity_y` | mutated `bodies` | **PRESENT** | `.cu` only (no `.ptx`). Called via RPN opcode `0x154` inside `modular_rpn_geometric_kernel`. |
| DECIDE | `cranium/cuda/device_functions.cuh:775` | `halting_gate_device` (__device__) | `float* scores`, `int num_scores`, `float min/gap/agreement_threshold` | `int` (converged flag) | **PRESENT** | Inlined `__device__` fn. Used inside `trm_step_fused.cu:815`. Standalone `cuda/halting_gate.cu` has `halting_gate_step` (different, `__device__` only). |
| TICK ENTRY | `cranium/ptx/trm_step_fused.cu:908` | `trm_step_fused` | See §2 below | mutated entity/ring state | **PRESENT** | Not a persistent/cooperative kernel. Per-query launch today. No `grid.sync()`, no `while(!shutdown)` loop. |

---

## 2. Kernel Signatures (PRESENT kernels)

### 2.1 `compute_morton_codes` — `cranium/ptx/morton_octree.cu:78`
```cuda
extern "C" __global__ void compute_morton_codes(
    const float* __restrict__ positions,  // [N, 3] — AoS interleaved xyz
    uint32_t node_count,
    uint32_t* __restrict__ morton_codes,  // [N] output
    float bbox_min_x, float bbox_min_y, float bbox_min_z,
    float bbox_size
)
```

### 2.2 `octree_query_morton` — `cranium/ptx/morton_octree.cu:172`
```cuda
extern "C" __global__ void octree_query_morton(
    const uint32_t* __restrict__ morton_sorted,
    const uint32_t* __restrict__ node_ids_sorted,
    uint32_t total_nodes,
    uint32_t query_morton,
    uint32_t query_radius,
    uint32_t* __restrict__ result_buffer,
    uint32_t* __restrict__ result_count,
    uint32_t max_results
)
```
**Wiring note:** Single-threaded (tid != 0 returns). Must be launched `<<<1,1>>>`.

### 2.3 `warp_frustum_cull_simd` — `cranium/ptx/frustum_cull_simd.ptx:15`
```ptx
.entry warp_frustum_cull_simd(
    .param .u64 positions_ptr,   // [N × float3] world-space
    .param .u64 candidates_ptr,  // [N × uint32] node indices
    .param .u32 node_count,
    .param .u64 visible_flags_ptr // [N × u8] output
)
```
Reads 6-plane frustum from `.const` `view_proj[16]` and `view_matrix[16]`.
Target: `sm_80`. Will fail to load on sm < 80 (e.g. sm_75 + Turing).

### 2.4 `led_astar_navigate` — `cranium/ptx/led_astar.cu:164`
```cuda
extern "C" __global__ void led_astar_navigate(
    const DependencyKernel* kernel,  // CSR adjacency: numVertices, edges[]
    uint32_t start,
    uint32_t goal,
    float alpha,                     // heuristic weight
    float beta,                      // edge cost weight
    uint32_t* path,
    uint32_t* pathLength,
    uint32_t maxPathLength
)
```
Uses dynamic shared memory: 5 arrays × `KERNEL_MAX_SIZE` elements.

### 2.5 `cosine_similarity_batch` — `cranium/kernels/cosine_similarity.cu:3`
```cuda
extern "C" __global__ void cosine_similarity_batch(
    const float* candidates,  // [N, D]
    const float* expected,    // [D]
    float* scores,            // [N] output
    int N, int D
)
```

### 2.6 `nine_chain_swarm_kernel` — `cranium/kernels/nine_chain_swarm_kernel.cu:111`
```cuda
extern "C" __global__ void nine_chain_swarm_kernel(
    const float* __restrict__ input_embedding,
    float* __restrict__ chain_states,
    float* __restrict__ output_embedding,
    float* __restrict__ resonance_scores,
    int num_iterations
)
```
Launched with `gridDim.x = 9` (one block per chain). `NUM_CHAINS = 9`, `CHAIN_STATE_DIM` hardcoded constant.

### 2.7 `nine_chain_swarm_device` — `cranium/cuda/device_functions.cuh:705` (__device__)
```cuda
__device__ void nine_chain_swarm_device(
    const float* query_embedding,
    float* chain_states,
    float* output_embedding,
    float* resonance_scores,
    int num_iterations
)
```
Inline device version (no block-level launch). Called by `trm_step_fused.cu:805` and `gpu_task_dispatch.cu:421`.

### 2.8 `gre_defeasible_resolver` — `cranium/kernels/gre_defeasible_resolver.cu:67`
```cuda
extern "C" __global__ void gre_defeasible_resolver(
    const float* __restrict__ conclusions,    // [num_workers × num_candidates]
    const int8_t* __restrict__ rule_strengths,
    const uint32_t* __restrict__ superiority, // [num_workers × max_superiors]
    float* __restrict__ verdicts,
    uint32_t* __restrict__ proof_tags,
    int num_workers,
    int num_candidates,
    int max_superiors
)
```
Single block. `MAX_WORKERS` hardcoded (check header). A second entry `gre_defeasible_resolver_ethical` at line 153.

### 2.9 `gre_graph_crystallizer` — `cranium/kernels/gre_graph_crystallizer.cu:13`
```cuda
extern "C" __global__ void gre_graph_crystallizer(
    const float* __restrict__ node_features,  // [N × D]
    const int* __restrict__ adjacency,        // [N × max_neighbors]
    const int* __restrict__ neighbor_counts,  // [N]
    float* __restrict__ refined_features,     // [N × D] output
    int node_count,
    int feature_dim,
    int max_neighbors,
    float self_weight,
    float neighbor_weight
)
```
**Critical note:** Source comment (lines 1–9) says multi-hop K rounds are orchestrated by the Python bridge via repeated launches + buffer swaps. Grid-wide sync is absent.

### 2.10 `modular_rpn_geometric_kernel` — `cranium/kernels/modular_rpn_kernel.cu:982`
```cuda
extern "C" __global__ void modular_rpn_geometric_kernel(
    uint32_t instance_id,
    const uint16_t* __restrict__ op_codes,
    const float* __restrict__ scalars,
    const float* __restrict__ vectors,
    InstanceState* __restrict__ states,
    uint32_t token_count
)
```
Reads constants from `__device__ __constant__` pointers (galaxy, query embedding, physics SoA, etc.). Physics opcodes 0x150–0x162 (19 cases). Opcode 0x163–0x17F: **absent** (no cases).

### 2.11 `physics_integrate` — `cranium/kernels/physics_integrate.cu:23`
```cuda
extern "C" __global__ void physics_integrate(
    PhysicsBodySOA* __restrict__ bodies,
    const PhysicsPredictedSOA predicted,
    float dt,
    float gravity_y
)
```

### 2.12 `trm_step_fused` (current, non-persistent) — `cranium/ptx/trm_step_fused.cu:908`
```cuda
extern "C" __global__ void trm_step_fused(
    const float* __restrict__ q,       // query embedding
    const float* __restrict__ y,       // entity state
    const float* __restrict__ z,       // workspace state
    const float* __restrict__ W1..W4,  // TRM weights
    float* __restrict__ z_new,
    float* __restrict__ y_new,
    float* __restrict__ workspace,
    const void* __restrict__ physics_soa_ptr,
    const void* __restrict__ contact_soa_ptr,
    unsigned int body_count,
    unsigned int solver_iterations,
    void* __restrict__ ring_buffer_ptr,
    uint32_t* __restrict__ head_ptr,
    uint32_t* __restrict__ tail_ptr,
    TRMStateMachine* __restrict__ state_machine_ptr,
    void* __restrict__ entity_hot_path_ptr,
    unsigned int entity_count,
    float delta_time,
    unsigned long long tick,
    int max_steps,
    float epsilon,
    int* __restrict__ steps_out,
    float* __restrict__ drift_out,
    const void* __restrict__ galaxy_table_ptr,
    unsigned int galaxy_star_count,
    void* __restrict__ action_buffer_out,
    const void* __restrict__ program_table_ptr,
    const void* __restrict__ action_buffer_in_ptr
)
```
**Not the spec §2.2 signature.** The spec defines a persistent-tick signature with input/output rings, `GalaxyUniverse*`, `MortonOctree*`, `TRMWeights*`, `SpecialistPool*`, `VramFreelist*`, `tick_counter`, `tick_status`, `shutdown_flag`. Those parameters do not exist in the current kernel.

### 2.13 `halting_gate_device` — `cranium/cuda/device_functions.cuh:775` (__device__)
```cuda
__device__ int halting_gate_device(
    const float* scores,
    int num_scores,
    float min_threshold,
    float gap_threshold,
    float agreement_threshold
)
```

---

## 3. Missing or Misnamed Kernels

| Spec Name | Path Claimed | Reality | Resolution |
|-----------|-------------|---------|------------|
| `frustum_cull.cu` | `cranium/ptx/frustum_cull.cu` or `kernels/frustum_cull.cu` | **Does not exist**. Near-match: `cranium/ptx/frustum_cull_simd.ptx` (PTX-only, `sm_80`). Python frustum logic is in `cranium/spatial_sovereign/frustum.py`. | Use `frustum_cull_simd.ptx` as the wiring target, or generate a `.cu` wrapper. Note: PTX targets sm_80 only. |
| `morton_octree.ptx` (as PERCEIVE) | `cranium/ptx/morton_octree.ptx` | **File exists** (`cranium/ptx/morton_octree.ptx` — compiled PTX) and source `cranium/ptx/morton_octree.cu`. But the persistent tick's PERCEIVE currently uses **inline Morton** inside `trm_step_fused.cu`, not the standalone kernel. | Standalone kernel is available; needs explicit wiring. |
| `physics_integrator.cu` | spec §3 "physics kernels" | **Does not exist at that name**. Actual file: `cranium/kernels/physics_integrate.cu`. | Rename reference in any doc to `physics_integrate.cu`. |
| `ring_atomics.cuh` | `cranium/ptx_kernels/ring_atomics.cuh` | **Does not exist**. Directory `ptx_kernels/` does not exist. Closest: `cranium/cuda/trm_game_loop.cuh` contains `trm_event_queue_push/pop` with `atomicCAS + __threadfence`. Does NOT have `ld.global.acquire.u32` or `st.global.release.u32` (uses older `atomicCAS + __threadfence` idiom). | Must be created per spec §2.6. |
| `persistent_tick.cu` | `cranium/ptx_kernels/persistent_tick.cu` | **Does not exist**. Directory `ptx_kernels/` absent. | Must be created per spec §2.2. |
| `wine_contract_scan.cu` | `cranium/ptx_kernels/wine_contract_scan.cu` | **Does not exist**. | Must be created per spec §4.2. |
| `matryoshka_prefix_dot.cu` | `cranium/ptx_kernels/matryoshka_prefix_dot.cu` | **Does not exist**. Near-match: `cranium/ptx/matryoshka_project.cu` (projection, not fused prefix dot). | Must be created per spec §5.2. |
| `vram_freelist.cu` | `cranium/ptx_kernels/vram_freelist.cu` | **Does not exist**. | Must be created per spec §2.1. |
| `log_ring.cu` | `cranium/ptx_kernels/log_ring.cu` | **Does not exist**. Near-match: `cranium/cuda/lane_perf_ring.cu` (warp-level LanePerf struct with `atomicInc`; no `ld.global.acquire.u32` semantics). | Must be created per spec §2.1. |
| `gre_embedding_extractor` | any path | **Does not exist** — no `.cu`, no `.ptx`. Listed in `FIXED_GRE_WORKERS` in `knowledgeverse.py:375`. | Missing kernel. Needs authoring or removal from loaded list. |

---

## 4. Orphaned Kernels

### 4.1 GRE Specialist Kernels — Loaded but Uncalled on Hot Path

15 `.ptx` files in `kernels/` constitute the GRE specialist pool. 9 appear in `FIXED_GRE_WORKERS`; all are loaded via Python bridge objects in `sovereign_bridges.py`. None are called inside a sovereign PTX hot path — they are dispatched by Python via `knowledgeverse.py:_dispatch_sovereign_task` (the method spec §8 mandates deleting).

| Kernel file | .cu source? | In FIXED_GRE_WORKERS | In sovereign_bridges.py | Hot-path call | Proposed tick phase |
|-------------|------------|----------------------|------------------------|---------------|---------------------|
| `gre_atomic_fission_fusion.ptx` | yes | yes | yes (line 1128) | no | REASON conflict |
| `gre_resonance_field.ptx` | yes | yes | yes (resonance) | no | REASON scoring |
| `gre_vector_resonator.ptx` | yes | yes | yes (line 1513) | no | NAVIGATE scoring |
| `gre_arc_reasoner.ptx` | **NO** (PTX-only) | yes | no | no | PERCEIVE (ARC3 frame) |
| `gre_geometry_router.ptx` | yes | yes | no | no | NAVIGATE |
| `gre_graph_crystallizer.ptx` | yes | yes | yes (line 1663) | no (Python multi-hop) | MULTI-HOP |
| `gre_temporal_reasoning.ptx` | yes | yes | yes (line 1398) | no | REASON temporal |
| `gre_fractal_emitter.ptx` | yes | yes | no | no | ACT synthesis |
| `gre_embedding_extractor` | **MISSING** | yes | no | no | NAVIGATE (missing) |
| `gre_cognitive_executive.ptx` | yes | no | yes (line 1001) | no | DECIDE |
| `gre_multimodal_halting_gate.ptx` | yes | no | yes (line 2048) | no | DECIDE |
| `gre_oom_spill.ptx` | yes | no | yes (line 680) | no | sleep-time OOM |
| `gre_world_model.ptx` | yes (via ptx/ dir) | no | yes (line 3767) | no | REASON world model |
| `gre_recursive_refiner.ptx` | **NO** (PTX-only) | no | no | no | sleep-time refiner |
| `gre_sub100micro_gate.ptx` | **NO** (PTX-only) | no | no | no | DECIDE latency gate |
| `gre_trm_core.ptx` | **NO** (PTX-only) | no | no | no | TRM core operations |

### 4.2 Other Orphaned Kernels Worth Noting

- `cranium/cuda/k3d_swarm_persistent.cu` — entry `k3d_swarm_sovereign` (line 109): a persistent-style sovereign swarm with `while(true)` loop, `K3D_SWARM_FLAG_SHUTDOWN` check. This is the closest existing match to what the spec §2 wants but it is a **separate kernel** from `trm_step_fused`. Not wired into persistent tick.
- `cranium/ptx/trm_recursive_fused.cu` — a recursive TRM variant. Separate from `trm_step_fused`.
- `cranium/ptx/trm_state_machine.cu` — standalone state machine kernel.
- `cranium/cuda/halting_gate.cu` — defines `halting_gate_step` (`__device__` only, no `__global__` entry). Different from `halting_gate_device` in `device_functions.cuh`.
- `cranium/kernels/gre_multimodal_halting_gate.cu` — standalone `__global__` entry, not currently wired.
- Physics kernels (11 `.cu` files: `physics_broad_phase_sap`, `physics_collision_event_write`, `physics_constraint_color`, `physics_constraint_generate`, `physics_integrate`, `physics_narrow_phase_gjk`, `physics_raycast`, `physics_sleep_island`, `physics_spawn`, `physics_xpbd_predict`, `physics_xpbd_solve`) — all reachable through RPN opcodes 0x150–0x162 in `modular_rpn_geometric_kernel`, but that kernel is dispatched by Python today.

---

## 5. Ring Buffer / Atomic Primitives Already Present

### 5.1 `cranium/cuda/trm_game_loop.cuh` — lines 103–168
**Primary ring primitive.** Contains:
- `trm_ring_next(index)` — modular increment with `TRM_EVENT_RING_MASK`
- `trm_event_queue_push(ring, head, tail, event)` — CAS spin loop (`atomicCAS`) + `__threadfence()` release before writing `event_type`
- `trm_event_queue_pop(ring, head, tail, out)` — CAS spin loop + read under CAS

**Gap vs spec §2.6:** Uses `atomicCAS + __threadfence()` (device-device fence). Spec requires `ld.global.acquire.u32` / `st.global.release.u32` / `membar.sys` (host-pinned zero-copy fences). These are different memory scopes: `__threadfence` covers device scope; `membar.sys` covers system (host+device). The new `ring_atomics.cuh` must use PTX inline `membar.sys` for host-pinned input/output rings. The existing primitives are reusable for device-side VRAM rings (log ring, event ring).

### 5.2 `cranium/cuda/lane_perf_ring.cu` — `lane_perf_write` function
Uses `atomicInc(ring_head, ...)` — simple overwrite ring without ordering guarantee. No acquire/release. Suitable for best-effort perf telemetry only.

### 5.3 `cranium/cuda/gpu_event_queue.cu`
Kernel wrappers (`gpu_event_queue_reset`, `gpu_event_queue_enqueue_stress`, etc.) that call `trm_event_queue_push/pop`. Re-usable as device-side VRAM event ring; not host-pinned.

### 5.4 `cranium/sovereign/loader.py:900` — `launch_cooperative`
Python wrapper for `cuLaunchCooperativeKernel` (Driver API). Fully implemented and callable. Used by `n_chain_swarm_bridge.py`. The `trm_boot.py` (confirmed present, 20 lines) calls `ctx.cuda.launch_cooperative_kernel` — verify it routes to this function.

**Summary:** The CAS-based device ring is present and correct for VRAM rings. A `ring_atomics.cuh` with `membar.sys` semantics must be written for the host-pinned zero-copy rings (input/output to Python).

---

## 6. Composition Order Inside the Persistent Tick

The following is the concrete call sequence a kernel author should follow when wiring `persistent_tick.cu`. Each step lists the **existing** symbol to call (not a new kernel to write), plus the mismatch that needs resolving first.

```
// === BOOT (Python, once) ===
// trm_boot.py: launch_cooperative(trm_step_fused_NEW, grid=SM_COUNT, block=256, shared=49152)

// === PERSISTENT LOOP (device, runs forever) ===
while (!ld.global.volatile.u32(shutdown_flag)) {
    grid.sync();    // cooperative groups: cuda::grid_group::sync()
                    // NOT bar.sync (block-local); requires cudaLaunchCooperativeKernel

    // --- POLL INPUT RING (block 0, thread 0) ---
    // Use ring_atomics.cuh (to be written): membar.sys acquire-load of input_ring_head
    // Populate shared work descriptor: star_id, query_bytes, paradigm_type

    grid.sync();

    // === PERCEIVE ===
    // Option A (standalone): launch compute_morton_codes + octree_query_morton
    //   cranium/ptx/morton_octree.cu: compute_morton_codes<<<N/128,128>>>(positions, N, morton, bbox)
    //   cranium/ptx/morton_octree.cu: octree_query_morton<<<1,1>>>(sorted, ids, N, qmorton, radius, buf, cnt, max)
    //   cranium/ptx/frustum_cull_simd.ptx: warp_frustum_cull_simd<<<N/32,32>>>(pos, cands, N, flags)
    //
    // Option B (current, lower overhead): call trm_perceiving_phase() device fn inline
    //   cranium/ptx/trm_step_fused.cu:339  (already inline Morton + frustum dot)
    //
    // BLOCKER: frustum_cull_simd.ptx targets sm_80; RTX 3070 is sm_86 (compatible).
    //          octree_query_morton is single-threaded — use as device function, not kernel.
    //          Recommended: refactor octree_query_morton into a __device__ fn callable inline.

    // === NAVIGATE ===
    // cranium/ptx/led_astar.cu: led_astar_navigate<<<1, BLOCK>>>(kernel, start, goal, alpha, beta, path, len, maxLen)
    //   requires DependencyKernel (CSR adjacency) resident in VRAM
    // cranium/kernels/cosine_similarity.cu: cosine_similarity_batch<<<N/128,128>>>(cands, query, scores, N, D)
    //   for scoring LED-A* candidate set
    //
    // BLOCKER: DependencyKernel (CSR) must be materialized in Galaxy VRAM at boot.
    //          led_astar uses dynamic shared memory; compatible with persistent kernel.

    // === REASON (swarm, 9 lanes) ===
    // Call nine_chain_swarm_device() inline (cranium/cuda/device_functions.cuh:705)
    //   OR launch nine_chain_swarm_kernel<<<9, BLOCK>>>(embed, states, out, scores, iters) as child kernel
    //   Inline device fn is preferred for cooperative persistent kernel (no nested launch).
    //
    // For specialist selection: meta_select_specialist_lane RPN program (Layer 4, to be seeded)
    //   Reads PERCEIVE output signal σ → returns 9 specialist IDs
    //   Specialists index SpecialistPool.descriptors[] (LoRA weight slices, VRAM-resident)
    //
    // BLOCKER: SpecialistPool struct not yet defined in persistent tick signature.

    // === REASON (conflict resolution) ===
    // Call gre_defeasible_resolver inline as __device__ fn
    //   OR: gre_defeasible_resolver<<<1, BLOCK>>>(conclusions, rule_strengths, superiority, verdicts, tags, W, C, K)
    //   NOTE: Currently __global__ only; needs a __device__ wrapper for inline call inside persistent kernel.
    //
    // BLOCKER: gre_defeasible_resolver.cu has no __device__ wrapper. Create one or use dynamic parallelism.

    // === MULTI-HOP ===
    // gre_graph_crystallizer: currently requires Python-orchestrated multi-launch.
    //   To move on-device: run K rounds via an inner loop WITHIN the persistent kernel body,
    //   double-buffering node_features_A / node_features_B in VRAM.
    //   cranium/kernels/gre_graph_crystallizer.cu:13 (single round)
    //
    // BLOCKER: Multi-hop loop must be on-device; existing kernel does one round only.
    //          Double-buffer allocation requires vram_freelist.cu (to be written).

    // === PHYSICS ===
    // Opcodes 0x150–0x162 in modular_rpn_geometric_kernel already call physics kernels.
    // For direct wiring in tick: physics_integrate, physics_xpbd_predict, physics_xpbd_solve, etc.
    //   cranium/kernels/physics_integrate.cu:23
    //   cranium/kernels/physics_xpbd_predict.cu (see entry)
    //   cranium/kernels/physics_xpbd_solve.cu (see entry)
    //
    // BLOCKER: No PHYSICS phase stub in current trm_step_fused. Wire trm_phase2_physics_step (line 557)
    //          which exists but uses inline SOA operations, not calling standalone kernels.

    // === DECIDE (halting gate) ===
    // halting_gate_device inline (device_functions.cuh:775) — already wired in trm_step_fused.cu:815
    // OR gre_multimodal_halting_gate for multi-modal convergence check
    //   cranium/kernels/gre_multimodal_halting_gate.cu:8

    // === ACT ===
    // trm_acting_phase() inline device fn (trm_step_fused.cu:519)
    // Write answer to output_ring using ring_atomics.cuh (membar.sys release store)

    atomicAdd(tick_counter, 1);
}
```

---

## 7. Open Questions / Ambiguities for Claude

1. **`frustum_cull_simd.ptx` targets `sm_80`; RTX 3070 is `sm_86`.**
   sm_86 is forward-compatible with sm_80 PTX, so it will JIT-compile. But if performance matters, recompile for `sm_86`. Confirm target before wiring.

2. **`octree_query_morton` is single-threaded (`<<<1,1>>>`).**
   Cannot run efficiently inside a persistent cooperative kernel. Recommend extracting its binary search into a `__device__` inline function and calling it from thread 0 of block 0. Claude should decide: standalone launch (device-side dynamic parallelism, requires sm_35+ compute capability 3.5) vs. inline refactor.

3. **`gre_defeasible_resolver` and `gre_graph_crystallizer` are `__global__` only.**
   The persistent tick is one cooperative kernel. Calling `__global__` from inside a `__global__` requires dynamic parallelism (CDP). CDP on sm_86 is available but adds latency and has depth limits. Alternative: extract core logic to `__device__` functions. Claude should specify which approach.

4. **`gre_graph_crystallizer` multi-hop orchestration.**
   Source comment says Python controls hop count. For sovereign operation, this must move on-device. The double-buffer scheme needs `vram_freelist.cu`. Claude should confirm the max hop count K and resulting VRAM budget.

5. **`gre_embedding_extractor` is missing entirely.**
   It appears in `FIXED_GRE_WORKERS` (`knowledgeverse.py:375`) but no `.cu` or `.ptx` file exists anywhere in the repo. Either author it or remove from the loaded list. The loaded list bootstraps specialist dispatch, so a missing kernel may cause a silent boot error. Claude should specify its interface.

6. **`trm_step_fused` signature mismatch with spec §2.2.**
   The current kernel (line 908) takes `q, y, z, W1..W4, physics_soa_ptr, ...`. The spec §2.2 wants `input_ring_head, input_ring_tail, QuerySlot*, output_ring_head, ..., GalaxyUniverse*, MortonOctree*, TRMWeights*, SpecialistPool*, VramFreelist*, tick_counter, tick_status, shutdown_flag`. These are completely different parameter lists. A new kernel `persistent_tick.cu` must be created in `ptx_kernels/`; the current `trm_step_fused` is the *per-entity game-NPC* kernel, not the *persistent query processor*. Claude should confirm naming convention and whether `trm_step_fused` stays as-is for entity simulation while `persistent_tick` handles the query path.

7. **`k3d_swarm_persistent.cu` (`k3d_swarm_sovereign`) is the closest existing persistent-loop kernel.**
   It has a `while(true)` + shutdown check + `K3D_SWARM_FLAG_RUN` polling. Claude should evaluate whether `persistent_tick.cu` should be built *from* this kernel vs. from scratch.

8. **Opcode gap 0x163–0x17F in `modular_rpn_geometric_kernel`.**
   19 physics opcodes exist (0x150–0x162). The spec mentions 0x150–0x17F as the physics range (48 opcodes total). 29 are absent. Claude should decide which physics sub-operations belong there before Codex adds cases.

9. **`trm_boot.py` exists (20 lines, confirmed) but calls `ctx.cuda.launch_cooperative_kernel`.**
   Verify `ctx.cuda` resolves to `cranium/sovereign/loader.py:launch_cooperative`. The boot file is a correct skeleton but the `ctx` contract is not formally specified.

10. **`ring_atomics.cuh` for host-pinned rings vs. device-side rings.**
    Two distinct semantic requirements: (a) `membar.sys` fences for zero-copy host-pinned I/O rings (input from Python, output to Python); (b) `__threadfence()` or device-scope fences for pure VRAM rings (log ring, event ring). The spec conflates these. Claude should clarify which fence scope each ring uses.

---

*Report generated by Agent 1. No source files modified.*
