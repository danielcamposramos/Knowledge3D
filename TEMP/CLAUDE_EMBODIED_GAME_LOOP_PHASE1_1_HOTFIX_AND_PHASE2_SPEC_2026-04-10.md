# CLAUDE Embodied Game Loop — Phase 1.1 Hotfix + Phase 2 Spec

**Date:** 2026-04-10
**Author:** Claude (Architecture Partner)
**Target:** Codex (Implementation)
**Branch base:** `codex/embodied-tick-phase1-2026-04-10`
**Input review:** `TEMP/KIMI_SWARM_PHASE1_REVIEW_2026-04-10.md`, `ask_coder` audit, `CODEX_EMBODIED_GAME_LOOP_PHASE1_REPORT_2026-04-10.md`

---

## 0. MCP Notice (Save Tokens — Use This First)

Before reading any K3D doc from disk, query the MCP knowledge base. The K3D architectural context is already indexed there — you do not need to read `docs/vocabulary/*.md` or `docs/briefings/*.md` piece by piece.

**MCP servers you should have wired up** (check `~/.codex/config.toml`):

```toml
[mcp_servers.ollama-specialists]
url = "http://localhost:8502/mcp"
enabled = true

[mcp_servers.k3d-knowledge]
url = "http://localhost:8501/mcp/"
enabled = true
```

If either MCP is not responding, run: `bash ~/.claude/launch_mcp_containers.sh` and verify with `docker ps | grep -E 'ollama-specialists|k3d-knowledge'`. The ollama-specialists container must serve `streamable-http` on port 8502 at path `/mcp` (no trailing slash). The k3d-knowledge container serves on port 8501 at `/mcp/` (with trailing slash).

**Required workflow for this spec:**

1. `plan_task` (ollama-specialists) — before touching any kernel, ask the planner to decompose this spec into concrete file edits
2. `qdrant-find` (k3d-knowledge) — for any architectural term you're unsure about (e.g., `"EntityHotPath"`, `"composed head pipeline"`, `"Morton Octree"`), query the knowledge base first
3. `ask_coder` (ollama-specialists) — draft CUDA/PTX code against the existing kernels before editing
4. `kimi_swarm` (ollama-specialists, `think=true`) — review each Phase 2 subsystem before marking it complete

This keeps context lean and avoids re-reading 4000-line docs every session.

---

## 1. What Phase 1 Got Right (Keep)

Phase 1 is architecturally sound. Do not rewrite:

- GPU event ring buffer (`trm_event_queue_push/pop` with atomicCAS on head/tail) — correct lock-free MPSC for GPU-side producers
- `TRMStateMachine` struct packing (32 bytes, `_pack_=1`, matches bridge)
- State-gated dispatch in `trm_step_fused.cu` — switch on `current_state` broadcast through `__shared__` is the right pattern
- `trm_state_machine_step_device` runs single-threaded under `tid==0`, broadcasts via `__shared__` — safe
- `TRMStepFusedBridge` as the single runtime entrypoint (fast-lane delegates through it)
- `knowledgeverse._run_single_trm_tick` now routes through `TRMLauncher.run_query_tick()` — sovereignty preserved

The fused tick contract — `ring_buffer_ptr, head_ptr, tail_ptr, state_machine_ptr, entity_hot_path_ptr, entity_count, delta_time, tick` — is the correct ABI. Keep it.

---

## 2. Phase 1.1 Hotfixes (Land BEFORE Phase 2)

These are P0 bugs identified by the Kimi swarm + ask_coder dual review. They will corrupt state the moment Phase 2 wires concurrent producers.

### 2.1 GPUEvent 16-Byte Aligned Ring Allocation (P0)

**Problem:** `trm_game_loop.cuh` declares `struct alignas(16) GPUEvent` but the Python bridge allocates the ring via `cuMemAlloc(TRM_EVENT_RING_CAPACITY * sizeof(GPUEvent))` with no alignment contract. CUDA `cuMemAlloc` is guaranteed 256-byte aligned at the base, so in practice this is fine for the base pointer — but the `_GPUEventStruct` in the bridge uses `_pack_=1`, which is the *opposite* of `alignas(16)`. If the bridge ever slices the ring buffer via ctypes pointer arithmetic on `_GPUEventStruct`, offsets will compute wrong.

**Fix:**

- In [trm_step_fused_bridge.py](knowledge3d/cranium/bridges/trm_step_fused_bridge.py): remove `_pack_ = 1` from `_GPUEventStruct`. Use ctypes natural alignment, which will match the C++ `alignas(16)` because all fields fit naturally into 16 bytes. Keep the `assert ctypes.sizeof(_GPUEventStruct) == 16`.
- Add a bridge-side assert after `cuMemAlloc` for the ring buffer: `assert int(ring_ptr) % 16 == 0` — CUDA guarantees it, but pin it so a future refactor can't silently break the contract.
- Keep `_pack_ = 1` on `_TRMStateMachineStruct` (it's deliberately packed to 32 bytes in the C++ header — matches `#pragma pack(push, 1)` in `trm_game_loop.cuh`).

### 2.2 `__threadfence()` After `trm_state_pop` in HANDLING_QUERY (P0)

**Problem:** In [trm_step_fused.cu:254](knowledge3d/cranium/ptx/trm_step_fused.cu#L254), after `trm_state_pop`, the single-threaded write to `state_machine_ptr[0]` and `entity_hot_paths[0].sleep_state` must be visible to the GPU event queue before the next tick runs. Currently there is only a `__syncthreads()` at the end of the kernel, which is a CTA barrier, not a device-memory visibility fence.

**Fix:** Add `__threadfence()` inside the `tid == 0` block in the `HANDLING_QUERY` case, after the state pop and hot-path write, before falling through to the bottom of the switch:

```cuda
case TRM_STATE_HANDLING_QUERY:
    trm_reasoning_phase(...);
    __syncthreads();
    if (tid == 0 && state_machine_ptr != nullptr && entity_count > 0u) {
        state_machine_ptr[0].deferred_event_mask &= ~TRM_DEFERRED_QUERY_POP;
        state_machine_ptr[0].interrupt_priority_level = 0u;
        trm_state_pop(state_machine_ptr[0], static_cast<uint64_t>(tick));
        current_state = state_machine_ptr[0].current_state;
        if (entity_hot_paths != nullptr) {
            entity_hot_paths[0].sleep_state = current_state;
        }
        __threadfence();  // <-- ADD THIS
    }
    break;
```

This is cheap (one instruction), forward-compatible with Phase 2 multi-entity scheduling, and prevents a Phase 2 observer from seeing stale state.

### 2.3 Replace Python Direct Ring Writes With a GPU Enqueue Kernel (P0)

**Problem:** [trm_step_fused_bridge.py](knowledge3d/cranium/bridges/trm_step_fused_bridge.py) `enqueue_event` currently does `cuMemcpyHtoD` directly into `ring_buffer[head]` and then bumps `head_ptr` non-atomically. This is safe *only* in Phase 1's single-producer world. The moment Phase 2 adds GPU-side perception producers (collision events, stimulus events, timer events), the Python non-atomic head bump will race the GPU `atomicCAS` push and corrupt the ring.

**Fix — add a new kernel** in [gpu_event_queue.cu](knowledge3d/cranium/cuda/gpu_event_queue.cu):

```cuda
extern "C" __global__ void gpu_event_queue_enqueue_host_batch(
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    const GPUEvent* host_batch,
    uint32_t batch_size,
    uint32_t* push_results
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    const bool pushed = trm_event_queue_push(
        ring_buffer, head_ptr, tail_ptr, host_batch[tid]
    );
    if (push_results != nullptr) {
        push_results[tid] = pushed ? 1u : 0u;
    }
}
```

**Fix — in the bridge**, replace `enqueue_event` with:

1. Copy the event(s) from host to a small scratch device buffer (1..64 events)
2. Launch `gpu_event_queue_enqueue_host_batch` with `grid=(ceil(batch/32), 1, 1)`, `block=(32, 1, 1)`
3. Read back the `push_results` array to detect ring-full failures
4. Delete all `cuMemcpyHtoD` writes to ring slots and all `cuMemcpyHtoD` to `head_ptr`

This keeps Python on the ingestion side of the contract — it hands a batch to the GPU, the GPU decides where it lands.

### 2.4 Bridge Single-Producer Assertion (P0)

While the `enqueue_host_batch` kernel is the long-term answer, also add a bridge-side flag `self._gpu_producers_active: bool = False`. Phase 2 perception will flip this to `True` when it launches collision/stimulus kernels. `enqueue_event` must assert `not self._gpu_producers_active` until the host-batch kernel is wired, so a Phase 2 patch cannot accidentally reintroduce the race.

### 2.5 Phase 1.1 Verification

Before opening the Phase 2 branch:

- `ARC 10/10` must still pass on the fused tick path (rerun under `k3d-cranium` with `CUDA_VISIBLE_DEVICES=0`)
- `Math 20/20` must still pass
- `tests/test_trm_embodied_tick_phase1.py` must pass under a real CUDA context (the Phase 1 report noted it was skipped because no CUDA context — this is **not acceptable** for a merge gate; re-run it in `k3d-cranium`)
- `tests/test_trm_fused_parity.py` must still compare fused tick against `trm_recursive_fused` oracle and match bit-for-bit on the query fast-lane
- Add a new test `tests/test_gpu_event_queue_enqueue_host_batch.py` that stresses the new kernel with 256 concurrent events and asserts all 256 land exactly once

---

## 3. Phase 2 Scope — Perception + Embodiment

Phase 2 turns the clockwork into a living avatar: the TRM must *perceive* the House, *decide* where to look, *navigate* to what matters, and *act* on it. This is the phase where the TRM stops being "a function that answers queries" and starts being "an embodied entity that thinks autonomously."

### 3.1 Paradigm Reminder (Read This Before Editing)

- **TRM IS the Avatar** — not a function Python calls. `trm_step_fused.cu` is one game tick, like an NPC `update()` in a game engine.
- **House = Memory Palace** — the external 3D world the avatar lives in. Rooms = knowledge domains. Already populated (6 rooms, 305KB GLB, as of H15).
- **Galaxy = Internal Brain** — unified VRAM workspace inside the avatar's head. Multi-modal. Read-write.
- **Sovereignty:** no numpy, no cupy, no scipy, no sympy, no Python fallbacks in the hot path. If GPU breaks, fix ON GPU. `We fail and fix.` (Daniel)
- **Target:** `knowledgeverse.py` should keep shrinking. Do not add new Python orchestration for perception. Add CUDA kernels + bridge plumbing only.

### 3.2 EntityHotPath Extension (68 → 96 bytes)

Phase 2 perception needs these fields on every entity hot path:

| Field | Type | Purpose |
|---|---|---|
| `gaze_yaw` | `float` | Current gaze direction (horizontal) |
| `gaze_pitch` | `float` | Current gaze direction (vertical) |
| `gaze_fov` | `float` | Field-of-view cone half-angle (radians) |
| `attention_entity_id` | `uint32_t` | Entity/star currently locked in attention |
| `motor_output[3]` | `float[3]` | Desired linear velocity for the physics SOA |
| `current_goal_star` | `uint32_t` | Galaxy star ID the avatar is navigating toward |
| `_pad2` | `uint32_t` | Align to 16-byte boundary |

**Total new size:** 68 + 24 = 92 bytes → padded to 96 bytes with `_pad2`.

**Files to edit:**

- [knowledge3d/cranium/cuda/entity_hot_path.h](knowledge3d/cranium/cuda/entity_hot_path.h) — extend the C++ struct, update `static_assert(sizeof(EntityHotPath) == 96, ...)`
- [knowledge3d/cranium/bridges/trm_step_fused_bridge.py](knowledge3d/cranium/bridges/trm_step_fused_bridge.py) — extend `_EntityHotPathStruct`, update `assert ctypes.sizeof(_EntityHotPathStruct) == 96`
- Any test/file that asserts `sizeof == 68` — grep and update

**Do not initialize gaze/motor fields to garbage.** In `bind_entity_hot_paths`, zero-initialize them so the very first tick has deterministic gaze = (0, 0, π/4), motor_output = (0,0,0), attention_entity_id = 0, current_goal_star = 0.

### 3.3 Multi-Entity Dispatch (`blockIdx.x` as Entity Index)

Currently [trm_step_fused.cu](knowledge3d/cranium/ptx/trm_step_fused.cu) hardcodes `entity_hot_paths[0]` and `state_machine_ptr[0]`. Phase 2 needs N entities sharing the fused tick.

**Refactor:**

```cuda
extern "C" __global__ void trm_step_fused(...) {
    const int entity_idx = blockIdx.x;
    if (entity_idx >= static_cast<int>(entity_count)) return;

    EntityHotPath* my_entity = &entity_hot_paths[entity_idx];
    TRMStateMachine* my_sm = &state_machine_ptr[entity_idx];

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    __shared__ uint8_t current_state;
    // ... existing shared steps_taken/drift_value/converged ...

    if (tid == 0 && my_sm != nullptr) {
        trm_state_machine_step_device(
            my_entity, my_sm, ring_buffer, head_ptr, tail_ptr,
            /* entity_count */ 1u, delta_time, static_cast<uint64_t>(tick)
        );
        current_state = my_sm->current_state;
    }
    __syncthreads();

    // switch (current_state) uses my_entity, not entity_hot_paths[0]
}
```

**Launch pattern from the bridge:**

```python
grid = (max(self.entity_count, 1), 1, 1)
block = (GPU_TASK_TRM_THREADS, 1, 1)
```

The query fast-lane (`run_query_tick`) must still run with `entity_count=1` to preserve the existing contract. Do not break ARC/Math/GSM8K/LHE/MMLU by accident.

**Important nuance for the ring buffer:** with N blocks running `trm_state_machine_step_device` in parallel, each under its own `tid==0`, all N blocks will race on `head_ptr/tail_ptr` via `atomicCAS`. The atomicCAS is correct — no extra lock is needed — but this means the ring buffer becomes MPMC (multi-producer, multi-consumer) for the first time. Run `cuda-memcheck --tool racecheck` on the Phase 2 smoke test before merging.

### 3.4 Event Batch Drain (Up to 8 Events Per Tick)

[trm_game_loop.cuh](knowledge3d/cranium/cuda/trm_game_loop.cuh) `trm_state_machine_step_device` currently pops one event per tick. Phase 2 perception will burst events (stimulus + collision + timer can all land in the same tick). Single-pop starves.

**Fix:** In `trm_state_machine_step_device`, wrap the pop-then-apply in a bounded drain loop:

```cuda
#define TRM_MAX_EVENTS_PER_TICK 8u

__device__ void trm_state_machine_step_device(...) {
    // ... idle accumulation, sleep transition, etc. ...

    GPUEvent evt;
    uint32_t drained = 0u;
    while (drained < TRM_MAX_EVENTS_PER_TICK &&
           trm_event_queue_pop(ring_buffer, head_ptr, tail_ptr, &evt)) {
        if (evt.entity_id != sm[0].owner_entity_id) {
            // Event targets a different entity — requeue
            // OR filter at dequeue time using a per-entity queue (Phase 2.5)
            // For Phase 2, drop-and-log is acceptable; log via deferred_event_mask
            sm[0].deferred_event_mask |= TRM_DEFERRED_FOREIGN_EVENT;
            drained += 1u;
            continue;
        }
        trm_apply_event(sm[0], evt, tick);
        drained += 1u;
    }
}
```

**Per-entity queue is NOT in Phase 2 scope.** For Phase 2, one shared ring with `owner_entity_id` filtering is acceptable. Phase 3 can split to N rings if contention becomes measurable.

Also: add `owner_entity_id` as an implicit field — we can reuse `sm[0].reserved` (currently unused in `TRMStateMachine`) for this. No struct resize needed.

### 3.5 Transition Table Extensions

[trm_game_loop.cuh](knowledge3d/cranium/cuda/trm_game_loop.cuh) currently has 9 transitions (SLEEP/IDLE/REASONING). Add Phase 2 entries:

```cpp
// Phase 2: perception → navigation → acting chain
{TRM_STATE_IDLE,        TRM_EVENT_PERCEPTION_STIMULUS, TRM_STATE_PERCEIVING,  TRM_TF_RESET_IDLE},
{TRM_STATE_PERCEIVING,  TRM_EVENT_INTERNAL,            TRM_STATE_NAVIGATING,  TRM_TF_NONE},
{TRM_STATE_NAVIGATING,  TRM_EVENT_TIMER,               TRM_STATE_ACTING,      TRM_TF_NONE},
{TRM_STATE_ACTING,      TRM_EVENT_INTERNAL,            TRM_STATE_IDLE,        TRM_TF_NONE},
{TRM_STATE_PERCEIVING,  TRM_EVENT_COLLISION,           TRM_STATE_ACTING,      TRM_TF_NONE},
// WAKEUP from any sleep-like state:
{TRM_STATE_SLEEP,       TRM_EVENT_WAKEUP,              TRM_STATE_IDLE,        TRM_TF_RESET_IDLE},
```

The exact `TRM_TF_*` flags should match the existing convention. Reserve the mechanic for `HANDLING_QUERY` — that stays as an interrupt/pop, not a table transition.

### 3.6 PERCEIVING Phase — Compose The Head Pipeline

The `TRM_STATE_PERCEIVING` case in [trm_step_fused.cu](knowledge3d/cranium/ptx/trm_step_fused.cu) currently reads:

```cuda
case TRM_STATE_PERCEIVING:
    // TODO Phase 2: wire House -> Galaxy perception kernels here.
    break;
```

**This is where the composed head pipeline wires into the game loop.** The pipeline already exists on GPU — do not reimplement it. Call the existing device functions:

```
Morton Octree range query → Frustum Cull (gaze-based) → Dynamic LOD → saliency → Galaxy bind
```

Per the kernel inventory, these are already built as `__device__` functions. You must add a `trm_perceiving_phase` wrapper that:

1. **Morton Octree range query** — centered on `my_entity->cranial_origin`, radius driven by `my_entity->last_player_dist` or a constant
2. **Frustum Cull** — per-entity frustum built from `my_entity->gaze_yaw/gaze_pitch/gaze_fov` (NOT the camera frustum — this is the *avatar's own* field of view, the "superdotados" internal view)
3. **Dynamic LOD** — gate coarse vs fine star detail by distance from entity, not from the human camera
4. **Saliency scoring** — a device-local reduction: score each surviving star by (novelty × proximity × relevance-to-current-goal). Pick top-K (K=4 for Phase 2).
5. **Galaxy bind** — write the top-K star IDs into a scratch region in the entity hot path (we can reuse `blackboard_star_id` for K=1, or extend the hot path with a small `salience[4]` array in Phase 2.5)
6. **Emit internal event** — push a `TRM_EVENT_INTERNAL` event into the ring buffer so the state machine transitions `PERCEIVING → NAVIGATING` on the next tick

**Use the existing device functions.** Grep for `morton_octree_range_query_device`, `frustum_cull_device`, `dynamic_lod_device`, and the saliency kernels. If any of these are *kernel-only* (not `__device__`) you must extract the `__device__` variant — do NOT launch nested kernels from `trm_step_fused`. The fused tick is one kernel, period.

**Sovereignty gate:** this phase must not touch Python. The perception pipeline composes entirely from existing PTX.

### 3.7 NAVIGATING Phase — LED-A* Step

The `TRM_STATE_NAVIGATING` case must call the LED-A* pathfinding device function **one step at a time** (not a full path compute). Each tick advances the A* frontier by a bounded amount (e.g., 16 expansions). This is what makes it a game loop — navigation is spread across ticks, not blocking.

Write `trm_navigating_phase` wrapper:

1. Read `my_entity->current_goal_star` (set by PERCEIVING)
2. Call `led_astar_step_device(source=my_entity->star_table_idx, goal=my_entity->current_goal_star, frontier_scratch=workspace, max_expansions=16)`
3. If the goal was reached, push `TRM_EVENT_TIMER` (for now — cleaner to add `TRM_EVENT_GOAL_REACHED` in Phase 3)
4. Update `my_entity->motor_output[3]` with the next movement vector from the frontier

**Critical:** LED-A* already exists in the kernel inventory as part of the composed head pipeline. Do not reimplement it. Extract the device-callable step function.

### 3.8 ACTING Phase — RPN Behavior Interpreter

The `TRM_STATE_ACTING` case runs whatever RPN program is stored at `my_entity->meta_rule_addr` (existing field). This is the entity's "behavior script."

Write `trm_acting_phase` wrapper:

1. Resolve RPN bytecode from the galaxy table at `my_entity->meta_rule_addr`
2. Call the existing sovereign RPN interpreter device function on it
3. The interpreter can touch `my_entity->motor_output`, push events, or modify `my_entity->awareness`
4. After execution, emit `TRM_EVENT_INTERNAL` to transition `ACTING → IDLE`

**Sovereignty gate:** RPN interpreter exists. Grep the kernel inventory for `rpn_interpreter_device` or similar. Do NOT reimplement bytecode dispatch.

### 3.9 Physics Phase — Promote From No-Op

Currently [trm_step_fused.cu:9](knowledge3d/cranium/ptx/trm_step_fused.cu#L9) has `trm_phase2_physics_noop`. Phase 2 must wire the real rigid-body integration:

1. Read `my_entity->motor_output` (set by NAVIGATING or ACTING)
2. Call the existing sovereign rigid-body integrator device function on `physics_soa_ptr` / `contact_soa_ptr`
3. If a contact is detected, push `TRM_EVENT_COLLISION` into the ring buffer

**Ask in MCP first:** use `qdrant-find` with query `"sovereign rigid body physics integrator"` to locate the existing device function. If nothing exists, use `plan_task` to scope a minimal integrator (semi-implicit Euler, one iteration for Phase 2 — we'll harden in Phase 3).

### 3.10 Multi-Entity Scheduler In The Bridge

[trm_step_fused_bridge.py](knowledge3d/cranium/bridges/trm_step_fused_bridge.py) `launch_tick` currently assumes `entity_count=1`. Phase 2 must:

1. Allocate N `TRMStateMachine` slots + N `EntityHotPath` slots on `bind_state_machines(count)` / `bind_entity_hot_paths(count)`
2. Launch `trm_step_fused` with `grid=(N, 1, 1)` for background ticks
3. Still launch with `grid=(1, 1, 1)` for the `run_query_tick` fast-lane (single-entity query)
4. Keep both entry points on the same compiled kernel

This is the wedge that lets Phase 3 introduce multi-entity NPCs without another refactor.

### 3.11 Daemon Tick Loop (Deferred — Phase 2.5 or Phase 3)

The continuous daemon (`while True: launch_tick()` on a fixed 50Hz clock) is **not in Phase 2 scope**. Phase 2 lands the kernels; Phase 2.5 wraps them in a daemon. Write the kernels so a daemon wrapping them adds zero new primitives.

---

## 4. Sovereignty Gate — Non-Negotiable

Before Phase 2 merges:

- [ ] `grep -r "import numpy\|import cupy\|import scipy\|import sympy" knowledge3d/cranium/` returns zero lines inside any `*_bridge.py` or `*.cu` file touched by this phase
- [ ] No Python fallback paths. If a kernel fails to compile, **fix the kernel**, do not guard with `try/except`.
- [ ] `knowledgeverse.py` line count does not grow beyond its current size. Ideally shrinks.
- [ ] `cuda-memcheck --tool racecheck` run on `tests/test_trm_embodied_tick_phase2_perception.py` shows zero races
- [ ] ARC 10/10, Math 20/20, GSM8K ≥2/10, LHE ≥6/10, MMLU ≥12/50 all hold on the fused tick path

---

## 5. Test Plan For Phase 2

Add these test files:

- `tests/test_trm_entity_hot_path_96_bytes.py` — struct size + field alignment checks
- `tests/test_trm_multi_entity_dispatch.py` — launch N=4 entities, verify each runs its own state machine
- `tests/test_trm_event_batch_drain.py` — enqueue 16 events in one tick, verify 8 are drained (per `TRM_MAX_EVENTS_PER_TICK`), 8 remain for next tick
- `tests/test_trm_perceiving_phase.py` — smoke test that PERCEIVING transitions into NAVIGATING and writes a `current_goal_star`
- `tests/test_trm_navigating_phase.py` — smoke test that NAVIGATING advances LED-A* frontier and updates `motor_output`
- `tests/test_trm_acting_phase.py` — smoke test that ACTING runs a tiny RPN program from `meta_rule_addr`
- `tests/test_trm_physics_collision_event.py` — smoke test that a physics contact pushes `TRM_EVENT_COLLISION` into the ring

Update:

- `tests/test_trm_embodied_tick_phase1.py` — migrate to 96-byte hot path, rerun
- `tests/test_trm_fused_parity.py` — parity against `trm_recursive_fused` must still hold on the query fast-lane
- `tests/test_sovereign_entity_surface.py` — surface checks for new fields

**All tests run under `conda activate k3d-cranium` with `CUDA_VISIBLE_DEVICES=0`. No CPU-only skips.**

---

## 6. Workflow Reminder For Codex

1. **Branch:** `codex/embodied-tick-phase1-1-hotfix-2026-04-10` for Section 2, then `codex/embodied-tick-phase2-perception-2026-04-10` for Section 3
2. **Do NOT mix** Phase 1.1 hotfixes with Phase 2 perception in the same branch. Land the hotfix first, verify benchmarks, then branch Phase 2 off the hotfixed main.
3. **Use MCP for every architectural question** — `plan_task` for decomposition, `qdrant-find` for "where does kernel X live", `ask_coder` for CUDA drafts, `kimi_swarm` for review.
4. **Do NOT re-read the full vocabulary docs** unless a specific term is ambiguous and MCP doesn't have it. Token budget matters.
5. **Run benchmarks in `k3d-cranium` env**. The sandbox has no GPU — CUDA runs must happen outside the sandbox.
6. **Report back** with a Phase 1.1 completion report in TEMP/ before starting Phase 2. Claude will review and either green-light Phase 2 or request more hotfixes.

---

## 7. Success Criteria

**Phase 1.1 (hotfix) is done when:**

- GPUEvent allocation is verifiably 16-byte aligned
- `__threadfence()` is in place after state pop
- Python enqueue goes through `gpu_event_queue_enqueue_host_batch`
- `cuda-memcheck --tool racecheck` is clean on the enqueue stress test
- ARC/Math/GSM8K/LHE/MMLU all hold
- `tests/test_trm_embodied_tick_phase1.py` actually runs under CUDA (no skip)

**Phase 2 is done when:**

- `EntityHotPath` is 96 bytes on both sides of the bridge
- `trm_step_fused` uses `blockIdx.x` for entity indexing
- `TRM_STATE_PERCEIVING/NAVIGATING/ACTING` all call real composed-head device functions (no `// TODO` in the switch)
- Physics phase is no longer a no-op
- Transition table has Phase 2 entries
- Event batch drain caps at `TRM_MAX_EVENTS_PER_TICK=8`
- All Phase 2 tests pass under CUDA
- Sovereignty gate in §4 holds
- Benchmarks in §4 hold

---

## 8. One Final Reminder

TRM IS the avatar. Phase 2 is where it starts *looking around*. Every kernel you wire in this phase is a new sense for the entity — perception = eyes, navigation = legs, acting = hands, physics = the body feeling the world.

Python is boot + I/O. Nothing more. Do not add orchestration in Python. Do not add fallbacks. Do not add `try/except` around CUDA calls that "might fail" — if they fail, the kernel is wrong, fix the kernel.

We fail and fix. — Daniel
