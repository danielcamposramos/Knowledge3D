# CODEX Embodied Game Loop Phase 1 Report — 2026-04-10

## Scope

Phase 1 landed the clockwork slice for embodied TRM ticking:

1. GPU event ring buffer in VRAM
2. VRAM lifecycle state machine (`TRMStateMachine`)
3. Fixed-timestep `delta_time` + `tick` plumbing on `trm_step_fused`
4. Single-entry runtime wiring through `TRMStepFusedBridge`

This work keeps the query fast-lane on the same fused tick entrypoint used for embodiment. `knowledgeverse.py` no longer launches `kernel_recursive_fused` directly for `_run_single_trm_tick`.

## What Was Built

### 1. GPU Event Ring Buffer

- Added shared game-loop definitions in `knowledge3d/cranium/cuda/trm_game_loop.cuh`
- Added `GPUEvent` (16 bytes) and the 7 requested event types
- Added lock-free queue helpers using `atomicCAS` head/tail management
- Added publication safety on the queue slots with `TRM_EVENT_NONE` sentinel clearing/publishing
- Added `knowledge3d/cranium/cuda/gpu_event_queue.cu`
  - `gpu_event_queue_reset`
  - `gpu_event_queue_enqueue_stress`
  - `gpu_event_queue_dequeue_all`

### 2. Lifecycle State Machine

- Added packed 32-byte `TRMStateMachine` in `trm_game_loop.cuh`
- Added 7 requested states
- Added constant-memory transition table
- Added device helpers for:
  - state push/pop
  - transition lookup/apply
  - query preemption
  - idle accumulation
  - `IDLE -> SLEEP` after 30 seconds
- Added kernel wrapper `knowledge3d/cranium/ptx/trm_state_machine.cu`

### 3. Refactored `trm_step_fused`

- Extended kernel ABI with:
  - `ring_buffer_ptr`
  - `head_ptr`
  - `tail_ptr`
  - `state_machine_ptr`
  - `entity_hot_path_ptr`
  - `entity_count`
  - `delta_time`
  - `tick`
- Replaced flat execution with lifecycle-gated dispatch
- Live Phase 1 paths:
  - `SLEEP`
  - `IDLE`
  - `REASONING`
  - `HANDLING_QUERY`
- `HANDLING_QUERY` now:
  - enters through the GPU event/state machine path
  - runs the existing recursive TRM math as the fast-lane
  - pops back to the interrupted state in the same fused tick
- `PERCEIVING`, `NAVIGATING`, `ACTING` are explicit Phase 2 stubs
- Removed the old void-cast style from the hot path and left an explicit GPU-side Phase 2 physics no-op slot

### 4. Unified Runtime Bridge

- Added `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
- Bridge responsibilities:
  - compile/load PTX modules
  - allocate/bind entity hot paths
  - allocate/bind lifecycle state machines
  - allocate/manage event ring buffers
  - expose one fused `launch_tick()` / `run_query_tick()` surface
- `TRMLauncher` fused path now delegates to the bridge instead of treating `trm_recursive_fused` as the runtime entrypoint
- `Knowledgeverse._run_single_trm_tick()` now delegates through `TRMLauncher.run_query_tick()`

## Tests and Verification

### Added / Updated

- Added `tests/test_trm_embodied_tick_phase1.py`
  - queue stress path
  - scripted lifecycle sequence
  - live Phase 1 state routing
- Updated `tests/test_trm_fused_parity.py`
  - runtime fused tick is compared against `trm_recursive_fused` as oracle
- Updated source-surface tests for the new fused tick contract

### Executed In This Shell

- `python3 -m py_compile ...` on the edited Python files: passed
- `python3 -m pytest -q tests/test_sovereign_entity_surface.py::test_entity_behavior_source_and_fused_step_slot_are_present tests/test_sovereign_physics_surface.py::test_trm_step_fused_source_has_explicit_physics_phase_slot`: passed
- Direct bridge import + PTX compilation smoke test: passed

### Not Fully Verified Here

- `tests/test_trm_embodied_tick_phase1.py` skipped in this shell because pytest could not acquire a CUDA context
- Full ARC / Math benchmark reruns were not completed in this shell
- Long-form `knowledgeverse` regression batch was still in progress at report time

## Benchmark Deltas

- ARC 10/10: not re-run in this shell
- Math 20/20: not re-run in this shell
- Query fast-lane contract: preserved at the API level (`query_embedding_512`, `y_new_vector_512`, `trm_latency_us`, `trm_recursion_steps`, `trm_drift`)
- `knowledgeverse.py` local diff is net-negative in this work slice (41 deletions, 26 insertions)

## What Was Skipped / Deferred To Phase 2

- Real perception kernels in the fused tick
- Real navigation kernels in the fused tick
- Real acting/materialization kernels in the fused tick
- Physics integration beyond the explicit GPU-side no-op slot
- Continuous daemon ownership of the tick loop
- Multi-entity queue consumption beyond the current single-entity Phase 1 path

## Phase 2 Ready Surface

Phase 2 can now build on:

- GPU-native event ingress
- GPU-native lifecycle gating
- Single fused tick entrypoint
- Query preemption/resume path
- Fixed timestep + tick plumbing already carried end-to-end

## Review Request

Before starting Phase 2 (Perception + Embodiment), Claude should review:

- `knowledge3d/cranium/cuda/trm_game_loop.cuh`
- `knowledge3d/cranium/cuda/gpu_event_queue.cu`
- `knowledge3d/cranium/ptx/trm_state_machine.cu`
- `knowledge3d/cranium/ptx/trm_step_fused.cu`
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
- `knowledge3d/cranium/sovereign/trm_launcher.py`
- `knowledge3d/knowledgeverse/knowledgeverse.py`
