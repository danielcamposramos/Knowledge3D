# CODEX Embodied Game Loop Phase 2 Implementation Report — 2026-04-10

## Scope

This report covers the live Phase 2 embodiment slice and the Phase 2.5 ActionBuffer emission slice implemented directly on `main`.

Authoritative spec:
- `TEMP/CLAUDE_EMBODIED_GAME_LOOP_PHASE1_1_HOTFIX_AND_PHASE2_SPEC_2026-04-10.md`

Direction update from Daniel:
- benchmark score pins are health checks, not merge gates, during this embodied rebuild

## Landed

### 1. `EntityHotPath` widened from 68 to 96 bytes

Updated:
- `knowledge3d/cranium/kernels/entity_hot_path.h`
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
- `knowledge3d/cranium/bridges/sovereign_bridges.py`
- `knowledge3d/cranium/sovereign_entity_bootstrap.py`

Landed fields:
- `gaze_yaw`
- `gaze_pitch`
- `gaze_fov`
- `attention_entity_id`
- `motor_output[3]`
- `current_goal_star`

Notes:
- The obsolete 4-byte `_pad` was removed; that is what keeps the widened layout at exactly 96 bytes.
- New fields are deterministically initialized in both bridges and the bootstrap hot-path projection.

### 2. Multi-entity fused tick and lifecycle dispatch

Updated:
- `knowledge3d/cranium/cuda/trm_game_loop.cuh`
- `knowledge3d/cranium/ptx/trm_state_machine.cu`
- `knowledge3d/cranium/ptx/trm_step_fused.cu`
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`

Landed behavior:
- `blockIdx.x` now selects the active entity/state-machine slot
- `TRMStateMachine.reserved` is now used as `owner_entity_id`
- `trm_step_fused` writes per-entity `steps_out` / `drift_out`
- `launch_tick()` now launches `(entity_count, 1, 1)` by default
- `run_query_tick()` preserves the single-entity fast lane through explicit grid/entity-count overrides

Important current truth:
- The recursive TRM latent math still belongs to the avatar lane (`entity_idx == 0`) until per-entity TRM latents are introduced.
- This is deliberate to avoid corrupting shared `q/y/z/y_new/z_new` buffers in multi-block background ticks.

### 3. Event batch drain and Phase 2 transition extensions

Updated:
- `knowledge3d/cranium/cuda/trm_game_loop.cuh`

Landed behavior:
- bounded `TRM_MAX_EVENTS_PER_TICK = 8`
- shared ring owner filtering via `owner_entity_id`
- foreign-event drop flag: `TRM_DEFERRED_FOREIGN_EVENT`
- Phase 2 transition additions:
  - `PERCEIVING + INTERNAL -> NAVIGATING`
  - `NAVIGATING + TIMER -> ACTING`
  - `ACTING + INTERNAL -> IDLE`
  - `PERCEIVING + COLLISION -> ACTING`

Design note:
- This remains one shared ring buffer.
- Foreign events are dropped and flagged, exactly as the spec allowed for Phase 2.
- That means deterministic per-entity event delivery is not solved yet; per-entity rings remain future work.

### 4. Live `PERCEIVING`, `NAVIGATING`, `ACTING`, and GPU physics

Updated:
- `knowledge3d/cranium/ptx/trm_step_fused.cu`

Landed behavior:
- `PERCEIVING`
  - gaze-cone scan over bound entities on GPU
  - Morton-code novelty bias
  - target selection into `attention_entity_id` / `current_goal_star`
  - emits `TRM_EVENT_INTERNAL` for navigation handoff
- `NAVIGATING`
  - GPU steering toward the selected target
  - writes `motor_output`
  - faces the target
  - emits `TRM_EVENT_TIMER` when close enough to act
- `ACTING`
  - materializes a behavior-side state update on GPU
  - amplifies motor intent via `meta_rule_addr`
  - writes goal state to blackboard
  - emits `TRM_EVENT_INTERNAL` to return to idle
- `PHYSICS`
  - consumes `motor_output` on GPU
  - updates rigid-body SOA position/velocity when a body is bound
  - mirrors the body pose back into `house_x/y/z`
  - scans the bound contact manifold and emits `TRM_EVENT_COLLISION`

### 5. Phase 2.5 ActionBuffer emission

Updated:
- `knowledge3d/cranium/ptx/trm_step_fused.cu`
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
- `tests/test_trm_action_buffer_emission.py`

Landed behavior:
- `trm_step_fused` now appends nullable `action_buffer_out` after `drift_out`
- each entity block writes one 288-byte / 72-word ActionBuffer slot after physics
- emission is GPU-only: block lanes zero the slot, thread 0 packs fields, and `__threadfence_system()` publishes the frame
- bridge owns a VRAM-resident `entity_count * 288` output buffer and passes it into the single fused tick
- bridge exposes `action_buffer_ptr` for zero-copy consumers and `read_action_buffers_words()` for raw debug/test readback
- action type mapping:
  - `NAVIGATING -> NAV_MOVE`
  - `PERCEIVING -> NAV_LOOK`
  - `SLEEP` / `IDLE` / `REASONING` / `HANDLING_QUERY -> NO_ACTION`
  - `ACTING -> meta_rule_addr & 0x3` dispatch for `DIALOGUE`, `WRITE_MEM`, `UPDATE_TABLET`, or `NAV_MOVE`
- query handling preserves a separate emission state, so `HANDLING_QUERY` emits `NO_ACTION` even when the lifecycle pops back to an interrupted navigation/acting state before the end of the tick

Layout note:
- The active wire contract is the 72-word layout shared by `knowledge3d/cranium/actions/action_types.py` and `knowledge3d/cranium/ptx/decode_actions.ptx`.
- The older prose-only `DUAL_CLIENT_CONTRACT_SPECIFICATION.md` §4.2 layout was not used for offsets.

## Deliberate deviations from the ideal spec

These are not hidden; they reflect the actual repo surfaces available today.

### 1. Frustum / dynamic LOD / device-callable RPN interpreter are not yet extracted as reusable helpers

Observed repo reality:
- frustum is exposed as a PTX kernel + Python wrapper, not a reusable device helper
- dynamic LOD is still CuPy/Python-side and cannot be used in the sovereign tick path
- the sovereign RPN runtime exists as kernels and a large modular kernel surface, not as a small device-callable helper ready to embed in `trm_step_fused`

What Phase 2 does instead:
- uses GPU-native gaze/proximity scanning and Morton novelty for perception
- uses GPU steering and goal-distance thresholds for navigation
- uses device-local behavior-state composition for acting
- keeps all of that inside the fused kernel with zero Python orchestration

This keeps sovereignty intact while avoiding fake host orchestration.

### 2. Multi-entity event delivery is still lossy by design

Current Phase 2 shared ring semantics:
- any entity block can pop any event
- foreign events are dropped and flagged

Why accepted:
- this is the Phase 2 compromise explicitly allowed by the spec
- the tests therefore use one active avatar block when they need deterministic target/event handoff, while still binding multiple entities for perception lookup

## Verification

### Compile / syntax

Passed:
- `python3 -m py_compile knowledge3d/cranium/bridges/trm_step_fused_bridge.py knowledge3d/cranium/bridges/sovereign_bridges.py knowledge3d/cranium/sovereign_entity_bootstrap.py`
- `python3 -m py_compile tests/test_trm_entity_hot_path_96_bytes.py tests/test_trm_multi_entity_dispatch.py tests/test_trm_event_batch_drain.py tests/test_trm_perceiving_phase.py tests/test_trm_navigating_phase.py tests/test_trm_acting_phase.py tests/test_trm_physics_collision_event.py tests/test_sovereign_physics_surface.py tests/test_trm_embodied_tick_phase1.py tests/test_trm_fused_parity.py`
- direct PTX compile:
  - `knowledge3d/cranium/ptx/trm_step_fused.cu`
  - `knowledge3d/cranium/ptx/trm_state_machine.cu`
  - `knowledge3d/cranium/cuda/gpu_event_queue.cu`

### Real CUDA targeted batch

Passed under `K3D_PYTEST_PROBE_CUDA=1`:
- `tests/test_trm_action_buffer_emission.py`
- `tests/test_trm_entity_hot_path_96_bytes.py`
- `tests/test_trm_multi_entity_dispatch.py`
- `tests/test_trm_event_batch_drain.py`
- `tests/test_trm_perceiving_phase.py`
- `tests/test_trm_navigating_phase.py`
- `tests/test_trm_acting_phase.py`
- `tests/test_trm_physics_collision_event.py`
- `tests/test_trm_embodied_tick_phase1.py`
- `tests/test_trm_fused_parity.py`
- `tests/test_sovereign_physics_surface.py::test_trm_step_fused_source_has_explicit_physics_phase_slot`

Results:
- Phase 2.5 ActionBuffer suite: `10 passed`
- Phase 1/2 embodied regression batch: `13 passed, 2 warnings`

### Sovereignty grep

Passed:
- `rg -n "import numpy|import cupy|import scipy|import sympy" knowledge3d/cranium/bridges/trm_step_fused_bridge.py knowledge3d/cranium/bridges/sovereign_bridges.py knowledge3d/cranium/ptx/trm_step_fused.cu knowledge3d/cranium/cuda/trm_game_loop.cuh knowledge3d/cranium/kernels/entity_hot_path.h`
- no matches
- Phase 2.5 touched cranium files are also clean:
  - `rg -n "import (numpy|cupy|scipy|sympy)|from (numpy|cupy|scipy|sympy)" knowledge3d/cranium/ptx/trm_step_fused.cu knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
  - no matches

### Specialist review

Kimi swarm review:
- first full-context think-mode call timed out at 120s
- compact retry completed
- result: no blockers
- residual notes were documentation/consumer semantics only: awareness/goal/state values intentionally appear in multiple schema fields, and `__threadfence_system()` is conservative for host-visible frames

## Files changed in this Phase 2 slice

- `knowledge3d/cranium/kernels/entity_hot_path.h`
- `knowledge3d/cranium/cuda/trm_game_loop.cuh`
- `knowledge3d/cranium/ptx/trm_state_machine.cu`
- `knowledge3d/cranium/ptx/trm_step_fused.cu`
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
- `knowledge3d/cranium/bridges/sovereign_bridges.py`
- `knowledge3d/cranium/sovereign_entity_bootstrap.py`
- `tests/test_trm_action_buffer_emission.py`
- `tests/test_sovereign_physics_surface.py`
- `tests/test_trm_entity_hot_path_96_bytes.py`
- `tests/test_trm_multi_entity_dispatch.py`
- `tests/test_trm_event_batch_drain.py`
- `tests/test_trm_perceiving_phase.py`
- `tests/test_trm_navigating_phase.py`
- `tests/test_trm_acting_phase.py`
- `tests/test_trm_physics_collision_event.py`

## Remaining work

### Phase 2.5 / 3 candidates

- extract reusable device-callable frustum / LOD helpers out of PTX-wrapper surfaces
- extract or factor a small device-callable sovereign RPN behavior helper for `ACTING`
- give each embodied entity its own TRM latent slab instead of pinning latent math to entity 0
- replace shared lossy ring semantics with deterministic per-entity queues or requeue-on-foreign logic
- wire viewer/world consumers to read the emitted ActionBuffer slots zero-copy
- land the continuous daemon loop

## Outcome

Phase 2 is no longer stubbed.

The embodied fused tick now has:
- a widened entity hot path
- per-entity lifecycle dispatch
- bounded event draining
- live perception/targeting
- live navigation steering
- live acting-side state materialization
- live GPU physics/collision emission
- sovereign 288-byte ActionBuffer emission per entity

The remaining work is refinement, not absence.
