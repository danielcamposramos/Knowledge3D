# CODEX Embodied Game Loop Phase 3 Living Tick Report - 2026-04-11

## Scope

This report covers the Phase 3 living tick slice implemented directly on `main`.

Goal:
- make the avatar tick continuously between queries
- keep Python as clock and I/O only
- route both background ticks and query ticks through `TRMStepFusedBridge`

## Landed

### 1. Bridge-owned tick loop

Updated:
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`

Landed behavior:
- `start_tick_loop()` starts a daemon thread named `k3d-trm-fused-tick`
- `stop_tick_loop()` stops and joins the thread
- the loop runs a fixed 50 Hz / `delta_time=0.02` cadence by default
- the thread calls exactly one fused GPU tick per iteration through `launch_tick(delta_time=0.02)`
- `tick_count`, `tick_loop_last_error`, and `tick_loop_status()` expose clock state for tests and runtime diagnostics
- `cleanup()` stops the tick loop before freeing VRAM buffers

Sovereignty note:
- the thread does not inspect tasks, states, events, or action semantics
- all lifecycle and phase decisions remain in `trm_step_fused.cu`

### 2. Query preemption lock

Updated:
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`

Landed behavior:
- `launch_tick()` and `run_query_tick()` share one `RLock`
- `run_query_tick()` holds the lock across optional runtime reset, query event enqueue, and fused kernel launch
- background ticks yield while query ticks run
- default zeroed VRAM q/y/z/W/workspace buffers allow clock ticks to run without host tensor assembly

### 3. TRMGameLoop routes to the fused bridge

Updated:
- `knowledge3d/knowledgeverse/trm_game_loop.py`
- `knowledge3d/knowledgeverse/knowledgeverse.py`

Landed behavior:
- `TRMGameLoop.tick()` no longer calls `_dispatch_sovereign_task`
- pending input presence routes to `bridge.run_query_tick(delta_time=0.02)`
- idle/background ticks route to `bridge.launch_tick(delta_time=0.02)`
- ActionBuffer words are read after ticks for consumer/debug visibility
- `TRMGameLoop.start()` / `stop()` delegate to `bridge.start_tick_loop()` / `stop_tick_loop()`
- `Knowledgeverse` now starts the TRM game loop after `_initialize_trm_launcher()`, so the bridge exists before live ticking begins

Compatibility note:
- the existing ring-buffer packet envelope remains for API compatibility
- the tick path no longer decodes semantic task payloads or dispatches Python task handlers

## Verification

Passed:
- `K3D_PYTEST_PROBE_CUDA=1 pytest -q tests/test_trm_tick_loop.py`
- `K3D_PYTEST_PROBE_CUDA=1 pytest -q tests/test_trm_tick_loop.py tests/test_trm_game_loop.py tests/test_trm_action_buffer_emission.py tests/test_trm_embodied_tick_phase1.py tests/test_trm_fused_parity.py`

Results:
- tick-loop suite: `3 passed`
- focused Phase 3 regression batch: `26 passed, 2 warnings`

Warnings:
- `tests/test_trm_fused_parity.py` emits existing Tesla resonance warnings for `n_steps=1` and `n_steps=3`

Specialist review:
- `ollama-specialists.ask_coder` reviewed the bridge/game-loop concurrency surface
- actionable check was clean tick-thread exit on launch error
- current implementation sets the stop event and breaks the loop on exceptions

## Remaining Work

- wire viewer/world consumers to read ActionBuffer slots zero-copy
- replace compatibility JSON queue envelopes with a GPU event ingestion surface when the viewer/input path is ready
- extract deeper composed-head device helpers out of Python orchestration
