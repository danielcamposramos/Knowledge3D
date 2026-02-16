# Codex Workflow Review + Game-Mode Cutover (No Benchmark Run)

Date: 2026-02-12  
Author: Codex

## Scope
- Per request: no benchmark execution before full workflow review.
- Focus: lifecycle/orchestration drift (script one-shot), GPU sovereignty path, fallback risk.

## End-to-End Workflow Review
1. Entry points currently used in practice are still script-centric:
   - `scripts/run_all_benchmarks.py`
   - `benchmarks/*.py`
2. Benchmark scripts instantiate `Knowledgeverse` and `TRMNavigator`, solve, then exit.
3. This keeps the system in library mode (load/solve/unload), not persistent game mode.
4. Query and solving are routed through `TRMNavigator` and `GalaxyManager`; this is the right logical path, but lifecycle is still one-shot.

## Hot-Path Findings (Static)
- `TRMNavigator._solve_math` is now Galaxy-first and no longer uses regex/eval fallback path.
- Major lifecycle gap remains: no persistent daemon command loop as primary runtime.
- Tiered execution still contains fallback-oriented behavior in bridge layers; this must remain under strict sovereignty checks as migration continues.

## Implemented in this pass
1. Persistent daemon bridge (functional now):
   - `knowledge3d/daemon/main.py`
   - `scripts/k3d_daemon.py`
   - Loads one `Knowledgeverse` once, keeps process alive, serves JSON commands (`STATUS`, `QUERY`, `ROUTE`, `SOLVE_MATH`, `CHAT`, `SHUTDOWN`).
   - Strict by default: `K3D_REQUIRE_PTX_QUERY=true`.
2. Native daemon lifecycle scaffold (C++):
   - `knowledge3d/daemon/main.cpp`
   - Implements persistent command loop for game-lifecycle cutover (`PING/STATUS/SHUTDOWN` + explicit not-implemented dispatch).

## Why this is aligned
- Moves orchestration to a persistent process model immediately.
- Stops one-shot script lifecycle as the only runtime pattern.
- Keeps strict sovereignty default at daemon startup.
- Establishes native daemon anchor for Phase-2 C++ dispatch migration.

## Next Required Cutover Steps
1. Wire native daemon dispatch to TRM/Galaxy/PTX runtime surface (replace placeholder error).
2. Route benchmark runners through daemon command protocol (no direct benchmark-side orchestration).
3. Add strict sovereignty test that asserts no forbidden fallback patterns in declared hot-path files.
4. Add GPU-call accounting in daemon responses (`gpu_calls`, `kernel_launches`) to make “GPU-used vs not-used” explicit per command.

## Notes
- No benchmark run was executed in this pass.
- This pass is lifecycle/cutover foundation only.
