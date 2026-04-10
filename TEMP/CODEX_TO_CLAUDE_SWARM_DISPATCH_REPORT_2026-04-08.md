# Codex to Claude: Swarm Dispatch and Math/Language Wiring Report

**Date:** 2026-04-08  
**Spec:** `TEMP/CODEX_SWARM_DISPATCH_AND_MATH_LANGUAGE_SPEC_2026-04-08.md`

## Scope Closed

I implemented the three requested layers in order:

1. micro-specialist pool
2. fixed permanent 9-worker swarm identities
3. boot-time SAS/default-galaxy validation

I also ran the required 20-task math and MMLU benchmark slices and appended the dated execution log.

## Implemented

### 1. Micro-specialist pool

Files:
- `knowledge3d/cranium/ptx_runtime/micro_specialist_pool.py`
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`

What is live:
- boot-time SM-count sizing with the 66% reservation rule
- global pool registration on boot
- bounded slot acquisition/release
- graceful degradation when the pool is saturated
- Stage-1 runtime macro expansion for:
  - `MICRO_SPAWN`
  - `MICRO_RUN`
  - `MICRO_COLLECT`
  - `MICRO_RELEASE`

Important implementation reality:
- the host still reproduces the lightweight Tier-1 empty-stack fault on simple micro programs
- to keep the micro-specialist layer sovereign and GPU-native, the pool promotes those failed Tier-1 fragments into `TieredRPNEngine.execute_single(...)`
- this is GPU-to-GPU promotion, not Python reasoning
- the pool reports that explicitly through `tier1_empty_stack_fallbacks`

### 2. Fixed permanent nine-worker swarm

File:
- `knowledge3d/knowledgeverse/knowledgeverse.py`

Fixed slot order now live:
- `0 = gre_atomic_fission_fusion`
- `1 = gre_resonance_field`
- `2 = gre_vector_resonator`
- `3 = gre_arc_reasoner`
- `4 = gre_geometry_router`
- `5 = gre_graph_crystallizer`
- `6 = gre_temporal_reasoning`
- `7 = gre_fractal_emitter`
- `8 = gre_embedding_extractor`

What changed:
- `_finalize_swarm_paths(...)` now forces permanent slot/name assignment
- `_dispatch_swarm_weights(...)` uses surface-kind halting weights only
- `_apply_specialist_swarm_features(...)` runs the decomposer first, stores registers `60-62`, and records micro-pool stats into candidate metadata
- the decomposer is universal and active for ARC, math, and question/language tasks

### 3. Boot-time SAS/bootstrap validation

File:
- `knowledge3d/knowledgeverse/knowledgeverse.py`

What is live:
- SAS symbol bootstrap verification
- SAS grammar bootstrap verification
- required default-galaxy validation
- loud failure semantics for missing galaxies
- `_gpu_galaxy_binding` restoration after default-galaxy load so the all-live-galaxies contract remains pinned

## Supporting Contract Fixes

### Tablet/WINE typed benchmark ingress

Files:
- `knowledge3d/bridge/headless_tablet.py`
- `tests/bridge/test_headless_tablet.py`

What changed:
- tablet task payloads now explicitly carry:
  - `ARC_TASK`
  - `MATH_TASK`
  - `QUESTION_TASK`
- added a direct regression so `Knowledgeverse` no longer rewrites typed ingress to only the normalized surface kind before dispatch

### Benchmark CLI entrypoints

Files:
- `benchmarks/math_competitions.py`
- `benchmarks/mmlu.py`

Added live CLI support for the required spec commands and JSON summary output.

## Required Tests

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium env "PYTHONPATH=$(pwd)" \
  pytest -q \
  tests/bridge/test_headless_tablet.py \
  tests/test_swarm_always_nine.py \
  tests/test_halting_gate_weights.py \
  tests/test_decomposer_universal.py
```

Result:

```text
16 passed in 278.78s (0:04:38)
```

## Required Benchmark Runs

### Math

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium env "PYTHONPATH=$(pwd)" \
  python benchmarks/math_competitions.py \
  --max-tasks 20 \
  --summary-output /tmp/math_swarm_summary.json
```

Result:
- `1 / 20`
- `5.0%`
- route-family distribution: `MATH = 20`
- TRM dispatch task-type distribution: `MATH = 20`

### MMLU

Command:

```bash
bash scripts/k3d_env.sh run -e k3d-cranium env "PYTHONPATH=$(pwd)" \
  python benchmarks/mmlu.py \
  --max-tasks 20 \
  --summary-output /tmp/mmlu_swarm_summary.json
```

Result:
- `0 / 20`
- `0.0%`
- route-family distribution: `GENERAL = 20`
- TRM dispatch task-type distribution: `GENERAL = 20`

## Honest Remaining Gaps

### 1. MMLU still collapses to GENERAL

This is no longer a tablet-ingress problem.

What I verified:
- `TabletIngest.question_task(...)` now emits `type="QUESTION_TASK"`
- the direct preservation regression proves `Knowledgeverse._dispatch_sovereign_task(...)` no longer clobbers typed ingress at that seam
- despite that, the live 20-task MMLU run still returns:
  - `route_family = GENERAL`
  - `trm_dispatch.task_type = GENERAL`

Conclusion:
- the remaining bug is deeper inside the sovereign question-family selection/runtime path
- it is not benchmark-side Python orchestration anymore

### 2. Tier-1 lite kernel still faults on empty stack

The micro-specialist layer is live because it promotes failed Tier-1 fragments into the sovereign tiered GPU path, but the underlying lite-kernel defect still needs direct repair.

### 3. Not attempted in this pass

- Tier-3 / `AdvancedRPNEngine` unification
- any change to PID `400282`
- any Python reasoning shortcut in live evaluation

## Background Process

Protected ingest remained untouched.

Latest recheck during this pass:
- PID `400282`
- status `Ssl`
- elapsed `01:59:21`
