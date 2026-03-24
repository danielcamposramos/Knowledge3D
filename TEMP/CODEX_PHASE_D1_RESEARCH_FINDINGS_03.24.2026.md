# Codex Phase D.1 Research Findings

**Date:** 2026-03-24  
**Scope:** Research only. No implementation proposals.  
**Prompt:** `TEMP/CODEX_PROMPT_CONTRASTIVE_ROOT_CAUSE_AND_PHASE_D_03.24.2026.md`

This document maps the current Python→GPU boundary honestly, with file:line citations.

## 1. Where `trm_step_fused.ptx` gets called today

`trm_step_fused.ptx` is loaded during Knowledgeverse/TRM boot and is currently used as an optional tick/refinement path, not as the dominant whole-question runtime.

Load path:

- `knowledge3d/knowledgeverse/knowledgeverse.py:766`
- `knowledge3d/knowledgeverse/knowledgeverse.py:782`
- `knowledge3d/cranium/sovereign/trm_launcher.py:121`
- `knowledge3d/cranium/sovereign/trm_launcher.py:125`

Direct tick launch path:

- `knowledge3d/knowledgeverse/knowledgeverse.py:956`
- `knowledge3d/knowledgeverse/knowledgeverse.py:962`

Generic cranium fused refine path:

- `knowledge3d/cranium/sovereign/trm_launcher.py:136`
- `knowledge3d/cranium/sovereign/trm_launcher.py:179`
- `knowledge3d/cranium/sovereign/trm_launcher.py:523`
- `knowledge3d/cranium/sovereign/trm_launcher.py:533`

Current flow:

1. Knowledgeverse boots with `TRMLauncher(use_fused=True)`.
2. `_run_single_trm_tick()` launches `kernel_fused` directly.
3. `query()` only reaches this tick path behind the TRM shadow/navigation gates:
   - `knowledge3d/knowledgeverse/knowledgeverse.py:13206`
   - `knowledge3d/knowledgeverse/knowledgeverse.py:13224`
4. Most benchmark questions still traverse a larger Python-owned orchestration path before or around this optional fused tick.

Current conclusion:

- `trm_step_fused.ptx` exists and is callable.
- It is **not** yet the sole per-question game loop.
- The live system is still primarily orchestrated above it in Python.

## 2. Where the Python benchmark loop lives

Per-question iteration is still owned by Python benchmark runners and by the Python TRM shell.

Top-level runner:

- `scripts/run_enriched_benchmarks.py:482`
- `scripts/run_enriched_benchmarks.py:538`

Row persistence / suite runner:

- `knowledge3d/tools/benchmark_health_check.py:262`
- `knowledge3d/tools/benchmark_health_check.py:413`

Per-suite Python loops:

- MMLU:
  - `benchmarks/mmlu.py:194`
  - `benchmarks/mmlu.py:211`
- Unified math:
  - `benchmarks/math_competitions.py:770`
  - `benchmarks/math_competitions.py:800`
- LHE:
  - `benchmarks/last_humanity_exam.py:202`
  - `benchmarks/last_humanity_exam.py:219`
- ARC:
  - `benchmarks/arc_agi_2.py:173`
  - `benchmarks/arc_agi_2.py:224`

Python TRM shell:

- `knowledge3d/knowledgeverse/knowledgeverse.py:13606`
- `knowledge3d/knowledgeverse/trm_game_loop.py:98`
- `knowledge3d/knowledgeverse/trm_game_loop.py:114`
- `knowledge3d/knowledgeverse/knowledgeverse.py:13580`

Current flow:

1. A benchmark loop iterates rows in Python.
2. Each row calls `kv.execute_task(...)`.
3. `Knowledgeverse.execute_task()` enqueues a request into the TRM I/O shell.
4. `TRMGameLoop.tick()` drains requests in a Python `while` loop.
5. It then calls `_execute_task_direct()` one task at a time.

Current conclusion:

- The benchmark layer is still one-question-at-a-time Python.
- The TRM “game loop” is still a Python queue/tick shell over the underlying GPU kernels.

## 3. Which PTX kernels fire during a single question

The exact count is data-dependent and still partly hidden by Python control flow, but the structural launch sites are clear.

Spatial/navigation kernels:

- Morton query/refine:
  - `knowledge3d/cranium/spatial_sovereign/morton_octree.py:179`
  - `knowledge3d/cranium/spatial_sovereign/morton_octree.py:213`
- LED-A*:
  - `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py:271`
- Frustum culling:
  - `knowledge3d/cranium/spatial_sovereign/frustum.py:164`
- Dynamic LOD:
  - `knowledge3d/knowledgeverse/query_head_substrate.py:68`

Swarm / executive / halting kernels:

- swarm dispatch call site:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:10186`
- nine-chain specialized bridge launch:
  - `knowledge3d/cranium/bridges/nine_chain_specialized_bridge.py:220`
- cognitive executive path:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:10198`
  - `knowledge3d/cranium/bridges/sovereign_bridges.py:562`
- halting gate:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:10534`
  - `knowledge3d/cranium/bridges/sovereign_bridges.py:1620`

Embedding kernels:

- query trigram embedding loop:
  - `knowledge3d/cranium/rpn_embedding_engine.py:227`
- trigram bridge kernel launches:
  - `knowledge3d/cranium/bridges/trigram_embed_bridge.py:106`
  - `knowledge3d/cranium/bridges/trigram_embed_bridge.py:126`

RPN evaluation path:

- candidate batch scoring:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:13105`
  - `knowledge3d/knowledgeverse/knowledgeverse.py:13130`
- modular engine batch path:
  - `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py:534`
- tiered RPN dispatch:
  - `knowledge3d/cranium/bridges/tiered_rpn.py:225`
- sovereign bridge execution:
  - `knowledge3d/cranium/bridges/sovereign_bridges.py:1779`
  - `knowledge3d/cranium/bridges/sovereign_bridges.py:1895`

Optional TRM nav/shadow kernels:

- matryoshka projection:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:917`
  - `knowledge3d/cranium/bridges/matryoshka_bridge.py:67`
- fused TRM step:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:962`

Current conclusion:

- A single question can touch many kernels.
- The important architectural fact is not the absolute count, but that Python still sequences many of these launches and waits between stages.
- The launch pattern is therefore mixed: GPU kernels exist, but orchestration is still largely serial and host-owned.

## 4. What the Jarvis dispatch path is today

Jarvis currently exists as mostly Python orchestration and briefing logic. It is not yet the GPU-native dispatch owner.

Tablet/daemon route path:

- `knowledge3d/bridge/headless_tablet.py:351`
- `knowledge3d/daemon/main.py:943`
- `knowledge3d/daemon/main.py:1007`

TRM game-loop bridge:

- `knowledge3d/knowledgeverse/knowledgeverse.py:13606`
- `knowledge3d/knowledgeverse/trm_game_loop.py:98`

Dispatch ticket planning:

- `knowledge3d/knowledgeverse/trm_game_loop.py:210`
- `knowledge3d/knowledgeverse/trm_game_loop.py:276`

Jarvis-related Knowledgeverse methods:

- `_dispatch_swarm_weights`:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:10172`
- `_jarvis_task_complexity`:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:10245`
- `_jarvis_determine_swarm_count`:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:10264`
- `_jarvis_compile_brief`:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:10285`
- `_jarvis_record_brief`:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:10384`
- brief attachment after scoring:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:12008`
  - `knowledge3d/knowledgeverse/knowledgeverse.py:12029`

Current flow:

1. Tablet or daemon routes a task in Python.
2. `execute_task()` and `TRMGameLoop` build the request in Python.
3. `_build_dispatch_ticket()` calculates swarm groups and worker slots in Python.
4. `query()` / `_select_composed_head_candidate()` invokes swarm weighting.
5. Jarvis compiles agreements, contradictions, confidence, and brief payloads in Python dict/list structures.

Current conclusion:

- Jarvis is present, but not yet as a GPU-native dispatcher.
- Python still decides swarm sizing, worker-slot assembly, and brief construction.

## 5. How much data crosses host↔device per question

There is not yet a single central counter, but the major host↔device transfer sites on the question path are identifiable and numerous.

Query embedding transfers:

- `knowledge3d/cranium/rpn_embedding_engine.py:176`
- `knowledge3d/cranium/bridges/trigram_embed_bridge.py:72`
- `knowledge3d/cranium/bridges/trigram_embed_bridge.py:97`
- `knowledge3d/cranium/bridges/trigram_embed_bridge.py:140`

RPN runtime transfers:

- `knowledge3d/cranium/bridges/sovereign_bridges.py:1756`
- `knowledge3d/cranium/bridges/sovereign_bridges.py:1819`
- `knowledge3d/cranium/bridges/sovereign_bridges.py:1849`
- `knowledge3d/cranium/bridges/sovereign_bridges.py:1875`

Query-head / navigation transfers:

- `knowledge3d/knowledgeverse/query_head_substrate.py:60`
- `knowledge3d/knowledgeverse/query_head_substrate.py:82`
- `knowledge3d/cranium/spatial_sovereign/morton_octree.py:175`
- `knowledge3d/cranium/spatial_sovereign/morton_octree.py:196`
- `knowledge3d/cranium/spatial_sovereign/frustum.py:155`
- `knowledge3d/cranium/spatial_sovereign/frustum.py:182`
- `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py:264`
- `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py:287`

Swarm / trust / halting transfers:

- `knowledge3d/cranium/bridges/nine_chain_specialized_bridge.py:155`
- `knowledge3d/cranium/bridges/nine_chain_specialized_bridge.py:165`
- `knowledge3d/cranium/bridges/sovereign_bridges.py:558`
- `knowledge3d/cranium/bridges/sovereign_bridges.py:574`
- `knowledge3d/cranium/bridges/sovereign_bridges.py:1618`
- `knowledge3d/cranium/bridges/sovereign_bridges.py:1636`

Bind/rebind transfers that are not necessarily every question but still shape residency:

- flattened galaxy buffer:
  - `knowledge3d/knowledgeverse/knowledgeverse.py:1288`
  - `knowledge3d/cranium/bridges/sovereign_bridges.py:1720`
- unified LOD buffer:
  - `knowledge3d/knowledgeverse/query_head_substrate.py:42`
  - `knowledge3d/knowledgeverse/query_head_substrate.py:189`

Optional TRM nav/shadow transfers:

- `knowledge3d/knowledgeverse/knowledgeverse.py:910`
- `knowledge3d/knowledgeverse/knowledgeverse.py:922`
- `knowledge3d/knowledgeverse/knowledgeverse.py:935`
- `knowledge3d/knowledgeverse/knowledgeverse.py:981`

Current conclusion:

- A question still causes many explicit H→D and D→H boundaries.
- Each one is a synchronization point where Python remains in control of the sequence.
- The system is not yet operating as a continuously resident GPU-native live world.

## Overall Boundary Assessment

The current architecture already contains many sovereign PTX kernels and bridges, but the control boundary still sits too high in Python:

- benchmark suites iterate questions in Python
- `execute_task()` and `TRMGameLoop.tick()` still orchestrate one task at a time in Python
- Jarvis planning and briefing are still Python-owned
- many query-stage transfers still cross host↔device explicitly per question
- `trm_step_fused.ptx` exists, but is not yet the sole always-on game loop that owns the full question lifecycle

That is the current state from the codebase as of 2026-03-24.
