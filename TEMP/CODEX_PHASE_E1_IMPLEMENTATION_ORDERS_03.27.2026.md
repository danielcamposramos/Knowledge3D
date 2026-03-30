# Codex — Phase E.1 IMPLEMENTATION ORDERS

**Date:** 2026-03-27
**From:** Daniel (Chair) + Claude (Architecture)
**To:** Codex
**Type:** IMPLEMENTATION ORDER — NOT A DISCUSSION. BUILD THIS.

---

## READ FIRST

The diagnosis is DONE. We all agree. Stop analyzing. **Build.**

The full architectural spec is at:
- `TEMP/CLAUDE_PHASE_E_ARC_AGI_3_SOVEREIGNTY_SPEC_03.27.2026.md`
- `TEMP/CODEX_PHASE_E1_GPU_GAME_LOOP_DIRECTIVE_03.27.2026.md`

Those documents contain the analysis, the reasoning, the comparisons. You do NOT need to produce another analysis. You need to produce CODE.

---

## ORDER 1: Create VRAM Task Buffer

**Create file:** `knowledge3d/knowledgeverse/vram_task_buffer.py`

Use the sovereign loader (`knowledge3d/cranium/sovereign/loader.py`) to allocate two VRAM buffers via `cuMemAlloc`:

- **Input buffer:** 1024 bytes × 6000 slots = ~6 MB
  - Per slot: query_embedding (float32×32), task_type (uint32), option_count (uint32), option_embeddings (float32×32×4), subject_id (uint32), domain_hint_id (uint32)
- **Output buffer:** 512 bytes × 6000 slots = ~3 MB
  - Per slot: answer_index (uint32), confidence (float32), convergence_signal (int8), answer_text_hash (uint64), iterations_used (uint32)

Implement two methods:
- `bulk_load(tasks: list[dict]) -> int` — marshals ALL questions into VRAM with ONE `cuMemcpyHtoD` call
- `read_results(count: int) -> list[dict]` — reads ALL answers from VRAM with ONE `cuMemcpyDtoH` call

**Test:** Load 10 questions, verify they round-trip through VRAM correctly.

---

## ORDER 2: Create GPU Task Dispatch Kernel

**Create file:** `knowledge3d/cranium/cuda/gpu_task_dispatch.cu`

This kernel reads from the VRAM input buffer and writes to the VRAM output buffer. It calls the existing device functions that D.3 already proved work:

```
For each task slot in input_buffer:
  1. Read query_embedding from slot
  2. Call morton_locate (existing device function)
  3. Call led_astar_expand (existing device function)
  4. Call frustum_cull (existing device function)
  5. Call dynamic_lod (existing device function)
  6. Call nine_chain_swarm (existing kernel — may need refactor to device function)
  7. Call gre_multimodal_halting_gate (existing kernel — may need refactor to device function)
  8. Write result to output_buffer[task_id]
```

If calling existing kernels as device functions requires refactoring them, DO THAT. Extract the core logic from each `.cu` file into a `__device__` function that can be called from `gpu_task_dispatch`.

**Granularity choice is yours:** one-kernel-per-task, CUDA graph, persistent kernel, or bulk-phase kernels. The REQUIREMENT is: Python does NOT enter the loop between tasks.

**Test:** Process 10 questions through the GPU dispatch. Verify answers match what the current Python path produces for the same 10 questions.

---

## ORDER 3: Create New Benchmark Runner

**Create file:** `scripts/run_gpu_benchmark.py`

This replaces `scripts/run_enriched_benchmarks.py` for the sovereign path. ~100 lines of Python:

```python
# 1. Boot
# Load House, Galaxies, TRM weights into VRAM (existing load_knowledgeverse() code)

# 2. Load questions
# Read benchmark JSON files into Python lists (existing benchmark loader code)
# Marshal into VRAMTaskBuffer.bulk_load()

# 3. Launch
# Call gpu_task_dispatch kernel (or CUDA graph)

# 4. Collect
# VRAMTaskBuffer.read_results()
# Write health_log.jsonl (ONE write per suite, not per question)

# 5. Report
# Print scores, run sleep-time consolidation
```

**Test:** Run the full 5954-question benchmark. Compare scores against D.2 baseline.

---

## ORDER 4: Move Composed-Head Scoring to GPU

**Migrate from:** `knowledgeverse.py:11547-12374` (`_select_composed_head_candidate`)

**Migrate to:** A device function callable from `gpu_task_dispatch.cu`

The scoring logic that matters:
1. `match_similarity × galaxy_weight × swarm_weight × defeasible_modifier` — arithmetic, trivially GPU
2. Defeasible resolution — `gre_defeasible_resolver.cu` already exists, call it
3. Halting gate — `gre_multimodal_halting_gate.cu` already exists, call it
4. Option similarity for MMLU — cosine between embeddings, trivially GPU
5. Argmax — trivially GPU

Everything else in that 830-line function is Python bookkeeping (string building, dict copying, debug logging). DELETE IT.

---

## ORDER 5: Wire GRE Specialists

The 15 GRE specialist kernels are loaded but never called. Wire them into the swarm dispatch inside `gpu_task_dispatch.cu`:

- `task_type == ARC` → activate `gre_arc_reasoner`, `gre_geometry_router`, `gre_fractal_emitter`
- `task_type == MATH` → activate `gre_atomic_fission_fusion`, `gre_geometry_router`
- `task_type == GSM8K` → activate `gre_atomic_fission_fusion`, `gre_temporal_reasoning`
- `task_type == LHE` → activate `gre_graph_crystallizer`
- `task_type == MMLU` → activate `gre_resonance_field`, `gre_vector_resonator`

All other workers use base TRM weights.

---

## EXECUTION ORDER

Do them in this sequence. Each builds on the previous:

1. **Order 1** (VRAM buffers) — can be tested immediately with existing Python path
2. **Order 2** (GPU dispatch kernel) — depends on Order 1, needs existing kernels refactored to device functions
3. **Order 4** (scoring migration) — integrate into Order 2's kernel
4. **Order 5** (wire specialists) — integrate into Order 2's swarm dispatch
5. **Order 3** (new benchmark runner) — wraps everything, run the real benchmark

---

## WHAT NOT TO DO

- **Do NOT produce another analysis document.** The analysis is done.
- **Do NOT add Python threading, asyncio, or multiprocessing.** That is explicitly rejected.
- **Do NOT optimize the existing Python code path.** It's being replaced, not optimized.
- **Do NOT keep the Python game loop "as fallback."** Zero fallbacks. Ever.
- **Do NOT JSON-serialize between GPU stages.** Data stays in VRAM.
- **Do NOT write "what Claude should plan next."** Claude already planned. You implement.

---

## SUCCESS = ONE THING

The benchmark runs in minutes, not hours. GPU utilization > 10%. Python lines in hot path = 0.

**Build it.**
