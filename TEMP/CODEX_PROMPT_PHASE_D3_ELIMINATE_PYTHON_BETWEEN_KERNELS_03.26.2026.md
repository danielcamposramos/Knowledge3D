# Codex: Phase D.3 — Eliminate Python Between GPU Kernels

**Date:** 2026-03-26
**Priority:** THIS IS WHERE THE 96% GPU IDLE TIME LIVES. The kernels work. The Python between them is the bottleneck.
**Binding specs:**
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` -- TRM IS the Avatar, game loop, Python = boot + I/O only (~200 lines)
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` SS4.1 -- fail-fast, ptx_fallback_rate = 0.0
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` SS3 -- VRAM-native, no CPU preprocessing
- `docs/vocabulary/HYPER_PARALLEL_PROCESSING.md` -- one-mind convergence, halting gate

**DO NOT STOP between parts. Execute ALL. The instructions are complete.**

---

## THE PROBLEM

Phase D.2 proved the recursive TRM fires (GPU spikes to 25%). But average GPU util is still 3.88% because the composed head pipeline — Morton, LED-A*, Frustum, LOD, Swarm, TRM — fires through a Python orchestration layer that:

1. **Reads back after EVERY kernel** — `morton_locate()` returns numpy array to Python, Python filters it, then passes to `frustum_visible()` which returns to Python, etc. Each readback is a `memcpy_dtoh` + `synchronize()` + Python processing.

2. **Builds Python dicts between kernels** — hundreds of lines of `dict.fromkeys()`, list comprehensions, `.get()` chains, `.strip()`, format strings. This is milliseconds of Python per query for what should be device-pointer passing.

3. **Sorts and filters on CPU** — `sorted()`, `max()`, list slicing, all on Python lists that were just read back from GPU.

The composed head pipeline in `_select_composed_head_candidate()` (line 10928 in knowledgeverse.py) calls these GPU kernels IN SEQUENCE with Python between each:

```
morton_locate()       → line 9745  → returns np.ndarray of candidate indices
[~150 lines of Python filtering/sorting]
LED-A* pathfind      → line ~10020 → returns path nodes
[~40 lines of Python filtering]
frustum_visible()    → line 10059  → returns np.ndarray of visible indices
[~40 lines of Python filtering/LOD]
lod_metrics()        → line 10079  → returns dict of LOD levels
[~100 lines of Python scoring/merging]
execute_swarm()      → line 10204  → returns resonance weights
[~40 lines of Python weight blending]
```

**Each GPU kernel takes microseconds. The Python between them takes MILLISECONDS.**

---

## THE FIX: Device-Side Index Buffers

### Principle

Instead of:
```
GPU kernel → memcpy_dtoh → Python processing → memcpy_htod → GPU kernel
```

Do:
```
GPU kernel → device buffer → GPU kernel → device buffer → GPU kernel → ONE readback
```

Keep candidate indices, visibility masks, LOD levels, and swarm weights ON THE GPU as device buffers. Only read back the FINAL answer.

### Part A: `query_head_substrate.py` — Device-Resident Pipeline

**File:** `knowledge3d/knowledgeverse/query_head_substrate.py`

This file contains `morton_locate()`, `frustum_visible()`, `lod_metrics()`. These methods currently return numpy arrays to the caller. Change them to OPTIONALLY return device pointers.

**Add device-output variants:**

```python
def morton_locate_device(
    self,
    query_embedding16: list[float],
    *,
    allowed_galaxy_indexes: set[int] | None = None,
    max_results: int = 128,
    morton_radius: int = 4,
    euclidean_radius: float = 5.0,
) -> tuple[int, int]:
    """Like morton_locate but returns (d_indices_ptr, count) — data stays on GPU."""
    # ... same kernel launch ...
    # Instead of memcpy_dtoh, return (device_pointer, result_count)
    return d_candidate_indices, count

def frustum_visible_device(
    self,
    query_embedding16: list[float],
    *,
    d_candidate_indices: int,
    candidate_count: int,
) -> tuple[int, int]:
    """Frustum cull using device-resident candidate indices. Returns (d_visible_ptr, count)."""
    # Read candidate indices FROM DEVICE, not from Python list
    # Return visible indices ON DEVICE
    return d_visible_indices, visible_count

def lod_metrics_device(
    self,
    query_embedding16: list[float],
    *,
    d_candidate_indices: int,
    candidate_count: int,
    saliency_threshold: float = 0.3,
) -> tuple[int, int]:
    """LOD filtering using device-resident indices. Returns (d_filtered_ptr, count)."""
    return d_filtered_indices, filtered_count
```

**The key:** These methods accept device pointers as INPUT and produce device pointers as OUTPUT. No Python lists. No memcpy between stages.

### Part B: Composed Head Device Pipeline

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`
**New method:** `_select_composed_head_candidate_device()`

Add a DEVICE-SIDE pipeline that chains the GPU kernels without Python readback between them:

```python
def _select_composed_head_candidate_device(
    self,
    *,
    query_embedding: list[float],
    target_galaxies: list[str],
    task_type: str,
    options: list[str] | None,
    domain_hint: str | None,
    task: dict[str, Any] | None,
    paths: list[dict[str, Any]],
    binding: dict[str, Any],
    selection_steps: list[str],
) -> dict[str, Any] | None:
    """Composed head pipeline with device-resident index buffers.

    GPU kernels pass device pointers to each other.
    Python only touches the FINAL result.
    """
    substrate = self.get_query_substrate()
    if substrate is None:
        return None

    # 1. Morton locate → device buffer
    d_morton_indices, morton_count = substrate.morton_locate_device(
        query_embedding16=query_embedding,
        allowed_galaxy_indexes=self._allowed_galaxy_indexes(target_galaxies),
        max_results=self._task_morton_max_results(task_type),
        morton_radius=self._task_morton_radius(task_type),
    )
    if morton_count == 0:
        return None  # fail-fast per spec

    # 2. Frustum cull → device buffer (reads morton output from device)
    d_visible_indices, visible_count = substrate.frustum_visible_device(
        query_embedding16=query_embedding,
        d_candidate_indices=d_morton_indices,
        candidate_count=morton_count,
    )

    # 3. LOD filter → device buffer (reads frustum output from device)
    d_lod_indices, lod_count = substrate.lod_metrics_device(
        query_embedding16=query_embedding,
        d_candidate_indices=d_visible_indices,
        candidate_count=visible_count,
        saliency_threshold=self._task_lod_saliency_threshold(task_type),
    )

    # 4. Nine-chain swarm → device scores
    swarm = self.get_swarm_bridge()
    if swarm is not None:
        # Swarm takes device-resident candidate indices
        swarm.execute_swarm_device(
            expand_embedding16_to128(query_embedding),
            d_candidate_indices=d_lod_indices,
            candidate_count=lod_count,
        )

    # 5. TRM recursive refinement (already device-native from D.1)
    trm_result = self._run_single_trm_tick(query_embedding)

    # 6. ONE readback: only the final scored candidates
    # Read the top-K scored candidates from device
    scored = substrate.read_top_candidates(
        d_indices=d_lod_indices,
        count=lod_count,
        top_k=24,
    )

    # Now Python only formats the final answer from scored results
    ...
```

**THIS IS THE ARCHITECTURAL CHANGE.** Steps 1-4 happen entirely on GPU. Python only sees the final scored candidates. The ~400 lines of Python filtering/sorting between kernels are replaced by device-to-device pointer passing.

### Part C: Keep the Old Path as Fallback (Temporarily)

Do NOT delete `_select_composed_head_candidate()` yet. Add a feature gate:

```python
# In query():
if os.getenv("K3D_DEVICE_PIPELINE", "0").strip().lower() in {"1", "true", "yes"}:
    best_candidate = self._select_composed_head_candidate_device(...)
else:
    best_candidate = self._select_composed_head_candidate(...)
```

This lets us A/B compare with the existing path.

### Part D: Remove numpy from knowledgeverse.py WHERE POSSIBLE

`knowledgeverse.py` is 13,616 lines and uses numpy HEAVILY. Do NOT try to eliminate all numpy in one pass. Focus on the HOT PATH only:

1. **`_select_composed_head_candidate`**: Replace `np.asarray(list(dict.fromkeys(candidate_indexes)), dtype=np.uint32)` (line 9753) with ctypes array or HostTensorF32
2. **`_decode_trm_galaxy_distribution`** (line 988): Replace `np.asarray` with HostTensorF32
3. **`_dispatch_swarm_weights`** (line 10224): Replace `np.asarray(trust_weights)` with list operations
4. **`_run_single_trm_tick`**: Already fixed in D.2, verify no numpy remnants

For each replacement: if the value came from a GPU kernel readback, it should stay as a device pointer or HostTensorF32. If it is pure Python logic (filtering, sorting), use plain Python lists.

**DO NOT touch the ~10,000 lines of Python that handle answer formatting, task routing, galaxy management.** Those are NOT in the hot path — they run ONCE per query after the composed head pipeline returns.

---

## Part E: Validate

```bash
# Compile check
python3 -m compileall knowledge3d/knowledgeverse/knowledgeverse.py
python3 -m compileall knowledge3d/knowledgeverse/query_head_substrate.py

# Focused tests
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py

# Feature-gate smoke: run with device pipeline ON
K3D_DEVICE_PIPELINE=1 python3 -c "
from knowledge3d.knowledgeverse import Knowledgeverse
kv = Knowledgeverse()
result = kv.query('What is 2+3?', specialist='math')
print(f'Answer: {result.get(\"answer\", result.get(\"predicted_answer\", \"none\"))}')
print(f'GPU execution: {result.get(\"gpu_execution\", False)}')
print('Device pipeline smoke: PASSED')
"
```

---

## Part F: Benchmark with Device Pipeline

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
export K3D_DEVICE_PIPELINE=1
conda activate k3d-cranium

nohup python3 -u benchmarks/run_all.py \
  --warm --sample-rate 0.35 \
  > /tmp/k3d_phaseD3_device_pipeline_warm_35pct_03.26.log 2>&1 &

echo "Phase D.3 benchmark launched. PID: $!"
```

**While it runs:** 2-minute live monitor focused on GPU utilization.

---

## Part G: Report

Write to `TEMP/CLAUDE_PHASE_D3_DEVICE_PIPELINE_REPORT_03.26.2026.md` with:

1. All 5 suite scores + combined
2. GPU utilization comparison:
   | Metric | Phase 3C | Phase D.2 | Phase D.3 |
   |--------|----------|-----------|-----------|
   | GPU avg | 0.17% | 3.88% | ? |
   | GPU max | 1.00% | 25.00% | ? |
   | CPU avg | 113% | 112% | ? |
3. Throughput per suite (seconds/question)
4. Device pipeline kernel chain timing (if measurable)
5. Contrastive/sleep-time outcome
6. numpy count: `rg "import numpy|from numpy" knowledge3d/knowledgeverse/query_head_substrate.py` — report but do NOT require zero yet (knowledgeverse.py is too big for one pass)

---

## THE VISION

**Before D.3:**
```
Python → kernel → Python → kernel → Python → kernel → Python → kernel → Python → answer
  10ms    50us    15ms     30us    10ms     20us    8ms     40us    5ms
```
Total: ~48ms Python + ~140us GPU = GPU doing 0.3% of the work

**After D.3:**
```
Python → kernel → kernel → kernel → kernel → Python → answer
  10ms    50us     30us     20us     40us    5ms
```
Total: ~15ms Python + ~140us GPU = GPU doing 0.9% of the work

This is STILL not the final state (the final state is TRM driving the loop from GPU), but it eliminates the BIGGEST source of Python overhead: the readback-process-reupload cycle between each kernel in the composed head pipeline.

---

## EXECUTION ORDER — DO NOT STOP

A -> B -> C -> D -> E -> F -> G

All in sequence. No pauses. The instructions are HERE.
