# Resident-World Lifecycle + Route-Family Fix

**Date:** 2026-04-15
**Author:** Claude (Architecture Partner)
**Branch:** codex/batch11-knowledge-waves-observability-game2d-2026-04-15
**Status:** P0 — Blocks all benchmark questioning within time budget

---

## Context

Codex confirmed: the 300s smoke run consumed its entire budget in boot/rebuild and produced no results. The swarm wiring from the previous spec is in place and tested, but it is unreachable because the system never finishes loading.

There are two separate bugs here. Fix A is blocking. Fix B is correctness. Both are small and surgical.

---

## Root Cause Analysis

### Bug A — Sovereign runtime is built before curriculum galaxies load, defeating the artifact cache

**The ordering problem:**

`Knowledgeverse.__init__` (with `eager_load_default_galaxies=True`) calls:
1. `self.ensure_default_galaxies_loaded()` — loads Drawing, Character, Word, etc.
2. `self._boot_sovereign_runtime(force_reload=True)` — builds the runtime from whatever is loaded **at that moment**

The runner then calls:
3. `load_canonical_curriculum_into_knowledgeverse(kv)` — adds benchmark stars to galaxies
4. `assert_canonical_curriculum_loaded(kv)` — verifies they're present

The sovereign runtime was built in step 2, BEFORE the benchmark stars were added in step 3. The runtime artifact saved to disk reflects **only the default galaxies** — the curriculum stars are not in it.

On the NEXT run:
- Step 2 calls `ensure_loaded` which checks artifact cache
- Cache signature does NOT match (curriculum stars are now in the galaxies but weren't in the cached build)
- Cache miss → full rebuild → consumes the entire time budget again

The artifact cache can **never** be warm because the build always happens before the full galaxy set exists.

**The `force_reload=True` amplifier:**

`_boot_sovereign_runtime(force_reload=True)` calls `runtime.invalidate_loaded_state()` first.
This zeroes `star_count` and `_catalog_signature`, bypassing even the `resident` fast path.
Even if the runtime were somehow already warm, `force_reload=True` would destroy it.

**`ensure_loaded` fast paths (sovereign_hot_path.py line 950):**
```
resident       → star_count > 0 and _catalog_signature set → instant
runtime_artifacts → valid artifact bundle on disk → load in seconds
full_build     → materialization from scratch → minutes
```
Both fast paths are always bypassed. Every run is `full_build`.

---

### Bug B — LHE routes as MMLU (route_family mismatch)

`_normalize_semantic_task_type` (knowledgeverse.py line 744):
```python
"MMLU_TASK": "QUESTION",
"LHE_TASK":  "QUESTION",   # ← wrong: LHE is multi-hop, not single-hop factual
```

LHE is multi-hop chained reasoning (graph crystallizer + temporal reasoning heavy).
MMLU is single-hop factual recall (vector similarity heavy).
Treating them identically means:
- LHE queries compete against MMLU stars for the same route slots
- MMLU stars win (more of them, higher base similarity)
- Warm-up shows `route_family: "MMLU"` on LHE trace → wrong star type selected
- Graph traversal kernels are not weighted up for LHE

---

## Codex Implementation Spec

---

### Fix A — Defer sovereign runtime boot in the runner (2 file changes)

#### Change A-1: Add `defer_sovereign_boot` parameter to `Knowledgeverse.__init__`

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`
**Location:** The class `__init__` signature and line 664

Current signature (find it near the top of `__init__`):
```python
def __init__(
    self,
    storage_root: ...
    ...
    eager_load_default_galaxies: bool = True,
    ...
):
```

Add new parameter:
```python
def __init__(
    self,
    storage_root: ...
    ...
    eager_load_default_galaxies: bool = True,
    defer_sovereign_boot: bool = False,   # ← ADD THIS
    ...
):
```

Then find line 662–669:
```python
if eager_load_default_galaxies:
    boot_stage_t0 = time.perf_counter()
    runtime_summary = self._boot_sovereign_runtime(force_reload=True)
    self._record_boot_stage(
        "sovereign_runtime_load",
        boot_stage_t0,
        summary=dict(runtime_summary or {}),
    )
```

Replace with:
```python
if eager_load_default_galaxies and not defer_sovereign_boot:
    boot_stage_t0 = time.perf_counter()
    runtime_summary = self._boot_sovereign_runtime(force_reload=False)   # ← False, not True
    self._record_boot_stage(
        "sovereign_runtime_load",
        boot_stage_t0,
        summary=dict(runtime_summary or {}),
    )
```

**Two changes in that block:**
1. Guard with `not defer_sovereign_boot`
2. `force_reload=True` → `force_reload=False`

The `force_reload=False` change is load-bearing even for callers that don't use `defer_sovereign_boot`. It stops `invalidate_loaded_state()` from being called on every boot, allowing the `resident` fast path to work for long-running sessions.

#### Change A-2: Runner creates world once, boots runtime after full galaxy load

**File:** `scripts/run_headless_tablet_benchmarks.py`
**Location:** `run_tablet_benchmark_suite` function, lines 831–848

Current:
```python
kv = Knowledgeverse(storage_root=storage_root)
if hasattr(kv, "suspend_auto_sleep"):
    kv.suspend_auto_sleep()
curriculum_summary = load_canonical_curriculum_into_knowledgeverse(
    kv,
    progress=lambda message: _log_section(message),
)
curriculum_assertion = assert_canonical_curriculum_loaded(kv)
if str(curriculum_assertion.get("status") or "").lower() != "ok":
    raise RuntimeError(...)
tablet = HeadlessTabletMPC(
    knowledgeverse=kv,
    ...
)
```

Replace with:
```python
kv = Knowledgeverse(storage_root=storage_root, defer_sovereign_boot=True)   # ← defer
if hasattr(kv, "suspend_auto_sleep"):
    kv.suspend_auto_sleep()
curriculum_summary = load_canonical_curriculum_into_knowledgeverse(
    kv,
    progress=lambda message: _log_section(message),
)
curriculum_assertion = assert_canonical_curriculum_loaded(kv)
if str(curriculum_assertion.get("status") or "").lower() != "ok":
    raise RuntimeError(...)

# Boot sovereign runtime AFTER full galaxy set is loaded.
# force_reload=False: hits resident fast path if already warm; artifact path on first run.
_log_section("booting sovereign runtime (post-curriculum)")
sovereign_boot_t0 = time.perf_counter()
sovereign_boot_summary = kv._boot_sovereign_runtime(force_reload=False)
_log_section(
    f"sovereign_runtime mode={sovereign_boot_summary.get('mode', '?')} "
    f"stars={sovereign_boot_summary.get('star_count', '?')} "
    f"elapsed={round(time.perf_counter() - sovereign_boot_t0, 2)}s"
)

tablet = HeadlessTabletMPC(
    knowledgeverse=kv,
    ...
)
```

**What this achieves:**
- Run 1: `ensure_loaded` finds no artifacts → full build from default + curriculum stars → saves artifacts → ~minutes but only once
- Run 2+: `ensure_loaded` finds valid artifacts (signature includes curriculum) → `runtime_artifacts` load → seconds → questioning starts well within budget
- Any run with resident memory (long-running session reuse): `resident` fast path → milliseconds

**Observability:** The `_log_section` on `mode=` makes it visible in the run log which path was taken.

#### Change A-3: Sleeptime save at end of run

After all suites complete (in the `finally` block around line 932), add a sleeptime consolidation save so the sovereign state is persisted for the next run:

**File:** `scripts/run_headless_tablet_benchmarks.py`
**Location:** in the `finally:` block after all suites complete (around line 932)

Find the `finally:` block and add before `execution_summary = _write_execution_artifacts(...)`:
```python
# Persist evolved sovereign state so next run benefits from this session.
try:
    if hasattr(kv, "_save_runtime_artifacts"):
        kv._save_runtime_artifacts()
        _log_section("sovereign runtime artifacts saved")
    elif hasattr(kv, "_get_sovereign_hot_path"):
        hot_path = kv._get_sovereign_hot_path()
        if hasattr(hot_path, "_save_runtime_artifacts"):
            hot_path._save_runtime_artifacts()
            _log_section("sovereign runtime artifacts saved")
except Exception as exc:
    _log_section(f"sovereign runtime artifact save skipped: {exc}")
```

This ensures subsequent runs find fresh artifacts and hit the fast path.

---

### Fix B — Give LHE its own surface kind (route-family correctness)

#### Change B-1: Add "LHE" to `_normalize_semantic_task_type`

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`
**Location:** line 744 in `_normalize_semantic_task_type`

Current:
```python
"MMLU_TASK": "QUESTION",
"LHE_TASK": "QUESTION",
```

Change to:
```python
"MMLU_TASK": "QUESTION",
"LHE_TASK": "LHE",
"LHE": "LHE",
```

#### Change B-2: Add LHE to route minima tables

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

Add to `MEANING_FAMILY_ROUTE_MINIMA` (around line 79):
```python
"LHE": {"routers": 1, "executors": 4, "validators": 3, "anti_patterns": 2},
```

Add to `MEANING_ROUTE_CLOSURE_MINIMA` (around line 88):
```python
"LHE": {"surface_bridges": 1, "routers": 1, "executors": 5, "materializers": 1, "validators": 3, "anti_patterns": 2},
```

#### Change B-3: Add LHE halting weights

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

Add to `HALTING_WEIGHT_TABLE` (around line 371):
```python
"LHE": (1.5, 1.5, 1.5, 0.5, 0.5, 3.0, 2.5, 0.5, 2.0),
```

Worker order (FIXED_GRE_WORKERS):
`gre_atomic_fission_fusion, gre_resonance_field, gre_vector_resonator, gre_arc_reasoner, gre_geometry_router, gre_graph_crystallizer, gre_temporal_reasoning, gre_fractal_emitter, gre_embedding_extractor`

Rationale for LHE weights:
- `gre_graph_crystallizer` → 3.0 (multi-hop traversal, most critical for LHE)
- `gre_temporal_reasoning` → 2.5 (chain ordering in multi-hop)
- `gre_embedding_extractor` → 2.0 (semantic recall for each hop)
- `gre_atomic_fission_fusion` → 1.5 (decompose multi-part questions)
- `gre_resonance_field`, `gre_vector_resonator` → 1.5 each (cross-galaxy lookup)
- `gre_arc_reasoner`, `gre_geometry_router`, `gre_fractal_emitter` → 0.5 (not primary for text multi-hop)

#### Change B-4: Add LHE surface bridge prefix

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`
**Location:** `_embed_query_batch_gpu` surface_bridge_prefix dict (around line 5461)

Current:
```python
"QUESTION": "question option evidence factual recall comparison",
```

Add:
```python
"LHE": "multi-hop chained reasoning evidence chain graph traversal inference",
```

This biases the query embedding toward multi-hop reasoning stars rather than single-hop factual stars.

---

## Success Criteria

### Fix A
- Run the smoke: `python scripts/run_headless_tablet_benchmarks.py --mmlu-count 1`
- Log must show: `sovereign_runtime mode=runtime_artifacts` or `mode=resident` (NOT `mode=full_build`) on the second run
- Questioning must begin within 60s of process start on the second run
- No results-absent timeout after 300s
- First run is allowed to be slow (full_build), subsequent runs must be fast

### Fix B
- After Fix A is working, run the warm-up probes
- LHE warm-up trace must show `route_family: "LHE"` (not `"MMLU"`)
- LHE warm-up `applied_kernels` must include `"gre_graph_crystallizer"` with non-trivial weight
- LHE and MMLU must select different candidate stars for the same semantic question

---

## What This Does NOT Change

- Galaxy ingestion, star population, GRE kernels — untouched
- The NChainSwarmBridge wiring from the previous spec — untouched
- The GAME_2D / MATH route path — untouched
- `force_reload=True` in the House import path (line 1721) — that's a full House rebuild, intentional
- `force_reload=True` exposed via the public API method (line 2425) — callers who pass `force=True` explicitly still get it

---

## Lifecycle Model (After Fix)

```
Runner start
│
├── Knowledgeverse(defer_sovereign_boot=True)
│     ├── ensure_default_galaxies_loaded()   [default galaxies in VRAM]
│     └── [NO runtime boot yet]
│
├── load_canonical_curriculum_into_knowledgeverse()
│     └── [benchmark stars added to galaxies]
│
├── assert_canonical_curriculum_loaded()     [verify full galaxy set]
│
├── kv._boot_sovereign_runtime(force_reload=False)
│     ├── run 1:  full_build (slow) → saves artifacts with full catalog signature
│     └── run 2+: runtime_artifacts (fast) OR resident (instant if still warm)
│
├── tablet = HeadlessTabletMPC(kv)
│
├── [warm-up probes]
├── [suite runs — many questions through one resident world]
│
└── finally:
      ├── sleeptime consolidation (strengthen correct paths)
      └── _save_runtime_artifacts() → warm for next run
```

This is the intended K3D lifecycle: **one resident AI, many internal questions, sleep at end**.

---

## Handoff to Codex

**Priority:** P0 for Fix A (questioning blocked). P1 for Fix B (correctness).

**Files to edit:**
1. `knowledge3d/knowledgeverse/knowledgeverse.py` — A-1, B-1, B-2, B-3, B-4
2. `scripts/run_headless_tablet_benchmarks.py` — A-2, A-3

**Do not touch:**
- `sovereign_hot_path.py` — `ensure_loaded` logic is correct as-is
- Any PTX kernel or bridge
- Any galaxy population script

**Test sequence:**
```bash
# 1. Compile check
python3 -m py_compile knowledge3d/knowledgeverse/knowledgeverse.py
python3 -m py_compile scripts/run_headless_tablet_benchmarks.py

# 2. Warm-up probes (should still all CONVERGE)
# run with --mmlu-count 0 --math-count 0 etc. (warm-up only)

# 3. Smoke — first run (will be slow, full_build)
python scripts/run_headless_tablet_benchmarks.py --mmlu-count 1
# expect: mode=full_build in log, questioning starts, result produced

# 4. Smoke — second run (must be fast)
python scripts/run_headless_tablet_benchmarks.py --mmlu-count 1
# expect: mode=runtime_artifacts in log, starts in <60s, result produced

# 5. LHE route family check
# warm-up LHE probe must show route_family="LHE" after Fix B
```

**Sovereignty check:**
```bash
grep -n "force_reload=True" knowledge3d/knowledgeverse/knowledgeverse.py
# Should only match the House-import path (line 1721) and the public API method (~line 2425)
# Must NOT match the __init__ eager_load path
```

---

*REMINDER: Claude does ARCHITECTURE, not implementation.*
*Codex implements. This spec is the handoff.*
