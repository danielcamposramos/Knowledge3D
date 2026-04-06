# CODEX E.56 — Spine Star Wiring, Role Cross-Reference Fix, Legacy Cleanup

**Date:** April 2, 2026
**Priority:** CRITICAL — non-game tasks (MATH, QUESTION, GRAMMAR) produce route_depth=1 because all foundational star cross-references are silently lost
**Blocked on:** E.56 must land before any benchmark or sleep-time training is meaningful for non-GAME_2D families
**Sovereignty:** No Python in hot path. No fallbacks. No stubs. We fail and fix.

---

## Root Cause (Read This Before Touching Anything)

The CUDA kernel (`gpu_task_dispatch.cu`) has full role-routing infrastructure. The E.55 smoke test proved it works: three manually-crafted stars with explicit router→executor→validator roles achieved route_depth=3. ARC3 achieves 30/30.

**Non-game tasks fail because foundational star cross-refs are silently zeroed out at load time.**

Trace of the bug:

1. `build_foundational_galaxy_table()` (foundational_galaxy_builder.py:450-453) pops `_ref_ids` from each star and replaces with `component_refs` as **locally pre-resolved integer indices** (valid only within the foundational table's own ordering, not the full sovereign table).

2. `galaxy_loader.load_all_galaxies_from_disk()` (galaxy_loader.py:246-252) processes ALL stars (foundational + disk-backed) by doing `ref_ids = list(star.pop("_ref_ids", []) or [])`. For foundational stars, `_ref_ids` was already popped — so `ref_ids = []` → **`component_refs` is overwritten with `[]`**. The pre-resolved integers are gone.

3. `sovereign_hot_path._build_stars_from_catalog()` maps `component_refs` → `router_refs`. With `component_refs = []`, all foundational stars land with no routing cross-refs at all. The executor/validator arrays on the VRAM table are empty for every foundational star.

4. No `selection_role` is set on foundational stars → roles are inferred from galaxy_name/layer_id only, with no cross-refs to back them up.

5. No spine router/executor/validator stars exist for MATH or QUESTION families → the kernel can't build a chain even if it finds a math-domain star.

**The result:** The kernel does cosine similarity, applies family_role_bias, picks the top ROLE_ROUTER candidate, then finds `executor_refs = []` → stays at route_depth=1. Generic Language/meaning_layer stars with weak embeddings win because everything has the same empty cross-ref structure.

---

## Three Tasks in Priority Order

---

### Task A (BLOCKING): Fix Foundational Star Ref Wiring

**File: `knowledge3d/knowledgeverse/foundational_galaxy_builder.py`**

**Change 1 — Remove pre-resolution at the end of `build_foundational_galaxy_table()` (lines 450-453):**

```python
# REMOVE THESE THREE LINES entirely:
id_to_index = {str(star["id"]): index for index, star in enumerate(stars)}
for star in stars:
    ref_ids = [str(ref_id) for ref_id in list(star.pop("_ref_ids", []))[:4]]
    star["component_refs"] = [id_to_index[ref_id] for ref_id in ref_ids if ref_id in id_to_index][:4]
```

After removal, `_ref_ids` remains as a string list on each foundational star. `galaxy_loader.load_all_galaxies_from_disk()` will resolve them correctly against the full sovereign table.

**Change 2 — Add `selection_role` and `answer_eligible` to existing foundational stars:**

In `_action_population()`: add `"selection_role": "executor"`, `"answer_eligible": True` to each action star dict.

In `_math_operation_stars()`: add `"selection_role": "executor"`, `"answer_eligible": True` to each op star dict.

In `_spatial_reasoning_stars()`: add `"selection_role": "executor"`, `"answer_eligible": True` to each spatial star dict.

In `_support_action_stars()`: add `"selection_role": "executor"` to each star dict.

In `_drawing_math_stars()`: add `"selection_role": "executor"` to drawing op stars.

**Change 3 — Update `_math_star_embedding()` to not reference `_ref_ids` (line 145-146):**

Lines 145-146 access `star_def.get("_ref_ids")` to set embedding bits. Replace `star_def.get("_ref_ids")` with `star_def.get("_ref_ids") or star_def.get("executor_refs") or star_def.get("validator_refs")` so the embedding signal survives even after the ref field names change.

Same fix in `_spatial_star_embedding()` lines 170-171.

---

**File: `knowledge3d/knowledgeverse/galaxy_loader.py`**

**Change 4 — Preserve role fields when resolving refs (lines 246-252):**

The current ref resolution loop overwrites `component_refs` but doesn't touch `selection_role`, `executor_refs`, `validator_refs`. This is CORRECT behavior for foundational stars that declare these explicitly. Verify the loop does NOT strip these keys.

**Change 5 — `_ref_ids_from_entry()` (lines 145-168): also read executor/validator/router refs as string sources:**

```python
# Add to the list of keys read by _ref_ids_from_entry:
"executor_refs",
"validator_refs",
"router_refs",
```

This ensures disk-backed entries that declare `executor_refs: ["some_id"]` have those IDs included in `_ref_ids` for full-table resolution.

**Change 6 — `_entry_to_star()`: pass through role fields:**

The function currently returns only `{id, _id, _ref_ids, embedding, galaxy_id, star_type, component_refs, flags}`. Add passthrough of:
- `selection_role` (from source or entry)
- `executor_refs` (as string list, from source or entry)
- `validator_refs` (as string list)
- `router_refs` (as string list)
- `answer_eligible` (bool)

These fields must survive into the catalog so `sovereign_hot_path._build_stars_from_catalog()` can read them via `_top_level_or_metadata_list`.

---

### Task B: Add Reasoning Family Spine Stars

**File: `knowledge3d/knowledgeverse/foundational_galaxy_builder.py`**

Add a new function `_reasoning_spine_stars()` and call it from `build_foundational_galaxy_table()` AFTER `_math_operation_stars()` so spine stars can reference op star IDs.

The spine provides the explicit router→executor→validator chain the kernel needs. The CUDA kernel's `choose_best_candidate_device` filters top-K by `desired_role` — without a ROUTER star in top-K with `executor_refs` pointing to an EXECUTOR star, routing never advances past depth=1.

**MATH family spine (4 stars):**

```
id: "math_question_router"
  selection_role: "router"
  answer_eligible: False
  _ref_ids: ["math_compute_executor", "math_word_problem_executor"]
  executor_refs: ["math_compute_executor", "math_word_problem_executor"]
  validator_refs: ["math_answer_validator"]
  galaxy_id: fnv1a32("math")
  star_type: STAR_TYPE_MATH
  flags: STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE
  attractive_prior: 0.4    ← router with executor refs gets strong prior
  embedding: math-broad signal
    dim[0] = 0.0   (neutral between add/mul families)
    dim[1] = 0.5   (mild trigonometric signal)
    dim[2] = 0.95  ← STRONG math domain marker (same as other math stars)
    dim[3] = 0.6   (moderate program depth)
    dim[4] = 0.8   (has refs)
    dim[5] = 0.7   (multiple refs)
    dims 8-31: _hash_tokens_into_embedding("math compute calculate solve number")
```

```
id: "math_compute_executor"
  selection_role: "executor"
  answer_eligible: True
  _ref_ids: ["add_op", "mul_op", "sub_op", "div_op"]
  executor_refs: ["add_op", "mul_op", "sub_op", "div_op"]
  validator_refs: ["math_answer_validator"]
  galaxy_id: fnv1a32("math")
  star_type: STAR_TYPE_MATH
  flags: STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE
  attractive_prior: 0.35
  embedding: arithmetic signal
    dim[2] = 0.95  ← math domain
    dim[0] = 0.7   (add/sum leaning)
    dims 8-31: _hash_tokens_into_embedding("arithmetic add multiply subtract divide compute result")
```

```
id: "math_word_problem_executor"
  selection_role: "executor"
  answer_eligible: True
  _ref_ids: ["add_op", "mul_op", "sub_op"]
  executor_refs: ["add_op", "mul_op", "sub_op"]
  validator_refs: ["math_answer_validator"]
  galaxy_id: fnv1a32("math")
  star_type: STAR_TYPE_MATH
  flags: STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE
  attractive_prior: 0.3
  embedding: word-problem signal
    dim[2] = 0.95  ← math domain
    dim[0] = 0.4   (mixed operations)
    dims 8-31: _hash_tokens_into_embedding("word problem total how many spent gave bought")
```

```
id: "math_answer_validator"
  selection_role: "validator"
  answer_eligible: True
  _ref_ids: []
  galaxy_id: fnv1a32("math")
  star_type: STAR_TYPE_MATH
  flags: STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE
  attractive_prior: 0.25
  embedding: validation signal
    dim[2] = 0.95  ← math domain
    dim[3] = 0.9   (validates programs)
    dims 8-31: _hash_tokens_into_embedding("answer result numeric value equals")
```

**QUESTION family spine (3 stars) for MMLU/LHE:**

```
id: "question_router"
  selection_role: "router"
  answer_eligible: False
  _ref_ids: ["knowledge_lookup_executor"]
  executor_refs: ["knowledge_lookup_executor"]
  validator_refs: ["question_answer_validator"]
  galaxy_id: fnv1a32("reality")   ← knowledge lives in reality galaxy
  star_type: STAR_TYPE_REALITY
  flags: STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE
  attractive_prior: 0.4
  embedding: question signal
    dim[2] = -0.9  ← NOT math domain (reality signal uses -0.9 per spatial embedding convention)
    dim[0] = 0.6   (factual/entity signal)
    dim[1] = 0.5   (question signal)
    dims 8-31: _hash_tokens_into_embedding("what which who when where why how knowledge fact")
```

```
id: "knowledge_lookup_executor"
  selection_role: "executor"
  answer_eligible: True
  _ref_ids: []
  galaxy_id: fnv1a32("reality")
  star_type: STAR_TYPE_REALITY
  flags: STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE
  attractive_prior: 0.35
  embedding:
    dim[2] = -0.9  ← reality domain
    dim[0] = 0.5   (factual)
    dims 8-31: _hash_tokens_into_embedding("answer choice option correct true false knowledge")
```

```
id: "question_answer_validator"
  selection_role: "validator"
  answer_eligible: True
  _ref_ids: []
  galaxy_id: fnv1a32("reality")
  star_type: STAR_TYPE_REALITY
  flags: STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE
  attractive_prior: 0.25
  embedding:
    dim[2] = -0.9
    dim[3] = 0.8   (validates)
    dims 8-31: _hash_tokens_into_embedding("answer correct option choice validate confirm")
```

**Embedding implementation:**

Use `_normalize()` on the raw embedding array. Use `_hash_tokens_into_embedding()` from the existing module (same function used by other stars). The token strings above should produce semantic proximity to actual queries in that family.

**Integration in `build_foundational_galaxy_table()`:**

```python
stars.extend(_action_population(reality_galaxy))
stars.extend(_drawing_math_stars())
stars.extend(_math_operation_stars())
stars.extend(_spatial_reasoning_stars())
stars.extend(_support_action_stars(reality_galaxy))
stars.extend(_reasoning_spine_stars())   # ← add this line LAST
```

Spine stars go last so their `_ref_ids` can resolve backward to the op stars that come before them in the list (for tests that still use local indexing).

---

### Task C: Delete Legacy bind/query-head Path

**File: `knowledge3d/knowledgeverse/knowledgeverse.py`**

**What to delete:**

1. The import at line 47: `from .query_head_substrate import DynamicLodDriverBridge, QueryHeadSubstrate, expand_embedding16_to128`

2. The instance attribute at line 491: `self._query_head_substrate: QueryHeadSubstrate | None = None`

3. The full method `bind_gpu_galaxy_runtime()` (~line 2548) and all its body. This is the CPU-heavy path that builds the legacy QueryHeadSubstrate.

4. The full method `get_query_head_substrate()` (~line 2829) and all its body.

5. Any call to `self.bind_gpu_galaxy_runtime()` (found at lines ~2499, ~3126, ~3993, ~10588). Replace each with a no-op or comment `# sovereign path via _dispatch_sovereign_task`.

6. Any caller of `get_query_head_substrate()` (lines ~12240, ~12745). These are likely in legacy helper query methods. For each:
   - If the method is a dead code path never called from `query()` or `execute_task()`, delete the entire method.
   - If the method IS reachable, replace the substrate call with `self._dispatch_sovereign_task(task)`.

7. The `QueryHeadSubstrate` close/cleanup in `close()` and `rebind()` methods.

**What NOT to delete:**
- `_dispatch_sovereign_task()` (the sovereign entrypoint)
- `query()` and `execute_task()` (these now call `_dispatch_sovereign_task`)
- `sovereign_hot_path` attribute and its initialization
- Anything in `sovereign_hot_path.py`, `galaxy_vram_table.py`, `gpu_task_dispatch.py`

**Verification after deletion:**
```bash
grep -n "bind_gpu_galaxy_runtime\|QueryHeadSubstrate\|get_query_head_substrate" knowledge3d/knowledgeverse/knowledgeverse.py
# Expected: 0 results
```

---

## Validation Protocol

After all changes, in this order:

### Step 1: Unit test foundational builder
```bash
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
stars = build_foundational_galaxy_table()
# Verify _ref_ids survived (not pre-resolved)
math_router = next((s for s in stars if s['id'] == 'math_question_router'), None)
assert math_router is not None, 'spine star missing'
assert '_ref_ids' in math_router, '_ref_ids were popped'
assert 'math_compute_executor' in math_router['_ref_ids'], 'executor ref missing'
assert math_router.get('selection_role') == 'router', 'role missing'
print('foundational builder: OK')
"
```

### Step 2: Unit test galaxy loader
```bash
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.knowledgeverse.galaxy_loader import load_all_galaxies_from_disk
stars = load_all_galaxies_from_disk()
math_router = next((s for s in stars if str(s.get('id','')).strip() == 'math_question_router'), None)
assert math_router is not None
# component_refs should be non-empty (executor IDs resolved to full-table indices)
assert math_router.get('component_refs') or math_router.get('executor_refs'), 'refs not resolved'
print(f'galaxy loader: OK — {len(stars)} stars, spine refs resolved')
"
```

### Step 3: Run existing passing tests
```bash
env PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -m pytest -q \
  tests/test_house_state.py tests/test_trm_game_loop.py \
  tests/test_galaxy_vram_table.py tests/test_galaxy_navigation_quality.py
# All must pass
```

### Step 4: GPU routing smoke test (MATH family)
Write a new test `tests/test_spine_routing.py`:
```python
"""Verify MATH and QUESTION family route_depth > 1 with spine stars."""
import pytest
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
from knowledge3d.knowledgeverse.galaxy_vram_table import GalaxyVRAMTable, ROLE_EXECUTOR, ROLE_ROUTER
from knowledge3d.knowledgeverse.vram_task_buffer import VRAMTaskBuffer
from knowledge3d.knowledgeverse.gpu_task_dispatch import GPUTaskDispatch

def test_math_route_has_executor():
    stars = build_foundational_galaxy_table()
    # Resolve _ref_ids to component_refs locally for test
    id_to_index = {str(s["id"]): i for i, s in enumerate(stars)}
    for star in stars:
        ref_ids = list(star.pop("_ref_ids", []) or [])
        star["executor_refs"] = [id_to_index[r] for r in star.get("executor_refs", ref_ids) if r in id_to_index]
        star["validator_refs"] = [id_to_index[r] for r in star.get("validator_refs", []) if r in id_to_index]
    table = GalaxyVRAMTable(max_stars=len(stars))
    table.load_stars(stars)
    math_router_idx = id_to_index.get("math_question_router", -1)
    assert math_router_idx >= 0
    # Verify executor_refs array is non-empty for the router star
    router_star = stars[math_router_idx]
    assert len(router_star.get("executor_refs") or []) > 0, "math router has no executor refs"
    table.close()

def test_question_router_exists():
    stars = build_foundational_galaxy_table()
    question_router = next((s for s in stars if s.get("id") == "question_router"), None)
    assert question_router is not None
    assert question_router.get("selection_role") == "router"
```

Step 5 (GPU required): Run full structural smoke with MATH query:
```bash
env CUDA_VISIBLE_DEVICES=0 conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium \
  python -m pytest -q tests/test_spine_routing.py tests/test_gpu_task_dispatch.py
```

**Success criteria:**
- All existing tests pass
- `test_spine_routing.py` passes
- In `test_gpu_task_dispatch.py`, any MATH family route test shows `route_depth >= 2` with `executor_star_index != 0xFFFFFFFF`

---

## Architecture Notes

**Why this is the right fix, not a hack:**

The CUDA kernel already has the correct architecture. `choose_best_candidate_device` does: find top-K by cosine sim → filter by `desired_role` → pick best by score+bias. Then follow `executor_refs` array on the winner. This is the sovereign routing path — no Python, no orchestration, pure GPU.

The spine stars ARE the knowledge about HOW to reason, not orchestration code. A star with `selection_role: "router"`, `executor_refs: ["math_compute_executor"]`, and a math-domain embedding IS the routing policy. It lives in the Galaxy (TRM's internal brain). When TRM processes a math query, cosine similarity + family_role_bias selects `math_question_router` as the router, and the kernel follows its `executor_refs` to `math_compute_executor`. This is TRM's cognition, not Python's orchestration.

**Sleep-time learning reinforces this:**
After a successful route (router→executor→validator), the lesson ring records positive traces. `sleep_tick()` calls `lesson_gpu.apply()` which updates `attractive_prior` on the chain stars. Next boot loads the updated table. The spine stars grow stronger with each correct route. This IS the training loop Daniel described: "sleep time compute is done on GPU and saved to the k3d file that's loaded next boot."

**Naming discipline:**
- Spine stars named by meaning: `math_question_router`, not `gsm8k_router`
- Operation stars named by operation: `add_op`, `mul_op`
- No benchmark names inside the Galaxy ever

**What this does NOT fix (deferred):**
- The answer emission is still winner star name (not RPN-computed result). That requires wiring RPN execution into `sovereign_hot_path.run_task()` answer extraction — deferred to E.57.
- The `_gsm8k_*` → `_math_*` rename in knowledgeverse.py — still needed, defer to E.57.

---

## Files to Change

| File | Change |
|------|--------|
| `knowledge3d/knowledgeverse/foundational_galaxy_builder.py` | Remove pre-resolution block, add selection_role+answer_eligible to existing stars, add `_reasoning_spine_stars()` |
| `knowledge3d/knowledgeverse/galaxy_loader.py` | Preserve role fields in `_entry_to_star`, add executor/validator/router keys to `_ref_ids_from_entry` |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Delete bind_gpu_galaxy_runtime + QueryHeadSubstrate + get_query_head_substrate |
| `tests/test_spine_routing.py` | New test: verify spine stars exist + executor refs populated |
| `tests/test_gpu_task_dispatch.py` | Add MATH family route_depth >= 2 assertion |

No other files need changes for this phase.
