# Codex — Phase E.19: Restore GPU Pipeline + ARC3 answer_index

**Date:** 2026-03-29
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation)
**Priority:** CRITICAL — Daniel: GPU at 100% before ARC3; now it's back to Python

---

## The Two Problems

### Problem 1: GPU path is gated behind unset env vars

The Knowledgeverse has a full GPU-accelerated pipeline — Morton/LOD navigation, LED-A*, device-side composed head — but it is gated behind:

```python
K3D_DEVICE_PIPELINE=0   # default → uses _compose_head_navigation_candidates() (CPU)
K3D_TRM_SHADOW=0         # default → TRM shadow probe skipped
K3D_TRM_NAVIGATE=0       # default → TRM navigation skipped
```

These are OFF by default. The benchmark was hitting 100% GPU before ARC3 because those flags were set. When Codex rewrote the runners, they were not carried forward.

**The fix is a single change to every runner that invokes Knowledgeverse:**

Every `conda run` command and every `Knowledgeverse()` init must be preceded by:
```bash
export K3D_DEVICE_PIPELINE=1
export K3D_TRM_SHADOW=1
export K3D_TRM_NAVIGATE=1
export CUDA_VISIBLE_DEVICES=0
```

### Where to set these

In `scripts/run_full_benchmark.py`, `scripts/run_arc3_agent.py`, `scripts/run_arc3_session.py`: at the top of `main()` or before `Knowledgeverse()` construction, call:

```python
import os
os.environ.setdefault("K3D_DEVICE_PIPELINE", "1")
os.environ.setdefault("K3D_TRM_SHADOW", "1")
os.environ.setdefault("K3D_TRM_NAVIGATE", "1")
```

`setdefault` means: set to 1 if not already set — so the user can still override with `K3D_DEVICE_PIPELINE=0` if needed, but the scripts default to GPU.

**Verification:** After setting these, run the benchmark and check that:
```bash
grep gpu_execution /K3D/Knowledge3D.local/logs/phase_e_*/mmlu.jsonl | head -5
```
shows `"gpu_execution": true` for ALL queries — not just the current mixed state.

---

### Problem 2: `_answer_arc_query` cannot return `answer_index` from navigation rules

The ARC3 knowledge is now in Galaxy (E.18). When Knowledgeverse runs an `ARC_TASK` query against a live game frame, it navigates Galaxy, finds `arc3_nav_move_down` or `arc3_rule_keyboard_game`. The match has `metadata.action_index = 1`.

BUT `_answer_arc_query()` (line 4026-4139) ONLY returns a result when `output_grid` is populated from `arc_primitive_plan` or `arc_transform_chain`. It has NO path to return `answer_index` from matched metadata. So it always returns `gpu_arc_no_output_grid`.

**Fix: add an `answer_index` extraction path to `_answer_arc_query()`.**

In `knowledgeverse.py`, in `_answer_arc_query()`, immediately after reading `output_grid`, `primitive_plan`, and `transform_chain` — add:

```python
# Check for ARC3 interactive navigation rule (answer_index in metadata)
metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
action_index_raw = (
    metadata.get("action_index")
    or match.get("action_index")
)
if action_index_raw is not None:
    try:
        answer_index = max(0, int(action_index_raw))
    except (TypeError, ValueError):
        answer_index = None
    if answer_index is not None and not primitive_plan and not transform_chain:
        thinking_trace = self._build_gpu_thinking_trace(
            binding=binding,
            program_id=str(reasoning_program.get("id", "")),
            match=match,
            similarity=similarity,
            specialist=specialist,
            read_field="answer_index",
            extra_steps=list(selection_steps),
        )
        return {
            "status": "ok",
            "answer_index": answer_index,
            "answer": str(answer_index),
            "response": str(answer_index),
            "result": answer_index,
            "thinking_trace": thinking_trace,
            "reasoning_trace": list(thinking_trace),
            "gpu_execution": True,
            "runtime": "knowledgeverse_gpu_query",
            "program_id": str(reasoning_program.get("id", "")),
            "program_type": "gpu_arc3_navigation_rule",
            "solver": "knowledgeverse_gpu_query",
            "patterns_used": 1,
            "query_text": query_text,
            "top_match_similarity": similarity,
            "route": {
                "specialist": specialist,
                "domain_hint": domain_hint,
                "galaxy_names": list(route_galaxies or binding.get("galaxies", [])),
            },
            "match": dict(match),
        }
```

Insert this block BEFORE the `if not isinstance(output_grid, list)` check at line 4102.

**Where to insert:** After line 4055 (after `color_mapping` is built), before line 4056 (`if primitive_plan:`). The full sequence becomes:
1. Read `output_grid`, `primitive_plan`, `transform_chain`, `color_mapping`
2. **NEW: Check for `answer_index` in metadata → return early if found and no transform plan**
3. If `primitive_plan` → execute GPU primitive plan
4. If `transform_chain` → execute GPU transform
5. If `output_grid` still None → return `gpu_arc_no_output_grid`

---

### Problem 3: Remove Python fallback from `arc_agi_3.py`

After Problem 2 is fixed, `_answer_arc_query()` returns `answer_index` for ARC3 navigation rule matches. The adapter `_derive_action_from_result()` picks it up at Priority 1 (line 111-114).

Delete `_derive_action_from_frames()` from `arc_agi_3.py`. Delete `_detect_game_type()`. Delete `_direction_preferences()`. Delete `_select_preferred_action()`. Delete `_grid_background_value()`, `_grid_value_counts()`, `_grid_centroid()`, `_find_primary_cell()`.

The adapter becomes:

```python
def _derive_action_from_result(
    frame: list[list[int]],
    result: dict[str, Any],
    *,
    goal_frame: list[list[int]] | None = None,
) -> tuple[int, dict[str, int]]:
    # Priority 1: Knowledgeverse returns answer_index (ARC3 navigation rule)
    raw = result.get("answer_index")
    if isinstance(raw, (int, float)):
        return max(0, min(int(raw), len(ACTION_NAMES) - 1)), {}

    # Priority 2: click coordinates
    if isinstance(result.get("x"), (int, float)) and isinstance(result.get("y"), (int, float)):
        return 5, {"x": int(result["x"]), "y": int(result["y"])}

    # Priority 3: output_grid geometric delta
    predicted = _normalize_grid(result.get("output_grid"))
    if predicted != [[]]:
        # ... existing grid-delta logic ...

    # No answer — Knowledgeverse has no matching knowledge yet.
    # Return 0 (neutral) and let sleep-time learn.
    return 0, {}
```

**No centroid. No game type detection. No Python geometry policy.**
When the Knowledgeverse finds no matching rule, `answer_index=0` (Move Up / neutral) and the outcome signal from `learn_from_outcome()` is −1. Sleep-time strengthens the rules that DID work and weakens the ones that didn't. Over many games the ARC3 navigation rules will evolve toward correct behavior.

If `answer_index=0` forever — that means the Galaxy navigation is not finding the ARC3 rules we ingested. That is a Galaxy navigation problem (wrong embeddings, wrong Galaxy in route), NOT a Python geometry problem.

---

### Problem 4: ARC3 task route must include Grammar, Reality, Tool galaxies

Currently `choose_action()` sends:
```python
route={"specialist": "visual", "domain_hint": "arc3_interactive",
       "galaxy_names": list(Knowledgeverse.GPU_ARC_TARGET_GALAXIES)}
```

`GPU_ARC_TARGET_GALAXIES = ("Language", "Drawing", "Grammar", "Tool")` — Grammar and Tool ARE included. Good. But Reality is not. The ARC3 meta-rules we ingested are in Reality. Add it:

```python
route={
    "specialist": "visual",
    "domain_hint": "arc3_interactive",
    "galaxy_names": list(Knowledgeverse.GPU_ARC_TARGET_GALAXIES) + ["Reality"],
}
```

---

## Execution Sequence

1. Add `os.environ.setdefault("K3D_DEVICE_PIPELINE", "1")` etc. to all three runner scripts
2. Add `answer_index` extraction block to `_answer_arc_query()` in `knowledgeverse.py`
3. Add "Reality" to the ARC3 task route in `arc_agi_3.py`
4. Delete all Python geometry policy from `arc_agi_3.py` (see above)
5. Run: `python3 -m py_compile benchmarks/arc_agi_3.py knowledge3d/knowledgeverse/knowledgeverse.py`
6. Run GPU benchmark: verify `gpu_execution=true` on 100% of rows, GPU util visible in `nvidia-smi`
7. Run live ARC3 session: verify `answer_index` appears in result (not `gpu_arc_no_output_grid`)

---

## What Drives Learning After This Fix

With the env vars set and `answer_index` returning from ARC3 navigation rules:

1. Live game step: frame → `_frame_to_query_text()` → "move colored cell down south navigate spatial grid"
2. Knowledgeverse GPU path navigates Grammar+Reality+Tool Galaxy
3. Finds `arc3_nav_move_down` (similarity ~0.7 with "down south" in embedding)
4. Returns `answer_index=1` (Move Down)
5. Agent sends ACTION2 to API
6. `levels_completed` increases → `learn_from_outcome()` returns `outcome=1`
7. Sleep-time: `jarvis_sleep_consolidation()` strengthens the path to `arc3_nav_move_down`
8. Next game: Galaxy finds `arc3_nav_move_down` faster, with higher confidence

This is the architecture. Knowledge in Galaxy. GPU navigates. Sleep-time learns. No Python policy.

---

## Files to Modify

| File | Change |
|------|--------|
| `scripts/run_full_benchmark.py` | Add env defaults at top of `main()` |
| `scripts/run_arc3_agent.py` | Add env defaults at top of `run_live_arc3()` |
| `scripts/run_arc3_session.py` | Add env defaults at top of `run_arc3_session()` |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Add `answer_index` path in `_answer_arc_query()` |
| `benchmarks/arc_agi_3.py` | Add Reality to route; delete Python geometry policy |

## Files NOT to Touch

| File | Why |
|------|-----|
| `knowledge3d/knowledgeverse/arc3_knowledge_builder.py` | Already correct |
| All other benchmark files | Already correct |
| All test files | Update tests to reflect removal of Python policy helpers |
