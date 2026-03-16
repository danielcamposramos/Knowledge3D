# Codex Directive: MMLU Shared-Session Navigation Fix

**Date:** 2026-03-16
**From:** Claude (Architecture)
**To:** Codex (Implementation)
**Priority:** Immediate — this is the last benchmark regression
**Baseline:** ARC 10/10, Math 20/20, GSM8K 10/10, LHE 7/10, MMLU 16/50 (isolated), 12-13/50 (shared session)
**Checkpoint:** fresh guard root

---

## The Problem

MMLU scores 16/50 when run in isolation but drops to 12-13/50 in the shared-session full guard (ARC + Math + GSM8K + LHE + MMLU on one Knowledgeverse instance).

**Root cause:** `reset_query_session()` (line 915) clears reasoning programs, query sequence, LED pathfinder, and RPN engine instances — but it does NOT clear the **navigation candidate cache** or **CSR graph neighborhood state**. After ARC/Math/GSM8K/LHE run 40+ queries, the LED-A* pathfinder and semantic CSR graph neighborhood have warmed toward those domains. When MMLU starts, the Morton locate + LED-A* navigation is seeded from a graph that has been shaped by prior benchmarks.

The subject-hint seeding (line 8162-8191) adds a `0.22` bias for subject-matched entries, but this isn't enough to overcome the accumulated neighborhood bias from prior benchmarks. The CSR graph's local structure (k=12 neighbors, max 2048 nodes) has been traversed heavily in math/ARC neighborhoods, making those paths "warm" while MMLU-relevant neighborhoods remain "cold."

**Evidence:**
- `gpu_bind_rebuilds=1` in full guard — the catalog is built once and shared
- Isolated MMLU: 16/50 (catalog starts cold, no bias)
- Shared MMLU: 12-13/50 (catalog warm from 40+ prior queries)
- The delta is exactly the ~3-4 questions that were identified as MMLU variance noise — but it's NOT noise, it's systematic shared-session bias

---

## What To Build

### Fix 1: Clear Navigation Neighborhood State on Task-Type Switch

In `reset_query_session()` (knowledgeverse.py line 915), also clear the navigation state that accumulates during queries:

```python
def reset_query_session(self) -> None:
    """Clear mutable per-benchmark state while keeping the GPU-bound galaxy snapshot assembled."""
    self._gpu_reasoning_programs.clear()
    self._query_sequence = 0
    self._led_pathfinder = None
    # NEW: clear accumulated navigation neighborhood state
    if self._semantic_csr_graph is not None and hasattr(self._semantic_csr_graph, "reset_traversal_state"):
        self._semantic_csr_graph.reset_traversal_state()
    # ... rest of existing reset
```

If `SemanticCSRGraph` doesn't have traversal state that accumulates, the bias may come from the LED pathfinder's internal caches. Check `led_pathfinder.py` for any cached paths or visited-node sets that persist across queries.

### Fix 2: Boost Subject-Hint Seed Weight for MMLU

The current subject-hint bias is `0.22` (line 8173). For MMLU specifically, this is too low to overcome the stale neighborhood bias. Two approaches:

**Option A (targeted):** When `task_type == "MMLU_TASK"`, increase the subject-hint bias from `0.22` to `0.35`. This is the simplest change and recovers the isolated-run score by making subject-matched entries more competitive against the neighborhood-warmed math/ARC entries.

**Option B (structural):** Before the similarity sort (line 8169), if `task_type == "MMLU_TASK"` and `domain_hint` is set, inject the top-K subject-matched Reality Galaxy entries directly into the candidate list, bypassing the Morton locate entirely for those entries. This ensures MMLU always has its domain-specific anchors regardless of what the CSR graph neighborhood looks like:

```python
if task_type == "MMLU_TASK" and str(domain_hint or "").strip():
    # Inject subject-matched entries as priority seeds
    for idx, entry in enumerate(catalog):
        if self._subject_anchor_match_score(entry, subject_hint=str(domain_hint), match_mode="mmlu") > 0.0:
            if idx not in candidate_index_set:
                candidate_index_list.append(idx)
                candidate_index_set.add(idx)
    # Re-compute embeddings for the expanded list
    candidate_embeddings = [
        list(catalog[int(index)].get("embedding16", []))
        for index in candidate_index_list
    ]
    candidate_similarities = self._embedding_similarities(query_embedding, candidate_embeddings)
```

**Recommendation:** Start with Option A (bump seed weight to 0.35). If that recovers 15-16/50 in shared session, stop. If not, add Option B.

### Fix 3: Verify Benchmark Ordering Doesn't Matter

After fixes 1+2, run the full guard twice:
1. Default order: ARC → Math → GSM8K → LHE → MMLU
2. MMLU first: MMLU → ARC → Math → GSM8K → LHE

If both produce the same MMLU score (±1), the shared-session bias is fixed. If MMLU-first scores higher, there's still residual state leaking.

---

## What NOT To Do

- Do NOT rebuild the GPU catalog between benchmarks — that's expensive and unnecessary. The catalog is correct; the navigation state is the problem.
- Do NOT add a separate Knowledgeverse instance per benchmark — that defeats the shared-session architecture (Phase C daemon will run everything on one instance).
- Do NOT increase the subject-hint bias above `0.40` — that risks making MMLU routing ignore semantic similarity entirely, which would break questions where the subject hint is wrong or ambiguous.
- Do NOT change the CSR graph structure (k=12, max 2048) — that's a global parameter affecting all benchmarks.

---

## Important Constraints

1. **Benchmark safety.** ARC 10/10, Math 20/20, GSM8K 10/10, LHE 7/10 MUST hold.
2. **MMLU target:** 15+/50 in shared session (from 12-13/50). Isolated should stay at 16/50 or improve.
3. **No new scoring terms.** This is a navigation/seeding fix, not a scoring change.
4. **Run from fresh guard root** to measure clean shared-session delta.

---

## Validation

1. `python3 -m compileall knowledge3d/` must pass
2. `pytest -q tests/test_trm_weight_persistence.py` — existing tests pass
3. Full shared-session guard:
   - ARC 10/10 (MUST hold)
   - Math 20/20 (MUST hold)
   - GSM8K 10/10 (MUST hold)
   - LHE 7/10 (MUST hold)
   - MMLU 15+/50 (target, from 12-13/50 shared session)
4. Isolated MMLU: 16+/50 (MUST hold or improve)

---

## Files Summary

| File | Change | Risk |
|------|--------|------|
| `knowledgeverse.py` | Clear navigation state in `reset_query_session()`, boost MMLU seed weight | Low — additive to existing reset |
| `semantic_csr_graph.py` | Add `reset_traversal_state()` if needed | Low — no structural change |
| `led_pathfinder.py` | Check for cached paths that persist across resets | Investigation |
