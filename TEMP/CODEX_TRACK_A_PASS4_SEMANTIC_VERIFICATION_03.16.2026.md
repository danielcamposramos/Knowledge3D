# Codex Directive: Track A — Pass 4 Semantic Verification

**Date:** 2026-03-16
**From:** Claude (Architecture)
**To:** Codex (Implementation)
**Priority:** Immediate — this is the highest-value improvement available
**Baseline:** ARC 10/10, Math 20/20, GSM8K 2/10, LHE 6/10, MMLU 15/50
**Checkpoint:** `/tmp/k3d_nsi_full_guard/checkpoints` (NSI-safe root)

---

## Context

The four-pass decomposition already runs for EVERY query via NavigatorSpecialist:
- **Pass 1** (Forward): `_forward_reading_path()` → clause-by-clause entities (line 405)
- **Pass 2** (Backward): `_backward_reading_path()` → goal-first dependencies (line 421)
- **Pass 3** (Fusion): `_fusion_reading_path()` → deduplicated merged entities (line 453)
- **Pass 4**: NOT IMPLEMENTED — this is what you're building

The `parse_bundle` carrying forward/backward/fusion data is already collected at query time (knowledgeverse.py line 10179) and threaded through `_build_gpu_reasoning_paths()` (line 10197).

The `compositional_consistency` field already feeds into the scoring expression at weight 0.12 gated by gsm8k_mode (line 9928). The AtomicFissionFusion bridge is already called at line 5924.

**What's missing:** The entity extraction is FLAT (numbers stripped from text with no semantic binding). Pass 4 should replace this with structured entities bound to role/unit/scope, enabling dimensional analysis.

---

## IMPORTANT CONSTRAINTS

1. **Benchmark safety first.** If any change drops ARC/Math/LHE/MMLU, narrow or revert immediately. GSM8K improvement must NOT regress others.
2. **Do NOT rewrite Passes 1-3.** They're live and working. Only add Pass 4 logic.
3. **Do NOT add new scoring terms.** Use the existing `compositional_consistency` field (already weighted at 0.12 in the scoring expression). Improve the QUALITY of what feeds into it.
4. **Checkpoint source:** Use `/tmp/k3d_nsi_full_guard/checkpoints` for baseline. Older checkpoints have drifted.
5. **Track B was reverted.** Pre-scoring crystallization regressed benchmarks. Do NOT move graph crystallizer. It stays post-hoc.

---

## What To Build

### Step 1: Structured Semantic Entity Extraction

**File:** `knowledge3d/knowledgeverse/navigator_specialist.py`

Enhance `_fusion_reading_path()` output to include structured semantic entities. Currently the fusion merge deduplicates variables by composite key `(value, surface, raw_block, offset)`. Add semantic role detection to each merged entity.

Add a helper method `_annotate_semantic_roles(merged_quantities)` that enriches each quantity with:

```python
{
    "value": 3,
    "surface": "3",          # existing
    "role": "frequency",     # NEW: count|rate|frequency|price|consume|result|goal
    "unit": "times",         # NEW: extracted from surrounding words
    "scope": "per_week",     # NEW: temporal/spatial qualifier
    "reference": None,       # NEW: "half"|"twice"|"double" if back-reference
}
```

**Role detection keywords** (simple, no regex in hot path — this is parsing, not reasoning):
- Noun after number → `role=count`, `unit=noun`
- "per Y" / "each Y" / "every Y" after number → `role=rate`, `scope=per_Y`
- "times" / "times a Y" → `role=frequency`, `scope=per_Y`
- "$" prefix / "dollars"/"cents" → `role=price`, `unit=currency`
- "remaining"/"left"/"after" → `role=result`
- "total"/"altogether"/"in all" → `role=goal`
- "half"/"twice"/"double"/"triple" + "that much"/"as many" → `reference=multiplier`

Thread the annotated entities into `fusion_parse["semantic_entities"]` so they flow through the existing `parse_bundle`.

### Step 2: Improve Compositional Atom Rows for GSM8K

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

The method `_candidate_compositional_atom_rows()` (used by `_apply_atomic_compositional_consistency` at line 5940) extracts atom embeddings for the fission/fusion kernel. Currently it pulls from candidate match embeddings.

Enhance it to ALSO consider `parse_bundle["fusion_parse"]["semantic_entities"]` when available:
- If semantic entities have resolved references (e.g., "half that much" → value=1), use the resolved value's embedding as an atom
- If entities have unit-scope annotations, group atoms by dimensional compatibility (same-unit atoms should be composed together before cross-unit composition)

This makes the `AtomicFissionFusion.decompose()` call more meaningful — it checks whether the candidate's embedding can be reconstructed from the SEMANTICALLY STRUCTURED atoms, not just raw number embeddings.

### Step 3: GSM8K Reference Expression Resolution

**File:** `knowledge3d/knowledgeverse/navigator_specialist.py`

In `_fusion_reading_path()`, after merging forward+backward entities, add reference resolution:

```python
# Resolve reference expressions
for entity in merged_quantities:
    ref = entity.get("reference")
    if ref and ref in ("half", "twice", "double", "triple"):
        # Find nearest previous entity with matching unit
        referent = _find_nearest_referent(entity, merged_quantities)
        if referent:
            multiplier = {"half": 0.5, "twice": 2.0, "double": 2.0, "triple": 3.0}[ref]
            entity["resolved_value"] = referent["value"] * multiplier
            entity["reference_source"] = referent
```

This is critical for GSM8K problems like "A robe takes 2 bolts of blue fiber and half that much white fiber" — "half" is NOT a standalone number, it's a reference expression that resolves to 1.

### Step 4: Dimensional Consistency Signal

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

In `_apply_atomic_compositional_consistency()` (line 5915), add a secondary check when semantic entities are available:

```python
# After the existing fission/fusion consistency check:
semantic_entities = parse_bundle.get("fusion_parse", {}).get("semantic_entities", [])
if semantic_entities and goal_entity:
    dimensional_ok = _check_dimensional_consistency(semantic_entities, goal_entity)
    if dimensional_ok:
        candidate["compositional_consistency"] *= 1.3  # boost consistent compositions
    elif dimensional_ok is False:  # explicitly inconsistent (not just unknown)
        candidate["compositional_consistency"] *= 0.5  # penalize
```

`_check_dimensional_consistency()` is a simple unit-chain check:
- Extract units from each entity's scope (e.g., "meters/sprint", "sprints/week")
- Check if chaining (multiplying) all entities' units cancels to the goal unit
- Return True (consistent), False (inconsistent), or None (can't determine)

This does NOT need to be on GPU — it's a simple string/unit check that runs on the few GSM8K candidates (typically 3-9). It enhances the SIGNAL that feeds into the existing sovereign scoring expression.

---

## What NOT To Do

- Do NOT add new scoring expression terms. Use existing `compositional_consistency` field.
- Do NOT move graph crystallizer (Track B was reverted).
- Do NOT modify Passes 1-3 core logic. Only ADD annotation/enrichment.
- Do NOT add external dependencies. String matching for role detection is fine.
- Do NOT break MMLU/LHE. If semantic entity annotation somehow affects non-GSM8K paths, gate it behind `gsm8k_mode > 0` or `goal_type_family == "gsm8k"`.

---

## Validation

1. `python3 -m compileall knowledge3d/` must pass
2. `pytest -q tests/test_trm_weight_persistence.py` — existing 13 NSI tests pass
3. Full benchmark from NSI-safe checkpoint:
   - ARC 10/10 (MUST hold)
   - Math 20/20 (MUST hold)
   - GSM8K: target 4+/10 (from 2/10)
   - LHE 6/10 (MUST hold)
   - MMLU 15/50 (MUST hold)
4. Add tests for semantic entity extraction (at least: "3 sprints 3 times a week" produces two entities with correct roles)
5. Add test for reference resolution ("half that much" resolves to 0.5× referent)

---

## Worked Example: Why This Fixes GSM8K

**GSM8K 3 (currently FAILING):** "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?"

**Current:** Numbers extracted as flat [3, 3, 60]. `_build_word_problem_rpn` uses `word_problem_rate` path → takes 3 × 60 = 180. Wrong.

**With Pass 4:**
```
Semantic entities after fusion:
  E1: {value:3, role:count, unit:sprints, scope:per_session}
  E2: {value:3, role:frequency, unit:sessions, scope:per_week}
  E3: {value:60, role:rate, unit:meters, scope:per_sprint}
  GOAL: {role:goal, unit:meters, scope:per_week}

Dimensional check:
  meters/sprint × sprints/session × sessions/week = meters/week ✓
  → E3 × E1 × E2 = 60 × 3 × 3 = 540

AtomicFissionFusion.decompose():
  compound = embedding(540_meters_per_week)
  atoms = [embedding(3_sprints), embedding(3_sessions), embedding(60_meters)]
  consistency = high (compound reconstructible from atoms)
  → compositional_consistency boosted by 1.3× (dimensional check passed)
```

The candidate with the correct composition (540) gets a higher `compositional_consistency` score than the incorrect one (180), because its atoms are dimensionally consistent with the goal.

---

## Files Summary

| File | Change | Risk |
|------|--------|------|
| `navigator_specialist.py` | `_annotate_semantic_roles()`, reference resolution in fusion | Low — additive only |
| `knowledgeverse.py` | Enhanced `_candidate_compositional_atom_rows()`, dimensional consistency check | Medium — touches scoring signal |
| `tests/test_trm_weight_persistence.py` | Add semantic entity + reference resolution tests | None |
