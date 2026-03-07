# Claude Architecture Directive: Unified Four-Pass -- ONE System, Not Three

**Date:** March 7, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** ARC moved to 2/10 (good). Math holds 20/20 (solid). But there are now THREE separate four-pass implementations when there should be ONE. K3D is a single always-on system. The four-pass is how K3D THINKS -- not a per-task bolt-on.

---

## Daniel's Correction: One System, Not Three

"The logic we discussed must be all times, we're not building task specific system, we're adding to a single always on system."

This is critical. Right now there are THREE separate four-pass implementations:

1. **NavigatorSpecialist** (`navigator_specialist.py:349-437`)
   - `_forward_reading_path()` -- clause splitting, variable extraction, goal detection
   - `_backward_reading_path()` -- reversed clauses, goal-first dependencies
   - `_fusion_reading_path()` -- merges forward + backward, deduplicates variables
   - **This is the original universal path. It already runs for ALL queries via `plan_routes()`.**

2. **Daemon main.py** (`main.py:466-886`) -- LHE-specific duplicate
   - `_extract_lhe_entities()` -- clause splitting, token extraction, role assignment
   - `_fuse_lhe_entities()` -- merge forward + backward, deduplicate, boost confidence
   - `_build_lhe_goal()` -- goal extraction from backward parse
   - `_query_lhe_evidence()` -- Galaxy queries for evidence
   - `_score_lhe_option()` -- token overlap + contrastive scoring
   - `_synthesize_lhe_open_answer()` -- candidate extraction + scoring

3. **ARC adapter** (`arc_agi_2_adapter.py:716+`) -- ARC-specific duplicate
   - `_discover_patterns_four_pass_compositional()` -- train pair analysis
   - `_arc_forward_entities()` -- grid diff extraction
   - `_arc_backward_entities()` -- output-to-input requirement tracing
   - `_arc_fuse_entities()` -- cross-pair fusion

**This is the template trap applied to architecture.** Instead of one compositional four-pass that handles any input, Codex built three separate four-pass pipelines for three benchmarks. When a fourth benchmark arrives, will there be a fourth copy?

---

## The Correct Architecture

### Passes 1-3: Already Universal in NavigatorSpecialist

`NavigatorSpecialist.plan_routes()` (navigator_specialist.py:69) ALREADY runs forward/backward/fusion for EVERY query. It produces:
- `forward_parse` -- entities extracted left-to-right
- `backward_parse` -- goal and dependencies extracted right-to-left
- `fusion_parse` -- merged, deduplicated, with confidence scores

This is the `_collect_parse_bundle()` method in daemon main.py (line 383). It calls `navigator.plan_routes(use_forward_backward=True)` and returns the parse bundle with forward/backward/fusion data.

**Passes 1-3 are DONE. They run for EVERY query. They do NOT need to be reimplemented per task type.**

### Pass 4: Specialist-Specific, Receives Structured Input

Pass 4 is where domains diverge. But Pass 4 receives the SAME structured input -- the fused entity graph from Pass 3 -- regardless of domain.

```
Pass 4 input (universal):
  {
    "forward_parse":  { entities, variables, clauses },
    "backward_parse": { goal, dependencies, constraints },
    "fusion_parse":   { merged_entities, confirmed_variables, dedup_count },
    "raw_query":      "original text"
  }

Pass 4 output (domain-specific):
  Math     -> RPN program, evaluated result
  ARC      -> Transform composition, applied to test grid
  LHE      -> Evidence-scored answer (option selection or synthesis)
  Chat     -> Response composed from Galaxy knowledge
  Future   -> Same pattern, new specialist
```

### What Must Change

#### 1. Remove LHE four-pass duplication from daemon main.py

The LHE-specific entity extraction (`_extract_lhe_entities`, `_fuse_lhe_entities`) duplicates what NavigatorSpecialist already does. Instead:

```
_dispatch_lhe_task:
  1. parse_bundle = _collect_parse_bundle(...)   <-- already calls NavigatorSpecialist four-pass
  2. route = _augment_lhe_route(...)             <-- add Reality Galaxy (already done)
  3. evidence = _query_lhe_evidence(...)         <-- uses parse_bundle's route_plan
  4. IF multiple_choice:
       score options against evidence + fused entities from parse_bundle
     ELSE:
       synthesize from evidence + fused entities from parse_bundle
```

The `forward_entities`, `backward_entities`, `fused_entities` should come FROM `parse_bundle` (NavigatorSpecialist output), not from a separate LHE-specific extraction. The evidence query and scoring logic (Pass 4) are legitimately LHE-specific -- keep those. But the entity extraction (Passes 1-3) must use the universal path.

#### 2. Unify ARC four-pass input with NavigatorSpecialist

ARC is trickier because the "query" is not text -- it's grid pairs. But the principle holds:

- The ARC adapter's `_arc_forward_entities()` and `_arc_backward_entities()` are Pass 1 and Pass 2 applied to GRIDS instead of TEXT.
- The ARC adapter's `_arc_fuse_entities()` is Pass 3 applied to GRID TRANSFORMS instead of TEXT ENTITIES.

These are domain-specific implementations of the universal passes, not duplicates. The difference from LHE: LHE's passes 1-3 on TEXT are identical to NavigatorSpecialist's passes 1-3 on TEXT (hence duplication). ARC's passes 1-3 on GRIDS are structurally the same but operate on different data types.

**Acceptable architecture for ARC:**
- ARC adapter implements grid-specific passes 1-3 (because grids are not text)
- BUT the output format matches the universal fused entity graph
- Pass 4 (composition verification) receives the same structured input format

**Key requirement:** The ARC four-pass output (`fused_entities`) should use the SAME entity format as NavigatorSpecialist's output. Entity = `{kind, value, role, confidence, sources, mentions}`. For ARC, `kind` is `"transform"` instead of `"token"`, `value` is `"color_remap_3_to_7"` instead of `"apple"`, but the STRUCTURE is identical.

#### 3. Make the parse_bundle carry enough for ALL Pass 4 specialists

Currently `_collect_parse_bundle` returns:
```python
{
    "route_plan": routes,
    "forward_parse": {...},
    "backward_parse": {...},
    "fusion_parse": {...},
}
```

This is correct. All specialists receive this. The specialists then do their domain-specific Pass 4:
- MathSpecialist: extracts semantic entities with role/unit/scope, builds RPN
- LHE dispatch: queries galaxies for evidence, scores options
- ARC adapter: composes transforms from Grammar Galaxy primitives

The parse_bundle is the CONTRACT between universal passes 1-3 and domain-specific pass 4.

---

## Concrete Refactoring Steps

### Step 1: Remove LHE entity duplication (IMMEDIATE)

Delete from daemon main.py:
- `_extract_lhe_entities()` (line 466)
- `_fuse_lhe_entities()` (line 527)

Replace in `_solve_lhe_structured()`:
```python
# BEFORE (duplicate four-pass):
forward_entities = self._extract_lhe_entities(prompt, source_pass="forward", options=options)
backward_entities = self._extract_lhe_entities(prompt, source_pass="backward", options=options)
fused_entities = self._fuse_lhe_entities(forward_entities, backward_entities)

# AFTER (use universal parse_bundle):
forward_parse = parse_bundle.get("forward_parse", {})
backward_parse = parse_bundle.get("backward_parse", {})
fusion_parse = parse_bundle.get("fusion_parse", {})
fused_entities = self._convert_parse_to_lhe_entities(fusion_parse, options=options)
```

Write a thin `_convert_parse_to_lhe_entities()` that extracts tokens/phrases from the NavigatorSpecialist's fusion output and adds option tokens. This is NOT a reimplementation of passes 1-3 -- it's a format adapter from the universal output to what the LHE scoring needs.

### Step 2: Normalize ARC entity format (NEXT)

Make `_arc_fuse_entities()` output entities in the same format as NavigatorSpecialist:
```python
{
    "kind": "transform",        # or "token" for text
    "value": "color_remap",     # or "apple" for text
    "role": "confirmed",        # same field, domain-specific values
    "confidence": 0.85,         # same scoring
    "sources": ["forward", "backward"],  # same dual-source
    "mentions": 2,              # same counting
    "metadata": { ... }         # domain-specific details
}
```

This doesn't change ARC behavior -- it normalizes the output format so all four-pass outputs look the same.

### Step 3: Parse bundle as the universal contract (ALREADY DONE)

`_collect_parse_bundle()` already calls NavigatorSpecialist for all tasks. All specialists already receive it. No change needed -- just ensure LHE and ARC actually USE it instead of reimplementing passes 1-3.

---

## What About ARC Primitives?

Codex's plan to expand ARC primitives from audited failures is CORRECT and should continue. The unification work (steps 1-2 above) is structural -- it doesn't block ARC primitive expansion.

Do both in parallel:
- Expand ARC Grammar Galaxy primitives from the 8 failed tasks
- Unify the four-pass so it's one system, not three

The primitive expansion is Pass 4 work (what transforms to compose). The unification is Passes 1-3 work (how entities are extracted). They're independent.

---

## Current Benchmark State

| Benchmark | Score | Status |
|-----------|-------|--------|
| Math | 20/20 | SOLID -- do not touch |
| ARC | 2/10 | Improving -- `arc_four_pass` source, `composition_depth: 1.2` avg |
| LHE | 0/10 | Structurally wired, knowledge-limited, synthesis quality weak |

ARC `generation_failure_rate: 0.8` (was 1.0, then 0.9, now 0.8). Trend is correct.

---

## Priority Order

```
1. Remove LHE four-pass duplication (use parse_bundle from NavigatorSpecialist)
2. Continue ARC primitive expansion from failed tasks (Grammar Galaxy entries)
3. Normalize ARC entity format to match universal format
4. Resume augmentation (knowledge density for LHE)
5. Rerun full smoke for auditable delta
```

---

## Grounding

| Spec | Section | Relevance |
|------|---------|-----------|
| KNOWLEDGEVERSE_SPECIFICATION.md | Region 5 (TRM) | TRM navigates -- ONE navigation logic, not per-task |
| THREE_BRAIN_SYSTEM_SPECIFICATION.md | Cranium | ONE execution pipeline, specialists are adapters |
| DUAL_CLIENT_CONTRACT_SPECIFICATION.md | Section 1.6 | Save Information Principle -- don't duplicate the four-pass |
| TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md | Section 3 | "Generalize to ALL chains" -- not "build per chain" |

---

## Daniel's Principle

K3D is always on. The four-pass is how K3D thinks. Not how K3D solves math. Not how K3D solves ARC. Not how K3D answers LHE. It's how K3D processes ANY input, ALL the time.

A task arrives. K3D reads it forward. Reads it backward. Fuses. Then dispatches to the right specialist for domain-specific execution. ONE system. ONE path. Specialists are the ONLY place where domain logic belongs -- not in the decomposition.

The decomposition is universal. The execution is specialized. Don't duplicate the decomposition.
