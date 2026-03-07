# Codex Directive: Universal Four-Pass Decomposition

**Date:** March 7, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** Math is now 20/20 on the smoke pack. The four-pass composition logic proved itself. Now generalize it across ALL benchmark paths -- ARC, LHE, and any future task -- as the standard problem decomposition strategy.

---

## Assessment of Current State

### Math: 20/20 -- SOLID

The four-pass word problem composition (forward / backward / fusion / semantic verification) works. This is now the reference implementation. Do not touch it.

### ARC: 0/10 -- Generator Coverage Gap (NOT routing)

From the smoke artifact:
- `generated_pattern_total = 100` -- generation IS running
- `oracle_at_all = 0.0` and `generation_failure_rate = 1.0` -- no correct candidate was generated
- `composition_depth: {"avg": 1.0, "distribution": {"1": 10}}` -- depth=1 means single transforms only
- Task 00576224: predicted block-scaled, expected phase-shifted tiling

Codex's diagnosis is correct: the ARC problem is **generator family coverage**, not ranking. But the proposed fix (adding more generator families one by one) is the same trap we fell into with math templates -- problem-specific instead of compositional.

### LHE: 0/10 -- No Reasoning Path

The LHE path currently routes through `_answer_question_via_tablet` which submits the question but has no decomposition, no multi-step reasoning, no domain-specific grammar navigation. It's a passthrough.

---

## Daniel's Direction

The four-pass decomposition is not a math technique. It is the STANDARD way K3D decomposes ANY problem into small steps. This is how humans solve hard problems across every domain:

1. Read forward -- what is given?
2. Read backward -- what is asked?
3. Merge -- what do we actually know (deduplicated)?
4. Verify -- translate to formal structure, check consistency, execute

This maps directly to existing K3D architecture:
- **TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md section 3** (line 591): "Generalize Week 18-19 pattern (math forward/backward reading) to ALL chains"
- **FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md section 3.4** (line 423): `TASK DECOMPOSE` is already an RPN opcode
- **NavigatorSpecialist** already has `_forward_reading_path`, `_backward_reading_path`, `_fusion_reading_path` for ALL queries, not just math

The infrastructure exists. It just isn't wired to ARC or LHE.

---

## Universal Four-Pass Architecture

### The Four Passes (Domain-Agnostic)

```
Pass 1: FORWARD READING
  Input:  raw query/task
  Output: ordered list of SEMANTIC ENTITIES with roles
  Method: parse left-to-right, extract entities from each clause
  Already exists: NavigatorSpecialist._forward_reading_path()

Pass 2: BACKWARD READING
  Input:  raw query/task
  Output: dependency chain from GOAL back to GIVENS
  Method: identify goal first, trace what it needs
  Already exists: NavigatorSpecialist._backward_reading_path()

Pass 3: FUSION (Deduplicate + Merge)
  Input:  forward entities + backward dependencies
  Output: fused entity graph with confirmed entities (higher confidence)
  Method: merge, deduplicate, resolve references
  Already exists: NavigatorSpecialist._fusion_reading_path()

Pass 4: SEMANTIC VERIFICATION + EXECUTION
  Input:  fused entity graph
  Output: formal plan (RPN for math, transform sequence for ARC, reasoning chain for LHE)
  Method: map entities to domain operations, verify consistency, execute
  Domain-specific -- this is where math/ARC/LHE diverge
```

Passes 1-3 are IDENTICAL across all domains. Pass 4 is domain-specific but follows the same pattern: map entities to operations, verify, execute.

### How It Applies to Each Benchmark

#### Math (Already Working)

```
Pass 4 (Math):
  Entities -> numbers with roles (count, rate, frequency)
  Operations -> arithmetic from Grammar Galaxy (ADD, SUB, MUL, DIV)
  Verification -> dimensional analysis (units cancel to goal)
  Execution -> RPN program on PTX stack
```

#### ARC (New Application)

ARC tasks have train examples (input/output grid pairs) and a test input. The "query" is implicit: "what transformation maps input to output?"

```
Pass 1 (Forward -- from train examples):
  For each train pair, parse input -> output left-to-right:
  Clause = each observable difference between input and output grid
  Entities:
    E1{type: "color_change", from: 3, to: 7, region: "row_2"}
    E2{type: "shape_move", object: "block_A", direction: "right", distance: 2}
    E3{type: "pattern_tile", source: "top_left_3x3", target: "full_grid"}

Pass 2 (Backward -- from output to input):
  Goal: output grid
  What operations on input produce this output?
  Trace backward:
    Output has tiling -> needs a source pattern + tile rule
    Source pattern is in input top-left corner
    Tiling rule is phase-shifted (offset per row)

Pass 3 (Fusion):
  Merge forward and backward observations across ALL train pairs.
  Entities confirmed across multiple train pairs get higher confidence.
  Entities seen in only one pair are uncertain (ternary 0 -- explore).
  Deduplicate: "color_change from 3 to 7" seen in pair 1 and pair 2 -> confirmed.

Pass 4 (ARC-specific):
  Entities -> transform primitives from Grammar Galaxy (ROTATE_90, MIRROR_H, TILE, COLOR_MAP)
  Operations -> composition of transforms (Grammar Galaxy rules)
  Verification -> apply composed transform to train inputs, check against train outputs
  If verification passes for ALL train pairs -> apply to test input
  If verification fails -> contrastive -1, try another composition
```

**The critical insight for ARC:** Instead of hardcoding generator families (phase-shift tiling, rotation-plus-layout, etc.), the four-pass decomposition DISCOVERS the transform by observing the train pairs. The Grammar Galaxy already has geometric transforms (rotate, mirror, transpose from `foundational_operations_bootstrap.py`). The TRM composes them.

What needs to be added to Grammar Galaxy for ARC:
```
Grammar entries (compositional, not family-specific):
  "tile_pattern"       -> extract source region, repeat across grid
  "phase_shift"        -> offset tiling by row/column index
  "color_remap"        -> map color A to color B
  "object_extract"     -> identify connected components in grid
  "object_place"       -> place object at position
  "grid_resize"        -> change output grid dimensions
  "conditional_fill"   -> fill region based on neighbor condition
  "symmetry_complete"  -> complete pattern by symmetry axis
```

These are COMPOSITIONAL primitives. The TRM chains them. "Phase-shift tiling" is not a generator family -- it's `tile_pattern` composed with `phase_shift`. "Rotation-plus-layout" is `object_extract` composed with `ROTATE_90` composed with `object_place`.

**This is the same principle as math:** don't add templates for every problem type. Add composable primitives and let the TRM chain them.

#### LHE (New Application)

LHE questions are multi-domain expert-level questions. Many are multiple-choice. The four-pass decomposition applies directly:

```
Pass 1 (Forward -- parse the question):
  Clause-by-clause entity extraction.
  Identify: domain (physics, math, biology, history...), concepts mentioned, constraints stated.
  For multiple-choice: each option is also an entity with a claim.

Pass 2 (Backward -- from answer requirements):
  What kind of answer is expected? (multiple-choice letter, exact number, open text)
  What knowledge domain does the answer come from?
  What would make each option correct/incorrect?

Pass 3 (Fusion):
  Merge forward and backward analysis.
  Confirmed entities: domain, key concepts, answer format.
  For multiple-choice: each option now has forward analysis (what it claims) + backward analysis (what would make it correct).

Pass 4 (LHE-specific):
  Navigate Reality Galaxy for physics/chemistry/biology knowledge.
  Navigate Grammar Galaxy for reasoning patterns.
  Navigate Math Galaxy if computation is needed.
  For multiple-choice: score each option against Galaxy knowledge.
    Option matches Galaxy knowledge with high confidence -> candidate answer.
    Option contradicts Galaxy knowledge -> eliminate (contrastive -1).
    Option has uncertain match -> explore (ternary 0).
  For open-ended: compose answer from Galaxy knowledge using Grammar rules.
  Verification: does the answer satisfy all constraints identified in Pass 3?
```

**LHE depends heavily on Galaxy knowledge density.** The four-pass decomposition structures the REASONING, but without knowledge in Reality/Grammar/Math galaxies, it has nothing to reason with. The augmentation process addresses this. The four-pass decomposition makes sure that WHEN the knowledge is present, it's used effectively.

---

## Implementation Plan

### Phase 1: Lift Four-Pass to NavigatorSpecialist Level (IMMEDIATE)

The four passes currently live split between NavigatorSpecialist (passes 1-3) and MathSpecialist (pass 4). Restructure:

1. Passes 1-3 stay in NavigatorSpecialist where they already are.
2. Pass 4 becomes a **specialist dispatch with structured input**. Instead of passing raw query text to specialists, pass the FUSED ENTITY GRAPH.
3. Each specialist receives structured input: `{entities: [...], relations: [...], goal: {...}, raw_query: "..."}`
4. Each specialist implements its own Pass 4 (semantic verification + execution).

**Key code locations:**
- `NavigatorSpecialist.plan_routes()` (line 69) -- already does passes 1-3
- `NavigatorSpecialist.navigate()` (line 141) -- calls plan_routes, then dispatches
- The `parse_bundle` already flows to specialists via `MathSpecialist._extract_parse_bundle()`

What changes: the `parse_bundle` should carry the fused entity graph from Pass 3, not just raw parse structures. The specialist then does Pass 4 with that structured input.

### Phase 2: ARC Four-Pass (NEXT PRIORITY)

Instead of adding generator families one by one, implement the four-pass decomposition for ARC:

1. **Passes 1-2:** Analyze train pairs forward (input->output diffs) and backward (output->input requirements). Extract TRANSFORM ENTITIES per train pair.
2. **Pass 3:** Fuse across ALL train pairs. Transforms confirmed across multiple pairs get high confidence.
3. **Pass 4:** Compose transforms from Grammar Galaxy primitives. Verify against train pairs. Apply to test input.

**Existing code to wire into:**
- `arc_agi_2_adapter.py:558` -- `discover_patterns()` already analyzes train pairs
- `arc_agi_2_adapter.py:593` -- `discover_patterns_contrastive()` already does ternary quality
- `foundational_operations_bootstrap.py` -- already has ROTATE_90, MIRROR_H, TRANSPOSE, etc.
- `arc_agi_2_adapter.py:2811` -- `_generate_candidate_from_pattern()` generates candidates from patterns

The four-pass structure replaces the ad-hoc pattern discovery with systematic decomposition. The `discover_patterns` method becomes passes 1-3 (extract transforms from train pairs, fuse across pairs). The `_generate_candidate_from_pattern` method becomes pass 4 (compose from Grammar primitives, verify against train outputs).

**New Grammar Galaxy entries needed for ARC:**
Add to `foundational_operations_bootstrap.py`:
```
tile_pattern, phase_shift, color_remap, object_extract, object_place,
grid_resize, conditional_fill, symmetry_complete, border_fill,
crop_region, overlay_grid, flood_fill, connected_components
```

These are the ARC compositional primitives. They go into the foundational bootstrap -- loaded at every system init, just like basic math. The TRM composes them.

### Phase 3: LHE Four-Pass (AFTER AUGMENTATION)

LHE requires knowledge density that the augmentation process provides. But the four-pass structure should be wired NOW so that as knowledge grows, LHE accuracy improves automatically:

1. Wire LHE through the same NavigatorSpecialist four-pass path.
2. LHE's Pass 4 navigates Reality/Grammar/Math galaxies based on the domain identified in Pass 3.
3. Multiple-choice: score options against Galaxy knowledge, eliminate contradictions.
4. Open-ended: compose answer from Galaxy entries using Grammar rules.

---

## What NOT to Do

1. **Do NOT add ARC generator families one by one.** This is the template trap. Add composable PRIMITIVES to Grammar Galaxy and let the TRM compose.

2. **Do NOT build separate four-pass implementations for each benchmark.** Passes 1-3 are universal (NavigatorSpecialist). Only Pass 4 is specialist-specific.

3. **Do NOT flatten entities into lists.** The math 20/20 proved that semantic entities with roles/units/scopes work. Apply the same principle to ARC (transform entities with type/region/direction) and LHE (concept entities with domain/claim/confidence).

4. **Do NOT add problem-specific logic anywhere.** Grammar Galaxy entries are the knowledge. TRM navigation is the reasoning. PTX execution is the computation. Python code is the plumbing.

---

## Priority Order

```
1. Lift four-pass to NavigatorSpecialist dispatch    -- makes it available to all specialists
2. ARC Grammar Galaxy primitives                     -- foundational, loaded at init
3. ARC four-pass wiring                              -- replace ad-hoc pattern discovery
4. LHE four-pass wiring                              -- ready for augmentation knowledge
5. Run full audited benchmark pack                   -- measure real delta
```

---

## Grounding in Existing Specs

| Spec | Section | What It Says | How It Applies |
|------|---------|-------------|----------------|
| TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md | 3 (line 591) | "Generalize forward/backward reading to ALL chains" | Exactly this directive -- four-pass for all benchmarks |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | 3.4 (line 423) | `TASK DECOMPOSE` as RPN opcode | Pass 4 uses TASK DECOMPOSE to break complex tasks |
| DUAL_CLIENT_CONTRACT_SPECIFICATION.md | 1.6 | Save Information Principle -- reference, don't duplicate | ARC primitives compose, not duplicate as families |
| THREE_BRAIN_SYSTEM_SPECIFICATION.md | Cranium | PTX kernels execute, Galaxy stores, TRM navigates | Four passes: Galaxy for knowledge, TRM for composition, PTX for execution |
| KNOWLEDGEVERSE_SPECIFICATION.md | Region 2 | Galaxy Universe is active AI memory | Transform primitives are Galaxy entries, not Python code |

---

## Success Criteria

1. NavigatorSpecialist passes structured entity graph to all specialists (not raw text)
2. ARC solver uses four-pass decomposition instead of ad-hoc family generators
3. ARC Grammar Galaxy has composable transform primitives (loaded at init)
4. ARC composition depth > 1 (currently locked at 1.0 -- needs multi-step transform chains)
5. LHE routed through four-pass with domain-specific Galaxy navigation
6. Math remains 20/20 (regression guard)
7. No numpy/cupy/scipy in hot path
8. All new primitives are Grammar Galaxy entries, not Python if/else branches

---

## Daniel's Principle

The four-pass decomposition is how K3D thinks. Not just for math. Not just for benchmarks. For everything. A complex task arrives. K3D reads it forward (what is given). Reads it backward (what is asked). Merges (what do we know). Verifies and executes (translate to formal structure, check, compute).

This is the `TASK DECOMPOSE` opcode made real across the entire system.
