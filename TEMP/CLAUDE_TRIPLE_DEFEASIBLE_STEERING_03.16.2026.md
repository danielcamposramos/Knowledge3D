# Triple Defeasible Resolution — Architecture Steering

**Date:** 2026-03-16
**Author:** Claude (Architecture Partner)
**Status:** Steering spec for Codex — three-stage defeasible reasoning

---

## The Insight

Daniel: "Maybe utilize it two times in the reasoning chain — early then where it is now, or, use it into the individual reasoning chains as well — the one that fits best or both so we use it three times."

**Why this is the missing piece:** Currently the defeasible resolver runs ONCE, at the end, on aggregated path records. But defeasible reasoning is most powerful when it prunes contradictions EARLY, so downstream processing doesn't waste work on defeated candidates.

Think of it like a game engine: you don't just collision-detect at the end — you cull early (broad phase), refine per-entity (narrow phase), and confirm at commit (resolution phase).

---

## Current Pipeline (single defeasible pass)

```
query()
  → _build_gpu_reasoning_paths()           ← paths assembled
  → _dispatch_swarm_weights()              ← swarm weights assigned
  → FOR EACH PATH:
      → _apply_specialist_swarm_features() ← geometry/temporal/fractal/resonance scored
      → _score_gpu_candidates_batch()      ← RPN scoring
  → _apply_defeasible_specialist_resolution()  ← SINGLE PASS (line 8575)
  → _halting_gate_converged()              ← convergence check
```

## Proposed Pipeline (triple defeasible)

```
query()
  → _build_gpu_reasoning_paths()
  → _dispatch_swarm_weights()
  → *** STAGE 1: EARLY DEFEASIBLE GATE ***     ← prune defeated paths BEFORE per-path work
  → FOR EACH PATH:
      → _apply_specialist_swarm_features()
      → *** STAGE 2: PER-PATH DEFEASIBLE ***   ← resolve conflicts WITHIN each path's candidates
      → _score_gpu_candidates_batch()
  → *** STAGE 3: FINAL DEFEASIBLE RESOLUTION ***  ← existing pass, now with upstream proof tags
  → _halting_gate_converged()
```

---

## Stage 1: Early Defeasible Gate (path-level pruning)

**Where:** After `_dispatch_swarm_weights()`, before the per-path scoring loop.
**What:** Apply defeasible resolution at the PATH level — if two paths use conflicting Grammar rules and one has superiority, defeat the inferior path early.

**Inputs:**
- `paths[]` — each path has a `program_id` linking to a Grammar rule
- `swarm_weights[]` — from nine-chain dispatch
- Grammar rule metadata (rule_strength, superior_to, trust_weight)

**Algorithm:**
1. For each path, look up its Grammar rule's `rule_strength` and `superior_to`
2. Build path-level superiority: if path_A's rule is `superior_to` path_B's rule, and they produce conflicting directions (different target galaxies or contradictory transform types), path_B's weight gets multiplied by a defeat factor (e.g., 0.3 — soft defeat, not hard kill)
3. If a path uses a `defeater` rule (strength = -1), it reduces competing paths' weights but doesn't contribute positively itself
4. Thread a `path_defeasible_tag` onto each path: `+1` (undefeated), `0` (soft-defeated), `-1` (hard-defeated by strict superiority)
5. Soft-defeated paths still run (they might have valuable candidates) but with reduced weight. Hard-defeated paths skip entirely.

**Why soft defeat:** At this stage we don't have per-candidate evidence yet. We're pruning based on rule-level superiority only. A path might have a defeated rule but contain the right candidate via a different mechanism. So we REDUCE weight, not kill.

**New method:** `_apply_early_defeasible_gate(paths, swarm_weights, selection_steps)` → modifies `swarm_weights` in place, adds `path_defeasible_tag` to each path.

**Uses:** The same `gre_defeasible_resolver.cu` kernel. Each path becomes a "worker" with its rule_strength. The "candidates" are the paths themselves (num_candidates = len(paths)). Conclusions = swarm_weights. This is a reuse of the existing kernel with different semantic framing.

---

## Stage 2: Per-Path Defeasible Resolution (candidate-level within each path)

**Where:** Inside the per-path loop, AFTER `_apply_specialist_swarm_features()`, BEFORE `_score_gpu_candidates_batch()`.
**What:** Apply defeasible resolution to the candidates WITHIN a single path — if two candidates use conflicting Grammar rules, resolve them before scoring.

**Inputs:**
- `local_candidates[]` — each candidate has `specialist_worker` (the Grammar rules applied)
- Per-candidate specialist scores (resonance, coherence, geometry, temporal, fractal)
- Grammar rule metadata from the rules that contributed to each candidate

**Algorithm:**
1. For each candidate, identify which Grammar rules contributed (from `specialist_worker` field and the reasoning program)
2. Build candidate-level superiority: if candidate_A's contributing rule has `superior_to` candidate_B's rule, and their specialist scores indicate conflict (high A vs high B), candidate_B gets a defeasible penalty
3. Compute per-candidate `intra_path_defeasible_verdict`: the verdict considering only within-path conflicts
4. Thread into candidate as `specialist_intra_defeasible` — feeds into the RPN scoring expression

**Why this matters:** Currently all candidates within a path are scored independently. But if one candidate was found via a strict rule (`2+3=5`) and another via a defeasible rule that contradicts it, the strict one should dominate. Stage 2 makes this explicit.

**New method:** `_apply_intra_path_defeasible(local_candidates, path, task_type, selection_steps)` → adds `specialist_intra_defeasible` to each candidate.

**Uses:** Same `gre_defeasible_resolver.cu` kernel. Workers = candidates, conclusions = specialist scores, rule_strengths from Grammar metadata.

---

## Stage 3: Final Defeasible Resolution (existing, enhanced)

**Where:** Line 8575, after all paths scored.
**What:** The existing `_apply_defeasible_specialist_resolution()` — now enhanced with upstream proof tags.

**Enhancement:**
- Candidates arriving at Stage 3 now carry `path_defeasible_tag` (from Stage 1) and `specialist_intra_defeasible` (from Stage 2)
- The final verdict combines all three stages:
  - If Stage 1 hard-defeated the path → verdict = 0 regardless
  - If Stage 2 found intra-path defeat → reduce verdict by defeat factor
  - Stage 3 applies cross-path defeasible resolution as today
- The `specialist_proof_tag` now encodes the FULL chain: was this candidate supported by strict rules that survived all three stages?

**Modification to existing method:** Add `path_defeasible_tag` and `specialist_intra_defeasible` as inputs to the verdict computation.

---

## Scoring Expression Integration

Currently (Stage 3 only):
```
specialist_defeasible_verdict * 0.04
```

With triple defeasible:
```
specialist_intra_defeasible * 0.03     ← Stage 2 (per-path candidate conflicts)
specialist_defeasible_verdict * 0.04   ← Stage 3 (cross-path, existing)
```

Stage 1 doesn't add a scoring term — it modifies swarm_weights directly, which propagates through all downstream scoring.

---

## Implementation Steps (for Codex)

### Step 1: Early Defeasible Gate

1. Add method `_apply_early_defeasible_gate(self, *, paths, swarm_weights, selection_steps)`
2. For each path, extract Grammar rule profile via `_defeasible_rule_profile(program_id)`
3. Build path-level conclusions array (from swarm_weights) and rule_strengths array
4. Build superiority adjacency from `superior_to` across path rules
5. Call `gre_defeasible_resolver` via bridge (reuse existing DefeasibleResolver)
6. Multiply defeated path weights by 0.3 (soft defeat). Tag paths with `path_defeasible_tag`.
7. Wire call at line ~7678, right after `_dispatch_swarm_weights()` returns

### Step 2: Per-Path Defeasible Resolution

1. Add method `_apply_intra_path_defeasible(self, *, local_candidates, path, task_type, selection_steps)`
2. For each candidate, determine contributing rule's strength and superiority
3. Build candidate-level conclusions (from specialist scores), rule_strengths, superiority
4. Call `gre_defeasible_resolver` via bridge
5. Thread `specialist_intra_defeasible` verdict into each candidate
6. Wire call inside per-path loops (lines ~7901, ~8283), after `_apply_specialist_swarm_features()`, before `_score_gpu_candidates_batch()`

### Step 3: Enhanced Final Resolution

1. In `_apply_defeasible_specialist_resolution()`, incorporate `path_defeasible_tag` from Stage 1
2. If a record's path was hard-defeated (tag = -1), set verdict to 0.0
3. If a record's path was soft-defeated (tag = 0), multiply verdict by 0.3
4. The existing Stage 3 logic continues as-is for undefeated paths

### Step 4: Scoring Expression

1. In `_build_gpu_candidate_score_expression()`, add `specialist_intra_defeasible` at weight 0.03
2. Existing `specialist_defeasible_verdict` stays at 0.04

### Step 5: Tests

1. Test Stage 1: strict-rule path defeats defeasible-rule path → weight reduction
2. Test Stage 2: within a path, strict candidate beats defeasible candidate
3. Test Stage 3: cross-path resolution honors upstream defeat tags
4. Test backward compat: no superiority defined → all three stages are no-ops (verdicts = raw scores)
5. Full benchmark: must hold ARC 10/10, Math 20/20, no regression

---

## Why Three Stages Is Correct

**Game engine analogy:**

| Stage | Game Engine | K3D Defeasible |
|-------|-------------|---------------|
| **Broad phase** | AABB bounding-box cull | Stage 1: path-level rule defeat (cheap, prunes whole paths) |
| **Narrow phase** | Per-entity collision | Stage 2: per-candidate intra-path defeat (medium cost, refines within path) |
| **Resolution** | Contact resolution | Stage 3: cross-path final verdict (expensive, produces final proof tags) |

Each stage uses the SAME kernel (`gre_defeasible_resolver.cu`) with different semantic inputs. Three launches of a single kernel. Zero new kernels needed.

**The key insight:** Defeasible logic is not just conflict resolution — it's a FILTERING mechanism. The earlier you filter, the less work downstream. Stage 1 can soft-defeat 2-3 paths before they each generate candidates. That's potentially 60% less per-path scoring work on conflicting strategies.

---

## Sovereignty Compliance

All three stages reuse `gre_defeasible_resolver.cu` (PTX). No new Python logic in hot path. The new methods are thin wiring that prepares arrays and calls the existing bridge.
