# Phase B+ Advancement — Three-Track Architecture Steering

**Date:** 2026-03-16
**Author:** Claude (Architecture Partner)
**Status:** Codex implementation directive — Track B REVERTED (regressed), advancing A → C

---

## Current Benchmark State

| Benchmark | Score | Infrastructure | Content Gap |
|-----------|-------|---------------|-------------|
| ARC | 10/10 | Solid | — |
| Math | 20/20 | Solid | — |
| GSM8K | 2/10 | Triple defeasible live | Pass 4 semantic verification not activated |
| LHE | 6/10 | Graph crystallizer wired (post-hoc) | Multi-hop needs iterative crystallization |
| MMLU | 15/50 | Defeasible deferred (exploratory regression) | Galaxy neighborhood coverage sparse |

**Infrastructure just completed:**
- NSI closed loop (five binary chokepoints → ternary routing)
- Triple defeasible pipeline (3-stage, GSM8K-only rollout)
- DefeasibleVerdictEvent in shadow copy
- Kernel contract map (24 bridges documented)
- 9 spec files updated

---

## Track B: LHE Multi-Hop via Iterative Graph Crystallization — REVERTED

**Target:** LHE 6/10 → 8+/10
**Result:** REVERTED. Pre-scoring crystallization dropped LHE 6→5, MMLU 15→8. Even LHE-only narrowing destabilized checkpoints. Graph crystallizer stays post-hoc. Multi-hop LHE improvement deferred to Phase D (TRM game loop).
**Learning:** Don't move kernels earlier in the pipeline without weight recalibration. Crystallized embeddings reduce candidate gap, making halting gate less decisive.

### The Problem

Graph crystallizer currently runs AFTER candidate ranking (knowledgeverse.py lines 6188-6262). It smooths embeddings but doesn't do iterative graph traversal during scoring. Multi-hop LHE questions need: query → get neighbor facts → re-query neighbors → aggregate.

### What Exists

- `GraphCrystallizer.crystallize_graph()` — GPU message passing kernel, supports K rounds
- `SemanticCSRGraph` — K-NN graph with Dijkstra-like local expansion (k=12, max 2048 nodes)
- `reasoning_multi_hop_top2` program — combines 2 Galaxy entries but is single-shot RPN
- Three call modes: local_graph, semantic_knn, compatibility (all post-hoc)

### Architecture Change: Pre-Scoring Crystallization

Move graph crystallizer from "post-hoc smoothing" to "pre-scoring enrichment":

```
CURRENT:
  candidates scored → graph crystallizer smooths → halting gate

PROPOSED:
  candidates collected → graph crystallizer enriches (2 rounds) → candidates scored → halting gate
```

**Specifically:**

1. **Before per-path scoring**, after `_apply_specialist_swarm_features()`:
   - Extract the local semantic neighborhood (CSR graph, max 2048 nodes)
   - Build dense adjacency from CSR for the candidate set
   - Run `crystallize_graph(rounds=2)` to propagate neighbor information into candidate embeddings
   - This gives each candidate the CONTEXT of its 2-hop neighborhood BEFORE scoring

2. **Wire into the per-path loop** (lines ~7900, ~8280):
   - After specialist features are computed but before `_score_gpu_candidates_batch()`
   - The crystallized embeddings replace the raw embeddings for scoring
   - Selection step: `"GRE graph crystallizer: pre-scoring rounds=2 nodes=N avg_neighbors=K"`

3. **Adjust rounds by task type:**
   - LHE_TASK: rounds=3 (multi-hop needs deeper propagation)
   - GSM8K/MATH_TASK: rounds=1 (compositional, not graph-heavy)
   - MMLU_TASK: rounds=2 (moderate cross-domain)
   - ARC_TASK: rounds=1 (transform-focused)

4. **Keep existing post-hoc crystallization** as a secondary pass (backward compatibility). The pre-scoring enrichment is additive.

### New Method

```python
def _apply_pre_scoring_crystallization(
    self,
    *,
    local_candidates: list[dict],
    path: dict,
    task_type: str,
    selection_steps: list[str],
) -> None:
    """Enrich candidate embeddings with graph-neighbor context before scoring."""
    rounds = {"LHE_TASK": 3, "MMLU_TASK": 2}.get(task_type, 1)
    # Build node features from candidate embeddings
    # Build adjacency from semantic CSR graph (local kernel extraction)
    # Call crystallize_graph(rounds=rounds, self_weight=0.6, neighbor_weight=0.4)
    # Thread crystallized embeddings back into candidates
```

### Success Criteria

- LHE 8+/10 (from 6/10)
- ARC/Math/GSM8K hold (no regression)
- Selection steps show pre-scoring crystallization activating for multi-hop queries

### Files to Modify

| File | Change |
|------|--------|
| `knowledgeverse.py` | Add `_apply_pre_scoring_crystallization()`, wire into per-path loop |
| `knowledgeverse.py` | Existing post-hoc crystallization stays (lines 6188-6262) |

---

## Track A: Four-Pass Semantic Verification (Pass 4 Activation)

**Target:** GSM8K 2/10 → 5+/10, benefits ALL benchmarks
**Why second:** The four-pass decomposition (forward/backward/fusion) already runs for every query via NavigatorSpecialist. Passes 1-3 are DONE. Pass 4 (domain-specific semantic verification) exists as spec but is NOT activated in the scoring pipeline.

### What Exists (DO NOT REWRITE)

| Pass | Method | Status |
|------|--------|--------|
| Pass 1: Forward | `NavigatorSpecialist._forward_reading_path()` (line 405) | LIVE for all queries |
| Pass 2: Backward | `NavigatorSpecialist._backward_reading_path()` (line 421) | LIVE for all queries |
| Pass 3: Fusion | `NavigatorSpecialist._fusion_reading_path()` (line 453) | LIVE for all queries, dedup via composite keys |
| Pass 4: Verify | Spec only (`CLAUDE_FOUR_PASS_MATH_COMPOSITION_DIRECTIVE_03.07.2026.md`) | NOT IMPLEMENTED |

### What Pass 4 Does (the "fourth option")

Pass 4 is **domain-specific semantic verification** on the fused entity graph from Pass 3:

**For Math/GSM8K (dimensional analysis):**
1. Take fused entities (each number bound to role + unit + scope)
2. Identify GOAL unit (e.g., "meters per week")
3. Chain entity relationships via dimensional cancellation (meters/sprint × sprints/week = meters/week)
4. If dimensions cancel consistently → build RPN program → score +1
5. If dimensions DON'T cancel → reject this composition path → score -1 (contrastive)

**For LHE (evidence scoring):**
1. Take fused entities (facts + goal)
2. Query Galaxy for evidence matching fused entities
3. Score each option against evidence coverage
4. Options contradicting fused entities → -1; supported → +1; neutral → 0

**For ARC (transform verification):**
1. Take fused transform entities (from grid-specific passes 1-3)
2. Compose transforms from Grammar Galaxy primitives
3. Apply to all training inputs, check against training outputs
4. Perfect match → +1; partial → 0; contradiction → -1

**For MMLU (knowledge verification):**
1. Take fused entities (question facts + domain)
2. Query domain-specific Galaxy neighborhood
3. Score options against Galaxy knowledge
4. Same evidence/contradiction/neutral trichotomy

### The Key Insight: Pass 4 = AtomicFissionFusion on Meaning

The `gre_atomic_fission_fusion` kernel already does exactly what Pass 4 needs:
- **Fission (decompose):** "Can the GOAL embedding be reconstructed from the entity embeddings?" → consistency score
- **Fusion (compose):** "Do the entity embeddings agree with each other?" → agreement score

Currently `_verify_compositional_consistency_for_math_task()` (line 5924) runs this ONLY for GSM8K candidates with positive focus. It should run for ALL candidates as the Pass 4 verification signal.

### Architecture Change: Thread Pass 4 Verdict into Scoring

1. **After per-path specialist scoring**, before halting gate:
   - Retrieve `parse_bundle` from NavigatorSpecialist (already available via `_collect_parse_bundle()`)
   - Run domain-specific Pass 4 verification on top candidates
   - Thread `pass4_verification_score` into candidate records
   - Add to RPN scoring expression at weight 0.05 (higher than defeasible because it's direct verification)

2. **For GSM8K specifically:**
   - Replace flat `_build_word_problem_rpn` (if/else tree based on problem type labels)
   - With dimensional-analysis composition from fused entities
   - Key: recognize reference expressions ("half that much", "twice as many", "the same amount")
   - Key: chain units via cancellation (meters/sprint × sprints/session × sessions/week)

3. **New scoring term:**
   ```
   specialist_pass4_verification * 0.05   ← domain-specific verification
   specialist_intra_defeasible * 0.03     ← Stage 2 (existing)
   specialist_defeasible_verdict * 0.04   ← Stage 3 (existing)
   ```

### Implementation Steps

#### Step 1: GSM8K Semantic Entity Extraction (math_specialist.py)

Replace `_extract_word_problem_entities()` flat number extraction with structured semantic entities:

```python
@dataclass
class SemanticEntity:
    value: float
    role: str       # count, rate, frequency, price, quantity, consume, result
    unit: str       # sprints, meters, dollars, eggs, etc.
    scope: str      # per_day, per_week, per_sprint, each, etc.
    clause_index: int
    reference: str | None  # "half", "twice", "double" → multiplier of previous entity
```

Role detection keywords:
- "X things/items" → count, unit=noun
- "X per Y" / "X each Y" → rate, unit=X, scope=per_Y
- "X times" / "X times a Y" → frequency, scope=per_Y
- "$X" / "X dollars" → price, unit=currency
- "remaining/left/after" → result (subtraction)
- "total/altogether" → goal_aggregate
- "half/twice/double/triple" → reference expression (NOT a standalone number)

#### Step 2: Dimensional Analysis Composer (math_specialist.py)

New method `_compose_via_dimensional_analysis()`:

1. Extract goal unit from fused entities
2. For each entity, compute its dimensional signature (numerator_units, denominator_units)
3. Greedily chain entities where output units of one cancel input units of next
4. If chain terminates at goal unit → build RPN program
5. If chain fails → report -1 (contrastive signal)
6. Verify via `AtomicFissionFusion.decompose()` — can the goal embedding be reconstructed from entity embeddings?

#### Step 3: Thread into Scoring Expression (knowledgeverse.py)

Add `specialist_pass4_verification` to the RPN scoring expression at weight 0.05.

#### Step 4: Universal Pass 4 Dispatch

In the per-path loop, after specialist features:
```python
if task_type in ("MATH_TASK", "GSM8K_TASK"):
    verification = self._pass4_dimensional_analysis(candidate, parse_bundle)
elif task_type == "LHE_TASK":
    verification = self._pass4_evidence_scoring(candidate, parse_bundle)
elif task_type == "MMLU_TASK":
    verification = self._pass4_knowledge_verification(candidate, parse_bundle)
else:
    verification = 0.0  # neutral for unknown domains
candidate["specialist_pass4_verification"] = verification
```

### Success Criteria

- GSM8K 5+/10 (from 2/10) — dimensional analysis solves multi-step problems
- Math 20/20 holds (strict axioms already have +D proof tags)
- LHE benefits from evidence-scored Pass 4
- MMLU benefits from knowledge-verified Pass 4

### Files to Modify

| File | Change |
|------|--------|
| `math_specialist.py` | `SemanticEntity`, `_compose_via_dimensional_analysis()`, replace flat extraction |
| `knowledgeverse.py` | Thread `specialist_pass4_verification` into scoring expression at 0.05 weight |
| `knowledgeverse.py` | Add `_pass4_*` dispatch methods in per-path loop |

---

## Track C: MMLU Galaxy Neighborhood Coverage

**Target:** MMLU 15/50 → 22+/50
**Why third:** Requires the most content work but benefits from the Pass 4 knowledge verification (Track A) being live first.

### The Problem

MMLU covers 50 diverse domains. Current Galaxy has strong math/grammar coverage but sparse Reality Galaxy entries for physics, chemistry, biology, history, geography, law, economics, etc.

### Architecture: Reality Galaxy Expansion

1. **Identify failing MMLU domains** — run the 50-question MMLU suite, categorize failures by domain
2. **For each failing domain:**
   - Add 5-10 core concept entries to Reality Galaxy (procedural RPN programs)
   - Add 3-5 Grammar Galaxy rules connecting concepts (with `rule_strength=0`, defeasible)
   - Add superiority relations where known (e.g., "conservation of energy is strict, not defeasible")
3. **Wire domain hints** — MMLU questions often contain domain keywords. Thread `domain_hint` into Galaxy navigation so LED-A* starts in the right neighborhood.
4. **Leverage Pass 4 knowledge verification** — once Track A is live, MMLU candidates get verified against Galaxy knowledge, improving option elimination.

### Key Constraint

Exploratory grammar insertion is DEFERRED to sleep-time only (Codex's finding — live insertion regressed MMLU). All new Galaxy content must be added via the ingestion path (foundational_operations_bootstrap.py), NOT generated during inference.

### Success Criteria

- MMLU 22+/50 (from 15/50)
- No regression on other benchmarks
- Reality Galaxy grows by 100+ entries across MMLU domains

### Files to Modify

| File | Change |
|------|--------|
| `foundational_operations_bootstrap.py` | Add Reality Galaxy entries per MMLU domain |
| `grammar_galaxy.py` | Add domain-specific Grammar rules with defeasible metadata |
| `knowledgeverse.py` | Improve `domain_hint` threading for MMLU LED-A* navigation |

---

## Execution Order

```
Track B (LHE multi-hop):
  1. Add _apply_pre_scoring_crystallization()
  2. Wire into per-path loop (before scoring, after specialist features)
  3. Adjust rounds by task type
  4. Validate: LHE 8+/10, no regression

Track A (Four-pass activation):
  1. SemanticEntity dataclass + structured extraction
  2. Dimensional analysis composer
  3. Thread pass4_verification into scoring at 0.05
  4. Universal Pass 4 dispatch (math → dim analysis, LHE → evidence, MMLU → knowledge)
  5. Validate: GSM8K 5+/10, Math 20/20 holds

Track C (MMLU Galaxy coverage):
  1. Audit MMLU failures by domain
  2. Add Reality Galaxy entries per domain
  3. Add Grammar rules with superiority relations
  4. Thread domain_hint into LED-A* navigation
  5. Validate: MMLU 22+/50
```

---

## Sovereignty Compliance

All three tracks maintain hot-path sovereignty:
- Track B: `gre_graph_crystallizer.cu` (existing PTX kernel, no new kernels)
- Track A: `gre_atomic_fission_fusion.cu` for verification (existing), RPN for dimensional analysis (existing)
- Track C: Galaxy entries via ingestion path (flexible), queried via sovereign Galaxy navigation

No new Python in the hot path. No external dependencies. The four-pass decomposition runs in NavigatorSpecialist (Python, but it's query parsing, not reasoning). All reasoning stays PTX + Galaxy + RPN.

---

## Reference Documents

- `CLAUDE_FOUR_PASS_MATH_COMPOSITION_DIRECTIVE_03.07.2026.md` — Pass 4 math design with worked examples
- `CLAUDE_UNIVERSAL_FOUR_PASS_DECOMPOSITION_DIRECTIVE_03.07.2026.md` — Domain-specific Pass 4 variants
- `CLAUDE_UNIFIED_FOUR_PASS_SINGLE_SYSTEM_DIRECTIVE_03.07.2026.md` — "One system, not three"
- `SOVEREIGN_NSI_SPECIFICATION.md §9` — Kernel function contracts
