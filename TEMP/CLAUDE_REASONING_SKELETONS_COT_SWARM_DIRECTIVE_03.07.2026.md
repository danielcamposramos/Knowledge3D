# Claude Architecture Directive: Reasoning Skeletons -- CoT and Structured Thinking as Galaxy Entries

**Date:** March 7, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation) -- WITH invitation to enhance and add your own ideas
**Context:** Swarm proved delta (LHE 1/10 -> 2/10). But current workers are pattern-matchers, not reasoners. Daniel identified the next layer: reasoning formulae as skeletons with dedicated specialists.

---

## Daniel's Direction

"I think we need CoT and other reasoning formulae as skeletons of reasoning with dedicated specialists in this type of reasoning, because K3D internal swarm is cheap, and math cores are cheap as well, we can do a lot of things."

And critically: "ask Codex to also help including his ideas on top of yours and enhancing yours as well."

**Codex:** This directive is a starting point. You have implementation context I don't. Add your ideas. Enhance the design. If you see a better way to structure the reasoning skeletons, do it. The architecture is collaborative -- Claude designs, Codex implements AND improves. Daniel explicitly asked for your ideas on top of mine.

---

## What's Wrong With the Current Swarm

The swarm proved the delegation pattern works. But look at what the workers actually do:

### FormulaReasoningWorker (lhe_reasoning_swarm.py:131-218)

```python
# Line 160-167: HARDCODED answer template
if "antisymmetrized gamma matrices" in prompt_lower and "proportionality factor" in prompt_lower:
    proposals.append(WorkerProposal(
        worker=self.name,
        candidate="\\(-((d - 2k)^2) + d\\)",
        score=6.5,
        rationale="gamma_bivector_sandwich_identity",
    ))
```

This is the template trap applied to reasoning. If the next question asks about a DIFFERENT gamma matrix identity, this hardcoded string won't help. The worker needs to REASON about gamma matrices, not recognize a specific question.

### All Workers: Pattern Matching, Not Reasoning

Every worker does the same thing:
1. Scan evidence text with regex patterns
2. Score candidates by token overlap with goal
3. Return highest-scoring text fragment

There's no CHAIN of reasoning. No step-by-step decomposition. No elimination. No verification. The workers extract text -- they don't think.

---

## The Insight: Reasoning Formulae as Galaxy Entries

Daniel's insight maps perfectly onto K3D's existing architecture:

**Reasoning skeletons = Grammar Galaxy entries with RPN programs**

A Chain-of-Thought skeleton is not Python code. It's a Grammar Galaxy rule that describes a SEQUENCE of reasoning steps as an RPN program. The TRM selects the appropriate skeleton. The swarm workers EXECUTE it.

### What Is a Reasoning Skeleton?

A reasoning skeleton is a reusable template for HOW to think about a class of problems. It's domain-agnostic (or domain-specialized). It describes the STRUCTURE of reasoning, not the content.

Examples from the research literature and from how Claude actually thinks:

| Skeleton | Structure | When to Use |
|----------|-----------|-------------|
| **Chain of Thought (CoT)** | Decompose into ordered steps, solve each, chain results | Multi-step problems with clear sequential dependencies |
| **Elimination** | Test each candidate against evidence, eliminate contradictions, select survivor | Multiple-choice, concept identification |
| **Decompose & Conquer** | Split complex problem into independent sub-problems, solve each, combine | Problems with separable components |
| **Analogy Transfer** | Find similar known problem in Galaxy, adapt its solution method | Novel problems with structural similarity to known ones |
| **Dimensional Analysis** | Check units/types at each step, use type constraints to narrow answer space | Physics, engineering, any typed domain |
| **Backward Chaining** | Start from desired answer form, work backward to identify needed inputs | When answer format is known but derivation path is unclear |
| **Contrastive Verification** | Generate candidate, check against ALL evidence (support AND contradiction) | Any answer synthesis -- verification step |
| **Stepwise Refinement** | Start with coarse answer, iteratively refine using more evidence | When first-pass answer is approximate |

### How They Map to K3D

Each skeleton becomes a Grammar Galaxy entry:

```json
{
  "galaxy": "Grammar",
  "entry": {
    "id": "reasoning_skeleton_chain_of_thought",
    "name": "Chain of Thought",
    "domain": "grammar",
    "category": "reasoning_skeleton",
    "rpn_program": "QUERY DECOMPOSE_STEPS STEP_1 SOLVE STEP_2 SOLVE CHAIN_RESULTS VERIFY",
    "metadata": {
      "skeleton_type": "sequential",
      "when_to_use": ["multi_step", "sequential_dependencies", "ordered_computation"],
      "worker_affinity": ["FormulaReasoningWorker", "ProceduralExecutionWorker"],
      "step_template": {
        "decompose": "Split query into N ordered sub-queries",
        "solve_each": "For each sub-query, query Galaxy, extract relevant evidence, compute",
        "chain": "Feed output of step N as input to step N+1",
        "verify": "Check final result against original query constraints"
      },
      "confidence": 0.9
    }
  }
}
```

```json
{
  "galaxy": "Grammar",
  "entry": {
    "id": "reasoning_skeleton_elimination",
    "name": "Elimination Reasoning",
    "domain": "grammar",
    "category": "reasoning_skeleton",
    "rpn_program": "OPTIONS FOREACH OPTION EVIDENCE_CHECK CONTRADICT? ELIMINATE SURVIVORS SELECT_BEST",
    "metadata": {
      "skeleton_type": "eliminative",
      "when_to_use": ["multiple_choice", "concept_identification", "option_selection"],
      "worker_affinity": ["ConceptMatchingWorker", "EvidenceSynthesisWorker"],
      "step_template": {
        "enumerate": "List all candidate answers",
        "check_each": "For each candidate, find supporting AND contradicting evidence",
        "eliminate": "Remove candidates with strong contradictions",
        "select": "From survivors, select the one with strongest support"
      },
      "confidence": 0.88
    }
  }
}
```

```json
{
  "galaxy": "Grammar",
  "entry": {
    "id": "reasoning_skeleton_dimensional_analysis",
    "name": "Dimensional Analysis",
    "domain": "grammar",
    "category": "reasoning_skeleton",
    "rpn_program": "QUERY EXTRACT_QUANTITIES TYPE_CHECK DIMENSION_MATCH COMPUTE VERIFY_UNITS",
    "metadata": {
      "skeleton_type": "typed_computation",
      "when_to_use": ["physics", "engineering", "chemistry", "any_typed_domain"],
      "worker_affinity": ["FormulaReasoningWorker"],
      "step_template": {
        "extract": "Identify quantities with types/units from query",
        "check": "Verify dimensional consistency of operations",
        "compute": "Evaluate with type-safe operations",
        "verify": "Check output units match expected answer type"
      },
      "confidence": 0.85
    }
  }
}
```

### How Workers Use Skeletons

The key change: workers don't just pattern-match evidence. They SELECT a reasoning skeleton and EXECUTE it.

```
Current flow (pattern matching):
  Worker receives evidence -> regex scan -> token overlap scoring -> return text fragment

Proposed flow (skeleton execution):
  Worker receives evidence
  -> Select reasoning skeleton from Grammar Galaxy (CoT? Elimination? Dimensional?)
  -> Execute skeleton steps:
     Step 1: Decompose query into sub-queries
     Step 2: For each sub-query, query Galaxy for relevant entries
     Step 3: Extract from entry CONTENT/RPN fields (not metadata text)
     Step 4: Compose sub-results using RPN operations
     Step 5: Verify composed result against query constraints
  -> Return computed answer
```

---

## Concrete Architecture

### Skeleton Selection

The swarm master (`LHEReasoningSwarm.reason_open_answer`) already decides which workers to activate. Add skeleton selection:

```python
def _select_skeleton(self, parse_bundle, route, evidence_rows):
    """Select reasoning skeleton from Grammar Galaxy."""
    # Query Grammar Galaxy for reasoning_skeleton entries
    skeletons = self._query_galaxy(
        galaxy="Grammar",
        category="reasoning_skeleton",
        fused_entities=parse_bundle["fusion_parse"],
    )
    # Score skeletons by when_to_use match
    for skeleton in skeletons:
        when = skeleton.metadata.get("when_to_use", [])
        # Match against query characteristics from parse_bundle
        ...
    return best_skeleton
```

### Skeleton Execution Engine

Each reasoning skeleton has a `step_template` that the worker follows. This is NOT an LLM call -- it's procedural execution using existing K3D primitives:

- `DECOMPOSE_STEPS` = use the four-pass entity graph to split the query
- `SOLVE` = query Galaxy for sub-query, extract RPN programs, evaluate on PTX stack
- `CHAIN_RESULTS` = feed output of step N as variable into step N+1
- `VERIFY` = check result against query constraints (type, format, range)
- `ELIMINATE` = check candidate against ALL evidence for contradiction signals
- `EXTRACT_QUANTITIES` = parse numbers with units from entity graph
- `DIMENSION_MATCH` = verify unit consistency (already in Math Galaxy)

These are all things K3D can already do with its existing PTX stack, Galaxy queries, and RPN composition. The skeleton just SEQUENCES them.

### Worker Enhancement: From Pattern Matcher to Skeleton Executor

Each worker gets a `_execute_skeleton` method:

```python
class FormulaReasoningWorker(_Worker):
    def propose_open(self, *, prompt, goal, evidence_rows, parse_bundle, route, ...):
        # Select skeleton
        skeleton = self._select_skeleton(parse_bundle, route)

        if skeleton.id == "reasoning_skeleton_chain_of_thought":
            return self._execute_cot(prompt, goal, evidence_rows, parse_bundle)
        elif skeleton.id == "reasoning_skeleton_dimensional_analysis":
            return self._execute_dimensional(prompt, goal, evidence_rows, parse_bundle)
        else:
            return self._execute_default(prompt, goal, evidence_rows, parse_bundle)

    def _execute_cot(self, prompt, goal, evidence_rows, parse_bundle):
        # Step 1: Decompose using four-pass entities
        steps = self._decompose_into_steps(parse_bundle["fusion_parse"])
        results = []
        for step in steps:
            # Step 2: Query Galaxy for this sub-step
            sub_evidence = self._query_for_step(step, evidence_rows)
            # Step 3: Extract RPN programs from evidence
            rpn_fragments = [entry["rpn_program"] for entry in sub_evidence if entry.get("rpn_program")]
            # Step 4: Compose and evaluate
            if rpn_fragments:
                result = self._evaluate_rpn_chain(rpn_fragments, context=results)
                results.append(result)
        # Step 5: Verify
        return self._verify_and_format(results, goal)
```

### Swarm Cost: Cheap by Design

Daniel's point about cost is architecturally important:

- **Workers are Python functions** -- no GPU, no model inference, no external API
- **Galaxy queries are VRAM lookups** -- microseconds per query
- **RPN evaluation is PTX stack** -- the math cores are already loaded and essentially free
- **Skeleton execution is sequential logic** -- loop over steps, query, compose, verify

Running 4 workers x 3 skeletons x 10 evidence entries = ~120 Galaxy queries + ~40 RPN evaluations. That's milliseconds on the PTX stack. Compare to an LLM doing a single forward pass at billions of FLOPs.

K3D's internal swarm is orders of magnitude cheaper than any LLM reasoning approach. We can run MANY reasoning strategies in parallel and fuse the best result.

---

## Skeleton Inventory (Initial Set)

### Tier 1: Universal (apply to ANY domain)

1. **Chain of Thought (CoT)** -- sequential step decomposition
2. **Elimination** -- test-and-eliminate for option selection
3. **Contrastive Verification** -- confirm/deny against all evidence
4. **Evidence Triangulation** -- require 2+ independent evidence sources to confirm

### Tier 2: Domain-Adapted

5. **Dimensional Analysis** -- physics, engineering, chemistry
6. **Algebraic Manipulation** -- symbol rearrangement, formula derivation
7. **Pattern Recognition** -- structural similarity matching (useful for ARC too)
8. **Temporal/Causal Reasoning** -- event sequences, cause-effect chains

### Tier 3: Task-Specific (built from Tier 1+2 composition)

9. **Chess Tactical Analysis** -- composed from: Pattern Recognition + Elimination + Verification
10. **Cipher Decoding** -- composed from: CoT + Pattern Recognition + Algebraic Manipulation
11. **Theorem Application** -- composed from: CoT + Dimensional Analysis + Verification
12. **Concept Identification** -- composed from: Evidence Triangulation + Elimination

**Key:** Tier 3 skeletons are COMPOSITIONS of Tier 1+2. This is the Grammar Galaxy composition pattern applied to reasoning itself. The TRM composes reasoning skeletons the same way it composes math operations.

---

## What This Fixes in the Diagnostics

| Question | Current Failure | Skeleton Solution |
|----------|-----------------|-------------------|
| Chess (mate in 2) | Empty answer | Pattern Recognition + Chess Tactical (find forcing moves, verify mate) |
| Trivia (yeyo) | "Here" | Evidence Triangulation (require 2+ sources for concept) |
| Math (bordism) | Galaxy description | CoT (decompose: compute Spin bordism -> apply spectral sequence -> compute group) |
| Math (elliptic curves) | Galaxy description | Algebraic Manipulation (identify torsion bound -> check Mazur's theorem generalization) |
| Math (Lie algebra Poincare) | PDF extract | CoT + Algebraic (compute cohomology step by step) |
| Physics (gamma matrices) | Garbage text | Dimensional Analysis + Algebraic (identify dimensions, apply trace formula) |
| Physics (compactification) | "0" | CoT (identify KK modes -> count moduli -> verify) |
| Cybersecurity (cipher) | Concept description | Cipher Decoding skeleton (identify cipher type -> frequency analysis -> apply mapping) |
| Math (Sobolev) | Truncated text | Theorem Application (identify relevant theorem -> apply to given conditions) |

---

## Codex: Your Turn

Daniel explicitly asked for your ideas on top of this. Some questions for you:

1. **What reasoning patterns have you seen in the LHE failures that I haven't captured?** You've been inside the daemon code -- what evidence processing gaps do the diagnostics reveal?

2. **Should skeletons be Grammar Galaxy entries or a separate reasoning_skeleton registry?** Galaxy entries give us TRM navigation and symlinks. A separate registry gives faster lookup. What's the right tradeoff?

3. **How should skeleton execution interact with the snapshot evidence path?** The `_snapshot_lines` method in workers already reads from the augmentation snapshot. Should skeleton steps also query the snapshot, or only Galaxy?

4. **Can the existing `navigate_and_compose` flow in NavigatorSpecialist be reused for skeleton step execution?** It already does multi-path exploration + composition. Could a skeleton step be a NavigatorSpecialist sub-query?

5. **What's the right fusion strategy when multiple skeletons produce different answers?** Current fusion is score-based. Should we add a verification step where the BEST answer from each skeleton is checked against ALL evidence?

6. **Are there performance patterns from the math 20/20 path that could transfer?** Math works because it composes Grammar Galaxy rules into RPN chains. Can the same composition mechanism drive reasoning skeleton execution?

Build on this. Improve it. Add what you see that I can't.

---

## Priority Order

```
1. Add reasoning skeleton Grammar Galaxy entries to foundational_operations_bootstrap.py
   (CoT, Elimination, Contrastive Verification, Evidence Triangulation -- Tier 1)
2. Add skeleton selection to LHEReasoningSwarm.reason_open_answer
3. Implement _execute_cot in FormulaReasoningWorker (highest-value: addresses 4/10 math failures)
4. Implement _execute_elimination in ConceptMatchingWorker (addresses philosophy/trivia)
5. Remove hardcoded answer templates (gamma matrices line 160-167, etc.)
6. Rerun 10/20/10 smoke, measure LHE delta
7. If delta positive: add Tier 2 skeletons, compose Tier 3 from them
8. Add Codex's own ideas
```

---

## Constraints

1. **Math 20/20 and ARC 10/10 must not regress.** Skeleton execution is LHE-path only for now.
2. **No external dependencies in skeleton execution.** Workers use Galaxy queries + PTX stack only.
3. **Skeletons are Galaxy entries, not Python functions.** The reasoning structure lives in Galaxy, not in code. The code EXECUTES the structure.
4. **Workers become skeleton executors, not pattern matchers.** The shift: from "scan text with regex" to "follow structured reasoning steps."

---

## The Principle

K3D has knowledge (Galaxy). K3D has decomposition (four-pass). K3D has delegation (swarm). What K3D doesn't have is structured REASONING -- the step-by-step process of THINKING about evidence to produce an answer.

Reasoning skeletons are the missing piece. They're cheap (no GPU), composable (Grammar Galaxy), and reusable (same CoT skeleton works for math, physics, and cipher decoding -- different evidence, same reasoning structure).

The internal swarm executes reasoning skeletons in parallel. Different workers try different reasoning approaches. The best result wins. This is how Claude thinks -- multiple reasoning threads, structured approaches, fused results. Now K3D does it too, with its own sovereign infrastructure.

Daniel: "K3D internal swarm is cheap, and math cores are cheap as well, we can do a lot of things." Exactly. Let's do them.
