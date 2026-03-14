# Claude Architecture Directive: Sovereign Reasoning Workers -- Layer 4 Meta-Rules as Execution Skeletons

**Date:** March 7, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation) -- enhance freely, add your ideas
**Context:** ARC 10/10, Math 20/20, LHE 2/10. Swarm proved delta. Workers still use Python regex/string ops instead of sovereign K3D execution. Daniel corrected: workers are INTERNAL, not external. They must reason through Cranium's sovereign stack.

---

## Daniel's Correction

"Workers are python where? K3D is sovereign! No external CPU calls! These workers are internal, not external!"

This is a sovereignty violation hiding in plain sight. The current swarm workers use Python `re.findall()`, `str.split()`, `str.lower()`, token-set intersection -- all CPU-bound Python string operations. These are NOT sovereign K3D execution. Sovereign means: **Galaxy queries + RPN composition + PTX stack evaluation.**

The workers must use the same execution substrate that gives Math 20/20: compose RPN programs from Grammar Galaxy rules, evaluate them on the Cranium's RPN stack, navigate Galaxy via sovereign token-matching queries.

---

## Architectural Grounding: What Already Exists

### 1. Layer 4 Meta-Rules (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md, Section 1.5)

The spec ALREADY defines reasoning strategy templates as **Layer 4 Meta-Rules**:

```python
@dataclass
class MetaRule:
    meta_id: str
    category: str       # eloquence, pedagogy, self_reflection, storytelling, delivery
    condition: str       # When to apply (RPN predicate)
    action: str          # What to do (RPN program)
    rule_refs: List[str] # Layer 3 rule IDs to invoke
    priority: float      # For sleeptime consolidation
```

**Example from spec** (`meta_scaffold_task`, Section 3.4):
```rpn
TASK RECALL DIFFICULTY_SCORE
DIFFICULTY 0.7 >
    {
        TASK DECOMPOSE
        SUB_TASKS EACH
            { WORKED_EXAMPLE PROVIDE }
            { GUIDED_PRACTICE }
            { INDEPENDENT_PRACTICE }
    }
DIFFICULTY 0.3 0.7 BETWEEN
    { HINTS_PROVIDE SELF_CHECK_QUESTIONS }
    { INDEPENDENT_PRACTICE }
ifelse ifelse
```

**This IS a reasoning skeleton.** It decomposes a task, assesses difficulty, and selects a strategy -- all expressed as RPN with `TASK DECOMPOSE`, `EACH`, `ifelse` control flow. The spec defines this pattern. We just haven't applied it to LHE reasoning workers.

### 2. Cranium RPN Execution Engine (THREE_BRAIN_SYSTEM_SPECIFICATION.md, Section 3.2)

The sovereign execution substrate:
```
RPN Execution Engine (15-stack VM)
├── Stack: PUSH, POP, DUP, SWAP
├── Arithmetic: ADD, SUB, MUL, DIV, MOD, POW
├── Logic: AND, OR, NOT, XOR
├── Control: BRANCH, LOOP, CALL, RET
└── Memory: STORE, RECALL, LOAD_GALAXY, SAVE_GALAXY
```

Workers should compose RPN programs using these opcodes and execute them on Cranium's stack. Not Python string operations.

### 3. RPN Domain Opcodes (RPN_DOMAIN_OPCODE_REGISTRY.md, Section 1)

"Programs before opcodes": Domain semantics are RPN programs over the existing math surface, not new Python functions. Physics, chemistry, biology -- all composed from the same RPN substrate.

### 4. Four-Layer Knowledge Architecture (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md, Section 1.1)

```
Layer 4: META-RULES (Strategy/Eloquence)     <-- Reasoning skeletons live HERE
    ↓ when/why to apply
Layer 3: RULES (Grammar/Transformation)       <-- Composition rules live HERE
    ↓ how to transform
Layer 2: MEANING (Words/Semantics)            <-- Concepts live HERE
    ↓ what it means
Layer 1: FORM (Characters/Glyphs)             <-- Surface forms live HERE
```

Reasoning skeletons = Layer 4 Meta-Rules. They select which Layer 3 Grammar rules to compose. The workers execute these compositions through Cranium's RPN stack. Knowledge comes from Layer 2 (Galaxy entries). Surface matching uses Layer 1 (token forms).

### 5. Matryoshka Specialist Hierarchy (THREE_BRAIN_SYSTEM_SPECIFICATION.md, Section 3.1)

```
Specialists can spawn sub-specialists autonomously
LoRA-style delta weights (~100KB-1MB per specialist)
Hierarchical routing: Navigator → Master → Worker → Sub-worker
```

The swarm workers ARE matryoshka sub-specialists. Each one carries a LoRA-style delta that biases the TRM's pattern recognition for its specific reasoning domain. The deltas are learned via Shadow Copy from successful reasoning traces.

---

## The Correct Architecture: Workers as Sovereign RPN Executors

### What Workers Must Do (Sovereign)

1. **Receive** the fused entity graph from the universal four-pass (parse_bundle)
2. **Query** Galaxy Universe for relevant entries (sovereign token-matching, VRAM lookup)
3. **Select** a Layer 4 Meta-Rule (reasoning skeleton) based on `condition` RPN predicate
4. **Compose** an RPN program from the skeleton's `action` template + Galaxy entry `rpn_program` fields
5. **Execute** the composed program on Cranium's RPN stack (PUSH/POP/BRANCH/LOOP/STORE/RECALL)
6. **Return** the stack result as the worker's proposal

### What Workers Must NOT Do (Violations)

- `re.findall(pattern, text)` -- this is CPU Python regex, not sovereign
- `set(tokens) & set(other_tokens)` -- this is CPU Python set ops
- `str.lower().split()` -- this is CPU Python string processing
- Hardcoded answer strings (gamma matrices line 160-167)
- Python `for evidence in evidence_rows:` scanning with if/else branches

### The Transition Path

We can't eliminate all Python overnight -- the entire codebase orchestrates in Python. But the REASONING logic inside workers should progressively move to sovereign execution:

**Phase 1 (Immediate):** Workers compose RPN programs from Galaxy entries and evaluate them. The composition logic can remain in Python (orchestration layer). The evaluation is sovereign (RPN stack).

**Phase 2 (Next):** Workers receive their own Meta-Rule skeletons from Grammar Galaxy at init time. Worker selection of which skeleton to apply becomes a sovereign `condition` predicate evaluation.

**Phase 3 (Target):** Workers ARE Meta-Rules -- Galaxy entries with `condition` + `action` RPN programs. The swarm master iterates them sovereignly. New reasoning strategies are Grammar Galaxy entries, not Python code.

---

## Concrete Implementation: Reasoning Meta-Rules as Grammar Galaxy Entries

### Add to `foundational_operations_bootstrap.py`:

These are Layer 4 Meta-Rules stored in Grammar Galaxy, loaded at init time.

#### CoT (Chain of Thought) Skeleton

```python
{
    "id": "meta_reasoning_chain_of_thought",
    "name": "Chain of Thought Reasoning",
    "domain": "grammar",
    "category": "reasoning_meta_rule",
    "rpn_program": (
        "QUERY RECALL "
        "TASK DECOMPOSE "           # Split into sub-steps using four-pass entities
        "SUB_TASKS EACH "
        "{ "
        "  STEP_QUERY RECALL "
        "  GALAXY_NAMES RECALL "
        "  LOAD_GALAXY "            # Query Galaxy for this sub-step
        "  TOP_K 5 "
        "  FIND_SIMILAR "           # Get relevant entries
        "  RPN_PROGRAM RECALL "     # Extract entry's RPN program
        "  EVAL "                   # Evaluate on stack
        "  STORE "                  # Store sub-result for next step
        "} "
        "CHAIN_RESULTS "            # Compose all sub-results
        "VERIFY"                    # Check against query constraints
    ),
    "metadata": {
        "layer": 4,
        "skeleton_type": "sequential",
        "when_to_use": ["multi_step", "sequential_dependencies", "ordered_computation"],
        "condition_rpn": "ENTITY_COUNT 2 > SEQUENTIAL_DEPS 0 >",
        "worker_affinity": ["FormulaReasoningWorker", "ProceduralExecutionWorker"],
        "confidence": 0.90,
    },
}
```

#### Elimination Skeleton

```python
{
    "id": "meta_reasoning_elimination",
    "name": "Elimination Reasoning",
    "domain": "grammar",
    "category": "reasoning_meta_rule",
    "rpn_program": (
        "OPTIONS RECALL "
        "OPTIONS EACH "
        "{ "
        "  OPTION RECALL "
        "  EVIDENCE_ALL RECALL "
        "  OPTION EVIDENCE SUPPORT_CHECK "     # Does evidence support this option?
        "  OPTION EVIDENCE CONTRADICT_CHECK "  # Does evidence contradict this option?
        "  SUPPORT CONTRADICTION SUB "          # Net score
        "  STORE "
        "} "
        "SCORES RECALL "
        "MAX_INDEX "                            # Select highest-scoring survivor
        "OPTIONS SWAP INDEX "                   # Return the winning option
    ),
    "metadata": {
        "layer": 4,
        "skeleton_type": "eliminative",
        "when_to_use": ["multiple_choice", "concept_identification"],
        "condition_rpn": "OPTIONS_COUNT 1 >",
        "worker_affinity": ["ConceptMatchingWorker"],
        "confidence": 0.88,
    },
}
```

#### Contrastive Verification Skeleton

```python
{
    "id": "meta_reasoning_contrastive_verification",
    "name": "Contrastive Verification",
    "domain": "grammar",
    "category": "reasoning_meta_rule",
    "rpn_program": (
        "CANDIDATE RECALL "
        "EVIDENCE_ALL RECALL "
        "EVIDENCE EACH "
        "{ "
        "  ENTRY RECALL "
        "  CANDIDATE ENTRY SUPPORT_CHECK "     # +1 if entry supports candidate
        "  CANDIDATE ENTRY CONTRADICT_CHECK "  # -1 if entry contradicts
        "  ADD "                                # Accumulate
        "} "
        "TOTAL_SCORE RECALL "
        "TOTAL_SCORE 0 > "
        "{ CANDIDATE EMIT } "                  # Confirmed
        "{ CANDIDATE REJECT } "                # Falsified
        "ifelse"
    ),
    "metadata": {
        "layer": 4,
        "skeleton_type": "verification",
        "when_to_use": ["any_answer_synthesis", "post_reasoning_check"],
        "condition_rpn": "CANDIDATE_EXISTS 1 ==",
        "worker_affinity": ["EvidenceSynthesisWorker"],
        "confidence": 0.85,
    },
}
```

#### Evidence Triangulation Skeleton

```python
{
    "id": "meta_reasoning_evidence_triangulation",
    "name": "Evidence Triangulation",
    "domain": "grammar",
    "category": "reasoning_meta_rule",
    "rpn_program": (
        "QUERY RECALL "
        "GALAXY_NAMES RECALL "
        "LOAD_GALAXY "
        "TOP_K 20 FIND_SIMILAR "               # Get evidence
        "CANDIDATES EXTRACT "                   # Extract candidate answers from evidence
        "CANDIDATES EACH "
        "{ "
        "  CANDIDATE RECALL "
        "  EVIDENCE_ALL RECALL "
        "  CANDIDATE INDEPENDENT_SOURCES_COUNT " # How many independent entries confirm this?
        "  2 >= "                                # Require 2+ independent sources
        "  { CANDIDATE SCORE_BOOST } "
        "  { CANDIDATE SCORE_PENALTY } "
        "  ifelse "
        "} "
        "BEST_CANDIDATE SELECT"
    ),
    "metadata": {
        "layer": 4,
        "skeleton_type": "triangulation",
        "when_to_use": ["open_ended", "concept_identification", "factual_retrieval"],
        "condition_rpn": "EVIDENCE_COUNT 3 >",
        "worker_affinity": ["EvidenceSynthesisWorker", "ConceptMatchingWorker"],
        "confidence": 0.82,
    },
}
```

#### Dimensional Analysis Skeleton (Physics/Math)

```python
{
    "id": "meta_reasoning_dimensional_analysis",
    "name": "Dimensional Analysis",
    "domain": "grammar",
    "category": "reasoning_meta_rule",
    "rpn_program": (
        "QUERY RECALL "
        "EXTRACT_QUANTITIES "                   # Parse numbers + units from entities
        "QUANTITIES EACH "
        "{ "
        "  QUANTITY RECALL "
        "  UNIT_TYPE RECALL "                   # What type is this? (length, mass, time, etc.)
        "  STORE "
        "} "
        "FORMULA RECALL "                       # From Galaxy evidence
        "FORMULA QUANTITIES DIMENSION_CHECK "   # Are units consistent?
        "DIMENSION_OK "
        "{ FORMULA QUANTITIES EVAL } "          # Compute
        "{ DIMENSION_MISMATCH_FLAG } "
        "ifelse "
        "RESULT UNIT_ATTACH "                   # Attach units to result
    ),
    "metadata": {
        "layer": 4,
        "skeleton_type": "typed_computation",
        "when_to_use": ["physics", "engineering", "chemistry"],
        "condition_rpn": "DOMAIN physics == DOMAIN engineering == OR",
        "worker_affinity": ["FormulaReasoningWorker"],
        "confidence": 0.85,
    },
}
```

#### Procedural Decode Skeleton (Cipher/Chess/Notation)

```python
{
    "id": "meta_reasoning_procedural_decode",
    "name": "Procedural Decode Execution",
    "domain": "grammar",
    "category": "reasoning_meta_rule",
    "rpn_program": (
        "INPUT RECALL "
        "TASK_TYPE RECALL "                     # cipher, chess, notation
        "GRAMMAR_RULES RECALL "                 # Rules for this task type from Grammar Galaxy
        "GRAMMAR_RULES EACH "
        "{ "
        "  RULE RECALL "
        "  INPUT RULE APPLY "                   # Apply transformation rule
        "  RESULT STORE "
        "} "
        "RESULT VERIFY "                        # Check result validity
        "RESULT EMIT"
    ),
    "metadata": {
        "layer": 4,
        "skeleton_type": "procedural",
        "when_to_use": ["cipher", "chess", "notation", "decode"],
        "condition_rpn": "GOAL_KIND procedural ==",
        "worker_affinity": ["ProceduralExecutionWorker"],
        "confidence": 0.80,
    },
}
```

---

## How Workers Execute Meta-Rules

### Current (Sovereignty Violation)

```python
class FormulaReasoningWorker:
    def propose_open(self, *, prompt, goal, evidence_rows, ...):
        # Python regex scan (CPU, not sovereign)
        for pattern in self._FORMULA_PATTERNS:
            for match in re.findall(pattern, field_text):
                # Python string ops (CPU, not sovereign)
                candidate = " ".join(str(match).split()).strip()
                # Python set intersection (CPU, not sovereign)
                overlap = len(goal_tokens & candidate_tokens)
```

### Target (Sovereign Execution)

```python
class FormulaReasoningWorker:
    def propose_open(self, *, prompt, goal, evidence_rows, parse_bundle, ...):
        # 1. Select Meta-Rule from Grammar Galaxy (sovereign query)
        skeleton = self._select_meta_rule(parse_bundle, goal)

        # 2. Compose RPN program from skeleton + evidence entry programs
        rpn_program = self._compose_from_skeleton(
            skeleton=skeleton,
            evidence_rpn_programs=[
                entry["rpn_program"]
                for entry in evidence_rows
                if entry.get("rpn_program")
            ],
            variables=parse_bundle["fusion_parse"].get("entities", []),
        )

        # 3. Execute on Cranium RPN stack (sovereign)
        result = self._evaluate_rpn(rpn_program)

        # 4. Return sovereign result
        return [WorkerProposal(
            worker=self.name,
            candidate=result,
            score=skeleton.metadata["confidence"],
            rationale=f"meta_rule:{skeleton.meta_id}",
        )]
```

The key shift: workers COMPOSE AND EXECUTE RPN programs. They don't scan text with regex. The evidence entries already have `rpn_program` fields. The Meta-Rules already have `action` RPN templates. The worker's job is to BIND the evidence programs into the skeleton template and EVALUATE.

---

## Addressing the 5 Remaining LHE Failure Bands

### 1. Chess -- Needs Compositional Reasoning, Not Text Extraction

**Problem:** Worker scans evidence text for chess notation regex. Finds fragments. Can't compose a mating sequence.

**Fix:** Add chess tactical Meta-Rules to Grammar Galaxy:
```rpn
"meta_chess_tactical" =>
    POSITION RECALL
    PIECES_ENUMERATE
    FORCING_MOVES_GENERATE           # Checks, captures, threats
    FORCING_MOVES EACH
    {
        MOVE RECALL
        POSITION MOVE APPLY_MOVE
        OPPONENT_RESPONSES_GENERATE
        OPPONENT_RESPONSES EACH
        {
            RESPONSE RECALL
            POSITION RESPONSE APPLY_MOVE
            FOLLOWUP_MOVES_GENERATE
            CHECKMATE_CHECK
        }
    }
    MATING_SEQUENCE SELECT
```

This composes from Grammar Galaxy primitives (already have `object_extract`, `conditional_fill`, `symmetry_complete` from ARC). Chess position analysis is structurally similar to ARC grid analysis -- both are spatial pattern reasoning.

**Note:** Full chess engine is Phase 3+. For now, the Meta-Rule structures the APPROACH even when the Galaxy knowledge is sparse. As chess concept entries grow (from augmentation), the reasoning skeleton becomes more effective.

### 2. Clue Chain -- Needs Step-by-Step Composition

**Problem:** Worker tries to match full answer. Can't compose intermediate variables.

**Fix:** CoT Meta-Rule applied to clue chains. Each clue becomes a sub-task. Sub-task queries Galaxy for the specific concept. Result stored. Next clue uses stored result. This is exactly `TASK DECOMPOSE → SUB_TASKS EACH → { STORE } → CHAIN_RESULTS`.

### 3. Math (Symbolic) -- Needs Formula Composition, Not Concept Retrieval

**Problem:** Worker returns Galaxy entry descriptions ("Meaning-first canonical concept...") instead of computing.

**Fix:** Dimensional Analysis + Contrastive Verification Meta-Rules. Worker extracts `rpn_program` from evidence entries (the formulas ARE there as RPN), composes them with the query variables, evaluates on stack, verifies dimensions. The answer is the COMPUTATION result, not the entry description.

### 4. Physics -- Needs Constraint Verification

**Problem:** Worker returns "0" instead of "3". Guessed without verification.

**Fix:** Contrastive Verification Meta-Rule. After generating a candidate, check it against ALL evidence. "0" contradicts evidence that says "non-trivial moduli count". The verification step catches this.

### 5. Cipher -- Needs Procedural Execution

**Problem:** Worker recognizes "cipher" but returns concept descriptions instead of decoding.

**Fix:** Procedural Decode Meta-Rule. Worker queries Grammar Galaxy for substitution cipher rules, applies them step-by-step to the ciphertext, scores result by language model (english_score is fine as a sovereign scoring function -- it's frequency analysis, which IS cryptographic reasoning).

**Note:** The `_solve_two_step_substitution` in ProceduralExecutionWorker is actually the RIGHT approach -- frequency analysis + hill-climbing key search IS how sovereign cipher decoding works. The issue is it's implemented as a monolithic Python function instead of composed from Grammar Galaxy cipher rules. Refactor the logic INTO Grammar Galaxy entries (substitution_cipher_rule, frequency_analysis_rule, hill_climbing_rule) and compose them via the Procedural Decode Meta-Rule.

---

## Codex: Build On This

Daniel asked for your ideas. Specific questions:

1. **The `_solve_two_step_substitution` function in ProceduralExecutionWorker is actually doing real reasoning** -- frequency analysis, key search, english scoring. How much of this can move into Grammar Galaxy rules vs. how much needs to stay as procedural execution? The hill-climbing loop might need to stay procedural (it's algorithmic, not declarative), but the frequency tables and english scoring could be Galaxy entries.

2. **The `_solve_clue_chain` function has a hardcoded answer** (line 395: `if "mars closer in mass" not in prompt_lower`). This needs to become compositional. Each clue should be a sub-query to Galaxy. How would you structure the clue decomposition?

3. **Shadow Copy integration:** When a Meta-Rule skeleton produces a correct LHE answer, that success should update the specialist's routing bias (SpecialistBase.update_routing_bias) AND store the successful RPN composition as a Shadow Copy pattern. How should this feedback loop connect?

4. **Math 20/20 uses Grammar Galaxy composition → RPN → PTX stack.** Can the FormulaReasoningWorker use the SAME composition mechanism that MathSpecialist uses? If so, the worker doesn't need its own formula extraction logic -- it delegates to the existing math composition pipeline with the evidence entries as input.

5. **What reasoning patterns do you see in the remaining 8 LHE failures that I haven't captured?** You have the full diagnostics. What would YOU add?

---

## Priority Order

```
1. Add Layer 4 Meta-Rule entries to foundational_operations_bootstrap.py
   (CoT, Elimination, Contrastive Verification, Evidence Triangulation,
    Dimensional Analysis, Procedural Decode -- 6 entries)
2. Workers load their Meta-Rules at init from Grammar Galaxy
3. FormulaReasoningWorker: compose evidence rpn_programs via CoT skeleton, evaluate on stack
   (remove hardcoded gamma matrices answer)
4. ConceptMatchingWorker: use Elimination skeleton for option scoring
5. ProceduralExecutionWorker: refactor cipher logic into Grammar Galaxy rules
   composed via Procedural Decode skeleton
6. EvidenceSynthesisWorker: use Evidence Triangulation (require 2+ independent sources)
7. Rerun 10/20/10 smoke, measure LHE delta
8. If delta: Shadow Copy stores successful reasoning traces
```

---

## Constraints (Hard)

1. **Math 20/20 and ARC 10/10 MUST NOT regress.** Meta-Rules are LHE-path additions.
2. **No numpy/cupy/scipy in workers.** Workers use Galaxy queries + RPN stack.
3. **Meta-Rules are Grammar Galaxy entries.** New reasoning strategies = new Galaxy entries, not new Python code.
4. **Workers are Cranium sub-specialists** (SpecialistBase nodes in the matryoshka tree). They carry LoRA-like deltas. They learn from Shadow Copy. They're internal to K3D, not external CPU processes.
5. **Programs before opcodes** (RPN_DOMAIN_OPCODE_REGISTRY.md, Section 1). Compose reasoning from existing RPN surface. Don't add new opcodes unless absolutely necessary.

---

## Grounding References

| Spec | Section | What It Provides |
|------|---------|------------------|
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | Section 1.5 (Layer 4) | Meta-Rule dataclass: condition + action RPN programs |
| FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md | Section 3.4 | `TASK DECOMPOSE`, `SUB_TASKS EACH`, `ifelse` -- the reasoning primitives |
| THREE_BRAIN_SYSTEM_SPECIFICATION.md | Section 3.1 | Cranium = sovereign reasoning engine, zero external dependencies |
| THREE_BRAIN_SYSTEM_SPECIFICATION.md | Section 3.2 | RPN stack: PUSH/POP/BRANCH/LOOP/STORE/RECALL/LOAD_GALAXY |
| THREE_BRAIN_SYSTEM_SPECIFICATION.md | Section 3.5 | Shadow Copy: learn from successful executions |
| KNOWLEDGEVERSE_SPECIFICATION.md | Section 1.1 | One persistent PTX context, sovereignty invariants |
| RPN_DOMAIN_OPCODE_REGISTRY.md | Section 1 | "Programs before opcodes" -- compose, don't extend |
| DUAL_CLIENT_CONTRACT_SPECIFICATION.md | Section 1.6 | Save Information Principle -- reasoning in Galaxy, not code |

---

## The Principle

K3D's intelligence is **executed, not described** (THREE_BRAIN_SYSTEM_SPECIFICATION.md: "Intelligence is executed, patterns are learned"). Workers must EXECUTE reasoning through the sovereign RPN stack, not DESCRIBE it through Python string operations.

The four-layer architecture already defines where reasoning skeletons live: Layer 4 (Meta-Rules). The RPN execution engine already has the control flow: BRANCH, LOOP, STORE, RECALL, CALL, RET. The Galaxy already has the knowledge entries with `rpn_program` fields. The matryoshka specialist tree already supports spawning and routing workers.

Everything exists. The workers just need to USE the sovereign stack instead of bypassing it with Python regex. Fill Grammar Galaxy with Meta-Rule reasoning skeletons. Workers compose from them. Cranium executes them. Shadow Copy learns from successes. The system improves.

Daniel: "K3D internal swarm is cheap, and math cores are cheap." The math cores are the RPN stack. The swarm is the matryoshka specialist tree. Connect them. Make the workers think through Cranium, not through Python.
