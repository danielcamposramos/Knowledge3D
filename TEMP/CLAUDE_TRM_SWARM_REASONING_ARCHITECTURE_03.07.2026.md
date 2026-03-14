# Claude Architecture Directive: TRM Swarm Reasoning -- Base Model + Specialist Delegation

**Date:** March 7, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Context:** ARC 10/10, Math 20/20, LHE 1/10. Foundational corpus added 5,565 entries. LHE did NOT improve. Knowledge density is necessary but not sufficient. The real gap: K3D finds relevant evidence but cannot REASON with it. Daniel identified the missing architectural pattern.

---

## Daniel's Insight

"Remember this 'head' is supposed to be composed of base model + specialists that are crafted as needed by the own model. We might need to craft some initially so he has an example, but you got the idea -- an internal swarm that the main model can spawn and delegate and later sum to actually answer, just like you do Claude."

This is the key. Daniel is describing what Claude does internally: decompose a problem into sub-tasks, delegate each to a focused reasoning thread, then fuse the results. K3D's TRM should do the same thing -- using the matryoshka specialist tree that already exists.

---

## Why Knowledge Density Alone Failed

The diagnostics tell the story. After adding 2,048 concept entries:

| Question | Evidence Found | Predicted | Correct | Problem |
|----------|---------------|-----------|---------|---------|
| Chess (mate in 2) | 12 | "" (empty) | "Rxf3, Rf1#" | No chess reasoning specialist |
| Philosophy | 19 | "Weak Non-Sadism" | "Weak Non-Sadism" | WORKS -- token match sufficient |
| Trivia | 12 | "Here" | "yeyo" | No reasoning, grabbed random word |
| Math (bordism) | 40 | "Meaning-first canonical concept..." | "Z+Z+Z+Z+Z" | **Returned Galaxy entry description** |
| Math (elliptic curves) | 20 | "Meaning-first canonical concept..." | "18" | **Returned Galaxy entry description** |
| Math (Lie algebra) | 52 | "Generating random samples..." | "$1 + 3x + ...$" | **Returned PDF extract text** |
| Physics (gamma matrices) | 39 | "n = {x = (xl9" | "$-((d-2k)^2)+d$" | Garbage -- no formula synthesis |
| Physics (compactification) | 50 | "0" | "3" | Guessed, didn't reason |
| Cybersecurity (cipher) | 12 | "Linux systems using..." | "Katie kicked..." | No cipher decoding, returned concept description |
| Math (Sobolev spaces) | 32 | "1E3 , and suppose..." | "$\mathcal{A}^{\alpha-}$" | Returned truncated evidence text |

**The pattern is clear:** K3D finds 12-52 relevant evidence entries per question. But the synthesis step (`_synthesize_lhe_open_answer`) does regex pattern matching over evidence TEXT, extracting surface-level candidates. It doesn't REASON with the evidence -- it just grabs text fragments.

**What's needed:** The TRM should decompose the question, delegate sub-tasks to specialist reasoning threads, and fuse their outputs into an answer. This is exactly the swarm pattern Daniel described.

---

## Existing Infrastructure (Already Built, Underused)

K3D already has the building blocks:

### 1. SpecialistBase (specialist_base.py) -- The Matryoshka Tree

```
SpecialistBase
├── name, domain, level
├── children: dict[str, SpecialistBase]     # sub-specialists
├── routing_bias: dict[str, float]          # learned routing weights
├── delta: SpecialistDelta (rank, alpha, seed, scale)  # LoRA-like adapter
├── spawn_child(name, domain, rank)         # CREATE new sub-specialist
├── route(query, domain_hint)               # DELEGATE to best child
├── mark_query(success)                     # LEARN from outcome
└── effective_delta_chain()                 # COMPOSE parent+child deltas
```

This is a self-similar fractal specialist tree with LoRA-like composition. It already supports spawning children, routing queries, and learning from outcomes.

### 2. SpecialistSpawner (specialist_spawner.py) -- Autonomous Spawning

```
SpecialistSpawner
├── observe(parent, query, confidence, success)   # Track subdomain pressure
├── _evaluate(parent, bucket, metrics)             # Decide when to spawn
├── frequency_threshold: 100 queries -> spawn      # High-frequency trigger
├── low_confidence_threshold: 0.6                  # Performance gap trigger
└── persist()                                      # Save spawning state
```

This already monitors query patterns and auto-spawns specialist children when a subdomain gets enough traffic OR shows poor confidence. But it's purely reactive -- it spawns from HISTORY, not from the current query.

### 3. NavigatorSpecialist (navigator_specialist.py) -- Multi-Path Exploration

```
NavigatorSpecialist
├── plan_routes()                          # Generate multiple route candidates
├── explore_multi_path()                   # Query + compose each route
├── compose_paths()                        # Fuse best candidates
├── navigate_and_compose()                 # Full pipeline
└── forward/backward/fusion reading paths  # Four-pass decomposition
```

This explores multiple specialist routes IN PARALLEL and composes the best result. But it composes ROUTES (which specialist to use), not REASONING (what the specialist concluded).

### 4. Matryoshka Routing in Daemon (main.py)

The diagnostics show `matryoshka_specialist` and `matryoshka_level: 2` in every route. The daemon already routes through the specialist tree. But the matryoshka specialist is used ONLY for routing, not for reasoning delegation.

---

## The Architecture: Swarm Reasoning via Specialist Delegation

### Core Principle

When the TRM encounters a question it cannot answer with a single specialist pass, it should:

1. **Decompose** the question into sub-tasks (already done by four-pass)
2. **Spawn** (or select) specialist workers for each sub-task
3. **Delegate** each sub-task to its worker
4. **Collect** results from all workers
5. **Fuse** results into a final answer

This is NOT a new system. It's activating the existing matryoshka infrastructure for REASONING, not just ROUTING.

### Concrete Example: Physics Question

**Question:** "Consider antisymmetrized gamma matrices... What is the trace?"

**Current flow (broken):**
```
1. Four-pass -> entities: [gamma_matrices, antisymmetrized, trace, dimensions]
2. Route -> MechanicsSpecialist, galaxies: [Reality, Grammar, Tool, 3DObjects, Math]
3. Query Galaxy -> 39 evidence entries about gamma matrices, Clifford algebras
4. Synthesize -> regex extracts "n = {x = (xl9" from evidence text
5. Return garbage
```

**Swarm flow (proposed):**
```
1. Four-pass -> entities: [gamma_matrices, antisymmetrized, trace, dimensions]
2. Route -> MechanicsSpecialist (parent)
3. Decompose into sub-tasks:
   a. "What is gamma matrix antisymmetrization?" -> spawn/select AlgebraSpecialist
   b. "What is the trace formula for antisymmetrized products?" -> spawn/select FormulaSpecialist
   c. "How does dimension d affect the result?" -> spawn/select DimensionalAnalysisWorker
4. Each worker:
   a. AlgebraSpecialist: queries Math+Reality Galaxy, finds Clifford algebra entries,
      extracts: "gamma_{mu1...muk} = gamma_{[mu1}...gamma_{muk]}"
   b. FormulaSpecialist: queries Grammar Galaxy for trace computation rules,
      composes RPN: "D 2 K MUL SUB SQ NEG D ADD"
   c. DimensionalAnalysisWorker: queries Reality Galaxy for dimension constraints,
      extracts: "d = spacetime dimension, k = number of indices"
5. Fuse: combine formula from (b) with variables from (a) and (c)
6. Return: "-((d - 2k)^2) + d"
```

### Concrete Example: Chess Question

**Question:** "Black to move. Which sequence is mate in 2?"

**Swarm flow:**
```
1. Four-pass -> entities: [black, move, mate, 2, chess_position]
2. Route -> ChessSpecialist (spawned or pre-built)
3. Decompose:
   a. "Parse the chess position" -> PositionParser worker
   b. "Find forcing moves for black" -> TacticsWorker
   c. "Verify mate in 2" -> VerificationWorker
4. Workers use Grammar Galaxy chess notation rules + Reality Galaxy chess concepts
5. Fuse: verified mating sequence
```

### Concrete Example: Cipher Question

**Question:** "Decipher the two-step substitution cipher: BD QZOT..."

**Swarm flow:**
```
1. Four-pass -> entities: [decipher, two-step, substitution_cipher, ciphertext]
2. Route -> CryptographySpecialist
3. Decompose:
   a. "What is a two-step substitution cipher?" -> ConceptWorker
   b. "Apply frequency analysis to ciphertext" -> FrequencyAnalysisWorker
   c. "Apply substitution mapping" -> DecodingWorker
4. Workers compose Grammar Galaxy cipher rules into RPN program
5. Fuse: decoded plaintext
```

---

## Implementation Design

### Phase 1: Pre-Built Seed Specialists (Immediate)

Build 3-4 seed specialists as EXAMPLES for the spawner to learn from. These demonstrate the swarm pattern at a concrete level.

#### Seed Specialist 1: FormulaReasoningWorker

**Purpose:** When evidence contains mathematical formulas/relationships, extract and compose them into an answer.

**What it does:**
- Receives: fused entities from four-pass + evidence entries from Galaxy query
- Scans evidence for `rpn_program` fields (mathematical formulas)
- Composes relevant programs using Grammar Galaxy composition rules
- Evaluates composed program on PTX stack
- Returns: computed result or formula expression

**Why this helps LHE:** 4/10 questions are math. Current synthesis grabs text descriptions. This worker would use the `rpn_program` fields to compute actual answers.

#### Seed Specialist 2: ConceptMatchingWorker

**Purpose:** When the question asks "which concept" or "what is", find the best-matching Galaxy concept star and extract its canonical name.

**What it does:**
- Receives: fused entities + evidence entries
- Scores evidence entries by semantic overlap with question entities (not just token overlap)
- Uses symlink navigation: Word Galaxy -> Reality Galaxy concept -> canonical name
- Returns: the concept's canonical name or definition

**Why this helps LHE:** The Philosophy question (1/10 correct) works because concept matching happens to align. This worker would make it systematic for ALL concept-identification questions.

#### Seed Specialist 3: ProceduralExecutionWorker

**Purpose:** When the question requires DOING something (decode a cipher, play a chess move, compute a value), execute a procedural program.

**What it does:**
- Receives: fused entities + Grammar Galaxy rules relevant to the task
- Identifies the TASK TYPE from entity roles (decode, compute, enumerate, verify)
- Composes a multi-step RPN program from Grammar Galaxy primitives
- Executes the program on the PTX stack
- Returns: execution result

**Why this helps LHE:** The cipher question, chess question, and physics computation questions all require EXECUTION, not just text extraction.

#### Seed Specialist 4: EvidenceSynthesisWorker

**Purpose:** When multiple evidence entries contribute partial information, synthesize a coherent answer from their CONTENT fields (not metadata).

**What it does:**
- Receives: ranked evidence entries from Galaxy query
- Extracts: `content`, `description`, `rpn_program`, `metadata.definition` from entries
- Filters: removes Galaxy infrastructure text (entry IDs, category names)
- Composes: merges partial information from multiple entries
- Returns: synthesized answer text

**Why this helps LHE:** Current synthesis returns "Meaning-first canonical concept for..." (Galaxy metadata). This worker would extract the actual knowledge from entry CONTENT fields.

### Phase 2: Swarm Dispatch in Daemon (After Seed Specialists Work)

Modify `_dispatch_lhe_task` to use the swarm pattern:

```python
def _dispatch_lhe_task(self, *, route, task, use_enriched):
    parse_bundle = self._collect_parse_bundle(...)
    evidence = self._query_galaxy_evidence(parse_bundle)

    # Identify which workers are relevant
    workers = self._select_workers(parse_bundle, evidence)

    # Delegate to each worker
    worker_results = []
    for worker in workers:
        result = worker.process(
            parse_bundle=parse_bundle,
            evidence=evidence,
            task=task,
        )
        worker_results.append(result)

    # Fuse worker results
    answer = self._fuse_worker_results(worker_results, parse_bundle)
    return answer
```

### Phase 3: Autonomous Spawning (When Pattern Proves)

Once the seed specialists demonstrate value, the SpecialistSpawner can learn to create new workers from query patterns. A new domain with enough queries and low confidence automatically spawns a specialist. The seed specialists serve as the TEMPLATE for what a specialist looks like.

---

## What This Changes

### Current: Single-Specialist Monolithic Path
```
Question -> Four-Pass -> Route -> Single Specialist -> Galaxy Query -> Regex Synthesis -> Answer
```

### Proposed: Swarm Delegation Path
```
Question -> Four-Pass -> Route -> Parent Specialist
                                    ├── Worker 1: Formula Reasoning
                                    ├── Worker 2: Concept Matching
                                    ├── Worker 3: Procedural Execution
                                    └── Worker 4: Evidence Synthesis
                                    -> Fusion -> Answer
```

The parent specialist DECIDES which workers to activate based on the fused entity graph from four-pass. Not all workers run for every query. A math formula question activates FormulaReasoningWorker + EvidenceSynthesisWorker. A concept question activates ConceptMatchingWorker. A procedural question activates ProceduralExecutionWorker.

---

## What NOT to Change

1. **The four-pass decomposition stays universal.** Workers receive the fused entity graph. They don't reimplement passes 1-3.
2. **Galaxy query stays sovereign.** Workers query Galaxy Universe through the existing TRM interface. No external APIs.
3. **Math 20/20 must not regress.** The MathSpecialist path is independent. Don't touch it.
4. **ARC 10/10 must not regress.** The ARC adapter path is independent. Don't touch it.
5. **Workers are lightweight.** Each worker is a thin function that processes evidence in a specific way. They are NOT full ML models or LLM calls. They are PROCEDURAL -- compose RPN, execute on PTX, navigate Galaxy.

---

## Connection to Existing Specs

| Spec | Section | Relevance |
|------|---------|-----------|
| THREE_BRAIN_SYSTEM_SPECIFICATION.md | Cranium (TRM) | TRM = base model + specialist adapters, fractal hierarchy |
| KNOWLEDGEVERSE_SPECIFICATION.md | Region 5 | TRM navigates, specialists process, results compose |
| specialist_base.py | spawn_child() | Matryoshka pattern already supports fractal worker creation |
| specialist_spawner.py | observe() | Autonomous spawning from performance signals |
| navigator_specialist.py | explore_multi_path() | Multi-path exploration already runs multiple specialists |

---

## Priority Order

```
1. Build seed FormulaReasoningWorker (immediate -- addresses 4/10 math failures)
2. Build seed ConceptMatchingWorker (immediate -- addresses philosophy/trivia failures)
3. Build seed ProceduralExecutionWorker (addresses cipher/chess failures)
4. Build seed EvidenceSynthesisWorker (addresses garbage-text synthesis)
5. Wire swarm dispatch into _dispatch_lhe_task
6. Rerun 10/20/10 smoke, measure LHE delta
7. If delta positive: let SpecialistSpawner learn to create more workers autonomously
```

---

## The Principle

K3D finds the right evidence. It just can't think with it yet. The four-pass decomposes. The Galaxy stores. But between "found 40 relevant entries" and "the answer is Z+Z+Z+Z+Z" there needs to be REASONING -- and reasoning is what specialists do.

The base model routes. The specialists reason. The swarm fuses. Just like Claude does internally -- decompose, delegate, fuse. K3D already has the tree (SpecialistBase), the spawner (SpecialistSpawner), and the multi-path explorer (NavigatorSpecialist). What's missing is connecting these to the LHE synthesis step so that evidence becomes computation, not text extraction.

Daniel's insight: "an internal swarm that the main model can spawn and delegate and later sum to actually answer." That's the architecture. Build the seed specialists. Let the system learn to spawn more.
