# Adaptive Reasoning Budget Specification — Ternary-Gated Recursive Computation Governance

**Version**: 1.0
**Status**: Candidate Standard (K3D Canonical Vocabulary)
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)
**Date**: March 26, 2026

---

## Abstract

This specification defines the **Adaptive Reasoning Budget** (ARB) — the computational governance mechanism that controls how deeply, how broadly, and for how long the TRM reasons about a query before emitting an answer. Unlike fixed-depth architectures (standard Transformers with L layers, classical RNNs with T time steps), the ARB dynamically allocates reasoning resources based on the **ternary knowledge signal** (+1/0/−1) carried by the knowledge the TRM encounters during navigation.

The core insight: **some knowledge is settled (+1) and requires shallow verification; some knowledge is uncertain (0) and requires moderate exploration; some knowledge is contradictory or deep (−1) and requires recursive decomposition into sub-tasks, each with their own budgets.** The system must never halt prematurely on hard problems (minimum budget enforcement), and must never waste computation on trivial ones (early termination for converged answers).

When recursive decomposition produces more sub-tasks than the parallel swarm can execute simultaneously, the system transitions from parallel to serial execution with priority scheduling — always emitting intermediate results as symlinked stars into the Knowledgeverse, so that no computation is wasted and future queries benefit from prior reasoning.

**Normative References:**
- Hyper-Parallel Processing Specification v1.0 (docs/vocabulary/HYPER_PARALLEL_PROCESSING.md)
- Three Brain System Specification v1.1 (docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md)
- Knowledgeverse Specification v5.1 (docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md)
- Formal Ontology Specification v1.0 (docs/vocabulary/FORMAL_ONTOLOGY_SPECIFICATION.md)
- Sovereign NSI Specification v2.0 (docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md)
- RPN Domain Opcode Registry v0.1 (docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md)
- Avatar Embodiment Specification v1.0 (docs/vocabulary/AVATAR_EMBODIMENT_SPECIFICATION.md)

**Informative References:**
- Graves, A. (2016). "Adaptive Computation Time for Recurrent Neural Networks." arXiv:1603.08983
- Banino, A., Balaguer, J., Blundell, C. (2021). "PonderNet: Learning to Ponder." arXiv:2107.05407
- Russell, S. & Wefald, E. (1991). "Principles of Metareasoning." Artificial Intelligence 49(1-3)
- Simon, H. (1955). "A Behavioral Model of Rational Choice." QJE 69(1)
- Zilberstein, S. (1996). "Using Anytime Algorithms in Intelligent Systems." AI Magazine 17(3)
- Blumofe, R.D. & Leiserson, C.E. (1999). "Scheduling Multithreaded Computations by Work Stealing." JACM 46(5)
- Kleene, S.C. (1952). "Introduction to Metamathematics." (strong three-valued logic)
- Kocsis, L. & Szepesvari, C. (2006). "Bandit Based Monte-Carlo Planning." ECML
- Laird, J.E. (2022). "Introduction to the Soar Cognitive Architecture." arXiv:2205.03854
- Nau, D. et al. (2003). "SHOP2: An HTN Planning System." JAIR 20

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [The Ternary Knowledge Signal](#3-the-ternary-knowledge-signal)
4. [Budget Computation Model](#4-budget-computation-model)
5. [Minimum Budget Enforcement](#5-minimum-budget-enforcement)
6. [Recursive Sub-Task Decomposition](#6-recursive-sub-task-decomposition)
7. [Parallel Saturation and Priority Serialization](#7-parallel-saturation-and-priority-serialization)
8. [Knowledge Persistence: Intermediate Results as Stars](#8-knowledge-persistence-intermediate-results-as-stars)
9. [Integration with the Composed Head Pipeline](#9-integration-with-the-composed-head-pipeline)
10. [Halting Gate Extension](#10-halting-gate-extension)
11. [Memory Pressure Governance](#11-memory-pressure-governance)
12. [Formal Definitions](#12-formal-definitions)
13. [RPN Budget Opcodes](#13-rpn-budget-opcodes)
14. [Normative Invariants](#14-normative-invariants)
15. [Conformance](#15-conformance)
16. [Future Extensions](#16-future-extensions)

---

## 1. Introduction

### 1.1 The Problem: Fixed-Depth Reasoning Is Wrong

Standard neural architectures apply a fixed amount of computation to every input:
- A Transformer with L layers applies exactly L attention + feedforward passes per token
- A standard RNN applies exactly T steps per sequence element
- Most LLM reasoning chains have a fixed token budget or no budget at all

This is architecturally wrong. "What is 2+3?" requires a single Galaxy lookup. "Prove that every even number greater than 2 is the sum of two primes" requires recursive decomposition, multi-hop navigation across Galaxy neighborhoods, specialist collaboration, and potentially thousands of reasoning steps. Applying the same computation to both is either wasteful (over-computing the easy case) or insufficient (under-computing the hard case).

### 1.2 The K3D Solution: Ternary-Gated Adaptive Budget

K3D solves this with three interconnected mechanisms:

**Mechanism 1: Ternary-Gated Minimum Budget**
Every knowledge node the TRM encounters during navigation carries a ternary assertion state (+1/0/−1). This signal determines the minimum computation budget allocated to the query:

| Signal | Knowledge State | Budget Regime | Analogy |
|--------|----------------|---------------|---------|
| +1 | Affirmed, well-connected, confident | **Shallow**: B_base steps, early termination allowed | Chess: clearly winning position — play fast |
| 0 | Uncertain, sparse connections, novel | **Moderate**: 2 × B_base steps, standard convergence required | Chess: balanced position — think normally |
| −1 | Contradictory, deeply nested, complex | **Deep**: 4 × B_base steps, full dialectical exploration | Chess: losing or tactically sharp — think deeply |

**Mechanism 2: Recursive Sub-Task Decomposition**
When the budget for a query exceeds a decomposition threshold (the query is too complex for a single reasoning pass), the system decomposes the query into sub-tasks — each with their own ternary-gated budget. Sub-tasks can recursively decompose further, creating a tree of reasoning tasks bounded only by the problem's intrinsic complexity.

**Mechanism 3: Parallel Saturation → Priority Serialization**
The nine-chain swarm provides fixed parallelism. When recursive decomposition generates more sub-tasks than the swarm can handle simultaneously, excess tasks are serialized into a priority queue. Tasks on the critical path of the dependency DAG get highest priority. Off-critical-path tasks wait. This graceful degradation ensures that the system never deadlocks and always makes progress on the most important reasoning chains.

### 1.3 The Persistence Invariant

ALL intermediate results — from sub-task decomposition, from partial swarm convergence, from serialized queue processing — are persisted as symlinked stars in the Knowledgeverse. No computation is ephemeral. Every reasoning step produces knowledge that future queries can reuse, implementing the SOAR cognitive architecture's "chunking" principle: **slow deliberative reasoning crystallizes into fast cached knowledge**.

---

## 2. Theoretical Foundations

### 2.1 Adaptive Computation Time (Graves, 2016)

Alex Graves demonstrated that recurrent neural networks can learn when to stop computing. A scalar halting unit h_t accumulates across time steps until crossing a threshold:

```
H_t = Σ_{i=1}^{t} h_i
halt at N = min{t : H_t ≥ 1}
```

The output is a weighted mean-field aggregation over all steps, with a "ponder cost" regularizer τ that biases toward fewer steps. **K3D's ternary signal is a discretized, knowledge-grounded version of this halting signal** — where the halting probability is determined not by a learned scalar but by the ontological state of the knowledge being navigated.

### 2.2 PonderNet and Geometric Priors (Banino et al., 2021)

PonderNet reformulates adaptive halting as a probabilistic model with a geometric prior distribution over halting steps. The expected computation depth is 1/λ_p, modulated by the input. **K3D maps this directly**: the ternary signal sets λ_p:

| Signal | λ_p | Expected Steps | Distribution |
|--------|-----|---------------|-------------|
| +1 | High (e.g., 0.5) | ~2 (shallow) | Geometric, steep decay |
| 0 | Medium (e.g., 0.2) | ~5 (moderate) | Geometric, moderate decay |
| −1 | Low (e.g., 0.05) | ~20 (deep) | Geometric, long tail |

### 2.3 Metareasoning: Reasoning About How Much to Reason (Russell & Wefald, 1991)

Stuart Russell and Eric Wefald formalized the principle that an agent should reason about the **value of computation** (VOC) before deciding whether to compute more:

```
VOC(C_j) = E[U(α* | result(C_j))] − U(α*)
```

The agent should continue computing if and only if VOC exceeds the cost of computation. **In K3D, the ternary signal is a pre-computed VOC estimate**: +1 knowledge has low VOC (further computation unlikely to improve the answer), while −1 knowledge has high VOC (further computation likely to resolve contradictions or deepen understanding).

### 2.4 Bounded Rationality and Satisficing (Simon, 1955)

Herbert Simon's satisficing principle: agents search until finding a solution that meets a minimum aspiration level, rather than searching for the optimal solution. **The minimum budget enforcement in K3D is a formalization of satisficing** — the system MUST compute at least B_min steps before it is allowed to halt, preventing premature termination on problems that appear easy but have hidden depth.

### 2.5 Anytime Algorithms (Zilberstein, 1996)

An anytime algorithm can be stopped at any point with a valid result, with quality monotonically improving over time. **Each swarm chain in K3D is an anytime reasoner** — it produces progressively better candidates at each iteration. The Halting Gate is the interruptibility mechanism that decides when the collective result is "good enough."

Zilberstein distinguishes:
- **Interruptible algorithms**: Can be stopped at ANY time. The K3D swarm is interruptible.
- **Contract algorithms**: Must know the total budget in advance. Sub-tasks with allocated budgets are contract algorithms.

### 2.6 Hierarchical Task Networks (Nau et al., 2003)

HTN planning decomposes compound tasks into subtasks via decomposition methods, recursively, until reaching primitive operators. **K3D's recursive sub-task decomposition is HTN planning applied to reasoning**: when a query is too complex for direct resolution, it decomposes into sub-queries (compound → subtask), each resolved by Galaxy navigation and swarm reasoning (primitive operators).

### 2.7 SOAR Subgoaling and Chunking (Laird, 2022)

The SOAR cognitive architecture automatically creates subgoals when an impasse arises (insufficient knowledge to proceed). After resolving the subgoal, SOAR "chunks" the solution as a new production rule for future reuse. **K3D implements the same pattern**: when reasoning stalls (budget exceeded, convergence not reached), the system decomposes into sub-tasks. Resolved sub-tasks are persisted as Galaxy stars — the chunked knowledge that prevents future impasses on the same pattern.

### 2.8 Kleene's Three-Valued Logic (K3) for Partial Computation

Kleene's strong three-valued logic models partial recursive functions where computation may not terminate. The third value U (undefined) represents "computation has not yet produced a result." **K3D's ternary assertion state maps directly to Kleene's K3**:

| K3D Ternary | Kleene K3 | Computation Meaning |
|-------------|-----------|-------------------|
| +1 (affirmed) | T (true) | Computation complete, result confident |
| 0 (unknown) | U (undefined) | Computation incomplete, more steps may help |
| −1 (negated) | F (false) | Computation reveals contradiction, deeper analysis required |

The "infectious" behavior of U in Kleene conjunction (T ∧ U = U) models how uncertainty propagates: if ANY knowledge along a reasoning chain is uncertain, the overall conclusion inherits uncertainty, triggering deeper computation.

### 2.9 MCTS Budget Allocation (Kocsis & Szepesvari, 2006)

Monte Carlo Tree Search allocates simulation budget across tree branches using the UCT formula:

```
UCT(node) = Q(node)/N(node) + C × √(ln(N(parent)) / N(node))
```

The first term (exploitation) favors promising nodes; the second term (exploration) favors under-visited nodes. **K3D's ternary signal modulates the exploration constant C**: −1 knowledge gets high C (explore deeply), +1 knowledge gets low C (exploit known paths).

---

## 3. The Ternary Knowledge Signal

### 3.1 Signal Sources

The ternary signal that gates computation depth comes from multiple sources, combined via Kleene conjunction:

**Source 1: Star Assertion State**
Every MeaningCentricStar carries a `ternary_state` field (+1/0/−1). When the TRM navigates to a star, its assertion state contributes to the budget computation.

**Source 2: Swarm Convergence State**
The Halting Gate's convergence assessment is ternary: converged (+1), still processing (0), diverging (−1). A diverging swarm signals that more computation is needed.

**Source 3: Galaxy Neighborhood Density**
Dense neighborhoods (many interconnected stars with high meaning mass) signal well-known territory (+1). Sparse neighborhoods signal novel or poorly-understood territory (0). Neighborhoods with high contradiction density signal contested knowledge (−1).

**Source 4: Cross-Domain Reference Depth**
Queries that require crossing multiple Galaxy domains (e.g., physics → chemistry → biology) carry higher intrinsic complexity. Each domain crossing increments the depth signal toward 0 or −1.

**Source 5: Defeasible Rule Chains**
When the defeasible resolver encounters long override chains (rule A defeats rule B which defeats rule C...), the chain length signals reasoning depth. Short chains → +1. Long chains → −1.

### 3.2 Signal Aggregation

The composite ternary signal σ(q) for a query q is computed by Kleene conjunction over all signal sources:

```
σ(q) = σ_star ∧_K σ_convergence ∧_K σ_density ∧_K σ_cross_domain ∧_K σ_defeasible
```

where ∧_K is Kleene's strong conjunction:
- +1 ∧_K +1 = +1 (all sources affirm → shallow budget)
- +1 ∧_K 0 = 0 (any uncertainty → moderate budget)
- +1 ∧_K −1 = −1 (any contradiction → deep budget)
- 0 ∧_K 0 = 0 (compound uncertainty → moderate budget)
- 0 ∧_K −1 = −1 (uncertainty + contradiction → deep budget)
- −1 ∧_K −1 = −1 (compound contradiction → deep budget)

This means the composite signal is **dominated by the most challenging source** — a single contradictory signal forces deep reasoning, matching the principle that a chain is only as strong as its weakest link.

### 3.3 Signal Refinement During Reasoning

The ternary signal is NOT static. It is recomputed at each iteration of the TRM game loop:

```
Iteration 1: σ(q) = −1 (initial contradiction detected)
             → Budget allocated: 4 × B_base
             → Swarm begins deep exploration

Iteration 3: Sub-contradiction resolved, new σ(q) = 0
             → Budget reduced: 2 × B_base remaining
             → Swarm narrows focus

Iteration 5: All sources converge, new σ(q) = +1
             → Budget satisfied, Halting Gate may fire
             → If quality threshold met: emit answer
```

This dynamic signal refinement prevents over-computation: the system starts deep when needed but shortcuts when the knowledge landscape clears.

---

## 4. Budget Computation Model

### 4.1 Base Budget

The base budget B_base is calibrated from empirical convergence data:

**Empirical baseline** (Sovereign NSI Specification validation): 87% of queries converge within 5 iterations; 9-iteration reasoning completes in 80.69µs.

```
B_base = 5 iterations  (87th percentile convergence point)
```

### 4.2 Ternary Budget Scaling

The allocated budget B(q) for query q with composite signal σ(q):

```
B(q) = B_base × 2^(1 − σ(q))
```

| σ(q) | B(q) | Iterations | Rationale |
|------|------|-----------|-----------|
| +1 | B_base × 2^0 = B_base | 5 | Well-known knowledge: verify and emit |
| 0 | B_base × 2^1 = 2 × B_base | 10 | Uncertain knowledge: explore and converge |
| −1 | B_base × 2^2 = 4 × B_base | 20 | Contradictory/deep: full dialectical analysis |

### 4.3 Budget Partitioning Across Swarm Workers

The nine-chain swarm partitions the budget:

```
Per-worker budget = B(q) / num_active_workers
```

But this is a soft partition — workers that converge early yield their remaining budget to workers still processing (work-stealing). The total computation is bounded by B(q) × num_active_workers, but no individual worker is forced to use its full allocation.

### 4.4 Adaptive Worker Activation

Not all nine workers are needed for every query (Hyper-Parallel Processing §3):

| σ(q) | Active Workers | Rationale |
|------|---------------|-----------|
| +1 | 1–3 | Simple retrieval or verification; few specialists needed |
| 0 | 3–6 | Moderate exploration; several domain specialists |
| −1 | 6–9 | Full swarm deployment for maximum coverage |

The number of active workers is the minimum of the signal-suggested count and the number of relevant specialists for the query's domain. A pure math query does not activate the audio specialist regardless of signal.

---

## 5. Minimum Budget Enforcement

### 5.1 The Anti-Premature-Halt Guarantee

The minimum budget is the most critical constraint: **the system MUST NOT halt before B_min steps**, even if early results appear satisfactory. This prevents a known failure mode in adaptive computation: the system learns to halt immediately on hard problems because early (wrong) answers have low loss during training.

```
B_min(q) = max(B_base, B(q) × min_fraction)
```

where `min_fraction` is a tunable parameter (default: 0.5). For σ(q) = −1:

```
B_min = max(5, 20 × 0.5) = 10 iterations minimum
```

The Halting Gate MUST NOT fire before iteration B_min, regardless of convergence score.

### 5.2 Formal Satisficing Threshold

The satisficing threshold (Simon, 1955) determines when the system has computed "enough":

```
function SATISFICE(query, budget):
    aspiration = aspiration_level(σ(query))
    for iteration = 1 to budget:
        result = TRM_STEP(query)
        if iteration >= B_min AND result.confidence >= aspiration:
            return result  // Satisficed: quality meets aspiration
    return result  // Budget exhausted: return best effort
```

Aspiration levels by ternary signal:

| σ(q) | Aspiration Level | Meaning |
|------|-----------------|---------|
| +1 | 0.70 | Moderate confidence sufficient (knowledge is reliable) |
| 0 | 0.85 | Higher confidence needed (knowledge is uncertain) |
| −1 | 0.95 | Near-certainty required (knowledge is contradictory) |

---

## 6. Recursive Sub-Task Decomposition

### 6.1 Decomposition Trigger

When the allocated budget B(q) exceeds a decomposition threshold T_decomp, the query is too complex for direct resolution and MUST be decomposed:

```
T_decomp = 4 × B_base = 20 iterations (default)
```

This threshold is reached when σ(q) = −1 at multiple stages of navigation, compounding the budget requirement beyond a single-pass capacity.

### 6.2 The Decomposition Process

Decomposition follows the HTN (Hierarchical Task Network) pattern:

```
function ADAPTIVE_REASON(query, budget, depth):
    // Phase 1: Attempt direct resolution
    result = SWARM_REASON(query, min(budget, T_decomp))

    if result.σ == +1:
        PERSIST_AS_STAR(query, result)
        return result  // Direct resolution succeeded

    // Phase 2: Budget exceeded or convergence failed → decompose
    if budget > T_decomp OR result.σ <= 0:
        subtasks = DECOMPOSE(query, result.partial_knowledge)
        sub_results = []

        for each subtask in subtasks:
            σ_sub = EVALUATE_SIGNAL(subtask)
            B_sub = B_base × 2^(1 − σ_sub)

            // Recursive: subtasks can decompose further
            sub_result = ADAPTIVE_REASON(subtask, B_sub, depth + 1)
            sub_results.append(sub_result)

            // Persist intermediate result as star
            PERSIST_AS_STAR(subtask, sub_result)

        // Phase 3: Compose sub-results
        composed = COMPOSE(sub_results, query)
        PERSIST_AS_STAR(query, composed)
        return composed
```

### 6.3 Decomposition Strategies

The DECOMPOSE function uses domain-specific strategies:

**Mathematical decomposition**: Break a proof into lemmas. Each lemma is a sub-task with its own budget. Lemma dependencies form a DAG.

**Multi-hop decomposition**: A question requiring knowledge from domains A, B, and C becomes three domain-specific sub-queries plus a composition query.

**Dialectical decomposition**: A contradictory assertion (σ = −1) decomposes into: (1) find arguments FOR, (2) find arguments AGAINST, (3) evaluate defeater chains, (4) synthesize judgment.

**Temporal decomposition**: A question about a process decomposes into initial state → transitions → final state, each queried independently.

### 6.4 Maximum Recursion Depth

To prevent unbounded decomposition, a maximum recursion depth D_max is enforced:

```
D_max = 8 (default)
```

At depth D_max, no further decomposition occurs — the system MUST produce a result from available knowledge, annotated with σ = 0 (uncertain) if convergence was not achieved.

### 6.5 Sub-Task Dependency Graphs

Sub-tasks may have dependencies: sub-task B requires the result of sub-task A. These dependencies form a Directed Acyclic Graph (DAG):

```
Query: "What is the boiling point of a novel alloy of copper and zinc?"

Sub-task DAG:
    [A] Look up copper properties (σ=+1, B=5)
    [B] Look up zinc properties (σ=+1, B=5)
    [C] Compute alloy phase diagram (σ=0, B=10) ← depends on A, B
    [D] Derive boiling point from phase diagram (σ=−1, B=20) ← depends on C

Critical path: A → C → D (or B → C → D), length = 35 iterations
Parallel path: A and B execute simultaneously
```

---

## 7. Parallel Saturation and Priority Serialization

### 7.1 The Saturation Problem

The nine-chain swarm provides fixed parallelism. When recursive decomposition generates N sub-tasks where N > 9 (or N > active_workers), the system cannot execute all sub-tasks in parallel.

### 7.2 Work-Stealing Phase (Soft Saturation)

Before serialization, the system attempts work-stealing (Blumofe & Leiserson, 1999):

```
function WORK_STEAL(swarm_workers, pending_tasks):
    // Workers maintain double-ended queues (deques)
    // Local work: push/pop from BOTTOM (LIFO — depth-first)
    // Steal: from TOP (FIFO — breadth-first)

    for each idle_worker in swarm_workers:
        if idle_worker.deque.is_empty():
            victim = random_choice(busy_workers)
            stolen_task = victim.deque.steal_from_top()
            if stolen_task is not None:
                idle_worker.execute(stolen_task)
```

Work-stealing naturally balances load: workers that finish early steal from overloaded workers. This extends effective parallelism without serialization overhead.

### 7.3 Hard Saturation: Priority Queue

When work-stealing cannot absorb all pending tasks (all workers busy, all deques have only one task), the system transitions to priority-queued serialization:

```
function SERIALIZE_OVERFLOW(active_chains, pending_tasks):
    // Step 1: Critical path analysis
    critical_path = FIND_CRITICAL_PATH(pending_tasks.dependency_DAG)

    // Step 2: Priority assignment
    for each task t in pending_tasks:
        t.priority = compute_priority(t)

    // Step 3: Queue by priority
    priority_queue.insert_all(pending_tasks)

    // Step 4: As workers complete, they pull from queue
    on_worker_complete(worker):
        if not priority_queue.is_empty():
            next_task = priority_queue.pop()
            worker.execute(next_task)
```

### 7.4 Priority Computation

Task priority is computed from multiple factors:

```
priority(t) = w_crit × on_critical_path(t)
            + w_signal × (1 − σ(t))        // Harder tasks get higher priority
            + w_depth × depth(t)             // Deeper decomposition = more invested
            + w_depend × num_dependents(t)   // Tasks others wait on = higher priority
            + w_budget × remaining_budget(t) // Tasks close to completion = higher priority
```

Default weights: w_crit = 10.0, w_signal = 3.0, w_depth = 1.0, w_depend = 5.0, w_budget = 2.0.

### 7.5 Serialization Guarantees

**Guarantee 1 (Progress):** At least one worker is always executing a task. The system never deadlocks.

**Guarantee 2 (Priority Ordering):** Higher-priority tasks execute before lower-priority tasks (pre-emption is not used — tasks run to completion or budget exhaustion).

**Guarantee 3 (Bounded Latency):** The maximum latency for any serialized task is bounded by:

```
T_max = Σ(B(t_i)) for all higher-priority tasks + B(t) for the task itself
```

### 7.6 Distributed Saturation (Future)

When local GPU saturation is reached AND the K3D system has access to distributed resources (other Houses via Doors protocol, cloud-hosted K3D instances on the same server), sub-tasks can be distributed:

```
if local_saturation AND distributed_resources_available:
    for each serialized_task in priority_queue:
        if task.can_distribute():  // No sovereignty violation
            remote_house = find_available_house()
            remote_house.submit(task)
            // Result arrives via Door as symlinked star
```

This is the software-as-a-space paradigm: each K3D House has its own AI avatar, and Houses can collaborate on reasoning tasks by sharing sub-tasks via the Doors protocol. The distributed result is received as a symlinked star in the originating House's Knowledgeverse — maintaining the persistence invariant.

---

## 8. Knowledge Persistence: Intermediate Results as Stars

### 8.1 The Persistence Invariant

**Every intermediate reasoning result MUST be persisted as a MeaningCentricStar** (or symlink reference to an existing star) in the Knowledgeverse. This is not optional — it is a normative invariant of the Adaptive Reasoning Budget.

Rationale: In SOAR, chunking converts deliberative reasoning into cached production rules, dramatically accelerating future similar reasoning. In K3D, persisting intermediate results as Galaxy stars serves the same function — future queries that traverse the same knowledge neighborhood will find pre-computed results with σ = +1, halting immediately rather than re-deriving.

### 8.2 Star Generation from Sub-Task Results

```
function PERSIST_AS_STAR(query, result):
    star = MeaningCentricStar(
        meaning_rpn = result.reasoning_program,
        visual_rpn = result.visualization_program,
        star_id = hash(result.meaning_rpn),  // Content-addressed
        layer = determine_layer(result),
        ternary_state = result.σ,
        confidence = result.confidence,
        domain = result.galaxy_domain,
        provenance = {
            source: "adaptive_reasoning",
            parent_query: query.id,
            depth: result.decomposition_depth,
            budget_used: result.iterations_used,
            budget_allocated: result.iterations_allocated,
            worker_id: result.swarm_worker_id,
        },
        taxonomy_refs = result.discovered_taxonomy_links,
        symlink_refs = result.referenced_stars,
    )

    // Check for existing star (content-addressed dedup)
    existing = Galaxy.find_by_star_id(star.star_id)
    if existing:
        // Merge: strengthen confidence, add symlink refs
        existing.confidence = max(existing.confidence, star.confidence)
        existing.symlink_refs.merge(star.symlink_refs)
    else:
        Galaxy.insert(star)

    return star
```

### 8.3 Cross-Domain Bridge Discovery

When sub-task results from different domains reference the same star, a cross-domain bridge is automatically created:

```
Sub-task A (Math domain): discovers star_∑ (summation symbol)
Sub-task B (Statistics domain): discovers star_∑ (same summation symbol)
Sub-task C (Finance domain): discovers star_∑ (same summation symbol)

Bridge: star_∑ is referenced by Math, Statistics, AND Finance
→ New cross-domain discovery star created:
   "summation_cross_domain_bridge" with refs to all three domains
```

This is the Compositional Ontology's cross-domain discovery (Formal Ontology Specification §11.3) emerging naturally from the Adaptive Reasoning Budget's persistence invariant.

### 8.4 Reasoning Trace Persistence

The full reasoning trace (which Galaxy neighborhoods were visited, which sub-tasks were created, which priorities were assigned, which results were composed) is persisted as a NavigationTrace star:

```
NavigationTrace(
    seed_stars = [query embedding location in Galaxy],
    hops = [(star_a, star_b, distance, method), ...],
    decomposition_tree = {root: query, children: [subtask_1, subtask_2, ...]},
    budget_profile = {allocated: 20, used: 14, signal_history: [−1, 0, +1]},
    worker_assignments = {worker_3: subtask_1, worker_7: subtask_2, ...},
)
```

This trace is available for sleep-time consolidation: the TRM learns which decomposition strategies worked for which query types, improving future budget allocation.

---

## 9. Integration with the Composed Head Pipeline

### 9.1 Budget in the Pipeline

The Adaptive Reasoning Budget integrates at the TRM game loop level, wrapping the existing composed head pipeline:

```
Composed Head Pipeline (existing):
    Morton → LED-A* → Frustum → LOD → Nine-Chain Swarm → Halting Gate

Adaptive Reasoning Budget (new wrapper):
    1. EVALUATE signal σ(q) from knowledge landscape
    2. COMPUTE budget B(q) = B_base × 2^(1−σ(q))
    3. SET B_min = max(B_base, B(q) × min_fraction)
    4. FOR iteration = 1 to B(q):
         a. RUN composed head pipeline (one tick)
         b. CHECK Halting Gate convergence
         c. IF iteration >= B_min AND converged: EMIT answer
         d. RECOMPUTE σ(q) from updated knowledge state
         e. IF σ(q) changed: ADJUST remaining budget
    5. IF budget exceeded AND not converged:
         a. DECOMPOSE into sub-tasks
         b. ALLOCATE sub-budgets
         c. SCHEDULE parallel/serial execution
         d. COMPOSE sub-results
         e. PERSIST all intermediates as stars
    6. EMIT final answer
```

### 9.2 Budget Registers

The budget state is maintained in STORE/RECALL registers accessible to all swarm workers:

```
STORE register 60: budget_allocated    (int32)
STORE register 61: budget_remaining    (int32)
STORE register 62: budget_min          (int32)
STORE register 63: composite_signal    (trit: +1/0/−1)
STORE register 64: decomposition_depth (int32)
STORE register 65: active_workers      (int32)
STORE register 66: pending_subtasks    (int32)
STORE register 67: completed_subtasks  (int32)
```

These registers are ternary-ready: when ternary hardware arrives, registers 63 natively carries the trit without binary encoding overhead.

---

## 10. Halting Gate Extension

### 10.1 Current Halting Gate (Before ARB)

The existing Halting Gate (Sovereign NSI Specification §9.2, `gre_multimodal_halting_gate` kernel) computes convergence from:

```
convergence = top_score × gap × agreement × dimensional_consistency
```

And fires when convergence exceeds a static threshold.

### 10.2 Extended Halting Gate (With ARB)

The Adaptive Reasoning Budget extends the Halting Gate with three additional inputs:

```
function EXTENDED_HALTING_GATE(
    // Existing inputs
    top_score, gap, agreement, dimensional_consistency,
    // ARB inputs
    budget_remaining, budget_min, composite_signal
) -> (halt: bool, convergence_state: trit):

    base_convergence = top_score × gap × agreement × dimensional_consistency

    // Minimum budget enforcement
    if iterations_used < budget_min:
        return (halt=false, convergence_state=0)  // Still processing

    // Aspiration-adjusted threshold
    aspiration = ASPIRATION_LEVEL(composite_signal)
    if base_convergence >= aspiration:
        return (halt=true, convergence_state=+1)  // Converged

    // Budget exhaustion
    if budget_remaining <= 0:
        if base_convergence >= 0.5:
            return (halt=true, convergence_state=0)  // Partial convergence
        else:
            return (halt=false, convergence_state=−1)  // Diverging → trigger decomposition

    // Continue processing
    return (halt=false, convergence_state=0)
```

### 10.3 Convergence State Semantics

The Halting Gate now emits a ternary convergence state that feeds back into the budget system:

| Convergence State | Meaning | Action |
|-------------------|---------|--------|
| +1 | Converged: answer meets aspiration | Emit answer, persist result as star |
| 0 | Processing: budget remaining, not yet converged | Continue reasoning |
| −1 | Diverging: budget exhausted, no convergence | Trigger recursive decomposition |

---

## 11. Memory Pressure Governance

### 11.1 Watermark Integration

The Knowledgeverse's region watermarks (KNOWLEDGEVERSE_SPECIFICATION §3.3) constrain the Adaptive Reasoning Budget:

| Watermark | Budget Adjustment | Rationale |
|-----------|------------------|-----------|
| GREEN (< 70% VRAM) | Full budget: B(q) as computed | Normal operation |
| YELLOW (70–85%) | Reduce budget by 25%: B(q) × 0.75 | Memory pressure — shorter reasoning |
| ORANGE (85–92%) | Reduce budget by 50%: B(q) × 0.50 | High pressure — prioritize completion |
| RED (> 92%) | Minimum budget only: B_min | Emergency — emit best effort immediately |

### 11.2 Decomposition Depth Capping Under Pressure

| Watermark | D_max Adjustment |
|-----------|-----------------|
| GREEN | D_max = 8 (full depth) |
| YELLOW | D_max = 4 (half depth) |
| ORANGE | D_max = 2 (shallow only) |
| RED | D_max = 0 (no decomposition) |

### 11.3 Worker Count Capping Under Pressure

| Watermark | Max Active Workers |
|-----------|-------------------|
| GREEN | 9 (full swarm) |
| YELLOW | 6 |
| ORANGE | 3 |
| RED | 1 (single-threaded reasoning) |

---

## 12. Formal Definitions

### 12.1 Adaptive Reasoning Budget (ARB)

**Definition:** An Adaptive Reasoning Budget is a tuple ARB = (B_base, σ, B_min, T_decomp, D_max) where:
- B_base ∈ ℕ⁺ is the base iteration count (default: 5)
- σ: Q → {+1, 0, −1} is the composite ternary signal function over queries
- B_min: Q → ℕ⁺ is the minimum budget function: B_min(q) = max(B_base, B(q) × min_fraction)
- T_decomp ∈ ℕ⁺ is the decomposition threshold (default: 4 × B_base)
- D_max ∈ ℕ⁺ is the maximum decomposition depth (default: 8)

### 12.2 Budget Allocation Function

**Definition:** The budget allocation function B: Q × {+1, 0, −1} → ℕ⁺ is:

```
B(q) = B_base × 2^(1 − σ(q))
```

### 12.3 Decomposition Tree

**Definition:** A decomposition tree T(q) is a rooted tree where:
- The root is the original query q
- Each internal node is a compound query that was decomposed
- Each leaf is a primitive query resolved by direct Galaxy navigation
- Each edge (parent, child) is labeled with the decomposition strategy used
- depth(T) ≤ D_max

### 12.4 Priority Function

**Definition:** The priority function π: Task → ℝ is:

```
π(t) = w_crit × 𝟙[t ∈ critical_path]
     + w_signal × (1 − σ(t))
     + w_depth × depth(t)
     + w_depend × |dependents(t)|
     + w_budget × (B(t) − remaining(t)) / B(t)
```

### 12.5 Persistence Function

**Definition:** The persistence function P: (Query, Result) → MeaningCentricStar maps every (query, result) pair to a Galaxy star with content-addressed identity:

```
P(q, r) = star with star_id = hash(r.meaning_rpn)
```

---

## 13. RPN Budget Opcodes

### 13.1 Opcode Classification

Per the RPN Domain Opcode Registry's admission pipeline (§6), ARB opcodes are classified as follows:

| Opcode | Class | Stage | Description |
|--------|-------|-------|-------------|
| `OP_BUDGET_ALLOC` | B | Recipe | Compute B(q) from ternary signal; store in register 60 |
| `OP_BUDGET_CHECK` | B | Recipe | Check if budget_remaining > 0; branch accordingly |
| `OP_BUDGET_DECOMPOSE` | B | Recipe | Trigger sub-task decomposition; push subtasks to queue |
| `OP_BUDGET_PERSIST` | B | Recipe | Persist current result as star; emit symlink |
| `OP_PRIORITY_COMPUTE` | B | Recipe | Compute priority π(t) for scheduling |

All are Class B (representable now as RPN recipes using existing `OP_LIMIT`, `OP_STORE`, `OP_RECALL`, `OP_BRANCH`). Promotion to Class A (PTX kernel) follows the admission pipeline when usage frequency and performance justify it.

### 13.2 Recipe Compositions

```rpn
// OP_BUDGET_ALLOC recipe:
RECALL_composite_signal    // σ(q) from register 63
1 SWAP SUB                 // 1 − σ(q)
2 SWAP POW                 // 2^(1 − σ(q))
RECALL_B_base MUL          // B_base × 2^(1 − σ(q))
STORE_budget_allocated     // → register 60
DUP 0.5 MUL               // B(q) × 0.5
RECALL_B_base MAX          // max(B_base, B(q) × 0.5)
STORE_budget_min           // → register 62

// OP_BUDGET_CHECK recipe:
RECALL_budget_remaining    // register 61
0 GT                       // budget > 0?
RECALL_iterations_used     // how many so far
RECALL_budget_min          // minimum required
GTE                        // iterations >= B_min?
AND                        // both conditions
? IF                       // if budget remaining AND minimum met
    RECALL_convergence     // check convergence score
    RECALL_aspiration      // check aspiration threshold
    GTE                    // converged?
THEN
```

---

## 14. Normative Invariants

### 14.1 Minimum Budget Invariant

**The Halting Gate MUST NOT emit an answer before B_min iterations have been executed.** No shortcut, no override, no exception. This prevents the known failure mode of premature termination on hard problems that superficially resemble easy ones.

### 14.2 Persistence Invariant

**Every intermediate result from sub-task decomposition MUST be persisted as a MeaningCentricStar** (or merged into an existing star) in the Knowledgeverse. Ephemeral computation that disappears after the query is answered is an architectural violation.

### 14.3 Priority Monotonicity Invariant

**Tasks on the critical path of the dependency DAG MUST have higher priority than off-critical-path tasks.** No scheduling policy may violate critical-path priority — this ensures that serialization does not increase total latency beyond the minimum possible.

### 14.4 Ternary Signal Dominance Invariant

**The composite ternary signal σ(q) is computed by Kleene conjunction.** Any single −1 source forces the composite to −1. The system MUST NOT average, vote, or otherwise dilute the most challenging signal.

### 14.5 Sovereignty Invariant

**All ARB computation — budget allocation, signal aggregation, decomposition, scheduling, persistence — MUST execute on sovereign substrates** (PTX kernels, RPN stacks, Galaxy VRAM). No Python orchestration in the budget loop. The budget loop IS part of the hot path.

### 14.6 Decomposition Depth Bound Invariant

**Recursive decomposition MUST NOT exceed D_max levels.** At D_max, the system produces a best-effort answer with appropriate confidence annotation. Unbounded recursion is a sovereignty violation (infinite computation is not sovereign).

### 14.7 Progress Invariant

**At least one swarm worker MUST be executing at all times during active reasoning.** The system MUST NOT enter a state where all workers are idle while tasks remain pending.

---

## 15. Conformance

### 15.1 Conformance Levels

**Level A: Budget-Aware Reasoning**
A conforming Level A system MUST:
- Compute ternary-gated budgets per §4
- Enforce minimum budget per §5
- Extend the Halting Gate per §10
- Persist final results as stars per §8

**Level B: Recursive Decomposition**
Level A plus:
- Decompose queries when budget exceeds T_decomp per §6
- Construct sub-task dependency DAGs
- Enforce maximum recursion depth D_max
- Persist ALL intermediate results as stars

**Level C: Parallel Governance**
Level B plus:
- Implement work-stealing for soft saturation per §7.2
- Implement priority serialization for hard saturation per §7.3
- Compute priorities per §7.4
- Respect memory watermarks per §11

**Level D: Distributed Reasoning**
Level C plus:
- Distribute sub-tasks to remote Houses via Doors protocol per §7.6
- Receive distributed results as symlinked stars
- Maintain provenance across distributed decomposition

### 15.2 Validation Tests

| Test | Description | Validates |
|------|-------------|-----------|
| T1 | Query with σ=+1 halts within B_base iterations | Budget scaling (§4) |
| T2 | Query with σ=−1 does NOT halt before B_min iterations | Minimum budget enforcement (§5) |
| T3 | Complex query decomposes into sub-tasks at depth > 0 | Recursive decomposition (§6) |
| T4 | Sub-task results persist as retrievable stars | Persistence invariant (§8) |
| T5 | 20 simultaneous sub-tasks serialize correctly on 9 workers | Priority serialization (§7) |
| T6 | Critical path tasks complete before off-path tasks | Priority monotonicity (§7.4) |
| T7 | Under RED watermark, budget reduces to B_min | Memory pressure governance (§11) |
| T8 | Decomposition depth never exceeds D_max | Depth bound invariant (§14.6) |
| T9 | Future query reuses persisted intermediate star | Knowledge caching (§8) |
| T10 | Full budget loop executes on sovereign substrates only | Sovereignty invariant (§14.5) |

---

## 16. Future Extensions

### 16.1 Learned Budget Allocation

The current budget function B(q) = B_base × 2^(1−σ(q)) is a hand-designed exponential. Future work should allow the TRM to LEARN the budget allocation via shadow copy: recording (query_type, σ(q), iterations_used, result_quality) tuples during inference and adjusting B_base and the scaling exponent during sleep-time consolidation. This implements PonderNet's learned halting on K3D's sovereign substrate.

### 16.2 Specialist-Specific Budgets

Currently, all swarm workers share the same budget. Future work should allow specialist-specific budgets: the math specialist may need more iterations for a calculus proof than the language specialist needs for a word lookup. The ternary signal per specialist's Galaxy neighborhood determines per-specialist budget allocation.

### 16.3 Ternary-Native Budget Registers

On ternary hardware, the composite signal register (register 63) natively carries a balanced trit (−1/0/+1) without binary encoding. The budget scaling function 2^(1−σ(q)) computes natively on trit-vector ALUs:
- σ = +1 → exponent = 0 → multiplier = 1 (single trit operation)
- σ = 0 → exponent = 1 → multiplier = 2 (shift operation)
- σ = −1 → exponent = 2 → multiplier = 4 (double shift)

### 16.4 Collective Budgets (SHGI)

When multiple TRM agents collaborate in shared Galaxy (SHGI), the budget system extends to collective reasoning: total budget is distributed across agents, not just workers. One agent may handle the math sub-task while another handles the physics sub-task, each with their own local budgets, serialization queues, and persistence to shared Galaxy.

### 16.5 Budget Visualization in House

The Adaptive Reasoning Budget is visualizable in the House via the avatar's Cranial Galaxy:
- Budget allocation appears as a glowing sphere around the avatar's head, its radius proportional to B(q)
- Sub-task decomposition appears as branching traces from the sphere
- Serialized tasks appear as queued dots orbiting the sphere
- Convergence appears as trace merging
- Budget exhaustion appears as sphere dimming

This makes the reasoning process inspectable by human observers — maintaining the Dual-Client transparency commitment.

---

## Appendix A: Empirical Calibration Data

From Sovereign NSI Specification validation (Phase G):

| Metric | Value | Source |
|--------|-------|--------|
| 87th percentile convergence | 5 iterations | gre_multimodal_halting_gate validation |
| 100th percentile convergence | 9 iterations | Same |
| 9-iteration latency | 80.69µs | Same |
| Per-iteration cost | ~8.97µs | Derived |
| B_base recommendation | 5 iterations | 87th percentile |
| B(σ=−1) maximum | 20 iterations | 4 × B_base |
| Expected latency at B_max | ~179.4µs | 20 × 8.97µs |

## Appendix B: Comparison with Related Approaches

| System | Computation Control | Ternary Signal | Decomposition | Persistence |
|--------|-------------------|---------------|---------------|-------------|
| **ACT (Graves)** | Learned halting scalar | No (continuous) | No | No |
| **PonderNet** | Probabilistic halting | No (continuous) | No | No |
| **Universal Transformer** | Per-position ACT | No | No | No |
| **SOAR** | Impasse → subgoal | No (binary: impasse/no) | Yes (automatic) | Yes (chunking) |
| **MCTS** | UCT budget allocation | No (continuous UCT) | Yes (tree expansion) | Partial (tree retained) |
| **Chess engines** | Time management | No (heuristic) | Yes (iterative deepening) | No |
| **K3D ARB** | Ternary-gated exponential | **Yes (+1/0/−1)** | **Yes (recursive HTN)** | **Yes (all intermediates as stars)** |

K3D's ARB is unique in combining all four properties: ternary signal gating, recursive decomposition, parallel-to-serial degradation, and full knowledge persistence of intermediate results.
