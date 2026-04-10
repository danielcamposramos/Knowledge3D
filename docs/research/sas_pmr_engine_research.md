**Knowledge3D PMR Engine Architecture Specification**
*Sovereign GPU Pattern-Match-Replace for Term Rewriting*

---

## 1. Unification Algorithms: GPU Suitability Analysis

### 1.1 Robinson's Unification vs. Pattern Matching

**Robinson's First-Order Unification** (symmetric, bidirectional) requires:
- Occurs check (O(n) complexity per variable)
- Substitution composition (merging binding environments)
- Bidirectional variable binding

**Pattern Matching** (one-way unification) characteristics:
- Pattern DAG is the **ground structure** (or has pre-bound variables)
- Target expression DAG may contain free variables but only the pattern binds them
- No occurs check required (pattern variables don't appear in target)

**GPU SIMT Verdict:** Pattern matching wins decisively.

**Rationale:**
- **Branch divergence**: Occurs check introduces unpredictable branching; pattern matching has uniform control flow within warps when matching same pattern against different targets
- **Memory coalescence**: Pattern DAG can be kept in shared memory/constants while target expressions stream through global memory
- **Binding representation**: One-way binding uses fixed-size arrays (variable_index → node_reference) vs. Robinson's triangular substitutions requiring pointer chasing

### 1.2 E-Unification Considerations

E-unification (equality modulo theories: AC, ACU, Boolean rings) is **not recommended** for the hot-path kernel due to:
- **Non-deterministic search spaces** (branching factor explosion)
- **Theory-specific algorithms** requiring divergent code paths

**Mitigation Strategy:**
- Pre-compile E-theories into **pattern sets** at Grammar Galaxy ingestion time
- Example: `f(x,y) = f(y,x)` becomes two pattern variants `f(x,y)` and `f(y,x)` with shared rule_id
- Use **discrimination nets** (automata) compiled offline, executed as GPU state machines

### 1.3 Memory Requirements for GPU Unification Stack

Per-warp (32 threads) shared memory budget:
```
struct BindingStack {
    uint16_t var_slots[64];      // Variable index → node index mapping
    uint8_t  depth;              // Current binding count
    uint32_t trail[64];          // For backtracking: (var_idx, old_val) pairs
    uint16_t trail_head;
};
```

- **64 variables max** per pattern (sufficient for Knowledge3D semantic patterns)
- **Trail-based backtracking**: O(redex_depth) memory, not O(expression_size)
- Total per-warp: ~640 bytes (fits comfortably in 48KB shared memory with 64 warps/SM)

---

## 2. Term Rewriting Systems: Strategies for SIMT

### 2.1 Existing Engine Analysis

**Maude** (rewriting logic):
- Represents rules as labeled sequents: `label : t => t' if cond`
- Uses **internal strategy** (fixed outermost/innermost)
- AC matching via diophantine equations (too complex for GPU)

**ELAN**:
- Separates **computation** (rewriting) from **strategy** (control)
- Strategies are first-class: `repeat`, `dc` (don't care), `if-then-else`
- Compiled to **abstract strategy machine** (ASM)

**Stratego**:
- **Programmable strategies**: `topdown(s)`, `bottomup(s)`, `innermost`, etc.
- **Congruence rules**: Automatically generate traversal strategies
- Uses **ATerms** (maximal sharing) for representation

**Adaptation for Knowledge3D:**
- Adopt Stratego's **maximal sharing** (DAGs, not trees) - already native to STAR nodes
- Adopt ELAN's strategy separation: fixed strategy kernel with rule selection via Grammar Galaxy metadata

### 2.2 Innermost vs. Outermost for GPU

**Innermost (Call-by-Value):**
```
Strategy: repeat(bottomup(try(rules)))
```
- **Advantages for SIMT:**
  - All threads converge to normal forms simultaneously
  - Memory locality: arguments reduced before parents (bottom-up spatial locality)
  - No suspension/thunk management (unlike outermost)
- **Disadvantage**: May do unnecessary reductions (but GPUs have compute to spare)

**Outermost (Call-by-Need):**
- Requires **sharing preservation** and **update pointers** (black holes for lazy eval)
- Divergent memory access patterns (some threads suspended, others active)
- **Verdict**: Too complex for SIMT; requires dynamic scheduling

**Recommendation:** **Innermost with sharing** (also called "leftmost-innermost" for ordered arguments)

### 2.3 Parallel Redex Strategy

**Theoretical Model:** Match **all redexes simultaneously** across the DAG, apply non-conflicting rules in parallel.

**GPU Implementation:**
1. **Mark phase**: Kernel marks all redex positions (parallel scan)
2. **Conflict detection**: Parent-child redexes conflict (innermost takes precedence)
3. **Apply phase**: Parallel substitution at independent redexes

**Critical Pair Resolution:**
When rule A and rule B match overlapping redexes:
- Use Grammar Galaxy `superior_to` partial order
- Atomic selection: thread with highest (rule_strength, quality_score, trust_weight) wins
- Losers retry next iteration (synchronous rounds)

---

## 3. GPU Kernel Design

### 3.1 Kernel: `k3d_pattern_match`

**Signature:**
```cuda
__global__ void k3d_pattern_match(
    // Input
    const StarNode* __restrict__ expr_dag,     // Expression to match
    const PatternDAG* __restrict__ pattern,    // Pattern template
    const uint32_t start_node,                 // Root of subexpression to match
    
    // Output
    MatchResult* results,                      // Per-warp results
    uint32_t* match_count                      // Atomic counter for successful matches
);
```

**Algorithm (per warp):**
```cuda
// Shared memory: pattern cached, binding table
__shared__ PatternNode pat_cache[PATTERN_MAX_SIZE];
__shared__ uint32_t bindings[MAX_VARS];        // var_id -> node_id

// Thread 0 loads pattern into SM, broadcast to warp
if (lane_id == 0) load_pattern(pattern, pat_cache);

__syncwarp();

// Recursive matching implemented as explicit stack (no recursion in CUDA)
MatchStack stack;
push(&stack, pattern_root, start_node);

while (!empty(&stack)) {
    PatternNode* p = pop_pattern(&stack);
    StarNode* e = pop_expr(&stack);
    
    switch (p->type) {
        case PAT_VAR:
            // Variable: bind or check consistency
            uint32_t existing = atomicCAS(&bindings[p->var_id], UNBOUND, e->index);
            if (existing != UNBOUND && existing != e->index) 
                goto FAIL; // Mismatch
            break;
            
        case PAT_SYMBOL:
            if (p->meaning_class != e->meaning_class) goto FAIL;
            if (p->arity != e->arity) goto FAIL;
            // Push children (right to left for stack order)
            for (int i = p->arity - 1; i >= 0; i--) {
                push(&stack, p->children[i], e->children[i]);
            }
            break;
            
        case PAT_WILDCARD:
            // "_" matches anything, no binding
            break;
            
        case PAT_GUARD:
            // Semantic condition check (GPU-evaluable predicate)
            if (!eval_guard(p->guard_rpn, bindings)) goto FAIL;
            break;
    }
}

// Success: write bindings to global memory
uint32_t idx = atomicAdd(match_count, 1);
write_bindings(&results[idx], bindings);

FAIL:
    mark_failure(&results[threadIdx.x / 32]);
```

**Key Optimizations:**
- **Warp-wide reduction** for early failure detection (`__ballot_sync`)
- **Shared memory binding table** (64 entries × 4 bytes = 256 bytes/warp)
- **Bitmask encoding** for variables (uint64_t supports 64 variables per pattern)

### 3.2 Kernel: `k3d_rule_apply`

**Signature:**
```cuda
__global__ void k3d_rule_apply(
    const ReplacementRPN* behavior,      // Replacement template (postfix)
    const BindingTable* bindings,        // From pattern_match
    const StarNode* source_dag,            // Original for shared subtrees
    
    // Output
    StarNode* new_dag,                     // Pre-allocated pool
    uint32_t* new_root_index               // Result pointer
);
```

**Algorithm (RPN evaluation in shared memory):**
```cuda
// Each warp builds one replacement expression
__shared__ StarNode* build_stack[STACK_SIZE];
__shared__ uint32_t node_counter;

int sp = 0;
for (int i = 0; i < behavior->length; i++) {
    RPNToken tok = behavior->tokens[i];
    
    switch (tok.opcode) {
        case OP_PUSH_VAR:
            // Look up binding
            StarNode* bound = resolve_binding(tok.var_id, bindings);
            build_stack[sp++] = bound;
            break;
            
        case OP_CONSTRUCT:
            // Pop 'arity' nodes, construct new node
            uint32_t first_child = sp - tok.arity;
            
            // Allocate new node in new_dag (atomicAdd on node_counter)
            uint32_t new_idx = atomicAdd(&node_counter, 1);
            StarNode* new_node = &new_dag[new_idx];
            
            // Copy metadata from pattern behavior
            new_node->meaning_class = tok.meaning_class;
            new_node->arity = tok.arity;
            
            // Set children (pointer stitching)
            for (int j = 0; j < tok.arity; j++) {
                new_node->children[j] = build_stack[first_child + j];
            }
            
            sp = first_child;
            build_stack[sp++] = new_node;
            break;
            
        case OP_COPY_SUBTREE:
            // Deep copy with sharing preservation
            StarNode* copied = copy_subtree(source_dag, tok.node_ref, new_dag);
            build_stack[sp++] = copied;
            break;
    }
}

// Result is sole stack item
*new_root_index = build_stack[0]->index;
```

**Memory Management:**
- **Arena allocation**: Pre-allocate 1MB per block for new nodes
- **Hash consing** (optional): Use warp-level hash table to preserve maximal sharing during construction
- **Compaction**: Parallel prefix sum to eliminate unused allocation slots

### 3.3 Rule Priority & Conflict Resolution

**Grammar Galaxy Metadata per rule:**
```cuda
struct GrammarRule {
    float quality_score;        // Match quality (0.0 - 1.0)
    int8_t rule_strength;       // +1 (strict), 0 (defeasible), -1 (defeater)
    uint16_t superior_to[8];    // Rule IDs this defeats (partial order)
    float trust_weight;         // Provenance confidence
    uint32_t rule_id;
};
```

**Conflict Resolution Kernel (`k3d_resolve_conflicts`):**
```cuda
// Input: Array of potential matches (rule_id, position, quality)
// Output: Bitmask of accepted matches (non-conflicting, highest priority)

// Phase 1: Group by position (redexes at same node)
// Phase 2: Within each group, sort by defeasible logic:
//   1. rule_strength: +1 > 0 > -1
//   2. superior_to: Check if any candidate defeats others (transitive closure precomputed)
//   3. trust_weight: Higher wins
//   4. quality_score: Higher wins

// Phase 3: Non-overlapping check (parent-child conflicts)
//   If parent and child both matched, keep child (innermost priority)
```

**Defeasible Logic Implementation:**
- Precompute **superiority matrix** (N×N boolean, N=max rules per meaning_class) in constant memory
- **Strict rules** (+1) automatically override defeasible (0) regardless of other metrics
- **Defeaters** (-1) block application of conflicting rules but don't apply themselves

---

## 4. Grammar Galaxy Integration

### 4.1 Defeasible Logic Connection

The Grammar Galaxy's `GrammarRule` structure (extended per Christoph Dorn's defeasible logic specification) maps directly to PMR conflict resolution:

**Dung's Abstract Argumentation Framework** adaptation:
- Each match is an **argument**
- `superior_to` defines **attack** relation
- **Grounded semantics**: Accept matches not defeated by accepted matches

**GPU-Optimized Calculation:**
Since full argumentation semantics is P-complete (hard for GPU), use **stratified approximation**:
1. **Stratum 0**: Strict rules (+1) - apply if match exists
2. **Stratum 1**: Defeasible rules not attacked by accepted strict rules
3. **Stratum 2**: Apply trust_weight ranking within stratum

### 4.2 Integration Architecture

```
Grammar Galaxy (Host/CPU)
    |
    v
k3d_grammar_query (OP 0xE9) → Returns active rule set for meaning_class
    |
    v
Pattern Match Kernel (OP 0x22C)
    - Loads pattern_rpn from GrammarRule
    - Matches against expression DAG
    - Outputs: (rule_id, bindings, quality_score)
    |
    v
Conflict Resolution Kernel
    - Applies defeasible logic using rule metadata
    - Filters overlapping redexes
    |
    v
Rule Apply Kernel (OP 0x22D)
    - Uses behavior_rpn from selected GrammarRule
    - Constructs new expression DAG
    |
    v
Result committed to STAR heap
```

### 4.3 Quality Score Calculation

During `k3d_pattern_match`, compute `quality_score` via:
```cuda
float quality = 1.0f;
// Structural similarity bonus
quality *= structural_similarity(pattern, expression);

// Variable binding specificity (prefer concrete matches)
quality *= (1.0f - (num_wildcards / pattern_size));

// Semantic proximity (if using MeaningCentricStar embeddings)
quality *= cosine_similarity(pattern.embedding, expression.embedding);
```

---

## 5. Implementation Roadmap

### Phase 1: Core PMR (Weeks 1-4)
- Implement `k3d_pattern_match` for ground patterns (no variables)
- Simple replacement without sharing preservation
- Single rule application per kernel launch

### Phase 2: Variable Binding & Backtracking (Weeks 5-6)
- Add PAT_VAR handling with trail-based backtracking
- Shared memory optimization for binding tables
- Support for non-linear patterns (same var multiple times)

### Phase 3: Defeasible Resolution (Weeks 7-8)
- Integration of Grammar Galaxy metadata
- Conflict resolution kernel
- Innermost strategy enforcement

### Phase 4: Optimization (Weeks 9-10)
- Discrimination nets for pattern indexing
- Persistent kernel for continuous rewriting
- Memory pool management for STAR node allocation

---

## Appendix: PTX-Level Optimizations

**Register Pressure Management:**
- Keep `pattern DAG` in constant memory (cached)
- Use `.reg .u64` for STAR node pointers (64-bit addressing)
- Unroll match loops for patterns < 32 nodes (common case)

**Memory Coalescing:**
- Store STAR nodes in **Structure of Arrays** (SoA) format:
  ```cuda
  struct StarNodeSoA {
      uint32_t* meaning_class;  // Separate array for each field
      uint16_t* arity;
      uint32_t** children;      // Indirect
  };
  ```

**Warp Primitives:**
```cuda
// Early termination if any thread in warp fails
int failed = __ballot_sync(0xFFFFFFFF, mismatch);
if (failed != 0) goto next_candidate;
```

This architecture provides a **sovereign** (Python-independent) GPU PMR engine capable of handling Knowledge3D's defeasible transformation rules at scale.