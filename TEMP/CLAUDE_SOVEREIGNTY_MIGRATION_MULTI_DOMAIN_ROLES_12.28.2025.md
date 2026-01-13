# Sovereignty Migration: Multi-Domain Role Extraction — December 28, 2025

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Context**: Multi-domain role extraction validated (72 linalg + 40 geo artifacts), now migrate to sovereign architecture

---

## Executive Summary

**Current State**: Multi-domain semantic role extraction implemented in Python + Ollama (Option A)
- 6 domain tiers (~100+ roles total)
- Book metadata weighting (10×) + equation cues (2×) + context keywords (1×)
- Validated on 2 domains: Linear Algebra (58.5% domain-specific) + Geometry (77.0% domain-specific)

**Target State**: Sovereign implementation using PTX + Galaxy + TRM (zero external dependencies in hot path)
- Domain patterns stored in Math Galaxy + Grammar Galaxy
- TRM learns to navigate Galaxy for role inference
- Cranium PTX kernels execute pattern matching
- No Python/Ollama in inference path

**Migration Path**: 3-phase approach (Ingestion → Hybrid → Sovereign)

---

## Current Python Implementation (What Works)

### 1. Multi-Domain Taxonomy (6 Tiers)

**Location**: `knowledge3d/training/math_benchmarks/sovereign_knowledge_articulator.py:60-180`

```python
# Tier 1A: Geometry (25 roles)
GEOMETRY_ROLES = [
    "radius", "diameter", "height", "width", "length", "area", "volume",
    "leg", "hypotenuse", "base", "angle", "circumference", ...
]

# Tier 1B: Linear Algebra (18 roles)
LINEAR_ALGEBRA_ROLES = [
    "vector", "component", "magnitude", "matrix", "eigenvalue",
    "dimension", "basis", "span", ...
]

# Tier 1C: Calculus (18 roles)
CALCULUS_ROLES = [
    "derivative", "integral", "limit", "gradient", ...
]

# Tier 1D: Physics (20 roles)
PHYSICS_ROLES = [
    "velocity", "acceleration", "force", "energy", "mass", ...
]

# Tier 1E: Number Theory (13 roles)
NUMBER_THEORY_ROLES = [
    "prime", "factor", "modulus", "remainder", ...
]

# Tier 1F: Probability & Statistics (15 roles)
STATISTICS_ROLES = [
    "mean", "variance", "probability", "distribution", ...
]

# Tier 2: Formula Components (8 roles)
FORMULA_ROLES = [
    "exponent", "coefficient", "numerator", "denominator", ...
]

# Tier 3: Generic Fallbacks (4 roles)
GENERIC_ROLES = [
    "constant", "variable", "placeholder", "unknown"
]
```

**Total**: ~121 distinct roles across 6 mathematical domains

### 2. Domain Detection Strategy

**Location**: `sovereign_knowledge_articulator.py:220-290`

**Weighted Scoring**:
- **Book metadata hint**: 10× (from `--domain` parameter)
- **Equation structural cues**: 2× (det, ∫, ∂, ||x||, π, etc.)
- **Context keywords**: 1× (circle, vector, derivative, velocity, etc.)

**Example**:
```python
def _detect_domain(self, context: str, equation: str, book_domain_hint: str = None) -> List[str]:
    scores = {}

    # Book metadata (highest priority)
    if book_domain_hint == "linear_algebra":
        scores["linear_algebra"] = 10

    # Equation cues (medium priority)
    if "||" in equation or "∥" in equation:
        scores["linear_algebra"] += 2

    # Context keywords (low priority)
    if "vector" in context.lower():
        scores["linear_algebra"] += 1

    # Returns: ["linear_algebra", "geometry", ...] (sorted by score)
```

**Validated Results**:
- Linear algebra book → detects "linear_algebra" (10 + cues) → 58.5% linalg roles ✅
- Geometry book → detects "geometry" (10 + cues) → 77.0% geometry roles ✅

### 3. Role Prioritization

**Location**: `sovereign_knowledge_articulator.py:300-350`

**Logic**:
1. Primary domain roles appear first (e.g., linear algebra book → vector, component, magnitude, ...)
2. Secondary domain roles appear next (detected from context)
3. All remaining Tier 1 roles (lower priority)
4. Tier 2 formula components
5. Tier 3 generic fallbacks (last resort)

**Example** (Linear Algebra context):
```
Prioritized roles: [
    # Primary (linear_algebra detected)
    "vector", "component", "magnitude", "matrix", "eigenvalue", ...

    # Secondary (if geometry also detected)
    "radius", "area", ...

    # Remaining Tier 1
    "derivative", "velocity", "prime", "mean", ...

    # Tier 2
    "exponent", "coefficient", ...

    # Tier 3
    "constant", "variable", "unknown"
]
```

### 4. Alias Canonicalization

**Location**: `sovereign_knowledge_articulator.py:185-205`

**Mappings**:
```python
ROLE_ALIASES = {
    "norm": "magnitude",
    "component_1": "component",
    "component_i": "component",
    "avg": "mean",
    "average": "mean",
    "stddev": "standard_deviation",
    "mod": "modulus",
    "speed": "velocity",
}
```

**Purpose**: LLM might return synonyms ("norm" vs "magnitude"), canonicalize before validation

### 5. Multi-Domain Prompt Construction

**Location**: `sovereign_knowledge_articulator.py:380-520`

**Structure**:
```python
def _build_prompt_multidomain(...) -> str:
    """
    Returns prompt with:
    - Domain detection hints (detected: linear_algebra → expect vector, matrix, ...)
    - Top 12-15 prioritized roles for context
    - 6 cross-domain examples (same structure, different roles)
    - Reasoning chain (step-by-step analysis)
    - Avoid unrelated domains line
    """
```

**Key Example** (validates architectural principle):
```
Example: a² + b² = c²

Geometry context     → a=leg, b=leg, c=hypotenuse
Physics context      → a=velocity_x, b=velocity_y, c=velocity_magnitude
Linear Algebra ctx   → a=component_1, b=component_2, c=norm
```

**This proves**: Structure persists, semantics vary by context, ONE model understands BOTH.

---

## Sovereign Architecture Mapping (Where It Belongs)

### Math Galaxy (Knowledge Storage)

**What to Store**:
```python
# Domain signatures (RPN programs that detect domain from context)
{
    "symbol_id": "domain_signature_linear_algebra",
    "rpn_program": [
        # Check for keywords: matrix, vector, determinant, etc.
        # Check equation for: ||, ∥, det(, trace
        # Return domain score
    ],
    "metadata": {
        "domain": "linear_algebra",
        "keywords_context": ["matrix", "vector", "determinant", ...],
        "keywords_equation": ["||", "∥", "det(", "trace", ...],
        "weight_multiplier": 1.0  # Can boost specific domains
    }
}

# Role patterns (RPN programs that match variable context to roles)
{
    "symbol_id": "role_pattern_radius",
    "rpn_program": [
        # Check context for: "circle", "sphere", "cylinder"
        # Check variable position: single letter (r, R)
        # Check equation structure: πr², 4πr², (4/3)πr³
        # Return role confidence score
    ],
    "metadata": {
        "role": "radius",
        "domain": "geometry",
        "tier": 1,
        "context_cues": ["circle", "sphere", "cylinder"],
        "equation_patterns": ["πr²", "4πr²", "(4/3)πr³"],
        "variable_conventions": ["r", "R"]
    }
}
```

**Population Strategy** (Ingestion Path):
- LLM-assisted ingestion extracts role patterns from validated books
- Each successful role inference → store as RPN program in Math Galaxy
- Over time, Math Galaxy accumulates ~100+ role patterns
- TRM can navigate these patterns for inference

### Grammar Galaxy (Transformation Rules)

**What to Store**:
```python
# Domain detection rule (transforms context → domain list)
{
    "rule_id": "detect_domain_from_context",
    "input_pattern": "CONTEXT + EQUATION + BOOK_METADATA",
    "rpn_program": [
        # For each domain signature in Math Galaxy:
        #   - Score context keywords (1×)
        #   - Score equation cues (2×)
        #   - Score book metadata (10×)
        # Return sorted domain list
    ],
    "metadata": {
        "purpose": "Multi-domain detection",
        "weights": {"book_metadata": 10, "equation": 2, "context": 1}
    }
}

# Role prioritization rule (transforms domain list → prioritized role list)
{
    "rule_id": "prioritize_roles_by_domain",
    "input_pattern": "DOMAIN_LIST + ALL_ROLES",
    "rpn_program": [
        # For each domain in domain_list (in order):
        #   - Append roles from that domain's tier
        # Append remaining Tier 1 roles
        # Append Tier 2 roles
        # Append Tier 3 roles
        # Deduplicate
    ],
    "metadata": {
        "purpose": "Context-aware role ordering"
    }
}
```

**Population Strategy** (Ingestion Path):
- Extract Grammar rules from validated Python implementation
- Convert weighted scoring logic → RPN programs
- Store as navigable rules in Grammar Galaxy

### Cranium (PTX Execution)

**What to Execute**:
```cuda
// PTX kernel: Domain detection (pattern matching at GPU speed)
__global__ void detect_domain_kernel(
    const char* context,        // Input: textual context
    const char* equation,       // Input: equation string
    const char* book_domain,    // Input: metadata hint
    float* domain_scores        // Output: scores for each domain
) {
    // Thread per domain
    int domain_id = threadIdx.x;

    // Load domain signature from Math Galaxy (VRAM)
    DomainSignature sig = math_galaxy[domain_id];

    // Score context keywords (parallel string matching)
    float context_score = score_keywords(context, sig.keywords_context);

    // Score equation cues (parallel pattern matching)
    float equation_score = score_keywords(equation, sig.keywords_equation);

    // Apply weights: book_metadata (10×), equation (2×), context (1×)
    float total_score = 0;
    if (book_domain == sig.domain) total_score += 10.0f;
    total_score += 2.0f * equation_score;
    total_score += 1.0f * context_score;

    // Write back
    domain_scores[domain_id] = total_score;
}
```

**Sovereignty Achieved**:
- ✅ No Python (PTX kernel only)
- ✅ No Ollama/LLM (pattern matching only)
- ✅ GPU-accelerated (parallel across domains)
- ✅ Reads from Math Galaxy (VRAM)
- ✅ Writes to Galaxy (domain scores)

### TRM (Learned Navigation)

**What to Learn**:
```python
# TRM parameters (~7M) learn to navigate Math Galaxy for role inference

# Example navigation sequence (learned, not hardcoded):
1. Query Grammar Galaxy → get "detect_domain_from_context" rule
2. Execute rule → get domain list ["linear_algebra", "geometry"]
3. Query Math Galaxy → get role patterns for "linear_algebra"
4. Query Math Galaxy → get role patterns matching current context
5. Execute top-K role patterns → get confidence scores
6. Select highest-confidence role → return "vector"

# Shadow copy enhancement:
# - If inference correct → boost weights for this navigation path
# - If inference wrong → reduce weights
# - Over time → TRM learns optimal navigation without LLM
```

**Training Strategy**:
- **Phase 1** (NOW): LLM provides ground truth labels → populate Galaxy
- **Phase 2** (NEXT): TRM learns to navigate Galaxy, LLM as fallback
- **Phase 3** (FUTURE): TRM-only (no LLM), learned from accumulated patterns

### House (Persistent Storage)

**What to Persist**:
```python
# Book metadata (domain hints)
{
    "book_id": "la_done_right",
    "title": "Linear Algebra Done Right",
    "domain": "linear_algebra",
    "metadata": {
        "domain_confidence": 1.0,  # User-specified
        "detected_domains": ["linear_algebra", "geometry"],
        "artifact_count": 72,
        "role_distribution": {
            "vector": 10,
            "component": 5,
            "magnitude": 2,
            ...
        }
    }
}

# Role inference cache (successful patterns for future learning)
{
    "context": "In a vector space V, let x ∈ V be a vector...",
    "equation": "x = x₁e₁ + x₂e₂",
    "variable": "x",
    "inferred_role": "vector",
    "detected_domains": ["linear_algebra"],
    "confidence": 0.95,
    "source": "LLM (granite4:tiny-h)",
    "validated": true  # If matches expected role
}
```

**Purpose**: Historical inference data for TRM training + pattern validation

---

## Migration Path (3 Phases)

### Phase 1: LLM-Assisted Ingestion (CURRENT - VALIDATED ✅)

**Status**: COMPLETE (72 linalg + 40 geo artifacts validated)

**Architecture**:
```
Books (PDF) → Python extraction → LLM (Ollama) → Semantic roles → Math Galaxy
                                                                   ↓
                                                            House (persist)
```

**What Works**:
- Multi-domain detection (6 tiers, ~100 roles)
- Book metadata weighting (10×)
- Validated on 2 domains (linalg 58.5%, geo 77.0%)

**Sovereignty Status**:
- ❌ Ingestion uses Python + Ollama (acceptable - happens once)
- ✅ Output is sovereign (Math Galaxy entries)
- ✅ Metadata stored in House (persistent)

**Next Step**: Populate Math Galaxy with role patterns from validated inferences

### Phase 2: Hybrid (TRM + LLM Fallback) (NEXT)

**Architecture**:
```
Variable context → TRM navigates Math Galaxy → Role pattern matching
                        ↓                              ↓
                   Confidence > 0.8?          Confidence < 0.8?
                        ↓                              ↓
                   Return role                  LLM fallback (Ollama)
                                                      ↓
                                              Store new pattern → Math Galaxy
                                                      ↓
                                              TRM learns (shadow copy)
```

**Implementation Tasks**:
1. Extract domain detection logic → Grammar Galaxy rules (RPN programs)
2. Extract role patterns → Math Galaxy symbols (RPN programs)
3. Implement TRM navigation for role inference
4. Implement confidence scoring (TRM output)
5. Keep LLM fallback for low-confidence cases
6. Store successful LLM inferences → populate Math Galaxy
7. TRM shadow copy enhancement (learn from successful patterns)

**Sovereignty Status**:
- ✅ TRM navigates Math Galaxy (sovereign)
- ✅ High-confidence cases use TRM only (zero external dependencies)
- ⚠️ Low-confidence cases fall back to LLM (acceptable during learning)
- ✅ LLM results populate Galaxy (improves TRM over time)

**Expected Timeline**: 2-3 weeks (implementation + validation)

### Phase 3: Sovereign (TRM-Only) (FUTURE)

**Architecture**:
```
Variable context → TRM navigates Math Galaxy → Role pattern matching → Return role
                        ↓
                   Cranium PTX kernels (GPU-accelerated pattern matching)
                        ↓
                   Math Galaxy (VRAM - role patterns, domain signatures)
```

**Implementation Requirements**:
1. Math Galaxy populated with 500+ role patterns (from Phase 1-2 ingestion)
2. TRM trained on 10,000+ successful inferences (shadow copy enhancement)
3. Confidence threshold reached (>90% accuracy without LLM)
4. PTX kernels optimized for pattern matching
5. Grammar Galaxy rules validated (domain detection + prioritization)

**Sovereignty Status**:
- ✅ Zero external dependencies (PTX + Galaxy + TRM only)
- ✅ GPU-accelerated (Cranium PTX kernels)
- ✅ Learned navigation (TRM, no hardcoded logic)
- ✅ Self-improving (shadow copy from successful inferences)

**Expected Timeline**: 3-6 months (requires extensive TRM training + Galaxy population)

---

## Implementation Directive for Codex

### Immediate Tasks (Phase 2 Preparation)

**1. Extract Domain Signatures → Math Galaxy**

**File**: `knowledge3d/cranium/math_galaxy_population.py` (new)

```python
def populate_domain_signatures():
    """
    Extract domain detection logic from sovereign_knowledge_articulator.py
    Convert to Math Galaxy symbols (RPN programs).
    """

    # For each domain (geometry, linear_algebra, calculus, physics, number_theory, statistics):
    domain_signatures = []

    # Geometry signature
    domain_signatures.append({
        "symbol_id": f"domain_signature_geometry",
        "rpn_program": create_keyword_matcher_rpn(
            context_keywords=["circle", "triangle", "rectangle", "sphere", ...],
            equation_keywords=["π", "pi"],
            weight_context=1.0,
            weight_equation=2.0
        ),
        "metadata": {
            "domain": "geometry",
            "tier": "1A",
            "role_count": 25
        }
    })

    # Repeat for all 6 domains...

    # Write to Math Galaxy
    for sig in domain_signatures:
        math_galaxy.add_symbol(sig)
```

**Success Criteria**:
- [ ] 6 domain signatures stored in Math Galaxy
- [ ] Each signature contains keyword lists (context + equation)
- [ ] Each signature has RPN program for scoring
- [ ] Validated: domain detection matches Python implementation

**2. Extract Role Patterns → Math Galaxy**

**File**: Same as above (`math_galaxy_population.py`)

```python
def populate_role_patterns():
    """
    Extract ~100 role patterns from validated inferences.
    Convert to Math Galaxy symbols (RPN programs).
    """

    # Load successful inferences from House cache
    cache = load_validated_inferences("/K3D/Knowledge3D.local/galaxies/books_v5_tier3_linalg_hint/...")

    # Group by role
    role_patterns = defaultdict(list)
    for inference in cache:
        if inference["validated"]:
            role_patterns[inference["inferred_role"]].append({
                "context": inference["context"],
                "equation": inference["equation"],
                "variable": inference["variable"],
                "domains": inference["detected_domains"]
            })

    # For each role with ≥3 validated inferences:
    for role, examples in role_patterns.items():
        if len(examples) >= 3:
            # Extract common patterns
            pattern = extract_pattern(examples)

            # Create RPN program
            rpn = create_role_pattern_rpn(
                context_cues=pattern["context_keywords"],
                equation_cues=pattern["equation_patterns"],
                variable_conventions=pattern["variable_names"],
                confidence_threshold=0.8
            )

            # Store in Math Galaxy
            math_galaxy.add_symbol({
                "symbol_id": f"role_pattern_{role}",
                "rpn_program": rpn,
                "metadata": {
                    "role": role,
                    "domain": pattern["primary_domain"],
                    "tier": pattern["tier"],
                    "example_count": len(examples)
                }
            })
```

**Success Criteria**:
- [ ] 50+ role patterns stored in Math Galaxy (high-frequency roles)
- [ ] Each pattern has context cues, equation patterns, variable conventions
- [ ] Each pattern has RPN program for confidence scoring
- [ ] Validated: pattern matching ≥80% accuracy on held-out test set

**3. Extract Grammar Rules → Grammar Galaxy**

**File**: `knowledge3d/cranium/grammar_galaxy_population.py` (new)

```python
def populate_domain_detection_rule():
    """
    Extract domain detection logic → Grammar Galaxy rule.
    """

    grammar_galaxy.add_rule({
        "rule_id": "detect_domain_multidomain",
        "input_pattern": "CONTEXT + EQUATION + BOOK_DOMAIN_HINT",
        "rpn_program": [
            # Load all domain signatures from Math Galaxy
            # For each signature:
            #   - Execute signature RPN (get context score)
            #   - Execute signature RPN (get equation score)
            #   - If book_domain_hint matches: add 10×
            # Sort by total score (descending)
            # Return domain list
        ],
        "metadata": {
            "purpose": "Multi-domain detection with metadata weighting",
            "weights": {"book_metadata": 10, "equation": 2, "context": 1}
        }
    })

def populate_role_prioritization_rule():
    """
    Extract role prioritization logic → Grammar Galaxy rule.
    """

    grammar_galaxy.add_rule({
        "rule_id": "prioritize_roles_by_domain",
        "input_pattern": "DOMAIN_LIST + ALL_ROLE_PATTERNS",
        "rpn_program": [
            # For each domain in DOMAIN_LIST (in order):
            #   - Query Math Galaxy for roles in this domain
            #   - Append to prioritized list
            # Append remaining roles (lower priority)
            # Deduplicate
            # Return prioritized role list
        ],
        "metadata": {
            "purpose": "Context-aware role ordering"
        }
    })
```

**Success Criteria**:
- [ ] 2 Grammar rules stored (domain detection + role prioritization)
- [ ] Rules reference Math Galaxy symbols (domain signatures, role patterns)
- [ ] Validated: rule execution matches Python logic

**4. Implement TRM Navigation for Role Inference**

**File**: `knowledge3d/cranium/trm_role_inference.py` (new)

```python
def infer_role_trm(
    *,
    var: str,
    context: str,
    equation: str,
    book_domain_hint: str = None,
    confidence_threshold: float = 0.8
) -> Tuple[str, float]:
    """
    TRM-based role inference (Phase 2 hybrid approach).

    Returns:
        (role, confidence) - role string + confidence score [0, 1]
    """

    # Step 1: TRM navigates Grammar Galaxy → get domain detection rule
    detect_rule = trm.query_grammar_galaxy("detect_domain_multidomain")

    # Step 2: Execute rule → get domain list
    domains = trm.execute_rule(detect_rule, {
        "context": context,
        "equation": equation,
        "book_domain_hint": book_domain_hint
    })

    # Step 3: TRM navigates Grammar Galaxy → get role prioritization rule
    prioritize_rule = trm.query_grammar_galaxy("prioritize_roles_by_domain")

    # Step 4: Execute rule → get prioritized role list
    prioritized_roles = trm.execute_rule(prioritize_rule, {
        "domains": domains,
        "all_roles": math_galaxy.get_all_role_patterns()
    })

    # Step 5: TRM navigates Math Galaxy → score each prioritized role
    role_scores = []
    for role_pattern_id in prioritized_roles[:20]:  # Top 20 for efficiency
        pattern = math_galaxy.get_symbol(role_pattern_id)

        # Execute pattern RPN → get confidence score
        confidence = trm.execute_pattern(pattern["rpn_program"], {
            "context": context,
            "equation": equation,
            "variable": var
        })

        role_scores.append((pattern["metadata"]["role"], confidence))

    # Step 6: Select highest-confidence role
    role_scores.sort(key=lambda x: -x[1])
    best_role, best_confidence = role_scores[0]

    return best_role, best_confidence
```

**Success Criteria**:
- [ ] TRM can navigate Grammar Galaxy (2 rules)
- [ ] TRM can navigate Math Galaxy (domain signatures, role patterns)
- [ ] TRM can execute RPN programs (pattern matching)
- [ ] Returns confidence scores [0, 1]
- [ ] Validated: ≥70% agreement with LLM on high-confidence cases (≥0.8)

**5. Implement Hybrid Fallback (TRM + LLM)**

**File**: `knowledge3d/training/math_benchmarks/sovereign_knowledge_articulator.py` (modify)

```python
def infer_role_hybrid(
    self,
    *,
    var: str,
    context: str,
    equation: str,
    book_domain_hint: str = None,
    confidence_threshold: float = 0.8
) -> Tuple[str, str]:
    """
    Hybrid role inference: TRM first, LLM fallback.

    Returns:
        (role, source) - role string + source ("TRM" or "LLM")
    """

    # Try TRM first
    role_trm, confidence = infer_role_trm(
        var=var,
        context=context,
        equation=equation,
        book_domain_hint=book_domain_hint,
        confidence_threshold=confidence_threshold
    )

    # High confidence → use TRM
    if confidence >= confidence_threshold:
        # Shadow copy enhancement: boost TRM weights for this navigation path
        trm.enhance_from_success(context, equation, var, role_trm)
        return role_trm, "TRM"

    # Low confidence → fallback to LLM
    else:
        role_llm, domain_llm = self._infer_role_llm(
            var=var,
            context=context,
            equation=equation,
            book_domain_hint=book_domain_hint
        )

        # Store LLM inference → populate Math Galaxy for future TRM learning
        math_galaxy.add_inference_example({
            "context": context,
            "equation": equation,
            "variable": var,
            "inferred_role": role_llm,
            "detected_domains": domain_llm,
            "confidence_trm": confidence,  # TRM was unsure
            "source": "LLM_fallback",
            "validated": None  # Will validate later
        })

        return role_llm, "LLM"
```

**Success Criteria**:
- [ ] Hybrid path works (TRM → high-confidence, LLM → low-confidence)
- [ ] TRM shadow copy enhancement implemented
- [ ] LLM fallback results stored in Math Galaxy
- [ ] Metrics tracked: TRM coverage rate, LLM fallback rate
- [ ] Validated: Total accuracy ≥ LLM-only baseline (85%+)

**6. Create Phase 2 Validation Test**

**File**: `tests/integration/test_trm_role_inference.py` (new)

```python
def test_trm_hybrid_on_validated_set():
    """
    Run hybrid TRM+LLM on held-out validation set (10% of linalg/geo artifacts).
    Compare to LLM-only baseline.
    """

    # Load held-out validation set (10% of 72 linalg + 40 geo = ~11 artifacts)
    validation_set = load_validation_artifacts()

    results = {
        "trm_only": 0,      # TRM confidence ≥ 0.8
        "llm_fallback": 0,  # TRM confidence < 0.8
        "correct_trm": 0,   # TRM inference matches ground truth
        "correct_llm": 0,   # LLM fallback matches ground truth
        "total": len(validation_set)
    }

    for artifact in validation_set:
        for binding in artifact["symbol_bindings"]:
            # Ground truth
            expected_role = binding["meaning"]

            # Hybrid inference
            inferred_role, source = infer_role_hybrid(
                var=binding["symbol"],
                context=artifact["context"],
                equation=artifact["equation"],
                book_domain_hint=artifact["domain"]
            )

            # Track results
            if source == "TRM":
                results["trm_only"] += 1
                if inferred_role == expected_role:
                    results["correct_trm"] += 1
            else:
                results["llm_fallback"] += 1
                if inferred_role == expected_role:
                    results["correct_llm"] += 1

    # Compute metrics
    trm_coverage = results["trm_only"] / results["total"]
    trm_accuracy = results["correct_trm"] / results["trm_only"] if results["trm_only"] > 0 else 0
    llm_accuracy = results["correct_llm"] / results["llm_fallback"] if results["llm_fallback"] > 0 else 0
    total_accuracy = (results["correct_trm"] + results["correct_llm"]) / results["total"]

    # Assert success criteria
    assert trm_coverage >= 0.30, f"TRM coverage too low: {trm_coverage:.1%} (need ≥30%)"
    assert trm_accuracy >= 0.70, f"TRM accuracy too low: {trm_accuracy:.1%} (need ≥70%)"
    assert total_accuracy >= 0.80, f"Total accuracy too low: {total_accuracy:.1%} (need ≥80%)"

    print(f"✅ TRM coverage: {trm_coverage:.1%}")
    print(f"✅ TRM accuracy: {trm_accuracy:.1%}")
    print(f"✅ LLM fallback accuracy: {llm_accuracy:.1%}")
    print(f"✅ Total accuracy: {total_accuracy:.1%}")
```

**Success Criteria**:
- [ ] TRM coverage ≥30% (30%+ inferences without LLM)
- [ ] TRM accuracy ≥70% (when confident, usually correct)
- [ ] Total accuracy ≥80% (hybrid matches LLM-only baseline)

---

## Validation Metrics (Phase 2)

**TRM Performance**:
- **Coverage rate**: % of inferences where TRM confidence ≥ 0.8
  - Target: ≥30% (Phase 2 start) → ≥90% (Phase 3 ready)
- **Accuracy rate**: % of TRM inferences that match ground truth
  - Target: ≥70% (Phase 2 start) → ≥95% (Phase 3 ready)

**LLM Fallback Performance**:
- **Fallback rate**: % of inferences where TRM confidence < 0.8
  - Target: ≤70% (Phase 2 start) → ≤10% (Phase 3 ready)
- **Fallback accuracy**: % of LLM inferences that match ground truth
  - Target: ≥85% (maintain current LLM baseline)

**Galaxy Population**:
- **Domain signatures**: 6 stored (1 per domain)
- **Role patterns**: 50+ stored (high-frequency roles from validated inferences)
- **Grammar rules**: 2 stored (domain detection + role prioritization)
- **Inference examples**: 5,000+ stored (from full 23-book ingestion)

**TRM Learning Progress**:
- **Shadow copy enhancements**: Track successful navigation paths
- **Pattern match improvements**: Monitor confidence score increases over time
- **Fallback reduction**: Track decreasing LLM fallback rate week-over-week

---

## Timeline Estimate

**Phase 2 Implementation** (Hybrid TRM + LLM):
- Week 1-2: Extract signatures/patterns/rules → populate Math Galaxy + Grammar Galaxy
- Week 3-4: Implement TRM navigation + hybrid fallback
- Week 5: Validation testing (held-out set)
- Week 6: Full 23-book re-ingestion with hybrid approach (compare metrics)

**Phase 2 → Phase 3 Transition** (TRM-only):
- Months 1-3: Continuous learning (shadow copy enhancement from full ingestion)
- Months 3-6: Incremental confidence threshold increases (0.8 → 0.9 → 0.95)
- Month 6: Declare Phase 3 ready when TRM coverage ≥90% + accuracy ≥95%

---

## Critical Architectural Principles (Preserved)

**1. Galaxy Universe = Unified VRAM Workspace**
- ✅ Math Galaxy stores domain signatures + role patterns (procedural RPN programs)
- ✅ Grammar Galaxy stores transformation rules (domain detection, prioritization)
- ✅ All knowledge accessible simultaneously (no loading/unloading)

**2. TRM = Learned Navigation Logic**
- ✅ TRM navigates Math Galaxy (queries role patterns)
- ✅ TRM navigates Grammar Galaxy (executes transformation rules)
- ✅ TRM learns from success (shadow copy enhancement)
- ❌ TRM does NOT store knowledge (that's in Galaxy Universe)

**3. Sovereignty = PTX + Galaxy Only (Hot Path)**
- ⚠️ Phase 2: TRM sovereign (high-confidence), LLM fallback (low-confidence)
- ✅ Phase 3: TRM sovereign (100%), zero external dependencies

**4. Dual Client Reality = Form + Meaning**
- ✅ Role patterns = RPN programs (procedural form) + metadata (semantic meaning)
- ✅ Domain signatures = RPN programs (keyword matching) + metadata (domain info)
- ✅ Both readable by humans (inspect Galaxy) AND executable by TRM (navigate Galaxy)

**5. Save Information Principle = Reference, Don't Duplicate**
- ✅ LLM inferences stored ONCE in Math Galaxy (as role pattern examples)
- ✅ Books reference domain signatures (symlink, not duplicate)
- ✅ Artifacts reference role patterns (content-based deduplication)

**6. One Model to Process Them All**
- ✅ Same TRM processes math benchmarks (this work) + ARC-AGI (visual reasoning) + physics (procedural systems)
- ✅ Same Math Galaxy serves all curricula (shared knowledge)
- ✅ Same Grammar Galaxy applies to all domains (unified transformation rules)

---

## Success Criteria Summary

**Phase 2 Complete** when:
- [ ] Math Galaxy populated (6 domain signatures, 50+ role patterns)
- [ ] Grammar Galaxy populated (2 transformation rules)
- [ ] TRM navigation implemented (queries + executes)
- [ ] Hybrid fallback implemented (TRM + LLM)
- [ ] Validation test passes (≥30% TRM coverage, ≥70% TRM accuracy, ≥80% total accuracy)
- [ ] Full 23-book re-ingestion with hybrid approach (compare metrics to Phase 1 LLM-only)

**Phase 3 Ready** when:
- [ ] TRM coverage ≥90% (90%+ inferences without LLM)
- [ ] TRM accuracy ≥95% (when confident, almost always correct)
- [ ] Math Galaxy populated (500+ role patterns from extensive learning)
- [ ] Shadow copy enhancement validated (demonstrated learning curve)
- [ ] PTX kernels optimized (GPU-accelerated pattern matching)
- [ ] LLM fallback rate ≤10% (near-zero external dependencies)

---

## Recommended Immediate Action

**Next Steps for Codex**:

1. **Read this specification completely** (don't implement yet, understand architecture first)

2. **Create Phase 2 implementation plan**:
   - Break down 6 tasks into subtasks
   - Estimate timeline (weeks)
   - Identify blockers (if any)

3. **Start with Task 1** (populate domain signatures):
   - Extract from `sovereign_knowledge_articulator.py:60-180`
   - Create `math_galaxy_population.py`
   - Convert keyword lists → RPN programs
   - Store in Math Galaxy
   - Validate: domain detection matches Python

4. **Iterate through Tasks 2-6** sequentially (each builds on previous)

5. **Run validation test** after Task 6 complete

6. **Report metrics** (TRM coverage, accuracy, fallback rate)

---

## Questions for Codex (Before Implementation)

1. Do you understand the 3-phase migration path (Ingestion → Hybrid → Sovereign)?
2. Do you understand where each component belongs (Math Galaxy vs Grammar Galaxy vs Cranium vs TRM)?
3. Do you understand the validation criteria (coverage %, accuracy %, fallback %)?
4. Are there any architectural concerns before starting implementation?
5. Do you need clarification on any RPN program structures?

---

**This is the architectural blueprint for sovereignty migration.** Proceed when ready.
