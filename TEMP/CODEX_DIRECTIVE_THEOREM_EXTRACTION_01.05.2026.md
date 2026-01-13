# Codex Directive: Theorem Application Extraction — January 5, 2026

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Priority**: CRITICAL (MATH benchmark regression fix)

---

## Context: Why Books Made MATH Worse (2% → 1%)

**Root Cause Identified**: We extracted SEMANTIC labels (what symbols mean) but MISSING PROCEDURAL knowledge (how to transform equations).

**User Correction**: "We need BOTH - semantic labels will be used on the search, procedural knowledge (and RPN formulations that work with our math cores) on solving it"

**Architecture Insight**:
1. **Semantic labels** → TRM routes via semantic closeness (find relevant knowledge)
2. **Procedural RPN** → Cranium executes transformation (solve the problem)
3. **Current state**: Have #1, missing #2 → TRM finds books but can't use them

---

## Your Mission: Extract Theorem Application Patterns

**Goal**: Extract PROCEDURAL transformation patterns from 21 books, link to semantic labels, format as RPN programs compatible with math cores.

**Output**: Populate Math Galaxy with:
- **Semantic index** (role patterns - already extracted ✅)
- **Procedural transformations** (theorem applications - NEW ⚠️)
- **Linkage** (semantic pattern → RPN program)

---

## Task 1: Analyze Current Artifact Structure

**File**: Inspect `/K3D/Knowledge3D.local/galaxies/books_v5_clean2/*/artifacts.jsonl`

**What to Find**: Do artifacts contain transformation sequences or just definitions?

**Look for**:
- Multi-step derivations (theorem A → theorem B → simplified form)
- Worked examples (problem → solution steps)
- Formula transformations (identity expansions, substitutions)

**Example Artifact to Inspect**:
```bash
# Sample from calculus book
jq 'select(.conclusion != null and (.conditions | length) > 0) | {conclusion, conditions, lhs, rhs}' \
  /K3D/Knowledge3D.local/galaxies/books_v5_clean2/numerical_analysis/artifacts.jsonl | head -20
```

**Report**:
- What % of artifacts are definitions vs. transformations?
- Are multi-step derivations preserved in artifact structure?
- Can we reconstruct theorem application sequences?

---

## Task 2: Define Theorem Application Pattern Schema

**Structure**: Each pattern links PRECONDITION → TRANSFORMATION → POSTCONDITION

```python
THEOREM_APPLICATION_PATTERN = {
    # Semantic routing (TRM uses for search)
    "pattern_id": "power_rule_polynomial",
    "domain": "calculus",
    "semantic_tags": ["derivative", "polynomial", "power_rule"],

    # Precondition (when to apply)
    "precondition": {
        "structure_match": "polynomial",  # Problem type
        "context_cues": ["derivative", "d/d", "∂"],  # Lexical signals
        "equation_pattern": r"x\^n",  # Structural pattern (regex)
    },

    # Transformation (how to apply)
    "transformation": {
        "theorem_name": "power_rule",
        "rpn_program": [
            # RPN sequence compatible with math cores
            "PUSH_n",           # Get exponent
            "PUSH_x",           # Get base
            "PUSH_n",           # Duplicate exponent
            "PUSH_1",
            "SUB",              # n - 1
            "POW",              # x^(n-1)
            "MULT",             # n * x^(n-1)
        ],
        "tier": 2,  # Tier 2 algebra (formula components)
        "dependencies": [],  # Other patterns needed
    },

    # Postcondition (expected result)
    "postcondition": {
        "result_type": "polynomial",
        "degree_reduction": 1,  # degree n → n-1
        "validation": "term_by_term",  # How to verify
    },

    # Provenance (where learned)
    "source": {
        "book": "numerical_analysis",
        "artifact_ids": ["art_123", "art_456"],  # Which artifacts
        "example_count": 47,  # How many times seen
    }
}
```

**Key Insight**: Semantic tags (derivative, polynomial) enable TRM routing → RPN program enables Cranium execution.

---

## Task 3: Extract Transformation Patterns from Artifacts

**File**: Extend `knowledge3d/cranium/math_galaxy_population.py`

**Add Function**: `extract_theorem_patterns(artifact_dirs, min_examples=3)`

**Logic**:

```python
def extract_theorem_patterns(
    artifact_dirs: Sequence[str],
    *,
    min_examples: int = 3,
) -> List[Dict[str, Any]]:
    """
    Extract theorem application patterns from validated artifacts.

    Looks for:
    - Transformation sequences (multi-condition → conclusion)
    - Formula identities (LHS ≡ RHS under conditions)
    - Derivation chains (step-by-step simplifications)
    """
    paths = [Path(p) for p in artifact_dirs]

    # Group by transformation type
    transformations: Dict[str, List[Dict]] = defaultdict(list)

    for art in _iter_artifacts(paths):
        # Skip pure definitions (no transformation)
        if not art.get("conditions") or not art.get("conclusion"):
            continue

        # Extract transformation signature
        pattern_sig = _identify_transformation_pattern(art)
        if pattern_sig:
            transformations[pattern_sig["pattern_id"]].append(art)

    # Synthesize RPN programs for frequent patterns
    patterns = []
    for pattern_id, artifacts in transformations.items():
        if len(artifacts) < min_examples:
            continue

        pattern = _synthesize_theorem_pattern(pattern_id, artifacts)
        if pattern:
            patterns.append(pattern)

    return patterns
```

**Helper Functions Needed**:

```python
def _identify_transformation_pattern(artifact: Dict) -> Optional[Dict]:
    """
    Identify transformation type from artifact structure.

    Returns pattern signature or None if not a transformation.
    """
    lhs = artifact.get("lhs")
    rhs = artifact.get("rhs")
    conclusion = artifact.get("conclusion")
    conditions = artifact.get("conditions", [])

    # Detect transformation types:
    # 1. Derivative rule (conditions mention "differentiable", conclusion has d/dx)
    # 2. Integral identity (conditions setup, conclusion has ∫)
    # 3. Algebraic simplification (LHS complex, RHS simplified)
    # 4. Substitution (variable mapping changes structure)
    # 5. Identity expansion (trig, logarithm, etc.)

    # Example: power rule detection
    if any("derivative" in str(c).lower() for c in conditions):
        if "d/dx" in str(conclusion) or "∂" in str(conclusion):
            if re.search(r"x\^\d+", str(lhs) or str(rhs) or ""):
                return {
                    "pattern_id": "power_rule_polynomial",
                    "type": "derivative_rule",
                    "domain": "calculus",
                }

    # ... more pattern detections
    return None

def _synthesize_theorem_pattern(
    pattern_id: str,
    artifacts: List[Dict]
) -> Optional[Dict]:
    """
    Synthesize theorem pattern from multiple artifact examples.

    Extracts:
    - Common preconditions (structure + context)
    - Transformation logic (RPN synthesis)
    - Postcondition validation
    """
    # Extract semantic tags from all artifacts
    semantic_tags = set()
    for art in artifacts:
        # From role patterns
        for binding in (art.get("symbol_bindings") or {}).values():
            role = binding.get("meaning")
            if role and role != "unknown":
                semantic_tags.add(role)

        # From context
        domain = art.get("domain")
        if domain:
            semantic_tags.add(domain)

    # Synthesize RPN program (CRITICAL: must work with math cores)
    rpn_program = _infer_rpn_from_artifacts(pattern_id, artifacts)
    if not rpn_program:
        return None

    # Extract preconditions (when to apply)
    precondition = _extract_precondition(artifacts)

    # Extract postcondition (expected result)
    postcondition = _extract_postcondition(artifacts)

    return {
        "pattern_id": pattern_id,
        "domain": artifacts[0].get("domain", "unknown"),
        "semantic_tags": sorted(semantic_tags),
        "precondition": precondition,
        "transformation": {
            "theorem_name": pattern_id,
            "rpn_program": rpn_program,
            "tier": _infer_tier(pattern_id),
            "dependencies": [],  # TODO: extract from conditions
        },
        "postcondition": postcondition,
        "source": {
            "book": Path(artifacts[0].get("source_file", "")).parts[-2],
            "artifact_ids": [a.get("id") for a in artifacts[:10]],
            "example_count": len(artifacts),
        }
    }
```

---

## Task 4: RPN Program Synthesis (CRITICAL)

**Challenge**: Convert transformation logic → RPN programs compatible with math cores

**Math Core Tiers** (from MATH_CORE_SPECIFICATION.md):
- **Tier 1**: High-school algebra (add, sub, mult, div, pow, sqrt, frac, parens)
- **Tier 2**: Formula components (subscript, nCr, factorial, piecewise, sum, product)
- **Tier 3**: Advanced (limit, derivative, integral, matrix, vector)

**RPN Operations Available**:
```python
# Tier 1 (basic algebra)
TIER1_OPS = [
    "PUSH", "POP", "DUP", "SWAP",  # Stack ops
    "ADD", "SUB", "MULT", "DIV",   # Arithmetic
    "POW", "SQRT", "ABS", "NEG",   # Power/root
    "EQ", "LT", "GT", "AND", "OR", # Logic
]

# Tier 2 (formula components)
TIER2_OPS = [
    "FACTORIAL", "BINOM", "SUM", "PRODUCT",
    "SUBSCRIPT", "PIECEWISE", "CASE",
]

# Tier 3 (advanced math)
TIER3_OPS = [
    "DERIVATIVE", "INTEGRAL", "LIMIT",
    "MATRIX_MULT", "DOT_PRODUCT", "CROSS_PRODUCT",
    "DET", "TRACE", "EIGENVALUE",
]
```

**Synthesis Strategy**:

```python
def _infer_rpn_from_artifacts(
    pattern_id: str,
    artifacts: List[Dict]
) -> Optional[List[str]]:
    """
    Infer RPN program from transformation examples.

    Strategy:
    1. Extract LHS → RHS transformations
    2. Identify common operation sequence
    3. Map to RPN opcodes (tier-appropriate)
    4. Validate against artifacts
    """
    # Example: power rule
    if pattern_id == "power_rule_polynomial":
        return [
            "DUP_EXPONENT",  # n
            "SWAP_BASE",     # x, n
            "DUP_EXPONENT",  # x, n, n
            "PUSH_1",        # x, n, n, 1
            "SUB",           # x, n, n-1
            "POW",           # x, n, x^(n-1)
            "MULT",          # n * x^(n-1)
        ]

    # Example: product rule (f*g)' = f'*g + f*g'
    if pattern_id == "product_rule":
        return [
            "PUSH_F", "DERIVATIVE", "PUSH_G", "MULT",     # f' * g
            "PUSH_F", "PUSH_G", "DERIVATIVE", "MULT",     # f * g'
            "ADD",                                         # f'*g + f*g'
        ]

    # Example: chain rule (f(g(x)))' = f'(g(x)) * g'(x)
    if pattern_id == "chain_rule":
        return [
            "PUSH_G", "DERIVATIVE",           # g'(x)
            "PUSH_F", "PUSH_G", "COMPOSE",    # f(g(x))
            "DERIVATIVE",                      # f'(g(x))
            "MULT",                            # f'(g(x)) * g'(x)
        ]

    # For unknown patterns, attempt synthesis
    return _synthesize_rpn_heuristic(artifacts)

def _synthesize_rpn_heuristic(artifacts: List[Dict]) -> Optional[List[str]]:
    """
    Heuristic RPN synthesis from transformation examples.

    PLACEHOLDER: This is complex - start with manual mapping for
    top 10-20 patterns, then add synthesis logic.
    """
    # TODO: Implement pattern-to-RPN synthesis
    # For now, return None for unknown patterns
    return None
```

**Start Simple**: Manually define RPN for top 10 theorem patterns (power rule, product rule, chain rule, quotient rule, trig identities, etc.)

---

## Task 5: Link Semantic Labels → Procedural RPN

**File**: Update `knowledge3d/cranium/math_galaxy_population.py`

**Add Global**:
```python
THEOREM_PATTERNS: List[Dict[str, Any]] = []
```

**Add Function**:
```python
def populate_theorem_patterns(
    artifact_dirs: Sequence[str],
    math_galaxy: Optional[object] = None,
    *,
    min_examples: int = 3,
) -> List[Dict[str, Any]]:
    """
    Populate Math Galaxy with theorem application patterns.

    Links semantic labels (from ROLE_PATTERNS) to procedural RPN.
    """
    patterns = extract_theorem_patterns(artifact_dirs, min_examples=min_examples)
    global THEOREM_PATTERNS
    THEOREM_PATTERNS = list(patterns)

    if math_galaxy is None:
        return list(patterns)

    # Populate Math Galaxy
    if hasattr(math_galaxy, "add_theorem_pattern"):
        for pat in patterns:
            math_galaxy.add_theorem_pattern(pat)

    return list(patterns)
```

**Update `__all__`**:
```python
__all__ = [
    "DOMAIN_SIGNATURES",
    "ROLE_PATTERNS",
    "THEOREM_PATTERNS",  # NEW
    "populate_domain_signatures",
    "extract_role_patterns",
    "populate_role_patterns",
    "extract_theorem_patterns",  # NEW
    "populate_theorem_patterns",  # NEW
]
```

---

## Task 6: Test Extraction on Calculus Book

**File**: Create `tests/integration/test_theorem_extraction.py`

**Test**: Extract theorem patterns from numerical_analysis book (44 calculus patterns available)

**Expected Patterns** (manual verification):
1. **power_rule_polynomial**: d/dx(x^n) = n*x^(n-1)
2. **product_rule**: (f*g)' = f'*g + f*g'
3. **quotient_rule**: (f/g)' = (f'*g - f*g')/g^2
4. **chain_rule**: (f∘g)' = f'(g) * g'
5. **sum_rule**: (f+g)' = f' + g'
6. **constant_multiple**: (c*f)' = c*f'
7. **integration_by_parts**: ∫u dv = uv - ∫v du
8. **fundamental_theorem**: ∫[a,b] f'(x) dx = f(b) - f(a)

**Validation**:
- Extracted patterns ≥ 5 (at minimum)
- Each has semantic_tags (for routing)
- Each has rpn_program (for execution)
- RPN uses only tier-appropriate opcodes

**Run**:
```bash
pytest tests/integration/test_theorem_extraction.py -v
```

---

## Task 7: Integrate with MATH Benchmark

**File**: Modify `knowledge3d/training/math_benchmarks/run_sovereign_math_benchmarks.py`

**Add**: Load THEOREM_PATTERNS alongside ROLE_PATTERNS

**Logic**:
```python
from knowledge3d.cranium.math_galaxy_population import (
    populate_role_patterns,
    populate_theorem_patterns,  # NEW
)

# During initialization
if args.load_all_galaxies:
    # Load semantic labels (existing)
    role_patterns = populate_role_patterns(
        artifact_dirs=["/K3D/Knowledge3D.local/galaxies/books_v5_clean2"],
        min_examples=3,
    )

    # Load procedural theorems (NEW)
    theorem_patterns = populate_theorem_patterns(
        artifact_dirs=["/K3D/Knowledge3D.local/galaxies/books_v5_clean2"],
        min_examples=3,
    )

    # TRM can now:
    # 1. Route via semantic_tags (role_patterns)
    # 2. Execute via rpn_program (theorem_patterns)
```

**TRM Navigation** (conceptual - already implemented via semantic closeness):
```
Problem: "Find derivative of 3x^4"
  ↓
TRM semantic search: "derivative" + "polynomial" → routes to theorem_patterns
  ↓
Finds: power_rule_polynomial (semantic_tags match)
  ↓
Executes: rpn_program [PUSH_4, PUSH_x, PUSH_3, SUB, POW, PUSH_4, MULT]
  ↓
Result: 4 * 3 * x^3 = 12x^3
```

**Run MATH Benchmark Again**:
```bash
python knowledge3d/training/math_benchmarks/run_sovereign_math_benchmarks.py \
  --datasets math \
  --max-problems 100 \
  --load-all-galaxies \
  --disable-retrieval \
  --shadow-readonly
```

**Success Criteria**:
- Accuracy ≥ 2% (match baseline, don't make worse)
- Book usage signals with CORRECT answers (not just access)
- Theorem pattern usage logged (which patterns helped)

---

## Success Criteria (Full Task)

### Extraction Quality
- [ ] ≥10 theorem patterns extracted from 21 books
- [ ] Each pattern has semantic_tags (routing) + rpn_program (execution)
- [ ] RPN programs use tier-appropriate opcodes (validate against math cores)
- [ ] Patterns span ≥3 domains (calculus, geometry, linear_algebra)

### Integration Quality
- [ ] MATH benchmark loads both role_patterns + theorem_patterns
- [ ] TRM routing uses semantic_tags (evidence in logs)
- [ ] Cranium executes rpn_programs (evidence in traces)
- [ ] No sovereignty violations (grep for numpy/sympy in hot path)

### Performance
- [ ] MATH accuracy ≥ 2% (don't regress from baseline)
- [ ] Book usage correlates with CORRECT answers (not just access)
- [ ] Theorem pattern usage ≥ 5 times in 100 problems (evidence of utilization)

### Documentation
- [ ] Update math_galaxy_population.py docstrings
- [ ] Add theorem extraction test with examples
- [ ] Log which theorem patterns were used (for debugging)

---

## Timeline Estimate

**Week 1**: Tasks 1-3 (analyze artifacts, define schema, extract patterns)
**Week 2**: Tasks 4-5 (RPN synthesis, linkage)
**Week 3**: Tasks 6-7 (test extraction, integrate with MATH benchmark)

**Total**: 3 weeks to theorem extraction + MATH validation

---

## Critical Reminders

**Dual Purpose Design**:
- Semantic labels (role_patterns) → TRM routing via semantic closeness ✅
- Procedural RPN (theorem_patterns) → Cranium execution via PTX kernels ⚠️ NEW

**Sovereignty Compliance**:
- Extraction = ingestion phase (can use any tools)
- Execution = hot path (PTX + Galaxy ONLY)
- RPN programs must work with existing math cores (tier 1-3)

**Math Core Integration**:
- RPN opcodes MUST match math_core_tier*.ptx definitions
- Test RPN execution on simple cases BEFORE full benchmark
- Document which tier each pattern uses (for GPU allocation)

**Shadow Copy Learning**:
- When TRM routes to theorem_pattern and succeeds → shadow copy enhancement
- When TRM routes to theorem_pattern and fails → log for analysis
- Over time, TRM learns which semantic_tags → which theorem_patterns

---

## Questions Before Starting?

1. Do you understand the dual need (semantic routing + procedural execution)?
2. Do you understand RPN synthesis requirements (must work with math cores)?
3. Do you understand linkage (semantic_tags connect role_patterns → theorem_patterns)?
4. Are there blockers with artifact structure (can we extract transformations)?

**If clear, proceed with Task 1** (analyze artifact structure).

**If unclear, ask questions before implementing** (architecture clarity > speed).

---

## Expected Outcome

**After theorem extraction complete**:
- Math Galaxy has BOTH semantic index (role_patterns) AND procedural knowledge (theorem_patterns)
- TRM routes via semantic closeness (finds relevant theorem)
- Cranium executes RPN program (applies transformation)
- MATH accuracy improves from 1% (current) to ≥3% (Phase 7 baseline)
- Book usage signals correlate with CORRECT answers (evidence knowledge helps)

**Path to 10%+ MATH accuracy**:
- Extract 50+ theorem patterns from 21 books
- TRM shadow copy learns optimal routing (problem type → theorem pattern)
- Grammar Galaxy stores composition sequences (theorem chains for multi-step problems)
- Sovereign solving: semantic routing → theorem selection → RPN execution

---

**This is the architecturally correct path.** Proceed with Task 1, partner! 🚀
