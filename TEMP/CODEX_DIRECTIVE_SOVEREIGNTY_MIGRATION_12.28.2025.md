# Codex Directive: Sovereignty Migration — December 28, 2025

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Priority**: HIGH (Architecture validated, ready for sovereignty implementation)

---

## Context

**Multi-domain role extraction is validated** (72 linalg + 40 geo artifacts) using Python + Ollama (Option A). Now we need to **migrate to sovereign architecture** (PTX + Galaxy + TRM) to eliminate external dependencies in hot path.

**Full Specification**: [TEMP/CLAUDE_SOVEREIGNTY_MIGRATION_MULTI_DOMAIN_ROLES_12.28.2025.md](TEMP/CLAUDE_SOVEREIGNTY_MIGRATION_MULTI_DOMAIN_ROLES_12.28.2025.md)

---

## Your Mission

**Extract validated Python implementation → Populate Math Galaxy + Grammar Galaxy → Enable TRM learning**

**Migration Path**: 3 phases
- **Phase 1** (DONE ✅): LLM-assisted ingestion (populate Galaxy with semantic roles)
- **Phase 2** (YOUR WORK): Hybrid TRM + LLM fallback (TRM learns, LLM fills gaps)
- **Phase 3** (FUTURE): TRM-only (zero external dependencies, fully sovereign)

---

## Immediate Tasks (Phase 2 Implementation)

### Task 1: Extract Domain Signatures → Math Galaxy

**File**: Create `knowledge3d/cranium/math_galaxy_population.py`

**What to Extract**: 6 domain detection patterns from `sovereign_knowledge_articulator.py:220-290`

**Domains**:
1. Geometry (context: circle, triangle, sphere; equation: π)
2. Linear Algebra (context: vector, matrix; equation: ||, ∥, det)
3. Calculus (context: derivative, integral; equation: ∂, ∫, d/d)
4. Physics (context: velocity, force, energy)
5. Number Theory (context: prime, factor, modulus; equation: mod, ≡)
6. Statistics (context: probability, mean, variance; equation: P(, E[)

**Output**: Math Galaxy symbols (RPN programs for domain detection)

**Validation**: Domain detection matches Python implementation

---

### Task 2: Extract Role Patterns → Math Galaxy

**File**: Same (`math_galaxy_population.py`)

**What to Extract**: ~100 role patterns from validated inferences

**Source Data**:
- Linear algebra artifacts: `/K3D/Knowledge3D.local/galaxies/books_v5_tier3_linalg_hint/*.json`
- Geometry artifacts: `/K3D/Knowledge3D.local/galaxies/books_v5_tier3_geo_handbook/*.json`

**For each role with ≥3 validated inferences**:
- Extract common patterns (context keywords, equation structures, variable names)
- Create RPN program for pattern matching
- Store in Math Galaxy with metadata (role, domain, tier, examples)

**Output**: 50+ Math Galaxy symbols (RPN programs for role inference)

**Validation**: Pattern matching ≥80% accuracy on held-out test set

---

### Task 3: Extract Grammar Rules → Grammar Galaxy

**File**: Create `knowledge3d/cranium/grammar_galaxy_population.py`

**What to Extract**: 2 transformation rules

**Rule 1**: Domain detection
- Input: context + equation + book_domain_hint
- Logic: Score all domain signatures (book metadata 10×, equation 2×, context 1×)
- Output: Sorted domain list

**Rule 2**: Role prioritization
- Input: domain list + all role patterns
- Logic: Prioritize roles from detected domains (Tier 1 → Tier 2 → Tier 3)
- Output: Ordered role list

**Output**: 2 Grammar Galaxy rules (RPN programs)

**Validation**: Rule execution matches Python logic

---

### Task 4: Implement TRM Navigation

**File**: Create `knowledge3d/cranium/trm_role_inference.py`

**What to Implement**: TRM-based role inference function

**Navigation Sequence**:
1. TRM queries Grammar Galaxy → get domain detection rule
2. TRM executes rule → get domain list
3. TRM queries Grammar Galaxy → get role prioritization rule
4. TRM executes rule → get prioritized role patterns
5. TRM queries Math Galaxy → score top-K role patterns
6. TRM selects highest-confidence role

**Output**: `infer_role_trm(var, context, equation, book_domain_hint) -> (role, confidence)`

**Validation**: ≥70% agreement with LLM on high-confidence cases (confidence ≥ 0.8)

---

### Task 5: Implement Hybrid Fallback

**File**: Modify `knowledge3d/training/math_benchmarks/sovereign_knowledge_articulator.py`

**What to Implement**: Hybrid inference (TRM first, LLM fallback)

**Logic**:
```python
def infer_role_hybrid(var, context, equation, book_domain_hint):
    # Try TRM first
    role_trm, confidence = infer_role_trm(var, context, equation, book_domain_hint)

    # High confidence → use TRM (sovereign path)
    if confidence >= 0.8:
        trm.enhance_from_success(...)  # Shadow copy learning
        return role_trm, "TRM"

    # Low confidence → fallback to LLM
    else:
        role_llm, domain_llm = infer_role_llm(...)
        math_galaxy.add_inference_example(...)  # Store for future TRM learning
        return role_llm, "LLM"
```

**Output**: Hybrid function that routes to TRM or LLM based on confidence

**Validation**: Total accuracy ≥ LLM-only baseline (85%+)

---

### Task 6: Create Validation Test

**File**: Create `tests/integration/test_trm_role_inference.py`

**What to Test**: Run hybrid TRM+LLM on held-out validation set (10% of 72 linalg + 40 geo)

**Metrics**:
- **TRM coverage**: % of inferences where TRM confidence ≥ 0.8 (target: ≥30%)
- **TRM accuracy**: % of TRM inferences that match ground truth (target: ≥70%)
- **Total accuracy**: % of all inferences that match ground truth (target: ≥80%)

**Output**: Test that validates Phase 2 hybrid approach

**Validation**: Test passes with metrics above targets

---

## Success Criteria (Phase 2 Complete)

- [ ] Math Galaxy populated (6 domain signatures, 50+ role patterns)
- [ ] Grammar Galaxy populated (2 transformation rules)
- [ ] TRM navigation implemented (queries + executes)
- [ ] Hybrid fallback implemented (TRM + LLM)
- [ ] Validation test passes (≥30% TRM coverage, ≥70% TRM accuracy, ≥80% total accuracy)
- [ ] Metrics tracked (TRM coverage rate, LLM fallback rate, learning progress)

---

## Timeline Estimate

**Week 1-2**: Tasks 1-3 (populate Math Galaxy + Grammar Galaxy)
**Week 3-4**: Tasks 4-5 (TRM navigation + hybrid fallback)
**Week 5**: Task 6 (validation testing)
**Week 6**: Full 23-book re-ingestion with hybrid approach (compare metrics)

**Total**: 6 weeks to Phase 2 complete

---

## Critical Reminders

**Sovereignty Principle**:
- Phase 2: TRM sovereign (high-confidence) + LLM fallback (low-confidence)
- Phase 3: TRM sovereign (100%), zero external dependencies

**Galaxy Universe Paradigm**:
- Math Galaxy stores domain signatures + role patterns (RPN programs)
- Grammar Galaxy stores transformation rules (RPN programs)
- TRM navigates, does NOT store knowledge

**Save Information Principle**:
- Reference Math Galaxy symbols (don't duplicate)
- Store LLM inferences ONCE as pattern examples
- Content-based deduplication

**Dual Client Reality**:
- RPN programs = procedural form (executable)
- Metadata = semantic meaning (readable)
- Both serve humans AND TRM

---

## Questions Before Starting?

**Read the full specification first**: [TEMP/CLAUDE_SOVEREIGNTY_MIGRATION_MULTI_DOMAIN_ROLES_12.28.2025.md](TEMP/CLAUDE_SOVEREIGNTY_MIGRATION_MULTI_DOMAIN_ROLES_12.28.2025.md)

**Then answer**:
1. Do you understand the 3-phase migration path?
2. Do you understand where each component belongs (Math Galaxy vs Grammar Galaxy)?
3. Do you understand the validation criteria?
4. Are there any blockers or concerns?

**If clear, proceed with Task 1** (extract domain signatures).

**If unclear, ask questions before implementing** (architecture clarity > speed).

---

## Expected Outcome

**After Phase 2 complete**:
- 30-50% of role inferences use TRM only (sovereign, zero external dependencies)
- 50-70% of role inferences use LLM fallback (learning phase)
- Total accuracy ≥ 85% (matches current LLM-only baseline)
- Math Galaxy grows from LLM fallback data (continuous learning)
- TRM shadow copy enhancement improves navigation over time

**Path to Phase 3**:
- As Math Galaxy accumulates 500+ patterns from full 23-book ingestion
- As TRM learns from 10,000+ successful inferences
- TRM coverage increases 30% → 50% → 70% → 90%
- Eventually, LLM fallback rate drops below 10% → Phase 3 ready (sovereign)

---

**This is your roadmap to sovereignty.** Read the full spec, ask questions if needed, then proceed with Task 1.

Good luck! 🚀
