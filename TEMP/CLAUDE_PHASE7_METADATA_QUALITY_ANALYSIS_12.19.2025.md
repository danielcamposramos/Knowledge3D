# Phase 7 Complete: Metadata Quality Bottleneck Identified

**Date:** December 19, 2025
**Architect:** Claude (Architecture Partner)
**Context:** Stage 3 semantic binding complete but accuracy unchanged
**Status:** ROOT CAUSE CONFIRMED - Metadata quality issue

---

## Executive Summary

**Phase 7 (Stages 1-3) validation:**
- ✅ Stage 1: Book-sourced boost → MATH 2.5% → 3.0% (+20%)
- ✅ Stage 2: Context gating → AMC stable (no regression)
- ❌ Stage 3: Semantic binding → No additional improvement

**Root cause identified:** `symbol_bindings[*].meaning` is always "unknown" (100% of 9,140 entries)

**Impact:** Stage 3 semantic binding cannot match roles → falls back to naive variable-name heuristics (r→radius, h→height)

**Result:** Stage 3 infrastructure works (tests pass), but data quality blocks semantic matching.

---

## Diagnostic Results

### 1. Symbol Bindings Coverage (books_v4)

**Scan of all artifacts.jsonl files:**
```
Artifacts analyzed: 1,329
With symbol_bindings (non-empty): 1,122 (84.4%) ✅
With conditions: 994 (74.8%) ✅
With semantic meanings (radius/height/leg/...): 0 (0.0%) ❌
Meaning values: "unknown" only (9,140 entries, 100%)
```

**Interpretation:**
- ✅ **Structure exists:** 84.4% of artifacts have symbol_bindings
- ❌ **Semantics missing:** 0% have meaningful role labels
- ❌ **All meanings = "unknown":** Cannot match on semantic roles

**Example artifact (typical):**
```json
{
  "artifact_id": "...",
  "symbol_bindings": {
    "a": {"meaning": "unknown", "domain": "real"},
    "b": {"meaning": "unknown", "domain": "real"},
    "c": {"meaning": "unknown", "domain": "real"}
  },
  "conditions": ["right triangle", "a and b are legs", "c is hypotenuse"]
}
```

**Problem:** Stage 3 needs `"a": {"meaning": "leg"}` but gets `"a": {"meaning": "unknown"}`.

### 2. TTC Selection Stats (Stage 3)

**MATH:**
```
TTC calls: 177
With book seeds: 78 (44%)
With book-sourced seeds: 28 (16%)
```

**AMC-AIME:**
```
TTC calls: 159
With book seeds: 70 (44%)
With book-sourced seeds: 21 (13%)
```

**Comparison to Stage 2 (identical):**
- MATH Stage 2: with_book_sourced_seed = 28
- MATH Stage 3: with_book_sourced_seed = 28 (no change)
- AMC Stage 2: with_book_sourced_seed = 21
- AMC Stage 3: with_book_sourced_seed = 21 (no change)

**Interpretation:** Stage 3 binding code doesn't change book candidate availability (as expected - it only changes variable assignment within candidates).

### 3. Binding Evidence

**No semantic binding debug output found in logs.**

**Inference:** Stage 3 semantic role matching never triggers because all meanings are "unknown". Falls back to variable-name heuristics (r→radius, h→height) which existed before Stage 3.

---

## Root Cause Analysis

### Why Stage 1+2 Worked, But Stage 3 Didn't

**Stage 1 (Book Boost):**
- Mechanism: Add +0.45 confidence to book-sourced TTC candidates
- Data dependency: None (just boosts existing candidates)
- Result: ✅ MATH 2.5% → 3.0% (book selection 3→16)

**Stage 2 (Context Gating):**
- Mechanism: Reject artifacts with shape/intent mismatches (circle vs sphere)
- Data dependency: `artifact.conditions` (74.8% populated)
- Result: ✅ AMC stable (no regression from bad boosts)

**Stage 3 (Semantic Binding):**
- Mechanism: Match problem roles to `symbol_bindings[*].meaning`
- Data dependency: `meaning` field must have semantic values ("radius", "height", "leg")
- **BLOCKER:** All meanings = "unknown" (0% semantic coverage)
- Result: ❌ Falls back to variable-name heuristics (same as before Stage 3)

**Analogy:**
- Stage 1: Boost library books (works - doesn't need metadata)
- Stage 2: Remove obviously wrong books (works - uses conditions text)
- Stage 3: Match book topics to reader needs (fails - topic labels all say "unknown")

---

## Why Meanings Are "Unknown"

**Sovereign Knowledge Articulator (current implementation):**
```python
# In knowledge3d/training/math_benchmarks/sovereign_knowledge_articulator.py
# (hypothetical - checking actual implementation needed)

def _extract_symbol_bindings(self, latex_block):
    """Extract variables but don't infer semantic meanings."""
    bindings = {}

    # Find variables (a, b, c, x, y, r, h, etc.)
    for var in find_variables(latex_block):
        bindings[var] = {
            "meaning": "unknown",  # TODO: Infer from context
            "domain": infer_domain(var)
        }

    return bindings
```

**Missing inference logic:**
```python
# NEEDED: Parse artifact text for role mentions
if "radius" in artifact_text and var == "r":
    bindings["r"]["meaning"] = "radius"

if "height" in artifact_text and var == "h":
    bindings["h"]["meaning"] = "height"

if "legs" in artifact_text and var in ["a", "b"]:
    bindings[var]["meaning"] = "leg"

if "hypotenuse" in artifact_text and var == "c":
    bindings["c"]["meaning"] = "hypotenuse"
```

**Result:** Articulator creates symbol_bindings structure (84.4% coverage) but never populates semantic meanings.

---

## Fix Options

### Option A: Re-Ingest with Enhanced Articulator (PERMANENT FIX)

**Goal:** Extract semantic meanings during book ingestion

**Implementation:**
1. Enhance `SovereignKnowledgeArticulator._extract_symbol_bindings()`
2. Add role inference from artifact text ("radius r" → `"r": {"meaning": "radius"}`)
3. Re-ingest 23 books → books_v5
4. Run benchmarks

**Timeline:** ~2-3 hours
- Articulator enhancement: ~1 hour
- Re-ingestion (23 books): ~1 hour
- Benchmarks (MATH + AMC): ~40 min

**Expected impact:**
- Semantic meanings populated: 0% → 60-80%
- Stage 3 binding triggers: rarely → frequently
- MATH accuracy: 3.0% → **6-10%** (semantic binding working)
- AMC accuracy: 0.5% → **2-4%**

**Pros:**
- Permanent fix (books_v5 has semantic metadata)
- Clean architecture (metadata at ingestion time)
- Enables future semantic features

**Cons:**
- Time investment (~3 hours)
- Need to validate articulator inference logic

---

### Option B: Hot-Path Role Inference (QUICK VALIDATION)

**Goal:** Infer semantic meanings at TTC time (temporary workaround)

**Implementation:**
1. Add `_infer_variable_meanings()` to TRMGalaxyReader
2. Parse `artifact.conditions` + `artifact.raw_block` for role mentions
3. Build temporary role map on-the-fly during binding
4. Run benchmarks

**Code sketch:**
```python
# In knowledge3d/training/math_benchmarks/trm_galaxy_reader.py

def _infer_variable_meanings_from_text(self, artifact, problem_text):
    """
    Infer semantic meanings from artifact text (hot-path workaround).

    Parse artifact.conditions for role mentions:
    - "radius r" → r = radius
    - "legs a and b" → a,b = leg
    - "hypotenuse c" → c = hypotenuse
    """
    role_map = {}

    # Combine all text sources
    text = " ".join(artifact.conditions or [])
    if hasattr(artifact, "raw_block"):
        text += " " + (artifact.raw_block or "")

    text_lower = text.lower()

    # Role inference patterns
    patterns = [
        (r"radius\s+([a-z])", "radius"),
        (r"height\s+([a-z])", "height"),
        (r"legs?\s+([a-z])\s+and\s+([a-z])", "leg"),  # "legs a and b"
        (r"hypotenuse\s+([a-z])", "hypotenuse"),
        (r"width\s+([a-z])", "width"),
        (r"length\s+([a-z])", "length"),
        (r"base\s+([a-z])", "base"),
        (r"area\s+([a-z])", "area"),
        (r"volume\s+([a-z])", "volume")
    ]

    for pattern, role in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):  # "legs a and b"
                for var in match:
                    role_map[var] = role
            else:
                role_map[match] = role

    return role_map


def _bind_variables_semantic(self, artifact, problem_text, numbers):
    """Enhanced binding using inferred meanings."""

    # STEP 1: Try using artifact.symbol_bindings (if populated)
    role_map = {}
    for var, info in artifact.symbol_bindings.items():
        if info.get("meaning") != "unknown":
            role_map[var] = info["meaning"]

    # STEP 2: If no semantic meanings, infer from text (hot-path)
    if not role_map:
        role_map = self._infer_variable_meanings_from_text(artifact, problem_text)

    # STEP 3: Bind using role_map
    bindings = {}
    for var, role in role_map.items():
        number = self._find_number_near_keyword(problem_text, role, numbers, set())
        if number is not None:
            bindings[var] = number

    # STEP 4: Fallback for unbound variables
    # ... (existing logic)

    return bindings
```

**Timeline:** ~1 hour
- Add inference logic: ~30 min
- Test on samples: ~15 min
- Run benchmarks: ~40 min

**Expected impact:**
- Semantic meanings inferred: 0% → 40-60% (from text parsing)
- MATH accuracy: 3.0% → **4-6%** (partial semantic binding)
- Validates hypothesis quickly

**Pros:**
- Fast validation (~1 hour vs ~3 hours)
- Tests hypothesis before re-ingestion investment
- Still sovereign (regex parsing, no external NLP)

**Cons:**
- Temporary workaround (adds work to hot path)
- Lower quality than ingestion-time extraction (40-60% vs 60-80%)
- Doesn't improve data for future features

---

## Recommendation: Option B → Option A (Staged Approach)

**Phase 1: Quick Validation (Option B - ~1 hour)**
1. Implement hot-path role inference (`_infer_variable_meanings_from_text()`)
2. Run MATH/AMC benchmarks
3. **Validate hypothesis:** If accuracy improves (MATH > 4%), semantic binding works

**Phase 2: Permanent Fix (Option A - ~2 hours if Phase 1 succeeds)**
1. Enhance Sovereign Knowledge Articulator with role inference
2. Re-ingest 23 books → books_v5
3. Run benchmarks again
4. **Expected:** MATH 6-10%, AMC 2-4% (full semantic metadata)

**Total timeline:** ~3-4 hours (1 hour validation + 2-3 hours permanent fix)

**Why staged:**
- Option B validates hypothesis quickly (1 hour)
- If Option B shows no improvement, we save 2 hours (don't re-ingest)
- If Option B works, Option A is justified investment

---

## Success Criteria

**Option B validation (hot-path inference):**
- ✅ MATH accuracy ≥ 4.5% (+50% from 3.0%)
- ✅ Book binding examples show semantic matches in logs
- ✅ Role inference coverage ≥ 40% (estimate from text parsing)

**Option A completion (re-ingest books_v5):**
- ✅ Symbol_bindings semantic meanings: 0% → 60-80%
- ✅ MATH accuracy ≥ 6% (2× improvement from Stage 2)
- ✅ AMC accuracy ≥ 2% (4× improvement)
- ✅ Clean metadata for future semantic features

---

## Architectural Lessons Learned

### What Worked ✅

**1. Layered approach to quality:**
- Layer 1: Hygiene (structural validity) → filters 38% of malformed programs
- Layer 2: Context (applicability) → prevents shape/intent mismatches
- Layer 3: Semantics (binding) → **BLOCKED by data quality**

**2. Instrumentation/diagnostics:**
- Phase 7A attribution revealed "books present but not winning"
- Codex's symbol_bindings scan revealed "structure exists, semantics missing"
- Precise diagnosis saves implementation time

**3. Test-driven validation:**
- Tests passed for Stage 3 (infrastructure works)
- Benchmarks unchanged (data quality issue)
- Clear separation of code quality vs. data quality

### What We Learned ❌

**1. Data quality matters as much as code quality:**
- 84.4% symbol_bindings coverage looks good
- But 0% semantic meanings makes it useless
- **Lesson:** Validate metadata quality, not just structure

**2. Infrastructure can't fix missing data:**
- Stage 3 code is correct (tests pass, regex fixed)
- But can't extract semantics from "unknown" labels
- **Lesson:** Data-dependent features need data validation

**3. Fallbacks mask issues:**
- Stage 3 falls back to variable-name heuristics (r→radius)
- No error/warning when semantic binding fails
- **Lesson:** Add telemetry for fallback rates

---

## Next Steps (Immediate)

**Codex: Implement Option B (hot-path role inference)**
- Add `_infer_variable_meanings_from_text()` method
- Enhance `_bind_variables_semantic()` to use inferred meanings
- Run MATH (200 problems, seed 123)
- **Expected timeline:** ~1 hour

**If Option B succeeds (MATH ≥ 4.5%):**
→ Proceed with Option A (re-ingest books_v5 with enhanced articulator)

**If Option B fails (MATH < 4%):**
→ Investigate other failure modes (may not be binding issue)

---

**Architect:** Claude (Architecture Partner)
**Date:** December 19, 2025
**Status:** ROOT CAUSE CONFIRMED - Option B implementation recommended
