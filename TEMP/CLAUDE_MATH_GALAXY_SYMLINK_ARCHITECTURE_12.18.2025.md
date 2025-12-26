# Math Galaxy Symlink Architecture - Lexical Mismatch Solution

**Date:** December 18, 2025
**Architect:** Claude (Architecture Partner)
**Priority:** CRITICAL - Architectural Correction
**Context:** Phase 4 book artifact validation (MATH/AMC-AIME lexical mismatch)

---

## Critical Architectural Correction

### The Problem

**Phase 4 validation results (TEMP/CODEX_PHASE4_VALIDATION_BOOK_ARTIFACTS_12.18.2025.md):**
- MATH: 2.5% accuracy (unchanged from baseline)
- AMC-AIME: 0.5% accuracy (unchanged from baseline)
- Root cause: Lexical mismatch between book text and problem LaTeX

**Specific issue:**
- Books ingested via `pdftotext` → plain text (e.g., "cos", "sin", "sqrt")
- MATH/AMC problems → LaTeX markup (e.g., "\cos", "\sin", "\sqrt")
- Query tokens don't match → book artifact lookups fail

**Proposed fix (INCORRECT):**
```python
# Normalize LaTeX to plain text during queries
def normalize_latex(text):
    return text.replace("\\cos", "cos").replace("\\sin", "sin")...
```

### Why This Violates Architecture

**User's feedback:**
> "No, this violate the architecture. You are right, we must link cos to /cos but not like that. These are two representations of the same thing - cos in RPN, right? so they must live in the same star - that's what's missing, one star contains meaning, so cos, cosine, /cos whatever it might be called must point to the same meaning - the RPN way of calculating it with what's needed and so on"

**Dual Client Contract violation:**
- External normalization in hot path (sovereignty violation)
- Duplicates data (doesn't follow symlink pattern)
- Loses semantic precision (different contexts for same symbol)

**From DUAL_CLIENT_CONTRACT_SPECIFICATION.md (section 1.6):**

> **Save Information Principle**
>
> **Don't duplicate!** Use references (symlink pattern):
> - Characters stored once with full metadata (font, language, pronunciation, meaning cluster)
> - Words reference character IDs (not duplicate glyphs)
> - Grammar metadata references word IDs (not duplicate strings)
> - Discoveries reference canonical programs (content-based deduplication)

> **Meaning-Based Identity**: One star per meaning, many glyph variants. Visually similar symbols with different meanings (Latin A vs Cyrillic А; π as Greek letter vs π as math constant) MUST remain separate nodes/galaxies; uppercase/lowercase of the same letter meaning stay in one node with variants.

---

## The Correct Solution: Math Galaxy Symlink Pattern

### Architectural Principle

**One star per meaning, multiple forms symlink to it.**

Just as Character Galaxy stores each letter ONCE with procedural rendering + metadata, Math Galaxy should store each mathematical concept ONCE with:
- **Procedural RPN** (how to compute it)
- **Semantic meaning** (what it represents)
- **Symbol variants** (all representations that map to this concept)

### Math Galaxy Star Structure

**Star: cosine_function**
```json
{
  "star_id": "cosine_function",
  "star_type": "trigonometric_function",

  "procedural_rpn": {
    "implementation": "TRIG_COS",  // PTX kernel opcode
    "fallback_rpn": "...",          // Taylor series if needed
    "domain": "angle in radians",
    "range": "[-1, 1]"
  },

  "semantic_meaning": {
    "description": "Ratio of adjacent side to hypotenuse in right triangle",
    "category": "trigonometry",
    "related_concepts": ["sine_function", "tangent_function", "pythagorean_identity"],
    "domain_math": "linear_algebra, calculus, geometry"
  },

  "symbol_variants": {
    "latex": "\\cos",           // LaTeX representation
    "plain": "cos",             // Plain text (pdftotext output)
    "word": "cosine",           // English word
    "rpn": "/cos",              // RPN-style operator
    "unicode": "⁡cos",          // Unicode math symbol (if applicable)
    "alternatives": ["Cos"]     // Case variants
  },

  "applicability": {
    "conditions": [
      "requires angle measurement",
      "output bounded [-1, 1]",
      "input in radians (or convert from degrees)"
    ],
    "common_contexts": [
      "right_triangle_geometry",
      "wave_functions",
      "circular_motion"
    ]
  },

  "provenance": {
    "books": [
      {"book_id": "advanced_calculus", "pages": [45, 67, 89]},
      {"book_id": "dmoi3", "pages": [234]}
    ]
  }
}
```

### Symbol Registry (Index Structure)

**Purpose:** Fast lookup from any variant to the canonical star.

**Implementation:** In-memory hash table (or Galaxy index structure)

```python
# Symbol Registry (conceptual structure)
MATH_SYMBOL_REGISTRY = {
    # Cosine function - all variants point to same star
    "\\cos":   "star:cosine_function",
    "cos":     "star:cosine_function",
    "cosine":  "star:cosine_function",
    "/cos":    "star:cosine_function",
    "Cos":     "star:cosine_function",

    # Sine function
    "\\sin":   "star:sine_function",
    "sin":     "star:sine_function",
    "sine":    "star:sine_function",
    "/sin":    "star:sine_function",

    # Square root
    "\\sqrt":  "star:square_root_function",
    "sqrt":    "star:square_root_function",
    "√":       "star:square_root_function",
    "/sqrt":   "star:square_root_function",

    # Fraction (division)
    "\\frac":  "star:division_operator",
    "/":       "star:division_operator",
    "÷":       "star:division_operator",

    # Summation
    "\\sum":   "star:summation_operator",
    "sum":     "star:summation_operator",
    "Σ":       "star:summation_operator",

    # ... all math symbols ...
}
```

**Key properties:**
- Constant-time lookup: O(1) from variant → star_id
- No duplication: Each star stored once
- Bidirectional: Can query "what variants exist for this star?"
- Extensible: New variants added without modifying star

---

## Architectural Integration

### 1. Math Galaxy Population (Ingestion Phase)

**During book ingestion (BookGalaxyIngester + SovereignKnowledgeArticulator):**

```python
class SovereignKnowledgeArticulator:
    def __init__(self, math_galaxy):
        self.math_galaxy = math_galaxy
        self.symbol_registry = SymbolRegistry()  # NEW

    def _extract_math_symbols(self, latex_content: str):
        """
        Extract LaTeX symbols and create/update Math Galaxy stars.

        IMPORTANT: This does NOT normalize - it creates symlinks!
        """
        # Parse LaTeX to find symbols
        symbols = self._parse_latex_symbols(latex_content)

        for symbol_info in symbols:
            latex_form = symbol_info["latex"]        # e.g., "\cos"
            plain_form = symbol_info["plain"]        # e.g., "cos"
            context = symbol_info["context"]

            # 1. Check if star already exists
            star_id = self.symbol_registry.lookup(latex_form)

            if star_id is None:
                # 2. Create new star (first time seeing this concept)
                star_id = self.math_galaxy.create_star(
                    procedural_rpn=self._derive_rpn(symbol_info),
                    semantic_meaning=context,
                    symbol_variants={
                        "latex": latex_form,
                        "plain": plain_form
                    }
                )

                # 3. Register all variants
                self.symbol_registry.register(latex_form, star_id)
                self.symbol_registry.register(plain_form, star_id)
            else:
                # 4. Star exists - add new variant if not already present
                star = self.math_galaxy.get_star(star_id)
                if plain_form not in star.symbol_variants.values():
                    star.symbol_variants[f"variant_{len(star.symbol_variants)}"] = plain_form
                    self.symbol_registry.register(plain_form, star_id)
```

**Result:**
- Book says "cos" → registers variant, links to star:cosine_function
- Problem has "\cos" → lookup in registry → finds same star:cosine_function
- NO normalization needed!

### 2. TRM Navigation (Inference Phase - Hot Path)

**During problem solving (TRMGalaxyReader):**

```python
class TRMGalaxyReader:
    def __init__(self, math_galaxy):
        self.math_galaxy = math_galaxy
        self.symbol_registry = math_galaxy.symbol_registry  # Reference to registry

    def _lookup_math_concept(self, query_token: str):
        """
        Look up math symbol using symlink registry.

        SOVEREIGN: No external parsing, just hash table lookup.
        """
        # 1. Direct registry lookup (O(1))
        star_id = self.symbol_registry.lookup(query_token)

        if star_id is None:
            # Not found - could be compound expression
            return None

        # 2. Retrieve star from Math Galaxy (VRAM)
        star = self.math_galaxy.get_star(star_id)

        # 3. Return procedural RPN + conditions
        return {
            "rpn": star.procedural_rpn,
            "conditions": star.applicability.conditions,
            "meaning": star.semantic_meaning
        }

    def _generate_book_galaxy_candidates(self, problem_text: str):
        """
        Generate RPN candidates from book knowledge using symlink lookup.
        """
        # Parse problem text to find math symbols
        symbols = self._tokenize_math_symbols(problem_text)

        candidates = []
        for symbol_token in symbols:
            # Symlink lookup (works for BOTH "cos" and "\cos")
            concept = self._lookup_math_concept(symbol_token)

            if concept:
                # Check if conditions are met
                if self._conditions_satisfied(concept["conditions"], problem_text):
                    candidates.append(concept["rpn"])

        return candidates
```

**Result:**
- Query has "\cos x" → tokenize → ["\cos", "x"]
- Lookup "\cos" in registry → star:cosine_function
- Retrieve RPN: "TRIG_COS"
- Check conditions: "requires angle" → satisfied
- Return candidate: "x TRIG_COS"

**CRITICAL: Zero external normalization in hot path! Just VRAM lookups (sovereign).**

### 3. Sovereign Knowledge Articulator Enhancement

**Update artifact extraction to create symlinks:**

```python
class SovereignKnowledgeArticulator:
    def _parse_theorem(self, block: dict) -> KnowledgeArtifact:
        """
        Enhanced to register symbol variants during parsing.
        """
        # Extract theorem statement
        latex_content = block["latex"]

        # OLD: Just extract "lhs = rhs"
        # NEW: Register all symbol variants encountered

        symbols_used = self._extract_math_symbols(latex_content)
        # This populates symbol_registry with symlinks

        # Parse conditions (as before)
        conditions = self._extract_conditions(block)

        # Parse conclusion (as before)
        conclusion_rpn = self._latex_to_rpn(block["conclusion"])

        # Symbol bindings (as before)
        bindings = self._extract_symbol_bindings(block)

        return KnowledgeArtifact(
            conditions=conditions,
            conclusion_rpn=conclusion_rpn,
            symbol_bindings=bindings,
            symbols_used=symbols_used  # NEW: references to Math Galaxy stars
        )
```

---

## Implementation Roadmap

### Phase 1: Math Galaxy Star Infrastructure (Week 1)

**File:** `knowledge3d/training/math_benchmarks/math_galaxy_core.py` (new)

**Components:**
1. **MathGalaxyStar** dataclass (star structure above)
2. **SymbolRegistry** class (hash table for variant → star_id)
3. **MathGalaxy** class (manages stars + registry)

**Deliverable:** Math Galaxy can store stars and register symbol variants

**Test:** Create star for "cos", register variants ["\cos", "cos", "cosine"], lookup all variants

### Phase 2: Sovereign Knowledge Articulator Integration (Week 1)

**File:** `knowledge3d/training/math_benchmarks/sovereign_knowledge_articulator.py` (update)

**Changes:**
1. Add `_extract_math_symbols()` method
2. During artifact parsing, register symbol variants
3. Store symbol references in artifacts (not normalized strings)

**Deliverable:** Book ingestion creates Math Galaxy stars + symlinks

**Test:** Ingest book with "cos", verify star created, verify variants registered

### Phase 3: TRM Navigation Integration (Week 2)

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py` (update)

**Changes:**
1. Add `_lookup_math_concept()` method using symbol registry
2. Update `_generate_book_galaxy_candidates()` to use symlink lookup
3. Remove any LaTeX normalization code (sovereignty compliance)

**Deliverable:** TRM navigates book galaxies using symlinks

**Test:** Query "\cos x" → retrieves same star as "cos x"

### Phase 4: Re-Ingestion + Validation (Week 2)

**Actions:**
1. Re-ingest 8 books with enhanced articulator
2. Validate symbol registry populated (count variants)
3. Run MATH/AMC-AIME benchmarks
4. Compare to baseline (Phase 4 results)

**Expected results:**
- MATH: 2.5% → **8-12%** (lexical mismatch fixed + better artifact coverage)
- AMC-AIME: 0.5% → **5-8%**
- no_rule_match reduction (book lookups now succeed)

---

## Sovereignty Compliance

### Ingestion Phase (NOT hot path)
✅ Can use LaTeX parsers (TexSoup, pyparsing)
✅ Can use pdftotext, PyMuPDF
✅ Can build symbol registry using any tools
✅ Output is sovereign (Math Galaxy stars in VRAM)

### Inference Phase (HOT path)
✅ Symbol registry lookup = hash table (O(1), VRAM)
✅ Math Galaxy star retrieval = VRAM access
✅ RPN execution = Cranium PTX kernels
❌ NO external LaTeX parsing
❌ NO normalization/string processing
❌ NO numpy/sympy/external libraries

**Result:** Fully sovereign hot path maintained.

---

## Comparison to Character Galaxy

**This is the SAME pattern already used for characters!**

### Character Galaxy Structure
```json
{
  "character_star_id": "letter_a",
  "procedural_glyph": "...",  // Bézier curves → line segments
  "semantic_meaning": {
    "unicode": "U+0061",
    "language": "latin",
    "pronunciation": "/eɪ/"
  },
  "variants": {
    "uppercase": "A",
    "lowercase": "a",
    "font_arial": "glyph_ref_1",
    "font_times": "glyph_ref_2"
  }
}
```

**Symbol Registry (for characters):**
```python
CHARACTER_REGISTRY = {
    "a": "star:letter_a",
    "A": "star:letter_a",  // Same star, different variant
    "U+0061": "star:letter_a"
}
```

**Math Galaxy should work EXACTLY the same way:**
```python
MATH_SYMBOL_REGISTRY = {
    "\\cos": "star:cosine_function",
    "cos": "star:cosine_function",  // Same star, different variant
    "cosine": "star:cosine_function"
}
```

**This is NOT a new pattern - it's applying existing architecture to Math Galaxy!**

---

## Example Scenario (Lexical Mismatch Fixed)

### Before (Normalization Approach - WRONG)

**Book text:** "The cosine of an angle..."
**Problem:** "Compute \cos(π/4)"

**Old approach:**
1. Extract from book: "cos" → template
2. Query problem: "\cos" → normalize to "cos"
3. Match template ❌ (external normalization in hot path)

### After (Symlink Approach - CORRECT)

**Book ingestion:**
1. Parse book: "The cosine of an angle..."
2. Register variants: "cos" → star:cosine_function
3. Star contains: procedural_rpn="TRIG_COS", variants={"plain": "cos"}

**Problem solving:**
1. Parse problem: "\cos(π/4)"
2. Tokenize: ["\cos", "(", "π", "/", "4", ")"]
3. Lookup "\cos" in registry → star:cosine_function
4. Retrieve RPN: "TRIG_COS"
5. Build candidate: "π 4 / TRIG_COS"
6. Execute in Cranium ✅

**No normalization! Just symlink lookup (sovereign).**

---

## Semantic Precision Advantage

**Additional benefit of symlink pattern: Context-aware symbol resolution**

### Example: π Symbol

**Multiple meanings:**
1. **Geometric constant** (π ≈ 3.14159...)
   - Star: "geometric_pi_constant"
   - RPN: "PI_CONSTANT" (PTX kernel returns 3.14159...)

2. **Product operator** (Π)
   - Star: "product_operator"
   - RPN: "PRODUCT" (iterative multiplication)

**Symbol Registry:**
```python
{
    "\\pi": "star:geometric_pi_constant",       // Lowercase pi
    "π": "star:geometric_pi_constant",
    "Pi": "star:geometric_pi_constant",

    "\\Pi": "star:product_operator",            // Uppercase Pi
    "Π": "star:product_operator",
    "PRODUCT": "star:product_operator"
}
```

**Context resolution:**
- Problem: "The area of a circle is πr²" → lookup "π" → star:geometric_pi_constant
- Problem: "Compute Π_{i=1}^{n} x_i" → lookup "Π" → star:product_operator

**Normalization would LOSE this distinction!**

---

## Success Criteria

### Technical Validation
1. ✅ Math Galaxy populated with 100+ stars (trigonometry, algebra, calculus operators)
2. ✅ Symbol registry has 300+ variant mappings
3. ✅ Lookup latency <100ns (hash table O(1))
4. ✅ Zero normalization in hot path (sovereignty test)

### Benchmark Improvement
1. ✅ MATH accuracy: 2.5% → 8-12% (lexical mismatch fixed)
2. ✅ AMC-AIME accuracy: 0.5% → 5-8%
3. ✅ no_rule_match reduction (book artifact lookups succeed)
4. ✅ wrong_computation reduction (better conditions + symbol resolution)

### Architectural Compliance
1. ✅ Follows Dual Client Contract (one star per meaning)
2. ✅ Follows Save Information Principle (no duplication)
3. ✅ Sovereignty maintained (VRAM-only hot path)
4. ✅ Same pattern as Character Galaxy (consistent architecture)

---

## Conclusion

**The lexical mismatch issue is NOT a normalization problem - it's a missing infrastructure problem.**

The architecture ALREADY defines the solution (symlink pattern from Dual Client Contract). We just need to:
1. **Apply it to Math Galaxy** (same as Character Galaxy)
2. **Populate symbol registry during ingestion**
3. **Use registry for lookups during inference**

**NO external parsing/normalization in hot path. Fully sovereign. Architecturally correct.**

---

**Next Step:** Hand off to Codex for implementation (Phases 1-4 above).

**Estimated Timeline:** 2 weeks (1 week infrastructure + 1 week integration/validation)

**Expected Impact:** MATH 2.5% → 8-12%, AMC 0.5% → 5-8% (lexical mismatch eliminated + better artifact coverage)

---

**Architect:** Claude (Architecture Partner)
**Date:** December 18, 2025
**Status:** Ready for Implementation
