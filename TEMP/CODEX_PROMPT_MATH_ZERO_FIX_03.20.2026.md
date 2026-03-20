# Codex Directive: Fix Math 0/500 — Galaxy Content + Answer Extraction

**Date:** 2026-03-20
**Priority:** CRITICAL
**Author:** Claude (Architecture), directed by Daniel
**Context:** Full benchmark returned Math 0/500. This is a knowledge + extraction problem, not a pipeline problem.

---

## Problem Summary

The full benchmark ran 500 real MATH competition problems (from `/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl`, 12,500 total). Result: **0/500 correct.** The synthetic guard problems (linear equations) scored 80/80. The composed head pipeline works — the Galaxy is empty of math knowledge.

---

## Root Cause Analysis (Two Problems)

### Problem 1: Math Galaxy Has NO Mathematical Knowledge for Real Problems

**Evidence:**

`Math.jsonl` has 4,749 entries, broken down as:
| Category | Count | Useful for MATH benchmark? |
|----------|-------|---------------------------|
| `arithmetic_chain_instance` | 2,028 | NO — these are pre-computed instances like "2+3=5", not general rules |
| `arithmetic_instance` | 1,409 | NO — same, pre-computed trivial arithmetic |
| `linear_equation_instance` | 693 | PARTIAL — only handles ax+b=c form (guard problems) |
| `meaning_star` | 475 | NO — these are MMLU abstract algebra concepts, not solvable math |
| `formula_fact` | 33 | YES but only 33! And no RPN programs to evaluate them |
| `template_program` | 17 | YES but only 17 templates |
| `benchmark_fact` | 20 | NO — these are the old synthetic 20/20 guard answers |
| Other | 74 | Mostly definitions, aliases, symbolic stubs |

**What's missing:** The MATH dataset contains problems in Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, and Precalculus. The Galaxy has:
- NO algebraic manipulation rules (factoring, expanding, completing the square)
- NO combinatorics rules (permutations, combinations, binomial coefficients)
- NO geometry rules (area, volume, angle theorems, coordinate geometry)
- NO number theory rules (modular arithmetic, divisibility, prime factorization)
- NO precalculus rules (trig identities, sequences, series, complex numbers)
- NO general equation-solving strategies beyond linear ax+b=c
- Only 33 formula facts and 17 template programs — need HUNDREDS

Additionally, the `meaning_layer_stars.jsonl` (117,497 entries) is ALL `Foundation/Language` with no galaxy assignment. Zero math entries make it into the meaning layer.

### Problem 2: Answer Extraction Returns Galaxy Entry Names Instead of Computed Answers

**Evidence from health_log.jsonl:**
```
Expected: 0     Got: "Sum All Values"          ← Galaxy entry NAME, not an answer
Expected: 98    Got: "3"                        ← Wrong number
Expected: 4     Got: "+ y(3») = 7ri T(xa..."    ← Corrupted LaTeX fragment
Expected: 17    Got: "P(A) = {∅, {1}, ...}"     ← Power set definition, not 17
Expected: 6     Got: "en_service_area"           ← A Galaxy entry ID
```

**Code location:** `knowledgeverse.py:4804`
```python
answer = str(match.get("answer_text") or match.get("name") or match.get("id") or "").strip()
```

When a real MATH problem is queried, the pipeline:
1. Finds the closest Galaxy entry by embedding similarity
2. That entry has no `answer_text` (0 out of 4,749 entries have one)
3. Falls back to `match.get("name")` → returns the entry's NAME (e.g., "Sum All Values")
4. The RPN evaluation path (`rpn_program`) fires but the RPN programs are grid operations like `GRID SUM`, not algebra solvers
5. The template matching path (`_evaluate_math_template`) fails because no template matches

**This means:** Even if the Galaxy HAD math knowledge, the answer extraction would still fail for problems requiring multi-step algebraic reasoning, because there's no RPN program that solves "find a+b if the piecewise function is continuous."

---

## Fix Plan

### Fix 1: Populate Math Galaxy with Comprehensive Mathematical Rules (CRITICAL)

Create a math knowledge ingestion script that loads procedural math rules into the Math Galaxy. These should be **general rules with RPN programs**, not pre-computed instances.

**Target categories to ingest (minimum):**

**Algebra (for MATH dataset "Algebra" type):**
- Quadratic formula: ax² + bx + c = 0 → x = (-b ± √(b²-4ac)) / 2a
- Factoring patterns: difference of squares, perfect square trinomials, sum/difference of cubes
- Completing the square
- Systems of equations (substitution, elimination)
- Polynomial division, synthetic division
- Rational expressions simplification
- Absolute value equations/inequalities
- Logarithm properties (product, quotient, power, change of base)
- Exponent rules (product, quotient, power, zero, negative)

**Counting & Probability:**
- Permutations: P(n,r) = n!/(n-r)!
- Combinations: C(n,r) = n!/(r!(n-r)!)
- Binomial theorem: (a+b)^n = Σ C(n,k) a^(n-k) b^k
- Probability rules (addition, multiplication, complement, conditional)
- Expected value, variance
- Pigeonhole principle
- Inclusion-exclusion

**Geometry:**
- Area formulas: triangle, rectangle, circle, trapezoid, regular polygon
- Volume formulas: sphere, cylinder, cone, prism, pyramid
- Pythagorean theorem and distance formula
- Angle relationships (supplementary, complementary, vertical, parallel lines)
- Similar triangles, congruence criteria
- Circle theorems (inscribed angle, tangent-radius, power of a point)
- Coordinate geometry (midpoint, slope, distance, equation of line)
- Trigonometric ratios and identities

**Number Theory:**
- Divisibility rules
- GCD/LCM algorithms (Euclidean)
- Modular arithmetic (addition, multiplication, inverse, Chinese Remainder Theorem)
- Prime factorization
- Euler's totient function
- Fermat's little theorem

**Precalculus:**
- Arithmetic/geometric sequence formulas (nth term, sum)
- Trig identities (Pythagorean, double angle, half angle, sum/difference)
- Complex numbers (addition, multiplication, modulus, De Moivre's theorem)
- Matrices (multiplication, determinant, inverse for 2×2 and 3×3)
- Vectors (dot product, cross product, magnitude)

**Each entry MUST have:**
- `id`: unique identifier
- `name`: human-readable rule name
- `category`: `"math_rule"` or `"math_formula"`
- `domain`: `"math"`
- `content`: the rule statement
- `rpn_program`: an RPN program that evaluates the rule given parameters (where applicable)
- `answer_text`: empty (answers are computed, not stored)
- `metadata.tags`: searchable keywords for embedding similarity matching
- `metadata.math_type`: matches MATH dataset types: `"Algebra"`, `"Counting & Probability"`, `"Geometry"`, `"Intermediate Algebra"`, `"Number Theory"`, `"Prealgebra"`, `"Precalculus"`

**Implementation path:**
- New script: `scripts/ingest_math_rules.py`
- Output: append to `/K3D/Knowledge3D.local/galaxies/Math.jsonl`
- Embeddings: use the same embedding pipeline as `ingest_meaning_layer.py`
- Target: minimum 500 rules covering all 7 MATH types

### Fix 2: Fix Answer Extraction for Non-Template Problems

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`, function `_answer_math_query` (line 4786)

The current answer cascade is:
1. GSM8K decomposition → only for GSM8K
2. RPN evaluation → only works if `match` has a working `rpn_program`
3. Template evaluation → only works for recognized templates
4. **Fallback: return `match.get("name")`** ← THIS IS THE BUG

The fallback should NEVER return a Galaxy entry name as a math answer. When no RPN program or template matches, the system needs a better strategy:

**Option A (short-term):** If none of the evaluation paths resolve, extract numeric values from the match context and the query, attempt basic arithmetic composition. If that fails, return the best numeric candidate from the reasoning trace rather than a Galaxy entry name.

**Option B (medium-term):** Build a math expression parser that can:
1. Parse the LaTeX in the problem
2. Find matching rules in the Galaxy
3. Compose an RPN program from matched rules
4. Evaluate the composed program

This is the "Galaxy navigation" approach — the TRM should navigate through math rules and compose them, not just look up the closest embedding match.

**For now, implement Option A** — stop returning Galaxy entry names as answers. At minimum, if no evaluation resolves, return `""` or the best numeric value found in the reasoning trace, not `match.get("name")`.

### Fix 3: Symlink Math-Related Meaning Stars to Math Galaxy (CRITICAL ARCHITECTURE)

**File:** `scripts/ingest_meaning_layer.py`

The meaning layer has 117,497 stars, ALL classified as `Foundation/Language` with no galaxy routing. The `GALAXY_BY_DOMAIN` mapping (line 36) and `_MATH_LEMMAS` (line 46) exist but apparently don't fire — either the domain field in the meaning stars doesn't match, or the routing logic isn't reached.

**Check:** Why do none of the 117,497 meaning stars get routed to Math/Reality/Drawing galaxies? The `ROUTED_MEANING_GALAXIES` tuple on line 35 lists Math, Reality, Drawing, Grammar, Language — but all stars end up as Language only.

**CRITICAL ARCHITECTURAL PRINCIPLE — Daniel's correction:**

Math-related stars should exist in BOTH galaxies via symlink, NOT be moved from Language to Math:

1. **Language Galaxy keeps the meaning star.** The word "addition" lives in Language with its definition, translations, surface forms, pronunciation — everything about what the word MEANS. This is the TRM's navigation layer.

2. **Math Galaxy gets a symlink + the executable content.** The Math Galaxy entry for "addition" holds the RPN program (`a b +`), the formula, the evaluation rules. It symlinks back to the Language star for meaning context.

3. **The Language star instructs the TRM to execute in Math Galaxy.** When the TRM navigates to "addition" in Language, the star's metadata says "for execution, go to Math Galaxy entry X." This is how meaning drives computation.

**This is the existing Save Information Principle (DUAL_CLIENT_CONTRACT_SPECIFICATION.md §1.6):**
- DON'T duplicate — use references (symlink pattern)
- Language star = WHAT it means (navigation/understanding)
- Math star = HOW to compute it (execution/RPN)
- Symlink connects them

**Implementation:**
- When a meaning star matches `_MATH_LEMMAS`, it STAYS in Language Galaxy (don't remove it)
- Additionally, create a Math Galaxy entry that:
  - Has `symlink_to`: the Language star's ID
  - Has `rpn_program`: the executable math operation
  - Has `metadata.language_star_ref`: pointer back to Language meaning
- The Language star gets `metadata.math_galaxy_ref`: pointer to the Math execution entry
- Same pattern applies to Reality (physics lemmas), Drawing (visual lemmas), etc.

**This is exactly the book pattern:** The word "book" lives in Language (what a book IS). When you open the book, its contents load as a Galaxy. Same here: "addition" lives in Language (what addition IS). When the TRM needs to DO addition, it follows the symlink to Math Galaxy where the RPN program lives.

---

## Files to Modify

| File | What |
|------|------|
| **NEW: `scripts/ingest_math_rules.py`** | Comprehensive math rule ingestion (500+ rules with RPN programs) |
| `knowledge3d/knowledgeverse/knowledgeverse.py:4804` | Fix answer fallback — stop returning Galaxy entry names |
| `scripts/ingest_meaning_layer.py` | Symlink math-related meaning stars to Math Galaxy (keep in Language + add Math entry with symlink) |
| `benchmarks/math_competitions.py` | No changes needed — scoring logic is correct |

## Files to Read First

- `knowledge3d/knowledgeverse/knowledgeverse.py` lines 4786-4887 (`_answer_math_query`)
- `scripts/ingest_meaning_layer.py` full file (especially galaxy routing logic)
- `/K3D/Knowledge3D.local/galaxies/Math.jsonl` (current content, 4749 entries)
- `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py` (how existing Math entries were created)
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` (available RPN opcodes for math programs)

## Success Criteria

1. Math Galaxy has 500+ general math rules covering all 7 MATH dataset types
2. Each rule has meaningful `rpn_program` and `metadata.tags` for similarity matching
3. Answer extraction never returns a Galaxy entry name — returns computed value or empty
4. Meaning layer stars with math keywords STAY in Language AND get symlinked Math Galaxy entries (Language = meaning, Math = execution)
5. Re-run Math benchmark: target >0/500 (any improvement proves the fix works)
6. Existing benchmarks don't regress (ARC 10/120, GSM8K 29/1319, MMLU 3211/14042)

## Sovereignty Compliance

- All math rules are RPN programs in the Galaxy (sovereign)
- Ingestion script can use any Python libraries (ingestion path is flexible)
- Hot path remains PTX + Galaxy + RPN only
- No Python math solvers in the inference loop

---

## Reference: MATH Dataset Types and Distribution

From `/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl` (12,500 problems):
- Algebra
- Counting & Probability
- Geometry
- Intermediate Algebra
- Number Theory
- Prealgebra
- Precalculus

Each problem has LaTeX formatting with `\boxed{answer}` in the solution field. The answer extraction (`_extract_math_answer`) correctly parses these.

## Reference: Current Math.jsonl Entry Format

```json
{
    "category": "formula_fact",
    "content": "sin^2(x) + cos^2(x) = 1",
    "description": "Pythagorean trigonometric identity",
    "domain": "math",
    "id": "math_identity_sin2_plus_cos2",
    "metadata": {
        "aliases": ["trigonometric identity", "sin squared plus cos squared"],
        "answer": "sin^2(x) + cos^2(x) = 1",
        "bootstrap": "deterministic_foundation_v5",
        "confidence": 0.9,
        "meaning_ref": "math_identity_sin2_plus_cos2"
    },
    "name": "sin²(x) + cos²(x) = 1",
    "rpn_program": ""
}
```

New entries should follow this format, with populated `rpn_program` fields.
