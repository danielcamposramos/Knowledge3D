# Codex Prompt: Math Galaxy Knowledge Expansion — Concept Anchors + Procedural Programs

**Date:** March 21, 2026
**Architecture:** Claude (analysis + spec) → Codex (implementation)
**Priority:** HIGH — Math went from 0/500 to 10/50. Plumbing fixed. Now it's a knowledge coverage problem.
**Constraint:** One-process-at-a-time discipline. No parallel benchmark runs.

---

## Context

The Math benchmark wall (0/500) is broken. Codex fixed the answer format and anchor injection. Current state:
- 10/50 on the first 50 MATH problems (all Algebra, Level 3-5)
- The 10 that pass have exact-question concept anchors in the bootstrap
- The 40 that fail have NO concept anchor → TRM can't find the right procedure

The sovereign engine works. The composed head pipeline works. The answer comparison works. **What's missing is knowledge in the Galaxy** — concept anchors that guide the TRM to the right procedure, and procedural programs that actually execute the math.

---

## MATH Dataset Analysis (12,500 problems)

### Distribution by Type
| Type | Count | % | Current Coverage |
|------|------:|--:|-----------------|
| Algebra | 2,931 | 23.4% | ~10 anchors, ~5 programs |
| Intermediate Algebra | 2,198 | 17.6% | ~2 anchors (log rules) |
| Prealgebra | 2,076 | 16.6% | ~0 anchors |
| Number Theory | 1,409 | 11.3% | ~0 anchors |
| Geometry | 1,349 | 10.8% | ~2 anchors (circle center) |
| Precalculus | 1,292 | 10.3% | ~0 anchors |
| Counting & Probability | 1,245 | 10.0% | ~5 anchors (factorial, binomial) |

### Distribution by Difficulty
| Level | Count | Strategy |
|-------|------:|----------|
| Level 1 | 1,001 | Highest ROI — simplest problems, most likely to solve with basic anchors |
| Level 2 | 2,242 | High ROI — one-step application of known rules |
| Level 3 | 2,723 | Medium ROI — multi-step, needs procedure chaining |
| Level 4 | 2,904 | Lower ROI — complex reasoning, needs deep programs |
| Level 5 | 3,628 | Long-term — competition-level, needs sophisticated solver |

### Top Concept × Type Pairs (highest-value anchors to create)
| Type | Concept | Problems | Priority |
|------|---------|-------:|----------|
| Geometry + angle | 714 | P1 |
| Number Theory + integer | 598 | P1 |
| Geometry + triangle | 536 | P1 |
| Geometry + area | 466 | P1 |
| Counting & Probability + probability | 442 | P1 |
| Algebra + integer | 385 | P1 |
| Precalculus + matrix | 358 | P2 |
| Intermediate Algebra + integer | 335 | P1 |
| Algebra + equation | 304 | P1 |
| Geometry + circle | 301 | P1 |
| Prealgebra + angle | 293 | P1 |
| Intermediate Algebra + root | 256 | P1 |
| Intermediate Algebra + polynomial | 235 | P2 |
| Number Theory + remainder | 176 | P1 |
| Algebra + fraction | 176 | P1 |
| Intermediate Algebra + complex | 153 | P2 |
| Counting & Probability + fraction | 185 | P1 |

### Answer Format Distribution
| Format | Count | % | Status |
|--------|------:|--:|--------|
| Simple integer/number | 8,073 | 64.6% | ✅ Comparison works |
| Fraction (\frac) | 1,964 | 15.7% | ✅ LaTeX normalizer handles |
| Other LaTeX (tuples, expressions) | 1,789 | 14.3% | ⚠️ Text comparison fallback |
| Square root (\sqrt) | 457 | 3.7% | ✅ LaTeX normalizer handles |
| Matrix/vector | 162 | 1.3% | ⚠️ Text comparison fallback |
| Simple float | 53 | 0.4% | ✅ Comparison works |

---

## What to Build

### Phase 1: Concept Anchors (Layer 2) — Highest ROI

Concept anchors are the bridge between natural-language question patterns and mathematical procedures. They tell the TRM: "this question pattern → use this rule/program."

**Target: 80 new concept anchors** covering the top question families.

Each anchor goes into `foundational_operations_bootstrap.py` following the existing pattern (see the `math_anchor_piecewise_continuity` pattern already there). Each anchor has:
- `id`: `math_anchor_{concept}_{subtype}`
- `category`: `"rule"`
- `query_anchor`: natural language pattern that matches question text
- `semantics`: what this concept means for the solver
- `metadata.subfield`: the mathematical subfield
- `metadata.math_type`: the MATH dataset type

#### Priority 1 Anchors: Prealgebra (2,076 problems, ~0 current coverage)

These are the EASIEST problems. Maximum score uplift per anchor.

```
1.  prealgebra_arithmetic_operations     — basic +,-,*,/ on integers and fractions
2.  prealgebra_fraction_simplification   — reduce fractions, find common denominators
3.  prealgebra_percentage_conversion     — convert between fractions, decimals, percentages
4.  prealgebra_ratio_proportion          — "if 3 faucets fill in 6 min, how long for 1?"
5.  prealgebra_area_rectangle            — length × width, composite shapes
6.  prealgebra_area_triangle             — (1/2) × base × height
7.  prealgebra_perimeter                 — sum of sides, circumference = 2πr
8.  prealgebra_angle_sum_triangle        — angles sum to 180°
9.  prealgebra_angle_complement          — complementary (90°) and supplementary (180°)
10. prealgebra_mean_median_mode          — average, middle value, most frequent
11. prealgebra_number_line_ordering      — compare, order integers and fractions
12. prealgebra_place_value               — digit positions, rounding
13. prealgebra_unit_conversion           — feet↔inches, hours↔minutes, etc.
14. prealgebra_divisibility_rules        — divisible by 2,3,4,5,6,8,9,10
15. prealgebra_prime_composite           — identify primes, prime factorization
```

#### Priority 1 Anchors: Number Theory (1,409 problems, ~0 current coverage)

```
16. number_theory_gcd_euclidean          — GCD via Euclidean algorithm
17. number_theory_lcm                    — LCM = a*b/GCD(a,b)
18. number_theory_modular_arithmetic     — a mod n, modular addition/multiplication
19. number_theory_remainder_theorem      — polynomial remainder when divided by (x-a)
20. number_theory_divisibility_counting  — "how many integers in range divisible by k"
21. number_theory_prime_factorization    — factor into primes, fundamental theorem
22. number_theory_base_conversion        — convert between decimal, binary, hex, base-n
23. number_theory_floor_ceiling          — ⌊x⌋, ⌈x⌉ evaluation
24. number_theory_digit_sum              — sum of digits, digital root
25. number_theory_congruence             — a ≡ b (mod n), Chinese Remainder Theorem
```

#### Priority 1 Anchors: Geometry (1,349 problems, ~2 current anchors)

```
26. geometry_angle_triangle_sum          — interior angles sum to 180°
27. geometry_angle_parallel_transversal  — alternate interior, corresponding angles
28. geometry_triangle_area               — (1/2)bh, Heron's formula
29. geometry_triangle_pythagorean        — a² + b² = c², 3-4-5, 5-12-13 triples
30. geometry_triangle_similar            — AA, SAS, SSS similarity, proportional sides
31. geometry_triangle_congruent          — SSS, SAS, ASA, AAS congruence
32. geometry_circle_area_circumference   — πr², 2πr
33. geometry_circle_arc_sector           — arc length = rθ, sector area = (1/2)r²θ
34. geometry_circle_inscribed_angle      — inscribed angle = (1/2) central angle
35. geometry_coordinate_distance         — d = √((x₂-x₁)² + (y₂-y₁)²)
36. geometry_coordinate_midpoint         — M = ((x₁+x₂)/2, (y₁+y₂)/2)
37. geometry_coordinate_slope            — m = (y₂-y₁)/(x₂-x₁)
38. geometry_polygon_area                — regular polygon area, shoelace formula
39. geometry_volume_prism_cylinder       — V = Bh
40. geometry_volume_cone_sphere          — V = (1/3)πr²h, V = (4/3)πr³
41. geometry_surface_area                — 2D surface area of 3D solids
```

#### Priority 1 Anchors: Algebra (2,931 problems, ~10 current anchors)

Expand beyond the existing 10:

```
42. algebra_linear_equation_one_var      — ax + b = c → x = (c-b)/a
43. algebra_linear_equation_two_var      — system of 2 equations, substitution/elimination
44. algebra_quadratic_formula            — x = (-b ± √(b²-4ac)) / 2a
45. algebra_quadratic_factoring          — factor ax² + bx + c
46. algebra_completing_the_square        — x² + bx → (x + b/2)² - (b/2)²
47. algebra_absolute_value               — |x-a| = b → x = a±b
48. algebra_function_composition         — f(g(x)), nested evaluation
49. algebra_piecewise_evaluation         — evaluate f(x) by selecting correct branch
50. algebra_domain_range                 — find where function is defined
51. algebra_inverse_function             — f⁻¹(x), swap x and y, solve
52. algebra_polynomial_degree            — highest power of x
53. algebra_polynomial_roots             — find zeros of polynomial
54. algebra_rational_expression          — simplify p(x)/q(x), find restrictions
55. algebra_inequality_linear            — ax + b > c, interval notation
56. algebra_inequality_quadratic         — ax² + bx + c > 0, sign chart
57. algebra_word_problem_rate            — distance = rate × time, work rate
58. algebra_word_problem_mixture         — concentration, combined quantities
59. algebra_arithmetic_sequence          — aₙ = a₁ + (n-1)d, sum = n(a₁+aₙ)/2
60. algebra_geometric_sequence           — aₙ = a₁rⁿ⁻¹, sum = a₁(1-rⁿ)/(1-r)
```

#### Priority 1 Anchors: Counting & Probability (1,245 problems, ~5 current anchors)

```
61. counting_permutation                 — P(n,r) = n!/(n-r)!
62. counting_combination                 — C(n,r) = n!/(r!(n-r)!)
63. counting_complement                  — P(A) = 1 - P(not A)
64. counting_multiplication_principle    — if A has m ways and B has n ways → m×n
65. counting_pigeonhole                  — n+1 items in n boxes → at least one box has 2
66. counting_inclusion_exclusion         — |A∪B| = |A| + |B| - |A∩B|
67. counting_conditional_probability     — P(A|B) = P(A∩B)/P(B)
68. counting_expected_value              — E[X] = Σ xᵢP(xᵢ)
69. counting_stars_and_bars              — distributing n identical items into k bins
70. counting_binomial_theorem            — (a+b)ⁿ = Σ C(n,k)aⁿ⁻ᵏbᵏ
```

#### Priority 2 Anchors: Intermediate Algebra + Precalculus

```
71. intermediate_algebra_polynomial_division  — synthetic division, long division
72. intermediate_algebra_partial_fractions    — decompose rational expressions
73. intermediate_algebra_complex_operations   — (a+bi)(c+di), magnitude, conjugate
74. intermediate_algebra_logarithm_solve      — log equations → exponential form
75. intermediate_algebra_exponential_growth   — A = P(1+r)ᵗ, continuous: Peʳᵗ
76. intermediate_algebra_vietas_formulas      — sum/product of roots from coefficients
77. precalculus_trig_identity                 — sin²+cos²=1, double angle, sum formulas
78. precalculus_trig_solve                    — solve sin(x)=k, find all solutions
79. precalculus_matrix_multiply               — row × column, 2×2 and 3×3
80. precalculus_matrix_determinant            — ad-bc for 2×2, cofactor expansion
81. precalculus_vector_operations             — dot product, cross product, magnitude
82. precalculus_parametric_polar              — x=rcosθ, y=rsinθ, conversions
```

---

### Phase 2: Procedural Programs (Layer 3) — The Solver

Each procedural program is an RPN template that the sovereign engine can EXECUTE on GPU. Currently 23 programs exist. **Target: 50 new programs** covering the highest-frequency question patterns.

Programs go into `ingest_math_rules.py` as new `_build_*()` functions, following the existing pattern (see `_build_exponent_rules`, `_build_factoring_rules`).

#### Priority 1 Programs: Direct Evaluation (can produce numeric answers)

```python
# These produce NUMBERS — directly comparable to expected answers

# 1. Linear equation solver
# ax + b = c  →  PUSH c, PUSH b, SUB, PUSH a, DIV
"ARG_C ARG_B - ARG_A /"

# 2. Quadratic formula (both roots)
# (-b ± √(b²-4ac)) / 2a
"ARG_B neg ARG_B 2 pow 4 ARG_A * ARG_C * - sqrt + ARG_A 2 * /"
"ARG_B neg ARG_B 2 pow 4 ARG_A * ARG_C * - sqrt - ARG_A 2 * /"

# 3. Distance formula
# √((x₂-x₁)² + (y₂-y₁)²)
"ARG_X2 ARG_X1 - 2 pow ARG_Y2 ARG_Y1 - 2 pow + sqrt"

# 4. Midpoint formula
# ((x₁+x₂)/2, (y₁+y₂)/2)
"ARG_X1 ARG_X2 + 2 / ARG_Y1 ARG_Y2 + 2 /"

# 5. Triangle area (base, height)
"ARG_BASE ARG_HEIGHT * 2 /"

# 6. Circle area
"ARG_R 2 pow 3.141592653589793 *"

# 7. Circle circumference
"ARG_R 2 * 3.141592653589793 *"

# 8. Pythagorean theorem (solve for c)
"ARG_A 2 pow ARG_B 2 pow + sqrt"

# 9. GCD (Euclidean) — needs loop, may need multi-step RPN
# For now: direct evaluation for small numbers

# 10. Modular arithmetic
# a mod n
"ARG_A ARG_N mod"

# 11. Floor function
"ARG_X floor"

# 12. Ceiling function
"ARG_X ceil"

# 13. Absolute value
"ARG_X abs"

# 14. Percentage
# (part/whole) × 100
"ARG_PART ARG_WHOLE / 100 *"

# 15. Slope formula
"ARG_Y2 ARG_Y1 - ARG_X2 ARG_X1 - /"

# 16. Simple interest
"ARG_P ARG_R * ARG_T *"

# 17. Compound interest
"ARG_P 1 ARG_R ARG_N / + ARG_N ARG_T * pow *"

# 18. Permutation P(n,r)
"ARG_N ! ARG_N ARG_R - ! /"

# 19. Combination C(n,r)
"ARG_N ! ARG_R ! ARG_N ARG_R - ! * /"

# 20. Arithmetic sequence nth term
"ARG_A1 ARG_N 1 - ARG_D * +"

# 21. Arithmetic sequence sum
"ARG_N ARG_A1 ARG_AN + * 2 /"

# 22. Geometric sequence nth term
"ARG_A1 ARG_R ARG_N 1 - pow *"

# 23. Geometric sequence sum
"ARG_A1 1 ARG_R ARG_N pow - * 1 ARG_R - /"

# 24. Completing the square → vertex
# x² + bx + c → vertex at (-b/2, c - b²/4)
"ARG_B neg 2 / ARG_C ARG_B 2 pow 4 / -"

# 25. Discriminant
"ARG_B 2 pow 4 ARG_A * ARG_C * -"
```

#### Priority 2 Programs: Pattern Application (need question parsing)

```python
# These need the concept anchor to extract parameters from question text first

# 26. System of 2 equations (Cramer's rule)
# a1x + b1y = c1, a2x + b2y = c2
# x = (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
"ARG_C1 ARG_B2 * ARG_C2 ARG_B1 * - ARG_A1 ARG_B2 * ARG_A2 ARG_B1 * - /"

# 27. Heron's formula (triangle area from sides)
# s = (a+b+c)/2, A = √(s(s-a)(s-b)(s-c))
"ARG_A ARG_B + ARG_C + 2 / DUP DUP ARG_A - * OVER ARG_B - * OVER ARG_C - * sqrt"

# 28. Regular polygon interior angle
# (n-2) × 180 / n
"ARG_N 2 - 180 * ARG_N /"

# 29. Sector area
# (θ/360) × πr²
"ARG_THETA 360 / 3.141592653589793 * ARG_R 2 pow *"

# 30. Arc length
# (θ/360) × 2πr
"ARG_THETA 360 / 2 * 3.141592653589793 * ARG_R *"

# 31. Volume: cylinder
"3.141592653589793 ARG_R 2 pow * ARG_H *"

# 32. Volume: cone
"3.141592653589793 ARG_R 2 pow * ARG_H * 3 /"

# 33. Volume: sphere
"4 3.141592653589793 * ARG_R 3 pow * 3 /"

# 34. Inclusion-exclusion (2 sets)
"ARG_A ARG_B + ARG_AB -"

# 35. Expected value (2 outcomes)
"ARG_X1 ARG_P1 * ARG_X2 ARG_P2 * +"

# 36. Base conversion (base b digit to decimal)
# d_n * b^n + d_(n-1) * b^(n-1) + ...
# Multi-step — needs parametric RPN with loop

# 37. Complex number magnitude
# |a+bi| = √(a²+b²)
"ARG_A 2 pow ARG_B 2 pow + sqrt"

# 38. Complex multiplication
# (a+bi)(c+di) = (ac-bd) + (ad+bc)i
"ARG_A ARG_C * ARG_B ARG_D * - ARG_A ARG_D * ARG_B ARG_C * +"

# 39. 2×2 matrix determinant
"ARG_A ARG_D * ARG_B ARG_C * -"

# 40. Dot product (2D)
"ARG_X1 ARG_X2 * ARG_Y1 ARG_Y2 * +"
```

---

### Phase 3: Galaxy Symlinks — Math ↔ Language Bridge

**Critical architectural point from Daniel:** Math stars should symlink to language meaning. The concept anchors need TWO entries:

1. **Math Galaxy entry**: the mathematical procedure (RPN program, rule, formula)
2. **Language Galaxy symlink**: the natural-language pattern that triggers it

This is how the TRM finds math procedures from natural-language questions: Language meaning → symlink → Math procedure → execute.

For each concept anchor, create a Language Galaxy entry:
```python
{
    "id": f"lang_math_symlink_{concept_id}",
    "domain": "language",
    "category": "meaning_symlink",
    "content": f"Questions about {natural_language_description}",
    "metadata": {
        "symlink_target": f"math_anchor_{concept_id}",
        "symlink_galaxy": "Math",
        "query_anchor": natural_language_patterns,  # "find the area", "how many ways", etc.
        "semantics": f"Language-to-math bridge for {concept}",
    }
}
```

**Key insight**: The TRM first navigates Language Galaxy (because the question IS natural language), finds the symlink, follows it to Math Galaxy, retrieves the procedure, and executes. Without the symlink, the TRM stays in Language Galaxy and never reaches the math program.

---

## Implementation Guide

### File Changes

| File | Action |
|------|--------|
| `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py` | Add 80 concept anchors to the bootstrap registry |
| `scripts/ingest_math_rules.py` | Add ~15 new `_build_*()` functions for each math domain |
| `scripts/ingest_meaning_layer.py` | Add Language↔Math symlink generation |
| `tests/test_math_zero_fix.py` | Expand with anchor coverage tests |

### Adding Concept Anchors to Bootstrap

Follow the existing pattern in `foundational_operations_bootstrap.py`. Look for `MATH_CONCEPT_ANCHORS` or the section with `math_anchor_piecewise_continuity`. Each anchor:

```python
{
    "id": "math_anchor_pythagorean_theorem",
    "name": "Pythagorean theorem concept anchor",
    "domain": "math",
    "category": "rule",
    "layer": 2,
    "description": "Layer-2 geometry concept anchor for right-triangle side-length questions.",
    "content": "In a right triangle, a² + b² = c² where c is the hypotenuse.",
    "rpn_program": "ARG_A 2 pow ARG_B 2 pow + sqrt",
    "metadata": {
        "math_type": "Geometry",
        "subfield": "triangles",
        "query_anchor": "right triangle hypotenuse pythagorean theorem side length legs",
        "semantics": "concept anchor for geometry questions about right triangle side relationships",
        "keywords": ["pythagorean", "right triangle", "hypotenuse", "legs"],
        "confidence": 0.91,
    },
}
```

### Adding Procedural Programs to Ingest

Follow the existing `_build_exponent_rules()` pattern in `ingest_math_rules.py`. Each new builder:

```python
def _build_geometry_basic_rules() -> list[dict[str, Any]]:
    items = [
        ("triangle_area_bh", "Triangle area (base×height)", "A = (1/2)bh", "ARG_BASE ARG_HEIGHT * 2 /"),
        ("circle_area", "Circle area", "A = πr²", "ARG_R 2 pow 3.141592653589793 *"),
        # ...
    ]
    return [_rule_spec(...) for ... in items]
```

Then add to `_all_rule_specs()` (or equivalent aggregator function).

### Adding Language Symlinks

In `scripts/ingest_meaning_layer.py`, after the meaning stars are loaded, create symlink entries for each math concept anchor:

```python
def _create_math_language_symlinks(kv: Knowledgeverse) -> int:
    """Create Language Galaxy entries that symlink to Math Galaxy concept anchors."""
    symlinks = [
        {
            "language_pattern": "find area triangle",
            "math_target": "math_anchor_triangle_area",
            "semantics": "area calculation for triangles",
        },
        # ... for each of the 80 concept anchors
    ]
    count = 0
    for s in symlinks:
        kv.galaxy_manager.add_entry("Language", {
            "id": f"lang_symlink_{s['math_target']}",
            "domain": "language",
            "category": "meaning_symlink",
            "content": s["language_pattern"],
            "metadata": {
                "symlink_target": s["math_target"],
                "symlink_galaxy": "Math",
                "query_anchor": s["language_pattern"],
                "semantics": s["semantics"],
            },
        })
        count += 1
    return count
```

---

## Execution Order

1. **Phase 1a: Prealgebra + Number Theory anchors** (25 anchors) — biggest score-per-anchor gain
2. **Phase 1b: Geometry + Algebra anchors** (20 anchors) — high frequency domains
3. **Phase 1c: Counting, Intermediate Algebra, Precalculus anchors** (37 anchors) — complete coverage
4. **Phase 2: Procedural programs** (40 programs) — enable actual solving
5. **Phase 3: Language symlinks** (80 symlinks) — bridge natural language to math
6. **Test: Run first 50 MATH problems** — verify score improvement
7. **Test: Run first 200 MATH problems** (mixed types) — verify cross-type coverage

---

## Success Criteria

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Math 50 (first 50, Algebra) | 10/50 | 25+/50 | Anchors + programs for common Algebra patterns |
| Math 200 (first 200, mixed) | ~20/200 | 60+/200 | Cross-type coverage |
| Math 500 (benchmark slice) | 0/500 | 80+/500 | Meaningful score on official benchmark |
| Concept anchors | ~12 | 80+ | Covering all 7 MATH types |
| Procedural programs | 23 | 60+ | Covering top question families |
| Language symlinks | 0 | 80+ | Every anchor has a language bridge |
| GSM8K regression | 30/1319 | 30+/1319 | No regression — same engine, more knowledge |

---

## Sovereignty Notes

- All concept anchors and procedural programs are **Galaxy entries** (VRAM, sovereign)
- RPN programs execute on **GPU via composed head pipeline** (sovereign)
- Language symlinks are **Galaxy navigation** (sovereign)
- The `ingest_math_rules.py` script is **ingestion-path** (Python, runs once)
- The `foundational_operations_bootstrap.py` is **boot-time** (Python, runs once at startup)
- Once loaded, ALL math reasoning is sovereign: TRM navigates Galaxy → finds anchor → follows symlink → retrieves program → executes RPN on GPU
- **No new Python in hot path**

---

## Key Principle: One Mind

After this expansion, Math and GSM8K use the SAME concept anchors and programs. A word problem about "three faucets filling a tub" (GSM8K-style) and a competition problem about "rate and work" (MATH-style) both match the `algebra_word_problem_rate` anchor and use the same RPN rate program. One mind, one Galaxy, one solver.
