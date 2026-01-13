# Multi-Domain Role Taxonomy — December 27, 2025

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Context**: Mid-size test revealed linear algebra ≠ geometry, need multi-domain roles

---

## Critical Architectural Insight (from Daniel)

> "We are aiming for generalization, and for that, we must have several categories with similar concepts so you can see, as I do, that **many formulae are the same across disciplines with small variances per context**."

**Example**: A² + B² = C²
- **Geometry**: A=leg, B=leg, C=hypotenuse (Pythagorean theorem)
- **Physics**: A=velocity_x, B=velocity_y, C=velocity_magnitude (vector decomposition)
- **Linear Algebra**: A=component_1, B=component_2, C=norm (Euclidean distance)
- **Complex Analysis**: A=real_part, B=imaginary_part, C=modulus

**The Pattern**: Same STRUCTURE, different SEMANTIC ROLES depending on domain context.

**Our Goal**: ONE model that understands context-dependent semantics while recognizing structural invariance.

---

## Multi-Domain Role Taxonomy

### Tier 1A: Geometry & Spatial Measurement
```python
GEOMETRY_ROLES = [
    # Length measurements
    "radius", "diameter", "chord", "arc_length",
    "height", "width", "length", "depth", "thickness",
    "distance", "perimeter", "circumference",

    # Triangle-specific
    "leg", "hypotenuse", "base", "altitude", "median", "side",

    # Angular measurements
    "angle", "central_angle", "inscribed_angle", "radian",

    # Areas & Volumes
    "area", "surface_area", "volume", "cross_section",

    # Coordinate geometry
    "slope", "intercept", "coordinate",
]
```

### Tier 1B: Linear Algebra & Vector Spaces
```python
LINEAR_ALGEBRA_ROLES = [
    # Vector operations
    "vector", "component", "magnitude", "direction",
    "dot_product", "cross_product", "projection",

    # Matrix operations
    "matrix", "element", "row", "column",
    "determinant", "trace", "rank",

    # Eigenanalysis
    "eigenvalue", "eigenvector", "characteristic_polynomial",

    # Spaces & transformations
    "dimension", "basis", "span", "kernel", "image",
    "linear_combination", "transformation",
]
```

### Tier 1C: Calculus & Analysis
```python
CALCULUS_ROLES = [
    # Differentiation
    "derivative", "differential", "rate_of_change",
    "gradient", "partial_derivative", "directional_derivative",
    "slope", "tangent", "secant",

    # Integration
    "integral", "antiderivative", "accumulation",
    "area_under_curve", "definite_integral", "indefinite_integral",

    # Limits & continuity
    "limit", "epsilon", "delta", "bound",
    "supremum", "infimum",

    # Series & sequences
    "sequence", "series", "term", "sum", "convergence",
]
```

### Tier 1D: Physics & Applied Math
```python
PHYSICS_ROLES = [
    # Mechanics
    "position", "velocity", "acceleration", "force",
    "mass", "momentum", "energy", "work", "power",
    "torque", "angular_velocity", "angular_acceleration",

    # Waves & oscillations
    "frequency", "wavelength", "amplitude", "period", "phase",

    # Thermodynamics
    "temperature", "pressure", "volume", "entropy",
    "heat_capacity", "internal_energy",

    # Electromagnetism
    "charge", "current", "voltage", "resistance",
    "electric_field", "magnetic_field", "flux",
]
```

### Tier 1E: Number Theory & Algebra
```python
NUMBER_THEORY_ROLES = [
    # Number properties
    "prime", "composite", "factor", "divisor", "multiple",
    "greatest_common_divisor", "least_common_multiple",

    # Modular arithmetic
    "modulus", "remainder", "quotient", "congruence",

    # Algebraic structures
    "group_element", "ring_element", "field_element",
    "order", "generator", "identity",
]
```

### Tier 1F: Probability & Statistics
```python
STATISTICS_ROLES = [
    # Descriptive statistics
    "mean", "median", "mode", "variance", "standard_deviation",
    "percentile", "quartile", "range",

    # Probability
    "probability", "event", "sample_space", "outcome",
    "expected_value", "distribution", "density",

    # Inference
    "parameter", "statistic", "estimate", "confidence_interval",
    "p_value", "significance_level",
]
```

### Tier 2: Formula Components (Domain-Agnostic)
```python
FORMULA_ROLES = [
    "exponent", "base",
    "coefficient", "constant_factor",
    "numerator", "denominator",
    "radicand", "index",
    "argument", "parameter",
]
```

### Tier 3: Generic Fallbacks (Last Resort)
```python
GENERIC_ROLES = [
    "constant",      # Known mathematical constants (π, e, g, c, h, φ)
    "variable",      # Generic unknown
    "placeholder",   # Temporary notation
    "unknown",       # Cannot determine
]
```

---

## Domain Detection Strategy

### Auto-Detect Domain from Context (with Book Metadata Priority)

```python
def detect_domain(context: str, equation: str, book_domain_hint: Optional[str] = None) -> List[str]:
    """
    Detect mathematical domain(s) from context.
    Returns ordered list of domains (most relevant first).
    """
    ctx_lower = context.lower()
    eq_lower = equation.lower()

    scores = {}
    scores_eq = {}
    scores_book = {}

    # Book metadata hint (highest priority)
    if book_domain_hint:
        scores_book[book_domain_hint] = scores_book.get(book_domain_hint, 0) + 1

    # Geometry indicators (context)
    geo_keywords = {
        "circle", "triangle", "rectangle", "sphere", "cylinder", "cone",
        "polygon", "angle", "perpendicular", "parallel", "tangent",
        "area", "volume", "perimeter", "circumference"
    }
    scores["geometry"] = sum(1 for kw in geo_keywords if kw in ctx_lower)
    scores_eq["geometry"] = sum(1 for kw in ["pi", "π"] if kw in eq_lower)

    # Linear algebra indicators (context + equation cues)
    linalg_keywords = {
        "matrix", "vector", "determinant", "eigenvalue", "eigenvector",
        "linear", "subspace", "basis", "span", "dimension", "rank"
    }
    scores["linear_algebra"] = sum(1 for kw in linalg_keywords if kw in ctx_lower)
    scores_eq["linear_algebra"] = sum(
        1 for kw in ["det(", "trace", "rank", "||", "∥"] if kw in eq_lower
    )

    # Calculus indicators (context + equation cues)
    calc_keywords = {
        "derivative", "integral", "limit", "differential", "gradient",
        "rate of change", "tangent line", "area under", "accumulation"
    }
    scores["calculus"] = sum(1 for kw in calc_keywords if kw in ctx_lower)
    scores_eq["calculus"] = sum(1 for kw in ["d/d", "∂", "∫", "lim"] if kw in eq_lower)

    # Physics indicators
    phys_keywords = {
        "velocity", "acceleration", "force", "energy", "momentum",
        "electric", "magnetic", "wave", "frequency", "mass"
    }
    scores["physics"] = sum(1 for kw in phys_keywords if kw in ctx_lower)
    scores_eq["physics"] = 0

    # Number theory indicators (context + equation cues)
    num_keywords = {
        "prime", "divisor", "factor", "gcd", "lcm", "modulo",
        "congruence", "integer", "rational", "irrational"
    }
    scores["number_theory"] = sum(1 for kw in num_keywords if kw in ctx_lower)
    scores_eq["number_theory"] = sum(1 for kw in ["mod", "≡", "gcd", "lcm"] if kw in eq_lower)

    # Probability/Statistics indicators (context + equation cues)
    stat_keywords = {
        "probability", "random", "distribution", "mean", "variance",
        "standard deviation", "expected value", "sample", "population"
    }
    scores["statistics"] = sum(1 for kw in stat_keywords if kw in ctx_lower)
    scores_eq["statistics"] = sum(1 for kw in ["p(", "e[", "var("] if kw in eq_lower)

    # Weighted total score:
    # - book metadata: 10x
    # - equation cues: 2x
    # - context keywords: 1x
    total = {
        k: (scores.get(k, 0) + 2 * scores_eq.get(k, 0) + 10 * scores_book.get(k, 0))
        for k in scores
    }

    # Sort by score (highest first), break ties by equation cues
    sorted_domains = sorted(
        total.items(),
        key=lambda x: (-x[1], -scores_eq.get(x[0], 0))
    )

    # Return domains with score > 0
    return [domain for domain, score in sorted_domains if score > 0]
```

### Construct Role Choices Based on Domain

```python
def get_role_choices(detected_domains: List[str]) -> List[str]:
    """
    Construct role choices prioritizing detected domains.

    Returns ordered list:
    1. Domain-specific Tier 1 roles (prioritized by domain detection)
    2. Formula components (Tier 2)
    3. Generic fallbacks (Tier 3)
    """
    role_choices = []

    # Add domain-specific roles (in order of relevance)
    for domain in detected_domains:
        if domain == "geometry":
            role_choices.extend(GEOMETRY_ROLES)
        elif domain == "linear_algebra":
            role_choices.extend(LINEAR_ALGEBRA_ROLES)
        elif domain == "calculus":
            role_choices.extend(CALCULUS_ROLES)
        elif domain == "physics":
            role_choices.extend(PHYSICS_ROLES)
        elif domain == "number_theory":
            role_choices.extend(NUMBER_THEORY_ROLES)
        elif domain == "statistics":
            role_choices.extend(STATISTICS_ROLES)

    # Add remaining Tier 1 roles (lower priority)
    all_tier1 = (
        GEOMETRY_ROLES + LINEAR_ALGEBRA_ROLES + CALCULUS_ROLES +
        PHYSICS_ROLES + NUMBER_THEORY_ROLES + STATISTICS_ROLES
    )
    for role in all_tier1:
        if role not in role_choices:
            role_choices.append(role)

    # Add Tier 2 (formula components)
    role_choices.extend(FORMULA_ROLES)

    # Add Tier 3 (generic fallbacks)
    role_choices.extend(GENERIC_ROLES)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for role in role_choices:
        if role not in seen:
            seen.add(role)
            deduped.append(role)

    return deduped
```

### Codex Enhancements (Recommended)

1. **Equation cues count double** (structural signal beats prose).
2. **Tie-break by equation cues** (deterministic ordering).
3. **Domain hint persists** (`symbol_bindings[*].domain_hint`) to disambiguate shared roles like `volume`.
4. **Alias folding** (e.g., `norm` → `magnitude`, `component_1` → `component`, `avg` → `mean`).
5. **Prompt focus**: show top 12–15 prioritized roles and add a brief "avoid unrelated domains unless explicitly mentioned" line.

---

## Enhanced Prompt Template

```python
def _build_prompt_multidomain(
    self,
    *,
    var: str,
    context: str,
    equation: str,
    detected_domains: List[str],
    role_choices: List[str]
) -> str:
    """
    Multi-domain enhanced prompt with:
    - Domain detection hints
    - Cross-domain examples
    - Context-aware role prioritization
    """

    # Detect which domains are relevant
    domain_hints = []
    if "geometry" in detected_domains:
        domain_hints.append("GEOMETRY detected → expect spatial measurements (radius, height, area, volume)")
    if "linear_algebra" in detected_domains:
        domain_hints.append("LINEAR ALGEBRA detected → expect vector/matrix concepts (component, eigenvalue, dimension)")
    if "calculus" in detected_domains:
        domain_hints.append("CALCULUS detected → expect rates/accumulation (derivative, integral, limit)")
    if "physics" in detected_domains:
        domain_hints.append("PHYSICS detected → expect physical quantities (velocity, force, energy)")
    if not domain_hints:
        domain_hints.append("No clear domain detected → use general mathematical roles")

    domain_hint_text = "\n  - ".join(domain_hints)

    # Organize roles by tier for display
    # (This gets complex with multi-domain, so we'll show top 20 prioritized)
    top_roles = role_choices[:20]
    remaining_count = len(role_choices) - 20

    return f"""You are a mathematical semantic role extractor. Your task is to identify the MOST SPECIFIC role of a variable in a mathematical context.

**CONTEXT-BASED ROLE DETECTION**:

Based on the context, the following mathematical domains were detected:
  - {domain_hint_text}

**ALLOWED ROLES** (prioritized by detected domain, total: {len(role_choices)}):

Top priority roles for this context:
  {', '.join(top_roles)}

... and {remaining_count} more roles available

**MULTI-DOMAIN EXAMPLES**:

Example 1 — GEOMETRY (Pythagorean theorem):
CONTEXT: "In a right triangle, the sum of squares of the legs equals the square of the hypotenuse: a² + b² = c²"
EQUATION: a² + b² = c²
VARIABLE: c
DOMAIN: Geometry (triangle keywords detected)
REASONING: Right triangle context, c is opposite the right angle → hypotenuse
ROLE: hypotenuse

Example 2 — PHYSICS (Vector magnitude):
CONTEXT: "The magnitude of velocity v is given by v = √(vₓ² + vᵧ²) where vₓ and vᵧ are components"
EQUATION: v = √(vₓ² + vᵧ²)
VARIABLE: vₓ
DOMAIN: Physics (velocity keyword detected)
REASONING: Physics context, vₓ is one part of velocity vector → component
ROLE: component

Example 3 — LINEAR ALGEBRA (Euclidean norm):
CONTEXT: "The Euclidean norm of vector x in ℝⁿ is ‖x‖ = √(x₁² + x₂² + ... + xₙ²)"
EQUATION: ‖x‖ = √(x₁² + x₂²)
VARIABLE: x₁
DOMAIN: Linear Algebra (vector, norm keywords detected)
REASONING: Vector space context, x₁ is one coordinate → component
ROLE: component

Example 4 — CALCULUS (Derivative):
CONTEXT: "The derivative of position x with respect to time t gives velocity: v = dx/dt"
EQUATION: v = dx/dt
VARIABLE: v
DOMAIN: Calculus (derivative keyword detected)
REASONING: Rate of change of position → velocity (but also a derivative)
ROLE: derivative

Example 5 — NUMBER THEORY (Modular arithmetic):
CONTEXT: "If a ≡ b (mod n), then a and b have the same remainder when divided by n"
EQUATION: a ≡ b (mod n)
VARIABLE: n
DOMAIN: Number Theory (modulo keyword detected)
REASONING: Modular arithmetic context, n is the modulus
ROLE: modulus

Example 6 — AMBIGUOUS STRUCTURE (Same formula, different domains):
CONTEXT: "The area of a circle is A = πr²"
EQUATION: A = πr²
VARIABLE: r
DOMAIN: Geometry (circle keyword detected)
REASONING: Circle geometry, r is distance from center → radius
ROLE: radius

Contrast with:
CONTEXT: "The kinetic energy is E = ½mv² where v is velocity"
EQUATION: E = ½mv²
VARIABLE: v
DOMAIN: Physics (energy, velocity keywords detected)
REASONING: Physics context, v is speed → velocity
ROLE: velocity

**NOTE**: Both use exponent 2, but semantic roles differ based on CONTEXT!

**YOUR TASK**:

CONTEXT:
{context}

EQUATION/FORMULA:
{equation}

VARIABLE: {var}

**STEP-BY-STEP ANALYSIS**:
1. What mathematical domain(s) are indicated by the context? (geometry, linear algebra, calculus, physics, etc.)
2. What role does '{var}' play in the equation '{equation}'?
3. Which domain-specific role from the prioritized list best describes this?
4. If no domain-specific role fits, use formula component (Tier 2) or generic (Tier 3)

**FINAL ANSWER** (respond with ONLY ONE WORD from the allowed roles list):
"""
```

---

## Implementation Changes Required

### File: `sovereign_knowledge_articulator.py`

**Add domain detection**:

```python
def _detect_domain(self, context: str, equation: str) -> List[str]:
    """Detect mathematical domains from context (see taxonomy above)."""
    # Implementation from detect_domain() above
    pass

def _get_role_choices_multidomain(self, detected_domains: List[str]) -> List[str]:
    """Get role choices prioritized by detected domains."""
    # Implementation from get_role_choices() above
    pass
```

**Add role alias mapping** (canonicalize before validation):
```python
ROLE_ALIASES = {
    "norm": "magnitude",
    "component_1": "component",
    "component_2": "component",
    "component_3": "component",
    "component_i": "component",
    "component_j": "component",
    "component_k": "component",
    "avg": "mean",
    "average": "mean",
    "stddev": "standard_deviation",
    "stdev": "standard_deviation",
    "prob": "probability",
    "mod": "modulus",
    "speed": "velocity",
}
```

**Update `infer_role()` method**:

```python
def infer_role(
    self,
    *,
    var: str,
    context: str,
    equation: str,
) -> Tuple[str, str]:
    """Returns (meaning, domain, domain_hint). Meaning is detected multi-domain role."""

    # Detect mathematical domains from context
    detected_domains = self._detect_domain(context, equation)

    # Get role choices prioritized by detected domains
    role_choices = self._get_role_choices_multidomain(detected_domains)

    # Build multi-domain prompt
    prompt = self._build_prompt_multidomain(
        var=var,
        context=context,
        equation=equation,
        detected_domains=detected_domains,
        role_choices=role_choices
    )

    # Run model (rest of implementation unchanged)
    role = self._run_model(prompt, model=self._cfg.model, role_choices=role_choices)
    # ... fallback logic, etc.
```

---

## Testing Strategy: Dual Validation

**Test 1: Geometry Book** (validates Tier 1A)
```bash
# Area_and_Volume.pdf (50 pages)
# Expected: High geometry roles (radius, diameter, area, volume, circumference)
# Success: ≥50% Tier 1A roles, ≥60% non-unknown
```

**Test 2: Linear Algebra Book** (validates Tier 1B)
```bash
# Linear.Algebra.Done.Right.pdf (50 pages)
# Expected: High linear algebra roles (vector, matrix, eigenvalue, dimension)
# Success: ≥50% Tier 1B roles, ≥60% non-unknown
```

**Test 3: Physics Book** (validates Tier 1D) [OPTIONAL]
```bash
# Physical Quantities or Mechanics book (50 pages)
# Expected: High physics roles (velocity, force, energy, mass)
# Success: ≥50% Tier 1D roles, ≥60% non-unknown
```

**Combined Success Criteria**:
- ✅ Each test achieves ≥60% non-unknown
- ✅ Each test shows ≥40% roles from its PRIMARY domain tier
- ✅ Total artifacts across all tests: ≥60
- ✅ Total distinct Tier 1 roles: ≥20

---

## Expected Results

### Per-Domain Coverage Projection

**Geometry Test**:
- Non-unknown: 70-80%
- Tier 1A (geometry): 55-65%
- Tier 2 (formula): 20-30%
- Tier 3 (generic): 10-15%

**Linear Algebra Test**:
- Non-unknown: 75-85%
- Tier 1B (linalg): 50-60%
- Tier 2 (formula): 25-35%
- Tier 3 (generic): 10-15%

**Physics Test** (if run):
- Non-unknown: 65-75%
- Tier 1D (physics): 45-55%
- Tier 2 (formula): 30-40%
- Tier 3 (generic): 10-15%

### Cross-Domain Validation

**Key Insight to Validate**: Same formula structure, different semantic roles

Example to watch for:
- **a² + b² = c²** appears in:
  - Geometry: a=leg, b=leg, c=hypotenuse
  - Physics: a=velocity_x, b=velocity_y, c=velocity_magnitude
  - Linear Algebra: a=component_1, b=component_2, c=norm

**Success**: System correctly assigns DIFFERENT roles to same structure based on context!

---

## Architectural Significance

**This validates the core K3D principle**:

> "Many formulae are the same across disciplines with small variances per context"

**What we're proving**:
- ✅ Structure is invariant (a² + b² = c²)
- ✅ Semantics are context-dependent (leg vs component vs velocity)
- ✅ ONE model can understand BOTH (structural similarity + contextual meaning)

**This is exactly what TRM needs**:
- Navigate Galaxy Universe based on STRUCTURE (find a² + b² = c² pattern)
- Understand semantic MEANING from context (this is geometry, so interpret as Pythagorean)
- Apply knowledge across domains (learned pattern in geometry helps physics)

---

## Timeline Estimate

**Implementation**: 3-4 hours
- Add multi-domain role taxonomy
- Implement domain detection
- Update prompt template
- Test on small subset (verify no syntax errors)

**Validation Testing**: 6-9 hours
- Geometry test: 2-3 hours
- Linear algebra test: 2-3 hours
- Physics test (optional): 2-3 hours
- Analysis: 30 minutes

**Total**: 9-13 hours to complete multi-domain validation

---

## Success Metrics

**Implementation Complete** when:
- [ ] All 6 domain tier definitions added
- [ ] Domain detection function implemented
- [ ] Multi-domain prompt template created
- [ ] Small subset test passes (no errors)

**Validation Complete** when:
- [ ] Geometry test: ≥60% non-unknown, ≥40% Tier 1A
- [ ] Linear algebra test: ≥60% non-unknown, ≥40% Tier 1B
- [ ] Combined: ≥20 distinct Tier 1 roles extracted
- [ ] Cross-domain examples verified (same structure, different roles)

**Architecture Validated** when:
- [ ] Same formula appears with different roles in different contexts
- [ ] Domain detection correctly prioritizes relevant roles
- [ ] Models successfully extract context-dependent semantics

---

## Critical Note

This is NOT just "adding more roles" - this is **validating the fundamental K3D insight**:

**Structure persists, semantics vary by context, ONE model understands BOTH.**

This is what separates K3D from traditional approaches that either:
- Ignore context (treat all a² as "exponent")
- Ignore structure (separate models per domain)

K3D does BOTH: Recognize structure + Understand context = True generalization

---

**Ready to implement? This is the architecturally correct path.** 🚀
