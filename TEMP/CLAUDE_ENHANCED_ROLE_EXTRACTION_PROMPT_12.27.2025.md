# Enhanced Role Extraction Prompt — December 27, 2025

**Purpose**: Improve LLM accuracy on semantic role extraction for mathematical variables

**Target Accuracy**: >60% non-unknown, >40% geometry roles (from current 39.4% / ~13%)

---

## Enhanced Prompt Template

```python
def _build_prompt_enhanced(self, *, var: str, context: str, equation: str, role_choices: Sequence[str]) -> str:
    """
    Enhanced prompt with:
    - Few-shot examples (3-4 cases)
    - Reasoning chain ("think step by step")
    - Geometric context cues
    - Ambiguity clarification
    """
    # Organize roles by tier for display
    tier1_geo = ["radius", "diameter", "height", "width", "length", "depth",
                 "leg", "hypotenuse", "base", "side", "angle", "arc_length",
                 "circumference", "area", "volume"]
    tier2_formula = ["exponent", "coefficient", "numerator", "denominator", "radicand"]
    tier3_fallback = ["constant", "variable", "unknown"]

    # Detect geometric context cues
    ctx_lower = context.lower()
    geo_cues = []
    if "circle" in ctx_lower or "round" in ctx_lower or "π" in context:
        geo_cues.append("CIRCLE geometry detected → expect: radius, diameter, circumference, area")
    if "triangle" in ctx_lower or "pythagorean" in ctx_lower:
        geo_cues.append("TRIANGLE geometry detected → expect: leg, hypotenuse, base, height, angle")
    if "rectangle" in ctx_lower or "square" in ctx_lower or "box" in ctx_lower:
        geo_cues.append("RECTANGULAR geometry detected → expect: length, width, height, area, volume")
    if "sphere" in ctx_lower or "ball" in ctx_lower:
        geo_cues.append("SPHERE geometry detected → expect: radius, diameter, volume, area")
    if "cylinder" in ctx_lower or "cone" in ctx_lower:
        geo_cues.append("CYLINDER/CONE geometry detected → expect: radius, height, volume")

    geo_hint = "\n".join(f"  - {cue}" for cue in geo_cues) if geo_cues else "  - No clear geometric shape detected"

    return f"""You are a mathematical semantic role extractor. Your task is to identify the MOST SPECIFIC role of a variable in a mathematical context.

**ALLOWED ROLES** (in priority order):

Tier 1 — GEOMETRY MEASUREMENTS (PRIORITIZE THESE):
  {', '.join(tier1_geo)}

Tier 2 — FORMULA COMPONENTS:
  {', '.join(tier2_formula)}

Tier 3 — GENERIC FALLBACKS (LAST RESORT):
  {', '.join(tier3_fallback)}

**INSTRUCTIONS**:
1. Read the context and equation carefully
2. Identify geometric shapes or formulas mentioned
3. Match the variable to the MOST SPECIFIC role from Tier 1 if possible
4. Only use Tier 2 if no geometric role fits
5. Only use 'constant' for known constants (π, e, g, c, φ)
6. Only use 'variable' or 'unknown' if nothing else fits

**GEOMETRIC CONTEXT HINTS**:
{geo_hint}

**EXAMPLES**:

Example 1:
CONTEXT: "The area of a circle is πr², where r is the radius."
EQUATION: A = πr²
VARIABLE: r
REASONING: Context mentions "circle" and "radius" explicitly. Variable r is the radius.
ROLE: radius

Example 2:
CONTEXT: "In a right triangle, a² + b² = c², where c is the hypotenuse."
EQUATION: a² + b² = c²
VARIABLE: c
REASONING: Context mentions "right triangle" and "hypotenuse" explicitly. Pythagorean theorem.
ROLE: hypotenuse

Example 3:
CONTEXT: "The volume of a cylinder is V = πr²h."
EQUATION: V = πr²h
VARIABLE: h
REASONING: Context mentions "cylinder". Formula has r² (radius squared) and h. In cylinder formulas, h is height.
ROLE: height

Example 4:
CONTEXT: "The exponential function e^x grows rapidly."
EQUATION: y = e^x
VARIABLE: x
REASONING: No geometric context. x is in the exponent position of e^x.
ROLE: exponent

**NOW ANALYZE THIS**:

CONTEXT:
{context}

EQUATION/FORMULA:
{equation}

VARIABLE: {var}

**STEP-BY-STEP REASONING**:
1. What geometric shapes or formulas are mentioned? [Think carefully]
2. What role does '{var}' play in the equation '{equation}'?
3. Which Tier 1 role best describes this? If none, which Tier 2? If none, Tier 3?

**FINAL ANSWER** (respond with ONLY ONE WORD from the allowed roles list):
"""
```

---

## Key Enhancements

### 1. **Few-Shot Examples** (Lines 45-75)
- Shows 4 diverse cases: circle (radius), triangle (hypotenuse), cylinder (height), exponential (exponent)
- Demonstrates reasoning process
- **Expected improvement**: +15-25% accuracy

### 2. **Geometric Context Cues** (Lines 22-35)
- Auto-detects shapes: circle, triangle, rectangle, sphere, cylinder, cone
- Provides hints: "CIRCLE geometry → expect radius, diameter, circumference"
- **Expected improvement**: +10-20% geometry role rate

### 3. **Reasoning Chain** (Lines 82-86)
- "STEP-BY-STEP REASONING" section
- Guides model through: shape detection → role analysis → tier selection
- **Expected improvement**: +5-15% accuracy (especially for granite4/qwen2.5)

### 4. **Tier Visibility** (Lines 17-25)
- Explicit tier grouping in prompt
- Clear priority ordering
- **Expected improvement**: +5-10% tier 1 selection rate

---

## Estimated Impact

**Baseline** (current):
- Non-unknown: 39.4%
- Geometry roles: ~13%

**With Enhanced Prompt**:
- Non-unknown: **60-70%** (+20-30%)
- Geometry roles: **45-55%** (+30-40%)
- Tier 1 selection: **>70%** of non-unknowns

---

## Implementation

**Replace in `sovereign_knowledge_articulator.py`**:
- Current `_build_prompt()` → `_build_prompt_enhanced()`
- Add geometric context detection
- Add few-shot examples

**Backward Compatibility**:
- Keep cache keys the same (include model in hash)
- Old cache entries still valid
- New prompt = new cache entries

---

## Testing Strategy

1. **Subset Test** (3 books: areavol, numbersets, physquantities)
   - Run with enhanced prompt
   - Validate: >60% non-unknown, >40% geometry roles

2. **Single Book Test** (Advanced Geometry.pdf, --max-pages 50)
   - High geometry density
   - Should show >70% Tier 1 roles

3. **Full Ingestion** (23 books)
   - Only if Steps 1-2 succeed
   - Expected: 65-75% overall non-unknown rate

---

**Confidence**: High (based on few-shot learning literature + geometric cue detection)
**Risk**: Low (prompt changes only, no model/architecture changes)
**Recommendation**: Implement immediately, validate on subset before full run
