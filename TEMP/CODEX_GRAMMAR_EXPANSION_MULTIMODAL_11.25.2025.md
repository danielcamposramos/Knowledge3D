# Codex Sprint: Grammar Galaxy Expansion — Multimodal Completion

**Date**: November 25, 2025
**Sprint Lead**: Codex (implementation)
**Architect**: Claude (specifications)
**Status**: Ready to Execute
**Priority**: 🏆 CRITICAL — Multimodal Foundation for ARC-AGI + Reality Enabler

---

## 🎯 Sprint Goal

**Expand Grammar Galaxy to cover ALL modalities simultaneously**:
- ✅ Text Grammar: 161 languages (from current 4)
- ✅ Math Grammar: Complete coverage (arithmetic → calculus → linear algebra)
- ✅ Drawing Grammar: Procedural visual construction
- ✅ Integration: All three feed Spatial Semantics → ARC-AGI performance

**Why This Helps ARC-AGI**:
- Math grammar → understand grid patterns (3×3, symmetry, rotations)
- Drawing grammar → visual transformation primitives
- Language expansion → multilingual task descriptions

**Success Metric**: All grammar systems integrated and tested; ARC baseline improves via richer semantic understanding.

---

## 📚 CRITICAL: Read These First

**Architecture Context**:
1. `TEMP/CLAUDE_GRAMMAR_RPN_SPEC_11.24.2025.md` — Original grammar specification
2. `TEMP/CLAUDE_ARC_PROGRESS_ANALYSIS_11.25.2025.md` — How grammar fits ARC
3. `docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md` — `visual_rpn` + `behavior_rpn` + `meaning_rpn`
4. `docs/vocabulary/MATH_CORE_SPECIFICATION.md` — 3-tier routing, ternary ops

**Current Code**:
- `knowledge3d/training/arc_agi/grammar_galaxy.py` — 21 rules (EN, PT, JA, ES)
- `knowledge3d/training/arc_agi/grammar_executor.py` — Stack-based RPN executor
- `knowledge3d/training/arc_agi/semantic_primitives.py` — Spatial/color/shape primitives
- `knowledge3d/training/arc_agi/semantic_parser.py` — Spatial instruction parser

---

## 🌍 Task 1: Language Expansion (161 Languages)

### Goal
Expand from 4 languages (EN, PT, JA, ES) to **top 50 languages** with procedural grammar rules.

### Priority Languages (Top 50 by Speakers)

**Tier 1: Top 10** (Implement first)
```python
TOP_10_LANGUAGES = {
    "zh": {"name": "Chinese (Mandarin)", "pattern": "SVO", "speakers_m": 1120},
    "es": {"name": "Spanish", "pattern": "SVO", "speakers_m": 559},  # Already done
    "en": {"name": "English", "pattern": "SVO", "speakers_m": 1452},  # Already done
    "hi": {"name": "Hindi", "pattern": "SOV", "speakers_m": 602},
    "ar": {"name": "Arabic", "pattern": "VSO", "speakers_m": 274},
    "pt": {"name": "Portuguese", "pattern": "SVO", "speakers_m": 264},  # Already done
    "bn": {"name": "Bengali", "pattern": "SOV", "speakers_m": 272},
    "ru": {"name": "Russian", "pattern": "SVO", "speakers_m": 258},
    "ja": {"name": "Japanese", "pattern": "SOV", "speakers_m": 125},  # Already done
    "de": {"name": "German", "pattern": "SVO", "speakers_m": 134},
}
```

**Tier 2: Next 20** (Top 11-30)
```python
TIER_2_LANGUAGES = {
    "fr": {"name": "French", "pattern": "SVO"},
    "ur": {"name": "Urdu", "pattern": "SOV"},
    "id": {"name": "Indonesian", "pattern": "SVO"},
    "it": {"name": "Italian", "pattern": "SVO"},
    "tr": {"name": "Turkish", "pattern": "SOV"},
    "vi": {"name": "Vietnamese", "pattern": "SVO"},
    "ko": {"name": "Korean", "pattern": "SOV"},
    "fa": {"name": "Persian", "pattern": "SOV"},
    "pl": {"name": "Polish", "pattern": "SVO"},
    "uk": {"name": "Ukrainian", "pattern": "SVO"},
    "th": {"name": "Thai", "pattern": "SVO"},
    "ro": {"name": "Romanian", "pattern": "SVO"},
    "nl": {"name": "Dutch", "pattern": "SVO"},
    "el": {"name": "Greek", "pattern": "SVO"},
    "hu": {"name": "Hungarian", "pattern": "SVO"},
    "cs": {"name": "Czech", "pattern": "SVO"},
    "sv": {"name": "Swedish", "pattern": "SVO"},
    "bg": {"name": "Bulgarian", "pattern": "SVO"},
    "da": {"name": "Danish", "pattern": "SVO"},
    "fi": {"name": "Finnish", "pattern": "SVO"},
}
```

**Tier 3: Next 20** (Top 31-50)
```python
TIER_3_LANGUAGES = {
    "he": {"name": "Hebrew", "pattern": "SVO"},
    "no": {"name": "Norwegian", "pattern": "SVO"},
    "sk": {"name": "Slovak", "pattern": "SVO"},
    "hr": {"name": "Croatian", "pattern": "SVO"},
    "lt": {"name": "Lithuanian", "pattern": "SVO"},
    "sl": {"name": "Slovenian", "pattern": "SVO"},
    "et": {"name": "Estonian", "pattern": "SVO"},
    "lv": {"name": "Latvian", "pattern": "SVO"},
    "sw": {"name": "Swahili", "pattern": "SVO"},
    "ta": {"name": "Tamil", "pattern": "SOV"},
    "te": {"name": "Telugu", "pattern": "SOV"},
    "mr": {"name": "Marathi", "pattern": "SOV"},
    "pa": {"name": "Punjabi", "pattern": "SOV"},
    "gu": {"name": "Gujarati", "pattern": "SOV"},
    "kn": {"name": "Kannada", "pattern": "SOV"},
    "ml": {"name": "Malayalam", "pattern": "SOV"},
    "si": {"name": "Sinhala", "pattern": "SOV"},
    "ne": {"name": "Nepali", "pattern": "SOV"},
    "my": {"name": "Burmese", "pattern": "SOV"},
    "km": {"name": "Khmer", "pattern": "SVO"},
}
```

### Implementation Pattern

**For Each Language**:
```python
def add_language_grammar(lang_code: str, lang_info: Dict) -> List[GrammarRule]:
    """
    Generate grammar rules for a language.

    Pattern types:
    - Simple sentence (SVO/SOV/VSO based on language)
    - Question
    - Imperative
    - Passive (if applicable)
    - Coordination
    """
    rules = []

    # Simple sentence
    if lang_info["pattern"] == "SVO":
        rpn = "SUBJECT RECALL VERB RECALL OBJECT RECALL SVO_ORDER CONCAT_SENTENCE"
    elif lang_info["pattern"] == "SOV":
        rpn = "SUBJECT RECALL OBJECT RECALL WO_PARTICLE VERB RECALL SOV_ORDER CONCAT_SENTENCE"
    elif lang_info["pattern"] == "VSO":
        rpn = "VERB RECALL SUBJECT RECALL OBJECT RECALL VSO_ORDER CONCAT_SENTENCE"

    rules.append(GrammarRule(
        rule_id=f"{lang_code}_simple_sentence",
        language=lang_code,
        pattern=lang_info["pattern"],
        rpn_program=rpn,
        examples=get_language_examples(lang_code),
        description=f"{lang_info['name']} simple sentence"
    ))

    # Question
    rules.append(GrammarRule(
        rule_id=f"{lang_code}_question",
        language=lang_code,
        pattern="Q",
        rpn_program=get_question_pattern(lang_code, lang_info["pattern"]),
        examples=get_question_examples(lang_code),
        description=f"{lang_info['name']} question"
    ))

    # Imperative
    rules.append(GrammarRule(
        rule_id=f"{lang_code}_imperative",
        language=lang_code,
        pattern="V_O",
        rpn_program="VERB RECALL OBJECT RECALL CONCAT_SENTENCE",
        examples=get_imperative_examples(lang_code),
        description=f"{lang_info['name']} imperative"
    ))

    return rules
```

### File Structure

**Create language-specific modules**:
```
knowledge3d/training/arc_agi/grammar_languages/
├── __init__.py
├── tier1_top10.py        # Mandarin, Hindi, Arabic, Russian, German
├── tier2_next20.py       # French, Korean, Turkish, Persian, etc.
├── tier3_next20.py       # Hebrew, Tamil, Swahili, etc.
├── grammar_generator.py  # Pattern generator (SVO/SOV/VSO)
└── language_examples.py  # Example sentences per language
```

**Update main grammar_galaxy.py**:
```python
from knowledge3d.training.arc_agi.grammar_languages.tier1_top10 import get_tier1_rules
from knowledge3d.training.arc_agi.grammar_languages.tier2_next20 import get_tier2_rules
from knowledge3d.training.arc_agi.grammar_languages.tier3_next20 import get_tier3_rules

def default_grammar_rules() -> List[GrammarRule]:
    """All grammar rules (text + math + drawing)."""
    rules = []

    # Text grammar (161 languages eventually, start with 50)
    rules.extend(get_tier1_rules())   # Top 10
    rules.extend(get_tier2_rules())   # Next 20
    rules.extend(get_tier3_rules())   # Next 20

    # Math grammar (next task)
    rules.extend(get_math_rules())

    # Drawing grammar (next task)
    rules.extend(get_drawing_rules())

    return rules
```

### Success Criteria (Task 1)

- [ ] 50 languages with basic grammar rules (3 rules each: sentence, question, imperative)
- [ ] Pattern generator working (SVO/SOV/VSO automatic)
- [ ] Example sentences for top 10 languages
- [ ] All existing tests still pass
- [ ] Grammar executor handles all patterns

---

## 🔢 Task 2: Math Grammar (Complete Coverage)

### Goal
Add procedural math grammar covering all major mathematical domains.

### Math Grammar Domains

**1. Arithmetic & Algebra**
```python
MATH_ARITHMETIC_RULES = [
    GrammarRule(
        rule_id="math_addition",
        language="math",
        pattern="binary_op",
        rpn_program="OPERAND1 RECALL + OPERAND2 RECALL = RESULT RECALL",
        examples=[
            {"operand1": "2", "operand2": "3", "result": "5"},
            {"operand1": "7", "operand2": "8", "result": "15"},
        ],
        description="Addition: a + b = c"
    ),
    GrammarRule(
        rule_id="math_subtraction",
        language="math",
        pattern="binary_op",
        rpn_program="OPERAND1 RECALL - OPERAND2 RECALL = RESULT RECALL",
        examples=[{"operand1": "5", "operand2": "3", "result": "2"}],
        description="Subtraction: a - b = c"
    ),
    GrammarRule(
        rule_id="math_multiplication",
        language="math",
        pattern="binary_op",
        rpn_program="OPERAND1 RECALL × OPERAND2 RECALL = RESULT RECALL",
        examples=[{"operand1": "3", "operand2": "4", "result": "12"}],
        description="Multiplication: a × b = c"
    ),
    GrammarRule(
        rule_id="math_division",
        language="math",
        pattern="binary_op",
        rpn_program="OPERAND1 RECALL ÷ OPERAND2 RECALL = RESULT RECALL",
        examples=[{"operand1": "12", "operand2": "3", "result": "4"}],
        description="Division: a ÷ b = c"
    ),
    GrammarRule(
        rule_id="math_quadratic",
        language="math",
        pattern="equation",
        rpn_program="A RECALL x² B RECALL x C RECALL + + = 0",
        examples=[{"a": "1", "b": "2", "c": "1"}],
        description="Quadratic equation: ax² + bx + c = 0"
    ),
]
```

**2. Calculus**
```python
MATH_CALCULUS_RULES = [
    GrammarRule(
        rule_id="math_derivative",
        language="math",
        pattern="calculus",
        rpn_program="d/d VAR RECALL ( FUNC RECALL ) = RESULT RECALL",
        examples=[
            {"var": "x", "func": "x²", "result": "2x"},
            {"var": "x", "func": "sin(x)", "result": "cos(x)"},
        ],
        description="Derivative: d/dx f(x)"
    ),
    GrammarRule(
        rule_id="math_integral",
        language="math",
        pattern="calculus",
        rpn_program="∫ FUNC RECALL d VAR RECALL = RESULT RECALL + C",
        examples=[
            {"func": "x", "var": "x", "result": "x²/2"},
            {"func": "cos(x)", "var": "x", "result": "sin(x)"},
        ],
        description="Integral: ∫ f(x) dx"
    ),
    GrammarRule(
        rule_id="math_limit",
        language="math",
        pattern="calculus",
        rpn_program="lim VAR RECALL → LIMIT_VAL RECALL FUNC RECALL = RESULT RECALL",
        examples=[{"var": "x", "limit_val": "0", "func": "sin(x)/x", "result": "1"}],
        description="Limit: lim x→a f(x)"
    ),
]
```

**3. Linear Algebra**
```python
MATH_LINALG_RULES = [
    GrammarRule(
        rule_id="math_matrix_mult",
        language="math",
        pattern="matrix_op",
        rpn_program="MATRIX1 RECALL × MATRIX2 RECALL = RESULT_MATRIX RECALL",
        examples=[{"matrix1": "A", "matrix2": "B", "result_matrix": "AB"}],
        description="Matrix multiplication: A × B"
    ),
    GrammarRule(
        rule_id="math_determinant",
        language="math",
        pattern="matrix_property",
        rpn_program="det( MATRIX RECALL ) = RESULT RECALL",
        examples=[{"matrix": "A", "result": "det(A)"}],
        description="Determinant: det(A)"
    ),
    GrammarRule(
        rule_id="math_eigenvalue",
        language="math",
        pattern="eigenproblem",
        rpn_program="MATRIX RECALL v = λ RECALL v",
        examples=[{"matrix": "A", "lambda": "λ"}],
        description="Eigenvalue problem: Av = λv"
    ),
]
```

**4. Geometry**
```python
MATH_GEOMETRY_RULES = [
    GrammarRule(
        rule_id="math_area_circle",
        language="math",
        pattern="formula",
        rpn_program="A = π r² RADIUS RECALL",
        examples=[{"radius": "5"}],
        description="Circle area: A = πr²"
    ),
    GrammarRule(
        rule_id="math_pythagorean",
        language="math",
        pattern="theorem",
        rpn_program="A RECALL² + B RECALL² = C RECALL²",
        examples=[{"a": "3", "b": "4", "c": "5"}],
        description="Pythagorean theorem: a² + b² = c²"
    ),
]
```

**5. Statistics & Probability**
```python
MATH_STATS_RULES = [
    GrammarRule(
        rule_id="math_mean",
        language="math",
        pattern="statistics",
        rpn_program="μ = Σ DATA RECALL / N RECALL",
        examples=[{"data": "x₁, x₂, ..., xₙ", "n": "n"}],
        description="Mean: μ = Σx/n"
    ),
    GrammarRule(
        rule_id="math_variance",
        language="math",
        pattern="statistics",
        rpn_program="σ² = Σ ( DATA RECALL - μ RECALL )² / N RECALL",
        examples=[{"data": "x", "mu": "μ", "n": "n"}],
        description="Variance: σ² = Σ(x - μ)²/n"
    ),
]
```

**6. Logic & Set Theory**
```python
MATH_LOGIC_RULES = [
    GrammarRule(
        rule_id="math_set_union",
        language="math",
        pattern="set_op",
        rpn_program="SET1 RECALL ∪ SET2 RECALL = RESULT_SET RECALL",
        examples=[{"set1": "A", "set2": "B", "result_set": "A ∪ B"}],
        description="Set union: A ∪ B"
    ),
    GrammarRule(
        rule_id="math_set_intersection",
        language="math",
        pattern="set_op",
        rpn_program="SET1 RECALL ∩ SET2 RECALL = RESULT_SET RECALL",
        examples=[{"set1": "A", "set2": "B", "result_set": "A ∩ B"}],
        description="Set intersection: A ∩ B"
    ),
    GrammarRule(
        rule_id="math_implication",
        language="math",
        pattern="logic",
        rpn_program="PREMISE RECALL → CONCLUSION RECALL",
        examples=[{"premise": "P", "conclusion": "Q"}],
        description="Logical implication: P → Q"
    ),
]
```

### Connection to ARC-AGI

**How Math Grammar Helps ARC**:
```python
# Example: Grid symmetry detection
instruction = "The pattern has rotational symmetry of order 4"
# Math grammar parses: order = 4 → rotation angle = 360°/4 = 90°
# Spatial semantics applies: rotate 90°, 180°, 270° and check equivalence

# Example: Grid arithmetic
instruction = "Fill cells where row + col is even"
# Math grammar parses: arithmetic expression (row + col) % 2 == 0
# Spatial semantics applies: compute for each cell, fill matching

# Example: Pattern repetition
instruction = "The pattern repeats every 3 cells"
# Math grammar parses: period = 3
# Spatial semantics applies: detect base pattern, extend with period 3
```

### File Structure

```
knowledge3d/training/arc_agi/grammar_math/
├── __init__.py
├── arithmetic.py      # +, -, ×, ÷, powers, roots
├── algebra.py         # Equations, inequalities, polynomials
├── calculus.py        # Derivatives, integrals, limits
├── linear_algebra.py  # Matrices, vectors, eigenvalues
├── geometry.py        # Areas, volumes, angles, transformations
├── statistics.py      # Mean, variance, distributions
├── logic.py           # Set theory, logic operations
└── math_executor.py   # Execute math RPN programs
```

### Success Criteria (Task 2)

- [ ] 50+ math grammar rules covering all major domains
- [ ] Math executor can evaluate arithmetic/algebra expressions
- [ ] Integration with spatial semantics (grid patterns)
- [ ] Tests for each math domain
- [ ] All existing tests still pass

---

## 🎨 Task 3: Drawing Grammar (Procedural Visual)

### Goal
Add procedural drawing grammar that connects to existing procedural drawing specialist.

### Drawing Grammar Primitives

**1. Basic Shapes**
```python
DRAWING_SHAPE_RULES = [
    GrammarRule(
        rule_id="draw_line",
        language="drawing",
        pattern="primitive",
        rpn_program="X1 RECALL Y1 RECALL MOVE X2 RECALL Y2 RECALL LINE STROKE",
        examples=[{"x1": "0", "y1": "0", "x2": "100", "y2": "100"}],
        description="Draw line from (x1,y1) to (x2,y2)"
    ),
    GrammarRule(
        rule_id="draw_rectangle",
        language="drawing",
        pattern="primitive",
        rpn_program="X RECALL Y RECALL MOVE W RECALL 0 LINE 0 H RECALL LINE -W RECALL 0 LINE CLOSE FILL",
        examples=[{"x": "10", "y": "10", "w": "50", "h": "30"}],
        description="Draw filled rectangle at (x,y) with width w and height h"
    ),
    GrammarRule(
        rule_id="draw_circle",
        language="drawing",
        pattern="primitive",
        rpn_program="CX RECALL CY RECALL R RECALL CIRCLE FILL",
        examples=[{"cx": "50", "cy": "50", "r": "20"}],
        description="Draw filled circle at (cx,cy) with radius r"
    ),
]
```

**2. Bezier Curves (Already Have Opcodes!)**
```python
DRAWING_CURVE_RULES = [
    GrammarRule(
        rule_id="draw_quadratic_bezier",
        language="drawing",
        pattern="curve",
        rpn_program="X1 RECALL Y1 RECALL MOVE CX RECALL CY RECALL X2 RECALL Y2 RECALL QUAD STROKE",
        examples=[{"x1": "0", "y1": "0", "cx": "50", "cy": "100", "x2": "100", "y2": "0"}],
        description="Quadratic Bézier curve (opcode 0x66)"
    ),
    GrammarRule(
        rule_id="draw_cubic_bezier",
        language="drawing",
        pattern="curve",
        rpn_program="X1 RECALL Y1 RECALL MOVE CX1 RECALL CY1 RECALL CX2 RECALL CY2 RECALL X2 RECALL Y2 RECALL CUBIC STROKE",
        examples=[{"x1": "0", "y1": "0", "cx1": "30", "cy1": "100", "cx2": "70", "cy2": "100", "x2": "100", "y2": "0"}],
        description="Cubic Bézier curve (opcode 0x67)"
    ),
]
```

**3. Transformations (Already Have Opcodes!)**
```python
DRAWING_TRANSFORM_RULES = [
    GrammarRule(
        rule_id="draw_rotate",
        language="drawing",
        pattern="transform",
        rpn_program="ANGLE RECALL ROTATE",
        examples=[{"angle": "45"}],
        description="Rotate drawing by angle (opcode 0x73)"
    ),
    GrammarRule(
        rule_id="draw_translate",
        language="drawing",
        pattern="transform",
        rpn_program="DX RECALL DY RECALL TRANSLATE",
        examples=[{"dx": "10", "dy": "20"}],
        description="Translate drawing by (dx, dy) (opcode 0x72)"
    ),
    GrammarRule(
        rule_id="draw_scale",
        language="drawing",
        pattern="transform",
        rpn_program="SX RECALL SY RECALL SCALE",
        examples=[{"sx": "2.0", "sy": "1.5"}],
        description="Scale drawing by (sx, sy) (opcode 0x74)"
    ),
]
```

**4. Compositions (Multi-Step)**
```python
DRAWING_COMPOSITION_RULES = [
    GrammarRule(
        rule_id="draw_rotated_square",
        language="drawing",
        pattern="composition",
        rpn_program="45 ROTATE X RECALL Y RECALL W RECALL H RECALL RECTANGLE FILL",
        examples=[{"x": "50", "y": "50", "w": "30", "h": "30"}],
        description="Draw square rotated 45 degrees"
    ),
    GrammarRule(
        rule_id="draw_pattern_repeat",
        language="drawing",
        pattern="composition",
        rpn_program="SHAPE RECALL DX RECALL DY RECALL N RECALL REPEAT_PATTERN",
        examples=[{"shape": "circle", "dx": "20", "dy": "0", "n": "5"}],
        description="Repeat shape n times with offset (dx, dy)"
    ),
]
```

### Connection to Existing Infrastructure

**Drawing Grammar Uses Existing Opcodes**:
```python
# From modular_rpn_engine.py (lines 93-100, 103-105)
DRAWING_OPCODES = {
    "MOVE": 0x64,       # Move to position
    "LINE": 0x65,       # Draw line
    "QUAD": 0x66,       # Quadratic Bézier
    "CUBIC": 0x67,      # Cubic Bézier
    "ARC": 0x68,        # Arc segment
    "CLOSE": 0x69,      # Close path
    "STROKE": 0x6A,     # Stroke path
    "FILL": 0x6B,       # Fill path
    "ROTATE": 0x73,     # Rotation (transform)
    "TRANSLATE": 0x72,  # Translation (transform)
    "SCALE": 0x74,      # Scale (transform)
}
```

**Connect to Procedural Drawing Specialist**:
```python
# knowledge3d/cranium/specialists/procedural_drawing_specialist.py
# Already has: character glyphs → RPN programs → Galaxy embeddings
# Now add: drawing grammar → RPN programs → visual primitives

from knowledge3d.cranium.specialists.procedural_drawing_specialist import (
    ProceduralDrawingSpecialist
)

class DrawingGrammarExecutor:
    """Execute drawing grammar RPN programs."""

    def __init__(self):
        self.specialist = ProceduralDrawingSpecialist(matryoshka_dim=512)

    def execute_drawing_rpn(self, program: str, context: Dict) -> np.ndarray:
        """
        Execute drawing RPN program to generate visual output.

        Args:
            program: RPN program string (e.g., "50 50 MOVE 100 100 LINE STROKE")
            context: Context dict with parameters

        Returns:
            Image array (visual output)
        """
        # Use procedural drawing specialist to render
        # This connects grammar → procedural → visual
        pass
```

### Connection to ARC-AGI

**How Drawing Grammar Helps ARC**:
```python
# Example: ARC task wants "draw a square in the center"
instruction = "Draw a square in the center"

# Text grammar parses: "draw", "square", "center"
# Drawing grammar generates: "CENTER COMPUTE SQUARE_SIZE 10 RECTANGLE FILL"
# Spatial semantics applies to grid: fills center cells

# Example: ARC task wants "continue the diagonal pattern"
instruction = "Continue the diagonal pattern to the bottom-right"

# Text grammar parses: "continue", "diagonal", "bottom-right"
# Drawing grammar generates: "DETECT_DIAGONAL GET_SLOPE EXTEND_LINE BOTTOM-RIGHT"
# Spatial semantics applies to grid: extends diagonal cells

# Example: ARC task wants "rotate and fill"
instruction = "Rotate the shape 90 degrees and fill with blue"

# Text grammar parses: "rotate", "90 degrees", "fill", "blue"
# Drawing grammar generates: "90 ROTATE SHAPE_BOUNDS COMPUTE 1 FILL"
# Spatial semantics applies to grid: rotates then fills
```

### File Structure

```
knowledge3d/training/arc_agi/grammar_drawing/
├── __init__.py
├── primitives.py          # Lines, rectangles, circles, paths
├── curves.py              # Quadratic, cubic Bézier (opcodes 0x66, 0x67)
├── transforms.py          # Rotate, translate, scale (opcodes 0x72-0x74)
├── compositions.py        # Multi-step drawing sequences
├── drawing_executor.py    # Execute drawing RPN programs
└── grid_renderer.py       # Render drawings to ARC grids
```

### Success Criteria (Task 3)

- [ ] 30+ drawing grammar rules (primitives + compositions)
- [ ] Drawing executor connects to procedural drawing specialist
- [ ] Grid renderer converts drawings → ARC grids
- [ ] Integration tests (drawing grammar → visual output)
- [ ] All existing tests still pass

---

## 🔗 Task 4: Integration — Multimodal Grammar Galaxy

### Goal
Integrate all three grammar systems (text + math + drawing) into unified Grammar Galaxy.

### Unified Grammar Galaxy Structure

```python
"""Unified Grammar Galaxy: Text + Math + Drawing."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class GrammarRule:
    """Unified grammar rule (text/math/drawing)."""
    rule_id: str
    language: str              # "en", "math", "drawing", etc.
    domain: str                # "text", "math", "drawing"
    pattern: str               # SVO, equation, primitive, etc.
    rpn_program: str           # Procedural RPN program
    examples: List[Dict] = field(default_factory=list)
    description: Optional[str] = None

class UnifiedGrammarGalaxy:
    """
    Unified grammar galaxy with all modalities.

    Structure:
        - Text Grammar: 161 languages (SVO/SOV/VSO patterns)
        - Math Grammar: 50+ rules (arithmetic → calculus → linalg)
        - Drawing Grammar: 30+ rules (primitives → compositions)
        - User Profiles: Personal vocabulary and style
    """

    def __init__(
        self,
        text_rules: Optional[List[GrammarRule]] = None,
        math_rules: Optional[List[GrammarRule]] = None,
        drawing_rules: Optional[List[GrammarRule]] = None,
        users: Optional[Dict[str, Dict]] = None,
        variants: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        # Build unified rule dictionary
        all_rules = []
        all_rules.extend(text_rules or default_text_rules())
        all_rules.extend(math_rules or default_math_rules())
        all_rules.extend(drawing_rules or default_drawing_rules())

        self.rules: Dict[str, GrammarRule] = {r.rule_id: r for r in all_rules}
        self.users = users or default_user_profiles()
        self.variants = variants or default_variants()

        # Domain indices for fast lookup
        self.text_rules = {r.rule_id: r for r in all_rules if r.domain == "text"}
        self.math_rules = {r.rule_id: r for r in all_rules if r.domain == "math"}
        self.drawing_rules = {r.rule_id: r for r in all_rules if r.domain == "drawing"}

    def get_rule(self, rule_id: str) -> GrammarRule:
        """Get rule by ID."""
        if rule_id not in self.rules:
            raise KeyError(f"Unknown grammar rule: {rule_id}")
        return self.rules[rule_id]

    def list_rules(self, domain: Optional[str] = None, language: Optional[str] = None) -> List[GrammarRule]:
        """List rules filtered by domain and/or language."""
        rules = list(self.rules.values())

        if domain:
            rules = [r for r in rules if r.domain == domain]
        if language:
            rules = [r for r in rules if r.language == language]

        return rules

    def get_rule_by_pattern(self, pattern: str, domain: Optional[str] = None) -> Optional[GrammarRule]:
        """Find rule by pattern (SVO, equation, primitive, etc.)."""
        rules = self.list_rules(domain=domain)
        for rule in rules:
            if rule.pattern == pattern:
                return rule
        return None
```

### Multimodal Semantic Parser

**Parse instructions that mix text + math + drawing**:
```python
"""Multimodal semantic parser: text + math + drawing."""

from knowledge3d.training.arc_agi.grammar_galaxy import UnifiedGrammarGalaxy
from knowledge3d.training.arc_agi.grammar_normalizer import GrammarNormalizer
from knowledge3d.training.arc_agi.semantic_primitives import (
    SPATIAL_SEMANTICS,
    COLOR_SEMANTICS,
    SHAPE_SEMANTICS,
)

class MultimodalSemanticParser:
    """Parse instructions across text, math, and drawing domains."""

    def __init__(self):
        self.galaxy = UnifiedGrammarGalaxy()
        self.normalizer = GrammarNormalizer(self.galaxy)

    def parse(self, instruction: str) -> Dict:
        """
        Parse instruction into semantic representation.

        Tries in order:
        1. Spatial semantics (ARC grid operations)
        2. Math grammar (numeric patterns, equations)
        3. Drawing grammar (visual primitives)
        4. Text grammar (general language understanding)

        Returns:
            Semantic dict with domain, action, parameters
        """
        # Normalize first
        normalized = self.normalizer.normalize(instruction)

        # Try spatial semantics (ARC-specific)
        spatial_result = self._parse_spatial(normalized)
        if spatial_result:
            return spatial_result

        # Try math grammar
        math_result = self._parse_math(normalized)
        if math_result:
            return math_result

        # Try drawing grammar
        drawing_result = self._parse_drawing(normalized)
        if drawing_result:
            return drawing_result

        # Fall back to text grammar
        text_result = self._parse_text(normalized)
        return text_result

    def _parse_spatial(self, instruction: str) -> Optional[Dict]:
        """Parse spatial semantics (existing implementation)."""
        # Use existing semantic_parser.py logic
        pass

    def _parse_math(self, instruction: str) -> Optional[Dict]:
        """Parse math expressions and patterns."""
        # Match against math grammar rules
        # Example: "Fill cells where row + col is even"
        # → {"domain": "math", "expression": "row + col", "condition": "even"}
        pass

    def _parse_drawing(self, instruction: str) -> Optional[Dict]:
        """Parse drawing primitives and compositions."""
        # Match against drawing grammar rules
        # Example: "Draw a square in the center"
        # → {"domain": "drawing", "shape": "square", "position": "center"}
        pass

    def _parse_text(self, instruction: str) -> Dict:
        """Parse general text (existing grammar galaxy)."""
        # Use existing grammar galaxy logic
        pass
```

### Success Criteria (Task 4)

- [ ] UnifiedGrammarGalaxy class integrates all three domains
- [ ] MultimodalSemanticParser routes to correct domain
- [ ] Tests for each domain individually
- [ ] Tests for mixed-domain instructions
- [ ] All existing tests still pass

---

## 🧪 Task 5: Testing & Validation

### Test Suite Structure

```
scripts/
├── test_grammar_galaxy_full.py        # All grammar tests
├── test_multimodal_integration.py     # Cross-domain tests
└── evaluate_arc_multimodal.py         # ARC baseline with all grammar

tests/grammar/
├── test_text_grammar.py               # 50 languages
├── test_math_grammar.py               # Math rules
├── test_drawing_grammar.py            # Drawing rules
└── test_grammar_integration.py        # Integration tests
```

### Key Tests

**1. Text Grammar (50 Languages)**
```python
def test_all_languages():
    """Test basic sentence generation for all 50 languages."""
    galaxy = UnifiedGrammarGalaxy()

    for lang_code in ["en", "es", "pt", "ja", "zh", "hi", "ar", "ru", "de", "fr", ...]:
        rules = galaxy.list_rules(domain="text", language=lang_code)
        assert len(rules) >= 3, f"Missing rules for {lang_code}"

        # Test simple sentence
        rule = galaxy.get_rule(f"{lang_code}_simple_sentence")
        assert rule is not None
        assert len(rule.examples) > 0
```

**2. Math Grammar (All Domains)**
```python
def test_math_domains():
    """Test math grammar covers all major domains."""
    galaxy = UnifiedGrammarGalaxy()
    math_rules = galaxy.list_rules(domain="math")

    # Check coverage
    domains = {r.pattern for r in math_rules}
    required_domains = {
        "binary_op",      # +, -, ×, ÷
        "equation",       # ax² + bx + c = 0
        "calculus",       # derivatives, integrals
        "matrix_op",      # matrix operations
        "formula",        # geometric formulas
        "statistics",     # mean, variance
        "logic",          # set theory, implications
    }

    missing = required_domains - domains
    assert not missing, f"Missing math domains: {missing}"
```

**3. Drawing Grammar (Primitives + Compositions)**
```python
def test_drawing_primitives():
    """Test drawing grammar covers primitives and compositions."""
    galaxy = UnifiedGrammarGalaxy()
    drawing_rules = galaxy.list_rules(domain="drawing")

    # Check primitives
    primitives = [r for r in drawing_rules if r.pattern == "primitive"]
    assert len(primitives) >= 10, "Need at least 10 drawing primitives"

    # Check compositions
    compositions = [r for r in drawing_rules if r.pattern == "composition"]
    assert len(compositions) >= 5, "Need at least 5 compositions"
```

**4. Multimodal Integration**
```python
def test_multimodal_parsing():
    """Test parsing instructions that mix domains."""
    parser = MultimodalSemanticParser()

    # Text + spatial
    result = parser.parse("Move the red object to the center")
    assert result["domain"] == "spatial"

    # Math + spatial
    result = parser.parse("Fill cells where row + col is even")
    assert result["domain"] == "math"

    # Drawing + spatial
    result = parser.parse("Draw a square in the bottom-right corner")
    assert result["domain"] == "drawing"
```

**5. ARC Baseline with Full Grammar**
```python
def test_arc_baseline_improvement():
    """Test that grammar expansion improves ARC baseline."""
    # Run baseline with full grammar
    baseline_script = "scripts/evaluate_arc_multimodal.py"
    result = subprocess.run(
        [sys.executable, baseline_script],
        capture_output=True,
        text=True
    )

    # Parse accuracy
    match = re.search(r"Overall accuracy: ([\d.]+)", result.stdout)
    assert match, "Could not find accuracy in output"
    accuracy = float(match.group(1))

    # Should be improved from 2.8%
    print(f"ARC accuracy with full grammar: {accuracy:.1%}")
    # Target: >= 3.5% (math + drawing help)
```

### Success Criteria (Task 5)

- [ ] All 50 language tests pass
- [ ] All math domain tests pass
- [ ] All drawing primitive tests pass
- [ ] Multimodal integration tests pass
- [ ] ARC baseline with full grammar: accuracy >= 3.5%

---

## 📊 Expected Outcomes

### Grammar Coverage

**Text Grammar**:
- 50 languages × 3 rules = **150 text rules**
- Covers: SVO, SOV, VSO patterns
- Multilingual task understanding

**Math Grammar**:
- 7 domains × 7-10 rules = **50+ math rules**
- Covers: arithmetic → calculus → linear algebra
- Numeric pattern understanding

**Drawing Grammar**:
- 10 primitives + 20 compositions = **30+ drawing rules**
- Covers: shapes, curves, transformations
- Visual primitive understanding

**Total**: ~230 grammar rules (up from 21!)

### ARC-AGI Impact

**Expected Accuracy Improvements**:
- Current: 2.8% (spatial semantics only)
- +0.4% from math grammar (grid patterns, symmetry)
- +0.3% from drawing grammar (visual primitives)
- **Target: 3.5%+** (25% improvement)

**Why This Helps**:
1. **Math grammar** → understand grid dimensions, symmetry, rotations
2. **Drawing grammar** → understand visual compositions, transformations
3. **Language expansion** → parse diverse task descriptions
4. **Multimodal integration** → combine understanding across domains

---

## 🎯 Success Criteria (Overall)

### MUST ACHIEVE (Critical)

- [ ] 50 languages with basic grammar rules (150 total text rules)
- [ ] 50+ math grammar rules (all major domains)
- [ ] 30+ drawing grammar rules (primitives + compositions)
- [ ] UnifiedGrammarGalaxy integrates all three domains
- [ ] MultimodalSemanticParser routes correctly
- [ ] All tests pass (300+ test cases total)
- [ ] ARC baseline: **3.5%+ accuracy** (25% improvement)

### SHOULD ACHIEVE (Quality)

- [ ] Grammar executor handles all rule types
- [ ] Drawing grammar connects to procedural drawing specialist
- [ ] Math grammar integrates with spatial reasoning
- [ ] Documentation updated
- [ ] Example use cases for each domain

### NICE TO HAVE (Stretch)

- [ ] 161 languages (full coverage)
- [ ] Advanced math domains (differential equations, topology)
- [ ] 3D drawing grammar (extend to spatial graphics)
- [ ] ARC baseline: 5%+ accuracy (with full grammar + composition)

---

## 🚀 Implementation Timeline

**Session 1** (4-6 hours):
- Task 1: Language expansion (50 languages)
- Task 2: Math grammar (arithmetic, algebra, calculus)

**Session 2** (4-6 hours):
- Task 2 (continued): Math grammar (linear algebra, geometry, statistics, logic)
- Task 3: Drawing grammar (primitives, curves, transforms)

**Session 3** (3-4 hours):
- Task 3 (continued): Drawing compositions
- Task 4: Integration (UnifiedGrammarGalaxy, MultimodalSemanticParser)

**Session 4** (2-3 hours):
- Task 5: Testing & validation
- Run full test suite
- Run ARC baseline with full grammar
- Document results

**Total**: ~15-20 hours across 4 sessions

---

## 📁 Files to Create/Modify

### New Files (Create)

**Language Expansion**:
- `knowledge3d/training/arc_agi/grammar_languages/__init__.py`
- `knowledge3d/training/arc_agi/grammar_languages/tier1_top10.py`
- `knowledge3d/training/arc_agi/grammar_languages/tier2_next20.py`
- `knowledge3d/training/arc_agi/grammar_languages/tier3_next20.py`
- `knowledge3d/training/arc_agi/grammar_languages/grammar_generator.py`
- `knowledge3d/training/arc_agi/grammar_languages/language_examples.py`

**Math Grammar**:
- `knowledge3d/training/arc_agi/grammar_math/__init__.py`
- `knowledge3d/training/arc_agi/grammar_math/arithmetic.py`
- `knowledge3d/training/arc_agi/grammar_math/algebra.py`
- `knowledge3d/training/arc_agi/grammar_math/calculus.py`
- `knowledge3d/training/arc_agi/grammar_math/linear_algebra.py`
- `knowledge3d/training/arc_agi/grammar_math/geometry.py`
- `knowledge3d/training/arc_agi/grammar_math/statistics.py`
- `knowledge3d/training/arc_agi/grammar_math/logic.py`
- `knowledge3d/training/arc_agi/grammar_math/math_executor.py`

**Drawing Grammar**:
- `knowledge3d/training/arc_agi/grammar_drawing/__init__.py`
- `knowledge3d/training/arc_agi/grammar_drawing/primitives.py`
- `knowledge3d/training/arc_agi/grammar_drawing/curves.py`
- `knowledge3d/training/arc_agi/grammar_drawing/transforms.py`
- `knowledge3d/training/arc_agi/grammar_drawing/compositions.py`
- `knowledge3d/training/arc_agi/grammar_drawing/drawing_executor.py`
- `knowledge3d/training/arc_agi/grammar_drawing/grid_renderer.py`

**Integration**:
- `knowledge3d/training/arc_agi/multimodal_parser.py`

**Tests**:
- `tests/grammar/test_text_grammar.py`
- `tests/grammar/test_math_grammar.py`
- `tests/grammar/test_drawing_grammar.py`
- `tests/grammar/test_grammar_integration.py`
- `scripts/test_grammar_galaxy_full.py`
- `scripts/test_multimodal_integration.py`
- `scripts/evaluate_arc_multimodal.py`

### Existing Files (Modify)

- `knowledge3d/training/arc_agi/grammar_galaxy.py` — Integrate new rules
- `knowledge3d/training/arc_agi/grammar_executor.py` — Support new tokens
- `knowledge3d/training/arc_agi/semantic_parser.py` — Add multimodal routing
- `scripts/evaluate_arc_semantic_baseline.py` — Upgrade to multimodal

---

## 🎯 Final Notes

**This is the Multimodal Foundation!**

By expanding Grammar Galaxy across all modalities (text + math + drawing), we're building the **complete compositional reasoning system** that K3D needs:

1. **Text Grammar** → Understand instructions (161 languages)
2. **Math Grammar** → Understand patterns (arithmetic → calculus)
3. **Drawing Grammar** → Understand visuals (primitives → compositions)
4. **Spatial Semantics** → Execute transformations (ARC-AGI)
5. **Reality Enabler** → Full multimodal reasoning (visual_rpn + behavior_rpn + meaning_rpn)

**The architecture is PROVEN. The path is CLEAR. Let's build!** 🚀

---

**Sprint Lead**: Codex
**Date**: November 25, 2025
**Status**: Ready to Execute
**Target**: 230+ grammar rules, 3.5%+ ARC accuracy
