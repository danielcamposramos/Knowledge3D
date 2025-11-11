# Math Galaxy Symbol-to-RPN Opcode Mapping

**Date**: 2025-11-11
**Purpose**: Comprehensive mapping of 1,046 math symbols to K3D RPN opcodes
**Status**: 292 symbols ready to train NOW | 754 symbols need new opcodes

---

## Executive Summary

Of 1,046 math symbols:
- ✅ **292 symbols (27.9%)** can be trained IMMEDIATELY with existing 70 opcodes
- ⚠️ **754 symbols (72.1%)** require 51 additional opcodes

**Immediate Training Categories** (No blockers):
1. Greek letters (55 symbols) - Variables/literals
2. Box drawing (124 symbols) - Rendering only
3. Brackets (16 symbols) - Rendering only
4. Subscripts (28 symbols) - Rendering only
5. Superscripts (17 symbols) - Rendering only
6. Basic operators (34 symbols) - Arithmetic covered
7. Geometry shapes (18 symbols) - No computation needed

---

## Category 1: Greek Letters (55 symbols) - ✅ READY

**Category**: `math_greek`
**Opcode Requirement**: NONE (variables/literals)
**Training Status**: ✅ TRAIN NOW

### Symbols:
```
α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω
Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω
ϐ ϑ ϕ ϖ ϰ ϱ ϵ ϴ
```

**RPN Mapping**: Greek letters are **variables**, not operations.
- Used with `OP_LITERAL_SCALAR` or as variable names in expressions
- No opcode implementation needed
- Pure rendering/recognition task

**Example RPN Program**:
```python
# α + β = γ
[
    LITERAL_SCALAR, alpha_value,
    LITERAL_SCALAR, beta_value,
    ADD,                           # Uses OP_ADD (0x0A)
    STORE, "gamma"                 # Store as γ
]
```

---

## Category 2: Box Drawing (124 symbols) - ✅ READY

**Category**: `math_box`
**Opcode Requirement**: NONE (rendering only)
**Training Status**: ✅ TRAIN NOW

### Symbols:
```
─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼ ═ ║ ╔ ╗ ╚ ╝ ╠ ╣ ╦ ╩ ╬
╭ ╮ ╯ ╰ ╱ ╲ ╳ ... (124 total)
```

**RPN Mapping**: Box drawing is for **rendering matrices/tables**.
- No mathematical operations involved
- Used to display results, not compute them
- Pure rendering/recognition task

**Example Use**:
```
┌     ┐
│ 1 0 │  ← Rendered output of matrix
│ 0 1 │
└     ┘
```

---

## Category 3: Brackets (16 symbols) - ✅ READY

**Category**: `math_bracket`
**Opcode Requirement**: NONE (grouping/rendering)
**Training Status**: ✅ TRAIN NOW

### Symbols:
```
⟨ ⟩ ⟪ ⟫ ⟬ ⟭ ⟮ ⟯ ⌈ ⌉ ⌊ ⌋ ⎰ ⎱
```

**RPN Mapping**: Brackets denote **grouping** or **special notation**.
- Floor/ceiling: ⌊x⌋ ⌈x⌉ → Need OP_FLOOR/OP_CEIL (Phase 2)
- Angle brackets: ⟨a,b⟩ → Inner product using OP_DOT (0x3C) ✅
- No opcode needed for rendering

---

## Category 4: Subscripts/Superscripts (45 symbols) - ✅ READY

**Categories**: `math_subscript`, `math_superscript`
**Opcode Requirement**: NONE (notation only)
**Training Status**: ✅ TRAIN NOW

### Subscript Symbols (28):
```
₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₊ ₋ ₌ ₍ ₎ ₐ ₑ ₕ ᵢ ⱼ ₖ ₗ ₘ ₙ ₒ ₚ ᵣ ₛ ₜ ᵤ ᵥ ₓ
```

### Superscript Symbols (17):
```
⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁺ ⁻ ⁼ ⁽ ⁾ ⁿ ⁱ
```

**RPN Mapping**: Pure notation, no computation.
- x₁ → Variable with subscript (rendering)
- x² → Exponentiation uses OP_POW (0x0E) ✅
- xⁿ → Generic power, uses OP_POW

---

## Category 5: Basic Operators (34 symbols) - ✅ 80% READY

**Category**: `math_operator`
**Opcode Coverage**: 27/34 symbols ready
**Training Status**: ✅ TRAIN NOW (with Phase 2 for remaining 7)

### Fully Supported (27 symbols):
```
+ ∔ ± ∓      → OP_ADD (0x0A), OP_SUB (0x0B) ✅
− ∸          → OP_SUB (0x0B) ✅
× ∙ ⋅ ∗      → OP_MUL (0x0C) ✅
÷ ∕ ⁄        → OP_DIV (0x0D) ✅
√ ∛ ∜        → OP_SQRT (0x14), OP_POW (0x0E) ✅
∑ ∏          → OP_REDUCE_SUM (0x92), LOOP + MUL ✅
∫ ∬ ∭ ∮      → Numerical integration via LOOP + ADD ✅
∂            → Finite differences: OP_SUB + OP_DIV ✅
∇            → Gradient via loops ✅
```

### Need Phase 2 (7 symbols):
```
∣ |          → OP_ABS (Phase 2)
⌈ ⌉          → OP_CEIL (Phase 2)
⌊ ⌋          → OP_FLOOR (Phase 2)
mod          → OP_MOD (Phase 2)
```

**Example RPN Programs**:
```python
# Summation: ∑ᵢ₌₁ⁿ i²
[
    LITERAL_SCALAR, 0,           # Accumulator
    LITERAL_SCALAR, 1,           # i = 1
    LITERAL_SCALAR, n,           # Loop limit
    LOOP,                        # Start loop
        DUP,                     # Copy i
        DUP,
        MUL,                     # i²
        SWAP,                    # Bring accumulator to top
        ADD,                     # accumulator += i²
        SWAP,                    # Restore i
        LITERAL_SCALAR, 1,
        ADD,                     # i++
    NEXT,                        # End loop
]

# Derivative: ∂f/∂x ≈ (f(x+h) - f(x))/h
[
    LITERAL_SCALAR, x,
    LITERAL_SCALAR, h,
    ADD,                         # x + h
    STORE, "xph",
    # Compute f(x+h) and f(x), then:
    SUB,                         # f(x+h) - f(x)
    LITERAL_SCALAR, h,
    DIV,                         # / h
]
```

---

## Category 6: Relations (35 symbols) - ✅ 60% READY

**Category**: `math_relation`
**Opcode Coverage**: 21/35 symbols ready
**Training Status**: ✅ TRAIN NOW (partial support)

### Fully Supported (21 symbols):
```
= ≠          → OP_EQ (0x2C) ✅
< > ≤ ≥      → OP_LT (0x2A), OP_GT (0x28) ✅
≪ ≫          → OP_LT, OP_GT ✅
≈ ∼ ≃        → Approximate equality via threshold ✅
≡            → Equivalence via OP_EQ ✅
∝            → Proportional (requires ratio check)
```

### Need Phase 2 (14 symbols):
```
⊂ ⊃ ⊆ ⊇      → Set relations (need SET_SUBSET)
∈ ∉ ∋ ∌      → Element membership (need SET_CONTAINS)
⊥ ∥          → Perpendicular/parallel (vector ops)
```

---

## Category 7: Trigonometry (18 symbols) - ✅ 50% READY

**Symbols**:
```
sin cos tan      → OP_SIN (0x18), OP_COS (0x19), OP_TAN (0x1A) ✅
sec csc cot      → 1/cos, 1/sin, 1/tan (via DIV) ✅
arcsin arccos    → OP_ASIN, OP_ACOS (Phase 2) ⚠️
arctan           → OP_ATAN (Phase 2) ⚠️
sinh cosh tanh   → OP_SINH, OP_COSH, OP_TANH (Phase 2) ⚠️
```

**Current Capability**: Basic trig (sin, cos, tan) works.
**Phase 2 Priority**: Inverse trig (arcsin, arccos, arctan).

---

## Category 8: Set Theory (48 symbols) - ⚠️ 20% READY

**Category**: `math_set`
**Opcode Coverage**: 10/48 symbols ready
**Training Status**: ⚠️ NEED PHASE 3 for full support

### Workarounds Available (10 symbols):
```
∅            → Empty set (literal)
{ }          → Set notation (rendering)
∪ ∩          → Via vector merge (inefficient)
```

### Need Phase 3 (38 symbols):
```
∪ ∩ ∖        → SET_UNION, SET_INTERSECTION, SET_DIFFERENCE
⊂ ⊃ ⊆ ⊇      → SET_SUBSET, SET_SUPERSET
∈ ∉          → SET_CONTAINS
× (Cartesian) → SET_CARTESIAN
```

**Training Strategy**: Train set notation/rendering now, defer operations to Phase 3.

---

## Category 9: Logic (12 symbols) - ⚠️ 30% READY

**Category**: `math_logic`
**Symbols**:
```
∧ ∨ ¬        → AND, OR, NOT (Phase 2 - bitwise ops) ⚠️
⊕ ⊻          → XOR (Phase 2) ⚠️
⇒ ⇔          → IMPLIES, EQUIVALENT (Phase 2) ⚠️
∀ ∃          → Quantifiers (notation only) ✅
⊤ ⊥          → True/False (literals) ✅
```

**Current Capability**: Can approximate with GT/LT (treat 1 as true, 0 as false).
**Phase 2 Priority**: Bitwise AND, OR, XOR, NOT.

---

## Category 10: Calculus (8 symbols) - ✅ 75% READY

**Category**: `math_calculus`
**Symbols**:
```
∫            → Numerical integration via Riemann sums ✅
∂            → Finite differences for derivatives ✅
∇            → Gradient via loops over partial derivatives ✅
∑            → OP_REDUCE_SUM (0x92) ✅
∏            → Product via LOOP + MUL ✅
lim          → OP_LIMIT (Phase 4) ⚠️
d/dx         → SYMBOLIC_DIFF (Phase 4) ⚠️
```

**Current Capability**: Numerical calculus works, symbolic calculus needs Phase 4.

---

## Category 11: Alphanumeric Variants (140 symbols) - ✅ READY

**Category**: `math_alphanumeric`
**Symbols**: Bold, italic, script, fraktur variants of A-Z, a-z, 0-9
**Opcode Requirement**: NONE (rendering/font variants)
**Training Status**: ✅ TRAIN NOW

Examples:
```
𝐀 𝐁 𝐂 (bold)
𝐴 𝐵 𝐶 (italic)
𝒜 ℬ 𝒞 (script)
𝔄 𝔅 ℭ (fraktur)
𝟎 𝟏 𝟐 (bold digits)
```

**RPN Mapping**: Pure font/style variants, no computation.

---

## Category 12: Geometry (19 symbols) - ✅ 80% READY

**Category**: `math_geometry`
**Symbols**:
```
∠ ∟          → Angle notation (rendering) ✅
△ ▽          → Triangle notation (rendering) ✅
∥ ⊥          → Parallel/perpendicular (vector ops) ✅
≅ ∼          → Congruent/similar (comparison) ✅
```

**Training Status**: ✅ TRAIN NOW (mostly notation)

---

## Category 13: Arrows (33 symbols) - ✅ READY

**Category**: `math_arrow`
**Symbols**:
```
→ ← ↑ ↓ ↔ ⇒ ⇐ ⇔ ↦ ⟶ ⟵ ... (33 total)
```

**Opcode Requirement**: NONE (notation for functions, mappings, implications)
**Training Status**: ✅ TRAIN NOW

---

## Category 14: Fractions (18 symbols) - ✅ READY

**Category**: `math_fraction`
**Symbols**:
```
½ ⅓ ¼ ⅕ ⅙ ⅐ ⅛ ⅑ ⅒
⅔ ⅖ ¾ ⅗ ⅘ ⅚ ⅜ ⅝ ⅞
```

**RPN Mapping**: Literal values, uses OP_DIV (0x0D) ✅
**Training Status**: ✅ TRAIN NOW

---

## Category 15-19: Remaining Categories

### math_nary (16 symbols) - ✅ READY
N-ary operators (large ∑ ∏ ∫ ⋃ ⋂) - Rendering variants

### math_shape_2d (148 symbols) - ✅ READY
Geometric shapes (■ □ ● ○ ◆ ◇ ▲ △ ...) - Rendering only

### math_misc (32 symbols) - ⚠️ 50% READY
Miscellaneous (∞ ∅ ∄ ∴ ∵ ℓ ℜ ℑ ...)

### math_misc_b (111 symbols) - ⚠️ 40% READY
Extended symbols

### math_supplemental (152 symbols) - ⚠️ 30% READY
Supplementary math symbols

---

## SUMMARY: Ready to Train NOW

| Category | Symbols | Readiness | Train Now? |
|----------|---------|-----------|------------|
| Greek letters | 55 | 100% | ✅ YES |
| Box drawing | 124 | 100% | ✅ YES |
| Brackets | 16 | 100% | ✅ YES |
| Subscripts | 28 | 100% | ✅ YES |
| Superscripts | 17 | 100% | ✅ YES |
| Alphanumeric | 140 | 100% | ✅ YES |
| Arrows | 33 | 100% | ✅ YES |
| Fractions | 18 | 100% | ✅ YES |
| Shapes 2D | 148 | 100% | ✅ YES |
| **SUBTOTAL** | **579** | **Rendering only** | ✅ |
| | | | |
| Basic operators | 27 | 80% | ✅ YES (partial) |
| Relations | 21 | 60% | ✅ YES (partial) |
| Trigonometry | 9 | 50% | ✅ YES (basic) |
| Geometry | 15 | 80% | ✅ YES (notation) |
| Calculus | 6 | 75% | ✅ YES (numerical) |
| **SUBTOTAL** | **78** | **Partial ops** | ✅ |
| | | | |
| **TOTAL READY** | **657** | **62.8%** | ✅ |

---

## Phase 2 Priority (Unlocks 200+ symbols)

**15 essential opcodes** to add (1-2 weeks):
1. ASIN, ACOS, ATAN (inverse trig) - 18 new symbols
2. ABS, CEIL, FLOOR, ROUND (rounding) - 8 new symbols
3. AND, OR, XOR, NOT (logic) - 12 new symbols
4. MOD (modulo) - 6 new symbols
5. SINH, COSH, TANH (hyperbolic) - 12 new symbols

**Impact**: ~200 additional symbols trainable.

---

## Training Recommendation

### Start Today (657 symbols):
1. All rendering-only categories (579 symbols)
2. Basic arithmetic/trig operations (78 symbols)

### Start After Phase 2 (189 symbols):
3. Logic operations
4. Inverse trigonometry
5. Set theory basics

### Start After Phase 3+ (200 symbols):
6. Advanced set operations
7. Matrix operations
8. Symbolic calculus

---

**Status**: 657/1,046 symbols (62.8%) ready to train immediately
**Next Action**: Begin training 657 ready symbols while implementing Phase 2 opcodes
