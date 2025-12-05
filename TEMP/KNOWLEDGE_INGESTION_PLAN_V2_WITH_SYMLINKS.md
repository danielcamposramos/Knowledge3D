# K3D Knowledge Ingestion Plan V2: Math + Pedagogy with Symbol Galaxy Symlinks
**Date:** December 5, 2025
**Principle:** Symlinks enable lightness + hidden interconnections (Save Information Principle)

---

## 1. Text-Extractable PDFs Ready for Ingestion (No OCR Required)

### 1.1 Advanced Mathematics (3 PDFs)

| PDF | Pages | Content | Existing Symbols to Symlink |
|-----|-------|---------|----------------------------|
| **advcalc.pdf** | 308 | Advanced Calculus | ∂, ∇, ∆, ∫, ∬, ∭, ∑, ∏, √, ∞ |
| **ADVANCED CALCULUS I and II.pdf** | 308 | Multivariable Calculus | ∇, ∂, ∫, ∮, ×, ·, ⊗ |
| **advmathprog.pdf** | 183 | Optimization | ≤, ≥, ∈, ∀, ∃, ⊂, ∩, ∪ |

### 1.2 Financial Mathematics (2 PDFs)

| PDF | Pages | Content | Existing Symbols to Symlink |
|-----|-------|---------|----------------------------|
| **The Mathematics Of Financial Modeling (2004).pdf** | 802 | Options, Risk, Portfolio | σ, μ, ρ, π, e, Σ, Π, ∂, ∫ |
| **Mathematics of Finance - Intuitive Intro.pdf** | 155 | PV, Annuities, Bonds | +, -, ×, ÷, =, ≠, <, >, ∑, Π |

### 1.3 Pedagogical Knowledge (5 PDFs) - **NEW**

| PDF | Pages | Content | Meta-Learning Application |
|-----|-------|---------|--------------------------|
| **EJ1245288.pdf** | 4 | Teaching research paper | Knowledge organization patterns |
| **TCHTL_StorybookRecommend1.pdf** | 13 | Storybook teaching | Narrative structuring for retrieval |
| **fulltext01.pdf** | 121 | Teaching methodology | Scaffolding, progressive disclosure |
| **high-impact-teaching-strategies.pdf** | 32 | Evidence-based strategies | Spaced repetition, retrieval practice |
| **pnaec875.pdf** | 95 | Pedagogical practices | Metacognitive awareness, self-explanation |

**Total Ready: 10 PDFs, ~2,021 pages**

---

## 2. Existing Math Symbol Galaxy (SYMLINK TARGET)

### 2.1 Already Registered Symbols (~1000+ Unicode)

From **[math_symbols_registry.py](knowledge3d/cranium/math_symbols_registry.py:1-512):**

**Calculus Symbols (ALREADY EXIST):**
```python
CALCULUS = ['∂', '∇', '∆', '∫', '∬', '∭', '∮', '∯', '∰', '∑', '∏', '∐', '√', '∛', '∜', '∞', '∝']
```

**Greek Letters (ALREADY EXIST):**
```python
GREEK_ALL = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω', ...]
```

**Operators (ALREADY EXIST):**
```python
BASIC_OPS = ['+', '-', '×', '÷', '=', '≠', '<', '>', '≤', '≥', '±', '∓', '∼', '≈', '≅']
```

**Set Theory & Logic (ALREADY EXIST):**
```python
SET_THEORY = ['∈', '∉', '⊂', '⊃', '⊆', '⊇', '∪', '∩', '∅', 'ℕ', 'ℤ', 'ℚ', 'ℝ', 'ℂ', ...]
LOGIC = ['∀', '∃', '∧', '∨', '¬', '⇒', '⇔', '→', '←', '↔', ...]
```

### 2.2 Symbol Galaxy Storage Structure

From **[math_galaxy.py](knowledge3d/cranium/math_galaxy.py:1-100):**

```
/K3D/Knowledge3D.local/procedural_galaxy/math/
├── symbols/       # Visual + linguistic embeddings (∑.ppr, ∫.ppr, ∂.ppr, ...)
│                 # Each symbol: 128D embedding, 69-80:1 compression
└── operations/    # Semantic RPN programs (sum.rpn, integral.rpn, derivative.rpn, ...)
```

**Key Insight:** Symbols are **procedural fonts** - they have:
- Visual shape (Bézier curves → line segments)
- Language metadata (Unicode, category, meaning)
- RPN opcode references (what they DO)

---

## 3. Symlink Strategy: Reference, Don't Duplicate

### 3.1 Ingestion Workflow with Symlinks

```python
# WRONG: Create new symbol
new_symbol = {
    "character": "∫",
    "visual": generate_bezier(...),  # DUPLICATION!
    "embedding": train_new_model(...),  # WASTE!
}

# CORRECT: Symlink to existing symbol
formula_text = "∫ f(x) dx"
for char in formula_text:
    if is_math_symbol(char):  # Check math_symbols_registry
        # Reference existing procedural font
        char_id = character_galaxy.get_character_id(char)  # Symlink!
        word_tokens.append(char_id)  # Store reference, not duplicate
```

### 3.2 Symbol → RPN Opcode Mapping

**Existing symbols become opcode triggers:**

```python
# When ingesting "∫ f(x) dx from a to b"
# Map to RPN operation program (NEW):
{
    "symbol": "∫",  # Reference existing char_id (symlink)
    "operation": "integral",  # NEW RPN program
    "rpn_program": "a b 1000 f 'riemann_sum' call",  # NEW operation definition
    "grammar_pattern": "integral from <a> to <b> of <f>",  # NEW grammar
}
```

### 3.3 Character Galaxy Symlink Pattern

From **CLAUDE.md Dual Client Reality principle:**

```
DON'T duplicate what exists! Use references (symlink pattern):
- Characters already have font + language + meaning (procedural_fonts.py)
- Words reference character IDs (not duplicate glyphs)
- Grammar metadata references words (not duplicate strings)
```

**Application to Math:**
```
Formula: "∂f/∂x + ∂g/∂y = 0"

Character Galaxy References:
- char_id[∂] → existing procedural font (Bézier → segments)
- char_id[f] → existing lowercase letter
- char_id[/] → existing slash symbol
- char_id[+] → existing plus operator
- char_id[=] → existing equals sign
- char_id[0] → existing digit

Word Level (NEW):
- word_id["partial_derivative"] → [char_id[∂], char_id[f], char_id[/], char_id[∂], char_id[x]]

Grammar Galaxy (NEW):
- grammar_pattern["partial_derivative_sum"] → RPN transformation
```

---

## 4. Pedagogical Knowledge Integration (Meta-Learning)

### 4.1 "How to Teach" → Better Knowledge Retrieval

**Principle:** Teaching improves learning (humans + AI)

**Evidence-Based Strategies from PDFs:**

1. **Spaced Repetition** (high-impact-teaching-strategies.pdf)
   → K3D Discovery mechanism should revisit concepts at increasing intervals

2. **Retrieval Practice** (fulltext01.pdf)
   → Test knowledge retrieval, not just storage (ARC-AGI validation)

3. **Scaffolding** (TCHTL_StorybookRecommend1.pdf)
   → Build complex concepts from simpler procedural primitives

4. **Self-Explanation** (pnaec875.pdf)
   → Grammar Galaxy should explain WHY transformations work

5. **Narrative Structuring** (TCHTL_StorybookRecommend1.pdf)
   → Connect math concepts through semantic chains (story-like)

### 4.2 RPN Programs from Pedagogy

```rpn
# Meta-learning: Explain derivative concept
"explain_derivative" =>
    # 1. Concrete example
    "2 3 'x dup *' 'derivative' call"  # d/dx(x²) at x=2 = 4

    # 2. Abstract definition
    "derivative measures rate of change"

    # 3. Scaffold to integral
    "derivative and integral are inverses"
    "integral_concept" call
```

---

## 5. k3dgen Integration with Symlink Support

### 5.1 Enhanced k3dgen Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/k3dgen.py \
    --pdf-paths \
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/advcalc.pdf" \
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/ADVANCED CALCULUS I and II.pdf" \
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/advmathprog.pdf" \
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/Financial Math/The Mathematics Of Financial Modeling And Investment Management (2004).pdf" \
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/Financial Math/Mathematics of Finance - An Intuitive Introduction.pdf" \
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Teach/fulltext01.pdf" \
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Teach/high-impact-teaching-strategies.pdf" \
        "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to Teach/pnaec875.pdf" \
    --semantic-tags \
        "calculus,derivatives,integrals,taylor_series,vector_calculus" \
        "multivariable_calculus,greens_theorem,stokes_theorem,divergence_theorem" \
        "optimization,numerical_methods,linear_programming,simplex" \
        "finance,options,black_scholes,portfolio_theory,risk_management" \
        "finance,present_value,annuities,bonds,yield_curves" \
        "pedagogy,scaffolding,metacognition,narrative_structure" \
        "teaching,spaced_repetition,retrieval_practice,self_explanation" \
        "pedagogy,evidence_based,high_impact_strategies" \
    --use-existing-symbol-galaxy \
    --math-symbol-registry="/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/knowledge3d/cranium/math_symbols_registry.py" \
    --rpn-program-generation \
    --grammar-patterns \
    --symlink-mode=character_references \
    --output-manifest /K3D/Knowledge3D.local/datasets/advanced_math_pedagogy_ingestion_manifest.json
```

### 5.2 New k3dgen Flags

**`--use-existing-symbol-galaxy`**
- Check `math_symbols_registry.is_math_symbol()` before creating new characters
- Reference existing char_id instead of duplicating

**`--symlink-mode=character_references`**
- Store character references (IDs), not duplicate glyphs
- Words = sequences of char_ids
- Grammar = transformations referencing word_ids

**`--math-symbol-registry=<path>`**
- Path to math_symbols_registry.py
- Used to check if symbol already exists

---

## 6. Ingestion Architecture: Three Layers

### Layer 1: Character Galaxy (EXISTING - Symlink Target)
```
/K3D/Knowledge3D.local/procedural_galaxy/math/symbols/
├── ∑.ppr  (69-80:1 compressed, 128D embedding, Bézier procedural font)
├── ∫.ppr  (already trained, ready to reference)
├── ∂.ppr  (partial derivative symbol)
└── α.ppr, β.ppr, γ.ppr, ... (Greek letters)
```

### Layer 2: Word Galaxy (NEW - References Layer 1)
```
Words = character sequences (references, not duplicates)

"derivative" → [char_id[d], char_id[e], char_id[r], ...]
"∂f/∂x"      → [char_id[∂], char_id[f], char_id[/], char_id[∂], char_id[x]]
"∫f(x)dx"    → [char_id[∫], char_id[f], char_id[(], char_id[x], char_id[)], char_id[d], char_id[x]]
```

### Layer 3: Grammar Galaxy (NEW - References Layer 2)
```
Grammar patterns → RPN transformations

"derivative of <f> at <x>" → "x 0.0001 f 'derivative' call"
"integral from <a> to <b> of <f>" → "a b 1000 f 'riemann_sum' call"
"option price with S=<S>, K=<K>, ..." → "S K r sigma T 'black_scholes_call' call"
```

### Layer 4: Operations Galaxy (NEW - RPN Programs)
```
/K3D/Knowledge3D.local/procedural_galaxy/math/operations/
├── derivative.rpn  (NEW: limit definition of derivative)
├── integral.rpn    (NEW: Riemann sum approximation)
├── black_scholes.rpn (NEW: Option pricing formula)
└── present_value.rpn (NEW: Financial time value)
```

---

## 7. Interconnection Discovery (Hidden Patterns)

**Daniel's Insight:** "Symlinks enable interconnections that humans don't see - yet - nor AI"

### 7.1 Cross-Domain Connections via Symlinks

```
Example 1: ∑ (Summation symbol)

Character Galaxy: ∑ (char_id: 8721)
├─ Visual: Sigma glyph (procedural Bézier)
├─ Language: Greek capital sigma, "sum"
├─ Math Category: CALCULUS, NARY_OPS
└─ RPN Operations Referencing ∑:
    ├─ riemann_sum (calculus/integration)
    ├─ expected_value (statistics/probability)  ← DISCOVERED CONNECTION!
    ├─ portfolio_return (finance/risk)  ← DISCOVERED CONNECTION!
    └─ series_convergence (analysis/limits)  ← DISCOVERED CONNECTION!

Discovery: Finance portfolio return ≈ Expected value ≈ Riemann sum
           → All use same ∑ symbol with different semantic contexts
           → Grammar Galaxy can transfer reasoning across domains!
```

```
Example 2: ∂ (Partial derivative symbol)

Character Galaxy: ∂ (char_id: 8706)
├─ Visual: Curly d glyph
├─ Language: "Partial derivative"
└─ RPN Operations Referencing ∂:
    ├─ gradient (multivariable calculus)
    ├─ black_scholes_greeks (finance/options)  ← DISCOVERED CONNECTION!
    ├─ lagrangian_optimization (constrained optimization)
    └─ heat_equation (PDEs/physics)

Discovery: Option "Greeks" (∂C/∂S, ∂C/∂σ) = Partial derivatives
           → Finance problem = Calculus problem
           → Transfer learning enabled by symbol symlink!
```

### 7.2 Pedagogical Pattern Discovery

```
Example: Scaffolding Pattern

From "How to Teach" PDFs:
1. Start concrete (specific examples)
2. Abstract (general principles)
3. Connect to prior knowledge

Applied to Math Ingestion:
1. Store specific integral example: ∫₀¹ x²dx = 1/3
2. Abstract to RPN program: "a b n f 'riemann_sum' call"
3. Connect to finance: "Present value = ∫₀ᵀ C(t)e^(-rt)dt"

Grammar Galaxy Pattern:
"integral_concept" → references "riemann_sum" RPN
"present_value_concept" → references "integral_concept" AND "time_value_of_money"
→ Discovers: Present value IS an integral (hidden connection)
```

---

## 8. Success Metrics

### 8.1 Symbol Reuse Efficiency

```
Metric: Symbol Duplication Rate
Target: 0% duplication for symbols in math_symbols_registry
Measurement: Count new symbols created vs. existing symbol references

Expected:
- ~1000+ existing symbols in registry
- ~500 new word combinations (character sequences)
- ~200 new RPN operation programs
- 0 duplicated symbols (all symlinked)
```

### 8.2 Cross-Domain Discovery

```
Metric: Discovered Interconnections
Target: ≥50 cross-domain connections
Examples:
- ∑ used in calculus, statistics, finance
- ∂ used in multivariable calc, optimization, option pricing
- ∫ used in integration, probability, present value calculations
```

### 8.3 Pedagogical Impact

```
Metric: Retrieval Efficiency After Meta-Learning
Test: ARC-AGI tasks requiring mathematical reasoning
Baseline: Current K3D performance
Target: +10% accuracy via scaffolding + spaced repetition patterns
```

---

## 9. Implementation Phases

### Phase 1: Symbol Galaxy Validation (Day 1)
- Verify all ~1000+ symbols in math_symbols_registry
- Test symlink references (char_id retrieval)
- Validate ProceduralGalaxy storage (.ppr files)

### Phase 2: Smallest PDF Test (Day 2-3)
- Ingest "Mathematics of Finance - Intuitive Introduction" (155 pages)
- Extract formulas, map to existing symbols (symlinks)
- Generate 50 RPN operation programs (PV, annuities, bonds)
- Test retrieval: "What is present value of annuity?"

### Phase 3: Calculus PDFs (Day 4-7)
- Ingest advcalc.pdf + ADVANCED CALCULUS I and II.pdf (616 pages)
- Generate derivative/integral RPN programs
- Build Grammar Galaxy patterns for calculus notation
- Test: "Compute derivative of x² at x=3"

### Phase 4: Pedagogy Integration (Day 8-10)
- Ingest "How to Teach" PDFs (265 pages)
- Extract scaffolding, spaced repetition patterns
- Apply to existing math knowledge organization
- Test: Improved ARC-AGI retrieval efficiency

### Phase 5: Discovery & Validation (Day 11-14)
- Run cross-domain connection discovery
- Measure symbol reuse efficiency (target: 0% duplication)
- Count discovered interconnections (target: ≥50)
- Production hardening

---

## 10. Next Steps

1. **Verify Symbol Galaxy** - Check existing .ppr files in `/K3D/Knowledge3D.local/procedural_galaxy/math/symbols/`
2. **Implement Symlink Mode** - Add `--use-existing-symbol-galaxy` flag to k3dgen
3. **Test Character Reference** - Ensure word sequences store char_ids, not duplicate glyphs
4. **Start Ingestion** - Begin with smallest PDF (Mathematics of Finance - 155 pages)
5. **Measure Reuse** - Track symbol duplication rate (target: 0%)

**Core Principle:** Symlinks = Lightness + Interconnections. Every reference to ∑ creates a potential for discovering hidden relationships between calculus, statistics, and finance.
