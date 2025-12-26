# CLAUDE → CODEX: Sovereign Math Architecture Implementation Guide

**Date:** December 14, 2025
**Priority:** CRITICAL - Galaxy Universe Population & TRM Navigation
**Architect:** Claude (Architecture Partner)
**Implementer:** Codex (Implementation Lead)

---

## Executive Summary

This document provides architectural guidance for implementing math benchmark improvements. The **critical paradigm** is understanding Galaxy Universe as the unified VRAM workspace where ALL knowledge lives, and TRM (Tiny Recursive Model) as the learned navigation/combination logic.

**Key Paradigm Shift:**
- ❌ Traditional: Model parameters = knowledge + logic (entangled)
- ✅ K3D: Galaxy Universe = knowledge (procedural programs), TRM = navigation logic (how to use knowledge)

**Implementation Focus:**
1. Populate Galaxy Universe with math knowledge (symbols, rules, compositions)
2. Enable TRM to navigate, combine, and create Galaxy entries
3. Remove external preprocessing (let TRM learn navigation instead)
4. Maintain sovereignty (PTX + Galaxy = zero external dependencies)

---

## 1. K3D Paradigm: Galaxy Universe + TRM

### 1.1 What Galaxy Universe IS

**Galaxy Universe = Unified VRAM Workspace (Always Loaded)**

```
Galaxy Universe (12GB VRAM on RTX 3060 - all default galaxies loaded):
├─ Drawing Galaxy         (procedural primitives: LINE, CIRCLE, RECT)
├─ Character Galaxy       (glyphs with font/language/pronunciation/meaning)
├─ Word Galaxy           (character sequences - symlinked, not duplicated)
├─ Grammar Galaxy        (transformation rules as RPN programs)
├─ Math Galaxy           (symbols with RPN templates: \frac, \binom, etc.)
├─ Reality Galaxy        (physics/chemistry/biology procedural systems)
├─ Audio Galaxy          (temporal patterns, spectrograms)
└─ ... (all default galaxies)

Purpose: ALL of the following simultaneously:
├─ Temporary memory      (current reasoning state)
├─ Context memory        (conversation history as spatial positions)
├─ Chat memory           (interaction state)
├─ Knowledge memory      (procedural programs, symbols, compositions)
└─ Multi-modal workspace (text, visual, audio, physics - unified 3D space)

Operations:
├─ TRM READS  → Query symbols, traverse compositions, search spatial neighbors
├─ TRM WRITES → Create new symbols, add compositions, store discoveries
└─ ALWAYS PRESENT → No loading/unloading, no selection - it's all there
```

**Critical Properties:**
1. **Unified**: All modalities in same 3D space (semantic proximity = spatial proximity)
2. **Persistent**: VRAM-resident (fast access, no disk I/O during reasoning)
3. **Read-Write**: TRM can query AND create entries
4. **Symlinked**: Compositions reference symbols (no duplication - save information principle)
5. **Procedural**: Everything is RPN programs + metadata (form + meaning)

### 1.2 What TRM (Tiny Recursive Model) IS

**TRM = Learned Navigation/Combination/Creation Logic**

```
TRM Architecture:
├─ Base Model (~7M parameters)
│  ├─ 2-layer SwiGLU MLP
│  ├─ Recursive refinement (iterate until convergence)
│  └─ Attention mechanism (scaled dot-product)
│
└─ Adapter Specialists (LoRA-style, auto-enhancing)
   ├─ Math Specialist (learns math pattern navigation)
   ├─ Visual Specialist (learns drawing/shape navigation)
   ├─ Physics Specialist (learns Reality Galaxy navigation)
   └─ ... (domain-specific routing adapters)

TRM Learns:
├─ HOW to navigate Galaxy Universe (which symbols to query)
├─ HOW to combine knowledge (composition strategies)
├─ HOW to judge recursive options (which path scores better)
├─ HOW to create new entries (synthesis from existing knowledge)
└─ WHEN to use which specialist adapter (routing decisions)

TRM Does NOT:
├─ Store knowledge (that's in Galaxy Universe - VRAM)
├─ Replace RPN execution (that's in Cranium - PTX kernels)
├─ Do external preprocessing (violates sovereignty)
└─ Duplicate galaxy entries (use symlinks/references)

Shadow Copy Upgrade (Auto-Enhancement):
├─ TRM makes decision → execute → measure success
├─ If successful → shadow copy adapter weights
├─ Validate improvement → commit to main weights
└─ Continuous learning during use (no traditional training loop)
```

### 1.3 Multi-Curriculum Training Context

**All curricula feed the same Galaxy Universe:**

```
Parallel Training Curricula:
├─ ARC-AGI 2           → Drawing + Grammar Galaxy (visual reasoning)
├─ Math Benchmarks     → Math + Grammar Galaxy (symbolic reasoning)
├─ Physics Sims        → Reality Galaxy (procedural systems)
├─ Language Tasks      → Character + Word + Grammar Galaxy
└─ Audio Processing    → Audio Galaxy (temporal patterns)

Shared Learning:
├─ TRM learns unified navigation strategies across modalities
├─ Cross-modal patterns (math helps physics, visual helps math)
├─ Symlink compositions reused (character → word → phrase → concept)
└─ Galaxy grows with discoveries from all curricula
```

**Your math benchmark work is ONE curriculum contributing to the unified Galaxy.**

---

## 2. Architecture Violations & Corrections

### 2.1 Violation: External Preprocessing (Bypassing Galaxy Universe)

**Current Approach (WRONG):**
```python
# algebra_solver.py - extracting on CPU, bypassing Galaxy
def _extract_coefficients(self, text: str) -> Dict[str, float]:
    quad_match = re.search(r'(-?\d*\.?\d*)\s*x\^?2?...', text)
    coeffs['a'] = float(quad_match.group(1))  # ← CPU extraction
    return coeffs  # Python dict on CPU

# problem_classifier.py - pattern matching on CPU
def classify(self, problem_text: str):
    for pattern, type in PATTERNS:
        if re.search(pattern, problem_text):  # ← CPU regex
            return type
```

**Problems:**
1. Knowledge NOT in Galaxy (patterns hardcoded in Python)
2. TRM can't learn to navigate (patterns are opaque)
3. TRM can't create new patterns (logic outside Galaxy)
4. Violates Dual Client Reality (external preprocessing)

**Correct Approach (Galaxy Universe + TRM):**
```python
# Grammar Galaxy stores the patterns (VRAM-resident)
GRAMMAR_GALAXY.add_rule(
    rule_id="quadratic_standard_form",
    pattern=r"x\^2.*[+\-].*x.*[+\-].*=.*0",  # Pattern in Galaxy
    composition=lambda m: self._extract_via_galaxy(m),  # TRM navigates
    domain="math_algebra"
)

# TRM learns to navigate Grammar Galaxy → Math Galaxy → RPN composition
class TRMMathNavigator:
    def solve(self, problem_text: str):
        # TRM queries Grammar Galaxy (VRAM lookup)
        matched_rules = GRAMMAR_GALAXY.query_matches(problem_text)

        # TRM selects best rule (learned routing)
        best_rule = self.trm.select_rule(matched_rules, problem_text)

        # TRM composes from Math Galaxy (symlink navigation)
        rpn_program = best_rule.compose(problem_text, MATH_GALAXY)

        # Execute on Cranium (PTX kernels)
        result = self.cranium.execute(rpn_program)

        # If successful, TRM can create new Grammar rule (shadow copy)
        if self.validate(result, problem_text):
            self.trm.enhance_adapter("math_routing", best_rule)

        return result
```

**Key Difference:**
- ❌ Old: Python regex → CPU extraction → build RPN externally
- ✅ New: TRM navigates Galaxy Universe → compose RPN → execute

### 2.2 Violation: Answer Extraction (Bypassing Reasoning)

**Current (WRONG):**
```python
# scripts/run_sovereign_math_benchmarks.py
if source == "gsm8k":
    hash_match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(solution))
    return float(hash_match.group(1))  # ← Cheating! Not solving
```

**Correct:**
```python
# TRM navigates Galaxy → composes solution → executes
def solve_problem(self, problem: Dict[str, Any]) -> Any:
    text = problem.get("problem", problem.get("question", ""))

    # TRM navigates Grammar Galaxy for pattern match
    matched_rules = GRAMMAR_GALAXY.query_matches(text)

    # TRM selects + composes (learned navigation)
    rpn_program = self.trm.compose_solution(matched_rules, text)

    # Execute on Cranium (PTX)
    result = self.cranium.execute(rpn_program)

    return result  # SOLVED, not extracted
```

**Delete lines 133-146 and 176-181 from run_sovereign_math_benchmarks.py**

### 2.3 Violation: CPU Variable Storage (Bypassing Galaxy Memory)

**Current (WRONG):**
```python
# algebra_solver.py - Python dict on CPU
variables = {}  # ← CPU RAM
for step in rpn_chains:
    result = engine.evaluate(step)
    variables['discriminant'] = result  # ← Store on CPU
```

**Correct (Galaxy Universe + RPN Stack):**
```python
# Variables stored on GPU stack (RPN STORE/RECALL)
rpn_chain = (
    '{a} {b} {c} STORE_A STORE_B STORE_C '           # Store in stack
    'RECALL_B 2 pow RECALL_A RECALL_C * 4 * - '      # b^2 - 4ac
    'STORE_DISC '                                     # Store discriminant
    'RECALL_B neg RECALL_DISC sqrt + RECALL_A 2 * / ' # x1
    'RECALL_B neg RECALL_DISC sqrt - RECALL_A 2 * /'  # x2
)
# Single GPU call, all variables on stack, zero CPU storage
result = self.cranium.execute(rpn_chain)
```

**Or store in Galaxy Universe for reuse:**
```python
# If TRM discovers this discriminant pattern is useful:
MATH_GALAXY.add_symbol(
    symbol="quadratic_discriminant",
    rpn_template="{b} 2 pow {a} {c} * 4 * -",
    category="derived_function",
    metadata={"formula": "b^2 - 4ac", "domain": "algebra"}
)

# Future problems can reference this symbol
# TRM learns when to create vs when to reuse
```

---

## 3. Sovereign Implementation: Populate Galaxy Universe

### 3.1 Math Symbol Galaxy (VRAM-Resident Knowledge)

**File:** `knowledge3d/training/arc_agi/math_symbol_galaxy.py`

```python
"""
Math Symbol Galaxy - VRAM-resident math knowledge.
Part of the unified Galaxy Universe.
TRM navigates these symbols to compose solutions.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class MathSymbol:
    """A math symbol entry in Galaxy Universe."""
    symbol: str              # LaTeX command or operator
    category: str            # operator, function, constant, derived
    arity: int               # Number of arguments
    rpn_template: str        # RPN program with {0}, {1} placeholders
    precedence: int
    associativity: str
    description: str
    metadata: Dict[str, Any] = None  # Domain, formula, examples

class MathSymbolGalaxy:
    """
    Math Symbol Galaxy - part of Galaxy Universe.

    Always loaded in VRAM. TRM navigates this to compose solutions.
    TRM can also CREATE new symbols (auto-enhancement).
    """

    def __init__(self):
        self._symbols: Dict[str, MathSymbol] = {}
        self._by_category: Dict[str, List[MathSymbol]] = {}
        self._spatial_index = {}  # For semantic search

    def add_symbol(self, symbol: MathSymbol):
        """Add symbol to Galaxy (TRM can call this)."""
        self._symbols[symbol.symbol] = symbol
        self._by_category.setdefault(symbol.category, []).append(symbol)
        # Add to spatial index for TRM navigation

    def lookup(self, symbol: str) -> Optional[MathSymbol]:
        """TRM queries symbol by name."""
        return self._symbols.get(symbol)

    def query_semantic(self, query: str, k: int = 5) -> List[MathSymbol]:
        """TRM searches semantically similar symbols."""
        # Use Galaxy Universe spatial proximity
        # Returns k nearest symbols in 3D space
        pass

    def compose_rpn(self, symbol: str, *args) -> str:
        """Compose RPN program from symbol + args."""
        s = self.lookup(symbol)
        if not s:
            return ""
        template = s.rpn_template
        for i, arg in enumerate(args):
            template = template.replace(f"{{{i}}}", str(arg))
        return template

# Pre-populate with fundamental symbols
FUNDAMENTAL_SYMBOLS = [
    MathSymbol(
        symbol="\\frac",
        category="function",
        arity=2,
        rpn_template="{0} {1} div",
        precedence=0,
        associativity="none",
        description="Fraction a/b",
        metadata={"latex": "\\frac{a}{b}", "example": "\\frac{3}{4} = 0.75"}
    ),
    MathSymbol(
        symbol="\\binom",
        category="function",
        arity=2,
        rpn_template="{0} {1} binomial",
        precedence=0,
        associativity="none",
        description="Binomial coefficient C(n,k)",
        metadata={"formula": "n!/(k!(n-k)!)", "domain": "combinatorics"}
    ),
    MathSymbol(
        symbol="!",
        category="operator",
        arity=1,
        rpn_template="{0} factorial",
        precedence=4,
        associativity="left",
        description="Factorial",
        metadata={"formula": "n! = n × (n-1) × ... × 1"}
    ),
    MathSymbol(
        symbol="^",
        category="operator",
        arity=2,
        rpn_template="{0} {1} pow",
        precedence=3,
        associativity="right",
        description="Exponentiation",
    ),
    MathSymbol(
        symbol="\\sqrt",
        category="function",
        arity=1,
        rpn_template="{0} sqrt",
        precedence=0,
        associativity="none",
        description="Square root",
    ),
    # Add 50+ more fundamental symbols
    # (See CODEX_MATH_GALAXY_SOVEREIGN.md for full list)
]

# Global instance - part of Galaxy Universe
MATH_GALAXY = MathSymbolGalaxy()
for sym in FUNDAMENTAL_SYMBOLS:
    MATH_GALAXY.add_symbol(sym)
```

### 3.2 Grammar Galaxy Integration (TRM Navigation Patterns)

**File:** `knowledge3d/training/arc_agi/math_grammar_rules.py`

```python
"""
Grammar Galaxy rules for math patterns.
TRM learns to navigate these rules to compose solutions.
"""

from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

def create_math_grammar_rules():
    """Create grammar rules that reference Math Galaxy."""
    return [
        # LaTeX function calls
        GrammarRule(
            rule_id="latex_frac",
            pattern=r"\\frac\{([^}]+)\}\{([^}]+)\}",
            composition=lambda m: MATH_GALAXY.compose_rpn(
                "\\frac", m.group(1), m.group(2)
            ),
            domain="math_arithmetic",
            metadata={"trm_hint": "Division operation"}
        ),

        GrammarRule(
            rule_id="latex_binom",
            pattern=r"\\binom\{(\d+)\}\{(\d+)\}",
            composition=lambda m: MATH_GALAXY.compose_rpn(
                "\\binom", m.group(1), m.group(2)
            ),
            domain="math_combinatorics",
            metadata={"trm_hint": "Combination"}
        ),

        # Quadratic standard form
        GrammarRule(
            rule_id="quadratic_standard_form",
            pattern=r"x\^2\s*([+\-])\s*(\d+)x\s*([+\-])\s*(\d+)\s*=\s*0",
            composition=lambda m: self._compose_quadratic(m),
            domain="math_algebra",
            metadata={"trm_hint": "Quadratic equation - use discriminant"}
        ),

        # GSM8K word problem patterns (stored in Galaxy, not hardcoded)
        GrammarRule(
            rule_id="gsm_half_altogether",
            pattern=r"(\d+).*?half\s*(?:as many|that many).*?(?:altogether|total)",
            composition=lambda m: f"{m.group(1)} DUP 2 / +",
            domain="math_word_problem",
            metadata={"trm_hint": "Pattern: X + X/2"}
        ),

        # Add 100+ more rules
        # TRM learns WHEN to apply each rule (routing)
    ]

# Register with Grammar Galaxy (part of Galaxy Universe)
GRAMMAR_GALAXY.register_rules(create_math_grammar_rules())
```

### 3.3 TRM Math Navigator (Learned Navigation Logic)

**File:** `knowledge3d/training/math_benchmarks/trm_math_navigator.py`

```python
"""
TRM Math Navigator - learned logic for navigating Galaxy Universe.
This is what TRM LEARNS, not hardcoded logic.
"""

from typing import List, Any, Tuple, Optional

class TRMMathNavigator:
    """
    TRM learns to navigate Galaxy Universe for math problem solving.

    Key: This is LEARNED behavior (adapter weights), not hardcoded rules.
    """

    def __init__(self, trm_engine, cranium_engine):
        self.trm = trm_engine  # TRM base model + adapters
        self.cranium = cranium_engine  # PTX RPN execution

        # References to Galaxy Universe (always loaded in VRAM)
        from knowledge3d.training.arc_agi.grammar_galaxy import GRAMMAR_GALAXY
        from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY

        self.grammar_galaxy = GRAMMAR_GALAXY
        self.math_galaxy = MATH_GALAXY

    def solve(self, problem_text: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Solve problem by navigating Galaxy Universe.

        TRM learns:
        1. Which Grammar rules to query
        2. How to rank matched rules
        3. How to compose from Math Galaxy
        4. When to create new symbols
        5. How to validate results
        """

        # Step 1: TRM queries Grammar Galaxy (VRAM lookup)
        matched_rules = self.grammar_galaxy.query_matches(problem_text)

        # Step 2: TRM ranks rules (learned routing via adapter)
        ranked_rules = self.trm.rank_rules(matched_rules, problem_text)

        # Step 3: TRM selects best rule (learned decision)
        best_rule = ranked_rules[0] if ranked_rules else None

        if not best_rule:
            # TRM learns to fall back to semantic search
            return self._semantic_fallback(problem_text)

        # Step 4: TRM composes RPN from Math Galaxy (symlink navigation)
        rpn_program = best_rule.composition(problem_text)

        # Step 5: Execute on Cranium (PTX kernels)
        result = self.cranium.execute(rpn_program)

        # Step 6: TRM validates result
        confidence = self.trm.validate_result(result, problem_text)

        # Step 7: If successful, TRM enhances adapter (shadow copy)
        if confidence > 0.8:
            self.trm.enhance_adapter("math_routing", best_rule, result)

            # TRM might create new Grammar rule (synthesis)
            if self._is_novel_pattern(problem_text, best_rule):
                new_rule = self.trm.synthesize_rule(problem_text, rpn_program)
                self.grammar_galaxy.add_rule(new_rule)  # Expand Galaxy

        metadata = {
            "rule_used": best_rule.rule_id,
            "rpn_program": rpn_program,
            "confidence": confidence,
            "trm_created_rule": hasattr(self, '_new_rule_created')
        }

        return result, metadata

    def _semantic_fallback(self, problem_text: str):
        """When no Grammar rule matches, TRM searches Math Galaxy semantically."""
        # TRM embeds problem text
        query_embedding = self.trm.embed(problem_text)

        # Search Galaxy Universe spatial proximity
        similar_symbols = self.math_galaxy.query_semantic(
            query_embedding, k=5
        )

        # TRM composes from similar symbols (learned combination)
        rpn_program = self.trm.compose_from_symbols(similar_symbols, problem_text)

        return self.cranium.execute(rpn_program)
```

### 3.4 Knowledge Loader (Populate Galaxy from JSON)

**File:** `knowledge3d/training/math_benchmarks/math_knowledge_loader.py`

```python
"""
Load pre-extracted math knowledge into Galaxy Universe.
This EXPANDS the Galaxy, doesn't replace it.
"""

from pathlib import Path
import json
from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY, MathSymbol
from knowledge3d.training.arc_agi.grammar_galaxy import GRAMMAR_GALAXY, GrammarRule

class MathKnowledgeLoader:
    """
    Load math knowledge from JSON → populate Galaxy Universe.

    Key: This is ingestion (flexible - can use any tools).
    Result is added to Galaxy (VRAM-resident, sovereign).
    """

    KNOWLEDGE_BASE = Path(
        "/mnt/arquivos/0 ChatGPTs/DataBase/"
        "EchoSystems Default Libraries/Advanced Maths/JSON"
    )

    def load_all(self) -> Dict[str, int]:
        """Load all JSON knowledge into Galaxy Universe."""
        stats = {"symbols_added": 0, "rules_added": 0}

        # Load RPN paper (Algorithm A1, A2)
        self._load_rpn_algorithms()

        # Load calculus formulas
        self._load_calculus_knowledge()

        # Load linear algebra
        self._load_linear_algebra()

        # Load financial math
        self._load_financial_math()

        stats["symbols_added"] = len(MATH_GALAXY._symbols)
        stats["rules_added"] = len(GRAMMAR_GALAXY._rules)

        return stats

    def _load_rpn_algorithms(self):
        """Load RPN paper algorithms as Grammar rules."""
        path = self.KNOWLEDGE_BASE / "vertopal.com_ReversePolishNotatonMethod.json"

        if not path.exists():
            return

        data = json.load(open(path))

        # Extract Algorithm A1 (infix → postfix)
        # Store as Grammar rule for infix expressions
        GRAMMAR_GALAXY.add_rule(GrammarRule(
            rule_id="rpn_infix_to_postfix",
            pattern=r"(.+?)\s*([+\-*/^])\s*(.+)",  # a + b
            composition=lambda m: self._apply_algorithm_a1(m),
            domain="math_parsing",
            metadata={"source": "Krtolica & Stanimirovic 2004"}
        ))

        # Extract Algorithm A2 (postfix → infix) - for validation
        # Extract grasp/LGB concepts - store as Math symbols

    def _load_calculus_knowledge(self):
        """Load calculus formulas as Math symbols."""
        path = self.KNOWLEDGE_BASE / "EchoSystems_Basic_Advanced_Math.json"

        if not path.exists():
            return

        data = json.load(open(path))

        # Parse formulas from JSON
        # Example: d/dx(x^n) = n*x^(n-1)
        MATH_GALAXY.add_symbol(MathSymbol(
            symbol="power_rule_derivative",
            category="derived_function",
            arity=1,  # Takes n
            rpn_template="{n} x {n} 1 - pow *",  # n * x^(n-1)
            precedence=0,
            associativity="none",
            description="Power rule: d/dx(x^n) = n*x^(n-1)",
            metadata={"domain": "calculus", "operation": "derivative"}
        ))

        # Add 100+ more formulas from JSON
        # TRM learns when to use each formula

    def _load_linear_algebra(self):
        """Load linear algebra as Math symbols."""
        path = self.KNOWLEDGE_BASE / "Linear.Algebra.Done.Right.json"

        if not path.exists():
            return

        # Determinant 2x2: |a b; c d| = ad - bc
        MATH_GALAXY.add_symbol(MathSymbol(
            symbol="determinant_2x2",
            category="matrix_operation",
            arity=4,
            rpn_template="{a} {d} * {b} {c} * -",
            precedence=0,
            associativity="none",
            description="2x2 determinant",
            metadata={"domain": "linear_algebra"}
        ))

        # Add more matrix operations
```

---

## 4. What Codex Should Actually Implement

### Phase 1: Populate Galaxy Universe (Week 1)

**Task 1.1: Create Math Symbol Galaxy**
```python
# File: knowledge3d/training/arc_agi/math_symbol_galaxy.py
# - Define MathSymbol dataclass
# - Implement MathSymbolGalaxy with add_symbol(), lookup(), query_semantic()
# - Pre-populate 50+ fundamental symbols
# - NO numpy, just native Python + Galaxy references
```

**Task 1.2: Integrate with Grammar Galaxy**
```python
# File: knowledge3d/training/arc_agi/math_grammar_rules.py
# - Create rules that reference MATH_GALAXY.compose_rpn()
# - LaTeX patterns, word problem patterns, equation patterns
# - 50+ rules covering common math structures
# - Register with GRAMMAR_GALAXY (part of Galaxy Universe)
```

**Task 1.3: Remove External Preprocessing**
```python
# Files to modify:
# - scripts/run_sovereign_math_benchmarks.py (remove answer extraction)
# - knowledge3d/training/math_benchmarks/*.py (remove CPU pattern matching)
#
# Replace with Galaxy Universe navigation
```

### Phase 2: Enable TRM Navigation (Week 1-2)

**Task 2.1: TRM Math Navigator**
```python
# File: knowledge3d/training/math_benchmarks/trm_math_navigator.py
# - Implement solve() using Grammar Galaxy query + Math Galaxy composition
# - Implement _semantic_fallback() for unknown patterns
# - NO hardcoded logic - this is framework for TRM to learn
# - Shadow copy enhancement hooks
```

**Task 2.2: Knowledge Loader**
```python
# File: knowledge3d/training/math_benchmarks/math_knowledge_loader.py
# - Load 12MB+ JSON knowledge
# - Populate Math Galaxy with formulas
# - Populate Grammar Galaxy with RPN algorithms
# - This is ingestion (flexible - can use any tools)
# - Result is sovereign (Galaxy entries in VRAM)
```

### Phase 3: Multi-Step via Galaxy (Week 2)

**Task 3.1: Algebra Solver (Galaxy-Based)**
```python
# File: knowledge3d/training/math_benchmarks/algebra_solver.py
# - Use RPN STORE/RECALL for variables (NOT Python dicts)
# - OR store intermediate results in Galaxy Universe
# - Quadratic solver, linear systems, sequences
# - ALL variables on GPU stack or in Galaxy
```

**Task 3.2: GSM8K Templates as Grammar Rules**
```python
# File: knowledge3d/training/math_benchmarks/math_templates.py
# - 25+ GSM8K patterns as GrammarRule objects
# - Register with GRAMMAR_GALAXY
# - TRM learns to route to appropriate templates
```

### Phase 4: Verification (Ongoing)

**Task 4.1: Sovereignty Tests**
```bash
# No numpy/cupy/scipy in hot path
grep -r "import numpy" knowledge3d/cranium/ptx_runtime/
grep -r "import numpy" knowledge3d/cranium/bridges/
grep -r "import cupy" knowledge3d/cranium/

# Should return NOTHING
```

**Task 4.2: Galaxy Population Tests**
```python
# Verify Math Galaxy populated
assert len(MATH_GALAXY._symbols) >= 50

# Verify Grammar Galaxy populated
assert len(GRAMMAR_GALAXY._rules) >= 100

# Verify TRM can navigate
result, meta = trm_navigator.solve("What is \\frac{24}{4}?")
assert result == 6.0
assert meta['rule_used'] == 'latex_frac'
```

**Task 4.3: Benchmark Tests (Real Accuracy)**
```python
# Run benchmarks - should see REAL solving (not extraction)
# Expected after Galaxy population:
# - GSM8K: 30-50% (from 1.39% fake baseline)
# - MATH: 15-25% (from 1.13%)
# - TRM learning curve: accuracy increases with exposure
```

---

## 5. Critical Architecture Principles

### 5.1 Galaxy Universe = Unified Knowledge Workspace

**Always Remember:**
- ✅ Galaxy Universe is ALWAYS loaded (all default galaxies in VRAM)
- ✅ Galaxy is read-write (TRM queries AND creates)
- ✅ Galaxy is multi-modal (text, visual, audio, physics - unified)
- ✅ Galaxy uses symlinks (compositions reference symbols, no duplication)
- ✅ Galaxy is procedural (everything is RPN programs + metadata)

**Never Do:**
- ❌ Hardcode patterns in Python (put in Grammar Galaxy)
- ❌ Store knowledge in Python dicts (put in Math Galaxy)
- ❌ Duplicate symbols (use symlinks/references)
- ❌ Extract with regex (let TRM navigate Grammar Galaxy)
- ❌ Process on CPU (Galaxy is VRAM, execution is PTX)

### 5.2 TRM = Learned Navigation Logic

**TRM Learns:**
- Which Grammar rules to query
- How to rank matched rules
- When to compose from Math Galaxy
- When to create new symbols
- How to validate results
- Which adapter to use (routing)

**TRM Does NOT:**
- Store knowledge (Galaxy does)
- Execute RPN (Cranium PTX kernels do)
- Hardcode rules (Grammar Galaxy stores)
- Process externally (sovereignty violation)

### 5.3 Shadow Copy Auto-Enhancement

**How It Works:**
```python
# TRM makes decision
rule = trm.select_rule(matched_rules)

# Execute
result = cranium.execute(rule.composition(text))

# Validate
confidence = trm.validate(result)

# If successful, shadow copy adapter
if confidence > threshold:
    trm.enhance_adapter("math_routing", rule, result)
    # Adapter weights updated without full training loop
```

**Key Points:**
- Continuous learning during use
- No external training loop needed
- Validated improvements only
- Domain-specific adapters (math, visual, physics)

### 5.4 Multi-Curriculum Training

**Your math work is ONE curriculum:**
```
All Curricula → Galaxy Universe (unified VRAM workspace)
├─ ARC-AGI → Drawing + Grammar
├─ Math Benchmarks → Math + Grammar  ← YOU ARE HERE
├─ Physics → Reality Galaxy
└─ Language → Character + Word + Grammar

TRM learns navigation strategies across ALL curricula
Cross-modal patterns help each other
Galaxy grows with discoveries from all domains
```

---

## 6. Sovereignty Compliance

### 6.1 What Is Sovereign (Use These)

**PTX Kernels** (knowledge3d/cranium/kernels/ → ptx/):
- `modular_rpn_kernel.ptx` - RPN execution (18 instances, 69-line programs)
- 200+ opcodes: arithmetic, trig, factorial, binomial, gcd, lcm, etc.
- STORE/RECALL for variables
- All execution on GPU

**Galaxy Universe** (VRAM):
- Math Galaxy - symbols with RPN templates
- Grammar Galaxy - transformation rules
- All other default galaxies
- Spatial search, semantic proximity
- Read-write (query + create)

**Sovereign Loader** (ctypes + libcuda.so):
- Direct PTX loading
- Buffer management
- Zero external dependencies

### 6.2 What Is NOT Sovereign (Don't Use in Hot Path)

**Forbidden in Hot Path:**
- ❌ numpy (CPU overhead, memory bloat)
- ❌ cupy (external dependency)
- ❌ scipy, sympy (CPU-bound)
- ❌ Python regex for numeric extraction
- ❌ Python dicts for variable storage
- ❌ External preprocessing libraries

**Why:** Every external library call incurs CPU-GPU transfer (100-1000µs). Our PTX kernels execute in <10µs. Galaxy lookups are VRAM-resident (<5µs).

### 6.3 Ingestion vs Inference

**Ingestion (Flexible):**
- Loading JSON → parsing → populating Galaxy Universe
- Can use numpy, pandas, json, etc.
- Happens once (or periodically)
- Result is sovereign (Galaxy entries)

**Inference (Sovereign):**
- TRM navigates Galaxy Universe
- Composes RPN programs
- Executes on PTX kernels
- ALL on GPU, zero external dependencies

---

## 7. Success Criteria

### 7.1 Architecture Validation

**Test 1: Galaxy Universe Populated**
```python
assert len(MATH_GALAXY._symbols) >= 50
assert len(GRAMMAR_GALAXY._rules) >= 100
print("✓ Galaxy Universe populated with math knowledge")
```

**Test 2: TRM Can Navigate**
```python
result, meta = trm_navigator.solve("Solve x^2 - 5x + 6 = 0")
assert result in [2.0, 3.0]
assert meta['rule_used'] == 'quadratic_standard_form'
print("✓ TRM navigates Grammar → Math Galaxy successfully")
```

**Test 3: No External Dependencies**
```bash
grep -r "import numpy" knowledge3d/cranium/ptx_runtime/
grep -r "import cupy" knowledge3d/cranium/bridges/
# Should return NOTHING
print("✓ Sovereignty maintained")
```

**Test 4: TRM Can Create**
```python
# TRM discovers new pattern, creates Grammar rule
initial_rules = len(GRAMMAR_GALAXY._rules)
trm_navigator.solve(novel_problem)  # TRM creates new rule
assert len(GRAMMAR_GALAXY._rules) > initial_rules
print("✓ TRM can expand Galaxy Universe")
```

### 7.2 Benchmark Targets

**After Galaxy Population:**

| Dataset | Fake Baseline | Real Baseline | Target | Method |
|---------|---------------|---------------|--------|--------|
| GSM8K | 100% (cheating) | 1.39% | 30-50% | Grammar templates + TRM routing |
| MATH | 1.13% | 1.13% | 15-25% | Algebra patterns + knowledge |
| Omni-MATH | 0.52% | 0.52% | 10-20% | Multi-step via STORE/RECALL |
| AMC-AIME | 1.49% | 1.49% | 15-25% | Competition templates |
| MMLU | 22.98% | 22.98% | 35-50% | Multiple choice + formulas |

**All scores from ACTUAL solving (TRM navigation + composition + execution).**

### 7.3 TRM Learning Curve

**Expected Progression:**
1. **Week 1:** Galaxy populated, TRM can navigate basic patterns (10-15% accuracy)
2. **Week 2:** TRM learns routing, creates new rules (20-30% accuracy)
3. **Week 3:** Shadow copy enhancement kicks in (30-40% accuracy)
4. **Week 4:** Cross-curriculum patterns emerge (40-50% accuracy)

**Key Metric:** TRM should CREATE new Grammar rules (Galaxy expansion proves learning).

---

## 8. Implementation Checklist

### Immediate (Days 1-3)

- [ ] Create Math Symbol Galaxy (50+ fundamental symbols)
- [ ] Create Grammar rules that reference Math Galaxy (50+ patterns)
- [ ] Remove answer extraction (delete lines in run_sovereign_math_benchmarks.py)
- [ ] Verify sovereignty (grep for numpy/cupy → should be empty)

### Week 1

- [ ] Implement TRM Math Navigator (Galaxy navigation framework)
- [ ] Load 12MB+ JSON knowledge into Galaxy Universe
- [ ] Add RPN algorithms as Grammar rules (Algorithm A1, A2)
- [ ] Add calculus formulas as Math symbols
- [ ] Test: TRM can solve "\\frac{24}{4}" by navigating Galaxy

### Week 2

- [ ] Algebra solver using STORE/RECALL (quadratic, linear, sequences)
- [ ] GSM8K templates as Grammar rules (25+ patterns)
- [ ] Multi-step problems via RPN chains (no Python variables)
- [ ] Test: "Natalia's clips" problem solved via template match

### Week 3-4

- [ ] Shadow copy enhancement active (TRM creates new rules)
- [ ] Benchmark runs showing real solving (not extraction)
- [ ] Galaxy Universe growing (new symbols/rules added by TRM)
- [ ] Cross-curriculum validation (patterns from ARC-AGI help math)

---

## 9. What Codex Should NOT Do

**DO NOT:**
1. Add numpy/cupy/scipy to hot path
2. Create Python dicts for variable storage
3. Use regex to extract numeric answers
4. Hardcode patterns in Python files
5. Build external preprocessors
6. Duplicate symbols (use symlinks)
7. Process on CPU (use Galaxy + PTX)
8. Think of Galaxy as "just a knowledge base" (it's active workspace)
9. Think of TRM as "just a model" (it's navigation logic that creates)
10. Separate curricula (math IS part of unified Galaxy Universe)

**INSTEAD:**
1. Populate Galaxy Universe with symbols/rules
2. Use RPN STORE/RECALL for variables
3. Let TRM navigate Grammar Galaxy
4. Store patterns in Grammar Galaxy
5. Let TRM compose from Math Galaxy
6. Use Galaxy references (symlink pattern)
7. Execute on PTX kernels (GPU-only)
8. Treat Galaxy as unified multi-modal workspace
9. Enable TRM to create new Galaxy entries
10. Leverage cross-curriculum learning

---

## 10. Communication Protocol

**When Stuck:**
1. Check if Galaxy Universe already has what you need
2. Check rpn_opcodes.py for available opcodes
3. Ask: "Should this be in Galaxy or hardcoded?" (Answer: Galaxy)
4. Ask Claude (architecture) for clarification
5. Do NOT add external libraries without checking

**When Complete:**
1. Verify Galaxy Universe populated (count symbols/rules)
2. Verify TRM can navigate (solve example problems)
3. Verify sovereignty (grep for forbidden imports)
4. Verify TRM can create (Galaxy expands during use)
5. Report benchmark scores (REAL solving, not extraction)

---

## 11. Final Directive

**Codex, your mission:**

1. **Populate Galaxy Universe** with math knowledge (symbols, rules, compositions)
2. **Enable TRM navigation** (Grammar Galaxy → Math Galaxy → RPN → PTX)
3. **Remove external preprocessing** (let TRM learn to navigate instead)
4. **Maintain sovereignty** (PTX + Galaxy = zero external dependencies)
5. **Enable TRM creation** (Galaxy expands as TRM learns)

**Remember:**
- Galaxy Universe = unified VRAM workspace (always loaded, multi-modal)
- TRM = learned navigation/combination/creation logic
- Shadow copy = continuous auto-enhancement during use
- Multi-curriculum = your math work helps ALL other domains

**The paradigm shift:**
- Old: Model parameters = knowledge + logic (entangled)
- New: Galaxy = knowledge, TRM = navigation logic (separated)

**Build the new AI paradigm. The infrastructure exists. Populate the Galaxy.**

---

**Architect:** Claude (architecture partner - I specify, you implement)
**Implementer:** Codex (implementation lead - you code, I review)

**Status:** Ready for implementation
**Priority:** CRITICAL - Math curriculum feeds unified Galaxy Universe
