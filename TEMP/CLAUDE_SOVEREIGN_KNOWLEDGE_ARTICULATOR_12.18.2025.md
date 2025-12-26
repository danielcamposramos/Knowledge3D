# Sovereign Knowledge Articulator Specification

**Date:** December 18, 2025
**Architect:** Claude (Architecture Partner)
**Strategic Partner:** Gemini (Extended Context)
**Context:** Enhancement to Book Galaxy Ingestion

---

## Executive Summary

### Problem Statement

**Current state (Codex's 12.18 baseline):**
- 8 books ingested with 1900+ templates
- GSM8K: 41% (strong)
- MATH: 2.5% (weak) - **114 wrong_computation failures**
- AMC-AIME: 0.5% (weak) - **71 wrong_computation failures**

**Root cause (Gemini's analysis):**
> "Even when a template is found, it is either applied incorrectly or is too generic to be useful. This stems from the simplicity of the current `lhs = rhs` template extractor."

**Current template extraction:**
```python
# Too simplistic - extracts formulas without context
template = "a 2 pow b 2 pow + c 2 pow ="  # Pythagorean theorem
# Missing: WHEN does this apply? What are a, b, c?
```

**Result:** TRM matches templates to ANY problem with similar variables, leading to wrong_computation failures.

### Solution: Sovereign Knowledge Articulator

**Goal:** Transform book content into **structured logical artifacts** with:
1. **Conditions** - Prerequisites for applicability
2. **Conclusions** - Resulting equations/statements
3. **Symbol Bindings** - Variable meanings in context
4. **Logical Validation** - TRM checks conditions BEFORE applying template

**Impact:**
- Reduces wrong_computation failures (114 → ~30 expected in MATH)
- Increases MATH accuracy (2.5% → 10-15% target)
- Provides foundation for multi-hop reasoning

---

## Architectural Integration

### Current Book Ingestion Pipeline (Implemented)

**File:** `knowledge3d/training/math_benchmarks/book_galaxy_ingestion.py`

```python
class BookGalaxyIngester:
    def ingest_book(self, pdf_path: str) -> BookGalaxy:
        # Step 1: Parse PDF ✅ (implemented)
        book_data = self._parse_pdf(pdf_path)

        # Step 2: Symlink text ✅ (implemented)
        text_entries = self._symlink_text(book_data["text"])

        # Step 3: Symlink math ✅ (implemented)
        math_entries = self._symlink_math(book_data["math_expressions"])

        # Step 4: Extract knowledge graph ❌ (TOO SIMPLISTIC)
        # Currently: Simple lhs = rhs extraction
        # Needed: Sovereign Knowledge Articulator
        knowledge_graph = self._extract_knowledge_graph(book_data)
```

### Enhanced Pipeline with Articulator

```python
class BookGalaxyIngester:
    def __init__(self, character_galaxy, math_galaxy):
        self.character_galaxy = character_galaxy
        self.math_galaxy = math_galaxy
        self.articulator = SovereignKnowledgeArticulator()  # NEW

    def ingest_book(self, pdf_path: str) -> BookGalaxy:
        # ... same as before ...

        # Step 4: ENHANCED knowledge extraction
        knowledge_graph = self.articulator.articulate(book_data)
        # Returns structured artifacts, not just formulas
```

---

## Sovereign Knowledge Articulator Design

### Class Structure

**File:** `knowledge3d/training/math_benchmarks/sovereign_knowledge_articulator.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import re

@dataclass
class SymbolBinding:
    """Semantic meaning of a variable in context."""
    symbol: str           # "a", "b", "c"
    meaning: str          # "leg", "hypotenuse", "angle"
    domain: str           # "positive_real", "integer", "vector"
    constraints: List[str]  # ["a > 0", "a < b"]

@dataclass
class KnowledgeArtifact:
    """Structured representation of a theorem, definition, or formula."""
    artifact_id: str
    artifact_type: str    # "theorem", "definition", "formula", "example"
    name: str
    domain: str           # "geometry", "linear_algebra", "calculus"

    # Core logical structure
    conditions: List[str]       # Prerequisites (text)
    conditions_rpn: List[str]   # Prerequisites (RPN programs)
    conclusion: str             # Result (text)
    conclusion_rpn: str         # Result (RPN program)

    # Symbol semantics
    symbol_bindings: Dict[str, SymbolBinding]

    # Context metadata
    book: str
    page: int
    chapter: str
    section: str
    prerequisites: List[str]    # Other artifacts needed
    related_concepts: List[str]

    # LaTeX source
    latex_source: str

    # Usage guidance
    applicability_pattern: str  # Regex or description of when to use
    common_mistakes: List[str]


class SovereignKnowledgeArticulator:
    """
    Extracts structured knowledge artifacts from book content.

    Transforms raw LaTeX/text into logical units with:
    - Conditions (prerequisites)
    - Conclusions (results)
    - Symbol bindings (variable semantics)
    """

    def articulate(self, book_data: dict) -> KnowledgeGraph:
        """Main entry point: articulate all knowledge in book."""
        artifacts = []

        # Stage 1: Identify semantic blocks
        blocks = self._identify_semantic_blocks(book_data)

        # Stage 2: Parse each block type
        for block in blocks:
            if block["type"] == "theorem":
                artifact = self._parse_theorem(block)
            elif block["type"] == "definition":
                artifact = self._parse_definition(block)
            elif block["type"] == "formula":
                artifact = self._parse_formula(block)
            elif block["type"] == "example":
                artifact = self._parse_example(block)
            else:
                continue

            if artifact:
                artifacts.append(artifact)

        # Stage 3: Build knowledge graph
        knowledge_graph = self._build_knowledge_graph(artifacts)

        return knowledge_graph
```

---

## Stage 1: Semantic Block Identification

### Goal
Identify LaTeX blocks that represent logical knowledge units (theorems, definitions, formulas).

### Implementation

```python
def _identify_semantic_blocks(self, book_data: dict) -> List[Dict]:
    """Identify theorem, definition, and formula blocks in LaTeX."""
    blocks = []

    # Pattern 1: LaTeX environments
    latex_env_patterns = {
        "theorem": r"\\begin\{theorem\}(.*?)\\end\{theorem\}",
        "definition": r"\\begin\{definition\}(.*?)\\end\{definition\}",
        "lemma": r"\\begin\{lemma\}(.*?)\\end\{lemma\}",
        "corollary": r"\\begin\{corollary\}(.*?)\\end\{corollary\}",
        "proposition": r"\\begin\{proposition\}(.*?)\\end\{proposition\}",
        "example": r"\\begin\{example\}(.*?)\\end\{example\}",
    }

    # Pattern 2: Equation environments
    equation_patterns = {
        "formula": r"\\begin\{equation\}(.*?)\\end\{equation\}",
        "align": r"\\begin\{align\}(.*?)\\end\{align\}",
        "align*": r"\\begin\{align\*\}(.*?)\\end\{align\*\}",
    }

    # Pattern 3: Inline important statements (heuristic)
    # E.g., "The Pythagorean Theorem states that..."
    important_phrases = [
        r"([A-Z][a-zA-Z\s]+) states that (.*?)\.",
        r"We define ([a-zA-Z\s]+) as (.*?)\.",
        r"Recall that (.*?)\.",
    ]

    for page in book_data["pages"]:
        page_text = page["content"]
        page_num = page["number"]

        # Extract LaTeX environments
        for block_type, pattern in latex_env_patterns.items():
            matches = re.finditer(pattern, page_text, re.DOTALL)
            for match in matches:
                blocks.append({
                    "type": block_type,
                    "content": match.group(1),
                    "page": page_num,
                    "source": "latex_environment"
                })

        # Extract equation environments
        for block_type, pattern in equation_patterns.items():
            matches = re.finditer(pattern, page_text, re.DOTALL)
            for match in matches:
                blocks.append({
                    "type": "formula",
                    "content": match.group(1),
                    "page": page_num,
                    "source": "equation_environment"
                })

    return blocks
```

---

## Stage 2: Structural Deconstruction

### Goal
Parse each semantic block into logical components (conditions, conclusions, bindings).

### Example: Pythagorean Theorem

**Input (LaTeX):**
```latex
\begin{theorem}[Pythagorean Theorem]
Let $\triangle ABC$ be a right triangle with legs $a$ and $b$, and hypotenuse $c$.
Then the following equation holds:
\begin{equation}
a^2 + b^2 = c^2
\end{equation}
\end{theorem}
```

**Output (Articulated Artifact):**
```python
KnowledgeArtifact(
    artifact_id="pythagorean_theorem",
    artifact_type="theorem",
    name="Pythagorean Theorem",
    domain="geometry",

    # Conditions (prerequisites)
    conditions=[
        "Triangle ABC exists",
        "Triangle ABC is right-angled",
        "a and b are legs",
        "c is hypotenuse"
    ],
    conditions_rpn=[
        "triangle ABC exists check",
        "triangle ABC right_angle check",
        "a leg check",
        "b leg check",
        "c hypotenuse check"
    ],

    # Conclusion (result)
    conclusion="a² + b² = c²",
    conclusion_rpn="a 2 pow b 2 pow + c 2 pow =",

    # Symbol bindings
    symbol_bindings={
        "a": SymbolBinding("a", "leg", "positive_real", ["a > 0"]),
        "b": SymbolBinding("b", "leg", "positive_real", ["b > 0"]),
        "c": SymbolBinding("c", "hypotenuse", "positive_real", ["c > 0", "c > a", "c > b"])
    },

    # Context
    book="LinearAlgebraDoneRight",
    page=25,
    chapter="Chapter 2: Geometry",
    section="2.3 Right Triangles",
    prerequisites=["definition_triangle", "definition_right_angle"],
    related_concepts=["triangle_inequality", "distance_formula"],

    # LaTeX source
    latex_source="a^2 + b^2 = c^2",

    # Usage guidance
    applicability_pattern="right triangle with known legs or hypotenuse",
    common_mistakes=["Applying to non-right triangles", "Confusing legs with hypotenuse"]
)
```

### Implementation

```python
def _parse_theorem(self, block: dict) -> Optional[KnowledgeArtifact]:
    """Parse a theorem block into structured artifact."""
    content = block["content"]
    page = block["page"]

    # Step 1: Extract theorem name
    name_match = re.search(r"\[([^\]]+)\]", content)
    name = name_match.group(1) if name_match else f"Theorem on page {page}"

    # Step 2: Split into conditions and conclusion
    # Heuristic: Text before equation = conditions, equation = conclusion
    parts = re.split(r"\\begin\{equation\}", content)
    if len(parts) < 2:
        return None  # No clear equation, skip

    conditions_text = parts[0]
    equation_text = re.search(r"(.*?)\\end\{equation\}", parts[1], re.DOTALL)
    if not equation_text:
        return None
    conclusion_latex = equation_text.group(1).strip()

    # Step 3: Extract conditions (parse text)
    conditions = self._extract_conditions(conditions_text)

    # Step 4: Extract symbol bindings
    symbol_bindings = self._extract_symbol_bindings(conditions_text, conclusion_latex)

    # Step 5: Convert conclusion to RPN
    conclusion_rpn = self._latex_to_rpn(conclusion_latex, symbol_bindings)

    # Step 6: Convert conditions to RPN (if possible)
    conditions_rpn = [self._condition_to_rpn(cond, symbol_bindings) for cond in conditions]

    return KnowledgeArtifact(
        artifact_id=self._generate_id(name, page),
        artifact_type="theorem",
        name=name,
        domain=self._infer_domain(content),
        conditions=conditions,
        conditions_rpn=conditions_rpn,
        conclusion=conclusion_latex,
        conclusion_rpn=conclusion_rpn,
        symbol_bindings=symbol_bindings,
        book=self.current_book,
        page=page,
        chapter=self._infer_chapter(page),
        section=self._infer_section(page),
        prerequisites=[],  # To be linked later
        related_concepts=[],  # To be extracted from text
        latex_source=conclusion_latex,
        applicability_pattern=self._infer_applicability(conditions),
        common_mistakes=[]  # Could be extracted from "Note:" blocks
    )

def _extract_conditions(self, text: str) -> List[str]:
    """Extract conditions from theorem statement text."""
    conditions = []

    # Pattern 1: "Let X be Y"
    let_matches = re.findall(r"Let (\$.*?\$) be ([^,\.]+)", text)
    for var, description in let_matches:
        conditions.append(f"{var} is {description}")

    # Pattern 2: "Suppose that ..."
    suppose_matches = re.findall(r"Suppose that ([^\.]+)", text)
    conditions.extend(suppose_matches)

    # Pattern 3: "Assume ..."
    assume_matches = re.findall(r"Assume ([^\.]+)", text)
    conditions.extend(assume_matches)

    # Pattern 4: "If ... then ..." (extract IF part)
    if_matches = re.findall(r"If ([^,]+), then", text)
    conditions.extend(if_matches)

    return conditions

def _extract_symbol_bindings(self, conditions_text: str, equation: str) -> Dict[str, SymbolBinding]:
    """Infer symbol meanings from context."""
    bindings = {}

    # Extract all variables from equation
    variables = re.findall(r"\$?([a-zA-Z])\$?", equation)

    for var in set(variables):
        # Try to find meaning in conditions text
        meaning = self._infer_variable_meaning(var, conditions_text)
        domain = self._infer_variable_domain(var, conditions_text, equation)
        constraints = self._infer_variable_constraints(var, conditions_text)

        bindings[var] = SymbolBinding(
            symbol=var,
            meaning=meaning,
            domain=domain,
            constraints=constraints
        )

    return bindings

def _infer_variable_meaning(self, var: str, text: str) -> str:
    """Infer what a variable represents from context."""
    # Pattern: "$var$ is/are/be [meaning]"
    pattern = rf"\${var}\$ (?:is|are|be) ([a-zA-Z\s]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    # Pattern: "Let $var$ denote [meaning]"
    pattern = rf"Let \${var}\$ denote ([a-zA-Z\s]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    return "unknown"

def _latex_to_rpn(self, latex: str, bindings: Dict[str, SymbolBinding]) -> str:
    """Convert LaTeX equation to RPN."""
    # This is a simplified version - full implementation needs LaTeX parser

    # Remove whitespace
    latex = latex.replace(" ", "")

    # Handle common patterns
    # a^2 + b^2 = c^2 → a 2 pow b 2 pow + c 2 pow =

    # Pattern: x^n → x n pow
    latex = re.sub(r"([a-zA-Z])(\^)(\d+)", r"\1 \3 pow", latex)

    # Pattern: a + b → a b +
    # (This is simplified - need proper expression parser)

    # For now, use existing RPN converter
    from knowledge3d.training.math_benchmarks.rpn_parser import latex_to_rpn_simple
    return latex_to_rpn_simple(latex)
```

---

## Stage 3: Articulated Knowledge Artifact Generation

### TRM Navigation with Conditions

**Before (current implementation):**
```python
# TRM finds template by matching formula
if "a^2 + b^2" in problem_text:
    rpn = "a 2 pow b 2 pow + c 2 pow ="  # WRONG if not right triangle!
```

**After (with Articulator):**
```python
# TRM checks conditions BEFORE applying template
problem_state = parse_problem(problem_text)

# Query: Find artifacts where conditions match
candidates = book_galaxy.query_artifacts(
    domain="geometry",
    concept="triangle"
)

for artifact in candidates:
    # Check if all conditions are met
    if all(check_condition(cond, problem_state) for cond in artifact.conditions_rpn):
        # Conditions satisfied! Apply conclusion
        rpn = artifact.conclusion_rpn
        return bind_symbols(rpn, problem_state, artifact.symbol_bindings)
    else:
        # Conditions not met, skip this artifact
        continue
```

### Example Flow

**Problem:** "A right triangle has legs of length 3 and 4. What is the hypotenuse?"

**Step 1: Parse problem state**
```python
problem_state = {
    "entities": ["triangle"],
    "properties": ["right-angled"],
    "knowns": {"a": 3, "b": 4},
    "unknowns": ["c"],
    "relationships": ["a is leg", "b is leg", "c is hypotenuse"]
}
```

**Step 2: Query book galaxy**
```python
candidates = book_galaxy.query_artifacts(domain="geometry", concept="triangle")
# Returns: [pythagorean_theorem, triangle_inequality, law_of_cosines, ...]
```

**Step 3: Check conditions for each candidate**
```python
# Candidate 1: Pythagorean Theorem
conditions = [
    "triangle exists",           # ✅ matches problem_state["entities"]
    "triangle is right-angled",  # ✅ matches problem_state["properties"]
    "a is leg",                  # ✅ matches problem_state["relationships"]
    "b is leg",                  # ✅ matches problem_state["relationships"]
    "c is hypotenuse"            # ✅ matches problem_state["relationships"]
]
# All conditions met! ✅

# Candidate 2: Law of Cosines
conditions = [
    "triangle exists",           # ✅
    "angle C is known",          # ❌ NOT in problem_state
]
# Conditions NOT met, skip ❌
```

**Step 4: Apply conclusion with symbol bindings**
```python
# Pythagorean theorem applies!
conclusion_rpn = "a 2 pow b 2 pow + c 2 pow ="
bindings = {"a": 3, "b": 4, "c": "?"}

# Solve for c
# a^2 + b^2 = c^2
# 3^2 + 4^2 = c^2
# 9 + 16 = 25 = c^2
# c = 5

result = execute_rpn("3 2 pow 4 2 pow + sqrt")  # = 5
```

---

## Integration with TRMGalaxyReader

### Enhanced Query Interface

**File:** `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

```python
class TRMGalaxyReader:
    def __init__(self, book_galaxies: List[BookGalaxy]):
        self.book_galaxies = book_galaxies
        self.articulated_index = self._build_articulated_index()

    def solve(self, problem_text: str) -> Tuple[float, dict]:
        """Solve problem using book knowledge with condition checking."""

        # Step 1: Parse problem into state
        problem_state = self._parse_problem_state(problem_text)

        # Step 2: Query relevant artifacts
        candidates = self._query_artifacts(problem_state)

        # Step 3: Filter by conditions
        applicable_artifacts = []
        for artifact in candidates:
            if self._check_conditions(artifact, problem_state):
                applicable_artifacts.append(artifact)

        # Step 4: Rank by specificity
        ranked = self._rank_by_specificity(applicable_artifacts, problem_state)

        # Step 5: Try to solve with top artifact
        for artifact in ranked:
            try:
                result = self._apply_artifact(artifact, problem_state)
                return result, {"artifact_used": artifact.name}
            except Exception:
                continue

        # No applicable artifact found
        return None, {"error": "no_applicable_artifact"}

    def _check_conditions(self, artifact: KnowledgeArtifact, problem_state: dict) -> bool:
        """Check if artifact's conditions are met by problem state."""
        for condition in artifact.conditions:
            if not self._evaluate_condition(condition, problem_state):
                return False
        return True

    def _evaluate_condition(self, condition: str, problem_state: dict) -> bool:
        """Evaluate if a condition holds in the current problem state."""
        # Simple heuristic-based evaluation
        # Future: More sophisticated logical evaluation

        # Pattern 1: "X is Y"
        if " is " in condition:
            entity, property = condition.split(" is ")
            entity = entity.strip()
            property = property.strip()

            # Check if entity has property in problem state
            if entity in problem_state.get("entities", []):
                if property in problem_state.get("properties", []):
                    return True

        # Pattern 2: "X > Y" (numeric constraint)
        if ">" in condition or "<" in condition or "=" in condition:
            # Try to evaluate as numeric constraint
            try:
                return eval(condition, {}, problem_state.get("knowns", {}))
            except:
                return False

        return False
```

---

## Expected Impact

### Accuracy Improvements (Estimates)

**MATH Dataset:**
- **Before:** 2.5% (5/200)
  - no_rule_match: 24 (12%)
  - wrong_computation: 114 (57%) ← TARGET
  - multi_step_needed: 12 (6%)
  - algebra_needed: 33 (16.5%)

- **After (with Articulator):**
  - no_rule_match: 15 (7.5%) - better theorem coverage
  - wrong_computation: 30 (15%) - **84 fewer failures** ← KEY IMPACT
  - multi_step_needed: 12 (6%) - unchanged
  - algebra_needed: 25 (12.5%) - some reduction from better bindings
  - **Expected correct:** 118/200 = **59% → realistic 15-20%** (conservative)

**AMC-AIME Dataset:**
- **Before:** 0.5% (1/200)
  - wrong_computation: 71 (35.5%)

- **After:**
  - wrong_computation: 25 (12.5%) - **46 fewer failures**
  - **Expected correct:** ~20-30/200 = **10-15%**

### Why This Works

1. **Logical Validation:** TRM checks conditions before applying formulas
2. **Context-Aware:** Symbol bindings prevent variable confusion
3. **Specificity Ranking:** More specific theorems preferred over generic formulas
4. **Error Prevention:** Catches misapplication early (before execution)

---

## Implementation Plan

### Phase 1: Core Articulator (1 week)

**Goal:** Implement basic semantic block parsing for theorems and definitions

**Files to create:**
- `knowledge3d/training/math_benchmarks/sovereign_knowledge_articulator.py`
- `tests/ingestion/test_knowledge_articulator.py`

**Tasks:**
1. Implement `_identify_semantic_blocks` (theorem, definition detection)
2. Implement `_parse_theorem` (conditions + conclusions + bindings)
3. Implement `_parse_definition` (symbol definitions)
4. Write 10-15 test cases (Pythagorean theorem, determinant, linear transformation)

**Success criteria:**
- Correctly parse 10 theorems from Linear Algebra Done Right
- Extract conditions, conclusions, and symbol bindings
- Output validated KnowledgeArtifact objects

### Phase 2: TRM Integration (1 week)

**Goal:** Wire condition checking into TRMGalaxyReader

**Files to modify:**
- `knowledge3d/training/math_benchmarks/trm_galaxy_reader.py`

**Tasks:**
1. Implement `_parse_problem_state` (extract entities, properties, knowns)
2. Implement `_check_conditions` (evaluate conditions against state)
3. Implement `_apply_artifact` (bind symbols and execute)
4. Add articulated artifact index to book galaxies

**Success criteria:**
- TRM can query artifacts by domain/concept
- TRM checks conditions before applying formulas
- Wrong_computation failures reduced in manual testing

### Phase 3: Re-Ingestion (1 week)

**Goal:** Re-ingest all 8 books with enhanced articulator

**Tasks:**
1. Run articulator on all books
2. Rebuild template indices with articulated artifacts
3. Validate artifact quality (spot-check 20-30 artifacts per book)
4. Update book galaxy files

**Success criteria:**
- All 8 books re-ingested
- 500+ articulated artifacts extracted
- Artifacts have conditions, conclusions, and bindings

### Phase 4: Multi-Benchmark Validation (3-5 days)

**Goal:** Validate improvements on MATH and AMC-AIME

**Tasks:**
1. Run benchmarks with enhanced book knowledge
2. Compare to baseline (MATH 2.5% → target 10-15%)
3. Analyze remaining failures (new bottlenecks?)
4. Write completion report

**Success criteria:**
- MATH: >10% accuracy (4x improvement)
- AMC-AIME: >5% accuracy (10x improvement)
- wrong_computation failures reduced by 50-70%

---

## Handoff to Codex

**Codex:** Implement Phase 1 (Core Articulator) with the following priority:

### Priority 1: Theorem Parser

**Focus on Linear Algebra Done Right** (81 templates currently, target 150+ articulated artifacts)

**Example input (from book):**
```latex
\begin{theorem}[Pythagorean Theorem]
Let $\triangle ABC$ be a right triangle with legs $a$ and $b$, and hypotenuse $c$.
Then:
\begin{equation}
a^2 + b^2 = c^2
\end{equation}
\end{theorem}
```

**Expected output:**
```python
KnowledgeArtifact(
    name="Pythagorean Theorem",
    conditions=["triangle is right-angled", "a is leg", "b is leg", "c is hypotenuse"],
    conclusion="a^2 + b^2 = c^2",
    conclusion_rpn="a 2 pow b 2 pow + c 2 pow =",
    symbol_bindings={
        "a": SymbolBinding("a", "leg", "positive_real", ["a > 0"]),
        "b": SymbolBinding("b", "leg", "positive_real", ["b > 0"]),
        "c": SymbolBinding("c", "hypotenuse", "positive_real", ["c > a", "c > b"])
    }
)
```

### Priority 2: Unit Tests

Write regression tests for 10 common theorems:
1. Pythagorean Theorem
2. Determinant formula (2×2, 3×3)
3. Linear transformation definition
4. Eigenvalue definition
5. Dot product formula
6. Cross product formula
7. Matrix multiplication
8. Inverse matrix formula
9. Quadratic formula
10. Distance formula

### Priority 3: Re-Ingest Linear Algebra

Re-run ingestion on `Linear Algebra Done Right` with enhanced articulator, targeting 150+ artifacts.

**Validation:** Spot-check 10 artifacts manually, ensure conditions + conclusions + bindings are correct.

---

**Architect:** Claude (Architecture Partner)
**Strategic Partner:** Gemini (Extended Context)
**Status:** Specification complete, ready for implementation
**Priority:** HIGH - This addresses the root cause of wrong_computation failures
