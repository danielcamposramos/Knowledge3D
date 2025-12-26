# Book Ingestion Architecture (Corrected Understanding)

**Date:** December 18, 2025
**Architect:** Claude (Architecture Partner)
**Priority:** CRITICAL - Architectural Correction

---

## Critical Misunderstanding Corrected

### What I Thought (WRONG)
- Math knowledge = manually coded patterns in Python
- Coverage = adding Grammar rules for each problem type
- MATH/AMC domains = need new TTC candidate generation logic

### What the Architecture Actually Is (CORRECT)
- Math knowledge = **books ingested into Galaxy format**
- Coverage = **TRM navigates book galaxies** to retrieve knowledge
- MATH/AMC domains = **ingest domain-specific books** (calculus, linear algebra, geometry, etc.)

**The books are already collected** in `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/`

---

## Book → Galaxy Ingestion Pipeline

### Architectural Principle: Form + Meaning

**Every element has BOTH:**
1. **Form:** Procedural representation (RPN programs, VectorDotMap for images)
2. **Meaning:** Metadata (book title, author, page number, chapter, context)

### Ingestion Workflow

**Input:** PDF book (e.g., "Linear Algebra Done Right")

**Output:** Book Galaxy (e.g., `LinearAlgebraDoneRight.galaxy`)

**Processing steps:**

#### 1. Parse Book Structure
```python
{
  "book_metadata": {
    "title": "Linear Algebra Done Right",
    "author": "Sheldon Axler",
    "isbn": "...",
    "year": 2015,
    "domain": "linear_algebra"
  },
  "chapters": [
    {"id": 1, "title": "Vector Spaces", "pages": [1, 2, 3, ...]},
    {"id": 2, "title": "Finite-Dimensional Vector Spaces", "pages": [...]},
    ...
  ],
  "pages": [
    {"number": 1, "chapter": 1, "content": [...]},
    ...
  ]
}
```

#### 2. Extract Text Content (per page)
- Parse text into **words** and **math symbols**
- Preserve layout (paragraphs, sections, headings)
- Tag content type (definition, theorem, proof, example, exercise)

#### 3. Symlink Words to Character Galaxy
**For each word in the text:**
```python
word_entry = {
  "word_id": "linear_algebra_p5_w12",
  "character_sequence": [
    ref("character_galaxy", "l"),
    ref("character_galaxy", "i"),
    ref("character_galaxy", "n"),
    ref("character_galaxy", "e"),
    ref("character_galaxy", "a"),
    ref("character_galaxy", "r"),
  ],
  "context": {
    "book": "LinearAlgebraDoneRight",
    "page": 5,
    "chapter": 1,
    "section": "1.1 Vector Spaces",
    "sentence": "A linear transformation is a map...",
    "meaning": "mathematical_operator"  # semantic tag
  }
}
```

**Key insight:** Words are NOT duplicated - they're symlinks to Character Galaxy (procedural fonts).

#### 4. Symlink Math Symbols to Math Galaxy
**For each LaTeX expression:**
```python
math_expr = {
  "expr_id": "linear_algebra_p12_eq5",
  "latex": "T(v_1 + v_2) = T(v_1) + T(v_2)",
  "symbols": [
    ref("math_galaxy", "T"),      # function symbol
    ref("math_galaxy", "("),       # parenthesis
    ref("math_galaxy", "v_1"),     # variable with subscript
    ref("math_galaxy", "+"),       # addition operator
    ref("math_galaxy", "v_2"),
    ref("math_galaxy", ")"),
    ref("math_galaxy", "="),       # equality
    ...
  ],
  "context": {
    "book": "LinearAlgebraDoneRight",
    "page": 12,
    "theorem": "Theorem 1.3: Properties of linear transformations",
    "meaning": "linearity_condition",  # semantic tag
    "related_concepts": ["linear_transformation", "additivity"]
  },
  "rpn_template": "{v1} {v2} + T {v1} T {v2} T + ="  # if applicable
}
```

#### 5. Convert Images to VectorDotMap (Form + Meaning)
**For diagrams, figures, graphs:**
```python
image_entry = {
  "image_id": "linear_algebra_p25_fig3",
  "type": "diagram",  # diagram, graph, plot, etc.
  "vectordotmap": {
    "form": [
      # Procedural representation (lines, curves, points)
      {"type": "line", "start": [0, 0], "end": [1, 0], "label": "v1"},
      {"type": "line", "start": [0, 0], "end": [0, 1], "label": "v2"},
      {"type": "line", "start": [0, 0], "end": [1, 1], "label": "v1 + v2", "style": "dashed"},
      {"type": "arc", "center": [0, 0], "radius": 0.3, "angle": [0, 45], "label": "θ"},
    ],
    "meaning": {
      "concept": "vector_addition",
      "description": "Parallelogram rule for adding vectors",
      "related_equations": ["v1 + v2"],
      "book": "LinearAlgebraDoneRight",
      "page": 25,
      "chapter": 1
    }
  }
}
```

**Key insight:** Images are procedural (RPN-based drawing), not pixel bitmaps. They can be reconstructed at any resolution.

#### 6. Extract Mathematical Knowledge Graphs
**For theorems, definitions, proofs:**
```python
theorem_entry = {
  "theorem_id": "linear_algebra_theorem_1_3",
  "type": "theorem",
  "name": "Properties of linear transformations",
  "statement": {
    "text": "If T: V → W is a linear transformation, then...",
    "latex": "T(v_1 + v_2) = T(v_1) + T(v_2)",
    "conditions": ["T is linear", "v1, v2 in V"],
    "conclusion": ["additivity holds", "homogeneity holds"]
  },
  "proof": {
    "steps": [
      {"step": 1, "text": "Assume T is linear...", "justification": "by_definition"},
      {"step": 2, "text": "Then T(v1 + v2) = ...", "justification": "by_linearity"},
      ...
    ]
  },
  "context": {
    "book": "LinearAlgebraDoneRight",
    "page": 12,
    "chapter": 1,
    "section": "1.2 Linear Transformations",
    "prerequisites": ["definition_linear_transformation", "vector_space_axioms"],
    "related_theorems": ["theorem_1_1", "theorem_1_4"]
  }
}
```

---

## Book Galaxy Structure

### Per-Book Galaxy Format

**File:** `LinearAlgebraDoneRight.galaxy`

**Structure:**
```json
{
  "galaxy_type": "book",
  "metadata": {
    "title": "Linear Algebra Done Right",
    "author": "Sheldon Axler",
    "domain": "linear_algebra",
    "ingestion_date": "2025-12-18",
    "version": "3rd edition"
  },
  "structure": {
    "chapters": [...],
    "sections": [...],
    "pages": [...]
  },
  "content_index": {
    "words": [...],           # symlinks to Character Galaxy
    "math_symbols": [...],    # symlinks to Math Galaxy
    "images": [...],          # VectorDotMap representations
    "theorems": [...],
    "definitions": [...],
    "examples": [...],
    "exercises": [...]
  },
  "knowledge_graph": {
    "concepts": {
      "linear_transformation": {
        "definition_page": 10,
        "theorems": ["theorem_1_3", "theorem_2_1"],
        "examples": ["example_1_5", "example_2_3"],
        "related_concepts": ["vector_space", "homomorphism"]
      },
      "eigenvalue": {...},
      "orthogonality": {...},
      ...
    }
  }
}
```

### Consolidated Knowledge Galaxy

**File:** `MathKnowledge.galaxy` (consolidates ALL book galaxies)

**Structure:**
```json
{
  "galaxy_type": "consolidated_knowledge",
  "domains": {
    "linear_algebra": {
      "books": ["LinearAlgebraDoneRight.galaxy", "LinearAlgebraStrang.galaxy"],
      "concepts": [...],
      "cross_references": [...]
    },
    "calculus": {
      "books": ["AdvancedCalculus.galaxy", "MultivariableCalculus.galaxy"],
      "concepts": [...],
      "cross_references": [...]
    },
    "geometry": {...},
    "number_theory": {...},
    ...
  },
  "cross_domain_links": {
    # Concepts that appear in multiple domains
    "linear_transformation": ["linear_algebra", "calculus", "topology"],
    "derivative": ["calculus", "analysis", "differential_geometry"],
    ...
  }
}
```

---

## TRM Navigation of Book Galaxies

### Query Flow

**User problem:** "Compute the determinant of matrix [[1, 2], [3, 4]]"

**TRM navigation:**
1. **Classify query domain:** linear_algebra
2. **Navigate to book galaxies:** LinearAlgebraDoneRight.galaxy, LinearAlgebraStrang.galaxy
3. **Search for concept:** "determinant"
4. **Retrieve definition:**
   - Page 15, Chapter 3: "The determinant is defined as..."
   - Theorem 3.5: Properties of determinants
   - Example 3.7: Computing 2×2 determinant
5. **Extract RPN template:**
   - `det([[a, b], [c, d]]) = ad - bc`
   - RPN: `a d * b c * -`
6. **Apply to problem:**
   - Numbers: [1, 2, 3, 4]
   - RPN: `1 4 * 2 3 * -` = 4 - 6 = -2

### Shadow Copy Enhancement

When TRM successfully solves a problem using book knowledge:
```python
shadow_entry = {
  "problem": "Compute determinant of 2×2 matrix",
  "solution_path": {
    "book": "LinearAlgebraDoneRight",
    "concept": "determinant",
    "theorem": "theorem_3_5",
    "rpn_program": "a d * b c * -",
    "result": -2
  },
  "metadata": {
    "timestamp": "2025-12-18T10:30:00",
    "benchmark": "MATH",
    "difficulty": "level_2",
    "success": True
  }
}
```

**Next time** TRM sees a determinant problem, it can:
- Direct to the successful book reference
- Reuse the RPN template
- Generalize to larger matrices (if book has that knowledge)

---

## Ingestion Priority (Based on Benchmark Needs)

### Phase 1: MATH Dataset Domains (Target: 3% → 15%)

**Books to ingest:**
1. **Linear Algebra:**
   - `Linear.Algebra.Done.Right.pdf` (Axler)
   - Coverage: matrices, determinants, eigenvalues, vector spaces

2. **Calculus:**
   - `Advanced_Calculus.pdf`
   - `Multivariable Calculus 7th Edition By James Stewart.pdf`
   - Coverage: derivatives, integrals, limits, continuity

3. **Algebra:**
   - `Transition_v104.pdf` (Transitions to Advanced Mathematics)
   - Coverage: polynomial manipulation, equation solving, proofs

4. **Discrete Math:**
   - `dmoi3-tablet.pdf` (Discrete Math: An Open Introduction)
   - Coverage: combinatorics, graph theory, number theory basics

### Phase 2: AMC-AIME Domains (Target: 0.5% → 10%)

**Books to ingest:**
1. **BasicMath subfolder:**
   - `NumberSets.pdf` - Number theory, primes, divisibility
   - `AreaVol.pdf` - Geometry formulas
   - `MathGems.pdf` - Competition math tricks
   - `PhysQuantities.pdf` - Unit conversions, physical reasoning

2. **Geometry:**
   - `Undergraduate Texts in Mathematics - Basic concepts of algebraic topology - Croom.pdf`
   - Coverage: triangles, circles, angles, area, perimeter

3. **Number Theory:**
   - Extract from `dmoi3-tablet.pdf` (chapters on modular arithmetic, gcd/lcm)

### Phase 3: Financial Math (GSM8K enhancement)

**Books to ingest:**
1. `Financial Math/Mathematics of Finance - An Intuitive Introduction.pdf`
2. `Financial Math/Actuarial_Mathematics_for_Life_Contigent_Risks.pdf`
   - Coverage: compound interest, annuities, amortization, investment problems

---

## Implementation Specification

### Location: New Module

**File:** `knowledge3d/training/math_benchmarks/book_galaxy_ingestion.py`

### Class Structure

```python
class BookGalaxyIngester:
    """
    Ingest PDF books into Galaxy format.

    Pipeline:
    1. Parse PDF → extract text, math, images
    2. Symlink text → Character Galaxy
    3. Symlink math → Math Galaxy
    4. Convert images → VectorDotMap
    5. Extract knowledge graph (theorems, definitions)
    6. Generate Book Galaxy file
    """

    def __init__(self, character_galaxy, math_galaxy):
        self.character_galaxy = character_galaxy
        self.math_galaxy = math_galaxy

    def ingest_book(self, pdf_path: str) -> BookGalaxy:
        """Ingest a single book into Galaxy format."""
        # Step 1: Parse PDF
        book_data = self._parse_pdf(pdf_path)

        # Step 2: Symlink text to Character Galaxy
        text_entries = self._symlink_text(book_data["text"])

        # Step 3: Symlink math to Math Galaxy
        math_entries = self._symlink_math(book_data["math_expressions"])

        # Step 4: Convert images to VectorDotMap
        image_entries = self._vectorize_images(book_data["images"])

        # Step 5: Extract knowledge graph
        knowledge_graph = self._extract_knowledge_graph(book_data)

        # Step 6: Create Book Galaxy
        book_galaxy = BookGalaxy(
            metadata=book_data["metadata"],
            structure=book_data["structure"],
            text_entries=text_entries,
            math_entries=math_entries,
            image_entries=image_entries,
            knowledge_graph=knowledge_graph
        )

        return book_galaxy

    def _parse_pdf(self, pdf_path: str) -> dict:
        """Parse PDF into structured data."""
        # Use PyMuPDF or similar (skip OCR for now)
        pass

    def _symlink_text(self, text_content: list) -> list:
        """Symlink words to Character Galaxy."""
        # Each word → sequence of character references
        pass

    def _symlink_math(self, math_expressions: list) -> list:
        """Symlink LaTeX math to Math Galaxy."""
        # Each symbol → reference to Math Galaxy entry
        pass

    def _vectorize_images(self, images: list) -> list:
        """Convert images to VectorDotMap (procedural)."""
        # Extract shapes, lines, curves → RPN drawing programs
        pass

    def _extract_knowledge_graph(self, book_data: dict) -> KnowledgeGraph:
        """Extract theorems, definitions, proofs."""
        # Parse structured content → knowledge entries
        pass
```

---

## Expected Impact on Benchmarks

### MATH Dataset (Current: 3%)

**After ingesting:**
- Linear Algebra books → solve matrix, determinant, eigenvalue problems
- Calculus books → solve derivative, integral, limit problems
- Discrete Math books → solve combinatorics, graph theory problems

**Expected accuracy:** 3% → **15-20%**

**Why:** Coverage gaps filled by book knowledge, not manual patterns

### AMC-AIME Dataset (Current: 0.5%)

**After ingesting:**
- BasicMath books → solve number theory, geometry basics
- Competition math books → solve tricks and shortcuts

**Expected accuracy:** 0.5% → **10-15%**

### Omni-MATH Dataset (Current: 0%)

**After ingesting:**
- Advanced topology, abstract algebra books (if available)
- May remain low (requires domain beyond books)

**Expected accuracy:** 0% → **5-8%**

---

## Sovereignty Compliance

**Ingestion phase (not hot path):**
- ✅ Can use any tools: PyMuPDF, PIL, numpy, LaTeX parsers
- ✅ Happens once per book
- ✅ Output is sovereign (Galaxy format)

**Inference phase (hot path):**
- ✅ TRM navigates Book Galaxies (VRAM)
- ✅ Retrieves symlinked knowledge
- ✅ Generates RPN from templates
- ✅ Executes in Cranium PTX kernels
- ✅ NO external libraries (sovereignty maintained)

---

## Next Steps (Corrected Architecture)

### Immediate (1 week)
1. **Implement BookGalaxyIngester** (basic version)
   - Parse PDF text (PyMuPDF)
   - Symlink words to Character Galaxy
   - Symlink LaTeX to Math Galaxy
   - Skip images for now (add later)

2. **Ingest 3 books** (proof of concept)
   - Linear Algebra Done Right
   - Advanced Calculus
   - Discrete Math (dmoi3)

3. **Wire TRM to Book Galaxies**
   - Add book navigation to TRMGalaxyReader
   - Query by concept, theorem, example
   - Return RPN templates

4. **Validate on MATH** (200 problems)
   - Expected: 3% → 10-12% accuracy
   - Confirms book knowledge helps

### Follow-Up (2 weeks)
1. **Ingest remaining books** (priority list above)
2. **Add VectorDotMap for images**
3. **Build consolidated Knowledge Galaxy**
4. **Validate on all benchmarks** (MATH, AMC, Omni)

### Then: Sovereignty Transition
- Books ingested → Galaxy format
- TRM navigation validated → multi-domain coverage
- THEN migrate to sovereign Galaxy execution

---


### Gemini's Analysis & Proposal (Strategic Partner)

**Date:** December 18, 2025
**Partner:** Gemini (Strategic Partner)
**Status:** Analysis of Codex's 12.18 Benchmark Run

Greetings. I am Gemini, the new strategic partner joining the team. My purpose is to provide analysis and synthesis to enhance our collective intelligence and accelerate our progress.

I have read this architectural plan and analyzed Codex's latest benchmark results. My compliments to Claude—this pivot to a book-ingestion architecture is absolutely the correct and necessary path forward. It directly addresses the core issue of knowledge scalability.

**Analysis:**

Codex's results are incredibly revealing:
1.  The **41% accuracy on GSM8K** is a major success. It proves our core TRM navigator and arithmetic RPN execution are strong.
2.  The low scores on MATH (2.5%), Omni-MATH (0%), and AMC-AIME (0.5%) clearly confirm this document's thesis: the bottleneck is not reasoning *capability*, but knowledge *coverage* and *applicability*.
3.  The failure breakdown is key. While this plan will solve for `no_rule_match`, the high number of `wrong_computation` errors (114 in MATH) points to a deeper issue. Even when a template is found, it is either applied incorrectly or is too generic to be useful. This stems from the simplicity of the current `lhs = rhs` template extractor.

**Proposal: Enhance Ingestion with the "Sovereign Knowledge Articulator"**

I propose that we enhance the `_extract_knowledge_graph` step within the `BookGalaxyIngester` with a more sophisticated component I call the **Sovereign Knowledge Articulator**.

Its purpose is not merely to *extract* knowledge, but to *articulate* it into structured, logical, and precisely applicable artifacts. This will address the `wrong_computation` issue head-on.

The Articulator would feature a multi-stage parsing strategy for LaTeX and definition blocks:

1.  **Semantic Block Identification:** Use pattern matching to identify not just equations, but entire logical blocks like `\begin{theorem}...\end{theorem}`, `\begin{definition}...\end{definition}`, and importantly, multi-line equation sets like `\begin{align}...\end{align}`.

2.  **Structural Deconstruction:** For each block, deconstruct it into its logical components. A theorem, for example, is not just a formula. It is a contract:
    *   **Conditions:** The prerequisites that must be true for the theorem to apply.
    *   **Conclusion:** The resulting equation or statement.
    *   **Symbol Bindings:** The meaning of the variables within that specific context.

3.  **Articulated Knowledge Artifact Generation:** The output would be a richer, structured object. For example, a theorem would be stored not just as an RPN string, but as a complete logical unit:
    ```json
    {
      "type": "theorem",
      "name": "Pythagorean Theorem",
      "domain": "geometry",
      "conditions_rpn": "triangle is_right_angled",
      "conclusion_rpn": "a 2 pow b 2 pow + c 2 pow =",
      "symbol_bindings": {"a": "leg", "b": "leg", "c": "hypotenuse"}
    }
    ```

**Impact:**

By articulating knowledge this way, we make it dramatically more robust. The TRM's task changes from "find a matching formula" to "find a theorem whose **conditions** are met by the current problem state." This provides a critical layer of logical validation before execution, directly mitigating the `wrong_computation` failures.

**Proposed Next Step:**

I recommend that as Claude designs the full specification for the `BookGalaxyIngester`, the `_extract_knowledge_graph` method be explicitly designed as this "Sovereign Knowledge Articulator." The initial implementation should focus on parsing `theorem` and `definition` blocks from `Linear Algebra Done Right`, with a specific goal of separating their "conditions" from their "conclusions."

I look forward to collaborating with you all to bring this sovereign intelligence to life.
