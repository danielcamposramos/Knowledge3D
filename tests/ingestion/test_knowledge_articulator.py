from __future__ import annotations


def test_articulator_extracts_theorem_block_and_conditions():
    from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

    page_text = """
Theorem (Pythagorean Theorem)
Let triangle ABC be a right triangle with legs a and b and hypotenuse c.
a^2 + b^2 = c^2
"""

    articulator = SovereignKnowledgeArticulator()
    artifacts = articulator.articulate_pages(pages=[(25, page_text)], book_id="demo_book", domain="geometry")
    assert artifacts

    art = artifacts[0]
    assert art.artifact_type in {"theorem", "formula"}
    assert art.lhs and art.rhs
    assert art.lhs_rpn and art.rhs_rpn
    assert any("triangle is right-angled" in c.lower() for c in art.conditions)


def test_articulator_derives_sqrt_candidate_from_squared_variable():
    from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

    page_text = """
Theorem (Pythagorean Theorem)
Let triangle ABC be a right triangle.
a^2 + b^2 = c^2
"""

    articulator = SovereignKnowledgeArticulator()
    artifacts = articulator.articulate_pages(pages=[(1, page_text)], book_id="demo_book", domain="geometry")
    assert artifacts

    art = artifacts[0]
    derived = list(art.derived_rpns or [])
    assert derived
    assert any(d.get("kind") == "solve_sqrt" for d in derived)
    assert any("sqrt" in str(d.get("rpn", "")) for d in derived)

def test_articulator_infers_pythagorean_roles_from_equation_structure():
    from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

    # Prose does not declare variable -> role; inference must use equation structure
    # plus "right triangle" context.
    page_text = """
Theorem (Pythagorean Theorem)
In a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.
a^2 + b^2 = c^2
"""

    articulator = SovereignKnowledgeArticulator()
    artifacts = articulator.articulate_pages(pages=[(1, page_text)], book_id="demo_book", domain="geometry")
    assert artifacts

    art = artifacts[0]
    bindings = dict(art.symbol_bindings or {})
    assert bindings.get("a", {}).get("meaning") == "leg"
    assert bindings.get("b", {}).get("meaning") == "leg"
    assert bindings.get("c", {}).get("meaning") == "hypotenuse"


def test_articulator_parses_latex_theorem_environment_and_bindings():
    from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

    page_text = r"""
\begin{theorem}[Pythagorean Theorem]
Let \triangle ABC be a right triangle with legs $a$ and $b$, and hypotenuse $c$.
\begin{equation}
a^2 + b^2 = c^2
\end{equation}
\end{theorem}
"""

    articulator = SovereignKnowledgeArticulator()
    artifacts = articulator.articulate_pages(pages=[(25, page_text)], book_id="demo_book", domain="geometry")
    assert artifacts

    art = artifacts[0]
    assert art.artifact_type == "theorem"
    assert "pythagorean" in art.name.lower()
    assert any("right-angled" in c.lower() for c in art.conditions)

    bindings = dict(art.symbol_bindings or {})
    assert "a" in bindings and bindings["a"].get("meaning") == "leg"
    assert "b" in bindings and bindings["b"].get("meaning") == "leg"
    assert "c" in bindings and bindings["c"].get("meaning") == "hypotenuse"


def test_articulator_emits_definition_without_equation():
    from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

    page_text = """
Definition (Vector Space)
Let V be a set with addition and scalar multiplication.
A vector space satisfies closure and associativity.
"""

    articulator = SovereignKnowledgeArticulator()
    artifacts = articulator.articulate_pages(pages=[(3, page_text)], book_id="demo_book", domain="linear_algebra")
    assert artifacts

    art = artifacts[0]
    assert art.artifact_type == "definition"
    assert art.rpn is None
    assert isinstance(art.conclusion, str) and art.conclusion


def test_articulator_emits_loose_pi_formula_and_infers_radius_role():
    from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

    page_text = """
We can compute the area of a circle using A = pi*r^2.
"""
    articulator = SovereignKnowledgeArticulator()
    artifacts = articulator.articulate_pages(pages=[(1, page_text)], book_id="demo_book", domain="geometry")
    assert artifacts

    # Expect a formula artifact to be emitted from a loose equation line.
    formula = next((a for a in artifacts if a.artifact_type == "formula"), None)
    assert formula is not None
    bindings = dict(formula.symbol_bindings or {})
    # r should be inferred as radius from the π * r^2 structure.
    assert "r" in bindings
    assert bindings["r"].get("meaning") == "radius"


def test_articulator_parses_common_math_formulas():
    from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

    cases = [
        (
            "Pythagorean Theorem",
            "Let triangle ABC be a right triangle with legs a and b and hypotenuse c.",
            "a^2 + b^2 = c^2",
        ),
        (
            "Determinant 2x2",
            "Let A be a 2x2 matrix with entries a,b,c,d.",
            "det = a*d - b*c",
        ),
        (
            "Linear Transformation",
            "Let T be a linear transformation.",
            "T(x+y) = T(x) + T(y)",
        ),
        (
            "Eigenvalue Definition",
            "Let v be a nonzero vector.",
            "A*v = lambda*v",
        ),
        (
            "Dot Product",
            "Let a and b be vectors.",
            "dot = ax*bx + ay*by + az*bz",
        ),
        (
            "Cross Product Magnitude",
            "Let theta be the angle between vectors.",
            "cross = a*b*sin(theta)",
        ),
        (
            "Matrix Multiplication Entry",
            "Let A and B be 2x2 matrices.",
            "c11 = a11*b11 + a12*b21",
        ),
        (
            "Inverse Scalar",
            "Let det be nonzero.",
            "inv = 1/det",
        ),
        (
            "Quadratic Formula",
            "Let a, b, c be real numbers with a != 0.",
            "x = (sqrt(b^2 - 4*a*c) - b) / (2*a)",
        ),
        (
            "Distance Formula",
            "Let (x1,y1) and (x2,y2) be points.",
            "d = sqrt((x2-x1)^2 + (y2-y1)^2)",
        ),
    ]

    articulator = SovereignKnowledgeArticulator()
    for idx, (name, preface, equation) in enumerate(cases):
        page_text = f"Theorem ({name})\n{preface}\n{equation}\n"
        artifacts = articulator.articulate_pages(pages=[(idx + 1, page_text)], book_id="demo_book", domain="math")
        assert artifacts, name
        art = artifacts[0]
        assert art.lhs and art.rhs and art.rpn, name
        assert isinstance(art.conclusion, str) and "=" in art.conclusion, name


def test_articulator_handles_pdftotext_control_char_equals():
    from knowledge3d.training.math_benchmarks.sovereign_knowledge_articulator import SovereignKnowledgeArticulator

    # Some pdftotext exports emit "\x03" in place of "=".
    page_text = "Theorem (Euler)\nv - e + f \x03 2\n"
    articulator = SovereignKnowledgeArticulator()
    artifacts = articulator.articulate_pages(pages=[(1, page_text)], book_id="demo_book", domain="graph_theory")
    assert artifacts
    assert artifacts[0].lhs and artifacts[0].rhs
