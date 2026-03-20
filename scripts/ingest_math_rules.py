#!/usr/bin/env python3
"""Populate the Math galaxy with general rules, formulas, and template anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse  # noqa: E402


DEFAULT_STORAGE_ROOT = Path("/K3D/Knowledge3D.local")
MATH_TYPES = (
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
)


def _slug(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "entry"


def _dedup_strs(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        resolved = str(value or "").strip()
        if not resolved:
            continue
        lowered = resolved.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(resolved)
    return result


def _rule_spec(
    *,
    ident: str,
    name: str,
    statement: str,
    math_type: str,
    subfield: str,
    aliases: list[str],
    keywords: list[str],
    rpn_program: str = "",
    semantics: str = "",
) -> dict[str, Any]:
    return {
        "id": ident,
        "name": name,
        "statement": statement,
        "math_type": math_type,
        "subfield": subfield,
        "aliases": _dedup_strs(aliases),
        "keywords": _dedup_strs(keywords),
        "rpn_program": str(rpn_program or "").strip(),
        "semantics": str(semantics or statement).strip(),
    }


def _base_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "math_type": spec["math_type"],
        "subject": "mathematics",
        "subfield": spec["subfield"],
        "aliases": list(spec["aliases"]),
        "keywords": list(spec["keywords"]),
        "tags": list(spec["keywords"]),
        "semantics": spec["semantics"],
        "confidence": 0.91,
        "ingest_source": "ingest_math_rules",
    }


def _formula_fact_entry(spec: dict[str, Any]) -> dict[str, Any]:
    metadata = _base_metadata(spec)
    metadata["query_anchor"] = f"{spec['name']} {spec['statement']}"
    return {
        "id": f"math_formula_{spec['id']}",
        "name": spec["name"],
        "domain": "math",
        "category": "formula_fact",
        "content": spec["statement"],
        "summary": spec["statement"],
        "description": f"{spec['math_type']} formula or identity.",
        "rpn_program": "",
        "answer_text": "",
        "metadata": metadata,
    }


def _symbolic_rule_entry(spec: dict[str, Any]) -> dict[str, Any]:
    metadata = _base_metadata(spec)
    metadata["query_anchor"] = f"{spec['name']} {' '.join(spec['keywords'])}"
    metadata["direct_eval"] = False
    metadata["rule_ref"] = spec["id"]
    return {
        "id": f"math_rule_{spec['id']}",
        "name": f"{spec['name']} rule",
        "domain": "math",
        "category": "math_rule",
        "content": spec["statement"],
        "summary": spec["name"],
        "description": f"General {spec['math_type']} execution rule.",
        "rpn_program": spec["rpn_program"],
        "answer_text": "",
        "metadata": metadata,
    }


def _concept_anchor_entry(spec: dict[str, Any]) -> dict[str, Any]:
    metadata = _base_metadata(spec)
    metadata["query_anchor"] = f"{spec['name']} concept {' '.join(spec['aliases'])}"
    metadata["direct_eval"] = False
    metadata["rule_ref"] = spec["id"]
    return {
        "id": f"math_anchor_{spec['id']}",
        "name": f"{spec['name']} concept anchor",
        "domain": "math",
        "category": "rule",
        "content": f"{spec['name']} is a reusable math concept for {spec['math_type'].lower()} tasks.",
        "summary": spec["name"],
        "description": spec["statement"],
        "rpn_program": spec["rpn_program"],
        "answer_text": "",
        "metadata": metadata,
    }


def _problem_anchor_entry(spec: dict[str, Any]) -> dict[str, Any]:
    metadata = _base_metadata(spec)
    metadata["query_anchor"] = (
        f"Use {spec['name']} to solve {' '.join(spec['keywords'])} problems in {spec['math_type']}"
    )
    metadata["direct_eval"] = False
    metadata["rule_ref"] = spec["id"]
    return {
        "id": f"math_problem_anchor_{spec['id']}",
        "name": f"{spec['name']} problem anchor",
        "domain": "math",
        "category": "rule",
        "content": f"Problem-solving anchor for {spec['name']}.",
        "summary": spec["name"],
        "description": f"Retrieve {spec['name']} for unseen {spec['math_type'].lower()} problems.",
        "rpn_program": spec["rpn_program"],
        "answer_text": "",
        "metadata": metadata,
    }


def _build_exponent_rules() -> list[dict[str, Any]]:
    items = [
        ("product_of_powers", "Product of powers", "x^a * x^b = x^(a+b)", "ARG_EXP_A ARG_EXP_B +"),
        ("quotient_of_powers", "Quotient of powers", "x^a / x^b = x^(a-b)", "ARG_EXP_A ARG_EXP_B -"),
        ("power_of_a_power", "Power of a power", "(x^a)^b = x^(ab)", "ARG_EXP_A ARG_EXP_B *"),
        ("power_of_a_product", "Power of a product", "(ab)^n = a^n b^n", "ARG_A ARG_N pow ARG_B ARG_N pow *"),
        ("power_of_a_quotient", "Power of a quotient", "(a/b)^n = a^n / b^n", "ARG_A ARG_N pow ARG_B ARG_N pow /"),
        ("zero_exponent", "Zero exponent rule", "x^0 = 1 for x != 0", ""),
        ("negative_exponent", "Negative exponent rule", "x^(-n) = 1 / x^n", "1 ARG_X ARG_N pow /"),
        ("fractional_exponent", "Fractional exponent rule", "x^(1/n) = n-th root of x", "ARG_X ARG_N / pow"),
    ]
    return [
        _rule_spec(
            ident=f"exponent_{ident}",
            name=name,
            statement=statement,
            math_type="Algebra",
            subfield="exponents",
            aliases=[name.lower(), "exponent rule", ident.replace("_", " ")],
            keywords=["exponent", "powers", *ident.split("_")],
            rpn_program=program,
            semantics=f"Apply {name.lower()} in exponent simplification.",
        )
        for ident, name, statement, program in items
    ]


def _build_log_rules() -> list[dict[str, Any]]:
    items = [
        ("product", "Log product rule", "log_b(xy) = log_b(x) + log_b(y)", "ARG_LOG_X ARG_LOG_Y +"),
        ("quotient", "Log quotient rule", "log_b(x/y) = log_b(x) - log_b(y)", "ARG_LOG_X ARG_LOG_Y -"),
        ("power", "Log power rule", "log_b(x^n) = n log_b(x)", "ARG_N ARG_LOG_X *"),
        ("change_base", "Change of base formula", "log_b(x) = log_k(x) / log_k(b)", "ARG_LOG_X ARG_LOG_B /"),
        ("inverse_exp", "Logarithm-exponential inverse", "log_b(b^x) = x", ""),
        ("inverse_exp_reverse", "Exponential-logarithm inverse", "b^(log_b(x)) = x", ""),
        ("one_rule", "Logarithm of one", "log_b(1) = 0", ""),
        ("base_rule", "Logarithm of the base", "log_b(b) = 1", ""),
    ]
    return [
        _rule_spec(
            ident=f"log_{ident}",
            name=name,
            statement=statement,
            math_type="Intermediate Algebra",
            subfield="logarithms",
            aliases=[name.lower(), "log rule", ident.replace("_", " ")],
            keywords=["logarithm", "log", *ident.split("_")],
            rpn_program=program,
            semantics=f"Apply {name.lower()} during logarithm manipulation.",
        )
        for ident, name, statement, program in items
    ]


def _build_factoring_rules() -> list[dict[str, Any]]:
    items = [
        ("difference_squares", "Difference of squares", "a^2 - b^2 = (a-b)(a+b)", ""),
        ("perfect_square_plus", "Perfect square trinomial (plus)", "a^2 + 2ab + b^2 = (a+b)^2", ""),
        ("perfect_square_minus", "Perfect square trinomial (minus)", "a^2 - 2ab + b^2 = (a-b)^2", ""),
        ("sum_cubes", "Sum of cubes", "a^3 + b^3 = (a+b)(a^2-ab+b^2)", ""),
        ("difference_cubes", "Difference of cubes", "a^3 - b^3 = (a-b)(a^2+ab+b^2)", ""),
        ("quadratic_monic", "Monic quadratic factoring", "x^2 + bx + c = (x+m)(x+n) when m+n=b and mn=c", ""),
        ("grouping", "Factoring by grouping", "ab + ac + db + dc = (a+d)(b+c)", ""),
        ("gcf", "Greatest common factor factoring", "ax + ay = a(x+y)", ""),
        ("expand_binomial_square", "Expand binomial square", "(a+b)^2 = a^2 + 2ab + b^2", ""),
        ("expand_binomial_difference", "Expand conjugates", "(a+b)(a-b) = a^2 - b^2", ""),
    ]
    return [
        _rule_spec(
            ident=f"factor_{ident}",
            name=name,
            statement=statement,
            math_type="Algebra",
            subfield="factoring",
            aliases=[name.lower(), "factoring", ident.replace("_", " ")],
            keywords=["factor", "factoring", *ident.split("_")],
            semantics=f"Use {name.lower()} while factoring or expanding expressions.",
        )
        for ident, name, statement, _program in items
    ]


def _build_equation_rules() -> list[dict[str, Any]]:
    items = [
        ("complete_square", "Completing the square", "x^2 + bx = (x + b/2)^2 - (b/2)^2", ""),
        ("quadratic_formula", "Quadratic formula", "x = (-b ± sqrt(b^2 - 4ac)) / (2a)", "0 ARG_B - ARG_B ARG_B * 4 ARG_A * ARG_C * - sqrt + 2 ARG_A * /"),
        ("slope_intercept", "Slope-intercept line form", "y = mx + b", "ARG_M ARG_X * ARG_B +"),
        ("point_slope", "Point-slope line form", "y - y1 = m(x - x1)", ""),
        ("distance_formula", "Distance formula", "d = sqrt((x2-x1)^2 + (y2-y1)^2)", "ARG_X2 ARG_X1 - 2 pow ARG_Y2 ARG_Y1 - 2 pow + sqrt"),
        ("midpoint_formula", "Midpoint formula", "M = ((x1+x2)/2, (y1+y2)/2)", ""),
        ("absolute_value_split", "Absolute value equation split", "|u| = c implies u = c or u = -c", ""),
        ("substitution_method", "System solving by substitution", "Substitute one solved variable into the other equation", ""),
        ("elimination_method", "System solving by elimination", "Add or subtract equations to eliminate one variable", ""),
        ("inequality_flip", "Inequality sign flip rule", "Multiplying or dividing an inequality by a negative flips the sign", ""),
    ]
    return [
        _rule_spec(
            ident=f"equation_{ident}",
            name=name,
            statement=statement,
            math_type="Intermediate Algebra",
            subfield="equations",
            aliases=[name.lower(), ident.replace("_", " "), "equation strategy"],
            keywords=["equation", "algebra", *ident.split("_")],
            rpn_program=program,
            semantics=f"Apply {name.lower()} to manipulate equations or inequalities.",
        )
        for ident, name, statement, program in items
    ]


def _build_counting_probability_rules() -> list[dict[str, Any]]:
    items = [
        ("factorial", "Factorial definition", "n! = n(n-1)(n-2)...1", "ARG_N factorial"),
        ("permutation", "Permutation formula", "P(n,r) = n!/(n-r)!", "ARG_N factorial ARG_N ARG_R - factorial /"),
        ("combination", "Combination formula", "C(n,r) = n!/(r!(n-r)!)", "ARG_N ARG_R binom"),
        ("binomial_theorem", "Binomial theorem", "(a+b)^n = sum_{k=0}^n C(n,k) a^(n-k) b^k", ""),
        ("complement", "Probability complement rule", "P(A^c) = 1 - P(A)", "1 ARG_P -"),
        ("addition_disjoint", "Disjoint addition rule", "P(A or B) = P(A) + P(B) for disjoint events", "ARG_P_A ARG_P_B +"),
        ("addition_general", "General addition rule", "P(A or B) = P(A) + P(B) - P(A and B)", "ARG_P_A ARG_P_B + ARG_P_INTERSECTION -"),
        ("multiplication_independent", "Independent multiplication rule", "P(A and B) = P(A)P(B) for independent events", "ARG_P_A ARG_P_B *"),
        ("conditional", "Conditional probability", "P(A|B) = P(A and B) / P(B)", "ARG_P_INTERSECTION ARG_P_B /"),
        ("expected_value", "Expected value", "E[X] = sum x_i p_i", ""),
        ("variance", "Variance formula", "Var(X) = E[X^2] - (E[X])^2", ""),
        ("pigeonhole", "Pigeonhole principle", "If more than n objects are placed into n boxes, some box has at least two objects", ""),
        ("inclusion_exclusion_two", "Inclusion-exclusion for two sets", "|A union B| = |A| + |B| - |A intersection B|", "ARG_A ARG_B + ARG_INTERSECTION -"),
        ("inclusion_exclusion_three", "Inclusion-exclusion for three sets", "|A union B union C| = sum singles - sum pairs + triple overlap", ""),
    ]
    return [
        _rule_spec(
            ident=f"counting_{ident}",
            name=name,
            statement=statement,
            math_type="Counting & Probability",
            subfield="counting_probability",
            aliases=[name.lower(), ident.replace("_", " "), "probability rule"],
            keywords=["counting", "probability", *ident.split("_")],
            rpn_program=program,
            semantics=f"Use {name.lower()} for counting or probability reasoning.",
        )
        for ident, name, statement, program in items
    ]


def _build_geometry_formula_rules() -> list[dict[str, Any]]:
    area_shapes = [
        ("triangle", "Area of a triangle", "A = bh/2", "ARG_BASE ARG_HEIGHT * 2 /"),
        ("rectangle", "Area of a rectangle", "A = lw", "ARG_LENGTH ARG_WIDTH *"),
        ("square", "Area of a square", "A = s^2", "ARG_SIDE 2 pow"),
        ("circle", "Area of a circle", "A = pi r^2", "pi ARG_RADIUS 2 pow *"),
        ("trapezoid", "Area of a trapezoid", "A = (b1+b2)h/2", "ARG_B1 ARG_B2 + ARG_HEIGHT * 2 /"),
        ("parallelogram", "Area of a parallelogram", "A = bh", "ARG_BASE ARG_HEIGHT *"),
        ("rhombus", "Area of a rhombus", "A = d1 d2 / 2", "ARG_D1 ARG_D2 * 2 /"),
        ("regular_polygon", "Area of a regular polygon", "A = ap/2", "ARG_APOTHEM ARG_PERIMETER * 2 /"),
        ("sector", "Area of a sector", "A = theta/360 * pi r^2", "ARG_THETA 360 / pi ARG_RADIUS 2 pow * *"),
        ("annulus", "Area of an annulus", "A = pi(R^2-r^2)", "pi ARG_R_OUTER 2 pow ARG_R_INNER 2 pow - *"),
    ]
    volume_shapes = [
        ("cube", "Volume of a cube", "V = s^3", "ARG_SIDE 3 pow"),
        ("rectangular_prism", "Volume of a rectangular prism", "V = lwh", "ARG_LENGTH ARG_WIDTH * ARG_HEIGHT *"),
        ("cylinder", "Volume of a cylinder", "V = pi r^2 h", "pi ARG_RADIUS 2 pow * ARG_HEIGHT *"),
        ("cone", "Volume of a cone", "V = pi r^2 h / 3", "pi ARG_RADIUS 2 pow * ARG_HEIGHT * 3 /"),
        ("sphere", "Volume of a sphere", "V = 4/3 pi r^3", "4 3 / pi * ARG_RADIUS 3 pow *"),
        ("pyramid", "Volume of a pyramid", "V = Bh/3", "ARG_BASE_AREA ARG_HEIGHT * 3 /"),
        ("triangular_prism", "Volume of a triangular prism", "V = (bh/2)L", "ARG_BASE ARG_HEIGHT * 2 / ARG_LENGTH *"),
        ("frustum", "Volume of a frustum", "V = h(A1 + A2 + sqrt(A1A2))/3", ""),
    ]
    specs: list[dict[str, Any]] = []
    for ident, name, statement, program in area_shapes + volume_shapes:
        specs.append(
            _rule_spec(
                ident=f"geometry_{ident}",
                name=name,
                statement=statement,
                math_type="Geometry",
                subfield="geometry_formulas",
                aliases=[name.lower(), ident.replace("_", " "), "geometry formula"],
                keywords=["geometry", *ident.split("_")],
                rpn_program=program,
                semantics=f"Use {name.lower()} in geometry calculation problems.",
            )
        )
    return specs


def _build_geometry_theorem_rules() -> list[dict[str, Any]]:
    items = [
        ("pythagorean", "Pythagorean theorem", "a^2 + b^2 = c^2", ""),
        ("triangle_sum", "Triangle angle sum", "The interior angles of a triangle sum to 180 degrees", ""),
        ("supplementary", "Supplementary angles", "Supplementary angles sum to 180 degrees", "180 ARG_ANGLE -"),
        ("complementary", "Complementary angles", "Complementary angles sum to 90 degrees", "90 ARG_ANGLE -"),
        ("vertical", "Vertical angles", "Vertical angles are congruent", ""),
        ("parallel_alternate", "Alternate interior angles", "Alternate interior angles are equal for parallel lines", ""),
        ("similar_triangles", "Similar triangles proportionality", "Corresponding sides of similar triangles are proportional", ""),
        ("congruence_sss", "Triangle congruence SSS", "Three equal sides imply triangle congruence", ""),
        ("circle_inscribed", "Inscribed angle theorem", "An inscribed angle measures half its intercepted arc", "ARG_ARC 2 /"),
        ("circle_tangent_radius", "Tangent-radius theorem", "A radius to a point of tangency is perpendicular to the tangent", ""),
        ("power_of_point", "Power of a point", "External secant and tangent lengths satisfy the power-of-a-point relation", ""),
        ("arc_length", "Arc length formula", "s = r theta (radians)", "ARG_RADIUS ARG_THETA *"),
    ]
    return [
        _rule_spec(
            ident=f"geometry_theorem_{ident}",
            name=name,
            statement=statement,
            math_type="Geometry",
            subfield="geometry_theorems",
            aliases=[name.lower(), ident.replace("_", " "), "geometry theorem"],
            keywords=["geometry", "theorem", *ident.split("_")],
            rpn_program=program,
            semantics=f"Apply {name.lower()} in geometry proofs or calculations.",
        )
        for ident, name, statement, program in items
    ]


def _build_coordinate_rules() -> list[dict[str, Any]]:
    items = [
        ("slope", "Slope formula", "m = (y2 - y1)/(x2 - x1)", "ARG_Y2 ARG_Y1 - ARG_X2 ARG_X1 - /"),
        ("midpoint", "Midpoint formula", "M = ((x1+x2)/2, (y1+y2)/2)", ""),
        ("distance", "Coordinate distance formula", "d = sqrt((x2-x1)^2 + (y2-y1)^2)", "ARG_X2 ARG_X1 - 2 pow ARG_Y2 ARG_Y1 - 2 pow + sqrt"),
        ("line_standard", "Standard line form", "Ax + By = C", ""),
        ("line_intercept", "Intercept form", "x/a + y/b = 1", ""),
        ("circle_standard", "Standard circle form", "(x-h)^2 + (y-k)^2 = r^2", ""),
        ("parabola_vertex", "Parabola vertex form", "y = a(x-h)^2 + k", ""),
        ("ellipse_standard", "Standard ellipse form", "(x-h)^2/a^2 + (y-k)^2/b^2 = 1", ""),
        ("hyperbola_standard", "Standard hyperbola form", "(x-h)^2/a^2 - (y-k)^2/b^2 = 1", ""),
        ("section_formula", "Section formula", "A point dividing a segment in ratio m:n can be found by weighted coordinates", ""),
    ]
    return [
        _rule_spec(
            ident=f"coordinate_{ident}",
            name=name,
            statement=statement,
            math_type="Geometry",
            subfield="coordinate_geometry",
            aliases=[name.lower(), ident.replace("_", " "), "coordinate geometry"],
            keywords=["coordinate", "geometry", *ident.split("_")],
            rpn_program=program,
            semantics=f"Use {name.lower()} in coordinate-geometry problems.",
        )
        for ident, name, statement, program in items
    ]


def _build_number_theory_rules() -> list[dict[str, Any]]:
    items = [
        ("divisible_2", "Divisibility rule for 2", "An integer is divisible by 2 when its last digit is even", ""),
        ("divisible_3", "Divisibility rule for 3", "An integer is divisible by 3 when its digit sum is divisible by 3", ""),
        ("divisible_4", "Divisibility rule for 4", "An integer is divisible by 4 when its last two digits form a multiple of 4", ""),
        ("divisible_5", "Divisibility rule for 5", "An integer is divisible by 5 when it ends in 0 or 5", ""),
        ("divisible_6", "Divisibility rule for 6", "An integer is divisible by 6 when it is divisible by both 2 and 3", ""),
        ("divisible_8", "Divisibility rule for 8", "An integer is divisible by 8 when its last three digits form a multiple of 8", ""),
        ("divisible_9", "Divisibility rule for 9", "An integer is divisible by 9 when its digit sum is divisible by 9", ""),
        ("divisible_11", "Divisibility rule for 11", "An integer is divisible by 11 when the alternating digit sum is divisible by 11", ""),
        ("gcd", "Greatest common divisor", "gcd(a,b) is the largest integer dividing both a and b", "ARG_A ARG_B gcd"),
        ("lcm", "Least common multiple", "lcm(a,b) = |ab| / gcd(a,b)", "ARG_A ARG_B * ARG_A ARG_B gcd / abs"),
        ("euclidean", "Euclidean algorithm", "Repeated remainder reduction computes the gcd", ""),
        ("prime_factorization", "Prime factorization", "Every integer greater than 1 factors uniquely into primes", ""),
        ("mod_add", "Modular addition", "(a+b) mod n = ((a mod n) + (b mod n)) mod n", ""),
        ("mod_mul", "Modular multiplication", "(ab) mod n = ((a mod n)(b mod n)) mod n", ""),
        ("fermat_little", "Fermat's little theorem", "If p is prime and gcd(a,p)=1 then a^(p-1) = 1 mod p", ""),
        ("totient", "Euler totient function", "phi(n) counts integers up to n that are coprime with n", ""),
    ]
    return [
        _rule_spec(
            ident=f"number_theory_{ident}",
            name=name,
            statement=statement,
            math_type="Number Theory",
            subfield="number_theory",
            aliases=[name.lower(), ident.replace("_", " "), "number theory"],
            keywords=["number", "theory", *ident.split("_")],
            rpn_program=program,
            semantics=f"Use {name.lower()} for number-theory reasoning.",
        )
        for ident, name, statement, program in items
    ]


def _build_sequence_series_rules() -> list[dict[str, Any]]:
    items = [
        ("arithmetic_nth", "Arithmetic sequence nth term", "a_n = a_1 + (n-1)d", "ARG_N 1 - ARG_D * ARG_A1 +"),
        ("arithmetic_sum", "Arithmetic sequence sum", "S_n = n(a_1 + a_n)/2", "ARG_N ARG_A1 ARG_AN + * 2 /"),
        ("geometric_nth", "Geometric sequence nth term", "a_n = a_1 r^(n-1)", "ARG_A1 ARG_R ARG_N 1 - pow *"),
        ("geometric_sum", "Geometric series sum", "S_n = a(1-r^n)/(1-r)", "ARG_A 1 ARG_R ARG_N pow - * 1 ARG_R - /"),
        ("sigma_linear", "Linear sigma sum", "sum_{k=1}^n k = n(n+1)/2", "ARG_N ARG_N 1 + * 2 /"),
        ("sigma_square", "Square sigma sum", "sum_{k=1}^n k^2 = n(n+1)(2n+1)/6", "ARG_N ARG_N 1 + * 2 ARG_N * 1 + * 6 /"),
        ("binomial_sequence", "Binomial recursive sequence", "Pascal-triangle recurrences build binomial coefficients", ""),
        ("recursive_linear", "Linear recurrence update", "a_n = a_(n-1) + d defines an arithmetic progression recursively", ""),
        ("recursive_geometric", "Geometric recurrence update", "a_n = r a_(n-1) defines a geometric progression recursively", ""),
        ("finite_difference", "Finite differences", "Constant first difference implies an arithmetic sequence", ""),
        ("complex_modulus", "Complex modulus", "|a+bi| = sqrt(a^2+b^2)", "ARG_A 2 pow ARG_B 2 pow + sqrt"),
        ("de_moivre", "De Moivre's theorem", "(cos x + i sin x)^n = cos(nx) + i sin(nx)", ""),
    ]
    return [
        _rule_spec(
            ident=f"precalc_{ident}",
            name=name,
            statement=statement,
            math_type="Precalculus",
            subfield="sequences_series_complex",
            aliases=[name.lower(), ident.replace("_", " "), "precalculus"],
            keywords=["sequence", "series", "precalculus", *ident.split("_")],
            rpn_program=program,
            semantics=f"Use {name.lower()} in sequence, series, or complex-number reasoning.",
        )
        for ident, name, statement, program in items
    ]


def _build_trig_rules() -> list[dict[str, Any]]:
    items = [
        ("pythagorean", "Pythagorean trig identity", "sin^2(x) + cos^2(x) = 1", ""),
        ("tan_ratio", "Tangent ratio identity", "tan(x) = sin(x)/cos(x)", ""),
        ("sec_relation", "Secant relation", "sec^2(x) = 1 + tan^2(x)", ""),
        ("csc_relation", "Cosecant relation", "csc^2(x) = 1 + cot^2(x)", ""),
        ("sin_sum", "Sine sum formula", "sin(a+b) = sin(a)cos(b) + cos(a)sin(b)", ""),
        ("cos_sum", "Cosine sum formula", "cos(a+b) = cos(a)cos(b) - sin(a)sin(b)", ""),
        ("sin_diff", "Sine difference formula", "sin(a-b) = sin(a)cos(b) - cos(a)sin(b)", ""),
        ("cos_diff", "Cosine difference formula", "cos(a-b) = cos(a)cos(b) + sin(a)sin(b)", ""),
        ("sin_double", "Sine double-angle formula", "sin(2x) = 2sin(x)cos(x)", ""),
        ("cos_double", "Cosine double-angle formula", "cos(2x) = cos^2(x) - sin^2(x)", ""),
        ("tan_double", "Tangent double-angle formula", "tan(2x) = 2tan(x)/(1-tan^2(x))", ""),
        ("sin_half", "Sine half-angle formula", "sin^2(x/2) = (1-cos x)/2", ""),
        ("cos_half", "Cosine half-angle formula", "cos^2(x/2) = (1+cos x)/2", ""),
        ("law_sines", "Law of sines", "a/sin(A) = b/sin(B) = c/sin(C)", ""),
        ("law_cosines", "Law of cosines", "c^2 = a^2 + b^2 - 2ab cos(C)", ""),
    ]
    return [
        _rule_spec(
            ident=f"trig_{ident}",
            name=name,
            statement=statement,
            math_type="Precalculus",
            subfield="trigonometry",
            aliases=[name.lower(), ident.replace("_", " "), "trigonometry"],
            keywords=["trig", "trigonometry", *ident.split("_")],
            semantics=f"Use {name.lower()} in trigonometric simplification and solving.",
        )
        for ident, name, statement, _program in items
    ]


def _build_matrix_vector_rules() -> list[dict[str, Any]]:
    items = [
        ("matrix_add", "Matrix addition", "(A+B)_{ij} = A_{ij} + B_{ij}", ""),
        ("matrix_mul", "Matrix multiplication", "(AB)_{ij} = sum_k A_{ik} B_{kj}", ""),
        ("det_2x2", "2x2 determinant", "det([[a,b],[c,d]]) = ad - bc", "ARG_A ARG_D * ARG_B ARG_C * -"),
        ("inverse_2x2", "2x2 matrix inverse", "A^-1 = 1/det(A) [[d,-b],[-c,a]]", ""),
        ("vector_dot", "Dot product", "u dot v = u1v1 + u2v2 + u3v3", ""),
        ("vector_cross", "Cross product", "u x v is perpendicular to both u and v", ""),
        ("vector_magnitude", "Vector magnitude", "|v| = sqrt(v1^2 + v2^2 + v3^2)", ""),
        ("projection", "Vector projection", "proj_u(v) = (v dot u)/(u dot u) u", ""),
        ("complex_add", "Complex addition", "(a+bi) + (c+di) = (a+c) + (b+d)i", ""),
        ("complex_mul", "Complex multiplication", "(a+bi)(c+di) = (ac-bd) + (ad+bc)i", ""),
        ("complex_conjugate", "Complex conjugate", "conj(a+bi) = a-bi", ""),
        ("polar_form", "Complex polar form", "z = r(cos theta + i sin theta)", ""),
    ]
    return [
        _rule_spec(
            ident=f"linear_algebra_{ident}",
            name=name,
            statement=statement,
            math_type="Precalculus",
            subfield="matrices_vectors_complex",
            aliases=[name.lower(), ident.replace("_", " "), "vector rule", "matrix rule"],
            keywords=["matrix", "vector", "complex", *ident.split("_")],
            rpn_program=program,
            semantics=f"Use {name.lower()} in matrix, vector, or complex-number problems.",
        )
        for ident, name, statement, program in items
    ]


def build_rule_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    catalog.extend(_build_exponent_rules())
    catalog.extend(_build_log_rules())
    catalog.extend(_build_factoring_rules())
    catalog.extend(_build_equation_rules())
    catalog.extend(_build_counting_probability_rules())
    catalog.extend(_build_geometry_formula_rules())
    catalog.extend(_build_geometry_theorem_rules())
    catalog.extend(_build_coordinate_rules())
    catalog.extend(_build_number_theory_rules())
    catalog.extend(_build_sequence_series_rules())
    catalog.extend(_build_trig_rules())
    catalog.extend(_build_matrix_vector_rules())
    return catalog


TEMPLATE_PROGRAM_SPECS = [
    {
        "id": "math_template_arithmetic_chain_gpu",
        "name": "Arithmetic chain template",
        "math_type": "Prealgebra",
        "template_program": "ARG0 ARG1 OP ARG2 OP ...",
        "aliases": ["arithmetic chain", "multi-step arithmetic", "evaluate expression"],
        "keywords": ["arithmetic", "chain", "expression"],
        "semantics": "evaluate multi-step arithmetic chains on the GPU stack",
        "anchors": [
            "evaluate an arithmetic expression step by step",
            "compute a multi-step arithmetic chain",
            "simplify a numerical expression",
            "evaluate ceiling or floor after simplifying arithmetic",
        ],
    },
    {
        "id": "math_template_linear_equation_ax_plus_b_eq_c_gpu",
        "name": "Linear equation solve template",
        "math_type": "Algebra",
        "template_program": "ARG_C ARG_B - ARG_A /",
        "aliases": ["solve ax+b=c", "linear equation", "solve for x"],
        "keywords": ["linear", "equation", "solve", "x"],
        "semantics": "solve linear equations of the form ax + b = c",
        "anchors": [
            "solve ax + b = c for x",
            "isolate x in a linear equation",
            "what value of x satisfies ax+b=c",
            "linear balance word problem",
        ],
    },
    {
        "id": "math_template_polynomial_degree_gpu",
        "name": "Polynomial degree template",
        "math_type": "Algebra",
        "template_program": "ARG_DEG_1 ARG_DEG_2 max ...",
        "aliases": ["degree of polynomial", "highest exponent"],
        "keywords": ["polynomial", "degree", "exponent"],
        "semantics": "recover the degree of a polynomial from its non-zero terms",
        "anchors": [
            "what is the degree of this polynomial",
            "find the highest exponent in a polynomial",
            "polynomial degree question",
            "highest power of x in the expression",
        ],
    },
    {
        "id": "math_template_polynomial_eval_gpu",
        "name": "Polynomial evaluation template",
        "math_type": "Algebra",
        "template_program": "ARG_TERM_1 ARG_TERM_2 + ...",
        "aliases": ["polynomial evaluation", "function substitution", "evaluate polynomial"],
        "keywords": ["polynomial", "evaluate", "substitute"],
        "semantics": "evaluate polynomial or function expressions after substitution",
        "anchors": [
            "evaluate the polynomial at a given value",
            "substitute x into the function",
            "compute f(a) from a polynomial",
            "function composition with explicit values",
        ],
    },
    {
        "id": "math_template_factorial_gpu",
        "name": "Factorial template",
        "math_type": "Counting & Probability",
        "template_program": "ARG_N factorial",
        "aliases": ["factorial", "n!", "compute factorial"],
        "keywords": ["factorial", "counting", "permutations"],
        "semantics": "compute factorial values on the GPU stack",
        "anchors": [
            "what is n factorial",
            "compute n!",
            "factorial counting problem",
            "arrangement count using factorial",
        ],
    },
    {
        "id": "math_template_binomial_gpu",
        "name": "Binomial coefficient template",
        "math_type": "Counting & Probability",
        "template_program": "ARG_N ARG_K binom",
        "aliases": ["n choose k", "binomial coefficient", "combination"],
        "keywords": ["binomial", "choose", "combination"],
        "semantics": "compute binomial coefficients on the GPU stack",
        "anchors": [
            "compute n choose k",
            "how many combinations are possible",
            "binomial coefficient problem",
            "choose function in combinatorics",
        ],
    },
    {
        "id": "math_template_permutation_gpu",
        "name": "Permutation template",
        "math_type": "Counting & Probability",
        "template_program": "ARG_N factorial ARG_N ARG_R - factorial /",
        "aliases": ["permutation", "ordered selection", "arrangement count"],
        "keywords": ["permutation", "arrange", "ordered"],
        "semantics": "compute permutations P(n,r) with GPU arithmetic",
        "anchors": [
            "how many permutations are possible",
            "ordered arrangement count",
            "arrange n objects taking r",
            "permutation formula problem",
        ],
    },
    {
        "id": "math_template_arithmetic_series_sum_gpu",
        "name": "Arithmetic series sum template",
        "math_type": "Precalculus",
        "template_program": "ARG_N ARG_A1 ARG_AN + * 2 /",
        "aliases": ["arithmetic series", "sum of first n integers", "finite progression sum"],
        "keywords": ["arithmetic", "series", "sum"],
        "semantics": "compute arithmetic-series sums on the GPU stack",
        "anchors": [
            "sum of an arithmetic series",
            "sum of the first n terms",
            "arithmetic progression total",
            "sum of first n integers",
        ],
    },
    {
        "id": "math_template_arithmetic_nth_term_gpu",
        "name": "Arithmetic sequence nth-term template",
        "math_type": "Precalculus",
        "template_program": "ARG_N 1 - ARG_D * ARG_A1 +",
        "aliases": ["arithmetic sequence nth term", "common difference", "sequence term"],
        "keywords": ["arithmetic", "sequence", "nth term"],
        "semantics": "compute arithmetic-sequence nth terms on the GPU stack",
        "anchors": [
            "find the nth term of an arithmetic sequence",
            "arithmetic sequence with common difference",
            "sequence term from first term and difference",
            "what is a_n in an arithmetic progression",
        ],
    },
    {
        "id": "math_template_geometric_series_sum_gpu",
        "name": "Geometric series sum template",
        "math_type": "Precalculus",
        "template_program": "ARG_A 1 ARG_R ARG_N pow - * 1 ARG_R - /",
        "aliases": ["geometric series", "ratio progression sum"],
        "keywords": ["geometric", "series", "sum"],
        "semantics": "compute finite geometric-series sums on the GPU stack",
        "anchors": [
            "sum of a geometric series",
            "finite geometric progression total",
            "geometric series with common ratio",
            "sum of n terms in a geometric sequence",
        ],
    },
    {
        "id": "math_template_geometric_nth_term_gpu",
        "name": "Geometric sequence nth-term template",
        "math_type": "Precalculus",
        "template_program": "ARG_A1 ARG_R ARG_N 1 - pow *",
        "aliases": ["geometric sequence nth term", "common ratio", "sequence term"],
        "keywords": ["geometric", "sequence", "nth term"],
        "semantics": "compute geometric-sequence nth terms on the GPU stack",
        "anchors": [
            "find the nth term of a geometric sequence",
            "geometric sequence with common ratio",
            "sequence term from first term and ratio",
            "what is a_n in a geometric progression",
        ],
    },
    {
        "id": "math_template_quadratic_discriminant_gpu",
        "name": "Quadratic discriminant template",
        "math_type": "Algebra",
        "template_program": "ARG_B ARG_B * 4 ARG_A * ARG_C * -",
        "aliases": ["discriminant", "b squared minus four a c"],
        "keywords": ["quadratic", "discriminant"],
        "semantics": "compute quadratic discriminants on the GPU stack",
        "anchors": [
            "what is the discriminant of ax^2+bx+c",
            "compute b squared minus four a c",
            "quadratic discriminant question",
            "classify roots using the discriminant",
        ],
    },
    {
        "id": "math_template_quadratic_roots_gpu",
        "name": "Quadratic roots template",
        "math_type": "Algebra",
        "template_program": "(-b ± sqrt(b^2-4ac)) / (2a)",
        "aliases": ["quadratic formula", "solve quadratic", "roots of quadratic"],
        "keywords": ["quadratic", "roots", "formula"],
        "semantics": "solve quadratic equations with the quadratic formula",
        "anchors": [
            "solve ax^2+bx+c=0",
            "find the roots of a quadratic",
            "quadratic formula problem",
            "what are the zeros of the polynomial",
        ],
    },
    {
        "id": "math_template_gcd_gpu",
        "name": "GCD template",
        "math_type": "Number Theory",
        "template_program": "ARG_A ARG_B gcd",
        "aliases": ["gcd", "greatest common divisor", "greatest common factor"],
        "keywords": ["gcd", "common divisor", "number theory"],
        "semantics": "compute greatest common divisors on the GPU stack",
        "anchors": [
            "find the greatest common divisor",
            "greatest common factor problem",
            "compute gcd of two integers",
            "number theory gcd question",
        ],
    },
    {
        "id": "math_template_lcm_gpu",
        "name": "LCM template",
        "math_type": "Number Theory",
        "template_program": "ARG_A ARG_B * ARG_A ARG_B gcd / abs",
        "aliases": ["lcm", "least common multiple"],
        "keywords": ["lcm", "least common multiple", "number theory"],
        "semantics": "compute least common multiples on the GPU stack",
        "anchors": [
            "find the least common multiple",
            "compute lcm of two integers",
            "least common multiple problem",
            "number theory lcm question",
        ],
    },
    {
        "id": "math_template_remainder_gpu",
        "name": "Remainder template",
        "math_type": "Number Theory",
        "template_program": "ARG_A ARG_B mod",
        "aliases": ["remainder", "modulo", "mod"],
        "keywords": ["remainder", "modulo", "mod"],
        "semantics": "compute remainders with modular arithmetic on the GPU stack",
        "anchors": [
            "what is the remainder when a is divided by b",
            "modulo arithmetic problem",
            "compute a mod b",
            "remainder after division",
        ],
    },
    {
        "id": "math_template_circle_center_gpu",
        "name": "Circle center template",
        "math_type": "Geometry",
        "template_program": "(-ARG_X_LINEAR/2, -ARG_Y_LINEAR/2)",
        "aliases": ["circle center", "complete the square", "center from equation"],
        "keywords": ["circle", "center", "equation"],
        "semantics": "recover a circle center from the linear coefficients of x and y",
        "anchors": [
            "find the center of the circle from its equation",
            "complete the square to locate the center",
            "circle equation center question",
            "geometry circle center problem",
        ],
    },
    {
        "id": "math_template_midpoint_formula_gpu",
        "name": "Midpoint template",
        "math_type": "Geometry",
        "template_program": "((ARG_X1+ARG_X2)/2, (ARG_Y1+ARG_Y2)/2)",
        "aliases": ["midpoint", "segment midpoint", "coordinate midpoint"],
        "keywords": ["midpoint", "coordinate", "segment"],
        "semantics": "compute coordinate midpoints from endpoint pairs",
        "anchors": [
            "find the midpoint between two points",
            "coordinate midpoint problem",
            "segment midpoint in the plane",
            "average the endpoint coordinates",
        ],
    },
    {
        "id": "math_template_slope_formula_gpu",
        "name": "Slope template",
        "math_type": "Geometry",
        "template_program": "(ARG_Y2-ARG_Y1)/(ARG_X2-ARG_X1)",
        "aliases": ["slope", "rate of change", "line slope"],
        "keywords": ["slope", "line", "coordinate"],
        "semantics": "compute line slopes from coordinate pairs",
        "anchors": [
            "find the slope between two points",
            "coordinate slope problem",
            "rate of change from coordinates",
            "line slope formula question",
        ],
    },
    {
        "id": "math_template_band_formation_max_gpu",
        "name": "Band formation template",
        "math_type": "Algebra",
        "template_program": "bounded_integer_factor_search",
        "aliases": ["band formation", "rows and columns", "factor pair search"],
        "keywords": ["band", "rows", "columns"],
        "semantics": "solve bounded factor-pair search problems on the GPU stack",
        "anchors": [
            "rectangular band formation problem",
            "rows and columns with leftovers",
            "maximize total members under a cap",
            "integer factor pair search in algebra",
        ],
    },
    {
        "id": "math_template_interval_upper_root_gpu",
        "name": "Interval upper-root template",
        "math_type": "Intermediate Algebra",
        "template_program": "lower_bound upper_bound interval",
        "aliases": ["interval notation", "upper bound root", "inequality interval"],
        "keywords": ["interval", "inequality", "root"],
        "semantics": "compute interval notation from a lower bound and an upper root",
        "anchors": [
            "express the answer in interval notation",
            "solve an inequality and report the interval",
            "upper bound interval question",
            "valid values in interval form",
        ],
    },
    {
        "id": "math_template_l_shaped_sequence_gpu",
        "name": "L-shaped sequence template",
        "math_type": "Algebra",
        "template_program": "linked_row_column_progressions",
        "aliases": ["L-shaped sequence", "grid progression", "missing N"],
        "keywords": ["sequence", "grid", "progression"],
        "semantics": "solve linked arithmetic-sequence diagrams on the GPU stack",
        "anchors": [
            "L-shaped arithmetic sequence puzzle",
            "grid progression with missing N",
            "row and column arithmetic sequence problem",
            "linked sequence diagram question",
        ],
    },
    {
        "id": "math_template_compound_interest_gpu",
        "name": "Compound interest template",
        "math_type": "Prealgebra",
        "template_program": "F / (1 + r/m)^(mt)",
        "aliases": ["compound interest", "present value", "future value"],
        "keywords": ["interest", "compound", "investment"],
        "semantics": "compute present value from future value under compound interest",
        "anchors": [
            "compound interest investment problem",
            "how much should be invested now",
            "present value with quarterly compounding",
            "future value bank question",
        ],
    },
]


def build_template_program_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in TEMPLATE_PROGRAM_SPECS:
        entries.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "domain": "math",
                "category": "template_program",
                "content": spec["semantics"],
                "summary": spec["name"],
                "description": f"{spec['math_type']} template program.",
                "rpn_program": spec["template_program"],
                "answer_text": "",
                "metadata": {
                    "meaning_ref": spec["id"],
                    "math_type": spec["math_type"],
                    "subject": "mathematics",
                    "subfield": "template_program",
                    "template_ref": spec["id"],
                    "template_program": spec["template_program"],
                    "aliases": list(spec["aliases"]),
                    "keywords": list(spec["keywords"]),
                    "tags": list(spec["keywords"]),
                    "query_anchor": f"{spec['name']} {' '.join(spec['keywords'])}",
                    "semantics": spec["semantics"],
                    "direct_eval": False,
                    "ingest_source": "ingest_math_rules",
                },
            }
        )
    return entries


def build_template_support_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in TEMPLATE_PROGRAM_SPECS:
        for index, anchor in enumerate(spec["anchors"], start=1):
            entries.append(
                {
                    "id": f"{spec['id']}_support_{index:02d}",
                    "name": f"{spec['name']} support {index}",
                    "domain": "math",
                    "category": "template_support",
                    "content": spec["semantics"],
                    "summary": spec["name"],
                    "description": f"Support anchor for {spec['name']}.",
                    "rpn_program": "",
                    "answer_text": "",
                    "metadata": {
                        "meaning_ref": spec["id"],
                        "math_type": spec["math_type"],
                        "subject": "mathematics",
                        "subfield": "template_support",
                        "template_ref": spec["id"],
                        "template_params": {},
                        "aliases": list(spec["aliases"]),
                        "keywords": list(spec["keywords"]),
                        "tags": list(spec["keywords"]),
                        "query_anchor": anchor,
                        "semantics": spec["semantics"],
                        "direct_eval": False,
                        "ingest_source": "ingest_math_rules",
                    },
                }
            )
    return entries


def build_math_rule_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in build_rule_catalog():
        entries.append(_formula_fact_entry(spec))
        entries.append(_symbolic_rule_entry(spec))
        entries.append(_concept_anchor_entry(spec))
        entries.append(_problem_anchor_entry(spec))
    entries.extend(build_template_program_entries())
    entries.extend(build_template_support_entries())
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        deduped[str(entry["id"])] = entry
    return [deduped[key] for key in sorted(deduped.keys())]


def ingest_math_rules(
    knowledgeverse: Knowledgeverse,
    *,
    entries: list[dict[str, Any]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    manager = knowledgeverse.galaxy_manager
    emit = progress or (lambda _message: None)
    payload = list(entries or build_math_rule_entries())
    emit(f"Math rules: staging {len(payload)} entries")
    status_counts = {"inserted": 0, "updated": 0}
    category_counts: dict[str, int] = {}
    math_type_counts: dict[str, int] = {}
    with manager.bulk_disk_sync():
        for entry in payload:
            status = manager.upsert_entry("Math", entry)
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            category = str(entry.get("category", "")).strip() or "<none>"
            category_counts[category] = int(category_counts.get(category, 0)) + 1
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            math_type = str(metadata.get("math_type", "")).strip() or "<none>"
            math_type_counts[math_type] = int(math_type_counts.get(math_type, 0)) + 1
    summary = {
        "total_entries": len(payload),
        "status": status_counts,
        "categories": category_counts,
        "math_types": math_type_counts,
    }
    emit(
        "Math rules: ingest complete "
        + json.dumps(summary["status"], ensure_ascii=False, sort_keys=True)
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    knowledgeverse = Knowledgeverse(storage_root=args.storage_root)
    summary = ingest_math_rules(knowledgeverse, progress=lambda message: print(message, flush=True))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
