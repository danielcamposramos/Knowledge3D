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
    template_ref: str = "",
    query_anchor: str = "",
    arg_keys: list[str] | None = None,
    eval_program: str = "",
    eval_programs: list[str] | None = None,
    output_kind: str = "",
    template_params: dict[str, Any] | None = None,
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
        "template_ref": str(template_ref or "").strip(),
        "query_anchor": str(query_anchor or "").strip(),
        "arg_keys": [str(value).strip().lower() for value in list(arg_keys or []) if str(value).strip()],
        "eval_program": str(eval_program or "").strip(),
        "eval_programs": [str(value).strip() for value in list(eval_programs or []) if str(value).strip()],
        "output_kind": str(output_kind or "").strip(),
        "template_params": dict(template_params or {}),
    }


def _base_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    metadata = {
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
    template_ref = str(spec.get("template_ref", "")).strip()
    query_anchor = str(spec.get("query_anchor", "")).strip()
    eval_program = str(spec.get("eval_program", "")).strip()
    eval_programs = [str(value).strip() for value in list(spec.get("eval_programs") or []) if str(value).strip()]
    output_kind = str(spec.get("output_kind", "")).strip()
    arg_keys = [str(value).strip().lower() for value in list(spec.get("arg_keys") or []) if str(value).strip()]
    template_params = spec.get("template_params") if isinstance(spec.get("template_params"), dict) else {}
    if template_ref:
        metadata["template_ref"] = template_ref
    if query_anchor:
        metadata["query_anchor"] = query_anchor
    if eval_program:
        metadata["eval_program"] = eval_program
    if eval_programs:
        metadata["eval_programs"] = list(eval_programs)
    if output_kind:
        metadata["output_kind"] = output_kind
    if arg_keys:
        metadata["arg_keys"] = list(arg_keys)
    if template_params:
        metadata["template_params"] = dict(template_params)
    return metadata


def _preferred_query_anchor(spec: dict[str, Any], fallback: str) -> str:
    query_anchor = str(spec.get("query_anchor", "")).strip()
    return query_anchor or fallback


def _template_metadata_extras(spec: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("eval_program", "output_kind"):
        value = str(spec.get(key, "")).strip()
        if value:
            metadata[key] = value
    eval_programs = [str(value).strip() for value in list(spec.get("eval_programs") or []) if str(value).strip()]
    if eval_programs:
        metadata["eval_programs"] = list(eval_programs)
    arg_keys = [str(value).strip().lower() for value in list(spec.get("arg_keys") or []) if str(value).strip()]
    if arg_keys:
        metadata["arg_keys"] = list(arg_keys)
    template_params = spec.get("template_params") if isinstance(spec.get("template_params"), dict) else {}
    if template_params:
        metadata["template_params"] = dict(template_params)
    return metadata


def _formula_fact_entry(spec: dict[str, Any]) -> dict[str, Any]:
    metadata = _base_metadata(spec)
    metadata["query_anchor"] = _preferred_query_anchor(spec, f"{spec['name']} {spec['statement']}")
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
    metadata["query_anchor"] = _preferred_query_anchor(spec, f"{spec['name']} {' '.join(spec['keywords'])}")
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
    metadata["query_anchor"] = _preferred_query_anchor(spec, f"{spec['name']} concept {' '.join(spec['aliases'])}")
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
    metadata["query_anchor"] = _preferred_query_anchor(
        spec,
        f"Use {spec['name']} to solve {' '.join(spec['keywords'])} problems in {spec['math_type']}",
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


def _build_prealgebra_rules() -> list[dict[str, Any]]:
    items = [
        {
            "ident": "prealgebra_arithmetic_operations",
            "name": "Prealgebra arithmetic operations",
            "statement": "Prealgebra arithmetic questions simplify integers, signed values, and fractions by applying the requested operation chain in order.",
            "aliases": ["basic arithmetic", "integer arithmetic", "fraction arithmetic", "order of operations"],
            "keywords": ["prealgebra", "arithmetic", "integers", "fractions", "operations"],
            "template_ref": "math_template_arithmetic_chain_gpu",
            "query_anchor": "evaluate arithmetic expression add subtract multiply divide integers fractions order of operations",
        },
        {
            "ident": "prealgebra_fraction_simplification",
            "name": "Fraction simplification and common denominators",
            "statement": "Fraction questions reduce ratios by common factors and combine terms by finding a common denominator.",
            "aliases": ["simplify fraction", "common denominator", "reduce ratio"],
            "keywords": ["prealgebra", "fraction", "simplify", "denominator"],
            "template_ref": "math_template_arithmetic_chain_gpu",
            "query_anchor": "simplify fraction reduce to lowest terms common denominator add subtract fractions",
        },
        {
            "ident": "prealgebra_percentage_conversion",
            "name": "Percentage conversion",
            "statement": "Percent questions convert between fractions, decimals, and percentages by scaling by 100 or dividing by 100.",
            "aliases": ["percent conversion", "fraction to percent", "decimal to percent"],
            "keywords": ["prealgebra", "percent", "percentage", "decimal", "fraction"],
            "template_ref": "math_template_percentage_gpu",
            "query_anchor": "convert fraction decimal percentage percent of whole percent conversion",
            "arg_keys": ["part", "whole"],
            "eval_program": "ARG_PART ARG_WHOLE / 100 *",
        },
        {
            "ident": "prealgebra_ratio_proportion",
            "name": "Ratio and proportion",
            "statement": "Ratio and proportion problems scale one quantity by matching multiplicative relationships between two known pairs.",
            "aliases": ["ratio", "proportion", "scale factor"],
            "keywords": ["prealgebra", "ratio", "proportion", "scale"],
            "template_ref": "math_template_ratio_proportion_gpu",
            "query_anchor": "ratio proportion if a corresponds to b what corresponds to c scale factor",
            "arg_keys": ["a", "b", "c"],
            "eval_program": "ARG_B ARG_C * ARG_A /",
        },
        {
            "ident": "prealgebra_area_rectangle",
            "name": "Rectangle area",
            "statement": "Rectangle area is length times width, and composite rectangles split into smaller rectangular parts.",
            "aliases": ["rectangle area", "length times width", "composite rectangle"],
            "keywords": ["prealgebra", "area", "rectangle", "length", "width"],
            "template_ref": "math_template_rectangle_area_gpu",
            "query_anchor": "find area of rectangle length width composite rectangle area",
            "arg_keys": ["length", "width"],
            "eval_program": "ARG_LENGTH ARG_WIDTH *",
        },
        {
            "ident": "prealgebra_area_triangle",
            "name": "Triangle area",
            "statement": "Triangle area is one half of base times height.",
            "aliases": ["triangle area", "one half base times height", "base height triangle"],
            "keywords": ["prealgebra", "area", "triangle", "base", "height"],
            "template_ref": "math_template_triangle_area_gpu",
            "query_anchor": "find area of triangle base height one half b h",
            "arg_keys": ["base", "height"],
            "eval_program": "ARG_BASE ARG_HEIGHT * 2 /",
        },
        {
            "ident": "prealgebra_perimeter",
            "name": "Perimeter and circumference",
            "statement": "Perimeter sums the outer side lengths, while circle perimeter uses circumference formulas such as 2 pi r.",
            "aliases": ["perimeter", "circumference", "sum of sides"],
            "keywords": ["prealgebra", "perimeter", "circumference", "sides"],
            "template_ref": "math_template_rectangle_perimeter_gpu",
            "query_anchor": "find perimeter sum of sides rectangle circumference circle perimeter",
        },
        {
            "ident": "prealgebra_angle_sum_triangle",
            "name": "Triangle angle sum",
            "statement": "The interior angles of a triangle add to 180 degrees, so a missing angle is found by subtracting the known angles from 180.",
            "aliases": ["triangle angles", "angles sum to 180", "missing triangle angle"],
            "keywords": ["prealgebra", "angle", "triangle", "180"],
            "template_ref": "math_template_triangle_missing_angle_gpu",
            "query_anchor": "triangle angle sum missing angle interior angles sum to 180",
            "arg_keys": ["angle_a", "angle_b"],
            "eval_program": "180 ARG_ANGLE_A ARG_ANGLE_B + -",
        },
        {
            "ident": "prealgebra_angle_complement",
            "name": "Complementary and supplementary angles",
            "statement": "Complementary angles add to 90 degrees and supplementary angles add to 180 degrees.",
            "aliases": ["complementary angles", "supplementary angles", "missing angle"],
            "keywords": ["prealgebra", "angle", "complementary", "supplementary"],
            "template_ref": "math_template_complementary_angle_gpu",
            "query_anchor": "complementary supplementary angles sum to 90 or 180 missing angle",
        },
        {
            "ident": "prealgebra_mean_median_mode",
            "name": "Mean median mode",
            "statement": "Basic statistics questions compute mean, identify the middle ordered value, or find the most frequent value.",
            "aliases": ["average", "mean median mode", "basic statistics"],
            "keywords": ["prealgebra", "mean", "median", "mode", "average"],
            "template_ref": "math_template_mean_average_gpu",
            "query_anchor": "average mean of numbers median mode basic statistics",
        },
        {
            "ident": "prealgebra_number_line_ordering",
            "name": "Number-line ordering",
            "statement": "Number-line questions compare signed integers and fractions by relative position and common denominator reasoning.",
            "aliases": ["order numbers", "compare integers", "number line"],
            "keywords": ["prealgebra", "number line", "order", "compare", "fractions"],
            "query_anchor": "order integers fractions least to greatest compare on number line",
        },
        {
            "ident": "prealgebra_place_value",
            "name": "Place value and rounding",
            "statement": "Place-value questions identify digit positions and round according to the next digit.",
            "aliases": ["place value", "rounding", "digit position"],
            "keywords": ["prealgebra", "place value", "rounding", "digits"],
            "query_anchor": "place value round nearest digit tens hundreds thousands",
        },
        {
            "ident": "prealgebra_unit_conversion",
            "name": "Unit conversion",
            "statement": "Unit conversion problems multiply or divide by a scale factor, and affine scales add an offset after rescaling.",
            "aliases": ["unit conversion", "convert units", "scale conversion"],
            "keywords": ["prealgebra", "unit", "conversion", "scale"],
            "template_ref": "math_template_unit_conversion_scale_gpu",
            "query_anchor": "convert feet inches hours minutes centimeters meters scale factor",
        },
        {
            "ident": "prealgebra_divisibility_rules",
            "name": "Divisibility rules",
            "statement": "Divisibility rules use digit patterns and digit sums to decide whether an integer is divisible by 2, 3, 4, 5, 6, 8, 9, or 10.",
            "aliases": ["divisibility", "digit test", "multiple rule"],
            "keywords": ["prealgebra", "divisibility", "digit sum", "multiple"],
            "query_anchor": "divisible by 2 3 4 5 6 8 9 10 divisibility rule",
        },
        {
            "ident": "prealgebra_prime_composite",
            "name": "Prime and composite numbers",
            "statement": "Prime/composite questions identify whether an integer has exactly two positive divisors and may ask for prime factorization.",
            "aliases": ["prime numbers", "composite numbers", "prime factorization"],
            "keywords": ["prealgebra", "prime", "composite", "factorization"],
            "query_anchor": "prime composite identify prime factorization smallest prime factor",
        },
    ]
    return [
        _rule_spec(
            ident=item["ident"],
            name=item["name"],
            statement=item["statement"],
            math_type="Prealgebra",
            subfield="prealgebra",
            aliases=item["aliases"],
            keywords=item["keywords"],
            semantics=f"Use {item['name'].lower()} as the prealgebra solver neighborhood.",
            template_ref=str(item.get("template_ref", "")).strip(),
            query_anchor=str(item.get("query_anchor", "")).strip(),
            arg_keys=item.get("arg_keys"),
            eval_program=str(item.get("eval_program", "")).strip(),
        )
        for item in items
    ]


def _build_high_roi_number_theory_rules() -> list[dict[str, Any]]:
    items = [
        ("number_theory_gcd_euclidean", "GCD via Euclidean algorithm", "Repeated remainder reduction computes the greatest common divisor.", ["gcd", "euclidean algorithm", "greatest common divisor"], ["number theory", "gcd", "euclidean", "divisor"], "math_template_gcd_gpu", "greatest common divisor gcd euclidean algorithm common factor"),
        ("number_theory_lcm_relation", "LCM relation", "The least common multiple satisfies lcm(a,b) = |ab| / gcd(a,b).", ["lcm", "least common multiple"], ["number theory", "lcm", "multiple"], "math_template_lcm_gpu", "least common multiple lcm of integers common multiple"),
        ("number_theory_modular_arithmetic", "Modular arithmetic", "Remainders track arithmetic classes modulo n and reduce large computations.", ["modular arithmetic", "mod", "remainder"], ["number theory", "mod", "remainder", "modulo"], "math_template_remainder_gpu", "modulo remainder when divided by modular arithmetic"),
        ("number_theory_remainder_theorem", "Polynomial remainder theorem", "The remainder of p(x) on division by x-a equals p(a).", ["remainder theorem", "polynomial remainder"], ["number theory", "remainder", "polynomial"], "", "remainder theorem polynomial divided by x minus a"),
        ("number_theory_divisibility_counting", "Divisibility counting", "Count multiples in an interval by comparing floor divisions at the upper and lower bounds.", ["count multiples", "divisible in range"], ["number theory", "divisible", "count", "range"], "", "how many integers in range divisible by k count multiples"),
        ("number_theory_prime_factorization", "Prime factorization", "Every positive integer factors uniquely into primes up to ordering.", ["prime factorization", "fundamental theorem of arithmetic"], ["number theory", "prime", "factorization"], "", "prime factorization unique prime factors"),
        ("number_theory_base_conversion", "Base conversion", "Base-n representations convert to decimal by weighted digit sums.", ["base conversion", "convert from base n"], ["number theory", "base", "conversion", "digits"], "math_template_base_to_decimal_two_digit_gpu", "convert number from base n to decimal base conversion"),
        ("number_theory_floor_ceiling", "Floor and ceiling evaluation", "Floor and ceiling problems round down or up to the nearest integer boundary.", ["floor function", "ceiling function"], ["number theory", "floor", "ceiling"], "math_template_floor_gpu", "evaluate floor ceiling greatest integer least integer"),
        ("number_theory_digit_sum", "Digit sum and digital root", "Digit-sum questions aggregate decimal digits and may reduce repeatedly to a digital root.", ["digit sum", "digital root"], ["number theory", "digits", "sum"], "", "sum of digits digital root decimal expansion"),
        ("number_theory_congruence", "Congruence classes", "Congruence statements compare whether two integers leave the same remainder modulo n.", ["congruence", "mod equivalence", "chinese remainder"], ["number theory", "congruence", "modulo"], "math_template_remainder_gpu", "a congruent to b mod n congruence classes same remainder"),
    ]
    return [
        _rule_spec(
            ident=ident,
            name=name,
            statement=statement,
            math_type="Number Theory",
            subfield="number_theory_problem_family",
            aliases=list(aliases),
            keywords=list(keywords),
            semantics=f"Use {name.lower()} when the question asks for a number-theory procedure.",
            template_ref=template_ref,
            query_anchor=query_anchor,
        )
        for ident, name, statement, aliases, keywords, template_ref, query_anchor in items
    ]


def _build_high_roi_geometry_rules() -> list[dict[str, Any]]:
    items = [
        ("geometry_angle_triangle_sum", "Geometry triangle-angle sum", "Triangle interior angles sum to 180 degrees.", ["triangle angles", "missing triangle angle"], ["geometry", "triangle", "angle"], "math_template_triangle_missing_angle_gpu", "triangle angle sum missing interior angle 180 degrees"),
        ("geometry_angle_parallel_transversal", "Parallel-line angle chasing", "Parallel lines cut by a transversal create corresponding, alternate-interior, and supplementary angle relations.", ["parallel lines", "transversal angles"], ["geometry", "parallel", "transversal", "angle"], "", "parallel lines transversal alternate interior corresponding supplementary angles"),
        ("geometry_triangle_area_family", "Triangle area family", "Triangle area questions use base-height or Heron's formula depending on the information given.", ["triangle area", "heron formula"], ["geometry", "triangle", "area"], "math_template_triangle_area_gpu", "triangle area base height heron formula"),
        ("geometry_triangle_pythagorean", "Right-triangle side relation", "Right-triangle side lengths satisfy the Pythagorean theorem.", ["pythagorean theorem", "hypotenuse"], ["geometry", "right triangle", "hypotenuse"], "math_template_pythagorean_hypotenuse_gpu", "right triangle hypotenuse pythagorean theorem side length"),
        ("geometry_triangle_similar", "Similar triangles", "Similar triangles preserve angle equality and side proportionality.", ["similar triangles", "proportional sides"], ["geometry", "triangle", "similar"], "", "similar triangles proportional sides scale factor"),
        ("geometry_triangle_congruent", "Congruent triangles", "Congruent triangles match side and angle data through SSS, SAS, ASA, or AAS.", ["triangle congruence", "sss sas asa"], ["geometry", "triangle", "congruent"], "", "triangle congruence sss sas asa aas"),
        ("geometry_circle_area_circumference", "Circle area and circumference", "Circle measurement problems use pi r squared for area and 2 pi r for circumference.", ["circle area", "circle circumference"], ["geometry", "circle", "area", "circumference"], "math_template_circle_area_gpu", "circle area circumference radius diameter pi"),
        ("geometry_circle_arc_sector", "Arc and sector measurement", "Arc length and sector area scale with central angle.", ["arc length", "sector area"], ["geometry", "circle", "arc", "sector"], "math_template_sector_area_gpu", "arc length sector area central angle radius"),
        ("geometry_circle_inscribed_angle", "Inscribed-angle theorem", "An inscribed angle equals half the measure of its intercepted arc.", ["inscribed angle", "central angle"], ["geometry", "circle", "inscribed", "angle"], "", "inscribed angle equals half intercepted arc"),
        ("geometry_coordinate_distance", "Coordinate distance", "Coordinate geometry distance is the square root of horizontal and vertical squared differences.", ["distance formula", "coordinate distance"], ["geometry", "coordinate", "distance"], "math_template_distance_formula_gpu", "distance between two points coordinate plane"),
        ("geometry_coordinate_midpoint", "Coordinate midpoint", "Midpoints average the endpoint coordinates.", ["midpoint formula", "segment midpoint"], ["geometry", "coordinate", "midpoint"], "math_template_midpoint_formula_gpu", "midpoint of segment between two points"),
        ("geometry_coordinate_slope", "Coordinate slope", "Slope compares vertical change to horizontal change.", ["slope formula", "rise over run"], ["geometry", "coordinate", "slope"], "math_template_slope_formula_gpu", "slope between two points rise over run"),
        ("geometry_polygon_area", "Polygon area", "Polygon area questions often decompose a figure or use a regular-polygon formula.", ["polygon area", "regular polygon area"], ["geometry", "polygon", "area"], "math_template_regular_polygon_area_gpu", "regular polygon area apothem perimeter polygon area"),
        ("geometry_volume_prism_cylinder", "Prism and cylinder volume", "Prism-like solids use base area times height, including cylinders.", ["prism volume", "cylinder volume"], ["geometry", "volume", "prism", "cylinder"], "math_template_cylinder_volume_gpu", "volume of prism cylinder base area times height"),
        ("geometry_volume_cone_sphere", "Cone and sphere volume", "Cones and spheres use specialized pi-based volume formulas.", ["cone volume", "sphere volume"], ["geometry", "volume", "cone", "sphere"], "math_template_cone_volume_gpu", "volume of cone sphere pi r squared h"),
        ("geometry_surface_area", "Surface area of solids", "Surface-area questions sum all exposed faces of a solid.", ["surface area", "net of solid"], ["geometry", "surface area", "solid"], "math_template_surface_area_rectangular_prism_gpu", "surface area of rectangular prism solid"),
    ]
    return [
        _rule_spec(
            ident=ident,
            name=name,
            statement=statement,
            math_type="Geometry",
            subfield="geometry_problem_family",
            aliases=list(aliases),
            keywords=list(keywords),
            semantics=f"Use {name.lower()} to route geometry questions into the right solver pattern.",
            template_ref=template_ref,
            query_anchor=query_anchor,
        )
        for ident, name, statement, aliases, keywords, template_ref, query_anchor in items
    ]


def _build_high_roi_algebra_rules() -> list[dict[str, Any]]:
    items = [
        ("algebra_linear_equation_one_var", "Linear equation in one variable", "Solve ax + b = c by undoing the additive and multiplicative steps.", ["linear equation", "solve for x"], ["algebra", "linear", "equation", "x"], "math_template_linear_equation_ax_plus_b_eq_c_gpu", "solve linear equation ax plus b equals c isolate x"),
        ("algebra_linear_equation_two_var", "Two-variable linear system", "Solve two equations in two variables by elimination, substitution, or determinant formulas.", ["system of equations", "two variables"], ["algebra", "system", "equations"], "math_template_system_2x2_x_gpu", "solve system of two equations two variables elimination substitution"),
        ("algebra_quadratic_formula_family", "Quadratic formula family", "Quadratic equations use the discriminant and quadratic formula to locate roots.", ["quadratic formula", "roots of quadratic"], ["algebra", "quadratic", "roots"], "math_template_quadratic_roots_gpu", "solve quadratic equation find roots zeros discriminant"),
        ("algebra_quadratic_factoring_family", "Quadratic factoring family", "Quadratic expressions may factor into binomials when coefficients align with integer root patterns.", ["quadratic factoring", "factor trinomial"], ["algebra", "quadratic", "factor"], "", "factor quadratic trinomial product sum"),
        ("algebra_completing_square_family", "Completing-the-square family", "Completing the square rewrites a quadratic into vertex form by adding and subtracting the same square.", ["completing the square", "vertex form"], ["algebra", "complete square", "vertex"], "math_template_completing_square_vertex_x_gpu", "complete the square vertex of parabola"),
        ("algebra_absolute_value_family", "Absolute-value equations", "Absolute-value questions split into positive and negative cases around a distance from zero.", ["absolute value", "distance from zero"], ["algebra", "absolute value"], "math_template_absolute_value_gpu", "absolute value equation distance from zero"),
        ("algebra_function_composition_family", "Function composition", "Function-composition problems evaluate an inner function before applying the outer function.", ["function composition", "f of g x"], ["algebra", "function", "composition"], "math_template_polynomial_eval_gpu", "function composition evaluate f g of x"),
        ("algebra_piecewise_evaluation_family", "Piecewise evaluation", "Piecewise functions select the branch whose interval contains the input.", ["piecewise function", "select branch"], ["algebra", "piecewise", "function"], "", "piecewise function evaluate correct branch interval"),
        ("algebra_domain_range_family", "Domain and range", "Domain/range questions find allowed inputs and resulting outputs for a function.", ["domain and range", "where function defined"], ["algebra", "domain", "range"], "", "domain range function defined allowed inputs outputs"),
        ("algebra_inverse_function_family", "Inverse function", "Inverse-function questions swap x and y and solve for the new output variable.", ["inverse function", "f inverse"], ["algebra", "inverse function"], "", "inverse function swap x and y solve"),
        ("algebra_polynomial_degree_family", "Polynomial degree", "Polynomial degree is the maximum exponent among non-zero terms.", ["polynomial degree", "highest exponent"], ["algebra", "polynomial", "degree"], "math_template_polynomial_degree_gpu", "degree of polynomial highest exponent nonzero term"),
        ("algebra_polynomial_roots_family", "Polynomial roots", "Polynomial roots are the inputs where the polynomial evaluates to zero.", ["polynomial roots", "zeros of polynomial"], ["algebra", "polynomial", "roots"], "math_template_quadratic_roots_gpu", "roots zeros polynomial equals zero"),
        ("algebra_rational_expression_family", "Rational expressions", "Rational-expression questions simplify numerator/denominator structure and track denominator restrictions.", ["rational expression", "denominator restriction"], ["algebra", "rational", "expression"], "", "simplify rational expression denominator restriction"),
        ("algebra_inequality_linear_family", "Linear inequalities", "Linear inequalities isolate the variable and flip the sign when multiplying or dividing by a negative.", ["linear inequality", "interval notation"], ["algebra", "inequality", "linear"], "", "solve linear inequality interval notation"),
        ("algebra_inequality_quadratic_family", "Quadratic inequalities", "Quadratic inequalities analyze sign changes across the roots to determine valid intervals.", ["quadratic inequality", "sign chart"], ["algebra", "inequality", "quadratic"], "math_template_interval_upper_root_gpu", "solve quadratic inequality sign chart interval notation"),
        ("algebra_word_problem_rate_family", "Rate and work problems", "Rate problems combine distance-rate-time or work-rate relations and inverse scaling.", ["rate problem", "work problem"], ["algebra", "rate", "time", "work"], "math_template_rate_scaling_gpu", "rate work time distance shared work inverse rate"),
        ("algebra_word_problem_mixture_family", "Mixture and shortfall problems", "Mixture and shortfall questions balance totals across concentrations, values, or currencies.", ["mixture problem", "shortfall"], ["algebra", "mixture", "shortfall"], "math_template_exchange_gap_gpu", "mixture concentration shortfall exchange not enough amount"),
        ("algebra_arithmetic_sequence_family", "Arithmetic sequence family", "Arithmetic sequences use a constant difference for nth-term and sum formulas.", ["arithmetic sequence", "common difference"], ["algebra", "sequence", "arithmetic"], "math_template_arithmetic_nth_term_gpu", "arithmetic sequence nth term common difference"),
        ("algebra_geometric_sequence_family", "Geometric sequence family", "Geometric sequences use a constant ratio for nth-term and sum formulas.", ["geometric sequence", "common ratio"], ["algebra", "sequence", "geometric"], "math_template_geometric_nth_term_gpu", "geometric sequence nth term common ratio"),
    ]
    return [
        _rule_spec(
            ident=ident,
            name=name,
            statement=statement,
            math_type="Algebra",
            subfield="algebra_problem_family",
            aliases=list(aliases),
            keywords=list(keywords),
            semantics=f"Use {name.lower()} to route algebra questions into the matching execution program.",
            template_ref=template_ref,
            query_anchor=query_anchor,
        )
        for ident, name, statement, aliases, keywords, template_ref, query_anchor in items
    ]


def _build_high_roi_counting_rules() -> list[dict[str, Any]]:
    items = [
        ("counting_permutation_family", "Permutation counting", "Permutation problems count ordered selections from a finite set.", ["permutation", "ordered arrangement"], ["counting", "probability", "permutation"], "math_template_permutation_gpu", "how many ordered arrangements permutations"),
        ("counting_combination_family", "Combination counting", "Combination problems count unordered selections using n choose k.", ["combination", "n choose k"], ["counting", "probability", "combination"], "math_template_binomial_gpu", "how many combinations choose n k"),
        ("counting_complement_family", "Probability complement", "Complement questions subtract the probability of the opposite event from 1.", ["probability complement", "not event"], ["counting", "probability", "complement"], "math_template_probability_complement_gpu", "probability complement not event one minus probability"),
        ("counting_multiplication_principle_family", "Multiplication principle", "Independent step counts multiply to get the total number of outcomes.", ["multiplication principle", "fundamental counting principle"], ["counting", "product rule", "ways"], "", "fundamental counting principle multiply number of ways"),
        ("counting_pigeonhole_family", "Pigeonhole principle", "More objects than boxes forces at least one shared box.", ["pigeonhole principle"], ["counting", "pigeonhole"], "", "pigeonhole principle objects boxes at least one repeated"),
        ("counting_inclusion_exclusion_family", "Inclusion-exclusion", "Union counts add the parts and subtract overlaps to avoid double counting.", ["inclusion exclusion", "overlap counting"], ["counting", "probability", "inclusion exclusion"], "math_template_inclusion_exclusion_two_gpu", "inclusion exclusion overlap count union of sets"),
        ("counting_conditional_probability_family", "Conditional probability", "Conditional probability divides joint probability by the conditioning event probability.", ["conditional probability", "given event"], ["counting", "probability", "conditional"], "math_template_conditional_probability_gpu", "conditional probability given b probability of a given b"),
        ("counting_expected_value_family", "Expected value", "Expected value sums outcome values weighted by their probabilities.", ["expected value", "weighted average"], ["counting", "probability", "expected value"], "math_template_expected_value_two_gpu", "expected value weighted average random variable"),
        ("counting_stars_and_bars_family", "Stars and bars", "Stars-and-bars distributes identical objects across bins by separator placement.", ["stars and bars", "distribute identical objects"], ["counting", "combinatorics", "distribution"], "", "stars and bars identical objects distribute into bins"),
        ("counting_binomial_theorem_family", "Binomial theorem", "Binomial expansions combine coefficients from Pascal-style counting and powers of each term.", ["binomial theorem", "expand binomial"], ["counting", "probability", "binomial"], "math_template_binomial_gpu", "binomial theorem expand a plus b to n"),
    ]
    return [
        _rule_spec(
            ident=ident,
            name=name,
            statement=statement,
            math_type="Counting & Probability",
            subfield="counting_problem_family",
            aliases=list(aliases),
            keywords=list(keywords),
            semantics=f"Use {name.lower()} to choose the counting/probability procedure.",
            template_ref=template_ref,
            query_anchor=query_anchor,
        )
        for ident, name, statement, aliases, keywords, template_ref, query_anchor in items
    ]


def _build_high_roi_intermediate_algebra_rules() -> list[dict[str, Any]]:
    items = [
        ("intermediate_algebra_polynomial_division_family", "Polynomial division", "Polynomial division reduces a rational expression or quotient term by term.", ["polynomial division", "synthetic division"], ["intermediate algebra", "polynomial", "division"], "", "synthetic division polynomial long division"),
        ("intermediate_algebra_partial_fractions_family", "Partial fractions", "Partial fractions decompose a rational function into simpler quotients over linear or quadratic factors.", ["partial fractions"], ["intermediate algebra", "partial fractions"], "", "partial fractions decompose rational expression"),
        ("intermediate_algebra_complex_operations_family", "Complex operations", "Complex-number questions add, multiply, conjugate, or find modulus values.", ["complex numbers", "complex operations"], ["intermediate algebra", "complex", "imaginary"], "math_template_complex_magnitude_gpu", "complex number modulus magnitude conjugate imaginary"),
        ("intermediate_algebra_logarithm_solve_family", "Logarithm solving", "Logarithm equations convert to exponential form and apply log identities.", ["solve logarithm equation", "change of base"], ["intermediate algebra", "logarithm", "exponential"], "", "solve logarithm equation exponential form"),
        ("intermediate_algebra_exponential_growth_family", "Exponential growth and decay", "Exponential-growth questions multiply by repeated growth factors across time.", ["exponential growth", "decay"], ["intermediate algebra", "exponential", "growth"], "math_template_compound_interest_gpu", "exponential growth decay compound change over time"),
        ("intermediate_algebra_vieta_family", "Vieta relations", "Vieta's formulas connect polynomial coefficients to sums and products of roots.", ["vieta formulas", "sum and product of roots"], ["intermediate algebra", "vieta", "roots"], "", "sum of roots product of roots coefficients"),
    ]
    return [
        _rule_spec(
            ident=ident,
            name=name,
            statement=statement,
            math_type="Intermediate Algebra",
            subfield="intermediate_algebra_problem_family",
            aliases=list(aliases),
            keywords=list(keywords),
            semantics=f"Use {name.lower()} for intermediate-algebra benchmark problems.",
            template_ref=template_ref,
            query_anchor=query_anchor,
        )
        for ident, name, statement, aliases, keywords, template_ref, query_anchor in items
    ]


def _build_high_roi_precalculus_rules() -> list[dict[str, Any]]:
    items = [
        ("precalculus_trig_identity_family", "Trigonometric identities", "Trig-identity questions rewrite expressions using Pythagorean, sum-difference, and double-angle formulas.", ["trig identities", "sin cos identities"], ["precalculus", "trigonometry", "identity"], "", "trigonometric identity simplify sine cosine tangent"),
        ("precalculus_trig_solve_family", "Trigonometric equation solving", "Trig equations solve for angles using inverse trig values and family solutions.", ["solve trig equation", "all solutions"], ["precalculus", "trigonometry", "solve"], "", "solve trig equation all solutions radians degrees"),
        ("precalculus_matrix_multiply_family", "Matrix multiplication", "Matrix multiplication combines rows and columns when inner dimensions align.", ["matrix multiplication"], ["precalculus", "matrix", "multiply"], "", "matrix multiplication row by column"),
        ("precalculus_matrix_determinant_family", "Matrix determinant", "Determinants summarize oriented scaling and invertibility for square matrices.", ["matrix determinant", "2x2 determinant"], ["precalculus", "matrix", "determinant"], "math_template_determinant_2x2_gpu", "determinant of matrix ad minus bc"),
        ("precalculus_vector_operations_family", "Vector operations", "Vector questions compute dot products, magnitudes, and directional relationships.", ["vector dot product", "vector magnitude"], ["precalculus", "vector", "dot product"], "math_template_dot_product_2d_gpu", "dot product vectors magnitude of vector"),
        ("precalculus_parametric_polar_family", "Parametric and polar conversions", "Polar and parametric forms convert between angle-radius and coordinate representations.", ["polar coordinates", "parametric equations"], ["precalculus", "polar", "parametric"], "", "polar coordinates convert to cartesian parametric form"),
    ]
    return [
        _rule_spec(
            ident=ident,
            name=name,
            statement=statement,
            math_type="Precalculus",
            subfield="precalculus_problem_family",
            aliases=list(aliases),
            keywords=list(keywords),
            semantics=f"Use {name.lower()} for precalculus benchmark problems.",
            template_ref=template_ref,
            query_anchor=query_anchor,
        )
        for ident, name, statement, aliases, keywords, template_ref, query_anchor in items
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
    catalog.extend(_build_prealgebra_rules())
    catalog.extend(_build_high_roi_number_theory_rules())
    catalog.extend(_build_high_roi_geometry_rules())
    catalog.extend(_build_high_roi_algebra_rules())
    catalog.extend(_build_high_roi_counting_rules())
    catalog.extend(_build_high_roi_intermediate_algebra_rules())
    catalog.extend(_build_high_roi_precalculus_rules())
    return catalog


def _direct_template_spec(
    *,
    ident: str,
    name: str,
    math_type: str,
    template_program: str,
    aliases: list[str],
    keywords: list[str],
    semantics: str,
    anchors: list[str],
    arg_keys: list[str],
    output_kind: str = "scalar",
    template_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": ident,
        "name": name,
        "math_type": math_type,
        "template_program": template_program,
        "aliases": list(aliases),
        "keywords": list(keywords),
        "semantics": semantics,
        "anchors": list(anchors),
        "arg_keys": [str(value).strip().lower() for value in arg_keys],
        "eval_program": template_program,
        "output_kind": output_kind,
        "template_params": dict(template_params or {}),
    }


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

TEMPLATE_PROGRAM_SPECS.extend(
    [
        _direct_template_spec(
            ident="math_template_triangle_area_gpu",
            name="Triangle area template",
            math_type="Geometry",
            template_program="ARG_BASE ARG_HEIGHT * 2 /",
            aliases=["triangle area", "base times height over two"],
            keywords=["triangle", "area", "base", "height"],
            semantics="compute triangle area from base and height",
            anchors=[
                "find the area of a triangle",
                "triangle with base and height",
                "one half base times height",
            ],
            arg_keys=["base", "height"],
        ),
        _direct_template_spec(
            ident="math_template_rectangle_area_gpu",
            name="Rectangle area template",
            math_type="Prealgebra",
            template_program="ARG_LENGTH ARG_WIDTH *",
            aliases=["rectangle area", "length times width"],
            keywords=["rectangle", "area", "length", "width"],
            semantics="compute rectangle area",
            anchors=[
                "find the area of a rectangle",
                "rectangle length width area",
                "length times width",
            ],
            arg_keys=["length", "width"],
        ),
        _direct_template_spec(
            ident="math_template_rectangle_perimeter_gpu",
            name="Rectangle perimeter template",
            math_type="Prealgebra",
            template_program="ARG_LENGTH ARG_WIDTH + 2 *",
            aliases=["rectangle perimeter", "perimeter of rectangle"],
            keywords=["rectangle", "perimeter", "length", "width"],
            semantics="compute rectangle perimeter",
            anchors=[
                "find the perimeter of a rectangle",
                "rectangle length width perimeter",
                "sum of all sides of rectangle",
            ],
            arg_keys=["length", "width"],
        ),
        _direct_template_spec(
            ident="math_template_triangle_missing_angle_gpu",
            name="Triangle missing-angle template",
            math_type="Geometry",
            template_program="180 ARG_ANGLE_A ARG_ANGLE_B + -",
            aliases=["triangle angle sum", "missing triangle angle"],
            keywords=["triangle", "angle", "180"],
            semantics="recover the third angle in a triangle",
            anchors=[
                "triangle angle sum problem",
                "find the missing angle in a triangle",
                "interior angles sum to 180",
            ],
            arg_keys=["angle_a", "angle_b"],
        ),
        _direct_template_spec(
            ident="math_template_complementary_angle_gpu",
            name="Complementary-angle template",
            math_type="Prealgebra",
            template_program="90 ARG_ANGLE -",
            aliases=["complementary angle", "angles sum to ninety"],
            keywords=["angle", "complementary", "90"],
            semantics="find a complementary angle",
            anchors=[
                "find the complementary angle",
                "angles sum to 90 degrees",
                "missing complementary angle",
            ],
            arg_keys=["angle"],
        ),
        _direct_template_spec(
            ident="math_template_supplementary_angle_gpu",
            name="Supplementary-angle template",
            math_type="Geometry",
            template_program="180 ARG_ANGLE -",
            aliases=["supplementary angle", "angles sum to 180"],
            keywords=["angle", "supplementary", "180"],
            semantics="find a supplementary angle",
            anchors=[
                "find the supplementary angle",
                "angles sum to 180 degrees",
                "missing supplementary angle",
            ],
            arg_keys=["angle"],
        ),
        _direct_template_spec(
            ident="math_template_mean_average_gpu",
            name="Arithmetic-mean template",
            math_type="Prealgebra",
            template_program="ARG_A ARG_B + ARG_C + 3 /",
            aliases=["average", "arithmetic mean"],
            keywords=["mean", "average", "statistics"],
            semantics="compute the arithmetic mean of three values",
            anchors=[
                "find the average of three numbers",
                "arithmetic mean question",
                "compute the mean",
            ],
            arg_keys=["a", "b", "c"],
        ),
        _direct_template_spec(
            ident="math_template_percentage_gpu",
            name="Percentage template",
            math_type="Prealgebra",
            template_program="ARG_PART ARG_WHOLE / 100 *",
            aliases=["percentage", "percent of whole"],
            keywords=["percent", "percentage", "fraction"],
            semantics="compute a percentage from part and whole",
            anchors=[
                "what percent is part of whole",
                "convert part over whole to percent",
                "percentage conversion problem",
            ],
            arg_keys=["part", "whole"],
        ),
        _direct_template_spec(
            ident="math_template_ratio_proportion_gpu",
            name="Ratio-proportion template",
            math_type="Prealgebra",
            template_program="ARG_B ARG_C * ARG_A /",
            aliases=["ratio proportion", "cross multiplication"],
            keywords=["ratio", "proportion", "scale"],
            semantics="solve a direct proportion a:b = c:x",
            anchors=[
                "ratio and proportion problem",
                "if a corresponds to b what corresponds to c",
                "cross multiply a proportion",
            ],
            arg_keys=["a", "b", "c"],
        ),
        _direct_template_spec(
            ident="math_template_circle_area_gpu",
            name="Circle area template",
            math_type="Geometry",
            template_program="3.141592653589793 ARG_RADIUS 2 pow *",
            aliases=["circle area", "pi r squared"],
            keywords=["circle", "area", "radius"],
            semantics="compute the area of a circle",
            anchors=[
                "find the area of a circle",
                "circle with radius r area",
                "pi r squared",
            ],
            arg_keys=["radius"],
        ),
        _direct_template_spec(
            ident="math_template_circle_circumference_gpu",
            name="Circle circumference template",
            math_type="Geometry",
            template_program="2 3.141592653589793 * ARG_RADIUS *",
            aliases=["circle circumference", "two pi r"],
            keywords=["circle", "circumference", "radius"],
            semantics="compute the circumference of a circle",
            anchors=[
                "find the circumference of a circle",
                "circle perimeter question",
                "two pi r",
            ],
            arg_keys=["radius"],
        ),
        _direct_template_spec(
            ident="math_template_pythagorean_hypotenuse_gpu",
            name="Pythagorean hypotenuse template",
            math_type="Geometry",
            template_program="ARG_A 2 pow ARG_B 2 pow + sqrt",
            aliases=["pythagorean theorem", "hypotenuse"],
            keywords=["pythagorean", "right triangle", "hypotenuse"],
            semantics="compute a right-triangle hypotenuse",
            anchors=[
                "find the hypotenuse of a right triangle",
                "apply the pythagorean theorem",
                "right triangle side length",
            ],
            arg_keys=["a", "b"],
        ),
        _direct_template_spec(
            ident="math_template_distance_formula_gpu",
            name="Distance-formula template",
            math_type="Geometry",
            template_program="ARG_X2 ARG_X1 - 2 pow ARG_Y2 ARG_Y1 - 2 pow + sqrt",
            aliases=["distance formula", "distance between points"],
            keywords=["distance", "coordinates", "points"],
            semantics="compute Euclidean distance between two points",
            anchors=[
                "distance between two points",
                "coordinate distance problem",
                "use the distance formula",
            ],
            arg_keys=["x1", "y1", "x2", "y2"],
        ),
        _direct_template_spec(
            ident="math_template_heron_area_gpu",
            name="Heron-area template",
            math_type="Geometry",
            template_program="ARG_A ARG_B + ARG_C + 2 / STORE_S RECALL_S RECALL_S ARG_A - * RECALL_S ARG_B - * RECALL_S ARG_C - * sqrt",
            aliases=["heron formula", "triangle area from sides"],
            keywords=["triangle", "heron", "sides"],
            semantics="compute triangle area from three side lengths",
            anchors=[
                "triangle area from side lengths",
                "apply heron formula",
                "three sides of triangle area",
            ],
            arg_keys=["a", "b", "c"],
        ),
        _direct_template_spec(
            ident="math_template_regular_polygon_interior_angle_gpu",
            name="Regular-polygon interior-angle template",
            math_type="Geometry",
            template_program="ARG_N 2 - 180 * ARG_N /",
            aliases=["interior angle of regular polygon", "polygon interior angle"],
            keywords=["polygon", "interior angle", "regular"],
            semantics="compute a regular polygon interior angle",
            anchors=[
                "interior angle of a regular polygon",
                "regular n gon interior angle",
                "polygon angle formula",
            ],
            arg_keys=["n"],
        ),
        _direct_template_spec(
            ident="math_template_sector_area_gpu",
            name="Sector-area template",
            math_type="Geometry",
            template_program="ARG_THETA 360 / 3.141592653589793 * ARG_RADIUS 2 pow *",
            aliases=["sector area", "central angle area"],
            keywords=["sector", "area", "circle", "theta"],
            semantics="compute sector area from central angle and radius",
            anchors=[
                "area of a sector",
                "central angle and radius sector area",
                "fraction of a circle area",
            ],
            arg_keys=["theta", "radius"],
        ),
        _direct_template_spec(
            ident="math_template_arc_length_gpu",
            name="Arc-length template",
            math_type="Geometry",
            template_program="ARG_THETA 360 / 2 * 3.141592653589793 * ARG_RADIUS *",
            aliases=["arc length", "length of arc"],
            keywords=["arc", "length", "circle", "theta"],
            semantics="compute arc length from central angle and radius",
            anchors=[
                "find the arc length",
                "central angle and radius arc length",
                "length of an arc on a circle",
            ],
            arg_keys=["theta", "radius"],
        ),
        _direct_template_spec(
            ident="math_template_cylinder_volume_gpu",
            name="Cylinder-volume template",
            math_type="Geometry",
            template_program="3.141592653589793 ARG_RADIUS 2 pow * ARG_HEIGHT *",
            aliases=["cylinder volume", "pi r squared h"],
            keywords=["cylinder", "volume", "radius", "height"],
            semantics="compute cylinder volume",
            anchors=[
                "volume of a cylinder",
                "cylinder radius and height",
                "pi r squared h",
            ],
            arg_keys=["radius", "height"],
        ),
        _direct_template_spec(
            ident="math_template_cone_volume_gpu",
            name="Cone-volume template",
            math_type="Geometry",
            template_program="3.141592653589793 ARG_RADIUS 2 pow * ARG_HEIGHT * 3 /",
            aliases=["cone volume", "one third pi r squared h"],
            keywords=["cone", "volume", "radius", "height"],
            semantics="compute cone volume",
            anchors=[
                "volume of a cone",
                "cone radius and height",
                "one third pi r squared h",
            ],
            arg_keys=["radius", "height"],
        ),
        _direct_template_spec(
            ident="math_template_sphere_volume_gpu",
            name="Sphere-volume template",
            math_type="Geometry",
            template_program="4 3 / 3.141592653589793 * ARG_RADIUS 3 pow *",
            aliases=["sphere volume", "four thirds pi r cubed"],
            keywords=["sphere", "volume", "radius"],
            semantics="compute sphere volume",
            anchors=[
                "volume of a sphere",
                "sphere radius volume",
                "four thirds pi r cubed",
            ],
            arg_keys=["radius"],
        ),
        _direct_template_spec(
            ident="math_template_rectangular_prism_volume_gpu",
            name="Rectangular-prism volume template",
            math_type="Geometry",
            template_program="ARG_LENGTH ARG_WIDTH * ARG_HEIGHT *",
            aliases=["rectangular prism volume", "box volume"],
            keywords=["prism", "volume", "length", "width", "height"],
            semantics="compute rectangular-prism volume",
            anchors=[
                "volume of a rectangular prism",
                "box length width height volume",
                "multiply length width height",
            ],
            arg_keys=["length", "width", "height"],
        ),
        _direct_template_spec(
            ident="math_template_surface_area_rectangular_prism_gpu",
            name="Rectangular-prism surface-area template",
            math_type="Geometry",
            template_program="ARG_LENGTH ARG_WIDTH * ARG_LENGTH ARG_HEIGHT * + ARG_WIDTH ARG_HEIGHT * + 2 *",
            aliases=["surface area rectangular prism", "box surface area"],
            keywords=["surface area", "prism", "box"],
            semantics="compute rectangular-prism surface area",
            anchors=[
                "surface area of a rectangular prism",
                "box surface area problem",
                "sum the areas of all six faces",
            ],
            arg_keys=["length", "width", "height"],
        ),
        _direct_template_spec(
            ident="math_template_inclusion_exclusion_two_gpu",
            name="Two-set inclusion-exclusion template",
            math_type="Counting & Probability",
            template_program="ARG_A ARG_B + ARG_AB -",
            aliases=["inclusion exclusion", "union with overlap"],
            keywords=["inclusion", "exclusion", "union", "overlap"],
            semantics="compute a two-set union size or probability",
            anchors=[
                "inclusion exclusion for two sets",
                "count the union with overlap",
                "add the parts and subtract intersection",
            ],
            arg_keys=["a", "b", "ab"],
        ),
        _direct_template_spec(
            ident="math_template_probability_complement_gpu",
            name="Probability-complement template",
            math_type="Counting & Probability",
            template_program="1 ARG_P -",
            aliases=["probability complement", "one minus p"],
            keywords=["probability", "complement", "not"],
            semantics="compute the complement of an event probability",
            anchors=[
                "probability of not a",
                "complement probability",
                "one minus the event probability",
            ],
            arg_keys=["p"],
        ),
        _direct_template_spec(
            ident="math_template_conditional_probability_gpu",
            name="Conditional-probability template",
            math_type="Counting & Probability",
            template_program="ARG_P_INTERSECTION ARG_P_GIVEN /",
            aliases=["conditional probability", "p of a given b"],
            keywords=["probability", "conditional", "given"],
            semantics="compute conditional probability from joint and given-event probability",
            anchors=[
                "probability of a given b",
                "conditional probability problem",
                "joint probability divided by given event",
            ],
            arg_keys=["p_intersection", "p_given"],
        ),
        _direct_template_spec(
            ident="math_template_expected_value_two_gpu",
            name="Two-outcome expected-value template",
            math_type="Counting & Probability",
            template_program="ARG_X1 ARG_P1 * ARG_X2 ARG_P2 * +",
            aliases=["expected value", "weighted average"],
            keywords=["expected value", "probability", "outcomes"],
            semantics="compute a two-outcome expected value",
            anchors=[
                "expected value with two outcomes",
                "weighted average of outcomes",
                "sum x times probability",
            ],
            arg_keys=["x1", "p1", "x2", "p2"],
        ),
        _direct_template_spec(
            ident="math_template_floor_gpu",
            name="Floor template",
            math_type="Number Theory",
            template_program="ARG_X floor",
            aliases=["floor function", "greatest integer"],
            keywords=["floor", "greatest integer"],
            semantics="evaluate a floor function",
            anchors=[
                "evaluate the floor of x",
                "greatest integer function",
                "floor function problem",
            ],
            arg_keys=["x"],
        ),
        _direct_template_spec(
            ident="math_template_ceiling_gpu",
            name="Ceiling template",
            math_type="Number Theory",
            template_program="ARG_X ceil",
            aliases=["ceiling function", "least integer above"],
            keywords=["ceiling", "least integer"],
            semantics="evaluate a ceiling function",
            anchors=[
                "evaluate the ceiling of x",
                "smallest integer greater than or equal",
                "ceiling function problem",
            ],
            arg_keys=["x"],
        ),
        _direct_template_spec(
            ident="math_template_absolute_value_gpu",
            name="Absolute-value template",
            math_type="Algebra",
            template_program="ARG_X abs",
            aliases=["absolute value", "distance from zero"],
            keywords=["absolute", "value", "distance"],
            semantics="evaluate absolute value",
            anchors=[
                "evaluate absolute value",
                "distance from zero",
                "absolute value problem",
            ],
            arg_keys=["x"],
        ),
        _direct_template_spec(
            ident="math_template_simple_interest_gpu",
            name="Simple-interest template",
            math_type="Prealgebra",
            template_program="ARG_P ARG_R * ARG_T *",
            aliases=["simple interest", "principal rate time"],
            keywords=["interest", "simple", "principal", "rate", "time"],
            semantics="compute simple interest",
            anchors=[
                "simple interest problem",
                "principal rate time interest",
                "interest earned without compounding",
            ],
            arg_keys=["p", "r", "t"],
        ),
        _direct_template_spec(
            ident="math_template_determinant_2x2_gpu",
            name="2x2 determinant template",
            math_type="Precalculus",
            template_program="ARG_A ARG_D * ARG_B ARG_C * -",
            aliases=["2x2 determinant", "ad minus bc"],
            keywords=["matrix", "determinant", "2x2"],
            semantics="compute a 2x2 determinant",
            anchors=[
                "determinant of a 2x2 matrix",
                "ad minus bc",
                "2 by 2 determinant",
            ],
            arg_keys=["a", "b", "c", "d"],
        ),
        _direct_template_spec(
            ident="math_template_dot_product_2d_gpu",
            name="2D dot-product template",
            math_type="Precalculus",
            template_program="ARG_X1 ARG_X2 * ARG_Y1 ARG_Y2 * +",
            aliases=["dot product", "vector dot product"],
            keywords=["vector", "dot", "product"],
            semantics="compute a 2D dot product",
            anchors=[
                "dot product of two vectors",
                "vector dot product problem",
                "multiply corresponding components and add",
            ],
            arg_keys=["x1", "y1", "x2", "y2"],
        ),
        _direct_template_spec(
            ident="math_template_system_2x2_x_gpu",
            name="2x2 linear-system x-solution template",
            math_type="Algebra",
            template_program="ARG_C1 ARG_B2 * ARG_C2 ARG_B1 * - ARG_A1 ARG_B2 * ARG_A2 ARG_B1 * - /",
            aliases=["solve system for x", "cramers rule x"],
            keywords=["system", "equations", "x", "cramers"],
            semantics="compute the x solution for a 2x2 linear system",
            anchors=[
                "solve a system of two equations for x",
                "cramers rule x value",
                "two variable linear system x",
            ],
            arg_keys=["a1", "b1", "c1", "a2", "b2", "c2"],
        ),
        _direct_template_spec(
            ident="math_template_system_2x2_y_gpu",
            name="2x2 linear-system y-solution template",
            math_type="Algebra",
            template_program="ARG_A1 ARG_C2 * ARG_A2 ARG_C1 * - ARG_A1 ARG_B2 * ARG_A2 ARG_B1 * - /",
            aliases=["solve system for y", "cramers rule y"],
            keywords=["system", "equations", "y", "cramers"],
            semantics="compute the y solution for a 2x2 linear system",
            anchors=[
                "solve a system of two equations for y",
                "cramers rule y value",
                "two variable linear system y",
            ],
            arg_keys=["a1", "b1", "c1", "a2", "b2", "c2"],
        ),
        _direct_template_spec(
            ident="math_template_completing_square_vertex_x_gpu",
            name="Completing-square vertex-x template",
            math_type="Algebra",
            template_program="0 ARG_B - 2 /",
            aliases=["vertex x coordinate", "negative b over two"],
            keywords=["quadratic", "vertex", "x coordinate"],
            semantics="compute the x-coordinate of a quadratic vertex from x^2 + bx + c",
            anchors=[
                "vertex x coordinate after completing the square",
                "negative b over two",
                "quadratic vertex x value",
            ],
            arg_keys=["b"],
        ),
        _direct_template_spec(
            ident="math_template_completing_square_vertex_y_gpu",
            name="Completing-square vertex-y template",
            math_type="Algebra",
            template_program="ARG_C ARG_B 2 pow 4 / -",
            aliases=["vertex y coordinate", "c minus b squared over four"],
            keywords=["quadratic", "vertex", "y coordinate"],
            semantics="compute the y-coordinate of a quadratic vertex from x^2 + bx + c",
            anchors=[
                "vertex y coordinate after completing the square",
                "c minus b squared over four",
                "quadratic vertex y value",
            ],
            arg_keys=["b", "c"],
        ),
        _direct_template_spec(
            ident="math_template_base_to_decimal_two_digit_gpu",
            name="Two-digit base-conversion template",
            math_type="Number Theory",
            template_program="ARG_D1 ARG_BASE * ARG_D0 +",
            aliases=["base conversion", "two digit base n"],
            keywords=["base", "conversion", "digits"],
            semantics="convert a two-digit base-n numeral into decimal",
            anchors=[
                "convert a two digit base n number to decimal",
                "base conversion with two digits",
                "weighted digit sum in base n",
            ],
            arg_keys=["d1", "d0", "base"],
        ),
        _direct_template_spec(
            ident="math_template_trapezoid_area_gpu",
            name="Trapezoid-area template",
            math_type="Geometry",
            template_program="ARG_B1 ARG_B2 + ARG_HEIGHT * 2 /",
            aliases=["trapezoid area"],
            keywords=["trapezoid", "area", "bases", "height"],
            semantics="compute trapezoid area",
            anchors=[
                "area of a trapezoid",
                "sum the bases times height over two",
                "trapezoid base and height problem",
            ],
            arg_keys=["b1", "b2", "height"],
        ),
        _direct_template_spec(
            ident="math_template_annulus_area_gpu",
            name="Annulus-area template",
            math_type="Geometry",
            template_program="3.141592653589793 ARG_R_OUTER 2 pow ARG_R_INNER 2 pow - *",
            aliases=["annulus area", "ring area"],
            keywords=["annulus", "area", "ring"],
            semantics="compute annulus area from outer and inner radii",
            anchors=[
                "area of an annulus",
                "ring shaped region area",
                "difference of outer and inner circles",
            ],
            arg_keys=["r_outer", "r_inner"],
        ),
        _direct_template_spec(
            ident="math_template_regular_polygon_area_gpu",
            name="Regular-polygon area template",
            math_type="Geometry",
            template_program="ARG_APOTHEM ARG_PERIMETER * 2 /",
            aliases=["regular polygon area", "apothem perimeter over two"],
            keywords=["polygon", "area", "apothem", "perimeter"],
            semantics="compute regular polygon area from apothem and perimeter",
            anchors=[
                "area of a regular polygon",
                "apothem and perimeter polygon area",
                "regular polygon formula",
            ],
            arg_keys=["apothem", "perimeter"],
        ),
        _direct_template_spec(
            ident="math_template_unit_conversion_scale_gpu",
            name="Scale unit-conversion template",
            math_type="Prealgebra",
            template_program="ARG_VALUE ARG_SCALE * ARG_OFFSET +",
            aliases=["unit conversion", "scale conversion"],
            keywords=["unit", "conversion", "scale"],
            semantics="apply a linear scale-and-offset unit conversion",
            anchors=[
                "convert one unit to another with a scale factor",
                "unit conversion scale problem",
                "multiply by scale and add offset",
            ],
            arg_keys=["value", "scale", "offset"],
        ),
        _direct_template_spec(
            ident="math_template_unit_conversion_affine_gpu",
            name="Affine unit-conversion template",
            math_type="Prealgebra",
            template_program="ARG_VALUE ARG_SCALE * ARG_OFFSET +",
            aliases=["temperature conversion", "affine conversion"],
            keywords=["unit", "conversion", "affine", "offset"],
            semantics="apply an affine unit conversion with both scale and offset",
            anchors=[
                "temperature conversion with offset",
                "affine unit conversion",
                "multiply then add offset",
            ],
            arg_keys=["value", "scale", "offset"],
        ),
        _direct_template_spec(
            ident="math_template_midpoint_coordinate_sum_gpu",
            name="Midpoint coordinate-sum template",
            math_type="Geometry",
            template_program="ARG_X1 ARG_X2 + ARG_Y1 ARG_Y2 + + 2 /",
            aliases=["midpoint coordinate sum", "average of coordinates"],
            keywords=["midpoint", "coordinates", "sum"],
            semantics="compute the sum of midpoint coordinates",
            anchors=[
                "sum of the midpoint coordinates",
                "midpoint coordinate sum problem",
                "average the endpoints then add coordinates",
            ],
            arg_keys=["x1", "y1", "x2", "y2"],
        ),
        _direct_template_spec(
            ident="math_template_exchange_gap_gpu",
            name="Exchange-gap template",
            math_type="Algebra",
            template_program="ARG_TOTAL_COST ARG_FOREIGN_AMOUNT ARG_EXCHANGE_RATE / -",
            aliases=["exchange shortfall", "currency gap"],
            keywords=["exchange", "shortfall", "currency"],
            semantics="compute the remaining amount after currency conversion",
            anchors=[
                "how much more money is needed after exchange",
                "currency shortfall problem",
                "exchange rate gap",
            ],
            arg_keys=["total_cost", "foreign_amount", "exchange_rate"],
        ),
        _direct_template_spec(
            ident="math_template_rate_scaling_gpu",
            name="Rate-scaling template",
            math_type="Algebra",
            template_program="ARG_BASE_TIME_MINUTES 60 * ARG_TARGET_VOLUME * ARG_BASE_UNITS * ARG_BASE_VOLUME / ARG_TARGET_UNITS /",
            aliases=["rate scaling", "work rate scaling"],
            keywords=["rate", "time", "work", "scaling"],
            semantics="scale time by unit count and target volume for a parallel-rate problem",
            anchors=[
                "if several workers finish in some time how long for fewer workers",
                "parallel rate scaling problem",
                "fill rate or work rate scaling",
            ],
            arg_keys=["base_units", "base_volume", "base_time_minutes", "target_units", "target_volume"],
        ),
    ]
)


def build_template_program_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in TEMPLATE_PROGRAM_SPECS:
        metadata = {
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
        }
        metadata.update(_template_metadata_extras(spec))
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
                "metadata": metadata,
            }
        )
    return entries


def build_template_support_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in TEMPLATE_PROGRAM_SPECS:
        for index, anchor in enumerate(spec["anchors"], start=1):
            metadata = {
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
            }
            metadata.update(_template_metadata_extras(spec))
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
                    "metadata": metadata,
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
