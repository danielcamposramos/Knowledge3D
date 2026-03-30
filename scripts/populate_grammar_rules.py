from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.galaxy_population_utils import upsert_entries  # noqa: E402


BOOTSTRAP_TAG = "phase_e28_grammar_population_v1"
DEFAULT_GALAXY_DIR = Path("/K3D/Knowledge3D.local/galaxies")


def _word_anchor(
    entry_id: str,
    name: str,
    *,
    forms: list[str] | None = None,
    related_character_refs: list[str] | None = None,
) -> dict[str, Any]:
    forms = list(forms or [name])
    return {
        "id": entry_id,
        "name": name,
        "domain": "word",
        "category": "concept_anchor",
        "content": name,
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "forms": forms,
            "language": "en",
            "related_character_refs": list(related_character_refs or []),
        },
        "rpn_program": f"TOKEN {name.replace(' ', '_').upper()}",
    }


def _rule(
    *,
    entry_id: str,
    name: str,
    category: str,
    domain: str,
    content: str,
    rpn_program: str,
    symbol_refs: list[str],
    word_refs: list[str],
    tags: list[str],
    examples: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "domain": "grammar",
        "category": category,
        "layer": 3,
        "content": content,
        "rpn_program": rpn_program,
        "symbol_refs": list(symbol_refs),
        "word_refs": list(word_refs),
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "domain": domain,
            "rule_strength": 1,
            "surface_forms": {
                "en": name.lower(),
                "pt": name.lower(),
            },
            "examples": list(examples or []),
        },
        "tags": list(tags),
    }


WORD_ANCHORS: list[dict[str, Any]] = [
    _word_anchor("word_derivative", "derivative", related_character_refs=["char_math_partial"]),
    _word_anchor("word_integral", "integral", related_character_refs=["char_math_integral"]),
    _word_anchor("word_chain", "chain"),
    _word_anchor("word_product", "product", related_character_refs=["char_math_product", "char_op_multiply"]),
    _word_anchor("word_quotient", "quotient", related_character_refs=["char_op_divide"]),
    _word_anchor("word_power", "power", related_character_refs=["char_op_power"]),
    _word_anchor("word_constant", "constant"),
    _word_anchor("word_sum", "sum", related_character_refs=["char_math_summation", "char_op_plus"]),
    _word_anchor("word_difference", "difference", related_character_refs=["char_op_minus"]),
    _word_anchor("word_substitution", "substitution"),
    _word_anchor("word_logic", "logic"),
    _word_anchor("word_implication", "implication"),
    _word_anchor("word_negation", "negation"),
    _word_anchor("word_conjunction", "conjunction"),
    _word_anchor("word_disjunction", "disjunction"),
    _word_anchor("word_meter", "meter"),
    _word_anchor("word_foot", "foot"),
    _word_anchor("word_celsius", "celsius"),
    _word_anchor("word_fahrenheit", "fahrenheit"),
    _word_anchor("word_kelvin", "kelvin"),
    _word_anchor("word_second", "second"),
    _word_anchor("word_minute", "minute"),
    _word_anchor("word_hour", "hour"),
    _word_anchor("word_radian", "radian"),
    _word_anchor("word_degree", "degree"),
    _word_anchor("word_sine", "sine"),
    _word_anchor("word_cosine", "cosine"),
    _word_anchor("word_tangent", "tangent"),
    _word_anchor("word_identity", "identity"),
    _word_anchor("word_factor", "factor"),
    _word_anchor("word_polynomial", "polynomial"),
]


GENERAL_CALCULUS_RULES = [
    (
        "grammar_math_power_rule",
        "Power Rule",
        "Power rule: d/dx(x^n) = n*x^(n-1)",
        "n RECALL x RECALL n RECALL 1 SUB POW MUL",
        ["char_math_partial", "char_op_power", "char_op_multiply"],
        ["word_derivative", "word_power"],
    ),
    (
        "grammar_math_chain_rule",
        "Chain Rule",
        "Chain rule: d/dx(f(g(x))) = f'(g(x))*g'(x)",
        "g_x RECALL f_prime RECALL APPLY g_prime RECALL x RECALL APPLY MUL",
        ["char_math_partial", "char_op_multiply"],
        ["word_derivative", "word_chain"],
    ),
    (
        "grammar_math_product_rule",
        "Product Rule",
        "Product rule: d/dx(f*g) = f'*g + f*g'",
        "f_prime RECALL g RECALL MUL f RECALL g_prime RECALL MUL ADD",
        ["char_math_partial", "char_op_multiply", "char_op_plus"],
        ["word_derivative", "word_product"],
    ),
    (
        "grammar_math_quotient_rule",
        "Quotient Rule",
        "Quotient rule: d/dx(f/g) = (f'*g - f*g') / g^2",
        "f_prime RECALL g RECALL MUL f RECALL g_prime RECALL MUL SUB g RECALL 2 POW DIV",
        ["char_math_partial", "char_op_divide", "char_op_multiply", "char_op_minus", "char_op_power"],
        ["word_derivative", "word_quotient"],
    ),
    (
        "grammar_math_constant_rule",
        "Constant Rule",
        "Constant rule: d/dx(c) = 0",
        "0",
        ["char_math_partial"],
        ["word_derivative", "word_constant"],
    ),
    (
        "grammar_math_sum_rule",
        "Sum Rule",
        "Sum rule: d/dx(f + g) = f' + g'",
        "f_prime RECALL g_prime RECALL ADD",
        ["char_math_partial", "char_op_plus"],
        ["word_derivative", "word_sum"],
    ),
    (
        "grammar_math_difference_rule",
        "Difference Rule",
        "Difference rule: d/dx(f - g) = f' - g'",
        "f_prime RECALL g_prime RECALL SUB",
        ["char_math_partial", "char_op_minus"],
        ["word_derivative", "word_difference"],
    ),
    (
        "grammar_math_constant_multiple_rule",
        "Constant Multiple Rule",
        "Constant multiple rule: d/dx(c*f) = c*f'",
        "c RECALL f_prime RECALL MUL",
        ["char_math_partial", "char_op_multiply"],
        ["word_derivative", "word_constant", "word_product"],
    ),
    (
        "grammar_math_integration_by_parts",
        "Integration By Parts",
        "Integration by parts: integral(u dv) = u*v - integral(v du)",
        "u RECALL v RECALL MUL integral_v_du RECALL SUB",
        ["char_math_integral", "char_op_multiply", "char_op_minus"],
        ["word_integral", "word_product"],
    ),
    (
        "grammar_math_substitution_rule",
        "Substitution Rule",
        "Substitution rule: integral(f(g(x))*g'(x) dx) = integral(f(u) du)",
        "f_of_u RECALL integral_u RECALL",
        ["char_math_integral", "char_op_multiply"],
        ["word_integral", "word_substitution"],
    ),
]


FUNCTION_DERIVATIVE_RULES = [
    ("exp_e", "Exponential e Rule", "Derivative of e^x is e^x", "exp_x RECALL", ["word_derivative", "word_power"]),
    ("natural_log", "Natural Log Rule", "Derivative of ln(x) is 1/x", "1 x RECALL DIV", ["word_derivative"]),
    ("sqrt", "Square Root Rule", "Derivative of sqrt(x) is 1 / (2*sqrt(x))", "1 2 x RECALL SQRT MUL DIV", ["word_derivative"]),
    ("reciprocal", "Reciprocal Rule", "Derivative of 1/x is -1/x^2", "1 NEG x RECALL 2 POW DIV", ["word_derivative"]),
    ("sin", "Sine Derivative", "Derivative of sin(x) is cos(x)", "x RECALL COS", ["word_derivative", "word_sine", "word_cosine"]),
    ("cos", "Cosine Derivative", "Derivative of cos(x) is -sin(x)", "x RECALL SIN NEG", ["word_derivative", "word_cosine", "word_sine"]),
    ("tan", "Tangent Derivative", "Derivative of tan(x) is sec(x)^2", "x RECALL SEC 2 POW", ["word_derivative", "word_tangent"]),
    ("sec", "Secant Derivative", "Derivative of sec(x) is sec(x)*tan(x)", "x RECALL SEC x RECALL TAN MUL", ["word_derivative", "word_tangent"]),
    ("csc", "Cosecant Derivative", "Derivative of csc(x) is -csc(x)*cot(x)", "x RECALL CSC x RECALL COT MUL NEG", ["word_derivative"]),
    ("cot", "Cotangent Derivative", "Derivative of cot(x) is -csc(x)^2", "x RECALL CSC 2 POW NEG", ["word_derivative"]),
]


ADVANCED_CALCULUS_RULES = [
    (
        "grammar_math_second_derivative",
        "Second Derivative",
        "Second derivative applies the derivative operator twice.",
        "f_prime RECALL DERIVE",
        ["char_math_partial"],
        ["word_derivative"],
    ),
    (
        "grammar_math_inverse_function_derivative",
        "Inverse Function Derivative",
        "Derivative of f^-1(x) is 1 / f'(f^-1(x)).",
        "1 inverse_inner_prime RECALL DIV",
        ["char_math_partial", "char_op_divide"],
        ["word_derivative"],
    ),
    (
        "grammar_math_arcsin_derivative",
        "Arcsin Derivative",
        "Derivative of arcsin(x) is 1 / sqrt(1 - x^2).",
        "1 1 x RECALL 2 POW SUB SQRT DIV",
        ["char_math_partial", "char_op_divide", "char_op_minus", "char_op_power"],
        ["word_derivative"],
    ),
    (
        "grammar_math_arccos_derivative",
        "Arccos Derivative",
        "Derivative of arccos(x) is -1 / sqrt(1 - x^2).",
        "1 NEG 1 x RECALL 2 POW SUB SQRT DIV",
        ["char_math_partial", "char_op_divide", "char_op_minus", "char_op_power"],
        ["word_derivative"],
    ),
    (
        "grammar_math_arctan_derivative",
        "Arctan Derivative",
        "Derivative of arctan(x) is 1 / (1 + x^2).",
        "1 1 x RECALL 2 POW ADD DIV",
        ["char_math_partial", "char_op_divide", "char_op_plus", "char_op_power"],
        ["word_derivative"],
    ),
    (
        "grammar_math_definite_integral_linearity",
        "Definite Integral Linearity",
        "Integral linearity distributes over addition.",
        "integral_f RECALL integral_g RECALL ADD",
        ["char_math_integral", "char_op_plus"],
        ["word_integral", "word_sum"],
    ),
    (
        "grammar_math_polynomial_term_derivative",
        "Polynomial Term Derivative",
        "Differentiate one polynomial term at a time.",
        "coefficient RECALL exponent RECALL x RECALL exponent RECALL 1 SUB POW MUL",
        ["char_math_partial", "char_op_power", "char_op_multiply"],
        ["word_derivative", "word_polynomial"],
    ),
    (
        "grammar_math_power_of_composition",
        "Power Of Composition",
        "Derivative of (g(x))^n is n*(g(x))^(n-1)*g'(x).",
        "n RECALL g_x RECALL n RECALL 1 SUB POW MUL g_prime RECALL MUL",
        ["char_math_partial", "char_op_power", "char_op_multiply"],
        ["word_derivative", "word_power", "word_chain"],
    ),
    (
        "grammar_math_total_differential",
        "Total Differential",
        "Total differential sums partial derivatives over each variable.",
        "df_dx RECALL dx RECALL MUL df_dy RECALL dy RECALL MUL ADD",
        ["char_math_partial", "char_op_multiply", "char_op_plus"],
        ["word_derivative", "word_sum"],
    ),
    (
        "grammar_math_fundamental_theorem_calculus",
        "Fundamental Theorem Of Calculus",
        "Derivative of an accumulated integral recovers the integrand.",
        "integrand RECALL",
        ["char_math_integral", "char_math_partial"],
        ["word_integral", "word_derivative"],
    ),
]


LOGIC_RULES = [
    ("grammar_logic_modus_ponens", "Modus Ponens", "If P implies Q and P is true, infer Q.", "P RECALL P_implies_Q RECALL AND Q STORE"),
    ("grammar_logic_modus_tollens", "Modus Tollens", "If P implies Q and not Q, infer not P.", "Q RECALL NOT P_implies_Q RECALL AND P RECALL NOT STORE"),
    ("grammar_logic_hypothetical_syllogism", "Hypothetical Syllogism", "If P implies Q and Q implies R, infer P implies R.", "P_implies_Q RECALL Q_implies_R RECALL COMPOSE"),
    ("grammar_logic_disjunctive_syllogism", "Disjunctive Syllogism", "From P or Q and not P, infer Q.", "P_or_Q RECALL P RECALL NOT AND Q STORE"),
    ("grammar_logic_constructive_dilemma", "Constructive Dilemma", "From P->R, Q->S, and P or Q, infer R or S.", "P_or_Q RECALL P_implies_R RECALL Q_implies_S RECALL APPLY_DILEMMA"),
    ("grammar_logic_simplification", "Simplification", "From P and Q, infer P.", "P_and_Q RECALL LEFT"),
    ("grammar_logic_conjunction_introduction", "Conjunction Introduction", "From P and Q separately, infer P and Q.", "P RECALL Q RECALL AND"),
    ("grammar_logic_addition", "Addition", "From P, infer P or Q.", "P RECALL Q RECALL OR"),
    ("grammar_logic_resolution", "Resolution", "From P or Q and not P or R, infer Q or R.", "P_or_Q RECALL not_P_or_R RECALL RESOLVE"),
    ("grammar_logic_contrapositive", "Contrapositive", "P implies Q is equivalent to not Q implies not P.", "Q RECALL NOT P RECALL NOT IMPLIES"),
    ("grammar_logic_double_negation", "Double Negation", "Not not P is equivalent to P.", "P RECALL NOT NOT"),
    ("grammar_logic_demorgan_and", "De Morgan And", "Not(A and B) becomes not A or not B.", "A RECALL NOT B RECALL NOT OR"),
    ("grammar_logic_demorgan_or", "De Morgan Or", "Not(A or B) becomes not A and not B.", "A RECALL NOT B RECALL NOT AND"),
    ("grammar_logic_implication_elimination", "Implication Elimination", "P implies Q becomes not P or Q.", "P RECALL NOT Q RECALL OR"),
    ("grammar_logic_biconditional_elimination", "Biconditional Elimination", "P iff Q becomes (P->Q) and (Q->P).", "P_implies_Q RECALL Q_implies_P RECALL AND"),
    ("grammar_logic_excluded_middle", "Excluded Middle", "P or not P is always true.", "P RECALL P RECALL NOT OR"),
    ("grammar_logic_noncontradiction", "Noncontradiction", "P and not P is always false.", "P RECALL P RECALL NOT AND"),
    ("grammar_logic_distributive_and", "Distributive And", "P and (Q or R) becomes (P and Q) or (P and R).", "P RECALL Q RECALL AND P RECALL R RECALL AND OR"),
    ("grammar_logic_distributive_or", "Distributive Or", "P or (Q and R) becomes (P or Q) and (P or R).", "P RECALL Q RECALL OR P RECALL R RECALL OR AND"),
    ("grammar_logic_absorption", "Absorption", "P or (P and Q) reduces to P.", "P RECALL"),
]


UNIT_CONVERSIONS = [
    ("meters_to_feet", "Meters To Feet", "Convert meters to feet using 3.28084.", "value RECALL 3.28084 MUL", ["word_meter", "word_foot"]),
    ("feet_to_meters", "Feet To Meters", "Convert feet to meters using 0.3048.", "value RECALL 0.3048 MUL", ["word_foot", "word_meter"]),
    ("kilometers_to_miles", "Kilometers To Miles", "Convert kilometers to miles using 0.621371.", "value RECALL 0.621371 MUL", ["word_meter"]),
    ("miles_to_kilometers", "Miles To Kilometers", "Convert miles to kilometers using 1.60934.", "value RECALL 1.60934 MUL", ["word_meter"]),
    ("centimeters_to_inches", "Centimeters To Inches", "Convert centimeters to inches using 0.393701.", "value RECALL 0.393701 MUL", ["word_meter"]),
    ("inches_to_centimeters", "Inches To Centimeters", "Convert inches to centimeters using 2.54.", "value RECALL 2.54 MUL", ["word_meter"]),
    ("kilograms_to_pounds", "Kilograms To Pounds", "Convert kilograms to pounds using 2.20462.", "value RECALL 2.20462 MUL", []),
    ("pounds_to_kilograms", "Pounds To Kilograms", "Convert pounds to kilograms using 0.453592.", "value RECALL 0.453592 MUL", []),
    ("grams_to_kilograms", "Grams To Kilograms", "Convert grams to kilograms by dividing by 1000.", "value RECALL 1000 DIV", []),
    ("kilograms_to_grams", "Kilograms To Grams", "Convert kilograms to grams by multiplying by 1000.", "value RECALL 1000 MUL", []),
    ("celsius_to_fahrenheit", "Celsius To Fahrenheit", "F = C * 9/5 + 32.", "value RECALL 9 MUL 5 DIV 32 ADD", ["word_celsius", "word_fahrenheit"]),
    ("fahrenheit_to_celsius", "Fahrenheit To Celsius", "C = (F - 32) * 5/9.", "value RECALL 32 SUB 5 MUL 9 DIV", ["word_fahrenheit", "word_celsius"]),
    ("celsius_to_kelvin", "Celsius To Kelvin", "K = C + 273.15.", "value RECALL 273.15 ADD", ["word_celsius", "word_kelvin"]),
    ("kelvin_to_celsius", "Kelvin To Celsius", "C = K - 273.15.", "value RECALL 273.15 SUB", ["word_kelvin", "word_celsius"]),
    ("seconds_to_minutes", "Seconds To Minutes", "Convert seconds to minutes by dividing by 60.", "value RECALL 60 DIV", ["word_second", "word_minute"]),
    ("minutes_to_hours", "Minutes To Hours", "Convert minutes to hours by dividing by 60.", "value RECALL 60 DIV", ["word_minute", "word_hour"]),
    ("hours_to_minutes", "Hours To Minutes", "Convert hours to minutes by multiplying by 60.", "value RECALL 60 MUL", ["word_hour", "word_minute"]),
    ("radians_to_degrees", "Radians To Degrees", "Degrees = radians * 180 / pi.", "value RECALL 180 MUL pi RECALL DIV", ["word_radian", "word_degree"]),
    ("degrees_to_radians", "Degrees To Radians", "Radians = degrees * pi / 180.", "value RECALL pi RECALL MUL 180 DIV", ["word_degree", "word_radian"]),
    ("meters_per_second_to_kph", "Meters Per Second To KPH", "Multiply meters per second by 3.6.", "value RECALL 3.6 MUL", ["word_meter"]),
]


ALGEBRA_RULES = [
    ("grammar_algebra_distributive", "Distributive Property", "a*(b + c) = a*b + a*c", "a RECALL b RECALL c RECALL ADD MUL"),
    ("grammar_algebra_factor_common", "Factor Common Term", "ab + ac = a(b + c)", "ab RECALL ac RECALL FACTOR_COMMON"),
    ("grammar_algebra_difference_squares", "Difference Of Squares", "a^2 - b^2 = (a-b)(a+b)", "a RECALL 2 POW b RECALL 2 POW SUB"),
    ("grammar_algebra_square_sum", "Square Of Sum", "(a+b)^2 = a^2 + 2ab + b^2", "a RECALL b RECALL ADD 2 POW"),
    ("grammar_algebra_square_difference", "Square Of Difference", "(a-b)^2 = a^2 - 2ab + b^2", "a RECALL b RECALL SUB 2 POW"),
    ("grammar_algebra_binomial_product", "Binomial Product", "(a+b)(a-b) = a^2 - b^2", "a RECALL b RECALL ADD a RECALL b RECALL SUB MUL"),
    ("grammar_algebra_common_denominator", "Common Denominator", "a/b + c/d = (ad + bc) / bd", "a RECALL d RECALL MUL b RECALL c RECALL MUL ADD b RECALL d RECALL MUL DIV"),
    ("grammar_algebra_reciprocal_multiplication", "Reciprocal Multiplication", "a / (b/c) = a * c / b", "a RECALL c RECALL MUL b RECALL DIV"),
    ("grammar_algebra_combine_like_terms", "Combine Like Terms", "ax + bx = (a+b)x", "a RECALL b RECALL ADD x RECALL MUL"),
    ("grammar_algebra_isolate_addition", "Isolate Variable By Addition", "x + a = b implies x = b - a", "b RECALL a RECALL SUB"),
    ("grammar_algebra_isolate_multiplication", "Isolate Variable By Multiplication", "a*x = b implies x = b / a", "b RECALL a RECALL DIV"),
    ("grammar_algebra_zero_product", "Zero Product Property", "If a*b = 0 then a = 0 or b = 0", "a RECALL b RECALL MUL 0 EQ"),
    ("grammar_algebra_exponent_product", "Exponent Product Rule", "a^m * a^n = a^(m+n)", "a RECALL m RECALL POW a RECALL n RECALL POW MUL"),
    ("grammar_algebra_exponent_quotient", "Exponent Quotient Rule", "a^m / a^n = a^(m-n)", "a RECALL m RECALL POW a RECALL n RECALL POW DIV"),
    ("grammar_algebra_power_of_power", "Power Of Power", "(a^m)^n = a^(m*n)", "a RECALL m RECALL POW n RECALL POW"),
]


TRIG_IDENTITIES = [
    ("grammar_trig_pythagorean", "Pythagorean Identity", "sin^2(x) + cos^2(x) = 1", "x RECALL SIN 2 POW x RECALL COS 2 POW ADD"),
    ("grammar_trig_tan_ratio", "Tangent Ratio", "tan(x) = sin(x) / cos(x)", "x RECALL SIN x RECALL COS DIV"),
    ("grammar_trig_cot_ratio", "Cotangent Ratio", "cot(x) = cos(x) / sin(x)", "x RECALL COS x RECALL SIN DIV"),
    ("grammar_trig_sec_reciprocal", "Secant Reciprocal", "sec(x) = 1 / cos(x)", "1 x RECALL COS DIV"),
    ("grammar_trig_csc_reciprocal", "Cosecant Reciprocal", "csc(x) = 1 / sin(x)", "1 x RECALL SIN DIV"),
    ("grammar_trig_angle_sum_sin", "Sine Angle Sum", "sin(a+b) = sin(a)cos(b) + cos(a)sin(b)", "a RECALL SIN b RECALL COS MUL a RECALL COS b RECALL SIN MUL ADD"),
    ("grammar_trig_angle_sum_cos", "Cosine Angle Sum", "cos(a+b) = cos(a)cos(b) - sin(a)sin(b)", "a RECALL COS b RECALL COS MUL a RECALL SIN b RECALL SIN MUL SUB"),
    ("grammar_trig_double_angle_sin", "Sine Double Angle", "sin(2x) = 2sin(x)cos(x)", "2 x RECALL SIN MUL x RECALL COS MUL"),
    ("grammar_trig_double_angle_cos", "Cosine Double Angle", "cos(2x) = cos^2(x) - sin^2(x)", "x RECALL COS 2 POW x RECALL SIN 2 POW SUB"),
    ("grammar_trig_half_angle_sin", "Sine Half Angle", "sin(x/2) = sqrt((1-cos(x))/2)", "1 x RECALL COS SUB 2 DIV SQRT"),
    ("grammar_trig_half_angle_cos", "Cosine Half Angle", "cos(x/2) = sqrt((1+cos(x))/2)", "1 x RECALL COS ADD 2 DIV SQRT"),
    ("grammar_trig_product_to_sum_sin_cos", "Product To Sum Sin Cos", "sin(a)cos(b) = 1/2[sin(a+b)+sin(a-b)]", "a RECALL b RECALL ADD SIN a RECALL b RECALL SUB SIN ADD 2 DIV"),
    ("grammar_trig_product_to_sum_sin_sin", "Product To Sum Sin Sin", "sin(a)sin(b) = 1/2[cos(a-b)-cos(a+b)]", "a RECALL b RECALL SUB COS a RECALL b RECALL ADD COS SUB 2 DIV"),
    ("grammar_trig_product_to_sum_cos_cos", "Product To Sum Cos Cos", "cos(a)cos(b) = 1/2[cos(a-b)+cos(a+b)]", "a RECALL b RECALL SUB COS a RECALL b RECALL ADD COS ADD 2 DIV"),
    ("grammar_trig_phase_shift", "Phase Shift Identity", "sin(x + pi/2) = cos(x)", "x RECALL pi RECALL 2 DIV ADD SIN"),
]


def _calculus_rules() -> list[dict[str, Any]]:
    rows = [
        _rule(
            entry_id=entry_id,
            name=name,
            category="math_transform",
            domain="calculus_differentiation",
            content=content,
            rpn_program=rpn_program,
            symbol_refs=symbol_refs,
            word_refs=word_refs,
            tags=["grammar", "math", "calculus"],
        )
        for entry_id, name, content, rpn_program, symbol_refs, word_refs in GENERAL_CALCULUS_RULES
    ]
    for suffix, name, content, rpn_program, word_refs in FUNCTION_DERIVATIVE_RULES:
        rows.append(
            _rule(
                entry_id=f"grammar_math_{suffix}_derivative",
                name=name,
                category="math_transform",
                domain="calculus_differentiation",
                content=content,
                rpn_program=rpn_program,
                symbol_refs=["char_math_partial", "char_op_power"],
                word_refs=word_refs,
                tags=["grammar", "math", "calculus", "function_rule"],
            )
        )
    rows.extend(
        [
            _rule(
                entry_id=entry_id,
                name=name,
                category="math_transform",
                domain="calculus_differentiation",
                content=content,
                rpn_program=rpn_program,
                symbol_refs=symbol_refs,
                word_refs=word_refs,
                tags=["grammar", "math", "calculus", "advanced"],
            )
            for entry_id, name, content, rpn_program, symbol_refs, word_refs in ADVANCED_CALCULUS_RULES
        ]
    )
    return rows


def _logic_rules() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry_id, name, content, rpn_program in LOGIC_RULES:
        rows.append(
            _rule(
                entry_id=entry_id,
                name=name,
                category="logic_rule",
                domain="formal_logic",
                content=content,
                rpn_program=rpn_program,
                symbol_refs=["char_op_equals", "char_op_greater_than"],
                word_refs=["word_logic", "word_implication", "word_negation", "word_conjunction", "word_disjunction"],
                tags=["grammar", "logic", "formal_rule"],
            )
        )
    return rows


def _unit_conversion_rules() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suffix, name, content, rpn_program, word_refs in UNIT_CONVERSIONS:
        rows.append(
            _rule(
                entry_id=f"grammar_unit_{suffix}",
                name=name,
                category="unit_conversion",
                domain="measurement_conversion",
                content=content,
                rpn_program=rpn_program,
                symbol_refs=["char_op_multiply", "char_op_divide", "char_op_plus", "char_op_minus", "char_math_pi"],
                word_refs=word_refs,
                tags=["grammar", "conversion", "units"],
            )
        )
    return rows


def _algebra_rules() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry_id, name, content, rpn_program in ALGEBRA_RULES:
        rows.append(
            _rule(
                entry_id=entry_id,
                name=name,
                category="algebra_identity",
                domain="algebraic_transformation",
                content=content,
                rpn_program=rpn_program,
                symbol_refs=["char_op_plus", "char_op_minus", "char_op_multiply", "char_op_divide", "char_op_power", "char_op_equals"],
                word_refs=["word_factor", "word_identity", "word_polynomial"],
                tags=["grammar", "math", "algebra"],
            )
        )
    return rows


def _trig_rules() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry_id, name, content, rpn_program in TRIG_IDENTITIES:
        rows.append(
            _rule(
                entry_id=entry_id,
                name=name,
                category="trig_identity",
                domain="trigonometric_transformation",
                content=content,
                rpn_program=rpn_program,
                symbol_refs=["char_math_pi", "char_math_theta", "char_op_plus", "char_op_minus", "char_op_divide", "char_op_power", "char_op_equals"],
                word_refs=["word_sine", "word_cosine", "word_tangent", "word_identity", "word_degree", "word_radian"],
                tags=["grammar", "math", "trigonometry"],
            )
        )
    return rows


def build_grammar_rule_entries() -> list[dict[str, Any]]:
    rows = _calculus_rules() + _logic_rules() + _unit_conversion_rules() + _algebra_rules() + _trig_rules()
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 grammar rules, generated {len(rows)}")
    return rows


def build_word_anchor_entries() -> list[dict[str, Any]]:
    return [dict(row) for row in WORD_ANCHORS]


def populate_grammar_rules(*, galaxy_dir: Path = DEFAULT_GALAXY_DIR) -> dict[str, dict[str, int]]:
    galaxy_dir = Path(galaxy_dir)
    return {
        "Grammar.jsonl": upsert_entries(
            galaxy_dir / "Grammar.jsonl",
            build_grammar_rule_entries(),
        ),
        "Word.jsonl": upsert_entries(
            galaxy_dir / "Word.jsonl",
            build_word_anchor_entries(),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate Grammar galaxy with foundational rules.")
    parser.add_argument(
        "--galaxy-dir",
        type=Path,
        default=DEFAULT_GALAXY_DIR,
        help="Directory containing galaxy JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = populate_grammar_rules(galaxy_dir=args.galaxy_dir)
    for name in ("Grammar.jsonl", "Word.jsonl"):
        stats = summary[name]
        print(
            f"{name}:"
            f" before={stats['before']}"
            f" after={stats['after']}"
            f" appended={stats['appended']}"
            f" replaced={stats['replaced']}"
            f" removed={stats['removed']}"
        )


if __name__ == "__main__":
    main()
