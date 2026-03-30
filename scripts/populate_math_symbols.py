from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.galaxy_population_utils import upsert_entries  # noqa: E402


BOOTSTRAP_TAG = "phase_e28_character_population_v1"
DEFAULT_GALAXY_DIR = Path("/K3D/Knowledge3D.local/galaxies")


def _char_entry(
    entry_id: str,
    name: str,
    glyph: str,
    unicode_code: str,
    category: str,
    visual_rpn: str,
    word_refs: list[str],
    tags: list[str],
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "domain": "character",
        "category": category,
        "layer": 1,
        "content": f"{name} {unicode_code}",
        "description": f"Procedural glyph for {glyph} ({unicode_code}).",
        "visual_rpn": visual_rpn,
        "word_refs": list(word_refs),
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "unicode": unicode_code,
            "glyph": glyph,
            "surface_forms": {
                "en": name.lower(),
                "pt": name.lower(),
            },
            "word_refs": list(word_refs),
        },
        "tags": list(tags),
    }


MATH_SYMBOLS: list[dict[str, Any]] = [
    _char_entry("char_math_summation", "Summation Symbol", "∑", "U+2211", "math_symbol", "32 8 DRAW_MOVE 8 8 DRAW_LINE 32 32 DRAW_LINE 8 56 DRAW_LINE 56 56 DRAW_LINE DRAW_STROKE", ["word_sum"], ["math", "symbol", "summation"]),
    _char_entry("char_math_product", "Product Symbol", "∏", "U+220F", "math_symbol", "12 8 DRAW_MOVE 12 56 DRAW_LINE 20 8 DRAW_MOVE 44 8 DRAW_LINE 44 8 DRAW_MOVE 44 56 DRAW_LINE DRAW_STROKE", ["word_product"], ["math", "symbol", "product"]),
    _char_entry("char_math_integral", "Integral Symbol", "∫", "U+222B", "math_symbol", "40 8 DRAW_MOVE 28 16 CURVE 24 32 CURVE 28 48 CURVE 16 56 DRAW_LINE DRAW_STROKE", ["word_integral"], ["math", "symbol", "integral"]),
    _char_entry("char_math_partial", "Partial Derivative Symbol", "∂", "U+2202", "math_symbol", "40 16 DRAW_MOVE 24 12 CURVE 16 28 CURVE 24 48 CURVE 40 44 DRAW_LINE 40 16 DRAW_LINE DRAW_STROKE", ["word_derivative"], ["math", "symbol", "partial"]),
    _char_entry("char_math_nabla", "Nabla Symbol", "∇", "U+2207", "math_symbol", "32 8 DRAW_MOVE 8 56 DRAW_LINE 56 56 DRAW_LINE 32 8 DRAW_LINE DRAW_STROKE", ["word_derivative"], ["math", "symbol", "nabla"]),
    _char_entry("char_math_infinity", "Infinity Symbol", "∞", "U+221E", "math_symbol", "16 32 DRAW_MOVE 8 20 CURVE 20 12 CURVE 32 32 CURVE 44 52 CURVE 56 32 CURVE 44 12 CURVE 32 32 CURVE 20 52 CURVE 8 32 CURVE DRAW_STROKE", ["word_identity"], ["math", "symbol", "infinity"]),
    _char_entry("char_math_pi", "Pi Symbol", "π", "U+03C0", "math_symbol", "12 16 DRAW_MOVE 52 16 DRAW_LINE 24 16 DRAW_MOVE 24 56 DRAW_LINE 40 16 DRAW_MOVE 40 56 DRAW_LINE DRAW_STROKE", ["word_identity"], ["math", "symbol", "pi"]),
    _char_entry("char_math_theta", "Theta Symbol", "θ", "U+03B8", "math_symbol", "32 8 DRAW_MOVE 16 16 CURVE 8 32 CURVE 16 48 CURVE 32 56 CURVE 48 48 CURVE 56 32 CURVE 48 16 CURVE 32 8 CURVE 16 32 DRAW_MOVE 48 32 DRAW_LINE DRAW_STROKE", ["word_identity"], ["math", "symbol", "theta"]),
    _char_entry("char_math_lambda", "Lambda Symbol", "λ", "U+03BB", "math_symbol", "20 8 DRAW_MOVE 36 32 DRAW_LINE 20 56 DRAW_LINE 28 36 DRAW_MOVE 44 56 DRAW_LINE DRAW_STROKE", ["word_identity"], ["math", "symbol", "lambda"]),
    _char_entry("char_math_mu", "Mu Symbol", "μ", "U+03BC", "math_symbol", "16 16 DRAW_MOVE 16 48 DRAW_LINE 32 24 DRAW_MOVE 32 48 DRAW_LINE 48 16 DRAW_MOVE 48 56 DRAW_LINE DRAW_STROKE", ["word_identity"], ["math", "symbol", "mu"]),
    _char_entry("char_math_sigma", "Sigma Symbol", "σ", "U+03C3", "math_symbol", "16 16 DRAW_MOVE 48 16 DRAW_LINE 56 32 CURVE 48 48 CURVE 16 48 DRAW_LINE 8 32 CURVE 16 16 CURVE DRAW_STROKE", ["word_identity"], ["math", "symbol", "sigma"]),
    _char_entry("char_math_delta", "Delta Symbol", "Δ", "U+0394", "math_symbol", "32 8 DRAW_MOVE 8 56 DRAW_LINE 56 56 DRAW_LINE 32 8 DRAW_LINE DRAW_STROKE", ["word_difference"], ["math", "symbol", "delta"]),
    _char_entry("char_math_alpha", "Alpha Symbol", "α", "U+03B1", "math_symbol", "28 20 DRAW_MOVE 16 28 CURVE 20 44 CURVE 36 44 CURVE 44 28 CURVE 36 16 CURVE 28 20 CURVE 44 20 DRAW_MOVE 44 56 DRAW_LINE DRAW_STROKE", ["word_identity"], ["math", "symbol", "alpha"]),
    _char_entry("char_math_beta", "Beta Symbol", "β", "U+03B2", "math_symbol", "28 8 DRAW_MOVE 28 56 DRAW_LINE 28 8 DRAW_MOVE 44 20 CURVE 28 32 CURVE 44 44 CURVE 28 56 CURVE DRAW_STROKE", ["word_identity"], ["math", "symbol", "beta"]),
    _char_entry("char_math_element_of", "Element Of Symbol", "∈", "U+2208", "math_symbol", "48 12 DRAW_MOVE 24 12 DRAW_LINE 12 32 DRAW_LINE 24 52 DRAW_LINE 48 52 DRAW_LINE 20 32 DRAW_MOVE 40 32 DRAW_LINE DRAW_STROKE", ["word_identity"], ["math", "symbol", "membership"]),
    _char_entry("char_math_not_equal", "Not Equal Symbol", "≠", "U+2260", "math_symbol", "12 20 DRAW_MOVE 52 20 DRAW_LINE 12 44 DRAW_MOVE 52 44 DRAW_LINE 20 52 DRAW_MOVE 44 12 DRAW_LINE DRAW_STROKE", ["word_identity"], ["math", "symbol", "inequality"]),
    _char_entry("char_math_approximately_equal", "Approximately Equal Symbol", "≈", "U+2248", "math_symbol", "12 22 DRAW_MOVE 20 18 CURVE 28 26 CURVE 36 22 CURVE 44 18 CURVE 52 22 CURVE 12 42 DRAW_MOVE 20 38 CURVE 28 46 CURVE 36 42 CURVE 44 38 CURVE 52 42 CURVE DRAW_STROKE", ["word_identity"], ["math", "symbol", "approximation"]),
]


DIGITS: list[dict[str, Any]] = [
    _char_entry("char_digit_0", "Digit Zero", "0", "U+0030", "digit_symbol", "20 12 DRAW_MOVE 12 32 CURVE 20 52 CURVE 44 52 CURVE 52 32 CURVE 44 12 CURVE 20 12 CURVE DRAW_STROKE", ["word_zero"], ["digit", "number", "zero"]),
    _char_entry("char_digit_1", "Digit One", "1", "U+0031", "digit_symbol", "28 16 DRAW_MOVE 36 8 DRAW_LINE 36 56 DRAW_LINE 24 56 DRAW_MOVE 48 56 DRAW_LINE DRAW_STROKE", ["word_one"], ["digit", "number", "one"]),
    _char_entry("char_digit_2", "Digit Two", "2", "U+0032", "digit_symbol", "16 20 DRAW_MOVE 28 8 CURVE 48 16 CURVE 44 32 CURVE 16 56 DRAW_LINE 52 56 DRAW_LINE DRAW_STROKE", ["word_two"], ["digit", "number", "two"]),
    _char_entry("char_digit_3", "Digit Three", "3", "U+0033", "digit_symbol", "16 16 DRAW_MOVE 40 12 CURVE 28 32 CURVE 40 52 CURVE 16 48 CURVE DRAW_STROKE", ["word_three"], ["digit", "number", "three"]),
    _char_entry("char_digit_4", "Digit Four", "4", "U+0034", "digit_symbol", "44 8 DRAW_MOVE 12 40 DRAW_LINE 52 40 DRAW_LINE 44 8 DRAW_MOVE 44 56 DRAW_LINE DRAW_STROKE", ["word_four"], ["digit", "number", "four"]),
    _char_entry("char_digit_5", "Digit Five", "5", "U+0035", "digit_symbol", "48 12 DRAW_MOVE 20 12 DRAW_LINE 16 32 DRAW_LINE 36 32 DRAW_LINE 48 40 CURVE 36 56 CURVE 16 48 CURVE DRAW_STROKE", ["word_five"], ["digit", "number", "five"]),
    _char_entry("char_digit_6", "Digit Six", "6", "U+0036", "digit_symbol", "44 12 DRAW_MOVE 24 24 CURVE 16 40 CURVE 28 56 CURVE 44 44 CURVE 32 28 CURVE 16 32 CURVE DRAW_STROKE", ["word_six"], ["digit", "number", "six"]),
    _char_entry("char_digit_7", "Digit Seven", "7", "U+0037", "digit_symbol", "12 12 DRAW_MOVE 52 12 DRAW_LINE 28 56 DRAW_LINE DRAW_STROKE", ["word_seven"], ["digit", "number", "seven"]),
    _char_entry("char_digit_8", "Digit Eight", "8", "U+0038", "digit_symbol", "28 8 DRAW_MOVE 16 20 CURVE 28 32 CURVE 16 44 CURVE 28 56 CURVE 44 44 CURVE 32 32 CURVE 44 20 CURVE 28 8 CURVE DRAW_STROKE", ["word_eight"], ["digit", "number", "eight"]),
    _char_entry("char_digit_9", "Digit Nine", "9", "U+0039", "digit_symbol", "20 52 DRAW_MOVE 40 40 CURVE 48 24 CURVE 36 8 CURVE 20 20 CURVE 32 36 CURVE 48 32 CURVE DRAW_STROKE", ["word_nine"], ["digit", "number", "nine"]),
]


OPERATORS: list[dict[str, Any]] = [
    _char_entry("char_op_plus", "Plus Operator", "+", "U+002B", "operator_symbol", "32 12 DRAW_MOVE 32 52 DRAW_LINE 12 32 DRAW_MOVE 52 32 DRAW_LINE DRAW_STROKE", ["word_sum"], ["operator", "plus", "addition"]),
    _char_entry("char_op_minus", "Minus Operator", "-", "U+2212", "operator_symbol", "12 32 DRAW_MOVE 52 32 DRAW_LINE DRAW_STROKE", ["word_difference"], ["operator", "minus", "subtraction"]),
    _char_entry("char_op_multiply", "Multiply Operator", "×", "U+00D7", "operator_symbol", "16 16 DRAW_MOVE 48 48 DRAW_LINE 48 16 DRAW_MOVE 16 48 DRAW_LINE DRAW_STROKE", ["word_product"], ["operator", "multiply"]),
    _char_entry("char_op_divide", "Divide Operator", "÷", "U+00F7", "operator_symbol", "12 32 DRAW_MOVE 52 32 DRAW_LINE 32 16 DRAW_MOVE 32 16 DRAW_LINE 32 48 DRAW_MOVE 32 48 DRAW_LINE DRAW_STROKE", ["word_quotient"], ["operator", "divide"]),
    _char_entry("char_op_equals", "Equals Operator", "=", "U+003D", "operator_symbol", "12 22 DRAW_MOVE 52 22 DRAW_LINE 12 42 DRAW_MOVE 52 42 DRAW_LINE DRAW_STROKE", ["word_identity"], ["operator", "equals"]),
    _char_entry("char_op_less_than", "Less Than Operator", "<", "U+003C", "operator_symbol", "44 12 DRAW_MOVE 16 32 DRAW_LINE 44 52 DRAW_LINE DRAW_STROKE", ["word_identity"], ["operator", "comparison", "less"]),
    _char_entry("char_op_greater_than", "Greater Than Operator", ">", "U+003E", "operator_symbol", "20 12 DRAW_MOVE 48 32 DRAW_LINE 20 52 DRAW_LINE DRAW_STROKE", ["word_identity"], ["operator", "comparison", "greater"]),
    _char_entry("char_op_power", "Power Operator", "^", "U+005E", "operator_symbol", "20 36 DRAW_MOVE 32 16 DRAW_LINE 44 36 DRAW_LINE DRAW_STROKE", ["word_power"], ["operator", "power", "exponent"]),
]


def build_math_symbol_entries() -> list[dict[str, Any]]:
    rows = [dict(row) for row in MATH_SYMBOLS + DIGITS + OPERATORS]
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 character entries, generated {len(rows)}")
    return rows


def populate_math_symbols(*, galaxy_dir: Path = DEFAULT_GALAXY_DIR) -> dict[str, dict[str, int]]:
    galaxy_dir = Path(galaxy_dir)
    return {
        "Character.jsonl": upsert_entries(
            galaxy_dir / "Character.jsonl",
            build_math_symbol_entries(),
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate Character galaxy with math symbols, digits, and operators.")
    parser.add_argument(
        "--galaxy-dir",
        type=Path,
        default=DEFAULT_GALAXY_DIR,
        help="Directory containing galaxy JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = populate_math_symbols(galaxy_dir=args.galaxy_dir)
    stats = summary["Character.jsonl"]
    print(
        "Character.jsonl:"
        f" before={stats['before']}"
        f" after={stats['after']}"
        f" appended={stats['appended']}"
        f" replaced={stats['replaced']}"
        f" removed={stats['removed']}"
    )


if __name__ == "__main__":
    main()
