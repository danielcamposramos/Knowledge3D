"""Bootstrap foundational deterministic operations into Grammar/Math galaxies."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_NUMBER_WORD_UNITS: dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

_NUMBER_WORD_TENS: dict[int, str] = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def _number_to_words(value: int) -> str:
    if value < 20:
        return _NUMBER_WORD_UNITS[value]
    if value < 100:
        tens = (value // 10) * 10
        unit = value % 10
        if unit == 0:
            return _NUMBER_WORD_TENS[tens]
        return f"{_NUMBER_WORD_TENS[tens]}-{_NUMBER_WORD_UNITS[unit]}"
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        if remainder == 0:
            return f"{_NUMBER_WORD_UNITS[hundreds]} hundred"
        return f"{_NUMBER_WORD_UNITS[hundreds]} hundred {_number_to_words(remainder)}"
    if value == 1000:
        return "one thousand"
    raise ValueError("foundational number bootstrap capped at 1000")


def _word_entry_id(word: str) -> str:
    return f"word_{word.replace('-', '_').replace(' ', '_')}"


def _number_word_entries(max_value: int = 1000) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    number_entries: list[dict[str, Any]] = []
    word_entries: list[dict[str, Any]] = []
    for value in range(max_value + 1):
        word = _number_to_words(value)
        word_id = _word_entry_id(word)
        number_id = f"num_{value}"
        number_entries.append(
            {
                "id": number_id,
                "name": str(value),
                "domain": "number",
                "category": "integer",
                "rpn_program": str(value),
                "metadata": {
                    "value": value,
                    "word_ref": word_id,
                    "char_refs": [f"char_{ch}" for ch in str(value)],
                    "forms": [str(value), word],
                    "bootstrap": "deterministic_foundation_v2",
                    "save_information_principle": True,
                },
            }
        )
        word_entries.append(
            {
                "id": word_id,
                "name": word,
                "domain": "word",
                "category": "numeric_lexeme",
                "rpn_program": f"LOOKUP {number_id}",
                "metadata": {
                    "value": value,
                    "number_ref": number_id,
                    "forms": [word, word.replace("-", " "), str(value)],
                    "is_numeric_word": True,
                    "bootstrap": "deterministic_foundation_v2",
                    "save_information_principle": True,
                },
            }
        )
    return number_entries, word_entries


def _entry(
    *,
    entry_id: str,
    name: str,
    domain: str,
    category: str,
    rpn_program: str,
    operation: str,
    tags: list[str],
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "domain": domain,
        "category": category,
        "rpn_program": rpn_program,
        "metadata": {
            "operation": operation,
            "tags": tags,
            "bootstrap": "deterministic_foundation_v1",
        },
    }


def _grammar_entries() -> list[dict[str, Any]]:
    base: list[dict[str, Any]] = [
        _entry(
            entry_id="rotate_90",
            name="Rotate 90",
            domain="grammar",
            category="geometric_transform",
            rpn_program="GRID ROTATE_90",
            operation="ROTATE_90",
            tags=["geometry", "transform"],
        ),
        _entry(
            entry_id="rotate_180",
            name="Rotate 180",
            domain="grammar",
            category="geometric_transform",
            rpn_program="GRID ROTATE_180",
            operation="ROTATE_180",
            tags=["geometry", "transform"],
        ),
        _entry(
            entry_id="mirror_h",
            name="Mirror Horizontal",
            domain="grammar",
            category="geometric_transform",
            rpn_program="GRID MIRROR_H",
            operation="MIRROR_H",
            tags=["geometry", "transform"],
        ),
        _entry(
            entry_id="mirror_v",
            name="Mirror Vertical",
            domain="grammar",
            category="geometric_transform",
            rpn_program="GRID MIRROR_V",
            operation="MIRROR_V",
            tags=["geometry", "transform"],
        ),
        _entry(
            entry_id="transpose",
            name="Transpose",
            domain="grammar",
            category="geometric_transform",
            rpn_program="GRID TRANSPOSE",
            operation="TRANSPOSE",
            tags=["geometry", "transform"],
        ),
        _entry(
            entry_id="pattern_alternating_next",
            name="Alternating Pattern Next",
            domain="grammar",
            category="pattern_rule",
            rpn_program="A B A B NEXT=A",
            operation="ALTERNATING_NEXT",
            tags=["pattern", "sequence"],
        ),
        _entry(
            entry_id="pattern_arithmetic_next",
            name="Arithmetic Progression Next",
            domain="grammar",
            category="pattern_rule",
            rpn_program="A D ADD",
            operation="ARITHMETIC_NEXT",
            tags=["pattern", "progression"],
        ),
        _entry(
            entry_id="pattern_geometric_next",
            name="Geometric Progression Next",
            domain="grammar",
            category="pattern_rule",
            rpn_program="A R MUL",
            operation="GEOMETRIC_NEXT",
            tags=["pattern", "progression"],
        ),
        _entry(
            entry_id="pattern_mirror_complete",
            name="Mirror Completion",
            domain="grammar",
            category="pattern_rule",
            rpn_program="HALF REVERSE APPEND",
            operation="MIRROR_COMPLETE",
            tags=["pattern", "symmetry"],
        ),
        _entry(
            entry_id="pattern_row_tile",
            name="Row Tiling",
            domain="grammar",
            category="pattern_rule",
            rpn_program="ROW PERIODIC_TILE",
            operation="ROW_TILE",
            tags=["pattern", "periodic"],
        ),
        _entry(
            entry_id="tile_pattern",
            name="Tile Pattern",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID TILE_PATTERN",
            operation="TILE_PATTERN",
            tags=["arc", "pattern", "composition"],
        ),
        _entry(
            entry_id="phase_shift",
            name="Phase Shift",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID PHASE_SHIFT",
            operation="PHASE_SHIFT",
            tags=["arc", "pattern", "composition"],
        ),
        _entry(
            entry_id="color_remap",
            name="Color Remap",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID COLOR_REMAP",
            operation="COLOR_REMAP",
            tags=["arc", "color", "composition"],
        ),
        _entry(
            entry_id="object_extract",
            name="Object Extract",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID CONNECTED_COMPONENTS",
            operation="OBJECT_EXTRACT",
            tags=["arc", "object", "composition"],
        ),
        _entry(
            entry_id="object_place",
            name="Object Place",
            domain="grammar",
            category="arc_transform",
            rpn_program="OBJECT PLACE",
            operation="OBJECT_PLACE",
            tags=["arc", "object", "composition"],
        ),
        _entry(
            entry_id="grid_resize",
            name="Grid Resize",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID RESIZE",
            operation="GRID_RESIZE",
            tags=["arc", "grid", "composition"],
        ),
        _entry(
            entry_id="conditional_fill",
            name="Conditional Fill",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID CONDITIONAL_FILL",
            operation="CONDITIONAL_FILL",
            tags=["arc", "fill", "composition"],
        ),
        _entry(
            entry_id="symmetry_complete",
            name="Symmetry Complete",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID SYMMETRY_COMPLETE",
            operation="SYMMETRY_COMPLETE",
            tags=["arc", "symmetry", "composition"],
        ),
        _entry(
            entry_id="border_fill",
            name="Border Fill",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID BORDER_FILL",
            operation="BORDER_FILL",
            tags=["arc", "fill", "composition"],
        ),
        _entry(
            entry_id="crop_region",
            name="Crop Region",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID CROP_REGION",
            operation="CROP_REGION",
            tags=["arc", "grid", "composition"],
        ),
        _entry(
            entry_id="overlay_grid",
            name="Overlay Grid",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID OVERLAY_GRID",
            operation="OVERLAY_GRID",
            tags=["arc", "grid", "composition"],
        ),
        _entry(
            entry_id="flood_fill",
            name="Flood Fill",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID FLOOD_FILL",
            operation="FLOOD_FILL",
            tags=["arc", "fill", "composition"],
        ),
        _entry(
            entry_id="connected_components",
            name="Connected Components",
            domain="grammar",
            category="arc_transform",
            rpn_program="GRID CONNECTED_COMPONENTS",
            operation="CONNECTED_COMPONENTS",
            tags=["arc", "object", "composition"],
        ),
        _entry(
            entry_id="consume_from_total",
            name="Consume From Total",
            domain="grammar",
            category="math_word_problem",
            rpn_program="STACK total consumed SUB",
            operation="CONSUME_FROM_TOTAL",
            tags=["math", "word_problem", "composition", "subtraction"],
        ),
        _entry(
            entry_id="rate_application",
            name="Rate Application",
            domain="grammar",
            category="math_word_problem",
            rpn_program="STACK quantity rate MUL",
            operation="RATE_APPLICATION",
            tags=["math", "word_problem", "composition", "multiplication"],
        ),
        _entry(
            entry_id="sequential_computation",
            name="Sequential Computation",
            domain="grammar",
            category="math_word_problem",
            rpn_program="STACK step_1 step_2 step_3 EVAL_CHAIN",
            operation="SEQUENTIAL_COMPUTATION",
            tags=["math", "word_problem", "composition", "multi_step"],
        ),
        _entry(
            entry_id="comparison_delta",
            name="Comparison Delta",
            domain="grammar",
            category="math_word_problem",
            rpn_program="STACK larger smaller SUB",
            operation="COMPARISON_DELTA",
            tags=["math", "word_problem", "comparison"],
        ),
        _entry(
            entry_id="percentage_application",
            name="Percentage Application",
            domain="grammar",
            category="math_word_problem",
            rpn_program="STACK base percent MUL 100 DIV",
            operation="PERCENTAGE_APPLICATION",
            tags=["math", "word_problem", "percentage"],
        ),
        _entry(
            entry_id="answer_final_stack",
            name="Answer Final Stack",
            domain="grammar",
            category="math_word_problem",
            rpn_program="STACK TOP EMIT",
            operation="ANSWER_FINAL_STACK",
            tags=["math", "word_problem", "composition", "emit"],
        ),
    ]

    # Compositional aliases to strengthen query/retrieval coverage.
    for a, b in (
        ("ROTATE_90", "MIRROR_H"),
        ("ROTATE_90", "MIRROR_V"),
        ("TRANSPOSE", "MIRROR_H"),
        ("TRANSPOSE", "MIRROR_V"),
        ("ROTATE_180", "MIRROR_H"),
    ):
        base.append(
            _entry(
                entry_id=f"compose_{a.lower()}_{b.lower()}",
                name=f"Compose {a} Then {b}",
                domain="grammar",
                category="compositional_rule",
                rpn_program=f"GRID {a} {b}",
                operation=f"{a}+{b}",
                tags=["composition", "chain", "deterministic"],
            )
        )
    # Alias entries to improve retrieval robustness during curriculum runs.
    alias_map = {
        "ROTATE_90": ("turn right", "clockwise quarter turn"),
        "ROTATE_180": ("turn around", "half turn"),
        "MIRROR_H": ("flip left right", "reflect vertical axis"),
        "MIRROR_V": ("flip up down", "reflect horizontal axis"),
        "TRANSPOSE": ("swap rows columns", "matrix transpose"),
        "ALTERNATING_NEXT": ("alternating continuation", "abab sequence next"),
        "ARITHMETIC_NEXT": ("linear sequence next", "constant difference next"),
        "GEOMETRIC_NEXT": ("ratio sequence next", "multiplicative progression next"),
        "MIRROR_COMPLETE": ("symmetry completion", "complete mirrored sequence"),
        "ROW_TILE": ("repeat row pattern", "periodic row extension"),
        "CONSUME_FROM_TOTAL": ("consume from total", "left over remaining after"),
        "RATE_APPLICATION": ("rate application", "each per every"),
        "SEQUENTIAL_COMPUTATION": ("multi step word problem", "first then after finally"),
        "COMPARISON_DELTA": ("word problem comparison", "more than less than difference"),
        "PERCENTAGE_APPLICATION": ("word problem percentage", "percent of percentage"),
        "ANSWER_FINAL_STACK": ("answer final stack", "emit final answer"),
    }
    for op, aliases in alias_map.items():
        for alias in aliases:
            slug = alias.replace(" ", "_")
            base.append(
                _entry(
                    entry_id=f"alias_{op.lower()}_{slug}",
                    name=f"{op} Alias: {alias}",
                    domain="grammar",
                    category="alias_rule",
                    rpn_program=f"{op}",
                    operation=op,
                    tags=["alias", "retrieval", "curriculum"],
                )
            )
    return base


def _math_entries() -> list[dict[str, Any]]:
    base: list[dict[str, Any]] = [
        _entry(
            entry_id="count_value",
            name="Count Value",
            domain="math",
            category="grid_arithmetic",
            rpn_program="GRID VALUE EQUAL COUNT",
            operation="COUNT_VALUE",
            tags=["count", "grid"],
        ),
        _entry(
            entry_id="sum_all",
            name="Sum All Values",
            domain="math",
            category="grid_arithmetic",
            rpn_program="GRID SUM",
            operation="SUM_ALL",
            tags=["sum", "grid"],
        ),
        _entry(
            entry_id="max_value",
            name="Maximum Value",
            domain="math",
            category="grid_arithmetic",
            rpn_program="GRID MAX",
            operation="MAX_VALUE",
            tags=["max", "grid"],
        ),
        _entry(
            entry_id="min_value",
            name="Minimum Value",
            domain="math",
            category="grid_arithmetic",
            rpn_program="GRID MIN",
            operation="MIN_VALUE",
            tags=["min", "grid"],
        ),
        _entry(
            entry_id="unique_count",
            name="Unique Value Count",
            domain="math",
            category="grid_arithmetic",
            rpn_program="GRID UNIQUE COUNT",
            operation="UNIQUE_COUNT",
            tags=["unique", "count", "grid"],
        ),
        _entry(
            entry_id="rpn_eval",
            name="Evaluate RPN",
            domain="math",
            category="symbolic_rpn",
            rpn_program="RPN EVAL",
            operation="RPN_EVAL",
            tags=["rpn", "symbolic"],
        ),
    ]
    # Symbolic operation vocabulary aliases.
    for op in ("ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MAX", "MIN", "ABS", "NEG"):
        base.append(
            _entry(
                entry_id=f"op_{op.lower()}",
                name=f"RPN Op {op}",
                domain="math",
                category="symbolic_rpn",
                rpn_program=op,
                operation=op,
                tags=["rpn", "operator"],
            )
        )
    for op, alias in (
        ("COUNT_VALUE", "count matching value"),
        ("SUM_ALL", "sum every cell"),
        ("MAX_VALUE", "largest cell"),
        ("MIN_VALUE", "smallest cell"),
        ("UNIQUE_COUNT", "count unique symbols"),
        ("RPN_EVAL", "evaluate reverse polish expression"),
        ("ADD", "plus operator"),
        ("SUB", "minus operator"),
        ("MUL", "times operator"),
        ("DIV", "division operator"),
        ("POW", "power operator"),
        ("MOD", "modulo operator"),
    ):
        slug = alias.replace(" ", "_")
        base.append(
            _entry(
                entry_id=f"alias_{op.lower()}_{slug}",
                name=f"{op} Alias: {alias}",
                domain="math",
                category="alias_rule",
                rpn_program=op,
                operation=op,
                tags=["alias", "retrieval", "curriculum"],
            )
        )
    return base


def _existing_ids(galaxy_entries: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for entry in galaxy_entries:
        if not isinstance(entry, dict):
            continue
        for key in ("id", "rule_id"):
            value = entry.get(key)
            if value:
                out.add(str(value))
    return out


def populate_foundational_operations(galaxy_manager: Any) -> dict[str, int]:
    """Populate foundational deterministic operations with idempotent semantics."""
    grammar = galaxy_manager.get_galaxy("Grammar")
    math = galaxy_manager.get_galaxy("Math")
    word = galaxy_manager.get_galaxy("Word")
    number = galaxy_manager.get_galaxy("Number")
    grammar_ids = _existing_ids(list(getattr(grammar, "entries", [])))
    math_ids = _existing_ids(list(getattr(math, "entries", [])))
    word_ids = _existing_ids(list(getattr(word, "entries", [])))
    number_ids = _existing_ids(list(getattr(number, "entries", [])))

    inserted_grammar = 0
    for entry in _grammar_entries():
        if entry["id"] in grammar_ids:
            continue
        galaxy_manager.add_entry("Grammar", entry)
        grammar_ids.add(entry["id"])
        inserted_grammar += 1

    inserted_math = 0
    for entry in _math_entries():
        if entry["id"] in math_ids:
            continue
        galaxy_manager.add_entry("Math", entry)
        math_ids.add(entry["id"])
        inserted_math += 1

    number_entries, word_entries = _number_word_entries()

    inserted_numbers = 0
    for entry in number_entries:
        if entry["id"] in number_ids:
            continue
        galaxy_manager.add_entry("Number", entry)
        number_ids.add(entry["id"])
        inserted_numbers += 1

    inserted_words = 0
    for entry in word_entries:
        if entry["id"] in word_ids:
            continue
        galaxy_manager.add_entry("Word", entry)
        word_ids.add(entry["id"])
        inserted_words += 1

    return {
        "inserted_grammar": inserted_grammar,
        "inserted_math": inserted_math,
        "inserted_numbers": inserted_numbers,
        "inserted_words": inserted_words,
        "total_inserted": inserted_grammar + inserted_math + inserted_numbers + inserted_words,
    }


def bootstrap_default(storage_root: str | Path = "../Knowledge3D.local") -> dict[str, Any]:
    """Convenience bootstrap entrypoint for scripts and CLI usage."""
    from .knowledgeverse import Knowledgeverse

    kv = Knowledgeverse(storage_root=storage_root)
    return {
        **kv.foundational_bootstrap_summary,
        "storage_root": str(kv.storage_root),
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Bootstrap deterministic foundational operations.")
    parser.add_argument("--storage-root", default="../Knowledge3D.local")
    args = parser.parse_args()

    summary = bootstrap_default(storage_root=args.storage_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
