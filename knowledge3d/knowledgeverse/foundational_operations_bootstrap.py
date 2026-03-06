"""Bootstrap foundational deterministic operations into Grammar/Math galaxies."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    grammar_ids = _existing_ids(list(getattr(grammar, "entries", [])))
    math_ids = _existing_ids(list(getattr(math, "entries", [])))

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

    return {
        "inserted_grammar": inserted_grammar,
        "inserted_math": inserted_math,
        "total_inserted": inserted_grammar + inserted_math,
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
