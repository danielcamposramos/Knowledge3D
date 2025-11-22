"""
Math Symbol Galaxy builder (procedural-first, meaning-separated from letters).

Purpose:
- Build math symbol stars with executable math_rpn (stack effect), visual_rpn variants,
  and optional audio_rpn (verbalization). No case variants, no word composition.
- Output JSONL ready for ProceduralGalaxy/GLB upsert.

Inputs:
- math_symbols_procedural.jsonl (expected fields: symbol, visual_rpn, math_rpn, optional
  size/font metadata, optional audio_rpn or verbalization).

Outputs:
- JSONL where each line is one math symbol star:
  {
    "symbol_concept": "ADDITION_OPERATOR",
    "symbol": "+",
    "glyph_variants": [...],
    "procedural_programs": {"visual_rpn": "...", "math_rpn": "...", "audio_rpn": "..."},
    "semantic_identity": {...}
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_symbols(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _concept_from_symbol(sym: str) -> str:
    """Simple mapping; extend with richer tables as needed."""
    mapping = {
        "+": "ADDITION_OPERATOR",
        "-": "SUBTRACTION_OPERATOR",
        "−": "SUBTRACTION_OPERATOR",
        "±": "PLUS_MINUS_OPERATOR",
        "*": "MULTIPLICATION_OPERATOR",
        "×": "MULTIPLICATION_OPERATOR",
        "/": "DIVISION_OPERATOR",
        "÷": "DIVISION_OPERATOR",
        "=": "EQUALITY_OPERATOR",
        "≠": "INEQUALITY_OPERATOR",
        "<": "LESS_THAN_OPERATOR",
        ">": "GREATER_THAN_OPERATOR",
        "≤": "LEQ_OPERATOR",
        "≥": "GEQ_OPERATOR",
        "^": "EXPONENTIATION_OPERATOR",
        "√": "SQUARE_ROOT_OPERATOR",
        "∑": "SUMMATION_OPERATOR",
        "∏": "PRODUCT_OPERATOR",
        "∫": "INTEGRATION_OPERATOR",
        "∂": "PARTIAL_DERIVATIVE_OPERATOR",
        "∇": "NABLA_OPERATOR",
        "π": "PI_CONSTANT",
        "e": "E_CONSTANT",
        "!": "FACTORIAL_OPERATOR",
    }
    return mapping.get(sym, f"MATH_SYMBOL_{sym}")


def build_math_symbols(jsonl_path: Path) -> List[Dict]:
    stars: List[Dict] = []
    for item in _iter_symbols(jsonl_path):
        sym = item.get("symbol") or item.get("char") or ""
        if not sym:
            continue
        concept = item.get("symbol_concept") or _concept_from_symbol(sym)
        glyph_variants = item.get("glyph_variants") or [
            {
                "visual_rpn": item.get("visual_rpn", ""),
                "size": item.get("size", "inline"),
                "font": item.get("font", "default"),
            }
        ]
        # Default math_rpn/metadata from concept if not provided
        default_ops = {
            "ADDITION_OPERATOR": {"math_rpn": "POP b POP a ADD a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "SUBTRACTION_OPERATOR": {"math_rpn": "POP b POP a SUB a b PUSH result", "arity": 2, "associative": False, "commutative": False},
            "MULTIPLICATION_OPERATOR": {"math_rpn": "POP b POP a MUL a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "DIVISION_OPERATOR": {"math_rpn": "POP b POP a DIV a b PUSH result", "arity": 2, "associative": False, "commutative": False},
            "EXPONENTIATION_OPERATOR": {"math_rpn": "POP b POP a POW a b PUSH result", "arity": 2, "associative": False, "commutative": False},
            "FACTORIAL_OPERATOR": {"math_rpn": "POP a FACT a PUSH result", "arity": 1, "associative": False, "commutative": False},
            "PLUS_MINUS_OPERATOR": {"math_rpn": "POP a DUP a NEG SWAP ADD PUSH result", "arity": 1, "associative": False, "commutative": False},
            "EQUALITY_OPERATOR": {"math_rpn": "POP b POP a EQ a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "INEQUALITY_OPERATOR": {"math_rpn": "POP b POP a NEQ a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "LEQ_OPERATOR": {"math_rpn": "POP b POP a LEQ a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "GEQ_OPERATOR": {"math_rpn": "POP b POP a GEQ a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "LESS_THAN_OPERATOR": {"math_rpn": "POP b POP a LT a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "GREATER_THAN_OPERATOR": {"math_rpn": "POP b POP a GT a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "SUMMATION_OPERATOR": {"math_rpn": "POP b POP a SUM a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "PRODUCT_OPERATOR": {"math_rpn": "POP b POP a PROD a b PUSH result", "arity": 2, "associative": True, "commutative": True},
            "INTEGRATION_OPERATOR": {"math_rpn": "POP f POP b POP a INTEGRATE f a b PUSH result", "arity": 3, "associative": False, "commutative": False},
            "PARTIAL_DERIVATIVE_OPERATOR": {"math_rpn": "POP var POP f PD f var PUSH result", "arity": 2, "associative": False, "commutative": False},
            "NABLA_OPERATOR": {"math_rpn": "POP f GRAD f PUSH result", "arity": 1, "associative": False, "commutative": False},
            "PI_CONSTANT": {"math_rpn": "PUSH CONST_PI", "arity": 0, "associative": None, "commutative": None},
            "E_CONSTANT": {"math_rpn": "PUSH CONST_E", "arity": 0, "associative": None, "commutative": None},
        }
        defaults = default_ops.get(concept, {})
        math_rpn = item.get("math_rpn") or defaults.get("math_rpn") or "PUSH 0"
        audio_rpn = item.get("audio_rpn")
        semantic_identity = {
            "operation": concept,
            "arity": item.get("arity", defaults.get("arity")),
            "commutative": item.get("commutative", defaults.get("commutative")),
            "associative": item.get("associative", defaults.get("associative")),
            "identity_element": item.get("identity_element", defaults.get("identity_element")),
        }
        stars.append(
            {
                "symbol_concept": concept,
                "symbol": sym,
                "semantic_identity": semantic_identity,
                "glyph_variants": glyph_variants,
                "procedural_programs": {
                    "visual_rpn": glyph_variants[0].get("visual_rpn", ""),
                    "math_rpn": math_rpn,
                    "audio_rpn": audio_rpn,
                },
                "usage": "mathematical_expression",
            }
        )
    return stars


def write_jsonl(stars: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for star in stars:
            f.write(json.dumps(star, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build math symbol stars (procedural-first).")
    ap.add_argument("--symbols-jsonl", type=Path, required=True, help="Input math_symbols_procedural.jsonl")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL of math symbol stars.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stars = build_math_symbols(args.symbols_jsonl)
    write_jsonl(stars, args.output)
    print(f"Wrote {len(stars)} math symbol stars -> {args.output}")


if __name__ == "__main__":
    main()
