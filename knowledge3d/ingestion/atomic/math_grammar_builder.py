"""
Math Grammar builder: constructs "grammar stars" for math expressions.

Purpose:
- Define operator precedence/associativity, valid variable symbol sets, and sequencing
  rules as compositional references to atomic math symbol stars and punctuation.
- Output JSONL of grammar rules (symlink style) to be stored in math/grammar galaxy.

Outputs (per grammar rule star):
{
  "grammar_id": "MATH_GRAMMAR_CORE_V1",
  "precedence": [
     {"group": "parentheses", "symbols": ["(", ")"], "level": 0},
     {"group": "exponentiation", "symbols": ["^"], "associativity": "right", "level": 1},
     {"group": "multiplication_division", "symbols": ["*", "×", "/", "÷"], "associativity": "left", "level": 2},
     {"group": "addition_subtraction", "symbols": ["+", "-"], "associativity": "left", "level": 3}
  ],
  "variables": {
     "latin": ["A-Z", "a-z"],
     "greek_math": ["α","β","γ","π","Σ","Δ","λ","μ","σ","φ","ψ","ω"],
     "digits": ["0-9"]
  },
  "punctuation": ["(", ")", ",", ";"],
  "procedural_programs": {
     "parse_rpn": "GRAMMAR MATH_CORE PRECEDENCE 4 ...",  # optional placeholder
  }
}

Notes:
- Uses symlink style: references math symbols/punctuation by literal glyph; resolution to
  math symbol galaxy/punctuation galaxy happens at runtime.
- No evaluation here; this is structural grammar metadata for routing/parsing/execution order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def build_math_grammar() -> List[Dict]:
    precedence = [
        {"group": "parentheses", "symbols": ["(", ")"], "level": 0, "associativity": None},
        {"group": "factorial", "symbols": ["!"], "level": 1, "associativity": "right", "unary": True},
        {"group": "unary", "symbols": ["-"], "level": 1, "associativity": "right", "unary": True},
        {"group": "exponentiation", "symbols": ["^"], "level": 2, "associativity": "right"},
        {
            "group": "multiplication_division",
            "symbols": ["*", "×", "/", "÷"],
            "level": 3,
            "associativity": "left",
        },
        {
            "group": "addition_subtraction",
            "symbols": ["+", "-"],
            "level": 4,
            "associativity": "left",
        },
    ]
    variables = {
        "latin": ["A-Z", "a-z"],
        "greek_math": ["α", "β", "γ", "π", "Σ", "Δ", "λ", "μ", "σ", "φ", "ψ", "ω"],
        "digits": ["0-9"],
    }
    punctuation = ["(", ")", ",", ";"]
    grammar = {
        "grammar_id": "MATH_GRAMMAR_CORE_V1",
        "precedence": precedence,
        "variables": variables,
        "punctuation": punctuation,
        "procedural_programs": {
            "parse_rpn": "GRAMMAR MATH_CORE PRECEDENCE 5",  # placeholder hook
        },
        "usage": "math_expression",
    }
    return [grammar]


def write_jsonl(stars: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for star in stars:
            f.write(json.dumps(star, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build math grammar stars (precedence, variables, punctuation).")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL for math grammar.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stars = build_math_grammar()
    write_jsonl(stars, args.output)
    print(f"Wrote {len(stars)} math grammar stars -> {args.output}")


if __name__ == "__main__":
    main()
