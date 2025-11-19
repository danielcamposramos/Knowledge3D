#!/usr/bin/env python3
"""
Generate Procedural Math Symbols (Dual-Modal)

Math symbols are BOTH:
1. Visual glyphs (drawn procedurally with RPN)
2. Semantic operations (executable RPN math)

Example: '√' (square root)
- Visual RPN: "0.3 0.5 MOVE 0.4 0.6 LINE 0.45 0.3 LINE 0.7 0.3 LINE STROKE"
- Math RPN: "SQRT"  (takes one arg from stack)
- Semantic: √16 → "16 SQRT" → 4

This enables the model to DO MATH in its mind (sovereignty)!
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

# Import RPN opcodes for semantic mapping
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import *


def generate_math_symbol_dual_modal(symbol: str) -> Tuple[str, str, str]:
    """
    Generate dual-modal representation for math symbol.

    Returns:
        (visual_rpn, math_rpn, semantic_meaning)

    Example:
        symbol='√' → (
            "0.3 0.5 MOVE ...",  # How to draw it
            "SQRT",              # How to execute it
            "Square root: pop x, push sqrt(x)"  # What it means
        )
    """

    # Dual-modal math symbols
    dual_symbols = {
        # Arithmetic operations
        '+': (
            "0.5 0.3 MOVE 0.5 0.7 LINE 0.3 0.5 MOVE 0.7 0.5 LINE STROKE",
            "ADD",
            "Addition: pop b, pop a, push a+b"
        ),
        '-': (
            "0.3 0.5 MOVE 0.7 0.5 LINE STROKE",
            "SUB",
            "Subtraction: pop b, pop a, push a-b"
        ),
        '×': (
            "0.3 0.3 MOVE 0.7 0.7 LINE 0.3 0.7 MOVE 0.7 0.3 LINE STROKE",
            "MUL",
            "Multiplication: pop b, pop a, push a*b"
        ),
        '÷': (
            "0.3 0.5 MOVE 0.7 0.5 LINE 0.5 0.3 MOVE 0.5 0.35 LINE 0.5 0.65 MOVE 0.5 0.7 LINE STROKE",
            "DIV",
            "Division: pop b, pop a, push a/b"
        ),

        # Relations (return 1.0 or 0.0)
        '=': (
            "0.3 0.45 MOVE 0.7 0.45 LINE 0.3 0.55 MOVE 0.7 0.55 LINE STROKE",
            "EQ",
            "Equality: pop b, pop a, push 1.0 if a==b else 0.0"
        ),
        '<': (
            "0.6 0.3 MOVE 0.4 0.5 LINE 0.6 0.7 LINE STROKE",
            "LT",
            "Less than: pop b, pop a, push 1.0 if a<b else 0.0"
        ),
        '>': (
            "0.4 0.3 MOVE 0.6 0.5 LINE 0.4 0.7 LINE STROKE",
            "GT",
            "Greater than: pop b, pop a, push 1.0 if a>b else 0.0"
        ),

        # Unary operations
        '√': (
            "0.3 0.5 MOVE 0.4 0.6 LINE 0.45 0.3 LINE 0.7 0.3 LINE STROKE",
            "SQRT",
            "Square root: pop x, push sqrt(x)"
        ),
        '|x|': (
            "0.4 0.3 MOVE 0.4 0.7 LINE 0.4 0.5 MOVE 0.6 0.5 LINE 0.6 0.3 MOVE 0.6 0.7 LINE STROKE",
            "ABS",
            "Absolute value: pop x, push |x|"
        ),

        # Trigonometric
        'sin': (
            "0.2 0.5 MOVE 0.3 0.3 0.5 0.5 QUAD 0.7 0.7 0.8 0.5 QUAD STROKE",
            "SIN",
            "Sine: pop x, push sin(x)"
        ),
        'cos': (
            "0.2 0.3 MOVE 0.4 0.5 0.6 0.5 QUAD 0.8 0.7 LINE STROKE",
            "COS",
            "Cosine: pop x, push cos(x)"
        ),
        'tan': (
            "0.3 0.2 MOVE 0.3 0.8 LINE 0.5 0.5 MOVE 0.7 0.3 LINE STROKE",
            "TAN",
            "Tangent: pop x, push tan(x)"
        ),

        # Exponential/logarithmic
        'exp': (
            "0.2 0.6 MOVE 0.4 0.4 0.6 0.4 QUAD 0.8 0.3 LINE STROKE",
            "EXP",
            "Exponential: pop x, push e^x"
        ),
        'ln': (
            "0.3 0.3 MOVE 0.3 0.7 LINE 0.5 0.5 MOVE 0.7 0.5 LINE STROKE",
            "LOG",  # Natural log
            "Natural log: pop x, push ln(x)"
        ),
        'log': (
            "0.3 0.3 MOVE 0.3 0.7 LINE 0.5 0.4 MOVE 0.7 0.6 LINE STROKE",
            "LOG10",
            "Log base 10: pop x, push log10(x)"
        ),

        # Constants (no pop, just push)
        'π': (
            "0.25 0.4 MOVE 0.75 0.4 LINE 0.4 0.4 MOVE 0.4 0.7 LINE 0.6 0.4 MOVE 0.6 0.7 LINE STROKE",
            "3.14159265359",  # Push constant
            "Pi constant: push π (3.14159...)"
        ),
        'e': (
            "0.3 0.3 MOVE 0.7 0.3 LINE 0.7 0.5 LINE 0.5 0.5 LINE 0.7 0.7 LINE 0.3 0.7 LINE STROKE",
            "2.71828182846",  # Push constant
            "Euler's number: push e (2.71828...)"
        ),

        # Stack operations
        'dup': (
            "0.3 0.4 MOVE 0.7 0.4 LINE 0.3 0.6 MOVE 0.7 0.6 LINE STROKE",
            "DUP",
            "Duplicate: pop x, push x, push x"
        ),
        'swap': (
            "0.3 0.3 MOVE 0.7 0.7 LINE 0.3 0.7 MOVE 0.7 0.3 LINE STROKE",
            "SWAP",
            "Swap: pop b, pop a, push b, push a"
        ),
        'drop': (
            "0.5 0.3 MOVE 0.5 0.7 LINE 0.4 0.6 MOVE 0.6 0.6 LINE STROKE",
            "DROP",
            "Drop: pop x (discard)"
        ),

        # Composite operations (min, max)
        'max': (
            "0.3 0.5 MOVE 0.5 0.3 LINE 0.7 0.5 LINE STROKE",
            "MAX",
            "Maximum: pop b, pop a, push max(a,b)"
        ),
        'min': (
            "0.3 0.3 MOVE 0.5 0.5 LINE 0.7 0.3 LINE STROKE",
            "MIN",
            "Minimum: pop b, pop a, push min(a,b)"
        ),

        # Modulo, rounding
        'mod': (
            "0.3 0.5 MOVE 0.5 0.5 LINE 0.5 0.3 MOVE 0.5 0.7 LINE 0.6 0.5 MOVE 0.8 0.5 LINE STROKE",
            "MOD",
            "Modulo: pop b, pop a, push a%b"
        ),
        'floor': (
            "0.3 0.7 MOVE 0.7 0.7 LINE 0.7 0.3 LINE STROKE",
            "FLOOR",
            "Floor: pop x, push floor(x)"
        ),
        'ceil': (
            "0.3 0.3 MOVE 0.7 0.3 LINE 0.7 0.7 LINE STROKE",
            "CEIL",
            "Ceiling: pop x, push ceil(x)"
        ),
        'round': (
            "0.3 0.5 MOVE 0.7 0.5 LINE 0.5 0.3 MOVE 0.5 0.7 LINE STROKE",
            "ROUND",
            "Round: pop x, push round(x)"
        ),
    }

    return dual_symbols.get(symbol, (
        "0.3 0.3 MOVE 0.7 0.7 LINE STROKE",  # Default visual
        "NOP",  # No operation
        "Unknown operation"
    ))


def generate_math_dataset_procedural(output_path: Path):
    """Generate procedural math symbols dataset (visual + semantic)."""
    print(f"\n{'='*60}")
    print("Generating Procedural Math Dataset (Dual-Modal)")
    print(f"{'='*60}")

    # Math symbols by category
    symbols_by_category = {
        'arithmetic': ['+', '-', '×', '÷'],
        'relations': ['=', '<', '>'],
        'unary': ['√', '|x|'],
        'trigonometric': ['sin', 'cos', 'tan'],
        'exponential': ['exp', 'ln', 'log'],
        'constants': ['π', 'e'],
        'stack': ['dup', 'swap', 'drop'],
        'comparison': ['max', 'min'],
        'rounding': ['mod', 'floor', 'ceil', 'round'],
    }

    dataset = []

    for category, symbols in symbols_by_category.items():
        for symbol in symbols:
            visual_rpn, math_rpn, semantic = generate_math_symbol_dual_modal(symbol)

            # Example usage for semantic understanding
            if symbol == '+':
                example = "3 5 ADD → 8"
            elif symbol == '√':
                example = "16 SQRT → 4"
            elif symbol == 'π':
                example = "PI DUP MUL → 9.8696 (π²)"
            else:
                example = f"{symbol} operation"

            entry = {
                'symbol': symbol,
                'visual_rpn': visual_rpn,      # How to DRAW it
                'math_rpn': math_rpn,          # How to EXECUTE it
                'semantic': semantic,          # What it MEANS
                'category': category,
                'example': example,
                'type': 'math_dual_modal',
                # Relation to text 'x' (if applicable)
                'text_relation': symbol if len(symbol) == 1 else None
            }
            dataset.append(entry)

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ Generated {len(dataset)} dual-modal math symbols")
    print(f"   Categories: {list(symbols_by_category.keys())}")
    print(f"   Each symbol has:")
    print(f"     - Visual RPN (how to draw)")
    print(f"     - Math RPN (how to execute)")
    print(f"     - Semantic meaning")
    print(f"   Output: {output_path}")

    # Print examples
    print(f"\n{'='*60}")
    print("Example Dual-Modal Symbols:")
    print(f"{'='*60}")
    for i, entry in enumerate(dataset[:5]):
        print(f"\n{i+1}. Symbol: {entry['symbol']}")
        print(f"   Visual: {entry['visual_rpn'][:50]}...")
        print(f"   Math RPN: {entry['math_rpn']}")
        print(f"   Semantic: {entry['semantic']}")
        print(f"   Example: {entry['example']}")

    return len(dataset)


def main():
    """Generate procedural math symbols dataset."""
    print("\n" + "="*60)
    print("PROCEDURAL MATH SYMBOLS GENERATION")
    print("Dual-Modal: Visual (draw) + Semantic (execute)")
    print("="*60)

    output_path = Path("/K3D/Knowledge3D.local/datasets/atomic/math_procedural.jsonl")
    count = generate_math_dataset_procedural(output_path)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Math symbols: {count} dual-modal entries")
    print()
    print("Model can now:")
    print("  1. DRAW '√' symbol (visual RPN)")
    print("  2. EXECUTE √16 (math RPN: '16 SQRT')")
    print("  3. UNDERSTAND 'square root' (semantic meaning)")
    print()
    print("Sovereignty: Math in the mind, no external tools!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
