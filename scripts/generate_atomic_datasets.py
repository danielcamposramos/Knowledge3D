#!/usr/bin/env python3
"""
Generate Atomic Datasets: Fonts + Math Symbols

Creates two core datasets that are always loaded (atomic units):
1. Character glyphs (all available fonts)
2. Math symbols (procedurally generated)

Both use RPN procedural representation for GPU-native execution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict
import subprocess

def generate_simple_glyph_rpn(char: str, variant: int = 0) -> str:
    """
    Generate simple procedural RPN for character glyph.

    Real font harvesting would trace actual font outlines.
    This creates synthetic placeholders for immediate testing.
    """
    # Hash character to get deterministic coordinates
    char_code = ord(char)

    # Variant offset for different "fonts"
    v_offset = variant * 0.1

    # Simple geometric shapes based on character
    if char.isdigit():
        # Numbers: rectangles with different aspect ratios
        digit = int(char)
        w = 0.3 + (digit * 0.05)
        h = 0.6
        x, y = 0.2 + v_offset, 0.2
        rpn = f"{x} {y} MOVE {x+w} {y} LINE {x+w} {y+h} LINE {x} {y+h} LINE CLOSE STROKE"

    elif char.isupper():
        # Uppercase: tall vertical lines + horizontals
        x_base = 0.2 + ((char_code % 10) * 0.03)
        y_base = 0.1
        # Vertical line
        rpn = f"{x_base} {y_base} MOVE {x_base} {y_base+0.7} LINE "
        # Horizontal crossbar
        rpn += f"{x_base+0.3} {y_base+0.35} LINE STROKE"

    elif char.islower():
        # Lowercase: shorter with more curves
        x = 0.2 + ((char_code % 15) * 0.02)
        y = 0.3
        # Use QUAD for curves
        cx, cy = x + 0.2, y - 0.1
        rpn = f"{x} {y} MOVE {x+0.3} {y+0.3} {cx} {cy} QUAD STROKE"

    else:
        # Default: simple diagonal
        rpn = f"0.2 0.2 MOVE 0.7 0.7 LINE STROKE"

    return rpn


def generate_math_symbol_rpn(symbol: str) -> str:
    """
    Generate procedural RPN for math symbols.

    Math symbols are drawn geometrically using RPN primitives.
    """
    symbol_programs = {
        # Arithmetic
        '+': "0.5 0.3 MOVE 0.5 0.7 LINE 0.3 0.5 MOVE 0.7 0.5 LINE STROKE",  # Plus
        '-': "0.3 0.5 MOVE 0.7 0.5 LINE STROKE",  # Minus
        '×': "0.3 0.3 MOVE 0.7 0.7 LINE 0.3 0.7 MOVE 0.7 0.3 LINE STROKE",  # Multiplication
        '÷': "0.3 0.5 MOVE 0.7 0.5 LINE 0.5 0.3 MOVE 0.5 0.35 LINE 0.5 0.65 MOVE 0.5 0.7 LINE STROKE",  # Division

        # Relations
        '=': "0.3 0.45 MOVE 0.7 0.45 LINE 0.3 0.55 MOVE 0.7 0.55 LINE STROKE",  # Equals
        '<': "0.6 0.3 MOVE 0.4 0.5 LINE 0.6 0.7 LINE STROKE",  # Less than
        '>': "0.4 0.3 MOVE 0.6 0.5 LINE 0.4 0.7 LINE STROKE",  # Greater than
        '≤': "0.6 0.3 MOVE 0.4 0.5 LINE 0.6 0.7 LINE 0.3 0.8 MOVE 0.7 0.8 LINE STROKE",
        '≥': "0.4 0.3 MOVE 0.6 0.5 LINE 0.4 0.7 LINE 0.3 0.8 MOVE 0.7 0.8 LINE STROKE",
        '≠': "0.3 0.4 MOVE 0.7 0.4 LINE 0.3 0.6 MOVE 0.7 0.6 LINE 0.4 0.3 MOVE 0.6 0.7 LINE STROKE",

        # Greek letters (simple approximations)
        'α': "0.6 0.5 MOVE 0.3 0.7 0.5 0.3 QUAD 0.6 0.5 LINE 0.6 0.7 LINE STROKE",  # alpha
        'β': "0.3 0.2 MOVE 0.3 0.8 LINE 0.5 0.8 0.6 0.65 QUAD 0.5 0.5 LINE 0.6 0.5 0.6 0.35 QUAD 0.5 0.2 LINE STROKE",  # beta
        'γ': "0.3 0.4 MOVE 0.5 0.4 LINE 0.4 0.7 LINE STROKE",  # gamma (simple)
        'π': "0.25 0.4 MOVE 0.75 0.4 LINE 0.4 0.4 MOVE 0.4 0.7 LINE 0.6 0.4 MOVE 0.6 0.7 LINE STROKE",  # pi
        'Σ': "0.6 0.3 MOVE 0.3 0.3 LINE 0.5 0.5 LINE 0.3 0.7 LINE 0.6 0.7 LINE STROKE",  # Sigma
        'Δ': "0.5 0.3 MOVE 0.3 0.7 LINE 0.7 0.7 LINE CLOSE STROKE",  # Delta (triangle)

        # Calculus
        '∫': "0.4 0.2 MOVE 0.45 0.3 0.4 0.4 QUAD 0.45 0.5 0.4 0.6 QUAD 0.45 0.7 0.5 0.8 QUAD STROKE",  # Integral
        '∂': "0.5 0.3 MOVE 0.3 0.5 0.5 0.7 QUAD 0.7 0.5 0.5 0.3 QUAD 0.5 0.2 MOVE 0.5 0.3 LINE STROKE",  # Partial derivative
        '∇': "0.5 0.3 MOVE 0.3 0.7 LINE 0.7 0.7 LINE CLOSE STROKE",  # Nabla (inverted delta)

        # Set theory
        '∈': "0.6 0.4 MOVE 0.4 0.5 0.6 0.6 QUAD 0.3 0.5 MOVE 0.6 0.5 LINE STROKE",  # Element of
        '∉': "0.6 0.4 MOVE 0.4 0.5 0.6 0.6 QUAD 0.3 0.5 MOVE 0.6 0.5 LINE 0.35 0.35 MOVE 0.65 0.65 LINE STROKE",
        '∪': "0.3 0.4 MOVE 0.4 0.6 0.6 0.6 QUAD 0.7 0.4 LINE STROKE",  # Union
        '∩': "0.3 0.6 MOVE 0.4 0.4 0.6 0.4 QUAD 0.7 0.6 LINE STROKE",  # Intersection
        '⊂': "0.6 0.35 MOVE 0.4 0.5 0.6 0.65 QUAD STROKE",  # Subset
        '∅': "0.5 0.35 MOVE 0.35 0.5 0.5 0.65 QUAD 0.65 0.5 0.5 0.35 QUAD 0.4 0.4 MOVE 0.6 0.6 LINE STROKE",  # Empty set

        # Logic
        '∧': "0.5 0.6 MOVE 0.35 0.4 LINE 0.5 0.6 MOVE 0.65 0.4 LINE STROKE",  # AND
        '∨': "0.35 0.6 MOVE 0.5 0.4 LINE 0.65 0.6 LINE STROKE",  # OR
        '¬': "0.3 0.5 MOVE 0.6 0.5 LINE 0.6 0.6 LINE STROKE",  # NOT
        '⇒': "0.3 0.5 MOVE 0.6 0.5 LINE 0.5 0.4 MOVE 0.6 0.5 LINE 0.5 0.6 LINE STROKE",  # Implies
        '⇔': "0.3 0.45 MOVE 0.6 0.45 LINE 0.3 0.55 MOVE 0.6 0.55 LINE 0.5 0.4 MOVE 0.6 0.45 LINE 0.5 0.6 LINE 0.5 0.4 MOVE 0.4 0.45 LINE 0.5 0.6 LINE STROKE",

        # Other
        '∞': "0.3 0.5 MOVE 0.4 0.4 0.45 0.5 QUAD 0.55 0.5 0.6 0.4 QUAD 0.7 0.5 0.6 0.6 QUAD 0.55 0.5 0.45 0.5 QUAD 0.4 0.6 0.3 0.5 QUAD STROKE",  # Infinity
        '√': "0.3 0.5 MOVE 0.4 0.6 LINE 0.45 0.3 LINE 0.7 0.3 LINE STROKE",  # Square root
    }

    return symbol_programs.get(symbol, "0.3 0.3 MOVE 0.7 0.7 LINE STROKE")


def generate_font_dataset(output_path: Path, max_samples: int = 50000):
    """Generate character glyph dataset from all available fonts."""
    print(f"\n{'='*60}")
    print("Generating Font Dataset (Character Glyphs)")
    print(f"{'='*60}")

    # Characters to generate
    chars = (
        # Uppercase A-Z
        [chr(i) for i in range(65, 91)] +
        # Lowercase a-z
        [chr(i) for i in range(97, 123)] +
        # Digits 0-9
        [chr(i) for i in range(48, 58)] +
        # Common punctuation
        list(".,;:!?\"'()-[]{}/@#$%^&*_+=<>")
    )

    # Generate synthetic "fonts" (variants)
    # In production, would use actual font tracing
    synthetic_fonts = [
        "Synthetic-Regular",
        "Synthetic-Bold",
        "Synthetic-Italic",
        "Synthetic-Light",
        "Synthetic-Condensed",
    ]

    dataset = []
    for font_idx, font_name in enumerate(synthetic_fonts):
        for char in chars:
            rpn = generate_simple_glyph_rpn(char, variant=font_idx)
            entry = {
                'char': char,
                'rpn': rpn,
                'font': font_name,
                'type': 'glyph',
                'category': 'alphanumeric' if char.isalnum() else 'punctuation'
            }
            dataset.append(entry)

            if len(dataset) >= max_samples:
                break
        if len(dataset) >= max_samples:
            break

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ Generated {len(dataset)} font samples")
    print(f"   Characters: {len(chars)}")
    print(f"   Fonts: {len(synthetic_fonts)}")
    print(f"   Output: {output_path}")

    return len(dataset)


def generate_math_dataset(output_path: Path):
    """
    Load dual-modal math symbols from procedurally extracted fonts.

    Uses real font outlines extracted from professional math fonts
    (Asana-Math, STIX, Libertinus, Fira, Noto, Latin Modern, DejaVu).

    Each symbol has:
    - visual_rpn: How to DRAW it (procedural outline)
    - math_rpn: How to EXECUTE it (RPN bytecode)
    - semantic: What it MEANS (mathematical description)
    """
    print(f"\n{'='*60}")
    print("Loading Math Symbols Dataset (Dual-Modal Procedural)")
    print(f"{'='*60}")

    # Path to extracted dual-modal math symbols
    extracted_path = Path("/K3D/Knowledge3D.local/datasets/math_symbols_procedural.jsonl")

    if not extracted_path.exists():
        print(f"⚠️  Extracted math symbols not found at {extracted_path}")
        print("   Run: python scripts/extract_math_fonts_procedural.py")
        print("   Falling back to synthetic symbols...")
        return generate_math_dataset_synthetic(output_path)

    # Load extracted symbols
    dataset = []
    with open(extracted_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)

            # Convert to training format
            entry = {
                'char': record['symbol'],
                'visual_rpn': record['visual_rpn'],  # How to DRAW
                'math_rpn': record['math_rpn'],      # How to EXECUTE
                'semantic': record['semantic'],       # What it MEANS
                'font': record['font_name'],
                'type': 'math_dual_modal',
                'category': record['category'],
                'name': record['name'],
                'unicode': record['unicode'],
                'multivariate': record['multivariate'],
            }
            dataset.append(entry)

    # Write training dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Statistics
    from collections import Counter
    category_counts = Counter(e['category'] for e in dataset)
    unique_symbols = len(set(e['char'] for e in dataset))

    print(f"✅ Loaded {len(dataset)} dual-modal math symbols")
    print(f"   Unique symbols: {unique_symbols}")
    print(f"   Source fonts: {len(set(e['font'] for e in dataset))}")
    print(f"   Categories:")
    for cat, count in sorted(category_counts.items()):
        print(f"     {cat:20s}: {count:4d}")
    print(f"   Output: {output_path}")

    return len(dataset)


def generate_math_dataset_synthetic(output_path: Path):
    """Fallback: Generate synthetic math symbols if extraction hasn't run."""
    print("   Generating synthetic math symbols...")

    math_symbols = {
        'arithmetic': ['+', '-', '×', '÷'],
        'relations': ['=', '<', '>', '≤', '≥', '≠'],
        'greek_letters': ['α', 'β', 'γ', 'π', 'Σ', 'Δ'],
        'calculus': ['∫', '∂', '∇'],
        'set_theory': ['∈', '∉', '∪', '∩', '⊂', '∅'],
        'logic': ['∧', '∨', '¬', '⇒', '⇔'],
        'other': ['∞', '√'],
    }

    dataset = []
    for category, symbols in math_symbols.items():
        for symbol in symbols:
            rpn = generate_math_symbol_rpn(symbol)

            symbol_names = {
                '+': 'plus', '-': 'minus', '×': 'times', '÷': 'divide',
                '=': 'equals', '<': 'less', '>': 'greater',
                'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'π': 'pi',
                'Σ': 'sigma', 'Δ': 'delta',
                '∫': 'integral', '∂': 'partial', '∇': 'nabla',
                '∈': 'element', '∪': 'union', '∩': 'intersection',
                '∧': 'and', '∨': 'or', '¬': 'not',
                '∞': 'infinity', '√': 'sqrt',
            }

            entry = {
                'char': symbol,
                'rpn': rpn,
                'font': 'Synthetic-Math',
                'type': 'math_synthetic',
                'category': category,
                'name': symbol_names.get(symbol, symbol),
                'text_relation': symbol_names.get(symbol, symbol)
            }
            dataset.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ Generated {len(dataset)} synthetic math symbols")
    return len(dataset)


def main():
    """Generate both atomic datasets."""
    print("\n" + "="*60)
    print("ATOMIC DATASETS GENERATION")
    print("Fonts + Math Symbols (Always Loaded)")
    print("="*60)

    # Output directory
    dataset_dir = Path("/K3D/Knowledge3D.local/datasets/atomic")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Generate font dataset
    font_path = dataset_dir / "fonts_procedural.jsonl"
    font_count = generate_font_dataset(font_path, max_samples=50000)

    # Generate math dataset
    math_path = dataset_dir / "math_symbols_procedural.jsonl"
    math_count = generate_math_dataset(math_path)

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Font glyphs:   {font_count:,} samples → {font_path}")
    print(f"Math symbols:  {math_count:,} samples → {math_path}")
    print(f"Total atomic:  {font_count + math_count:,} samples")
    print()
    print("These are ATOMIC UNITS - always loaded (like RPN opcodes)")
    print("Both text 'x' and math 'x' coexist, can relate but are separate galaxies")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
