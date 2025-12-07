#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract procedural glyph definitions for canonical math symbols.

Converts font Bézier curves → RPN drawing programs.
Stores results in Math Galaxy as canonical knowledge (NOT trained weights).

Usage:
    python scripts/extract_math_symbol_glyphs.py --font /path/to/STIXTwoMath-Regular.otf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from knowledge3d.cranium.math_galaxy import get_math_galaxy


def extract_glyph_bezier(char: str, font_path: Path) -> List[Tuple]:
    """
    Extract Bézier curve segments for a glyph using fontTools.
    Falls back to placeholder if fontTools is unavailable.
    """
    try:
        from fontTools.ttLib import TTFont
        from fontTools.pens.recordingPen import RecordingPen
    except ImportError:
        print("  [WARN] fontTools not installed, using placeholder RPN")
        return []

    font = TTFont(str(font_path))
    cmap = font.getBestCmap()
    codepoint = ord(char)
    if codepoint not in cmap:
        print(f"  [WARN] Character {char} (U+{codepoint:04X}) not in font")
        return []

    glyph_name = cmap[codepoint]
    glyph_set = font.getGlyphSet()
    pen = RecordingPen()
    glyph_set[glyph_name].draw(pen)

    segments: List[Tuple] = []
    for op, args in pen.value:
        if op == "moveTo":
            segments.append(("MOVE", args[0]))
        elif op == "lineTo":
            segments.append(("LINE", args[0]))
        elif op == "curveTo":
            segments.append(("CURVE", args))
        elif op == "qCurveTo":
            segments.append(("QCURVE", args))
        elif op == "closePath":
            segments.append(("CLOSE", None))
    return segments


def bezier_to_rpn(segments: List[Tuple], canvas_size: int = 64) -> str:
    """
    Convert Bézier segments to a procedural RPN drawing program using engine opcodes.
    """
    if not segments:
        # Simple placeholder shape with valid opcodes
        return "32 32 MOVE 32 48 LINE 48 48 LINE 48 32 LINE CLOSE STROKE"

    rpn_ops: List[str] = []
    scale = canvas_size / 1000.0
    for op_type, args in segments:
        if op_type == "MOVE" and args:
            x, y = int(args[0] * scale), int(args[1] * scale)
            rpn_ops.append(f"{x} {y} MOVE")
        elif op_type == "LINE" and args:
            x, y = int(args[0] * scale), int(args[1] * scale)
            rpn_ops.append(f"{x} {y} LINE")
        elif op_type == "CURVE" and args:
            # Cubic Bézier: two control points + end point
            if len(args) >= 3:
                c1x = int(args[0][0] * scale)
                c1y = int(args[0][1] * scale)
                c2x = int(args[1][0] * scale)
                c2y = int(args[1][1] * scale)
                x = int(args[2][0] * scale)
                y = int(args[2][1] * scale)
                rpn_ops.append(f"{c1x} {c1y} {c2x} {c2y} {x} {y} CUBIC")
        elif op_type == "QCURVE" and args:
            # Quadratic Bézier: control point + end point
            if len(args) >= 2:
                cx = int(args[0][0] * scale)
                cy = int(args[0][1] * scale)
                x = int(args[1][0] * scale)
                y = int(args[1][1] * scale)
                rpn_ops.append(f"{cx} {cy} {x} {y} QUAD")
        elif op_type == "CLOSE":
            rpn_ops.append("CLOSE")
    rpn_ops.append("STROKE")
    return " ".join(rpn_ops)


def extract_all_math_symbols(font_path: Path, output: Path | None = None) -> None:
    galaxy = get_math_galaxy()
    extracted = 0
    failed = 0

    for codepoint, symbol in galaxy.symbols.items():
        segments = extract_glyph_bezier(symbol.char, font_path)
        if segments:
            symbol.bezier_segments = segments
            symbol.rpn_program = bezier_to_rpn(segments)
            extracted += 1
            print(f"  ✓ {symbol.char} (U+{codepoint:04X}): {len(segments)} segments")
        else:
            failed += 1
            print(f"  ✗ {symbol.char} (U+{codepoint:04X}): using placeholder RPN")

    target = output or (galaxy.storage_path / "math_galaxy.json")
    galaxy.save(target)
    total = len(galaxy.symbols)
    print(f"\n{'='*50}")
    print(f"Extracted: {extracted}/{total} symbols")
    print(f"Failed: {failed} (placeholders kept)")
    print(f"Saved to: {target}")
    print(f"{'='*50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract math symbol glyphs to RPN")
    parser.add_argument("--font", type=Path, required=True, help="Path to math font (OTF/TTF)")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if not args.font.exists():
        raise SystemExit(f"ERROR: Font not found: {args.font}")

    print(f"Extracting math symbols from: {args.font}")
    extract_all_math_symbols(args.font, args.output)


if __name__ == "__main__":
    main()
