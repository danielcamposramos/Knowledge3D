"""
Procedural font utilities.

Converts vector glyph outlines from TrueType/OpenType fonts into lightweight
segment descriptors that can be consumed by the GPU rasterizer. Each glyph is
represented as a list of line segments normalized to the range [-1, 1] in both
axes so the GPU kernel can operate without inspecting font units.
"""

from dataclasses import dataclass
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from fontTools import ttLib
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont


@dataclass(frozen=True)
class GlyphDescriptor:
    """CPU-side representation of a glyph outline."""

    segments: np.ndarray  # shape:(N,4) containing x0,y0,x1,y1 in [-1,1]
    advance: float
    weight: int | None = None
    italic: bool | None = None


def _evaluate_quadratic(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[float, float]:
    """Evaluate a quadratic Bézier curve at parameter t."""
    mt = 1.0 - t
    x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
    y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
    return x, y


def _evaluate_cubic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    """Evaluate a cubic Bézier curve at parameter t."""
    mt = 1.0 - t
    x = (
        mt * mt * mt * p0[0]
        + 3 * mt * mt * t * p1[0]
        + 3 * mt * t * t * p2[0]
        + t * t * t * p3[0]
    )
    y = (
        mt * mt * mt * p0[1]
        + 3 * mt * mt * t * p1[1]
        + 3 * mt * t * t * p2[1]
        + t * t * t * p3[1]
    )
    return x, y


def _flatten_curve(points: Sequence[Tuple[float, float]], steps: int) -> Iterable[Tuple[float, float]]:
    """Yield sampled positions along a curve."""
    if len(points) == 3:
        for i in range(steps + 1):
            yield _evaluate_quadratic(points[0], points[1], points[2], i / steps)
    elif len(points) == 4:
        for i in range(steps + 1):
            yield _evaluate_cubic(points[0], points[1], points[2], points[3], i / steps)
    else:
        for pt in points:
            yield pt


def _segments_from_pen(pen: RecordingPen, steps: int = 12) -> List[Tuple[float, float, float, float]]:
    """Convert pen commands into polyline segments."""
    segments: List[Tuple[float, float, float, float]] = []
    current_start: Tuple[float, float] | None = None
    current_pos: Tuple[float, float] | None = None

    for command, coords in pen.value:
        if command == "moveTo":
            current_start = coords[0]
            current_pos = coords[0]
        elif command == "lineTo":
            for pt in coords:
                if current_pos is not None:
                    segments.append((*current_pos, *pt))
                current_pos = pt
        elif command in {"curveTo", "qCurveTo"}:
            pts = [current_pos] + list(coords)
            if any(p is None for p in pts):
                continue
            samples = list(_flatten_curve(pts, steps))
            for idx in range(len(samples) - 1):
                start = samples[idx]
                end = samples[idx + 1]
                segments.append((*start, *end))
            current_pos = samples[-1]
        elif command == "closePath":
            if current_pos is not None and current_start is not None:
                segments.append((*current_pos, *current_start))
            current_pos = current_start
        else:
            continue

    return segments


@lru_cache(maxsize=1024)
def _load_font(font_path: str) -> TTFont:
    """Load and cache fonts for reuse."""
    return TTFont(font_path)


def _get_curve_steps() -> int:
    """Curve sampling granularity for Béziers.

    Tunable via env var K3D_GLYPH_STEPS (int). Defaults to 12 for high fidelity.
    Lower values (e.g., 8) improve training throughput with minor quality impact.
    """
    try:
        return max(4, int(os.environ.get("K3D_GLYPH_STEPS", "12")))
    except Exception:
        return 12


def extract_glyph(font_path: str, char: str) -> GlyphDescriptor:
    """
    Convert a glyph to normalized segments.

    Args:
        font_path: Path to font file.
        char: Single-character string.

    Returns:
        GlyphDescriptor with segments normalized to [-1, 1].
    """
    try:
        ttfont = _load_font(font_path)
        glyph_set = ttfont.getGlyphSet()
    except ttLib.TTLibError:
        return GlyphDescriptor(segments=np.zeros((0, 4), dtype=np.float32), advance=0.0, weight=None, italic=None)

    cmap_table = ttfont["cmap"] if "cmap" in ttfont else None
    cmap = cmap_table.getBestCmap() if cmap_table else None
    glyph_key = char[0] if char else " "
    glyph_name = cmap.get(ord(glyph_key)) if cmap else None
    if glyph_name is None:
        glyph_name = ttfont.getGlyphOrder()[0]
    glyph = glyph_set[glyph_name]

    pen = RecordingPen()
    glyph.draw(pen)
    raw_segments = _segments_from_pen(pen, steps=_get_curve_steps())

    if not raw_segments:
        return GlyphDescriptor(segments=np.zeros((0, 4), dtype=np.float32), advance=float(glyph.width or 0), weight=_get_weight(ttfont), italic=_get_italic(ttfont))

    units_per_em = ttfont["head"].unitsPerEm or 1000
    scale = 2.0 / units_per_em  # map font units to [-1,1]
    normalized: List[Tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in raw_segments:
        normalized.append(
            (
                (x0 * scale) - 1.0,
                (y0 * scale) - 1.0,
                (x1 * scale) - 1.0,
                (y1 * scale) - 1.0,
            )
        )

    segments = np.asarray(normalized, dtype=np.float32)
    return GlyphDescriptor(
        segments=segments,
        advance=float(glyph.width or units_per_em),
        weight=_get_weight(ttfont),
        italic=_get_italic(ttfont),
    )


@lru_cache(maxsize=16384)
def extract_glyph_cached(font_path: str, char: str) -> GlyphDescriptor:
    """LRU-cached glyph extraction to avoid repeated CPU work per epoch."""
    return extract_glyph(font_path, char)


def build_descriptor_batch(jobs: Sequence[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble batched descriptor buffers for GPU consumption.

    Args:
        jobs: Iterable of (font_path, character) pairs.

    Returns:
        segments, offsets, lengths arrays ready for GPU transfer.
    """
    all_segments: List[np.ndarray] = []
    offsets = np.zeros(len(jobs), dtype=np.int32)
    lengths = np.zeros(len(jobs), dtype=np.int32)

    cursor = 0
    for idx, (font_path, ch) in enumerate(jobs):
        descriptor = extract_glyph_cached(font_path, ch)
        segs = descriptor.segments
        all_segments.append(segs)
        offsets[idx] = cursor
        lengths[idx] = segs.shape[0]
        cursor += segs.shape[0]

    if cursor == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            offsets,
            lengths,
        )

    stacked = np.vstack(all_segments).astype(np.float32, copy=False)
    return stacked, offsets, lengths


def segments_to_rpn(
    segments: np.ndarray,
    *,
    stroke_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    stroke_width: float = 1.0,
) -> str:
    """
    Convert segment array (x0,y0,x1,y1) into an RPN drawing program.

    Returns a string with SET_COLOR/SET_LINE_WIDTH followed by MOVE/LINE and STROKE.
    """
    if segments.size == 0:
        return ""
    segs = np.asarray(segments, dtype=np.float32)
    program_tokens: List[str] = []
    r, g, b, a = stroke_color
    program_tokens.extend([str(r), str(g), str(b), str(a), "SET_COLOR"])
    program_tokens.extend([str(stroke_width), "STROKE_WIDTH"])
    # Start at first segment start
    x0, y0, x1, y1 = segs[0]
    program_tokens.extend([str(x0), str(y0), "MOVE", str(x1), str(y1), "LINE"])
    for seg in segs[1:]:
        program_tokens.extend([str(seg[2]), str(seg[3]), "LINE"])
    program_tokens.append("STROKE")
    return " ".join(program_tokens)


def glyph_to_rpn(
    font_path: str,
    char: str,
    *,
    stroke_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    stroke_width: float = 1.0,
) -> str:
    """
    Convert a font glyph into an RPN drawing program.
    """
    descriptor = extract_glyph(font_path, char)
    inferred_width = stroke_width * _infer_width_from_font(font_path)
    inferred_color = _infer_color_from_font(font_path, stroke_color)
    return segments_to_rpn(descriptor.segments, stroke_color=inferred_color, stroke_width=inferred_width)


def _infer_width_from_font(font_path: str) -> float:
    """Heuristic: adjust stroke width based on font weight keywords."""
    name = Path(font_path).name.lower()
    if "black" in name or "heavy" in name:
        return 1.8
    if "bold" in name or "semi" in name:
        return 1.4
    if "light" in name or "thin" in name:
        return 0.7
    return 1.0


def _infer_color_from_font(font_path: str, base_color: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Optional: tweak tint for italic/oblique to mark style."""
    name = Path(font_path).name.lower()
    r, g, b, a = base_color
    if "italic" in name or "oblique" in name:
        return (r, min(1.0, g * 0.9), min(1.0, b * 1.1), a)
    return base_color


def _get_weight(ttfont: TTFont) -> int | None:
    try:
        return int(ttfont["OS/2"].usWeightClass)
    except Exception:
        return None


def _get_italic(ttfont: TTFont) -> bool | None:
    try:
        mac_style = ttfont["head"].macStyle
        return bool(mac_style & 0x02)
    except Exception:
        return None


__all__ = [
    "GlyphDescriptor",
    "extract_glyph",
    "extract_glyph_cached",
    "build_descriptor_batch",
    "segments_to_rpn",
    "glyph_to_rpn",
]
