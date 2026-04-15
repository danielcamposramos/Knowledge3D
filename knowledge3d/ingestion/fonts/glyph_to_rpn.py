"""System font manifest and glyph-outline to Drawing Galaxy RPN extraction.

This module is ingestion-path only. It scans host-installed text fonts, records
their script coverage in the local runtime workspace, and converts individual
font glyph outlines into existing Drawing Galaxy RPN path operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from time import gmtime, strftime
from typing import Iterable, Iterator, Sequence
import unicodedata

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTCollection, TTFont

from knowledge3d.ingestion.canonical_lookup import canonical_slug


SYSTEM_FONT_ROOT = Path("/usr/share/fonts")
LOCAL_FONT_ASSET_DIR = Path("/K3D/Knowledge3D.local/assets/fonts/text")
MANIFEST_PATH = LOCAL_FONT_ASSET_DIR / "MANIFEST.json"
TEXT_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
TARGET_EM_SQUARE = 2048

_SYMBOL_EXCLUSION_TOKENS = (
    "wingdings",
    "webdings",
    "fontawesome",
    "powerlinesymbols",
    "standardsymbolsps",
    "d050000l",
    "symbola",
    "povraylogo",
    "stmary10",
    "wasy10",
    "rsfs10",
    "dingbats",
    "emoji",
)

_EXACT_SYMBOL_NAMES = {"symbol"}

_SCRIPT_RANGES = (
    ("latn", ((0x0000, 0x024F), (0x1E00, 0x1EFF))),
    ("grek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrl", ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ("arab", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("deva", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("beng", ((0x0980, 0x09FF),)),
    ("guru", ((0x0A00, 0x0A7F),)),
    ("gujr", ((0x0A80, 0x0AFF),)),
    ("taml", ((0x0B80, 0x0BFF),)),
    ("mlym", ((0x0D00, 0x0D7F),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("tibt", ((0x0F00, 0x0FFF),)),
    ("geor", ((0x10A0, 0x10FF), (0x1C90, 0x1CBF))),
    ("hang", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("hira", ((0x3040, 0x309F),)),
    ("kana", ((0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("hani", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
    ("hebr", ((0x0590, 0x05FF),)),
    ("ethi", ((0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F))),
    ("math", ((0x2150, 0x218F), (0x2200, 0x22FF), (0x27C0, 0x27EF), (0x2980, 0x29FF), (0x2A00, 0x2AFF), (0x1D400, 0x1D7FF))),
)


@dataclass(frozen=True)
class GlyphMetrics:
    advance_width: int
    lsb: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    em_square: int = TARGET_EM_SQUARE
    source_units_per_em: int = TARGET_EM_SQUARE

    def to_dict(self) -> dict[str, int]:
        return {
            "advance_width": self.advance_width,
            "lsb": self.lsb,
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "em_square": self.em_square,
            "source_units_per_em": self.source_units_per_em,
        }


@dataclass(frozen=True)
class GlyphRPN:
    rpn_program: str
    rpn_bytes: bytes
    metrics: GlyphMetrics
    contour_count: int
    opcode_count: int


def _compact_token(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def is_symbol_font(family: str, path: str | Path, *, style: str = "") -> bool:
    """Return True for symbol/pictogram fonts excluded from text letter stars."""

    family_key = _compact_token(family)
    style_key = _compact_token(style)
    path_key = _compact_token(Path(path).stem)
    haystack = f"{family_key} {style_key} {path_key}"
    if family_key in _EXACT_SYMBOL_NAMES or path_key in _EXACT_SYMBOL_NAMES:
        return True
    return any(token in haystack for token in _SYMBOL_EXCLUSION_TOKENS)


def iter_system_font_files(root: str | Path = SYSTEM_FONT_ROOT) -> Iterator[Path]:
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_FONT_EXTENSIONS:
            yield path


def _name_record(font: TTFont, name_ids: Sequence[int]) -> str | None:
    names = font.get("name")
    if names is None:
        return None
    for name_id in name_ids:
        record = names.getName(name_id, 3, 1, 0x409) or names.getName(name_id, 1, 0, 0)
        if record is not None:
            value = record.toUnicode().strip()
            if value:
                return value
    return None


def font_family_style(font: TTFont, path: str | Path) -> tuple[str, str]:
    family = _name_record(font, (16, 1)) or Path(path).stem
    style = _name_record(font, (17, 2)) or "Regular"
    return family, style


def _font_count(path: Path) -> int:
    if path.suffix.lower() != ".ttc":
        return 1
    collection = TTCollection(str(path), lazy=True)
    try:
        return len(collection.fonts)
    finally:
        for font in collection.fonts:
            font.close()


def _open_font(path: str | Path, font_index: int = 0) -> TTFont:
    return TTFont(str(path), fontNumber=font_index, lazy=True)


def script_for_codepoint(codepoint: int) -> str:
    for script, ranges in _SCRIPT_RANGES:
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    category = unicodedata.category(chr(codepoint))
    if category.startswith("L"):
        return "letter_other"
    if category.startswith("N"):
        return "number"
    return "common"


def scripts_covered(codepoints: Iterable[int]) -> list[str]:
    scripts = {script_for_codepoint(cp) for cp in codepoints}
    scripts.discard("common")
    return sorted(scripts or {"common"})


def _compress_codepoint_ranges(codepoints: Iterable[int]) -> list[list[str]]:
    sorted_points = sorted(set(codepoints))
    if not sorted_points:
        return []
    ranges: list[list[str]] = []
    start = prev = sorted_points[0]
    for cp in sorted_points[1:]:
        if cp == prev + 1:
            prev = cp
            continue
        ranges.append([f"U+{start:04X}", f"U+{prev:04X}"])
        start = prev = cp
    ranges.append([f"U+{start:04X}", f"U+{prev:04X}"])
    return ranges


def _font_manifest_entry(path: Path, font_index: int = 0) -> dict | None:
    font = _open_font(path, font_index)
    try:
        cmap = font.getBestCmap() or {}
        family, style = font_family_style(font, path)
        if not cmap or is_symbol_font(family, path, style=style):
            return None
        codepoints = sorted(int(cp) for cp in cmap)
        return {
            "path": str(path),
            "file": path.name,
            "font_index": font_index,
            "family": family,
            "style": style,
            "scripts": scripts_covered(codepoints),
            "codepoint_count": len(codepoints),
            "codepoint_ranges": _compress_codepoint_ranges(codepoints),
            "unreadable_codepoints": [],
        }
    finally:
        font.close()


def build_system_font_manifest(
    *,
    font_root: str | Path = SYSTEM_FONT_ROOT,
    output_path: str | Path = MANIFEST_PATH,
    font_paths: Sequence[str | Path] | None = None,
) -> dict:
    """Scan system fonts and write the local text-font manifest."""

    paths = [Path(p) for p in font_paths] if font_paths is not None else list(iter_system_font_files(font_root))
    entries: list[dict] = []
    excluded: list[dict] = []
    unreadable: list[dict] = []
    for path in paths:
        try:
            count = _font_count(path)
            for font_index in range(count):
                entry = _font_manifest_entry(path, font_index)
                if entry is None:
                    excluded.append({"path": str(path), "font_index": font_index})
                else:
                    entries.append(entry)
        except Exception as exc:
            unreadable.append({"path": str(path), "error": str(exc)})

    payload = {
        "schema": "k3d_system_text_font_manifest_v1",
        "generated_at_utc": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "font_root": str(font_root),
        "font_count": len(entries),
        "excluded_count": len(excluded),
        "unreadable_count": len(unreadable),
        "symbol_exclusion_tokens": list(_SYMBOL_EXCLUSION_TOKENS) + sorted(_EXACT_SYMBOL_NAMES),
        "fonts": entries,
        "excluded": excluded[:200],
        "unreadable": unreadable[:200],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mark_unreadable_codepoints(
    manifest: dict,
    failures: Iterable[Mapping[str, str]],
) -> dict:
    """Annotate manifest entries with codepoints whose glyph outlines failed."""

    by_font: dict[tuple[str, int], set[str]] = {}
    for failure in failures:
        font = str(failure.get("font") or failure.get("path") or "").strip()
        if not font:
            continue
        try:
            font_index = int(failure.get("font_index") or 0)
        except Exception:
            font_index = 0
        codepoint = str(failure.get("codepoint") or "").strip().upper()
        if codepoint and not codepoint.startswith("U+"):
            codepoint = f"U+{int(codepoint, 16):04X}"
        if codepoint:
            by_font.setdefault((font, font_index), set()).add(codepoint)
    for entry in manifest.get("fonts", []) or []:
        key = (str(entry.get("path") or ""), int(entry.get("font_index") or 0))
        additions = by_font.get(key)
        if not additions:
            continue
        existing = {str(value).upper() for value in entry.get("unreadable_codepoints", []) or []}
        entry["unreadable_codepoints"] = sorted(existing | additions)
    return manifest


def update_manifest_unreadable_codepoints(
    manifest_path: str | Path,
    failures: Iterable[Mapping[str, str]],
) -> dict:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    updated = mark_unreadable_codepoints(manifest, failures)
    manifest_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated


def glyph_key(family: str, style: str, codepoint: int) -> str:
    return f"{family}::{style}::U+{codepoint:04X}"


def glyph_star_id(family: str, style: str, codepoint: int) -> str:
    return f"font_glyph_{canonical_slug(family)}_{canonical_slug(style)}_u{codepoint:04x}"


def _normalize_coord(value: float, units_per_em: int) -> int:
    scaled = int(round(float(value) * TARGET_EM_SQUARE / float(units_per_em or TARGET_EM_SQUARE)))
    return max(-32768, min(32767, scaled))


def _normalize_point(point: tuple[float, float], units_per_em: int) -> tuple[int, int]:
    return (_normalize_coord(point[0], units_per_em), _normalize_coord(point[1], units_per_em))


def _midpoint(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)


def _append_point_command(tokens: list[str], units_per_em: int, point: tuple[float, float], opcode: str) -> None:
    x, y = _normalize_point(point, units_per_em)
    tokens.extend((str(x), str(y), opcode))


def _append_quad(tokens: list[str], units_per_em: int, control: tuple[float, float], end: tuple[float, float]) -> None:
    cx, cy = _normalize_point(control, units_per_em)
    ex, ey = _normalize_point(end, units_per_em)
    tokens.extend((str(cx), str(cy), str(ex), str(ey), "QUAD"))


def _append_cubic(
    tokens: list[str],
    units_per_em: int,
    c1: tuple[float, float],
    c2: tuple[float, float],
    end: tuple[float, float],
) -> None:
    c1x, c1y = _normalize_point(c1, units_per_em)
    c2x, c2y = _normalize_point(c2, units_per_em)
    ex, ey = _normalize_point(end, units_per_em)
    tokens.extend((str(c1x), str(c1y), str(c2x), str(c2y), str(ex), str(ey), "CUBIC"))


def _emit_qcurve(
    tokens: list[str],
    units_per_em: int,
    current: tuple[float, float] | None,
    points: Sequence[tuple[float, float] | None],
) -> tuple[float, float] | None:
    if not points:
        return current
    if len(points) == 1 and points[0] is not None:
        _append_point_command(tokens, units_per_em, points[0], "LINE")
        return points[0]
    off_curves = [p for p in points[:-1] if p is not None]
    end = points[-1]
    if end is None:
        if not off_curves:
            return current
        end = _midpoint(off_curves[-1], off_curves[0])
    if not off_curves:
        _append_point_command(tokens, units_per_em, end, "LINE")
        return end
    for idx, control in enumerate(off_curves):
        quad_end = end if idx == len(off_curves) - 1 else _midpoint(control, off_curves[idx + 1])
        _append_quad(tokens, units_per_em, control, quad_end)
        current = quad_end
    return current


def _glyph_metrics(font: TTFont, glyph_name: str, units_per_em: int, bounds: tuple[float, float, float, float] | None) -> GlyphMetrics:
    advance_width, lsb = (0, 0)
    hmtx = font.get("hmtx")
    if hmtx is not None and glyph_name in hmtx.metrics:
        advance_width, lsb = hmtx.metrics[glyph_name]
    if bounds is None:
        xmin = ymin = xmax = ymax = 0
    else:
        xmin, ymin, xmax, ymax = bounds
    return GlyphMetrics(
        advance_width=_normalize_coord(advance_width, units_per_em),
        lsb=_normalize_coord(lsb, units_per_em),
        xmin=_normalize_coord(xmin, units_per_em),
        ymin=_normalize_coord(ymin, units_per_em),
        xmax=_normalize_coord(xmax, units_per_em),
        ymax=_normalize_coord(ymax, units_per_em),
        source_units_per_em=units_per_em,
    )


def extract_glyph_rpn(font_file: str | Path, codepoint: int, *, font_index: int = 0) -> GlyphRPN:
    """Extract a single glyph outline as textual Drawing Galaxy RPN bytes."""

    font_file = Path(font_file)
    font = _open_font(font_file, font_index)
    try:
        cmap = font.getBestCmap() or {}
        glyph_name = cmap.get(codepoint)
        if glyph_name is None:
            raise KeyError(f"glyph_missing:U+{codepoint:04X}:{font_file}")
        glyph_set = font.getGlyphSet()
        glyph = glyph_set[glyph_name]
        units_per_em = int(font["head"].unitsPerEm) if "head" in font else TARGET_EM_SQUARE

        pen = DecomposingRecordingPen(glyph_set)
        glyph.draw(pen)

        bounds_pen = BoundsPen(glyph_set)
        glyph.draw(bounds_pen)
        metrics = _glyph_metrics(font, glyph_name, units_per_em, bounds_pen.bounds)

        tokens: list[str] = []
        contour_count = 0
        current: tuple[float, float] | None = None
        contour_open = False
        for operator, operands in pen.value:
            if operator == "moveTo":
                point = operands[0]
                _append_point_command(tokens, units_per_em, point, "MOVE")
                current = point
                contour_count += 1
                contour_open = True
            elif operator == "lineTo":
                point = operands[0]
                _append_point_command(tokens, units_per_em, point, "LINE")
                current = point
            elif operator == "qCurveTo":
                current = _emit_qcurve(tokens, units_per_em, current, operands)
            elif operator == "curveTo":
                points = list(operands)
                for idx in range(0, len(points), 3):
                    chunk = points[idx : idx + 3]
                    if len(chunk) == 3:
                        _append_cubic(tokens, units_per_em, chunk[0], chunk[1], chunk[2])
                        current = chunk[2]
            elif operator == "closePath":
                tokens.append("CLOSE")
                contour_open = False
            elif operator == "endPath":
                contour_open = False
        if contour_open:
            tokens.append("CLOSE")
        if not tokens:
            raise ValueError(f"glyph_empty:U+{codepoint:04X}:{font_file}")
        tokens.append("STROKE")
        rpn_program = " ".join(tokens)
        opcode_count = sum(1 for tok in tokens if tok.isalpha() or "_" in tok)
        return GlyphRPN(
            rpn_program=rpn_program,
            rpn_bytes=rpn_program.encode("utf-8"),
            metrics=metrics,
            contour_count=contour_count,
            opcode_count=opcode_count,
        )
    finally:
        font.close()


def font_glyph_metadata(family: str, style: str, codepoint: int, glyph: GlyphRPN) -> dict:
    return {
        "family": family,
        "style": style,
        "script": script_for_codepoint(codepoint),
        "em_square": TARGET_EM_SQUARE,
        "advance_width": glyph.metrics.advance_width,
        "contour_count": glyph.contour_count,
        "opcode_count": glyph.opcode_count,
        "glyph_metrics": glyph.metrics.to_dict(),
    }


def register_font_glyph(
    lookup,
    *,
    family: str,
    style: str,
    codepoint: int,
    glyph: GlyphRPN,
) -> str:
    """Register one extracted glyph in ``k3d_canonical`` as ``font_glyph``."""

    star_id = glyph_star_id(family, style, codepoint)
    lookup.register(
        kind="font_glyph",
        key=glyph_key(family, style, codepoint),
        star_id=star_id,
        metadata=font_glyph_metadata(family, style, codepoint, glyph),
    )
    return star_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the K3D local system text-font manifest.")
    parser.add_argument("--font-root", default=str(SYSTEM_FONT_ROOT))
    parser.add_argument("--out", default=str(MANIFEST_PATH))
    args = parser.parse_args(argv)
    manifest = build_system_font_manifest(font_root=args.font_root, output_path=args.out)
    print(
        f"font_count={manifest['font_count']} "
        f"excluded_count={manifest['excluded_count']} "
        f"unreadable_count={manifest['unreadable_count']} "
        f"manifest={args.out}"
    )
    return 0


__all__ = [
    "GlyphMetrics",
    "GlyphRPN",
    "LOCAL_FONT_ASSET_DIR",
    "MANIFEST_PATH",
    "SYSTEM_FONT_ROOT",
    "TARGET_EM_SQUARE",
    "build_system_font_manifest",
    "extract_glyph_rpn",
    "font_family_style",
    "font_glyph_metadata",
    "glyph_key",
    "glyph_star_id",
    "is_symbol_font",
    "iter_system_font_files",
    "load_manifest",
    "main",
    "mark_unreadable_codepoints",
    "register_font_glyph",
    "script_for_codepoint",
    "scripts_covered",
    "update_manifest_unreadable_codepoints",
]


if __name__ == "__main__":
    raise SystemExit(main())
