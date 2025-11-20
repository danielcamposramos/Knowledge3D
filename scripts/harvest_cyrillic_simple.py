#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural glyph harvester for multiple scripts.

Scans installed fonts (system + .local) and converts vector outlines into
MOVE/LINE/QUAD/CUBIC RPN programs with full font metadata and language coverage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from fontTools.ttLib import TTFont

from knowledge3d.cranium.procedural_fonts import extract_glyph_cached, segments_to_rpn
from knowledge3d.cranium.specialists.character_languages import get_character_languages

BASE_DATASET_DIR = Path("/K3D/Knowledge3D.local/datasets/atomic")
EXTRA_FONT_ROOT = Path("/K3D/Knowledge3D.local/fonts")
DEFAULT_FONT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / ".local/share/fonts"),
    str(EXTRA_FONT_ROOT),
)

LEGACY_OUTPUT_PATH = BASE_DATASET_DIR / "fonts_cyrillic_simple.jsonl"


@dataclass(frozen=True)
class ScriptDefinition:
    name: str
    label: str
    ranges: Tuple[Tuple[int, int], ...]
    extras: Tuple[str, ...] = ()
    max_chars: int | None = None
    description: str = ""

    def output_path(self, root: Path) -> Path:
        return root / f"fonts_{self.name}_procedural.jsonl"


SCRIPT_DEFINITIONS: Dict[str, ScriptDefinition] = {
    "latin_basic": ScriptDefinition(
        name="latin_basic",
        label="Latin (Basic)",
        ranges=(
            (0x0041, 0x005A),
            (0x0061, 0x007A),
        ),
        extras=tuple("0123456789"),
    ),
    "latin_extended": ScriptDefinition(
        name="latin_extended",
        label="Latin Extended",
        ranges=(
            (0x0100, 0x024F),
            (0x1E00, 0x1EFF),
        ),
    ),
    "cyrillic": ScriptDefinition(
        name="cyrillic",
        label="Cyrillic + Supplement",
        ranges=((0x0400, 0x052F),),
        extras=('Ё', 'ё', 'Є', 'є', 'Ї', 'ї', 'Ґ', 'ґ', 'І', 'і', 'Ў', 'ў'),
    ),
    "greek": ScriptDefinition(
        name="greek",
        label="Greek",
        ranges=((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    ),
    "hebrew": ScriptDefinition(
        name="hebrew",
        label="Hebrew",
        ranges=((0x0590, 0x05FF),),
    ),
    "arabic": ScriptDefinition(
        name="arabic",
        label="Arabic + Presentation Forms",
        ranges=((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
        max_chars=768,
    ),
    "devanagari": ScriptDefinition(
        name="devanagari",
        label="Devanagari",
        ranges=((0x0900, 0x097F),),
    ),
    "bengali": ScriptDefinition(
        name="bengali",
        label="Bengali / Assamese",
        ranges=((0x0980, 0x09FF),),
    ),
    "gurmukhi": ScriptDefinition(
        name="gurmukhi",
        label="Gurmukhi",
        ranges=((0x0A00, 0x0A7F),),
    ),
    "gujarati": ScriptDefinition(
        name="gujarati",
        label="Gujarati",
        ranges=((0x0A80, 0x0AFF),),
    ),
    "oriya": ScriptDefinition(
        name="oriya",
        label="Oriya",
        ranges=((0x0B00, 0x0B7F),),
    ),
    "tamil": ScriptDefinition(
        name="tamil",
        label="Tamil",
        ranges=((0x0B80, 0x0BFF),),
    ),
    "telugu": ScriptDefinition(
        name="telugu",
        label="Telugu",
        ranges=((0x0C00, 0x0C7F),),
    ),
    "kannada": ScriptDefinition(
        name="kannada",
        label="Kannada",
        ranges=((0x0C80, 0x0CFF),),
    ),
    "malayalam": ScriptDefinition(
        name="malayalam",
        label="Malayalam",
        ranges=((0x0D00, 0x0D7F),),
    ),
    "sinhala": ScriptDefinition(
        name="sinhala",
        label="Sinhala",
        ranges=((0x0D80, 0x0DFF),),
    ),
    "thai": ScriptDefinition(
        name="thai",
        label="Thai",
        ranges=((0x0E00, 0x0E7F),),
    ),
    "lao": ScriptDefinition(
        name="lao",
        label="Lao",
        ranges=((0x0E80, 0x0EFF),),
    ),
    "tibetan": ScriptDefinition(
        name="tibetan",
        label="Tibetan",
        ranges=((0x0F00, 0x0FFF),),
    ),
    "myanmar": ScriptDefinition(
        name="myanmar",
        label="Myanmar",
        ranges=((0x1000, 0x109F), (0xA9E0, 0xA9FF)),
    ),
    "georgian": ScriptDefinition(
        name="georgian",
        label="Georgian",
        ranges=((0x10A0, 0x10FF), (0x2D00, 0x2D2F), (0x1C90, 0x1CBF)),
    ),
    "armenian": ScriptDefinition(
        name="armenian",
        label="Armenian",
        ranges=((0x0530, 0x058F),),
    ),
    "ethiopic": ScriptDefinition(
        name="ethiopic",
        label="Ethiopic",
        ranges=((0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF)),
        max_chars=640,
    ),
    "cherokee": ScriptDefinition(
        name="cherokee",
        label="Cherokee",
        ranges=((0x13A0, 0x13FF), (0xAB70, 0xABBF)),
    ),
    "canadian": ScriptDefinition(
        name="canadian",
        label="Canadian Aboriginal Syllabics",
        ranges=((0x1400, 0x167F),),
        max_chars=512,
    ),
    "braille": ScriptDefinition(
        name="braille",
        label="Braille Patterns",
        ranges=((0x2800, 0x28FF),),
    ),
    "bopomofo": ScriptDefinition(
        name="bopomofo",
        label="Bopomofo",
        ranges=((0x3100, 0x312F), (0x31A0, 0x31BF)),
    ),
    "hiragana": ScriptDefinition(
        name="hiragana",
        label="Hiragana",
        ranges=((0x3040, 0x309F),),
    ),
    "katakana": ScriptDefinition(
        name="katakana",
        label="Katakana",
        ranges=((0x30A0, 0x30FF), (0x31F0, 0x31FF)),
    ),
    "hangul": ScriptDefinition(
        name="hangul",
        label="Hangul Syllables",
        ranges=((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)),
        max_chars=768,
    ),
    "cjk_common": ScriptDefinition(
        name="cjk_common",
        label="CJK Unified Ideographs (sampled)",
        ranges=((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)),
        max_chars=1024,
    ),
}

DEFAULT_SCRIPTS = ["cyrillic"]


def available_script_names() -> List[str]:
    return sorted(SCRIPT_DEFINITIONS.keys())


def _load_extra_chars(file_path: Path | None) -> List[str]:
    if not file_path:
        return []
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    chars: List[str] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chars.append(line[0])
    return chars


def _iter_font_files(font_dirs: Sequence[str], limit: int | None) -> List[Path]:
    fonts: List[Path] = []
    seen: Set[Path] = set()
    for directory in font_dirs:
        base = Path(directory).expanduser()
        if not base.exists():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for path in sorted(base.rglob(pattern)):
                if path in seen:
                    continue
                seen.add(path)
                fonts.append(path)
                if limit is not None and len(fonts) >= limit:
                    return fonts
    return fonts


def _name_record(name_table, name_id: int) -> str | None:
    if name_table is None:
        return None
    candidates = [
        (3, 1, 0x409),
        (1, 0, 0),
        (0, 3, 0),
        (0, 0, 0),
    ]
    for platform_id, enc_id, lang_id in candidates:
        record = name_table.getName(name_id, platform_id, enc_id, lang_id)
        if record:
            try:
                return str(record)
            except Exception:
                try:
                    return record.toUnicode()
                except Exception:
                    continue
    return None


def _extract_font_metadata(font_path: Path) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "font_family": font_path.stem,
        "font_name": font_path.stem,
        "font_weight": 400,
        "font_style": "normal",
        "font_variant": "regular",
        "font_source": "system",
    }
    ttfont: TTFont | None = None
    try:
        ttfont = TTFont(str(font_path))
        name_table = ttfont["name"] if "name" in ttfont else None
        family = _name_record(name_table, 1)
        subfamily = _name_record(name_table, 2)
        full_name = _name_record(name_table, 4)
        if family:
            meta["font_family"] = family
        if full_name:
            meta["font_name"] = full_name
        if subfamily:
            meta["font_variant"] = subfamily.lower()
        if "OS/2" in ttfont:
            try:
                meta["font_weight"] = int(ttfont["OS/2"].usWeightClass)
            except Exception:
                pass
        if "head" in ttfont:
            mac_style = getattr(ttfont["head"], "macStyle", 0)
            if mac_style & 0x02:
                meta["font_style"] = "italic"
            elif mac_style & 0x20:
                meta["font_variant"] = "condensed"
            elif mac_style & 0x01:
                meta["font_variant"] = "bold"
    except Exception as exc:
        logging.debug("Failed to parse metadata for %s: %s", font_path, exc)
    finally:
        if ttfont is not None:
            ttfont.close()
    return meta


def _infer_width_from_font(name: str) -> float:
    lowered = name.lower()
    if "black" in lowered or "heavy" in lowered:
        return 1.8
    if "bold" in lowered or "semi" in lowered:
        return 1.4
    if "light" in lowered or "thin" in lowered:
        return 0.7
    return 1.0


def _infer_color_from_font(name: str, base: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> Tuple[float, float, float, float]:
    lowered = name.lower()
    r, g, b, a = base
    if "italic" in lowered or "oblique" in lowered:
        return (r, min(1.0, g * 0.9), min(1.0, b * 1.1), a)
    return base


@dataclass
class HarvestStats:
    script: ScriptDefinition
    fonts_scanned: int = 0
    fonts_with_script: int = 0
    glyph_entries: int = 0
    char_counts: Counter = field(default_factory=Counter)
    language_hits: Counter = field(default_factory=Counter)

    def register(self, char: str, languages: Sequence[str]) -> None:
        self.glyph_entries += 1
        self.char_counts[char] += 1
        for lang in languages or []:
            self.language_hits[lang] += 1


def _build_charset(defn: ScriptDefinition, global_extras: Sequence[str], extra_file_chars: Sequence[str], limit_override: Dict[str, int]) -> List[str]:
    chars: List[str] = []
    seen: Set[str] = set()
    for start, end in defn.ranges:
        for code in range(start, end + 1):
            ch = chr(code)
            if unicodedata.category(ch).startswith("C"):
                continue
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    for group in (defn.extras, tuple(global_extras), tuple(extra_file_chars)):
        for ch in group:
            if not ch:
                continue
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    limit = limit_override.get(defn.name, defn.max_chars)
    if limit and len(chars) > limit:
        chars = chars[:limit]
    return chars


def _harvest_font(font_path: Path, charset: Sequence[str], min_segments: int, script: ScriptDefinition) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    metadata = _extract_font_metadata(font_path)
    stroke_width = _infer_width_from_font(font_path.name)
    stroke_color = _infer_color_from_font(font_path.name)

    for char in charset:
        try:
            descriptor = extract_glyph_cached(str(font_path), char)
        except Exception:
            continue
        segments = descriptor.segments
        if segments.shape[0] < min_segments:
            continue
        rpn = segments_to_rpn(segments, stroke_width=stroke_width, stroke_color=stroke_color)
        if not rpn:
            continue
        languages = get_character_languages(char)
        entry: Dict[str, object] = {
            "char": char,
            "rpn": rpn,
            "visual_rpn": rpn,
            "font": metadata.get("font_family"),
            "font_path": str(font_path),
            "type": "glyph",
            "category": script.name,
            "script": script.name,
            "script_label": script.label,
            "languages": languages,
            "segments": int(segments.shape[0]),
            "unicode_codepoint": f"U+{ord(char):04X}",
        }
        entry.update(metadata)
        entries.append(entry)
    return entries


def harvest_script(
    font_paths: Sequence[Path],
    charset: Sequence[str],
    script: ScriptDefinition,
    min_segments: int,
) -> Tuple[List[Dict[str, object]], HarvestStats]:
    stats = HarvestStats(script=script, fonts_scanned=len(font_paths))
    dataset: List[Dict[str, object]] = []
    for font_path in font_paths:
        font_entries = _harvest_font(font_path, charset, min_segments, script)
        if not font_entries:
            continue
        stats.fonts_with_script += 1
        for entry in font_entries:
            dataset.append(entry)
            stats.register(entry["char"], entry.get("languages", []))
    return dataset, stats


def _write_dataset(entries: Sequence[Dict[str, object]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _mirror_to_legacy(new_path: Path, legacy_path: Path) -> None:
    try:
        if legacy_path.exists() or legacy_path.is_symlink():
            legacy_path.unlink()
        os.link(new_path, legacy_path)
    except OSError:
        shutil.copyfile(new_path, legacy_path)


def _parse_limit_overrides(values: Sequence[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for item in values or []:
        if "=" not in item:
            continue
        name, raw = item.split("=", 1)
        try:
            overrides[name.strip()] = max(1, int(raw))
        except ValueError:
            logging.warning("Invalid script limit value: %s", item)
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest procedural glyph datasets for multiple scripts.")
    parser.add_argument("--scripts", nargs="*", default=DEFAULT_SCRIPTS, help="Scripts to harvest (use 'all' for every available definition).")
    parser.add_argument("--max-fonts", type=int, default=250, help="Maximum number of fonts to scan.")
    parser.add_argument("--min-segments", type=int, default=4, help="Minimum segments per glyph.")
    parser.add_argument("--font-dirs", nargs="*", default=DEFAULT_FONT_DIRS, help="Extra font directories to include.")
    parser.add_argument("--output-dir", type=Path, default=BASE_DATASET_DIR, help="Directory for dataset JSONL files.")
    parser.add_argument("--legacy-output", type=Path, default=LEGACY_OUTPUT_PATH, help="Legacy path for Cyrillic mirroring.")
    parser.add_argument("--no-legacy-mirror", action="store_true", help="Disable mirroring to the legacy Cyrillic dataset.")
    parser.add_argument("--charset-file", type=Path, help="File with one character per line to append to every charset.")
    parser.add_argument("--chars", type=str, default="", help="Extra characters to append to every charset.")
    parser.add_argument("--script-limit", action="append", default=[], help="Override max chars per script (e.g., cjk_common=4096).")
    parser.add_argument("--list-scripts", action="store_true", help="List available script names and exit.")
    return parser.parse_args()


def resolve_scripts(requested: Sequence[str]) -> List[ScriptDefinition]:
    if not requested or requested == ["all"]:
        names = available_script_names()
    else:
        names: List[str] = []
        for item in requested:
            if item.lower() == "all":
                names = available_script_names()
                break
            if item not in SCRIPT_DEFINITIONS:
                raise ValueError(f"Unknown script '{item}'. Use --list-scripts to see options.")
            names.append(item)
    return [SCRIPT_DEFINITIONS[name] for name in names]


def main() -> None:
    args = parse_args()
    if args.list_scripts:
        print("Available scripts:")
        for name in available_script_names():
            cfg = SCRIPT_DEFINITIONS[name]
            limit = cfg.max_chars if cfg.max_chars else "∞"
            print(f" - {name:16s} :: {cfg.label} (limit={limit})")
        return

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    scripts = resolve_scripts(args.scripts)
    font_paths = _iter_font_files(args.font_dirs, args.max_fonts)
    if not font_paths:
        raise RuntimeError("No fonts found. Install fonts or adjust --font-dirs.")

    limit_overrides = _parse_limit_overrides(args.script_limit)
    extra_chars = list(args.chars) if args.chars else []
    extra_file_chars = _load_extra_chars(args.charset_file)

    for script in scripts:
        charset = _build_charset(script, extra_chars, extra_file_chars, limit_overrides)
        logging.info("Harvesting %s (%d chars, %d fonts)...", script.label, len(charset), len(font_paths))
        dataset, stats = harvest_script(font_paths, charset, script, args.min_segments)
        output_path = script.output_path(args.output_dir)
        if not dataset:
            logging.warning("No glyphs harvested for %s. Check font coverage.", script.name)
            continue
        _write_dataset(dataset, output_path)
        logging.info("Wrote %d glyph entries to %s", len(dataset), output_path)
        if script.name == "cyrillic" and not args.no_legacy_mirror:
            _mirror_to_legacy(output_path, args.legacy_output)
            logging.info("Mirrored Cyrillic dataset to legacy path %s", args.legacy_output)

        top_chars = stats.char_counts.most_common(5)
        top_langs = stats.language_hits.most_common(5)
        logging.info(
            "[Summary][%s] Fonts:%d Glyphs:%d Distinct chars:%d",
            script.name,
            stats.fonts_with_script,
            stats.glyph_entries,
            len(stats.char_counts),
        )
        if top_chars:
            logging.info("  Top chars: %s", ", ".join(f"{ch}:{cnt}" for ch, cnt in top_chars))
        if top_langs:
            logging.info("  Top languages: %s", ", ".join(f"{lang}:{cnt}" for lang, cnt in top_langs))


if __name__ == "__main__":
    main()
