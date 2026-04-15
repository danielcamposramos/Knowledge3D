"""Math Symbol Galaxy builder for Phase 7.A.1 Slice 3.

Ingestion-path only. It builds meaning-centric math symbol stars from Unicode
math blocks and links them to existing character stars without introducing new
RPN opcodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping
import unicodedata

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id, canonical_slug
from knowledge3d.ingestion.symlink_helpers import link
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


LOCAL_MATH_SYMBOL_STAR_PATH = Path("/K3D/Knowledge3D.local/assets/math/MATH_SYMBOL_STARS.jsonl")
MATH_CLASS_EX_PATH = Path("/K3D/Knowledge3D.local/assets/math/MathClassEx-15.txt")

MATH_BLOCKS = (
    (0x2200, 0x22FF),
    (0x27C0, 0x27EF),
    (0x2980, 0x29FF),
    (0x2A00, 0x2AFF),
    (0x1D400, 0x1D7FF),
    (0x2150, 0x218F),
)

PINNED_CODEPOINTS = {
    0x00B1,
    0x00D7,
    0x00F7,
    0x221A,
    0x221E,
    0x2260,
    0x2264,
    0x2265,
    0x2208,
    0x2209,
    0x2282,
    0x2286,
    0x222B,
    0x2211,
    0x220F,
    0x221B,
    0x221C,
    0x002B,
    0x002D,
    0x003D,
    0x003C,
    0x003E,
    0x005E,
    0x03C0,
}

LATEX_COMMANDS = {
    0x002B: ["+"],
    0x002D: ["-"],
    0x003D: ["="],
    0x003C: ["<"],
    0x003E: [">"],
    0x005E: ["^"],
    0x00B1: ["\\pm"],
    0x00D7: ["\\times"],
    0x00F7: ["\\div"],
    0x03C0: ["\\pi"],
    0x2202: ["\\partial"],
    0x2207: ["\\nabla"],
    0x2208: ["\\in"],
    0x2209: ["\\notin"],
    0x220F: ["\\prod"],
    0x2211: ["\\sum"],
    0x2212: ["-"],
    0x221A: ["\\sqrt"],
    0x221B: ["\\sqrt[3]"],
    0x221C: ["\\sqrt[4]"],
    0x221E: ["\\infty"],
    0x222B: ["\\int"],
    0x2260: ["\\ne"],
    0x2264: ["\\le"],
    0x2265: ["\\ge"],
    0x2282: ["\\subset"],
    0x2286: ["\\subseteq"],
}

MATH_CLASS_BY_CODEPOINT = {
    0x002B: "B",
    0x002D: "B",
    0x005E: "B",
    0x00D7: "B",
    0x00F7: "B",
    0x2212: "B",
    0x2211: "O",
    0x220F: "O",
    0x222B: "O",
    0x003D: "R",
    0x003C: "R",
    0x003E: "R",
    0x2260: "R",
    0x2264: "R",
    0x2265: "R",
    0x2208: "R",
    0x2209: "R",
    0x2282: "R",
    0x2286: "R",
    0x03C0: "A",
}

EXECUTABLE_PROGRAMS = {
    0x002B: ("rpn_program_addition", "OPERAND_0 OPERAND_1 ADD STORE_RESULT RET"),
    0x2212: ("rpn_program_subtraction", "OPERAND_0 OPERAND_1 SUB STORE_RESULT RET"),
    0x002D: ("rpn_program_subtraction", "OPERAND_0 OPERAND_1 SUB STORE_RESULT RET"),
    0x00D7: ("rpn_program_multiplication", "OPERAND_0 OPERAND_1 MUL STORE_RESULT RET"),
    0x00F7: ("rpn_program_division", "OPERAND_0 OPERAND_1 DIV STORE_RESULT RET"),
    0x005E: ("rpn_program_power", "OPERAND_0 OPERAND_1 POW STORE_RESULT RET"),
}

MATH_CLASS_NAMES = {
    "N": "Normal",
    "A": "Alphabetic",
    "B": "Binary",
    "C": "Closing",
    "D": "Diacritic",
    "F": "Fence",
    "G": "Glyph_Part",
    "L": "Large_Operator",
    "O": "Large_Operator",
    "P": "Punctuation",
    "R": "Relation",
    "U": "Unary",
    "V": "Variable",
}


@dataclass
class MathSymbolBuild:
    stars: dict[str, dict]
    target_updates: dict[str, dict]
    skipped_links: list[dict[str, str]]
    followups: list[dict[str, str]]


def math_symbol_star_id(char: str) -> str:
    name = unicodedata.name(char, f"U+{ord(char):04X}")
    return f"math_symbol_{canonical_slug(name)}"


def iter_math_symbol_codepoints() -> Iterable[int]:
    seen: set[int] = set()
    for codepoint in sorted(PINNED_CODEPOINTS):
        seen.add(codepoint)
        yield codepoint
    for start, end in MATH_BLOCKS:
        for codepoint in range(start, end + 1):
            if codepoint in seen:
                continue
            try:
                name = unicodedata.name(chr(codepoint))
            except ValueError:
                continue
            if name:
                seen.add(codepoint)
                yield codepoint


def parse_math_class_ex(path: str | Path = MATH_CLASS_EX_PATH) -> dict[int, str]:
    path = Path(path)
    if not path.exists():
        return {}
    classes: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ";" not in line:
            continue
        code, cls, *_rest = [part.strip() for part in line.split(";")]
        if ".." in code:
            start, end = [int(part, 16) for part in code.split("..", 1)]
            for codepoint in range(start, end + 1):
                classes[codepoint] = cls
        else:
            classes[int(code, 16)] = cls
    return classes


def _math_class(codepoint: int, classes: Mapping[int, str]) -> str:
    return str(classes.get(codepoint) or MATH_CLASS_BY_CODEPOINT.get(codepoint) or "N")


def _selection_role(math_class: str, codepoint: int) -> str:
    if codepoint in EXECUTABLE_PROGRAMS or math_class in {"B", "O", "L", "U"}:
        return "operator"
    if math_class == "R":
        return "relation"
    if math_class in {"C", "F", "P"}:
        return "delimiter"
    return "operand"


def _meaning_rpn(char: str, math_class: str) -> str:
    executable = EXECUTABLE_PROGRAMS.get(ord(char))
    if executable:
        return executable[1]
    if ord(char) == 0x03C0:
        return "PI CONSTANT STORE"
    role = _selection_role(math_class, ord(char)).upper()
    return f"SYMBOL U+{ord(char):04X} {role} {MATH_CLASS_NAMES.get(math_class, math_class).upper()}"


def _surface_forms(codepoint: int, commands: list[str]) -> dict[str, dict[str, str]]:
    primary = commands[0] if commands else f"U+{codepoint:04X}"
    return {
        "tex": {"word_ref": f"word_tex_{canonical_slug(primary)}"},
        "unicode": {"word_ref": f"word_unicode_u{codepoint:04x}"},
    }


def make_math_symbol_star(codepoint: int, classes: Mapping[int, str] | None = None) -> MeaningCentricStar:
    char = chr(codepoint)
    resolved_classes = dict(classes or {})
    math_class = _math_class(codepoint, resolved_classes)
    program = EXECUTABLE_PROGRAMS.get(codepoint)
    return MeaningCentricStar(
        star_id=math_symbol_star_id(char),
        meaning_class="concept",
        domain=f"Math/{MATH_CLASS_NAMES.get(math_class, math_class)}",
        meaning_rpn=_meaning_rpn(char, math_class),
        meta_refs=[program[0]] if program else [],
        lod_class="LOD_SUMMARY",
    )


def _star_payload(star: MeaningCentricStar, codepoint: int, classes: Mapping[int, str]) -> dict:
    char = chr(codepoint)
    math_class = _math_class(codepoint, classes)
    commands = LATEX_COMMANDS.get(codepoint, [])
    program = EXECUTABLE_PROGRAMS.get(codepoint)
    payload = star.to_dict()
    payload.update(
        {
            "name": unicodedata.name(char, f"U+{codepoint:04X}"),
            "codepoint": codepoint,
            "unicode_char": char,
            "math_class": math_class,
            "latex_commands": commands,
            "surface_forms": _surface_forms(codepoint, commands),
            "selection_role": _selection_role(math_class, codepoint),
            "answer_eligible": False,
            "meaning_program": star.meaning_rpn if program else None,
            "meta_rule_addr": 0,
            "program_ref": program[0] if program else "",
            "has_executable_program": bool(program),
        }
    )
    return payload


def build_math_symbol_galaxy(
    *,
    existing_char_stars: Mapping[str, MeaningCentricStar] | None = None,
    codepoints: Iterable[int] | None = None,
    math_classes: Mapping[int, str] | None = None,
) -> MathSymbolBuild:
    targets = dict(existing_char_stars or {})
    classes = dict(math_classes or parse_math_class_ex())
    stars: dict[str, dict] = {}
    target_updates: dict[str, dict] = {}
    skipped_links: list[dict[str, str]] = []
    followups: list[dict[str, str]] = []
    points = list(codepoints) if codepoints is not None else list(iter_math_symbol_codepoints())
    for codepoint in sorted(set(points)):
        try:
            char = chr(codepoint)
            unicodedata.name(char)
        except (ValueError, TypeError):
            continue
        star = make_math_symbol_star(codepoint, classes)
        char_id = canonical_char_star_id(char)
        target = targets.get(char_id)
        if target is None:
            skipped_links.append({"source": star.star_id, "target": char_id, "reason": "target_missing"})
        else:
            link(star, target, "char_refs", "mathematical_role")
            target_updates[char_id] = target.to_dict()
        math_class = _math_class(codepoint, classes)
        if math_class in {"O", "L"} and not star.meta_refs:
            followups.append(
                {
                    "star_id": star.star_id,
                    "codepoint": f"U+{codepoint:04X}",
                    "reason": "large_operator_template_deferred",
                }
            )
        stars[star.star_id] = _star_payload(star, codepoint, classes)
    return MathSymbolBuild(stars=stars, target_updates=target_updates, skipped_links=skipped_links, followups=followups)


def write_math_symbol_build(build: MathSymbolBuild, output_path: str | Path = LOCAL_MATH_SYMBOL_STAR_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for star_id in sorted(build.stars):
            handle.write(json.dumps(build.stars[star_id], ensure_ascii=False) + "\n")
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "star_count": len(build.stars),
                "target_update_count": len(build.target_updates),
                "skipped_link_count": len(build.skipped_links),
                "followup_count": len(build.followups),
                "skipped_links": build.skipped_links[:500],
                "followups": build.followups[:500],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def canonical_math_symbol_entry(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "kind": "math_symbol",
        "key": f"U+{int(row['codepoint']):04X}",
        "star_id": str(row["star_id"]),
        "metadata": {
            "latex_commands": list(row.get("latex_commands") or []),
            "math_class": str(row.get("math_class") or ""),
            "has_executable_program": bool(row.get("has_executable_program")),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build local Math Symbol Galaxy stars from Unicode math blocks.")
    parser.add_argument("--out", default=str(LOCAL_MATH_SYMBOL_STAR_PATH))
    args = parser.parse_args(argv)
    build = build_math_symbol_galaxy()
    output_path = write_math_symbol_build(build, args.out)
    print(
        f"math_symbol_stars={len(build.stars)} "
        f"target_updates={len(build.target_updates)} "
        f"skipped_links={len(build.skipped_links)} "
        f"followups={len(build.followups)} "
        f"out={output_path}"
    )
    return 0


__all__ = [
    "EXECUTABLE_PROGRAMS",
    "LOCAL_MATH_SYMBOL_STAR_PATH",
    "MATH_BLOCKS",
    "MATH_CLASS_EX_PATH",
    "MathSymbolBuild",
    "build_math_symbol_galaxy",
    "canonical_math_symbol_entry",
    "iter_math_symbol_codepoints",
    "make_math_symbol_star",
    "math_symbol_star_id",
    "parse_math_class_ex",
    "write_math_symbol_build",
]


if __name__ == "__main__":
    raise SystemExit(main())
