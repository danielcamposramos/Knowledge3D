"""Non-blocking Batch 8 RPN sketch lexer."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
from typing import Iterable


DOCUMENTARY_OPCODES = frozenset({"STORE", "RECALL", "GALAXY_LOOKUP", "OP_BRANCH"})
_RPN_MODULE = importlib.import_module("knowledge3d.cranium.ptx_runtime.rpn_opcodes")
REAL_OPCODES = frozenset(name[3:] for name in getattr(_RPN_MODULE, "__all__", ()) if name.startswith("OP_"))
REPORT_PATH = Path("/K3D/Knowledge3D.local/reports/batch8_rpn_sketch_coverage.json")


@dataclass(frozen=True)
class RpnSketchToken:
    opcode: str
    args: tuple[str, ...]
    raw: str


def lex_rpn_sketch(sketch: str) -> list[RpnSketchToken]:
    tokens: list[RpnSketchToken] = []
    for raw in re.findall(r"\[([^\]]+)\]", str(sketch or "")):
        parts = tuple(part for part in raw.strip().split() if part)
        if not parts:
            continue
        tokens.append(RpnSketchToken(opcode=parts[0], args=parts[1:], raw=raw.strip()))
    return tokens


def classify_opcode(opcode: str) -> str:
    key = str(opcode or "").strip().upper()
    if key in REAL_OPCODES:
        return "real"
    if key in DOCUMENTARY_OPCODES:
        return "documentary"
    return "unknown"


def write_coverage_report(sketches: Iterable[str], output_path: Path = REPORT_PATH) -> dict[str, object]:
    histogram: dict[str, int] = {}
    real_hits = 0
    documentary_hits = 0
    unknown: dict[str, int] = {}
    rows = 0
    for sketch in sketches:
        rows += 1
        for token in lex_rpn_sketch(sketch):
            histogram[token.opcode] = histogram.get(token.opcode, 0) + 1
            bucket = classify_opcode(token.opcode)
            if bucket == "real":
                real_hits += 1
            elif bucket == "documentary":
                documentary_hits += 1
            else:
                unknown[token.opcode] = unknown.get(token.opcode, 0) + 1
    payload = {
        "rows_scanned": rows,
        "opcode_histogram": histogram,
        "real_opcode_hits": real_hits,
        "documentary_hits": documentary_hits,
        "unknown_opcodes": unknown,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


__all__ = ["DOCUMENTARY_OPCODES", "REAL_OPCODES", "RpnSketchToken", "classify_opcode", "lex_rpn_sketch", "write_coverage_report"]
