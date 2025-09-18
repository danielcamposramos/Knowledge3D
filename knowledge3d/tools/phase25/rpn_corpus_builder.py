"""Build a Reverse Polish Notation corpus from curated advanced math sources.

The builder scans JSON documents within the Architect's curated math library and
extracts infix expressions, converts them to RPN, and writes structured
entries. Expressions are filtered to avoid malformed content; each record
includes provenance for auditing.

Usage (inside k3d-cranium env):

    PYTHONPATH=. python -m knowledge3d.tools.phase25.rpn_corpus_builder \
        --source "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/JSON" \
        --output viewer/public/galaxy/working/rpn_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Supported functions/operators for conversion
FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "ln",
    "sqrt",
    "abs",
}

OPERATORS: Dict[str, Tuple[int, str]] = {
    "+": (2, "left"),
    "-": (2, "left"),
    "*": (3, "left"),
    "/": (3, "left"),
    "^": (4, "right"),
}

TOKEN_PATTERN = re.compile(
    r"(?P<number>\b\d+(?:\.\d+)?\b)"  # numbers
    r"|(?P<identifier>[A-Za-zφΦπΠ]+(?:_\d+)?)"  # identifiers/variables
    r"|(?P<operator>[+\-*/^])"  # operators
    r"|(?P<bracket>[()])"  # parentheses
    r"|(?P<comma>,)"  # comma
)

EXPRESSION_CANDIDATE = re.compile(r"[A-Za-z0-9πφΦΠ]+[\s]*[+\-*/^][^=]*")
EQUALS_SPLIT = re.compile(r"=|≈|≃|~=|:=")


@dataclass
class CorpusEntry:
    source_file: str
    page: Optional[str]
    infix: str
    rpn_tokens: List[str]
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_file": self.source_file,
            "page": self.page,
            "infix": self.infix,
            "rpn": " ".join(self.rpn_tokens),
            "tokens": self.rpn_tokens,
            "notes": self.notes,
        }


class RPNCorpusBuilder:
    def __init__(self, source_dir: Path, output_path: Path) -> None:
        self.source_dir = source_dir
        self.output_path = output_path

    # -------------------------- Parsing helpers --------------------------
    def _tokenize(self, expr: str) -> List[str]:
        tokens: List[str] = []
        idx = 0
        while idx < len(expr):
            match = TOKEN_PATTERN.match(expr, idx)
            if not match:
                ch = expr[idx]
                if ch.isspace():
                    idx += 1
                    continue
                # unsupported symbol -> abort expression
                raise ValueError(f"Unsupported token '{ch}' in expression '{expr}'")
            kind = match.lastgroup
            value = match.group()
            if kind == "identifier":
                value_lower = value.lower()
                if value_lower in FUNCTIONS:
                    tokens.append(value_lower)
                elif value_lower in {"pi", "π"}:
                    tokens.append("π")
                elif value_lower in {"phi", "φ"}:
                    tokens.append("φ")
                else:
                    tokens.append(value)
            elif kind == "number":
                tokens.append(value)
            elif kind == "operator":
                # Distinguish unary minus when preceding token is operator or '('
                if value == "-":
                    prev = tokens[-1] if tokens else None
                    if prev in OPERATORS or prev in ("(", ",") or prev in FUNCTIONS or prev is None:
                        tokens.append("unary_minus")
                    else:
                        tokens.append(value)
                else:
                    tokens.append(value)
            else:
                tokens.append(value)
            idx = match.end()
        return tokens

    def _shunting_yard(self, tokens: Sequence[str]) -> List[str]:
        output: List[str] = []
        stack: List[str] = []
        for token in tokens:
            if token in OPERATORS:
                prec, assoc = OPERATORS[token]
                while stack:
                    top = stack[-1]
                    if top in OPERATORS:
                        top_prec, _ = OPERATORS[top]
                        if (assoc == "left" and prec <= top_prec) or (assoc == "right" and prec < top_prec):
                            output.append(stack.pop())
                            continue
                    break
                stack.append(token)
            elif token == "unary_minus":
                stack.append(token)
            elif token == "(":
                stack.append(token)
            elif token == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Mismatched parentheses")
                stack.pop()
                if stack and stack[-1] in FUNCTIONS:
                    output.append(stack.pop())
            elif token == ",":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Misplaced comma")
            elif token in FUNCTIONS:
                stack.append(token)
            else:
                output.append(token)
        while stack:
            top = stack.pop()
            if top in ("(", ")"):
                raise ValueError("Mismatched parentheses")
            output.append(top)
        # Replace unary-minus symbol with explicit NEG op in RPN
        result: List[str] = []
        for tok in output:
            if tok == "unary_minus":
                result.append("neg")
            else:
                result.append(tok)
        return result

    def _normalize_expression(self, expr: str) -> Optional[str]:
        cleaned = expr.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace("−", "-").replace("·", "*").replace("÷", "/")
        cleaned = cleaned.replace(" ", " ")  # non-breaking space
        cleaned = cleaned.replace("…", "")
        # collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _split_equation(self, expr: str) -> List[str]:
        if "=" not in expr:
            return [expr]
        parts = [p.strip() for p in EQUALS_SPLIT.split(expr) if p.strip()]
        return parts if len(parts) <= 2 else parts[:2]

    # -------------------------- Extraction --------------------------
    def _extract_from_text(self, text: str, source: str, page: Optional[str]) -> Iterable[CorpusEntry]:
        candidates = [line.strip() for line in text.splitlines() if line and any(op in line for op in "+-*/^")]
        for cand in candidates:
            if not EXPRESSION_CANDIDATE.search(cand):
                continue
            cleaned = self._normalize_expression(cand)
            if not cleaned:
                continue
            split_parts = self._split_equation(cleaned)
            for part in split_parts:
                try:
                    tokens = self._tokenize(part)
                    rpn = self._shunting_yard(tokens)
                    if len(rpn) < 2:
                        continue
                    yield CorpusEntry(source_file=source, page=page, infix=part, rpn_tokens=rpn)
                except ValueError:
                    continue

    # -------------------------- IO --------------------------
    def build(self) -> List[CorpusEntry]:
        entries: List[CorpusEntry] = []
        for json_path in self.source_dir.glob("*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue

            def consume_text(obj: object, page: Optional[str] = None) -> Iterable[Tuple[str, Optional[str]]]:
                if obj is None:
                    return
                if isinstance(obj, str):
                    yield obj, page
                elif isinstance(obj, dict):
                    page_here = page
                    if "page" in obj and isinstance(obj["page"], (str, int)):
                        page_here = str(obj["page"])
                    if "content" in obj:
                        yield from consume_text(obj["content"], page_here)
                    else:
                        for value in obj.values():
                            yield from consume_text(value, page_here)
                elif isinstance(obj, list):
                    for item in obj:
                        yield from consume_text(item, page)

            for text_block, page in consume_text(data):
                for entry in self._extract_from_text(text_block, json_path.name, page):
                    entries.append(entry)
        return entries

    def write(self, entries: Sequence[CorpusEntry]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fh:
            for item in entries:
                fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def build_corpus(source: str, output: str) -> None:
    builder = RPNCorpusBuilder(Path(source), Path(output))
    entries = builder.build()
    unique = {(e.infix, e.source_file): e for e in entries}
    builder.write(list(unique.values()))
    print(f"📚 Built RPN corpus with {len(unique)} unique expressions → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RPN corpus from advanced maths library")
    parser.add_argument("--source", type=str, required=True, help="Source directory with JSON math texts")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    args = parser.parse_args()
    build_corpus(args.source, args.output)


if __name__ == "__main__":
    main()
