from __future__ import annotations

import ast
import io
import os
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PyStats:
    files: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


def _find_docstring_ranges(src: str):
    """Return list of (start_line, end_line, col) for docstrings and a set of
    (insert_pass_line, indent) where a pass needs to be inserted (docstring-only bodies).
    """
    module = ast.parse(src)
    ranges = []
    pass_points = []

    def handle(node: ast.AST):
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
            ranges.append((first.lineno, first.end_lineno or first.lineno, first.col_offset))
            if len(body) == 1 and not isinstance(node, ast.Module):
                # function/class with only a docstring: we need a pass
                indent = first.col_offset
                pass_points.append((first.lineno, indent))

    for n in ast.walk(module):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            handle(n)
    return ranges, pass_points


def _remove_comments_tokens(src: str) -> str:
    out = io.StringIO()
    prev_end = (1, 0)
    tokgen = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok in tokgen:
        ttype, tstr, start, end, line = tok
        if ttype == tokenize.COMMENT:
            continue
        if ttype == tokenize.NL and start[1] == 0:
            # allow empty NL
            out.write("\n")
            prev_end = end
            continue
        # write whitespace between prev_end and start to preserve layout
        (sline, scol), (eline, ecol) = start, end
        if prev_end[0] < sline:
            out.write("\n" * (sline - prev_end[0]))
            out.write(" " * scol)
        else:
            out.write(" " * max(0, scol - prev_end[1]))
        out.write(tstr)
        prev_end = end
    return out.getvalue()


def process_python(src: str) -> str:
    # 1) Remove comments
    no_comments = _remove_comments_tokens(src)
    lines = no_comments.splitlines()

    # 2) Remove/replace docstrings
    ranges, pass_points = _find_docstring_ranges(no_comments)
    to_replace = set()
    for (start, end, col) in ranges:
        for ln in range(start, end + 1):
            to_replace.add(ln - 1)

    for idx in sorted(to_replace):
        if 0 <= idx < len(lines):
            lines[idx] = ""

    for (line_no, indent) in pass_points:
        idx = line_no - 1
        if 0 <= idx < len(lines):
            lines[idx] = (" " * indent) + "pass"

    # 3) Compress blank lines and trailing spaces
    compact: list[str] = []
    blank = False
    for ln in lines:
        s = ln.rstrip()
        if s == "":
            if not blank:
                compact.append("")
                blank = True
        else:
            compact.append(s)
            blank = False
    return "\n".join(compact) + "\n"


def write_python(in_path: Path, out_path: Path, stats: PyStats) -> None:
    src = in_path.read_text(encoding="utf-8")
    dst = process_python(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dst, encoding="utf-8")
    stats.files += 1
    stats.bytes_in += len(src.encode("utf-8"))
    stats.bytes_out += len(dst.encode("utf-8"))


