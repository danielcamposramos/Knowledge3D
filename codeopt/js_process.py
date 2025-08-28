from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


BLOCK_RE = re.compile(r"/\*[\s\S]*?\*/", re.MULTILINE)


@dataclass
class JSStats:
    files: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


def _strip_line_comments(code: str) -> str:
    out_lines = []
    for line in code.splitlines():
        i = 0
        in_single = False
        in_double = False
        in_template = False
        escaped = False
        cut = len(line)
        while i < len(line):
            ch = line[i]
            if in_template:
                if ch == '`' and not escaped:
                    in_template = False
                escaped = (ch == '\\' and not escaped)
                i += 1
                continue
            if in_single:
                if ch == "'" and not escaped:
                    in_single = False
                escaped = (ch == '\\' and not escaped)
                i += 1
                continue
            if in_double:
                if ch == '"' and not escaped:
                    in_double = False
                escaped = (ch == '\\' and not escaped)
                i += 1
                continue
            if ch == '`':
                in_template = True
                i += 1
                continue
            if ch == "'":
                in_single = True
                i += 1
                continue
            if ch == '"':
                in_double = True
                i += 1
                continue
            if ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
                cut = i
                break
            i += 1
        out_lines.append(line[:cut].rstrip())
    return "\n".join(out_lines) + "\n"


def process_js(code: str) -> str:
    no_block = re.sub(BLOCK_RE, "", code)
    no_line = _strip_line_comments(no_block)
    # compress multiple blank lines
    out = []
    blank = False
    for ln in no_line.splitlines():
        s = ln.rstrip()
        if s == "":
            if not blank:
                out.append("")
                blank = True
        else:
            out.append(s)
            blank = False
    return "\n".join(out) + "\n"


def write_js(in_path: Path, out_path: Path, stats: JSStats) -> None:
    src = in_path.read_text(encoding="utf-8")
    dst = process_js(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dst, encoding="utf-8")
    stats.files += 1
    stats.bytes_in += len(src.encode("utf-8"))
    stats.bytes_out += len(dst.encode("utf-8"))

