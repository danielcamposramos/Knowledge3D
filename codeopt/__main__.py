from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from .py_process import write_python, PyStats
from .js_process import write_js, JSStats


def iter_files(paths: Iterable[Path], exts: set[str]) -> Iterable[Path]:
    for p in paths:
        if p.is_file():
            if p.suffix in exts:
                yield p
        elif p.is_dir():
            for root, _, files in os.walk(p):
                for name in files:
                    fp = Path(root) / name
                    if fp.suffix in exts:
                        yield fp


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MR sources from HR code")
    ap.add_argument("--in", dest="inputs", nargs="*", default=["."], help="Input files or directories")
    ap.add_argument("--out", dest="out", default="../Knowledge3D.local/mr", help="Output root directory")
    ap.add_argument("--lang", choices=["auto", "py", "js", "ts"], default="auto", help="Language filter")
    ap.add_argument("--stats", action="store_true", help="Show summary stats")
    args = ap.parse_args()

    out_root = Path(args.out)
    inputs = [Path(x) for x in args.inputs]

    if args.lang == "py":
        exts = {".py"}
    elif args.lang in ("js", "ts"):
        exts = {".js", ".ts"}
    else:
        exts = {".py", ".js", ".ts"}

    pystats = PyStats()
    jsstats = JSStats()
    for src in iter_files(inputs, exts):
        rel = None
        try:
            rel = src.resolve().relative_to(Path.cwd().resolve())
        except Exception:
            rel = src.name
        out_path = out_root / rel
        if src.suffix == ".py":
            write_python(src, out_path, pystats)
        else:
            write_js(src, out_path, jsstats)

    if args.stats:
        def pct(a,b):
            return (1 - (b/(a or 1))) * 100
        print("Python:", pystats.files, "files,", pystats.bytes_in, "→", pystats.bytes_out, f"({pct(pystats.bytes_in, pystats.bytes_out):.1f}% saved)")
        print("JS/TS:", jsstats.files, "files,", jsstats.bytes_in, "→", jsstats.bytes_out, f"({pct(jsstats.bytes_in, jsstats.bytes_out):.1f}% saved)")


if __name__ == "__main__":  # pragma: no cover
    main()

