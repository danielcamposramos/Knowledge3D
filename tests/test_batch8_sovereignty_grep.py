from __future__ import annotations

from pathlib import Path
import re


def test_batch8_files_do_not_import_forbidden_modules() -> None:
    files = [
        Path("knowledge3d/ingestion/hs_math_parser.py"),
        Path("knowledge3d/ingestion/math_canonical_id.py"),
        Path("knowledge3d/ingestion/math_semantic_aliases.py"),
        Path("knowledge3d/ingestion/math_symlink_resolver.py"),
        Path("knowledge3d/ingestion/rpn_sketch_lexer.py"),
        Path("scripts/ingest_phase7a1_seed_audit.py"),
        Path("scripts/ingest_hs_math_cluster1.py"),
    ]
    forbidden = re.compile(r"\b(import numpy|import cupy|import scipy|import sympy)\b")
    forbidden_runtime = re.compile(r"knowledge3d\.cranium\.(cuda|kernels)")
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not forbidden.search(text)
        assert not forbidden_runtime.search(text)
