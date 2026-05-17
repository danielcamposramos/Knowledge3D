#!/usr/bin/env python3
"""Prepare math competition benchmark files from AMC-AIME raw datasets.

Produces benchmark-ready JSON files expected by benchmarks/math_competitions.py:
- amc_problems.json
- aime_problems.json
- imo_problems.json

Input defaults to /K3D/K3D_llama_cpp/datasets/AMC-AIME/data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _pick_text(obj: dict[str, Any]) -> str:
    for key in ("problem", "Problem", "question", "Question", "prompt"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _pick_answer(obj: dict[str, Any]) -> str:
    for key in ("answer", "Answer", "final_answer", "FinalAnswer", "finalAnswer"):
        val = obj.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s:
            return s
    return ""


def _normalize(rows: list[dict[str, Any]], competition: str, id_prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        text = _pick_text(row)
        answer = _pick_answer(row)
        if not text or not answer:
            continue
        raw_id = row.get("id", row.get("ID", idx))
        out.append(
            {
                "id": f"{id_prefix}_{raw_id}",
                "competition": competition,
                "problem_text": text,
                "answer": answer,
                "source": "AMC-AIME",
            }
        )
    return out


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare benchmark math competition files from AMC-AIME JSONL")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/K3D/K3D_llama_cpp/datasets/AMC-AIME/data"),
        help="Input directory containing aimo_train.jsonl, aimo_test.jsonl, aime_2024.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/math_competitions"),
        help="Output directory for amc_problems.json, aime_problems.json, imo_problems.json",
    )
    parser.add_argument("--max-amc", type=int, default=0, help="Optional cap for AMC records (0 = all)")
    parser.add_argument("--max-aime", type=int, default=0, help="Optional cap for AIME records (0 = all)")
    parser.add_argument(
        "--imo-source",
        type=Path,
        default=None,
        help="Optional JSON/JSONL source for IMO problems; if omitted writes empty imo_problems.json",
    )
    args = parser.parse_args()

    train = _load_jsonl(args.input_dir / "aimo_train.jsonl")
    test = _load_jsonl(args.input_dir / "aimo_test.jsonl")
    aime_2024 = _load_jsonl(args.input_dir / "aime_2024.jsonl")

    amc_rows = _normalize(train + test, competition="AMC", id_prefix="amc")
    aime_rows = _normalize(aime_2024, competition="AIME", id_prefix="aime")

    if args.max_amc > 0:
        amc_rows = amc_rows[: args.max_amc]
    if args.max_aime > 0:
        aime_rows = aime_rows[: args.max_aime]

    imo_rows: list[dict[str, Any]] = []
    if args.imo_source and args.imo_source.exists():
        if args.imo_source.suffix.lower() == ".jsonl":
            imo_rows = _normalize(_load_jsonl(args.imo_source), competition="IMO", id_prefix="imo")
        else:
            try:
                payload = json.loads(args.imo_source.read_text(encoding="utf-8"))
                rows = payload if isinstance(payload, list) else payload.get("problems", [])
                if isinstance(rows, list):
                    imo_rows = _normalize([r for r in rows if isinstance(r, dict)], competition="IMO", id_prefix="imo")
            except Exception:
                imo_rows = []

    _write_json(args.output_dir / "amc_problems.json", amc_rows)
    _write_json(args.output_dir / "aime_problems.json", aime_rows)
    _write_json(args.output_dir / "imo_problems.json", imo_rows)

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "counts": {"amc": len(amc_rows), "aime": len(aime_rows), "imo": len(imo_rows)},
        "sources": {
            "aimo_train": str(args.input_dir / "aimo_train.jsonl"),
            "aimo_test": str(args.input_dir / "aimo_test.jsonl"),
            "aime_2024": str(args.input_dir / "aime_2024.jsonl"),
            "imo_source": str(args.imo_source) if args.imo_source else None,
        },
    }
    _write_json(args.output_dir / "prepare_math_competitions_report.json", [report])

    print(f"[math-prepare] AMC={len(amc_rows)} AIME={len(aime_rows)} IMO={len(imo_rows)}")
    print(f"[math-prepare] output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
