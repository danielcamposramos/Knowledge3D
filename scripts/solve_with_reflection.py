#!/usr/bin/env python3
"""
Reflective inference for navigation specialist with confidence head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from knowledge3d.training.math_benchmarks.reflective_inference import ReflectiveSolver


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_problem(entry: Dict[str, Any]) -> str:
    for key in ("problem_text", "problem", "question", "prompt"):
        value = entry.get(key)
        if value:
            return str(value)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflective inference with confidence head.")
    parser.add_argument(
        "--checkpoint",
        default="/K3D/Knowledge3D.local/checkpoints/v7_sovereign",
        help="Converted checkpoint directory with confidence head.",
    )
    parser.add_argument("--problem", type=str, default=None, help="Single problem text.")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL with problems.")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path.")
    parser.add_argument("--max-steps", type=int, default=64, help="Max decoding steps.")
    parser.add_argument("--confident-threshold", type=float, default=0.9, help="Confident threshold.")
    parser.add_argument("--verify-threshold", type=float, default=0.5, help="Verify threshold.")
    parser.add_argument("--quiet", action="store_true", help="Silence solver logs.")
    args = parser.parse_args()

    solver = ReflectiveSolver(
        args.checkpoint,
        max_steps=int(args.max_steps),
        confident_threshold=float(args.confident_threshold),
        verify_threshold=float(args.verify_threshold),
        quiet=bool(args.quiet),
    )

    if args.problem:
        result, trace, meta = solver.solve(str(args.problem))
        print(json.dumps({"result": result, "trace": trace, "reflection": meta}, ensure_ascii=True))
        return

    if not args.input:
        raise SystemExit("Provide --problem or --input.")

    output = Path(args.output) if args.output else None
    handle = output.open("w", encoding="utf-8") if output else None

    for entry in _iter_jsonl(args.input):
        problem_text = _extract_problem(entry)
        if not problem_text:
            continue
        result, trace, meta = solver.solve(problem_text)
        payload = dict(entry)
        payload["result"] = result
        payload["trace"] = trace
        payload["reflection"] = meta
        line = json.dumps(payload, ensure_ascii=True)
        if handle:
            handle.write(line + "\n")
        else:
            print(line)

    if handle:
        handle.close()
        print(f"[Saved] {output}")


if __name__ == "__main__":
    main()
