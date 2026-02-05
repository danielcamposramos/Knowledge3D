#!/usr/bin/env python3
"""
Seed RouterGalaxy with initial labeled training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge3d.training.math_benchmarks.router_galaxy import RouterGalaxy


def _load_seed(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Router Galaxy from labeled JSONL.")
    parser.add_argument("--input", required=True, help="Input labeled JSONL (text/label).")
    parser.add_argument("--output", required=True, help="Router Galaxy JSONL to write/append.")
    parser.add_argument("--append", action="store_true", help="Append to existing output.")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--positive-logit", type=float, default=2.0, help="Logit for positive seeds.")
    parser.add_argument("--negative-logit", type=float, default=-2.0, help="Logit for negative seeds.")
    args = parser.parse_args()

    rows = _load_seed(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    galaxy = RouterGalaxy(embedding_dim=int(args.embedding_dim))
    seeded = 0
    for row in rows:
        text = str(row.get("text", "")).strip()
        label = row.get("label")
        if not text or label not in (0, 1):
            continue
        logit = float(args.positive_logit if label == 1 else args.negative_logit)
        use_specialist = bool(label == 1)
        dataset = str(row.get("source") or "seed")
        galaxy.add_event(
            problem_text=text,
            router_logit=logit,
            router_use_specialist=use_specialist,
            solver="seed",
            correct=True,
            dataset=dataset,
            label=int(label),
            metadata={"seed_source": args.input},
        )
        seeded += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and output.exists() else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for entry in galaxy.entries:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")

    print(f"[RouterGalaxy] Seeded {seeded} entries into {output}")


if __name__ == "__main__":
    main()
