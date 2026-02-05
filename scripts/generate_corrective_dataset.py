#!/usr/bin/env python3
"""
Generate corrective training samples from Feedback Galaxy entries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _load_feedback(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _load_registry(meta_path: Path) -> List[str]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    registry = meta.get("rule_registry") or []
    if not isinstance(registry, list):
        raise ValueError("rule_registry missing or invalid in log galaxy metadata.")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate corrective dataset from feedback galaxy.")
    parser.add_argument(
        "--feedback",
        type=str,
        default="data/feedback_galaxy_v1.jsonl",
        help="Feedback Galaxy JSONL path.",
    )
    parser.add_argument(
        "--log-meta",
        type=str,
        default="data/log_galaxy_neural_v1.json",
        help="Log Galaxy metadata JSON containing rule registry.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/corrective_tuning_v1.pt",
        help="Output corrective dataset (.pt).",
    )
    args = parser.parse_args()

    feedback_path = Path(args.feedback)
    if not feedback_path.exists():
        raise SystemExit(f"Feedback Galaxy JSONL not found: {feedback_path}")
    meta_path = Path(args.log_meta)
    if not meta_path.exists():
        raise SystemExit(f"Log Galaxy metadata not found: {meta_path}")

    registry = _load_registry(meta_path)
    registry_lookup = {str(rule).lower(): idx for idx, rule in enumerate(registry)}

    entries = _load_feedback(feedback_path)
    samples: List[Dict[str, Any]] = []
    missing_rule = 0
    missing_embed = 0

    for entry in entries:
        suggested_rule = str(entry.get("suggested_rule") or "").strip().lower()
        if not suggested_rule:
            continue
        if suggested_rule not in registry_lookup:
            missing_rule += 1
            continue
        embedding = entry.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            missing_embed += 1
            continue
        rule_id = registry_lookup[suggested_rule]
        samples.append(
            {
                "embedding": torch.tensor(embedding, dtype=torch.float32),
                "rule_id": int(rule_id),
            }
        )

    payload = {
        "samples": samples,
        "rule_registry": registry,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(
        json.dumps(
            {
                "entries": len(entries),
                "samples": len(samples),
                "missing_rule": missing_rule,
                "missing_embedding": missing_embed,
                "output": str(output_path),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
