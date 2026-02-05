#!/usr/bin/env python3
"""
Generate router precision/recall analytics from router event logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_events(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _accumulate(events: List[Dict[str, Any]]) -> Tuple[Dict[str, int], List[float], Dict[str, int]]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}
    logits: List[float] = []
    datasets: Dict[str, int] = {}

    for event in events:
        label = event.get("label")
        use_specialist = event.get("router_use_specialist")
        if label not in (0, 1):
            continue
        if use_specialist is None:
            continue
        pred = 1 if bool(use_specialist) else 0
        counts["total"] += 1
        dataset = str(event.get("dataset") or "unknown")
        datasets[dataset] = datasets.get(dataset, 0) + 1
        if isinstance(event.get("router_logit"), (int, float)):
            logits.append(float(event["router_logit"]))
        if label == 1 and pred == 1:
            counts["tp"] += 1
        elif label == 0 and pred == 1:
            counts["fp"] += 1
        elif label == 0 and pred == 0:
            counts["tn"] += 1
        elif label == 1 and pred == 0:
            counts["fn"] += 1
    return counts, logits, datasets


def _metrics(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    total = counts["total"]
    return {
        "precision": tp / (tp + fp) if (tp + fp) > 0 else None,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else None,
        "accuracy": (tp + tn) / total if total > 0 else None,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else None,
    }


def _build_report(events: List[Dict[str, Any]]) -> str:
    counts, logits, datasets = _accumulate(events)
    overall = _metrics(counts)
    avg_logit = sum(logits) / len(logits) if logits else None

    lines = [
        "# Phase 3.1 Router Performance",
        "",
        f"Total labeled events: {counts['total']}",
        "",
        "## Confusion Matrix",
        "",
        "| | Predicted Positive | Predicted Negative |",
        "| --- | --- | --- |",
        f"| Actual Positive | {counts['tp']} | {counts['fn']} |",
        f"| Actual Negative | {counts['fp']} | {counts['tn']} |",
        "",
        "## Metrics",
        "",
        f"- Precision: {_fmt_pct(overall['precision'])}",
        f"- Recall: {_fmt_pct(overall['recall'])}",
        f"- Accuracy: {_fmt_pct(overall['accuracy'])}",
        f"- False Positive Rate: {_fmt_pct(overall['fpr'])}",
    ]

    if avg_logit is not None:
        lines.append(f"- Avg Router Logit: {avg_logit:.4f}")

    if datasets:
        lines.append("")
        lines.append("## Dataset Counts")
        for name, count in sorted(datasets.items()):
            lines.append(f"- {name}: {count}")

    # Per-dataset metrics
    per_dataset = {}
    for event in events:
        dataset = str(event.get("dataset") or "unknown")
        per_dataset.setdefault(dataset, []).append(event)

    if per_dataset:
        lines.append("")
        lines.append("## Per-Dataset Metrics")
        for dataset, items in sorted(per_dataset.items()):
            d_counts, d_logits, _ = _accumulate(items)
            d_metrics = _metrics(d_counts)
            d_avg_logit = sum(d_logits) / len(d_logits) if d_logits else None
            lines.append("")
            lines.append(f"### {dataset}")
            lines.append(f"- Precision: {_fmt_pct(d_metrics['precision'])}")
            lines.append(f"- Recall: {_fmt_pct(d_metrics['recall'])}")
            lines.append(f"- Accuracy: {_fmt_pct(d_metrics['accuracy'])}")
            lines.append(f"- False Positive Rate: {_fmt_pct(d_metrics['fpr'])}")
            if d_avg_logit is not None:
                lines.append(f"- Avg Router Logit: {d_avg_logit:.4f}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Router analytics report.")
    parser.add_argument("--input", required=True, help="Path to router events JSONL.")
    parser.add_argument("--output", help="Optional output Markdown path.")
    args = parser.parse_args()

    events = _load_events(args.input)
    report = _build_report(events)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"[RouterReport] Wrote {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
