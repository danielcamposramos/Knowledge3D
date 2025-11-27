#!/usr/bin/env python3
"""
Capture ARC training metrics for documentation.

Usage:
  PYTHONPATH=. python scripts/capture_arc_metrics.py \
    --log /tmp/arc_run_001.log \
    --output metrics/arc_run_001_metrics.json
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime

def parse_log(log_path: Path) -> dict:
    """Extract metrics from training log."""
    with open(log_path, 'r', encoding='latin-1') as f:
        log_content = f.read()

    # Extract epoch stats
    epoch_pattern = r"Epoch \d+ \(cycle \d+\): (\d+)/(\d+) correct \(([\d.]+)%\)"
    epochs = re.findall(epoch_pattern, log_content)

    accuracy_progression = []
    for correct, total, pct in epochs:
        accuracy_progression.append({
            "correct": int(correct),
            "total": int(total),
            "accuracy": float(pct) / 100.0
        })

    # Extract final state
    # Use the final occurrence in the log (state can update mid-run)
    library_matches = re.findall(r"Shadow entries: (\d+)", log_content)
    shapes_matches = re.findall(r"Drawing shapes: (\d+)", log_content)
    rules_matches = re.findall(r"Grammar rules: (\d+)", log_content)
    patterns_matches = re.findall(r"Pattern types: (\d+)", log_content)

    # Extract curriculum distribution
    curriculum_match = re.search(
        r"\[CURRICULUM\] Total mixed: (\d+) tasks \(easy=(\d+), mid=(\d+), hard=(\d+)\)",
        log_content
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "log_file": str(log_path),
        "accuracy_progression": accuracy_progression,
        "final_state": {
            "library_programs": int(library_matches[-1]) if library_matches else 0,
            "drawing_shapes": int(shapes_matches[-1]) if shapes_matches else 0,
            "grammar_rules": int(rules_matches[-1]) if rules_matches else 0,
            "pattern_types": int(patterns_matches[-1]) if patterns_matches else 0,
        },
        "curriculum": {
            "total": int(curriculum_match.group(1)) if curriculum_match else 0,
            "easy": int(curriculum_match.group(2)) if curriculum_match else 0,
            "mid": int(curriculum_match.group(3)) if curriculum_match else 0,
            "hard": int(curriculum_match.group(4)) if curriculum_match else 0,
        } if curriculum_match else None,
        "peak_accuracy": max(ep["accuracy"] for ep in accuracy_progression) if accuracy_progression else 0.0,
        "final_accuracy": accuracy_progression[-1]["accuracy"] if accuracy_progression else 0.0,
    }

def load_checkpoint_metrics(checkpoint_dir: Path) -> dict:
    """Load metrics from checkpoint files."""
    shadow_path = checkpoint_dir / "shadow_copy.json"
    dedup_path = checkpoint_dir / "deduplication_index.json"
    semantic_path = checkpoint_dir / "semantic_context.json"

    metrics = {
        "library": {},
        "deduplication": {},
        "semantic": {}
    }

    # Shadow copy
    if shadow_path.exists():
        with open(shadow_path, 'r') as f:
            shadow = json.load(f)
        metrics["library"] = {
            "programs": len(shadow.get("library", [])),
            "total_references": sum(e.get("reference_count", 0) for e in shadow.get("library", [])),
            "avg_quality": sum(e.get("quality_score", 0) for e in shadow.get("library", [])) / max(1, len(shadow.get("library", []))),
        }

    # Deduplication
    if dedup_path.exists():
        with open(dedup_path, 'r') as f:
            dedup = json.load(f)
        # New format (canonical_programs dict + totals)
        if "canonical_programs" in dedup:
            unique_programs = dedup.get("total_unique", len(dedup.get("canonical_programs", {})))
            total_refs = dedup.get("total_references", 0)
            # Fallback: derive references from usage_metadata counts if totals are missing
            if total_refs == 0 and isinstance(dedup.get("usage_metadata"), dict):
                total_refs = sum(len(v) for v in dedup["usage_metadata"].values() if isinstance(v, list))
        else:
            # Legacy format (list of programs with reference_count)
            unique_programs = len(dedup.get("programs", []))
            total_refs = sum(e.get("reference_count", 0) for e in dedup.get("programs", []))

        metrics["deduplication"] = {
            "unique_programs": unique_programs,
            "total_references": total_refs,
            "dedup_efficiency": (1.0 - unique_programs / max(1, total_refs)) if total_refs > 0 else 0.0,
        }

    # Semantic context
    if semantic_path.exists():
        with open(semantic_path, 'r') as f:
            semantic = json.load(f)
        vocab = semantic.get("vocabulary", {})
        metrics["semantic"] = {
            "contexts": len(semantic.get("contexts", [])),
            "vocabulary_words": len(vocab.get("words", [])),
            "vocabulary_refs": vocab.get("total_refs", 0),
            "storage_savings": vocab.get("storage_savings", 0.0),
        }

    return metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Training log file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--checkpoints", type=Path, default=Path("/K3D/Knowledge3D.local/checkpoints/arc_agi"))
    args = parser.parse_args()

    # Parse log
    log_metrics = parse_log(args.log)

    # Load checkpoint metrics
    checkpoint_metrics = load_checkpoint_metrics(args.checkpoints)

    # Combine
    combined = {
        **log_metrics,
        "checkpoints": checkpoint_metrics,
    }

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"Metrics saved to {args.output}")
    print(f"  Peak accuracy: {combined['peak_accuracy']:.2%}")
    print(f"  Final accuracy: {combined['final_accuracy']:.2%}")
    print(f"  Library programs: {combined['checkpoints']['library']['programs']}")
    print(f"  Dedup efficiency: {combined['checkpoints']['deduplication']['dedup_efficiency']:.1%}")

if __name__ == "__main__":
    main()
