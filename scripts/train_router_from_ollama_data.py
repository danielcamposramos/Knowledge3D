#!/usr/bin/env python3
"""
Train router specialist from Ollama-generated synthetic routing decisions.

Outputs:
1) Router specialist checkpoint (MatryoshkaTRM save_all)
2) JSON weights map for TheoremRouter (semantic tags -> rule weights)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.cranium.math_galaxy_population import populate_theorem_patterns
from knowledge3d.cranium.router_specialist import RoutingDecision, RouterSpecialistTrainer
from knowledge3d.training.math_benchmarks.calculus_grammar_rules import get_calculus_rules


_SAMPLE_TEXT_BY_PATTERN = {
    "power_rule_polynomial": "derivative of x^2 at x=3",
    "product_rule": "derivative of x^2 * x^3 at x=2",
    "quotient_rule": "derivative of x^2 / x^3 at x=2",
    "chain_rule": "derivative of (x^2)^3 at x=2",
    "sum_rule": "derivative of x^2 + x^3 at x=2",
    "constant_multiple_rule": "derivative of 3*x^2 at x=2",
    "integration_by_parts": "integral x e^x from 1 to 2",
    "fundamental_theorem_calculus": "integral x^2 from 1 to 3",
    "pythagorean_identity": "sin^2(theta) + cos^2(theta)",
}


def _load_decisions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Routing data must be a list of decisions")
    return data


def _to_routing_decisions(rows: List[Dict[str, Any]]) -> List[RoutingDecision]:
    decisions: List[RoutingDecision] = []
    for row in rows:
        input_data = row.get("input_data")
        if input_data is None:
            continue
        decisions.append(
            RoutingDecision(
                input_data=np.asarray(input_data, dtype=np.float32),
                task_description=row.get("task_description"),
                specialist_weights=row.get("specialist_weights", {}) or {},
                outcome_performance=float(row.get("outcome_performance", 0.0)),
                timestamp=str(row.get("timestamp") or ""),
            )
        )
    return decisions


def _build_tag_key(tags: Iterable[str]) -> str:
    tokens = [str(t or "").strip().lower() for t in tags if str(t or "").strip()]
    return "|".join(sorted(tokens))


def _aggregate_weight_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        tags = row.get("semantic_tags") or []
        key = _build_tag_key(tags)
        if not key:
            continue
        weights = row.get("specialist_weights") or {}
        if key not in merged:
            merged[key] = {str(k): float(v) for k, v in weights.items()}
            counts[key] = 1
            continue
        counts[key] += 1
        for rule_id, weight in weights.items():
            merged[key][str(rule_id)] = merged[key].get(str(rule_id), 0.0) + float(weight)
    for key, weights in merged.items():
        total = sum(weights.values())
        if total > 0:
            merged[key] = {rid: val / total for rid, val in weights.items()}
    return merged


def _build_pattern_map(
    *,
    artifact_dirs: Iterable[str],
    min_examples: int,
) -> Dict[str, Dict[str, Any]]:
    patterns = populate_theorem_patterns(
        artifact_dirs=list(artifact_dirs),
        min_examples=min_examples,
    )
    return {p["pattern_id"]: p for p in patterns}


def _filter_decisions(
    rows: List[Dict[str, Any]],
    pattern_map: Dict[str, Dict[str, Any]],
    grammar_rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    rejected = 0
    for row in rows:
        pattern_id = row.get("pattern_id")
        selected_rule = row.get("selected_rule")
        if not pattern_id or not selected_rule:
            rejected += 1
            continue
        pattern = pattern_map.get(pattern_id)
        if not pattern:
            rejected += 1
            continue
        expected_rule = pattern.get("grammar_rule")
        if expected_rule and not (
            selected_rule == expected_rule or selected_rule.startswith(f"{expected_rule}_")
        ):
            rejected += 1
            continue
        rule = grammar_rules.get(selected_rule)
        if rule is None:
            rejected += 1
            continue
        sample_text = _SAMPLE_TEXT_BY_PATTERN.get(pattern_id, "")
        if sample_text:
            try:
                import re as _re

                if not _re.search(rule.pattern, sample_text, _re.IGNORECASE | _re.DOTALL):
                    rejected += 1
                    continue
            except Exception:
                rejected += 1
                continue
        filtered.append(row)
    if rejected:
        print(f"[RouterTraining] Filtered out {rejected} hallucinated decisions")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Train router specialist from Ollama data.")
    parser.add_argument(
        "--input",
        type=str,
        default="/K3D/Knowledge3D.local/logs/ollama_router_training_data.json",
        help="Input JSON produced by generate_router_training_data_ollama.py",
    )
    parser.add_argument("--router-dims", type=int, default=256, help="Router input dimension.")
    parser.add_argument("--router-rank", type=int, default=16, help="Router LoRA rank.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Training learning rate.")
    parser.add_argument("--filter-threshold", type=float, default=0.8, help="Performance threshold.")
    parser.add_argument(
        "--artifact-dirs",
        nargs="+",
        default=["/K3D/Knowledge3D.local/galaxies/books_v5_clean2"],
        help="Artifact directories for theorem pattern validation.",
    )
    parser.add_argument("--min-examples", type=int, default=1, help="Min examples per theorem pattern.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output checkpoint directory (defaults to K3D_LOCAL_DIR/checkpoints/router_specialist_ollama).",
    )
    parser.add_argument(
        "--router-weights-output",
        type=str,
        default=None,
        help="Output JSON path for TheoremRouter weights.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = _load_decisions(input_path)
    grammar_rules_list = get_calculus_rules()
    grammar_rules = {r.rule_id: r for r in grammar_rules_list}
    pattern_map = _build_pattern_map(
        artifact_dirs=args.artifact_dirs,
        min_examples=int(args.min_examples),
    )
    rows = _filter_decisions(rows, pattern_map, grammar_rules)
    decisions = _to_routing_decisions(rows)
    if not decisions:
        raise SystemExit("No valid routing decisions found in input.")

    rule_ids = [r.rule_id for r in grammar_rules_list]
    if not rule_ids:
        raise SystemExit("No calculus grammar rules available.")

    config = SwarmConfig(
        base_dims=int(args.router_dims),
        min_dims=int(args.router_dims),
        enable_auto_expansion=False,
    )
    swarm = AdaptiveSwarmTRM(config)
    for rule_id in rule_ids:
        swarm.register_specialist(rule_id, required_dims=int(args.router_dims), rank=int(args.router_rank))

    trainer = RouterSpecialistTrainer(swarm)
    trainer.register_router_specialist(
        num_specialists=len(rule_ids),
        router_dims=int(args.router_dims),
        router_rank=int(args.router_rank),
    )

    stats = trainer.train_from_history(
        routing_history=decisions,
        epochs=int(args.epochs),
        filter_threshold=float(args.filter_threshold),
        learning_rate=float(args.learning_rate),
    )
    if "error" in stats:
        raise SystemExit(f"Router specialist training failed: {stats['error']}")

    if args.output_dir is None:
        local_dir = Path(os.getenv("K3D_LOCAL_DIR", "/K3D/Knowledge3D.local")) / "checkpoints" / "router_specialist_ollama"
    else:
        local_dir = Path(args.output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    swarm.base.save_all(local_dir)
    print(f"[OllamaBootstrap] Saved router specialist checkpoint to {local_dir}")

    weight_map = _aggregate_weight_map(rows)
    weights_path = Path(args.router_weights_output or (local_dir / "theorem_router_weights.json"))
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(weight_map, indent=2), encoding="utf-8")
    print(f"[OllamaBootstrap] Wrote TheoremRouter weights to {weights_path}")


if __name__ == "__main__":
    main()
