#!/usr/bin/env python3
"""
Bootstrap and optionally train the router specialist for theorem routing.

Phase 1: heuristic routing to collect decisions.
Phase 2: train router specialist from routing history.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np

from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.cranium.router_specialist import RouterBootstrap, RouterSpecialistTrainer
from knowledge3d.training.math_benchmarks.calculus_grammar_rules import get_calculus_rules
from knowledge3d.cranium.math_galaxy_population import extract_theorem_patterns


@dataclass
class TheoremHeuristicRouter:
    rule_ids: List[str]

    def route_blend(self, task_description: str | None = None) -> Dict[str, float]:
        tags = []
        for token in (task_description or "").replace(",", " ").split():
            tok = token.strip().lower()
            if tok:
                tags.append(tok)
        if not tags:
            return {}
        tag_set = set(tags)
        scores: Dict[str, float] = {}
        for rule_id in self.rule_ids:
            tokens = [t for t in rule_id.replace("apply_", "").split("_") if t]
            overlap = len(tag_set.intersection(tokens))
            scores[rule_id] = float(overlap) if overlap else 0.05
        total = sum(scores.values())
        return {rid: score / total for rid, score in scores.items()} if total else {}


def _tags_to_vec(tags: List[str], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tag in tags:
        idx = abs(hash(tag)) % dim
        vec[idx] += 1.0
    return vec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="/K3D/Knowledge3D.local/galaxies/books_v5_clean2")
    parser.add_argument("--min-examples", type=int, default=2)
    parser.add_argument("--router-dims", type=int, default=256)
    parser.add_argument("--router-rank", type=int, default=16)
    parser.add_argument("--output", default="/K3D/Knowledge3D.local/logs/theorem_router_history.json")
    parser.add_argument("--train", action="store_true", help="Train router specialist after bootstrapping.")
    args = parser.parse_args()

    patterns = extract_theorem_patterns([args.artifact_dir], min_examples=args.min_examples)
    if not patterns:
        raise SystemExit("No theorem patterns found; aborting.")

    calculus_rules = get_calculus_rules()
    rule_ids = [r.rule_id for r in calculus_rules]
    if not rule_ids:
        raise SystemExit("No calculus grammar rules found; aborting.")

    swarm = AdaptiveSwarmTRM(SwarmConfig(base_dims=args.router_dims, min_dims=args.router_dims))
    for rule_id in rule_ids:
        swarm.register_specialist(rule_id, required_dims=args.router_dims, rank=args.router_rank)

    trainer = RouterSpecialistTrainer(swarm)
    trainer.register_router_specialist(num_specialists=len(rule_ids), router_dims=args.router_dims, router_rank=args.router_rank)

    heuristic_router = TheoremHeuristicRouter(rule_ids=rule_ids)
    bootstrap = RouterBootstrap(swarm, heuristic_router=heuristic_router)

    tasks = []
    for pattern in patterns:
        tags = list(pattern.get("semantic_tags") or [])
        description = " ".join(tags)
        tasks.append(
            {
                "input": _tags_to_vec(tags, args.router_dims),
                "description": description,
                "target_rule": pattern.get("grammar_rule"),
            }
        )

    def outcome_fn(task, weights: Dict[str, float]) -> float:
        target = task.get("target_rule")
        if not target:
            return 0.0
        return float(weights.get(target, 0.0))

    routing_history = bootstrap.collect_routing_data(tasks, outcome_fn, num_samples=None)
    output_path = Path(args.output)
    bootstrap.save_history(output_path)

    if args.train:
        trainer.train_from_history(routing_history, epochs=5)


if __name__ == "__main__":
    main()
