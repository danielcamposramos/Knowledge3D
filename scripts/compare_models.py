#!/usr/bin/env python3
"""
Compare multiple navigation checkpoints on the calculus microbench.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from knowledge3d.training.math_benchmarks.navigation_model import NavigationSeqModel
from knowledge3d.training.math_benchmarks.recursive_solver import RecursiveSolver


def _load_bench(path: Path) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _load_model(path: Path) -> Tuple[NavigationSeqModel, List[str]]:
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    embedding_dim = int(ckpt.get("embedding_dim") or 0)
    hidden_dim = int(ckpt.get("hidden_dim") or 0)
    vocab_size = int(ckpt.get("vocab_size") or 0)
    rule_registry = ckpt.get("rule_registry") or []

    if not embedding_dim or not hidden_dim or not vocab_size:
        raise SystemExit(f"Checkpoint missing model dimensions: {path}")

    model = NavigationSeqModel(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(state)
    model.eval()
    return model, list(rule_registry)


def _eval_model(
    solver: RecursiveSolver,
    bench: List[Dict[str, str]],
    *,
    tolerance: float,
) -> Dict[str, float]:
    total = 0
    correct = 0
    autonomy_sum = 0.0
    total_mismatches = 0
    total_policy_steps = 0
    with_trace = 0

    for item in bench:
        problem = item.get("problem", "")
        expected_raw = item.get("answer", "")
        try:
            expected = float(expected_raw)
        except Exception:
            expected = None

        pred = solver.solve(problem)
        total += 1

        if pred is not None and expected is not None and abs(pred - expected) <= tolerance:
            correct += 1

        trace = solver.get_last_trace()
        steps = len(trace.get("step_sequence") or [])
        if steps > 0:
            policy_steps = int(trace.get("policy_steps", 0))
            mismatches = int(trace.get("policy_mismatches", 0))
            autonomy_sum += policy_steps / float(steps)
            total_policy_steps += policy_steps + mismatches
            total_mismatches += mismatches
            with_trace += 1

    avg_autonomy = autonomy_sum / float(with_trace) if with_trace else 0.0
    avg_drift = total_mismatches / float(total_policy_steps) if total_policy_steps else 0.0
    return {
        "total": float(total),
        "correct": float(correct),
        "accuracy": float(correct) / float(total) if total else 0.0,
        "avg_autonomy": avg_autonomy,
        "avg_drift": avg_drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare navigation models on a microbench.")
    parser.add_argument("--models", nargs="+", required=True, help="List of model checkpoints to compare.")
    parser.add_argument("--names", nargs="+", required=True, help="Names for the models (e.g. V1 V3 V4).")
    parser.add_argument("--bench", required=True, help="Microbench JSONL path.")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Answer tolerance.")
    args = parser.parse_args()

    if len(args.models) != len(args.names):
        raise SystemExit("Number of models must match number of names.")

    bench = _load_bench(Path(args.bench))
    if not bench:
        raise SystemExit("Bench dataset is empty.")

    results: Dict[str, Dict[str, float]] = {}
    base_registry = None

    for name, model_path in zip(args.names, args.models):
        print(f"--- Evaluating {name} ({model_path}) ---")
        model, registry = _load_model(Path(model_path))
        if base_registry is None:
            base_registry = registry
        elif base_registry != registry:
            print(f"Warning: Registry mismatch for {name}. Skipping.")
            continue

        solver = RecursiveSolver(verbose=False, policy_model=model, policy_registry=registry)
        stats = _eval_model(solver, bench, tolerance=float(args.tolerance))
        results[name] = stats

    # Print report
    print("\n" + "=" * 60)
    print(" " * 15 + "=== RLWHF Impact Report ===")
    header = f"{'Metric':<12}| " + " | ".join(f"{name:<10}" for name in args.names)
    print(header)
    print("-" * len(header))

    acc_row = f"{'Accuracy':<12}| " + " | ".join(f"{results.get(n, {}).get('accuracy', 0.0):<10.1%}" for n in args.names)
    print(acc_row)

    autonomy_row = f"{'Autonomy':<12}| " + " | ".join(f"{results.get(n, {}).get('avg_autonomy', 0.0):<10.1%}" for n in args.names)
    print(autonomy_row)

    drift_row = f"{'Drift':<12}| " + " | ".join(f"{results.get(n, {}).get('avg_drift', 0.0):<10.1%}" for n in args.names)
    print(drift_row)
    print("-" * len(header))


if __name__ == "__main__":
    main()
