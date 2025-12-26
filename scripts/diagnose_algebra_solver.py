#!/usr/bin/env python3
"""
Diagnose algebra solver coverage on MATH / AMC-AIME / Omni-MATH.

Pure Python, no numpy/cupy.
"""

import json
import random
from pathlib import Path


def load_samples(dataset: str, n: int = 10):
    """Load n sample problems from a dataset."""
    base = Path("/K3D/K3D_llama_cpp/datasets")

    if dataset == "math":
        path = base / "math/data/train.jsonl"
        problems = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    problems.append(json.loads(line))
        return random.sample(problems, min(n, len(problems)))

    if dataset == "amc_aime":
        path = base / "AMC-AIME/data"
        problems = []
        if path.exists():
            for jsonl in path.glob("*.jsonl"):
                with open(jsonl) as f:
                    for line in f:
                        problems.append(json.loads(line))
        return random.sample(problems, min(n, len(problems)))

    if dataset == "omni_math":
        path = base / "Omni-MATH/Omni-Math.jsonl"
        problems = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    problems.append(json.loads(line))
        return random.sample(problems, min(n, len(problems)))

    return []


def diagnose_problem(problem: dict, solver):
    """Run full diagnostic on a single problem."""
    text = problem.get("problem", problem.get("question", ""))[:500]
    answer = problem.get("answer", problem.get("solution", ""))

    classification = solver.classifier.classify(text)
    result, metadata = solver.solve(text)

    return {
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "ground_truth": str(answer)[:100],
        "classification": {
            "type": classification.problem_type,
            "subtype": classification.subtype,
            "confidence": classification.confidence,
            "coefficients": classification.coefficients,
        },
        "strategy": metadata.get("strategy", "none"),
        "result": result,
        "variables_found": metadata.get("variables", {}),
    }


def main():
    from knowledge3d.training.math_benchmarks.algebra_solver import AlgebraSolver, DEPRECATION_MSG

    try:
        solver = AlgebraSolver()
    except RuntimeError as exc:
        print(DEPRECATION_MSG)
        print(f"Diagnostic halted: {exc}")
        return

    print("=" * 80)
    print("ALGEBRA SOLVER DIAGNOSTICS")
    print("=" * 80)

    for dataset in ["math", "amc_aime", "omni_math"]:
        print(f"\n{'=' * 40}")
        print(f"DATASET: {dataset.upper()}")
        print("=" * 40)

        samples = load_samples(dataset, n=5)

        classification_counts = {}
        strategy_counts = {}
        solved_count = 0

        for i, problem in enumerate(samples):
            diag = diagnose_problem(problem, solver)

            print(f"\n--- Problem {i+1} ---")
            print(f"Text: {diag['text_preview']}")
            print(f"Ground Truth: {diag['ground_truth']}")
            print(f"Classification: {diag['classification']['type']}/{diag['classification']['subtype']} "
                  f"(conf={diag['classification']['confidence']:.2f})")
            print(f"Coefficients found: {diag['classification']['coefficients']}")
            print(f"Strategy: {diag['strategy']}")
            print(f"Result: {diag['result']}")

            ctype = diag["classification"]["type"]
            classification_counts[ctype] = classification_counts.get(ctype, 0) + 1
            strategy_counts[diag["strategy"]] = strategy_counts.get(diag["strategy"], 0) + 1
            if diag["result"] is not None:
                solved_count += 1

        total = len(samples) if samples else 1
        print(f"\n--- {dataset.upper()} SUMMARY ---")
        print(f"Classification distribution: {classification_counts}")
        print(f"Strategy distribution: {strategy_counts}")
        print(f"Solved: {solved_count}/{total}")


if __name__ == "__main__":
    main()
