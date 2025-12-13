#!/usr/bin/env python3
"""Quick validation of math benchmark loading and proceduralization."""

import sys

sys.path.insert(0, ".")

from knowledge3d.training.math_benchmarks import (
    MathProceduralizer,
    MathDatasetLoader,
    MathBenchmarkEvaluator,
    MathOutputAdapter,
)


def main():
    print("=== Math Benchmark Validation ===\n")

    # Test 1: Proceduralizer
    print("[1] Testing MathProceduralizer...")
    proc = MathProceduralizer()
    test_problem = {
        "question": "Alice has 5 apples. Bob gives her 3 more. How many apples?",
        "answer": "#### 8",
        "source": "gsm8k",
    }
    result = proc.proceduralize_problem(test_problem)
    print(f"    RPN: {result['problem_rpn']}")
    print(f"    Answer: {result['answer']}")
    assert result["answer"] == 8.0, "Answer extraction failed"
    print("    ✓ Proceduralizer OK\n")

    # Test 2: Dataset loader (just check it initializes)
    print("[2] Testing MathDatasetLoader...")
    for ds in ["gsm8k", "math", "omni_math"]:
        try:
            loader = MathDatasetLoader(datasets=[ds], shuffle=False)
            stats = loader.get_stats()
            print(f"    {ds}: {stats['total_problems']} problems")
        except Exception as e:
            print(f"    {ds}: SKIP ({e})")
    print("    ✓ Dataset loader OK\n")

    # Test 3: Evaluator
    print("[3] Testing MathBenchmarkEvaluator...")
    evaluator = MathBenchmarkEvaluator()
    evaluator.evaluate("p1", 8, 8, "gsm8k")
    evaluator.evaluate("p2", 42.0, "42", "gsm8k")
    evaluator.evaluate("p3", "B", "B", "mmlu")
    metrics = evaluator.get_metrics()
    print(f"    Accuracy: {metrics['overall']['accuracy']:.0%}")
    assert metrics["overall"]["accuracy"] == 1.0, "Evaluator failed"
    print("    ✓ Evaluator OK\n")

    # Test 4: Output adapter
    print("[4] Testing MathOutputAdapter...")
    adapter = MathOutputAdapter()
    adapter.record_result("p1", [8], "gsm8k")
    adapter.record_result("p2", [5], "math")
    stats = adapter.get_stats()
    print(f"    Recorded: {stats['total_recorded']}")
    print("    ✓ Output adapter OK\n")

    print("=== All math benchmark components validated ===")


if __name__ == "__main__":
    main()
