#!/usr/bin/env python3
"""Quick test of MMLU benchmark integration."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.mmlu import MMLUBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def main():
    print("=== MMLU Quick Test ===\n")

    # Initialize Knowledgeverse
    print("[1/4] Initializing Knowledgeverse...")
    kv = Knowledgeverse(storage_root="../Knowledge3D.local")
    print(f"✓ Knowledgeverse ready (ID: {id(kv)})\n")

    # Initialize MMLU benchmark
    print("[2/4] Loading MMLU benchmark (10 questions)...")
    mmlu = MMLUBenchmark(
        knowledgeverse=kv,
        dataset_path="/K3D/K3D_llama_cpp/datasets/MMLU/data",
        max_questions=10,
        subjects="all",  # All subjects, but only first 10 questions
    )

    print(f"✓ MMLU loaded:")
    print(f"  - Total questions: {len(mmlu.questions)}")
    print(f"  - Dataset source: {mmlu.dataset_source}")
    print(f"  - Synthetic fallback: {mmlu.synthetic_fallback}")
    print(f"  - Dataset file: {mmlu.dataset_file}\n")

    if mmlu.synthetic_fallback:
        print("⚠️  WARNING: Using synthetic fallback (real dataset not found)")
        print("   This is OK for development, but NOT for paper claims!\n")

    # Show sample questions
    print("[3/4] Sample questions:")
    for i, q in enumerate(mmlu.questions[:3], 1):
        print(f"\n  Q{i} [{q['subject']}]:")
        print(f"     {q['question_text'][:80]}...")
        print(f"     Options: {len(q['options'])}")
        print(f"     Correct: {q['correct_answer'][:30]}...")

    # Run benchmark
    print("\n[4/4] Running benchmark (enriched mode)...")
    results = mmlu.run_benchmark(use_enriched=True)

    print(f"\n✓ Benchmark complete:")
    print(f"  - Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total_questions']})")
    print(f"  - Subjects tested: {results.get('subjects_tested', 'N/A')}")
    print(f"  - Dataset source: {results['dataset_source']}")

    if 'domain_breakdown' in results:
        print(f"\n  Domain breakdown:")
        for domain, stats in results['domain_breakdown'].items():
            acc = stats['accuracy']
            total = stats['total']
            print(f"    - {domain}: {acc:.2%} ({total} questions)")

    print("\n=== Test Complete ===")
    print("✅ MMLU benchmark is working!")

    if results['synthetic_fallback']:
        print("\n⚠️  REMINDER: This test used synthetic fallback.")
        print("   For paper claims, ensure real MMLU dataset is available.")
    else:
        print(f"\n✅ Using real MMLU dataset ({results['total_questions']} questions)")
        print("   Ready for scientific validation!")


if __name__ == "__main__":
    main()
