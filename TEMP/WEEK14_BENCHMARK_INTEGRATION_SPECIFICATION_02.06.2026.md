# Week 14: Benchmark Integration Specification — Phase 1C

**Created:** February 6, 2026
**Author:** Claude (Architecture Partner)
**Priority:** CRITICAL (Prize-Winning Validation)
**Status:** Architecture Specification
**Context:** Post-Week 13 (38/38 tests, hardened ingestion, enriched Galaxies)

---

## Executive Summary

**Goal:** Integrate benchmark evaluation pipelines to measure performance with enriched knowledge base.

**Benchmarks:**
1. **ARC-AGI 2** — Visual reasoning (46.7% baseline → 55%+ target)
2. **Math Competitions** — AMC, AIME, IMO (0% baseline → 30%+ target)
3. **Last Humanity Exam** — Multi-domain reasoning (0% baseline → 40%+ target)

**Key Insight:** We've completed knowledge preparation (Phase 1B) and hardening (Week 13). Now we validate that the enriched Galaxies (500+ Grammar rules, 1000+ Math symbols, 200+ Reality procedures) actually improve benchmark performance.

**Success Criteria:**
- ✅ All benchmarks integrated and runnable
- ✅ Baseline measurements captured (enriched vs empty mind)
- ✅ Performance improves over empty mind baseline
- ✅ Identify gaps for iterative improvement

---

## Benchmark 1: ARC-AGI 2 Integration

### Background

**Previous Result:** 46.7% on ARC-AGI validation set (Sovereign TRM v7, "empty mind" with minimal Drawing/Grammar rules)

**Target:** 55%+ (prize threshold for ARC-AGI 2)

**Hypothesis:** Enriched Drawing Galaxy (300+ geometric patterns) + Grammar Galaxy (500+ transformation rules) should improve visual reasoning.

### Architecture

```python
# In benchmarks/arc_agi_2.py

class ARCAGI2Benchmark:
    """
    ARC-AGI 2 benchmark integration with enriched Galaxies.
    """

    def __init__(self, knowledgeverse, dataset_path: str):
        self.kv = knowledgeverse
        self.dataset_path = dataset_path
        self.tasks = self._load_tasks()
        self.results = []

    def _load_tasks(self) -> List[Dict]:
        """Load ARC-AGI 2 evaluation tasks."""
        import json

        tasks = []
        task_files = glob.glob(f"{self.dataset_path}/*.json")

        for task_file in task_files:
            with open(task_file, 'r') as f:
                task = json.load(f)
                tasks.append({
                    'id': os.path.basename(task_file).replace('.json', ''),
                    'train': task['train'],
                    'test': task['test']
                })

        return tasks

    def run_benchmark(self, use_enriched: bool = True) -> Dict:
        """
        Run ARC-AGI 2 benchmark.

        Args:
            use_enriched: Use enriched Galaxies (True) or empty mind (False)

        Returns:
            dict: {
                "total_tasks": int,
                "correct": int,
                "accuracy": float,
                "results": List[dict]
            }
        """
        correct = 0

        for task in self.tasks:
            result = self._solve_task(task, use_enriched=use_enriched)

            if result['correct']:
                correct += 1

            self.results.append(result)

        accuracy = correct / len(self.tasks) if self.tasks else 0.0

        return {
            "benchmark": "ARC-AGI 2",
            "use_enriched": use_enriched,
            "total_tasks": len(self.tasks),
            "correct": correct,
            "accuracy": accuracy,
            "results": self.results
        }

    def _solve_task(self, task: Dict, use_enriched: bool) -> Dict:
        """
        Solve single ARC-AGI task.

        Args:
            task: Task dict with 'train' and 'test'
            use_enriched: Use enriched Galaxies

        Returns:
            dict: {
                "task_id": str,
                "correct": bool,
                "predicted": array,
                "expected": array,
                "reasoning_trace": List[str]
            }
        """
        from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator

        navigator = TRMNavigator(self.kv)

        # 1. Query enriched Drawing + Grammar Galaxies
        if use_enriched:
            relevant_patterns = navigator.query(
                "visual pattern transformation",
                galaxy_names=["Drawing", "Grammar"],
                top_k=20
            )
        else:
            # Empty mind: only baseline patterns
            relevant_patterns = navigator.query(
                "visual pattern transformation",
                galaxy_names=["Drawing"],
                top_k=5  # Much fewer patterns
            )

        # 2. Compose RPN program from patterns
        composed_program = navigator.compose(
            task_examples=task['train'],
            patterns=relevant_patterns,
            specialist='visual'
        )

        # 3. Execute on test input
        test_input = task['test'][0]['input']
        predicted = navigator.execute(composed_program, test_input)

        # 4. Verify correctness
        expected = task['test'][0]['output']
        correct = self._grids_match(predicted, expected)

        return {
            "task_id": task['id'],
            "correct": correct,
            "predicted": predicted.tolist(),
            "expected": expected,
            "reasoning_trace": navigator.get_reasoning_trace(),
            "patterns_used": len(relevant_patterns)
        }

    def _grids_match(self, predicted: np.ndarray, expected: np.ndarray) -> bool:
        """Check if predicted grid matches expected."""
        if predicted.shape != expected.shape:
            return False
        return np.array_equal(predicted, expected)

    def save_results(self, output_path: str):
        """Save benchmark results to JSON."""
        import json

        summary = {
            "benchmark": "ARC-AGI 2",
            "total_tasks": len(self.tasks),
            "correct": sum(1 for r in self.results if r['correct']),
            "accuracy": sum(1 for r in self.results if r['correct']) / len(self.tasks),
            "results": self.results
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[ARCAGI2] Results saved to {output_path}")
```

### Comparison Script

```python
# In scripts/benchmark_arc_agi_comparison.py

"""
Compare ARC-AGI 2 performance: empty mind vs enriched.
"""

import asyncio
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

async def main():
    # Initialize Knowledgeverse
    kv = Knowledgeverse()

    # Load benchmark
    benchmark = ARCAGI2Benchmark(
        knowledgeverse=kv,
        dataset_path="../Knowledge3D.local/datasets/arc_agi_2/evaluation"
    )

    print("="*60)
    print("ARC-AGI 2 BENCHMARK COMPARISON")
    print("="*60)

    # Run with empty mind (baseline)
    print("\n[1/2] Running with EMPTY MIND (baseline)...")
    empty_results = benchmark.run_benchmark(use_enriched=False)
    print(f"Empty Mind Accuracy: {empty_results['accuracy']:.2%}")
    benchmark.save_results("../Knowledge3D.local/results/arc_agi_2_empty_mind.json")

    # Clear results for next run
    benchmark.results = []

    # Run with enriched Galaxies
    print("\n[2/2] Running with ENRICHED GALAXIES...")
    enriched_results = benchmark.run_benchmark(use_enriched=True)
    print(f"Enriched Accuracy: {enriched_results['accuracy']:.2%}")
    benchmark.save_results("../Knowledge3D.local/results/arc_agi_2_enriched.json")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Empty Mind:  {empty_results['accuracy']:.2%} ({empty_results['correct']}/{empty_results['total_tasks']})")
    print(f"Enriched:    {enriched_results['accuracy']:.2%} ({enriched_results['correct']}/{enriched_results['total_tasks']})")

    improvement = enriched_results['accuracy'] - empty_results['accuracy']
    print(f"Improvement: {improvement:+.2%}")

    if enriched_results['accuracy'] >= 0.55:
        print("\n✅ TARGET ACHIEVED (55%+)")
    else:
        print(f"\n⚠️  Target not yet reached (need {0.55 - enriched_results['accuracy']:.2%} more)")

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Benchmark 2: Math Competitions

### Background

**Baseline:** 0% (no prior math benchmark integration)

**Target:** 30%+ (competitive baseline for AMC/AIME)

**Hypothesis:** Enriched Math Galaxy (1000+ symbols) + Grammar Galaxy (500+ rules) should enable symbolic reasoning.

### Architecture

```python
# In benchmarks/math_competitions.py

class MathCompetitionBenchmark:
    """
    Math competition benchmark (AMC, AIME, IMO).
    """

    def __init__(self, knowledgeverse, dataset_path: str):
        self.kv = knowledgeverse
        self.dataset_path = dataset_path
        self.problems = self._load_problems()
        self.results = []

    def _load_problems(self) -> List[Dict]:
        """Load math competition problems."""
        import json

        problems = []

        # Load AMC problems
        with open(f"{self.dataset_path}/amc_problems.json", 'r') as f:
            amc = json.load(f)
            problems.extend([{**p, 'competition': 'AMC'} for p in amc])

        # Load AIME problems
        with open(f"{self.dataset_path}/aime_problems.json", 'r') as f:
            aime = json.load(f)
            problems.extend([{**p, 'competition': 'AIME'} for p in aime])

        # Load IMO problems (optional)
        imo_path = f"{self.dataset_path}/imo_problems.json"
        if os.path.exists(imo_path):
            with open(imo_path, 'r') as f:
                imo = json.load(f)
                problems.extend([{**p, 'competition': 'IMO'} for p in imo])

        return problems

    def run_benchmark(self, use_enriched: bool = True) -> Dict:
        """
        Run math competition benchmark.

        Args:
            use_enriched: Use enriched Math Galaxy

        Returns:
            dict: Results by competition level
        """
        results_by_competition = {}

        for problem in self.problems:
            result = self._solve_problem(problem, use_enriched=use_enriched)

            competition = problem['competition']
            if competition not in results_by_competition:
                results_by_competition[competition] = {
                    "total": 0,
                    "correct": 0,
                    "results": []
                }

            results_by_competition[competition]["total"] += 1
            if result['correct']:
                results_by_competition[competition]["correct"] += 1

            results_by_competition[competition]["results"].append(result)

        # Calculate accuracies
        for competition, data in results_by_competition.items():
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0.0

        return {
            "benchmark": "Math Competitions",
            "use_enriched": use_enriched,
            "results_by_competition": results_by_competition,
            "overall_accuracy": sum(d["correct"] for d in results_by_competition.values()) /
                               sum(d["total"] for d in results_by_competition.values())
        }

    def _solve_problem(self, problem: Dict, use_enriched: bool) -> Dict:
        """
        Solve single math problem.

        Args:
            problem: {
                "id": str,
                "competition": str,
                "problem_text": str,
                "answer": str or number,
                "solution_steps": List[str] (optional)
            }

        Returns:
            dict: Solution result
        """
        from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator

        navigator = TRMNavigator(self.kv)

        # 1. Query Math + Grammar Galaxies
        if use_enriched:
            relevant_symbols = navigator.query(
                problem['problem_text'],
                galaxy_names=["Math", "Grammar"],
                top_k=30
            )
        else:
            # Empty mind: minimal symbols
            relevant_symbols = navigator.query(
                problem['problem_text'],
                galaxy_names=["Math"],
                top_k=5
            )

        # 2. Compose solution RPN program
        solution_program = navigator.compose(
            query=problem['problem_text'],
            patterns=relevant_symbols,
            specialist='math'
        )

        # 3. Execute solution
        try:
            predicted_answer = navigator.execute(solution_program)
        except Exception as e:
            predicted_answer = None
            error = str(e)

        # 4. Verify correctness
        expected_answer = problem['answer']
        correct = self._answers_match(predicted_answer, expected_answer)

        return {
            "problem_id": problem['id'],
            "competition": problem['competition'],
            "correct": correct,
            "predicted_answer": predicted_answer,
            "expected_answer": expected_answer,
            "reasoning_trace": navigator.get_reasoning_trace(),
            "symbols_used": len(relevant_symbols)
        }

    def _answers_match(self, predicted, expected) -> bool:
        """Check if predicted answer matches expected."""
        if predicted is None:
            return False

        # Normalize answers
        pred_str = str(predicted).strip().lower()
        exp_str = str(expected).strip().lower()

        # Try exact match
        if pred_str == exp_str:
            return True

        # Try numeric comparison (with tolerance)
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 1e-6
        except (ValueError, TypeError):
            return False

    def save_results(self, output_path: str):
        """Save results to JSON."""
        import json

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
```

### Comparison Script

```python
# In scripts/benchmark_math_comparison.py

"""
Compare math competition performance: empty mind vs enriched.
"""

import asyncio
from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

async def main():
    kv = Knowledgeverse()

    benchmark = MathCompetitionBenchmark(
        knowledgeverse=kv,
        dataset_path="../Knowledge3D.local/datasets/math_competitions"
    )

    print("="*60)
    print("MATH COMPETITIONS BENCHMARK COMPARISON")
    print("="*60)

    # Empty mind
    print("\n[1/2] Running with EMPTY MIND...")
    empty_results = benchmark.run_benchmark(use_enriched=False)
    print(f"Empty Mind Overall: {empty_results['overall_accuracy']:.2%}")

    # Enriched
    print("\n[2/2] Running with ENRICHED GALAXIES...")
    enriched_results = benchmark.run_benchmark(use_enriched=True)
    print(f"Enriched Overall: {enriched_results['overall_accuracy']:.2%}")

    # Summary by competition
    print("\n" + "="*60)
    print("SUMMARY BY COMPETITION")
    print("="*60)

    for comp in ['AMC', 'AIME', 'IMO']:
        if comp in enriched_results['results_by_competition']:
            empty_acc = empty_results['results_by_competition'][comp]['accuracy']
            enrich_acc = enriched_results['results_by_competition'][comp]['accuracy']
            improvement = enrich_acc - empty_acc

            print(f"{comp}:")
            print(f"  Empty Mind: {empty_acc:.2%}")
            print(f"  Enriched:   {enrich_acc:.2%}")
            print(f"  Improvement: {improvement:+.2%}")

    if enriched_results['overall_accuracy'] >= 0.30:
        print("\n✅ TARGET ACHIEVED (30%+)")
    else:
        print(f"\n⚠️  Target not yet reached")

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Benchmark 3: Last Humanity Exam

### Background

**Baseline:** 0% (no prior integration)

**Target:** 40%+ (multi-domain reasoning baseline)

**Hypothesis:** Cross-domain enrichment (Math + Physics + Grammar) enables complex reasoning.

### Architecture

```python
# In benchmarks/last_humanity_exam.py

class LastHumanityExamBenchmark:
    """
    Last Humanity Exam benchmark (multi-domain reasoning).
    """

    def __init__(self, knowledgeverse, dataset_path: str):
        self.kv = knowledgeverse
        self.dataset_path = dataset_path
        self.questions = self._load_questions()
        self.results = []

    def _load_questions(self) -> List[Dict]:
        """Load Last Humanity Exam questions."""
        import json

        with open(f"{self.dataset_path}/last_humanity_exam.json", 'r') as f:
            data = json.load(f)

        return data['questions']

    def run_benchmark(self, use_enriched: bool = True) -> Dict:
        """Run Last Humanity Exam benchmark."""

        correct = 0

        for question in self.questions:
            result = self._answer_question(question, use_enriched=use_enriched)

            if result['correct']:
                correct += 1

            self.results.append(result)

        accuracy = correct / len(self.questions) if self.questions else 0.0

        return {
            "benchmark": "Last Humanity Exam",
            "use_enriched": use_enriched,
            "total_questions": len(self.questions),
            "correct": correct,
            "accuracy": accuracy,
            "results": self.results
        }

    def _answer_question(self, question: Dict, use_enriched: bool) -> Dict:
        """
        Answer single exam question.

        Args:
            question: {
                "id": str,
                "domain": str,  # "math", "physics", "logic", "multi"
                "question_text": str,
                "options": List[str],
                "correct_answer": str
            }

        Returns:
            dict: Answer result
        """
        from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator

        navigator = TRMNavigator(self.kv)

        # 1. Determine relevant Galaxies based on domain
        domain = question.get('domain', 'multi')
        galaxy_names = self._get_galaxies_for_domain(domain)

        # 2. Query relevant Galaxies
        if use_enriched:
            relevant_knowledge = navigator.query(
                question['question_text'],
                galaxy_names=galaxy_names,
                top_k=40
            )
        else:
            relevant_knowledge = navigator.query(
                question['question_text'],
                galaxy_names=galaxy_names[:1],  # Only one galaxy
                top_k=5
            )

        # 3. Determine specialist
        specialist = self._get_specialist_for_domain(domain)

        # 4. Compose reasoning
        reasoning = navigator.compose(
            query=question['question_text'],
            patterns=relevant_knowledge,
            specialist=specialist
        )

        # 5. Select answer
        predicted_answer = navigator.select_answer(
            reasoning=reasoning,
            options=question['options']
        )

        # 6. Verify correctness
        correct = (predicted_answer == question['correct_answer'])

        return {
            "question_id": question['id'],
            "domain": domain,
            "correct": correct,
            "predicted_answer": predicted_answer,
            "correct_answer": question['correct_answer'],
            "reasoning_trace": navigator.get_reasoning_trace(),
            "knowledge_used": len(relevant_knowledge)
        }

    def _get_galaxies_for_domain(self, domain: str) -> List[str]:
        """Map domain to relevant Galaxies."""
        domain_map = {
            "math": ["Math", "Grammar"],
            "physics": ["Reality", "Math", "Grammar"],
            "logic": ["Grammar"],
            "visual": ["Drawing", "Grammar"],
            "multi": ["Math", "Reality", "Grammar", "Drawing"]
        }
        return domain_map.get(domain, ["Grammar"])

    def _get_specialist_for_domain(self, domain: str) -> str:
        """Map domain to TRM specialist."""
        specialist_map = {
            "math": "math",
            "physics": "physics",
            "logic": "grammar",
            "visual": "visual",
            "multi": "cartographer"  # Router specialist for multi-domain
        }
        return specialist_map.get(domain, "grammar")

    def save_results(self, output_path: str):
        """Save results to JSON."""
        import json

        summary = {
            "benchmark": "Last Humanity Exam",
            "total_questions": len(self.questions),
            "correct": sum(1 for r in self.results if r['correct']),
            "accuracy": sum(1 for r in self.results if r['correct']) / len(self.questions),
            "results_by_domain": self._summarize_by_domain(),
            "results": self.results
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def _summarize_by_domain(self) -> Dict:
        """Summarize results by domain."""
        by_domain = {}

        for result in self.results:
            domain = result['domain']
            if domain not in by_domain:
                by_domain[domain] = {"total": 0, "correct": 0}

            by_domain[domain]["total"] += 1
            if result['correct']:
                by_domain[domain]["correct"] += 1

        for domain, data in by_domain.items():
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0.0

        return by_domain
```

### Comparison Script

```python
# In scripts/benchmark_lhe_comparison.py

"""
Compare Last Humanity Exam performance: empty mind vs enriched.
"""

import asyncio
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

async def main():
    kv = Knowledgeverse()

    benchmark = LastHumanityExamBenchmark(
        knowledgeverse=kv,
        dataset_path="../Knowledge3D.local/datasets/last_humanity_exam"
    )

    print("="*60)
    print("LAST HUMANITY EXAM BENCHMARK COMPARISON")
    print("="*60)

    # Empty mind
    print("\n[1/2] Running with EMPTY MIND...")
    empty_results = benchmark.run_benchmark(use_enriched=False)
    print(f"Empty Mind Accuracy: {empty_results['accuracy']:.2%}")

    # Enriched
    print("\n[2/2] Running with ENRICHED GALAXIES...")
    enriched_results = benchmark.run_benchmark(use_enriched=True)
    print(f"Enriched Accuracy: {enriched_results['accuracy']:.2%}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Empty Mind:  {empty_results['accuracy']:.2%}")
    print(f"Enriched:    {enriched_results['accuracy']:.2%}")
    print(f"Improvement: {enriched_results['accuracy'] - empty_results['accuracy']:+.2%}")

    if enriched_results['accuracy'] >= 0.40:
        print("\n✅ TARGET ACHIEVED (40%+)")
    else:
        print(f"\n⚠️  Target not yet reached")

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Unified Benchmark Runner

```python
# In scripts/run_all_benchmarks.py

"""
Run all benchmarks in sequence with unified reporting.
"""

import asyncio
import json
from datetime import datetime
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

async def main():
    print("="*60)
    print("KNOWLEDGE3D BENCHMARK SUITE")
    print("Week 14: Enriched vs Empty Mind Comparison")
    print("="*60)

    # Initialize Knowledgeverse
    print("\nInitializing Knowledgeverse...")
    kv = Knowledgeverse()

    results_dir = "../Knowledge3D.local/results/week14"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {}
    }

    # 1. ARC-AGI 2
    print("\n" + "="*60)
    print("BENCHMARK 1: ARC-AGI 2")
    print("="*60)

    arc_benchmark = ARCAGI2Benchmark(
        knowledgeverse=kv,
        dataset_path="../Knowledge3D.local/datasets/arc_agi_2/evaluation"
    )

    arc_empty = arc_benchmark.run_benchmark(use_enriched=False)
    arc_benchmark.results = []
    arc_enriched = arc_benchmark.run_benchmark(use_enriched=True)

    all_results["benchmarks"]["arc_agi_2"] = {
        "empty_mind": arc_empty,
        "enriched": arc_enriched,
        "improvement": arc_enriched["accuracy"] - arc_empty["accuracy"]
    }

    print(f"\nARC-AGI 2:")
    print(f"  Empty Mind: {arc_empty['accuracy']:.2%}")
    print(f"  Enriched:   {arc_enriched['accuracy']:.2%}")
    print(f"  Improvement: {all_results['benchmarks']['arc_agi_2']['improvement']:+.2%}")

    # 2. Math Competitions
    print("\n" + "="*60)
    print("BENCHMARK 2: MATH COMPETITIONS")
    print("="*60)

    math_benchmark = MathCompetitionBenchmark(
        knowledgeverse=kv,
        dataset_path="../Knowledge3D.local/datasets/math_competitions"
    )

    math_empty = math_benchmark.run_benchmark(use_enriched=False)
    math_enriched = math_benchmark.run_benchmark(use_enriched=True)

    all_results["benchmarks"]["math_competitions"] = {
        "empty_mind": math_empty,
        "enriched": math_enriched,
        "improvement": math_enriched["overall_accuracy"] - math_empty["overall_accuracy"]
    }

    print(f"\nMath Competitions:")
    print(f"  Empty Mind: {math_empty['overall_accuracy']:.2%}")
    print(f"  Enriched:   {math_enriched['overall_accuracy']:.2%}")
    print(f"  Improvement: {all_results['benchmarks']['math_competitions']['improvement']:+.2%}")

    # 3. Last Humanity Exam
    print("\n" + "="*60)
    print("BENCHMARK 3: LAST HUMANITY EXAM")
    print("="*60)

    lhe_benchmark = LastHumanityExamBenchmark(
        knowledgeverse=kv,
        dataset_path="../Knowledge3D.local/datasets/last_humanity_exam"
    )

    lhe_empty = lhe_benchmark.run_benchmark(use_enriched=False)
    lhe_enriched = lhe_benchmark.run_benchmark(use_enriched=True)

    all_results["benchmarks"]["last_humanity_exam"] = {
        "empty_mind": lhe_empty,
        "enriched": lhe_enriched,
        "improvement": lhe_enriched["accuracy"] - lhe_empty["accuracy"]
    }

    print(f"\nLast Humanity Exam:")
    print(f"  Empty Mind: {lhe_empty['accuracy']:.2%}")
    print(f"  Enriched:   {lhe_enriched['accuracy']:.2%}")
    print(f"  Improvement: {all_results['benchmarks']['last_humanity_exam']['improvement']:+.2%}")

    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print("\nTargets vs Actual:")
    print(f"ARC-AGI 2:         {arc_enriched['accuracy']:.2%} (target: 55%+)")
    print(f"Math Competitions: {math_enriched['overall_accuracy']:.2%} (target: 30%+)")
    print(f"Last Humanity Exam: {lhe_enriched['accuracy']:.2%} (target: 40%+)")

    targets_met = 0
    if arc_enriched['accuracy'] >= 0.55:
        targets_met += 1
    if math_enriched['overall_accuracy'] >= 0.30:
        targets_met += 1
    if lhe_enriched['accuracy'] >= 0.40:
        targets_met += 1

    print(f"\nTargets Met: {targets_met}/3")

    # Save unified results
    results_path = f"{results_dir}/week14_all_benchmarks.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    asyncio.run(main())
```

---

## tmux Orchestration for Week 14

```bash
#!/bin/bash
# scripts/week14_benchmark_tmux.sh

tmux new-session -d -s k3d_week14

# Window 0: GPU Monitor
tmux rename-window -t k3d_week14:0 'gpu_monitor'
tmux send-keys -t k3d_week14:0 'watch -n 1 nvidia-smi' C-m

# Window 1: ARC-AGI 2
tmux new-window -t k3d_week14:1 -n 'arc_agi'
tmux send-keys -t k3d_week14:1 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m
tmux send-keys -t k3d_week14:1 '# Ready for ARC-AGI 2 benchmark' C-m

# Window 2: Math Competitions
tmux new-window -t k3d_week14:2 -n 'math'
tmux send-keys -t k3d_week14:2 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m
tmux send-keys -t k3d_week14:2 '# Ready for Math Competitions benchmark' C-m

# Window 3: Last Humanity Exam
tmux new-window -t k3d_week14:3 -n 'lhe'
tmux send-keys -t k3d_week14:3 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m
tmux send-keys -t k3d_week14:3 '# Ready for Last Humanity Exam benchmark' C-m

# Window 4: Unified Runner
tmux new-window -t k3d_week14:4 -n 'all_benchmarks'
tmux send-keys -t k3d_week14:4 'cd /mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D' C-m
tmux send-keys -t k3d_week14:4 '# Ready to run ALL benchmarks' C-m

tmux attach-session -t k3d_week14
```

**Usage:**

```bash
# Run all benchmarks
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py

# Or run individually (different tmux windows):
# Window 1:
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/benchmark_arc_agi_comparison.py

# Window 2:
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/benchmark_math_comparison.py

# Window 3:
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/benchmark_lhe_comparison.py
```

---

## Testing Strategy

### Test 1: Benchmark Loading

```python
def test_arc_agi_2_loading():
    """Test ARC-AGI 2 dataset loads correctly."""
    benchmark = ARCAGI2Benchmark(
        knowledgeverse=None,
        dataset_path="../Knowledge3D.local/datasets/arc_agi_2/evaluation"
    )

    assert len(benchmark.tasks) > 0
    assert all('train' in task and 'test' in task for task in benchmark.tasks)

def test_math_competitions_loading():
    """Test math competition dataset loads correctly."""
    benchmark = MathCompetitionBenchmark(
        knowledgeverse=None,
        dataset_path="../Knowledge3D.local/datasets/math_competitions"
    )

    assert len(benchmark.problems) > 0
    assert all('competition' in p for p in benchmark.problems)

def test_lhe_loading():
    """Test Last Humanity Exam dataset loads correctly."""
    benchmark = LastHumanityExamBenchmark(
        knowledgeverse=None,
        dataset_path="../Knowledge3D.local/datasets/last_humanity_exam"
    )

    assert len(benchmark.questions) > 0
    assert all('domain' in q for q in benchmark.questions)
```

### Test 2: Empty Mind Baseline

```python
async def test_arc_agi_empty_mind_baseline():
    """Test ARC-AGI 2 runs with empty mind (should match previous 46.7%)."""
    kv = Knowledgeverse()
    benchmark = ARCAGI2Benchmark(kv, dataset_path="...")

    results = benchmark.run_benchmark(use_enriched=False)

    # Should be close to previous baseline
    assert 0.40 <= results['accuracy'] <= 0.50  # Allow some variance
```

### Test 3: Enriched Improvement

```python
async def test_enriched_improves_over_empty():
    """Test that enriched Galaxies improve performance."""
    kv = Knowledgeverse()
    benchmark = ARCAGI2Benchmark(kv, dataset_path="...")

    empty_results = benchmark.run_benchmark(use_enriched=False)
    benchmark.results = []
    enriched_results = benchmark.run_benchmark(use_enriched=True)

    # Enriched should improve
    assert enriched_results['accuracy'] > empty_results['accuracy']
```

---

## Week 14 Implementation Timeline

### Day 1-2: Benchmark Infrastructure

**Files to Create:**
1. `benchmarks/__init__.py`
2. `benchmarks/arc_agi_2.py`
3. `benchmarks/math_competitions.py`
4. `benchmarks/last_humanity_exam.py`
5. `tests/test_benchmarks.py`

**Success Criteria:**
- ✅ All benchmarks load datasets correctly
- ✅ 3/3 loading tests passing

### Day 3: Individual Benchmark Runs

**Scripts to Create:**
1. `scripts/benchmark_arc_agi_comparison.py`
2. `scripts/benchmark_math_comparison.py`
3. `scripts/benchmark_lhe_comparison.py`

**Execution:**
- Run each benchmark individually
- Capture empty mind vs enriched results
- Save results to JSON

**Success Criteria:**
- ✅ All benchmarks complete without errors
- ✅ Results saved to `../Knowledge3D.local/results/week14/`

### Day 4: Unified Runner

**File to Create:**
- `scripts/run_all_benchmarks.py`

**Execution:**
- Run all benchmarks in sequence
- Generate unified report

**Success Criteria:**
- ✅ All 3 benchmarks run successfully
- ✅ Unified JSON report generated

### Day 5: Analysis & Iteration

**Tasks:**
1. Analyze results
2. Identify gaps (which tasks fail most)
3. Propose iterative improvements

**Deliverable:**
- `TEMP/CODEX_WEEK14_BENCHMARK_ANALYSIS_02.XX.2026.md`

---

## Success Metrics

**Targets:**
1. **ARC-AGI 2:** 55%+ (from 46.7% empty mind)
2. **Math Competitions:** 30%+ (from 0% empty mind)
3. **Last Humanity Exam:** 40%+ (from 0% empty mind)

**Baseline Comparison:**
- Enriched accuracy > Empty mind accuracy (for all benchmarks)
- Average improvement: +10% or more across all benchmarks

**Code Quality:**
- All benchmark classes implement consistent interface
- Results saved in unified JSON format
- Reasoning traces captured for debugging

---

## Codex Implementation Directive

**Priority:** CRITICAL (Week 14, Phase 1C)

**What to Implement:**

1. **Benchmark Infrastructure (Day 1-2)**
   - Create 3 benchmark classes
   - Implement dataset loading
   - Create loading tests

2. **Comparison Scripts (Day 3)**
   - Create 3 comparison scripts
   - Run individual benchmarks
   - Save results to JSON

3. **Unified Runner (Day 4)**
   - Create all-in-one benchmark script
   - Generate unified report

4. **Analysis (Day 5)**
   - Analyze results
   - Identify gaps
   - Write completion report

**Testing:**
- 3/3 loading tests passing
- All benchmarks complete without errors
- Results match expected format

---

## End of Specification

**Next Steps:**
1. Codex implements benchmark infrastructure
2. Codex runs all benchmarks with tmux orchestration
3. Codex analyzes results and writes completion report
4. Claude reviews results and proposes iterative improvements

**Remember:** This is baseline measurement. Performance may not hit targets immediately. The goal is to establish measurement infrastructure and identify gaps for iterative improvement.

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

---

**Claude (Architecture Partner)**
February 6, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
