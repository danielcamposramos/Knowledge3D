"""Run benchmark suites as natural query health checks and log the results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable


HealthQueryFn = Callable[[dict[str, Any]], Any]


def _canonical_suite_name(name: str) -> str:
    key = str(name or "").strip().lower()
    aliases = {
        "arc": "arc",
        "arc_agi_2": "arc",
        "gsm8k": "gsm8k",
        "math": "math",
        "math_competitions": "math",
        "lhe": "lhe",
        "last_humanity_exam": "lhe",
        "mmlu": "mmlu",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported suite: {name}")
    return aliases[key]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        try:
            cleaned = str(value).strip().replace(",", "")
            if cleaned.startswith("$"):
                cleaned = cleaned[1:]
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1]
            return float(cleaned)
        except Exception:
            return None


def evaluate_answer(suite: str, answer: Any, expected: Any) -> bool:
    """Evaluate answer correctness for the health-check logger."""
    canonical = _canonical_suite_name(suite)
    if canonical in {"gsm8k", "math"}:
        answer_float = _to_float(answer)
        expected_float = _to_float(expected)
        if answer_float is not None and expected_float is not None:
            return abs(answer_float - expected_float) <= 1e-3
    if canonical == "arc":
        return json.dumps(answer, sort_keys=True) == json.dumps(expected, sort_keys=True)
    return str(answer).strip().lower() == str(expected).strip().lower()


def _synthetic_questions(canonical: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "gsm8k": [
            {
                "id": "gsm8k_synth_0",
                "question": "Janet has 16 eggs, uses 7, and sells the rest for $2 each. How much does she make?",
                "expected": "18",
                "payload": {"question_text": "Janet has 16 eggs, uses 7, and sells the rest for $2 each. How much does she make?"},
            }
        ],
        "math": [
            {
                "id": "math_synth_0",
                "question": "What is 7 * (3 + 2)?",
                "expected": "35",
                "payload": {"problem_text": "What is 7 * (3 + 2)?"},
            }
        ],
        "lhe": [
            {
                "id": "lhe_synth_0",
                "question": "If all A are B and all B are C, what is true?",
                "expected": "All A are C",
                "payload": {"question_text": "If all A are B and all B are C, what is true?"},
            }
        ],
        "mmlu": [
            {
                "id": "mmlu_synth_0",
                "question": "What is 7 * (3 + 2)?",
                "expected": "35",
                "payload": {"question_text": "What is 7 * (3 + 2)?"},
            }
        ],
        "arc": [
            {
                "id": "arc_synth_0",
                "question": "ARC task synthetic_flip_h",
                "expected": [[0, 9], [2, 1]],
                "payload": {"id": "synthetic_flip_h"},
            }
        ],
    }
    questions = list(rows[canonical])
    if limit is not None:
        questions = questions[: int(limit)]
    return questions


def load_questions(suite: str, count: int | None = None) -> list[dict[str, Any]]:
    """Load normalized benchmark questions without running them."""
    canonical = _canonical_suite_name(suite)
    limit = int(count) if count is not None else None
    try:
        if canonical == "gsm8k":
            from benchmarks.gsm8k import GSM8KBenchmark

            bench = GSM8KBenchmark(knowledgeverse=object(), max_questions=limit)
            return [
                {
                    "id": question["id"],
                    "question": question["question_text"],
                    "expected": question["correct_answer"],
                    "payload": question,
                }
                for question in bench.questions[:limit]
            ]
        if canonical == "math":
            from benchmarks.math_competitions import MathCompetitionBenchmark

            bench = MathCompetitionBenchmark(knowledgeverse=object(), max_problems=limit)
            return [
                {
                    "id": problem["id"],
                    "question": problem["problem_text"],
                    "expected": problem["answer"],
                    "payload": problem,
                }
                for problem in bench.problems[:limit]
            ]
        if canonical == "lhe":
            from benchmarks.last_humanity_exam import LastHumanityExamBenchmark

            bench = LastHumanityExamBenchmark(knowledgeverse=object(), max_questions=limit)
            return [
                {
                    "id": question["id"],
                    "question": question["question_text"],
                    "expected": question["correct_answer"],
                    "payload": question,
                }
                for question in bench.questions[:limit]
            ]
        if canonical == "mmlu":
            from benchmarks.mmlu import MMLUBenchmark

            bench = MMLUBenchmark(knowledgeverse=object(), max_questions=limit)
            return [
                {
                    "id": question["id"],
                    "question": question["question_text"],
                    "expected": question["correct_answer"],
                    "payload": question,
                }
                for question in bench.questions[:limit]
            ]
        from benchmarks.arc_agi_2 import ARCAGI2Benchmark

        bench = ARCAGI2Benchmark(knowledgeverse=object(), max_tasks=limit)
        return [
            {
                "id": task["id"],
                "question": f"ARC task {task['id']}",
                "expected": task["test"][0].get("output"),
                "payload": task,
            }
            for task in bench.tasks[:limit]
        ]
    except Exception:
        return _synthetic_questions(canonical, limit)


def _run_suite_via_benchmark(suite: str, count: int, knowledgeverse: Any | None = None) -> list[dict[str, Any]]:
    canonical = _canonical_suite_name(suite)
    if canonical == "gsm8k":
        from benchmarks.gsm8k import GSM8KBenchmark

        bench = GSM8KBenchmark(knowledgeverse=knowledgeverse, max_questions=count)
        bench.run_benchmark(use_enriched=True)
        return [
            {
                "question_id": row["question_id"],
                "suite": canonical,
                "question": source["question_text"],
                "answer": row.get("predicted_answer"),
                "expected": source["correct_answer"],
                "correct": bool(row.get("correct", False)),
                "elapsed_s": 0.0,
                "timestamp": time.time(),
            }
            for source, row in zip(bench.questions, bench.results)
        ]
    if canonical == "math":
        from benchmarks.math_competitions import MathCompetitionBenchmark

        bench = MathCompetitionBenchmark(knowledgeverse=knowledgeverse, max_problems=count)
        bench.run_benchmark(use_enriched=True)
        return [
            {
                "question_id": row["problem_id"],
                "suite": canonical,
                "question": source["problem_text"],
                "answer": row.get("predicted_answer"),
                "expected": source["answer"],
                "correct": bool(row.get("correct", False)),
                "elapsed_s": 0.0,
                "timestamp": time.time(),
            }
            for source, row in zip(bench.problems, bench.results)
        ]
    if canonical == "lhe":
        from benchmarks.last_humanity_exam import LastHumanityExamBenchmark

        bench = LastHumanityExamBenchmark(knowledgeverse=knowledgeverse, max_questions=count)
        bench.run_benchmark(use_enriched=True)
        return [
            {
                "question_id": row["id"],
                "suite": canonical,
                "question": source["question_text"],
                "answer": row.get("predicted_answer"),
                "expected": source["correct_answer"],
                "correct": bool(row.get("correct", False)),
                "elapsed_s": 0.0,
                "timestamp": time.time(),
            }
            for source, row in zip(bench.questions, bench.results)
        ]
    if canonical == "mmlu":
        from benchmarks.mmlu import MMLUBenchmark

        bench = MMLUBenchmark(knowledgeverse=knowledgeverse, max_questions=count)
        bench.run_benchmark(use_enriched=True)
        return [
            {
                "question_id": row["id"],
                "suite": canonical,
                "question": source["question_text"],
                "answer": row.get("predicted_answer"),
                "expected": source["correct_answer"],
                "correct": bool(row.get("correct", False)),
                "elapsed_s": 0.0,
                "timestamp": time.time(),
            }
            for source, row in zip(bench.questions, bench.results)
        ]
    from benchmarks.arc_agi_2 import ARCAGI2Benchmark

    bench = ARCAGI2Benchmark(knowledgeverse=knowledgeverse, max_tasks=count)
    bench.run_benchmark(use_enriched=True)
    return [
        {
            "question_id": row["task_id"],
            "suite": canonical,
            "question": f"ARC task {source['id']}",
            "answer": row.get("predicted"),
            "expected": source["test"][0].get("output"),
            "correct": bool(row.get("correct", False)),
            "elapsed_s": 0.0,
            "timestamp": time.time(),
        }
        for source, row in zip(bench.tasks, bench.results)
    ]


def run_health_check(
    suite: str,
    count: int,
    log_path: str | Path,
    *,
    query_fn: HealthQueryFn | None = None,
    knowledgeverse: Any | None = None,
) -> dict[str, Any]:
    """Run a health check and append the resulting log rows."""
    canonical = _canonical_suite_name(suite)
    if query_fn is None:
        results = _run_suite_via_benchmark(canonical, count, knowledgeverse=knowledgeverse)
    else:
        results = []
        for row in load_questions(canonical, count):
            start = time.monotonic()
            response = query_fn(dict(row))
            elapsed = time.monotonic() - start
            if isinstance(response, dict):
                answer = response.get("answer", response.get("result"))
                correct = response.get("correct")
                if correct is None:
                    correct = evaluate_answer(canonical, answer, row["expected"])
            else:
                answer = response
                correct = evaluate_answer(canonical, answer, row["expected"])
            results.append(
                {
                    "question_id": row["id"],
                    "suite": canonical,
                    "question": row["question"],
                    "answer": answer,
                    "expected": row["expected"],
                    "correct": bool(correct),
                    "elapsed_s": round(float(elapsed), 3),
                    "timestamp": time.time(),
                }
            )

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    correct_count = sum(1 for row in results if row["correct"])
    return {
        "suite": canonical,
        "total": len(results),
        "correct": correct_count,
        "score": f"{correct_count}/{len(results)}",
        "log_path": str(path),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="arc, gsm8k, math, lhe, or mmlu")
    parser.add_argument("--count", type=int, default=10, help="Number of questions/tasks to run.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("../Knowledge3D.local/logs/health_log.jsonl"),
        help="JSONL path for health results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_health_check(args.suite, args.count, args.log)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
