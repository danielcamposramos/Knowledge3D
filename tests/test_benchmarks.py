from __future__ import annotations

import json

from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
from benchmarks.math_competitions import MathCompetitionBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_arc_benchmark_empty_vs_enriched(tmp_path):
    dataset_dir = tmp_path / "arc_eval"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
            {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
        ],
        "test": [{"input": [[9, 0], [1, 2]], "output": [[0, 9], [2, 1]]}],
    }
    (dataset_dir / "task_flip.json").write_text(json.dumps(task), encoding="utf-8")

    benchmark = ARCAGI2Benchmark(
        knowledgeverse=Knowledgeverse(storage_root=tmp_path / "kv_arc"),
        dataset_path=dataset_dir,
    )
    empty_result = benchmark.run_benchmark(use_enriched=False)
    enriched_result = benchmark.run_benchmark(use_enriched=True)

    assert empty_result["total_tasks"] == 1
    assert enriched_result["total_tasks"] == 1
    assert enriched_result["accuracy"] >= empty_result["accuracy"]
    assert enriched_result["accuracy"] == 1.0


def test_arc_adapter_preserves_trm_shadow_for_training_collection():
    class _FakeKnowledgeverse:
        def execute_task(self, **_kwargs):
            return {
                "output_grid": [[1]],
                "gpu_execution": True,
                "runtime": "knowledgeverse_gpu_query",
                "solver": "knowledgeverse_gpu_query",
                "program_id": Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID,
                "route": {"specialist": "visual", "galaxy_names": ["Drawing", "Grammar"]},
                "trm_shadow": {
                    "python_galaxies": ["Drawing", "Grammar"],
                    "python_program": Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID,
                    "query_embedding_512": [0.0] * 512,
                    "y_new_vector_512": [0.0] * 512,
                },
            }

    adapter = ArcAgi2Adapter(knowledgeverse=_FakeKnowledgeverse())
    solved = adapter.solve_task(
        {
            "id": "arc_shadow",
            "train": [],
            "test": [{"input": [[1]], "output": [[1]]}],
        }
    )

    assert solved["correct"] is True
    assert solved["task_result"]["trm_shadow"]["python_galaxies"] == ["Drawing", "Grammar"]
    assert solved["trm_shadow"]["python_program"] == Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID


def test_math_benchmark_empty_vs_enriched(tmp_path):
    dataset_dir = tmp_path / "math_competitions"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    amc = [
        {
            "id": "m1",
            "problem_text": "Solve linear equation 2x + 3 = 11.",
            "answer": r"\frac{4}{1}",
        },
        {
            "id": "m2",
            "problem_text": "What is 8 choose 2?",
            "answer": r"\frac{28}{1}",
        },
    ]
    (dataset_dir / "amc_problems.json").write_text(json.dumps(amc), encoding="utf-8")

    benchmark = MathCompetitionBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=tmp_path / "kv_math"),
        dataset_path=dataset_dir,
    )
    empty_result = benchmark.run_benchmark(use_enriched=False)
    enriched_result = benchmark.run_benchmark(use_enriched=True)

    assert empty_result["total"] == 2
    assert enriched_result["total"] == 2
    assert enriched_result["overall_accuracy"] >= empty_result["overall_accuracy"]
    assert enriched_result["overall_accuracy"] > 0.0


def test_lhe_benchmark_empty_vs_enriched(tmp_path):
    dataset_dir = tmp_path / "lhe_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    questions = {
        "questions": [
            {
                "id": "q_math",
                "domain": "math",
                "question_text": "What is 7 * (3 + 2)?",
                "options": ["30", "35", "40", "45"],
                "correct_answer": "35",
            },
            {
                "id": "q_logic",
                "domain": "logic",
                "question_text": "If all A are B and all B are C, which statement is true?",
                "options": ["Some A are not C", "All A are C", "No B are C"],
                "correct_answer": "All A are C",
            },
        ]
    }
    (dataset_dir / "last_humanity_exam.json").write_text(
        json.dumps(questions),
        encoding="utf-8",
    )

    benchmark = LastHumanityExamBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=tmp_path / "kv_lhe"),
        dataset_path=dataset_dir,
    )
    empty_result = benchmark.run_benchmark(use_enriched=False)
    enriched_result = benchmark.run_benchmark(use_enriched=True)

    assert empty_result["total_questions"] == 2
    assert enriched_result["total_questions"] == 2
    assert enriched_result["accuracy"] >= empty_result["accuracy"]
    assert all(row.get("gpu_execution") is True for row in enriched_result["results"])
    assert all(row.get("runtime") == "knowledgeverse_gpu_query" for row in enriched_result["results"])


def test_lhe_benchmark_accepts_direct_json_file_path(tmp_path):
    dataset_file = tmp_path / "last_humanity_exam.json"
    payload = {
        "questions": [
            {
                "id": "q_math",
                "domain": "math",
                "question_text": "What is 2 + 2?",
                "options": ["3", "4", "5"],
                "correct_answer": "4",
            }
        ]
    }
    dataset_file.write_text(json.dumps(payload), encoding="utf-8")

    benchmark = LastHumanityExamBenchmark(dataset_path=dataset_file)

    assert benchmark.synthetic_fallback is False
    assert benchmark.dataset_file == str(dataset_file)
    assert len(benchmark.questions) == 1


def test_lhe_benchmark_uses_gpu_query_path(tmp_path):
    dataset_dir = tmp_path / "lhe_gpu_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    questions = {
        "questions": [
            {
                "id": "q_physics",
                "domain": "physics",
                "question_text": "An object at rest remains at rest unless acted on by which quantity?",
                "options": ["Force", "Mass", "Time", "Temperature"],
                "correct_answer": "Force",
            }
        ]
    }
    (dataset_dir / "last_humanity_exam.json").write_text(
        json.dumps(questions),
        encoding="utf-8",
    )

    benchmark = LastHumanityExamBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=tmp_path / "kv_lhe_gpu"),
        dataset_path=dataset_dir,
        max_questions=1,
    )
    result = benchmark.run_benchmark(use_enriched=True)

    assert result["total_questions"] == 1
    row = result["results"][0]
    assert row["gpu_execution"] is True
    assert row["runtime"] == "knowledgeverse_gpu_query"
