from __future__ import annotations

import json
from pathlib import Path

from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.mmlu import MMLUBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_math_benchmark_uses_gpu_math_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "GSM8K" / "grade_school_math" / "data"
    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "test.jsonl").write_text(
        json.dumps(
            {
                "question": (
                    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
                    "bakes muffins for her friends every day with four. She sells the remainder at the "
                    "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
                    "every day at the farmers' market?"
                ),
                "answer": "Janet sells 9 eggs for $2 each.\\n#### 18",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    benchmark = GSM8KBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=tmp_path / "kv_gsm8k"),
        dataset_path=dataset_path / "test.jsonl",
        max_questions=1,
    )

    result = benchmark.run_benchmark(use_enriched=True)

    assert result["total_questions"] == 1
    assert result["results"][0]["gpu_execution"] is True
    assert result["results"][0]["runtime"] == "knowledgeverse_gpu_query"


def test_mmlu_benchmark_uses_gpu_chat_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MMLU" / "data" / "test"
    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "college_physics_test.csv").write_text(
        (
            "An object at rest remains at rest unless acted on by which quantity?,"
            "Force,Mass,Time,Temperature,A\n"
        ),
        encoding="utf-8",
    )

    benchmark = MMLUBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=tmp_path / "kv_mmlu"),
        dataset_path=dataset_path.parent,
        subjects=["college_physics"],
        max_questions=1,
    )

    result = benchmark.run_benchmark(use_enriched=True)

    assert result["total_questions"] == 1
    assert result["results"][0]["gpu_execution"] is True
    assert result["results"][0]["runtime"] == "knowledgeverse_gpu_query"


def test_mmlu_benchmark_supports_resume_offset_and_row_callback(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MMLU" / "data" / "test"
    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "college_physics_test.csv").write_text(
        (
            "An object at rest remains at rest unless acted on by which quantity?,"
            "Force,Mass,Time,Temperature,A\n"
            "What is 7 * (3 + 2)?,35,30,42,28,A\n"
        ),
        encoding="utf-8",
    )

    class _FakeKnowledgeverse:
        def execute_task(self, task, **_kwargs):
            return {
                "response": task["expected_answer"],
                "gpu_execution": True,
                "runtime": "knowledgeverse_gpu_query",
                "solver": "knowledgeverse_gpu_query",
                "program_id": Knowledgeverse.GPU_CHAT_REASONING_PROGRAM_ID,
                "route": {"specialist": "chat", "galaxy_names": ["Reality", "Grammar"]},
            }

    captured: list[tuple[str, str]] = []
    benchmark = MMLUBenchmark(
        knowledgeverse=_FakeKnowledgeverse(),
        dataset_path=dataset_path.parent,
        max_questions=2,
    )

    result = benchmark.run_benchmark(
        use_enriched=True,
        start_index=1,
        initial_correct=1,
        row_cb=lambda source, row: captured.append((source["id"], row["id"])),
    )

    assert result["total_questions"] == 2
    assert result["correct"] == 2
    assert len(captured) == 1
    assert captured[0][0] == "mmlu_college_physics_1"
