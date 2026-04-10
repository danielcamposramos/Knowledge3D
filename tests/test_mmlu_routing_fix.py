from __future__ import annotations

from pathlib import Path

from benchmarks.mmlu import MMLUBenchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _kv_stub() -> Knowledgeverse:
    return Knowledgeverse.__new__(Knowledgeverse)


def test_infer_query_mode_trusts_typed_question_envelope() -> None:
    kv = _kv_stub()
    mode = Knowledgeverse._infer_query_mode(
        kv,
        task={"type": "QUESTION_TASK", "surface_kind": "QUESTION", "question": "Who is Marie Curie?"},
        route=None,
        query_text="Who is Marie Curie?",
        options=None,
    )
    assert mode == "QUESTION_TASK"


def test_infer_query_mode_recognizes_choices_field_for_question_surface() -> None:
    kv = _kv_stub()
    mode = Knowledgeverse._infer_query_mode(
        kv,
        task={"surface_kind": "QUESTION", "choices": ["A", "B", "C", "D"]},
        route=None,
        query_text="Which choice is correct?",
        options=None,
    )
    assert mode == "MMLU_TASK"


def test_infer_query_mode_question_surface_without_choices_becomes_lhe() -> None:
    kv = _kv_stub()
    mode = Knowledgeverse._infer_query_mode(
        kv,
        task={"surface_kind": "QUESTION", "question": "Explain the causal chain."},
        route=None,
        query_text="Explain the causal chain.",
        options=None,
    )
    assert mode == "LHE_TASK"


def test_infer_query_mode_options_field_for_question_surface_becomes_mmlu() -> None:
    kv = _kv_stub()
    mode = Knowledgeverse._infer_query_mode(
        kv,
        task={"surface_kind": "QUESTION", "options": ["A", "B", "C", "D"]},
        route=None,
        query_text="Pick the right answer.",
        options=None,
    )
    assert mode == "MMLU_TASK"


def test_mmlu_benchmark_run_routes_no_rows_to_general(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MMLU" / "data" / "test"
    dataset_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(20):
        rows.append(
            "Which element has atomic number 6?,Hydrogen,Carbon,Oxygen,Nitrogen,B\n"
        )
    (dataset_path / "college_chemistry_test.csv").write_text("".join(rows), encoding="utf-8")

    class _RouteOnlyKnowledgeverse:
        def __init__(self) -> None:
            self._router = Knowledgeverse.__new__(Knowledgeverse)

        def execute_task(self, task, **_kwargs):
            query_text = " ".join(
                str(task.get(key, "")).strip()
                for key in ("query", "question", "prompt")
                if str(task.get(key, "")).strip()
            )
            task_type = Knowledgeverse._infer_query_mode(
                self._router,
                task=task,
                route={"specialist": "chat"},
                query_text=query_text,
                options=list(task.get("options") or []),
            )
            return {
                "response": task["expected_answer"],
                "gpu_execution": True,
                "runtime": "knowledgeverse_gpu_query",
                "solver": "knowledgeverse_gpu_query",
                "program_id": Knowledgeverse.GPU_CHAT_REASONING_PROGRAM_ID,
                "route": {
                    "specialist": "chat",
                    "route_family": task_type.replace("_TASK", ""),
                    "task_type": task_type,
                    "galaxy_names": ["Reality", "Grammar", "Word", "Character"],
                },
            }

    benchmark = MMLUBenchmark(
        knowledgeverse=_RouteOnlyKnowledgeverse(),
        dataset_path=dataset_path.parent,
        subjects=["college_chemistry"],
        max_questions=20,
    )

    result = benchmark.run_benchmark(use_enriched=False)

    assert result["total_questions"] == 20
    assert all(str((row.get("route") or {}).get("route_family") or "") != "GENERAL" for row in result["results"])
