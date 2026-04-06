from __future__ import annotations

from pathlib import Path

from scripts import run_gpu_benchmark as run_gpu_module
from scripts.run_gpu_benchmark import run_gpu_benchmark
from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a) + 1.0e-12)
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b) + 1.0e-12)
    return dot / (norm_a * norm_b)


def test_embed_text_sovereign_is_normalized_and_semantically_coherent():
    similar_a = embed_text_sovereign("what is the capital city of france")
    similar_b = embed_text_sovereign("which city is the capital of france")
    unrelated = embed_text_sovereign("move left across the blue grid cell")

    assert len(similar_a) == 32
    assert any(abs(value) > 1.0e-8 for value in similar_a)
    assert abs(sum(value * value for value in similar_a) - 1.0) < 1.0e-5
    assert _cosine(similar_a, similar_b) > _cosine(similar_a, unrelated)


def test_run_gpu_benchmark_mmlu_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_gpu_module,
        "_mmlu_tasks",
        lambda kv, count: (
            [
                {
                    "type": "MMLU_TASK",
                    "query_embedding": [1.0] + ([0.0] * 31),
                    "option_embeddings": [[1.0] + ([0.0] * 31)] + ([[0.0] * 32] * 3),
                    "subject": "unit_test",
                    "domain_hint": "unit_test",
                }
                for _ in range(count)
            ],
            [
                {
                    "id": f"mmlu_{index}",
                    "suite": "mmlu",
                    "mode": "multiple_choice",
                    "options": ["a", "b", "c", "d"],
                    "correct_index": 0,
                    "correct_answer": "a",
                    "subject": "unit_test",
                }
                for index in range(count)
            ],
        ),
    )
    monkeypatch.setattr(
        run_gpu_module,
        "_dispatch_tasks",
        lambda tasks, dispatcher, brain=None, galaxy_table=None: [
            {
                "answer_index": 0,
                "confidence": 0.99,
                "convergence_signal": 1,
                "iterations_used": 5,
                "answer_text_hash": 0,
                "goal_progress": 0.0,
            }
            for task in tasks
        ],
    )

    summary = run_gpu_benchmark(
        suite="mmlu",
        count=10,
        storage_root=tmp_path / "gpu_bench_smoke",
        log_path=tmp_path / "gpu_bench_smoke.jsonl",
    )

    assert summary["suite"] == "mmlu"
    assert summary["total"] == 10
    assert summary["correct"] == 10
    assert summary["accuracy"] == 1.0


def test_run_gpu_benchmark_supports_math_and_arc2(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_gpu_module,
        "_math_tasks",
        lambda kv, count: (
            [{"type": "GSM8K_TASK", "query_embedding": [0.0] * 32, "option_embeddings": [], "subject": "gsm8k", "domain_hint": "word_problem"}],
            [{"id": "gsm8k_0", "suite": "gsm8k", "mode": "open_ended_hash", "correct_answer": "12"}],
        ),
    )
    monkeypatch.setattr(
        run_gpu_module,
        "_arc2_tasks",
        lambda kv, count: (
            [{"type": "ARC_TASK", "query_embedding": [0.0] * 32, "option_embeddings": [], "subject": "arc_agi_2", "domain_hint": "visual_reasoning"}],
            [{"id": "arc2_0", "suite": "arc2", "mode": "grid_generation_tbd", "input_grid": [[1]], "expected_grid": [[1]]}],
        ),
    )
    monkeypatch.setattr(
        run_gpu_module,
        "_dispatch_tasks",
        lambda tasks, dispatcher, brain=None, galaxy_table=None: [
            {"answer_index": 0, "confidence": 0.4, "convergence_signal": 1, "iterations_used": 5, "answer_text_hash": 0}
            for _ in tasks
        ],
    )

    gsm8k = run_gpu_benchmark(
        suite="gsm8k",
        count=1,
        storage_root=tmp_path / "storage",
        log_path=None,
    )
    arc2 = run_gpu_benchmark(
        suite="arc2",
        count=1,
        storage_root=tmp_path / "storage",
        log_path=None,
    )

    assert gsm8k["suite"] == "gsm8k"
    assert gsm8k["total"] == 1
    assert gsm8k["correct"] == 0
    assert gsm8k["results"][0]["mode"] == "open_ended_hash"

    assert arc2["suite"] == "arc2"
    assert arc2["total"] == 1
    assert arc2["scoring_tbd"] is True
    assert arc2["results"][0]["scoring_tbd"] is True


def test_benchmark_runners_drop_transformer_embedder_references():
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in ("scripts/run_gpu_benchmark.py", "scripts/run_full_benchmark.py"):
        content = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "_embed_query_gpu" not in content
        assert "sentence_transform" not in content.lower()
        assert "embed_sentence" not in content.lower()
