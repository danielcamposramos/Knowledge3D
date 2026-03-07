from __future__ import annotations

import json

from benchmarks.deterministic_foundation import DeterministicFoundationBenchmark
from benchmarks.tasks import (
    generate_arithmetic_tasks,
    generate_compositional_tasks,
    generate_geometric_tasks,
    generate_pattern_tasks,
    generate_rpn_tasks,
)
from knowledge3d.knowledgeverse.foundational_operations_bootstrap import (
    populate_foundational_operations,
)
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from scripts.train_deterministic_foundation import train_deterministic_foundation


def test_task_generators_counts():
    assert len(generate_geometric_tasks(7)) == 7
    assert len(generate_arithmetic_tasks(9)) == 9
    assert len(generate_pattern_tasks(11)) == 11
    assert len(generate_compositional_tasks(13)) == 13
    assert len(generate_rpn_tasks(15)) == 15


def test_foundational_bootstrap_is_idempotent(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    first = populate_foundational_operations(kv.galaxy_manager)
    second = populate_foundational_operations(kv.galaxy_manager)

    # Knowledgeverse now boots foundational deterministic operations eagerly.
    assert first["total_inserted"] >= 0
    assert second["total_inserted"] == 0


def test_foundational_number_word_symlinks_exist(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_num")
    populate_foundational_operations(kv.galaxy_manager)

    number_entries = kv.galaxy_manager.get_galaxy("Number").entries
    word_entries = kv.galaxy_manager.get_galaxy("Word").entries

    num_five = next(entry for entry in number_entries if entry.get("id") == "num_5")
    word_five = next(entry for entry in word_entries if entry.get("id") == "word_five")

    assert num_five["metadata"]["word_ref"] == "word_five"
    assert word_five["metadata"]["number_ref"] == "num_5"
    assert word_five["metadata"]["is_numeric_word"] is True


def test_deterministic_benchmark_smoke(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    populate_foundational_operations(kv.galaxy_manager)

    benchmark = DeterministicFoundationBenchmark(tasks_per_category=5, seed=7)
    result = benchmark.run_benchmark(kv, iteration=0, log_events=False)

    assert result["overall"]["total"] == 25
    assert 0.0 <= result["overall"]["accuracy"] <= 1.0
    assert "geometric_transforms" in result
    assert "grid_arithmetic" in result
    assert "pattern_completion" in result
    assert "compositional" in result
    assert "symbolic_rpn" in result


def test_deterministic_benchmark_with_system_literacy(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_sys")
    populate_foundational_operations(kv.galaxy_manager)
    benchmark = DeterministicFoundationBenchmark(
        tasks_per_category=4,
        seed=21,
        stage="A",
        include_system_literacy=True,
    )
    result = benchmark.run_benchmark(kv, iteration=0, log_events=False)
    assert "system_literacy" in result
    assert result["system_literacy"]["total"] == 4
    assert 0.0 <= result["system_literacy"]["accuracy"] <= 1.0


def test_stage_b_uses_alias_only_queries(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    populate_foundational_operations(kv.galaxy_manager)

    benchmark = DeterministicFoundationBenchmark(tasks_per_category=5, seed=8, stage="B")
    sample = benchmark.tasks["geometric_transforms"][0]
    assert "operation" not in sample
    assert "target_operation" in sample
    assert isinstance(sample["query"], str)
    assert sample["query"]

    result = benchmark.run_benchmark(kv, iteration=0, log_events=False)
    assert result["stage"] == "B"
    assert result["overall"]["total"] == 25
    assert 0.0 <= result["overall"]["accuracy"] <= 1.0


def test_train_deterministic_foundation_one_iteration(tmp_path):
    output_dir = tmp_path / "results"
    payload = train_deterministic_foundation(
        iterations=1,
        tasks_per_category=5,
        seed=42,
        storage_root=tmp_path / "kv",
        output_dir=output_dir,
    )
    history_path = output_dir / "training_history.json"
    assert history_path.exists()
    loaded = json.loads(history_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["iterations"] == 1
    assert loaded["history"][0]["results"]["overall"]["total"] == 25
    assert payload["summary"]["iterations"] == 1


def test_train_deterministic_foundation_stage_advancement(tmp_path):
    output_dir = tmp_path / "results_adv"
    payload = train_deterministic_foundation(
        iterations=4,
        tasks_per_category=5,
        seed=99,
        storage_root=tmp_path / "kv_adv",
        output_dir=output_dir,
    )
    history = payload["history"]
    assert len(history) == 4
    assert history[0]["stage"] == "A"
    advanced_events = [row["stage_advanced"] for row in history if row.get("stage_advanced")]
    assert advanced_events, "expected at least one stage promotion event"
    assert payload["final_stage"] in {"B", "C", "D"}


def test_train_deterministic_foundation_with_quality_and_system_literacy(tmp_path):
    output_dir = tmp_path / "results_quality"
    payload = train_deterministic_foundation(
        iterations=1,
        tasks_per_category=3,
        seed=7,
        storage_root=tmp_path / "kv_quality",
        output_dir=output_dir,
        enable_ternary_quality=True,
        include_system_literacy=True,
    )
    assert payload["ternary_quality_enabled"] is True
    first = payload["history"][0]
    assert first["quality_memory"] is not None
    assert first["teacher_feedback"]["pool_id"].startswith("ternary_pool_")


def test_train_deterministic_foundation_contrastive_mode(tmp_path):
    output_dir = tmp_path / "results_contrastive"
    payload = train_deterministic_foundation(
        iterations=2,
        tasks_per_category=2,
        seed=17,
        storage_root=tmp_path / "kv_contrastive",
        output_dir=output_dir,
        enable_transfer_gates=True,
        transfer_probe_arc_tasks=1,
        enable_contrastive_learning=True,
    )
    assert payload["contrastive_learning_enabled"] is True
    assert len(payload["history"]) == 2
    for row in payload["history"]:
        assert "pool_drift" in row
        assert "contrastive_feedback" in row
