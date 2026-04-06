from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_augment_entries_are_meaning_based_and_route_capable() -> None:
    module = _load_script_module("fundamental_augment_benchmarks_test", "scripts/fundamental_augment_benchmarks.py")
    rows = module._build_question_entries(
        {
            "id": "mmlu_astronomy_0",
            "subject": "astronomy",
            "question_text": "Which planet is known as the Red Planet?",
            "correct_answer": "Mars",
        },
        source_name="mmlu",
        include_ollama_hint=None,
    )

    grammar_entry = dict(rows[0]["entry"])
    general_entry = dict(rows[1]["entry"])

    assert grammar_entry["id"].startswith("question_reasoning_anchor_")
    assert grammar_entry["category"] == "question_reasoning_anchor"
    assert grammar_entry["route_family"] == "QUESTION"
    assert grammar_entry["selection_role"] == "executor"
    assert grammar_entry["layer_id"] == 3
    assert grammar_entry["answer_eligible"] is False
    assert "mmlu" not in grammar_entry["id"].lower()
    assert "mmlu" not in grammar_entry["category"].lower()

    assert general_entry["id"].startswith("general_reasoning_bridge_")
    assert general_entry["category"] == "general_reasoning_bridge"
    assert general_entry["route_family"] == "GENERAL"
    assert general_entry["validator_refs"] == ["general_consistency_validator", "general_answer_validator"]
    assert general_entry["anti_pattern_refs"] == [
        "anti_pattern_missing_evidence_consistency",
        "anti_pattern_generic_language_factual_winner",
    ]
    assert "mmlu" not in general_entry["id"].lower()


def test_ingest_validation_rejects_benchmark_name_leakage_and_missing_route_contract() -> None:
    module = _load_script_module("fundamental_ingest_payloads_test", "scripts/fundamental_ingest_payloads.py")

    valid_entry = module._normalize_route_contract(
        {
            "id": "general_reasoning_bridge_abcd1234",
            "category": "general_reasoning_bridge",
            "metadata": {
                "source": "benchmark_augmentation_mmlu",
                "route_family": "GENERAL",
                "selection_role": "executor",
                "layer_id": 3,
                "answer_eligible": False,
                "route_policy": {"requires_validator": True, "answer_gate": True, "branch_topk": 2},
            },
        }
    )
    assert module._requires_benchmark_route_contract("Reality", valid_entry) is True
    assert module._missing_route_fields(valid_entry) == []
    assert module._benchmark_name_leakage(valid_entry) == []

    leaked_entry = {
        "id": "mmlu_reality_bridge_bad",
        "category": "mmlu_reality_bridge",
        "metadata": {"source": "benchmark_augmentation_mmlu"},
    }
    assert module._benchmark_name_leakage(leaked_entry) == ["id", "category"]

    broken_entry = {
        "id": "general_reasoning_bridge_bad",
        "category": "general_reasoning_bridge",
        "metadata": {"source": "benchmark_augmentation_mmlu"},
    }
    assert set(module._missing_route_fields(broken_entry)) == {
        "route_family",
        "selection_role",
        "layer_id",
        "answer_eligible",
        "route_policy",
    }
