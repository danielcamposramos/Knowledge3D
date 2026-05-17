from __future__ import annotations

from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.navigator_specialist import HALTING_WEIGHT_PRIOR_UNIFORM, MEANING_CLASSES


class _FakeRuntime:
    def dispatch_task(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "result": "ok",
            "task_result": {
                "status": "ok",
                "task_snapshot": dict(task),
            },
        }


def test_unknown_class_guard_falls_back_to_factual_recall(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("K3D_BYPASS_GAME_LOOP", raising=False)
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_unknown_guard",
        eager_load_default_galaxies=False,
        bootstrap_foundational_galaxies=False,
        include_runtime_artifacts=False,
        include_runtime_language_enrichment=False,
    )
    monkeypatch.setattr(
        kv.navigator_specialist,
        "emit",
        lambda *args, **kwargs: (
            [1.0 / len(MEANING_CLASSES)] * len(MEANING_CLASSES),
            list(HALTING_WEIGHT_PRIOR_UNIFORM),
        ),
    )
    monkeypatch.setattr(kv, "_get_sovereign_hot_path", lambda: _FakeRuntime())

    result = kv.query(
        "This prompt is intentionally ambiguous.",
        task={"task_id": "unknown_guard_case", "query": "This prompt is intentionally ambiguous."},
    )

    assert result["status"] == "ok"
    assert result["mode"] == "query_tick"
    assert result["meaning_class"] == "FACTUAL_RECALL"
    assert result["low_confidence_routing"] is True
    assert result["task_result"]["meaning_class"] == "FACTUAL_RECALL"
    assert result["task_result"]["low_confidence_routing"] is True
    assert result["task_result"]["task_snapshot"]["meaning_class"] == "FACTUAL_RECALL"
    assert result["task_result"]["task_snapshot"]["low_confidence_routing"] is True
    assert "UNKNOWN" not in str(result)
