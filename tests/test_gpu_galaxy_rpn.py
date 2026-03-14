from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _unit_embedding(axis: int) -> list[float]:
    values = [0.0] * 16
    values[axis] = 1.0
    return values


@pytest.mark.cuda
def test_gpu_rpn_can_load_and_scan_galaxy_entries(tmp_path) -> None:
    try:
        kv = Knowledgeverse(
            storage_root=tmp_path / "kv_gpu_runtime",
            eager_load_default_galaxies=False,
            bootstrap_foundational_galaxies=False,
            include_runtime_artifacts=False,
            include_runtime_language_enrichment=False,
        )
        kv.galaxy_manager.add_entry(
            "Math",
            {
                "id": "math_top_match",
                "name": "Top Match",
                "domain": "math",
                "category": "formal_result",
                "metadata": {
                    "confidence": 0.95,
                    "embedding16": _unit_embedding(0),
                },
            },
        )
        kv.galaxy_manager.add_entry(
            "Math",
            {
                "id": "math_off_axis",
                "name": "Off Axis",
                "domain": "math",
                "category": "formal_result",
                "metadata": {
                    "confidence": 0.40,
                    "embedding16": _unit_embedding(1),
                },
            },
        )
        binding = kv.bind_gpu_galaxy_runtime(galaxy_names=["Math"], force=True)
        engine = kv.get_gpu_reasoning_engine()
    except RuntimeError as exc:
        if "Sovereign loader error" in str(exc) or "GPU path failed" in str(exc):
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        raise

    assert binding["entry_count"] == 2
    assert kv.metrics.gpu_galaxy_entries == 2

    instance_id = engine.store_embedding(embedding=_unit_embedding(0), instance_id=0)

    confidence = engine.evaluate("0 load_galaxy drop drop drop", instance_id=instance_id)
    similarity = engine.evaluate("0 galaxy_similarity", instance_id=instance_id)
    best_index = engine.evaluate("1 galaxy_scan drop", instance_id=instance_id)

    assert confidence == pytest.approx(0.95, abs=1e-5)
    assert similarity == pytest.approx(1.0, abs=1e-5)
    assert best_index == pytest.approx(0.0, abs=1e-5)


@pytest.mark.cuda
def test_knowledgeverse_query_runs_gpu_factual_lookup_with_thinking_trace(tmp_path) -> None:
    try:
        kv = Knowledgeverse(
            storage_root=tmp_path / "kv_gpu_query",
            eager_load_default_galaxies=False,
            include_runtime_artifacts=False,
            include_runtime_language_enrichment=False,
        )
        result = kv.query("An object at rest remains at rest unless acted on by which quantity?")
    except RuntimeError as exc:
        if "Sovereign loader error" in str(exc) or "GPU path failed" in str(exc):
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        raise

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["program_id"] == "reasoning_factual_lookup_top1"
    assert result["response"] == "Force"
    assert "thinking_trace" in result
    assert result["thinking_trace"]
    assert result["thinking_xml"].startswith("<thinking>\n")


@pytest.mark.cuda
def test_gpu_runtime_binding_includes_books_v5_artifacts(tmp_path) -> None:
    books_root = Path("/K3D/Knowledge3D.local/galaxies/books_v5_clean2")
    if not books_root.exists():
        pytest.skip("books_v5_clean2 runtime corpus is not available")
    try:
        kv = Knowledgeverse(
            storage_root=tmp_path / "kv_gpu_books_runtime",
            eager_load_default_galaxies=False,
        )
        binding = kv.bind_gpu_galaxy_runtime(
            galaxy_names=["Math", "Reality", "Word", "Grammar"],
            force=True,
        )
    except RuntimeError as exc:
        if "Sovereign loader error" in str(exc) or "GPU path failed" in str(exc):
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        raise

    assert binding["entry_count"] >= 30_000
    assert binding["runtime_artifact_entries"] >= 30_000
    assert kv.metrics.gpu_runtime_artifact_entries >= 30_000
