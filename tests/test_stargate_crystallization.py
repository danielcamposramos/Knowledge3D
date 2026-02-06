from __future__ import annotations

from pathlib import Path

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.knowledgeverse.stargate import IngestionStargate


def test_stargate_returns_real_crystallization_counts(tmp_path: Path):
    source = tmp_path / "algorithmic_thinking.md"
    source.write_text(
        "If then logic. Derivative of x^2 is 2x. Integral of x is x^2/2.",
        encoding="utf-8",
    )

    galaxy_manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    stargate = IngestionStargate(
        manifest_version="kv-test",
        ring_size_mb=8,
        storage_root=tmp_path / "stargate_jobs",
        galaxy_manager=galaxy_manager,
        use_local_models=False,
    )

    job_id = stargate.submit_ingestion_job(
        data_path=str(source),
        data_type="markdown",
        target_galaxies=["Math", "Grammar"],
        metadata={"domain": "math"},
    )
    result = stargate.wait_for_job(job_id=job_id, timeout=2.0)

    assert result["status"] == "completed"
    assert result["galaxy_entries_created"] >= 1
    assert result["rpn_programs_created"] >= 1
    assert result["embeddings_stored"] > 0
    assert "embedding_count" not in result

    assert len(galaxy_manager.get_galaxy("Math").entries) >= 1
    assert (tmp_path / "stargate_jobs" / job_id / "enriched.json").exists()
    assert (tmp_path / "stargate_jobs" / job_id / "crystallization.json").exists()


def test_stargate_missing_source_completes_without_synthetic_placeholder(tmp_path: Path):
    stargate = IngestionStargate(
        manifest_version="kv-test",
        ring_size_mb=4,
        storage_root=tmp_path / "stargate_jobs",
        galaxy_manager=GalaxyManager(storage_root=tmp_path / "galaxies"),
        use_local_models=False,
    )

    job_id = stargate.submit_ingestion_job(
        data_path=str(tmp_path / "missing_file.pdf"),
        data_type="pdf",
        target_galaxies=["Grammar"],
        metadata={"domain": "logic"},
    )
    result = stargate.wait_for_job(job_id=job_id, timeout=2.0)

    assert result["status"] == "completed"
    assert result["embeddings_stored"] > 0
    assert "embedding_count" not in result
