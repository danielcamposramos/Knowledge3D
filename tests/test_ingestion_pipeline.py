from __future__ import annotations

import asyncio
from pathlib import Path

from knowledge3d.ingestion.batch_orchestrator import BatchOrchestrator
from knowledge3d.ingestion.corpus_manifest import (
    CorpusEntry,
    CorpusManifest,
    CorpusTier,
    CorpusType,
)
from knowledge3d.ingestion.enrichment_pipeline import EnrichmentPipeline
from knowledge3d.knowledgeverse.stargate import IngestionStargate


def test_corpus_manifest_integrity(tmp_path: Path):
    manifest = CorpusManifest(load_defaults=False)
    p1 = tmp_path / "algorithmic_thinking.pdf"
    p2 = tmp_path / "math_foundations.pdf"
    p3 = tmp_path / "logic_reasoning.pdf"
    p1.write_text("algorithms", encoding="utf-8")
    p2.write_text("math", encoding="utf-8")
    p3.write_text("logic", encoding="utf-8")

    manifest.add_entry(
        CorpusEntry(
            id="algorithmic_thinking",
            name="Algorithmic Thinking",
            path=p1.name,
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Grammar"],
        )
    )
    manifest.add_entry(
        CorpusEntry(
            id="math_foundations",
            name="Math Foundations",
            path=p2.name,
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Math"],
            dependencies=["algorithmic_thinking"],
        )
    )
    manifest.add_entry(
        CorpusEntry(
            id="logic_reasoning",
            name="Logic Reasoning",
            path=p3.name,
            tier=CorpusTier.TIER_2_DOMAIN,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Grammar"],
            dependencies=["algorithmic_thinking"],
        )
    )

    ok, missing = manifest.validate_paths(root=tmp_path)
    assert ok
    assert missing == []

    order = manifest.get_ingestion_order()
    by_id = {entry.id: idx for idx, entry in enumerate(order)}
    assert by_id["algorithmic_thinking"] < by_id["math_foundations"]
    assert by_id["algorithmic_thinking"] < by_id["logic_reasoning"]


def test_batch_ingestion(tmp_path: Path):
    manifest = CorpusManifest(load_defaults=False)
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("A", encoding="utf-8")
    p2.write_text("B", encoding="utf-8")

    manifest.add_entry(
        CorpusEntry(
            id="a",
            name="A",
            path=str(p1),
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Grammar"],
        )
    )
    manifest.add_entry(
        CorpusEntry(
            id="b",
            name="B",
            path=str(p2),
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Math"],
            dependencies=["a"],
        )
    )

    stargate = IngestionStargate(manifest_version="kv-test", ring_size_mb=4)
    orchestrator = BatchOrchestrator(manifest=manifest, stargate=stargate)
    results = asyncio.run(orchestrator.ingest_tier(tier=1, max_parallel=2))

    assert "error" not in results["a"]
    assert "error" not in results["b"]
    assert manifest.entries["a"].ingested
    assert manifest.entries["b"].ingested
    assert len(stargate.completed_jobs) == 2


def test_enrichment_symlinks():
    pipeline = EnrichmentPipeline(use_local_models=False)

    content = "The derivative of x^2 is 2x"
    id1 = pipeline.find_or_create_symlink(content)
    id2 = pipeline.find_or_create_symlink(content)
    assert id1 == id2

    content2 = "The integral of 2x is x^2 + C"
    id3 = pipeline.find_or_create_symlink(content2)
    assert id3 != id1

    enriched = pipeline.enrich_document(
        content=content,
        metadata={"domain": "math"},
    )
    assert enriched["entry_id"] == id1
    assert set(enriched["embeddings"].keys()) == {64, 128, 512, 2048}
    assert len(enriched["patterns"]) >= 1


def test_end_to_end_pdf_to_galaxy(tmp_path: Path):
    pdf_like = tmp_path / "algorithmic_thinking.pdf"
    pdf_like.write_text(
        "If then logic. Derivative and integral. Rotate and reflect.",
        encoding="utf-8",
    )
    manifest = CorpusManifest(load_defaults=False)
    manifest.add_entry(
        CorpusEntry(
            id="algorithmic_thinking",
            name="Algorithmic Thinking",
            path=str(pdf_like),
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            corpus_type=CorpusType.PDF,
            target_galaxies=["Grammar", "Math", "Drawing"],
            metadata={"domain": "math"},
        )
    )
    stargate = IngestionStargate(manifest_version="kv-test", ring_size_mb=4)
    orchestrator = BatchOrchestrator(manifest=manifest, stargate=stargate)
    pipeline = EnrichmentPipeline(use_local_models=False)

    ingest_result = asyncio.run(orchestrator.ingest_all(max_parallel=1))
    assert ingest_result["stats_after"]["ingested_count"] == 1
    assert ingest_result["stats_after"]["pending_count"] == 0

    enriched = pipeline.enrich_document(
        content=pdf_like.read_text(encoding="utf-8"),
        metadata={"domain": "math", "source": str(pdf_like)},
    )
    assert enriched["entry_id"].startswith("entry_")
    assert len(enriched["embeddings"][2048]) == 2048
    assert len(enriched["patterns"]) >= 1
