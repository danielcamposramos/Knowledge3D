#!/usr/bin/env python3
"""Execute Phase 1B: prepare corpus, enrich, and ingest Tier 1-3."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from knowledge3d.ingestion.batch_orchestrator import BatchOrchestrator
from knowledge3d.ingestion.corpus_manifest import (
    CorpusEntry,
    CorpusManifest,
    CorpusTier,
    CorpusType,
)
from knowledge3d.ingestion.enrichment_pipeline import EnrichmentPipeline
from knowledge3d.knowledgeverse.stargate import IngestionStargate


@dataclass
class EntrySpec:
    entry_id: str
    name: str
    tier: CorpusTier
    target_galaxies: list[str]
    dependencies: list[str]
    domain: str
    sources: list[str]
    priority: int


def _entry_specs() -> list[EntrySpec]:
    return [
        EntrySpec(
            entry_id="algorithmic_thinking",
            name="Algorithmic Thinking",
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            target_galaxies=["Grammar", "Math", "Reality"],
            dependencies=[],
            domain="logic",
            sources=[
                "docs/RPN_MATHEMATICAL_FOUNDATIONS.md",
                "docs/PHILOSOPHY.md",
                "docs/INSPIRATION.md",
            ],
            priority=1,
        ),
        EntrySpec(
            entry_id="math_foundations",
            name="Mathematics Foundations",
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            target_galaxies=["Math", "Grammar"],
            dependencies=["algorithmic_thinking"],
            domain="math",
            sources=[
                "docs/vocabulary/MATH_CORE_SPECIFICATION.md",
                "docs/RPN_MATHEMATICAL_FOUNDATIONS.md",
            ],
            priority=2,
        ),
        EntrySpec(
            entry_id="logic_reasoning",
            name="Logic and Reasoning",
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            target_galaxies=["Grammar", "Math"],
            dependencies=["algorithmic_thinking"],
            domain="logic",
            sources=[
                "docs/CRANIUM_CORE.md",
                "docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md",
            ],
            priority=3,
        ),
        EntrySpec(
            entry_id="cs_fundamentals",
            name="Computer Science Fundamentals",
            tier=CorpusTier.TIER_1_FOUNDATIONAL,
            target_galaxies=["Grammar", "Reality"],
            dependencies=["algorithmic_thinking"],
            domain="computer_science",
            sources=[
                "docs/ENV_POLICY.md",
                "docs/TRM_INTEGRATION.md",
                "docs/ROADMAP.md",
            ],
            priority=4,
        ),
        EntrySpec(
            entry_id="competition_math",
            name="Competition Math",
            tier=CorpusTier.TIER_2_DOMAIN,
            target_galaxies=["Math", "Grammar"],
            dependencies=["math_foundations", "logic_reasoning"],
            domain="math",
            sources=[
                "docs/tests/LLM_EVAL_PLAN.md",
                "docs/PHASE_2_BIDIRECTIONAL_REASONING.md",
            ],
            priority=5,
        ),
        EntrySpec(
            entry_id="undergraduate_math",
            name="Undergraduate Math",
            tier=CorpusTier.TIER_2_DOMAIN,
            target_galaxies=["Math", "Grammar"],
            dependencies=["math_foundations"],
            domain="math",
            sources=[
                "docs/vocabulary/MATH_CORE_SPECIFICATION.md",
                "docs/Jules_K3D_Whitepaper.md",
            ],
            priority=6,
        ),
        EntrySpec(
            entry_id="geometry_theorems",
            name="Geometry Theorems",
            tier=CorpusTier.TIER_2_DOMAIN,
            target_galaxies=["Drawing", "Math"],
            dependencies=["math_foundations"],
            domain="visual",
            sources=[
                "docs/research/DRAWING_GRAMMAR_SPEC.md",
                "docs/research/Procedural_Vector_Drawing.md",
            ],
            priority=7,
        ),
        EntrySpec(
            entry_id="arc_agi_training",
            name="ARC AGI Training",
            tier=CorpusTier.TIER_2_DOMAIN,
            target_galaxies=["Drawing", "Grammar"],
            dependencies=["geometry_theorems"],
            domain="visual",
            sources=[
                "docs/tests/ARK_AGI_PLAN.md",
                "docs/SCENE_GEN_INSPIRATION.md",
            ],
            priority=8,
        ),
        EntrySpec(
            entry_id="classical_mechanics",
            name="Classical Mechanics",
            tier=CorpusTier.TIER_2_DOMAIN,
            target_galaxies=["Reality", "Math"],
            dependencies=["math_foundations", "cs_fundamentals"],
            domain="physics",
            sources=[
                "docs/Reality_Enabler.md",
                "docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md",
            ],
            priority=9,
        ),
        EntrySpec(
            entry_id="grammar_rules",
            name="Grammar Rules",
            tier=CorpusTier.TIER_2_DOMAIN,
            target_galaxies=["Grammar", "Word"],
            dependencies=["logic_reasoning"],
            domain="logic",
            sources=[
                "docs/research/DRAWING_GRAMMAR_SPEC.md",
                "docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md",
            ],
            priority=10,
        ),
        EntrySpec(
            entry_id="problem_solving_strategies",
            name="Problem Solving Strategies",
            tier=CorpusTier.TIER_3_INTEGRATION,
            target_galaxies=["Grammar", "Math", "Reality", "Drawing"],
            dependencies=["algorithmic_thinking", "competition_math"],
            domain="reasoning",
            sources=[
                "docs/PHILOSOPHY.md",
                "docs/SCIENTIFIC_MANIFESTO_COLLECTIVE_INTELLIGENCE_CHAIN.md",
                "docs/NEXT_STEPS.md",
            ],
            priority=11,
        ),
    ]


def _materialize_entry(spec: EntrySpec, output_root: Path) -> tuple[Path, list[str]]:
    entry_dir = output_root / spec.entry_id
    entry_dir.mkdir(parents=True, exist_ok=True)
    output_file = entry_dir / f"{spec.entry_id}.md"

    missing: list[str] = []
    parts: list[str] = [
        f"# {spec.name}",
        "",
        f"entry_id: {spec.entry_id}",
        f"tier: {spec.tier.value}",
        f"domain: {spec.domain}",
        f"priority: {spec.priority}",
        "",
    ]
    for src in spec.sources:
        src_path = Path(src)
        if not src_path.exists():
            missing.append(src)
            continue
        text = src_path.read_text(encoding="utf-8", errors="ignore")
        parts.append(f"## Source: {src}")
        parts.append("")
        parts.append(text[:120_000])
        parts.append("")
    output_file.write_text("\n".join(parts), encoding="utf-8")
    return output_file, missing


def _build_manifest(materialized: dict[str, Path]) -> CorpusManifest:
    manifest = CorpusManifest(load_defaults=False)
    for spec in _entry_specs():
        manifest.add_entry(
            CorpusEntry(
                id=spec.entry_id,
                name=spec.name,
                path=str(materialized[spec.entry_id]),
                tier=spec.tier,
                corpus_type=CorpusType.MARKDOWN,
                target_galaxies=spec.target_galaxies,
                dependencies=spec.dependencies,
                metadata={
                    "priority": spec.priority,
                    "domain": spec.domain,
                },
            )
        )
    return manifest


def _enrich_manifest_entries(
    manifest: CorpusManifest,
    enrichment: EnrichmentPipeline,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    enrichment_dir = output_root / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for entry in manifest.get_ingestion_order():
        content = Path(entry.path).read_text(encoding="utf-8", errors="ignore")
        enriched = enrichment.enrich_document(content=content, metadata=entry.metadata)
        payload = {
            "entry_id": enriched["entry_id"],
            "patterns": enriched["patterns"],
            "related_concepts": enriched["related_concepts"],
            "embedding_dims": sorted(list(enriched["embeddings"].keys())),
            "embedding_count": int(sum(len(v) for v in enriched["embeddings"].values())),
        }
        (enrichment_dir / f"{entry.id}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        results[entry.id] = payload
    return results


async def _run_ingestion(manifest: CorpusManifest, max_parallel: int) -> dict[str, Any]:
    stargate = IngestionStargate(manifest_version="kv-2026-02-06", ring_size_mb=64)
    orchestrator = BatchOrchestrator(manifest=manifest, stargate=stargate)
    return await orchestrator.ingest_all(max_parallel=max_parallel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Knowledge Preparation Phase 1B")
    parser.add_argument(
        "--output-root",
        default="../Knowledge3D.local/datasets/knowledge_prep_phase1b",
        help="Output root for materialized corpus and reports",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Parallel ingestion workers",
    )
    parser.add_argument(
        "--use-local-models",
        action="store_true",
        help="Enable Ollama-assisted enrichment",
    )
    args = parser.parse_args()

    started = time.time()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    materialized: dict[str, Path] = {}
    missing_sources: dict[str, list[str]] = {}
    for spec in _entry_specs():
        path, missing = _materialize_entry(spec=spec, output_root=output_root)
        materialized[spec.entry_id] = path
        if missing:
            missing_sources[spec.entry_id] = missing

    manifest = _build_manifest(materialized=materialized)
    enrichment = EnrichmentPipeline(use_local_models=args.use_local_models)
    enrichment_results = _enrich_manifest_entries(
        manifest=manifest,
        enrichment=enrichment,
        output_root=output_root,
    )

    ingestion_results = asyncio.run(
        _run_ingestion(manifest=manifest, max_parallel=max(1, args.max_parallel))
    )

    dedup_total = len(manifest.entries)
    dedup_unique = len(enrichment.symlink_registry)
    dedup_ratio = 0.0
    if dedup_total > 0:
        dedup_ratio = 1.0 - (dedup_unique / dedup_total)

    report = {
        "started_at": started,
        "finished_at": time.time(),
        "duration_sec": time.time() - started,
        "output_root": str(output_root),
        "use_local_models": args.use_local_models,
        "manifest_stats": manifest.get_stats(),
        "missing_sources": missing_sources,
        "dedup_unique_entries": dedup_unique,
        "dedup_ratio": dedup_ratio,
        "enrichment_summary": {
            "entry_count": len(enrichment_results),
            "entries": enrichment_results,
        },
        "ingestion": ingestion_results,
        "manifest_order": [entry.id for entry in manifest.get_ingestion_order()],
    }

    report_path = output_root / "knowledge_prep_phase1b_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    main()
