"""Enhanced PDF ingestion module with optional OCR extras.

The canonical proceduralizer path only needs lightweight transport modules such
as ``ollama_manager``. Keep OCR-heavy imports optional so ingestion scripts can
run even when the local environment does not include PDF OCR dependencies.
"""

from __future__ import annotations

from typing import Any

GPUPDFOCRProcessor = None
OllamaSecondPassEnricher = None
PDFKnowledgePipeline = None
ModularPDFProcessor = None
enhanced_pdf_ingest_main = None
ImportedWarpScene = None
import_warp_model = None
import_warp_modelbuilder = None
load_warp_scene_from_file = None

try:
    from knowledge3d.ingestion.ocr_gpu_processor import (
        GPUPDFOCRProcessor,
        OllamaSecondPassEnricher,
    )

    from knowledge3d.ingestion.pdf_ocr_pipeline import (
        PDFKnowledgePipeline,
        ModularPDFProcessor,
    )

    from knowledge3d.ingestion.enhanced_pdf_ingest import (
        main as enhanced_pdf_ingest_main,
    )
except ModuleNotFoundError:
    # Optional OCR/pdfium stack is not required for the proceduralizer-driven
    # PDF ingestion path used in the current base-knowledge restart.
    pass

from knowledge3d.ingestion.warp_importer import (
    ImportedWarpScene,
    import_warp_model,
    import_warp_modelbuilder,
    load_warp_scene_from_file,
)
from knowledge3d.cranium.sovereign_physics_bootstrap import (
    build_default_gravity_force_law,
    build_default_sleep_policy,
    build_physical_constant_stars,
    build_physics_material_stars,
)


def _physics_bootstrap_rows() -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for star in build_physical_constant_stars():
        rows.append(
            (
                "Reality",
                {
                    "id": str(star["star_id"]),
                    "name": str(star.get("surface_forms", {}).get("en") or star["star_id"]),
                    "galaxy": "Reality",
                    "domain": "reality",
                    "category": "physical_constant",
                    "layer": 2,
                    "content": str(star.get("symbol") or star["star_id"]),
                    "summary": str(star.get("surface_forms", {}).get("en") or star["star_id"]),
                    "description": str(star.get("si_units") or ""),
                    "rpn_program": "",
                    "metadata": dict(star),
                },
            )
        )
    for star in build_physics_material_stars():
        rows.append(
            (
                "Reality",
                {
                    "id": str(star["star_id"]),
                    "name": str(star["star_id"]).replace("physics_material_", "").replace("_", " "),
                    "galaxy": "Reality",
                    "domain": "reality",
                    "category": "physical_material",
                    "layer": 2,
                    "content": str(star["star_id"]),
                    "summary": str(star["star_id"]).replace("physics_material_", "").replace("_", " "),
                    "description": f"density={star['density']} restitution={star['restitution']}",
                    "rpn_program": "",
                    "metadata": dict(star),
                },
            )
        )
    force_law = build_default_gravity_force_law()
    rows.append(
        (
            "Grammar",
            {
                "id": str(force_law["star_id"]),
                "name": "default gravity force law",
                "galaxy": "Grammar",
                "domain": "grammar",
                "category": "force_law",
                "layer": 3,
                "content": str(force_law.get("physics_rpn_addr") or ""),
                "summary": str(force_law.get("summary") or ""),
                "description": str(force_law.get("summary") or ""),
                "rpn_program": str(force_law.get("physics_rpn_addr") or ""),
                "metadata": dict(force_law),
            },
        )
    )
    sleep_policy = build_default_sleep_policy()
    rows.append(
        (
            "Grammar",
            {
                "id": str(sleep_policy["star_id"]),
                "name": "default physics sleep policy",
                "galaxy": "Grammar",
                "domain": "grammar",
                "category": "meta_rule",
                "layer": 4,
                "content": str(sleep_policy.get("strategy_rpn_addr") or ""),
                "summary": "Default sleep/wake policy for sovereign rigid-body islands.",
                "description": "Layer-4 sleep/wake meta-rule for sovereign rigid-body execution.",
                "rpn_program": str(sleep_policy.get("strategy_rpn_addr") or ""),
                "metadata": dict(sleep_policy),
            },
        )
    )
    return rows


def ingest_physics_bootstrap(galaxy_manager: Any) -> int:
    """Ingest the foundational sovereign-physics stars once via the live manager."""
    count = 0
    rows = _physics_bootstrap_rows()
    bulk_disk_sync = getattr(galaxy_manager, "bulk_disk_sync", None)
    if callable(bulk_disk_sync):
        context = bulk_disk_sync()
    else:
        from contextlib import nullcontext

        context = nullcontext()
    with context:
        for galaxy_name, entry in rows:
            galaxy_manager.upsert_entry(galaxy_name, entry)
            count += 1
    return count


def ingest_entity_bootstrap(galaxy_manager: Any) -> int:
    """Ingest foundational entity stars through the canonical meaning-star path."""
    from knowledge3d.cranium.sovereign_entity_bootstrap import build_entity_stars

    count = 0
    bulk_disk_sync = getattr(galaxy_manager, "bulk_disk_sync", None)
    if callable(bulk_disk_sync):
        context = bulk_disk_sync()
    else:
        from contextlib import nullcontext

        context = nullcontext()
    with context:
        for star in build_entity_stars():
            galaxy_manager.store_meaning_star(
                "Reality",
                star,
                category=str(star.meaning_class or "entity"),
                metadata={"bootstrap": "sovereign_entity_v1"},
            )
            count += 1
    return count


def ingest_cas_grammar(galaxy_manager: Any) -> int:
    """Ingest foundational CAS transformation rules into the Grammar galaxy."""
    from knowledge3d.cranium.cas_grammar_bootstrap import build_cas_rule_stars

    count = 0
    bulk_disk_sync = getattr(galaxy_manager, "bulk_disk_sync", None)
    if callable(bulk_disk_sync):
        context = bulk_disk_sync()
    else:
        from contextlib import nullcontext

        context = nullcontext()
    with context:
        for star in build_cas_rule_stars():
            galaxy_manager.store_meaning_star(
                "Grammar",
                star,
                category=str(star.meaning_class or "cas_rule"),
                metadata={"bootstrap": "sovereign_cas_grammar_v1"},
            )
            count += 1
    return count


def ingest_sas_bootstrap(galaxy_manager: Any | None = None) -> tuple[list[float], list[int], list[Any]]:
    """Bootstrap the SAS layer: symbol table plus Grammar-galaxy SAS rules."""
    from knowledge3d.cranium.sas_grammar_bootstrap import build_sas_rule_stars
    from knowledge3d.cranium.sas_symbol_bootstrap import build_symbol_table

    values, star_ids = build_symbol_table(galaxy_manager)
    stars = build_sas_rule_stars()
    if galaxy_manager is not None:
        bulk_disk_sync = getattr(galaxy_manager, "bulk_disk_sync", None)
        if callable(bulk_disk_sync):
            context = bulk_disk_sync()
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            for star in stars:
                galaxy_manager.store_meaning_star(
                    "Grammar",
                    star,
                    category=str(star.meaning_class or "sas_rule"),
                    metadata={"bootstrap": "sovereign_sas_v1"},
                )
    return values, star_ids, stars


def ingest_router_cartographer_bootstrap(galaxy_manager: Any | None = None) -> list[Any]:
    """Bootstrap Router Cartographer stars into the Grammar galaxy."""
    from knowledge3d.cranium.router_cartographer_bootstrap import build_router_cartographer_stars

    stars = build_router_cartographer_stars()
    if galaxy_manager is not None:
        bulk_disk_sync = getattr(galaxy_manager, "bulk_disk_sync", None)
        if callable(bulk_disk_sync):
            context = bulk_disk_sync()
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            for star in stars:
                galaxy_manager.store_meaning_star(
                    "Grammar",
                    star,
                    category=str(star.meaning_class or "routing_signal"),
                    metadata={"bootstrap": "router_cartographer_v1"},
                )
    return stars


def ingest_arc_task_rules(
    task_json: dict[str, Any],
    *,
    task_id: str,
    galaxy_manager: Any | None = None,
) -> list[Any]:
    """Seed one ARC task's demonstration rules into the canonical Grammar path."""
    from benchmarks.arc_task_galaxy_seeder import seed_task

    return seed_task(task_json, galaxy_manager=galaxy_manager, task_id=str(task_id))

__all__ = [
    # GPU OCR Processing
    "GPUPDFOCRProcessor",
    "OllamaSecondPassEnricher",
    
    # PDF Pipeline
    "PDFKnowledgePipeline", 
    "ModularPDFProcessor",
    
    # Main Entry Point
    "enhanced_pdf_ingest_main",
    # Ingestion-only Warp adapter
    "ImportedWarpScene",
    "import_warp_model",
    "import_warp_modelbuilder",
    "load_warp_scene_from_file",
    "ingest_physics_bootstrap",
    "ingest_entity_bootstrap",
    "ingest_cas_grammar",
    "ingest_sas_bootstrap",
    "ingest_router_cartographer_bootstrap",
    "ingest_arc_task_rules",
]

__version__ = "2.0.0"
