"""
Knowledge ingestion subsystem.

Provides pipelines that transform external datasets into K3D-compatible
representations (Galaxy nodes, House artifacts, Garden growth inputs).

Modules are imported lazily to avoid pulling optional dependencies (e.g. librosa)
when only a subset of pipelines are required.
"""

from importlib import import_module
from typing import Any

_MODULE_EXPORTS = {
    "language": "knowledge3d.ingestion.language",
    "lexicons": "knowledge3d.ingestion.lexicons",
    "documents": "knowledge3d.ingestion.documents",
    "corpus_manifest": "knowledge3d.ingestion.corpus_manifest",
    "batch_orchestrator": "knowledge3d.ingestion.batch_orchestrator",
    "enrichment_pipeline": "knowledge3d.ingestion.enrichment_pipeline",
    "pdf_classifier": "knowledge3d.ingestion.pdf_classifier",
    "pdf_augmenter": "knowledge3d.ingestion.pdf_augmenter",
}

_SYMBOL_EXPORTS = {
    "CorpusEntry": ("knowledge3d.ingestion.corpus_manifest", "CorpusEntry"),
    "CorpusManifest": ("knowledge3d.ingestion.corpus_manifest", "CorpusManifest"),
    "CorpusTier": ("knowledge3d.ingestion.corpus_manifest", "CorpusTier"),
    "CorpusType": ("knowledge3d.ingestion.corpus_manifest", "CorpusType"),
    "BatchOrchestrator": ("knowledge3d.ingestion.batch_orchestrator", "BatchOrchestrator"),
    "EnrichmentPipeline": ("knowledge3d.ingestion.enrichment_pipeline", "EnrichmentPipeline"),
    "PDFKnowledgeClassifier": ("knowledge3d.ingestion.pdf_classifier", "PDFKnowledgeClassifier"),
    "PDFKnowledgeAugmenter": ("knowledge3d.ingestion.pdf_augmenter", "PDFKnowledgeAugmenter"),
}

__all__ = sorted([*list(_MODULE_EXPORTS.keys()), *list(_SYMBOL_EXPORTS.keys())])


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        module = import_module(_MODULE_EXPORTS[name])
        globals()[name] = module
        return module
    if name in _SYMBOL_EXPORTS:
        module_path, symbol_name = _SYMBOL_EXPORTS[name]
        module = import_module(module_path)
        symbol = getattr(module, symbol_name)
        globals()[name] = symbol
        return symbol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
