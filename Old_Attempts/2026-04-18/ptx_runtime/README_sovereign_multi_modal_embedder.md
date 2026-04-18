# Archived: sovereign_multi_modal_embedder.py

**Original Path**: `knowledge3d/cranium/ptx_runtime/sovereign_multi_modal_embedder.py`  
**Archive Date**: 2026-04-18  
**Reason**: Bulk-library sovereignty violation (line 10 imports numpy, sentence_transformers)  

## Why Archived

This module violates K3D sovereignty rules:
- Imports `sentence_transformers` in hot path (game-loop runtime)
- Uses `numpy` for embedding operations
- Pre-embeds text corpus at runtime instead of pre-caching during galaxy load

K3D paradigm: ALL text embeddings must be pre-computed during ingestion and cached in Galaxy; game loop must index only, never call SentenceTransformer.

## Replacement Strategy

**Phase 1**: Pre-compute embedding cache during `build_galaxy.py` ingestion  
**Phase 2**: Load embedding index into Galaxy during game initialization  
**Phase 3**: Replace game-loop calls with Galaxy index lookup (zero SentenceTransformer at runtime)

See: Audit report Table 2.4 "Sentence-Transformers (Embedding Bridge)" and Phase 4 "Cache Pre-computation"

## Active Importers

- `multi_modal_world_generator.py` (also archived, same reasoning)

If you need to restore usage, update that importer's reference path.

---

*This file was moved as part of bulk-library purge Phase 1. See TEMP/bulk_lib_audit_04.18.2026.md.*
