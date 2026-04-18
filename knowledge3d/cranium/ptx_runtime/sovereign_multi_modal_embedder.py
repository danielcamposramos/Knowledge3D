"""
Shim for archived sovereign_multi_modal_embedder.py

This module has been archived due to K3D sovereignty violations.
Original location: Old_Attempts/2026-04-18/ptx_runtime/sovereign_multi_modal_embedder.py

See: Old_Attempts/2026-04-18/ptx_runtime/README_sovereign_multi_modal_embedder.md
"""

raise ImportError(
    "SovereignMultiModalEmbedder has been archived (2026-04-18). "
    "Reason: bulk-library sovereignty violation (sentence_transformers in hot path). "
    "See: Old_Attempts/2026-04-18/ptx_runtime/README_sovereign_multi_modal_embedder.md "
    "\n\nReplacement strategy: Pre-compute embedding cache during ingestion; "
    "load Galaxy index at boot; game loop indexes only (zero SentenceTransformer at runtime)."
)
