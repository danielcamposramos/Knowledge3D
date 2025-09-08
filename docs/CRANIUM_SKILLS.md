K3D Cranium: Skills Bus (Integrated Logic)

Intent
- Provide a single, coherent logic layer (the “Cranium”) that integrates small, specialized models (skills) directly with House memory, without relying on external adapters at runtime. This engine is multimodal by design: chat, vision (images/video), audio, and 3D shape understanding/manipulation live behind one interface.

Skills (initial set)
- Intent (text → action): HF classifier trained from live logs.
- Vision (images): OpenCLIP embeddings + retrieval; thumbnails in tooltips.
- Audio: LAION‑CLAP embeddings + retrieval.
- Video: frame sampling via PyAV + OpenCLIP aggregation per clip.
- 3D Shapes: shape features and embeddings for objects/leaves; supports “direct vector manipulation” to change semantics/placement.
- Dynamics (optional): a tiny RSSM for sequence prediction over interaction traces. This is not a “navigation AI”; it is a helper for temporal reasoning when useful.
- RPN (policy): precise, auditable rules gating actions and parameters.

Memory Interface
- All skills operate on the same House memory (glTF/GLB + extras.k3d), sharing one embedding “galaxy”. Objects may carry multiple rays: text, image, audio, video, and spatial vectors.

Runtime
- Skills are registered in a small orchestrator (Cranium) and invoked by intent/needs. RPN checks all proposed actions. The live server logs every step (for consolidation and training).

Sleep‑Compute
- During consolidation, the Cranium can:
  - Regenerate or augment objects (when heavy generators are available),
  - Embed new assets (CLIP/CLAP), update neighbors and metadata,
  - Retrain the small skills (intent, RSSM) from fresh logs.

Progressive Capability
- Larger generators (e.g., HunyuanWorld, TRELLIS) are integrated as optional producers and only invoked when hardware permits. The House memory format remains stable. The Cranium remains the unified, multimodal logic layer (no hard external dependencies at runtime).
