K3D Cranium: Skills Bus (Integrated Logic)

Intent
- Provide a single, coherent logic layer (the “Cranium”) that integrates small, specialized models (skills) directly with House memory, without relying on external adapters at runtime.

Skills (initial set)
- Intent (text → action): Hugging Face classifier trained from live logs.
- Vision (images): OpenCLIP embeddings + retrieval; thumbnails in tooltips.
- Audio: LAION‑CLAP embeddings + retrieval.
- Video: frame sampling via PyAV + OpenCLIP aggregation per clip.
- Dynamics: Tiny RSSM predicts next 3D step from recent trajectory.
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
- Larger generators (e.g., HunyuanWorld, TRELLIS) are integrated as optional skills and only invoked when hardware permits. The House memory format remains stable.

