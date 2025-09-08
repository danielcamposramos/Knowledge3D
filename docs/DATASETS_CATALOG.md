# Public Datasets Catalog (Curated)

Purpose
- Track high‑value, open datasets across modalities that map cleanly to K3D’s memory‑first pipeline (embeddings → embedded glTF/GLB via `extras.k3d`).
- Serve as an actionable index when expanding beyond our 50k bootstraps.

Authoritative index for broad discovery
- Wikipedia: List of datasets for machine‑learning research — https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
  - Use this as a source of truth to find candidates, then select sized subsets that fit local disk and bandwidth budgets.

Modality Buckets (starter picks)
- Text
  - Wiki/Books excerpts or domain corpora (e.g., AI books). Ingest via simple line files → `k3dgen --text`.
  - HuggingFace corpora are also viable; keep curated CSV/JSONL lines to ≤1–5M for local runs.
- Image (captions)
  - COCO Captions (train2017) — supported by `knowledge3d/tools/ingest_coco.py`.
  - WIT sample TSV — supported by `knowledge3d/tools/ingest_wit.py`.
- Audio (captions)
  - Clotho, AudioCaps — supported by `knowledge3d/tools/ingest_audio.py` (LAION‑CLAP).
- Video (captions)
  - VATEX, MSR‑VTT, WebVid mini — supported by `knowledge3d/tools/ingest_video.py` (OpenCLIP).
- Multimodal
  - Cross‑modal matching from audio/video pools — `knowledge3d/tools/match_crossmodal.py`.
  - Knowledge Gardens ontology source files (paths) — `knowledge3d/tools/gardens.py --paths ...`.
- 3D/Scenes (reference; adapter targets)
  - HunyuanWorld outputs (text→room/scene), DreamFusion family (text→NeRF/mesh). Convert to K3D GLB with an adapter (see “Scene Generation Inspiration”).

Pipeline Mapping
- Ingest → embeddings: image/video via OpenCLIP, audio via CLAP, text via HF (or hashing for quick demos).
- k‑NN + reduction: FAISS (GPU if available) and UMAP (GPU via RAPIDS) or PCA fallback.
- Export: `k3dgen` produces embedded GLTF/GLB with `primitive.extras.k3d` containing ids, vectorsView, embeddingsView, metadata, neighbors, temporal, and optional AI flags.

Local Paths & Disk Policy
- Primary curated store: `../Knowledge3D.local/datasets/`
- Raw large media (alt disk): `/K3D/K3D_llama_cpp/datasets/` (symlinked nearby)
  - Favor symbolic links into `../Knowledge3D.local/datasets` for smaller curated subsets.
  - Keep GLBs lightweight when possible (use `--emb-precision f16`).

See Also
- docs/DATASETS_50K.md — small bootstraps + runbook
- docs/RUNBOOK_MULTIMODAL_50K.md — operator steps for COCO/Clotho/VATEX
- docs/HUNYUANWORLD_INTEGRATION.md — external scene model study

