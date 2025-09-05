Inspiration Sources

- NVIDIA AI Blueprints — 3D Object Generation
  - URL: https://github.com/NVIDIA-AI-Blueprints/3d-object-generation
  - Why it matters: elevates human‑facing quality and gives the AI client grounded “knowledge→object” mappings. We can port exemplar assets/pipelines where licensing permits and feed simplified geometry + metadata into K3D’s embedded glTF (`extras.k3d`).

Ideas to incorporate
- Knowledge→Object: Represent canonical concepts (e.g., “book”, “door”, “graph node”) as stylized meshes for the human client; keep AI view as embeddings + neighbors. Maintain identity across both.
- LOD Assets: Provide multi‑LOD meshes consistent with AI LOD: far billboards/low‑poly, near detailed geometry.
- Generation Hooks: Add a converter that takes generated meshes and injects K3D payload (ids, vectors, embeddings, neighbors) so the object participates in routing and explainability.

Notes
- Keep large/generated binaries out of the repo; save under `../Knowledge3D.local/datasets/assets/...` and document exact steps to reproduce.
- Align with K3D dual‑client principle (human visuals vs AI embeddings).
