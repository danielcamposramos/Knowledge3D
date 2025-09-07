HunyuanWorld Integration (Study + Leverage)

Goal
- Stand on the shoulders of giants: reuse HunyuanWorld’s open world‑model ideas, training data preparation, and scene generation capabilities, while keeping K3D’s memory‑first design (permanent glTF/GLB + extras.k3d) and a small, modular logic layer.

Repo Location
- Cloned under `ext/HunyuanWorld-1.0` (shallow clone). Use it as a reference and for local experiments; do not vendor large weights into this repo.

Why This Fits K3D
- K3D separates memory from logic: the House is permanent memory; small “brain regions” (intent, vision, audio, dynamics/RSSM, RPN) do the active work. HunyuanWorld provides powerful generative priors (3D scene synthesis) that we can adapt to populate House rooms while we keep our logic minimal.

High‑Level Path
1) Study & Datasets
   - Follow HunyuanWorld’s README and scripts to replicate data preparation. Prefer the listed open datasets (images/videos/scenes). Mirror minimally needed subsets (e.g., 20k–100k) for initial runs.
   - Reuse those sets in K3D’s ingest tools (text/image/video/audio) so embeddings and logs integrate with our training flows.
2) Inference → K3D
   - Add an adapter (`knowledge3d/tools/hunyuan_adapter.py`) that runs HunyuanWorld inference (text→scene or panogen) and converts outputs into glTF 2.0 with embedded K3D payload (`extras.k3d`).
   - Populate room‑scale scenes (Library/Garden/Workshop) as GLBs with vectors/embeddings/metadata (ids, labels, thumbnails). Store them under `../Knowledge3D.local/datasets/` and reference them in `viewer/public/condo.json`.
3) Learning Dynamics
   - Train K3D’s tiny world‑model (RSSM) on navigation logs over these generated scenes. This complements generative priors with observed dynamics.

Adapter Responsibilities (planned)
- Normalize HunyuanWorld outputs (meshes/scene graphs) → glTF 2.0.
- Construct buffers for POSITION (default + optional LOD) and embeddings (e.g., CLIP for textures/objects).
- Fill `primitive.extras.k3d` with ids, neighbors, metadata (labels, image URLs), AI flags, temporal tags.

Runtime & Containers
- Extend `docker/Dockerfile.k3d-gpu` or add a dedicated file to install HunyuanWorld dependencies (Torch/xFormers/weights). Keep weights out of the repo; place them under `../Knowledge3D.local/models/`.

Licenses
- HunyuanWorld’s LICENSE and NOTICE are in `ext/HunyuanWorld-1.0`. Adhere to their terms for any derived assets.

Status
- Repo cloned under `ext/`. Adapter stub to be added; dataset replication to follow their docs. We will start by leveraging their dataset recipes to seed K3D’s multimodal ingest and keep the House memory permanent and navigable.

