Local Development Environment (Reference)

- Hardware: AMD Ryzen 5 5600G, 96 GB RAM
- GPU: NVIDIA GeForce RTX 3070 (11 GB VRAM), CUDA target
- Storage: Standard HDD
- Local folder (not in repo): ../Knowledge3D.local
  - Structure: datasets/, logs/, models/, mr/, repos/
  - Alternative raw store (bigger disk): `/K3D/K3D_llama_cpp/datasets/` (often symlinked nearby). Prefer keeping curated subsets under `../Knowledge3D.local/datasets` and link to large raw trees to save space.

Acceleration settings
- Set `K3D_ACCEL=auto` (default) to prefer GPU when available; override with `gpu` or `cpu`.
- Optional router override: `K3D_ROUTER=bfs|dijkstra|astar` (default: bfs unless positions provided).

GPU setup guides
- See `docs/GPU_ACCEL.md` for Docker-based RAPIDS UMAP + FAISS GPU instructions.

Training containment (Debian guard)
- On Debian-like hosts (including Debian 13), heavy training commands are blocked unless running under Conda or Docker. This avoids unstable native builds and GPU driver mismatches. Prefer the RAPIDS Docker path in `docs/GPU_ACCEL.md`.
- To override (not recommended), set `K3D_ALLOW_NATIVE=1`.

Affected commands
- `python -m knowledge3d.tools.train_all`
- `python -m knowledge3d.models.intent_hf train ...`
- `python -m knowledge3d.models.spatial_memory_trainer ...`
- `python -m knowledge3d.rl.unsloth_adapter train ...`

Conda environments
- GPU (RAPIDS + FAISS GPU):
  - `conda env create -f envs/k3d-rapids.yml && conda activate k3d-rapids`
  - `export K3D_ACCEL=gpu K3D_FAISS_DEVICE=gpu`
- CPU (lightweight dev/testing):
  - `conda env create -f envs/k3d-cpu.yml && conda activate k3d-cpu`
  - `export K3D_ACCEL=cpu`

Notes
- RAPIDS requires a matching CUDA runtime; see `docs/GPU_ACCEL.md` for alternatives using Docker containers.

Large datasets
- Typical vector sizes: 180k–240k rows (e.g., 768‑dim embeddings). Use GPU paths when possible.
- k3dgen now prefers FAISS GPU for k‑NN and cuML UMAP for 3D reduction when installed.

Workflows
- Viewer: `cd viewer && npm run dev`
- Live server: `python -m knowledge3d.bridge.live_server`
- GPU smoke: `python -m knowledge3d.tools.gpu_smoke`

Log maintenance (server)
- Env: `K3D_LOG_ROTATE_BYTES` (default 1GB), `K3D_LOG_COMPRESS_AGE_HOURS` (default 24), `K3D_LOG_MAINT_PERIOD` (default 60s).
- Chat commands: `/logs status`, `/logs rotate`, `/logs compress`.

Traditional Chat (Tablet/CLI)
- Server loads the model automatically when available: `K3D_MODEL=../Knowledge3D.local/models/intent_hf K3D_MODEL_AUTO=1 python -m knowledge3d.bridge.live_server`
- Headless chat client (auto replies): `python -m knowledge3d.bridge.cli_client --nick tester --auto --model ../Knowledge3D.local/models/intent_hf --threshold 0.7`
- The 3D Tablet mirrors this “old Q&A” style, while the agent remains spatial‑first.

Long Dual‑Agent Runs (Multilingual)
- Generate training logs in parallel using two agents (scout + gardener):
```bash
python -m knowledge3d.tools.multi_instance \
  --gltf ../Knowledge3D.local/datasets/ai_compendium.500k.umap.ivfpq.glb \
  --count 500 --delay 0.08 --langs en,pt,es,fr,de,it,zh,ja,ar,ru,ko,hi,tr
```
- Logs land in `../Knowledge3D.local/logs/`. Retrain HF with fp16 after runs.

Knowledge Garden
- Every House includes the room “Knowledge Garden” (ontology greenhouse).
- Build a demo glb: `python3 -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb`
- Ethics tree: `python3 -m knowledge3d.tools.gardens --paths data/ontology/ethics_paths.txt --gltf viewer/public/knowledge_garden.ethics.glb`
