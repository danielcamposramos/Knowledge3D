Local Development Environment (Reference)

- Hardware: AMD Ryzen 5 5600G, 96 GB RAM
- GPU: NVIDIA GeForce RTX 3070 (11 GB VRAM), CUDA target
- Storage: Standard HDD
- Local folder (not in repo): ../Knowledge3D.local
  - Structure: datasets/, logs/, models/, mr/, repos/

Acceleration settings
- Set `K3D_ACCEL=auto` (default) to prefer GPU when available; override with `gpu` or `cpu`.
- Optional router override: `K3D_ROUTER=bfs|dijkstra|astar` (default: bfs unless positions provided).

Large datasets
- Typical vector sizes: 180k–240k rows (e.g., 768‑dim embeddings). Use GPU paths when possible.
- k3dgen now prefers FAISS GPU for k‑NN and cuML UMAP for 3D reduction when installed.

Workflows
- Viewer: `cd viewer && npm run dev`
- Live server: `python -m knowledge3d.bridge.live_server`
- GPU smoke: `python -m knowledge3d.tools.gpu_smoke`

Knowledge Garden
- Every House includes the room “Knowledge Garden” (ontology greenhouse).
- Build a demo glb: `python3 -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb`

