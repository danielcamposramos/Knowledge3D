K3D Work Session Report — 2025-09-05

Scope
- GPU acceleration (RAPIDS UMAP, FAISS IVF), AI LOD routing, per‑hop trace, Ethics Garden, large compendium builds (80k/180k/240k), multilingual model fine‑tune, retrieval/routing evaluation, RL (Unsloth‑style) scaffold, and inspiration plan from NVIDIA 3D object generation.

Deliverables
- Code: IVF switch in k3dgen, dynamic LOD A*, per‑hop trace, Ethics Garden ingestion, evaluation tools, RL adapter scaffold.
- Assets: `k3d_foundation.6k.umap.glb`, `knowledge_garden.ethics.glb`, Memory House GLTF, 80k/180k/240k GLBs (local).
- Models: sklearn baseline, multilingual HF fine‑tuned model (EN/PT/ES) saved to `../Knowledge3D.local/models/intent_hf`.
- Docs: GPU guide, RL integration notes, progress + evaluation summaries.

Results (high‑level)
- Retrieval: IVF recall@10 ≈ 0.9994 on 80k; GPU UMAP for 3D projection; FAISS IVF guarded to CPU for stability.
- Routing: On unit‑cost edges, BFS has lower hops and latency; A* and A* LOD provide geometric guidance and will benefit from weighted edges.
- LOD: Viewer HUD added; server uses LOD A* by default when positions are present.

Inspiration & Next Steps
- NVIDIA 3D Object Generation: Plan a knowledge→object converter with multi‑LOD assets for human‑pleasant rendering while preserving AI embeddings.
- Weighted routing: add edge weights (1 - cosine) for A*; evaluate.
- IVF‑PQ: add as k3dgen `--ann ivfpq` mode and benchmark.
- RL: hook Unsloth/TRL PPO with reward shaping from logs (sim, success, hop penalties) and export models for `/model on`.

How to resume next session
- Big builds are in `../Knowledge3D.local/datasets`. The evaluation JSONs are under `docs/reports/status/`.
- Use `python -m knowledge3d.tools.eval_retrieval` and `python -m knowledge3d.tools.eval_routing` to re‑run tests.
- For multilingual model, run `python -m knowledge3d.models.intent_hf train ...` and `/model on` in live server.
