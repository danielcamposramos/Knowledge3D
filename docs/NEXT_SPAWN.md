Next Spawn Playbook
===================

Environment
-----------
- Use GPU env via `scripts/k3d_env.sh bootstrap-rapids` and run all commands with `scripts/k3d_env.sh run ...`.
- Enforce `K3D_STRICT_GPU=1` to prevent CPU fallbacks for policy training and embeddings.

Galaxy
------
- Current unified Galaxy: `viewer/public/galaxy.v8.glb` (+ `.cross.glb`).
- Keep per‑modality balanced growth, by topic. New video batches arrive under `/K3D/K3D_llama_cpp/datasets/msrvtt_dl_more`.
- Rebuild v8 periodically:
  - `knowledge3d.tools.trellis_adapter to-k3d` for new CSV/meta pairs → GLBs
  - `knowledge3d.tools.unify_glbs ... --out viewer/public/galaxy.v8.glb --dims 256 --k 10 --reducer umap`
  - `knowledge3d.tools.add_crossmodal_edges`

House Vertical (Architecture & Furniture)
----------------------------------------
- Sources:
  - Images: COCO filter (chair, sofa, bed, table, lamp, door, window, kitchen, room, interior, furniture)
  - Audio: Clotho filter (door, knock, kitchen, vacuum, appliances, steps, home, wood, tools)
  - 3D: curated interior assets under `../Knowledge3D.local/datasets/gltf_house` (expand as curated)
  - Text: `knowledge3d.tools.gen_text_ollama_multi` (exaone, exaone‑deep, granite, deepseek‑r1)
- Build/update:
  - `viewer/public/house/image_house.glb`, `audio_house.glb`, `shapes_house.glb`, `text_house_all.glb`
  - `viewer/public/galaxy.house.glb` + `.cross.glb`

RLWHF Policy
------------
- Train (GPU only):
  - `scripts/k3d_env.sh run python -m knowledge3d.tools.train_rlwhf_policy --dataset docs/reports/training/rlwhf_dataset_unified_v8.jsonl --out ../Knowledge3D.local/models/rlwhf_policy_v8eXX --model distilgpt2 --epochs 10 --batch 4 --max_len 384 --lr 5e-5`
- Evaluate:
  - `scripts/k3d_env.sh run python -m knowledge3d.tools.eval_rlwhf_policy --dataset docs/reports/training/rlwhf_dataset_unified_v8.jsonl --model ../Knowledge3D.local/models/rlwhf_policy_v8eXX --out docs/reports/status/rlwhf_policy_eval_v8eXX.json --limit 500`
  - Optionally re‑run with `--limit 1000` overnight.

Generative Seeding
------------------
- Multi‑model text (Ollama):
  - `scripts/k3d_env.sh run python -m knowledge3d.tools.gen_text_ollama_multi --ollama http://192.168.0.4:11434 --models "exaone3.5:latest,exaone-deep:latest,granite3.3:8b,deepseek-r1:14b" --topics "architecture,furniture,rooms,doors,windows,kitchen,bedroom,bathroom,living room,materials,textures" --n 60 --out-dir ../Knowledge3D.local/datasets`
- Embed on GPU and build GLB via `embed_text_sharded` + `trellis_adapter to-k3d`.

Logging & Reports
-----------------
- Training summary: `docs/reports/status/rlwhf_training_v8e10.md`
- Eval reports: `docs/reports/status/rlwhf_policy_eval_*.json` and `..._report.md`
- Balanced expansion policy: `docs/EXPANSION_POLICY.md`
- Local models and roles: `docs/LOCAL_OLLAMA_MODELS.md`

Notes
-----
- Avoid CPU fallbacks; if CUDA is unavailable, abort and fix env.
- Keep disk usage in check: prune obsolete subsets after GLBs are built; stage raw media to `/K3D/K3D_llama_cpp/datasets`.
