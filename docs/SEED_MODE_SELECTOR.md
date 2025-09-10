Mode Selector Seeding & Training (GPU‑Only)
===========================================

Overview
--------
We auto‑route between `compose` (grounded factual; retrieval+stitching) and `compose_generate` (grounded generation) via a small learned selector. Outcomes are logged live, then we train a TF‑IDF + RandomForest classifier.

Components
----------
- Live server (WS): `knowledge3d.bridge.live_server` (use `--auto-port`)
- Outcome logging: `compose_auto` writes to `docs/reports/training/mode_selector_outcomes.jsonl`
- Seeder (remote LLM, slow‑GPU tolerant): `knowledge3d.tools.seed_mode_selector`
- Trainer: `knowledge3d.models.mode_selector` + `scripts/train_mode_selector.sh`

Prereqs
-------
- GPU‑ready env via `scripts/k3d_env.sh` (RAPIDS + Torch CUDA)
- Optional: FAISS GPU in `k3dfaiss` (see `docs/ENV_FAISS.md`) for KNN steps

Run (Step‑by‑Step)
------------------
1) Start the live server, enable outcome logging:

```
export K3D_USE_COMPOSE_AUTO=1 K3D_MODE_LOG_SIM=1
scripts/k3d_env.sh run python -m knowledge3d.bridge.live_server --auto-port
```

The chosen port is written to `docs/reports/status/live_server_ports.json` (field `chosen`).

2) Seed with Granite 2B on the GTX 970 box (long timeouts + retry built‑in):

```
WS=ws://127.0.0.1:<chosen>
scripts/k3d_env.sh run python -m knowledge3d.tools.seed_mode_selector \
  --ws "$WS" \
  --gltf viewer/public/galaxy.cross.glb \
  --ollama http://192.168.0.60:11434 \
  --model granite3.3:2b \
  --n 60
```

Tip: keep the viewer open (`cd viewer && npm run dev`) so it shares dataset graph/snippets with the server.

3) Verify logs:

```
wc -l docs/reports/training/mode_selector_outcomes.jsonl
```

4) Train the selector:

```
chmod +x scripts/train_mode_selector.sh
./scripts/train_mode_selector.sh
```

Model saved to `../Knowledge3D.local/models/mode_selector.pkl` and auto‑loaded by `compose_auto`.

Notes
-----
- The seeder only asks the remote LLM to produce a JSON list of prompts; it’s intentionally non‑agentic.
- On very old GPUs, the first call may be slow. The seeder uses 240s HTTP timeout and a WS retry loop with open_timeout=90s.
- For very large galaxies, cap the `dataset_graph` payload size to avoid WS frame limits:
  - `export K3D_SEED_GRAPH_MAX=1200` (default ~1200). The seeder will subsample nodes and keep neighbor links within the subsample.
