# Runbook — Build 1M LOD House + Garden (Container)

Prereqs
- NVIDIA GPU + Docker (with `--gpus all`).
- Repo checked out and sibling local folder: `../Knowledge3D.local/{datasets,logs,models,mr,conda_pkgs}`.

Build the GPU image
```bash
bash scripts/docker_build_gpu.sh
```

Launch an interactive container (optional)
```bash
bash scripts/docker_run_gpu.sh
# Inside container: repo is at /workspace, local artifacts at /k3dlocal
```

One‑click 1M LOD pipeline (detached)
```bash
# From host
docker rm -f k3d-build1m >/dev/null 2>&1 || true
docker run -d --gpus all --name k3d-build1m \
  -v "$PWD":"/workspace" -w /workspace \
  -v "$(dirname "$PWD")/Knowledge3D.local/conda_pkgs":"/opt/conda/pkgs" \
  -v "$(dirname "$PWD")/Knowledge3D.local":"/k3dlocal" \
  k3d-gpu:latest bash -lc 'python -m pip install -e . && bash scripts/pipeline_1m.sh'

docker logs -f k3d-build1m
```

Outputs
- `../Knowledge3D.local/datasets/ai_compendium.1m.umap.ivfpq.doors.glb`
  - POSITION: PCA (fast far LOD)
  - `extras.k3d.lods`: `umap_fast`, `umap_high` (mid/near LODs)

Serve datasets (optional)
```bash
docker rm -f k3d-datasets >/dev/null 2>&1 || true
docker run -d --gpus all --name k3d-datasets --network host \
  -v "$PWD":"/workspace" -w /workspace \
  -v "$(dirname "$PWD")/Knowledge3D.local/conda_pkgs":"/opt/conda/pkgs" \
  -v "$(dirname "$PWD")/Knowledge3D.local":"/k3dlocal" \
  k3d-gpu:latest bash -lc 'python -m pip install -e . >/dev/null 2>&1 && python -m knowledge3d.tools.serve_datasets --port 8766'

# http://127.0.0.1:8766/exams_index.json
```

Launch live server (optional)
```bash
docker rm -f k3d-live >/dev/null 2>&1 || true
docker run -d --gpus all --name k3d-live --network host \
  -e K3D_MODEL="$(dirname "$PWD")/Knowledge3D.local/models/intent_hf" -e K3D_MODEL_AUTO=1 \
  -v "$PWD":"/workspace" -w /workspace \
  -v "$(dirname "$PWD")/Knowledge3D.local/conda_pkgs":"/opt/conda/pkgs" \
  -v "$(dirname "$PWD")/Knowledge3D.local":"/k3dlocal" \
  k3d-gpu:latest bash -lc 'ln -s /k3dlocal /workspace.local || true; python -m pip install -e . >/dev/null 2>&1 && python -m knowledge3d.bridge.live_server'
```

Generate multilingual logs (example small scene)
```bash
docker run --rm --gpus all --network host \
  -v "$PWD":"/workspace" -w /workspace \
  -v "$(dirname "$PWD")/Knowledge3D.local/conda_pkgs":"/opt/conda/pkgs" \
  -v "$(dirname "$PWD")/Knowledge3D.local":"/k3dlocal" \
  k3d-gpu:latest bash -lc 'python -m pip install -e . >/dev/null 2>&1 && \
    python -m knowledge3d.tools.multi_instance --url ws://127.0.0.1:8765 \
      --gltf data/ai_books_basic.4k.umap.doors.glb --count 120 --delay 0.08 \
      --langs en,pt,es --workers 1 --rounds 2'
```

Train/refine HF model
```bash
docker run --rm --gpus all \
  -v "$PWD":"/workspace" -w /workspace \
  -v "$(dirname "$PWD")/Knowledge3D.local/conda_pkgs":"/opt/conda/pkgs" \
  -v "$(dirname "$PWD")/Knowledge3D.local":"/k3dlocal" \
  k3d-gpu:latest bash -lc 'python -m pip install -e . >/dev/null 2>&1 && pip install -q "accelerate>=0.26.0" && \
    python -m knowledge3d.models.intent_hf train --out /k3dlocal/models/intent_hf --pretrained xlm-roberta-base --epochs 1 --batch-size 16'
```

Garden build (demo)
```bash
python -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb
```

Cleanup / rerun
- Stop containers: `docker rm -f k3d-build1m k3d-live k3d-datasets`.
- Image retained: `k3d-gpu:latest` (rebuild with `scripts/docker_build_gpu.sh`).

