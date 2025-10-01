# NEXT SPAWN — 2025‑10‑01

A precise handoff with commands, paths, and checkpoints to continue training a single, PTX‑first, multi‑modal fused head using House GLB appliances (Tablet‑first) and expanding datasets across text, images, audio, video, shapes, and PDFs.

## 0) Environment

- Repo root: `/K3D/Knowledge3D`
- Conda env: `k3d-cranium`

```

- Recommended 50-epoch variant (aligns with expansion policy):

```
tmux new-session -d -s k3d_av_50 "bash -lc '\n  source /home/daniel/miniforge/bin/activate k3d-cranium; cd /K3D/Knowledge3D;\n  export PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1;\n  python -m knowledge3d.tools.phase25.consistency_trainer --epochs 50 --limit 3000 --lr 1e-3'"
```

- Optional second pass (uses OCR text when present):

```
tmux new-session -d -s k3d_av_50_text "bash -lc '\n  source /home/daniel/miniforge/bin/activate k3d-cranium; cd /K3D/Knowledge3D;\n  export PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1 K3D_CONSISTENCY_FALLBACK_TEXT=1;\n  python -m knowledge3d.tools.phase25.consistency_trainer --epochs 50 --limit 3000 --lr 1e-3'"
```
source /home/daniel/miniforge/bin/activate k3d-cranium
cd /K3D/Knowledge3D
export PYTHONPATH=.
```

- Core env flags (default behaviors):
  - `K3D_PTX_STRICT=1` (PTX‑only features; no external model fallbacks)
  - `K3D_FORCE_PTX_FUSE=1` (build fused vector inside the head)
  - `K3D_RPN_BEAM=1` and `K3D_RPN_BEAM_WIDTH=5` (robust numeric decoding)

## 1) House / Appliances Policy

- All head weights are stored as GLB appliances inside the House and are NOT committed to Git.
- Pack/unpack via:

```
# Example (pack fused_core after a trainer saves sidecar)
python -m knowledge3d.tools.weights_in_glb \
  --glb viewer/public/houses/default/memory_house.glb \
  --pt  viewer/public/house/house_core_heads.pt \
  --appliance fused_core
```

## 2) Data Roots (local)

- HF/General: `/home/daniel/.cache/huggingface/datasets`
- Project datasets: `/K3D/Knowledge3D.local/datasets`
- Llama‑cpp datasets: `/home/daniel/K3D_llama_cpp/datasets`
- External PDFs/JSON:
  - `/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias`
  - `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries`

## 3) Ingestion (Shapes + PDFs/JSON)

- Open shapes (ModelNet10 → GLBs + manifest)

```
python -m knowledge3d.tools.phase25.ingest_open_shapes --dataset modelnet10 --limit 500
```

- External PDFs/JSON (skips index/TOC/copyright pages, generates previews)

```
python -m knowledge3d.tools.phase25.ingest_pdf_corpus \
  --roots "/mnt/arquivos/0 ChatGPTs/DataBase/Encyclopedias,/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries" \
  --max-depth 4 --limit 500
```

- House manifest to consult/verify:
  - `viewer/public/house/materialized_objects/manifest.json`

## 4) Long‑Run Trainers (tmux)

> Rationale: long‑lived processes + PTX warm‑up reduce driver teardown faults. Beam decoding improves numeric robustness.

- Launch two chained 50‑epoch runs (keys expanded) in tmux:

```
tmux new-session -d -s k3d_multi_x2_full "bash -lc '
  source /home/daniel/miniforge/bin/activate k3d-cranium; cd /K3D/Knowledge3D;
  export PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1 K3D_RPN_BEAM=1 K3D_RPN_BEAM_WIDTH=5;
  python -m knowledge3d.tools.phase25.long_run \
    --epochs 50 --limit 300 --eval-every 5 \
    --dims "64,64,64,64" \
    --keys "math,gsm8k,metamath,aime,amc,olympiad,algebra,arc,openbook,geometry,number,theorem,logic,iq,reasoning,science,physics,chemistry,biology,probability,combinatorics" \
  && \
  python -m knowledge3d.tools.phase25.long_run \
    --epochs 50 --limit 300 --eval-every 5 \
    --dims "64,64,64,64" \
    --keys "math,gsm8k,metamath,aime,amc,olympiad,algebra,arc,openbook,geometry,number,theorem,logic,iq,reasoning,science,physics,chemistry,biology,probability,combinatorics"'"
```

- Shapes consistency (previews) — 100 epochs:

```
tmux new-session -d -s k3d_shapes_100 "bash -lc '
  source /home/daniel/miniforge/bin/activate k3d-cranium; cd /K3D/Knowledge3D;
  export PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1;
  python -m knowledge3d.tools.phase25.shapes_trainer --epochs 100 --limit 5000'"
```

- AV consistency — 30 epochs across images/audio/video:

```
tmux new-session -d -s k3d_av_30 "bash -lc '
  source /home/daniel/miniforge/bin/activate k3d-cranium; cd /K3D/Knowledge3D;
  export PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1;
  python -m knowledge3d.tools.phase25.consistency_trainer --epochs 30 --limit 3000 --lr 1e-3'"
```

- ARC grid (invariances enabled) — optional:

```
python -m knowledge3d.tools.phase25.arc_grid_trainer \
  --arc-root /K3D/Knowledge3D.local/datasets/exams/arc-src/data/training \
  --limit 1000 --epochs 8
```

- Monitor dashboards (every 10 min):

```
tmux new-session -d -s k3d_monitor "bash -lc '
  source /home/daniel/miniforge/bin/activate k3d-cranium; cd /K3D/Knowledge3D; export PYTHONPATH=.;
  while true; do python -m knowledge3d.tools.phase25.progress_dashboard; sleep 600; done'"
```

- tmux quick refs: `tmux ls`; `tmux attach -t k3d_multi_x2_full`; detach with `Ctrl-b d`.

## 5) Periodic Evaluations (automatic inside long_run)

- Every 5 epochs (configurable via `--eval-every`):
  - RPN stack unit eval (beam) — logs to Tablet (Learning Memory)
  - Mini math bench (30 samples across AIME/MetaMathQA/GSM8K) — writes to `docs/benchmarks/math_bench_epoch_<ep>.json` and appends mean accuracy to progress log.

- Manual runs:

```
# RPN stack eval (beam)
PYTHONPATH=. K3D_PTX_STRICT=1 K3D_FORCE_PTX_FUSE=1 K3D_RPN_BEAM=1 K3D_RPN_BEAM_WIDTH=5 \
python -m knowledge3d.tools.phase25.rpn_stack_eval --limit 1000 --beam

# Aggregate progress to summaries
PYTHONPATH=. python -m knowledge3d.tools.phase25.progress_dashboard
```

- Files to consult:
  - `docs/benchmarks/progress_log.json` (epoch records)
  - `docs/benchmarks/progress_summary.{json,md}` (aggregates)
  - `docs/benchmarks/math_bench_epoch_<ep>.json` (mini benches)
  - `docs/benchmarks/math_bench_report_{pre,post}.json` (full benches)

## 6) RLWHF (Reinforced Learning With Honesty and Feedback)

> Use RLWHF to refine honesty‑driven behaviors and thinking tags without merging low‑density memory into weights.

- Build RLWHF dataset (either live logs or offline benchmark):

```
# From scripts (if using project wrappers)
./scripts/run_live_benchmark.sh
# or
./scripts/train_rlwhf_policy.sh
```

- Direct module pattern (adjust paths if needed):

```
# Build dataset from offline runs
python -m knowledge3d.tools.rlwhf_from_offline_benchmark \
  --out docs/reports/training/rlwhf_dataset.jsonl

# Train RLWHF policy
python -m knowledge3d.tools.train_rlwhf_policy \
  --dataset docs/reports/training/rlwhf_dataset.jsonl \
  --out     ../Knowledge3D.local/models/rlwhf_policy

# Evaluate RLWHF policy
python -m knowledge3d.tools.eval_rlwhf_policy \
  --dataset docs/reports/training/rlwhf_dataset.jsonl \
  --model   ../Knowledge3D.local/models/rlwhf_policy \
  --out     docs/reports/status/rlwhf_policy_eval.json
```

- Refresh Phase 10 thinking tags after long RLWHF sessions:

```
python -m knowledge3d.tools.phase10.thinking_tag_trainer --mode rlwhf \
  --output_model viewer/public/models/thinking_tag_embedder_rlwhf.pth \
  --output_tags  viewer/public/models/tag_names_rlwhf.json
```

## 7) Evaluation / Reports

- Tablet (Learning Memory) receives:
  - RPN stack eval summaries
  - Training “thinking tags” and consistency records
- House manifest grows with:
  - GLB shapes (internal/external)
  - Document previews
- Progress dashboards:
  - Run `python -m knowledge3d.tools.phase25.progress_dashboard` periodically to refresh summaries.

## 8) Notes & Constraints

- Keep appliances in House GLB; do not commit them.
- Prefer tmux for long runs.
- PTX only; no CPU/external model fallbacks in core training.
- Skipping indices/copyright/TOC pages is heuristic; adjust roots and limits as desired.

---

## Appendix — Key Files

- Fused head: `knowledge3d/cranium/fused_head.py`
- Trainers:
  - Multi: `knowledge3d/tools/phase25/fused_multi_trainer.py`
  - Long-run driver: `knowledge3d/tools/phase25/long_run.py`
  - Shapes (consistency): `knowledge3d/tools/phase25/shapes_trainer.py`
  - Consistency (images/audio/video): `knowledge3d/tools/phase25/consistency_trainer.py`
  - ARC grid trainer: `knowledge3d/tools/phase25/arc_grid_trainer.py`
  - RPN stack eval: `knowledge3d/tools/phase25/rpn_stack_eval.py`
  - Progress dashboard: `knowledge3d/tools/phase25/progress_dashboard.py`
  - Ingest shapes: `knowledge3d/tools/phase25/ingest_open_shapes.py`
  - Ingest PDFs/JSON: `knowledge3d/tools/phase25/ingest_pdf_corpus.py`
- House manifest: `viewer/public/house/materialized_objects/manifest.json`
- Benchmarks: `docs/benchmarks/`
