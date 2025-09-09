# K3D Daily Wrap — 2025-09-09

This document records everything we built and ran today, with exact file paths and commands, so work can be replicated tomorrow. It also captures rationale and decisions (e.g., ditching TinyLlama for now and prioritizing K3D’s internal paradigm).

## Environment
- OS: Debian 13 (GPU‑only policy)
- GPU: NVIDIA RTX 3060 12GB
- Python env: Conda `k3dml` (CUDA 12.1 PyTorch)
- WebSockets: pinned `websockets==10.4` for server/client stability
- Repo: `K3D/Knowledge3D`
- Local artifacts: `../Knowledge3D.local/`

## Live Mode Stability
- Avoids ComfyUI port 8787 and uses a safe port set.
- Adds fast‑start and healthz to prevent opening‑handshake timeouts.

Files changed:
- `scripts/run_live_benchmark.sh` — safe PID shutdown; skips 8787; WS readiness probe; offline fallback
- `knowledge3d/bridge/live_server.py` — fast‑start (`K3D_LIVE_FAST=1` or `--fast-start`), background warmup, `healthz` event
- `envs/k3d-rapids.yml`, `envs/k3d-cpu.yml`, `scripts/k3d_env.sh` — pin `websockets==10.4`

Docs:
- `docs/ENV_POLICY.md`
- `docs/reports/status/live_run_notes_2025-09-09.md`

## Benchmarks and Artifacts
- Offline baseline (TinyLlama): `docs/reports/status/chat_benchmark_offline_default.json`
- RLWHF policy (distilgpt2): `docs/reports/status/chat_benchmark_offline_rlwhf.json`
- RLWHF LoRA TinyLlama (Q4, t=128): `docs/reports/status/chat_benchmark_offline_rlwhf_lora_q4_t128.json`
- RLWHF LoRA TinyLlama (GLB‑200, Q4, t=128): `docs/reports/status/chat_benchmark_offline_rlwhf_lora_glb_q4_t128.json`
- Comparisons:
  - `docs/reports/status/rlwhf_comparison.md` (Default vs distilgpt2 policy)
  - `docs/reports/status/rlwhf_lora_comparison.md` (Default vs TinyLlama LoRA)

Key summary (20 queries, same Galaxy):
- K3D compose (internal retrieval+composition) — sim≈0.776; p50≈50–60 ms
- TinyLlama baseline — sim≈0.15–0.19; slower
- RLWHF policy (distilgpt2) — sim≈0.19; faster than TinyLlama; still below K3D compose
- RLWHF LoRA TinyLlama — sim in ~0.16–0.20 range with 200 GLB‑grounded rows; latencies higher under 4‑bit

Conclusion: K3D compose remains superior for grounded answers and latency. We keep compose as primary; generative is auxiliary.

## RLWHF Pipeline (Internal)
- Ranker (compose context scoring): `knowledge3d/models/answer_ranker.py`
- Policy RW‑SFT (tiny): `knowledge3d/models/rlwhf_policy.py`
- LoRA training (TinyLlama): `knowledge3d/models/rlwhf_lora.py`
- Datasets:
  - From live logs/offline: `docs/reports/training/rlwhf_dataset.jsonl`
  - From GLB (grounded only, no external calls): `docs/reports/training/rlwhf_dataset_glb.jsonl`

Models (kept):
- Compose ranker: `../Knowledge3D.local/models/answer_ranker.pkl`
- RW‑SFT policy: `../Knowledge3D.local/models/rlwhf_policy`

Models (removed per decision):
- `../Knowledge3D.local/models/rlwhf_lora_tinyllama` — removed
- `../Knowledge3D.local/models/rlwhf_lora_tinyllama_glb` — removed

## Grounded Generative (Internal; no external RAG)
- Added `compose_generate` to `knowledge3d/skills/spatial_text.py`:
  - Builds an instruction + context prompt and calls the internal LLM skill.
  - Internal policy enforces: only use provided contexts, admit “I don’t know”, and cite labels.
- PEFT adapter support + caching in `knowledge3d/skills/llm.py` (if we reintroduce adapters later).

## Commands (Reference)
- Offline baseline benchmark (TinyLlama):
  `scripts/k3d_env.sh run python -m knowledge3d.tools.benchmark_offline --gltf viewer/public/galaxy.cross.glb --queries 20 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out-json docs/reports/status/chat_benchmark_offline_default.json --out-md docs/reports/status/chat_benchmark_offline_default.md`

- Train RW‑SFT policy (distilgpt2):
  `scripts/train_rlwhf_policy.sh`

- Evaluate policy grounding (ST cosine):
  `scripts/k3d_env.sh run python -m knowledge3d.tools.eval_rlwhf_policy --dataset docs/reports/training/rlwhf_dataset.jsonl --model ../Knowledge3D.local/models/rlwhf_policy --out docs/reports/status/rlwhf_policy_eval.json --limit 20`

- Build GLB‑grounded RLWHF rows (no external calls):
  `scripts/k3d_env.sh run python -m knowledge3d.tools.rlwhf_from_glb --gltf viewer/public/galaxy.cross.glb --out docs/reports/training/rlwhf_dataset_glb.jsonl --queries 200`

- Compare two reports:
  `scripts/k3d_env.sh run python -m knowledge3d.tools.compare_benchmarks --a docs/reports/status/chat_benchmark_offline_default.json --b docs/reports/status/chat_benchmark_offline_rlwhf.json --out-json docs/reports/status/rlwhf_comparison.json --out-md docs/reports/status/rlwhf_comparison.md`

- Publish local logs/models index:
  `scripts/k3d_env.sh run python -m knowledge3d.tools.publish_local_artifacts`

- Stop live servers (safe):
  `pkill -f knowledge3d.bridge.live_server || true`

- Remove TinyLlama LoRA adapters (decision for now):
  `rm -rf ../Knowledge3D.local/models/rlwhf_lora_tinyllama ../Knowledge3D.local/models/rlwhf_lora_tinyllama_glb`

## To Resume Tomorrow
- K3D compose remains primary; generative auxiliary. We’ll expand high‑quality grounded RLWHF data next.
- Quick smoke:
  - `scripts/k3d_env.sh run python -m knowledge3d.tools.benchmark_offline --gltf viewer/public/galaxy.cross.glb --queries 10 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out-json docs/reports/status/chat_benchmark_quick.json --out-md docs/reports/status/chat_benchmark_quick.md`

---

All work adheres to the Project Roadmap priorities: internal memory (House) first, compose + grounded generative as core, and transparent, cite‑able answers.
