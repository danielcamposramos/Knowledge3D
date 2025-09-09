# RLWHF Training (Reward‑Weighted SFT)

This repository implements a practical RLWHF pipeline that runs on a single consumer GPU and integrates natively with K3D’s House memory and compose engine.

Overview
- Dataset: Built from live session logs (`../Knowledge3D.local/logs/session-*.jsonl`) via `knowledge3d.tools.build_rlwhf_dataset` or from the offline chat benchmark via `knowledge3d.tools.rlwhf_from_offline_benchmark`.
- Reward: Combines explicit feedback (good/partial/bad) with a grounding proxy (semantic similarity between answer and selected contexts). Honesty earns partial rewards when context is weak.
- Ranker: `knowledge3d/models/answer_ranker.py` learns a linear regression on ST cosine features to prioritize grounded contexts for compose.
- Policy: `knowledge3d/models/rlwhf_policy.py` trains a small causal LM (default `distilgpt2`) using reward‑weighted supervised fine‑tuning (RWSF). Loss is computed only over the Answer tokens and scaled by a monotonic mapping of the scalar reward to [0.1, 1.0].

Run end‑to‑end
- Live benchmark and ranker (avoids port 8787):
  - `scripts/run_live_benchmark.sh`
- Train RLWHF policy (ensures dataset exists, then trains):
  - `scripts/train_rlwhf_policy.sh`

Artifacts
- Dataset: `docs/reports/training/rlwhf_dataset.jsonl`
- Ranker: `../Knowledge3D.local/models/answer_ranker.pkl`
- Policy: `../Knowledge3D.local/models/rlwhf_policy/`

Use the policy at runtime
- Point the LLM skill to the trained local policy:
  - `export K3D_LLM_BACKEND=transformers`
  - `export K3D_LLM_MODEL=../Knowledge3D.local/models/rlwhf_policy`
- Or in chat: `/llm backend transformers ../Knowledge3D.local/models/rlwhf_policy`

Notes
- WebSockets: pinned to `websockets==10.4` for stability; the live server supports fast‑start and a `healthz` event to reduce handshake failures on Debian 13.
- GPU: Training runs on GPU if available; otherwise it will fall back to CPU (but the project encourages GPU‑only runs).
- Extending to pairwise RLHF: This RWSF baseline can be extended to DPO/RRHF/GRPO with pair generation or using TRL; the current implementation is intentionally simple, efficient, and works well with small GPUs.
