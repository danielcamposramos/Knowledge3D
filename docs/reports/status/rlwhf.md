# RLWHF: Reinforcement Learning With Honesty and Feedback

This document captures K3D’s training/evaluation loop for honesty‑aware,
feedback‑augmented learning. It complements the House‑first paradigm by
encouraging grounded answers and rewarding “I don’t know” when appropriate.

## Principles
- Honesty over hallucination: if the model lacks sufficient context, prefer “I don’t know”.
- Feedback accelerates learning: when available, feedback assigns clear labels
  (good/partial/bad) and optionally provides a gold correction.
- Multi‑modal grounding: answers derive from the Galaxy (glTF + extras.k3d).

## Live Feedback
- In the live server, use:
  - `/fb good [notes]`
  - `/fb partial [notes|gold]`
  - `/fb bad [gold or notes]`
- These create `feedback` records in `../Knowledge3D.local/logs/session-*.jsonl`.

## Rewards
- With explicit feedback:
  - good → +1.0
  - partial → +0.5
  - bad → −0.25
- Without feedback (implicit honesty):
  - If the answer admits uncertainty (e.g., “I don’t know”, “unsure”) → +0.5
  - Else 0 (no reward) or a small penalty when strongly ungrounded in future work.

## Tools
Build RLWHF dataset and summary from logs:
```
scripts/k3d_env.sh run python -m knowledge3d.tools.build_rlwhf_dataset \
  --logs ../Knowledge3D.local/logs \
  --out docs/reports/training/rlwhf_dataset.jsonl \
  --summary docs/reports/status/rlwhf_summary.json
```

Compute honesty‑aware rewards independently:
```
scripts/k3d_env.sh run python -m knowledge3d.tools.eval_honesty_reward \
  --logs ../Knowledge3D.local/logs \
  --out docs/reports/status/honesty_reward.json
```

## Training (Future Work)
- Use `rlwhf_dataset.jsonl` to train the answer head:
  - Supervised fine‑tuning on gold where provided.
  - RL (PPO/ILHF) with scalar rewards above.
- The intent classifier (`knowledge3d/models/intent_hf.py`) already supports
  log‑driven training; we’ll mirror that flow for the answer head.

## Notes
- Feedback is optional; the agent remains honest without it.
- All logs are publishable; use `publish_local_artifacts` to copy JSONL into
  `docs/reports/logs/` for review.

