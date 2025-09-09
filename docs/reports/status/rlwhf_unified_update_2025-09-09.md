# RLWHF Unified Expansion — 2025-09-09

This note captures the expansion of grounded RLWHF data, OASST ingestion, unified dataset builds, answer ranker retrain, and the latest small policy evaluation. It also records the new Algorithmic Thinking dataset.

## New Datasets
- Anthropic grounded (compose):
  - `docs/reports/training/rlwhf_dataset_open_4000_anthropic.jsonl` (4,000)
- OpenAssistant grounded (compose):
  - `docs/reports/training/rlwhf_dataset_open_3000_oasst.jsonl` (3,000)
- Algorithmic Thinking RL (no contexts):
  - `docs/reports/training/rl_dataset_algo_2000.jsonl` (2,000)
- Honesty RL (no contexts):
  - `docs/reports/training/rl_dataset_honesty_1000.jsonl` (1,000)

## Build Commands
- OASST grounded (compose):
```
scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_rl_open \
  --gltf viewer/public/galaxy.cross.glb \
  --out docs/reports/training/rlwhf_dataset_open_3000_oasst.jsonl \
  --n 3000 --dataset oasst --mode compose
```
- Merge (v2):
```
scripts/k3d_env.sh run python -m knowledge3d.tools.merge_jsonl \
  --out docs/reports/training/rlwhf_dataset_unified_v2.jsonl --dedup query \
  docs/reports/training/rlwhf_dataset_unified.jsonl \
  docs/reports/training/rlwhf_dataset_open_3000_oasst.jsonl \
  docs/reports/training/rlwhf_dataset_open_4000_anthropic.jsonl
```
- Algorithmic Thinking RL:
```
scripts/k3d_env.sh run python -m knowledge3d.tools.build_algorithmic_thinking \
  --out docs/reports/training/rl_dataset_algo_2000.jsonl --n 2000 --mode rl
```
- Merge (v3, unified RL+RLWHF):
```
scripts/k3d_env.sh run python -m knowledge3d.tools.merge_jsonl \
  --out docs/reports/training/rlwhf_dataset_unified_v3.jsonl --dedup query \
  docs/reports/training/rlwhf_dataset_unified_v2.jsonl \
  docs/reports/training/rl_dataset_algo_2000.jsonl \
  docs/reports/training/rl_dataset_honesty_1000.jsonl
```

## Unified Dataset Sizes
- v2: 7,220 rows (compose‑grounded + OASST + Anthropic)
- v3: 9,220 rows (adds RL algorithmic + honesty)

## Compose Answer Ranker Retrain
```
scripts/k3d_env.sh run python -m knowledge3d.models.answer_ranker \
  --dataset docs/reports/training/rlwhf_dataset_unified_v2.jsonl \
  --out ../Knowledge3D.local/models/answer_ranker.pkl
```
- Info: `{samples: 40137, rows: 7220, mse: 0.1426, coef≈1.258, bias≈0.203}`

## Small Generative Policy (RW‑SFT)
- Retrained on unified v3 (RL + RLWHF combined):
```
scripts/k3d_env.sh run python -m knowledge3d.models.rlwhf_policy \
  --dataset docs/reports/training/rlwhf_dataset_unified_v3.jsonl \
  --out ../Knowledge3D.local/models/rlwhf_policy \
  --model distilgpt2 --epochs 2 --batch 4 --max_len 384 --lr 5e-5
```
- Eval (50 samples):
```
scripts/k3d_env.sh run python -m knowledge3d.tools.eval_rlwhf_policy \
  --dataset docs/reports/training/rlwhf_dataset_unified_v3.jsonl \
  --model ../Knowledge3D.local/models/rlwhf_policy \
  --out docs/reports/status/rlwhf_policy_eval_v3.json --limit 50
```
- Result (summary): `sim_avg≈0.351`, `sim_p50≈0.358` (context cosine). Compose remains stronger; the small head is an auxiliary.

## Notes
- Compose vs Compose‑Generate (routing plan):
  - Compose → RL (honesty; no contexts)
  - Compose‑Generate → RLWHF (with contexts; grounded creative)
- Next: add a learned selector to let the “one head” choose between compose and compose‑generate. For now, we keep compose as primary and enable compose‑generate for grounded creative tasks.
- Sleep‑time compute: After policy training (~912s), we ran an equivalent "sleep compute" window. As we progress, we’ll make this consolidation step integrate House/Diary reflection tasks.
