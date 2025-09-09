# RLWHF Open Build — 2025-09-09 (9/9/9)

We built a 1,000‑row RLWHF dataset from open RL prompts, grounded strictly in K3D’s House memory.

## Output
- Dataset (JSONL): `docs/reports/training/rlwhf_dataset_open_1000.jsonl`
- Rows: 1000
- Row schema: `{query, answer, contexts[], reward}`

## Source prompts
- Dataset: `Anthropic/hh-rlhf` (harmless-base split)
- Loader: Hugging Face `datasets` library (installed in `k3dml`)
- Postprocessing: deduplicated prompts; filtered for length >= 16; capped at 1000 unique prompts

## Grounding & Answering
- House GLB: `viewer/public/galaxy.cross.glb`
- Context selection: TF‑IDF over labels + embedded snippet text
- Answer path: `compose` (retrieval + stitching, fully internal)
- Reward: cosine similarity between the answer and the concatenated contexts using ST `all-MiniLM-L6-v2` on GPU
  - mapping: `sim >= 0.70 → +1.0`, `0.40..0.70 → +0.5`, else `-0.25`

## Command used
```
scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_rl_open \
  --gltf viewer/public/galaxy.cross.glb \
  --out docs/reports/training/rlwhf_dataset_open_1000.jsonl \
  --n 1000 --dataset anthropic --mode compose
```

## Notes
- Grounded generative is also available: set `--mode generate` to use `compose_generate` (internal, cite‑able, honesty policy). For speed and reproducibility we used `compose`.
- This dataset is safe, auditable, and reproducible end‑to‑end with local House memory.
- Next: mix these rows with live logs to train improved rankers and, if desired, a small internal policy.

## Expansion (4k Anthropic + GLB)
- We scaled to 4,000 anthropic prompts:
  - `docs/reports/training/rlwhf_dataset_open_4000_anthropic.jsonl` (4000 rows)
- We also include GLB‑only grounded rows (200):
  - `docs/reports/training/rlwhf_dataset_glb.jsonl`

Unified RLWHF dataset (deduped by query):
```
scripts/k3d_env.sh run python -m knowledge3d.tools.merge_jsonl \
  --out docs/reports/training/rlwhf_dataset_unified.jsonl --dedup query \
  docs/reports/training/rlwhf_dataset_open_4000_anthropic.jsonl \
  docs/reports/training/rlwhf_dataset_open_1000.jsonl \
  docs/reports/training/rlwhf_dataset_glb.jsonl \
  docs/reports/training/rlwhf_dataset.jsonl
```
Result: 4,220 rows.

Retrained compose Answer Ranker on unified dataset:
```
scripts/k3d_env.sh run python -m knowledge3d.models.answer_ranker \
  --dataset docs/reports/training/rlwhf_dataset_unified.jsonl \
  --out ../Knowledge3D.local/models/answer_ranker.pkl
```
Info: `{samples: 25092, rows: 4220, mse: 0.1358, coef≈1.043, bias≈0.120}`
