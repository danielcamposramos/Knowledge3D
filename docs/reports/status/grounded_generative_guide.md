# Grounded Generative — K3D Internal Paradigm

This guide explains how K3D generates answers using only House memory, and how to grow RLWHF datasets by merging real sources with controlled synthetic prompts (grounded, cite‑able, no external RAG).

## Principles
- Memory‑first: The House (glTF+extras.k3d) is the source of truth.
- Grounded only: Answers must be derived from provided contexts; admit “I don’t know” otherwise.
- Cite labels: Refer to the labels used (e.g., `(sources: labelA, labelB)`).
- Transparent: Artifacts and logs are published into the repo for audit.

## Two Answering Modes
- Compose (retrieval + composition):
  - Picks snippets via TF‑IDF and/or the Answer Ranker, then stitches a concise summary.
  - File: `knowledge3d/skills/spatial_text.py` (function `compose_answer`).
  - Strengths: very fast and highly grounded.
- Grounded Generative (internal):
  - Builds an instruction + context prompt, calls the internal LLM skill with a grounded policy.
  - File: `knowledge3d/skills/spatial_text.py` (function `compose_generate`).
  - Strengths: more free‑form expression, still grounded and cite‑able.

## Using Grounded Generative
- Programmatically:
  - `from knowledge3d.skills.spatial_text import compose_generate`
  - `answer = compose_generate(question, [(label, text), ...], max_tokens=256)`
- Chat (if enabled): switch `/llm` backend to a local adapter/policy, but we do not use external RAG.

## Growing RLWHF Datasets (Grounded)
- From GLB only (no external calls):
  - Tool: `knowledge3d/tools/rlwhf_from_glb.py`
  - Command: `scripts/k3d_env.sh run python -m knowledge3d.tools.rlwhf_from_glb --gltf viewer/public/galaxy.cross.glb --out docs/reports/training/rlwhf_dataset_glb.jsonl --queries 1000`
  - It samples labels, retrieves contexts, composes an internal answer, and assigns reward via similarity.
- From open RL prompts (future step):
  - Use prompts/questions from open RLHF datasets as “intent shells”; for each prompt, retrieve K3D contexts, then generate grounded answers with `compose_generate`.
  - This yields rows {query, answer, contexts[], reward} where all content is tied to real House memory.

## Training Internal Generative Policies
- RW‑SFT (small policy):
  - File: `knowledge3d/models/rlwhf_policy.py`
  - One‑liner: `scripts/train_rlwhf_policy.sh`
- LoRA adapters (paused for now):
  - File: `knowledge3d/models/rlwhf_lora.py`
  - We removed TinyLlama adapters per decision and retain the compose path as primary.

## Notes & Decisions
- No LLM+RAG: we do not use external retrieval; everything is in‑House.
- Compose is primary; grounded generative is auxiliary and uses only internal contexts.
- LoRA on TinyLlama is removed; future scaling will focus on better grounded data and compose/ranker upgrades.

## Artifacts
- Dataset: `docs/reports/training/rlwhf_dataset_glb.jsonl`
- Ranker: `../Knowledge3D.local/models/answer_ranker.pkl`
- Policy (small): `../Knowledge3D.local/models/rlwhf_policy`
- Benchmarks/Comparisons: `docs/reports/status/chat_benchmark_offline_*.json(md)` and `docs/reports/status/rlwhf_*comparison*.md`

This guide will evolve as we scale grounded datasets and improve compose + ranker strategies.
