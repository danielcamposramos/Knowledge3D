Plain-English Results — 2025‑09‑10
=================================

What we ran
-----------
- Built a new, modality‑balanced Galaxy (v7) with equal counts for text (exaone) and 3D (glTF samples). See `docs/reports/status/balanced_galaxy_v7.md`.
- Started the live server with compose_auto routing; seeded prompts via local Ollama (exaone3.5).
- Trained the small Mode Selector on the live outcome logs.
- Expanded RLWHF training set to v6 and trained a small policy (distilgpt2), then evaluated on a 100‑item slice.

Key numbers and what they mean
------------------------------
- Mode Selector training: Accuracy 1.000 on the held‑out fold, but with a catch: the log slice only contained one label (`compose_generate`).
  - Plain English: it got 100% on a test set that only had one type of decision. That’s not meaningful yet. We need more diverse, factual prompts so some cases clearly favor `compose` (retrieval + stitching). We’ll seed more factual queries next and retrain.

- RLWHF policy eval: sim_avg ≈ 0.285 (cosine similarity on embeddings, n=100).
  - Plain English: this is early‑stage alignment. Random answers would cluster near 0; strong alignment for this setup would be ≳0.4. We trained a tiny model for two epochs; with more balanced, grounded contexts (from the balanced Galaxy) and longer training, the score should climb.

Low‑Dim, High‑Density Memory vs Traditional
-------------------------------------------
- Traditional approach: high‑dimension, low‑density (e.g., 768–1024D per node; separate indices per modality). Pros: high capacity per embedding. Cons: heavy memory/ANN costs; weaker cross‑modal alignment unless explicitly bridged; more duplication.
- K3D approach: low‑dimension, high‑density (target ~256D shared space; LOD + galaxy unification). Pros: lighter GPU kNN, faster routing, one place for every modality to meet, stronger cross‑modal neighborhoods. Cons: if the set is tiny, PCA rank limits the effective dims (e.g., n=110 → 109D). As data grows, the effective dimension rises to the target while staying compact.

Why this matters
----------------
- The balanced Galaxy makes it easier to compare apples‑to‑apples across media types. With equal counts and coherent topics, we can measure whether neighbors and paths stay semantically consistent (text ↔ 3D) under tight dimensions.
- This reduces reliance on large LLM weights for “memory.” Instead, we preserve knowledge as geometry + embeddings in one House/Galaxy and use small models to reason over it.

Next steps
----------
1) Seed more factual prompts (exaone3.5) to balance `compose` vs `compose_generate` in outcome logs, then retrain the Mode Selector.
2) Extend v7 balancing to audio/video/images using the existing ingest tools; keep topics aligned (e.g., vehicles, sports, gardens).
3) Retrain the small RLWHF policy on the v7‑grounded data and re‑evaluate. Track sim_avg and p50 over time.

