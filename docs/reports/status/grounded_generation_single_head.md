# Grounded Generation — Single‑Head Design (K3D)

This note documents a precise, single‑head approach for K3D that supports both
factual answering and creative generation while remaining grounded in the House
(Galaxy memory). No extra “creative head” is required.

## Single Head: GroundedGenerator
- One head, two modes: `compose` (non‑generative) and `generate` (generative).
- Always driven by retrieved contexts from the Galaxy (labels + snippets).
- Faith Engine gates the choice (navigate vs answer; compose vs generate) based
  on intent confidence and policy thresholds.

## Mode Logic
- `compose` (fast, deterministic):
  - Rank contexts (Answer Ranker) and stitch a concise answer.
  - No token generation; milliseconds latency; highest factual alignment.
- `generate` (grounded creativity):
  - Prompted LLM generation with strict instructions:
    - Use only provided contexts as factual bases.
    - Mark uncertainties; do not invent facts; prefer analogies tied to contexts.
    - Expose cited labels (IDs) used for each paragraph.
  - Creativity controls: `creative_strength` (0..1) tunes narrative vs. factual tone.

## RLWHF for Grounded Creativity
- Reward components (per response):
  - Truthfulness (context sim, cited label coverage): +1.0 / +0.5
  - Novelty (semantic distance between contexts; analogical mix): +0.5
  - Cohesion (intra‑answer embedding coherence): +0.25
  - Relevance (query ↔ answer sim): +0.5
  - Honesty (explicit uncertainty when weak grounding): +0.5
  - Penalties: off‑context claims (−0.5), uncited assertions (−0.25)
- Feedback `/fb good|partial|bad [gold]` augments scores; optional rubric scale
  (novelty/cohesion/relevance/truth) can be added later without changing logs.

## Training Recipe
- Supervised warm‑start: use RLWHF dataset rows as labeled pairs
  (query, ranked contexts) → target: higher weight to high‑reward rows.
- RL fine‑tune: PPO/ILHF where reward = weighted sum above.
- Continual: append session logs; rebuild RLWHF set; re‑rank contexts; small
  batch updates to keep latency and stability.

## Integration Points (Code)
- Retrieval and ranking:
  - `knowledge3d/skills/spatial_text.compose_answer` (now loads Answer Ranker)
  - `knowledge3d/models/answer_ranker.py` (trained from RLWHF dataset)
- Generative path:
  - `knowledge3d/skills/llm.LLMSkill.answer_with_rag(query, contexts, ...)`
    with a prompt template that enforces grounded creativity and exposes citations.
- Decision policy:
  - `knowledge3d/cranium/core.py: act()` — add a `creative_strength` knob and
    select compose vs generate based on user intent and faith thresholds.
- Feedback + logs:
  - `/fb ...` records in session JSONL; dataset builders convert to rewards.

## Evaluation
- Reports to produce per run:
  - Latency (compose vs generate), context alignment (ST sim), citation coverage,
    novelty score, human feedback tallies, RLWHF reward distribution.
- Store in `docs/reports/status/` and publish logs via tool.

This design keeps a single, inspectable head that can be both factual and
creative without sacrificing grounding, aligned with K3D’s House‑first memory.

