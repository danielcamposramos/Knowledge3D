Local Ollama Models — Roles and Use
===================================

Hosts
-----
- Primary: `http://192.168.0.4:11434` (RTX 3060 12GB)
- Secondary (slow, backup/parallel): `http://192.168.0.60:11434` (GTX 970 4GB)

Reasoning / “Thinking” Models
-----------------------------
- `deepseek-r1:14b` — chain‑of‑thought strength, good for stepwise reasoning. Use to generate or critique rationales and derive factual vs generative routing seeds.
- `exaone-deep` — robust reasoning + instruct style; pairs well with grounded context for concise explanations.
- `granite3.3:8b` — strong generalist; use when speed/VRAM is tighter than 14B.

Vision (Images)
---------------
- `qwen2.5vl:7b-q8_0` — image understanding; use to caption or answer questions about images (not video). Ground answers and trim to short, factual statements for embedding.

Embeddings / Reranking
----------------------
- `dengcao/Qwen3-Embedding-4B:Q4_K_M` — powerful 2D text embeddings; use to validate or enrich topic splits and similarity checks.
- `dengcao/Qwen3-Reranker-4B:Q4_K_M` — re‑rank candidate contexts for improved grounding.
- `embeddinggemma` — compact, reliable embeddings; good for quick checks or memory‑constrained boxes.
- Secondary host options: `snowflake-arctic-embed2`, `bge-m3`, `nomic-embed-text`, plus `deepseek-r1:1.5b` for light reasoning.

Operational Guidance
--------------------
- One‑at‑a‑time loading: load a model, finish its batch, unload before switching. This keeps VRAM use predictable and avoids swap thrash.
- Roles by task:
  - Topic‑coherent text generation: `exaone3.5:latest` or `granite3.3:8b`.
  - Honesty + error‑feedback RLWHF: `exaone3.5:latest` or `exaone-deep` with an instruction template enforcing “I don’t know” when context is insufficient.
  - CoT/rationale drafts for seeds: `deepseek-r1:14b` (fall back to `granite3.3:8b`).
  - Image understanding: `qwen2.5vl:7b-q8_0` (caption Q&A, no video).
  - Embedding checks / re‑ranking: `Qwen3-Embedding-4B`, `Qwen3-Reranker-4B`, `embeddinggemma`.

Toward “Super‑Embeddings”
--------------------------
- Idea: combine multiple embedding views of the same text/image into a compact shared space that captures the strengths of each model.
- Approach: collect aligned pairs `(text, e_model1, e_model2, …)` across models; learn a small projection/adapter that maps these into K3D’s shared space (rank‑constrained PCA/CCA or a tiny MLP). Evaluate by cross‑modal retrieval and policy reward gains.
- Cross‑model reading: teach the policy to interpret embeddings from different sources by training on aligned examples; verify that identical text maps to consistent neighborhoods across encoders.

Notes
-----
- Use the balanced expansion policy in `docs/EXPANSION_POLICY.md` when generating or selecting data.
- The secondary host is much slower; leverage it for background embedding/rerank passes or for parallel long‑running batches when the primary is busy.

