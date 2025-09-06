Assessments and Metrics (Plain Language)

- HF Intent Evaluation
  - What it means: We hold back a portion of examples and check how often the model predicts the right action (goto/show/move…)
  - Why it matters: Shows the model understands phrasing in different languages. Very high scores on templates are expected; we validate with real chat logs to avoid overfitting.

- Routing Scoreboard (Viewer Training Panel)
  - What it means: The success rate and median hops for reaching labels/doors in a given house.
  - Why it matters: Measures the connectivity/quality of the neighborhood graph. Fewer hops and higher success indicates better UMAP layout, FAISS neighbors, and door placement.

- Acceleration Logs
  - What it means: Messages confirming which path ran (e.g., “UMAP via RAPIDS cuML (GPU)” or “FAISS IVF‑PQ (GPU)”).
  - Why it matters: Confirms we’re taking GPU fast paths; helps debug fallback situations.

Comparison: K3D Spatial Memory vs. GPT‑OSS High‑Dimensional Memory

- GPT‑OSS:
  - Knowledge primarily inside the model weights; retrieval comes from attending over internal states & KV cache; optional RAG adds external docs.
  - Strengths: General reasoning, long‑context pattern synthesis.
  - Trade‑offs: Heavier to update; limited transparency of “why this answer?”

- K3D:
  - Knowledge stored externally in a 3D space (embeddings + metadata). The model learns to navigate/query this space (small swappable logic).
  - Strengths: Easy updates (add rooms), transparent paths (hop‑by‑hop traces), scalable shared memory for multi‑agents.
  - Trade‑offs: Requires good spatial neighborhoods (UMAP/FAISS) and policy learning for navigation.

How They Complement
- GPT‑OSS can provide rich linguistic reasoning; K3D provides transparent, scalable, shared memory and action planning. Combined, they form a powerful system with both reasoning and grounded memory navigation.
