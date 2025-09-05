Evaluation Summary — 2025-09-05

Retrieval (kNN)
- 6k Foundation (dims=384, k=10, queries=512): recall@10 = 1.000, candidate (IVF) time = 0.22 s, truth time = 0.04 s
- 80k Compendium (dims=384, k=10, queries=512): recall@10 ≈ 0.9994, candidate (IVF) time ≈ 23.84 s, truth time ≈ 0.23 s

Notes
- IVF accuracy is near‑perfect at k=10 on 80k with our heuristic nlist/nprobe.
- GPU UMAP was used for 3D embedding; FAISS IVF was on CPU (guarded) for stability.

Routing (pairs=256; undirected adjacency for fairness)
- 6k:
  - BFS: success=0.988, median hops=7, avg=3.41 ms/route
  - A*: success=0.988, median hops=14, avg=27.60 ms/route
  - A* LOD: success=0.984, median hops=17, avg=26.90 ms/route
- 80k:
  - BFS: success=1.000, median hops=8, avg=143.31 ms/route
  - A*: success=1.000, median hops=27, avg=565.09 ms/route
  - A* LOD: success=0.977, median hops=34, avg=562.29 ms/route

Interpretation
- On unit‑cost graphs, BFS optimizes hop count by design and is faster. A* uses geometric heuristics that do not perfectly match topological cost, so it explores longer routes.
- A* (and A* LOD) remain valuable when the goal is geometric guidance or when future edges are weighted by similarity/geometry. LOD reduces expansions; additional gains are expected once edge weights reflect semantic/geometric costs.

Next steps
- Add edge weights (e.g., 1 - cosine(emb)) and evaluate A* on weighted graphs.
- Introduce IVF‑PQ for faster kNN at very large N and run per‑query latency benchmarks.
- Extend per‑hop traces to include embedding cosine where available.
