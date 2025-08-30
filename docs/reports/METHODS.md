# Methods: K3D HR/MR Paradigm

This document captures the concrete formulas and methods we use to turn text into spatial knowledge, and how the Sleep/Wake loop consolidates long‑term memory.

- Embeddings: SentenceTransformer encodes lines into R^d (default: all‑MiniLM‑L6‑v2). Denote embeddings E ∈ R^{n×d}.
- Similarity: Cosine s(a,b) = a·b / (||a||·||b||). Used for step explanations and KNN linking in Memory House.
- Dimensionality reduction: UMAP (fallback PCA) projects E → P ∈ R^{n×3}. UMAP uses n_neighbors≈min(15, n−1).
- Neighbor graph: KNN over original embeddings; we store neighbor ids per node for routing.
- TF‑IDF search (open‑vocab goto): TfidfVectorizer builds label vectors; cosine used for best match; gazetteer canonicalization (NFKD + stopword/clitic removal) helps cross‑lingual.
- Doors: Special nodes (metadata.type=door, optional metadata.address) seed inter‑house navigation.
- LOD (screen‑space): pixelRadius ≈ (R / d) · pixelsPerUnit where R = bounding sphere radius and d = camera distance; devicePixelRatio and FOV included; hysteresis + ease‑in‑out fades for level transitions.
- Sleep/Wake loop: /sleep consolidate pauses; then merges diary/reflections/training into House Memory, adds KNN links among objects/doors (K=3) by cosine; exports embedded glTF. /resume wakes.

This HR/MR split ensures human‑readable artifacts (HR) and machine‑runtime graphs (MR) evolve together.
