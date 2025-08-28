# Core Changes: Impact on Memory and Performance

This note captures the effects of two fundamental changes in K3D’s core:

1) Embedding data inside GLB bufferViews (vs. JSON extras/sidecars)
2) Adopting an RPN execution substrate for agent‑level numeric reasoning

The goal is to make tradeoffs explicit and guide where each approach should be used.

## 1) GLB BufferViews vs JSON Extras

Change: Embeddings and vectors are stored in binary `bufferView`s inside GLB (or data URI in glTF), and referenced in `primitive.extras.k3d` via indices (`vectorsView`, `embeddingsView`, `embeddingDims`). Previously, embeddings often traveled as JSON arrays inside `extras.k3d` or via a sidecar `.k3d`.

Why it helps:
- Size: Binary floats store exactly 4 bytes per value; JSON floats typically cost ~5–12 bytes plus separators. For moderate dimensionality, JSON payloads are ~5× larger.
- Parse time: GLB avoids JSON parsing of large numeric arrays; loaders can map binary directly to typed arrays.
- Streamability: BufferViews scale better to large scenes and progressive loading.

Measured (synthetic, embeddings only):

```
n=200  d=64   JSON 0.25 MB   GLB 0.05 MB   ratio ≈ 4.9×
n=2000 d=256  JSON 10.05 MB  GLB 2.05 MB   ratio ≈ 4.9×
n=10000 d=256 JSON 50.27 MB  GLB 10.24 MB  ratio ≈ 4.9×
```

Implication: For any non‑trivial embedding set, GLB bufferViews yield substantial memory and bandwidth reductions, and reduce parse/GC pressure in the viewer.

Recommendation:
- Use GLB with bufferViews for positions and embeddings (current default).
- Keep a lightweight `extras.k3d` index (ids, metadata, neighbors, indices to bufferViews).

## 2) RPN vs Direct Arithmetic

Change: Agent‑level numeric operations use an RPN (Reverse Polish Notation) stack machine (JS and Python) for deterministic, parse‑free execution. This unifies introspection and replay across environments.

Pros:
- Deterministic, stepwise execution with a small opcode set.
- Easy to instrument and log (good for imitation learning and debugging).
- Language‑portable (same behavior in JS and Python).

Cost: Throughput vs vectorized math. Micro‑benchmark (Python):

```
n=1000, d=64   direct ≈ 0.2 ms   RPN ≈ 211.9 ms   max diff ~ 3e-16
n=2000, d=64   direct ≈ 0.3 ms   RPN ≈ 375.0 ms   max diff ~ 3e-16
n=1000, d=256  direct ≈ 0.5 ms   RPN ≈ 734.7 ms   max diff ~ 2e-16
```

Interpretation:
- RPN is ~10^3× slower than vectorized NumPy for bulk cosine computations. Accuracy is identical (floating‑point noise).

Recommendation:
- Use RPN for agent decisions, explanation traces, and small compute steps where interpretability matters.
- Use vectorized math (NumPy/TypedArrays/WebGPU) for large‑scale operations (clustering, batch similarity, layout).

## Bottom Line

- Data: Move large numeric tensors to GLB bufferViews — it’s smaller, faster, and future‑proof.
- Compute: Use RPN for explainable, stepwise agent logic; reserve vectorized backends for heavy lifting.

