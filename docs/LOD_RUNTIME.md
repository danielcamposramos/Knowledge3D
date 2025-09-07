# Spatial LOD for K3D (Positions Only)

Goal: embed multiple 3D layouts for the same knowledge graph in a single GLB and let the client ("game engine") choose which one to render based on distance/perf — just like graphical LOD.

What’s inside the GLB
- `primitive.attributes["POSITION"]`: default positions (e.g., PCA). This is what viewers see if they ignore LODs.
- `extras.k3d.lods`: list of alternative position sets, each with a name and a `vectorsView` index pointing to an extra bufferView that holds positions.
- One embeddings buffer (`embeddingsView`) and one neighbor graph shared by all LODs (semantics stay constant; only display geometry changes).

LOD presets (accelerator)
- `pca`: very fast/overview (default POSITION recommended for far distance).
- `umap_fast`: faster/coarser UMAP (good mid‑range detail).
- `umap_high`: higher‑fidelity UMAP (good close‑up views).

Generator flags (k3dgen)
- `--reducer <pca|umap|umap_fast|umap_high>`: chooses the default POSITION.
- `--lod-levels <comma list>`: extra layouts to embed as LODs. Example: `--lod-levels umap_fast,umap_high`.
- Result: `extras.k3d.lods = [{ name, method, vectorsView }]`.

Client guidance (viewer/engine)
- Near: use `umap_high` for local structure fidelity.
- Medium: use `umap_fast` for mid detail.
- Far: use default POSITION (typically `pca`) for fast overview.
- Swap strategy: when camera distance thresholds change, rebind POSITION to the selected `vectorsView` from `extras.k3d.lods` (or copy the chosen buffer once per LOD change).

Why not per‑LOD neighbors?
- To reduce size/complexity, the GLB has one neighbor graph. Optional per‑LOD neighbors can be added later if needed for route costs to match geometry at each level.

