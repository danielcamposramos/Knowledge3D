# Knowledge Gardens (Ontology Greenhouse)

Concept
- A covered greenhouse room where crystallized human + AI knowledge is organized as trees (roots → branches → leaves). It lives in fixed house space but expands internally as needed.
- Stores knowledge that exceeds a note/book/multimedia surface. Each training step can prune to books, notes, media, or persistent ontology trees.

Format
- Implemented as a K3D GLB with:
  - `ids`, `vectorsView` (3D positions), `embeddingsView` (dense vectors)
  - `metadata[i] = { label, type: 'root'|'branch'|'leaf', path }`
  - `neighbors` for navigation
  - `edges = [[parentId, childId], ...]` (explicit tree edges)

Rendering
- The viewer draws nodes as colored points and edges as translucent green line segments behind points to suggest plant stems.

Build Locally
```bash
python3 -m knowledge3d.tools.gardens --gltf viewer/public/knowledge_garden.glb
```

Use in Viewer
- Start: `cd viewer && npm run dev`
- Select: `knowledge-garden` in the expert dropdown.

Extending
- Edit `knowledge3d/tools/gardens.py` to add domains and branches. The script produces deterministic layout and embeddings for reproducibility.

Meaning‑Guided Fractal Trees (Design)
- Goal: The greenhouse should look and feel like trees of meaning.
- Approach: Use a fractal branching algorithm guided by semantic similarity so geometry reflects ontology and local neighborhoods.
  - Fractal base: start with an L‑system or the “space colonization” algorithm (Runions 2007). Colonization works well: place attraction points at child/neighbor node positions and grow branches toward them while avoiding collisions.
  - Meaning guidance: derive attraction points and branching weights from embeddings and k‑NN structure (neighbors in `extras.k3d.neighbors`). Higher cosine similarity → stronger attraction and thicker branches.
  - Branch width: proportional to subtree mass (number of descendants) and/or average similarity; taper toward leaves.
  - Layout seeding: initialize root trunk direction using PCA of child vectors to align trees with their semantic span.
- Export: persist final node positions as `vectorsView`; keep edges as explicit parent→child pairs. Optionally add a lightweight instanced branch mesh layer in the viewer for near‑field detail.

Planned Implementation (phased)
- Phase A (current): hierarchy→points with explicit edges (already available).
- Phase B: add colonization growth in `knowledge3d/tools/gardens.py` with parameters:
  - `--fractal colonization` (default) or `--fractal lsystem`
  - `--branch-max-angle`, `--branch-step`, `--taper`, `--thickness-scale`
  - `--respect-similarity` to weight growth by cosine similarity
- Phase C: optional viewer overlay for branch geometry using instanced cylinders/curves behind points.

Media Shapes (Galaxy alignment)
- Near‑field stars use modality‑aware shapes and rays (text/image/audio/video). See `viewer/src/shapes.ts`. The Garden can reuse these cues for leaves, so humans quickly read modality at a glance while AIs read embeddings directly.
