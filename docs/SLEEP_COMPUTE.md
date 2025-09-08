# Sleep‑Time Compute — Consolidation and Garden Generation

Purpose
- During “sleep”, the Cranium consolidates short‑term observations (Galaxy memory) into persistent House memory. It also grows Knowledge Garden trees from meaning.

Inputs
- Galaxy snapshot: ids, positions (3D), embeddings, neighbors, labels (from `extras.k3d`).
- Logs: navigation/chat/events for optional temporal signals.

Outputs
- Updated House `memory_house.gltf` (rooms, objects, books, doors).
- `knowledge_garden.glb` with trees grown from meaning (embedded `extras.k3d`, explicit edges).
- Optional retrained small models (intent classifier; optional dynamics).

Garden Growth (model‑driven)
- The “garden model” runs at sleep to transform Galaxy meaning into trees:
  1) Cluster related concepts (embedding space; cosine similarity).
  2) Build hierarchy (roots → branches → leaves) via community detection or label ontology.
  3) Grow geometry (trunk/branches) toward cluster centroids using a space‑colonization procedure informed by similarity (thicker branches for stronger links).
  4) Export as GLB with `extras.k3d` (ids, vectorsView, embeddingsView, metadata, neighbors, edges).

Pipeline Sketch
- Cranium.sleep():
  - snapshot Galaxy → `../Knowledge3D.local/logs/`
  - run garden builder on snapshot → write `viewer/public/knowledge_garden.glb`
  - consolidate House notes/books/doors → write `viewer/public/memory_house.gltf`
  - (optional) retrain tiny intent model from fresh logs

Implementation Notes
- Builder can start with deterministic clustering (e.g., k‑NN graph + communities) and a simple growth algorithm; later swap with a learned generator.
- Keep GLB embedded: all semantics inside `primitive.extras.k3d`.

See Also
- docs/KNOWLEDGE_GARDENS.md — design details
- knowledge3d/tools/consolidate_from_galaxy.py — sample consolidation to House
- knowledge3d/tools/gardens.py — baseline Garden generator (ontology prototype)

