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

