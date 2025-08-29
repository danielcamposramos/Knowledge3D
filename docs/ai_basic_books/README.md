AI Books – Basic Knowledge

This folder seeds the “AI Books” concept: human/AI knowledge curated as book-like JSON. The initial source is `EchoSystems_Humans_Compendium.json`.

Build outputs
- `data/ai_books_basic.json` — normalized schema (entries array with title, text, references, tags).
- `data/ai_books_basic.txt` — K3D-ready lines for `k3dgen --text`.
- `data/ai_care_multilang.txt` — tiny EN/PT/ES self‑knowledge care pack.

Generate
1) Normalize and emit datasets:
   - `python -m knowledge3d.tools.build_ai_books`

2) Produce a K3D GLB from text (embeddings via sentence-transformers):
   - `python -m k3dgen --text data/ai_books_basic.txt --gltf data/ai_books_basic.glb --k 5 --reducer umap`
   - Small care pack: `python -m k3dgen --text data/ai_care_multilang.txt --gltf data/ai_care_multilang.umap.glb --k 5 --reducer umap`

Notes
- K3D embeds node ids, vectors, embeddings, metadata, and neighbors in `meshes[*].primitives[*].extras.k3d` (no sidecar files).
- For tiny datasets, the reducer will fall back to PCA as needed.
- See `docs/k3d-research.md` and `docs/ROADMAP.md` for the environment context (where the AI “lives”).
- Viewer has PCA and UMAP houses pre-wired in `viewer/public/condo.json`, including a small `ai-care-umap` option.
- Live mode pause/resume is available from the viewer chat box:
  - `/pause <reason>` — suspend navigation/commands, keep chat/logging.
  - `/resume` — resume actions.
  - `/status` — show channel state.
  - Pause/resume events are logged to `../Knowledge3D.local/logs/` and appended to `docs/reports/advancement_log.md`.
