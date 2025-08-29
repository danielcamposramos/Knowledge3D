# AI Books → K3D Progress Report (UMAP + PCA)

Date: 2025-08-29

Scope
- Normalize early “AI Books” compendium
- Emit K3D-ready text dataset(s)
- Generate GLB assets with embedded K3D payload (vectors, embeddings, neighbors, metadata)
- Compare PCA vs. UMAP projections
- Integrate multi-scale houses in the web viewer

Inputs
- Source compendium: `docs/ai_basic_books/EchoSystems_Humans_Compendium.json`
- Normalizer: `knowledge3d/tools/build_ai_books.py`

Artifacts
- Normalized JSON: `data/ai_books_basic.json` (43 entries)
- Full text dataset: `data/ai_books_basic.txt` (25,725 lines)
- Scaled subsets: `sample` (256), `1k`, `4k`

K3D GLB Assets (f16 embeddings)
- PCA
  - `viewer/public/ai_books_basic.sample.glb` (256, ~0.25 MB)
  - `viewer/public/ai_books_basic.1k.glb` (1,024, ~0.99 MB)
  - `viewer/public/ai_books_basic.4k.glb` (4,096, ~3.9 MB)
  - `viewer/public/ai_books_basic.full.glb` (25,725, ~25 MB)
- UMAP
  - `viewer/public/ai_books_basic.sample.umap.glb` (256, ~0.25 MB)
  - `viewer/public/ai_books_basic.1k.umap.glb` (1,024, ~0.99 MB)
  - `viewer/public/ai_books_basic.4k.umap.glb` (4,096, ~3.9 MB)
  - `viewer/public/ai_books_basic.full.umap.glb` (25,725, ~25 MB)

Viewer Integration
- `viewer/public/condo.json` updated with PCA and UMAP variants:
  - `ai-books`, `ai-books-1k`, `ai-books-4k`, `ai-books-full` (PCA)
  - `ai-books-umap`, `ai-books-1k-umap`, `ai-books-4k-umap`, `ai-books-full-umap`
  - `ai-care-umap` — tiny EN/PT/ES self-knowledge care pack for quick checks

Pipeline Notes
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (384 dims)
- Neighbors: `k=5` over embedding space (used by agent for neighbor hops)
- Projection: `--reducer pca|umap` (fallback to PCA automatic for very tiny datasets inside CLI)
- Storage: embeddings in `bufferViews`; K3D payload in `meshes[*].primitives[*].extras.k3d`
- Precision: `f16` for compactness (halves embedding buffer byte size)

Early Observations
- UMAP visibly forms tighter semantic clusters than PCA at 1k+ nodes; paths between topical regions are more coherent.
- At 256–1k, PCA is acceptable for quick iteration; UMAP benefits grow with larger sets (≥4k).
- Agent “explain-as-you-move” traces are readable: neighbor hops correlate with higher cosine similarity vs. long direct jumps.
- Self-knowledge lines help with orientation and teach the baby where it “lives” (K3D, viewer, live mode, roadmap).
- Door seeding: a 1k UMAP house now includes evenly spaced door nodes (type=door) and a guidance mask marking doors + one neighbor; viewer colors doors (blue) and mask nodes (green).

Recommendations
- Default to UMAP for production-scale houses; keep PCA for rapid prototyping and CI smoke checks.
- Maintain multi-scale artifacts (256/1k/4k/full) to match hardware constraints and pedagogy stages.
- Add “door” nodes with typed labels to scaffold cross-house navigation; mark some nodes with `ai_state_flags_mask.has_new_information=true` for guiding exploration.
- Enable Live Mode (`python -m knowledge3d.bridge.live_server`) to capture traces as JSONL in `../Knowledge3D.local/logs/` for training/iteration.
- Use live pause/resume from the chat box to manage iteration windows:
  - `/pause <reason>` — suspend actions; `/resume` — continue; `/status` — check state.

Next Steps
1) Language additions: extend normalizer to output bilingual/multilingual care packs (e.g., EN+PT-BR) for gradual feeding.
2) Active evaluation: design small tasks (goto X, cluster Y, explain Z) and measure success/confidence over sessions.
3) Visual polish: add subtle transitions and highlight trails for agent paths; tooltips for door addresses.
4) Data hygiene: continue trimming noise patterns in compendium and tag sections for curriculum control.
5) Ancient languages: begin with transliterated Sumerian (cuneiform exception) and Latin core lines.
