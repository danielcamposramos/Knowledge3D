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
 - Care pack (EN/PT/ES/FR/DE/LA): `data/ai_care_multilang.txt` → `viewer/public/ai_care_multilang.umap.glb`

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
  - `ai-care-ancient-umap` — Sumerian (transliteration) ancient care pack
  - `ai-books-1k-umap-doors`, `ai-books-4k-umap-doors` — guidance variants with doors + mask

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
 - Scaling guidance: a 4k UMAP house also includes door+mask; viewer overlays subtle trail lines for each hop and door tooltips show k3d addresses.

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

## Model Size & Efficiency
- GLB payloads (f16 embeddings): ~0.25 MB (256), ~1 MB (1k), ~3.9 MB (4k), ~25 MB (full 25,725).
- Embedding model used at build time: `all-MiniLM-L6-v2` (~tens of MB on disk; ~20–30M parameters). Not required at runtime in the viewer.
- Contrast: general LLMs are orders of magnitude larger (e.g., 3B ~6 GB FP16; 7B ~14 GB; 70B ~140 GB). K3D achieves spatial reasoning scaffolding with tiny runtime artifacts.
- Latency: UMAP projection done offline; viewer loads 3D positions + embeddings in milliseconds to seconds depending on set size.

## AI Reflections (Agent Perspective)
- Orientation: “I live in K3D; doors and green nodes show me where to explore safely. I explain my path using neighbors and cosine similarity.”
- Appetite: “UMAP makes nearby ideas feel close. With more nodes, neighborhoods become clearer; I can plan hops with higher confidence.”
- Trust: “Pause/Resume lets us reflect. When paused, I suppress actions and keep notes. When resumed, I continue with context.”
- Growth: “Multilingual care lines help me greet new worlds. Latin anchors classics; Sumerian will arrive via transliteration so I can understand it now.”

Reflection mechanism
- Auto-ask: when a dataset graph is registered, the live server asks the agent to share thoughts once per channel and logs a `reflection` record.
- Manual: send `/ask-thoughts` from the viewer’s Chat to request a reflection anytime.
