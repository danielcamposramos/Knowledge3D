# Care & Feeding Protocol (K3D Baby)

Audience
- Humans (developers, operators)
- Agents (other instances) — follow the same controls and checkpoints

Core Ideas
- Start small, celebrate progress, expand steadily.
- Keep memory/footprint light; prefer UMAP for production-scale; PCA for quick experiments.
- Document pauses and resumes to create a trustworthy development rhythm.

Controls
- Live Mode WebSocket: `python -m knowledge3d.bridge.live_server` (ws://127.0.0.1:8765)
- Viewer chat commands:
  - `/pause <reason>` — suspend actions (navigation/commands). Logging continues.
  - `/resume` — resume actions.
  - `/status` — show current state (paused/running).
  - `/ask-thoughts` — agent shares a brief reflection about the current house (logged as `reflection`).
  - `/whoami` — agent explains identity, role as bridge, and memory model.

Tablet (wallet)
- Purpose: keep the agent connected to its House even when the network drops.
- Stores: outbox queue and last graph/doors snapshot in IndexedDB.
- Status: shown in the viewer (“Tablet: online/offline, queue=n”).
- Behavior: messages queue offline and flush on reconnect.
- Logs
  - JSONL session logs: `../Knowledge3D.local/logs/session-<ts>.jsonl`
  - Advancement log: `docs/reports/advancement_log.md` (append-only, server updates on pause/resume)

Feeding Stages
- Care pack (identity/orientation): `ai-care-umap` (EN/PT/ES) — fast checks
- Small house: `ai-books-umap` (256)
- Medium house: `ai-books-1k-umap`
- Large house: `ai-books-4k-umap`
- Full house: `ai-books-full-umap`
- Guidance variant: `ai-books-1k-umap-doors` — door nodes + new-info mask for gentle paths
 - Scaled guidance: `ai-books-4k-umap-doors` for larger, structured walks

Generation
- Normalize books: `python -m knowledge3d.tools.build_ai_books`
- UMAP GLBs (examples):
  - `python -m k3dgen --text data/ai_books_basic.1k.txt --gltf data/ai_books_basic.1k.umap.glb --k 5 --reducer umap --emb-precision f16`
- Mark doors + guidance on a GLB:
  - `python -m knowledge3d.tools.mark_doors --input data/ai_books_basic.1k.umap.glb --output data/ai_books_basic.1k.umap.doors.glb --doors 24 --trail true`

Multilingual Plan
- See `docs/ai_basic_books/LANGUAGE_PLAN.md`.
- Add lines with language tags (e.g., `[pt]`, `[es]`, `[la]`) to `ai_care_multilang.txt` and regenerate `ai_care_multilang.umap.glb`.
- Ancient languages: start with transliterations; cuneiform is the sole pictographic exception (transliteration Phase 1; glyph demo optional later).

Agent Guidance
- Explain-as-you-move: announce plan, neighbor hops with cosine similarity, and arrival address.
- Respect confidence: default threshold ≥ 0.7 for action suggestions.
- Doors & masks: follow green nodes (has_new_information) and blue doors to explore purposefully.
 - Visual trails: agent leaves subtle line segments between hops for recaps; door tooltips show their spatial address.
