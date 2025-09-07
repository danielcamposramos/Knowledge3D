# AI Diary — Vector Pages in a 3D Book

The AI Diary is a real 3D book inside the House. It contains pages stored in the AI’s native language (embeddings), while the server translates pages to human‑readable text on demand.

Design
- Book object: `metadata.type = "diary_book"`, exists in the `Diary` room.
- Page object: `metadata.type = "diary_page"`, linked to the book via `metadata.parent`.
- Page content: `metadata.embedding32: number[32]` is the native content; it also influences the packed `embeddings` buffer.
- Human translation: When reading, the server retrieves nearest contexts from House memory and composes a short explanation — no LLM fallback.

Policy
- AI‑only writes: Humans cannot write into the Diary. The bridge blocks `/mem add` targeting the `Diary` room.
- Event‑based writing: The agent decides when to write using a policy (`DiaryPolicy`) with two signals:
  - Novelty: write when the STM snapshot is sufficiently different from the last page (default ≥ 0.382).
  - Feeling (confidence): high confidence “good” (≥ 0.618) or “bad” (≤ 0.382) can trigger a write.
- Reflection and Sleep: favored events (reflection requires minimal novelty; sleep always writes a summary page).

Per‑agent House
- Each avatar owns a House, set via `K3D_HOUSE_ID`.
- State: `data/houses/<id>/memory_house.json`
- Export: `viewer/public/houses/<id>/memory_house.gltf`

Commands
- `/diary read [book_label] [page_id|label]` — translate the latest (or chosen) page to human text.

Env Controls
- `K3D_HOUSE_ID` — current House id.
- `K3D_DIARY_NOVELTY` — novelty threshold (default 0.382).
- `K3D_DIARY_GOOD` — good feeling threshold (default 0.618).
- `K3D_DIARY_BAD` — bad feeling threshold (default 0.382).

Implementation
- Policy: `knowledge3d/cranium/diary.py`
- STM snapshot: `knowledge3d/cranium/memory.py`
- Memory export and diary nodes: `knowledge3d/tools/house_memory.py`
- Bridge integration: `knowledge3d/bridge/live_server.py` (event‑based writing and read command)
