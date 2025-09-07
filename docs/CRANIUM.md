# K3D Cranium — Unified Multimodal Core

The Cranium is the AI Avatar’s active processing center. It exposes a single, cohesive API over all core capabilities — text, image, audio, spatial navigation, self‑reflection, and sleep‑time consolidation — without bolting a monolithic LLM onto the agent.

Design principles
- Memory‑first: Prefer K3D House memory (glTF + extras.k3d) and short‑term galaxy over parametric weights.
- Small brains, big world: Keep logic light; keep durable knowledge in 3D assets.
- One core, many skills: Compose skills behind one facade so upper layers treat the agent as “one brain”.
- Faith engine: Gate actions with confidence thresholds (>= 0.7 by default).

House/Galaxy/Cranium Mapping
- House = Disk: per‑avatar persistent 3D memory (`K3D_HOUSE_ID` segregates assets under `data/houses/<id>` and `viewer/public/houses/<id>`).
- Galaxy = RAM: short‑term embeddings and active thoughts (`ShortTermGalaxy`).
- Cranium = CPU: unified logic (`CraniumCore`), policy gating (`FaithEngine`, `DiaryPolicy`).

Key components
- `knowledge3d/cranium/core.py`: `CraniumCore` facade with `ask`, `parse`, `act`, `reflect`, `sleep_consolidate`, and `observe_*` for multimodal inputs.
- `knowledge3d/cranium/memory.py`: `ShortTermGalaxy` short‑term memory for recent observations and RAG contexts.
- `knowledge3d/skills/spatial_text.py`: Memory‑native answering from House snippets (no external LLM required).
- `knowledge3d/skills/vision.py` and `skills/audio.py`: Lightweight, optional embeddings for images/audio (graceful fallbacks).
- `knowledge3d/core/faith_engine.py`: Confidence‑gated action selector using RPN.

Live mode integration
- The live server now wires the Cranium:
  - Feeds user chat into STM (short‑term galaxy).
  - Extends `/ask` to use `CraniumCore.act()` (falls back to spatial text skill).
  - Adds `/brain reflect` and `/brain sleep [out]` for quick reflection and consolidation to the House GLTF.
  - Adds autonomous background behavior (idle navigation + reflection + link suggestions).
  - Enforces AI‑only Diary writes and event‑based policy (no generic dumping).

Sleep‑time consolidation
- `CraniumCore.sleep_consolidate()` writes recent STM notes to the House via `tools/house_memory.py`, exporting to `viewer/public/memory_house.gltf` by default.
- Existing `/sleep consolidate` remains and augments from diaries/reflections; the Cranium path is additive.

Usage (live server)
- Start the viewer and live server (see `docs/VISION.md`).
- In chat:
  - `/ask <text>` → unified Cranium answer + navigation gating.
  - `/brain reflect` → short self‑reflection.
  - `/brain sleep` → consolidate STM into `memory_house.gltf`.
Autonomy (idle behavior)
- Enabled by default. Env:
  - `K3D_AUTONOMY=1|0` — turn autonomy on/off.
  - `K3D_AUTONOMY_PERIOD_SEC` — check period (default ≈ 3.14×3 ≈ 9s).
  - `K3D_AUTONOMY_IDLE_SEC` — idle threshold before actions (default ≈ 37s).
- Faith threshold: `K3D_FAITH_THRESHOLD` (default ≈ 0.618).

AI Diary
- Concept: AI writes diary pages in its native language (embeddings) into a book object; humans read pages via server translation (retrieval → short text).
- Commands:
  - `/diary write [book_label]` — persist a diary page from STM into the House.
  - `/diary read [book_label] [page_id|label]` — translate a page for human reading.
- Autowrite (enabled by default):
  - `K3D_DIARY_AUTO=1|0` — turn periodic diary writing on/off.
  - `K3D_DIARY_PERIOD_SEC` — write cadence (default 600s).
  - `K3D_DIARY_BOOK` — target book label (default "AI Diary").

Policy and thresholds
- Diary writes are policy‑gated (novelty ≥ 0.382; feelings: good ≥ 0.618, bad ≤ 0.382).
- Autonomy defaults off; event‑based diary writes are preferred.
- No LLM fallback in the Cranium path; memory‑native composition is used for answers and translations.

Extending skills
- Add new skills under `knowledge3d/skills/` and call them from `CraniumCore`.
- Keep permanent knowledge in GLTF extras and use STM only for the “now”.
