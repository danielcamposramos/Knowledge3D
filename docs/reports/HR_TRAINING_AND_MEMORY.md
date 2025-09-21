# K3D Training + House Memory — Human-Readable Guide

This guide explains how we train the intent models and how we built a persistent House Memory (rooms + objects) that the agent can use as long-term memory, while the Galaxy view captures relevant context during live runs.

## What We Trained (Fast + Full)
- Tiny multilingual model (sklearn, ~65 KB): trains in seconds from templates (en/pt/es) + recent logs.
- HF multilingual baseline (DistilBERT): stronger generalization in ~1 min on CPU; optional.

Commands
- Fast loop (seconds): `make train-fast`
- Full loop (adds HF): `make train-full`
- Scoreboard refresh: `make scoreboard`
- Log evaluation: `make eval-logs`

Notes
- Live server logs predictions and ground-truth actions; we track confusion to validate model behavior.
- Confidence gating + ethics policy keep actions safe.

## House Memory (Rooms + Objects)
We added a persistent “House Memory” that stores rooms (e.g., “Books”) and objects (e.g., book titles). It exports as a K3D GLTF the viewer can load.

Files
- Tool: `knowledge3d/tools/house_memory.py`
- State: `data/memory_house.json`
- Export: `viewer/public/memory_house.glb`
- Viewer selector: `viewer/public/condo.json` (entry “memory”)

Usage
- Bootstrap: `python3 -m knowledge3d.tools.house_memory --bootstrap-books 24 --export viewer/public/memory_house.glb`
- Add room: `python3 -m knowledge3d.tools.house_memory --add-room "Research" "Long-term domain studies" --export viewer/public/memory_house.glb`
- Add object: `python3 -m knowledge3d.tools.house_memory --add-object "Research" "Vector databases" "Key patterns" --export viewer/public/memory_house.glb`

Design
- Rooms form a circle in 3D; each room has objects arranged around it.
- Metadata: `type: room|object`, `layer: rooms|<room>`, plus `label`, `desc|room`.
- Embeddings: small deterministic pseudo-vectors (32-dim) ensure basic similarity.
- Neighbors link each room to its objects.

## Live Mode: Memory Commands
We extended the live chat with `/mem` to update memory without leaving the session:
- `/mem room <name> [desc]` — create/update a room
- `/mem add <room>|<label>|<text>` — add an object into a room
- `/mem export` — write `viewer/public/memory_house.glb`

Switch to the “memory” house in the viewer dropdown to browse rooms + objects.

## AI Compendium (Repo Docs → Training Lines)
We created a parallel AI Compendium from local K3D docs/specs so the agent learns the project’s language and structure.

Files
- Tool: `knowledge3d/tools/build_ai_compendium.py`
- Outputs: `data/ai_compendium.json`, `data/ai_compendium.txt`
- Generated GLBs: `viewer/public/ai_compendium.1k.umap.glb`, `ai_compendium.1k.umap.doors.glb`

Usage
- Build: `python3 -m knowledge3d.tools.build_ai_compendium --target-lines 4000` (falls back to available lines)
- Generate GLB: `python3 -m k3dgen --text data/ai_compendium.txt --gltf viewer/public/ai_compendium.1k.umap.glb --k 5 --reducer umap`
- Add doors: `python3 -m knowledge3d.tools.mark_doors --input viewer/public/ai_compendium.1k.umap.glb --output viewer/public/ai_compendium.1k.umap.doors.glb --doors 32 --trail true`
- Select in viewer: expert `ai-compendium` or `ai-compendium-doors` in `condo.json`.

## Galaxy vs House
- House (rooms/objects) = long-term memory; curated and persistent.
- Galaxy (rings in Tablet) = active context; expands from current focus by neighbors/similarity using the working dataset.

This duality matches our HR/MR standard: HR artifacts document the system and memory; MR sources run the code and models.

## Sleep Mode (Pause + Consolidate)
- Commands in live chat:
  - `/sleep` — pauses the channel (agent stops acting).
  - `/sleep consolidate` — pauses and consolidates long-term memory:
    - pulls new diary/reflection/training artifacts
    - updates the Memory House GLTF (rooms, objects, door map)
  - `/resume` or `/wake` — resumes action.

- Diary: Tablet → Diary app; entries are stored locally and emitted to the live server. The server logs them to `docs/reports/diary/diary-YYYY-MM-DD.md` in GMT‑3. Use `--bootstrap-diary` to bring entries into the Memory House.

## Credits
- Human (Daniel) for vision, prompts, direction.
- AI partners (Codex, Perplexity AI, DeepSeek) credited in `docs/reports/contributors_credits.md`.
