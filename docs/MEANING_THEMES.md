# 🧭 MEANING THEMES — TAILORED DATA INGESTION

Ground the habitat by curating data that matches the Galaxy’s geometry and the House’s rooms. Each theme specifies which modalities we want to ingest, where they live locally, and how they should flow into `extras.k3d` GLBs.

---

## THEME 1 · GALAXY GEOMETRY
- **Text** · PTX kernel docs, ray encoding specs, honesty scoring rules, embeddings/φ notes.
- **Image** · Ray diagrams, star embeddings, honesty heatmaps, galaxy visualisations.
- **Audio** · Spoken explanations of φ recursion, ray thickness, modality fusion.
- **Video** · Spatial chain-of-thought walkthroughs, sleep-time compute animations.
- **3D** · Canonical solids (tetrahedron, cube, octahedron, icosahedron, dodecahedron, hypersphere) and ray bundles.

**Recommended local sources**
- Text: `docs/CRANIUM_CORE.md`, `docs/HOUSE_MEMORY.md`, `docs/RPN_RUNTIME.md`, PTX specs.
- Image: `docs/images/*.png`, rendered galaxy previews.
- Audio: `spatial_web_k3d_discussion.mpga`, narrated explanations.
- Video: Live compute demos (place under `/home/daniel/K3D_llama_cpp/datasets/galaxy_geometry/video`).
- 3D: `viewer/public/house/workshop/*.glb`, `viewer/public/house/garden/*.glb` (core shapes).

---

## THEME 2 · HOUSE ITEMS — ZONE 5 (KNOWLEDGE GARDEN)
- **Text** · Book titles, tree growth logs, garden manifests, shape metadata.
- **Image** · Garden layouts, leaf/bark macro shots, book covers.
- **Audio** · Rustling leaves, fountains, page flips, ambient bird calls.
- **Video** · Tree growth time-lapses, book placement, knowledge garden tours.
- **3D** · Fractal trees, pedestals, bookshelf GLBs, garden artifacts.

**Recommended local sources**
- Text: `docs/KNOWLEDGE_GARDENS.md`, Zone 5 diaries.
- Image: `/home/daniel/K3D_llama_cpp/datasets/garden_images/` (add-on), `docs/images/cognitive_house.png`.
- Audio: `/home/daniel/K3D_llama_cpp/datasets/garden_audio/` (ambient captures).
- Video: `/home/daniel/K3D_llama_cpp/datasets/garden_video/`.
- 3D: `viewer/public/house/garden/*.glb`, `viewer/public/house/materialized_objects/*garden*.json` (convert to GLB).

---

## THEME 3 · HOUSE ITEMS — ZONE 7 (MIRROR ROOM)
- **Text** · Diary entries, self-critiques, reflection prompts, honesty scores.
- **Image** · Mirror selfies, honesty graphs, reflection diagrams.
- **Audio** · Whispered self-critiques, honesty affirmations, ambient mirror-room hum.
- **Video** · Self-reflection sessions, critique replays, honesty calibration walkthroughs.
- **3D** · Mirror GLBs, reflection spheres, diary stands, memory shards.

**Recommended local sources**
- Text: `docs/DIARY.md`, `viewer/public/house/materialized_objects/diary_*.json`.
- Image: Rendered honesty charts, to be exported under `/home/daniel/K3D_llama_cpp/datasets/mirror_images/`.
- Audio: `/home/daniel/K3D_llama_cpp/datasets/mirror_audio/` (to record/collect).
- Video: `/home/daniel/K3D_llama_cpp/datasets/mirror_video/`.
- 3D: `viewer/public/house/library/*.glb` (diary props), `viewer/public/house/workshop/mirror_*.glb`.

---

## PIPELINE · FOUR STEPS
1. **Download raw datasets** → HDD `/home/daniel/K3D_llama_cpp/datasets/<theme>/<modality>/...`
2. **Curate themed subsets** → SSD `/K3D/Knowledge3D.local/datasets/<theme>/<modality>/` (symlink to raw).
3. **Embed + produce GLBs** → run `build_theme_glbs(theme)` (exports stars with `extras.k3d`).
4. **Train** → `meaning_cluster_trainer --resume` (Galaxy stars mutate with real multi-modal corrections).

Keep logs per the Covenant, keep raw >99 MB assets off-repo with `.instruction` files, and grow the habitat meaning-first.
