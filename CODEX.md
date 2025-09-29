# Codex Tasks

This file lists actionable tasks for AI agents working on the Knowledge3D repository. All tasks are derived from the official **[Project Roadmap](docs/ROADMAP.md)** and should be executed in accordance with the vision outlined in the **[K3D Research Report](docs/k3d-research.md)**.

⚠️ **Memory Policy Reminder**: every task touching Galaxy, House, or Museum must follow [`docs/HOUSE_GALAXY_TABLET.md`](docs/HOUSE_GALAXY_TABLET.md). The Memory Tablet is now the primary interface for consolidated knowledge.

---

## Phase 1 Tasks: The MVP - The Static Knowledge Graph (Historical)

**Objective:** Prove the core data-to-spatial pipeline by creating a non-interactive, read-only visualization of a knowledge corpus.

1.  **Define Core Schema:**
    -   [ ] **Task:** Create and finalize the `spec/k3d_node_schema.json` file.
    -   **Details:** The schema must be robust and extensible, incorporating principles of spatial and propositional knowledge representation as discussed in the research report.

2.  **Implement Data-to-Spatial Pipeline:**
    -   [ ] **Task:** Enhance the `k3dgen` Python tool.
    -   **Details:**
        -   Integrate `umap-learn` to replace the existing PCA implementation for dimensionality reduction.
        -   Add functionality to process raw text data using embedding models from the Hugging Face library (e.g., BERT).
        -   Output a single embedded `.gltf` file where `primitive.extras.k3d` carries ids, vectors, embeddings, metadata, and neighbors. Sidecar `.k3d` is deprecated.

3.  **Develop Static 3D Visualization:**
    -   [ ] **Task:** Create an initial MVP viewer.
    -   **Details:**
        -   Load the embedded `.gltf` output directly and construct point clouds from `extras.k3d.vectors`.
        -   Implement basic camera controls to navigate the static point cloud.

4.  **Update Documentation:**
    -   [X] **Task:** Align core project documents with the new vision.
    -   **Details:** Update `README.md`, `docs/ROADMAP.md`, `AGENTS.md`, and `CODEX.md`.

---

## Phase 2 Tasks: The Interactive Agent (In Progress)

**Objective:** Transform the passive viewing experience into an active, conversational one by integrating an embodied AI agent.

1.  **Integrate Embodied AI Agent:**
    -   [ ] **Task:** Add an agent to the viewer.
    -   **Details:** Provide a visible avatar and minimal policy (seek-by-label, neighbor traversal), with a UI to issue simple commands (e.g., “go to <label>”). Later integrate voice/LLM when ready.

2.  **Develop Agent World Model:**
    -   [ ] **Task:** Implement the agent's spatial reasoning capabilities.
    -   **Details:** Create data structures that allow the agent to understand the location, properties, and relationships of the knowledge nodes in its environment. Ensure the fused head consumes the house-memory index produced during SleepTime before querying modality galaxies.

3.  **Implement Basic Interactivity:**
    -   [ ] **Task:** Develop the user-to-agent interaction loop.
    -   **Details:** Implement voice command recognition and the ability for the agent to provide multimodal feedback (e.g., speaking while highlighting a cluster of nodes).

4.  **AI Diary (Vector‑Native):**
    - [ ] Add viewer UI to open a diary book, list pages, and call `/diary read` to render translated text.
    - [ ] Enforce AI‑only writes; ensure `/mem add` cannot target the `Diary` room.
    - [ ] Add “promote to Garden” action to curate a diary page into the Knowledge Garden.
    - [ ] Surface diaries through the tablet inventory so the avatar can review consolidated reflections without leaving the workspace.

5.  **Doors & Address Bar:**
    - [ ] Add an address bar UI to door meshes; integrate `/open <label|k3d://...>` and show route previews.
    - [ ] Provide helpers to create inter‑House (LAN) doors given two `K3D_HOUSE_ID`s.

---

## Phase 3 Tasks: The Collaborative Knowledge Habitat (Upcoming)

**Objective:** Expand the single-user experience into a fully-featured, multiplayer, real-time collaborative environment.

1.  **Transition to High-Fidelity Engine (Optional but Recommended):**
    -   [ ] **Task:** Evaluate and potentially migrate the project to Unreal Engine.
    -   **Details:** The research report suggests Unreal Engine for its superior graphics and scalability for large, collaborative environments.

2.  **Implement Multiplayer Functionality:**
    -   [ ] **Task:** Add networking capabilities to the viewer.
    -   **Details:** Use a networking solution (e.g., Photon, Netcode for GameObjects) to enable multiple users to share the same space.

3.  **Build Collaborative Tools:**
    -   [ ] **Task:** Develop features for shared interaction.
    -   **Details:** Implement tools for real-time annotations, data manipulation, and other collaborative activities.

## Phase 4 Tasks: Memory Tablet & Dual-Space Operations

**Objective:** Make the House “disk” and Galaxy “RAM” operate through the Memory Tablet, with SleepTime consolidating and relocating knowledge automatically.

1. **House Memory Builder**
    - [ ] **Task:** Extend SleepTime to emit a PTX-ready `house_memory.glb` + manifest summarising consolidated artifacts (books, diaries, learning insights, dream records).
    - **Details:** Mirror `learning_memory_builder` but source `viewer/public/house/materialized_objects/`. Tag nodes with artifact type and zone for tablet filtering.

2. **Tablet Viewer & Search**
    - [ ] **Task:** Implement the 3D tablet UI in the viewer. Support search, filters, LOD toggles, and on-demand loads from House → Galaxy.
    - **Details:** The tablet must query the house-memory index first, show provenance (Galaxy / House / Museum), and expose actions to promote, archive, or relocate artifacts.

3. **Browser Bridge**
    - [ ] **Task:** Integrate the Firefox-based (or existing) browser container with the tablet so the avatar can open legacy web pages and capture them as structured notes.
    - **Details:** Each browsing session should produce a tablet note that SleepTime can consolidate into the House and tag with source metadata.

4. **Prompt Hygiene & Verification Loop**
    - [ ] **Task:** Remove nonsensical prompts (e.g., “What happened in 000?”), retire prompts after two perfect runs, and ensure every session logs timezone-aware timestamps.
    - **Details:** Update Phase18/Phase25 trainers to maintain mastered vs active pools, record retirement decisions, and follow `docs/TRAINING_DIRECTIVES.md`.

5. **Museum Relocation Automation**
    - [ ] **Task:** Automate calls to `relocate_to_museum` (or a refactored helper) whenever SleepTime supersedes an artifact.
    - **Details:** Deprecation records should include `relocated_at`, `previous_zone`, and a pointer to the superseding artifact. The tablet must surface these in “Museum mode”.

6. **LOD Streaming Pipeline**
    - [ ] **Task:** Implement game-style LOD tiers for memory streaming (coarse centroids, medium embeddings, full GLBs).
    - **Details:** Respect GPU budgets; allow the tablet to request upgrades/downgrades per artifact. Expose current LOD state to the fused head so PTX kernels know what data is loaded.

7. **Telemetry & Confidence**
    - [ ] **Task:** Instrument tablet interactions so we can measure how often the avatar consults House vs Galaxy vs Museum, including timezone-aware timestamps.
    - **Details:** Store telemetry alongside learning-memory entries to help tune SleepTime thresholds and critical reflection.

---

## Ongoing Research Tasks

-   **XAI Integration:** Investigate and implement methods like SHAP to provide explanations for data patterns.
-   **Knowledge Graph & RAG:** Research and develop a robust backend using a knowledge graph and a Retrieval-Augmented Generation (RAG) system.
-   **3D Data Standards:** Continue to monitor and align with standards like glTF, OpenUSD, and 3D Tiles.
-   **Graph Database Standards:** Continue to align K3D's data model with standards like openCypher and RDF.
-   **Phase 25 Sleep Consolidation:** Keep the `k3d-cranium` CUDA toolchain (with `cuda-python`) ready so `SleepTimeCompute` continues persisting Phase 25 knowledge into the House. Verify that each cycle rebuilds both `learning_memory.glb` and `house_memory.glb`, absorbs trusted teacher feedback immediately, and records timezone-aware timestamps.
-   **Lexicon Refinement:** Rebuild Q&A material from the cleaned book sources using exaone3.5 with full-document context to avoid page-break artifacts (e.g., `ma- chine`).
-   **Galaxy Coverage:** Ingest the balanced EN/ES/PT_PT/ZH Wikipedia splits into `viewer/public/galaxy/working/` while keeping mastered prompts out of the active drill queues.
-   **Time & Math Enrichment:** Pull curated time and math corpora from `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/...` directories, augment with exaone models, and feed them through the Phase18/Phase25 trainers following `docs/TRAINING_DIRECTIVES.md`.
-   **Thinking Tags Visibility:** Run the Phase 10 thinking-tag trainer after long RLWHF sessions to keep reasoning labels exposed during evaluation.
-   **Generalisation Benchmarks:** After consolidation, measure zero-shot math reasoning on `Maxwell-Jia/AIME_2024` to confirm the model generalises beyond retrievable content.

---

## Assets

- **Location:** Store all visual assets in `docs/images/`.
- **Naming:** Use descriptive, kebab-case names, e.g., `cognitive_house.png`, `avatar_workshop.png`.
- **Prompt Pairing:** For each asset, add a matching prompt file named `<image>_prompt.md` containing the generation prompt and any parameters.
- **Referencing:** When adding an asset, embed it in relevant docs (`README.md`, `docs/k3d-research.md`, `docs/ROADMAP.md`, `AGENTS.md`) and link to its prompt file.
- **Attribution & License:** Ensure images comply with the repo license or include source/usage notes in the prompt file if different.

---

## Recent Implementation Notes (2025‑09‑10)

- Live server stability: fixed an event‑loop starvation issue in log maintenance by yielding with an async sleep each pass. This restored reliable WS handshakes for local/remote clients.
- Mode seeding: added `K3D_SEED_GRAPH_MAX` to cap the size of the `dataset_graph` payload sent by the seeder to avoid WebSocket frame overflow (1009). Seeder now uses a context‑managed connect (`websockets==10.4`).
- Mode selector training: made classification report robust when a fold has only one class. Current log set is skewed toward `compose_generate`; expand seeding for factual prompts to balance labels.
- New generator: `knowledge3d.tools.gen_text_ollama` to produce topic‑coherent text via local Ollama (e.g., `exaone3.5:latest`).
- Built a modality‑balanced Galaxy (v7) with ~55 text (exaone) + ~55 3D assets. Unify projects embeddings to a common subspace (effective dims = min(target, max_dim, n‑1)). Cross‑modal edges added for navigation.
- Core head transition: The external LLM wrapper path is deprecated for core runs. Use the single in‑process Cranium Core head (docs/CRANIUM_CORE.md). TTS is first‑class. CPU fallbacks are invalid. See `docs/DEPRECATIONS.md`.
- Balanced Expansion Policy: apply per‑modality and per‑topic balancing for new galaxies; degrade gracefully only when open‑source data is exhausted (see `docs/EXPANSION_POLICY.md`).
- Local models playbook: roles/hosts and orchestration are documented in `docs/LOCAL_OLLAMA_MODELS.md`.

### Recent Implementation Notes (2025‑09‑29)
- Fused head RPN trace: optional explanation blocks (tokens + register map) are appended to math answers when `K3D_RPN_TRACE=1`.
- ARC/HLE tester hardening: supports unlimited `--limit 0` and a lightweight `--teacher` feedback score without influencing answers; avoids crashes on empty predictions.
- Math bench auto‑discovery: new flags in `phase25/math_bench_evaluator.py` — `--auto`, `--list`, `--repos` — to run multiple locally cached HF math suites.
- Wikipedia sweep evaluator: `knowledge3d.tools.wiki_sweep_evaluator` checks non‑math routing and extractive summaries across a large AI topics corpus.
- Fused‑head fallback fixed: memory/tablet lookup and neural fallback now always produce an answer (no `None`). Summarize‑inline fallback added for `Summarize: <text>` prompts.
