> **Note: This document is outdated.**
>
> The information in this file has been superseded by the new, consolidated vision document. For the most current and authoritative information on the project's vision, architecture, and roadmap, please see:
>
> **[`docs/VISION.md`](docs/VISION.md)**
---

# Codex Tasks

This file lists actionable tasks for AI agents working on the Knowledge3D repository. All tasks are derived from the official **[Project Roadmap](docs/ROADMAP.md)** and should be executed in accordance with the vision outlined in the **[K3D Research Report](docs/k3d-research.md)**.

---

## Phase 1 Tasks: The MVP - The Static Knowledge Graph

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

## Phase 2 Tasks: The Interactive Agent

**Objective:** Transform the passive viewing experience into an active, conversational one by integrating an embodied AI agent.

1.  **Integrate Embodied AI Agent:**
    -   [ ] **Task:** Add an agent to the viewer.
    -   **Details:** Provide a visible avatar and minimal policy (seek-by-label, neighbor traversal), with a UI to issue simple commands (e.g., “go to <label>”). Later integrate voice/LLM when ready.

2.  **Develop Agent World Model:**
    -   [ ] **Task:** Implement the agent's spatial reasoning capabilities.
    -   **Details:** Create data structures that allow the agent to understand the location, properties, and relationships of the knowledge nodes in its environment.

3.  **Implement Basic Interactivity:**
    -   [ ] **Task:** Develop the user-to-agent interaction loop.
    -   **Details:** Implement voice command recognition and the ability for the agent to provide multimodal feedback (e.g., speaking while highlighting a cluster of nodes).

4.  **AI Diary (Vector‑Native):**
    - [ ] Add viewer UI to open a diary book, list pages, and call `/diary read` to render translated text.
    - [ ] Enforce AI‑only writes; ensure `/mem add` cannot target the `Diary` room.
    - [ ] Add “promote to Garden” action to curate a diary page into the Knowledge Garden.

5.  **Doors & Address Bar:**
    - [ ] Add an address bar UI to door meshes; integrate `/open <label|k3d://...>` and show route previews.
    - [ ] Provide helpers to create inter‑House (LAN) doors given two `K3D_HOUSE_ID`s.

---

## Phase 3 Tasks: The Collaborative Knowledge Habitat

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

---

## Ongoing Research Tasks

-   **XAI Integration:** Investigate and implement methods like SHAP to provide explanations for data patterns.
-   **Knowledge Graph & RAG:** Research and develop a robust backend using a knowledge graph and a Retrieval-Augmented Generation (RAG) system.
-   **3D Data Standards:** Continue to monitor and align with standards like glTF, OpenUSD, and 3D Tiles.
-   **Graph Database Standards:** Continue to align K3D's data model with standards like openCypher and RDF.
-   **Phase 25 Sleep Consolidation:** Ensure `cuda-python` is available inside `k3d-cranium` (or a dedicated CUDA env) so `SleepTimeCompute` can persist Phase 25 knowledge into the House.
-   **Lexicon Refinement:** Rebuild Q&A material from the cleaned book sources using exaone3.5 with full-document context to avoid page-break artifacts (e.g., `ma- chine`).
-   **Galaxy Coverage:** Ingest the balanced EN/ES/PT_PT/ZH Wikipedia splits into `viewer/public/galaxy/working/` so algorithmic runs have broad factual grounding.
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
