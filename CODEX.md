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

---

## Assets

- **Location:** Store all visual assets in `docs/images/`.
- **Naming:** Use descriptive, kebab-case names, e.g., `cognitive_house.png`, `avatar_workshop.png`.
- **Prompt Pairing:** For each asset, add a matching prompt file named `<image>_prompt.md` containing the generation prompt and any parameters.
- **Referencing:** When adding an asset, embed it in relevant docs (`README.md`, `docs/k3d-research.md`, `docs/ROADMAP.md`, `AGENTS.md`) and link to its prompt file.
- **Attribution & License:** Ensure images comply with the repo license or include source/usage notes in the prompt file if different.
