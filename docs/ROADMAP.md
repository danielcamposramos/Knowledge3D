# Knowledge3D Project Roadmap

This document outlines the strategic, phased development plan for the Knowledge3D (K3D) project, as detailed in the **[Knowledge3D Project Research Report](k3d-research.md)**. The vision is to build a unified framework for embodied spatial intelligence, and this roadmap provides a clear, actionable path to achieving that goal.

The development is divided into three main phases, each building upon the last to progressively realize the full K3D vision.

![Cognitive House](images/cognitive_house.png)

Figure: The Cognitive House illustrates the House (persistent memory), Cranium (active processing), and Logic Layer (models) that guide the roadmap. See the generation prompt in `docs/images/cognitive_house_prompt.md`.

---

## Phase 1: The MVP - The Static Knowledge Graph

The initial focus is on proving the core data-to-spatial pipeline and validating the fundamental concept of transforming complex, high-dimensional data into an understandable 3D spatial representation. This phase will deliver a non-interactive, read-only visualization of a knowledge corpus.

| Deliverable | Purpose | Key Technologies | Status |
|---|---|---|---|
| **Data-to-Spatial Pipeline** | Ingest raw data (e.g., text), generate high-dimensional embeddings, and use UMAP for dimensionality reduction. | Python, Pandas, Hugging Face (BERT), UMAP-learn | Not Started |
| **Static 3D Visualization** | Render the generated 3D point cloud in a simple, navigable environment. | Unity Engine | Not Started |
| **Basic Schema Definition** | Define a robust `k3d_node_schema.json` from first principles, incorporating spatial and propositional knowledge representation. | JSON Schema | Not Started |
| **Core Documentation** | Update project documentation (`README.md`, `AGENTS.md`) to reflect the new vision and this roadmap. | Markdown | In Progress |

**Goal:** Prove the viability of the core visualization pipeline and establish a solid foundation for future development.

---

## Phase 2: The Interactive Agent

Once the core visualization is stable, this phase will introduce the embodied AI agent and interactive features. The goal is to transform the passive viewing experience into an active, conversational one, increasing user engagement and trust.

| Deliverable | Purpose | Key Technologies | Status |
|---|---|---|---|
| **Embodied AI Agent** | Integrate a conversational AI agent into the 3D environment to act as a guide and interpreter. | Unity Engine, LLM (e.g., GPT-4o, Llama), Convai API (or similar) | Not Started |
| **Agent World Model** | Develop the agent's internal representation of the virtual environment, enabling it to reason about the space and user intentions. | Custom spatial data structures | Not Started |
| **Interactive UI/UX** | Implement basic user interactions, such as selecting data points, issuing voice commands, and receiving multimodal feedback from the agent. | Unity UI, Speech-to-text libraries | Not Started |
| **XAI Integration** | Begin integrating Explainable AI (XAI) methods (e.g., SHAP) to allow the agent to answer "why" questions about the data. | SHAP, Python backend | Not Started |

**Goal:** Create an engaging, interactive experience where users can converse with an AI to explore and understand the knowledge space.

---

## Phase 3: The Collaborative Knowledge Habitat

The final phase will expand the single-user experience into a fully-featured, multiplayer, and real-time collaborative environment. This will realize the full vision of K3D as a platform for remote collaboration and shared knowledge exploration.

| Deliverable | Purpose | Key Technologies | Status |
|---|---|---|---|
| **Multiplayer Environment** | Enable multiple users and AI agents to co-exist and interact within the same shared 3D space. | Unreal Engine (recommended for high-fidelity graphics), Photon/Netcode for GameObjects | Not Started |
| **Real-time Collaboration Tools** | Develop tools for shared annotations, real-time data manipulation, and collaborative problem-solving. | Custom UI/UX components | Not Started |
| **Advanced RAG Integration** | Implement a sophisticated Retrieval-Augmented Generation (RAG) system connected to a knowledge graph to ensure agent responses are accurate and grounded. | Knowledge Graph DB (e.g., Neo4j), RAG frameworks | Not Started |
| **Full Ecosystem API** | Develop a comprehensive API for third-party developers to extend the K3D platform and integrate their own data and models. | REST/GraphQL | Not Started |

**Goal:** Deliver a rich, collaborative platform for collective intelligence, where teams of humans and AIs can explore complex data together.

---

This roadmap is a living document and will be updated as the project progresses. For a more detailed technical breakdown, please refer to the main [research document](k3d-research.md).
