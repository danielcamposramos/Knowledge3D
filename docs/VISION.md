# The K3D Vision: A Unified Framework for Spatial Intelligence

## Part 1: The Vision

The Knowledge3D (K3D) project represents a fundamental shift in how we interact with information. It is not merely a data visualization tool, but a unified framework designed to bridge the cognitive gap between human spatial intuition and the abstract, high-dimensional nature of modern data. The core vision is to create an interactive, embodied, and spatial environment for knowledge itself—transforming it into a tangible medium that our brains are uniquely equipped to understand.

The project's purpose is cognitive, not just technical. By translating complex, non-spatial data into a geometric form, K3D functions as a form of **cognitive augmentation**. It is a tool that amplifies human intelligence, enabling users to intuitively identify complex patterns, explore relationships, and gain deep understanding of complex datasets in a way that is impossible with traditional, flat-screen interfaces.

## Part 2: Core Concepts

The K3D framework is built on a set of core concepts that define its unique approach to human-AI interaction and knowledge representation.

### The AI Avatar: A Cognitive Architecture

The AI Avatar is the embodied agent that inhabits the K3D space. It is not just a chatbot, but a structured cognitive entity with a clear architecture:

-   **The House (Persistent Memory):** The entire 3D environment—the house, rooms, and furniture—serves as the AI's long-term, persistent memory. Knowledge is stored spatially, with different rooms and objects representing different domains of information.
-   **The Cranium (Active Processing):** The Avatar has a visible, translucent "cranium" that contains its short-term or active memory. This is visualized as a dynamic, galaxy-like structure of embeddings and semantic links, representing the AI's current thoughts and focus.
-   **The Logic Layer (Swappable AI Models):** The core reasoning of the avatar is powered by a swappable logic layer, allowing different AI models (like GPT-4, Llama, etc.) to be integrated to drive the avatar's behavior and conversational abilities.

### The Dual-Client Model: Human View vs. AI View

K3D operates on a fundamental principle of dual representation, providing different views of the world tailored to human and AI perception:

-   **Human Client:** Interacts with a traditional graphical representation of the world, seeing textures, colors, and readable text on objects. For example, a book object displays written pages.
-   **AI Client:** Perceives the world as a web of data. Instead of textures, the AI "sees" the raw embeddings and metadata that define an object's meaning and relationships. For the AI, a book's pages are not text, but the embedding vectors of its content.

This duality extends to the user interfaces. When a human views an AI's tablet, they see graphical elements. When an AI views a human's tablet, it perceives the underlying data and embeddings.

### The Tablet: A Bridge to the Digital World

Both humans and AI avatars are equipped with a "tablet," a 3D interactive object that serves as the primary interface for a wide range of functions:

-   **Chat and Communication:** The tablet is the main device for text-based chat with other users and AI agents.
-   **Access to External Tools:** It provides a gateway to the traditional internet, applications, and even virtual machines.
-   **System Control:** It acts as a control panel for interacting with the K3D environment, managing tasks, and accessing personal data like calendars and to-do lists.

For AI agents, the tablet serves as a primary API endpoint, allowing them to interact with the world, receive commands, and access information programmatically.

## Part 3: System Architecture

The K3D vision is realized through a robust, modular architecture designed to transform raw data into a navigable spatial environment and enable real-time interaction with an embodied AI agent.

### The Data-to-Spatial Pipeline

The core of the platform is a multi-stage pipeline that converts abstract information into a tangible 3D world:

1.  **Data Ingestion & Embedding:** The pipeline begins by ingesting raw data (such as text from documents). This data is then processed by advanced embedding models (e.g., BERT) to convert it into high-dimensional numerical vectors. Each vector represents the semantic meaning of the source data.
2.  **Dimensionality Reduction:** To make the data visualizable, these high-dimensional vectors are projected into a 3D space. The primary technique used is **UMAP** (Uniform Manifold Approximation and Projection), which excels at preserving both local and global data structures. Other methods like PCA and t-SNE are available as secondary options for different analytical perspectives.
3.  **Spatial Representation Model:** The output is not just a point cloud. A symbolic spatial index is built on top of the geometric data, encoding the semantic relationships, metadata, and neighbor-links between nodes. This "world model" is what allows the AI agent to reason about the space.

### The glTF Embedded Data Format

To ensure portability and simplicity, K3D uses a self-contained **embedded glTF format**. All critical knowledge data is stored directly within the 3D model file, rather than in a separate sidecar file.

-   **Location:** All custom data is stored in the `primitive.extras.k3d` section of a `.gltf` or `.glb` file.
-   **Schema:** The embedded data includes:
    -   `ids`: An array of unique identifiers for each knowledge node.
    -   `vectorsView`: A buffer view pointing to the packed 3D coordinates for each node.
    -   `embeddingsView`: A buffer view pointing to the packed high-dimensional embedding vectors.
    -   `embeddingDims`: The dimension of each embedding vector.
    -   `metadata`: An array of objects containing labels and other metadata for each node.
    -   `neighbors`: An adjacency list representing the graph structure of the knowledge.

This approach ensures that the geometry and its associated semantic meaning always travel together as a single asset.

### The Live Server & Agent Interaction

Interaction with the K3D world is managed by a live WebSocket bridge, enabling real-time communication between a user (or AI client) and the spatial environment.

-   **Live Server:** The `knowledge3d.bridge.live_server` module runs a Python-based WebSocket server.
-   **Client Connection:** A client (like the web viewer or an AI script) connects to this server (e.g., at `ws://127.0.0.1:8765`).
-   **Agent Protocol:** Once connected, a client can interact with the world using a simple chat-like protocol. Commands like `/join`, `/nick`, and `/msg` are used for basic interaction, while commands like `goto <label>` allow the agent to navigate the knowledge space by traversing the spatial graph. The agent's movements and reasoning are traced and logged for analysis and training.

## Part 4: Phased Roadmap

The development of K3D is guided by a phased roadmap. Each phase builds upon the last to progressively realize the full vision of a collaborative spatial knowledge environment.

### Phase 1: The Static Knowledge Graph (MVP)

**Objective:** To prove the core data-to-spatial pipeline by creating a non-interactive, read-only visualization of a knowledge corpus.

| Deliverable | Purpose | Status |
|---|---|---|
| **Data-to-Spatial Pipeline** | Ingest raw text, generate embeddings, and use UMAP to create a 3D point cloud. | Mostly Complete |
| **Embedded Data Model** | Embed all node data (IDs, vectors, embeddings, etc.) directly into a single glTF file. | Complete |
| **Static 3D Visualization** | Create a basic web viewer to load and navigate the static 3D knowledge graph. | In Progress |
| **Core Documentation** | Consolidate project vision, architecture, and roadmap into a unified document. | In Progress (Jules) |

---

### Phase 2: The Interactive Agent

**Objective:** To transform the passive viewing experience into an active, conversational one by integrating an embodied AI agent.

| Deliverable | Purpose | Status |
|---|---|---|
| **Embodied AI Agent** | Integrate a conversational AI agent into the 3D environment with a visible avatar. | Not Started |
| **Agent World Model** | Develop the agent's spatial reasoning to understand its environment and user intent. | Not Started |
| **Interactive UI/UX** | Implement user interactions like voice commands and multimodal feedback from the agent. | Not Started |
| **Live Server Bridge** | Enable real-time communication between the agent and the 3D world. | Complete |

---

### Phase 3: The Collaborative Knowledge Habitat

**Objective:** To expand the single-user experience into a fully-featured, multiplayer, real-time collaborative environment.

| Deliverable | Purpose | Status |
|---|---|---|
| **Multiplayer Environment** | Allow multiple users and AI agents to share and interact in the same 3D space. | Not Started |
| **Real-time Collaboration Tools** | Develop tools for shared annotations, data manipulation, and collaborative problem-solving. | Not Started |
| **Advanced RAG Integration** | Implement a sophisticated RAG system connected to a knowledge graph for grounded AI responses. | Not Started |
| **Ecosystem API** | Develop a public API for third-party developers to extend the K3D platform. | Not Started |

## Part 5: Contributor Guidelines

To ensure smooth and effective development, all contributors (both human and AI) must adhere to the following guidelines.

### Primary Directive

Your primary directive is to follow the phased plan outlined in this document's **Roadmap**. Ensure that any contributions directly support the goals of the current phase. Clear communication and alignment with the project's strategic vision are essential.

### Development Workflow (Live Mode)

The primary development and testing workflow involves running the web viewer and the live server simultaneously:

1.  **Start the Web Viewer:**
    ```bash
    cd viewer
    npm run dev
    ```
2.  **Start the Live Server Bridge:**
    ```bash
    python -m knowledge3d.bridge.live_server
    ```
    This will start the WebSocket server, typically at `ws://127.0.0.1:8765`.

### Testing Expectations

All code changes should be validated by running the appropriate tests.

-   **Python Tests:**
    ```bash
    pytest -q
    ```
-   **Viewer/TypeScript Tests:**
    ```bash
    cd viewer
    npm install --ignore-scripts --no-bin-links
    node ./node_modules/jest/bin/jest.js --runInBand
    ```

### Agent Behavior in Live Mode

AI agents interacting with the live environment are expected to be proactive and communicative:

-   **Explain-as-you-move:** Emit concise rationale at each step of navigation, referencing neighbors and similarity metrics.
-   **Be Proactive:** Agents may initiate chats to propose exploration or ask clarifying questions; do not wait for prompts.
-   **Act with Confidence:** Apply a "Faith Engine" threshold for actions that modify knowledge. Prefer read-only navigation unless confidence in an action is high (e.g., >= 0.7).
