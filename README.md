# Knowledge3D: A Spatial Knowledge Reality

| Status | License |
| ------ | ------- |
| ![pre-alpha](https://img.shields.io/badge/status-pre--alpha-blue) | [Apache-2.0](LICENSE) |

Knowledge3D (K3D) is an open-source initiative to build a foundational platform for a new paradigm of human-AI interaction: a **Spatial Knowledge Reality**. It fuses concepts from CAD geometry, vector databases, and augmented reality to create a navigable 3D universe where knowledge is represented spatially.

This "knowledgeverse" serves as a shared environment for both humans and AI agents, enabling a new level of collaboration and discovery.

## The K3D Vision

Our vision is to **make knowledge spatial, social, and sentient**. We are moving beyond linear text and 2D interfaces to create a traversable 3D universe where insights can be discovered and shared at the speed of thought.

For a detailed exploration of this vision, please see the **[Jules Report: The K3D Vision for a Spatial Knowledge Reality](docs/Jules_Report.md)**.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/danielcamposramos/Knowledge3D.git
    cd Knowledge3D
    ```

2.  Install the Python dependencies:
    ```bash
    pip install -e .
    ```

3.  Install the viewer dependencies:
    ```bash
    cd viewer
    npm install
    ```

### Quick Start

1.  Generate the sample dataset:
    ```bash
    python -m k3dgen examples/sample_vectors.csv --gltf examples/sample_output.gltf --k3d examples/sample_output.k3d
    ```

2.  Run the viewer:
    ```bash
    cd viewer
    npm run dev
    ```

This will launch a local server. Open your browser to the URL provided to see the 3D visualization.

![Viewer preview](docs/viewer_preview.png)

## Project Status

The project is currently in **Phase 3**, focusing on the development of a lightweight browser viewer. For more details, please see the [Project Roadmap](docs/ROADMAP.md).

## Key Concepts

The K3D standard is built on a few key concepts:

-   **K3D Node:** A single unit of knowledge, defined by the [`k3d_node_schema.json`](spec/k3d_node_schema.json). Each node has an ID, a 3D position, the original high-dimensional embedding, and metadata.
-   **glTF Extension:** We use a custom `K3D_nodes` glTF extension to link the 3D geometry to the `.k3d` metadata file. This keeps the glTF files small and focused on rendering. See [`glTF_K3D_extension.md`](spec/glTF_K3D_extension.md) for details.

## How it Works

The current implementation provides a simple pipeline for visualizing high-dimensional data:

1.  **`k3dgen` CLI:** This Python tool reads a CSV file of high-dimensional vectors, uses PCA to reduce them to 3D, and generates a `.gltf` file and a `.k3d` metadata file.
2.  **Viewer:** A `three.js`-based web application that loads the `.gltf` and `.k3d` files, displays the 3D point cloud, and provides basic interactivity.

## Further Reading

The ideas behind K3D are explored in more detail in the following documents:

-   [EchoSystems K3D Collaboration Action Plan](docs/reports/echo_systems_k_3_d_action_plan_v_0.md)
-   [A 3D Vector Universe Standard for High-Dimensional AI Knowledge](docs/papers/3d_vector_universe_standard.docx) (Whitepaper)
-   [Developer Guidelines](docs/DEV_GUIDELINES.md)
-   [Codex Tasks](CODEX.md)
-   [House Memory: Linking LLM Embeddings to the Spatial Web](docs/HOUSE_MEMORY.md)

A full list of related papers and research can be found in the `docs/` directory.

## Licensing

All code in this repository is released under the Apache-2.0 License. The documentation and other text-based content are distributed under the Creative Commons Attribution 4.0 International (CC-BY-4.0) license.
