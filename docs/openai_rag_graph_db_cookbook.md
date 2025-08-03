# RAG with Graph Databases

This document summarizes insights from the OpenAI cookbook notebook ["Retrieval Augmented Generation with a Graph Database"](https://github.com/openai/openai-cookbook/blob/main/examples/RAG_with_graph_db.ipynb) and relates them to the Knowledge3D (K3D) project.

> "This notebook shows how to use LLMs in combination with [Neo4j](https://neo4j.com/), a graph database, to perform Retrieval Augmented Generation (RAG)."

## Insights for K3D

1. **Relationship-aware retrieval** – Store K3D nodes and edges in a graph database so RAG queries can traverse explicit relationships instead of flat vectors.
2. **Context injection** – Fetch graph neighborhoods around a query node and pass them to models to reduce hallucinations during spatial exploration.
3. **Temporal layering** – Combine graph RAG with the temporal agent approach to ensure responses reflect the most recent knowledge state.

