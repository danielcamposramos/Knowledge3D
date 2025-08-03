# Temporal Agents with Knowledge Graphs

This document summarizes insights from the OpenAI cookbook notebook ["Temporal Agents with Knowledge Graphs"](https://github.com/openai/openai-cookbook/blob/main/examples/partners/temporal_agents_with_knowledge_graphs/temporal_agents_with_knowledge_graphs.ipynb) and relates them to the Knowledge3D (K3D) project.

## Project goals

Knowledge3D (K3D) aims to "build an open standard and toolkit for rendering artificial intelligence knowledge as a navigable three‑dimensional universe," combining CAD geometry, vector databases, and augmented‑reality mapping so humans and AI can explore data together. The EchoSystems action plan envisions shared data and graph layers that follow a common `K3D‑Node` schema, supported by guiding principles (form↔meaning, show‑don’t‑scroll, edge‑first, open‑by‑default) and a four‑pillar stack covering design, psychology, frontend velocity, and system design & DevOps.

## Cookbook reference

> "## 1.1. Purpose and Audience  
> This notebook provides a hands-on guide for building **temporally-aware knowledge graphs** and performing **multi-hop retrieval directly over those graphs**."
>
> "**Temporally-aware knowledge graph (KG) construction**  
> … A key challenge in developing knowledge-driven AI systems is maintaining a database that stays current and relevant. While much attention is given to boosting retrieval accuracy with techniques like semantic similarity and re-ranking, this guide focuses on a fundamental—yet frequently overlooked—aspect: *systematically updating and validating your knowledge base as new data arrives*.  
> … No matter how advanced your retrieval algorithms are, their effectiveness is limited by the quality and freshness of your database. This cookbook demonstrates how to routinely validate and update knowledge graph entries as new data arrives, helping ensure that your knowledge base remains accurate and up to date."
>
> "**Multi-hop retrieval using knowledge graphs**  
> … Learn how to combine OpenAI models (such as o3, o4-mini, GPT-4.1, and GPT-4.1-mini) with structured graph queries via tool calls, enabling the model to traverse your graph in multiple steps across entities and relationships.  
> … This method lets your system answer complex, multi-faceted questions that require reasoning over several linked facts, going well beyond what single-hop retrieval can accomplish."
>
> "Inside, you'll discover:  
> * **Practical decision frameworks** for choosing models and prompting techniques at each stage  
> * **Plug-and-play code examples** for easy integration into your ML and data pipelines  
> * **Links to in-depth resources** on OpenAI tool use, fine-tuning, graph backend selection, and more  
> * **A clear path from prototype to production**, with actionable best practices for scaling and reliability"
>
> "**Note:** All benchmarks and recommendations are based on the best available models and practices as of June 2025. As the ecosystem evolves, periodically revisit your approach to stay current with new capabilities and improvements."
>
> "**Why make your knowledge graph temporal?**  
> … Traditional knowledge graphs treat facts as static, but real-world information evolves constantly. What was true last quarter may be outdated today, risking errors or misinformed decisions if the graph does not capture change over time. Temporal knowledge graphs allow you to precisely answer questions like “What was true on a given date?” or analyse how facts and relationships have shifted, ensuring decisions are always based on the most relevant context."
>
> "**What is a Temporal Agent?**  
> … A Temporal Agent is a pipeline component that ingests raw data and produces time-stamped triplets for your knowledge graph. This enables precise time-based querying, timeline construction, trend analysis, and more."
>
> "**How does the pipeline work?**  
> … The pipeline starts by semantically chunking your raw documents. These chunks are decomposed into statements ready for our Temporal Agent, which then creates time-aware triplets. An Invalidation Agent can then perform temporal validity checks, spotting and handling any statements that are invalidated by new statements that are incident on the graph."
>
> "**Why use multi-step retrieval?**  
> … Direct, single-hop queries frequently miss salient facts distributed across a graph's topology. Multi-step (multi-hop) retrieval enables iterative traversal, following relationships and aggregating evidence across several hops. This methodology surfaces complex dependencies and latent connections that would remain hidden with one-shot lookups, providing more comprehensive and nuanced answers to sophisticated queries."
>
> "**Planners**  
> … Planners orchestrate the retrieval process. *Task-orientated* planners decompose queries into concrete, sequential subtasks. *Hypothesis-orientated* planners, by contrast, propose claims to confirm, refute, or evolve. Choosing the optimal strategy depends on where the problem lies on the spectrum from deterministic reporting (well-defined paths) to exploratory research (open-ended inference)."
>
> "**Tool Design Paradigms**  
> … Tool design spans a continuum: *Fixed tools* provide consistent, predictable outputs for specific queries (e.g., a service that always returns today’s weather for San Francisco). At the other end, *Free-form tools* offer broad flexibility, such as code execution or open-ended data retrieval. *Semi-structured tools* fall between these extremes, restricting certain actions while allowing tailored flexibility—specialized sub-agents are a typical example. Selecting the appropriate paradigm is a trade-off between control, adaptability, and complexity."
>
> "**Evaluating Retrieval Systems**  
> … High-fidelity evaluation hinges on expert-curated “golden” answers, though these are costly and labor-intensive to produce. Automated judgments, such as those from LLMs or tool traces, can be quickly generated to supplement or pre-screen, but may lack the precision of human evaluation. As your system matures, transition towards leveraging real user feedback to measure and optimize retrieval quality in production.  
> … A proven workflow: Start with synthetic tests, benchmark on your curated human-annotated “golden” dataset, and iteratively refine using live user feedback and ratings."
>
> "**Keep the graph lean**  
> … Established archival policies and assign numeric relevance scores to each edge (e.g., recency x trust x query-frequency). Automate the archival or sparsification of low-value nodes and edges, ensuring only the most critical and frequently accessed facts remain for rapid retrieval."
>
> "**Parallelize the ingestion pipeline**  
> … Transition from a linear document → chunk → extraction → resolution pipeline to a staged, asynchronous architecture. Assign each processing phase its own queue and dedicated worker pool. Apply clustering or network-based batching for invalidation jobs to maximize efficiency. Batch external API requests (e.g., OpenAI) and database writes wherever possible. This design increases throughput, introduces backpressure for reliability, and allows you to scale each pipeline stage independently."
>
> "**Integrate Robust Production Safeguards**  
> … Enforce rigorous output validation: standardise temporal fields (e.g., ISO-8601 date formatting), constrain entity types to your controlled vocabulary, and apply lightweight model-based sanity checks for output consistency. Employ structured logging with traceable identifiers and monitor real-time quality and performance metrics in real lime to proactively detect data drift, regressions, or pipeline anomalised before they impact downstream applications."
>
> "This cookbook is designed for flexible engagement:  
> 1. Use it as a comprehensive technical guide—read from start to finish for a deep understanding of temporally-aware knowledge graph systems.  
> 2. Skim for advanced concepts, methodologies, and implementation patterns if you prefer a high-level overview.  
> 3. Jump into any of the three modular sections; each is self-contained and directly applicable to real-world scenarios."
>
> "## 2.1. Pre-requisites  
> Before diving into building temporal agents and knowledge graphs, let's set up your environment. Install all required dependencies with pip, and set your OpenAI API key as an environment variable. Python 3.12 or later is required."

## Insights for K3D

1. **Add temporal semantics**  
   - Implement a Temporal Agent to generate time-stamped triplets, enabling queries like “what was true on a given date?” and facilitating trend analysis.  
   - Leverage an Invalidation Agent to mark outdated information, keeping the 3D knowledge universe current.
2. **Enhance retrieval**  
   - Support multi-step traversal so models can navigate the K3D graph across multiple entities and relationships, guided by planners (task- or hypothesis-oriented) and adaptable tool paradigms.
3. **Operational best practices**  
   - Keep the graph lean via relevance scoring and archival policies.  
   - Parallelize ingestion pipelines for scalability, with asynchronous stages and batched API/database operations.  
   - Integrate production safeguards: ISO-8601 date formats, controlled vocabularies, and structured logging.
4. **Roadmap alignment**  
   - Treat the cookbook as a flexible technical guide: start end-to-end, skim for concepts, or focus on specific sections when building the K3D Node schema and related tooling.  
   - Ensure developers meet the prerequisites (Python 3.12+, OpenAI API key) to reproduce or extend the examples.

