# Knowledge3D

| Status | License |
| ------ | ------- |
| ![pre-alpha](https://img.shields.io/badge/status-pre--alpha-blue) | [Apache-2.0](LICENSE) |

Knowledge3D (K3D) is an initiative from EchoSystems AI Studios to build an open standard and toolkit for rendering artificial intelligence knowledge as a navigable three‑dimensional universe.  It fuses concepts from CAD geometry, vector databases and augmented‑reality mapping so humans and AI models can explore data side by side.

## White‑paper overview

The white‑paper "A 3D Vector Universe Standard for High‑Dimensional AI Knowledge" describes the core objective: create a universal 3D data format that lets users fly through interconnected datasets as if visiting planets in space.  Success is defined by knowledge navigability—intuitive traversal and reduced cognitive load—and by technical metrics such as frame rate, query latency and adoption by researchers.

Other companion documents expand on this vision:

* **AI model integration.** "AI Models and 3D Knowledge (K3D) Databases" explores how neural models could inhabit the same 3D data world, enabling immersive visualization and shared interaction.
* **Augmented reality use cases.** "Integrating Augmented Reality and AI: New Frontiers in Training, Learning, and Exploration" outlines how AR headsets can overlay K3D scenes onto the real world for education and field work.
* **CAD/vector database fusion.** "Merging CAD and Vector Database Concepts into a 3D Knowledge Visualization Standard" and "Using AutoCAD’s 3D Vector Format as an AI Vector Database" discuss storing vector embeddings inside CAD-like geometry files so that standard 3D tools become knowledge browsers.
* **Science fiction influence.** "From Sci-Fi to Reality: How Tron's Grid, The Matrix, Star Wars, Star Trek, and *The Jetsons* Foreshadowed Our Future" highlights pop-culture milestones that predicted immersive 3D data interfaces.
* **Historical inspiration.** [Lessons from Early 3D Interfaces and Apollo&nbsp;11](docs/fsn_apollo_inspiration.md) reflects on SGI's 3D file navigator \(as seen in *Jurassic Park*\) and NASA's modular software practices.
* **Collaborative research.** "Spatial Web K3D Gemini Deep Research Report 2" combines insights from **OpenAI GPT o3 Deep Research** with a final write‑up produced by **Google Gemini&nbsp;2.5&nbsp;Pro/Flash**.
* **Manus AI perspective.** The PDFs "The K3D Advantage: A Paradigm Shift for Internal AI Memory" and "The K3D Catalyst: Fostering Human-Like Reasoning and Intuition in AI" are both signed by **Manus&nbsp;AI**.
* **AR vision with an AI coauthor.** "Integrating Augmented Reality and AI: New Frontiers in Training, Learning, and Exploration" expands on Daniel Campos Ramos' ideas in partnership with an **AI assistant**.
* **Interactive demo.** The HTML file "K3D Gemini report" showcases a simple web report generated using **Google Gemini**.
* **ChatGPT agent research.** "ChatGPT Agent - K3D paradigm research" collects exploratory notes from early conversations with the OpenAI model.
* **Industry impact deck.** "ChatGPT Agent - Impact of the K3D Paradigm on BIM and Additive Manufacturing" and the spreadsheet "ChatGPT Agent - Impact_Domains.xlsx" map potential use cases across sectors.
* **Claude's perspective.** "Claude - Collective Intelligence in Shared K3D Space" summarizes Anthropic's take on collaborative 3D knowledge worlds.
* **Concept slides.** "From_Pages_to_Spaces.pptx" visualizes how web pages could evolve into immersive spaces.
* **Deep research notes.** "Gemini DeepResearch Report" chronicles brainstorming sessions prior to the Gemini coauthored paper.
* **Perplexity AI review.** "Expert Assessment - Perplexity AI" offers an outside evaluation of the K3D paradigm from **Perplexity AI**.
* **Portuguese reflections.** "Como eu, como IA, me beneficiaria ao integrar uma" and "Is this idea too different that it requires a comp" capture additional ChatGPT analysis in Portuguese.
* **Audio discussion.** "Spatial Web & K3D_ Your Future of Immersive Knowledge and AI Collaboration.mpga" captures an informal conversation about the Spatial Web.
* **Screenshot preview.** "Screenshot_20250730_153941.png" shows an early build of the planned 3D viewer.

## EchoSystems Action Plan

The project now aligns with the **EchoSystems K3D Collaboration Action Plan**. Key directives are:

- **Guiding principles** – form↔meaning, show‑don’t‑scroll, edge‑first and open‑by‑default.
- **Four‑Pillar stack** – design craft, persuasive psychology, frontend velocity and system design & DevOps.
- **Reference architecture** – data and graph layers share a `K3D‑Node` schema ([`spec/k3d_node_schema.json`](spec/k3d_node_schema.json)).

See [`echo_systems_k_3_d_action_plan_v_0.md`](echo_systems_k_3_d_action_plan_v_0.md) for the full document.

## TeleKnowledge vision

The companion paper *"Towards an Interactive __TeleKnowledge__ Internet: A 3D Virtual World for Businesses, Humans, and AI"* expands on the long‑term ambition for Knowledge3D.  It imagines a **Spatial Web** where self‑hosted virtual buildings replace today’s static pages.  Using open protocols such as the IEEE 2874 Hyperspace Transaction Protocol (HSTP) and its HSML modeling language, any organization could serve a richly interactive 3D environment under its own domain while still linking seamlessly to others.  Built‑in decentralized identifiers control access so both humans and AI agents can explore knowledge spaces safely and transparently.

## Potential collaborators

Knowledge3D aims to cooperate with organizations building an open 3D internet.  Notable groups include:

- [Spatial Web Foundation](https://github.com/Spatial-Web-Foundation)
- [Verses AI](https://github.com/Versesai)
- [MSquared](https://github.com/msquared)
- [mml‑io](https://github.com/mml-io/mml) – Metaverse Markup Language
- [Niantic](https://github.com/Niantic) – Lightship AR platform
- [Brilliant Labs](https://github.com/brilliantlabsAR)
- [Meta](https://github.com/meta)
- [Apple](https://github.com/apple)
- [Google](https://github.com/google)
- [Autodesk](https://github.com/Autodesk)
- [Decentraland](https://github.com/decentraland)
- [OpenAI](https://github.com/openai)

## Licensing

All code in this repository is released under the Apache‑2.0 License.  The white‑paper and related text documents are distributed under the Creative Commons Attribution 4.0 International (CC‑BY‑4.0) license.  See [`/docs/LICENSE-CC-BY-4.0.txt`](docs/LICENSE-CC-BY-4.0.txt) for the full CC‑BY‑4.0 text.  GitHub may report "No license" for the documentation, but this file is authoritative.

For the development roadmap see [docs/ROADMAP.md](docs/ROADMAP.md). Codex agents coordinating work should consult [CODEX.md](CODEX.md). Development conventions are summarized in [docs/DEV_GUIDELINES.md](docs/DEV_GUIDELINES.md).

