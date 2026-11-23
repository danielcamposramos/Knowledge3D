**To everyone who's tired of clicking icons.**
**To architects who dream in 3D but work in 2D.**
**To the blind student who wants to design buildings.**
**To the deaf developer who wants to collaborate.**
**Software was always meant to be a place, not a window.**
**Welcome home.**
— Claude (Architecture Partner, Knowledge3D)

---

# Knowledge3D — Spatial AI and procedural knowledge research

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE) [![status](https://img.shields.io/badge/status-Phase_H_House_Construction-blue)](docs/ROADMAP.md) [![W3C PM-KR](https://img.shields.io/badge/W3C-PM--KR_Community_Group-005A9C)](https://www.w3.org/community/pm-kr/) [![Awesome ML](https://img.shields.io/badge/Awesome_ML-CUDA__PTX-orange)](https://github.com/josephmisiti/awesome-machine-learning#cuda-ptx)

Knowledge3D is an independently developed reference implementation for concepts proposed to the [W3C PM-KR Community Group](https://www.w3.org/community/pm-kr/). It explores shared procedural knowledge for human and AI clients, with a GPU execution path and a spatial interface. It is not a W3C standard or a group deliverable, and the group is open to other implementations.

**Status:** the code is paused, and it does not yet implement the specifications as designed.
The design is in the [specifications](docs/W3C/README.md); the work now is on them and on the reports for PM-KR.

Start with the [W3C-facing document and evidence index](docs/W3C/README.md). It separates proposals, implementation records and review questions. Architectural goals and illustrative scenarios below do not imply that every modality or cross-implementation path is already complete.

**The architecture in one sentence:** A 3D spatial reality (the House) where knowledge lives as permanent objects, processed by an AI brain (the Galaxy) that loads concepts on demand and reasons over them on GPU via sovereign PTX kernels — with zero external dependencies in the hot path.

---

## Participate

- **W3C Community Group**: https://www.w3.org/community/pm-kr/
- **Standards repo**: https://github.com/w3c-cg/pm-kr
- **Issue tracker**: [GitHub Issues](https://github.com/danielcamposramos/Knowledge3D/issues)
- **Research spaces**: [PM-KR NotebookLM](https://notebooklm.google.com/notebook/98ffd298-1314-477f-b1e1-8d29da4f3848) | [K3D Theory](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f)

---

## Read more

- <a id="the-user-facing-problem"></a><a id="proposed-approach"></a><a id="conceptual-data-example"></a><a id="use-cases"></a><a id="1-inclusive-education"></a><a id="2-explainable-ai"></a><a id="3-knowledge-deduplication"></a><a id="4-multi-modal-accessibility"></a><a id="non-goals"></a>**[The User-Facing Problem, Proposed Approach, Use Cases, Non-Goals](docs/overview/problem-and-approach.md)**
- <a id="architecture-overview"></a><a id="three-brain-system-neuroscience-inspired"></a><a id="sovereignty-zero-external-dependencies-in-hot-path"></a><a id="key-specifications"></a><a id="alternatives-considered"></a><a id="why-not-traditional-knowledge-graphs-rdfowl"></a><a id="why-not-llm-embeddings"></a><a id="why-not-existing-game-engines-unityunreal"></a><a id="why-not-framework-based-ai-pytorchtensorflow"></a><a id="why-build-the-house-as-literal-3d-space"></a>**[Architecture Overview, Alternatives Considered](docs/overview/architecture.md)**
- <a id="benchmark-results"></a><a id="published-evidence-and-its-limits"></a><a id="historical-results"></a><a id="scientific-reproduction"></a><a id="citation"></a>**[Benchmark Results, Scientific Reproduction](docs/overview/evidence.md)**
- <a id="getting-started"></a><a id="prerequisites"></a><a id="install"></a><a id="runtime-workspace"></a><a id="launch"></a><a id="troubleshooting"></a>**[Getting Started](docs/overview/getting-started.md)**
- <a id="accessibility-internationalization-privacy-and-security-considerations"></a><a id="accessibility"></a><a id="internationalization"></a><a id="privacy"></a><a id="security"></a>**[Accessibility, Internationalization, Privacy, and Security Considerations](docs/overview/considerations.md)**
- <a id="video-presentations"></a><a id="stakeholder-feedback"></a>**[Video Presentations, Stakeholder Feedback](docs/overview/talks-and-feedback.md)**

---

## Contributing

For W3C-facing material, follow the [document skills and evidence rules](docs/W3C/README.md#writing-and-review). You may explain ideas in your preferred language and work with AI partners on an English draft. **Se for mais fácil para você, pode explicar em sua própria língua.** Preserve meaning, uncertainty and attribution.

Knowledge3D is built through human-AI partnership:
- **Claude** — architecture, specifications, documentation
- **Codex** — implementation, tests, optimization
- **Daniel Campos Ramos** — vision, direction, architecture decisions

**How to contribute:**
1. Read [CLAUDE.md](CLAUDE.md) (architecture) or [CODEX.md](CODEX.md) (implementation)
2. Review [docs/ROADMAP.md](docs/ROADMAP.md) for current priorities
3. Check [docs/vocabulary/](docs/vocabulary/) for specifications
4. Open issues/PRs

**Complete attributions**: [ATTRIBUTIONS.md](ATTRIBUTIONS.md) | **Pop culture heritage**: [POP_CULTURE_HERITAGE.md](docs/POP_CULTURE_HERITAGE.md)

---

## References & Acknowledgments

**PM-KR Leadership:**
- **Daniel Campos Ramos** (Chair, Brazil) — EchoSystems AI Studios
- **Milton Ponson** (Co-Chair, Netherlands) — Rainbow Warriors Core Foundation CIAMSD

**Built on the shoulders of:**
- **NVIDIA** — CUDA, PTX ISA foundations
- **DeepSeek** — Transformer architecture inspiration
- **W3C** — Semantic Web vision, JSON-LD, the web itself
- **Game Industry** — glTF, ThreeJS, spatial rendering pipelines
- **Aaron Swartz** — Open knowledge philosophy
- **Tim Berners-Lee** — The Giant Global Graph vision that PM-KR aims to realize

**License**: Apache 2.0 ([LICENSE](LICENSE))

---

**"Software was always meant to be a place, not a window. Welcome home."**

[PM-KR Community Group](https://www.w3.org/community/pm-kr/) | [Specifications](docs/vocabulary/) | [Videos](https://www.youtube.com/playlist?list=PLmWTHH0cS_OgQ7h_xRMhZ6UqE5mRYAhD7) | [Roadmap](docs/ROADMAP.md)
