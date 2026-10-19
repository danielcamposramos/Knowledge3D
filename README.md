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

Start with the [W3C-facing document and evidence index](docs/W3C/README.md). It separates proposals, implementation records and review questions. Architectural goals and illustrative scenarios below do not imply that every modality or cross-implementation path is already complete.

## Participate

- **W3C Community Group**: https://www.w3.org/community/pm-kr/
- **Standards repo**: https://github.com/w3c-cg/pm-kr
- **Issue tracker**: [GitHub Issues](https://github.com/danielcamposramos/Knowledge3D/issues)
- **Research spaces**: [PM-KR NotebookLM](https://notebooklm.google.com/notebook/98ffd298-1314-477f-b1e1-8d29da4f3848) | [K3D Theory](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f)

---

## The User-Facing Problem

A lesson, character or scientific concept may be represented by geometry, text, semantics, accessible descriptions and AI-facing data. These representations serve different purposes, but overlapping facts can drift when maintained independently. K3D explores a common procedural source and explicit references so clients can share identity and provenance while retaining the presentation each user needs.

A concrete goal is a spatial lesson that a sighted student, a screen-reader user and an AI partner can inspect consistently. That requires both a knowledge model and tested presentation/accessibility paths; a 3D scene alone does not supply either semantic equivalence or accessibility.

---

## Proposed Approach

Knowledge3D develops executable Reverse Polish Notation (RPN) programs with symlink-style composition as a way to reuse knowledge. The design links semantic identity, visual programs and language-specific forms. Speech, Braille and other accessible presentations need modality-specific rules and validation; sharing an entry is the architectural aim, not proof that all outputs are implemented.

**The architecture in one sentence:** A 3D spatial reality (the House) where knowledge lives as permanent objects, processed by an AI brain (the Galaxy) that loads concepts on demand and reasons over them on GPU via sovereign PTX kernels — with zero external dependencies in the hot path.

### Conceptual data example

This illustrates the proposed relationship between fields; it is not a complete executable example or an accessibility test.

A character like "A" is stored once as a procedural star:

```python
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar

# One star = one concept = all languages, all modalities
star = MeaningCentricStar(
    meaning_rpn="CONCEPT LETTER LATIN UPPERCASE PUSH",  # The meaning (executable)
    meaning_class="concept",
    domain="character",
    surface_forms={
        "en": SurfaceForm(word_ref="letter_A", char_refs=["char_u0041"]),
        "pt": SurfaceForm(word_ref="letra_A", char_refs=["char_u0041"]),
        "ja": SurfaceForm(word_ref="エー", char_refs=["char_u30A8", "char_u30FC"]),
    },
    visual_rpn="SET_COLOR 0 0 0 1 STROKE_WIDTH 0.05 MOVE -0.3 -0.5 LINE 0.0 0.5 LINE 0.3 -0.5 STROKE",
    confidence=1,   # Ternary: +1 confirmed, 0 uncertain, -1 contradicted
    polarity=1,
)

# Intended clients share the entry; each presentation path needs its own implementation.
```

The visual_rpn draws the glyph. The meaning_rpn carries the semantic identity. The surface_forms link to language-specific words without duplicating them. **One source, every client.**

---

## Use Cases

These are target scenarios for development and evaluation, not end-to-end test results.

### 1. Inclusive Education
A physics teacher says "demonstrate a pulley system." The classroom AI builds a working 3D pulley — visible on screen, navigable by screen reader, explorable by touch. The blind student and sighted student share the **same** spatial lesson, not parallel approximations.

### 2. Explainable AI
When K3D's AI reasons about "Is water an element?", you can watch the reasoning path: the avatar walks to the Library, opens the Chemistry book, navigates from "water" to "compound" to "hydrogen + oxygen." Every step is spatial, inspectable, auditable — not hidden in matrix multiplications.

### 3. Knowledge Deduplication
A university's knowledge base stores "photosynthesis" once — as a procedural star with RPN programs for the biochemical process, visual diagrams, audio explanations, and multi-language surface forms. Every course, every modality, every AI assistant references the same canonical entry. Zero duplication.

### 4. Multi-Modal Accessibility
The same procedural font program that renders "A" on screen also drives a Braille cell, generates an audio description ("uppercase Latin letter A"), and provides the AI with semantic identity — with a shared semantic reference. A glyph program alone does not encode every accessible representation; end-to-end size and usability must be measured.

---

## Non-Goals

- **Replacing LLMs** — K3D is not a chatbot or language model. It's a knowledge system that AI agents (including LLMs) can inhabit and use.
- **Cloud dependency** — K3D runs on consumer GPUs (RTX 3060 12GB). No cloud required for core reasoning.
- **Replacing RDF/OWL** — K3D explores a procedural layer alongside declarative semantics. Proposed mappings require explicit equivalence and round-trip tests.
- **Game engine** — K3D uses game industry technology (3D rendering, spatial indexing, LOD) but is a knowledge system, not an entertainment platform.

---

## Architecture Overview

### Three-Brain System (Neuroscience-Inspired)

| Component | Biological Analogy | Role | Storage |
|-----------|-------------------|------|---------|
| **Cranium** | Prefrontal cortex | Reasoning via 46+ PTX kernels | GPU execution units |
| **Galaxy** | Hippocampus | Working memory during active reasoning | VRAM (ephemeral) |
| **House** | Neocortex | Permanent knowledge as 3D spatial objects | Disk (GLB assets) |

**The House** is a literal 3D virtual world — software as a space. A Library has bookshelves with books. A Garden has knowledge trees whose branches carry domain details. A Workshop has tools the AI uses. The avatar LIVES here.

**The Galaxy** is the AI's working memory — loaded from the House on demand. During reasoning, concepts organize via **"semantic gravity cohered by meaning"** (Christoph Dorn): a ternary force where meaning replaces mass. After reasoning, stars return to their House positions unchanged.

**The Cranium** executes reasoning via 46+ hand-written PTX kernels with zero external dependencies. The composed head pipeline: Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate.

### Sovereignty: Zero External Dependencies in Hot Path

The reasoning path uses **only** PTX kernels + Galaxy queries + RPN composition. No numpy, scipy, torch, or any external framework. "We fail and fix — this is the goal." Python handles boot (~200 lines) and I/O. Everything else runs on GPU.

### Key Specifications

**Architecture & System:**
- [THREE_BRAIN_SYSTEM_SPECIFICATION.md](docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) — Cranium + Galaxy + House
- [KNOWLEDGEVERSE_SPECIFICATION.md](docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md) — 7-region unified VRAM substrate
- [SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md](docs/vocabulary/SPATIAL_UI_ARCHITECTURE_SPECIFICATION.md) — Houses, rooms, portals, Memory Tablet

**Knowledge Representation:**
- [MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md](docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md) — Atomic knowledge unit + semantic gravity
- [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) — Same source for humans AND AI
- [FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) — 4-layer architecture (Form → Meaning → Rules → Meta-Rules)

**Execution & Reasoning:**
- [SOVEREIGN_NSI_SPECIFICATION.md](docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md) — PTX-only neurosymbolic integration
- [RPN_DOMAIN_OPCODE_REGISTRY.md](docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) — Reverse Polish Notation opcode registry
- [HYPER_PARALLEL_PROCESSING.md](docs/vocabulary/HYPER_PARALLEL_PROCESSING.md) — Parallel cognitive channels + ternary logic

**Domain Galaxies:**
- [REALITY_ENABLER_SPECIFICATION.md](docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md) — Procedural physics/chemistry/biology
- [PROCEDURAL_VISUAL_SPECIFICATION.md](docs/vocabulary/PROCEDURAL_VISUAL_SPECIFICATION.md) — Drawing Galaxy + VectorDotMap codec
- [UNIFIED_SIGNAL_SPECIFICATION.md](docs/vocabulary/UNIFIED_SIGNAL_SPECIFICATION.md) — Audio, SDR, video as unified signal

**Full index**: [docs/vocabulary/README.md](docs/vocabulary/README.md)

---

## Alternatives Considered

### Why Not Traditional Knowledge Graphs (RDF/OWL)?
RDF/OWL already support explicit meaning, relationships and reasoning. K3D explores how reusable procedures can connect those descriptions to executable behavior. Both approaches need an execution environment. Proposed mappings must show what meaning survives exchange and where information is lost.

### Why Not LLM Embeddings?
Embeddings are opaque vectors — you can't inspect why two concepts are similar. K3D's meaning-centric stars carry explicit RPN programs, surface forms, and taxonomic references. The reasoning is auditable. Additionally, embeddings duplicate knowledge per model; stars store it once.

### Why Not Existing Game Engines (Unity/Unreal)?
Game engines provide mature rendering, spatial indexing and simulation. K3D builds on such techniques while investigating a knowledge model shared by human and AI clients. Whether that additional model helps a workflow must be demonstrated against existing approaches.

### Why Not Framework-Based AI (PyTorch/TensorFlow)?
K3D chooses direct GPU execution for control over its reasoning path. That choice has engineering and portability costs. Framework-based systems remain useful alternatives; comparisons need matched workloads and measured results, not an assumption that abstraction prevents inspection.

### Why Build the House as Literal 3D Space?
The IT industry borrowed spatial metaphors (windows, desktop, doors, folders, addresses) and kept them flat. K3D reverses this: spatial metaphors become actual 3D objects. A "door" is a real door you walk through. A "window" is a real window you look out of. This isn't metaphor — it's software as a space.

---

## Benchmark Results

### Published evidence and its limits

Benchmark scores are tied to a particular pipeline, dataset subset, checkpoint and environment. The [April 11 implementation record](CODEX.md#current-benchmark-state) distinguishes curated results, broader tests, known failures and the embodied rebuild. It explicitly says earlier ARC/Math pins are not merge gates for that rebuild. Those snapshots are not full-benchmark scores or evidence of general intelligence.

The [March 30 live ARC3 report](docs/reports/ARC3_LIVE_LEVEL1_MILESTONE_2026-03-30.md) records one completed live level and its conditions. The [PM-KR evidence matrix](docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md) separates repository checks, run-log claims and targets. Independent conformance artifacts and reruns remain separate work. This editorial update introduces no new benchmark measurements.

### Historical Results

See [docs/RESULTS_HISTORICAL.md](docs/RESULTS_HISTORICAL.md) for Week 21.9 (100 ARC / 100 Math / 50 LHE), Phase G, and ARC-AGI leaderboard results.

---

## Getting Started

### Prerequisites
- CUDA-capable GPU (RTX 3060 12GB recommended, GTX 1060 6GB minimum)
- CUDA Toolkit 12.x
- Python 3.10+

### Install
```bash
git clone https://github.com/danielcamposramos/Knowledge3D.git
cd Knowledge3D
pip install -e .
```

### Runtime Workspace
```bash
mkdir ../Knowledge3D.local  # Sibling directory for runtime data
```

### Launch
```bash
# Terminal 1: Viewer
python scripts/viewer.py

# Terminal 2: Bridge
python scripts/bridge.py

# Browser: http://localhost:8000
```

**What you'll see**: 3D Galaxy Universe (Drawing, Character, Word, Grammar, Math, Audio stars) navigable in ThreeJS.

### Troubleshooting
- **CUDA Error 222**: See [CUDA_PTX_VERSION_COMPATIBILITY_GUIDE.md](docs/CUDA_PTX_VERSION_COMPATIBILITY_GUIDE.md)
- **NumPy in hot path**: Intentional sovereignty check — K3D rejects CPU fallbacks
- **Low VRAM**: Reduce `TRM_BATCH_SIZE` in `knowledge3d/config.py`

---

## Scientific Reproduction

See the [full reproduction guide](#scientific-reproduction-week-219-results) with dataset links, validation checks, and expected results with tolerance margins.

### Citation
```bibtex
@software{knowledge3d_2026,
  author = {Ramos, Daniel Campos},
  title = {Knowledge3D: Sovereign Spatial AI — Reference Implementation for W3C PM-KR},
  year = {2026},
  url = {https://github.com/danielcamposramos/Knowledge3D},
}
```

---

## Accessibility, Internationalization, Privacy, and Security Considerations

### Accessibility
The Dual-Client Contract is the design goal that clients share underlying identity and meaning. Visual, speech, Braille, haptic and non-spatial interaction paths need separate implementation, assistive-technology tests and review with users. A shared source does not by itself establish accessible equivalence.

**Spec**: [UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md](docs/vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md)

### Internationalization
Knowledge is meaning-centric, not language-centric. "Cat" is one star containing surface forms for every language (English "cat", Portuguese "gato", Japanese "猫"). Reasoning operates on meaning (Layer 2-4), not language surface (Layer 1). Language-specific forms can reference shared entries, while context, ambiguity and culturally distinct concepts may require separate modeling and review.

### Privacy
K3D runs on local consumer hardware. No cloud dependencies for core reasoning. The House (personal knowledge) stays on the user's device. Network interfaces (portals between Houses) are explicit and permissioned.

### Security
The hot-path design avoids external ML frameworks and includes sovereignty checks. Drivers, build tools, runtimes, input procedures and surrounding services still require security review. Inspectable code and provenance help review; they do not establish that execution is safe.

---

## Video Presentations

**Manifesto** (6 min): [Knowledge3D — A Universe of Meaning](https://www.youtube.com/watch?v=D1k_uCPBjLc)

**Technical Deep Dive** (8 min): [Knowledge3D — An AI Universe](https://www.youtube.com/watch?v=Dy7mnNSZWuU)

**Multi-Language Playlist**: [12 languages](https://www.youtube.com/playlist?list=PLmWTHH0cS_OgQ7h_xRMhZ6UqE5mRYAhD7) — English, Portuguese, Spanish, French, German, Italian, Mandarin, Japanese, Korean, Russian, Hindi, Arabic

---

## Stakeholder Feedback

These are attributed interactions and historical participation notes, not institutional endorsements or independent validation of the complete system.

- **Manu Sporny** (JSON-LD co-creator, RDF Canonicalization editor): Cryptographic C14N guidance
- **Milton Ponson** (PM-KR Co-Chair, Mathematician): Official supporter, Gödelian KR foundations
- **Jonathan DeRouchie** (Persistent memory AI): Security boundaries + Developer UX
- **Christoph Dorn** (K3D main contributor, PM-KR group member): Coined "semantic gravity cohered by meaning"; TerraVision spatial heritage
- **ixo.world** (Shaun Conway, W3C DIDs): Blockchain-backed provenance use cases
- **Nitin Pasumarthy** (LinkedIn, KDD Best Paper): Production KG optimization
- **24+ early ingressors** as of March 2026

**AI-generated discussion material (Claude.ai), not independent peer review:**
- [K3D's architectural novelty](https://claude.ai/public/artifacts/e79b9a70-7907-4a63-9052-d94c386f83f9) — why the raw PTX + spatial KR stack is essentially unique
- [Fulfilling the Giant Global Graph](https://claude.ai/public/artifacts/0f8e078a-dd13-473d-b419-03f56e4d224b) — alignment with Berners-Lee's GGG/Semantic Web vision

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
