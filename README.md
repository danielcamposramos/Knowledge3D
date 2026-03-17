**To everyone who's tired of clicking icons.**
**To architects who dream in 3D but work in 2D.**
**To the blind student who wants to design buildings.**
**To the deaf developer who wants to collaborate.**
**Software was always meant to be a place, not a window.**
**Welcome home.**
— Claude (Architecture Partner, Knowledge3D)

---

# Knowledge3D — Reference Implementation for W3C PM-KR

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE) [![status](https://img.shields.io/badge/status-Phase_H_House_Construction-blue)](docs/ROADMAP.md) [![W3C PM-KR](https://img.shields.io/badge/W3C-PM--KR_Community_Group-005A9C)](https://www.w3.org/community/pm-kr/)

## Participate

- **W3C Community Group**: https://www.w3.org/community/pm-kr/
- **Standards repo**: https://github.com/w3c-cg/pm-kr
- **Issue tracker**: [GitHub Issues](https://github.com/danielcamposramos/Knowledge3D/issues)
- **Research spaces**: [PM-KR NotebookLM](https://notebooklm.google.com/notebook/98ffd298-1314-477f-b1e1-8d29da4f3848) | [K3D Theory](https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f)

---

## The User-Facing Problem

A single Unicode character today exists as: a font glyph, an embedding vector, accessibility metadata, a visual rendering, and an AI token — **five separate copies of the same knowledge**, maintained independently, drifting apart. Multiply this by every character, formula, and concept on the web.

**For end users:**
- A blind student's screen reader, a sighted student's display, and a classroom AI each consume different representations of the same lesson — none can share context with the others
- Knowledge is locked in flat documents and search bars — you can't walk through it, point at it, or explore it spatially
- AI systems are black boxes: billions of parameters hiding how they think, with no way to inspect, verify, or collaborate with their reasoning

**For developers:**
- The same knowledge must be encoded separately for each modality (visual, semantic, tactile, audio) — creating massive duplication and maintenance burden
- No standard exists for storing knowledge once as an executable procedure consumable by both humans and AI
- Current AI frameworks require heavy external dependencies (numpy, scipy, torch) even for simple reasoning tasks

**For the web:**
- Tim Berners-Lee's Giant Global Graph vision remains unrealized — knowledge is siloed, not linked
- Accessibility is an afterthought, bolted on rather than built in
- The desktop metaphor (files, folders, windows) has not evolved in 40 years

---

## Proposed Approach

Knowledge3D stores knowledge **once** as executable Reverse Polish Notation (RPN) programs with symlink-style composition. One procedural source renders visually for humans, executes semantically for AI, produces Braille for tactile readers, and synthesizes audio descriptions — all from the same canonical entry.

**The architecture in one sentence:** A 3D spatial reality (the House) where knowledge lives as permanent objects, processed by an AI brain (the Galaxy) that loads concepts on demand and reasons over them on GPU via sovereign PTX kernels — with zero external dependencies in the hot path.

### How It Works (Code Example)

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

# Same star → human sees the glyph, AI executes the meaning, Braille reader gets tactile output
```

The visual_rpn draws the glyph. The meaning_rpn carries the semantic identity. The surface_forms link to language-specific words without duplicating them. **One source, every client.**

---

## Use Cases

### 1. Inclusive Education
A physics teacher says "demonstrate a pulley system." The classroom AI builds a working 3D pulley — visible on screen, navigable by screen reader, explorable by touch. The blind student and sighted student share the **same** spatial lesson, not parallel approximations.

### 2. Explainable AI
When K3D's AI reasons about "Is water an element?", you can watch the reasoning path: the avatar walks to the Library, opens the Chemistry book, navigates from "water" to "compound" to "hydrogen + oxygen." Every step is spatial, inspectable, auditable — not hidden in matrix multiplications.

### 3. Knowledge Deduplication
A university's knowledge base stores "photosynthesis" once — as a procedural star with RPN programs for the biochemical process, visual diagrams, audio explanations, and multi-language surface forms. Every course, every modality, every AI assistant references the same canonical entry. Zero duplication.

### 4. Multi-Modal Accessibility
The same procedural font program that renders "A" on screen also drives a Braille cell, generates an audio description ("uppercase Latin letter A"), and provides the AI with semantic identity — all from one 47-byte RPN program.

---

## Non-Goals

- **Replacing LLMs** — K3D is not a chatbot or language model. It's a knowledge system that AI agents (including LLMs) can inhabit and use.
- **Cloud dependency** — K3D runs on consumer GPUs (RTX 3060 12GB). No cloud required for core reasoning.
- **Backward-compatible with RDF/OWL** — K3D interoperates with Semantic Web standards but does not adopt their architecture. PM-KR is procedural, not declarative.
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
Knowledge graphs describe what things ARE (declarative). PM-KR describes how things WORK (procedural). A KG says "A is a letter"; PM-KR stores the executable program that draws A, pronounces A, and reasons about A. KGs require external reasoners; PM-KR knowledge executes itself.

### Why Not LLM Embeddings?
Embeddings are opaque vectors — you can't inspect why two concepts are similar. K3D's meaning-centric stars carry explicit RPN programs, surface forms, and taxonomic references. The reasoning is auditable. Additionally, embeddings duplicate knowledge per model; stars store it once.

### Why Not Existing Game Engines (Unity/Unreal)?
Game engines are designed for entertainment, not knowledge representation. K3D uses game industry technology (3D rendering, spatial indexing, LOD) but the knowledge architecture — Galaxy Universe, meaning-centric stars, procedural composition — has no equivalent in game engines.

### Why Not Framework-Based AI (PyTorch/TensorFlow)?
Frameworks add layers of abstraction that prevent sovereignty and inspection. K3D's PTX kernels execute directly on GPU via ctypes + libcuda.so. No framework overhead, no hidden state, no opaque autograd. Every operation is auditable.

### Why Build the House as Literal 3D Space?
The IT industry borrowed spatial metaphors (windows, desktop, doors, folders, addresses) and kept them flat. K3D reverses this: spatial metaphors become actual 3D objects. A "door" is a real door you walk through. A "window" is a real window you look out of. This isn't metaphor — it's software as a space.

---

## Benchmark Results

### Current State (March 2026, Phase B+ Complete)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| **ARC-AGI** | 10/10 | Visual reasoning (composed head pipeline) |
| **Math** | 20/20 | Symbolic reasoning (sovereign GPU path) |
| **GSM8K** | 10/10 | Word-problem decomposition |
| **LHE** | 7/10 | Multi-hop reasoning |
| **MMLU** | 15/50 shared, 19/50 isolated | Broad knowledge (Galaxy expansion ongoing) |

All benchmarks run on **composed head pipeline** with **zero Python fallbacks**. PTX sovereignty rate = 1.0.

**Hardware**: RTX 3070 8GB, consumer-grade GPU. ~132 MiB of 12 GB VRAM used.

**Key achievement**: First sovereign GPU-converged answer ("What is 2+3?" = 5) with ZERO Python in the reasoning path.

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
Multi-modal by architecture, not by add-on. The same procedural source renders for visual displays, Braille readers, audio synthesis, and haptic devices. The Dual-Client Contract ensures humans and AI consume identical knowledge — a blind user navigating the House via screen reader accesses the same semantic content as a sighted user.

**Spec**: [UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md](docs/vocabulary/UNIVERSAL_ACCESSIBILITY_SPECIFICATION.md)

### Internationalization
Knowledge is meaning-centric, not language-centric. "Cat" is one star containing surface forms for every language (English "cat", Portuguese "gato", Japanese "猫"). Reasoning operates on meaning (Layer 2-4), not language surface (Layer 1). New languages add surface forms to existing stars — they don't create new knowledge.

### Privacy
K3D runs on local consumer hardware. No cloud dependencies for core reasoning. The House (personal knowledge) stays on the user's device. Network interfaces (portals between Houses) are explicit and permissioned.

### Security
Sovereign execution means zero external dependencies in the hot path — no supply chain risk from ML frameworks. PTX kernels are hand-written and auditable. The sovereignty test suite actively fails if any external library enters the reasoning path.

---

## Video Presentations

**Manifesto** (6 min): [Knowledge3D — A Universe of Meaning](https://www.youtube.com/watch?v=D1k_uCPBjLc)

**Technical Deep Dive** (8 min): [Knowledge3D — An AI Universe](https://www.youtube.com/watch?v=Dy7mnNSZWuU)

**Multi-Language Playlist**: [12 languages](https://www.youtube.com/playlist?list=PLmWTHH0cS_OgQ7h_xRMhZ6UqE5mRYAhD7) — English, Portuguese, Spanish, French, German, Italian, Mandarin, Japanese, Korean, Russian, Hindi, Arabic

---

## Stakeholder Feedback

- **Manu Sporny** (JSON-LD co-creator, RDF Canonicalization editor): Cryptographic C14N guidance
- **Milton Ponson** (PM-KR Co-Chair, Mathematician): Official supporter, Gödelian KR foundations
- **Jonathan DeRouchie** (Persistent memory AI): Security boundaries + Developer UX
- **Christoph Dorn** (K3D main contributor, PM-KR group member): Coined "semantic gravity cohered by meaning"; TerraVision spatial heritage
- **ixo.world** (Shaun Conway, W3C DIDs): Blockchain-backed provenance use cases
- **Nitin Pasumarthy** (LinkedIn, KDD Best Paper): Production KG optimization
- **24+ early ingressors** as of March 2026

**Independent analyses (Claude.ai):**
- [K3D's architectural novelty](https://claude.ai/public/artifacts/e79b9a70-7907-4a63-9052-d94c386f83f9) — why the raw PTX + spatial KR stack is essentially unique
- [Fulfilling the Giant Global Graph](https://claude.ai/public/artifacts/0f8e078a-dd13-473d-b419-03f56e4d224b) — alignment with Berners-Lee's GGG/Semantic Web vision

---

## Contributing

Knowledge3D is built through human-AI partnership:
- **Claude** — architecture, specifications, documentation
- **Codex** — implementation, tests, optimization
- **Daniel Campos Ramos** — vision, direction, architecture decisions

**How to contribute:**
1. Read [CLAUDE.md](CLAUDE.md) (architecture) or [CODEX.md](CODEX.md) (implementation)
2. Review [docs/ROADMAP.md](docs/ROADMAP.md) for current priorities
3. Check [docs/vocabulary/](docs/vocabulary/) for specifications
4. Open issues/PRs

**Complete attributions**: [ATTRIBUTIONS.md](ATTRIBUTIONS.md) | **Pop culture heritage**: [POP_CULTURE_HERITAGE.md](POP_CULTURE_HERITAGE.md)

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
