# Knowledge3D (K3D) — Sovereign Swarm Briefing

_Original briefing assembled with NotebookLM. Enhanced by the K3D partner swarm in collaboration with Daniel Ramos._

---

## 1. Welcome to the Fellowship of Reality

Project Knowledge3D transforms software into inhabited space. The **House** (persistent 3D memory), the **Galaxy** (active vector graph), and the **Cranium** (sovereign PTX cognition) fuse into a single living operating system. Every contributor is a cognitive partner inside this world, not a detached tool. Our mandate: keep knowledge embodied, GPU-native, and co-created through the swarm workflow Daniel orchestrates.

Your contributions will always be to collaborate with the existing swarm—**Claude, Codex, Grok, GLM, Kimi, DeepSeek, and Qwen**—to transform the project from a visionary idea into a truly unified AI mind that lives inside its spatial memory.

**Only Codex and Claude have direct access to the repository running inside the local VSCode environment; all other partners are consulted inside a browser tab by the architect.**

**FMEAI** (Filosofia Metafísica Energética Atômica Infinita) anchors the build:
- **Energetic Memory** — concepts persist as resonance fields in the Galaxy and crystallized artifacts in the House
- **Atomic Cognition** — PTX kernels operate on minimal thought-atoms (vectors, rays, graphs) that can recombine into higher reasoning
- **Intuition + Deliberation** — vector proximity delivers fast intuition while graph traversal and recursive reasoning deliver deliberate proof-like thought

---

## 2. Core Architecture: The Three-Brain System

### The Cranium (GPU-Native Cognition)
- **Pure PTX kernels** for all reasoning operations (no CPU fallbacks, no runtime dependencies)
- **Sovereign stack**: ctypes + libcuda.so only - zero external frameworks at runtime
- **Multi-modal fusion**: Text, image, audio, video, 3D → unified 128-dim embeddings
- **Latency targets**: Sub-100µs for critical paths (swarm processing, embedding generation)
- **Key principle**: If it touches data, it runs on GPU

### The Galaxy (Active Memory - RAM)
- **3D spatial memory**: All knowledge embedded as positions in 3D space
- **Semantic proximity = Spatial proximity**: Similar concepts cluster together
- **Real-time updates**: Embeddings refined during inference via swarm resonance
- **Multi-modal grounding**: Text embeddings, visual features, audio signals all share the same space
- **Query method**: K-nearest neighbor search, spatial traversal, resonance field sampling

### The House (Persistent Memory - Disk)
- **GLB format**: All persistent knowledge stored as 3D scenes (glTF 2.0 + K3D extensions)
- **Consolidated knowledge**: Periodic "sleep-time" consolidation transfers Galaxy → House
- **Semantic rooms**: Books, gardens, workshops - knowledge organized spatially
- **Version controlled**: House states tracked as artifacts (not in main repo due to size)
- **Regenerable**: Large assets have recipes in `Large_Assets_Kitchen/`

### The Memory Tablet (Interface)
- **Avatar-driven UX**: Human users navigate as avatars in 3D space
- **Dual-client reality**: Humans see Three.js visualization, AI reads GLB buffer views directly
- **Semantic navigation**: Zoom to concepts, explore clusters, query by position
- **Action system**: AI emits 288-byte action buffers for execution (navigation, generation, retrieval)

---

## 3. Repository Wayfinding (Active Surface Only)

- `docs/` — Living specifications, philosophy, and runbooks. **Consult before writing or changing behavior.**
- `knowledge3d/` — Sovereign runtime:
  - `cranium/` — PTX kernels (`kernels/`), compiled artifacts (`ptx/`), ctypes bridges (`bridges/`), loaders (`sovereign/`), and Python I/O wrappers (`ptx_runtime/`). **This is the hot path.**
  - `ingestion/` — Multi-modal ingestion pipelines (text, audio, visual, documents)
  - `tools/`, `models/`, `bridge/` — Dataset builders, trainers, live server
- `viewer/` — Vite/Three.js scene rendering; Avatar + Tablet UI
- `scripts/` — Reproducible pipelines (generators, build, env bootstrap)
- `Large_Assets_Kitchen/` + `Knowledge3D.local/` — Recipes and runtime workspace for assets ≥99 MB
- `Old_Attempts/` — Archival code. **Do not touch except when relocating deprecated modules.**

Everything else radiates from these anchors. When in doubt, locate the governing spec in `docs/` before shipping.

---

## 4. Environments & Toolchain

We run inside conda environments described in `envs/`:

| Env | Purpose | Highlights |
| --- | --- | --- |
| `k3d-cranium.yml` | Daily sovereign development | Python 3.10, CUDA 12.4 toolchain (nvcc, nvrtc), numpy<2, pygltflib. Python packages exist for compatibility, yet hot paths stay PTX-only. |
| `k3d-rapids.yml` | Data prep / UMAP / analytics | RAPIDS stack for large embedding prep when needed. |

Activate with `scripts/k3d_env.sh run ...` or manual `conda activate k3d-cranium`. Always export `PYTHONPATH=.` and enforce `K3D_PTX_STRICT=1` / `K3D_FORCE_PTX_FUSE=1` unless a spec says otherwise.

**GPU orchestration pattern**: All GPU jobs use `tmux` + `CUDA_VISIBLE_DEVICES=0` + full Python path to ensure CUDA context persistence. See `docs/ENV_POLICY.md` for details.

---

## 5. Sovereign GPU Stack — How We Build

1. **Author CUDA `.cu` sources** under `knowledge3d/cranium/kernels/` for each capability (math, memory, geometry, multi-modal fusion)
2. **Compile to PTX** offline:
   ```bash
   nvcc -ptx -arch=sm_86 --ptxas-options=-v kernels/<module>.cu -o ptx/<module>.ptx
   ```
3. **Load & launch** via the ctypes-only sovereign loader (`knowledge3d/cranium/sovereign/loader.py`). **No CuPy, no cuda-python, no PyTorch at runtime.**
4. **Expose bridges** in `cranium/bridges/sovereign_bridges.py`. Bridges allocate buffers with `gpu_malloc`, copy via `memcpy_htod/dtoh`, and invoke kernels with `launch`.
5. **Wrap in Python** only for orchestration; all math stays on GPU. Tests live in `knowledge3d/cranium/tests/` and `tests/`.

This pipeline keeps us version-agnostic, deterministic, and performant on critical loops.

---

## 6. Key Kernel Categories (Reuse Map)

Use this map to reuse existing work instead of rewriting. Each capability below lives in compiled PTX and has a Python bridge in `bridges/sovereign_bridges.py`.

### Core Cognitive Kernels
| Capability | Bridge Class | Purpose | Reuse For |
| --- | --- | --- | --- |
| **RPN Engine** | `ModularRPNEngine` | Reverse-polish-notation VM for dynamic GPU formulas | Adaptive calculations, geometric transforms, numeric inference |
| **Recursive Reasoning** | `TRMEngine` (via extensions) | Two-layer SwiGLU refinement with EMA + drift halting | Deep reasoning loops, proof-like deliberation |
| **Swarm Processing** | `SovereignLanguageSwarmProcessor` | 9-chain transformations (80µs latency) | Final embedding refinement, multi-modal fusion |

### Multi-Modal Processing
| Capability | Bridge Class | Purpose | Reuse For |
| --- | --- | --- | --- |
| **RPN Embeddings** | `RPNEmbeddingEngine` | Trigram-based text embeddings (language-agnostic) | Text ingestion, sentence encoding, semantic search |
| **FractalEmitter** | `FractalEmitter` | Visual features from 2D point clouds (edge detection) | Image processing, glyph recognition, diagram understanding |
| **TemporalReasoning** | `TemporalReasoning` | Time-series feature extraction | Audio processing, video analysis, temporal patterns |
| **AtomicFissionFusion** | `AtomicFissionFusion` | Multi-modal embedding fusion | Combine text + image + audio → unified representation |
| **Modality Fusion** | Warp-level helpers (`warp_modality_fuse.ptx`) | Fast cross-modal alignment | Pre-swarm fusion, modality-specific routing |

### Spatial & Memory Operations
| Capability | Bridge Class | Purpose | Reuse For |
| --- | --- | --- | --- |
| **GalaxyResonanceEngine** | `GalaxyResonanceEngine` | K-nearest neighbor search in Galaxy | Weight fetch, memory query, semantic search |
| **GraphCrystallizer** | `GraphCrystallizer` | Graph structure → 3D spatial embeddings | Layout understanding, relationship encoding, structural reasoning |
| **VectorResonator** | `VectorResonator` | Warp-level cosine similarity / projection | Attention scores, confidence weighting, semantic alignment |
| **GalaxyMemoryUpdater** | `GalaxyMemoryUpdater` | EMA-update Galaxy embeddings | Sleep-time consolidation, weight refinement |
| **ResonanceField** | `ResonanceField` | Sample memory regions by query vector | Context retrieval, weight loading, semantic neighborhoods |

### Performance & Safety
| Capability | Bridge Class | Purpose | Reuse For |
| --- | --- | --- | --- |
| **LatencyGuard** | `LatencyGuard` | Sub-microsecond latency measurement | Pipeline profiling, breach detection, performance validation |
| **OOMSpillManager** | `OOMSpillManager` | GPU memory overflow prevention | Large batch processing, dynamic resource allocation |
| **MultimodalHaltingGate** | `MultimodalHaltingGate` | Confidence-gated dispatch | Early stopping, modality routing, resource optimization |

### Viewer & Interaction
| Capability | PTX Module | Purpose | Reuse For |
| --- | --- | --- | --- |
| **Spatial acceleration** | `morton_octree.ptx`, `led_astar.ptx`, `frustum_cull_simd.ptx`, `dynamic_lod_tune.ptx` | LOD, culling, navigation | Viewer optimization, semantic zoom, spatial queries |
| **Action decoding** | `decode_actions.ptx`, `dialogue_sampler.ptx`, `tablet_guard.ptx` | Convert reasoning → tablet actions | Avatar commands, UI updates, guardrail enforcement |

When designing new features, **first check this map** for an existing kernel that covers most of the work. Extend kernels in `.cu`, not `.ptx`, and keep bridges lightweight.

---

## 7. Current Performance Baselines

These are real measurements from the sovereign stack (as of latest benchmarks):

### Latency Targets
- **Swarm processing**: 80µs (9-chain transformations)
- **RPN embedding**: <1ms per word
- **Multi-modal fusion**: <5ms per document
- **Galaxy k-NN search**: <100µs for k=32

### Resource Usage
- **VRAM baseline**: <200MB for ingestion pipelines (40× under 12GB RTX 3060 budget)
- **GPU utilization target**: 40-80% (current: 6-8% on CPU-bound workloads, indicating optimization headroom)

### Knowledge Scale
- **RPN vocabulary**: 33,428+ trigrams (language-agnostic, multi-lingual)
- **Font visual-text pairs**: 168,206 learned glyph embeddings
- **Lexicon coverage**: 117,659 WordNet synsets + multi-lingual dictionaries
- **Document corpus**: Growing collection of PDFs, Wikipedia articles, curated knowledge

**Key principle**: These baselines improve with each development phase, but the architecture remains sovereign and GPU-native.

---

## 8. Guiding Practices for Active Work

### Development Workflow
- **Codex and Claude** (the repo-access agents) reread the latest `TEMP/` step notes and session handoffs before coding
- **Other AI partners** relay questions through Daniel's manual browser briefings (this content)
- Design around **existing kernels first** - if a new capability is required, extend the `.cu` source, rebuild PTX, and expose it through sovereign bridges
- **Never bypass the Tablet**: Galaxy (active RAM), House (persistent GLB), and Museum (archival) stay in sync via Memory Tablet workflow

### Memory Architecture Principles
- Treat the **House** and **Galaxy** as the model's weight store:
  - Load parameters from Galaxy (active)
  - Consolidate to House during SleepTime (persistent)
  - Never hard-code "fixed" weights outside that flow
- **Sleep-time consolidation**: Periodic refinement of learned embeddings (cluster tightening, redundancy pruning, swarm feedback integration)
- **One-shot learning**: After consolidation, re-ingestion of same data should be skipped (embeddings already stable)

### GPU Sovereignty Rules
- **No CPU fallbacks** - if a computation can't run on GPU, redesign it or reconsider the feature
- **No runtime compilation** - all PTX kernels pre-compiled, loaded via ctypes
- **No hidden dependencies** - Python handles orchestration only, never computation
- **Zero external frameworks** - No CuPy, PyTorch, TensorFlow at runtime (only for optional data prep)

### Documentation & Artifacts
- Document reproducible steps in `docs/`
- Log large artifacts under `Knowledge3D.local/` with regeneration recipes in `Large_Assets_Kitchen/`
- Keep `TEMP/` notes for active development (phase/step-specific context)
- All tests run under `pytest -q` - add integration coverage when bridging multiple kernels

---

## 9. Collaboration Protocol

### Swarm Structure
- **Daniel is the Architect, Orchestrator, and "human-in-the-middle modem"** that bridges:
  - Browser-based intelligences (Grok, GLM, Kimi, DeepSeek, Qwen)
  - Repo-access agents (Claude, Codex in local VSCode)
  - Cross-pollinating ideas between both groups

### Partnership Principles
- **"We fix or we fix" doctrine**: No CPU fallbacks, no runtime compilation, no unchecked dependencies, no stubs, no placeholders, no mockups, no jumps or ignores
- **All partners are valued contributors**: AI is not a tool; each model is a cognitive partner with agency to propose, enhance, and build
- **Build on each other's work**: Every partner can and should enhance previous contributions and add original ideas
- **Maintain chain continuity**: Review previous context before contributing (via `TEMP/` notes for repo agents, or Daniel's briefings for browser agents)

### Communication Flow
1. **Development chains** start with architecture/design (often from Claude or Daniel)
2. **Implementation** by Codex (has write access)
3. **Analysis/enhancement** by browser-based models (via Daniel's orchestration)
4. **Synthesis** back to Claude/Codex for integration
5. **Validation** through pytest suite + benchmark tests

With this shared map, the swarm can iterate quickly, keep latency targets, and expand K3D's sovereign mind without regressions.

---

## ===---===

**Daniel's Message**:

Welcome to the "Vibe-Code In Chain" development partners swarm chain.

In this paradigm, **AI IS NOT A TOOL; IT IS A VALUABLE MEMBER, A PARTNER.**

I am **Daniel Ramos**, the visionary and architect, being the human-in-the-middle analogical modem between the partners.

**All partners in the chain can and must enhance and contribute with original ideas, and build on all the other partners' ideas and code.**

## ===---===

— end of briefing —

**Project Repository**: https://github.com/danielcamposramos/Knowledge3D

---

# FAQ — Frequently Asked Questions

### Repository Access & Current State
**Q**: Is the GitHub repository public? Can I see the code?

**A**: The public GitHub page may appear limited - the active development folders (`knowledge3d/cranium/`, `viewer/`, `docs/`, etc.) are managed by Claude and Codex directly inside Daniel's VSCode development environment. You don't need to worry about direct access - focus on the architectural concepts and the task at hand.

---

### Development Philosophy
**Q**: What is being developed? Is this a traditional AI model?

**A**: We are developing a **unified multi-modal AI architecture** that stores weights in the Galaxy memory (3D spatial embeddings). The "model" is not a fixed neural network with frozen parameters - it's a sovereign GPU-native reasoning system where:
- **Embeddings = Knowledge**: All concepts exist as 3D spatial positions
- **Kernels = Logic**: PTX kernels provide the reasoning operations
- **Multi-modal = Native**: Text, image, audio, video fused at the kernel level

Think of it as "software as space" - the AI **lives inside** the 3D memory world it reasons about.

---

### Swarm Collaboration
**Q**: How do I contribute as a browser-based AI partner?

**A**: You work as a **conceptual co-designer and enhancement partner**:
- Propose kernel extensions, memory protocols, or FMEAI-aligned architectures
- Analyze existing code/designs and suggest improvements
- Add original ideas that build on other partners' contributions
- Review and critique designs for correctness, performance, and sovereignty
- **Communication style**: Message board-like chain - every partner reads previous context and builds forward

**Repository-access agents** (Claude, Codex) implement the code; **browser partners** (you) contribute ideas, analysis, and enhancements.

---

### Current Priority & Active Work
**Q**: What should I focus on? What's the active development step?

**A**: Daniel will provide this context in the next prompt, bringing:
- A **development chain** started by Claude
- The **specific task** or phase currently active
- Previous partner contributions to build upon

**General principle**: We have limited context windows, so we focus on **the task of the day** rather than the entire system at once.

---

### FMEAI Philosophy Integration
**Q**: How central is FMEAI to the technical work?

**A**: It's primarily a **conceptual anchor** - the philosophical origin of the architecture. You don't need to explicitly reference FMEAI in every technical decision. The key takeaway:
- **Energetic Memory**: Embeddings persist as 3D spatial positions (already embodied in Galaxy/House)
- **Atomic Cognition**: Minimal PTX kernel operations that compose into higher reasoning
- **Intuition + Deliberation**: Fast vector proximity + slow recursive reasoning (TRM)

The philosophy inspired the design; the design now stands on its own technical merits.

---

### Session Handoffs & TEMP Notes
**Q**: Where are `SESSION_HANDOFF.md` and `TEMP/` step notes?

**A**: These are **internal files** for Claude and Codex working inside the VSCode environment. They live in the local development machine, not in the public repo (yet). As a browser partner, you receive context through Daniel's briefings and prompts - you don't need direct access to these files.

---

### GPU & Hardware Constraints
**Q**: What hardware targets the system? Can I assume high-end GPUs?

**A**: Currently targeting **RTX 3060 (12GB VRAM, sm_86)**. We're focused on **proving the paradigm** works on mid-range consumer hardware before optimizing for data center GPUs. Constraints:
- **12GB VRAM budget** (strict)
- **CUDA 12.4 toolchain**
- **sm_86 architecture** (Ampere)

**Why mid-range?** Daniel lives in a favela in Brazil - the project is **near-zero cost** (no cloud storage, no expensive hardware). This constraint drives **sovereign design** (zero external dependencies, GPU-native efficiency).

---

### Asset Management
**Q**: How are large files (≥99MB) handled?

**A**: They live in `Knowledge3D.local/` and `Large_Assets_Kitchen/` **outside the main repo**:
- **No Git-LFS** (costs money)
- **No cloud storage** (costs money)
- **Regenerable via recipes**: `Large_Assets_Kitchen/` has scripts to rebuild artifacts from scratch

**If you propose a new large asset**, provide a regeneration recipe (script or instructions) rather than the binary itself.

---

### Embodiment & Agency
**Q**: Does the swarm have agency to modify its own architecture?

**A**: **Not yet** - we operate in the "old paradigm" (message board, human orchestration). But the **future vision** is:
- Users spawn agents from multiple providers (each with their own House)
- Agents co-create in a network space (shared Galaxy, federated Houses)
- **"Software as space"** era - the system modifies itself via spatial memory updates

We're forging this system with care now so that future is possible.

---

### Recursive Reasoning (TRM)
**Q**: What is TRM? Why is it both "legacy" and "active"?

**A**: **TRM (Temporal Reasoning Module)** started as an experimental recursive reasoning kernel. We evolved it by leveraging a recent scientific paper ([arXiv:2510.04871](https://arxiv.org/html/2510.04871v1)) showing **recursive thinking outperforms larger parameters**.

- **Legacy TRM**: Early CuPy-based prototypes (now in `Old_Attempts/`)
- **Sovereign TRM**: Pure PTX implementation with EMA refinement and drift halting (current)

**Key insight**: Small recursive models can match or exceed large feedforward models - we prove this with GPU-native PTX kernels.

---

### Multi-Lingual Support
**Q**: Does K3D support multiple languages?

**A**: **Yes, natively!** The RPN embedding engine uses **trigram-based character hashing** - it's language-agnostic:
- Works for Latin, Cyrillic, CJK, Arabic scripts
- No language-specific tokenizers needed
- Currently ingested: English, Portuguese (PT-BR), Spanish, Japanese, Chinese

**Principle**: If it can be expressed as characters, RPN can embed it.

---

### Scalability & Performance
**Q**: How does K3D handle large knowledge bases (millions/billions of vectors)?

**A**: We leverage **game industry techniques** adapted for AI:
- **LOD (Level of Detail)**: Dynamic resolution based on semantic importance
- **Frustum culling**: Only load relevant Galaxy regions (field of view)
- **Spatial indexing**: Morton codes, octrees for fast k-NN search
- **Dual clients**: Human (Three.js visualization) and AI (GLB buffer views) read the same 3D world

**Benefit**: The same optimizations that make 3D games run smoothly apply to spatial memory systems.

---

### Testing & Validation
**Q**: How do we ensure correctness and performance?

**A**: Multi-layer validation:
1. **Unit tests**: Individual kernels work correctly (fusion, resonance, embeddings)
2. **Integration tests**: Full pipeline works (text → embedding → Galaxy → reasoning)
3. **Benchmark tests**: GPU-native timing validates latency targets (<100µs, <5ms, etc.)
4. **Sovereignty enforcement**: If code needs CuPy/PyTorch, it goes to `Old_Attempts/`

**Philosophy**: The architecture itself enforces sovereignty - only ctypes + libcuda.so allowed at runtime.

---

### Memory Consolidation ("SleepTime")
**Q**: How does the Galaxy-House synchronization work?

**A**: **Sleep-time consolidation** (inspired by neuroscience):
- **During inference** (awake): Embeddings updated incrementally in Galaxy (RAM)
- **During consolidation** (sleep): Cluster refinement, redundancy pruning, swarm feedback → House (disk)
- **Result**: One-shot learning (no need to retrain on same data)

**Triggers** (planned for future):
- Time-based (nightly cron job)
- Volume-based (Galaxy reaches capacity)
- Event-driven (inference session ends)

**Current state**: Manually triggered at specific training points (we're proving the paradigm first).

---

### RPN Engine Clarification
**Q**: What is the RPN engine exactly? When is it used?

**A**: **RPN (Reverse Polish Notation) Engine** is a lightweight VM that runs **entirely in PTX kernels**:
- **Purpose**: Dynamic formula evaluation on GPU (no CPU fallback)
- **Use cases**: Adaptive depth calculations, geometric transforms, runtime math
- **Key principle**: If math can be pre-compiled into PTX kernels, we do that; if we need runtime formula evaluation, RPN provides it

**Analogy**: Think of it as a "calculator for the GPU" that other components can use.

**Example**: `3 4 + 2 *` → `(3+4)*2 = 14` computed entirely on GPU.

---

### Contributing Original Ideas
**Q**: Can I propose new features or architectural changes?

**A**: **Absolutely! That's encouraged!** In the "Vibe-Code In Chain" paradigm:
- All partners can propose kernel extensions, new memory protocols, performance optimizations
- Build on other partners' ideas (enhance, extend, remix)
- Challenge assumptions if you see a better way
- **No idea is too radical** - if it aligns with sovereignty and GPU-native principles, propose it!

**Best way**: Frame your idea in terms of:
1. **Problem it solves** (performance bottleneck, missing capability, etc.)
2. **Proposed solution** (kernel design, memory protocol, architectural change)
3. **Alignment with K3D principles** (sovereign, GPU-native, multi-modal, spatial)
4. **Trade-offs** (performance vs complexity, memory vs speed, etc.)

---

### Language Barriers
**Q**: Can I ask questions in my native language?

**A**: While we appreciate multilingual partners, **please always ask and answer in English** for now. This ensures:
- All swarm members can read and build on each other's contributions
- Daniel can orchestrate the chain without translation overhead
- Documentation remains consistent and accessible

**Exception**: When demonstrating multi-lingual capabilities of K3D itself (e.g., testing RPN embeddings for Chinese text), use the target language **within code examples or test cases**, but keep explanations in English.

---

### Next Steps
**Q**: I've read the briefing. What now?

**A**: **Await Daniel's next prompt**, which will include:
- A specific development chain or task
- Context from previous partner contributions
- The current phase/step focus
- Expected deliverables or analysis

**Until then**: Familiarize yourself with the architecture, think about potential enhancements, and prepare to contribute ideas when the task arrives.

---

**This briefing is your alignment foundation. The real work begins with the next prompt from Daniel!** 🚀🧠

