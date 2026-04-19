# Paper A (ARC Prize 2026) — Prior-Art Literature Sweep

**Date**: 2026-04-19
**Source**: Kimi K2.5 via `ask_cloud` MCP, two consecutive sweeps (clusters A-E then F-H)
**Purpose**: Citation-ready Prior Work material for Paper A §3 and Paper B/C/D §2. Preserved here so the lit sweep does not evaporate with tool-call state.
**Status**: Raw lit sweep — needs Daniel/Claude verification per citation before inclusion in any public document.

> **⚠️ Verification note**: These are LLM-generated citations. Before any of these reach a public paper, each citation must be verified against the actual paper (DOI resolution, correct venue/year, title exact match). Prior experience with LLMs and citations says ~10-20 % may be misattributed or partially hallucinated. Flag `[UNVERIFIED]` in Paper A draft until confirmed.

---

## Cluster A — Memory Palace / Method of Loci in AI

1. **Yates, F. A. (1966).** *The Art of Memory.* Routledge.
   - *Contribution:* Historical and cognitive foundations of Method of Loci — Simonides → Cicero → Renaissance memory theaters.
   - *Gap:* Human cognitive framework only; no computational substrate for loci-based machine memory addressing.

2. **Kremers, S., et al. (2020).** "The Method of Loci in Virtual Reality: A Comparative Study of Immersion and Agency." *Proc. ACM Human-Computer Interaction (PACMHCI),* 4(CSCW), 1–22. https://doi.org/10.1145/3392866
   - *Contribution:* VR memory palaces improve human recall via egocentric spatial navigation.
   - *Gap:* Human performance augmentation; does not propose an AI-native memory architecture using spatial coordinates as primary memory addresses.

3. **Qiu, T., et al. (2024).** "Enhancing Memory Recall Through AI-Assisted Method of Loci in Virtual Environments." *Proc. ACM Designing Interactive Systems Conference (DIS) 2024,* 1–15. https://doi.org/10.1145/3772363.3798815
   - *Contribution:* AI assistant curates loci objects and routes for human users.
   - *Gap:* AI as curator for human memory, not as a reasoning agent treating spatial indexing as its own native memory substrate.

**K3D gap:** No existing AI architecture employs hierarchical spatial indexing (palaces → halls → rooms → nodes) as the fundamental memory-addressing scheme for *the AI itself*. Current systems rely on flat vector spaces or sequential attention.

---

## Cluster B — Procedural Knowledge Representation

1. **Reed, S., & de Freitas, N. (2016).** "Neural Programmer-Interpreters." *ICML 2016.*
   - *Contribution:* Neural network learning to execute programs with stack-based execution traces.
   - *Gap:* Execution mediated through neural attention; no raw programmable stack machine or RPN substrate.

2. **Graves, A., et al. (2016).** "Hybrid Computing Using a Neural Network with Dynamic External Memory." *Nature,* 538(7626), 471–476. https://doi.org/10.1038/nature20101
   - *Contribution:* Differentiable Neural Computer (DNC) binds vectors to addressable memory via content-based lookup.
   - *Gap:* Memory operations are soft (differentiable) and CPU-simulated; no native GPU execution of discrete stack operations or RPN bytecode.

3. **Dyer, C., et al. (2015).** "Transition-Based Dependency Parsing with Stack Long Short-Term Memory." *ACL 2015,* 334–343. https://doi.org/10.3115/v1/P15-1033
   - *Contribution:* Push-down automaton via LSTM controllers.
   - *Gap:* Stack is a neural abstraction; no system treats RPN programs as persistent, editable memory objects.

4. **Cai, Q., & Yao, A. (2023).** "Reasoning with Program Execution: Strengthening Mental Models in Language Agents." *arXiv:2310.08367.*
   - *Contribution:* LLMs generate and execute intermediate procedural representations for multi-step reasoning.
   - *Gap:* Relies on external Python/CPU symbolic engines; lacks GPU-native, resident procedural memory without PCIe latency.

**K3D gap:** Existing systems embed procedural logic either as differentiable soft operations or external CPU code; none use RPN stack machines as the native, GPU-resident representation for declarative AND procedural knowledge.

---

## Cluster C — Small Reasoning Models with External Memory

1. **Khandelwal, U., et al. (2020).** "Generalization through Memorization: Nearest Neighbor Language Models." *ICLR 2020.*
   - *Contribution:* kNN-LM — 100M-param model + non-parametric datastore matches multi-billion-param dense models.
   - *Gap:* Flat, exhaustive kNN index; no hierarchical or spatial addressing; does not scale to trillion-token corpora on consumer GPUs.

2. **Lample, G., et al. (2019).** "Large Memory Layers with Product Keys." *NeurIPS 2019.*
   - *Contribution:* Product Key Memory scales to billions of sparse memory slots with small active parameter count.
   - *Gap:* Learned-hashing addressing, not content-addressable symbolic/spatial queries; soft attention retrieval, not discrete defeasible inference.

3. **Borgeaud, S., et al. (2022).** "Improving Language Models by Retrieving from Trillions of Tokens." *ICML 2022,* 139, 2206–2240.
   - *Contribution:* RETRO — 7B encoder + chunked retrieval; frozen external knowledge replaces parametric memory.
   - *Gap:* Distributed CPU retrieval infrastructure (FAISS over terabytes); not designed for sovereign single-GPU deployment with 7M-100M active parameters.

4. **Wang, S., et al. (2024).** "MemLong: Memory-Augmented Retrieval for Long Text Modeling." *ACL 2024,* 12345–12360.
   - *Contribution:* RAG for small decoders via frozen retrieval + adapter training; 20× context extension.
   - *Gap:* Retrieval memory is passive text chunks; no write-back, procedural execution, or shared human-AI mutation of the knowledge base.

**K3D gap:** Current small-memory architectures treat external knowledge as read-only static corpora indexed by vectors, lacking the active, symbolic, and writable memory palaces required for autonomous cognitive architectures.

---

## Cluster D — GPU-Native Symbolic Reasoning

1. **Dal Palù, A., et al. (2015).** "Parallel SAT Solving on GPUs." *JAIR,* 53, 293–343.
   - *Contribution:* Clause learning and unit propagation in CUDA kernels; 10–100× speedup for SAT.
   - *Gap:* Limited to NP-complete propositional search; no higher-order logic, no defeasible reasoning, no cognitive architecture.

2. **Jordan, H., et al. (2019).** "Radish: Scalable GPU-Based Datalog." *LPNMR 2019,* 134–139.
   - *Contribution:* Compiles stratified Datalog into CUDA kernels; real-time recursive queries over large knowledge graphs.
   - *Gap:* Monotonic / stratified logic only; cannot handle negation-as-failure, priorities, or rule defeat.

3. **Theissen-Lipp, J., & Deering, S. (2013).** "Parallel Production Systems on the GPU." *AAAI 2013,* 27(1), 948–954.
   - *Contribution:* Rete for CUDA; parallelized match-recognize-act for expert systems.
   - *Gap:* GPU as pattern-matching accelerator for CPU-hosted systems (CPU-GPU ping-pong); not a sovereign resident cognitive substrate with unified memory addressing.

4. **Zhang, M., et al. (2021).** "GPU-Accelerated Reasoning for Description Logics." *CIKM 2021,* 2457–2466.
   - *Contribution:* Parallelizes tableaux reasoning for OWL ontologies via CUDA streams.
   - *Gap:* Static TBox/ABox classification; no runtime integration with neural components or procedural execution environments.

**K3D gap:** No existing system implements defeasible, non-monotonic logic OR cognitive production systems directly in PTX/SASS; current approaches offload to CPU or restrict to monotonic subsets (Datalog, SAT).

---

## Cluster E — Hierarchical Spatial Indexing for Semantic Search

1. **Morton, G. M. (1896).** *A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing.* IBM Technical Report.
   - *Contribution:* Z-order (Morton) curve; hierarchical spatial hashing with locality preservation.
   - *Gap:* Designed for geodetic data; no mechanism for binding semantic embeddings to spatial octree nodes.

2. **Riegler, G., et al. (2017).** "OctNet: Learning Deep 3D Representations at High Resolutions." *CVPR 2017,* 6620–6629.
   - *Contribution:* Hybrid grid-octree structures for sparse 3D convolutional features.
   - *Gap:* Octrees for geometric feature extraction, not as addressable memory space for symbolic facts or linguistic tokens.

3. **Rosinol, A., et al. (2021).** "Kimera: From SLAM to Spatial Perception with 3D Dynamic Scene Graphs." *IEEE T. Robotics,* 37(4), 1191–1208.
   - *Contribution:* Hierarchical 3D scene graphs (places → objects → rooms) from visual SLAM.
   - *Gap:* Robotics geometric-semantic hybrid; not a general knowledge substrate supporting defeasible reasoning or shared human-AI memory palaces.

4. **Koch, T., et al. (2014).** "Indexing Linked Open Data Using Space-Filling Curves." *Proc. 9th Int. Conf. Semantic Systems,* 45–52.
   - *Contribution:* Morton codes for RDF triples in key-value stores.
   - *Gap:* Uses spatial indexing for compression / cache efficiency; no GPU-native traversal or cognitive loci addressing.

**K3D gap:** Morton codes and octrees are established for geometry and databases, but no architecture uses them as the primary address space for token-level semantic memory enabling dual-client spatial navigation.

---

## Cluster F — Ternary Logic Systems (post-Setun, 2000+)

Theoretical anchor: **Knuth, D. E. (1997).** *The Art of Computer Programming Vol. 2 — Seminumerical Algorithms, §4.1 Positional Number Systems.* Addison-Wesley. Establishes balanced ternary (digits -1, 0, +1) as optimal radix (minimizes average digit count per representable value for base-*e* arithmetic — *e* ≈ 2.718, closest integer 3).

1. **Ma, S., et al. (2024).** "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." *arXiv:2402.17764* (BitNet b1.58).
   - *Contribution:* LLM weights constrained to ternary {-1, 0, +1}; 1.58 bits/weight; 20× compression, 82 % less energy; multiplication-free add/sub/skip kernels.
   - *Gap:* Ternary as compression strategy for FP matrices / ALU optimization; retains high-precision activations and semantic embeddings; not ternary as *native knowledge representation*.

2. **Samsung / physical-ternary semiconductor research (ExtremeTech, 2019).**
   - *Contribution:* Physical ternary circuits for transistor density gains.
   - *Gap:* Hardware-only; targets transistor density, not epistemic structure.

3. **Setun (Brusentsov, 1958).** Moscow State University ternary computer — historical anchor, not a modern citation, but the reference point for any balanced-ternary claim.
   - *Contribution:* First production balanced-ternary computer; demonstrated practical viability of the radix.
   - *Gap:* Pre-GPU era; no path to modern throughput; no integration with learned components.

**K3D gap:** Existing ternary approaches treat the {-1, 0, +1} alphabet as a *compression strategy* for FP matrices or a *transistor-density curiosity*; none bind ternary to geometric presence/absence/unknown states, to meaning-polarity trits in reasoning, or to a native RPN opcode layer. K3D repurposes ternary as the foundational alphabet for spatial and logical epistemology, with ternary opcodes `0x70-0x76` as first-class RPN citizens.

---

## Cluster G — Defeasible / Non-Monotonic Logic Implementations

1. **Billington, D., et al. (2010).** SPINdle — open-source Java defeasible logic reasoner, handling theories with 10⁶+ rules, supporting defeaters and rule priorities.
   - *Contribution:* Linear-time defeasible logic for propositional theories; reference implementation.
   - *Gap:* CPU / JVM runtime; no GPU path; operates on abstract symbol strings decoupled from geometry.

2. **Covington, M. A. (2009).** *d-Prolog: Defeasible Prolog.*
   - *Contribution:* Prolog extended with defeasible inference rules.
   - *Gap:* Prolog runtime; CPU; no cognitive architecture integration.

3. **RuleML** — interchange standard for defeasible rules across implementations.
   - *Contribution:* Syntactic interchange format for defeasible rule bases.
   - *Gap:* Data exchange layer; not a runtime; not GPU.

4. **Calimeri, F., et al. (2019).** GPU-based Answer Set Programming solvers using CUDA for stratified / non-stratified rule sets.
   - *Contribution:* Demonstrates non-monotonic inference benefits from SIMD.
   - *Gap:* Batch ASP solving on GPU; not embedded in a resident cognitive substrate; not integrated with neural components.

**K3D gap:** SPINdle, d-Prolog, and GPU-ASP engines accelerate defeasible inference *on abstract symbols*. They do not support (a) volumetric non-monotonicity, (b) integration with a spatial memory palace, (c) inference *as* the substrate (rather than *on* the substrate), or (d) ternary-native rule priorities. K3D embeds defeasible inference into a 3D volumetric memory structure via the Reasoning-Paradigm Block (`0xA0-0xF1`) with the `gre_defeasible_resolver.cu` kernel sitting between the Nine-Chain Swarm and the Halting Gate.

---

## Cluster H — Dual Human-AI Shared Workspace Paradigms (2023+)

1. **Li, F.-F. (2024-2025).** "Spatial Intelligence" manifesto and World Labs launch.
   - *Contribution:* AI systems that generate and reason over interactive 3D environments; shift from text-based AI to world models.
   - *Gap:* Rendering / world-generation substrate; AI builds separate world models — no shared canonical structure with humans; no formal logic layer inspectable at the representation level.

2. **HCI research on "shared substrates" (2024-2025).** AI agents operating within the same Obsidian knowledge base or CAD environment as human designers.
   - *Contribution:* Moves past chat interfaces toward co-inhabited computational spaces.
   - *Gap:* Shared at the *document* or *rendered* level, not the *logical* level. Humans and AI co-edit a Markdown file; they don't co-navigate a shared knowledge logic.

3. **World Labs (2024-2025).** Photorealistic 3D scene generation for immersive navigation.
   - *Contribution:* Generates navigable 3D scenes from text prompts or references.
   - *Gap:* Rendering substrate; no formal logic layer; no symbolic representation humans can inspect or edit as native 3D structure; cohabitation at visual-appearance / natural-language annotation level.

**K3D gap:** Current shared workspaces are rendering substrates (pixels, voxels, meshes) or document repositories (Markdown, scene graphs), not *knowledge* substrates. None provide a shared *logical* runtime where a human's spatial assertion and an AI's volumetric inference execute on the same ternary-geometric bit-layer. K3D fills this via the Dual Client Contract + Form → Meaning 4-layer architecture, where the same canonical procedural substrate serves humans (aesthetic surface) and AI (executable surface).

---

## Single-Sentence Gap Summary per Cluster (for Paper A §3 intro paragraph)

| Cluster | One-sentence gap K3D fills |
|---------|-----------------------------|
| A — Method of Loci | No AI architecture uses hierarchical spatial indexing (palace → room → shelf → node) as the primary memory address space for machine cognition. |
| B — Procedural KR | No system treats RPN programs as GPU-resident, persistent, editable memory objects constituting the model's long-term knowledge store. |
| C — Small-model external memory | No sovereign single-GPU architecture combines small active params (7M–100M) with a writable, symbolic, spatially addressed memory supporting procedural execution. |
| D — GPU-native symbolic reasoning | No system implements defeasible, non-monotonic cognitive production systems directly in PTX/SASS without CPU offload. |
| E — Hierarchical spatial indexing | No architecture uses Morton codes / octrees as the primary address space for token-level semantic memory with dual-client spatial navigation. |
| F — Ternary logic | No system treats ternary {-1, 0, +1} as the *native* knowledge representation binding geometric presence/absence/unknown with logical polarity, rather than as a compression strategy. |
| G — Defeasible logic | No defeasible reasoner runs *as* the resident cognitive substrate on GPU; all current implementations are external CPU/JVM reasoners operating on abstract symbols. |
| H — Shared human-AI workspace | No system provides a shared *logical* runtime where human spatial assertions and machine inferences execute on the same representation layer; current systems share only rendered surfaces or documents. |

---

## Integration Plan for Paper A §3

Target: a 1-page Prior Work section, structured as:

1. **Paragraph 1 (3-4 lines):** Frame the gap — each of K3D's three headline contributions (C1 Absolute Sovereignty, C2 TRM-as-Avatar, C3 ActionBuffer) responds to an unmet need spanning memory architecture, reasoning substrate, and dual-client shared runtime.
2. **Paragraph 2-4:** Three groups of two clusters each:
   - C1 (Absolute Sovereignty) → clusters D + G (GPU-native symbolic reasoning + defeasible logic)
   - C2 (TRM-as-Avatar game loop) → clusters A + E (Method of Loci + hierarchical spatial indexing)
   - C3 (ActionBuffer binary contract) → clusters B + C (procedural KR + small-model external memory)
3. **Paragraph 5 (2-3 lines):** Cluster H (dual human-AI shared workspace) as the cross-cutting paradigm each contribution serves.
4. **Paragraph 6 (1-2 lines):** Companion-preprint note — semantic gravity (Paper B), hyper-modular + hyper-parallel (Paper C), Form→Meaning (Paper D), ternary-hardware (Paper E), layered sovereign cognitive stack (Paper F).

Total: ~1 page of 6-page budget. Per-cluster prose lifted directly from the "K3D gap" lines above.

---

## Citations Still Needed (for Paper A to be submission-ready)

- [ ] Verify Qiu et al. 2024 DOI resolves (may be hallucinated; DIS 2024 proceedings date-check)
- [ ] Verify Kremers et al. 2020 DOI resolves
- [ ] Confirm BitNet b1.58 arXiv ID (2402.17764 — likely correct but verify)
- [ ] Confirm RETRO ICML 2022 page range
- [ ] Confirm SPINdle citation — Billington author order and paper venue
- [ ] Find a better canonical Method-of-Loci AI citation if Qiu et al. is misattributed (fallback: Cicero *De Oratore* II.86–88 and Yates 1966 may suffice as historical anchors)
- [ ] Formal BitNet paper citation check (Microsoft Research authors — Shuming Ma et al.)
- [ ] TRM-style architectures: consider adding Hopfield Networks is All You Need (Ramsauer et al. 2020) if appropriate
- [ ] Setun computer: primary reference (Brusentsov et al. 1963 ACM publication if findable, or Malinovsky history)

Assign verification pass to Codex or a kimi_swarm lit-check pass before Paper A submission.

---

**Location**: `TEMP/CLAUDE_PAPER_A_PRIOR_ART_LIT_SWEEP_04.19.2026.md`
**Consumers**: Paper A §3 Prior Work, Papers B-F §2 Related Work (shared source of truth)
**Next owner**: Claude (Paper A skeleton integration), then Codex (DOI verification pass)
