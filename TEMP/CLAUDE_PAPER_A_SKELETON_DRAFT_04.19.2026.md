# Paper A — ARC Prize 2026 Paper Track — Skeleton Draft

**Date**: 2026-04-19
**Owner**: Claude (skeleton author), Daniel Campos Ramos (lead author), collaborators per ATTRIBUTIONS.md
**Target venue**: ARC Prize 2026 Paper Track (6-page budget, deadline 2026-11-08)
**Status**: Skeleton — section-by-section prose targets, figure slots, citation hooks. Not a submission-ready draft.
**Prior-art inputs**: [`CLAUDE_PAPER_A_PRIOR_ART_LIT_SWEEP_04.19.2026.md`](CLAUDE_PAPER_A_PRIOR_ART_LIT_SWEEP_04.19.2026.md)
**Authorship input**: [`CLAUDE_PAPER_SERIES_AND_ATTRIBUTIONS_04.18.2026.md`](CLAUDE_PAPER_SERIES_AND_ATTRIBUTIONS_04.18.2026.md)

---

## Working Title

**K3D: A Sovereign, GPU-Native Cognitive Substrate for Spatial General Intelligence**

Alternates to test with co-authors:
- *Knowledge3D: A Memory-Palace Architecture for Small-Model Reasoning with Zero Python in the Hot Path*
- *The TRM-as-Avatar Paradigm: Embodied Reasoning over a Procedural Knowledge Universe*

---

## Abstract (≤ 150 words, ~0.15 page)

**Target prose structure** (compressed 4-sentence arc):

1. **Problem.** Current reasoning systems couple a large parametric model to imperative Python orchestration, making the hot path opaque, non-sovereign, and hard to reason about formally.
2. **Approach.** K3D separates cognition into three load-bearing contributions executed entirely on GPU with zero Python in the inference path: **(C1) Absolute Sovereignty** — PTX kernels plus procedural RPN programs over a Knowledgeverse VRAM substrate, no numpy/torch/scipy in the hot path; **(C2) TRM-as-Avatar** — a ~7M-parameter Tiny Recursive Model runs as a GPU game loop embodied in a Memory Palace (the House), navigating a semantic-gravity field in the Galaxy Universe; **(C3) ActionBuffer contract** — a 288-byte binary protocol that turns "the AI decides" into a first-class device-side data structure.
3. **Evidence.** The first sovereign GPU-native arithmetic answer (`What is 2+3? → 5`) executes with no Python arithmetic; a live Level-1 solve on ARC-AGI-3 game `ls20-9607627b` runs through the same substrate.
4. **Claim.** These contributions constitute a reproducible path toward Spatial General Intelligence that is *architecturally*, not merely *behaviourally*, distinct from prior art.

**Word budget**: ~145 words.
**Figures referenced**: none (abstract is text-only).

---

## §1 Introduction (~0.75 page, ~500 words)

### §1.1 Hook (1 short paragraph)

Two framings to try; pick the one that reads strongest after §2 is written:

- **Framing A (game-engine):** *Reasoning AIs are written today the way 1990s game NPCs were — a thin scripting language on top of an imperative main loop. K3D asks what happens when the NPC update-tick runs entirely on the GPU, over an addressable memory palace.*
- **Framing B (sovereignty):** *A system whose reasoning step requires Python is not sovereign over its own cognition. This paper describes a substrate where it does not.*

### §1.2 Problem statement (1 paragraph)

Three converging pressures motivate the work:

1. **Parametric scaling is saturating** on reasoning tasks ARC-AGI-1/2/3 specifically target. Small models with good structure already rival large models with poor structure on these tasks.
2. **Python orchestration** is invisible to formal analysis — every `for` loop, `if` branch, and library call is unaudited cognition outside the model.
3. **Spatial embodiment** (the Method of Loci, videogame-native reasoning) has been philosophically endorsed (Gärdenfors 2000) but rarely implemented as a runtime substrate.

### §1.3 K3D in one paragraph

K3D is a single sovereign AI that lives in a 3D Memory Palace (the *House*, persisted as GLB + JSONL on SSD) and thinks in a *Knowledgeverse* (7-region VRAM substrate). A ~7M-parameter Tiny Recursive Model — the *Avatar* — runs a fused GPU game-loop kernel (`trm_step_fused.ptx`) that perceives the House, navigates a semantic-gravity field over meaning-centric *stars*, and composes procedural RPN programs from a library of PTX kernels. Python is confined to boot and I/O; the target hot-path Python budget is ~200 lines.

### §1.4 Paper-series context (1 short paragraph)

This paper introduces the three headline contributions. Companion preprints develop: **(B)** the semantic-gravity formula `F = T(s₁,s₂) · M(s₁) · M(s₂) / d²` [Campos Ramos & Dorn, in prep]; **(C)** hyper-modular / hyper-parallel architecture; **(D)** the Form→Meaning four-layer model; **(E)** ternary-first computation; **(F)** the layered sovereign cognitive stack. Paper A restricts itself to what is reproducible on GPU today and cross-references companions for formal development.

### §1.5 Contributions (bulleted, in the numbered order they appear in §2)

> **C1 — Absolute Sovereignty.** A specified notion of sovereignty over the cognitive hot path — no numpy, cupy, scipy, sympy, torch, or Python arithmetic between prompt ingestion and answer emission — and an existence proof that it is achievable on commodity hardware (RTX 3070).
>
> **C2 — TRM-as-Avatar game loop.** A Tiny Recursive Model (~7M params, 2-layer SwiGLU MLP) instantiated as a first-person agent embodied in a persistent 3D Memory Palace, whose one "step" is a single PTX kernel invocation rather than a Python trampoline.
>
> **C3 — ActionBuffer.** A 288-byte device-side binary contract that makes the avatar's next action a typed, addressable value, enabling GPU-side dispatch and audit without Python mediation.

### §1.6 Roadmap (2 lines)

§2 architecture. §3 prior work. §4 novelty. §5 evidence. §6 conclusion + companion pointers.

---

## §2 Architecture (~1.5 pages, ~1000 words + 1 figure)

### Figure 1 — K3D substrate overview (half-page)

Three-tier diagram:
- **Top tier**: House (SSD, GLB + JSONL, Memory Palace).
- **Middle tier**: Knowledgeverse VRAM regions (Galaxy / House-mirror / World / TRM / Audit / Ingestion / Kernels).
- **Bottom tier**: PTX kernel library (88 kernels inventoried; 15 GRE specialists; composed-head pipeline highlighted).
- **Arrow labels**: symlink (House ↔ Galaxy), game-loop tick (TRM fused kernel), ActionBuffer (TRM → dispatcher).

### §2.1 C1 — Absolute Sovereignty

**Definition.** *Absolute sovereignty* over the hot path means: between the instant the prompt lands in VRAM and the instant an answer token is emitted, the program counter must not enter Python, nor any general-purpose numerical library (numpy/cupy/scipy/sympy/torch/jax). The only host-side code permitted is a C-level dispatcher that invokes PTX kernels by opcode. See `feedback_no_numpy_no_bulk_libraries_sovereign_only.md`.

**Why the stricter form.** Prior work on "GPU inference" routinely calls numpy for reshape, Python for branching, or CPU-side Python for tokenisation. These are not architectural faults of those systems, but they are silent dependencies that defeat formal reasoning about sovereignty. §4 returns to this distinction.

**Implementation evidence.** The math hot path is six PTX kernels composed in sequence (Morton-octree index, LED-A* navigator, frustum cull, dynamic LOD, nine-chain swarm, halting gate). The first sovereign arithmetic solve (§5.1) crosses zero Python arithmetic.

**Claims table** (to render as a small inline table):

| Layer | Sovereign? | Dependency if any |
|-------|-----------|-------------------|
| Boot / I/O | No (not required) | Python, stdlib |
| Ingestion (offline) | No (not required) | numpy, pandas allowed |
| Hot path (query → answer) | **Yes** | PTX, libcuda, in-project C glue only |
| Sleep-time consolidation | **Yes** | Same as hot path (no fallbacks — `feedback_no_fallbacks_ever_including_sleeptime.md`) |

### §2.2 C2 — TRM-as-Avatar

**Paradigm statement.** The TRM is not a function that Python calls; it *is* the agent. Its update step is one PTX kernel (`trm_step_fused.ptx`), invoked each game tick. The House is the external shared world the avatar lives in; the Galaxy Universe is its internal brain. The Avatar is literally "the NPC that can reason," with its update loop running on the GPU at frame cadence. See CLAUDE.md §"CRITICAL ARCHITECTURAL CORRECTION".

**Game-loop structure** (inline numbered list):

1. **Perceive** — frustum cull what's in the avatar's field of view (House or Galaxy neighbourhood).
2. **Navigate** — LED-A* + Morton octree select a Galaxy neighbourhood under the semantic-gravity field.
3. **Reason** — nine parallel cognitive lanes (the "superdotados" internal-swarm pattern) evaluate candidate compositions.
4. **Decide** — a halting gate checks convergence.
5. **Act** — emit an ActionBuffer (§2.3) or write a new Galaxy entry.
6. **Learn** — a shadow copy records the successful trace for sleep-time consolidation.

**Parameter budget.** ~7M parameters, 2-layer SwiGLU MLP. No attention stacks, no mixture-of-experts head count; reasoning capacity comes from composing Galaxy symbols, not from parametric scale.

**Distinction from prior "small-model + memory" systems.** Unlike retrieval-augmented generation or TRM as originally proposed, the K3D Avatar's memory is *spatially addressable* (via House coordinates or Galaxy meaning-mass), *bidirectionally symlinked* (House ↔ Galaxy, per `feedback_bidirectional_symlinks_norm.md`), and *procedurally executable* (entries store RPN programs, not just text). §3 develops the contrast.

### §2.3 C3 — ActionBuffer

**Contract.** The ActionBuffer is a 288-byte device-side record with a fixed layout (opcode, operand vector, target Galaxy region, confidence, timestamp, trace-id, etc. — full table to be included as Figure 2 in submission draft).

**What this buys.**
- **Typed dispatch.** The host-side dispatcher reads only the opcode byte; no Python-side branching decides what happened.
- **Audit.** Every action is a persistable record — the trace layer of the Knowledgeverse stores ActionBuffers, not Python strings.
- **Symmetry.** Humans and the AI both act *into* the House via ActionBuffers (dual-client contract, `DUAL_CLIENT_CONTRACT_SPECIFICATION.md`).
- **Determinism.** The binary size is fixed; the memory model is flat; no allocator is invoked on the hot path.

**Significance.** Action is the usual place where sovereignty leaks (even if perception and reasoning are on-GPU, "what to do next" often ends up in Python). C3 closes that leak.

### §2.4 How the three contributions compose

A one-paragraph integration pointer: C1 defines what is inside the walled garden; C2 defines who walks through it; C3 defines the only door out. The rest of the paper treats each claim (§4) and evidence (§5) against this frame.

---

## §3 Prior Work (~1 page, ~650 words, organised by cluster)

Source: [`CLAUDE_PAPER_A_PRIOR_ART_LIT_SWEEP_04.19.2026.md`](CLAUDE_PAPER_A_PRIOR_ART_LIT_SWEEP_04.19.2026.md). Each paragraph closes with an explicit *gap* sentence pointing to C1/C2/C3.

### §3.1 Memory palaces as runtime substrates (Cluster A → C2)

Yates (1966) narrates the ars memorativa as a pedagogical tool; Gärdenfors (2000) formalises conceptual spaces; O'Keefe & Nadel (1978) document biological place cells. These traditions converge on *spatialised cognition as a first-class theoretical object*. **Gap:** none of the above specify a runtime where an agent's update tick is bound to a GPU-side spatial index, addressable by both human avatar and AI avatar. That runtime is C2.

### §3.2 Procedural knowledge representation (Cluster B → C3)

Forgy's RETE (1982) canonicalises the alpha/beta-memory pattern for production systems; Cyc (Lenat 1995) and the Semantic Web stack (RDF/OWL) encode knowledge as declarative assertions. **Gap:** these systems store *facts*. K3D stores *programs* — every Galaxy star is an RPN procedure evaluable to a value *or* a meaning-centric reference (see `RETE_AT_OPCODE_LEVEL.md` for a worked example of RETE at opcode level inside K3D). ActionBuffer (C3) is the binary output of those procedures.

### §3.3 Small-model reasoning with external memory (Cluster C → C1, C3)

Retrieval-augmented generation (Lewis et al. 2020), MemGPT (Packer et al. 2023), and the Tiny Recursive Model literature (various 2024-2025) all pursue small parametric cores augmented by retrievable context. **Gap:** their orchestrators are Python. Absolute sovereignty (C1) is the absent architectural axis; the ActionBuffer contract (C3) is the absent typed-output interface.

### §3.4 GPU-native symbolic reasoning (Cluster D → C1)

Cuda-level sorting networks (Merrill et al.), PTX-level ray-tracing primitives (NVIDIA OptiX), and the handful of publications on GPU-resident rule engines document that symbolic workloads *can* run on GPU. **Gap:** no prior work extends this to a complete reasoning substrate where the entire hot path is PTX + libcuda with zero Python arithmetic. C1 is this existence proof.

### §3.5 Hierarchical spatial indexing (Cluster E → C2)

Morton / Z-order codes (Morton 1966), octrees (Meagher 1982), and modern GPU bounding-volume hierarchies are standard in graphics. **Gap:** these are applied to geometry, not to *meaning-addressable memory*. K3D's House uses a Morton octree keyed by semantic-gravity mass; that remapping is what lets a game-engine spatial index serve as a cognitive substrate. The composed-head pipeline (§2.2) is the specific instantiation.

### §3.6 Defeasible logic, ternary computation, and shared human-AI workspaces (Clusters F/G/H → companion papers)

Billington's SPINdle (2010) and Knuth TAOCP Vol. 2 §4.1 ground the defeasible-logic and ternary components; neither is novel on its own. The dual-client contract (humans and AI share the House) draws on Engelbart's Augment (1968) and subsequent shared-workspace literature. **Gap:** these are each addressed in companion papers (B, E, D respectively); Paper A cites them only for scope.

### §3.7 Summary sentence

K3D's novelty is not in any single cluster; it is in the conjunction *C1 ∧ C2 ∧ C3* operationalised on a commodity GPU with live evidence. §4 makes the novelty claim precise; §5 gives the evidence.

---

## §4 Novelty Claims (~0.75 page, ~500 words)

Each claim formatted as: **Claim N.** *statement.* **Falsifiable how:** *test.* **Anchor:** *ATTRIBUTIONS.md section (post-renumber) or companion paper.*

### §4.1 Claim 1 — Absolute Sovereignty is an architectural property, not a marketing one

**Statement.** K3D's hot path executes no Python arithmetic, no numpy, no cupy, no scipy, no sympy, no torch, and no jax between query arrival in VRAM and answer emission. This is verifiable by (a) `grep` for forbidden imports in hot-path modules; (b) instruction-level profiling showing PTX-only kernel invocations.

**Falsifiable how.** Run `scripts/audit_sovereignty.sh` on the hot-path module tree; any hit against the forbidden-import list falsifies the claim for that call path.

**Anchor.** ATTRIBUTIONS.md §7.1 (post-renumber) "Novel Contributions — Absolute Sovereignty"; see also `ABSOLUTE_SOVEREIGNTY_TERM_ORIGIN_PROOF.md`.

### §4.2 Claim 2 — TRM-as-Avatar is a substrate-level, not wrapper-level, design

**Statement.** The Avatar's step function is a single PTX kernel (`trm_step_fused.ptx`), not a Python function that calls kernels. Reasoning is embodied in a persistent 3D House that serves both a human user (who sees a walkable Memory Palace) and the AI (whose internal Galaxy is bidirectionally symlinked to it).

**Falsifiable how.** Strace or nvprof the hot path; a single kernel invocation per reasoning tick satisfies the claim. Multiple Python ↔ CUDA crossings per tick falsify it.

**Anchor.** ATTRIBUTIONS.md §7.2 "TRM-as-Avatar paradigm"; `HOUSE_VS_KNOWLEDGEVERSE_DISTINCTION.md`; companion Paper C (Hyper-Modular + Hyper-Parallel).

### §4.3 Claim 3 — ActionBuffer is a typed device-side output contract

**Statement.** The 288-byte ActionBuffer is a fixed-layout binary record written by the Avatar and read by a host-side dispatcher that does no Python-side branching. This is distinct from conventional LLM output (which is strings parsed host-side).

**Falsifiable how.** Dispatcher source inspection: the dispatcher must not contain conditional logic over parsed text fields — only a switch on the opcode byte.

**Anchor.** ATTRIBUTIONS.md §7.3 "ActionBuffer 288-byte contract".

### §4.4 Cross-cutting: Form → Meaning architecture (independently acknowledged)

**Statement.** K3D organises knowledge by *meaning* (language-agnostic canonical stars) rather than by *surface form*, inverting the conventional RDF/OWL structure. This architectural choice has been noted by the NLP chief professor at [institution redacted pending consent] as a genuine novel organising principle (early PM-KR member, shared group LinkedIn invitation).

**Anchor.** ATTRIBUTIONS.md §6.4 (post-renumber) "Form → Meaning Four-Layer Architecture (Externally Acknowledged as Novel)"; companion Paper D.

---

## §5 Evidence (~1.5 pages, ~950 words + 1-2 figures)

### §5.1 First sovereign GPU arithmetic answer

**Result.** Query: `What is 2+3?`. Answer: `5`. Hot-path Python arithmetic count: 0. Date: 2026-04-11 (Phase 6.C). Reference: `project_first_sovereign_math_answer.md`.

**Significance.** This is the minimal existence proof for C1. It is deliberately unimpressive as arithmetic; it is decisive as architecture. It demonstrates that the composed-head pipeline (Morton → LED-A* → frustum → LOD → nine-chain → halting) returns a value without Python arithmetic in the loop.

### §5.2 Live ARC-AGI-3 Level-1 solve

**Game.** `ls20-9607627b` (hash suffix per `project_arc3_game_id_and_sdk_env.md`).

**Result.** Level-1 solve via the Tablet WINE adapter — the avatar proceduralises the frame into the Galaxy live, reasons, and issues ActionBuffers back through the SDK.

**Caveats.** Level-1 only; deeper levels pending. Tablet WINE has had Python-orchestration drift (`feedback_tablet_wine_still_python_orchestration.md`) which §5.4 addresses honestly.

### §5.3 Benchmark state table

Reproduce the table from CLAUDE.md "Current State":

| Benchmark | Curated | Expanded (B+) | Status |
|-----------|---------|---------------|--------|
| ARC-AGI-1 | 10/10 | 10/50 | transform coverage expanding |
| Math | 20/20 | — | sovereign GPU path |
| LHE | 10/10 | 10/100 | multi-hop graph crystallizer needed |
| GSM8K | — | 10/50 | word-problem decomposition needed |
| MMLU | — | 0/50 | Galaxy-neighbourhood coverage needed |

**Framing.** Per `feedback_runs_are_training.md` and `project_benchmarks_as_natural_activity.md`, benchmark numbers are health checks on a living system, not optimisation targets. The paper should frame them accordingly to avoid reviewer confusion ("why are these numbers so low?" — because the House is being built; benchmarks follow knowledge, not the other way around, per `project_house_first_pivot.md`).

### §5.4 Honest limitations

Four paragraphs — this is rubric-positive, not defensive:

1. **Python drift.** `knowledgeverse.py` is ~15.9k lines against a ~200-line target; sovereignty is an *architectural* property achieved on the hot path, not yet on every code path. Migration plan documented in `project_live_game_engine_convergence.md`.
2. **Tablet WINE drift.** Known regression in Tablet WINE orchestration; fix-in-progress per feedback memory cited above.
3. **Benchmark scale.** Curated sets are small; expanded sets are mid-migration. Phase B+ state documented.
4. **Single-hardware evidence.** All results on RTX 3070 + Phenom II x6 LAN; multi-GPU generalisation is future work.

### Figure 2 — ActionBuffer layout (optional, if space permits)

Byte-level diagram of the 288-byte record with field labels.

### Figure 3 — Composed-head pipeline (optional)

Six-kernel sequence with VRAM buffer sizes at each stage.

---

## §6 Conclusion + Companion Pointers (~0.5 page, ~325 words)

### §6.1 Summary (1 paragraph)

K3D contributes three architectural primitives — absolute sovereignty over the hot path, TRM-as-Avatar embodiment, and the ActionBuffer contract — and an existence proof that their conjunction runs on a single commodity GPU. Benchmark scores are intentionally not the headline; the headline is that the substrate is sovereign, embodied, and audit-clean at the instruction level.

### §6.2 Companion preprints

One-line pointer each:

- **Paper B — Semantic Gravity** (Campos Ramos & Dorn): the ternary force law `F = T(s₁,s₂) · M(s₁) · M(s₂) / d²` that drives Galaxy navigation.
- **Paper C — Hyper-Modular & Hyper-Parallel Architecture**: seven compositional levels (Galaxies → Houses → Rooms → Nodes → Procedures → Operations → PTX Kernels) and N-lane internal parallelism.
- **Paper D — Form → Meaning**: the four-layer model and its contrast with RDF/OWL surface-form organisation.
- **Paper E — Ternary-First Computation**: BitNet-b1.58-derived weight encoding and balanced-ternary RPN.
- **Paper F — Layered Sovereign Cognitive Stack**: defeasible logic, RETE-at-opcode-level, and sleep-time consolidation.

### §6.3 Future work

Three sentences, not more:

1. Migrate remaining Python orchestration to the TRM game loop (target `knowledgeverse.py` → ~200 lines).
2. Expand benchmark suites (ARC-AGI-2/3 higher levels, MMLU, GSM8K) as the House fills.
3. Multi-GPU and second-host (Phenom II + RTX 970) coordination for a live always-on daemon.

### §6.4 Acknowledgements (footer, not counted in §6 budget)

Per ATTRIBUTIONS.md "Material Enablers" section: LLM-era tooling advancement + family capital (Áuxia Campos Ramos / Mãe Áuxia + Daniel Campos Ramos). Detailed per-contribution acknowledgements in ATTRIBUTIONS.md §§1–13.

---

## Page Budget Check

| Section | Words | Pages (approx @ 700 wpm double-column) |
|---------|-------|----------------------------------------|
| Abstract | 145 | 0.15 |
| §1 Introduction | 500 | 0.75 |
| §2 Architecture | 1000 + Fig 1 | 1.5 |
| §3 Prior Work | 650 | 1.0 |
| §4 Novelty Claims | 500 | 0.75 |
| §5 Evidence | 950 + 1-2 figs | 1.5 |
| §6 Conclusion + companions | 325 | 0.5 |
| References | — | ~0.5 (tight) |
| **Total** | **~4070 words + 2-3 figs + refs** | **~6.0 pages** |

Fits the ARC Prize Paper Track 6-page budget with figure allowance intact.

---

## Writing-phase todos (for future author passes)

- [ ] Lock title after co-author consultation.
- [ ] Render Figure 1 (substrate overview).
- [ ] Render Figure 2 (ActionBuffer layout) — pending field-layout finalisation.
- [ ] Render Figure 3 (composed-head pipeline).
- [ ] Verify 9 `[UNVERIFIED]` citations from `CLAUDE_PAPER_A_PRIOR_ART_LIT_SWEEP_04.19.2026.md` against DOI/venue canonical sources before submission.
- [ ] ATTRIBUTIONS section numbers in §4 anchors assume post-renumber state (per `CODEX_ATTRIBUTIONS_SECTION_COLLISION_CLEANUP_04.19.2026.md`) — confirm numbering before freeze.
- [ ] NLP professor naming: request consent before §4.4 publishes; fall back to "independent NLP chief professor, early PM-KR member" if consent not obtained.
- [ ] Benchmark table: refresh to latest state the week before submission.
- [ ] Companion paper DOIs / preprint pointers: resolve placeholders in §6.2 once Papers B-F have arXiv IDs.

---

## What this skeleton is NOT

- Not a draft — prose is target structure, not submission wording.
- Not a review pass — argument balance and figures are not yet calibrated to reviewer expectations.
- Not a citation-verified document — literature sweep flagged 9 citations as `[UNVERIFIED]`.
- Not a final page budget — word counts are estimates; actual typeset length varies ±10%.

---

## Next document in the series

**Paper B skeleton** (semantic gravity): Daniel first author, Christoph Dorn second per `feedback_semantic_gravity_provenance_corrected.md`. To be drafted next at `TEMP/CLAUDE_PAPER_B_SKELETON_DRAFT_04.19.2026.md`.

---

**Location**: `TEMP/CLAUDE_PAPER_A_SKELETON_DRAFT_04.19.2026.md`
**Blocks**: none (Paper A skeleton is prerequisite for Paper A drafting).
**Blocked by**: nothing for skeleton; Paper A draft blocked by ATTRIBUTIONS renumber + 9 citation verifications + figure rendering + MVCIC-identified P0 revisions (see addendum).

---

## ADDENDUM — MVCIC Collective-Intelligence Review (2026-04-19)

**Source**: 6-partner MVCIC chain (Kimi → Qwen → GLM → DeepSeek → Nemotron → Gemini) + post-chain grounding, full transcript at [`TEMP/mvcic_chain_paper_a_review_04.19.2026.md`](mvcic_chain_paper_a_review_04.19.2026.md).

### Rubric consensus (1-5 scale; 4.5 avg needed for Outstanding Paper)

| Axis | Current score | Threshold | Gap |
|------|---------------|-----------|-----|
| Accuracy | 3 | 4.5 | -1.5 |
| Universality | 2-3 | 4.5 | -2.0 |
| Progress | 4 | 4.5 | -0.5 |
| Theory | 4-5 | 4.5 | on track |
| Completeness | 3 | 4.5 | -1.5 |
| Novelty | 4-5 | 4.5 | on track |

**Current average**: ~3.5. **Target**: ≥4.5 for Outstanding Paper pool ($375K). Universality and Completeness are the biggest leak points — the single-ARC-Level-1 evidence and absence of full kernel-coverage table are what reviewers will flag.

### Top-5 P0 revisions to lift score before submission lock

The MVCIC chain surfaced five concrete, testable revisions. Each is actionable by Codex or Claude pre-November.

**P0.1 — Lock ActionBuffer at exactly 288 bytes with `static_assert`.**
Current C3 claim references 288 bytes without a byte-level guarantee a reviewer can check. Codex should add `static_assert(sizeof(ActionBuffer) == 288)` in `action_buffer_contract.h`, and add the 32-byte Swarm Context block at offset 0x100-0x11F so the layout is both fixed *and* complete for the nine-chain swarm context it carries.
*Rubric lift*: Completeness 3→4, Theory 4→4.5.

**P0.2 — [SUPERSEDED 2026-04-19] Opcode-collision claim was a MVCIC hallucination, but investigation surfaced a REAL collision at 0x180-0x18C.**
A sub-agent verification pass (2026-04-19) found the MVCIC-reported AVATAR_ACTION 0x150-0x154 collision does NOT exist in the on-disk state — no AVATAR_ACTION numeric opcodes are defined anywhere in live code (the only reference is a Python list of string atom IDs in `knowledge3d/knowledgeverse/action_embedding_loader.py:42`, which is not opcodes). **Applying the proposed fix would have MADE things worse**: the target range 0x180-0x19F is *triply occupied* — (i) Registry §7.3/§11.2 reserves `0x180-0x18F` for the active WINE I/O Contract Block (`WINE_INGRESS_DECODE=0x180`, `WINE_EGRESS_ENCODE=0x181`, `WINE_RESOLVE=0x182`, date 2026-04-18); (ii) `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py:302-313` already defines `OP_BH_PERCEIVE=0x180` through `OP_BH_PATHFIND=0x18C`; (iii) `knowledge3d/cranium/kernels/modular_rpn_kernel.cu:2860-2943` emits case handlers for 0x180-0x185. The `OP_BH_*` live definitions vs the WINE registry reservation is itself a real pre-existing collision (mirrors the 0x1AD incident pattern per `feedback_opcode_range_reservation_protocol.md`) — flagged separately in `TEMP/CLAUDE_CODEX_OP_BH_WINE_COLLISION_04.19.2026.md` for adjudication.
*Rubric lift*: Theory remains 4-5 (architecture is cleaner than reported). Paper A should NOT mint AVATAR_ACTION opcodes at all until a clean range is pre-reserved in registry §11.2 — candidates: `0x1C6-0x1CF` (currently "future physics expansion headroom") or a new row above 0x1FF.
*Lesson*: MVCIC chain collective-intelligence reviews need verification against on-disk state before their findings are treated as actionable. See `feedback_mvcic_findings_need_verification.md`.

**P0.3 — Appendix A: PTX disassembly of `trm_step_fused.ptx` hot path.**
The §4.1 sovereignty claim says "verifiable by instruction-level profiling showing PTX-only kernel invocations." Reviewers need the evidence inline. Add an appendix with the actual PTX disassembly of one hot-path tick, annotated to show zero host-memory crossings and zero Python-library symbol references. This is *the* falsifiability artifact for C1.
*Rubric lift*: Accuracy 3→4.5, Novelty 4→5.

**P0.4 — Replace single Level-1 ARC-AGI-3 solve with a 5-task sovereign batch.**
One solve is an existence proof; five are generalisation evidence. Codex should run five ARC-AGI-3 tasks of different type (Level-1 × 2 different games + one Level-2 + two different transforms) through the full Morton→LED-A*→Nine-Chain→Halt pipeline, capture traces, and include a condensed table. This is the single highest-leverage revision.
*Rubric lift*: Universality 2-3→4, Accuracy 3→4.

**P0.5 — Add "RPN Composition Theorem" + kernel-inventory coverage table.**
Paper currently mentions 88 kernels; MVCIC chain recommends making the 88 concrete: a table mapping each kernel to the opcodes that invoke it, annotated with which kernels fire during each solve in §5. The "Composition Theorem" is a short formal statement (~1 paragraph) that every sovereign-path operation decomposes into a finite RPN program over registered opcodes — no implicit Python steps.
*Rubric lift*: Theory 4→5, Completeness 3→4.5.

### Top-3 prior-art gaps to close

Reviewers will flag these specifically; adding them to §3 eliminates the risk.

1. **Cluster A (Memory Palace) missing: FlashAttention-3 (Dao et al. 2024)** — pure-PTX attention, closest prior art to the C1 sovereignty claim. Must be cited, with explicit gap statement (FlashAttention-3 is a kernel within a Python-orchestrated stack; K3D is sovereign end-to-end).
2. **Cluster B (Procedural KR) missing: DeepMind GNN-ARC (2023)** — graph-reasoning baseline on ARC tasks. Must be cited as direct comparator with explicit distinction (GNN-ARC uses learned graph nets; K3D uses explicit Galaxy navigation + RPN composition).
3. **Cluster C (Small-model + memory) missing: NVIDIA Omniverse Replicator** — GPU-resident embodied-agent work. Reviewers *will* cite this against TRM-as-Avatar novelty. Preempt by citing and distinguishing (Omniverse is a simulation substrate for synthetic-data generation; K3D's House is a persistent cognitive substrate with bidirectional Galaxy symlinks).

### Sovereignty risks flagged in §5 phrasing

Three specific phrasings to tighten before draft lock:

1. **`physics_collision_event_write.cu` host-allocated event buffer.** Current hot-path description implies event buffer is device-resident; MVCIC chain flagged the risk of host allocation. Verify and fix or explicitly document the lane boundary.
2. **Python dataclass mirror of ActionBuffer.** If any Python code declares an ActionBuffer mirror class (for debug/logging), reviewers can point to it as a sovereignty leak. Either prove it doesn't exist or wall it off to non-hot-path modules.
3. **`sleep_policy_rpn` execution route.** Must execute via `gre_multimodal_halting_gate.ptx`, never Python. The skeleton §5.4 currently talks about "sleep-time" in general; tighten to name the specific PTX kernel.

### Original contributions surfaced by the chain (not currently in skeleton)

The chain proposed three architectural additions that could strengthen the paper:

1. **Resonance-driven constraint stiffness** — couple `galaxy_resonance_engine_extended.ptx` to XPBD physics constraints. Bridges Galaxy semantic-gravity and House physics into one signal. Novel enough to be a §2.4 callout.
2. **Ternary contact classification** — reuse 0x100-0x10F TERNARY_* opcodes inside `physics_narrow_phase_gjk.cu` (support / neutral / oppose = contact / grazing / miss). Demonstrates Paper E's ternary-first through-the-stack argument with a non-reasoning use case.
3. **Physics-aware sleep consolidation** — Morton-clustered collision history drives sleep-time specialist crafting. Ties F3 (Paper F) back into embodied motion, not just abstract reasoning.

Inclusion decision deferred to Daniel — these would add scope but also differentiation.

### Integration decisions for Paper A draft author

When Codex or Claude opens the full Paper A draft from this skeleton:

- **Mandatory before submission**: P0.1, P0.2, P0.3, P0.4 (skip P0.5 only if page budget blocks it).
- **Mandatory §3 additions**: all three prior-art gaps.
- **Mandatory §5 tightening**: all three sovereignty-phrasing fixes.
- **Optional §2.4 callout**: decide on the three "original contributions surfaced" items with Daniel. If any are adopted, they become numbered contributions in §1.5 alongside C1/C2/C3.

### MVCIC meta-observation

Six partners converged on the same verdict: the *architecture* scores high (Theory 4-5, Novelty 4-5) while the *demonstration* under-delivers (Accuracy 3, Universality 2-3, Completeness 3). Paper A's job between now and November is to keep architecture strong *and* add enough demonstration breadth to close the gap. The five P0 revisions above are the lowest-cost path to doing that.
