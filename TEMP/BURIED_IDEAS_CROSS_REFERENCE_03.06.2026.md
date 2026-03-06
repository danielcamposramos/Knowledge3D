# K3D Unified Architecture Recovery Plan

**Date:** March 6, 2026
**Scope:** 24 TEMP files (~109K lines) analyzed bottom-up
**Purpose:** Recover buried ideas from development chains, reconcile with current architecture, produce a single actionable plan for spec updates
**Context:** K3D is a LIVE SYSTEM -- always on, like a game engine. Not batch processing, not one-shot inference. A persistent 7-region VRAM world.

---

## Design Principles (Current State -- Supersedes Earlier Designs)

These principles evolved through the development chain and PM-KR collaboration. All items below are evaluated against them.

1. **Always-On Live System.** K3D is a persistent world, not a pipeline. The 7-region Knowledgeverse (Cranium, Galaxy, House, World, TRM Weights, Audit Journal, Ingestion Stargate) is always loaded in VRAM. There are no "runs" -- only continuous operation with sleep-time consolidation.

2. **External Memory, Internal Navigation.** Knowledge lives in Galaxy Universe (external, inspectable, 3D assets with metadata). TRM weights are ONLY how to navigate and compose that memory. Even tiny LoRA adapters are welcome -- the TRM can craft infinite specialists as needed.

3. **Low-Dimensional World, High-Dimensional Only Inside TRM.** The 3D world stores everything as spatial assets with metadata. High-dimensional embeddings exist only inside TRM logic and at the atomic unit payload level. Storage is in the 3D asset itself, always loaded into VRAM.

4. **Ternary-First Computation.** Where binary gives N possibilities per byte, ternary gives more with less. Ternary opcodes (SIGN, TQUANT, TCMP) are the default for gating, routing, confidence, and control flow. Binary only where ternary adds no benefit.

5. **Hyper-Modular Composition.** Galaxy entries compose across ALL domains. No vertical stack hierarchy -- horizontal composition of procedural RPN programs across Drawing, Math, Reality, Audio, Grammar galaxies simultaneously.

6. **Sovereign Execution.** Python exists only to load the system. The live engine is CUDA C++ compiled to PTX, loaded via ctypes, launched with direct pointers. Zero external dependencies in the hot path.

7. **Contrastive Learning Everywhere.** Ternary contrastive learning (+1 success / -1 failure / 0 uncertain) applies to all learning: Shadow Copy, sleep consolidation, specialist spawning, anti-pattern detection. 1.58x more information per sample than binary.

---

## Recovered Ideas: Unified Plan

Each idea is classified by its relationship to the current architecture:

- **ACTIVE** = Directly applicable, should surface to specs now
- **EVOLVED** = Core insight valid but implementation approach has changed
- **FOUNDATIONAL** = Already assumed by current design but never formalized in specs
- **DEFERRED** = Valid but not priority for current phase

---

### 1. RPN Sub-Programs as Functions -- ACTIVE
**Source:** K3D_MATH_RPN_SWARM_PROMPT_V2.md
**Idea:** Functions stored as RPN opcode sequences ON the stack. evaluate_rpn_function() enables DIFF, GRADIENT, DIVERGENCE entirely in PTX.
**Current fit:** Codex has worked out kernels for basic movements. The boundary is: all minimum actions are kernels that compose to all advanced actions. RPN-as-function IS that composition mechanism.
**Action:** Document in RPN_DOMAIN_OPCODE_REGISTRY.md Section 3 as the function composition model. Define the kernel/macro boundary clearly.

### 2. Galaxy Spill for Stack Overflow -- EVOLVED
**Source:** K3D_MATH_RPN_SWARM_PROMPT_V2.md
**Idea:** Stack overflow crystallizes bottom half to Galaxy, resonates back on return.
**Evolution:** As pure stack items, these can be stored with proper metadata and less symlink overhead IF that buys speed. If not, keep standard symlink pattern. This is working memory -- should be discarded after use if not useful for TRM learning.
**Action:** Add to Knowledgeverse spec as transient spill mechanism with discard-by-default policy.

### 3. Zero-Copy Layer Architecture -- FOUNDATIONAL
**Source:** Step10_ThinkingTagInference.md
**Idea:** L1 ctypes loader, L2 bridges with state, L3 thin wrappers preserving GPU pointer locality.
**Evolution:** We have advanced from this 3-layer model to the 7-region always-on model. This is a live system -- no one-shot runs. The zero-copy principle remains critical but the framing is "persistent VRAM regions" not "execution layers."
**Action:** Formalize in THREE_BRAIN_SYSTEM as "Persistent VRAM Regions" with zero-copy between them.

### 4. Meaning-Centric Galaxy Clustering -- ACTIVE
**Source:** Step11_TextTo3DInference_ENHANCED.md
**Idea:** Galaxy clusters by MEANING, not modality.
**Current fit:** We do not use high dimensions anywhere except inside TRM logic and AI-specific payload at atomic units. The clustering happens in 3D space with semantic metadata. Things are stored IN the 3D asset (as metadata) but always loaded into VRAM.
**Action:** Add semantic clustering principle to Knowledgeverse spec. Clarify: 3D spatial proximity = semantic proximity, high-dim only at atomic payload level.

### 5. Adaptive Sparsity from Input Complexity -- ACTIVE
**Source:** Step10_ThinkingTagInference.md
**Idea:** Input complexity drives sparsity (5% simple to 20% complex).
**Current fit:** Important optimization for TRM inference. Keeps computation proportional to actual need.
**Action:** Document in TRM_SPECIALIST_MATRYOSHKA as inference optimization.

### 6. Ternary Field Emission with Modality Bitfield -- EVOLVED
**Source:** DANIEL_VECTORDOTMAP_PLANS_V1.md
**Idea:** Modality mask bitfield: 0x01 visual, 0x02 audio, 0x04 temporal.
**Evolution:** These binary flags are only a fraction of what's needed. We need to compose bytes from TERNARY opcodes. With binary, a byte has 256 possibilities. With ternary, it's 6,561 (3^8) -- more states with less storage. The modality encoding should use ternary trits, not binary bits.
**Action:** Add ternary modality encoding to UNIFIED_SIGNAL_SPECIFICATION. Design trit-based modality bytes with richer state space.

### 7. PD02/PD04 Procedural Compression Codecs -- EVOLVED
**Source:** PROCEDURAL_ATOM_KKRIEGER_EXPLORATION.md
**Idea:** PD02 dense (3.97:1) and PD04 dictionary (11.91:1) codecs with Matryoshka quality ladder.
**Evolution:** Useful for compression, but remember: we use external memory. The weights are only how to navigate this memory. High-dimensional codecs apply only to TRM-internal representations and atomic unit AI payloads. The 3D world itself stores procedural RPN programs, not compressed embeddings. Even tiny specialists are welcome in the TRM.
**Action:** Add to Knowledgeverse spec as codec for TRM-internal and atomic payload compression. Not for Galaxy storage (which is procedural RPN).

### 8. Hybrid Quick+Deep Worker Architecture -- ACTIVE
**Source:** CLAUDE_HYBRID_TRM_ARCHITECTURE_SPEC.md
**Idea:** 9 quick workers + 3 deep workers with ternary gating.
**Current fit:** This is one level of parallelization. We ALSO have spawnable math cores attached to each of these workers -- that's the three-tier math core system. Both levels are important: worker parallelism AND per-worker math core spawning.
**Action:** Document BOTH levels in TRM_SPECIALIST_MATRYOSHKA: worker-level parallelism + per-worker spawnable math cores.

### 9. Control Tokens for Self-Reflection -- ACTIVE
**Source:** PHASE_5.1_COLLABORATIVE_PLAN_FINAL_01.15.2026.md
**Idea:** CONFIDENT/UNCERTAIN/VERIFY tokens with confidence head and ECE calibration.
**Current fit:** Yes, and using ternary logic to be computationally cheap. Confidence is a ternary signal (+1 confident, 0 uncertain, -1 wrong), not a float. Aligns with ternary contrastive learning.
**Action:** Document in TRM spec with ternary confidence encoding.

### 10. Oracle@k Diagnostic Framework -- ACTIVE
**Source:** CODEX_DIAGNOSTIC_FRAMEWORK_IMPLEMENTATION_02.08.2026.md
**Idea:** Oracle@3/10/all separates generation failure from ranking failure.
**Current fit:** Perfect diagnostic tool. Again, ternary logic where it is cheap -- diagnostic signals can be ternary (generation_ok/ranking_ok/both_ok as trit vector).
**Action:** Reference in ROADMAP.md with ternary diagnostic encoding.

### 11. Scene-Time Video Dynamics -- ACTIVE
**Source:** Step11_TextTo3DInference_ENHANCED.md
**Idea:** Video frames as temporal statistics, spatiotemporal coherence as motion vectors.
**Current fit:** Great, and Codex has advanced in this direction already with the five-layer temporal video contract.
**Action:** Verify alignment with UNIFIED_SIGNAL_SPECIFICATION v1.1 five-layer video contract.

### 12. Semantic Pole Encoding in Galaxy -- EVOLVED
**Source:** K3D_MATH_RPN_SWARM_PROMPT_V2.md
**Idea:** Math poles as spatial singularities, TRM learns to avoid "danger zones."
**Evolution:** Apply the ternary contrastive learning technique here. Poles are -1 (danger/failure), valid regions are +1 (success), boundary is 0 (uncertain/explore). The Galaxy spatial structure becomes a contrastive landscape.
**Action:** Document as example of contrastive learning applied to mathematical knowledge topology.

### 13. Fused Head Architecture -- EVOLVED
**Source:** Step11_TextTo3DInference_ENHANCED.md (4-head attention), Procedural_Galaxy_Universe_Composition.md (router model)
**Idea:** Initially 4-head attention fusion in PTX. Later evolved to "fused head = router, not fuser."
**Evolution:** This was an initial idea leading to what we have now: main TRM with infinite craftable specialists. The TRM can craft new specialists as needed via LoRA-like adapters. The "fused head" IS the TRM routing logic. The latest layered definition is hyper-modular.
**Action:** Clarify in TRM spec that specialist creation is dynamic and unlimited, not a fixed set of heads.

### 14. Math Symbol Representation -- EVOLVED
**Source:** Procedural_Galaxy_Universe_Composition.md
**Idea:** Math symbols as separate always-loaded galaxy with execution RPN.
**Evolution:** Right direction, wrong framing. Math symbols are form+meaning like ALL the system. The math "grammar" contains the rules and ways of solving/composing solutions -- there are several ways to calculate things. The routing (pi in math context vs pi in text) should be inside TRM weights. This is the part we can afford to not know HOW it learns to navigate, because we can still see WHAT it's doing since the memory itself is external and inspectable.
**Action:** Document math symbols as standard Galaxy entries with form+meaning. Math grammar rules as Grammar Galaxy entries. TRM learns routing internally.

### 15. Swarm-Galaxy Resonance Feedback Loop -- EVOLVED
**Source:** SLEEP_TIME_CONSOLIDATION_DESIGN.md
**Idea:** Spatial neighbors average embeddings to refine quality. Bidirectional Galaxy-embedding loop.
**Evolution:** Yes, but needs system time synchronization. Real time and timezone are important and crucial for networking -- K3D network doors depend on this. The resonance feedback should be time-stamped and sync-aware for multi-instance coordination.
**Action:** Add to Knowledgeverse spec with time-sync requirement. Connect to Doors protocol for network coordination.

### 16. PDF-as-RPN-Bytecode -- ACTIVE
**Source:** MULTIMODEL_CHAIN_PROMPT_PHASE_C.md
**Idea:** PDF operators (Td, Tj) map directly to RPN stack operations. Sub-500us/page in PTX.
**Evolution:** YES. Leverage ternary opcodes where it makes computational sense. PDF operator -> ternary RPN translation could use trit-encoded operator classes.
**Action:** Add to RPN_DOMAIN_OPCODE_REGISTRY as ingestion optimization target with ternary opcode mapping.

### 17. K3D_NODE Foundational Struct -- FOUNDATIONAL + EVOLVED
**Source:** Step7.the_chain.md
**Idea:** embed[768], quality, curiosity, shape, checksum.
**Evolution:** Apply ternary contrastive learning technique to enhance this. Quality and curiosity should be ternary signals. The checksum enables contrastive verification (match = +1, mismatch = -1, partial = 0). The embedding dimension should reflect the "high-dim only inside TRM" principle -- the node stores 3D position + metadata, not a 768D embedding directly.
**Action:** Formalize in Knowledgeverse spec with ternary quality/curiosity fields and 3D-first storage model.

### 18. House Room Quality Thresholds -- ACTIVE
**Source:** Step7.the_chain.md
**Idea:** Library >= 0.7, Garden >= 0.4, Museum < 0.4.
**Current fit:** Nice! Needs revision -- Christoph's ideas from PM-KR group will join with this initial thought. Keep registered as baseline.
**Action:** Add to Knowledgeverse spec as initial House policy. Mark as pending PM-KR alignment with Christoph's contributions.

### 19. Sleep-Time Consolidation Triggers -- EVOLVED
**Source:** Step9.md (sovereign pattern), SLEEP_TIME_CONSOLIDATION_DESIGN.md
**Idea:** Originally: idle detection triggers consolidation. Sovereign execution: CUDA C++ -> PTX -> ctypes -> direct pointers.
**Evolution:** This is part of sleep-time compute, but it can ALSO be triggered by near-full Galaxy memory for consolidation into House, freeing Galaxy space for more workloads. Python should be eliminated -- used only to load the system if needed. The system itself is a live game engine, always on.
**Action:** Formalize in THREE_BRAIN_SYSTEM: (1) Sovereign execution as the ONLY execution model, (2) Sleep-time consolidation triggered by BOTH idle AND memory pressure.

### 20. Sovereign OCR via PTX -- EVOLVED
**Source:** Step7.2, PHASE_F_DEEPSEEK_OCR_KERNELS_MASTER_PLAN.md
**Idea:** ActionBuffer 256-byte GPU struct, warp-cooperative OCR, hierarchical NMS.
**Evolution:** This was an attempt to do OCR natively. With the real PTX kernels being implemented by Codex, we'll be able to make this work properly. Foundational for enabling any user (including disabilities -- accessibility is core).
**Action:** Document ActionBuffer as foundational I/O struct. Mark OCR-via-PTX as enabled by current Codex kernel work. Accessibility is a first-class requirement.

### 21. Modality-to-Platonic-Solid Mapping -- FOUNDATIONAL
**Source:** Step8.md, Step7.the_chain.md
**Idea:** text=tetrahedron, image=cube, audio=octahedron, video=icosahedron, mixed=dodecahedron. Shape determines recursion depth, memory layout, kernel dispatch.
**Current fit:** TRUE, foundational. This is the geometric organizing principle for the multi-modal 3D world.
**Action:** Formalize in PROCEDURAL_VISUAL_SPECIFICATION and THREE_BRAIN_SYSTEM.

### 22. GPU Sleep Consolidation Kernels -- FOUNDATIONAL
**Source:** SLEEP_TIME_CONSOLIDATION_DESIGN.md
**Idea:** Three PTX kernels: cluster tightening, quality validation, glyph dedup. Currently CPU-only via scikit-learn.
**Current fit:** FOUNDATIONAL for sleep-time compute and shadow learning. Remember: we do not use any library, all in-house. These must be sovereign PTX kernels.
**Action:** Add to Codex implementation backlog as high-priority sovereign kernels. No scikit-learn in production.

### 23. Grammar Galaxy Cross-Modal Rule Synthesis -- ACTIVE
**Source:** CODEX_SOVEREIGN_SWARM_ARCHITECTURE_12.12.2025.md
**Idea:** Cross-modal observation triggers rule synthesis via dimension mapping. Bayesian quality for promotion.
**Current fit:** Yes, and THIS is where AGI will emerge from. Autonomous cross-modal pattern recognition creating new Grammar Galaxy rules is the path to general intelligence.
**Action:** Document as core Grammar Galaxy evolution mechanism. This is the AGI emergence pathway.

### 24. TRM Anti-Patterns from Paper -- EVOLVED
**Source:** Step8.md
**Idea:** MoE destroyed generalization, partial backprop didn't help, removing ACT dropped generalization, weight tying too constraining, TorchDEQ slower.
**Evolution:** This is contrastive learning applied -- match the technique using ternary opcodes. Each anti-pattern is a -1 signal. Each success is +1. The TRM learns WHAT NOT TO DO as efficiently as what to do.
**Action:** Add to TRM spec as ternary-encoded anti-patterns. Connect to contrastive learning framework.

---

### Additional from Gap Files

### 11b. Deep-Stack Procedural Reasoning -- EVOLVED
**Source:** Procedural_Galaxy_Universe_Composition.md
**Idea:** 18 dedicated stacks with 6-stage pipeline.
**Evolution:** This idea advanced to deeper stacks and the three-tier math cores + internal swarm design. But the core insight is correct: TRM operates on PROGRAMS, not vectors. These programs can be semantic (grammar-based text composing/understanding) as well as mathematical.
**Action:** Document the "programs not vectors" principle in RPN_DOMAIN_OPCODE_REGISTRY and THREE_BRAIN_SYSTEM.

### 12b. Meaning-Based Star Identity -- FOUNDATIONAL (KEY)
**Source:** Procedural_Galaxy_Universe_Composition.md
**Idea:** Stars grouped by semantic meaning, not visual form.
**Current fit:** THIS IS KEY. The visual form is a layer UNDER this -- at the base. This is the HIGHEST abstraction layer where multi-modal fuses into a single visual representation of knowledge, composed from atomic units from ground up. Form+meaning layering: atoms (Drawing Galaxy) -> composition (Grammar Galaxy) -> meaning (semantic identity at the star level).
**Action:** Document as the fundamental Galaxy identity principle in KNOWLEDGEVERSE_SPECIFICATION. This is the capstone of the form+meaning hierarchy.

---

## Cross-Reference: What Specs Need Updates

| Spec | # Items | Key Additions |
|------|---------|---------------|
| **KNOWLEDGEVERSE_SPECIFICATION** | 10 | Galaxy spill (#2), meaning clustering (#4), codecs (#7), meaning-based identity (#12b), resonance feedback (#15b), K3D_NODE (#17b), House thresholds (#18b), GPU sleep kernels (#22b), grammar evolution (#23b), Math Symbol routing (#14) |
| **TRM_SPECIALIST_MATRYOSHKA** | 7 | Adaptive sparsity (#5), quick+deep+spawnable cores (#8), ternary control tokens (#9), infinite specialist creation (#13), ActionBuffer (#20), anti-patterns (#24), fused head as router (#13b) |
| **THREE_BRAIN_SYSTEM** | 5 | Persistent VRAM regions (#3), programs-not-vectors (#11b), sovereign exec pattern (#19), Platonic solids (#21), sleep triggers (#19b) |
| **RPN_DOMAIN_OPCODE_REGISTRY** | 3 | RPN-as-function (#1), programs-not-vectors (#11b), PDF-as-RPN (#16b) |
| **PROCEDURAL_VISUAL** | 2 | Meaning clustering (#4), Platonic solid mapping (#21) |
| **MATH_CORE** | 2 | RPN-as-function (#1), math form+meaning (#14) |
| **UNIFIED_SIGNAL** | 1 | Ternary modality encoding (#6) |

**Total: 30 items across 7 specs.** Hold for Codex conclusions before executing updates.

---

## Files Analyzed (Complete)

**Round 1 (16 files, ~70K lines):**
Step11_TextTo3DInference_ENHANCED (16,340), Step10_ThinkingTagInference (13,037), K3D_MATH_RPN_SWARM_PROMPT_V2 (9,135), DANIEL_VECTORDOTMAP_PLANS_V1 (6,532), PHASE_F_DEEPSEEK_OCR_KERNELS_MASTER_PLAN (6,347), PROCEDURAL_ATOM_KKRIEGER_EXPLORATION (5,682), KnowledgeVerse_Browser_Partners_Development (3,382), CODEX_DIAGNOSTIC_FRAMEWORK_IMPLEMENTATION (2,351), CLAUDE_HYBRID_TRM_ARCHITECTURE_SPEC (1,926), CLAUDE_LOADING_STAGE_ARCHITECTURE_ENHANCED (1,909), PHASE_5.1_COLLABORATIVE_PLAN_FINAL (1,727)

**Round 2 (8 files, ~39K lines):**
MULTIMODEL_CHAIN_PROMPT_PHASE_C (8,453), Step7.2 - Original (7,831), Step7.the_chain (4,805), Step9 (4,397), CODEX_SOVEREIGN_SWARM_ARCHITECTURE (4,294), Procedural_Galaxy_Universe_Composition (4,161), Step8 (2,896), SLEEP_TIME_CONSOLIDATION_DESIGN (2,424)

**Not analyzed (~30K lines):**
~80+ files under 1,700 lines (handoffs, status reports, focused prompts) -- derivative/summary documents.
