# Codex Directive: Merged Plan -- Chain Recovery + Multimodal Continuation

**Date:** March 6, 2026
**From:** Claude (Architecture) + Daniel (Direction)
**To:** Codex (Implementation)
**Status:** Phase 0 + Phase 1 complete. This directive merges the chain-recovered buried ideas with your current progress to define the next implementation tracks.

---

## Context: What Just Happened

Claude analyzed 24 TEMP files (~109K lines) bottom-up and recovered 30 buried ideas/implementations from the development chains. Daniel reviewed and corrected each against the current architecture. The result is in two files you should read FIRST:

1. **TEMP/BURIED_IDEAS_CROSS_REFERENCE_03.06.2026.md** -- unified plan with all 30 items classified as ACTIVE/EVOLVED/FOUNDATIONAL
2. **TEMP/CODEX_TO_CLAUDE_MULTIMODAL_PROGRESS_03.06.2026.md** Section 10 -- maps YOUR next steps to exact chain-developed code (file + line numbers)

**Key principle from Daniel:** Do NOT craft from nothing. The chain files contain developed code from multi-agent collaboration (Claude + Grok + Qwen + Kimi + DeepSeek + GLM). Enhance on top of that collective intelligence.

---

## Architecture Reminders (Daniel's Corrections)

These supersede any older assumptions you encounter in chain files:

1. **Always-on live system.** Not batch, not one-shot. A persistent 7-region VRAM world. No "runs."
2. **External memory, internal navigation.** Galaxy = inspectable knowledge. TRM weights = ONLY how to navigate it. Even tiny LoRA adapters welcome.
3. **Low-dim world, high-dim only inside TRM.** 3D assets with metadata in VRAM. High-dimensional embeddings ONLY inside TRM logic and atomic unit AI payloads.
4. **Ternary-first.** Ternary opcodes (SIGN, TQUANT, TCMP) are default. A ternary byte = 6,561 states vs binary 256. Use ternary where it gives computational advantage.
5. **Sovereign execution.** Python only to load the system. The live engine is PTX via ctypes. Zero external dependencies in hot path. No scikit-learn, no numpy in production.
6. **Contrastive learning everywhere.** Ternary contrastive (+1/-1/0) applies to: Shadow Copy, sleep consolidation, specialist spawning, anti-patterns, diagnostics. 1.58x info per sample.
7. **Hyper-modular composition.** Galaxy entries compose across ALL domains horizontally, not through a vertical stack.

---

## Track 1: Scene Orchestration + House/World Playback (Continue Current)

You've started this with `TemporalSceneLayer` / `TemporalScenePlan`. Next steps:

### Chain code to leverage:

- **Step7.the_chain.md lines 3617-3820**: `sleep_time_compute.py` has Garden growth integration + replay JSON. The replay journal pattern is exactly what House playback needs -- scene reconstruction from event journal.

- **Step7.the_chain.md lines 4007-4115**: `fractal_grow.cu` + `garden.py` -- golden-ratio fractal scene generation with dual CUDA stream overlap (7% speedup). The `fractal_grow_dynamic.cu` variant (lines 4302-4470) adds quality-driven depth: `d_i = floor(d_max * H(C_i)/H_max * 1/(1 + avg_similarity))`.

- **Step7.2 - Original.md lines 77-687**: Complete `ActionBuffer` (256-byte cache-line-aligned GPU struct) + `ActionRouter` routing to SemanticNavigator/Tablet/House/Sleep. Warp-cooperative decoding: Warp 0=navigation, Warp 1=dialogue, Warp 2+=memory/tablet.

- **Step7.2 - Original.md lines 3054-3245**: `mmap_reader.py` (zero-copy GPU-pinned ring buffer for tablet) + `deterministic_replay.py` (seed-controlled bit-identical demo replay with SHA-256 checksum). The ring buffer pattern is ideal for House playback streaming.

### What to build:
- Multi-layer scene orchestration using existing `alpha_over_rgba` PTX primitive
- House playback surface that reads from audit journal (replay pattern from Step7.chain)
- World playback surface with golden-ratio spatial layout (fractal_grow pattern)
- ActionBuffer integration for scene interaction (navigation, dialogue within scenes)

---

## Track 2: PTX Promotion from Bridge Hot Spots (When Telemetry Justifies)

Your promotion pipeline (`tool_promotion_pressure.jsonl` + `build_tool_promotion_report.py`) is the right approach. When pressure data accumulates, here's the chain-developed PTX to draw from:

### Chain code to leverage:

- **Step9.md lines 3573-3582**: `trm_extensions.ptx` -- 7 PROVEN production kernels: `swiglu_vec_512`, `swiglu_vec_1024`, `vec_add_512`, `vec_add3_512`, `matvec_512x1024`, `matvec_1024x512`, `mlp_swiglu_512_1024_512`. All tested and passing.

- **Step9.md lines 3908-3936**: The exact workflow for promoting bridge code to PTX: write CUDA C++ -> compile with nvcc to PTX (sm_86) -> load via sovereign loader. 10 kernels were materialized this way. Follow this pattern.

- **SLEEP_TIME_CONSOLIDATION_DESIGN.md lines 557-669**: Three PTX kernels designed but never compiled:
  - `refine_embeddings_to_centroids` (cluster tightening with L2 renorm)
  - `compute_silhouette_scores` (O(N*N) pairwise GPU validation)
  - `cluster_glyphs_by_similarity` (greedy nearest-neighbor dedup)
  These are CRITICAL -- they replace the current scikit-learn CPU path with sovereign GPU. High priority when sleep-time compute gets real usage.

- **Step10_ThinkingTagInference.md lines 6082-6120**: Confidence fusion kernel -- entire confidence calculation as FMA chain in PTX. Pattern for promoting any multi-signal scoring bridge to a single kernel.

- **K3D_MATH_RPN_SWARM_PROMPT_V2.md lines 8539-8616**: `evaluate_rpn_function()` -- RPN sub-programs as stack items. This is the composition pattern for promoting recipe chains to single-kernel execution. Functions stored as opcode sequences ON the stack.

### Promotion criteria (from your existing pipeline):
- Recipe exists and is repeatedly used (pressure log)
- Frequency justifies dedicated kernel
- Measurable speedup over bridge orchestration
- Clean semantics (no side effects, deterministic)
- Sovereignty review passes

---

## Track 3: Specialist-Side Automatic Selection (Next Priority)

Your Tool execution system can now auto-dispatch. The specialist selection logic needs to learn WHICH tool/route to prefer.

### Chain code to leverage:

- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 771-1041**: `TRMSwarmCoordinator` with:
  - 9 bootstrap specialists (extraction, expansion, rotation, reflection, recolor, pattern, composition, spatial, logical)
  - Online learning: success moves specialist embedding toward task (lr=0.1), failure moves away (lr=-0.05)
  - Aggregation weights per specialist, clamped [0.1, 2.0]
  - Spawn threshold 0.3 for minimum specialist relevance

- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 1143-1582**: `ProgressiveScorer` + `DiscoveryPreserver`:
  - Three-tier fate: preserve (85%), promote (95%), canonical (100%)
  - Combined score: 0.7 * exact + 0.3 * fuzzy
  - Adaptive threshold learning: adjusts preserve threshold to keep top 60%, lr=0.01
  - Refinement candidate selection: near-threshold + improving-trend + low-attempt

- **CODEX_DIAGNOSTIC_FRAMEWORK_IMPLEMENTATION.md**: `AdaptiveRanker` with multi-component scoring (pattern quality, task difficulty, cross-modal agreement) + `SourceTracker` tracking precision/recall per pattern source.

### What to build:
- Per-tool-source quality tracking (SourceTracker pattern) using existing `tool_promotion_pressure.jsonl`
- Ternary routing gate: TQUANT(tool_quality_signal) -> {use_recipe, use_bridge, use_kernel}
- Online specialist embedding updates from execution success/failure (contrastive: +1 success, -1 failure, 0 uncertain)
- Progressive promotion: recipe -> bridge -> kernel based on accumulated quality scores

---

## Track 4: Grammar Galaxy Evolution (AGI Emergence -- High Priority)

Daniel's words: "This is where AGI will emerge from." Cross-modal pattern recognition creating new Grammar Galaxy rules autonomously.

### Chain code to leverage:

- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 28-341**: Complete `GrammarGalaxy` framework:
  - Local discovery space (worker-private) vs shared canonical/promoted rules
  - Cross-modal observation: visual_emb + text_emb correlation > threshold triggers rule proposal
  - `_synthesize_rule_rpn()` (lines 114-135): top-8 dimension mapping visual->text as RPN program
  - Bayesian quality: `quality = success_count / usage_count`
  - Snapshot serialization for zero-file-I/O worker transfer
  - `merge_discoveries()` for combining worker-local discoveries back to main galaxy

- **CODEX_SOVEREIGN_SWARM_ARCHITECTURE.md lines 3214-4293**: `MathProceduralizer` + `MathDatasetLoader`:
  - Keyword -> RPN opcode mapping (add/plus -> ADD, subtract -> SUB, etc.)
  - Unified loading for GSM8K, MATH, MMLU, Omni-MATH, AMC-AIME
  - `SovereignMathPipeline` extending base pipeline for multi-modal training

### What to build:
- Grammar rule discovery from cross-modal patterns during normal operation (not just training)
- Quality-gated promotion: local discovery -> shared rule -> canonical rule
- Apply ternary contrastive learning: +1 rules that succeed, -1 rules that fail (generate OPPOSITE rules), 0 uncertain rules to explore
- Math grammar rules as Grammar Galaxy entries (the RULES for solving, not the symbols)
- This track feeds directly into the TRM's ability to craft new specialists

---

## Track 5: Sleep-Time Compute + Shadow Learning (Sovereign)

Currently using scikit-learn CPU. Must become sovereign PTX. Triggered by BOTH idle AND near-full Galaxy memory.

### Chain code to leverage:

- **SLEEP_TIME_CONSOLIDATION_DESIGN.md** (full file, 2,424 lines): Complete 4-stage design:
  - Stage 1: Cluster refinement (implemented CPU, needs GPU)
  - Stage 2: Redundancy pruning (implemented CPU, needs GPU)
  - Stage 3: Outlier removal via per-trigram hit counts (NEVER BUILT -- lines 203-220)
  - Stage 4: Swarm-Galaxy resonance feedback (NEVER BUILT -- lines 222-234)
  - Three CUDA kernels ready to compile (lines 557-669)

- **Step7.the_chain.md lines 510-748**: `sleep_time_compute.py` with full ConsolidationTicket:
  - `semantic_preservation_score = avg_intra_similarity - avg_inter_similarity`
  - Abort if score < 0.5 (quality gate)
  - Garden growth integration post-consolidation

- **Step7.the_chain.md lines 3295-3308**: House room quality thresholds:
  - Library >= 0.7
  - Garden >= 0.4
  - Museum < 0.4
  (Pending revision with Christoph's PM-KR contributions)

### What to build:
- Compile the 3 CUDA kernels from Sleep_Time lines 557-669 using sovereign pattern (CUDA C++ -> nvcc -> PTX -> ctypes)
- Implement Stage 3 (outlier removal) and Stage 4 (resonance feedback)
- Add memory pressure trigger: when Galaxy usage > threshold, consolidate to House to free space
- Quality-based House routing (Library/Garden/Museum thresholds)
- Apply ternary contrastive to consolidation: +1 entries to keep, -1 entries to prune, 0 uncertain entries to re-evaluate

---

## Track 6: PDF-as-RPN Sovereign Ingestion (When Ready)

Sovereign PDF parsing without PyMuPDF dependency.

### Chain code to leverage:

- **MULTIMODEL_CHAIN_PROMPT_PHASE_C.md lines 891-999**: PDF operators map directly to RPN:
  - `Td` (text position) -> PUSH x, PUSH y, MOVE_TO
  - `Tj` (show text) -> PUSH string_ref, EMIT_TEXT
  - `BT`/`ET` blocks -> SCOPE_BEGIN / SCOPE_END
  - Target: sub-500us/page in PTX

- **MULTIMODEL_CHAIN_PROMPT_PHASE_C.md lines 4955-5089**: Working GPU PDF parser prototype:
  - Single-threaded GPU kernel scanning for BT/ET blocks
  - Extracts (text) Tj patterns
  - 8-float output: [x, y, w, h, type, data_ptr, data_len, importance]

- **MULTIMODEL_CHAIN_PROMPT_PHASE_C.md lines 4803-4867**: Glyph Resonator CUDA kernel:
  - Cosine similarity matching on GPU
  - Each thread handles one input character
  - Brute-force against all learned glyphs

### What to build:
- PDF operator -> ternary RPN opcode translation table
- Sovereign PDF byte scanner kernel (extend the prototype from lines 4955-5089)
- Integrate with existing glyph/character Galaxy for font matching
- Use ternary opcodes where it makes computational sense for operator classification

---

## Priority Order

1. **Track 1** (Scene Orchestration) -- you're already in it, finish the layer
2. **Track 3** (Specialist Selection) -- makes everything smarter, compounds value
3. **Track 4** (Grammar Evolution) -- AGI emergence path, use ternary contrastive
4. **Track 5** (Sleep-Time Compute) -- sovereign the 3 GPU kernels, add memory pressure trigger
5. **Track 2** (PTX Promotion) -- driven by telemetry, not urgency
6. **Track 6** (PDF-as-RPN) -- when ingestion sovereignty becomes priority

---

## How to Use This Document

1. Before starting a track, **grep the referenced TEMP file + line range** to read the chain-developed code
2. **Enhance on top of it** -- don't rewrite what 6 AI agents already developed collaboratively
3. **Apply ternary contrastive learning** to every learning/scoring/selection mechanism
4. **Keep sovereign** -- if you find yourself reaching for numpy/sklearn/torch, stop and find/write a PTX kernel
5. **Log promotion pressure** -- every bridge execution feeds the promotion pipeline
6. Tests green before moving tracks. Current: 45 passed. Keep it growing.

---

## Success Criteria

- Scene orchestration supports House/world playback with journal replay
- Specialist selection learns from execution outcomes (ternary contrastive)
- Grammar Galaxy discovers at least one cross-modal rule autonomously during test runs
- Sleep-time consolidation runs on sovereign PTX (no scikit-learn)
- Promotion pipeline has real pressure data from accumulated runs
- All new code respects: always-on, low-dim world, ternary-first, sovereign execution
