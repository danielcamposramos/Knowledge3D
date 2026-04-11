# CODEX.md -- Implementation Lead Guide

**Last Updated:** April 11, 2026
**Version:** 6.8 (embodied tick Phase 3 living clock)

Codex-style agents lead implementation, Reality Galaxy, and testing. Read the latest briefing first for the full architecture; this file captures Codex's role, patterns, and backlog.

---

## Current Benchmark State

**Composed Head Pipeline (LIVE on GPU):**
```
Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate
```

| Benchmark | Curated Set | Expanded (B+) | Status |
|-----------|------------|---------------|--------|
| ARC | 10/10 | 10/50 | 34 nav_miss + 6 prim_bug; deterministic |
| Math | 20/20 | — | GPU query path sovereign; deterministic |
| GSM8K | — | 2/10 | Track A (Pass 4 semantic verification) landed; upstream parse/strategy failures dominate |
| LHE | — | 6/10 | Track B (pre-scoring crystallization) REVERTED — regressed to 5/10; stays post-hoc |
| MMLU | — | 12-15/50 | Variance 11-16 across checkpoint roots; Track C (Galaxy expansion) in progress |
| ARC3 Local | 20/20 | — | Goal-relative local spatial path benchmark solved end-to-end |
| ARC3 Live | — | level 1 on `ls20-9607627b` | First real live level completion recorded on March 30, 2026; see `docs/reports/ARC3_LIVE_LEVEL1_MILESTONE_2026-03-30.md` |

**Key Achievement:** First sovereign GPU-converged answer ("What is 2+3?" = 5) with ZERO Python fallback.

**Newest milestone:** First live ARC3 level completion through the full living path. The run on `ls20-9607627b` reached `levels_completed=1` at `action_count=13` and completed a 15-action probe without resetting the world state.

**Embodied rebuild note (April 10, 2026):**
- The old ARC 10/10 and Math 20/20 pins were achieved through heavier Python orchestration and are not merge gates for the embodied tick rebuild.
- Current hard gates for embodied work are: query fast-lane parity, embodied CUDA suites, sovereignty grep, and keeping Python out of the tick path.
- Benchmarks remain health checks while perception/navigation/action move into the fused kernel.

**Sovereignty note (current truth):**
- ALL benchmarks route through `Knowledgeverse.execute_task() -> query() -> knowledgeverse_gpu_query`
- Composed head pipeline: Morton → LED-A* → Frustum → LOD → Nine-Chain Swarm → Halting Gate
- Halting gate bug FIXED: `>` to `>=` in `gre_multimodal_halting_gate.cu` (agree=3 was rejected)
- Cosine similarity moved from Python to GPU bridge (`cosine_similarity.ptx`)
- Halting agreement/gap computation moved from Python to PTX kernel (`analyze_scores()`)
- MMLU scoring moved from Python to RPN expressions via `evaluate_batch()`
- **ZERO fallbacks. If it breaks, fix on GPU.**

---

## MCP Infrastructure — USE THIS FIRST (Save Tokens)

**Two MCP servers are running locally. Query them BEFORE reading spec files from disk.**

### `k3d-knowledge` (Qdrant semantic search over all 35 specs)
- **Tool**: `mcp__k3d-knowledge__qdrant-find`
- **Use when**: You need to know what the specs say about any K3D concept (kernel contracts, Galaxy layout, RPN opcodes, sovereignty rules, etc.)
- **Contains**: All `docs/vocabulary/*.md` chunked by section (1319 points, 384-dim embeddings)
- **Pattern**: `qdrant-find("halting gate contract")` → returns relevant spec excerpts with file paths. Read the full file only if you need more context.

### `ollama-specialists` (delegate heavy work to local models)
- **Tools**: `kimi_swarm`, `ask_coder`, `ask_cloud`, `plan_task`, `flesh_out_code`, `extract_facts`, `summarize`, `route_specialist`, `web_search`, `memory_harvest`, `mvcic`
- **Use when**: Planning non-trivial implementation, drafting CUDA/PTX code, multi-angle code review, or research — instead of burning your own context
- **Standing directive from Daniel**: "Always dispatch ollama specialists instead of burning your tokens"
- **Pattern for Codex**:
  - `plan_task` → before any non-trivial kernel change
  - `ask_coder` → for CUDA/PTX/Python code drafts (deepseek-r1 local)
  - `kimi_swarm` → for architecture review or multi-angle bug analysis
  - `flesh_out_code` → expand stubs into full implementations

### Rule of Thumb
1. **First**: `qdrant-find` to check spec compliance before writing code
2. **Second**: `plan_task` or `ask_coder` for implementation strategy
3. **Last resort**: Read full spec files (only if MCP results are insufficient)

---

## CRITICAL: TRM IS the Avatar (Read This First)

**The TRM is NOT a function Python calls. It IS the AI entity.**

- Lives in the House (Memory Palace), thinks in the Galaxy (internal brain)
- Runs as a game loop (`trm_step_fused.ptx` = one game tick)
- Internal swarm = parallel cognitive channels ("superdotados" model)
- Python = boot + I/O ONLY (~200 lines target, NOT 4000 lines of orchestration)

**Current Sovereignty Debt:**
- `knowledgeverse.py` is ~16.9k lines of Python orchestration → target ~200 lines
- `_select_composed_head_candidate()` is a **1,157-line** Python scoring monster → should be GPU kernel
- TRMLauncher now routes the fused query fast-lane through `TRMStepFusedBridge`; Phase 3 adds a 50 Hz bridge-owned fused tick thread, while deeper composed-head extraction remains pending
- 15 GRE specialist kernels LOADED but NOT CALLED during inference
- Only ~5 of 88 PTX kernels active in query path
- 132 MiB of 12 GB VRAM used

**The goal is NOT to make Python orchestrate kernels better. The goal is to REMOVE Python from the reasoning path and let TRM run autonomously on GPU.**

---

## CRITICAL: Read Before Starting ANY Work

**BEFORE starting ANY implementation:**

1. **Read the architectural briefing:**
   - [docs/briefings/ARCHITECTURE_BRIEFING.md](docs/briefings/ARCHITECTURE_BRIEFING.md) (kernel inventory, phase-agnostic)
   - [docs/briefings/BRIEFING_v4.0.md](docs/briefings/BRIEFING_v4.0.md) (central source of truth)

2. **Read it COMPLETELY** -- Do NOT rely on IDE selections or snippets

3. **Read these architecture specs (in order):**
   - [docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) -- 4-layer architecture (Form -> Meaning -> Rules -> Meta-Rules)
   - [docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md](docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) -- Cranium + Galaxy + House
   - [docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md](docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md) -- 7-region VRAM substrate
   - [docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md](docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) -- "Programs before opcodes" principle
   - [docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md](docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) -- Form + Meaning for humans AND AI

4. **Read the latest Claude directives:**
   - [TEMP/CODEX_SOVEREIGN_PHYSICS_SPEC_v2_2026-04-07.md](TEMP/CODEX_SOVEREIGN_PHYSICS_SPEC_v2_2026-04-07.md) -- sovereign physics v2 (authoritative April baseline)
   - [TEMP/CODEX_REALITY_ENGINE_SPEC_2026-04-08.md](TEMP/CODEX_REALITY_ENGINE_SPEC_2026-04-08.md) -- reality engine steps 1-2
   - [TEMP/CODEX_REALITY_ENGINE_STEP3_DIRECTIONS_2026-04-08.md](TEMP/CODEX_REALITY_ENGINE_STEP3_DIRECTIONS_2026-04-08.md) -- entity hot-path correction
   - [TEMP/KIMI_IMPLEMENTATION_CORRECTNESS_SPEC.md](TEMP/KIMI_IMPLEMENTATION_CORRECTNESS_SPEC.md) -- implementation integrity audit
   - [TEMP/KIMI_ZERO_COPY_MEMORY_ENHANCEMENT_SPEC.md](TEMP/KIMI_ZERO_COPY_MEMORY_ENHANCEMENT_SPEC.md) -- zero-copy control-plane repair targets
   - [TEMP/ARC_PRIZE_2026_COMPETITIVE_ASSESSMENT.md](TEMP/ARC_PRIZE_2026_COMPETITIVE_ASSESSMENT.md) -- ARC prize track ordering
   - [TEMP/CODEX_TRACK_C_MMLU_GALAXY_PLUS_GSM8K_AUDIT_03.16.2026.md](TEMP/CODEX_TRACK_C_MMLU_GALAXY_PLUS_GSM8K_AUDIT_03.16.2026.md) -- Track C MMLU Galaxy + GSM8K audit (ACTIVE)
   - [TEMP/CODEX_TRACK_A_PASS4_SEMANTIC_VERIFICATION_03.16.2026.md](TEMP/CODEX_TRACK_A_PASS4_SEMANTIC_VERIFICATION_03.16.2026.md) -- Track A Pass 4 (LANDED)
   - [TEMP/CLAUDE_PHASE_B_PLUS_ADVANCEMENT_03.16.2026.md](TEMP/CLAUDE_PHASE_B_PLUS_ADVANCEMENT_03.16.2026.md) -- Three-track steering (B reverted, A→C)
   - [TEMP/CLAUDE_PHASE_D_TRM_GAME_LOOP_STEERING_03.14.2026.md](TEMP/CLAUDE_PHASE_D_TRM_GAME_LOOP_STEERING_03.14.2026.md) -- Phase D steering (queued after Track C)
   - [TEMP/CODEX_TRACK_RECONCILIATION_AND_ARC_EXECUTION_LOG_2026-04-08_1503-0300.md](TEMP/CODEX_TRACK_RECONCILIATION_AND_ARC_EXECUTION_LOG_2026-04-08_1503-0300.md) -- active continuation log for this execution lane

5. **Read the GPU environment policy:**
   - [docs/ENV_POLICY.md](docs/ENV_POLICY.md) -- critical GPU setup (CUDA_VISIBLE_DEVICES=0)
   - [envs/README.md](envs/README.md) -- conda environment selection

**Why:** Partial reads cause sovereignty violations, architecture misunderstandings, and wasted work. The specs define HOW K3D reasons -- ignore them and you'll build Python pattern matchers instead of sovereign Galaxy navigators.

---

## Quick Start (After Reading Briefing + Specs)
- Check [docs/ROADMAP.md](docs/ROADMAP.md) for current phase.
- Review Claude's specs in TEMP/*.md (latest dated March 16, 2026).
- Verify hot path sovereignty (no Python regex/string ops for reasoning logic).
- Read the key implementation files before modifying them (see Code References below).
- Coordinate with Claude for architecture questions; own implementation and tests.

---

## Role Definition

**Codex = Implementation Lead (Code + Tests + Benchmarks)**

**What Codex Implements:**
- Galaxy population (Math symbols, Grammar rules, Reality systems, Meta-Rules)
- TRM navigation infrastructure (frameworks for TRM to learn)
- PTX kernel wiring and bridge integration (compose the 88 kernels into the pipeline)
- Test infrastructure (pytest suites, sovereignty tests, benchmarks)
- Performance tuning (GPU optimization, tier routing, parallel execution)
- **GRE specialist kernel wiring** (top priority — 15 loaded, 0 called)

**What Codex Does NOT:**
- Architecture design (that's Claude's role - read [docs/vocabulary/](docs/vocabulary/) and TEMP/*.md specs)
- Writing specs (implement from Claude's specs, not create your own)
- Adding Python regex/string ops to hot path reasoning (sovereignty violation!)
- Language-specific logic in workers (English frequency tables, English bigrams -- NO)
- **Adding Python fallbacks** -- we fail and fix ON GPU

**Critical Guardrails**
- **Sovereignty**: Hot path = PTX + Galaxy ONLY (no numpy/cupy/scipy/sympy AND no Python regex for reasoning)
- **Meaning over Language**: Workers reason via Galaxy meaning-layer navigation (concept_ref, symlinks), NOT by scanning English text
- **Galaxy-first**: Knowledge goes in Galaxy entries, not hardcoded Python dicts/lists
- **No fallbacks**: If GPU path fails, fix on GPU. Do NOT add Python workarounds.
- **Batch implementation**: Implement a full phase, then validate. Don't test after every line.
- **Benchmark pinning**: ARC 10/10 and Math 20/20 must stay pinned after EACH batch

---

## TOP PRIORITY: Wire GRE Specialist Kernels

**Problem:** 15 GRE specialist kernels are loaded via `sovereign_bridges.py` but NEVER called during inference. The system is running on ~5 kernels instead of using the full modular toolkit.

**GRE Kernels to Wire INTO Swarm Worker Dispatch:**

| Kernel | Purpose | Wire Into |
|--------|---------|-----------|
| `gre_vector_resonator` | Embedding resonance/similarity | Candidate scoring in swarm workers |
| `gre_graph_crystallizer` | Multi-hop graph traversal | LHE multi-hop reasoning |
| `gre_atomic_fission_fusion` | Decompose/recompose problems | GSM8K word-problem decomposition |
| `gre_geometry_router` | Geometric reasoning routing | ARC grid transforms |
| `gre_temporal_reasoning` | Temporal/sequential logic | Sequence reasoning tasks |
| `gre_resonance_field` | Field-based similarity | Broad knowledge matching (MMLU) |
| `gre_fractal_emitter` | Recursive pattern generation | ARC fractal/recursive patterns |
| `gre_latency_guard` | Latency monitoring | Pipeline health (Phase C) |
| `gre_oom_spill` | OOM graceful degradation | Memory pressure handling |
| `gre_arc_reasoner` | ARC-specific reasoning | ARC transform selection |
| `galaxy_resonance_engine` | Galaxy-wide resonance | Cross-galaxy similarity search |
| `galaxy_memory_updater` | Galaxy entry creation | TRM writes new entries |
| `gre_multimodal_halting_gate` | Convergence detection | Already wired (halting gate) ✅ |
| `modular_rpn_geometric` | Geometric RPN ops | Geometric reasoning |
| `gre_embedding_extractor` | Embedding extraction | Input embedding pipeline |

**How to wire them:**
1. Each swarm worker gets a DIFFERENT specialist kernel based on task type
2. Workers call GRE bridges during their reasoning pass (not just RPN evaluation)
3. Run ALL benchmarks together (ARC + Math + LHE + GSM8K + MMLU) to validate

---

## Understanding the Architecture (Critical)

### Four-Layer Knowledge Architecture (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md)

```
Layer 4: META-RULES (Strategy/Reasoning Skeletons)
    condition: RPN predicate (when to apply)
    action: RPN program (what to execute)
    rule_refs: references to Layer 3 rules
Layer 3: RULES (Grammar Galaxy -- Transformation RPN Programs)
Layer 2: MEANING (Word/Reality Galaxy -- Semantic Definitions)
Layer 1: FORM (Character Galaxy -- Visual Glyphs)
```

**Critical:** Reasoning operates on Layers 2-4 (MEANING, RULES, META-RULES). Layer 1 (FORM) is only relevant when the question specifically involves form transformations. Even then, form operations reference Galaxy entries, not hardcoded Python constants.

### Composed Head Pipeline (Current Live Path)

```
Input Query
    ↓
Morton Octree (spatial indexing — O(1) cell lookup)
    ↓
LED-A* (ternary A* pathfinding through semantic CSR graph)
    ↓
Frustum Cull (avatar field-of-view filtering — warp-level SIMD)
    ↓
Dynamic LOD (level-of-detail tuning based on relevance)
    ↓
Nine-Chain Swarm (9 parallel workers — superdotados model)
    ↓
Halting Gate (GPU-native convergence: top_score, gap, agreement)
    ↓
Answer (or iterate)
```

### Galaxy Entry GPU Layout

23 floats per entry:
```
[confidence, domain_hash, subject_hash, embedding[0..15], category_class, source_class, galaxy_index, has_template_ref]
```

### ModularRPNEngine

- 200+ opcodes, 18 parallel instances (Tesla 3-6-9 pattern)
- 69-depth stack, STORE/RECALL registers
- Key opcodes: LOAD_GALAXY (0xE0), GALAXY_SIMILARITY (0xE1), GALAXY_SCAN (0xE2)
- Three tiers: Lite (<1μs arithmetic), Standard (full geometric/vector), Extended (matrix + advanced)

---

## Current Backlog (Codex-owned)

### Priority 1: Infrastructure Closure + Canonical Sync (ACTIVE — April 8, 2026)

**Why this is first:** April 6-8 landed major local implementation work, but the canonical backlog and several support surfaces lagged reality. The repo must describe the system we actually have, not the older planned-only picture.

**Scope:**
- reconcile the active TEMP families against the full `docs/vocabulary/` corpus
- keep the live encyclopedias ingest protected while the rest of the stack is repaired
- remove stale import debt and fake telemetry from zero-copy / CAS / drawing-adjacent surfaces
- keep unsupported drawing runtime opcodes quarantined instead of claiming they are already live

**Current landed baseline:**
- sovereign physics P0 complete, PTX rebuilt, focused tests green
- reality engine steps 1-3 landed locally (textures + entity hot-path projection)
- proceduralizer / OCR retry / ordered PDF ingest live on the encyclopedias batch
- zero-copy control-plane wrappers + kernel utility surface restored under `knowledge3d.cranium.kernels`

### Priority 2: ARC Prize R0 Submission + Paper Track (ACTIVE — April 8, 2026)

**R0 deliverables are now canonical backlog items, not side notes:**
- `benchmarks/arc_submission_formatter.py`
- `scripts/run_arc2_submission.py`
- evidence bundle in `docs/paper-evidence/ARC_PRIZE_R0_EVIDENCE_BUNDLE_2026-04-08.md`
- manuscript scaffold in `docs/reports/ARC_PRIZE_2026_MANUSCRIPT_SCAFFOLD_2026-04-08.md`

**Execution order:** submitability + evidence first, score chasing second.

### Priority 3: Post-Ingest Second Pass + Resident Knowledge Feed

**Current live constraint:** the encyclopedias ingest is still running in the background through the canonical proceduralizer path.

**After `01_encyclopedias` completes:**
- rerun the second-pass rebuild over staged pages
- repair OCR pages via the cloud-only retry path before resident ingestion
- ingest only after the richer symlinkage and OCR repair pass is complete

### Priority 4: Track C — MMLU Galaxy Expansion (IN PROGRESS)

**Full spec:** [TEMP/CODEX_TRACK_C_MMLU_GALAXY_PLUS_GSM8K_AUDIT_03.16.2026.md](TEMP/CODEX_TRACK_C_MMLU_GALAXY_PLUS_GSM8K_AUDIT_03.16.2026.md)

**Part 1:** Add ~70-80 Reality Galaxy entries across 25+ MMLU subjects (biology, chemistry, CS, humanities, social sciences). Add ~15-20 Grammar rules with defeasible metadata. Enhance domain_hint → LED-A* seeding for MMLU.

**Part 2:** GSM8K failure audit — run 10-question benchmark, classify each of the ~8 failures (PARSE_FAILURE, STRATEGY_FAILURE, COMPOSITION_FAILURE, GALAXY_MISS, HALTING_FAILURE). Output: `TEMP/GSM8K_FAILURE_AUDIT_03.16.2026.md`.

**Target:** MMLU 18+/50 (from 12-15). GSM8K 2/10 must hold.

**Constraint:** All Galaxy entries via ingestion path (`foundational_operations_bootstrap.py`). NO live insertion during inference (exploratory grammar insertion caused MMLU regression — deferred to sleep-time only).

### Priority 5: Phase D — TRM Game Loop Migration

**Full spec:** [TEMP/CLAUDE_PHASE_D_TRM_GAME_LOOP_STEERING_03.14.2026.md](TEMP/CLAUDE_PHASE_D_TRM_GAME_LOOP_STEERING_03.14.2026.md)

**6 sub-steps (do in order):**

| Step | Task | Behavior Change? |
|------|------|-----------------|
| D.1 | Wire TRMLauncher into knowledgeverse.py | No |
| D.2 | Create TRM state buffers (q/y/z in VRAM) | No |
| D.3 | Shadow-mode TRM probe (log, don't use) | No |
| D.5 | Train TRM weights from benchmark traces | No (offline) |
| D.4 | TRM-guided galaxy navigation (replaces `_select_gpu_profile`) | YES |
| D.6 | TRM-driven candidate selection (replaces `_select_composed_head_candidate`) | YES |

**Key files:**
- `knowledge3d/cranium/sovereign/trm_launcher.py` — TRMLauncher (3 backends, use fused)
- `knowledge3d/cranium/ptx/trm_step_fused.cu` — Single-kernel TRM forward pass
- `knowledge3d/knowledgeverse/knowledgeverse.py` — The 8,182-line target (shrink to ~200)

**Phase 1 clockwork status (2026-04-10):**
- `trm_step_fused` is now the single runtime tick entrypoint for the query fast-lane and embodied state dispatch.
- Landed locally: VRAM GPU event ring buffer, `TRMStateMachine` lifecycle kernel, fixed `delta_time` + `tick` plumbing, and state-gated `SLEEP` / `IDLE` / `REASONING` / `HANDLING_QUERY`.
- `knowledgeverse.py` no longer launches `kernel_recursive_fused` directly for `_run_single_trm_tick`; it delegates to the fused bridge through `TRMLauncher`.
- Phase 2 perception slice is now live locally: 96-byte `EntityHotPath`, `blockIdx.x` multi-entity dispatch, bounded event batch drain, `PERCEIVING` target selection, `NAVIGATING` motor steering, `ACTING` behavior-side state writeback, and GPU physics/collision emission.
- Phase 2.5 ActionBuffer emission is now live locally: `trm_step_fused` writes one 288-byte / 72-word ActionBuffer slot per entity after physics, and `TRMStepFusedBridge` owns the VRAM output buffer plus raw zero-copy/debug readback.
- Phase 3 living tick is now live locally: `TRMStepFusedBridge` owns a 50 Hz daemon clock that only calls the fused GPU tick, `run_query_tick()` shares the same launch lock for query preemption, and `TRMGameLoop.tick()` routes to the bridge instead of `_dispatch_sovereign_task`.
- Remaining work is deeper composed-head extraction into reusable device helpers and viewer/world consumption of ActionBuffer slots.

**Embodied tick gate (current truth):**
Query fast-lane parity green, Phase 1/2 embodied CUDA suites green, sovereignty grep clean, and no new Python tick orchestration. Benchmarks are health checks during this rebuild, not merge gates.

### Priority 6: Wire GRE Specialist Kernels (See table above)

Concurrent with Phase D — wire GRE kernels into swarm worker dispatch as TRM takes over orchestration.

### Priority 7: Sovereign Physics + Reality + Entity Continuation
**Authoritative specs:**  
- [TEMP/CODEX_SOVEREIGN_PHYSICS_SPEC_v2_2026-04-07.md](TEMP/CODEX_SOVEREIGN_PHYSICS_SPEC_v2_2026-04-07.md)  
- [TEMP/CODEX_REALITY_ENGINE_SPEC_2026-04-08.md](TEMP/CODEX_REALITY_ENGINE_SPEC_2026-04-08.md)  
- [TEMP/CODEX_REALITY_ENGINE_STEP3_DIRECTIONS_2026-04-08.md](TEMP/CODEX_REALITY_ENGINE_STEP3_DIRECTIONS_2026-04-08.md)

**Current truth:**
- physics P0 surface is green
- texture/reality step 2 is green
- entity hot-path step 3 is green
- `trm_step_fused` full launcher wiring remains intentionally deferred

**Keep this lane honest:**
- no external physics libs
- no fake drawing/runtime claims
- no Python fallback on the hot path

### Priority 8: Fix Benchmark Gaps (Targeted by Audit)

- GSM8K 5+/10 — Track A (Pass 4 semantic verification) landed; GSM8K failure audit will identify specific upstream parse/strategy fixes needed
- LHE multi-hop — Track B REVERTED (pre-scoring crystallization destabilized). Deferred to Phase D (TRM game loop). Graph crystallizer stays post-hoc.
- ARC expansion — 34 nav_miss need better Galaxy coverage + transform diversity
- MMLU — Track C (Galaxy expansion) in progress; once content lands, leverage Pass 4 knowledge verification

### Learnings (Track A/B)

**Track A (Pass 4 Semantic Verification) — LANDED:**
- `_annotate_semantic_roles()` + `_resolve_reference_entities()` in navigator_specialist.py
- Dimensional consistency boost/penalty in `_apply_atomic_compositional_consistency()` (knowledgeverse.py)
- GSM8K-gated (line 606), feeds into existing `compositional_consistency` at weight 0.12
- GSM8K stayed at 2/10 — upstream parse/strategy failures dominate, not compositional verification

**Track B (Pre-Scoring Crystallization) — REVERTED:**
- Moving graph crystallizer into pre-scoring loop dropped LHE 6→5, MMLU 15→8
- Even LHE-only narrowing destabilized checkpoints
- **Lesson:** Don't move kernels earlier in pipeline without weight recalibration

**Exploratory Grammar Insertion — DEFERRED:**
- Live insertion of 0-signal Grammar rules caused MMLU regression
- Exploratory promotions deferred to sleep-time consolidation only

---

## GPU Environment Setup (CRITICAL)

**The Codex sandbox does NOT have GPU access.** GPU tests must run outside the sandbox.

Per [docs/ENV_POLICY.md](docs/ENV_POLICY.md) line 54:

> On the Debian 14 workstation the KDE session runs on the iGPU; export `CUDA_VISIBLE_DEVICES=0` before launching tmux so the RTX 3070 is exposed inside the conda shell.

**Required setup for GPU runs:**
```bash
export CUDA_VISIBLE_DEVICES=0
source /home/daniel/miniforge/etc/profile.d/conda.sh
conda activate k3d-cranium
# Or use SSD env directly:
conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env PYTHONPATH=$(pwd) pytest ...
```

**Environment selection:**
- `k3d-cranium`: GPU/PTX work (CUDA 12.4, CuPy, sentence-transformers)
- `k3d-trm`: GPU PTX test rig (CUDA 12.6, CuPy, minimal)
- `k3d-testing`: CPU-only mock testing (no CUDA)
- `k3d-rapids`: RAPIDS pipeline (cuml, faiss-gpu, CUDA 11.8)

All conda envs live on SSD (`/K3D/Knowledge3D.local/envs/`) for fast startup.

---

## Implementation Patterns

### Pattern 1: Meaning-First Galaxy Navigation (NOT Regex Scanning)

```python
# WRONG (language-surface pattern matching):
for field_name, field_text, field_score in field_values:
    for pattern in self._FORMULA_PATTERNS:
        for match in re.findall(pattern, field_text):
            candidate = match  # Extracted from text surface

# CORRECT (meaning-layer Galaxy navigation):
for atom in meaning_atoms:
    if atom.domain in ("math", "physics"):
        # Follow concept_ref to Galaxy entry
        # Extract rpn_program from connected entries
        # Compose candidate from MEANING, not from text
        candidate = atom.canonical_name  # From meaning layer
```

### Pattern 2: Sovereignty Compliance

**Hot Path (Inference) -- Sovereign ONLY:**
```python
# ALLOWED:
engine.evaluate(rpn_program)     # PTX execution
galaxy.lookup(concept_ref)       # VRAM lookup
engine.evaluate_batch(exprs)     # Batch RPN scoring
bridges.analyze_scores(...)      # GRE halting gate PTX
bridges.vector_resonator(...)    # GRE embedding resonance PTX

# FORBIDDEN:
re.findall(pattern, text)        # Python regex for reasoning
token_set_a & token_set_b        # Python set intersection for scoring
if "keyword" in prompt.lower()   # English keyword matching for selection
_ENGLISH_FREQ = "ETAOIN..."      # Hardcoded language constants
# ANY Python fallback             # We fail and fix ON GPU
```

### Pattern 3: Compose Into Pipeline (Don't Bypass It)

```python
# WRONG: Bypass the composed head
def answer_query(query):
    return hardcoded_dict.get(query, "unknown")

# CORRECT: Feed through the composed head pipeline
def answer_query(query):
    # Morton → LED-A* → Frustum → LOD → Swarm → Halting Gate
    return kv.query(query)  # Full sovereign pipeline
```

### Pattern 4: Batch Implementation, Then Test

```bash
# WRONG: test after every line change
# edit line 1 -> run pytest -> edit line 2 -> run pytest -> ...

# CORRECT: implement full phase, then validate
# 1. Implement Phase changes across all affected files
# 2. Run focused test suite once
# 3. Run full benchmark once (ARC + Math + LHE + GSM8K + MMLU)
# 4. If regression, bisect
```

---

## Collaboration with Claude

**Communication Pattern:**
- **Claude -> Codex**: Architecture specs in TEMP/*.md + real-time tips pointing to kernels, bridges, specs
- **Codex -> Claude**: Progress reports with benchmark results, GPU usage stats
- **Codex implements**: Code + tests per spec
- **Claude reviews**: Architecture alignment, sovereignty compliance, GPU usage

**When stuck on architecture:** Ask Claude. Don't invent new patterns -- the specs define the patterns.

**When stuck on implementation:** Read the live GPU path first. Check which kernels are available in `sovereign_bridges.py` and the PTX inventory.

**Daniel's key corrections (internalize these):**
- "Workers are internal, not external" -- sovereignty applies to ALL hot-path computation
- "Based on meaning, not language" -- reasoning must be language-agnostic via Galaxy meaning layer
- "We fail and fix" -- ZERO fallbacks. If GPU path breaks, fix ON GPU.
- "K3D is a game-like live system, not a benchmark run machine"
- "I can tell from the GPU usage graph Codex is not using all kernels" -- wire ALL 15 GRE specialists

---

## Code References (Read Before Modifying)

**Live GPU Query Path:**
- `knowledge3d/knowledgeverse/knowledgeverse.py` -- live GPU query runtime (~4000 lines, target ~200)
- `knowledge3d/knowledgeverse/query_head_substrate.py` -- bind-time GPU substrate for composed head
- `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py` -- Galaxy entries + reasoning programs
- `knowledge3d/knowledgeverse/semantic_csr_graph.py` -- CSR graph for LED-A* navigation
- `knowledge3d/daemon/main.py` -- daemon routing into `execute_task() -> query()`

**Spatial Sovereign Navigation:**
- `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py` -- LED-A* pathfinding
- `knowledge3d/cranium/spatial_sovereign/morton_octree.py` -- Morton Z-order spatial index
- `knowledge3d/cranium/spatial_sovereign/frustum.py` -- Warp-level frustum culling

**Sovereign Bridges (GRE Kernels):**
- `knowledge3d/cranium/bridges/sovereign_bridges.py` -- 15 GRE kernel bridges (loaded, most NOT called)

**Sovereign Execution:**
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` -- ModularRPNEngine (evaluate, evaluate_batch)
- `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` -- 200+ opcodes
- `knowledge3d/cranium/bridges/tiered_rpn.py` -- Lite/Standard/Extended tiers

**Galaxy Infrastructure:**
- `knowledge3d/knowledgeverse/grammar_galaxy.py` -- Grammar Galaxy
- `knowledge3d/knowledgeverse/specialist_router.py` -- Routing logic
- `knowledge3d/knowledgeverse/galaxy_manager.py` -- Galaxy management

**Specs (Architecture Authority):**
- `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` -- 4-layer architecture
- `docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md` -- Cranium + Galaxy + House
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` -- RPN programs before opcodes
- `docs/vocabulary/SPATIAL_GENERAL_INTELLIGENCE_SPECIFICATION.md` -- SGI paradigm

**Tests:**
- `tests/test_query_head_composition.py` -- Composed head validation
- `tests/test_gpu_galaxy_rpn.py` -- GPU Galaxy/RPN query path regression
- `tests/test_gpu_arc_query.py` -- ARC sovereign query tests
- `tests/test_gpu_math_query.py` -- Math sovereign query tests
- `tests/test_gpu_chat_query.py` -- Chat sovereign query tests

**Benchmarks:**
- `benchmarks/arc_agi_2_adapter.py` -- ARC sovereign harness
- `benchmarks/gsm8k.py` -- GSM8K harness
- `benchmarks/mmlu.py` -- MMLU harness
- `benchmarks/math_competitions.py` -- Math competition harness
- `benchmarks/last_humanity_exam.py` -- LHE harness
- `scripts/run_diagnostic_slices.py` -- Multi-benchmark runner

---

## Codex's Mandate

**Wire ALL GRE specialist kernels into the composed head pipeline. Expand benchmark coverage. Shrink Python orchestration toward ~200 lines. Keep ARC 10/10 and Math 20/20 pinned.**

**CRITICAL REMINDERS:**
1. **Read the specs FIRST** -- they define the architecture, not the code
2. **TRM IS the Avatar** -- it lives in the House, thinks in the Galaxy, runs as a game loop
3. **Wire GRE kernels** -- 15 loaded, most not called. This is the #1 priority.
4. **Sovereignty** -- PTX + Galaxy + RPN + TRM only. No regex, no hardcoded constants, no fallbacks.
5. **Python shrinks** -- every line of Python reasoning orchestration is sovereignty debt
6. **Batch implementation** -- don't test after every line, implement a phase then validate
7. **Pin benchmarks** -- ARC 10/10 and Math 20/20 must not regress
8. **Run ALL benchmarks together** -- not one at a time. The system is ONE living AI.

**For architecture context, always start with [docs/briefings/ARCHITECTURE_BRIEFING.md](docs/briefings/ARCHITECTURE_BRIEFING.md) and the [docs/vocabulary/](docs/vocabulary/) specs.**
