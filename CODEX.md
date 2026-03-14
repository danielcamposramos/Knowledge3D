# CODEX.md -- Implementation Lead Guide

**Last Updated:** March 10, 2026
**Version:** 5.0 (Composed Head Sovereign + GRE Kernel Wiring)

Codex-style agents lead implementation, Reality Galaxy, and testing. Read the latest briefing first for the full architecture; this file captures Codex's role, patterns, and backlog.

---

## Current Benchmark State

**Composed Head Pipeline (LIVE on GPU):**
```
Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate
```

| Benchmark | Curated Set | Expanded (B+) | Status |
|-----------|------------|---------------|--------|
| ARC | 10/10 | 10/50 | 9 distinct worker programs; 34 transform misses + 6 primitive bugs |
| Math | 20/20 | — | GPU query path sovereign |
| LHE | 10/10 | 10/100 | Factual lookup dominates; multi-hop needed |
| GSM8K | — | 10/50 | Word-problem decomposition weak |
| MMLU | — | 0/50 | Galaxy has no relevant entries in target neighborhoods |

**Key Achievement:** First sovereign GPU-converged answer ("What is 2+3?" = 5) with ZERO Python fallback.

**Sovereignty note (current truth):**
- ALL benchmarks route through `Knowledgeverse.execute_task() -> query() -> knowledgeverse_gpu_query`
- Composed head pipeline: Morton → LED-A* → Frustum → LOD → Nine-Chain Swarm → Halting Gate
- Halting gate bug FIXED: `>` to `>=` in `gre_multimodal_halting_gate.cu` (agree=3 was rejected)
- Cosine similarity moved from Python to GPU bridge (`cosine_similarity.ptx`)
- Halting agreement/gap computation moved from Python to PTX kernel (`analyze_scores()`)
- MMLU scoring moved from Python to RPN expressions via `evaluate_batch()`
- **ZERO fallbacks. If it breaks, fix on GPU.**

---

## CRITICAL: TRM IS the Avatar (Read This First)

**The TRM is NOT a function Python calls. It IS the AI entity.**

- Lives in the House (Memory Palace), thinks in the Galaxy (internal brain)
- Runs as a game loop (`trm_step_fused.ptx` = one game tick)
- Internal swarm = parallel cognitive channels ("superdotados" model)
- Python = boot + I/O ONLY (~200 lines target, NOT 4000 lines of orchestration)

**Current Sovereignty Debt:**
- `knowledgeverse.py` is ~4000 lines of Python orchestration → target ~200 lines
- `_select_composed_head_candidate()` is a 200-line Python for-loop → should be GPU kernel
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

4. **Read the latest Claude directive:**
   - [TEMP/CLAUDE_COMPOSED_HEAD_CONVERGENCE_PLAN_03.10.2026.md](TEMP/CLAUDE_COMPOSED_HEAD_CONVERGENCE_PLAN_03.10.2026.md) -- THE active plan. Supersedes all previous directives.

5. **Read the GPU environment policy:**
   - [docs/ENV_POLICY.md](docs/ENV_POLICY.md) -- critical GPU setup (CUDA_VISIBLE_DEVICES=0)
   - [envs/README.md](envs/README.md) -- conda environment selection

**Why:** Partial reads cause sovereignty violations, architecture misunderstandings, and wasted work. The specs define HOW K3D reasons -- ignore them and you'll build Python pattern matchers instead of sovereign Galaxy navigators.

---

## Quick Start (After Reading Briefing + Specs)
- Check [docs/ROADMAP.md](docs/ROADMAP.md) for current phase.
- Review Claude's specs in TEMP/*.md (latest dated March 10, 2026).
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

### Priority 1: Wire GRE Specialist Kernels (See table above)

### Priority 2: Fix Benchmark Gaps

**MMLU 0/50 (abstract_algebra):**
- Galaxy has no entries in abstract_algebra neighborhood
- Need to populate Math Galaxy with group theory, ring theory, field theory entries
- Use `gre_resonance_field` for broad knowledge matching

**GSM8K 10/50:**
- Word-problem decomposition weak
- Wire `gre_atomic_fission_fusion` for problem decomposition
- Need Grammar Galaxy rules for extracting numeric relationships from text

**LHE 10/100:**
- Factual lookup dominates; multi-hop reasoning weak
- Wire `gre_graph_crystallizer` for multi-hop graph traversal
- Expand Reality Galaxy coverage

**ARC 10/50:**
- 34 transform misses + 6 primitive bugs (`masked_patch_requires_color_8_bbox`)
- All 9 workers now have distinct programs (diversity fixed)
- Need transform-specific coverage expansion

### Priority 3: Shrink Python Orchestration (Phase D Prep)

**Target:** `knowledgeverse.py` from ~4000 → ~200 lines

**What stays in Python:** Boot, I/O, display, network
**What moves to GPU:** All reasoning orchestration, scoring, candidate selection, answer formatting

### Priority 4: Cross-Benchmark Session Management

**Bug:** Single Knowledgeverse instance shared across benchmarks in `run_diagnostic_slices.py` causes cross-contamination through `_gpu_reasoning_programs` cache, `_gpu_galaxy_binding`, `trm_navigator`, `_query_sequence`.

**Fix:** Implement `reset_query_session()` clearing per-benchmark mutable state between sections.

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
