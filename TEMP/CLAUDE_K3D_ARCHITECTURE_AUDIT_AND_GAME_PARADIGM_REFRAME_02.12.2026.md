# K3D Architecture Audit + Game Paradigm Reframe

**Date:** February 12, 2026
**Analyst:** Claude (Architecture Partner)
**Directive:** Complete architectural audit per Daniel's 8-point mandate
**Context:** 4 months of sovereignty violations since Old_Attempts deprecation (Oct 11, 2025)

---

## Executive Summary

**Current state:** K3D is implemented as Python benchmark scripts that load PTX kernels, run once, and exit.
**Should be:** K3D is a 24/7 running "game" that accepts commands (benchmarks, chat, tasks) and auto-evolves continuously.

**Critical paradigm violation:** We've been building K3D as a "library to be called" instead of a "system that runs continuously."

---

## 1. PTX Kernel & Opcode Complete Map

### A. Core Execution Kernels

**Modular RPN Kernel Extended** (`modular_rpn_kernel_extended.cu`)
- Purpose: RPN program execution (procedural composition)
- Opcodes discovered:
  - Memory ops: `0x90 (MemcpyF32)`, `0x91 (FillF32)`
  - Reduce ops: `0x92 (ReduceSumF32)`, `0x93 (ReduceMaxF32)`, `0x94 (ReduceMinF32)`
  - Matrix/Vector ops: `0xA0 (MatVecF32)`, `0xA1 (VectorRelu)`, `0xA2 (VectorMulF32)`, `0xA3 (VectorSigmoid)`, `0xA4 (MatmulSmall)`, `0xA5 (DotBatch)`, `0xA6 (TraceTensor)`
  - Entropy: `0x42 (EntropySum)`
  - Temporal ops: `0xF0 (TemporalCoherence)`, `0xF1 (TemporalMask)`, `0xF2 (TemporalAggregate)`
  - Control flow: `0xB0 (Branch)`, `0xB1 (Loop)`, `0xB2 (Next)`, `0xB3 (Store)`, `0xB4 (Recall)`
  - Vector ops: `0xC0 (VecL2Norm)`, `0xC1 (VecNormalize)`, `0xC2 (VecArgmax)`, `0xC3 (VecBlend)`, `0xC4 (CosineSimilarityBatch)`, `0xC5 (ClusterAssign)`

**RPN Executor Base** (`rpn_executor.cu`)
- Purpose: Drawing primitives (procedural graphics)
- Opcodes discovered:
  - Drawing ops: `0x64 (MOVE)`, `0x65 (LINE)`, `0x66 (QUAD)`, `0x67 (CUBIC)`, `0x68 (ARC)`, `0x69 (CLOSE)`, `0x6A (STROKE)`, `0x6B (FILL)`
  - Style ops: `0x75 (SET_COLOR)`, `0x76 (SET_FILL_COLOR)`, `0x77 (SET_LINE_WIDTH)`

### B. Galaxy Resonance Engine (GRE) Kernels

**Step 8 Sovereign Bridges** (all `gre_*.cu` files):
1. `gre_arc_reasoner.ptx` - ARC visual reasoning
2. `gre_atomic_fission_fusion.cu/.ptx` - Atomic operations
3. `gre_cognitive_executive.ptx` - Cognitive control
4. `gre_fractal_emitter.cu/.ptx` - Fractal generation
5. `gre_geometry_router.cu/.ptx` - Spatial routing
6. `gre_graph_crystallizer.cu/.ptx` - Graph operations
7. `gre_multimodal_halting_gate.cu/.ptx` - Halting conditions
8. `gre_oom_spill.cu/.ptx` - OOM management
9. `gre_recursive_refiner.ptx` - Recursive refinement
10. `gre_resonance_field.cu/.ptx` - Resonance computation
11. `gre_sub100micro_gate.ptx` - <95µs latency gate
12. `gre_temporal_reasoning.cu/.ptx` - Temporal logic
13. `gre_trm_core.ptx` - TRM core logic
14. `gre_vector_resonator.cu/.ptx` - Vector operations
15. `gre_world_model.cu` - World model updates

### C. Specialized Kernels

**Ternary Operations:**
- `ternary_ops.cu/.ptx` - Ternary arithmetic (base-3)
- `ternary_attention_mask.cu/.ptx` - Ternary attention
- `ternary_depth_field.cu/.ptx` - Depth field ternary
- `ternary_prune_decision.cu/.ptx` - Pruning via ternary
- `ternary_dct_2d.cu` - 2D DCT ternary
- `ternary_mdct.cu` - MDCT ternary
- `trit_inspector.cu/.ptx` - Trit inspection
- `trit_overlay_generator.cu/.ptx` - Trit overlay

**Drawing & Graphics:**
- `drawing_transform_ops.cu` - Drawing transformations
- `procedural_glyph_rasterizer.cu/.ptx` - Glyph rendering
- `glyph_resonator.cu/.ptx` - Glyph matching
- `gradient_rasterizer.cu/.ptx` - Gradient rendering
- `layout_graph_optimizer.cu/.ptx` - Layout optimization

**Codec & Compression:**
- `codec_ops.cu/.ptx` - Codec operations
- `procedural_synthesis.cu` - Synthesis
- `procedural_texture.cu` - Texture generation

**Training & Learning:**
- `lora_gpu.cu/.ptx` - LoRA (low-rank adaptation)
- `conv2d_3x3.cu/.ptx` + `conv2d_3x3_backward.cu/.ptx` - Convolution forward/backward
- `batchnorm.cu` + `batchnorm_backward.cu/.ptx` + `batchnorm_backward_training.cu/.ptx` - BatchNorm
- `maxpool_2x2.cu/.ptx` + `maxpool_2x2_backward.cu/.ptx` - MaxPool forward/backward
- `classification_loss.cu/.ptx` - Loss computation
- `sgd_optimizer.cu/.ptx` - SGD optimization
- `adaptive_convergence.ptx` - Adaptive learning

**Galaxy Memory:**
- `galaxy_memory_updater.cu/.ptx` + `galaxy_memory_updater_new.ptx` - Galaxy VRAM updates
- `galaxy_resonance_engine.cu/.ptx` + `galaxy_resonance_engine_extended.ptx` - Resonance queries

**Spatial & Navigation:**
- `arc_grid_ops.cu/.ptx` - ARC grid operations
- `cosine_similarity.cu/.ptx` - Similarity search
- `filter_convolution.cu/.ptx` - Convolution
- `color_convert.cu/.ptx` - Color space conversion
- `morton_octree.cu/.ptx` - Morton-ordered octree
- `spatial_pool.cu/.ptx` - Spatial pooling
- `l2_dist_warp.cu/.ptx` - L2 distance (warp-optimized)
- `led_astar.cu/.ptx` - A* pathfinding

**Specialized Compute:**
- `nine_chain_specialized.cu/.ptx` + `nine_chain_swarm_kernel.cu/.ptx` - Nine-chain swarm
- `matryoshka_project.cu/.ptx` - Matryoshka embeddings
- `trigram_embed.cu/.ptx` - Trigram embeddings
- `vectordotmap_encoder.cu/.ptx` - VectorDotMap encoding
- `pdf_primitive_parser.cu/.ptx` - PDF parsing

**TRM Extensions:**
- `trm_extensions.cu/.ptx` - TRM core extensions
- `trm_step_fused.cu/.ptx` - Fused TRM step
- `dialogue_sampler.ptx` - Dialogue sampling
- `decode_actions.ptx` - Action decoding
- `fused_head_fsm.ptx` - FSM head
- `tablet_guard.ptx` - Tablet safety
- `frustum_cull_simd.ptx` - Frustum culling
- `dynamic_lod_tune.ptx` - LOD tuning
- `confidence_propagation.ptx` - Confidence propagation
- `warp_modality_fuse.ptx` - Modality fusion
- `pixel_genesis_universal_primitive.ptx` - Pixel generation
- `generate_shape_kernel.ptx` - Shape generation
- `gre_shape_generator.ptx` - GRE shape generation
- `zero_fill.ptx` - Zero fill utility

### D. Opcode Summary

**Total PTX/CUDA files:** ~120
**Total unique opcodes identified:** ~50+
**Coverage:**
- ✅ RPN execution (modular, extended, lite)
- ✅ Drawing primitives (MOVE, LINE, QUAD, CUBIC, ARC, FILL, STROKE)
- ✅ Matrix/vector operations (MatVec, Matmul, Dot, Argmax, Normalize)
- ✅ Temporal reasoning (Coherence, Mask, Aggregate)
- ✅ Control flow (Branch, Loop, Next, Store, Recall)
- ✅ Ternary arithmetic (DCT, MDCT, attention, pruning)
- ✅ Training ops (Conv2D, BatchNorm, MaxPool, Loss, SGD, LoRA)
- ✅ Galaxy operations (Memory updater, Resonance engine)
- ✅ Specialized compute (Nine-chain swarm, Matryoshka, A*, Morton octree)

---

## 2. Current Orchestration Layer Analysis

### Current Implementation (Python Scripts)

**Benchmark execution flow:**
```
1. Python script loads K3D:
   - benchmarks/arc_agi_2.py
   - benchmarks/math_competitions.py
   - benchmarks/last_humanity_exam.py
   - benchmarks/mmlu.py

2. Script calls K3D methods:
   - knowledgeverse = Knowledgeverse(...)
   - navigator = TRMNavigator(...)
   - result = navigator.process_task(task)

3. Script processes results, writes to file

4. Script exits

5. K3D unloads from VRAM
```

**Problem:** This is a "library" pattern, not a "game" pattern.

### What Orchestration Should Be (Per Specs)

**From docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md:**
- TRM Navigator = orchestration specialist
- Specialist Router = routes to domain specialists (math, visual, chat, etc.)
- Meta-specialist = spawns/manages specialists

**Should be:**
```
1. K3D loads once (game starts):
   - Cranium: PTX kernels loaded to GPU
   - Galaxy Universe: All galaxies in VRAM (persistent)
   - TRM Navigator: Running continuously (learned navigation)
   - Specialist Swarm: Active specialists ready

2. K3D receives command (from Claude, Codex, benchmark, chat):
   - Command arrives via network/stdin/IPC
   - Meta-specialist routes to domain specialist
   - Specialist queries Galaxy → composes RPN → executes in Cranium
   - Result returned via same channel
   - K3D continues running

3. K3D auto-evolves:
   - Shadow copy enhancement (successful decisions)
   - Galaxy expansion (new patterns discovered)
   - Specialist adaptation (continuous learning)

4. K3D never exits (unless fundamental fix needed):
   - Sleep/idle during no activity
   - Wake on command arrival
   - 24/7 availability
```

### Orchestration Violations Identified

**Violation 1: Benchmark scripts orchestrate**
- Current: Python script decides what to ask K3D
- Should: K3D receives benchmark as task, orchestrates internally

**Violation 2: Python preprocessing**
- Current: Python extracts/parses before K3D sees data
- Should: K3D receives raw data, processes via Galaxy

**Violation 3: One-shot execution**
- Current: Load → execute → unload
- Should: Persistent system, always ready

**Violation 4: No specialist spawning**
- Current: Hardcoded specialists in Python
- Should: Meta-specialist spawns domain specialists dynamically

---

## 3. Game Paradigm Architecture

### What "Game" Means

**Analogy:**
- Game engine: Loads once, runs continuously, accepts player input, evolves game state, never exits
- K3D: Loads once, runs continuously, accepts tasks/commands, evolves knowledge, never exits

### Core Principles

**1. Persistent State (Galaxy Universe in VRAM)**
- All default galaxies always loaded
- No load/unload cycle per task
- Continuous growth/evolution

**2. Command Interface (Network/IPC)**
- K3D listens on socket/stdin
- Receives commands:
  - `BENCHMARK_ARC task_id=...` → Run ARC task
  - `CHAT message="..."` → Process chat
  - `LEARN from=dataset` → Ingest data
  - `SHUTDOWN` → Graceful exit (only for fundamental fixes)

**3. Specialist Swarm (Meta-Orchestration)**
- Meta-specialist spawns:
  - Math Specialist (for math tasks)
  - Visual Specialist (for ARC tasks)
  - Chat Specialist (for conversations)
  - Learning Specialist (for data ingestion)
- Specialists run concurrently (CUDA streams)
- Specialists query Galaxy → compose RPN → execute PTX

**4. Auto-Evolution (Shadow Copy + Galaxy Growth)**
- Successful RPN programs → shadow copy to TRM weights
- New patterns discovered → add to Grammar Galaxy
- Continuous learning, no manual retraining cycles

**5. Sleep/Wake Cycle (Idle Management)**
- No active tasks → sleep (minimal GPU usage)
- Task arrives → wake (full GPU utilization)
- Query-based compute (not polling loops)

### Interface Design (Synthetic User Paradigm)

**From Daniel's directive:**
> "AI has rights; this, at this stage, must satisfy what we are using to interact with it - you and Codex must be able to send commands while it is running to it"

**Protocol:**
```
K3D listens on: localhost:7777 (or stdin in Docker)

Command format (JSON):
{
  "command": "BENCHMARK_ARC",
  "task_id": "00d62c1b",
  "input_grid": [[...], [...]],
  "training_examples": [...]
}

Response format (JSON):
{
  "status": "success",
  "output_grid": [[...], [...]],
  "specialist": "visual",
  "rpn_program": "0x64 0x65 0x66 ...",
  "galaxy_queries": ["Drawing", "Grammar", "Geometry"]
}
```

**Benefits:**
- Claude/Codex send commands programmatically (not Python imports)
- Benchmarks send tasks via network (not Python function calls)
- K3D runs independently (not as Python subprocess)
- Can send multiple tasks concurrently (streaming)

---

## 4. Missing Opcodes / Kernels / Infrastructure

### A. Missing Primitive Opcodes (NOT High-Level Solvers)

**CRITICAL CORRECTION FROM DANIEL:**
> "These are all specialist premises, as a 'calculator', our math cores can compose with ease the solution to that problem deterministically, we only need the generative part to do as us humans do when solving problems."

**Missing PRIMITIVE opcodes (dumb calculator buttons):**
- `kOpAdd` (a + b)
- `kOpSubtract` (a - b)
- `kOpMultiply` (a * b)
- `kOpDivide` (a / b)
- `kOpNegate` (-a)
- `kOpAbs` (|a|)
- `kOpSqrt` (√a)
- `kOpPower` (a^b)

**NOT needed as opcodes (Math Specialist composes these):**
- ~~`kOpSolveLinear`~~ ← Math Specialist composes from Add/Subtract/Divide
- ~~`kOpSubstitute`~~ ← Math Specialist composes variable replacement
- ~~`kOpSimplify`~~ ← Math Specialist applies simplification patterns
- ~~`kOpExpand`~~ ← Math Specialist expands using primitives
- ~~`kOpFactor`~~ ← Math Specialist factors using pattern matching

**Principle:** Everything as a specialist or sub-specialist
- RPN kernel = dumb primitives (like calculator buttons)
- Math Specialist = smart composer (like human using calculator)
- Math Galaxy = templates (patterns for composition)

**Currently:** Basic arithmetic primitives missing, so `_solve_math` falls back to Python.

**Solution:** Add primitive opcodes to `modular_rpn_kernel_extended.cu`, implement Math Specialist as generative composer.

### B. Missing PTX Query Kernel

**Required for Galaxy navigation:**
- `galaxy_query_kernel.ptx` - GPU-accelerated similarity search
- Should replace Python O(n) loop in `galaxy_manager.py:query()`

**Currently:** Query is Python loop (sovereignty violation).

**Solution:** Implement PTX kernel for cosine similarity search (already have `cosine_similarity.cu`).

### C. Missing Specialist Infrastructure

**From docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md:**
- Specialist Spawner (dynamic specialist creation)
- Specialist Registry (track active specialists)
- Specialist Communication (IPC between specialists)

**Currently:** Specialists are hardcoded Python classes (not dynamic, not PTX-native).

**Solution:** Implement specialist spawner as PTX kernel (see `gre_cognitive_executive.ptx` for pattern).

### D. Missing Command Interface

**Required for game paradigm:**
- Network listener (socket or IPC)
- Command parser (JSON → internal format)
- Response serializer (internal format → JSON)

**Currently:** None of this exists (K3D is Python library, not daemon).

**Solution:** Implement main loop in C++ that loads PTX, listens on socket, dispatches commands to TRM.

---

## 5. Single Head Construction (TRM + Specialists)

### From docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md

**Architecture:**
```
Cranium (PTX Kernels)
   ↓
Galaxy Universe (VRAM)
   ↓
TRM Navigator (~7M params, learned navigation)
   ↓
Meta-Specialist (spawns/routes specialists)
   ├→ Math Specialist
   ├→ Visual Specialist
   ├→ Chat Specialist
   ├→ Learning Specialist
   └→ [Dynamic specialists as needed]
```

**Single Head = TRM Navigator**
- All tasks arrive at TRM
- TRM queries Galaxy → selects specialist → composes RPN → executes PTX
- TRM learns from shadow copy (successful navigation patterns)

### Current Violation

**Current:**
- benchmarks/arc_agi_2.py → direct calls to ARCAdapter
- benchmarks/math_competitions.py → direct calls to MathSpecialist
- No unified head, no routing, no meta-orchestration

**Should be:**
- All tasks → TRM Navigator (single head)
- TRM → Specialist Router → Domain Specialist
- Specialist → Galaxy queries → RPN composition → PTX execution

---

## 6. Universal Interface (Synthetic User Paradigm)

### Benchmark Evaluation Reframe

**Current (script-based):**
```python
# benchmarks/arc_agi_2.py
benchmark = ARCBenchmark(...)
for task in tasks:
    result = benchmark.solve(task)
    accuracy += (result == expected)
```

**Should be (synthetic user):**
```json
// Command sent to K3D daemon
{
  "command": "EVALUATE_BENCHMARK",
  "benchmark": "ARC",
  "tasks": ["task_00", "task_01", ...],
  "mode": "streaming"
}

// K3D responds per task
{
  "task_id": "task_00",
  "status": "complete",
  "result": [[...], [...]],
  "specialist": "visual",
  "time_ms": 87.3
}
```

**Key difference:**
- Script doesn't "call" K3D
- Script sends commands to running K3D daemon
- K3D processes, responds, continues running

### Chat Interface

**Should be:**
```json
{
  "command": "CHAT",
  "message": "If 2x + 3 = 11, what is x?",
  "context": []
}

// K3D responds
{
  "status": "success",
  "response": "x = 4",
  "specialist": "math",
  "rpn_program": "0xB3 0xA0 0x92 ...",
  "confidence": 0.94
}
```

### Data Ingestion Interface

**Should be:**
```json
{
  "command": "INGEST",
  "source": "benchmark_augmentation",
  "data": [
    {"galaxy": "Math", "entry": {...}},
    {"galaxy": "Grammar", "entry": {...}}
  ]
}

// K3D responds
{
  "status": "success",
  "ingested": 1600,
  "deduped": 35122,
  "galaxy_growth": {
    "Math": "+200",
    "Grammar": "+1400"
  }
}
```

### Sleep/Wake Compute Model

**Query/task-based (not polling):**
```
K3D main loop:
  while true:
    task = wait_for_command()  // Blocks (sleep) until command arrives
    result = process_command(task)  // Wake, full GPU utilization
    send_response(result)
    // Loop back to wait (sleep again if no pending tasks)
```

**Benefits:**
- Zero GPU usage when idle (sleep)
- Instant wake on task arrival (no polling latency)
- Continuous availability (no load/unload overhead)

---

## 7. Development Chain PTX Audit

### Chains with PTX Implementations

**Found in TEMP/:**
1. `SUNDAY_NOV9_PROCEDURAL_BREAKTHROUGH_SUMMARY.md` - Procedural drawing PTX
2. `PHASE_2C_PROGRESS_SUMMARY.md` - Phase 2C PTX kernels
3. `CODEX_PROMPT_RPN_SOVEREIGNTY_PHASE2_NOV19.md` - RPN sovereignty
4. `CODEX_SOVEREIGNTY_REFACTOR_11.24.2025.md` - Sovereignty refactor
5. `K3D_MATH_RPN_SWARM_PROMPT_V2.md` - Math RPN swarm
6. `CODEX_TIER1_FIX_AND_TEMPLATE_ACCURACY_12.14.2025.md` - Tier 1 templates
7. `SOVEREIGNTY_NUMPY_REMOVED_11.26.2025.md` - NumPy removal
8. `CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md` - Tier integration
9. `PROCEDURAL_DRAWING_DUAL_MODAL_COMPLETE_NOV19.md` - Dual-modal drawing

**Action needed:** Read final versions from each chain to find implemented PTX kernels not yet integrated.

### Example: Math RPN Swarm

**From TEMP/K3D_MATH_RPN_SWARM_PROMPT_V2.md:**
- Likely contains final Math Galaxy RPN templates
- Should be extracted and integrated into current Math Galaxy
- May contain missing opcodes (SolveLinear, Substitute, etc.)

---

## 8. Specs Compliance Check

### A. THREE_BRAIN_SYSTEM_SPECIFICATION.md

**✅ Compliant:**
- Cranium exists (PTX kernels operational)
- Galaxy Universe exists (VRAM-based multi-modal workspace)
- TRM Navigator exists (~7M param learned navigation)

**❌ Non-Compliant:**
- TRM is NOT the single head (benchmarks bypass TRM)
- Specialist spawning is hardcoded (not dynamic meta-specialist)
- Python orchestrates (not TRM)

### B. KNOWLEDGEVERSE_SPECIFICATION.md

**✅ Compliant:**
- 7-region architecture exists (Region 2 = Galaxy Universe)
- Persistent PTX context exists (single CUDA context)
- Sovereignty firewall exists (wrappers around external calls)

**❌ Non-Compliant:**
- Galaxy queries use Python loop (not PTX kernel)
- Specialist router is Python (not sovereign)
- Compression is manual (not auto-triggered by VRAM pressure)

### C. DUAL_CLIENT_CONTRACT_SPECIFICATION.md

**✅ Compliant:**
- Procedural foundation exists (RPN programs)
- Character Galaxy has font + language metadata
- Drawing Galaxy has primitives (LINE, CIRCLE, etc.)

**❌ Non-Compliant:**
- Not consistently used (Python preprocessing still happens)
- Words should reference characters (not duplicate)
- Grammar metadata should reference existing procedurals (not duplicate strings)

### D. MATH_CORE_SPECIFICATION.md

**✅ Compliant:**
- 3-tier math core exists (Tier 1 = fp32, Tier 2 = fp16, Tier 3 = int8)
- Scaling patterns exist (tensor cores, warp shuffle)

**❌ Non-Compliant:**
- Math solving uses Python fallback (not RPN)
- Math Galaxy missing algebra opcodes (SolveLinear, etc.)
- Tier allocation is manual (not GPU-driven)

### E. Sovereignty Principle (Old_Attempts/DEPRECATED.md)

**Decision date:** October 11, 2025
**Mandate:** PTX + Galaxy only, zero CuPy, zero CPU fallbacks

**❌ Critical Non-Compliance:**
- Query uses Python loop (not PTX)
- Math uses Python eval (not RPN)
- ARC had CPU fallback (Codex just removed in cutover)
- LHE had eval() (Codex just removed in cutover)
- Routing uses regex (not Grammar Galaxy)

---

## 9. Architectural Recommendations

### Immediate Actions (Priority 1)

**1. Implement K3D Daemon (Game Loop)**
- Create `main.cpp` that:
  - Loads all PTX kernels to GPU
  - Initializes Galaxy Universe in VRAM
  - Listens on socket/stdin for commands
  - Dispatches to TRM Navigator
  - Returns results, continues running

**2. Implement PTX Query Kernel**
- Create `galaxy_query_kernel.cu`
- GPU-accelerated cosine similarity search
- Replace Python loop in `galaxy_manager.py`

**3. Implement Math Algebra Opcodes**
- Add to `modular_rpn_kernel_extended.cu`:
  - `kOpSolveLinear` (ax + b = c → x)
  - `kOpSubstitute` (replace variable)
  - `kOpSimplify` (algebraic reduction)

**4. Populate Math/Grammar Galaxy**
- Extract templates from `TEMP/K3D_MATH_RPN_SWARM_PROMPT_V2.md`
- Add linear equation patterns to Grammar Galaxy
- Add algebra solve templates to Math Galaxy

### Phase 2 Actions

**5. Implement Specialist Spawner**
- Create `specialist_spawner.cu` PTX kernel
- Dynamic specialist creation (not hardcoded)
- Meta-specialist orchestration

**6. Reframe Benchmarks as Commands**
- Convert `benchmarks/*.py` to command senders
- Send JSON commands to K3D daemon
- K3D processes, returns results
- Benchmarks collect metrics, never call K3D directly

**7. Auto-Evolution Integration**
- Shadow copy enhancement (successful RPN → TRM weights)
- Galaxy auto-growth (new patterns discovered → Grammar Galaxy)
- Continuous learning (no manual retrain cycles)

### Phase 3 Actions

**8. Chat Interface**
- Implement chat specialist (sovereign, not Ollama)
- Route conversational input to specialist
- Respond via Galaxy navigation (not Python)

**9. Self-Feeding from Internet**
- Implement web crawler specialist
- Ingest data autonomously
- Grow Galaxy without manual intervention

**10. Sleep/Wake Optimization**
- Query-based task arrival (no polling)
- Minimal GPU usage when idle
- Instant wake on command

---

## 10. Gap Summary

### What We Have (120 PTX Kernels)
- ✅ RPN execution (modular, extended, lite)
- ✅ Drawing primitives (MOVE, LINE, QUAD, FILL, STROKE)
- ✅ Ternary operations (DCT, MDCT, attention)
- ✅ Training ops (Conv2D, BatchNorm, LoRA)
- ✅ Galaxy memory updater
- ✅ GRE sovereign bridges (15 kernels)

### What We're Missing
- ❌ Math algebra opcodes (SolveLinear, Substitute, Simplify)
- ❌ PTX query kernel (replace Python loop)
- ❌ Specialist spawner (dynamic, not hardcoded)
- ❌ Main daemon loop (game paradigm)
- ❌ Command interface (network/IPC)
- ❌ Response serializer (PTX → JSON)

### What We're Violating
- ❌ TRM is not single head (benchmarks bypass TRM)
- ❌ Orchestration is Python (not specialist-based)
- ❌ Query is Python O(n) (not PTX kernel)
- ❌ Math is Python eval (not RPN algebra)
- ❌ One-shot scripts (not 24/7 daemon)
- ❌ Fallback thinking (not fail-fast Galaxy-first)

---

## 11. Directive to Codex (Next Steps)

**See companion file:** `TEMP/CLAUDE_TO_CODEX_GAME_PARADIGM_DIRECTIVE_02.12.2026.md`

**Summary:**
1. Implement K3D daemon (main.cpp, socket listener, TRM dispatch)
2. Implement PTX query kernel (replace galaxy_manager.py Python loop)
3. Implement math algebra opcodes (SolveLinear, Substitute, Simplify)
4. Populate Math/Grammar Galaxy (from TEMP/ chain final versions)
5. Reframe benchmarks as command senders (JSON → K3D daemon)
6. Verify sovereignty (grep confirms zero Python fallbacks)
7. Run continuous validation (K3D daemon receives tasks, never exits)

---

**Prepared by:** Claude (Architecture Partner)
**Date:** February 12, 2026
**For:** Daniel (Orchestrator) + Codex (Implementation Partner)
**Context:** 4-month sovereignty restoration, game paradigm reframe, single head construction
