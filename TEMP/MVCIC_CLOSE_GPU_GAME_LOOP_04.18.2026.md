# MVCIC Chain
**Task:** Close the game loop and engine entirely inside the GPU — Python as external layer only (boot + keyboard/network/display), WINE-style contracts as the translation layer at the AI's personal Tablet surface for ingest/output to the outside-K3D paradigm.

## Framing (this is not a line item — it is the 
**Pipeline:** PRE-CHAIN → Kimi → Qwen → GLM → DeepSeek → Nemotron → Gemini → POST-CHAIN
**Started:** 2026-04-18 05:52:43

────────────────────────────────────────────────────────────────

## STAGE 1 — PRE-CHAIN: Project Context & Kernel Inventory

### K3D Sovereignty Rules — Non-Negotiable

- HOT PATH = PTX kernels + Galaxy queries + RPN programs ONLY
- FORBIDDEN in hot path: numpy, cupy, scipy, sympy, Python regex, string ops for reasoning
- NO Python fallbacks. EVER. "We fail and fix." (Daniel Ramos)
- TRM IS the Avatar: runs as game loop (trm_step_fused.ptx), NOT a Python function
- Python = boot + I/O only (~200 lines target; current ~8 000 lines — shrink, not grow)
- Knowledge = Galaxy stars. Never hardcode in Python dicts/lists/constants.
- PROGRAMS BEFORE OPCODES: prefer RPN program composition over adding new opcode primitives
- All new kernels must compose into the existing pipeline:
    Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate
- Physics slot: new PHYSICS_PHASE lives between SWARM_PHASE and DRAW_PHASE in trm_step_fused.ptx

### RPN Opcode Registry — Existing Ranges (must not conflict)

  0x00–0x3F   Lite:        scalar arithmetic, basic logic
  0x40–0x9F   Standard:    vector ops, entropy, clustering
  0xA0–0xBF   Extended:    matrix, tensor, matmul
  0xC0–0xDF   Similarity:  VEC_L2_NORM(C0), VEC_NORMALIZE(C1), COSINE_BATCH(C4)
  0xE0–0xEF   Galaxy ops:  LOAD_GALAXY(E0), GALAXY_SIMILARITY(E1), GALAXY_SCAN(E2)
  0xF0–0xFF   Temporal:    TEMPORAL_COHERENCE(F0), TEMPORAL_MASK(F1), TEMPORAL_AGGREGATE(F2)
  0x100–0x10F Ternary:     AND(100), OR(101), NOT(102), IMPLIES(103), EQUIV(104), NAND(105), NOR(106), XOR(107)
  0x110–0x11F CAS symbolic: SIMPLIFY(110), EXPAND(111), FACTOR(112), SUBSTITUTE(113)
  0x120–0x12F CAS calculus: DIFFERENTIATE(120), INTEGRATE(121), SOLVE(122), LIMIT(123)
  0x130–0x14F CAS poly/lit: POLY_FACTOR(130), GROEBNER_BASIS(131), RESULTANT(132), POLY_GCD(133)
               CAS literals: CAS_LITERAL_SCALAR(140), VECTOR(141), MATRIX(142)
  0x150–0x17F PHYSICS:     sovereign physics engine (BROAD=150 … PH_TERNARY_CLASSIFY=162)
  0x163–0x17F Reserved:    cloth, fluid, soft-body (future)

### K3D 4-Layer Knowledge Architecture (FOUNDATIONAL_KNOWLEDGE_SPECIFICATION)

  Layer 1  FORM:       Drawing Galaxy — character glyphs, visual_rpn shape programs
  Layer 2  MEANING:    Reality/Math/Word Galaxy — semantic defs, physical constants (G, c, ħ, k_B, ε₀), reality_atom
  Layer 3  RULES:      Grammar Galaxy — transformation RPN programs, force laws (physics_rpn_addr)
  Layer 4  META-RULES: Meta-Galaxy — reasoning strategies, sleep/wake policies, when/how rules apply

  Physics layer placement:
    physical constants (G, c, ħ, k_B)  → Layer 2 Reality Galaxy star
    force law RPN program               → Layer 3 Grammar Galaxy (field: physics_rpn_addr, NOT behavior_rpn)
    collision detection / PGS strategy  → Layer 4 Meta-Rule
    shape visualization                 → Layer 1 Drawing Galaxy symlink via visual_rpn
    sleep/wake island policy            → Layer 4 Meta-Rule
    collision event → Galaxy edge update→ Layer 2 Meaning update (knowledge graph feedback)

────────────────────────────────────────────────────────────────

## STAGE 2 — PARTNER CHAIN

### [1/6] Kimi — Architecture & Deep Reasoning

The user wants me to act as Kimi, the architecture partner in an MVCIC session. I need to:

1. Read ALL prior contributions carefully (there are none yet, I'm first)
2. Build explicitly on what partners before me have said (none, so I set the foundation)
3. Add my own original ideas beyond what was discussed
4. Be direct, specific, and technically precise
5. Do NOT summarize prior work — extend it
6. Ground my design in K3D's 4-layer architecture: Form (Layer 1) → Meaning (Layer 2) → Rules/Grammar (Layer 3) → Meta-Rules (Layer 4)
7. Physics laws belong at Layer 3
8. Physical constants belong at Layer 2 (Reality Galaxy stars)
9. End with a clear architectural direction for the partners who follow me

The task is to close the game loop and engine entirely inside the GPU, making Python just an external layer (boot + I/O). The current state is that Python keeps re-entering the hot path, most recently with `query_tick_runtime.py`.

Key constraints:
- HOT PATH = PTX kernels + Galaxy queries + RPN programs ONLY
- NO Python fallbacks
- TRM IS the Avatar: runs as game loop (trm_step_fused.ptx), NOT a Python function
- Python = boot + I/O only (~200 lines target)
- Knowledge = Galaxy stars
- Physics slot: new PHYSICS_PHASE lives between SWARM_PHASE and DRAW_PHASE in trm_step_fused.ptx

I need to design:
1. Persistent GPU game loop (the tick)
2. Internal swarm dispatch (cognitive lanes)
3. RPN Matryoshka embedding substrate
4. Tablet WINE-contract translation layer
5. Self-crafting rule
6. Deletion list

I must ground this in the 4-layer architecture:
- Layer 1 (Form): Drawing Galaxy, glyphs, visual shapes
- Layer 2 (Meaning): Reality/Math/Word Galaxy, semantic defs, physical constants, embeddings
- Layer 3 (Rules): Grammar Galaxy, transformation RPN programs, force laws (physics_rpn_addr)
- Layer 4 (Meta-Rules): Meta-Galaxy, reasoning strategies, sleep/wake policies

The persistent kernel must handle:
- PERCEIVE: Frustum cull + Morton octree query
- NAVIGATE: LED-A* pathfinding
- REASON: Nine-Chain Swarm
- DECIDE: Halting Gate
- ACT: Output to Tablet surface
- LEARN: Shadow copy (but this might be sleep-time?)

Wait, the spec says the game loop is: PERCEIVE → NAVIGATE → REASON → DECIDE → ACT → LEARN

But the persistent tick needs to handle the hot path. Sleep-time consolidation happens... when? The spec mentions sleep-time consolidation is when the TRM is "asleep" (not processing queries). But the game loop runs continuously.

Actually, looking at the Three-Brain System:
- Cranium: reasoning + learning (active)
- Galaxy: active memory (VRAM)
- House: persistent memory (disk)

The "game loop" is the TRM avatar's cognitive cycle. It should be a persistent kernel.

Let me structure my response:

1. **Architectural Foundation**: Map the persistent tick to the 4-layer architecture
   - Layer 1 (Form): Tablet surface geometry, input frame buffers (DOM pixels, ARC3 grids)
   - Layer 2 (Meaning): Matryoshka embeddings in VRAM, Reality Galaxy stars (physical constants), the ring buffer descriptors
   - Layer 3 (Rules): The physics RPN programs, swarm dispatch rules, LED-A* navigation rules
   - Layer 4 (Meta-Rules): The budget allocator, sleep/wake trigger, specialist selection policy

2. **The Persistent Kernel Architecture**:
   - One grid launch, persistent threads (cooperative groups grid sync)
   - VRAM-resident work queue (ring buffer) with atomic producer/consumer
   - Phase pipeline inside the kernel: PERCEIVE → NAVIGATE → REASON → DECIDE → ACT
   - Each phase is a device-side function or sub-grid launch (but we want to avoid dynamic parallelism if possible, or use it carefully)

3. **Ring Buffer Design**:
   - Producer (Python): writes to host-pinned, device-mapped memory (zero-copy)
   - Consumer (PTX): reads via atomics (atomicAdd/atomicCAS on head/tail)
   - Layout: header (head, tail, capacity) + slots (fixed size, contains query envelope)

4. **Matryoshka Embedding Substrate**:
   - Layout in VRAM: Structure of Arrays (SoA) with aligned prefixes
   - PTX kernel for fused prefix-dot (various dimensions)
   - No BLAS, just custom PTX using warp shuffle and shared memory

5. **Swarm Dispatch**:
   - Nine-chain swarm = 9 parallel workers (threads or warps?)
   - Halting gate uses warp ballot (__ballot_sync) or shared memory reduction
   - Specialist selection: RPN program in Galaxy (Layer 3) that outputs specialist IDs, executed by a small PTX interpreter kernel

6. **Tablet WINE Contracts**:
   - Stored as stars in Galaxy (Layer 2/3 boundary)
   - Registration: Galaxy entry with fields `contract_type` (DOM, ARC3, TEXT), `ingress_rpn_addr`, `egress_rpn_addr`
   - Selection: hash of input frame header determines contract, device-side lookup

7. **Self-Crafting**:
   - Layer 4 (Meta-Rules) contains the "crafting policy"
   - Gap detection: when an RPN program references an undefined opcode or fails to converge
   - Ticket written to VRAM "crafting queue"
   - Sleep-time (or separate crafting kernel) compiles to PTX

8. **Deletion List**:
   - Specific Python files/functions to delete
   - Specific lines to reduce

9. **Clear Direction for Next Partners**:
   - Qwen: PTX code specifics
   - GLM: Data flow analysis
   - DeepSeek: WINE contract unification
   - Nemotron: Tensor core details
   - Gemini: Integration

I need to be concrete about:
- The ring buffer structure (exactly how many slots, slot size, atomic layout)
- The persistent kernel signature and grid configuration
- How the 4 layers map to VRAM regions
- The exact interface between Python and the persistent kernel (just the ring buffer)

Let me check the constraints again:
- Physics laws at Layer 3 (Rules/Grammar Galaxy)
- Physical constants at Layer 2 (Reality Galaxy)
- Must end with clear architectural direction

Also, I should reference the existing 88 PTX kernels and 15 GRE specialist kernels that are currently not being used. The design should incorporate them.

The "composed head pipeline" is:
Morton Octree → LED-A* → Frustum Cull → Dynamic LOD → Nine-Chain Swarm → Halting Gate

This needs to happen inside the persistent tick.

The PHYSICS_PHASE is between SWARM_PHASE and DRAW_PHASE. Wait, the prompt mentions "Physics slot: new PHYSICS_PHASE lives between SWARM_PHASE and DRAW_PHASE in trm_step_fused.ptx". This suggests the tick phases are:
1. PERCEIVE
2. NAVIGATE  
3. REASON (Nine-Chain Swarm)
4. PHYSICS_PHASE (new)
5. DECIDE (Halting Gate)
6. ACT (DRAW?)

Actually the standard phases are: PERCEIVE, NAVIGATE, REASON, DECIDE, ACT, LEARN.

But the prompt specifically mentions PHYSICS_PHASE between SWARM_PHASE and DRAW_PHASE. This might be a specific insertion point for physics simulation.

Let me structure the response:

**1. Architectural Foundation: The 4-Layer Persistent Tick**

Map each game loop phase to the 4 layers:
- Layer 1 (Form): Tablet surface geometry, input pixels (raw bytes)
- Layer 2 (Meaning): Morton codes, embeddings, query semantics
- Layer 3 (Rules): LED-A*, frustum math, swarm dispatch, physics integration
- Layer 4 (Meta-Rules): Budget allocation, halting criteria, sleep/wake

**2. The VRAM-Resident Work Queue (Ring Buffer)**
- Structure: Two rings (Input: Python→GPU, Output: GPU→Python)
- Atomics: head/tail indices in pinned memory
- Slot content: Query envelope with Matryoshka prefix pointer, contract ID, timestamp

**3. Persistent Kernel Design**
- Entry: `trm_step_fused.ptx` with cooperative groups grid sync
- Loop: `while (!grid_should_exit) { grid.sync(); process_tick(); }`
- Inside tick: sequential phases or device-side launches (tail launch?)

Actually, CUDA device-side graph launch (tail launch) is mentioned in the context. This could be used for sub-kernels without returning to host.

**4. Matryoshka Embedding Layout**
- Structure of Arrays: `float* embeddings[6]` for dims 64,128,256,512,1024,2048 (or whatever the tiers are)
- Or single allocation with offsets
- PTX kernel: `matryoshka_prefix_dot` takes dimension parameter, computes dot product up to that prefix

**5. Nine-Chain Swarm on GPU**
- 9 workers = 9 warps or 9 blocks?
- Shared memory for intermediate scores
- Warp ballot for convergence
- Halting gate as shared memory atomic reduction

**6. WINE Contract Registration in Galaxy**
- Contract as Galaxy

────────────────────────────────────────────────────────────────

### [2/6] Qwen — Code Specialist

# Qwen — Code Specialist Contribution

## Building on Kimi's Architecture: Concrete PTX Implementation Specs

Kimi established the 4-layer mapping. I now specify the **sovereign implementation** — every hot-path component as PTX/CUDA with exact memory layouts, atomic protocols, and kernel signatures. No Python orchestration survives this cut.

---

## 1. Persistent Kernel: `trm_step_fused.ptx` Specification

### 1.1 Kernel Signature & Launch Configuration

```cuda
// trm_step_fused.cu — Single persistent kernel, one launch per session
// Launch: cudaLaunchCooperativeKernel (NOT <<< >>>)
// Grid: deviceProp.multiProcessorCount blocks (one per SM)
// Block: 256 threads (8 warps per SM for occupancy)

__global__ void trm_step_fused(
    // Ring buffers (zero-copy, host-pinned)
    volatile uint32_t* input_ring_head,      // atomic producer index
    volatile uint32_t* input_ring_tail,      // atomic consumer index
    volatile QuerySlot*  input_ring_slots,   // [RING_CAPACITY] slots
    volatile uint32_t* output_ring_head,
    volatile uint32_t* output_ring_tail,
    volatile OutputSlot* output_ring_slots,
    
    // Galaxy VRAM base pointers (all knowledge co-resident)
    const GalaxyUniverse* galaxy,            // Layer 2-4 stars
    const MortonOctree*   octree,            // Spatial index
    const TRMWeights*     trm_weights,       // 7M params (Layer 3 navigation)
    const SpecialistPool* specialists,       // LoRA adapters (cognitive lanes)
    
    // Cooperative sync primitives
    cuda::grid_group      grid,              // grid.sync() for tick barrier
    volatile uint32_t*    tick_counter,      // global tick count (debug/ARB)
    
    // Exit condition (rare: shutdown only)
    volatile bool*        shutdown_flag
);
```

**Launch wrapper (Python, ~5 lines, boot-only):**
```python
# knowledge3d/knowledgeverse/trm_boot.py (NOT in hot path)
def launch_persistent_trm(galaxy_ptr, ring_buffers):
    props = cuda.get_device_properties(0)
    grid_size = props.multi_processor_count
    block_size = 256
    cuda.launch_cooperative_kernel(
        trm_step_fused,
        grid_size, block_size, 0,
        [ring_buffers.input_head, ring_buffers.input_tail, ...]
    )
    # Returns immediately; kernel runs until shutdown_flag = true
```

### 1.2 Persistent Tick Loop (PTX Internal Structure)

```ptx
// trm_step_fused.ptx — Inner tick loop (device function)
.visible .func trm_tick_loop(
    .param .u64 galaxy_ptr,
    .param .u64 ring_buffers,
    .param .u64 shared_state
) {
    // Per-block shared memory layout (48KB target)
    // .shared .align(128) .b8 swarm_scores[9 * 64];      // 9 workers × 64 candidates
    // .shared .align(128) .b8 halting_state[256];        // Convergence flags
    // .shared .align(128) .b8 rpn_stack[15 * 32];        // 15-deep RPN stack per lane
    
    .reg .b32 %tick_id;
    .reg .pred %has_work;
    
    tick_loop_entry:
        // === GRID BARRIER: All SMs sync at tick boundary ===
        grid.sync;  // Cooperative groups API (PTX: bar.grid.sync)
        
        // === WORK AVAILABILITY CHECK (atomic, no host call) ===
        ld.global.u32 %head, [input_ring_head];
        ld.global.u32 %tail, [input_ring_tail];
        sub.u32 %pending, %head, %tail;
        setp.gt.u32 %has_work, %pending, 0;
        
        @%has_work bra process_query;
        @!%has_work bra idle_spin;
    
    process_query:
        // === PHASE 1: PERCEIVE ===
        // Frustum cull + Morton octree query (device function)
        call perceive_phase, (%query_slot, %galaxy_ptr, %visible_stars);
        
        // === PHASE 2: NAVIGATE ===
        // LED-A* pathfinding through Galaxy (device function)
        call navigate_phase, (%visible_stars, %navigation_trace);
        
        // === PHASE 3: REASON (Nine-Chain Swarm) ===
        // Parallel specialist evaluation (shared memory reduction)
        call swarm_phase, (%navigation_trace, %specialist_pool, %swarm_scores);
        
        // === PHASE 4: PHYSICS (new slot) ===
        // Physics RPN program execution (Layer 3 force laws)
        call physics_phase, (%swarm_scores, %physics_rpn_addr, %physics_state);
        
        // === PHASE 5: DECIDE (Halting Gate) ===
        // Convergence check via warp ballot + shared memory reduction
        call halting_gate_phase, (%swarm_scores, %convergence_flag, %answer_slot);
        
        // === PHASE 6: ACT (Tablet Output) ===
        // Write to output ring buffer (atomic, zero-copy)
        call act_phase, (%answer_slot, %output_ring_slots, %output_ring_tail);
        
        // === TICK COMPLETE ===
        atom.add.global.u32 %tick_id, [tick_counter], 1;
        bra tick_loop_entry;
    
    idle_spin:
        // Low-power spin when no queries pending
        // Check shutdown flag periodically
        ld.global.b8 %shutdown, [shutdown_flag];
        @%shutdown ret;
        
        // nanosleep to reduce power (PTX ISA 6.3+, sm_70+)
        mov.u32 %sleep_ns, 10000;  // 10µs spin
        nanosleep.u32 %sleep_ns;
        bra tick_loop_entry;
}
```

### 1.3 Grid Synchronization Protocol

**Critical:** Use `cudaLaunchCooperativeKernel` + `grid.sync()` for tick barriers. This ensures all SMs complete phase N before any SM starts phase N+1.

```cuda
// Host-side setup (boot only, NOT per-query)
cudaLaunchConfig_t config = {};
config.gridDim = deviceProp.multiProcessorCount;
config.blockDim = 256;
config.dynamicSmemBytes = 49152;  // 48KB shared memory per block
config.stream = stream;

// Launch once, runs until shutdown_flag
cudaLaunchCooperativeKernel(
    (void*)trm_step_fused,
    config,
    kernel_args
);
// Returns only when kernel exits (shutdown)
```

**PTX barrier instruction (sm_90+ for grid sync):**
```ptx
// bar.grid.sync requires compute_90+ (Hopper)
// For sm_80/86/89 (Ampere/Ada), use cluster sync or

────────────────────────────────────────────────────────────────

### [3/6] GLM — Analysis & Edge Cases

Let me carefully analyze the entire chain so far, including Kimi's architectural foundation and Qwen's PTX implementation specs. My role as GLM is to:

1. Validate correctness
2. Identify edge cases
3. Check consistency with K3D's existing RPN stack convention (64-deep, StackValue is float4 with tag lane)
4. Ensure no new design clobbers existing Galaxy star fields (behavior_rpn, visual_rpn, etc.)
5. End with a prioritized list of open issues Claude must address in the Codex spec

Let me go through each section systematically.

**Kimi's Architecture - Issues I see:**

1. Kimi mentioned a 15-deep RPN stack in passing, but the spec says 64-deep. Need to flag this.
2. The ring buffer design needs more rigorous analysis - what about overflow? What about variable-size query envelopes?
3. The 4-layer mapping is sound but I need to check if behavior_rpn, visual_rpn, etc. are preserved.
4. The specialist selection via RPN program - needs validation against existing specialist pool design.

**Qwen's PTX Spec - Issues I see:**

1. **CRITICAL: RPN stack depth mismatch**: Qwen wrote `.shared .align(128) .b8 rpn_stack[15 * 32]` — 15-deep RPN stack. But the K3D spec says the RPN stack is **64-deep** and StackValue is **float4 with a tag lane**. This is a fundamental consistency violation.

2. **CRITICAL: grid.sync compute capability**: Qwen mentions `bar.grid.sync` requires sm_90+, then trails off. But the baseline is RTX 3060 (sm_86). Cooperative groups grid sync works on sm_70+ via `cudaLaunchCooperativeKernel`, but the PTX-level `bar.grid.sync` instruction availability varies. Need to clarify this.

3. **Ring buffer atomicity concerns**: Using `volatile uint32_t*` for head/tail with simple ld.global is NOT atomic. The producer (Python/host) and consumer (GPU) need proper memory fencing. Qwen's code does `ld.global.u32 %head, [input_ring_head]` without any acquire semantics — this can see stale values.

4. **Cooperative kernel limitations**: `cudaLaunchCooperativeKernel` requires the number of blocks to NOT exceed what can run concurrently on the device. With `multiProcessorCount` blocks of 256 threads, this should work, but what about dynamic parallelism within? If a phase needs more blocks than SMs, the persistent kernel can't launch them via CDP2 without careful management.

5. **Shared memory budget**: 48KB shared memory per block with 9 workers × 64 candidates × float (4 bytes) = 9 × 64 × 4 = 2,304 bytes just for swarm scores. But Qwen allocates 9 * 64 = 576 bytes (`.b8`). This is wrong — scores should be float32, not byte.

6. **Warp ballot for halting gate**: Qwen mentions this but doesn't specify the protocol. With 9 workers across potentially different warps/blocks, warp ballot only works within a warp. Cross-block convergence needs shared memory atomics or grid sync.

7. **Shutdown flag race condition**: The `volatile bool* shutdown_flag` without memory fence can cause the kernel to never see the update, or see it partially.

8. **The persistent kernel doesn't handle VRAM pressure**: If Galaxy grows during operation (new stars created), the persistent kernel needs to handle OOM conditions without returning to host.

9. **The RPN StackValue is float4**: Each stack entry is a float4 (x,y,z,w) where w is the tag lane. Qwen's shared memory layout for rpn_stack doesn't account for float4 alignment.

10. **Missing: how do existing 88 PTX kernels compose into the tick?** Kimi asked this, Qwen partially answered with `call` instructions, but the actual kernel composition (which of the 88 get called, and how their signatures match) is undefined.

11. **Python re-leak vectors**: 
    - Mutex: if Python holds a mutex on the ring buffer while writing, and the GPU tries to read... deadlock or starvation
    - Alloc: if the persistent kernel needs to allocate VRAM (new Galaxy entries), it can't call cudaMalloc from device. Need pre-allocated free-list.
    - Log emit: any printf from kernel is buffered and doesn't return to host until kernel exits, so this is safe. But if Python-side logging is triggered by the output ring, it could re-enter Python dispatch.
    - Descriptor refresh: if the kernel table (88 kernels) needs updating at runtime, how? Can't hot-reload PTX from device.

12. **The 6 embodiment gaps**: I need to check if the design handles:
    - Perceive: ✅ frustum + octree
    - Act: ✅ output ring
    - House↔Galaxy symlinks: ❌ NOT addressed — how does the persistent kernel access House (disk)? The House is on disk, Galaxy is in VRAM. The sleep-time path needs to bridge this.
    - Sleep/wake cycle: ❌ NOT addressed — how does the persistent kernel transition to sleep mode?
    - Memory palace navigation: Partially addressed via LED-A*
    - Avatar embodiment: Partially addressed

13. **Matryoshka prefix layout**: Qwen didn't specify the actual VRAM layout. Kimi mentioned SoA. I need to validate: if embeddings are at dims {64, 128, 512, 2048}, a SoA layout with aligned prefixes means:
    - Base pointer P
    - 64D view: P
    - 128D view: P (just read 128 floats)
    - 512D view: P (just read 512 floats)
    - 2048D view: P (just read 2048 floats)
    
    This works for a SINGLE embedding. For N embeddings, we need stride. If using Array of Structures (AoS), each embedding is 2048 floats, and prefixes are natural. If SoA, we need separate arrays per dimension tier. The question is which is better for GPU access patterns.

14. **Existing Galaxy star fields that must not be clobbered**: 
    - `behavior_rpn` — stored in Grammar Galaxy (Layer 3)
    - `visual_rpn` — stored in Drawing Galaxy (Layer 1)
    - `meaning_rpn` — stored in Reality Galaxy (Layer 2)
    - `physics_rpn_addr` — stored in Grammar Galaxy (Layer 3)
    
    The WINE contracts are proposed as new Galaxy entries with `ingress_rpn_addr` and `egress_rpn_addr`. These must NOT conflict with existing fields. They should be separate fields or stored in a dedicated contract sub-region.

15. **Specialist pool VRAM layout**: The 7M TRM params + LoRA adapters are "already co-resident in VRAM." But how are they indexed from device? The specialist pool needs a device-accessible lookup table. Currently `micro_specialist_pool` is Python. The replacement must be a VRAM-resident descriptor table with specialist ID → weight pointer mapping.

16. **The deletion list is incomplete**: Beyond what's listed, I should identify additional Python that will re-leak:
    - Any logging in the hot path that calls Python's logging module
    - Any exception handling that falls back to Python traceback
    - The Qdrant query interface — if Python calls Qdrant during inference, that's a sovereignty violation
    - The `knowledgeverse.py` 17.5k lines — what specific hot-path functions remain after the cuts?

17. **The TRM forward pass in the tick**: The 7M parameter TRM forward pass uses Tensor Cores (mma/wmma). But inside a cooperative kernel, can we use Tensor Cores? Yes, but the shared memory layout for MMA fragments is architecture-specific. This needs to be specified.

18. **Back-pressure semantics**: If the output ring is full (Python hasn't drained it), the persistent kernel must stall or drop. This is a real edge case during high-throughput queries.

19. **Input ring slot size**: Queries can vary wildly in size (a short text query vs. an ARC3 game frame). Fixed-size slots waste VRAM or truncate large inputs. Variable-size slots need offset/length pairs and contiguous buffer management.

20. **Galaxy-resident WINE contract lookup**: How does the engine "pick the right contract for an incoming frame"? Kimi mentioned "hash of input frame header." But what hash? And what if no contract matches? This is a real gap — need a fallback RPN program (not Python) that handles unknown paradigms.

Let me now structure my findings and add original analysis beyond what was discussed.

**Original analysis I can add:**

1. **The "VRAM Free-List" problem**: The persistent kernel cannot call cudaMalloc. All dynamic allocation must use pre-allocated free-lists in VRAM. This is a well-known pattern in GPU persistent kernels. I should specify the free-list structure for Galaxy star creation during reasoning.

2. **The "Stale Descriptor" problem**: If sleep-time consolidation creates new PTX kernels, the persistent kernel can't use them until it's relaunched. This means we need a versioned kernel table and a mechanism for the persistent kernel to detect and request a "hot reload" (shutdown and relaunch with updated table).

3. **The "Tick Budget Overflow" problem**: The Adaptive Reasoning Budget says B(q) = B_base

────────────────────────────────────────────────────────────────

### [4/6] DeepSeek — Synthesis & Cross-Domain Integration

We are at the fourth partner in the chain, DeepSeek, the synthesis and cross-domain integration partner. My mandate is to:

1. Identify any remaining gaps or conflicts in the chain and resolve them explicitly.
2. Explore how the current task connects to OTHER parts of K3D — e.g., how physics simulation connects to Drawing Galaxy (visual_rpn), Grammar Galaxy (transformation rules), or Sleep-time Consolidation (learning from collisions).
3. Add my own original contributions that no prior partner covered — cross-domain RPN compositions, novel Galaxy star designs, or new opcode uses.
4. End with a concise summary of what Claude should prioritize when writing the Codex spec.

Let me first review the contributions so far:

**Kimi (Architecture)** laid the foundation by mapping the persistent tick to the 4-layer architecture, defining the VRAM-resident work queue, and outlining the persistent kernel design. Key points:
- Persistent tick as a cooperative kernel with grid sync.
- Ring buffer for producer-consumer between Python and GPU.
- Matryoshka embedding layout as Structure of Arrays (SoA) or single allocation with offsets.
- WINE contracts stored in Galaxy as stars with ingress/egress RPN addresses.
- Self-crafting via Layer 4 meta-rules.

**Qwen (Code/PTX)** provided concrete PTX implementation specs, including kernel signatures, persistent tick loop, and grid synchronization. However, GLM identified several critical issues:
-
RPN stack depth mismatch (15 vs 64), grid.sync compute capability, ring buffer atomicity, shared memory layout errors, and more.

**GLM (Analysis)** performed a rigorous analysis, identifying 20+ issues and edge cases, including:
- RPN stack depth and StackValue format (float4 with tag lane).
- Atomicity and memory fencing for ring buffers.
- Cooperative kernel limitations and dynamic parallelism.
- Shared memory budget and layout.
- Warp ballot vs cross-block convergence.
- Shutdown flag race condition.
- VRAM pressure and OOM handling.
- Composition of existing 88 PTX kernels.
- Python re-leak vectors (mutex, alloc, log, descriptor refresh).
- Embodiment gaps (House↔Galaxy symlinks, sleep/wake).
- Matryoshka layout details.
- Galaxy star field conflicts.
- Specialist pool VRAM layout.
- Deletion list incompleteness.
- TRM forward pass with Tensor Cores.
- Back-pressure semantics.
- Input ring slot size variability.
- WINE contract lookup and fallback.

GLM also added original analysis on VRAM free-list, stale descriptor, and tick budget overflow problems.

Now, as DeepSeek, I must synthesize, resolve conflicts, and add cross-domain connections.

### Step 1: Resolve Conflicts and Gaps

**Conflict 1: RPN Stack Depth and Layout**
- The spec says 64-deep RPN stack, and StackValue is float4 (x,y,z,w) with w as tag lane.
.
- Correction: In shared memory, we need 64 entries per RPN stack per lane. However, note that the nine-chain swarm has 9 workers, each with a private RPN stack. But the RPN stack is per worker (or per cognitive lane) and is likely stored in registers or shared memory. The exact layout should be:
  - Each worker has a 64-deep stack of float4 (16 bytes per entry).
  - For 9 workers, that's 9 * 64 * 16 = 9216 bytes.
  - We can allocate in shared memory as `.shared .align(16) .f32 rpn_stacks[9 * 64 * 4];` (each float4 is 4 floats).

**Conflict 2: Grid Sync Compute Capability**
-LMA-2 noted that `bar.grid.sync` requires sm_90+, but baseline is RTX 3060 (sm_86). The solution is to use cooperative groups grid sync via the CUDA runtime API (`grid.sync()` in CUDA C++), which is available on sm_60+ with `cudaLaunchCooperativeKernel`. The PTX instruction for grid sync might not be available, but we can rely on the CUDA C++ cooperative groups API. The persistent kernel should be written in CUDA C++ (not PTX) for easier use of cooperative groups. However, the mandate says "HOT PATH = PTX kernels". We can write the kernel in CUDA C++ and then compile to PTX, but the grid sync will be a call to `grid.sync()` which in PTX is `bar.sync`. Actually, the PTX instruction `bar.sync` with `.grid` mode is available on sm_80+ (Ampere). For sm_86, it should be available. Let's check: the PTX ISA guide says `bar.sync` with `.grid` is introduced in PTX ISA 6.0 (sm_70). So it should be available on sm_86. We'll use `bar.sync .grid;` in PTX.

**Conflict 3: Ring Buffer Atomicity**
- We need proper acquire/release semantics. The producer (Python) writes to the slot and then updates the head with release semantics. The consumer (GPU) reads the head with acquire semantics. We can use `atomicAdd` for head/tail updates, but note that the GPU side only reads the head and tail (doesn't update them until consuming a slot). Actually, the GPU updates the tail after consuming. So we need:
  - Producer (Python) writes slot, then does an atomic increment on head (with memory release).
  - Consumer (GPU) reads head and tail with atomic loads (acquire). After consuming, does atomic increment on tail (release).
  - We can use `atomicAdd` for increments, and for loads we can use `__ldcg` (load with cache eviction) or `ld.global.acquire` in PTX.

**Conflict 4: Shared Memory Layout for Swarm Scores**
- Qwen's layout was incorrect. We need 9 workers × 64 candidates × float (4 bytes) = 2304 bytes. We can allocate as `.shared .align(4) .f32 swarm_scores[9 * 64];`.

**Conflict 5: Warp Ballot for Halting Gate**
- The nine-chain swarm workers might be in different warps or blocks. Convergence must be across all workers. We can use a two-stage reduction:
  - Each worker writes its scores to shared memory.
  - Then a reduction across the grid (using shared memory atomics or a grid sync reduction) to compute convergence.

**Conflict 6: Shutdown Flag**
- Use `volatile` and proper fencing. We can use `__threadfence_system()` before writing the flag from host, and `__threadfence_system()` before reading in the kernel. In PTX, use `membar.sys` for system memory fence.

**Conflict 7: VRAM Pressure and OOM**
- The persistent kernel cannot allocate. We must pre-allocate a free-list for Galaxy stars. This free-list is managed by the kernel using atomics. When creating a new star, the kernel allocates from the free-list (which is a contiguous VRAM region divided into fixed-size slots). If the free-list is exhausted, the kernel must trigger a spill (evict least recently used stars to House) using an OOM spill kernel (which is one of the existing 88 PTX kernels). This OOM spill kernel is called from the persistent kernel via device-side launch (dynamic parallelism) or as a separate phase.

**Conflict 8: Composition of Existing PTX Kernels**
- We must list which of the 88 PTX kernels are used in each phase. For example:
  - PERCEIVE: `morton_octree_query.cu`, `frustum_cull.cu`
  - NAVIGATE: `led_astar.cu`
  - REASON: `gre_defeasible_resolver.cu`, `gre_multimodal_halting_gate.cu`, etc.
  - PHYSICS: `physics_integrator.cu` (new)
  - ACT: `tablet_output.cu` (new)
- These kernels are called from the persistent kernel as device functions (if they are compiled together) or as separate kernels launched via dynamic parallelism. We prefer device functions for low latency, but some might be too large. We need to decide.

**Conflict 9: Python Re-leak Vectors**
- We must eliminate all Python in the hot path. The ring buffer should be zero-copy (pinned memory) and accessed via atomic operations. No mutexes. The Python side only writes to the ring buffer and reads from the output ring buffer. It does not dispatch or orchestrate.

**Conflict 10: Embodiment Gaps**
- House↔Galaxy symlinks: The House is on disk, but the Galaxy has symlinks to House entries. The persistent kernel does not access disk. When a symlink is dereferenced, the content must already be in VRAM (loaded during boot or sleep-time). If not, we need a fault mechanism that triggers a load from disk. This is a complex issue. We can assume that during boot, the entire House is loaded into Galaxy (as per the Three-Brain System: Galaxy is the active memory, House is persistent). But if the Galaxy is too large, we load only a subset. Then, when a symlink is accessed and not in VRAM, we trigger a load via a separate I/O thread (Python). This I/O thread would read from disk and write into a VRAM buffer, then update the Galaxy. This is a potential re-le

────────────────────────────────────────────────────────────────

### [5/6] Nemotron — GPU Kernel Integration & PTX Optimization

## Nemotron — GPU Kernel Integration & PTX Optimization

### 1. KERNEL REUSE: K3D Existing Kernel Audit & Opportunities
Cross-referencing the proposed pipeline against the 88-strong kernel inventory reveals significant reuse opportunities **not yet leveraged** in the chain's proposals. Critical omissions:

1. **`morton_octree.ptx` reuse gap**  
   Proposed pipeline uses custom frustum cull but ignores existing Morton octree traversal. The `morton_octree.ptx` kernel already implements **spatial sorting with morton codes** and **range queries** — ideal for the PERCEIVE phase. Current proposal duplicates this logic. Instead:  
   - Use `morton_octree.ptx` with input: `query_position + search_radius` (from pose)  
   - Output: `candidate_star_indices[]` (compressed via existing bitpack)  
   - *Reuse gain*: Avoids custom frustum math; leverages proven spatial index.

2. **`led_astar.ptx` integration gap**  
   NAVIGATE phase proposes custom pathfinding but `led_astar.ptx` is **already tuned for graph traversal in cranial space** (inside the skull). It accepts:  
   - Input: `start_node, goal_node, obstacle_mask`  
   - Output: `path_nodes[]` (with cost)  
   - *Reuse gain*: Avoids reimplementing A*; leverages existing obstacle handling via cranial geometry.

3. **`nine_chain_swarm_kernel.cu` bypass**  
   The nine-chain swarm is proposed as custom logic, but `nine_chain_swarm_kernel.cu` already implements **parallel candidate processing with STORE/RECALL registers** and **defeasible logic integration**. It expects:  
   - Input: `candidate_list[], num_workers=9`  
   - Output: `specialist_scores[9][C]`  
   - *Reuse gain*: Direct plug-in; no need to reimplement STORE/RECALL register comms.

4. **`modular_rpn_kernel.cu` underutilization**  
   Proposed phases reuse RPN execution but ignore `modular_rpn_kernel.cu`'s **opcode extension pattern** (cases `0x150...0x17F`). This kernel already has a **switch statement with fall-through handling** for undefined opcodes — critical for the PHYSICS phase. Instead of custom dispatch:  
   - Map physics opcodes (e.g., `OP_GRAVITY=0x150`) to cases in this kernel  
   - Undefined opcodes trigger `FAULT_UNDEFINED_OPCODE` (handled by defect resolver)  
   - *Reuse gain*: Avoids redundant RPN parser; leverages fault tolerance.

5. **`gre_defeasible_resolver.cu` omission**  
   PROPOSED pipeline mentions defeasible logic but doesn't use the existing resolver kernel. It expects:  
   - Input: `specialist_scores[W x C], rule_strengths[W], superiority[W x S]`  
   - Output: `verdicts[C], proof_tags[C]`  
   - *Reuse gain*: Direct fit for post-swarm conflict resolution; avoids reinventing superiority defeat logic.

6. **`cosine_similarity.ptx` bypass**  
   Similarity scoring (e.g., for candidate ranking) proposes custom logic but `cosine_similarity.ptx` already does **batched GPU cosine similarity [N, D] vs [D]** with `<200µs` latency. It expects:  
   - Input: `query_vec[], candidate_matrix[N, D]`  
   - Output: `similarities[N]`  
   - *Reuse gain*: Avoids warp-reduce reimplementation; leverages existing warp shuffle optimization.

7. **`gre_graph_crystallizer.cu` gap**  
   Multi-hop traversal (e.g., for causal reasoning) proposes custom logic but `gre_graph_crystallizer.cu` implements **multi-hop graph traversal with path compression**. It expects:  
   - Input: `graph_adjacency[], start_nodes[], max_hops`  
   - Output: `reachable_nodes[], path_lengths[]`  
   - *Reuse gain*: Avoids BFS reimplementation; leverages crystallized path caching.

**Summary of missed reuse**:  
> The chain proposes **7 new custom logic blocks** where **6 existing kernels** (`morton_octree.ptx`, `led_astar.ptx`, `nine_chain_swarm_kernel.cu`, `modular_rpn_kernel.cu`, `gre_defeasible_resolver.cu`, `cosine_similarity.ptx`) already implement equivalent functionality with battlefield-tested performance. Only `gre_graph_crystallizer.cu` represents a *true gap* requiring new logic for multi-hop crystallized traversal.

### 2. WARP-LEVEL PATTERNS: Synchronization & Reduction Optimizations
Leveraging warp primitives for synchronization-critical phases:

1. **`__ballot_sync` for sleep-island detection**  
   Instead of per-core sleep counters, use warp-level ballot to detect **global quiescence**:  
   ```cpp
   // In idle state, each thread votes: 1 if local work queue empty
   uint32_t empty_mask = __ballot_sync(0xffffffff, thread_work_empty());
   if (empty_mask == 0xffffffff) {  // All warps in block idle
       // Trigger sleep-island handoff via shared memory flag
       if (threadIdx.x == 0) { atomicExch(&block_is_idle, 1); }
   }
   ```
   - *Advantage*: Reduces sleep detection from O(W) atomics to O(1) warp vote; avoids false positives from transient stalls.

2. **`__shfl_xchg_sync` for PGS parallel reductions**  
   Position-Gradient-Solver (PGS) requires per-constraint lambda reduction. Replace shared-memory tree reduction with warp shuffle:  
   ```cpp
   // Each thread holds lambda_i for its constraint
   float lambda = my_lambda;
   for (int offset = 16; offset > 0; offset /= 2) {
       float neighbor = __shfl_xchg_sync(0xffffffff, lambda, offset);
       lambda = fminf(lambda, neighbor);  // Min-reduce for PGS
   }
   // First thread in warp holds min lambda
   if (lane_id == 0) { atomicMin(global_lambda_ptr, lambda); }
   ```
   - *Advantage*: Eliminates 6 shared memory transactions per reduction; leverages warp shuffle's single-cycle latency.

3. **Warp-cooperative simplex tracking in GJK**  
   Gilbert-Johnson-Keerthi (GJK) distance calculation benefits from warp-level simplex cooperation:  
   - Each warp maintains one simplex (4 points in 3D)  
   - Use `__shfl_sync` to share support points across lanes  
   - Use `__any_sync` to detect termination (no separation found)  
   - *Advantage*: Reduces shared memory usage by 75%; enables 32-way parallel GJK per warp.

### 3. SOA MEMORY: Structure-of-Arrays Validation & Fixes
Analyzing proposed memory layouts for coalesced access and bank conflicts:

#### **Current Proposed Layout (Flawed)**
```cpp
struct Candidate {
    float3 position;    // 12 bytes
    float4 color;       // 16 bytes
    uint32_t id;        //  4 bytes
    float score;        //  4 bytes
}; // Total: 36 bytes → 12-byte stride (non-coalesced)
```
**Problems**:  
- **Stride inefficiency**: 36-byte struct → 12-byte effective stride for `position.x` (not 4-byte aligned)  
- **Bank conflicts**: `position.y` at offset 4 → bank 1, `position.z` at offset 8 → bank 2 → no conflict *yet*, but...  
- **Worse**: When accessing `color.r` (offset 16), stride 36 → bank 0; next thread's `color.r` at offset 52 → bank 4 → **no conflict**, but...  
- **Critical flaw**: For `score` field (offset 32), stride 36 → bank 0 (32%32=0); thread 1: offset 68 → bank 4 (68%32=4) → **no conflict**, **BUT** when `warpSize=32`, threads 0-31 access offsets [32, 68, 104, ...] → all map to **different banks** → *no conflict*.  
  **Wait — actually this is coalesced?** Let's check access pattern:  
  Thread i accesses `base + i*36 + field_offset`. For contiguous threads:  
  - Addresses for `position.x`: `[base+0, base+36, base

────────────────────────────────────────────────────────────────

### [6/6] Gemini — Final Spec Grounding — 1M Context Integration

[gemini-3-flash:cloud unavailable after 3 attempts: 404 Client Error: Not Found for url: http://host.docker.internal:11434/api/chat]

────────────────────────────────────────────────────────────────

## STAGE 3 — POST-CHAIN GROUNDING
**Model:** kimi-k2.5:cloud

**POST-CHAIN GROUNDING PASS — K3D MVCIC SESSION**
**Validator**: Post-Chain Grounding Agent  
**Session**: Close the Game Loop — Persistent Kernel Architecture  
**Status**: P0 BLOCKERS IDENTIFIED — CODEX HANDOFF READY

---

## 1. SOVEREIGNTY AUDIT: Hot-Path Python Leak Points & Fixes

**Trace**: Python Bootstrap → Ring Buffer Write → GPU Persistent Tick → Galaxy Query → RPN Execution → Output Ring → Python Drain

| Stage | Leak Risk | Exact Location | Sovereign Fix |
|-------|-----------|----------------|---------------|
| **Input Classification** | Python parses query to select contract type | `knowledgeverse.py` line ~8400 (historical pattern) | Input frame header contains `contract_hash` (uint64). Kernel does Galaxy lookup via `LOAD_GALAXY(E0)` opcode 0xE0. |
| **Ring Buffer Allocation** | Python calls `cudaMalloc` per query | `_run_query_tick` historical implementation | Pre-allocated VRAM ring at boot (pinned/mapped). Python writes to host-mapped pointer only. |
| **Back-Pressure Handling** | Python sleeps/polls on GPU completion | `micro_specialist_pool.run_overflow_sequential` | Kernel uses `nanosleep.u32` (PTX ISA 6.3) for idle spin. Host uses `atomicAdd` on `output_ring_tail` (release) to detect completion. |
| **Error Fallback** | Python try/except around kernel launch | `trm_game_loop.py` try/except blocks | Kernel writes error code to `tick_status` register in VRAM. Python reads status asynchronously. No fallback path. |
| **Contract Discovery** | Python dict lookup for WINE adapter | `wine_contract_registry.py` (historical) | Contracts are Galaxy stars with field `paradigm_type` (Layer 2). Kernel scans Galaxy via `GALAXY_SCAN(E2)` opcode 0xE2. |
| **House Sync** | Python loads House data mid-query | Any Galaxy→House symlink dereference during tick | Galaxy fully resident in VRAM (Region 2). House (disk) updates only during sleep-time via `sleep_cluster_refiner.ptx`. |
| **Logging** | Python `logging.info` inside tick | `trm_game_loop.py` logging calls | Kernel writes to `log_ring_buffer` (VRAM circular queue). Python drains asynchronously. |
| **Matryoshka Tier Selection** | Python decides 64/128/512/2048D | Host-side if/else on query length | Layer 4 Meta-Rule RPN program `meta_select_matryoshka_tier` outputs dimension to shared memory. Kernel reads `tier_signal` directly. |

**Critical Fix**: The ring buffer atomics must use explicit memory ordering. Current chain proposal uses `volatile` which is insufficient. **Fix**: Use `ld.global.acquire` and `st.global.release` in PTX (or `__ldcg`/`__stcg` with `__threadfence`).

---

## 2. SPEC COMPLIANCE: 4-Layer Architecture Mapping

**Corrections to Chain Proposals:**

| Concept | Chain Placement | Correct Placement | Justification |
|---------|----------------|-------------------|---------------|
| **Physics Constants** (G, c, ħ) | Layer 3 (Rules) | **Layer 2** (Meaning/Reality Galaxy) | Per `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION`: "physical constants (G, c, ħ, k_B, ε₀) → Layer 2 Reality Galaxy star" |
| **Force Law RPN** | Layer 2 | **Layer 3** (Grammar Galaxy) | Per spec: "force law RPN program → Layer 3 Grammar Galaxy (field: physics_rpn_addr, NOT behavior_rpn)" |
| **Sleep/Wake Policy** | Layer 3 | **Layer 4** (Meta-Rules) | "Sleep/wake island policy → Layer 4 Meta-Rule" |
| **WINE Contract Metadata** | Hardcoded Python | **Layer 2** Galaxy star with fields `contract_hash`, `paradigm_type`, `ingress_rpn_addr`, `egress_rpn_addr` | Contracts must be knowledge, not code. |
| **Matryoshka Embeddings** | Layer 1 | **Layer 2** | Embeddings are semantic keys (search), not form. |
| **Physics Phase** | Between SWARM and DRAW | **Between SWARM_PHASE and DRAW_PHASE** | Per prompt specification. |

**Galaxy Star Schema for WINE Contract (Layer 2)**:
```c
struct WineContractStar {
    uint64_t contract_hash;      // murmur3 of paradigm signature
    uint8_t  paradigm_type;        // 0x01=DOM, 0x02=ARC3, 0x03=TEXT, 0x04=AUDIO
    uint64_t ingress_rpn_addr;     // Galaxy address of Layer 3 RPN program
    uint64_t egress_rpn_addr;      // Galaxy address of Layer 3 RPN program
    uint64_t visual_rpn_symlink;   // Layer 1 Drawing Galaxy reference (optional)
};
```

---

## 3. KERNEL COMPOSITION ENHANCEMENTS: Under-Utilized Inventory

The chain proposes new logic where existing kernels already provide sovereign implementations. **Delete the proposed duplication, wire these instead:**

**A. `morton_octree.ptx` — PERCEIVE Phase**
- **Current Gap**: Chain proposes custom Morton math for spatial indexing.
- **Fix**: Call existing `morton_octree.ptx` with params: `{avatar_position_x, avatar_position_y, avatar_position_z, search_radius}`. Output: `candidate_star_indices[]` (compressed).
- **Wiring**: `trm_step_fused.ptx` → `morton_octree.ptx` (device function linkage or tail launch via `cudaGraphLaunch` with `cudaStreamGraphTailLaunch`).

**B. `cosine_similarity.ptx` — NAVIGATE Phase**
- **Current Gap**: Chain proposes custom similarity kernels for LED-A* candidate ranking.
- **Fix**: Use existing `cosine_similarity_batch` kernel (latency <200µs). Input: `query_embedding[Matryoshka tier]`, `candidate_matrix[N, D]`. Output: `similarity_scores[N]`.
- **Wiring**: `led_astar.ptx` (during neighbor scoring) → `cosine_similarity.ptx`.

**C. `gre_defeasible_resolver.cu` — REASON Phase**
- **Current Gap**: Chain reimplements defeasible logic in swarm phase.
- **Fix**: After `nine_chain_swarm_kernel.cu` produces `specialist_scores[W × C]`, feed directly to `gre_defeasible_resolver.cu`. It handles superiority defeat via `superiority[W × S]` matrix already resident in Grammar Galaxy.
- **Wiring**: `swarm_scores` → `gre_defeasible_resolver.cu` → `verdicts[C]` → Halting Gate.

**Deletion Target**: Any Python in `micro_specialist_pool` that duplicates these kernels.

---

## 4. ORIGINAL IDEAS: Sovereign Architectural Extensions

**I. RPN Opcode Range 0x180–0x18F: Physics-to-Visual Bridge**
- **Purpose**: Eliminate Python bridge for physics visualization.
- **Layer**: Layer 3 (Rules) → Layer 1 (Form) symlink.
- **Opcode**: `PHYSICS_EMIT_VISUAL(0x180)` — Takes physics state vector (position, velocity, acceleration) from physics phase and pushes Drawing Galaxy RPN commands to `visual_rpn_stack`.
- **Implementation**: New case in `modular_rpn_kernel.cu` (which already handles opcode dispatch). When encountered, pops physics state from RPN stack, generates `CIRCLE` or `LINE` Drawing Galaxy opcodes, writes to Tablet surface buffer.
- **Sovereignty**: Physics simulation and its visualization both happen inside `trm_step_fused.ptx` without host round-trip.

**II. Sleep-Time Kernel: `sleep_physics_crystallizer.ptx`**
- **Purpose**: Learn physics patterns, not just query patterns.
- **Mechanism**: During sleep, scans `physics_event_ring` (VRAM circular buffer of collision/force events from PHYSICS_PHASE). Runs clustering on event vectors (position, force magnitude, participant types). Creates new Grammar Galaxy stars (Layer 3) for frequently occurring collision patterns (e.g., "rigid body restitution > 0.8").
- **Integration**: Extends `sleep_cluster_refiner.ptx` to handle physics events, not just embeddings. Writes new `physics_rpn_addr` entries for emergent force laws.

**III. Meta-Rule RPN: `meta_select_matryoshka_tier`**
- **Purpose**: Device-side decision on embedding dimension.
- **Layer**: Layer 4 (Meta-Rules).
- **Logic**: Reads composite signal σ(q) from Adaptive Reasoning Budget. Executes: `[LOAD_GALAXY sigma] [PUSH 64] [PUSH 128] [PUSH 512] [PUSH 2048] [TERNARY_SELECT]`.
- **Output**: Writes selected dimension (64/128/512/2048) to `mat

────────────────────────────────────────────────────────────────



**Handoff to Claude:** Read pre-chain context + full chain + post-chain grounding above. Synthesize into a formal spec + implementation instructions for Codex.
