/*
 * persistent_tick.cu — Persistent cooperative TRM game-loop kernel
 *
 * Spec: CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md
 *   §2.2  Kernel signature (authoritative)
 *   §2.4  Inner loop phase order
 *   §2.5  Grid sync on sm_86 via cooperative groups (NOT bar.grid.sync PTX)
 *   §2.6  Ring atomics: ring_atomics.cuh for host-pinned I/O rings
 *   §2.7  Shared memory layout: 9×64 float4 RPN stacks, 9×64 swarm scores
 * Audit: AGENT1_KERNEL_AUDIT_04.18.2026.md
 *   §6    Composition order, phase→kernel wiring table
 *   §7    Open questions resolved in comments below
 *
 * Architecture:
 *   One cudaLaunchCooperativeKernel at boot starts trm_step_fused on all SMs.
 *   Inside: while(!shutdown) { grid.sync(); phases...; tick_counter++; }
 *   Python never launches a kernel per query. Python = boot + I/O only.
 *
 * Phase stubs:
 *   Phase bodies currently call log_emit(PHASE_STUB) and proceed. This is
 *   intentional for the first cut: the kernel must compile and the ring must
 *   cycle. Wiring to existing kernels (audit §6 table) is the NEXT cut.
 *   See PHASE WIRING NOTE in each phase below for the exact wiring target.
 *
 * Grid sync on sm_86 (audit §7, question 2):
 *   cudaLaunchCooperativeKernel + cuda::grid_group::sync() is available from
 *   sm_60 upward via the cooperative groups API. We do NOT use the PTX
 *   instruction bar.grid.sync (requires sm_90+ Hopper clusters). The driver
 *   JIT-compiles grid.sync() to the appropriate PTX for sm_86.
 *
 * Dynamic parallelism (audit §7, question 3 — gre_defeasible_resolver etc.):
 *   For this cut, __global__ kernels that need to run inside the tick are
 *   declared as forward __device__ stubs. Real wiring will either:
 *   (a) extract core logic to __device__ functions and call inline, OR
 *   (b) use CDPv2 (available on sm_86) with careful depth management.
 *   The choice is deferred to the next cut; stubs are correct for now.
 *
 * Opaque struct forward declarations:
 *   GalaxyUniverse, MortonOctree, TRMWeights, SpecialistPool are new types
 *   defined by this persistent-tick boundary (spec §2.2). They do NOT exist
 *   in the repo yet — this file forward-declares them as incomplete types.
 *   The next cut will define them in a shared header in ptx_kernels/.
 *   The existing `const unsigned char* galaxy_table` pattern (device_functions.cuh)
 *   will be wrapped behind GalaxyUniverse for the persistent tick.
 *
 * QuerySlot / OutputSlot sizes:
 *   GPU_TASK_INPUT_SLOT_BYTES  = 2688 (from device_functions.cuh)
 *   GPU_TASK_OUTPUT_SLOT_BYTES = 640  (from device_functions.cuh)
 *   Ring capacity = 256 slots (power of 2 for modulo masking).
 *
 * Target: sm_86 (RTX 3070). sm_86 is forward-compatible with sm_80 PTX.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>

#include "ring_atomics.cuh"
#include "log_ring.cu"
#include "vram_freelist.cu"
#include "wine_contract_scan.cu"
#include "matryoshka_prefix_dot.cu"

namespace cg = cooperative_groups;

/* -------------------------------------------------------------------------
 * Ring sizes (power-of-2 for mask-based modulo)
 * ---------------------------------------------------------------------- */
#define K3D_INPUT_RING_CAPACITY  256u
#define K3D_OUTPUT_RING_CAPACITY 256u
#define K3D_LOG_RING_CAPACITY    4096u

#define K3D_INPUT_SLOT_BYTES  2688u  /* GPU_TASK_INPUT_SLOT_BYTES  */
#define K3D_OUTPUT_SLOT_BYTES 640u   /* GPU_TASK_OUTPUT_SLOT_BYTES */

/* -------------------------------------------------------------------------
 * tick_status codes written to the volatile tick_status register.
 * Python reads this asynchronously. No try/except. No fallback.
 * ---------------------------------------------------------------------- */
#define K3D_TICK_STATUS_OK            0u
#define K3D_TICK_STATUS_IDLE          1u
#define K3D_TICK_STATUS_FREELIST_OOM  2u
#define K3D_TICK_STATUS_GALAXY_MISS   3u
#define K3D_TICK_STATUS_OUTPUT_FULL   4u
#define K3D_TICK_STATUS_NEEDS_HOUSE_LOAD 5u  /* symlink to non-resident House star */

/* -------------------------------------------------------------------------
 * Shared memory layout (per block, 48 KB budget) — spec §2.7
 *
 *   rpn_stacks[9][64][4]  — 9 swarm lanes × 64-deep stack × float4 (x,y,z,w=tag)
 *                           = 9 × 64 × 16 bytes = 9216 bytes
 *   swarm_scores[9][64]   — 9 workers × 64 candidate scores (float)
 *                           = 9 × 64 × 4 bytes  = 2304 bytes
 *   halting_state[32]     — 32 uint32_t words for convergence flags
 *                           = 128 bytes
 *   tier_signal           — 1 uint32_t for Matryoshka tier from meta-rule RPN
 *                           = 4 bytes
 *   work_desc[16]         — shared work descriptor: star_id, query ptr, paradigm
 *                           = 64 bytes (16 × uint32_t)
 *   scratch[rest]         — aligned scratch for phase sub-computations
 *
 *   Total reserved: 9216 + 2304 + 128 + 4 + 64 = 11716 bytes
 *   Scratch available: 49152 - 11716 = 37436 bytes
 * ---------------------------------------------------------------------- */
struct K3DSharedMem {
    float    rpn_stacks[9][64][4];   /* 9216 bytes — float4 per StackValue */
    float    swarm_scores[9][64];    /* 2304 bytes */
    uint32_t halting_state[32];      /*  128 bytes */
    uint32_t tier_signal;            /*    4 bytes */
    uint32_t work_desc[16];          /*   64 bytes */
    uint8_t  scratch[37436];         /* remainder  */
};

static_assert(sizeof(K3DSharedMem) <= 49152,
              "K3DSharedMem exceeds 48 KB shared memory budget");

/* -------------------------------------------------------------------------
 * Opaque types for the persistent tick boundary (to be fully defined next cut)
 * ---------------------------------------------------------------------- */
struct GalaxyUniverse;  /* wraps: const unsigned char* galaxy_table + star_count */
struct MortonOctree;    /* wraps: VRAM-resident sorted Morton array + node IDs   */
struct TRMWeights;      /* wraps: 7M-param TRM weight tensors in VRAM            */
struct SpecialistPool;  /* wraps: LoRA adapter descriptor table + weight slices  */

/* Slot types — raw byte arrays matching GPU_TASK_*_SLOT_BYTES */
struct alignas(16) QuerySlot  { uint8_t data[K3D_INPUT_SLOT_BYTES];  };
struct alignas(16) OutputSlot { uint8_t data[K3D_OUTPUT_SLOT_BYTES]; };

/* -------------------------------------------------------------------------
 * Phase device-function forward declarations.
 * Bodies are stubs for this cut; each emits log code K3D_LOG_PHASE_STUB.
 * PHASE WIRING NOTE in each body cites the audit §6 kernel to call next.
 * ---------------------------------------------------------------------- */
__device__ void perceive_phase(
    const QuerySlot* slot,
    const GalaxyUniverse* galaxy,
    const MortonOctree* octree,
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id);

__device__ void navigate_phase(
    const GalaxyUniverse* galaxy,
    const TRMWeights* weights,
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id);

__device__ void reason_phase(
    const GalaxyUniverse* galaxy,
    const SpecialistPool* specialists,
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id);

__device__ void physics_phase(
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id);

__device__ void decide_phase(
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id);

__device__ void act_phase(
    volatile uint32_t* output_ring_head,
    volatile uint32_t* output_ring_tail,
    OutputSlot* output_ring_slots,
    volatile uint32_t* tick_status,
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id);

/* =========================================================================
 * trm_step_fused — the persistent cooperative TRM kernel
 *
 * Spec §2.2 authoritative signature.
 * Launch: cudaLaunchCooperativeKernel from trm_boot.py (boot only, once).
 * Grid: one block per SM (RTX 3070 has 46 SMs). Block: 256 threads.
 * Shared: 48 KB (K3DSharedMem).
 *
 * Lifecycle:
 *   1. Boot: trm_boot.py calls cudaLaunchCooperativeKernel → kernel starts.
 *   2. Run:  while(!shutdown) { grid.sync; poll input ring; 6 phases; tick++ }
 *   3. Exit: host sets *shutdown_flag = 1 with membar.sys; kernel sees it,
 *            runs to grid.sync boundary, all blocks exit together.
 * ========================================================================= */
__global__ void trm_step_fused(
    /* Zero-copy rings (host-pinned, device-mapped) — spec §2.2 */
    volatile uint32_t* input_ring_head,   /* atomic producer idx, host write  */
    volatile uint32_t* input_ring_tail,   /* atomic consumer idx, device write */
    QuerySlot*         input_ring_slots,  /* K3D_INPUT_RING_CAPACITY slots     */
    volatile uint32_t* output_ring_head,  /* atomic producer idx, device write */
    volatile uint32_t* output_ring_tail,  /* atomic consumer idx, host write   */
    OutputSlot*        output_ring_slots, /* K3D_OUTPUT_RING_CAPACITY slots    */
    volatile uint32_t* log_ring_head,     /* device log emit (VRAM ring)       */
    LogRecord*         log_ring_slots,    /* K3D_LOG_RING_CAPACITY records     */

    /* Galaxy + indices (VRAM-resident) */
    const GalaxyUniverse* galaxy,
    const MortonOctree*   octree,
    const TRMWeights*     trm_weights,
    const SpecialistPool* specialists,

    /* VRAM free-list for in-tick Galaxy star creation */
    VramFreelist* freelist,

    /* Control */
    volatile uint32_t* tick_counter,
    volatile uint32_t* tick_status,  /* kernel-to-host error code, no try/except */
    volatile uint32_t* shutdown_flag /* membar.sys fenced: spec §2.6 */
)
{
    /* ------------------------------------------------------------------
     * Cooperative groups: grid-scope synchronisation (spec §2.5)
     * cudaLaunchCooperativeKernel is required for grid.sync() to work.
     * On sm_86 this uses a device-resident barrier, not PTX bar.grid.sync
     * (the latter requires sm_90+ Hopper). The cooperative groups API
     * handles the target-specific PTX emission at JIT time.
     * ------------------------------------------------------------------ */
    cg::grid_group grid = cg::this_grid();

    /* ------------------------------------------------------------------
     * Shared memory — typed overlay on dynamic smem
     * Using the struct overlay: declare one __shared__ array and cast.
     * ------------------------------------------------------------------ */
    __shared__ K3DSharedMem smem;

    /* Thread identity */
    const bool is_block0  = (blockIdx.x == 0u);
    const bool is_thread0 = (threadIdx.x == 0u);

    /* Monotone local tick counter — atomicAdd into tick_counter each tick */
    uint64_t local_tick = 0ull;

    /* ------------------------------------------------------------------
     * Main persistent loop — spec §2.4
     * ------------------------------------------------------------------ */
    while (true) {
        /* --- TICK BARRIER: all SMs synchronise before polling --- */
        grid.sync();

        /* --- SHUTDOWN CHECK (after sync, before any work) ---
         * Use system-scope acquire-load so the device sees the host's
         * release-store that wrote shutdown_flag = 1. (spec §2.6)
         */
        uint32_t sdown = ring_load_acquire_u32(
            reinterpret_cast<const volatile uint32_t*>(shutdown_flag));
        if (sdown != 0u) {
            break;
        }

        /* --- POLL INPUT RING (block 0, thread 0 only) --- spec §2.4
         * Acquire-load ensures we see any slot payload written by the host
         * before it incremented the head.
         */
        if (is_block0 && is_thread0) {
            uint32_t head = ring_load_acquire_u32(input_ring_head);
            uint32_t tail = ring_load_acquire_u32(input_ring_tail);

            if (head == tail) {
                /* No work: idle spin indicator */
                ring_store_release_u32(tick_status, K3D_TICK_STATUS_IDLE);
                log_emit_1(log_ring_slots, log_ring_head,
                           K3D_LOG_RING_CAPACITY, K3D_LOG_INPUT_EMPTY,
                           0u, local_tick);
                smem.work_desc[0] = 0u; /* signal: no work this tick */
            } else {
                /* Claim the slot at tail */
                uint32_t slot_idx = tail & (K3D_INPUT_RING_CAPACITY - 1u);
                /* Store slot pointer info for all blocks */
                smem.work_desc[0] = 1u;          /* has_work flag */
                smem.work_desc[1] = slot_idx;    /* which input slot */
                /* Release-advance the tail so host sees slot consumed */
                ring_store_release_u32(tick_status, K3D_TICK_STATUS_OK);
                ring_atomic_add_release_u32(input_ring_tail, 1u);
            }
        }

        /* --- SYNC: all blocks see the work_desc written above --- */
        grid.sync();

        /* Skip phases if no work this tick */
        if (smem.work_desc[0] == 0u) {
            /*
             * nanosleep.u32: available on sm_70+ (PTX ISA 6.3).
             * 1000 ns idle sleep reduces SM power during quiescence.
             * Verified available on sm_86.
             */
            if (is_thread0) {
                asm volatile("nanosleep.u32 1000;" :::);
            }
            atomicAdd(const_cast<uint32_t*>(tick_counter), 1u);
            local_tick++;
            continue;
        }

        /* Get the input slot pointer for phases */
        const QuerySlot* work_slot =
            &input_ring_slots[smem.work_desc[1]];

        /* ============================================================
         * PHASE 1: PERCEIVE
         * Spec §2.4. Audit §6 PERCEIVE wiring target:
         *   compute_morton_codes (morton_octree.cu:78) + octree_query_morton
         *   + warp_frustum_cull_simd (frustum_cull_simd.ptx, sm_80 compat)
         * Current: stub that emits trace and loads work descriptor.
         * ========================================================== */
        perceive_phase(work_slot, galaxy, octree, &smem,
                       log_ring_slots, log_ring_head, local_tick);
        grid.sync();

        /* ============================================================
         * PHASE 2: NAVIGATE
         * Audit §6 NAVIGATE wiring target:
         *   led_astar_navigate (led_astar.cu:164) + cosine_similarity_batch
         * Current: stub.
         * ========================================================== */
        navigate_phase(galaxy, trm_weights, &smem,
                       log_ring_slots, log_ring_head, local_tick);
        grid.sync();

        /* ============================================================
         * PHASE 3: REASON (nine-chain swarm)
         * Audit §6 REASON wiring target:
         *   nine_chain_swarm_device (device_functions.cuh:705) [inline __device__]
         *   gre_defeasible_resolver (gre_defeasible_resolver.cu:67) [needs __device__ wrapper]
         * Current: stub.
         * ========================================================== */
        reason_phase(galaxy, specialists, &smem,
                     log_ring_slots, log_ring_head, local_tick);
        grid.sync();

        /* ============================================================
         * PHASE 4: PHYSICS
         * Audit §6 PHYSICS wiring target:
         *   physics_integrate (physics_integrate.cu:23) via RPN opcode 0x154
         *   physics_xpbd_predict / physics_xpbd_solve (separate .cu files)
         * Current: stub.
         * ========================================================== */
        physics_phase(&smem, log_ring_slots, log_ring_head, local_tick);
        grid.sync();

        /* ============================================================
         * PHASE 5: DECIDE (halting gate)
         * Audit §6 DECIDE wiring target:
         *   halting_gate_device (device_functions.cuh:775) [already __device__]
         *   gre_multimodal_halting_gate (gre_multimodal_halting_gate.cu) [optional]
         * Current: stub.
         * ========================================================== */
        decide_phase(&smem, log_ring_slots, log_ring_head, local_tick);
        grid.sync();

        /* ============================================================
         * PHASE 6: ACT (write to output ring)
         * Writes answer to output_ring_slots using ring_atomics.cuh release store.
         * ========================================================== */
        act_phase(output_ring_head, output_ring_tail, output_ring_slots,
                  tick_status, &smem,
                  log_ring_slots, log_ring_head, local_tick);
        grid.sync();

        /* --- TICK COMPLETE --- */
        atomicAdd(const_cast<uint32_t*>(tick_counter), 1u);
        local_tick++;

    } /* end while(!shutdown) */
}

/* =========================================================================
 * Phase stub implementations
 * ========================================================================= */

__device__ void perceive_phase(
    const QuerySlot* slot,
    const GalaxyUniverse* /*galaxy*/,
    const MortonOctree*   /*octree*/,
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id)
{
    /*
     * PHASE WIRING NOTE (next cut):
     *   1. Extract query position from slot->data at GPU_TASK_QUERY_EMBEDDING_OFFSET
     *   2. Call compute_morton_codes device fn for avatar position
     *   3. Call octree_query_morton as inline __device__ for neighbour candidates
     *   4. Load frustum_cull_simd PTX (extern __device__ link) for visible flags
     *   5. Write candidate_star_indices into smem.work_desc[4..15]
     *
     * Audit §7 Q1: frustum_cull_simd.ptx targets sm_80; sm_86 is binary-compatible.
     * Audit §7 Q2: octree_query_morton is single-threaded — extract as __device__ fn.
     */
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        uint32_t payload[2] = { smem->work_desc[1], 0u };
        log_emit(log_slots, log_head, K3D_LOG_RING_CAPACITY,
                 K3D_LOG_PHASE_STUB | 0x01u,  /* PHASE_STUB | PERCEIVE */
                 payload, 2, tick_id);
        /* Stub: mark 0 candidates found */
        smem->work_desc[3] = 0u;
    }
    (void)slot;
}

__device__ void navigate_phase(
    const GalaxyUniverse* /*galaxy*/,
    const TRMWeights*     /*weights*/,
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id)
{
    /*
     * PHASE WIRING NOTE (next cut):
     *   1. Load DependencyKernel (CSR adjacency) from galaxy/octree VRAM
     *   2. Call led_astar_navigate as __device__ (or CDPv2) with start/goal from perceive
     *   3. Score LED-A* candidates with cosine_similarity_batch
     *   4. Write nav_trace into smem.work_desc and swarm_scores[0..8]
     *
     * Audit §7: DependencyKernel (CSR) must be materialised in Galaxy VRAM at boot.
     *   led_astar uses dynamic shared memory — compatible with persistent kernel.
     */
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        log_emit_1(log_slots, log_head, K3D_LOG_RING_CAPACITY,
                   K3D_LOG_PHASE_STUB | 0x02u, /* PHASE_STUB | NAVIGATE */
                   smem->work_desc[3], tick_id);
    }
}

__device__ void reason_phase(
    const GalaxyUniverse* /*galaxy*/,
    const SpecialistPool* /*specialists*/,
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id)
{
    /*
     * PHASE WIRING NOTE (next cut):
     *   1. Read tier_signal from shared memory (written by meta_select_matryoshka_tier RPN)
     *   2. Call matryoshka_prefix_dot_batch for candidate scoring
     *   3. Call nine_chain_swarm_device (device_functions.cuh:705) inline
     *   4. Call gre_defeasible_resolver: extract __device__ wrapper from .cu:67
     *   5. Write scores into smem.swarm_scores[0..8][0..63]
     *
     * Audit §7 Q3: gre_defeasible_resolver is __global__ only. Options:
     *   (a) Extract core logic to __device__ fn — preferred for low latency
     *   (b) CDPv2 launch — available sm_86, adds ~10 µs overhead
     *   Decision deferred to next cut.
     */
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        log_emit_1(log_slots, log_head, K3D_LOG_RING_CAPACITY,
                   K3D_LOG_PHASE_STUB | 0x03u, /* PHASE_STUB | REASON */
                   0u, tick_id);
        /* Stub: write dummy score for lane 0, candidate 0 */
        smem->swarm_scores[0][0] = 0.5f;
        smem->halting_state[0]   = 0u; /* not converged */
    }
}

__device__ void physics_phase(
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id)
{
    /*
     * PHASE WIRING NOTE (next cut):
     *   Physics RPN opcode 0x154 inside modular_rpn_geometric_kernel calls
     *   physics_integrate (physics_integrate.cu:23) — the kernel already
     *   exists and is wired through RPN dispatch. To call it inline here:
     *   Extract physics_integrate's per-body logic to a __device__ fn and
     *   loop over active bodies in smem.
     *   Full physics stack: physics_xpbd_predict → physics_xpbd_solve →
     *   physics_collision_event_write (all in cranium/kernels/).
     */
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        log_emit_1(log_slots, log_head, K3D_LOG_RING_CAPACITY,
                   K3D_LOG_PHASE_STUB | 0x04u, /* PHASE_STUB | PHYSICS */
                   0u, tick_id);
    }
    (void)smem;
}

__device__ void decide_phase(
    K3DSharedMem* smem,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id)
{
    /*
     * PHASE WIRING NOTE (next cut):
     *   Call halting_gate_device (device_functions.cuh:775) with swarm_scores.
     *   Signature: halting_gate_device(scores, n, min_thr, gap_thr, agree_thr)
     *   Write result to smem.halting_state[0]: 1 = converged, 0 = not.
     *   On convergence: copy winning answer embedding to smem.work_desc[8..15].
     */
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        log_emit_1(log_slots, log_head, K3D_LOG_RING_CAPACITY,
                   K3D_LOG_PHASE_STUB | 0x05u, /* PHASE_STUB | DECIDE */
                   smem->halting_state[0], tick_id);
        /* Stub: always converge */
        smem->halting_state[0] = 1u;
    }
}

__device__ void act_phase(
    volatile uint32_t* output_ring_head,
    volatile uint32_t* output_ring_tail,
    OutputSlot* output_ring_slots,
    volatile uint32_t* tick_status,
    K3DSharedMem* /*smem*/,
    LogRecord* log_slots,
    volatile uint32_t* log_head,
    uint64_t tick_id)
{
    /*
     * PHASE WIRING NOTE (next cut):
     *   1. Construct OutputSlot from decide_phase winner (answer index, confidence, etc.)
     *   2. Check output ring has space: if full, emit K3D_LOG_OUTPUT_FULL and set
     *      tick_status = K3D_TICK_STATUS_OUTPUT_FULL, then skip write.
     *   3. Write slot payload, then ring_store_release_u32 to advance output_ring_head.
     *
     * egress_rpn_addr from the WineContractStar transforms the Galaxy answer
     * into bytes before this write (contract selected during perceive).
     */
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        /* Check output ring space */
        uint32_t o_head = ring_load_acquire_u32(output_ring_head);
        uint32_t o_tail = ring_load_acquire_u32(output_ring_tail);
        uint32_t used   = o_head - o_tail;

        if (used >= K3D_OUTPUT_RING_CAPACITY) {
            ring_store_release_u32(tick_status, K3D_TICK_STATUS_OUTPUT_FULL);
            log_emit_1(log_slots, log_head, K3D_LOG_RING_CAPACITY,
                       K3D_LOG_OUTPUT_FULL, o_head, tick_id);
            return;
        }

        /* Write stub output slot (all-zero answer for now) */
        uint32_t slot_idx = o_head & (K3D_OUTPUT_RING_CAPACITY - 1u);
        OutputSlot* out_slot = &output_ring_slots[slot_idx];
        for (uint32_t i = 0u; i < K3D_OUTPUT_SLOT_BYTES / 4u; ++i) {
            reinterpret_cast<uint32_t*>(out_slot->data)[i] = 0u;
        }

        /* Release fence before advancing head: host must see payload before index */
        ring_membar_sys();
        ring_atomic_add_release_u32(output_ring_head, 1u);

        log_emit_1(log_slots, log_head, K3D_LOG_RING_CAPACITY,
                   K3D_LOG_ACT_ENTER, slot_idx, tick_id);
    }
    (void)output_ring_slots;
}
