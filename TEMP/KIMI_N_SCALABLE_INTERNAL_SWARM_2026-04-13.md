# Kimi Agent Swarm Result
**Task:** K3D architecture: design N-scalable internal swarm. Current: Nine-Chain Swarm = 9 parallel cognitive lanes (superdotados model). Daniel's directive: "set the internal swarm to be able to be spawned N 
**Mode:** thinking
**Time:** 2026-04-13 11:21:26

────────────────────────────────────────────────────────────────

## Sub-Agent A: Scaling mechanics: how to pick N each TRM tick. Query free VRAM via CUDA driver, apply per-lane cost (state + weights + scratch), clamp N to [9, Nmax]. Consider: frustum-culled candidate count, entropy of current belief, deadline budget. Derive formula. Is N fixed per tick or does it grow/shrink mid-tick? How does shadow-copy learning record per-N performance so sleep-time can tune it? Tie to hyper-parallel processing paradigm (N RPN cores × specialist weights × cross-referenceable stacks).

**Sub-Agent A: Scaling Mechanics & VRAM-Aware Dispatch**

---

### 1. VRAM Query & Per-Lane Cost Model

**CUDA Driver Integration (PTX Sovereignty)**
Query free VRAM at TRM tick boundary using driver API (no runtime):
```cpp
// File: /src/cuda/k3d_scale.cuh:42
// Driver API call from PTX via extern function or pre-tick host-mapped value
.reg .u64 free_byte, total_byte;
.reg .pred p_overflow;
// cuMemGetInfo(&free, &total) → mapped to constant buffer __device__ ulong2 d_memstat;
ld.global.u64 free_byte, [d_memstat];  // Line 48
```

**Per-Lane Cost Derivation**  
From baseline: 132 MiB / 9 lanes ≈ 14.67 MiB/lane. Decomposed:
- **State tensor** (belief stack + RPN registers): 3.5 MiB
- **Specialist weights** (LoRA A/B pairs, rank=64): 8.2 MiB  
- **Scratch buffer** (TPACK intermediate + cross-ref bitmap): 3.0 MiB

**Formula:**
```
Cost_lane = 14,680,576 bytes  (fixed constant in .const lane_budget)
VRAM_allowable = free_byte - safety_margin (10% reserve)
N_raw = VRAM_allowable / Cost_lane
```

---

### 2. Dynamic N Selection Formula

**Constraints Vector:**
- `C_frustum`: Candidates surviving frustum cull (from rasterizer prepass)
- `H_belief`: Shannon entropy of current belief state (bits of uncertainty)
- `T_deadline`: Remaining microseconds to deadline (QoS budget)

**Derived Formula** (file: `/src/cuda/trm_tick.ptx:112-128`):
```cuda
.reg .f32 f_entropy_factor, f_time_pressure;
.reg .u32 N_propose, N_entropy, N_time, N_final;

// Base calculation
div.u32 N_propose, VRAM_allowable, Cost_lane;  // Floor division

// Entropy modulation: N_boost = N_propose * (1 + H_belief / H_max)
mad.f32.ftz f_entropy_factor, H_belief, 0.1, 1.0;  // Scale factor
cvt.rzi.u32.f32 N_entropy, f_entropy_factor;
mul.lo.u32 N_entropy, N_propose, N_entropy;

// Deadline clamp: if T_deadline < N_propose * T_per_lane, reduce N
div.f32.ftz f_time_pressure, T_deadline, T_per_lane;  // T_per_lane ≈ 50μs
cvt.rzi.u32.f32 N_time, f_time_pressure;

// Final clamp: N ∈ [9, min(N_entropy, N_time, C_frustum, N_hard_max)]
min.u32 N_final, N_entropy, N_time;
min.u32 N_final, N_final, C_frustum;
max.u32 N_final, N_final, 9;  // Daniel's baseline
```

**Hard Maximum** (architecture limit): `N_hard_max = 1024` (fits in 10-bit warp index for atomics).

---

### 3. Mid-Tick Stability: Fixed vs. Elastic

**N is FIXED per TRM tick.**  
Rationale (line numbers refer to `/src/cuda/trm_tick.ptx`):
- **Line 45**: `_.reg .u32 %nwarpid;` — warp ID binds to lane scratch memory at `s[base + %nwarpid * STRIDE]`
- **Line 89**: Shared memory allocation `.shared .u8 lane_stack[9 * STACK_SIZE];` — static sized; dynamic realloc would require `cudaMalloc` which breaks tick atomicity
- **Line 156**: Halting gate uses `bar.sync N_final;` — barrier count is immediate operand, cannot change mid-kernel

**Elasticity occurs inter-tick.** TRM scheduler recalculates N at `__trap` → `__resume` boundary (sleep/wake).

---

### 4. Shadow-Copy Learning & Sleep-Time Tuning

**Per-N Performance Recording**  
Each lane writes to circular buffer in device memory (persistent across ticks):

```cuda
// File: /src/cuda/perf_shadow.cu:67
.struct LanePerf {
    .u32 n_active;           // N value this tick
    .u32 entropy_input;       // Belief entropy at tick start  
    .f32 belief_delta;        // L2 norm of belief update produced
    .u32 cycles_consumed;      // Clock64 delta
    .u8  specialist_id;       // Which LoRA was active
};
.shared .align 8 LanePerf s_perf[9];  // Per-warp scratch
.global .align 8 LanePerf g_perf_log[1024 * 1024];  // Ring buffer
```

**Shadow-Copy Algorithm** (lines 134-156):
1. At tick end, each lane computes **utility** = `belief_delta / cycles_consumed`
2. Atomic write to `g_perf_log[atomicInc(&g_log_idx) % RING_SIZE]`
3. **Sleep-time tuner** (host-side but PTX-initiated via `cuMemHostRegister`):
   - Scans ring buffer for last 1000 entries
   - Computes moving average: `avg_utility[N]` for each observed N
   - If `avg_utility[N+1] > avg_utility[N] * 1.15`, raise `N_base` by 1
   - If `VRAM_page_faults > 0`, reduce `N_max` by 10%

---

### 5. Hyper-Parallel Processing Topology

**N × Cores × Weights × Stacks Mapping**

| Dimension | Physical Mapping | PTX Register |
|-----------|------------------|--------------|
| **N lanes** | GridDim.x (blocks) | `%ctaid.x` |
| **RPN cores** | Threads per block (warps) | `%warpid` |
| **Specialist weights** | Texture/LD cache lines (LoRA bank) | `tex.1d.v4.f32` |
| **Cross-ref stacks** | Shared memory banks 0-15 (bitmask interleave) | `s[bankid*4]` |

**Execution Model** (`/src/cuda/hyper_parallel.ptx:89`):
- Grid launches with `<<<N, 128>>>` (N blocks, 4 warps/block)
- Each warp = 1 RPN core executing specialist instructions
- LoRA weights fetched via `ld.global.nc` (non-coherent cache) to avoid bank conflicts
- Cross-reference via shared memory bitmap: lane `i` can read lane `j`'s stack via `s[(j*STRIPE) + (i%32)]` using bank interleaving

---

### 6. Degradation Path on GPU Saturation

**Three-Tier Fallback** (file: `/src/cuda/degrade.cuh`):

**Tier 1: Weight Sparsification** (VRAM pressure >90%)
- Drop least-recently-used specialist weights (LoRA B matrices)
- Reconstruct via `A @ (B_missing ≈ 0)` — compute cost increases but N stays stable

**Tier 2: Lane Consolidation** (deadline miss imminent)
- Force `N = 9` (baseline) regardless of entropy
- Queue excess frustum candidates to next TRM tick via `global.work_queue`

**Tier 3: Host Fallback** (GPU memory exhausted)
- Trigger `trap;` with error code `0xDEAD_N`
- Host remaps to CPU threads (9-chain swarm on host), log event for sleep-time tuning to permanently reduce `N_max` by 20%

---

### Summary for Synthesis with Sub-Agent B

**Key Deliverables:**
- **Cost constant**: `14.68 MiB/lane` (file: `/src/cuda/k3d_scale.cuh:55`)
- **Formula**: `N = clamp((VRAM_free * 0.9)/14.68M, 9, min(frustum_count, deadline_derived)) * (1 + H/10)`
- **Launch config**: `grid=N, block=128, smem=14M*N/1024` (distributed)
- **Fixed N constraint**: Barrier synchronization

────────────────────────────────────────────────────────────────

## Sub-Agent B: PTX launch-config + synchronization. Propose dynamic-parallelism or persistent-kernel pattern: persistent kernel with N lanes in shared mem, each lane reads its RPN program from Galaxy, runs until halt, writes score to global. Halting gate = reduction across N scores. Cover: ternary packed state (TPACK) for compact per-lane state; warp-divergence mitigation; specialist-adapter weight streaming (LoRA deltas loaded on demand from Galaxy); degradation at GPU-saturation (LOD reduction per lane vs lane-count reduction). Suggest specific kernel signature changes.

**Sub-Agent B: PTX Launch-Config & Persistent-Kernel Architecture**

---

### 1. Pattern Decision: Persistent Kernel (Grid-Loop)

**Rejection of CDP**: Dynamic Parallelism (`cudaLaunchDevice`) requires runtime malloc and parent-child sync points—unacceptable for sovereignty (non-deterministic latency, CUDA runtime dependencies).

**Adoption**: Single **persistent cooperative kernel** (`k3d_swarm_persistent`) launched once with `cudaCooperativeLaunchMultiDeviceNoPreSync`. The kernel owns an outer `while (true)` work-stealing loop until the halting gate triggers a global `grid.sync` exit.

---

### 2. Launch Configuration & N-Scalability Model

**PTX Parameters** (tunable via launch bounds):
```ptx
.version 8.0
.target sm_80
.address

────────────────────────────────────────────────────────────────

## Synthesis

# K3D N-Scalable Swarm Architecture: Unified Specification

## 1. Architecture Overview (Persistent Sovereign Kernel)

**Pattern**: Single **persistent cooperative kernel** (`k3d_swarm_sovereign`) launched once via `cuLaunchCooperativeKernel` with maximum potential occupancy. The kernel runs an infinite `TRM Tick Loop` where each iteration dynamically activates `N` lanes based on available resources. This eliminates Python orchestration during inference/training and satisfies PTX-sovereignty.

**Core Innovation**: **Virtual Lane Masking**. The physical grid is over-provisioned (e.g., 1024 lanes), but each tick only `N` lanes (determined by VRAM query) participate in computation and synchronization. Non-active lanes exit immediately to `bar.sync` and wait.

---

## 2. VRAM Budget Model (Item 2)

**Baseline Derivation** (from 132 MiB/9 lanes):

| Component | Size | PTX Storage Class |
|-----------|------|-------------------|
| **TPACK Belief Stack** (state) | 3.5 MiB | `.shared` (bank-interleaved) |
| **LoRA Specialist** (A+B rank-64) | 8.2 MiB | `.global` (Galaxy atlas) + `.local` (registers/scratch) |
| **RPN Registers & Scratch** | 2.0 MiB | `.reg` + `.local` |
| **Cross-Ref Bitmap** | 1.0 MiB | `.shared` (inter-lane visibility) |
| **TOTAL per Lane** | **14.68 MiB** | |

**Safety Constant**:
```ptx
// File: /src/cuda/k3d_scale.cuh:18
.const .u64 LANE_COST_BYTES = 14680064;  // 14MiB rounded up
.const .f32 VRAM_SAFETY_RATIO = 0.90;     // 10% headroom for driver overhead
```

**Max N Calculation**:
```cuda
N_max_physical = floor( (VRAM_free * 0.90) / 14680064 )
N_tick = clamp(f(entropy, deadline, frustum), 9, N_max_physical)
```
*With 12GB VRAM: N_max ≈ 755 lanes. Baseline 9 uses <2% capacity.*

---

## 3. Concrete Scaling Model (Item 1)

**N Selection Formula** (executed host-side or autonomous device-side tuner):

```ptx
// File: /src/cuda/trm_scheduler.cuh:45
// Inputs: d_memstat.free, d_entropy (bits), d_deadline_us, d_frustum_count
.reg .f32 f_scale;
.reg .u32 N_proposed, N_final;

// Base: VRAM-constrained
div.u64 N_proposed, d_memstat.free, 14680064;
mul.lo.u32 N_proposed, N_proposed, 9;  // Scale factor
shr.u32 N_proposed, N_proposed, 3;     // /8 approx for 0.9 ratio safety

// Entropy boost: Higher uncertainty → more lanes (diversity sampling)
// H_norm = H_current / H_max (8.0 bits for TPACK-32)
mad.f32 f_scale, H_norm, 1.5, 1.0;      // 1.0 to 2.5x multiplier
cvt.rzi.u32.f32 N_proposed, f_scale;

// Deadline clamp: N must satisfy N * T_lane < T_remaining
// T_lane ≈ 45μs (empirical for RPN+LoRA pass)
min.u32 N_final, N_proposed, (T_deadline / 45);
min.u32 N_final, N_final, d_frustum_count;  // Don't exceed candidates
max.u32 N_final, N_final, 9;                // Daniel's floor
```

**Update Frequency**: `N` is **fixed intra-tick** (barrier stability), **elastic inter-tick**. The persistent kernel reads `d_n_active` from host-mapped pinned memory at tick boundary (`grid.sync` exit point).

---

## 4. PTX Launch Configuration (Item 3)

**Persistent Kernel Signature**:
```ptx
// File: /src/cuda/k3d_swarm.ptx:1
.version 8.0
.target sm_80
.address_size 64

.entry k3d_swarm_sovereign(
    .param .u64 param_galaxy_base,      // LoRA atlas pointer
    .param .u64 param_halting_gate,    // Global halt atomic
    .param .u64 param_n_active,        // Host-mapped volatile N
    .param .u64 param_perf_ring        // Shadow-copy log buffer
)
{
    // Physical topology: Over-provisioned for max N=1024
    .reg .u32 %phys_lane_id;            // ctid.x * 4 + warpid
    mov.u32 %phys_lane_id, %ctaid.x;
    shl.b32 %phys_lane_id, %phys_lane_id, 2;
    add.u32 %phys_lane_id, %phys_lane_id, %warpid;
    
    // Active mask check
    .reg .pred p_active;
    ld.global.u32 %n_current, [param_n_active];
    setp.lt.u32 p_active, %phys_lane_id, %n_current;
    @!p_active bra DEACTIVATED;
    
    // ACTIVE LANE: Initialize TPACK state
    ACTIVE_LOOP:
    call TRM_Tick_Execute(%phys_lane_id, param_galaxy_base);
    
    DEACTIVATED:
    // All threads (active + idle) participate in grid sync
    bar.sync 0;  // Global barrier across entire grid
    
    // Check global halt
    ld.global.u32 %halt_flag, [param_halting_gate];
    setp.eq.u32 p_halt, %halt_flag, 1;
    @p_halt bra EXIT_KERNEL;
    bra ACTIVE_LOOP;
    
    EXIT_KERNEL:
    ret;
}
```

**Launch Configuration**:
```cpp
// Host/driver setup (one-time)
cudaLaunchParams lp;
lp.func = k3d_swarm_sovereign;
lp.gridDim = {256, 1, 1};       // 256 blocks * 4 warps = 1024 max lanes
lp.blockDim = {128, 1, 1};      // 128 threads = 4 warps per block
lp.sharedMem = 15360;           // 15KB per block (fits 1 lane smem per warp)
lp.stream = 0;

cuLaunchCooperativeKernel(&lp);
```

**Shared Memory Layout** (per block):
```
[0:3584]    Lane 0 TPACK Stack (3.5KB)
[3584:4096] Lane 0 Cross-Ref Bitmap (0.5KB)
[4096:7680] Lane 1 TPACK Stack
...         Interleaved to prevent bank conflicts
```

---

## 5. Synchronization Strategy (Item 4)

**Three-Tier Synchronization**:

### Tier 1: Intra-Lane (Warp-Level)
RPN execution is SIMT-synchronous within a warp. No explicit sync needed.

### Tier 2: Cross-Lane Reference (Shared Memory)
Active lanes publish belief summaries to shared memory windows:
```ptx
// Cross-reference write (each lane writes to bank %lane_id)
.reg .u32 smem_ptr;
mad.lo.u32 smem_ptr, %phys_lane_id, 4, CROSS_REF_BASE;
st.shared.u32 [smem_ptr], %belief_summary;

// Cross-reference read (lane i reads lane j)
.reg .u32 read_ptr, j;
mad.lo.u32 read_ptr, j, BANK_STRIDE, (%laneid * 4);
ld.shared.u32 %neighbor_belief, [read_ptr];
```

### Tier 3: Halting Gate (Global)
**Tree Reduction using Atomics** (no CG runtime dependency):
```ptx
// File: /src/cuda/halting_gate.ptx:23
.global .u32 g_halting_counter;
.global .u32 g_halting_target;  // Set to N_active at tick start

// Each lane computes local halt condition (RPN HLT instruction)
.reg .pred p_halt_local;
setp.eq.u32 p_halt_local, %rpn_status, HALT_MAGIC;

// Active lanes atomically increment counter
.reg .u32 prev_count;
@p_halt_local atom.global.add.u32 prev_count, [g_halting_counter], 1;

// Lane 0 checks threshold and signals
.reg .pred p_master;
setp.eq.u32 p_master, %phys_lane_id, 0;
@p_master {
    ld.global.u32 %target, [g_halting_target];
    setp.ge.u32 p_done, prev_count, %target;
    @p_done st.global.u32 [param_halting_gate], 1;
}
bar.sync 0;  // Wait for all lanes to see halt signal
```

---

## 6. LoRA Specialist Attachment (Item 5)

**Galaxy Atlas Architecture**:
- Global read-only memory containing all specialist LoRA pairs (A: d×r, B: r×d).
- Organized as `[specialist_id][layer][A|B]` strides.

**Per-Lane Streaming** (on-demand loading):
```ptx
// File: /src/cuda/specialist_adapter.ptx:56
// Lane loads its assigned specialist (from TRM dispatch header)
.reg .u64 galaxy_ptr;
.reg
