# CODEX SPEC — Close the GPU Game Loop (one cut, no new Python modules)

**Date:** 2026-04-18
**Author:** Claude (architecture), synthesizing MVCIC chain (Kimi → Qwen → GLM → DeepSeek → Nemotron → post-chain Kimi)
**Source:** `TEMP/MVCIC_CLOSE_GPU_GAME_LOOP_04.18.2026.md`
**Status:** P0. This supersedes `CLAUDE_KILL_PYTHON_DISPATCH_04.18.2026.md` and any "runtime helper" interpretation of it.

---

## 0. Non-Negotiable Preamble (read before you touch anything)

**Do NOT create any new Python module.** The previous spec produced `query_tick_runtime.py +407 lines` — another Python orchestrator under a new name. This one is different: every numbered deliverable below is either (a) a CUDA/PTX kernel, (b) a VRAM layout, (c) an RPN program registered in a Galaxy star, (d) a Galaxy star schema change, or (e) a **deletion**. If you find yourself opening a new `.py` file, stop and re-read this section.

`query_tick_runtime.py` must be deleted as part of this cut. It is not refactored, not renamed, not kept "for compatibility." Deleted.

Python's total contribution to the hot path after this cut: push bytes into `input_ring`, poll `output_ring`, write bytes to stdout/socket. That is it.

---

## 1. The Architecture In One Paragraph

One `cudaLaunchCooperativeKernel` call at boot starts `trm_step_fused` across every SM. It runs until shutdown. Inside, a `while(!shutdown)` loop does `grid.sync()` each tick, reads the VRAM-resident input ring, runs phases PERCEIVE → NAVIGATE → REASON → PHYSICS → DECIDE → ACT, writes to the output ring, and loops. Python never launches a kernel per query. Contracts (DOM, ARC3, stdin, …) are Galaxy stars; selection is a device-side `GALAXY_SCAN(0xE2)`. Specialist selection is a Layer 4 meta-rule RPN program. Matryoshka tier selection is a Layer 4 meta-rule RPN program. Nothing on the hot path calls Python, regex, or a library.

---

## 2. The Hot-Path Persistent Kernel

### 2.1 Files to create (CUDA/PTX only)

- `knowledge3d/cranium/ptx_kernels/persistent_tick.cu` — kernel entry, `cudaLaunchCooperativeKernel` target, inner `while` loop.
- `knowledge3d/cranium/ptx_kernels/ring_atomics.cuh` — acquire/release load/store helpers (PTX inline: `ld.global.acquire.u32`, `st.global.release.u32`, `membar.sys`).
- `knowledge3d/cranium/ptx_kernels/wine_contract_scan.cu` — Galaxy scan by `paradigm_type` → `ingress_rpn_addr`.
- `knowledge3d/cranium/ptx_kernels/matryoshka_prefix_dot.cu` — variable-width fused prefix dot with `__shfl_sync` warp reduce.
- `knowledge3d/cranium/ptx_kernels/vram_freelist.cu` — device-side slab allocator for new Galaxy stars (no `cudaMalloc` on hot path).
- `knowledge3d/cranium/ptx_kernels/log_ring.cu` — VRAM circular buffer for kernel log records.

### 2.2 Kernel signature (authoritative)

```cuda
__global__ void trm_step_fused(
    // Zero-copy rings (host-pinned, device-mapped)
    volatile uint32_t* input_ring_head,      // atomic producer idx (host write)
    volatile uint32_t* input_ring_tail,      // atomic consumer idx (device write)
    QuerySlot*         input_ring_slots,     // fixed-size slots
    volatile uint32_t* output_ring_head,     // atomic producer idx (device write)
    volatile uint32_t* output_ring_tail,     // atomic consumer idx (host write)
    OutputSlot*        output_ring_slots,
    volatile uint32_t* log_ring_head,        // device log emit
    LogRecord*         log_ring_slots,

    // Galaxy + indices (VRAM-resident)
    const GalaxyUniverse* galaxy,
    const MortonOctree*   octree,
    const TRMWeights*     trm_weights,
    const SpecialistPool* specialists,       // LoRA adapters, device-indexed

    // VRAM free-list for in-tick Galaxy star creation
    VramFreelist*      freelist,

    // Control
    volatile uint32_t* tick_counter,
    volatile uint32_t* tick_status,          // kernel-to-host error code (NO try/except)
    volatile uint32_t* shutdown_flag);       // membar.sys fenced
```

### 2.3 Launch (Python boot-only, ~8 lines, NOT hot path)

Put the launch in `knowledge3d/knowledgeverse/trm_boot.py` (new small file, boot-only; not a runtime module). The launch returns immediately; the kernel stays on the device until `shutdown_flag` is set.

```python
# trm_boot.py — BOOT ONLY, no per-query code here
def launch_persistent_trm(ctx):
    cuda.launch_cooperative_kernel(
        trm_step_fused,
        grid_dim=ctx.device.sm_count,
        block_dim=256,
        shared_bytes=49152,
        stream=ctx.stream,
        args=ctx.kernel_args,
    )
```

Nothing else goes in `trm_boot.py`. No "helpers," no "runtime," no "dispatcher."

### 2.4 Inner loop (authoritative phase order)

```
while (!*shutdown_flag) {
    grid.sync();

    if (block 0, thread 0) poll input ring, populate shared work descriptor;
    grid.sync();

    perceive_phase(work, galaxy, octree);            // -> candidate_stars
    navigate_phase(candidate_stars, trm_weights);    // -> nav_trace
    reason_phase(nav_trace, specialists);            // -> swarm_scores[9][C]
    physics_phase(swarm_scores);                     // -> physics_state
    decide_phase(swarm_scores, physics_state);       // halting gate
    act_phase(answer);                                // -> output ring

    atomicAdd((unsigned*)tick_counter, 1);
}
```

Every phase is a device function that composes *existing* kernels. See §3.

### 2.5 Grid sync on sm_86 (RTX 3070 baseline)

`cudaLaunchCooperativeKernel` + `cuda::grid_group::sync()` is available from sm_60 upward. **Do not** use `bar.grid.sync` PTX unless the target is sm_90+; use the cooperative groups API and let the driver pick the right PTX. GLM correctly flagged Qwen's `sm_90+` claim; ignore it.

### 2.6 Ring atomics (correct, not Qwen's stale-read draft)

- Producer increment: `atomicAdd(head, 1)` then release-fence before data read by consumer.
- Consumer load of head: `ld.global.acquire.u32` (or `__ldcg` + `__threadfence_system`).
- Shutdown flag: `ld.global.volatile.u8` + `membar.sys` before branch.

Put these as inline helpers in `ring_atomics.cuh`. No other file writes ring atomics by hand.

### 2.7 Shared memory layout (per block, 48 KB budget)

```
.shared .align(16) .f32 rpn_stacks[9][64][4];   // 9 lanes × 64-deep × float4 = 9216 B
.shared .align(4)  .f32 swarm_scores[9][64];    // 9 workers × 64 candidates   = 2304 B
.shared .align(4)  .u32 halting_state[32];      //                              =  128 B
.shared .align(128).b8  scratch[remainder];
```

GLM flag acknowledged: RPN stack is **64-deep `float4`**, not 15×32 bytes. StackValue's `w` lane is the tag.

---

## 3. Kernel Reuse (stop reinventing — 6 kernels already exist)

Nemotron caught this. For each phase, wire the existing kernel; do NOT write a new one.

| Phase | Existing kernel (wire) | Current status |
|-------|------------------------|----------------|
| PERCEIVE | `morton_octree.ptx` + `frustum_cull.cu` | In repo, not in tick |
| NAVIGATE | `led_astar.ptx`, scored by `cosine_similarity.ptx` | In repo, not in tick |
| REASON (swarm) | `nine_chain_swarm_kernel.cu` | In repo, not in tick |
| REASON (conflict resolution) | `gre_defeasible_resolver.cu` | Loaded, never called |
| MULTI-HOP | `gre_graph_crystallizer.cu` | Loaded, never called |
| RPN dispatch | `modular_rpn_kernel.cu` (opcode switch incl. 0x150–0x17F) | In repo, dispatch Python today |

The 15 GRE specialist kernels loaded at boot but uncalled during inference all live in this list. Wire them.

The only *genuinely new* kernel the chain identified is multi-hop crystallized traversal — and that's `gre_graph_crystallizer.cu`, already in the repo.

---

## 4. Tablet WINE Contracts — Galaxy Stars, Not Python Adapters

### 4.1 New Galaxy star schema (Layer 2)

```c
struct WineContractStar {
    uint64_t contract_hash;        // murmur3 of paradigm signature
    uint8_t  paradigm_type;        // 0x01=DOM, 0x02=ARC3, 0x03=TEXT, 0x04=AUDIO, 0x05=IMAGE
    uint64_t ingress_rpn_addr;     // Layer 3 Grammar Galaxy program (bytes → Galaxy form)
    uint64_t egress_rpn_addr;      // Layer 3 Grammar Galaxy program (Galaxy form → bytes)
    uint64_t visual_rpn_symlink;   // optional Layer 1 Drawing Galaxy ref
};
```

Add to `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md` as a new star kind.

### 4.2 Device-side contract resolution

```
E0: LOAD_GALAXY         // load Galaxy base into RPN register
E2: GALAXY_SCAN         // scan by field; predicate = (paradigm_type == in.header.paradigm_type)
```

No Python dict, no `re.search`, no contract registry module. Delete any Python `wine_contract_registry` scaffold if Codex finds one.

### 4.3 First three contracts to ship (as Galaxy stars, not Python)

1. **DOM `<p>` output** (Christoph's target): `egress_rpn_addr` = RPN program that emits an HTML tag envelope around text bytes produced by the tick.
2. **ARC3 game-frame ingress**: `ingress_rpn_addr` = RPN that reads the ARC3 frame header, maps grid cells to Galaxy atoms via `tablet/wine/game2d_wine.py` logic — but the logic moves into an RPN program, and `game2d_wine.py` shrinks to a boot-time bytes pipe.
3. **stdin/stdout text**: `ingress_rpn_addr` = tokenize-by-whitespace RPN; `egress_rpn_addr` = UTF-8 write RPN.

All three ship as Galaxy star payloads (JSONL seed consumed at boot by the Galaxy loader, then resident in VRAM). Python on the hot path never sees them.

---

## 5. Matryoshka Substrate (no numpy, no torch)

### 5.1 VRAM layout

Single base pointer per embedding. Dim tiers {64, 128, 256, 512, 1024, 2048} are prefix views — `ptr + k*4` for a k-dim view. Store as SoA across embeddings for coalesced access:

```
emb_base : float[N][2048]   // flat row-major, row = embedding, col = dim
```

A k-dim prefix view is the first k columns of each row.

### 5.2 Fused prefix dot (authoritative)

`matryoshka_prefix_dot.cu`: one warp per candidate embedding, each thread handles `k/32` floats, partial sum, `__shfl_xor_sync` log2 reduction, lane 0 writes out. Takes `k` as a runtime parameter loaded from a Layer 4 meta-rule RPN program output (§5.3), not a host `if/else`.

### 5.3 Meta-rule for tier selection

New Galaxy star: `meta_select_matryoshka_tier` in Layer 4. RPN body reads query signal σ (entropy, length, domain) produced in PERCEIVE, returns `k ∈ {64,128,256,512,1024,2048}`. Result written to `tier_signal` shared register; `matryoshka_prefix_dot` reads it directly.

No Python decides tier. Ever.

---

## 6. Specialist Dispatch (cognitive lanes, not multi-agent)

`micro_specialist_pool` is deleted from the hot path. Specialist selection becomes a Layer 4 meta-rule RPN program (`meta_select_specialist_lane`) that reads the PERCEIVE output and returns up to 9 specialist IDs for the nine-chain swarm. IDs index a VRAM-resident descriptor table `SpecialistPool.descriptors[]` with pointers to LoRA weight slices — co-resident since boot.

`nine_chain_swarm_kernel.cu` already handles the parallel evaluation. No new kernel. No new Python.

---

## 7. Self-Crafting (sleep-time only, never hot path)

When the tick hits an undefined opcode or a contract lookup miss, it writes a `CraftTicket` to a VRAM crafting queue (just another ring). Sleep-time kernels (`sleep_cluster_refiner.ptx` + new `sleep_physics_crystallizer.ptx` per Nemotron/post-chain) drain the queue and create new Galaxy stars / emit new kernels.

Python's role in crafting: when a new PTX kernel is emitted to the queue, a boot-like helper (runs only during sleep, never mid-tick) writes the `.ptx` file to disk and re-loads the kernel table. The hot path continues with the old table until the next relaunch window.

No new Python runtime module for this. The sleep path already exists in `knowledge3d/knowledgeverse/sleeptime.py`; extend it there if needed, then stop.

---

## 8. Deletion List (P0, grep-verifiable)

```
rm  knowledge3d/knowledgeverse/query_tick_runtime.py
```

In `knowledge3d/knowledgeverse/trm_game_loop.py`:

- Replace the body of `_run_query_tick` (currently at line ~315, 34 lines) with the 12-line version below.
- Delete any call from `_run_query_tick` into `_dispatch_sovereign_task`.

```python
def _run_query_tick(self, bridge: Any, record: TRMQueuedInput) -> dict[str, Any]:
    tick_result = dict(bridge.run_query_tick(delta_time=0.02))
    action_buffers = self._action_buffer_payload(bridge)
    self._last_tick_result = dict(tick_result)
    self._last_action_buffers = [list(row) for row in action_buffers]
    return {
        "status": "ok",
        "mode": "query_tick",
        "trm_tick": tick_result,
        "action_buffers": action_buffers,
    }
```

In `knowledge3d/knowledgeverse/knowledgeverse.py`:

- Delete `_dispatch_sovereign_task`.
- Delete `_build_universal_decomposer_programs` (the `re.findall`).
- Delete any `micro_specialist_pool.run_overflow_sequential` invocation from the hot-path call graph. The class may remain only if nothing on the ring path imports it; if still imported, delete the callers too.

In the codebase globally:

- Delete any `import re` reachable from `enqueue_task` → `wait_output_buffer`. Grep-verify.
- Delete any `logging.*` call reachable from that same graph. Route to `log_ring_buffer` (§2.1).
- Delete any `try/except` around kernel launches on the hot path. Kernel writes `tick_status`; host reads it asynchronously.

If deletion breaks imports, the fix is to delete the import, not to add a stub. "We fail and fix." (Daniel)

---

## 9. Acceptance (grep + nvidia-smi, NOT scores)

Ordered gate. Each step must pass before the next is claimed.

1. `test -e knowledge3d/knowledgeverse/query_tick_runtime.py` → **false**.
2. `wc -l` of `_run_query_tick` body → **≤ 15 lines**.
3. `grep -rn "re\\.\\(findall\\|search\\|compile\\|match\\)" knowledge3d/knowledgeverse/ knowledge3d/cranium/` → **zero hits in the ring tick call graph** (`enqueue_task` → `wait_output_buffer`). Use a small trace to generate the reachable set; if unclear, grep the entire two dirs and demonstrate each surviving hit is in ingestion/sleep, not hot path.
4. `grep -rn "_dispatch_sovereign_task\\|_build_universal_decomposer_programs\\|run_overflow_sequential" knowledge3d/` → **zero hits**.
5. Boot the daemon, issue one query. `nvidia-smi dmon -s u -c 30` during the query → **sustained >50 % `utilization.gpu`** (not flatline 0, not single spikes).
6. `gpu_calls_this_command` counter → **hundreds per query**, not 0/1/2.
7. A single `cudaLaunchCooperativeKernel` at boot; **no further kernel launches per query** (confirmed by `nsys profile` or by adding a boot-time counter).
8. ARC, Math, LHE benchmarks run end-to-end. Scores may drop. That is honest and expected. Do not tune scores before #1–#7 are green.

Do not claim #1–#7 piecewise — they land together or the cut is incomplete.

---

## 10. Ordered Handoff Checklist for Codex

- [ ] **D1.** Delete `query_tick_runtime.py`.
- [ ] **D2.** Replace `_run_query_tick` body with the 12-line version above.
- [ ] **D3.** Delete `_dispatch_sovereign_task` and `_build_universal_decomposer_programs`; remove their callers.
- [ ] **D4.** Grep-verify gates #3, #4.
- [ ] **K1.** Create `knowledge3d/cranium/ptx_kernels/ring_atomics.cuh` (inline acquire/release/fence helpers).
- [ ] **K2.** Create `persistent_tick.cu` with the §2.2 signature and §2.4 loop; phases call the existing kernels from §3.
- [ ] **K3.** Create `wine_contract_scan.cu`; add `WineContractStar` schema to canonical registry.
- [ ] **K4.** Create `matryoshka_prefix_dot.cu`; add `meta_select_matryoshka_tier` Galaxy star.
- [ ] **K5.** Create `vram_freelist.cu` (slab allocator, no `cudaMalloc` on hot path).
- [ ] **K6.** Create `log_ring.cu` and migrate hot-path log emits to it.
- [ ] **B1.** Create `trm_boot.py` (≤ 12 lines) with the cooperative launch. Nothing else.
- [ ] **G1.** Seed the three initial WINE contracts (DOM `<p>`, ARC3 frame, stdin text) as Galaxy stars in the seed JSONL.
- [ ] **V1.** Run the full acceptance gate (§9). Report each step pass/fail with evidence.

---

## 11. What Codex Must NOT Do

- Must not create a new Python runtime/helper/dispatcher/orchestrator module. Anything that looks like one is a regression.
- Must not keep `query_tick_runtime.py` "until the kernels are ready." Delete it first; kernels can be stubbed to return a canned envelope until the phase kernels wire in.
- Must not reintroduce `re.*` anywhere on the hot path. Contract hashing is murmur3 in a kernel; query routing is `GALAXY_SCAN(0xE2)`.
- Must not add fallbacks. No `try/except` around kernel launches. No "if kernel failed, call Python." Kernel writes `tick_status`; host surfaces the status; the fix is the kernel, not a shim.
- Must not "preserve backwards compatibility" with the old Python dispatch. Nothing in the public surface of this system depends on it. Delete and move on.

---

## 12. Notes from the chain (non-blocking, do not expand scope)

- **Gemini slot failed** (404 — `gemini-3-flash:cloud` was routed to localhost:11434 instead of the cloud endpoint). Not blocking; the remaining five partners + post-chain covered the ground. Fix the routing in `ollama_specialists.py` separately.
- **Nemotron's warp-shuffle patterns** (`__shfl_xor_sync` reductions, GJK per-warp) are banked for the physics phase deepening, not this cut. Keep physics phase in the tick but wire the stub first.
- **DeepSeek's House↔Galaxy symlink concern**: Galaxy is fully VRAM-resident post-boot. Mid-tick disk fetch is out of scope for this cut; if a symlink points to non-resident content, kernel writes `tick_status = NEEDS_HOUSE_LOAD` and returns a "not-yet" envelope. Sleep-time handles the load. No Python I/O mid-tick.
- **GLM's 20 flagged issues**: all the correctness ones (RPN depth, stack format, fence semantics, cooperative limits, free-list) are folded into §2–§5 above. The rest are either deferred (Tensor Core layout for TRM forward) or already addressed (log ring, contract fallback).

---

## 13. Bottom Line (for Daniel)

One launch at boot. Rings in VRAM. Phases compose existing kernels (6 of them already in the repo, 15 more loaded). Contracts are stars, not Python. Specialists and matryoshka tiers are meta-rule RPN, not host branches. Python is 200-ish lines of boot + I/O + stdout.

If Codex reads this and opens `a_new_python_file.py`, stop him. The spec has no such file in it.
