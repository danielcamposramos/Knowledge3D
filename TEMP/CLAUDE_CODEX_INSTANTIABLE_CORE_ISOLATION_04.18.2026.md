# Claude → Codex Spec: Instantiable Core Isolation

**Date**: 2026-04-18
**Author**: Claude (architecture)
**Implementer**: Codex
**Daniel's Ruling (verbatim)**: "Keep it simple, since they are instantiable — math cores (three tiers connected and self-referring able internally) are cores because they are also instantiable — they MUST be isolated (the cores, not the internal stacks of the tiers)."

---

## 1. The Contract — What "Isolated" Means for a Math Core

A math core is an **independently instantiable GPU compute unit** bound to one CUDA block. Isolation means:

1. **Zero cross-core mutable shared memory.** A core's `__shared__` declarations are private to its block. No `__shared__` symbol is declared in a kernel's global scope where two blocks could alias it.

2. **Zero cross-core synchronization barriers.** `__syncthreads()` and `__syncwarp()` are intra-block only — this already holds by CUDA semantics. The violation to guard against is `cudaDeviceSynchronize()` or `cudaStreamSynchronize()` called *inside* a per-core kernel. Those are host-side and must only appear in the Python launch scaffold, never embedded in PTX logic.

3. **One VRAM queue per core.** Each core has its own bounded-capacity queue slot in a global queue array `core_queues[MAX_CORES]`. The core reads from `core_queues[my_core_id]` and writes results to `core_output_tiles[my_core_id]`. No two cores share a queue slot.

4. **Results written to per-core output tile.** Each core writes its computed result to `core_output_tiles[my_core_id][...]` — a pre-allocated VRAM region. The host Python bridge reads the tile *after* kernel completion. No results are written to shared regions during execution.

5. **Cross-core coordination only via global memory queues.** If Core A needs to pass intermediate results to Core B, it does so by writing a message to Core B's queue via `QUEUE_PUSH`. Core B reads it via `QUEUE_POP`. There is no direct memory path between cores other than these queue slots.

---

## 2. What Is NOT Isolated — Tiers Inside a Core

The three tiers (Tier 1, Tier 2, Tier 3) **within a core share the yard substrate**. This is intentional and correct per Daniel's ruling:

> "the cores, not the internal stacks of the tiers"

Inside a single core (single CUDA block), the following is valid and correct:

- **Tier 3 calls a Tier 2 macro** which calls a **Tier 1 primitive** — all inside the same shared-memory `yards[9][9][69]`.
- The nine yards (`bank_id` 0-8) are the tiers' shared workspace. A Tier 3 physics integrator may deposit intermediate values in Bank 3, call a Tier 2 linear-algebra macro that reads Bank 3 and writes Bank 4, which calls a Tier 1 float4 op on Bank 4.
- Self-reference is permitted: an RPN program in Bank 2 may push an opcode that causes a recursive sub-program execution in Bank 5, provided the yard depth (69 slots) is not exceeded.

**Summary:** Tiers form a deep call hierarchy *inside* a core. Cores are peers — flat, isolated, communicating only through queues.

```
Core 0 [block 0]          Core 1 [block 1]
┌─────────────────────┐   ┌─────────────────────┐
│ yards[9][9][69]     │   │ yards[9][9][69]      │
│ sp[9][9]            │   │ sp[9][9]             │
│ active_bank[9]      │   │ active_bank[9]       │
│                     │   │                      │
│ Tier3 → Tier2 → T1  │   │ Tier3 → Tier2 → T1   │
│   (shared banks)    │   │   (shared banks)     │
└────────┬────────────┘   └──────────┬───────────┘
         │  QUEUE_PUSH to Core 1              │
         └──────────────────────────────────►│
                                    QUEUE_POP │
```

---

## 3. Core-Count Math — RTX 3070 (sm_86)

| Parameter | Value | Source |
|---|---|---|
| SM count on RTX 3070 | 46 | sm_86 physical |
| `cores_per_sm` | 9 | Tesla-9 compliance; digit-sum 9 |
| Concurrent cores | **414** | 46 × 9 |
| Per-core yard memory | 9 × 69 × float4 = 9,936 B ≈ 10 KB | §4.3 of Transfer Yard spec |
| Total yard memory | 414 × 10 KB = **4.03 MB** | Negligible vs 12 GB VRAM |
| Per-SM shared memory used | 22.3 KB (yards + sp + active_bank) | Within Ampere 100 KB/SM budget |

**Current state (Codex: verify by reading source):**
- `knowledge3d/cranium/bridges/advanced_rpn.py` — check for `MAX_INSTANCES` hard-code; likely 10 or 18.
- `knowledge3d/cranium/bridges/transfer_yard_tiered.py` — check `TransferYardTier2Engine` instance count.
- `knowledge3d/cranium/bridges/lightweight_rpn.py` — check Tier 1 concurrency setting.

After this spec lands: all three bridges derive concurrency from `MicroSpecialistPool.query_sm_count() * 9`, not hard-coded constants.

---

## 4. Instantiation API — Host-Side Bridge

The host-side bridge exposes one spawning primitive per engine tier. Codex writes these; the contract is defined here.

```
CoreHandle = CoreRegistry.spawn(
    tier   : int,          # 1, 2, or 3
    program: bytes,        # serialized RPN opcode+operand pairs
    inputs : list[float],  # initial values pushed to Bank 0 before launch
) -> CoreHandle
```

### Contract

- `CoreHandle` holds: `core_id: int`, `output_tile_ptr: ctypes.c_void_p`, `queue_slot_ptr: ctypes.c_void_p`.
- `CoreRegistry` tracks allocated cores against the 414-core hard ceiling (from `query_sm_count() * 9`). Attempting to spawn beyond the ceiling raises `CorePoolExhausted` immediately — no queuing, no blocking.
- **No shared state between handles.** `CoreHandle` objects are independent. Reading from one handle's output tile is safe while another handle is executing.
- `block_group` in CUDA = one core instance. Each `spawn()` allocates one CUDA block from the pool. The pool is pre-allocated at system start from a single `cuMemAlloc` of size `MAX_CORES × (YARD_MEM + OUTPUT_TILE_SIZE)`.

### Release

```
CoreRegistry.release(handle: CoreHandle)
```

Marks the core slot as available for reuse. Non-blocking. Codex may pool and reuse core slots without returning VRAM to the OS.

---

## 5. Cross-Core Coordination via Global Queues

### Queue ABI

The global queue array lives in VRAM Region 2 (Galaxy Universe — active working memory).

```cuda
// Global layout — allocated once at boot, never reallocated
struct CoreQueueHeader {
    uint32_t write_idx;   // producer advances this
    uint32_t read_idx;    // consumer advances this
    uint32_t capacity;    // fixed at boot (suggest: 64 slots per core)
    uint32_t _pad;
};

struct CoreQueueSlot {
    uint32_t src_core_id;
    uint32_t msg_type;     // RESULT=0, QUERY=1, SIGNAL=2
    float4   payload[16];  // 256 bytes of payload (4 float4 rows)
};

// Full layout in VRAM
CoreQueueHeader headers[MAX_CORES];        // one header per core
CoreQueueSlot   slots[MAX_CORES][64];      // 64 message slots per core
```

Total queue memory: 414 cores × (16 B header + 64 × 272 B slots) = **7.2 MB VRAM**.

### Opcodes for Queue I/O (extend RPN_DOMAIN_OPCODE_REGISTRY.md — range 0x178-0x17A)

| Opcode | Name | Semantics |
|---|---|---|
| `0x178` | `QUEUE_PUSH` | Pop `dst_core_id` (scalar), pop `payload_f4[16]` from active bank (top 16 float4 values), write message to `core_queues[dst_core_id]`. Spins on CAS if queue is full (bounded spin). |
| `0x179` | `QUEUE_POP` | Block until a message arrives in `core_queues[my_core_id]`. Push `payload_f4[16]` onto active bank. Push `src_core_id` scalar. |
| `0x17A` | `QUEUE_PEEK` | Non-blocking. Push 1 if `core_queues[my_core_id]` has pending messages, push 0 otherwise. Does not consume message. |

**Placement in the opcode registry:** These three opcodes follow the yard ops (0x170-0x177) and precede the reserved range (0x17B-0x17F). Register them in `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` and in `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`.

### Queue semantics notes

- `QUEUE_PUSH` is a device-side operation executed within the core's kernel. Bounded spin (maximum 1024 cycles) avoids deadlock; if the destination queue is full after 1024 cycles, the push is dropped and a flag bit is set in the core's status register. The Jarvis coordinator specialist polls status registers.
- `QUEUE_POP` with blocking is only valid in a speculative thread — Codex must ensure that a core waiting on `QUEUE_POP` does not hold the only thread group needed for the waiting core to produce its message (deadlock prevention by design — the nine-chain swarm topology avoids cycles).
- Cores NEVER write to another core's `core_output_tiles`. Output tiles are for host-side collection only.

---

## 6. Isolation Acceptance Gates

Codex runs these five checks before declaring core isolation complete:

### Gate 1 — No `__shared__` outside block scope
```bash
grep -n "__shared__" knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu \
                    knowledge3d/cranium/kernels/advanced_rpn_kernel_transfer_yard.cu
# Every hit must be inside a kernel function body, NOT at file/namespace scope.
# File-scope __shared__ → isolation violation.
```

### Gate 2 — No `cudaDeviceSynchronize` inside a core kernel
```bash
grep -rn "cudaDeviceSynchronize\|cuStreamSynchronize" \
    knowledge3d/cranium/kernels/ \
    knowledge3d/cranium/ptx/
# → 0 hits. These calls are host-only and live in Python bridges.
```

### Gate 3 — No hard-coded MAX_INSTANCES in bridge files
```bash
grep -rn "MAX_INSTANCES\s*=\s*[0-9]" knowledge3d/cranium/bridges/
# → 0 hits. All concurrency is derived from query_sm_count() * 9.
```

### Gate 4 — Queue opcodes registered in opcode registry
```bash
grep -nE "QUEUE_(PUSH|POP|PEEK)" docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md
# → 3 hits minimum (one per opcode with address 0x178, 0x179, 0x17A).
grep -nE "QUEUE_(PUSH|POP|PEEK)" knowledge3d/cranium/ptx_runtime/rpn_opcodes.py
# → 3 hits minimum.
```

### Gate 5 — Each core has distinct queue slot address
```bash
# Runtime check (Codex writes a test):
# Spawn 2 cores, verify their queue_slot_ptr values differ by exactly sizeof(CoreQueueSlot)*64.
python -m pytest tests/test_core_isolation.py::test_queue_slots_non_overlapping -xvs
# → PASSED
```

---

## 7. Codex Handoff Checklist

1. Read Transfer Yard spec (`CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md`) §2-§4 for the `yards[9][9][69]` layout that the cores share internally.
2. Grep current bridge files: `grep -rn "MAX_INSTANCES" knowledge3d/cranium/bridges/` — note all hits, they all get replaced.
3. Read `knowledge3d/cranium/ptx_runtime/micro_specialist_pool.py` — find `query_sm_count()` or equivalent. Confirm it returns 46 on RTX 3070. If not present, add it.
4. Add `CoreQueueHeader` and `CoreQueueSlot` struct definitions to `knowledge3d/cranium/cuda/rpn_execute_device.cuh`.
5. Allocate global queue array in `knowledge3d/cranium/sovereign/loader.py` at boot — one `cuMemAlloc` call for headers + slots, total ≈ 7.2 MB. Store `queue_base_ptr` in a module-level ctypes pointer.
6. Implement `QUEUE_PUSH` (`0x178`), `QUEUE_POP` (`0x179`), `QUEUE_PEEK` (`0x17A`) as device-inline functions in `rpn_execute_device.cuh` — they are called from within kernel code, not as separate PTX kernels.
7. Replace all `MAX_INSTANCES = N` hard-codes in `bridges/lightweight_rpn.py`, `bridges/transfer_yard_tiered.py`, `bridges/advanced_rpn.py` with `MicroSpecialistPool.query_sm_count() * 9`.
8. Write `CoreRegistry` class in `knowledge3d/cranium/bridges/core_registry.py` — `spawn()` and `release()` following §4 contract. Use ctypes, no numpy.
9. Write `tests/test_core_isolation.py` with at minimum: `test_queue_slots_non_overlapping`, `test_no_cross_core_shared_memory` (static grep check), `test_spawn_beyond_ceiling_raises`.
10. Run all five acceptance gates (§6). Report pass/fail per gate with evidence.

---

## 8. Must-NOT-Do List

- ❌ Do NOT declare `__shared__` at file scope in any kernel. All shared memory is per-block, declared inside kernel function bodies.
- ❌ Do NOT call `cudaDeviceSynchronize` from within a core kernel. This stalls ALL cores on the device.
- ❌ Do NOT have two cores write to overlapping VRAM regions for their results. Each core has its own output tile.
- ❌ Do NOT implement cross-core communication by having Core A directly read Core B's yard memory. The yard is private to the block. Use `QUEUE_PUSH` / `QUEUE_POP`.
- ❌ Do NOT keep `MAX_INSTANCES = 18` (or any constant) after this spec lands. Always derive from hardware.
- ❌ Do NOT add numpy anywhere in `core_registry.py` or `micro_specialist_pool.py`. ctypes only.
