# Yard Kernel Design Memo

**Date**: 2026-04-18
**Author**: Claude (architecture lane)
**For**: Codex (implementer)
**Files**: `reference_modular_rpn_kernel_transfer_yard.cu`, `reference_advanced_rpn_kernel_transfer_yard.cu`
**Supersedes**: shared-memory budget claim in `CLAUDE_CODEX_TRANSFER_YARD_AND_EMBEDDING_SOVEREIGNTY_04.18.2026.md §4.3`

---

## 1. CRITICAL CORRECTION — Shared Memory Budget

The handoff spec §4.3 states `float4 yards[9][9][69] = 22.2 KB`.  
**This is wrong.** The Kimi swarm confirmed the correct calculation:

```
9 lanes × 9 banks × 69 slots × 16 bytes/float4
= 5,589 float4s × 16 bytes
= 89,424 bytes = 87.3 KB
```

The 22.2 KB figure treated each slot as 4 bytes (one float), not 16 bytes (float4).  
The 22.2 KB is what `float yards[9][9][69]` (scalar slots) would cost.

**Full per-block shared budget:**
| Region | Size |
|--------|------|
| `float4 yards[9][9][69]` | 87,264 B |
| `uint8_t sp[9][9]` | 81 B |
| `uint8_t active_bank[9]` | 9 B |
| `uint32_t error_code[9]` | 36 B |
| `uint32_t scalar_idx[9]` | 36 B |
| `uint32_t vector_idx[9]` | 36 B |
| Tier 3: `float mat_scratch[9][16]` | 576 B |
| **Total (Tier 2)** | **~87.4 KB** |
| **Total (Tier 3)** | **~88.0 KB** |

**sm_86 occupancy consequence:**
- 1 block per SM (87.4 KB > 82 KB = half of 164 KB).
- This is acceptable for sovereign hot-path where latency per program matters more than SM saturation.
- If 2-block occupancy is needed, switch to `float yards[9][9][69]` (22 KB) — but this breaks the float4 ABI and halves per-slot precision. That is a design decision requiring a new spec from Daniel.

**Decision for Codex**: Proceed with `float4` as specified. Accept 1-block/SM occupancy. Tier 3 advanced physics is never expected to fill all 46 SMs simultaneously.

---

## 2. Bank-Conflict Analysis — sm_86 (32 banks, 4 B wide)

### 2.1 Address Formula

For `yards[lane][bank][slot]` (using C row-major layout):

```
byte_offset  = (lane * 9 * 69 + bank * 69 + slot) * 16
first_float_bank = (byte_offset / 4) % 32
             = ((lane * 621 + bank * 69 + slot) * 4) % 32
             = (lane * 2484 + bank * 276 + slot * 4) % 32
```

Reducing modulo 32:
- `2484 mod 32 = 20`  → lane stride = 20
- `276 mod 32 = 20`   → bank stride = 20
- `4`                 → slot stride = 4

So: **`first_float_bank = (20 * lane + 20 * bank + 4 * slot) % 32`**

### 2.2 Key Observation: Banks 0 and 8 Are Hardware Aliases

The combined term `20 * (lane + bank)`:

```
20 * 0 mod 32 =  0
20 * 1 mod 32 = 20
20 * 2 mod 32 =  8  (40 - 32)
20 * 3 mod 32 = 28  (60 - 32)
20 * 4 mod 32 = 16  (80 - 64)
20 * 5 mod 32 =  4
20 * 6 mod 32 = 24
20 * 7 mod 32 = 12
20 * 8 mod 32 =  0  ← ALIAS of value 0 (= 160 - 160)
```

`gcd(20, 32) = 4`, so the sequence repeats with period `32/4 = 8`. Logical bank indices 0 and 8 produce the **same hardware bank address** for any fixed `(lane, slot)`.

**Practical consequence**: `yards[lane][0][s]` and `yards[lane][8][s]` map to the same hardware bank. Simultaneous access to both (e.g., a `YARD_TRANSFER` between bank 0 and bank 8) will cause a **2-way bank conflict** (one stall cycle) even though the logical addresses are distinct.

### 2.3 Per-Opcode Conflict Analysis

| Opcode | Access Pattern | Conflict Assessment |
|--------|---------------|---------------------|
| `YARD_SELECT` | One read from active bank sp-1, head-thread only | **Zero conflicts.** Single thread, single access. |
| `YARD_PUSH_BANK` | One write to yard[bank_id][sp] | **Zero conflicts.** Single access. |
| `YARD_POP_BANK` | One read from yard[bank_id][sp-1] | **Zero conflicts.** Single access. |
| `YARD_PEEK_ADDR` | One read from yard[bank_id][slot_id] | **Zero conflicts.** Random-access read, single access. |
| `YARD_TRANSFER` n≤4 | Sequential head-thread loop over slots | **Zero conflicts.** Sequential single accesses. |
| `YARD_TRANSFER` n>4 with `\|src-dst\|=8` | Any concurrent access to banks 0+8 | **2-way conflict** (one stall). Serialize. |
| `YARD_SP` | Read of `sp[bank_id]` (uint8, not float4) | **Zero conflicts.** sp array is tiny, no contention. |
| `YARD_CLEAR` | Write `sp[bank_id] = 0` | **Zero conflicts.** |

### 2.4 Slot-Level Bank Cycling

For fixed `(lane, bank)`, consecutive slots hit banks `{base, base+4, base+8, base+12, base+16, base+20, base+24, base+28, base (slot 8 aliases slot 0), ...}`.

The cycle period is **8 slots**. If the head-thread sequentially accesses `n` consecutive slots, there are zero intra-access conflicts (one thread, one access at a time). If 8 threads cooperatively load 8 consecutive slots (Oct-thread pattern), they hit 8 distinct hardware banks — also zero conflicts.

### 2.5 Cross-Lane Analysis

Each of the 9 lanes is a separate warp (`threadIdx.y`). Bank conflicts are defined **per warp**, not across warps. Since lane `i` only reads `yards[i][*][*]`, and lane `j` only reads `yards[j][*][*]`, different lanes never conflict regardless of their active bank.

Even when `(lane_i + bank_i) mod 8 == (lane_j + bank_j) mod 8` (same hardware bank group), the warps execute on separate execution units and their shared-memory requests are arbitrated by the SM's shared memory controller — this is **serialization**, not warp divergence, and only occurs if the SM schedules both warps simultaneously to the same bank group. Given the head-thread pattern (31 of 32 threads are idle), the SM will typically pipeline these without conflict.

---

## 3. Why 9 Yards (Not 8 or 16)

| Option | Rationale for rejection |
|--------|------------------------|
| 8 yards | Power-of-two tempting but breaks Tesla 3-6-9 symmetry. 8 yards cannot isolate Newton iteration, pressure projection, and trace log simultaneously while keeping a spare register bank. |
| 16 yards | Doubles the `20*(lane+bank)` stride frequency. Bank-alias period shrinks: `gcd(20, 32)=4`, period=8. With 16 logical banks, banks 0 and 8 alias, 1 and 9 alias, etc. — same collision geometry, more wasted banks. Memory cost: 16 banks × 69 × 16 B = 155 KB per block, exceeds sm_86 limit. |
| **9 yards** | Exactly spans the alias period (0-8 covers one complete cycle of `20*k mod 32`). All 9 logical bank indices produce distinct hardware bank groups at slot 0. Isolation exactly matches the nine-chain swarm lanes. 9 × 9 × 69 = 5589 entries; depth-pressure is horizontal not vertical. |

The `gcd(20*9, 32) = gcd(180, 32) = 4` means `(20*k) mod 32` for `k ∈ {0..8}` yields 9 distinct values `{0, 20, 8, 28, 16, 4, 24, 12, 0}` — banks 0 and 8 alias (as shown above). This is the **one inescapable alias** in the 9-bank design. It is documented and mitigated by the bank-alias guard in `YARD_TRANSFER`.

---

## 4. YARD_TRANSFER Bank-Conflict Avoidance

### 4.1 Head-Thread Serialization (Implemented)

The reference kernels use the head-thread (thread 0) serialized loop for `YARD_TRANSFER`. This is conflict-free by construction (sequential single-thread accesses) and avoids the warp-cooperative complexity.

For `n ≤ 16`: one staging buffer of 16 StackValues in registers, one pass.
For `n > 16`: chunked 16-at-a-time with register staging.

### 4.2 Bank-Alias Guard

When `src_bank` and `dst_bank` satisfy `(20*(lane+src_bank) % 32) == (20*(lane+dst_bank) % 32)`, both banks map to the same hardware bank group. This occurs when:

```
20 * (src_bank - dst_bank) ≡ 0 (mod 32)
⟺ (src_bank - dst_bank) ≡ 0 (mod 8)
⟺ src_bank == dst_bank  OR  |src_bank - dst_bank| == 8
```

In the 9-bank design (indices 0-8), the only alias pair is `(0, 8)`. The reference kernels already serialize YARD_TRANSFER; the bank-alias guard is relevant if Codex later introduces warp-cooperative transfer. If that optimization is pursued, insert:

```cuda
// Before warp-cooperative transfer:
bool alias = ((scalar_to_bank(sf) - scalar_to_bank(df) + 9) % 9 == 0 ||
              (scalar_to_bank(sf) - scalar_to_bank(df) + 9) % 9 == 8);
if (alias || n <= 4) {
    /* serialize via head-thread loop */
} else {
    /* 8-thread cooperative copy */
}
```

### 4.3 8-Thread Cooperative Pattern (Future Optimization)

Kimi swarm analysis recommends 8-thread cooperative copy for `n > 4` when no alias exists. The pattern:

```cuda
// Only first 8 threads of the warp participate
if (threadIdx.x < 8) {
    int slot = base + threadIdx.x;
    if (slot < n) {
        // src and dst are different non-alias banks → 0 conflicts
        yards[lane][dst][dst_sp + slot] = yards[lane][src][src_sp - 1 - slot];
    }
}
__syncwarp(0xFF);  // sync first 8 threads only
```

This is NOT implemented in the reference kernels (head-thread serialization is simpler and correct). Codex should implement the cooperative version as a performance optimization after the kernel is functional.

---

## 5. Warp-Divergence Analysis for YARD_SELECT

### 5.1 Intra-warp divergence: NONE

Within a lane-warp (32 threads with the same `threadIdx.y`):
- All 32 threads execute the same opcode stream (SPMD).
- The RPN executor runs on thread 0 only (head-thread pattern).
- `YARD_SELECT` pops a scalar from shared memory (same address for all threads in the warp), writes to `active_bank[lane]` (same address for all threads in the warp).
- No conditional branch based on per-thread data.
- **Conclusion**: zero divergence.

### 5.2 Inter-lane "divergence": NOT DIVERGENCE

9 lanes = 9 warps. Different warps may have different `active_bank[lane]` values. This is **independent warp scheduling**, not warp divergence. The hardware warp scheduler issues each warp's instruction stream independently. There is no penalty analogous to SIMT-mask predication.

### 5.3 __syncwarp() vs __syncthreads() discipline

| Barrier | When to use | Cost |
|---------|-------------|------|
| `__syncwarp()` | After head-thread writes shared memory that other threads in the same warp will read. Example: after `YARD_SELECT` writes `active_bank[lane]`, before any thread other than 0 reads it. | ~2-4 cycles |
| `__syncthreads()` | Cross-lane: after block-level init (before token loop), at write-back. Required because threads from different warps (different lanes) must see consistent yard state. | ~20-40 cycles |
| Neither | Within the head-thread-only opcode loop (no other thread reads state between opcodes) | 0 cycles |

The reference kernels use:
1. `__syncthreads()` after block-level init (before token loop) — mandatory.
2. No sync inside the token loop (thread 0 is the sole writer and reader).
3. `__syncthreads()` at the top of the write-back section — mandatory for all 9 head-threads to agree on final state before global write.

If Codex adds cooperative operations inside the loop (e.g., warp-reduce for YARD_TRANSFER), add `__syncwarp()` after each cooperative section, before thread 0 updates `sp`.

---

## 6. Opcode Collision Check (0x170-0x177 vs Registry)

The spec reserves 0x170-0x17F. Verified against `RPN_DOMAIN_OPCODE_REGISTRY.md` and `rpn_opcodes.py`:

| Range | Status |
|-------|--------|
| 0x00-0x16F | Existing opcodes (arithmetic, ternary, CBR, physics, draw, temporal) |
| **0x170-0x176** | **NEW — YARD_* opcodes (this spec)** |
| 0x177-0x17F | Reserved for `YARD_FOLD`, `YARD_SIMD_MAP` (future) |
| 0x180-0x18F | WINE_* I/O contract — NO COLLISION |
| 0x190 | PHYSICS_EMIT_VISUAL — NO COLLISION |
| 0x1A0-0x1A8 | Tier 3 advanced ops (reference kernel, not yet in registry) |

**Codex action required**: register 0x170-0x176 in `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` and `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` before landing.

The Tier 3 opcodes 0x1A0-0x1A8 are in the reference kernel but NOT yet in the opcode registry. These need a separate registry entry in a new spec batch. For now, Codex can land them as internal constants in the kernel without adding to the shared registry.

---

## 7. Uncertainties for Codex to Resolve

1. **float4 vs float ABI decision**: The spec says `float4 yards[9][9][69]` but 87.4 KB forces 1-block/SM occupancy. If Daniel wants 2-block occupancy, switch to `float yards[9][9][69]` (22 KB) but all StackValue pushes/pops must be rewritten to use scalar slots. This is a design decision that needs explicit sign-off, not an implementation choice.

2. **Matrix row encoding**: Tier 3 uses `StackValue` (4 floats) as matrix rows, with `w` field repurposed as column 3. This means the `is_vector()` check (w == 1.0) cannot distinguish a mat4 row from a vector. Codex should define a matrix-row tag value (e.g., `w = 2.0` for mat4 rows) and update `make_vec4` / `push_mat4` / `pop_mat4` accordingly before landing.

3. **Newton opcode is single-step**: `OP_NEWTON_ITER` does one Newton-Raphson step. For convergence, the caller must wrap it in a `BRANCH / LOOP` RPN program and re-evaluate `f(x_new)` externally. This is intentional (the kernel cannot call the user's function), but Codex should document this in the bridge-level API so callers know to construct the loop.

4. **Tier 3 opcodes 0x1A0-0x1A8 in registry**: Not yet registered. Before landing, either extend the registry in a follow-up spec, or land these as kernel-internal defines. Do not collide with future WINE/PHYSICS expansions.

5. **YARD_TRANSFER n > 16 staging buffer**: The reference kernels stage in a register buffer of 16 StackValues per chunk. On sm_86, a StackValue is 16 bytes = 4 registers. 16 × 4 = 64 registers for the staging buffer alone. RTX 3070 has 65536 registers per SM, split across resident warps. With 1 block/SM (9 warps), each warp can use up to ~7000 registers. 64 registers for staging is safe, but Codex should run `nvcc --ptxas-options=-v` to confirm actual register usage.

---

## 8. Summary Verification Checklist

Before Codex lands these kernels:

```bash
# 1. Shared memory matches spec (corrected: float4 = 87 KB)
grep -n "float4 yards\[9\]\[9\]\[69\]" knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu  # 1 hit

# 2. sp array is uint8_t (not int32 — saves 252 bytes)
grep -n "uint8_t  sp\[9\]\[9\]" knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu         # 1 hit

# 3. New opcodes in registry
grep -nE "YARD_(SELECT|PUSH_BANK|POP_BANK|PEEK_ADDR|TRANSFER|SP|CLEAR)" docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md  # 7 hits

# 4. No LIFO fallback
grep -n "modular_rpn_kernel_lite\|legacy\|fallback" knowledge3d/cranium/bridges/lightweight_rpn.py     # 0 hits

# 5. No Python TransferYardStack
grep -n "class TransferYardStack" knowledge3d/cranium/bridges/transfer_yard_tiered.py                   # 0 hits

# 6. No numpy in cranium
grep -rn "import numpy\|np\.random" knowledge3d/cranium/                                                # 0 hits

# 7. Compile check
nvcc -arch=sm_86 -ptx knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu \
     -o knowledge3d/cranium/ptx/modular_rpn_kernel_transfer_yard.ptx

nvcc -arch=sm_86 -ptx knowledge3d/cranium/kernels/advanced_rpn_kernel_transfer_yard.cu \
     -o knowledge3d/cranium/ptx/advanced_rpn_kernel_transfer_yard.ptx
```
