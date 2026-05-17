# Async YARD_TRANSFER Patch Guide

**Date**: 2026-04-18
**Author**: Claude (architecture lane)
**For**: Codex — estimated implementation time: 25-35 minutes
**Prerequisites**:
- Read `yard_async_parallel_design.md` (15 min) — understand the pattern
- Both reference kernels already read and understood
- Header `reference_yard_transfer_async.cuh` written to `TEMP/` — copy to
  `knowledge3d/cranium/cuda/yard_transfer_async.cuh` before applying patches

---

## Step 0 — Copy the Header

```bash
cp TEMP/reference_yard_transfer_async.cuh \
   knowledge3d/cranium/cuda/yard_transfer_async.cuh
```

Include it in both kernel sources at the top (after existing includes):

```cuda
// ADD THIS LINE to both .cu files:
#include "knowledge3d/cranium/cuda/yard_transfer_async.cuh"
```

---

## Step 1 — Add Shared Args Arrays to Both Kernels

**Location:** Immediately after the existing `__shared__` declarations, before `__syncthreads()`.

**Patch (identical for Tier 2 and Tier 3):**

```diff
 __shared__ uint32_t   scalar_idx[kLanesPerBlock];
 __shared__ uint32_t   vector_idx[kLanesPerBlock];
+    // Cooperative YARD_TRANSFER argument channels (one slot per lane, 27 bytes total)
+    __shared__ uint8_t xfer_src[kLanesPerBlock];
+    __shared__ uint8_t xfer_dst[kLanesPerBlock];
+    __shared__ uint8_t xfer_n[kLanesPerBlock];
```

**File:** `reference_modular_rpn_kernel_transfer_yard.cu` line 185
**File:** `reference_advanced_rpn_kernel_transfer_yard.cu` line 371

---

## Step 2 — Replace YARD_TRANSFER Case in Tier 2 Kernel

**File:** `reference_modular_rpn_kernel_transfer_yard.cu`

**Remove:** lines 577–615 (the entire `case kOpYardTransfer:` block inside `if (thread == 0)`)

**Replace with** (this block spans OUTSIDE the `if(thread==0)` guard — see structural
note in `yard_async_parallel_design.md §3.4`):

```diff
-            case kOpYardTransfer: {
-                // 0x174  YARD_TRANSFER
-                // Stack layout: [..., src_bank, dst_bank, n_slots]  (n_slots on top)
-                // Move top-n from src to dst within the same lane.
-                // Order: pop from src, push to dst, preserving LIFO order.
-                float n_f; if (!pop_scalar(n_f)) break;
-                float dst_f; if (!pop_scalar(dst_f)) break;
-                float src_f; if (!pop_scalar(src_f)) break;
-                int src = scalar_to_bank(src_f);
-                int dst = scalar_to_bank(dst_f);
-                int n   = static_cast<int>(n_f + 0.5f);
-
-                // Clamp n to available slots and available dst space
-                int avail_src = LANE_SP(src);
-                int avail_dst = kYardDepth - LANE_SP(dst);
-                if (n > avail_src) n = avail_src;
-                if (n > avail_dst) n = avail_dst;
-
-                // Transfer: copy into a temporary register window, then push to dst.
-                // We stage into a small on-register buffer (max 16 slots staged at once)
-                // to preserve stack order.  For n > 16, iterate in chunks.
-                StackValue tmp[16];
-                int remaining = n;
-                while (remaining > 0 && LANE_ERROR == kErrorNone) {
-                    int chunk = (remaining < 16) ? remaining : 16;
-                    // Pop chunk items from src (they come out LIFO → reversed order)
-                    for (int k = chunk - 1; k >= 0; --k) {
-                        yard_pop(src, tmp[k]);
-                    }
-                    // Push in original order to dst
-                    for (int k = 0; k < chunk; ++k) {
-                        yard_push(dst, tmp[k]);
-                    }
-                    remaining -= chunk;
-                }
-                break;
-            }
```

**Replace with** (place this block at the same switch position, but note the structural
change: this case MUST be handled by breaking out of the `if(thread==0)` dispatch loop
and using a two-phase pattern. The cleanest implementation is a goto-free restructuring
of the dispatch loop — see "Structural Option B" below if a full refactor is out of scope):

```diff
+            case kOpYardTransfer: {
+                // 0x174  YARD_TRANSFER — 8-thread cooperative register-staged copy.
+                // Phase 1: thread 0 decodes operands, writes to shared args channel.
+                float n_f; if (!pop_scalar(n_f)) break;
+                float dst_f; if (!pop_scalar(dst_f)) break;
+                float src_f; if (!pop_scalar(src_f)) break;
+                int src = scalar_to_bank(src_f);
+                int dst = scalar_to_bank(dst_f);
+                int n   = yard_transfer_n_clamp(sp[lane], src, dst,
+                              static_cast<int>(n_f + 0.5f));
+                xfer_src[lane] = (uint8_t)src;
+                xfer_dst[lane] = (uint8_t)dst;
+                xfer_n[lane]   = (uint8_t)n;
+                // Signal threads 1-7 that args are ready.
+                // FALLS THROUGH to cooperative section below (see XFER_COOP label).
+                break;
+            }
```

Then, immediately AFTER the closing `} // end thread==0 guard` and BEFORE `__syncthreads()`
write-back, add the cooperative section:

```diff
+    // =========================================================================
+    // COOPERATIVE YARD_TRANSFER — outside thread==0 guard.
+    // Threads 0-7 of each lane cooperate on YARD_TRANSFER.
+    // The opcode loop above set xfer_src/xfer_dst/xfer_n and broke out;
+    // we detect the pending transfer via a shared flag.
+    //
+    // NOTE: This approach requires tracking "last opcode was YARD_TRANSFER"
+    // in the dispatch loop.  See Structural Option A (flag) or Option B
+    // (refactored loop) in the diff guide below.
+    // =========================================================================
+    if (xfer_n[lane] > 0 && /* pending_transfer_flag[lane] */ false) {
+        __syncwarp();  // thread 0 wrote xfer_* args; 1-7 must see them
+        if (thread < 8) {
+            yard_transfer_async(
+                reinterpret_cast<float4(*)[kYardDepth]>(yards[lane]),
+                sp[lane],
+                (int)xfer_src[lane], (int)xfer_dst[lane], (int)xfer_n[lane],
+                thread
+            );
+        }
+        __syncwarp(0xFF);
+        // sp already updated inside yard_transfer_async by thread 0.
+    }
```

---

## Structural Options for Integrating Cooperative Execution

The core challenge is that the current kernel structure wraps the entire opcode loop inside
`if (thread == 0)`, then writes back. For YARD_TRANSFER we need threads 0-7 to participate
mid-loop. Two clean options:

### Option A — Pending-Transfer Flag (Minimal Diff)

Add `__shared__ bool pending_transfer[kLanesPerBlock]` initialized to false.

Inside `case kOpYardTransfer` (within `thread==0` guard): write xfer args, set
`pending_transfer[lane] = true`, break.

After the `} // end thread==0 guard`, add:

```cuda
// Cooperative YARD_TRANSFER execution (outside thread==0 guard)
if (pending_transfer[lane]) {
    __syncwarp();   // thread 0 wrote args; 1-7 see them
    if (thread < 8) {
        yard_transfer_async(
            reinterpret_cast<float4(*)[kYardDepth]>(yards[lane]),
            sp[lane],
            (int)xfer_src[lane], (int)xfer_dst[lane], (int)xfer_n[lane],
            thread
        );
    }
    __syncwarp(0xFF);
    // Thread 0 clears the flag for the next iteration
    if (thread == 0) pending_transfer[lane] = false;
}
```

**Issue:** this executes ONCE per opcode-loop iteration (i.e., after every opcode, we check
the flag). This adds one shared-memory read per opcode. At 4 cycles per flag read across
all tokens, for a typical 64-token program: 64 × 4 = 256 cycles overhead.
Acceptable for a first-pass implementation; optimize in Option B once functional.

**Shared memory cost of flag:** 1 byte × 9 lanes = 9 bytes. Negligible.

### Option B — Refactored Dispatch Loop (Preferred, ~45 min)

Lift the opcode dispatch out of the `if(thread==0)` guard. Structure the loop so that:
- Scalar opcodes: only thread 0 does work (guarded by `if(thread==0)` inside each case).
- Cooperative opcodes (YARD_TRANSFER): all 8 threads participate (guarded by `if(thread<8)`).

```cuda
for (uint32_t i = 0; i < token_count; ++i) {
    // Thread 0 fetches the opcode and broadcasts via shared memory:
    if (thread == 0) shared_opcode[lane] = op_codes[i];
    __syncwarp();   // all threads see the opcode
    uint16_t opcode = shared_opcode[lane];

    if (error_code[lane] != kErrorNone) break;

    switch (opcode) {
    case kOpYardTransfer:
        // Thread 0 decodes, all 8 participate:
        if (thread == 0) {
            // ... pop operands, clamp, write xfer_* ...
        }
        __syncwarp();
        if (thread < 8) {
            yard_transfer_async(..., thread);
        }
        __syncwarp(0xFF);
        break;

    default:
        // All scalar ops: guard with thread==0 inside
        if (thread == 0) {
            // ... existing opcode handling ...
        }
        break;
    }
}
```

This requires adding `__shared__ uint16_t shared_opcode[kLanesPerBlock]` (18 bytes).
The `__syncwarp()` on every opcode costs ~12 cycles × 64 tokens = 768 cycles per program.
For a 300-opcode physics program: 3600 cycles overhead vs the current zero. This is the
trade-off: cleaner structure but more warp-sync overhead. Option A avoids the per-opcode
sync by only paying it on YARD_TRANSFER.

**Recommendation for Codex: start with Option A (flag), verify correctness, then profile
and decide if Option B's cleanliness justifies its overhead budget.**

---

## Step 3 — Apply Same Pattern to Tier 3 Kernel

**File:** `reference_advanced_rpn_kernel_transfer_yard.cu`

**Remove:** lines 562–584 (the `case kOpYardTransfer:` block inside `if (thread == 0)`)

**Apply:** identical substitution as Step 2. The Tier 3 kernel has the same structure
(`if(thread==0)` dispatch loop). Use Option A (flag) for consistency.

The one Tier-3-specific note: `pivot_buf[lane]` is in registers, not shared memory.
This does not affect YARD_TRANSFER. The `mat_scratch[lane]` shared region (576 bytes)
adds no bank conflict for yard operations (disjoint address range).

---

## Step 4 — Add YARD_TRANSFER Args to Shared Memory Budget Comment

Update the shared memory budget comment in both kernels:

**Tier 2:**
```diff
-//   Total                                              ≈ 22,275 bytes  (< 100 KB Ampere budget)
+//   float4  yards[9][9][69]                            = 87,264 bytes
+//   uint8_t sp[9][9]                                   =     81 bytes
+//   uint8_t active_bank[9] + error_code[9]×4           =     45 bytes
+//   uint32_t scalar_idx[9] + vector_idx[9]             =     72 bytes
+//   uint8_t xfer_src[9], xfer_dst[9], xfer_n[9]        =     27 bytes
+//   bool pending_transfer[9]                           =      9 bytes
+//   ─────────────────────────────────────────────────────────────────
+//   Total (Tier 2 + async)                             ≈ 87,498 bytes  (~85.4 KB)
+//   sm_86 limit with 1 block/SM: 164 KB.  1 block/SM confirmed.
```

---

## Step 5 — Verify Compilation

```bash
nvcc -arch=sm_86 -std=c++17 -ptx \
  -I knowledge3d/cranium/cuda \
  knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu \
  -o knowledge3d/cranium/ptx/modular_rpn_kernel_transfer_yard.ptx

nvcc -arch=sm_86 -std=c++17 -ptx \
  -I knowledge3d/cranium/cuda \
  knowledge3d/cranium/kernels/advanced_rpn_kernel_transfer_yard.cu \
  -o knowledge3d/cranium/ptx/advanced_rpn_kernel_transfer_yard.ptx
```

Expected: zero errors, zero warnings about `__syncwarp` misuse.

Check register pressure:
```bash
nvcc -arch=sm_86 --ptxas-options=-v \
  knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu 2>&1 | grep "registers"
# Expected: < 100 registers/thread (headroom: 65536 regs / 288 threads = 227 max)
# The 8 float4 staging registers per thread = 32 registers; well within limit.
```

---

## Step 6 — Verify Correctness (Mini Test Program)

Before wiring into the bridge, validate the cooperative copy with a standalone test:

```cuda
// test_yard_transfer_async.cu — standalone validation
// Launch: <<<1, dim3(32,9,1)>>>
// Verify: after YARD_TRANSFER of n=16 slots from bank 0 to bank 2,
//         sp[0] decremented by 16, sp[2] incremented by 16,
//         contents preserved in correct LIFO order.
```

Expected output per lane:
- `sp[0]` decreases by `min(16, initial_sp[0])`
- `sp[2]` increases by same amount
- `yards[lane][2][new_sp2 - k] == yards_old[lane][0][old_sp0 - 1 - k]` for k in [0, n)

---

## Summary of Changes (grep-verifiable after apply)

```bash
# 1. Header included in both kernels
grep -n "yard_transfer_async.cuh" knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu  # 1 hit
grep -n "yard_transfer_async.cuh" knowledge3d/cranium/kernels/advanced_rpn_kernel_transfer_yard.cu  # 1 hit

# 2. xfer_src/dst/n shared arrays present
grep -n "xfer_src\[kLanesPerBlock\]" knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu  # 1 hit

# 3. Serial inner loop GONE
grep -n "while (remaining > 0" knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu  # 0 hits
grep -n "while (rem > 0"       knowledge3d/cranium/kernels/advanced_rpn_kernel_transfer_yard.cu  # 0 hits

# 4. Cooperative call present
grep -n "yard_transfer_async" knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu  # >= 1 hit
grep -n "__syncwarp(0xFF)"    knowledge3d/cranium/kernels/modular_rpn_kernel_transfer_yard.cu  # >= 2 hits

# 5. No __syncthreads inside opcode loop (only at init and write-back)
# Manually verify: grep -n "__syncthreads" ... → only lines before token loop and after
```

---

## What NOT to Change

- Do NOT change `float4 yards[9][9][69]` layout. Bank conflicts are accepted per spec.
- Do NOT introduce Bank-4 shared staging buffer. It does not eliminate conflicts (proven
  in `yard_async_parallel_design.md §3.2`).
- Do NOT use `cp.async` for yard transfers. It is global→shared only on sm_86.
- Do NOT use `__syncthreads()` inside the opcode loop. Full block sync kills performance.
- Do NOT change YARD_DEPTH to 72. This collapses all banks to the same hardware bank.
- Do NOT add Python fallbacks anywhere. "We fail and fix."
