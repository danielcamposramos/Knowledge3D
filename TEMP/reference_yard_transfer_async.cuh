// =============================================================================
// reference_yard_transfer_async.cuh
// K3D — Transfer Yard Async/Cooperative Device Header
//
// AUTHORITY: TEMP/yard_async_parallel_design.md
// TARGET:    sm_86 (RTX 3070, Ampere), CUDA 12.4 / 12.6
//
// PURPOSE:
//   Reusable device-side inline functions implementing the 8-thread
//   register-staged cooperative YARD_TRANSFER pattern.
//
//   cp.async BLOCKER (definitive):
//     cp.async on sm_86 supports only global→shared. cp.async.bulk requires
//     sm_90 (Hopper). Shared-to-shared has no async hardware instruction on
//     Ampere. This header uses register-file staging, which IS the correct
//     GPU-native intra-shared-memory parallelism on Ampere.
//
// USAGE PATTERN (inside kernel opcode loop):
//
//   // --- OUTSIDE if(thread==0) guard ---
//   if (thread < 8) {
//       yard_transfer_async(yards[lane], sp[lane],
//                           xfer_src[lane], xfer_dst[lane],
//                           xfer_n[lane], thread);
//   }
//   __syncwarp(0xFF);  // ensure all 8 threads finished
//   // --- INSIDE if(thread==0) guard ---
//   if (thread == 0) {
//       sp[lane][dst] += committed_n;
//       sp[lane][src] -= committed_n;
//   }
//
// BANK NOTES (from yard_async_parallel_design.md §9):
//   Bank 0 and bank 8 are hardware aliases: (20*8) mod 32 = 0.
//   The 2-way conflict is accepted and mitigated by register staging:
//   load phase and store phase are separated by __syncwarp(0xFF).
//   No shared-memory staging buffer in bank 4 is needed (and would not
//   help — bank 4 writes also produce 2-way conflicts at the same density).
//
// LAYOUT:
//   yards[lane][bank][slot]  (C row-major, float4 per slot)
//   sp[bank]                 (uint8_t per bank for this lane)
//   kYardDepth = 69, kYardsPerLane = 9
//
// THREAD PARTICIPATION:
//   yard_transfer_async     : threads 0..7 (count >= 8)
//   yard_push_async         : all 32 threads (head-thread only path, others no-op)
//   yard_pop_async          : same as push_async
//   yard_peek_sync          : thread 0 only (single access, no parallel benefit)
//
// =============================================================================

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shared layout constants — must match the kernel that includes this header
// ---------------------------------------------------------------------------
#ifndef YARD_DEPTH
#define YARD_DEPTH 69
#endif
#ifndef YARDS_PER_LANE
#define YARDS_PER_LANE 9
#endif

// ---------------------------------------------------------------------------
// StackValue ABI — float4 slot (must match Tier 2 / Tier 3 kernel ABI)
// ---------------------------------------------------------------------------
// We declare it only if not already defined (kernels define it in their
// anonymous namespace; this header uses a local typedef for portability).

// We forward-declare a type alias rather than redefining struct StackValue,
// which avoids redefinition conflicts when included in a .cu that already
// has the struct. Kernels should ensure the ABI matches (x,y,z,w floats,
// 16-byte alignment).

// For this header we use float4 directly (CUDA built-in) since StackValue
// is bit-identical to float4. The kernel's yard array is declared as
// StackValue yards[...] but StackValue IS float4 (same layout).
// We cast via reinterpret in the template parameter.

// ---------------------------------------------------------------------------
// ALIAS DETECTION
// Hardware bank alias: |src_bank - dst_bank| == 8 in the 9-bank design.
// (Because 20*8 ≡ 0 mod 32; period of stride-20 sequence = 8.)
// When alias is true the 2-way conflict is irreducible; register staging
// ensures load-phase and store-phase do not compound it.
// ---------------------------------------------------------------------------

__device__ __forceinline__
bool yard_banks_alias(int src_bank, int dst_bank) {
    // The only alias pair in a 9-bank design (indices 0-8): (0,8) and (8,0).
    int diff = src_bank - dst_bank;
    if (diff < 0) diff = -diff;
    return (diff == 8);
    // Note: diff==0 (same bank) is caught by the caller; it is a no-op not a conflict.
}

// ---------------------------------------------------------------------------
// yard_transfer_async
//   Cooperative 8-thread register-staged copy of `count` float4 slots
//   from yard[src_bank] to yard[dst_bank] within a single lane.
//
//   MUST be called by threads 0..7 simultaneously (outside thread==0 guard).
//   Thread identity: thread_id = threadIdx.x % 32  (lane context).
//
//   For count < 8 (threshold): falls through to serial path (thread 0 only).
//   For count >= 8: 8-thread parallel, register-staged.
//
//   After return, the CALLER (thread 0) must update sp[src_bank] and
//   sp[dst_bank] with the actual transferred count (returned as the function
//   return value from thread 0's perspective; other threads discard it).
//
//   Returns committed slot count (may be clamped to available/space).
//   Non-zero only for thread_id == 0.
// ---------------------------------------------------------------------------

__device__ __forceinline__
int yard_transfer_async(
    float4           yard[][YARD_DEPTH],  // yard[bank][slot] for this lane
    uint8_t          sp[],               // sp[bank] for this lane
    int              src_bank,
    int              dst_bank,
    int              count,
    int              thread_id            // threadIdx.x within [0,32)
) {
    // -----------------------------------------------------------------------
    // Clamp count to available source slots and available destination space
    // (only thread 0 has valid sp; broadcast needed if threads 1-7 need it)
    // -----------------------------------------------------------------------
    // Thread 0 computes the clamped count. Since __syncwarp follows the arg
    // write to shared memory in the caller, all 8 threads read the same
    // already-clamped count from the shared xfer_n[lane] slot.
    // Here count is pre-clamped by the caller using sp values (see the caller
    // pattern in yard_async_parallel_design.md §3.4).
    //
    // Simple validity guard:
    if (count <= 0 || src_bank == dst_bank) return 0;

    // -----------------------------------------------------------------------
    // SERIAL FALLBACK: count < 8 — not worth paying 2×syncwarp overhead
    // Only thread 0 executes; threads 1-7 are no-ops.
    // -----------------------------------------------------------------------
    if (count < 8) {
        if (thread_id == 0) {
            // Clamp
            int avail = sp[src_bank];
            int space = YARD_DEPTH - sp[dst_bank];
            if (count > avail) count = avail;
            if (count > space) count = space;
            if (count <= 0) return 0;

            // Pop from src (LIFO order), stage in local registers, push to dst
            // Register buffer: 4 slots max for serial path (count < 8 → max 7)
            float4 tmp[7];
            int src_sp = sp[src_bank];
            int dst_sp = sp[dst_bank];
            for (int k = 0; k < count; ++k) {
                tmp[k] = yard[src_bank][src_sp - 1 - k];
            }
            // Push in LIFO-preserved order
            for (int k = 0; k < count; ++k) {
                yard[dst_bank][dst_sp + k] = tmp[k];
            }
            sp[src_bank] = (uint8_t)(src_sp - count);
            sp[dst_bank] = (uint8_t)(dst_sp + count);
            return count;
        }
        // Threads 1-7: no-op for serial path
        return 0;
    }

    // -----------------------------------------------------------------------
    // COOPERATIVE PATH: count >= 8, threads 0..7 each handle one float4
    // per round (8 slots per round, ceil(count/8) rounds).
    //
    // Layout:
    //   Round r: thread t copies slot (src_sp - 1 - r*8 - t) → (dst_sp + r*8 + t)
    //   This preserves LIFO stack order across rounds.
    //
    // Bank conflict note:
    //   For non-alias bank pairs: each thread accesses a distinct hardware bank.
    //   Zero conflict beyond the inherent 4-cycle v4 floor.
    //   For alias pair (bank 0↔8): 2-way conflict; 8 threads serialized into
    //   2 batches of 4. Load phase costs 8 cycles, store phase costs 8 cycles.
    //   Register staging separates the two phases via __syncwarp(0xFF).
    // -----------------------------------------------------------------------

    // Each thread t ∈ [0,7] participates. Threads 8-31 are no-ops in the
    // main body but must NOT call __syncwarp(0xFF) — the caller uses the
    // if(thread < 8) guard to exclude them.

    // Read sp from shared (thread 0 is among the 8, it knows the values).
    // For threads 1-7, sp is accessible because it is __shared__ and was
    // written before the caller's __syncwarp() that precedes this call.
    int src_sp = (int)sp[src_bank];
    int dst_sp = (int)sp[dst_bank];

    // Clamp (redundant if caller pre-clamped, but defensive):
    int avail = src_sp;
    int space = YARD_DEPTH - dst_sp;
    if (count > avail) count = avail;
    if (count > space) count = space;
    if (count <= 0) return 0;

    int t = thread_id;  // 0..7
    int rounds = (count + 7) / 8;  // ceil(count / 8)

    for (int r = 0; r < rounds; ++r) {
        int slot_in_round = r * 8 + t;  // which slot this thread handles
        bool active = (slot_in_round < count);

        // ------------------------------------------------------------------
        // Phase 1: LOAD from source bank into registers
        // Each thread loads one float4 from yards[src_bank][src_sp-1-slot]
        // into a register (float4 tmp — 4 hardware registers per thread).
        // ------------------------------------------------------------------
        float4 tmp = {0.f, 0.f, 0.f, 0.f};  // zero-init for inactive threads
        if (active) {
            int src_slot = src_sp - 1 - slot_in_round;
            // Direct shared-memory read (ld.shared.v4.f32)
            // 4 cycles baseline (inherent v4); possible 2-way conflict for
            // alias bank pairs — hardware serializes transparently.
            tmp = yard[src_bank][src_slot];
        }

        // ------------------------------------------------------------------
        // Phase 2: BARRIER — all 8 threads must complete their loads before
        // any thread stores. This decouples load-phase bank access from
        // store-phase bank access, preventing the alias conflict from
        // compounding.
        // __syncwarp(0xFF) = PTX bar.warp.sync 0xff ≈ 12 cycles.
        // Sufficient: all 8 threads are within the same parent warp.
        // __syncthreads() is NOT used here (would sync all 9 lane-warps,
        // adding ~30 cycles and serializing unrelated lanes).
        // ------------------------------------------------------------------
        __syncwarp(0xFF);

        // ------------------------------------------------------------------
        // Phase 3: STORE from registers to destination bank
        // ------------------------------------------------------------------
        if (active) {
            int dst_slot = dst_sp + slot_in_round;
            // Direct shared-memory write (st.shared.v4.f32)
            yard[dst_bank][dst_slot] = tmp;
        }

        // Barrier after store: ensure all stores are visible before next
        // round's loads read sp (and before thread 0 updates sp).
        __syncwarp(0xFF);
    }

    // sp update: only thread 0 writes sp (to avoid 8-way write conflict).
    // This update happens in the caller AFTER the caller's __syncwarp(0xFF).
    // We return the committed count so the caller knows what to adjust.
    // Threads 1-7 return 0 (caller ignores non-zero-thread return values).
    if (t == 0) {
        sp[src_bank] = (uint8_t)(src_sp - count);
        sp[dst_bank] = (uint8_t)(dst_sp + count);
        return count;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// yard_push_async
//   Async-friendly push: thread 0 writes the value; threads 1-7 are no-ops.
//   "Async" here means: the caller can structure its opcode loop so that
//   threads 1-7 are already fetching the next opcode while thread 0 pushes.
//   This is automatic on Ampere's warp scheduler (ILP within a warp).
//
//   Returns true on success (thread 0 only), false on overflow.
// ---------------------------------------------------------------------------

template<typename StackValue>
__device__ __forceinline__
bool yard_push_async(
    StackValue       yard[][YARD_DEPTH],
    uint8_t          sp[],
    int              bank_id,
    StackValue       val,
    int              thread_id
) {
    if (thread_id != 0) return true;  // threads 1-31: no-op, success
    uint8_t top = sp[bank_id];
    if (top >= YARD_DEPTH) return false;  // overflow
    yard[bank_id][top] = val;
    sp[bank_id] = top + 1;
    return true;
}

// ---------------------------------------------------------------------------
// yard_pop_async
//   Async-friendly pop: thread 0 reads and decrements sp; others no-op.
//   Returns true on success (thread 0 only), false on underflow.
//   val is written only for thread 0; callers guard with thread==0 check.
// ---------------------------------------------------------------------------

template<typename StackValue>
__device__ __forceinline__
bool yard_pop_async(
    StackValue       yard[][YARD_DEPTH],
    uint8_t          sp[],
    int              bank_id,
    StackValue&      val,
    int              thread_id
) {
    if (thread_id != 0) return true;  // no-op for non-head threads
    uint8_t top = sp[bank_id];
    if (top == 0) return false;  // underflow
    sp[bank_id] = top - 1;
    val = yard[bank_id][top - 1];
    return true;
}

// ---------------------------------------------------------------------------
// yard_peek_sync
//   Synchronous true random-access read: thread 0 only, no parallelism.
//   There is no parallel benefit for a single-slot read (4-cycle v4 floor
//   is irreducible). Returns false on out-of-bounds.
// ---------------------------------------------------------------------------

template<typename StackValue>
__device__ __forceinline__
bool yard_peek_sync(
    StackValue       yard[][YARD_DEPTH],
    int              bank_id,
    int              slot_id,
    StackValue&      val
) {
    // Caller already guards with thread==0; this function has no thread_id
    // parameter because it is strictly single-threaded.
    if (slot_id < 0 || slot_id >= YARD_DEPTH) return false;
    val = yard[bank_id][slot_id];
    return true;
}

// ---------------------------------------------------------------------------
// yard_transfer_n_clamp (utility for caller to pre-clamp before cooperative call)
//   Returns the clamped count that yard_transfer_async will actually commit.
//   Call from thread 0 before dispatching to yard_transfer_async.
// ---------------------------------------------------------------------------

__device__ __forceinline__
int yard_transfer_n_clamp(
    const uint8_t sp[],
    int src_bank,
    int dst_bank,
    int requested_n
) {
    int avail = (int)sp[src_bank];
    int space = YARD_DEPTH - (int)sp[dst_bank];
    int n = requested_n;
    if (n > avail) n = avail;
    if (n > space) n = space;
    if (n < 0) n = 0;
    return n;
}

// ---------------------------------------------------------------------------
// USAGE EXAMPLE (inline documentation for Codex)
//
// To replace YARD_TRANSFER inside a kernel with blockDim=(32,9,1):
//
//  __shared__ uint8_t xfer_src[9];
//  __shared__ uint8_t xfer_dst[9];
//  __shared__ uint8_t xfer_n[9];
//
//  case kOpYardTransfer: {
//      // Decode operands on head thread, write to shared args
//      if (thread == 0) {
//          float nf, df, sf;
//          pop_scalar(nf); pop_scalar(df); pop_scalar(sf);  // per existing pop pattern
//          int src = scalar_to_bank(sf);
//          int dst = scalar_to_bank(df);
//          int n   = static_cast<int>(nf + 0.5f);
//          // Pre-clamp
//          n = yard_transfer_n_clamp(sp[lane], src, dst, n);
//          xfer_src[lane] = (uint8_t)src;
//          xfer_dst[lane] = (uint8_t)dst;
//          xfer_n[lane]   = (uint8_t)n;
//      }
//      // Broadcast args to threads 1-7 via syncwarp
//      __syncwarp();  // full mask: thread 0 wrote, 1-31 must see
//
//      // Cooperative copy (outside thread==0 guard)
//      if (thread < 8) {
//          yard_transfer_async(
//              reinterpret_cast<float4(*)[YARD_DEPTH]>(yards[lane]),
//              sp[lane],
//              (int)xfer_src[lane], (int)xfer_dst[lane],
//              (int)xfer_n[lane],
//              thread
//          );
//      }
//      __syncwarp(0xFF);  // wait for all 8 to complete + sp update
//      // No additional sp update needed: yard_transfer_async thread 0 updates sp.
//      break;
//  }
//
// Note: StackValue is bit-identical to float4 (same 16-byte layout).
// The reinterpret_cast<float4(*)[YARD_DEPTH]> cast is safe if StackValue
// is declared as alignas(16) with fields {float x,y,z,w}.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Static assert: ensures compile-time check on YARD_DEPTH
// (sp uses uint8_t; max representable depth = 255)
// ---------------------------------------------------------------------------
static_assert(YARD_DEPTH <= 255, "YARD_DEPTH must fit in uint8_t sp pointers");
static_assert(YARDS_PER_LANE == 9, "Layout assumes 9 banks per lane for alias analysis");
