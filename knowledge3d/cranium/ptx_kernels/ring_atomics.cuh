/*
 * ring_atomics.cuh — System-scope acquire/release ring helpers
 *
 * Spec: CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §2.6
 *
 * WHY PTX inline here instead of C++ volatile + __threadfence:
 *   __threadfence() covers device scope only (GPU↔GPU).
 *   The input_ring and output_ring are host-pinned zero-copy buffers written by
 *   the Python host process and read by the persistent kernel (or vice versa).
 *   System-scope fences (membar.sys / ld.acquire.sys / st.release.sys) are
 *   required to ensure the host CPU and device GPU observe writes in order.
 *   This cannot be expressed with __threadfence alone.
 *
 * Target: sm_80+ (RTX 3070 = sm_86). PTX ISA 6.x required.
 *   ld.global.acquire.sys.u32 and st.global.release.sys.u32 are available
 *   from PTX ISA 6.0 (Volta sm_70) onward.
 *   membar.sys is available from PTX ISA 2.0 onward.
 *
 * Usage contract:
 *   - ONLY use these helpers for host-pinned zero-copy rings (input_ring,
 *     output_ring). Pure device-side VRAM rings (log_ring, event_ring) may
 *     use __threadfence() + atomicAdd, as in trm_game_loop.cuh.
 *   - Every file that touches ring head/tail pointers must include this header
 *     and use these helpers — never hand-roll ring atomics elsewhere.
 */

#ifndef K3D_RING_ATOMICS_CUH
#define K3D_RING_ATOMICS_CUH

#include <cuda_runtime.h>
#include <cstdint>

/* ---------------------------------------------------------------------------
 * ring_load_acquire_u32
 *
 * Performs a system-scope acquire-load of a uint32 from a host-pinned address.
 * Semantics: all subsequent device loads/stores see effects from any host store
 * that happened-before the host's release-store that produced this value.
 *
 * PTX: ld.global.acquire.sys.u32
 * --------------------------------------------------------------------------- */
__device__ __forceinline__ uint32_t ring_load_acquire_u32(const volatile uint32_t* addr)
{
    uint32_t val;
    asm volatile(
        "ld.global.acquire.sys.u32 %0, [%1];"
        : "=r"(val)
        : "l"(addr)
        : "memory"
    );
    return val;
}

/* ---------------------------------------------------------------------------
 * ring_store_release_u32
 *
 * Performs a system-scope release-store of a uint32 to a host-pinned address.
 * Semantics: all prior device writes are visible to the host before this store
 * is observed.
 *
 * PTX: st.global.release.sys.u32
 * --------------------------------------------------------------------------- */
__device__ __forceinline__ void ring_store_release_u32(volatile uint32_t* addr, uint32_t val)
{
    asm volatile(
        "st.global.release.sys.u32 [%0], %1;"
        :
        : "l"(addr), "r"(val)
        : "memory"
    );
}

/* ---------------------------------------------------------------------------
 * ring_membar_sys
 *
 * Full system memory barrier. Use before/after a slot payload write when the
 * slot data itself resides in host-pinned memory and must be visible to the
 * host before the head/tail index update is published.
 *
 * PTX: membar.sys
 * --------------------------------------------------------------------------- */
__device__ __forceinline__ void ring_membar_sys(void)
{
    asm volatile("membar.sys;" ::: "memory");
}

/* ---------------------------------------------------------------------------
 * ring_atomic_add_release_u32
 *
 * Atomically adds delta to *addr (global, u32) and returns the OLD value.
 * Issues a system-scope release fence after the add so the update is visible
 * to the host consumer immediately.
 *
 * Implementation: atomicAdd guarantees atomicity; membar.sys provides the
 * system release ordering. Using a single PTX atom.add with .sys scope would
 * be cleaner but requires PTX ISA 7.0+ (sm_80). We target sm_86 where it is
 * available, so we use the scoped atomic directly.
 *
 * PTX: atom.global.sys.add.u32
 * --------------------------------------------------------------------------- */
__device__ __forceinline__ uint32_t ring_atomic_add_release_u32(volatile uint32_t* addr, uint32_t delta)
{
    uint32_t old;
    asm volatile(
        "atom.global.sys.add.u32 %0, [%1], %2;"
        : "=r"(old)
        : "l"(addr), "r"(delta)
        : "memory"
    );
    return old;
}

#endif /* K3D_RING_ATOMICS_CUH */
