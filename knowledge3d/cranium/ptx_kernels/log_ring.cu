/*
 * log_ring.cu — Device-side circular log buffer for persistent tick observability
 *
 * Spec: CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §2.1
 * Audit: AGENT1_KERNEL_AUDIT_04.18.2026.md §5 (existing lane_perf_ring context)
 * Feedback: feedback_note_taking_everywhere.md — every solve must emit a trace;
 *           silence is a bug. This ring is the device-side trace mechanism.
 *
 * Design:
 *   Fixed-size records (64 bytes each) so the ring head is a simple slot index.
 *   No variable-length records — avoids fragmentation, avoids host coordination.
 *   On overflow (head wraps around to tail), the OLDEST record is silently
 *   overwritten. The alternative (drop new records) is worse: recent events
 *   would be lost during bursts. Host drains the ring asynchronously.
 *
 * Fence scope:
 *   log_ring is a VRAM-resident ring (device memory, not host-pinned).
 *   We use __threadfence() (device scope) for ordering between writer and any
 *   device-side reader (e.g., a sleep-time kernel that reads the log).
 *   ring_atomics.cuh membar.sys is NOT needed here: the host reads via
 *   cudaMemcpy which imposes its own ordering, not via mapped zero-copy.
 *
 * Log codes (lower 8 bits = phase, upper 24 bits = detail):
 *   0x01xx  PERCEIVE phase events
 *   0x02xx  NAVIGATE phase events
 *   0x03xx  REASON phase events
 *   0x04xx  PHYSICS phase events
 *   0x05xx  DECIDE phase events
 *   0x06xx  ACT phase events
 *   0xF0xx  STUB (phase not yet wired; expected during initial integration)
 *   0xFF00  FREELIST_OOM
 *   0xFF01  INPUT_RING_EMPTY (idle spin)
 *   0xFF02  OUTPUT_RING_FULL (back-pressure, record dropped)
 *
 * Target: sm_86 (RTX 3070).
 */

#include <cuda_runtime.h>
#include <cstdint>

/*
 * LogRecord — 64 bytes, cache-line aligned.
 *
 * tick_id   : monotone tick counter from trm_step_fused at time of emit
 * code      : log code (see table above)
 * payload   : up to 14 uint32_t words of free-form context data
 *             Callers fill only the words they need; remaining words are 0.
 */
struct alignas(64) LogRecord {
    uint64_t tick_id;       /*  8 bytes */
    uint32_t code;          /*  4 bytes */
    uint32_t payload[14];   /* 56 bytes — total = 64 bytes */
};

static_assert(sizeof(LogRecord) == 64, "LogRecord must be exactly 64 bytes");

/* Convenience log codes exported for callers */
#define K3D_LOG_PERCEIVE_ENTER    0x0100u
#define K3D_LOG_NAVIGATE_ENTER    0x0200u
#define K3D_LOG_REASON_ENTER      0x0300u
#define K3D_LOG_PHYSICS_ENTER     0x0400u
#define K3D_LOG_DECIDE_ENTER      0x0500u
#define K3D_LOG_ACT_ENTER         0x0600u
#define K3D_LOG_PHASE_STUB        0xF000u   /* phase is a stub, not yet wired */
#define K3D_LOG_FREELIST_OOM      0xFF00u
#define K3D_LOG_INPUT_EMPTY       0xFF01u
#define K3D_LOG_OUTPUT_FULL       0xFF02u
#define K3D_LOG_GALAXY_MISS       0xFF03u   /* contract or star lookup miss   */

/* ---------------------------------------------------------------------------
 * log_emit — write one LogRecord to the circular log.
 *
 * Parameters:
 *   slots    — device pointer to LogRecord array of length `capacity`
 *   head     — device pointer to the ring head (next-write index, monotone)
 *   capacity — number of slots (must be power of 2 for efficient masking)
 *   code     — log code identifying the event
 *   payload  — caller-supplied uint32_t data words (may be nullptr if nwords==0)
 *   nwords   — number of payload words to copy (clamped to 14)
 *   tick_id  — current tick counter value
 *
 * Concurrency:
 *   Uses atomicAdd on head to claim a unique slot index before writing.
 *   The __threadfence() before the payload write ensures the slot is not
 *   observed half-written by a concurrent reader.
 *   Overflow is silent: the modulo masks the index, overwriting the oldest entry.
 *
 * Called from multiple threads: only one thread per logical event should call
 * log_emit (typically block 0, thread 0 for per-tick events; any thread for
 * per-thread events). Concurrent calls are safe but unordered.
 * --------------------------------------------------------------------------- */
__device__ void log_emit(
    LogRecord*        slots,
    volatile uint32_t* head,
    uint32_t           capacity,
    uint32_t           code,
    const uint32_t*    payload,
    int                nwords,
    uint64_t           tick_id)
{
    if (slots == nullptr || head == nullptr || capacity == 0u) return;

    /* Claim slot — monotone increment, wrap via modulo */
    const uint32_t slot_idx = atomicAdd(const_cast<uint32_t*>(head), 1u)
                              & (capacity - 1u);

    /* Fence: ensure prior writes in this phase are visible to readers */
    __threadfence();

    LogRecord* rec = &slots[slot_idx];
    rec->tick_id = tick_id;
    rec->code    = code;

    /* Zero payload then copy caller-provided words */
    const int copy_words = (nwords < 14) ? nwords : 14;
    for (int i = 0; i < 14; ++i) {
        rec->payload[i] = (payload != nullptr && i < copy_words)
                         ? payload[i]
                         : 0u;
    }

    /* Release fence: the completed record is visible to readers */
    __threadfence();
}

/*
 * log_emit_1 — convenience wrapper for single-word payload.
 * Avoids requiring a stack array for the common case of logging one value.
 */
__device__ __forceinline__ void log_emit_1(
    LogRecord*        slots,
    volatile uint32_t* head,
    uint32_t           capacity,
    uint32_t           code,
    uint32_t           word0,
    uint64_t           tick_id)
{
    uint32_t p[1] = { word0 };
    log_emit(slots, head, capacity, code, p, 1, tick_id);
}
