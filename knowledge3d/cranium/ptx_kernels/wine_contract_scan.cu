/*
 * wine_contract_scan.cu — Device-side Galaxy scan for WINE contract resolution
 *
 * Spec: CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §4.1, §4.2
 * Audit: AGENT1_KERNEL_AUDIT_04.18.2026.md §3 (missing wine_contract_scan)
 *
 * What is a WINE contract?
 *   A Layer 2 Galaxy star that describes a translation paradigm at the avatar's
 *   Tablet surface. The star holds ingress and egress RPN program addresses so
 *   the persistent tick can route incoming bytes (DOM frames, ARC3 game frames,
 *   text, audio, image) through the correct Grammar Galaxy programs.
 *   No Python dict. No Python re.search. No Python routing module.
 *   The kernel scans WineContractStar[] entirely on device.
 *
 * Paradigm type values (spec §4.1):
 *   0x01  DOM    — HTML/DOM output (Christoph's target: <p> envelope)
 *   0x02  ARC3   — ARC-AGI-3 game frame ingress
 *   0x03  TEXT   — stdin/stdout text (whitespace tokenize → UTF-8 write)
 *   0x04  AUDIO  — audio frame ingress
 *   0x05  IMAGE  — image frame ingress
 *
 * Scan strategy — warp-cooperative first-match:
 *   Each warp collaborates to scan a 32-element window of the star array.
 *   __ballot_sync finds which lanes matched; __ffs elects the lowest lane as
 *   winner. The scan advances in 32-element strides until a match is found or
 *   all stars are exhausted. Result is broadcast to all lanes in the warp.
 *
 *   This is NOT a parallel reduction — we want the FIRST match (lowest index)
 *   because contract precedence is defined by insertion order in the Galaxy.
 *   warp-cooperative ballot gives us 32× throughput vs sequential scan while
 *   preserving first-match semantics.
 *
 * Thread assignment:
 *   Callers should assign one warp per contract query.
 *   The caller's thread 0 (lane 0) receives the valid result pointer.
 *   Other lanes get the same pointer (broadcast via __shfl_sync).
 *
 * Target: sm_86 (RTX 3070). Requires __ballot_sync, __ffs, __shfl_sync.
 *   All available on sm_70+ (Volta) and onward.
 */

#include <cuda_runtime.h>
#include <cstdint>

/*
 * WineContractStar — Layer 2 Galaxy star schema for WINE paradigm contracts.
 *
 * Total: 40 bytes. Stored as a flat array in VRAM, seeded at boot from JSONL.
 *
 * Fields:
 *   contract_hash      — murmur3(paradigm signature) for fast equality check
 *   paradigm_type      — enumerated contract type (0x01–0x05, see above)
 *   _pad[7]            — alignment padding to 8-byte boundary
 *   ingress_rpn_addr   — device ptr to Layer 3 Grammar Galaxy RPN program
 *                        that transforms incoming bytes → Galaxy form
 *   egress_rpn_addr    — device ptr to Layer 3 Grammar Galaxy RPN program
 *                        that transforms Galaxy form → outgoing bytes
 *   visual_rpn_symlink — optional Layer 1 Drawing Galaxy symlink address
 *                        (0 if unused; non-zero triggers visual rendering)
 */
struct alignas(8) WineContractStar {
    uint64_t contract_hash;       /*  8 bytes */
    uint8_t  paradigm_type;       /*  1 byte  */
    uint8_t  _pad[7];             /*  7 bytes */
    uint64_t ingress_rpn_addr;    /*  8 bytes */
    uint64_t egress_rpn_addr;     /*  8 bytes */
    uint64_t visual_rpn_symlink;  /*  8 bytes */
                                  /* = 40 bytes */
};

static_assert(sizeof(WineContractStar) == 40,
              "WineContractStar layout must be exactly 40 bytes");

/* ---------------------------------------------------------------------------
 * wine_scan_by_paradigm — find the first WineContractStar with matching type.
 *
 * Spec §4.2: GALAXY_SCAN(E2) predicate = (paradigm_type == in.header.paradigm_type)
 *
 * Called by one warp (32 consecutive threads). All lanes participate.
 * Returns a pointer to the matching star, or nullptr if not found.
 * The returned pointer is valid on ALL lanes (broadcast via __shfl_sync).
 *
 * Parameters:
 *   stars          — device ptr to WineContractStar array
 *   n              — number of stars
 *   paradigm_type  — the paradigm to search for
 * --------------------------------------------------------------------------- */
__device__ const WineContractStar* wine_scan_by_paradigm(
    const WineContractStar* stars,
    uint32_t                n,
    uint8_t                 paradigm_type)
{
    if (stars == nullptr || n == 0u) return nullptr;

    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t FULL    = 0xFFFFFFFFu;

    /* Scan in 32-element strides, warp-cooperative */
    for (uint32_t base = 0u; base < n; base += 32u) {
        const uint32_t idx = base + lane;

        /* Each lane checks its candidate (guard out-of-bounds lanes) */
        bool match = (idx < n) && (stars[idx].paradigm_type == paradigm_type);

        /* Ballot: which lanes have a match? */
        uint32_t ballot = __ballot_sync(FULL, match);
        if (ballot != 0u) {
            /* Elect lowest matching lane */
            int winner_lane = __ffs(static_cast<int>(ballot)) - 1;
            uint32_t winner_idx = base + static_cast<uint32_t>(winner_lane);

            /* Compute pointer on winner lane, broadcast to all lanes */
            const WineContractStar* result =
                (lane == static_cast<uint32_t>(winner_lane)) ? &stars[winner_idx] : nullptr;

            /* Broadcast: lane winner_lane sends its pointer to all lanes */
            uint64_t ptr_as_u64;
            if (lane == static_cast<uint32_t>(winner_lane)) {
                ptr_as_u64 = reinterpret_cast<uint64_t>(result);
            }
            /* __shfl_sync on 64-bit: split into two 32-bit halves */
            uint32_t lo = (lane == static_cast<uint32_t>(winner_lane))
                          ? static_cast<uint32_t>(ptr_as_u64 & 0xFFFFFFFFu) : 0u;
            uint32_t hi = (lane == static_cast<uint32_t>(winner_lane))
                          ? static_cast<uint32_t>(ptr_as_u64 >> 32u) : 0u;

            lo = __shfl_sync(FULL, lo, winner_lane);
            hi = __shfl_sync(FULL, hi, winner_lane);

            uint64_t broadcast_ptr =
                (static_cast<uint64_t>(hi) << 32u) | static_cast<uint64_t>(lo);

            return reinterpret_cast<const WineContractStar*>(broadcast_ptr);
        }
    }

    return nullptr; /* no match */
}

/* ---------------------------------------------------------------------------
 * wine_scan_by_hash — find the first WineContractStar with matching hash.
 *
 * Spec §4.2: secondary lookup path when paradigm_type alone is ambiguous
 * (e.g., two DOM contracts with different subtype hashes).
 *
 * Same warp-cooperative ballot pattern as wine_scan_by_paradigm.
 * --------------------------------------------------------------------------- */
__device__ const WineContractStar* wine_scan_by_hash(
    const WineContractStar* stars,
    uint32_t                n,
    uint64_t                contract_hash)
{
    if (stars == nullptr || n == 0u) return nullptr;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t FULL = 0xFFFFFFFFu;

    for (uint32_t base = 0u; base < n; base += 32u) {
        const uint32_t idx = base + lane;

        bool match = (idx < n) && (stars[idx].contract_hash == contract_hash);

        uint32_t ballot = __ballot_sync(FULL, match);
        if (ballot != 0u) {
            int winner_lane = __ffs(static_cast<int>(ballot)) - 1;
            uint32_t winner_idx = base + static_cast<uint32_t>(winner_lane);

            uint64_t ptr_as_u64 =
                (lane == static_cast<uint32_t>(winner_lane))
                ? reinterpret_cast<uint64_t>(&stars[winner_idx])
                : 0ull;

            uint32_t lo = static_cast<uint32_t>(ptr_as_u64 & 0xFFFFFFFFu);
            uint32_t hi = static_cast<uint32_t>(ptr_as_u64 >> 32u);

            lo = __shfl_sync(FULL, lo, winner_lane);
            hi = __shfl_sync(FULL, hi, winner_lane);

            uint64_t broadcast_ptr =
                (static_cast<uint64_t>(hi) << 32u) | static_cast<uint64_t>(lo);

            return reinterpret_cast<const WineContractStar*>(broadcast_ptr);
        }
    }

    return nullptr;
}
