/*
 * D3 sovereign procedural basis + token-id derivation.
 *
 * Replaces the Python hashlib.shake_256 / blake2b fallback paths that
 * fabricated per-token basis bytes and numeric ids on CPU. The resulting
 * .bin is now the deterministic output of on-GPU arithmetic over the
 * token's UTF-8 bytes -- no numpy, no Python hash library, no cupy.
 *
 * Contract preserved so Codex's matryoshka_accumulator.cu consumes the
 * output unchanged:
 *   - basis_packed[] has 5 ternary digits per byte in base-3 (values 0..242);
 *     matryoshka_accumulator::unpack_trit reads these as-is.
 *   - token_ids[] has the MSB set for non-canonical tokens so that
 *     matryoshka_accumulator::ternary_affinity pulls block byte
 *     ((id >> 8) & 0xFF) out of GPU-derived entropy instead of a Python
 *     hash. Canonical opcode overrides are applied host-side post-kernel
 *     (KNOWN_OPCODE_MAP is the canonical opcode registry, which IS the
 *     sovereign source of truth for named opcodes).
 */

#include <stdint.h>

#define MATRYOSHKA_DIM       2048
#define PACKED_TRITS_STRIDE  410  /* ceil(2048/5) */
#define BASIS_SEED          0xD3BA515D3BA515DULL
#define TOKEN_ID_SEED       0x1DD3BA515DD3BA5ULL

static __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static __device__ __forceinline__ uint64_t absorb_token(
    const uint8_t* bytes,
    int length,
    uint64_t seed
) {
    uint64_t state = seed;
    for (int i = 0; i < length; ++i) {
        state ^= (uint64_t)bytes[i];
        state *= 0x00000100000001B3ULL;  /* FNV-1a 64-bit prime */
        state  = splitmix64(state);
    }
    return splitmix64(state ^ (uint64_t)length);
}

/*
 * One CUDA block per vocabulary token; threads cooperate on PACKED_TRITS_STRIDE bytes.
 * Grid: (vocab_count, 1, 1).  Block: (64, 1, 1) is enough (410 bytes / 64 thr ~= 7 iters).
 */
extern "C" __global__ void procedural_basis_fingerprint(
    const uint8_t*  __restrict__ token_bytes,
    const uint32_t* __restrict__ token_offsets,
    int vocab_count,
    uint8_t*        __restrict__ basis_packed
) {
    const int token_idx = (int)blockIdx.x;
    if (token_idx >= vocab_count) {
        return;
    }

    const uint32_t start  = token_offsets[token_idx];
    const uint32_t end    = token_offsets[token_idx + 1];
    const int      length = (int)(end - start);
    const uint8_t* bytes  = token_bytes + start;

    const uint64_t seed = absorb_token(bytes, length, BASIS_SEED);

    for (int byte_idx = (int)threadIdx.x;
         byte_idx < PACKED_TRITS_STRIDE;
         byte_idx += (int)blockDim.x) {

        uint64_t mixed = splitmix64(seed + (uint64_t)byte_idx);
        /* Re-stir so neighboring bytes do not share obvious structure. */
        mixed ^= splitmix64(mixed * 0xD1B54A32D192ED03ULL + (uint64_t)byte_idx);

        uint8_t packed = 0;
        uint8_t place  = 1;
        #pragma unroll
        for (int k = 0; k < 5; ++k) {
            const int dim_in_byte = byte_idx * 5 + k;
            if (dim_in_byte < MATRYOSHKA_DIM) {
                /* Pull an independent 12-bit lane, reduce to a trit {0,1,2}. */
                const uint32_t lane = (uint32_t)((mixed >> (k * 12)) & 0xFFFu);
                const uint8_t  trit = (uint8_t)(lane % 3u);
                packed = (uint8_t)(packed + trit * place);
            }
            place = (uint8_t)(place * 3u);
        }
        basis_packed[(size_t)token_idx * (size_t)PACKED_TRITS_STRIDE + (size_t)byte_idx] = packed;
    }
}

/*
 * One thread per vocabulary token.
 * Grid: (ceil(vocab_count/256), 1, 1).  Block: (256, 1, 1).
 */
extern "C" __global__ void procedural_token_id(
    const uint8_t*  __restrict__ token_bytes,
    const uint32_t* __restrict__ token_offsets,
    int vocab_count,
    uint32_t*       __restrict__ token_ids
) {
    const int token_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (token_idx >= vocab_count) {
        return;
    }

    const uint32_t start  = token_offsets[token_idx];
    const uint32_t end    = token_offsets[token_idx + 1];
    const int      length = (int)(end - start);
    const uint8_t* bytes  = token_bytes + start;

    const uint64_t state = absorb_token(bytes, length, TOKEN_ID_SEED);
    const uint32_t id32  = (uint32_t)((state ^ (state >> 32)) & 0x7FFFFFFFu);

    /* MSB = 1 marks "not a canonical opcode" (Python side overrides this
     * host-side for tokens that hit KNOWN_OPCODE_MAP). Same convention as
     * the old _token_numeric_id fallback, so ternary_affinity() keeps its
     * block-byte semantics. */
    token_ids[token_idx] = 0x80000000u | id32;
}
