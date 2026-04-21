// sleeptime_lane_b.cu — Sleeptime Lane B: wake-cycle weight consolidation
//
// Opcodes (RPN_DOMAIN_OPCODE_REGISTRY §11, reserved 2026-04-21):
//   0x310  SLEEPTIME_LANE_B_DELTA_AGGREGATE
//   0x311  SLEEPTIME_LANE_B_TRIT_ACCEPT_REJECT
//   0x312  SLEEPTIME_LANE_B_INPLACE_UPDATE
//   0x313  SLEEPTIME_LANE_B_TILE_QUANTIZE_F32_TO_BITNET  (Gap 2, 2026-04-21)
//
// Fused three-stage kernel: B.1 → B.2 → B.3 per tile, one thread per tile,
// grid-stride loop.  Each "tile" is one contiguous group of TILE_TRITS ternary
// weights stored in ceil(TILE_TRITS/5) packed bytes.
//
// Stage 0 (0x313) — per-tile format check: if tile_format[tile_id]==WEIGHT_FORMAT_F32,
// read TILE_TRITS float32 scalars, quantise each via quantize_trit(), repack via
// pack5(), overwrite weight_tiles in place, set tile_format[tile_id]=WEIGHT_FORMAT_BITNET.
// Idempotent: already-packed tiles are a no-op.  Runs live during sleeptime ticks —
// convert-on-first-touch, never at boot.
//
// ─── Algorithm (one thread per tile) ──────────────────────────────────────
//
//  Stage B.1 — Delta aggregation (0x310)
//    Input : shadow_deltas[T × TILE_TRITS]  float32  (per-trit delta per trace)
//            trace_weights[T]               float32  (success confidence per trace)
//    For each trace t ∈ [0, T):
//      agg_delta[j] += trace_weights[t] × shadow_deltas[t × TILE_TRITS + tile×TILE_TRITS + j]
//    Output: per-trit aggregated delta for this tile (register-only, no write).
//
//  Stage B.2 — Trit accept/reject (0x311)
//    Aggregate the per-trit deltas into a single scalar magnitude for the tile:
//      tile_signal = sum(|agg_delta[j]|) / TILE_TRITS
//    Use quantize_trit() from gre_defeasible_resolver.cu to classify:
//      tile_trit = quantize_trit(tile_signal - ACCEPT_THRESH)
//        +1  → tile_signal > ACCEPT_THRESH         (accept: signal strong enough)
//         0  → |tile_signal| ≤ ACCEPT_THRESH       (pending: insufficient evidence)
//        -1  → tile_signal < -ACCEPT_THRESH         (reject: destabilising trace)
//    Note: reject (-1) is not possible under the current formulation because
//    tile_signal is an absolute magnitude.  We reserve -1 for future extensions
//    where a negative-confidence trace can actively contradict existing weights.
//    For this implementation, tile_trit ∈ {0, +1} only.
//
//  Stage B.3 — In-place BitNet b1.58 update (0x312)
//    If tile_trit == +1:
//      For each trit j in the tile:
//        1. Unpack current BitNet byte via unpack5() (symlinked from bitnet_attention.cu).
//        2. Apply delta: new_float = (float)(current_trit) + alpha × agg_delta[j]
//        3. Re-quantise to ternary: new_trit = quantize_trit(new_float)
//        4. Repack 5 trits into 1 byte via pack5() (symlinked from bitnet_attention.cu).
//        5. Write updated packed byte back to weight_tiles[tile × TILE_BYTES + byte_idx].
//    If tile_trit == 0: no write (leave tile unchanged).
//
// ─── Launch parameters ────────────────────────────────────────────────────
//   TILE_TRITS = 20  (one uint32 word = 4 bytes = 20 trits at 1.6 bits/trit)
//   TILE_BYTES = 4   (packed bytes per tile)
//   block_size = 128 (tunable, matches Lane A)
//   grid_size  = ceil(N_tiles / block_size)
//   Grid-stride handles N_tiles > max_grid × block_size.
//   Shared memory: none required (each thread works on its own tile independently).
//
// ─── Per-tile format metadata (tile_format[] device buffer) ───────────────
//   WEIGHT_FORMAT_F32    = 0  — tile holds TILE_TRITS float32 scalars (4 bytes each)
//                               i.e., each tile occupies TILE_TRITS × 4 = 80 bytes
//                               in the f32 weight buffer.
//   WEIGHT_FORMAT_BITNET = 1  — tile holds TILE_BYTES = 4 packed bytes (1.6 bits/trit)
//
//   The tile_format[] buffer is N_tiles uint8, allocated in Python alongside
//   weight_tiles.  Loaded checkpoints from .bin (float32) are initialised to
//   WEIGHT_FORMAT_F32.  Checkpoints loaded from .bitnet are WEIGHT_FORMAT_BITNET.
//   On first Lane B touch the kernel converts and sets WEIGHT_FORMAT_BITNET;
//   subsequent touches skip quantisation (idempotent).
//
// ─── Sovereignty ──────────────────────────────────────────────────────────
//   Pure PTX kernel — no external libraries, no CPU fallback, no Python logic.
//   Device functions symlinked from gre_defeasible_resolver.cu and bitnet_attention.cu.
//   Python reads out_tile_trits[] after synchronisation and writes checkpoint.

#include <math.h>
#include <stdint.h>
#include "gre_defeasible_resolver.cu"
#include "bitnet_attention.cu"

#define LANE_B_TILE_TRITS    20      /* 1 uint32 = 4 bytes = 20 trits */
#define LANE_B_TILE_BYTES    4       /* packed bytes per tile */
#define LANE_B_ACCEPT_THRESH 0.10f   /* min mean absolute delta to accept tile */
#define LANE_B_ALPHA         0.01f   /* learning rate for delta application */
#define LANE_B_MAX_TRACES    64      /* upper bound on wake-cycle traces */

// Per-tile format constants (opcode 0x313, Gap 2, 2026-04-21)
#define WEIGHT_FORMAT_F32    0       /* tile holds TILE_TRITS float32 scalars (80 bytes) */
#define WEIGHT_FORMAT_BITNET 1       /* tile holds TILE_BYTES packed 1.6-bit bytes (4 bytes) */

// ---------------------------------------------------------------------------
// lane_b_tile_quantize_f32_to_bitnet — opcode 0x313
//
// Convert one float32 tile to BitNet b1.58 1.6-bit packing in place.
// Idempotent: if tile_format[tile_id] == WEIGHT_FORMAT_BITNET, returns immediately.
//
// Parameters:
//   weight_tiles_f32  [N_tiles × TILE_TRITS]  float32  — input f32 tile buffer
//   weight_tiles      [N_tiles × TILE_BYTES]  uint8    — output packed tile buffer (in/out)
//   tile_format       [N_tiles]               uint8    — per-tile format byte (in/out)
//   tile_id                                   int      — which tile this thread owns
//
// Called as a device function from the fused kernel's Stage B.3 prologue.
// Symlinks: quantize_trit() from gre_defeasible_resolver.cu,
//           pack5()         from bitnet_attention.cu.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void lane_b_tile_quantize_f32_to_bitnet(
    const float*   __restrict__ weight_tiles_f32,   // [N_tiles × TILE_TRITS] float32 in
    uint8_t*       __restrict__ weight_tiles,        // [N_tiles × TILE_BYTES] uint8  in/out
    uint8_t*       __restrict__ tile_format,         // [N_tiles]              uint8  in/out
    int tile_id
) {
    // Idempotent: skip if already packed
    if (tile_format[tile_id] == (uint8_t)WEIGHT_FORMAT_BITNET) return;

    const float* src = weight_tiles_f32 + tile_id * LANE_B_TILE_TRITS;
    uint8_t*     dst = weight_tiles     + tile_id * LANE_B_TILE_BYTES;

    // Process TILE_BYTES (= TILE_TRITS / 5) groups of 5 floats → 1 packed byte.
#pragma unroll
    for (int byte_idx = 0; byte_idx < LANE_B_TILE_BYTES; ++byte_idx) {
        const int base = byte_idx * 5;
        int8_t t0 = quantize_trit(src[base + 0]);
        int8_t t1 = quantize_trit(src[base + 1]);
        int8_t t2 = quantize_trit(src[base + 2]);
        int8_t t3 = quantize_trit(src[base + 3]);
        int8_t t4 = quantize_trit(src[base + 4]);
        dst[byte_idx] = pack5(t0, t1, t2, t3, t4);
    }

    // Mark tile as packed
    tile_format[tile_id] = (uint8_t)WEIGHT_FORMAT_BITNET;
}

// ---------------------------------------------------------------------------
// sleeptime_lane_b_step — fused B.1+B.2+B.3, one thread per tile
//
// Parameters:
//   weight_tiles_f32 [N_tiles × TILE_TRITS]  float32  f32 source for 0x313 quantise (in)
//   weight_tiles     [N_tiles × TILE_BYTES]  uint8    1.6-bit-packed weight tiles (in/out)
//   tile_format      [N_tiles]               uint8    per-tile format byte WEIGHT_FORMAT_* (in/out)
//   shadow_deltas    [T × N_tiles × TILE_TRITS]  float32  per-trace per-tile per-trit deltas
//   trace_weights    [T]  float32  per-trace success confidence in [0,1]
//   T                int  number of wake-cycle traces
//   N_tiles          int  total number of weight tiles across all matrices
//   out_tile_trits   [N_tiles]  int8  output: {-1 reject, 0 pending, +1 accept} per tile
// ---------------------------------------------------------------------------

extern "C" __global__ void sleeptime_lane_b_step(
    const float*   __restrict__ weight_tiles_f32, // [N_tiles × TILE_TRITS] float32 (0x313 source)
    uint8_t*       __restrict__ weight_tiles,     // [N_tiles × TILE_BYTES] in/out
    uint8_t*       __restrict__ tile_format,      // [N_tiles] per-tile format byte in/out
    const float*   __restrict__ shadow_deltas,    // [T × N_tiles × TILE_TRITS] float32
    const float*   __restrict__ trace_weights,    // [T] float32
    int T,                                        // number of traces
    int N_tiles,                                  // total tiles

    int8_t*        __restrict__ out_tile_trits    // [N_tiles] output trit per tile
) {
    const int capped_T = (T < LANE_B_MAX_TRACES) ? T : LANE_B_MAX_TRACES;

    // grid-stride loop — one logical thread per tile
    for (int tile = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         tile < N_tiles;
         tile += (int)(gridDim.x * blockDim.x))
    {
        // ── Stage B.1: Delta aggregation ──────────────────────────────────
        float agg[LANE_B_TILE_TRITS];
#pragma unroll
        for (int j = 0; j < LANE_B_TILE_TRITS; ++j) {
            agg[j] = 0.0f;
        }

        for (int t = 0; t < capped_T; ++t) {
            const float tw = trace_weights[t];
            if (tw <= 0.0f) continue;  // zero-weight trace contributes nothing
            const int base = (t * N_tiles + tile) * LANE_B_TILE_TRITS;
#pragma unroll
            for (int j = 0; j < LANE_B_TILE_TRITS; ++j) {
                agg[j] += tw * shadow_deltas[base + j];
            }
        }

        // ── Stage B.2: Trit accept/reject ─────────────────────────────────
        float abs_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < LANE_B_TILE_TRITS; ++j) {
            abs_sum += fabsf(agg[j]);
        }
        const float tile_signal = abs_sum / (float)LANE_B_TILE_TRITS;

        // quantize_trit() from gre_defeasible_resolver.cu:
        //   +1 if value > 1e-6f, -1 if value < -1e-6f, 0 otherwise.
        // We centre by subtracting the threshold so that only tiles with
        // sufficient average delta magnitude are accepted.
        const int8_t tile_trit = quantize_trit(tile_signal - LANE_B_ACCEPT_THRESH);

        out_tile_trits[tile] = tile_trit;

        // ── Stage B.3: In-place BitNet update ─────────────────────────────
        // Opcode 0x313 prologue: convert float32 tile to BitNet on first touch.
        // Idempotent — no-op if tile_format[tile] == WEIGHT_FORMAT_BITNET already.
        lane_b_tile_quantize_f32_to_bitnet(weight_tiles_f32, weight_tiles, tile_format, tile);

        if (tile_trit != (int8_t)1) {
            // pending (0) or reject (-1): leave tile unchanged
            continue;
        }

        // Process TILE_BYTES (= TILE_TRITS / 5) packed bytes in this tile.
        // Each packed byte encodes 5 trits via pack5/unpack5.
        uint8_t* tile_ptr = weight_tiles + tile * LANE_B_TILE_BYTES;

        for (int byte_idx = 0; byte_idx < LANE_B_TILE_BYTES; ++byte_idx) {
            // Unpack current 5 trits from this byte.
            // unpack5() from bitnet_attention.cu (uses constant-memory LUT).
            int8_t trits[5];
            unpack5(tile_ptr[byte_idx], trits);

            const int trit_base = byte_idx * 5;
#pragma unroll 5
            for (int k = 0; k < 5; ++k) {
                const int j = trit_base + k;
                if (j >= LANE_B_TILE_TRITS) break;

                // Apply delta at learning rate ALPHA.
                // current_trit ∈ {-1, 0, +1}; agg[j] is the weighted delta signal.
                const float updated = (float)trits[k] + LANE_B_ALPHA * agg[j];

                // Re-quantise to ternary via quantize_trit().
                trits[k] = quantize_trit(updated);
            }

            // Re-pack updated 5 trits back into 1 byte.
            // pack5() from bitnet_attention.cu (host+device __forceinline__).
            tile_ptr[byte_idx] = pack5(trits[0], trits[1], trits[2], trits[3], trits[4]);
        }
    }
}
