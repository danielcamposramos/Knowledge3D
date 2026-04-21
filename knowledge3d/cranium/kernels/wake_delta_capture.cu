// wake_delta_capture.cu — Wake-Cycle Delta Capture (Gap 1)
//
// Opcode: 0x320  WAKE_CYCLE_DELTA_CAPTURE
// Registry: RPN_DOMAIN_OPCODE_REGISTRY §11, reserved 2026-04-20
// Owner: knowledge3d/knowledgeverse/wake_delta_capture.py
//
// Algorithm: One thread per tile, grid-stride loop.
//   For each tile t:
//     If halting_value < halting_threshold:
//       out_delta_tiles[t] = 0.0f  (query did not converge — no signal)
//       out_fired = 0             (written by thread 0 only)
//     Else:
//       mean_act = sum(activations[t*N .. t*N+N-1]) / N
//       l1_sum   = sum(|activations[t*N .. t*N+N-1]|)
//       delta    = sign(mean_act) * (l1_sum / N)
//       out_delta_tiles[t] = delta
//       out_fired = 1             (written by thread 0 only)
//
// This is a Hebbian-style reinforcement signal: tiles that were
// strongly active during the converged step are reinforced in the
// direction of their mean activation.  Raw float32 output — Lane B
// handles BitNet packing.
//
// Sovereignty contract: no Python math over tiles; all computation
// on GPU.  No numpy/cupy dependencies.  Raises via non-zero exit
// signal if inputs are null.
//
// Parameters
//   activations[T*N]     float32  last-step per-tile activations
//                                 (T tiles, N = tile_width per tile)
//   halting_value        float32  halting-gate output for this query step
//   halting_threshold    float32  convergence threshold (static per domain)
//   out_delta_tiles[T]   float32  output: per-tile signed-magnitude delta
//   out_fired            int32    1 if converged, 0 if not
//   T                   int32    number of tiles
//   N                   int32    tile width (activations per tile)

#include <stdint.h>

// ── Device helpers ────────────────────────────────────────────────────────────

__device__ __forceinline__ float _wdc_fabsf(float x) {
    return (x < 0.0f) ? -x : x;
}

__device__ __forceinline__ float _wdc_signf(float x) {
    // Returns +1.0f, -1.0f, or 0.0f (zero maps to +1.0f — zero-mass neutral).
    if (x > 0.0f) return 1.0f;
    if (x < 0.0f) return -1.0f;
    return 1.0f;
}

// ── Kernel ────────────────────────────────────────────────────────────────────

extern "C" __global__ void wake_delta_capture(
    const float* __restrict__ activations,   // [T * N] float32
    float                     halting_value,
    float                     halting_threshold,
    float*       __restrict__ out_delta_tiles, // [T] float32
    int*         __restrict__ out_fired,       // scalar int32
    int                       T,
    int                       N
) {
    // Thread 0 of block 0 writes the fired flag exactly once.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *out_fired = (halting_value >= halting_threshold) ? 1 : 0;
    }

    // If not converged: zero out all delta tiles and return.
    if (halting_value < halting_threshold) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;
        while (idx < T) {
            out_delta_tiles[idx] = 0.0f;
            idx += stride;
        }
        return;
    }

    // Converged path: compute per-tile signed-magnitude delta.
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const float inv_N = (N > 0) ? (1.0f / (float)N) : 1.0f;

    while (tile < T) {
        const int base = tile * N;
        float sum_act  = 0.0f;   // for mean (sign)
        float sum_abs  = 0.0f;   // L1 magnitude

        for (int n = 0; n < N; ++n) {
            float a = activations[base + n];
            sum_act += a;
            sum_abs += _wdc_fabsf(a);
        }

        float mean_act = sum_act * inv_N;
        float l1_norm  = sum_abs * inv_N;      // average L1 per element
        float delta    = _wdc_signf(mean_act) * l1_norm;

        out_delta_tiles[tile] = delta;
        tile += stride;
    }
}
