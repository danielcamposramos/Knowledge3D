// Ternary prune decision - maps scores to {-1,0,+1} keep flags
// Encoding: -1 (discard), 0 (neutral), +1 (keep)

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ int8_t to_trit(float v, float keep_thresh, float drop_thresh) {
    if (v >= keep_thresh) return 1;
    if (v <= drop_thresh) return -1;
    return 0;
}

extern "C" __global__ void ternary_prune_decision(
    const float* __restrict__ scores,  // importance scores per item
    int8_t* __restrict__ out,          // ternary decisions per item
    int n,
    float keep_thresh,
    float drop_thresh
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = to_trit(scores[idx], keep_thresh, drop_thresh);
}
