// Ternary Depth Field - computes per-node ternary proximity to a query embedding
// Encoding matches balanced ternary utilities: 00=-1, 01=0, 10=+1 (11 unused)
// Input: embeddings [n_nodes x dim], query [dim], thresholds
// Output: packed 2-bit trits (ceil(n_nodes/16) uint32 words)

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t encode_trit(int8_t t) {
    // Map {-1,0,+1} -> {0,1,2}
    return (t > 0) ? 2u : ((t == 0) ? 1u : 0u);
}

extern "C" __global__ void ternary_depth_field(
    const float* __restrict__ embeddings, // [n_nodes * dim]
    const float* __restrict__ query,      // [dim]
    int n_nodes,
    int dim,
    float attract_thresh,                 // >= => +1
    float repel_thresh,                   // <= => -1
    uint32_t* __restrict__ out_packed     // packed trits output
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_nodes) return;

    // Dot product (assumes embeddings may be pre-normalized; no norm to keep latency low)
    float dot = 0.0f;
    const int base = idx * dim;
    for (int d = threadIdx.y; d < dim; d += blockDim.y) {
        dot += embeddings[base + d] * query[d];
    }
    // Reduce over blockDim.y (use warp-level since blockDim.y typically small)
    // Here blockDim.y==1 by launch config; keep code simple

    int8_t trit = 0;
    if (dot >= attract_thresh) trit = 1;
    else if (dot <= repel_thresh) trit = -1;
    else trit = 0;

    const uint32_t bits = encode_trit(trit);
    const int word = idx >> 4;
    const int shift = (idx & 0xF) << 1;
    const uint32_t mask = bits << shift;
    // Pack with atomic OR to avoid races between threads in same word
    atomicOr(out_packed + word, mask);
}
