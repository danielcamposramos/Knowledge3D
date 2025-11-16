// Trit Inspector - summarizes packed ternary fields for given nodes
// Encoding: 2 bits per trit (00=-1, 01=0, 10=+1, 11 unused)

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

struct TritSummary {
    int32_t count;
    int32_t sum;
    float mean;
    float var;
    int32_t bottlenecks; // flag: 1 if trit == 0
};

__device__ __forceinline__ int8_t decode_trit(const uint32_t* buf, int idx) {
    const uint32_t word = buf[idx >> 4];
    const int shift = (idx & 0xF) << 1;
    const uint32_t bits = (word >> shift) & 0x3u;
    // Map to {-1, 0, +1}
    return bits == 2 ? 1 : (bits == 1 ? 0 : -1);
}

extern "C" __global__ void trit_inspector(
    const uint32_t* __restrict__ trit_buf,   // packed trits
    const int32_t* __restrict__ node_indices,// node indices to inspect
    int n,                                   // number of nodes
    int field_stride,                        // stride between fields per node
    TritSummary* __restrict__ out            // one summary per node
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int idx = node_indices[i] * field_stride;
    const int8_t t = decode_trit(trit_buf, idx);
    // Single-value summary (diagnostic scope keeps it simple)
    out[i].count = 1;
    out[i].sum = static_cast<int32_t>(t);
    out[i].mean = static_cast<float>(t);
    out[i].var = 0.0f;
    out[i].bottlenecks = (t == 0) ? 1 : 0;
}
