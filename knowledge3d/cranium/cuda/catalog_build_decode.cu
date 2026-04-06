#include <stdint.h>

#define BUILD_ROW_BYTES 176u

extern "C" __global__ void catalog_build_decode(
    const unsigned char* __restrict__ build_rows,
    unsigned char* __restrict__ raw_input,
    unsigned int entry_count
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entry_count) {
        return;
    }
    const unsigned char* src = build_rows + (i * BUILD_ROW_BYTES);
    unsigned char* dst = raw_input + (i * BUILD_ROW_BYTES);
    #pragma unroll
    for (unsigned int offset = 0u; offset < BUILD_ROW_BYTES; ++offset) {
        dst[offset] = src[offset];
    }
}
