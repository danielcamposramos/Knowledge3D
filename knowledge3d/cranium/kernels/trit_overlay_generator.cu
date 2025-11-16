// Trit Overlay Generator - renders ternary fields into an RGBA8 overlay texture
// Encoding: 2 bits per trit (00=-1, 01=0, 10=+1, 11 unused)
// Color map: -1 -> blue, 0 -> transparent, +1 -> red

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ int8_t decode_trit(const uint32_t* buf, int idx) {
    const uint32_t word = buf[idx >> 4];
    const int shift = (idx & 0xF) << 1;
    const uint32_t bits = (word >> shift) & 0x3u;
    // Map to {-1, 0, +1}
    return bits == 2 ? 1 : (bits == 1 ? 0 : -1);
}

extern "C" __global__ void trit_overlay_generator(
    const uint32_t* __restrict__ trit_buf, // packed trits
    uint8_t* __restrict__ rgba,            // RGBA8 overlay buffer
    int gx, int gy, int gz,                // overlay grid resolution
    int field_stride,                      // stride between fields per node
    int field_type,                        // which field to visualize (0-based)
    float threshold                        // min |trit| to visualize
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= gx || y >= gy || z >= gz) return;

    const int flat = (z * gy * gx) + (y * gx) + x;
    const int idx = flat * field_stride + field_type;
    int8_t t = decode_trit(trit_buf, idx);
    if (threshold > 0.0f && fabsf(static_cast<float>(t)) < threshold) {
        t = 0;
    }

    const int out = flat << 2;
    rgba[out + 0] = (t == 1) ? 255 : 0;    // R
    rgba[out + 1] = 0;                     // G
    rgba[out + 2] = (t == -1) ? 255 : 0;   // B
    rgba[out + 3] = (t != 0) ? 96 : 0;     // A (low alpha)
}
