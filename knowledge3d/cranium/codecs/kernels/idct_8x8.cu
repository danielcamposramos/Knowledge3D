/**
 * K3D Loeffler 8×8 IDCT — opcode 0x248 IDCT_8X8 (critical JPEG-equivalent path).
 *
 * Reference: Loeffler, Ligtenberg, Moschytz (1989), "Practical fast 1-D DCT
 * algorithms with 11 multiplications", ICASSP. 11 muls + 29 adds per 8-point.
 * Matches libjpeg-turbo's jidctint.c algorithmic shape (FP32 variant).
 *
 * Determinism requirements (sovereignty):
 *   - Compile with -fmad=false on the codec path (no FMA reordering).
 *   - No atomics, no -ffast-math.
 *   - 9-wide shared tile dodges 8-way bank conflict on column pass.
 *
 * Forward DCT (0x247) is the transpose of this kernel; drafted inline so
 * encode/decode ship together.
 */

#include <cuda_runtime.h>
#include <stdint.h>

// Loeffler cosine constants — cos(k*pi/16).
#define LOEFFLER_C1  0.98078528040323044912f
#define LOEFFLER_C3  0.83146961230254523708f
#define LOEFFLER_C6  0.38268343236508977173f
#define LOEFFLER_R2  1.41421356237309504880f

extern "C" {

__device__ __forceinline__ void loeffler_idct_1d(float* v) {
    // Even half.
    float b0 = v[0] + v[4];
    float b1 = v[0] - v[4];
    float b2 = v[2] * LOEFFLER_C6 - v[6] * LOEFFLER_C3 * LOEFFLER_R2;
    float b3 = v[2] * LOEFFLER_C3 * LOEFFLER_R2 + v[6] * LOEFFLER_C6;
    float a0 = b0 + b3, a1 = b1 + b2, a2 = b1 - b2, a3 = b0 - b3;

    // Odd half.
    float o0 = v[1] * LOEFFLER_C1 + v[7] * LOEFFLER_C3;
    float o1 = v[5] * LOEFFLER_C3 + v[3] * LOEFFLER_C1;
    float o2 = v[5] * LOEFFLER_C1 - v[3] * LOEFFLER_C3;
    float o3 = v[1] * LOEFFLER_C3 - v[7] * LOEFFLER_C1;
    float t  = (o0 - o1) * LOEFFLER_R2 * 0.5f;
    float u  = (o2 + o3) * LOEFFLER_R2 * 0.5f;

    v[0] = a0 + o0 + o1;
    v[1] = a1 + t + u;
    v[2] = a2 + t - u;
    v[3] = a3 + o2 - o3;
    v[4] = a3 - o2 + o3;
    v[5] = a2 - t + u;
    v[6] = a1 - t - u;
    v[7] = a0 - o0 - o1;
}

__device__ __forceinline__ void loeffler_dct_1d(float* v) {
    // Forward transform: input v[0..7] replaced with DCT coefficients.
    // Butterfly + Loeffler rotator structure, 11 multiplies.
    float b0 = v[0] + v[7];
    float b1 = v[1] + v[6];
    float b2 = v[2] + v[5];
    float b3 = v[3] + v[4];
    float c0 = v[0] - v[7];
    float c1 = v[1] - v[6];
    float c2 = v[2] - v[5];
    float c3 = v[3] - v[4];

    float d0 = b0 + b3;
    float d1 = b1 + b2;
    float d2 = b0 - b3;
    float d3 = b1 - b2;

    v[0] = d0 + d1;
    v[4] = d0 - d1;
    v[2] = d2 * LOEFFLER_C6 + d3 * LOEFFLER_C3 * LOEFFLER_R2;
    v[6] = d3 * LOEFFLER_C6 - d2 * LOEFFLER_C3 * LOEFFLER_R2;

    float e0 = (c1 + c2) * LOEFFLER_R2 * 0.5f;
    float e1 = (c1 - c2) * LOEFFLER_R2 * 0.5f;
    float f0 = c0 + e0;
    float f1 = c3 + e1;
    float f2 = c0 - e0;
    float f3 = c3 - e1;

    v[1] = f0 * LOEFFLER_C1 + f1 * LOEFFLER_C3;
    v[7] = f0 * LOEFFLER_C3 - f1 * LOEFFLER_C1;
    v[5] = f2 * LOEFFLER_C1 + f3 * LOEFFLER_C3;
    v[3] = f2 * LOEFFLER_C3 - f3 * LOEFFLER_C1;
}

// 0x248 IDCT_8X8 — grid=(n_blocks), block=(8,8).
__global__ void idct_8x8_kernel(
    const int16_t* __restrict__ coeffs,   // [n_blocks, 64]
    uint8_t*       __restrict__ pixels,    // [n_blocks, 64]
    int n_blocks)
{
    __shared__ float tile[8][9];   // +1 pad to kill column-bank conflicts.
    int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int tid = r * 8 + c;

    tile[r][c] = (float)coeffs[blk * 64 + tid];
    __syncthreads();

    // Row pass — one thread per row.
    if (c == 0) {
        float v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = tile[r][i];
        loeffler_idct_1d(v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) tile[r][i] = v[i];
    }
    __syncthreads();

    // Column pass — one thread per column.
    if (r == 0) {
        float v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = tile[i][c];
        loeffler_idct_1d(v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) tile[i][c] = v[i];
    }
    __syncthreads();

    // Scale 1/8, level-shift +128, clamp.
    float p = tile[r][c] * 0.125f + 128.0f;
    p = fminf(255.0f, fmaxf(0.0f, p + 0.5f));
    pixels[blk * 64 + tid] = (uint8_t)p;
}

// 0x247 DCT_8X8_FORWARD — grid=(n_blocks), block=(8,8).
// Mirror of IDCT; produces raster-order DCT coefficients.
__global__ void dct_8x8_kernel(
    const uint8_t* __restrict__ pixels,   // [n_blocks, 64]
    int16_t*       __restrict__ coeffs,    // [n_blocks, 64]
    int n_blocks)
{
    __shared__ float tile[8][9];
    int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int tid = r * 8 + c;

    // Load and level-shift.
    tile[r][c] = (float)pixels[blk * 64 + tid] - 128.0f;
    __syncthreads();

    if (c == 0) {
        float v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = tile[r][i];
        loeffler_dct_1d(v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) tile[r][i] = v[i];
    }
    __syncthreads();

    if (r == 0) {
        float v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = tile[i][c];
        loeffler_dct_1d(v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) tile[i][c] = v[i];
    }
    __syncthreads();

    // Normalize 1/8 (symmetric with IDCT scale) and emit int16.
    float q = tile[r][c] * 0.125f;
    int iv = (int)(q >= 0.0f ? q + 0.5f : q - 0.5f);
    if (iv >  32767) iv =  32767;
    if (iv < -32768) iv = -32768;
    coeffs[blk * 64 + tid] = (int16_t)iv;
}

}  // extern "C"
