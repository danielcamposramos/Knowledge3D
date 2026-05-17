/**
 * K3D JPEG-equivalent line-scan support kernels.
 * Opcodes: 0x243 BLOCK_8X8_ZIGZAG, 0x244 BLOCK_8X8_INV_ZIGZAG,
 *          0x245 QUANT_APPLY, 0x246 QUANT_INVERT,
 *          0x249 CHROMA_SUBSAMPLE_422, 0x24A CHROMA_UPSAMPLE_422.
 *
 * Pairs with idct_8x8.cu (0x247/0x248) and huff_decode.cu (0x24B/0x24C) to
 * form the sovereign JPEG decode path. No libjpeg, no PIL, no numpy — pure
 * CUDA + ctypes.
 *
 * Reference: ITU T.81 §F.1.4 (zigzag), §F.1.2 (quantization), §A.2.1 (4:2:2).
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

// Standard JPEG zigzag LUT. In device constant memory for L1 caching.
__constant__ int c_zigzag_fwd[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// 0x243 / 0x244 — one warp per block.
//   reverse=0 : raster→zigzag
//   reverse=1 : zigzag→raster
__global__ void zigzag_scan_kernel(
    const int16_t* __restrict__ in,   // [N,64]
    int16_t*       __restrict__ out,   // [N,64]
    int n_blocks,
    int reverse)
{
    int blk = blockIdx.x;
    int tid = threadIdx.x;   // 0..31
    if (blk >= n_blocks) return;

    const int16_t* src = in  + blk * 64;
    int16_t*       dst = out + blk * 64;

    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        int i = tid + 32 * k;
        if (reverse) dst[c_zigzag_fwd[i]] = src[i];
        else         dst[i] = src[c_zigzag_fwd[i]];
    }
}

// 0x246 QUANT_INVERT — per-element multiply.
__global__ void quant_invert_kernel(
    const int16_t*  __restrict__ zz_coeffs,   // [N,64]
    const uint16_t* __restrict__ quant_table,  // [64]
    int16_t*        __restrict__ out,           // [N,64]
    int n_blocks)
{
    int blk = blockIdx.x;
    int tid = threadIdx.x;
    if (blk >= n_blocks || tid >= 64) return;
    int coef = zz_coeffs[blk * 64 + tid];
    int q    = (int)quant_table[tid];
    int v    = coef * q;
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;
    out[blk * 64 + tid] = (int16_t)v;
}

// 0x245 QUANT_APPLY — forward quantize with rounding toward zero.
__global__ void quant_apply_kernel(
    const int16_t*  __restrict__ coeffs,      // [N,64] raster-order DCT
    const uint16_t* __restrict__ quant_table,  // [64]
    int16_t*        __restrict__ out,           // [N,64] quantized
    int n_blocks)
{
    int blk = blockIdx.x;
    int tid = threadIdx.x;
    if (blk >= n_blocks || tid >= 64) return;
    int coef = coeffs[blk * 64 + tid];
    int q    = (int)quant_table[tid];
    if (q == 0) { out[blk * 64 + tid] = 0; return; }
    int half = q >> 1;
    int v;
    if (coef >= 0) v = (coef + half) / q;
    else           v = -(((-coef) + half) / q);
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;
    out[blk * 64 + tid] = (int16_t)v;
}

// 0x249 CHROMA_SUBSAMPLE_422 — horizontal 2:1 box average.
__global__ void chroma_subsample_422_kernel(
    const uint8_t* __restrict__ cb_in,   // W×H
    const uint8_t* __restrict__ cr_in,
    uint8_t*       __restrict__ cb_out,   // (W/2)×H
    uint8_t*       __restrict__ cr_out,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_w = W >> 1;
    if (x >= half_w || y >= H) return;
    int x0 = x * 2;
    int x1 = x0 + 1;
    if (x1 >= W) x1 = W - 1;
    int cb = (cb_in[y * W + x0] + cb_in[y * W + x1] + 1) >> 1;
    int cr = (cr_in[y * W + x0] + cr_in[y * W + x1] + 1) >> 1;
    cb_out[y * half_w + x] = (uint8_t)cb;
    cr_out[y * half_w + x] = (uint8_t)cr;
}

// 0x24A CHROMA_UPSAMPLE_422 — bilinear horizontal (center-cosited per JPEG).
__global__ void chroma_upsample_422_kernel(
    const uint8_t* __restrict__ cb_in,   // (W/2)×H
    const uint8_t* __restrict__ cr_in,
    uint8_t*       __restrict__ cb_out,   // W×H
    uint8_t*       __restrict__ cr_out,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int half_w = W >> 1;
    int hx  = x >> 1;
    int hx1 = hx + 1;
    if (hx1 >= half_w) hx1 = half_w - 1;
    float w = (x & 1) ? 0.75f : 0.25f;
    float cb = (1.0f - w) * (float)cb_in[y * half_w + hx]
             +        w   * (float)cb_in[y * half_w + hx1];
    float cr = (1.0f - w) * (float)cr_in[y * half_w + hx]
             +        w   * (float)cr_in[y * half_w + hx1];
    cb_out[y * W + x] = (uint8_t)(cb + 0.5f);
    cr_out[y * W + x] = (uint8_t)(cr + 0.5f);
}

}  // extern "C"
