/**
 * K3D Frame Codec / Temporal Video RPN — opcodes 0x260-0x26F.
 *
 * A video in K3D is a sequence of DotMap keyframes (0x260) plus ternary-packed
 * deltas (0x261) with per-region motion vectors (0x262). Sprites, cells, and
 * 2-D Morton sit in the same range because they share the frame surface.
 *
 * Ternary-first retarget:
 *   - Frame deltas across consecutive frames are dominated by zero trits
 *     (static background). Encode dot-position Δ and dot-color-ref Δ as trits
 *     packed via 0x1AB PACK5 / 0x1AC UNPACK5. Skip-zero decode path.
 *   - Motion vectors use two trit channels (dx, dy in {-1,0,+1}) per block;
 *     fractional motion goes through a bicubic resampler only when magnitude > 0.
 *
 * Dual-client contract:
 *   - Encoder (arc3_frame_encoder.cu) continues emitting the 64-D embedding.
 *   - This file adds the *drawing program* half: each frame is a DotMap +
 *     deltas + sprite ops, readable by humans (render) AND AI (Galaxy ops).
 *   - Retrofit to arc3_frame_encoder.cu is a separate lane (spec'd by frame-
 *     codec research agent; dual-emit pattern captured in memory).
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

// ---------------------------------------------------------------------------
// 0x260 FRAME_KEYFRAME — tag a frame as an independent DotMap root.
// Writes a 16-byte header to `out_tag` with {magic, frame_idx, W, H, dot_count}.
// ---------------------------------------------------------------------------

__global__ void frame_keyframe_tag(
    uint8_t* __restrict__ out_tag,   // [16]
    int frame_idx, int W, int H, int dot_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // 'KFRM' = 0x4D52464B little-endian.
    out_tag[0] = 0x4B; out_tag[1] = 0x46; out_tag[2] = 0x52; out_tag[3] = 0x4D;
    *((int*)(out_tag + 4))  = frame_idx;
    out_tag[8]  = (uint8_t)(W & 0xFF);
    out_tag[9]  = (uint8_t)((W >> 8) & 0xFF);
    out_tag[10] = (uint8_t)(H & 0xFF);
    out_tag[11] = (uint8_t)((H >> 8) & 0xFF);
    *((int*)(out_tag + 12)) = dot_count;
}

// ---------------------------------------------------------------------------
// 0x261 FRAME_DELTA — compute per-dot Δ between frame_t and frame_t-1,
// normalize to {-1, 0, +1} via a caller-supplied threshold.
//
// Input:  prev[N*2], curr[N*2] — dot x/y pairs (aligned dot indices).
// Output: trits[2*N] — dx and dy trits interleaved.
//         nonzero_count — scalar, # of non-zero trits (for compression ratio).
// ---------------------------------------------------------------------------

__global__ void frame_delta_ternary(
    const float* __restrict__ prev,       // [N, 2]
    const float* __restrict__ curr,       // [N, 2]
    int8_t*      __restrict__ trits,       // [N, 2]
    int*         __restrict__ nonzero_count,
    float threshold,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float dx = curr[i * 2 + 0] - prev[i * 2 + 0];
    float dy = curr[i * 2 + 1] - prev[i * 2 + 1];
    int8_t tx = (dx >  threshold) ?  1 : ((dx < -threshold) ? -1 : 0);
    int8_t ty = (dy >  threshold) ?  1 : ((dy < -threshold) ? -1 : 0);
    trits[i * 2 + 0] = tx;
    trits[i * 2 + 1] = ty;
    // Atomic counter of non-zero trits — used to size downstream RLE.
    if (tx != 0) atomicAdd(nonzero_count, 1);
    if (ty != 0) atomicAdd(nonzero_count, 1);
}

// ---------------------------------------------------------------------------
// 0x262 MOTION_VECTOR — block-level motion estimation via exhaustive search
// within a ±search_radius window. Ternary-clamped output (dx, dy ∈ {-1,0,+1}
// per block when the best match is within the tolerance; otherwise a full
// int16 MV is stored in the overflow stream).
//
// Block size: 8×8 pixels (matches DotMap cell size). One CUDA block per frame
// block; 64 threads per CUDA block cooperatively scan the search window.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int abs_diff(uint8_t a, uint8_t b) {
    return a > b ? a - b : b - a;
}

__global__ void motion_vector_8x8(
    const uint8_t* __restrict__ ref_frame,   // [H*W]
    const uint8_t* __restrict__ cur_frame,    // [H*W]
    int8_t*        __restrict__ mv_trits,     // [n_blocks_x*n_blocks_y, 2]
    int16_t*       __restrict__ mv_overflow,  // [n_blocks_x*n_blocks_y, 2]
    int W, int H, int search_radius)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int n_blocks_x = (W + 7) / 8;
    int block_idx = by * n_blocks_x + bx;
    int x0 = bx * 8;
    int y0 = by * 8;

    __shared__ int s_best_sad;
    __shared__ int s_best_dx;
    __shared__ int s_best_dy;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_best_sad = 0x7FFFFFFF;
        s_best_dx = 0;
        s_best_dy = 0;
    }
    __syncthreads();

    int win = 2 * search_radius + 1;
    int total = win * win;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int off = tid; off < total; off += blockDim.x * blockDim.y) {
        int dx = (off % win) - search_radius;
        int dy = (off / win) - search_radius;
        int sad = 0;
        #pragma unroll
        for (int py = 0; py < 8; ++py) {
            int cy = y0 + py;
            int ry = cy + dy;
            if (ry < 0 || ry >= H || cy >= H) { sad = 0x7FFFFFFF; break; }
            #pragma unroll
            for (int px = 0; px < 8; ++px) {
                int cx = x0 + px;
                int rx = cx + dx;
                if (rx < 0 || rx >= W || cx >= W) { sad = 0x7FFFFFFF; break; }
                sad += abs_diff(cur_frame[cy * W + cx], ref_frame[ry * W + rx]);
            }
            if (sad == 0x7FFFFFFF) break;
        }
        atomicMin(&s_best_sad, sad);
        __syncthreads();
        if (sad == s_best_sad) {
            s_best_dx = dx;
            s_best_dy = dy;
        }
    }
    __syncthreads();

    if (tid == 0) {
        int dx = s_best_dx, dy = s_best_dy;
        // Ternary-clamp: if |dx|, |dy| ≤ 1 emit trits; otherwise stash overflow.
        if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1) {
            mv_trits[block_idx * 2 + 0] = (int8_t)dx;
            mv_trits[block_idx * 2 + 1] = (int8_t)dy;
            mv_overflow[block_idx * 2 + 0] = 0;
            mv_overflow[block_idx * 2 + 1] = 0;
        } else {
            mv_trits[block_idx * 2 + 0] = 0;
            mv_trits[block_idx * 2 + 1] = 0;
            mv_overflow[block_idx * 2 + 0] = (int16_t)dx;
            mv_overflow[block_idx * 2 + 1] = (int16_t)dy;
        }
    }
}

// ---------------------------------------------------------------------------
// 0x265 / 0x266 FRAME_SPRITE_EMIT / FRAME_SPRITE_BATCH — blit a sprite into
// the frame buffer. Single-emit and batched. Sprite data is RGBA8 with alpha
// test (α=0 pixels are skipped — ternary-aware write path).
// ---------------------------------------------------------------------------

__global__ void frame_sprite_batch(
    const uint8_t* __restrict__ sprites,    // [n_sprites * sprite_stride * 4]
    const int*     __restrict__ spr_x,       // [n_sprites]
    const int*     __restrict__ spr_y,       // [n_sprites]
    const int*     __restrict__ spr_w,       // [n_sprites]
    const int*     __restrict__ spr_h,       // [n_sprites]
    uint8_t*       __restrict__ frame,       // [H*W*4]
    int frame_W, int frame_H, int sprite_stride,
    int n_sprites)
{
    int s   = blockIdx.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (s >= n_sprites) return;
    int w = spr_w[s], h = spr_h[s];
    int total = w * h;
    int base_px = s * sprite_stride * sprite_stride * 4;
    int x0 = spr_x[s], y0 = spr_y[s];
    for (int i = tid; i < total; i += blockDim.x * blockDim.y) {
        int px = i % w;
        int py = i / w;
        int fx = x0 + px;
        int fy = y0 + py;
        if (fx < 0 || fx >= frame_W || fy < 0 || fy >= frame_H) continue;
        int src = base_px + (py * sprite_stride + px) * 4;
        uint8_t a = sprites[src + 3];
        if (a == 0) continue;                 // alpha-0 ⇒ skip (free pixel)
        int dst = (fy * frame_W + fx) * 4;
        if (a == 255) {
            frame[dst + 0] = sprites[src + 0];
            frame[dst + 1] = sprites[src + 1];
            frame[dst + 2] = sprites[src + 2];
            frame[dst + 3] = 255;
        } else {
            // Source-over alpha blend.
            int ia = 255 - a;
            frame[dst + 0] = (uint8_t)((sprites[src + 0] * a + frame[dst + 0] * ia + 127) / 255);
            frame[dst + 1] = (uint8_t)((sprites[src + 1] * a + frame[dst + 1] * ia + 127) / 255);
            frame[dst + 2] = (uint8_t)((sprites[src + 2] * a + frame[dst + 2] * ia + 127) / 255);
            frame[dst + 3] = 255;
        }
    }
}

// ---------------------------------------------------------------------------
// 0x268 FRAME_CELL_FILL — paint a rect region with a palette color (common
// for game_2d tile rendering; avoids a full sprite for solid cells).
// ---------------------------------------------------------------------------

__global__ void frame_cell_fill(
    uint8_t* __restrict__ frame,    // [H*W*4]
    int x0, int y0, int w, int h,
    uint8_t r, uint8_t g, uint8_t b, uint8_t a,
    int frame_W, int frame_H)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;
    int fx = x0 + px, fy = y0 + py;
    if (fx < 0 || fx >= frame_W || fy < 0 || fy >= frame_H) return;
    int dst = (fy * frame_W + fx) * 4;
    frame[dst + 0] = r;
    frame[dst + 1] = g;
    frame[dst + 2] = b;
    frame[dst + 3] = a;
}

// ---------------------------------------------------------------------------
// 0x269 FRAME_MORTON_2D — interleave x/y bits for 2-D Morton (Z-order) codes.
// Preserves locality for RLE and segmented scan stages.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t spread_bits(uint16_t x) {
    uint32_t v = x;
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

__global__ void frame_morton_2d(
    const uint16_t* __restrict__ xs,   // [N]
    const uint16_t* __restrict__ ys,   // [N]
    uint32_t*       __restrict__ morton,   // [N]
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint32_t mx = spread_bits(xs[i]);
    uint32_t my = spread_bits(ys[i]);
    morton[i] = mx | (my << 1);
}

// ---------------------------------------------------------------------------
// 0x26A FRAME_SEQUENCE_RENDER — materialize a frame from (keyframe + deltas +
// motion vectors). Integer-only reconstruction so decode is deterministic.
// One CUDA block per output dot; threads cooperate to resolve Δ and color.
// ---------------------------------------------------------------------------

__global__ void frame_sequence_render(
    const float*   __restrict__ key_dots,      // [N, 2]
    const int8_t*  __restrict__ pos_trits,     // [N, 2]
    const int16_t* __restrict__ mv_overflow,   // [n_mv_blocks, 2]
    const int*     __restrict__ block_of_dot,  // [N] → mv block index
    float scale,                                // trit magnitude in pixels
    float*         __restrict__ out_dots,       // [N, 2]
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float kx = key_dots[i * 2 + 0];
    float ky = key_dots[i * 2 + 1];
    int8_t tx = pos_trits[i * 2 + 0];
    int8_t ty = pos_trits[i * 2 + 1];
    int blk = block_of_dot[i];
    int16_t mvx = mv_overflow[blk * 2 + 0];
    int16_t mvy = mv_overflow[blk * 2 + 1];
    // Ternary-first decode: if trit is 0, mv_overflow may contribute;
    // zero path is one fetch + two adds (no multiplies when trit=0).
    float dx = (tx != 0) ? (scale * (float)tx) : (float)mvx;
    float dy = (ty != 0) ? (scale * (float)ty) : (float)mvy;
    out_dots[i * 2 + 0] = kx + dx;
    out_dots[i * 2 + 1] = ky + dy;
}

}  // extern "C"
