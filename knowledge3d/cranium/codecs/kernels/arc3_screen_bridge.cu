/**
 * K3D ARC3 Live Screen Bridge — opcodes 0x2A0-0x2AF.
 *
 * Purpose: close the "human sees what AI sees" loop. ARC3 produces a 64×64
 * palette-indexed bitmap per tick (NOT a rolling camera — discrete frame per
 * step per `docs/architecture/arc3_frame_format.md`). Previously this frame
 * was consumed only by `arc3_frame_encoder.cu` for 64-D embedding. The screen
 * pipeline (`projection_screen.cu`) and DotMap codec (`dotmap_codec.cu`)
 * already exist — this file is the bridge that wires them together.
 *
 * Novel kernels defined here:
 *   - 0x2A0 arc3_frame_decode_kernel      — palette idx → RGBA raster
 *   - 0x2A4 arc3_click_invert_kernel       — screen pixel → grid cell
 *   - 0x2A5 arc3_action_emit_kernel        — grid cell + action → tuple
 *   - 0x2A7 arc3_diff_highlight_kernel     — XOR changed-cell overlay
 *   - 0x2A8 arc3_lives_hud_kernel          — HUD composition
 *
 * Delegating opcodes (no new kernel — dispatch to existing):
 *   - 0x2A1 ARC3_PALETTE_SET     → host-side cuMemcpyToSymbol (c_arc3_palette)
 *   - 0x2A2 ARC3_FRAME_TO_DOTMAP → dotmap_codec.cu::dot_place_procedural
 *   - 0x2A3 ARC3_PROJECT_TO_SCREEN → projection_screen.cu::screen_project_kernel
 *   - 0x2A6 ARC3_REPLAY_STEP     → host-side frame advance + frame_decode
 *   - 0x2A9 ARC3_GAME_ID_BIND    → host-side hash store (no GPU state)
 *
 * Sovereignty: no numpy, no Python loops, no preprocessing. All decode math
 * is one kernel launch; palette lives in constant memory (16 × uint32 = 64 B).
 * Per CLAUDE_TEXTURE_FORGE_IMAGE_TO_3D_ARC3_SCREEN_04.20.2026.md §3.4.
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

// 16-entry RGBA palette, loaded via cuMemcpyToSymbol by the launcher.
// ARC3 spec defines 16 discrete colors (indexed 0..15, per arc3_frame_encoder.cu:5).
// Packing: (R << 0) | (G << 8) | (B << 16) | (A << 24). Alpha=0xFF for opaque.
__constant__ uint32_t c_arc3_palette[16];

// 0x2A0 ARC3_FRAME_DECODE — 64×64 uint8 palette indices → RGBA8 raster.
// Grid: ceil(W/16) × ceil(H/16) blocks of 16×16 threads. Typical call: W=H=64.
__global__ void arc3_frame_decode_kernel(
    const uint8_t* __restrict__ frame_indices,   // [W*H] palette indices 0..15
    uint8_t*       __restrict__ rgba_out,         // [W*H*4]
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int pidx = y * W + x;
    // Mask to 4 bits to match arc3_frame_encoder.cu's `color & 0x0Fu` contract.
    uint32_t packed = c_arc3_palette[frame_indices[pidx] & 0x0Fu];
    int dst = pidx * 4;
    rgba_out[dst + 0] = (uint8_t)((packed >>  0) & 0xFFu);
    rgba_out[dst + 1] = (uint8_t)((packed >>  8) & 0xFFu);
    rgba_out[dst + 2] = (uint8_t)((packed >> 16) & 0xFFu);
    rgba_out[dst + 3] = (uint8_t)((packed >> 24) & 0xFFu);
}

// 0x2A4 ARC3_CLICK_INVERT — inverse of screen_project_kernel's rect-scale.
// Input: screen pixel (sx, sy). Output: grid cell (gx, gy) in [0, grid_w) × [0, grid_h).
// Clamps out-of-rect clicks to (-1, -1) so the caller can discard safely.
__global__ void arc3_click_invert_kernel(
    int sx, int sy,
    int rect_x, int rect_y, int rect_w, int rect_h,
    int grid_w, int grid_h,
    int* __restrict__ cell_out)            // [2] = {gx, gy}
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int lx = sx - rect_x;
    int ly = sy - rect_y;
    if (lx < 0 || ly < 0 || lx >= rect_w || ly >= rect_h) {
        cell_out[0] = -1;
        cell_out[1] = -1;
        return;
    }
    // Inverse of screen_project_kernel's (x * Vw) / rect_w mapping.
    cell_out[0] = (lx * grid_w) / rect_w;
    cell_out[1] = (ly * grid_h) / rect_h;
}

// 0x2A5 ARC3_ACTION_EMIT — pack (action_id, gx, gy) into a tuple record.
// Action ids follow the ARC3 SDK: 1=Up, 2=Down, 3=Left, 4=Right, 5=Perform,
// 6=Click, 7=Undo. For non-Click actions gx/gy are ignored by the API but we
// still write them for trace completeness.
__global__ void arc3_action_emit_kernel(
    int action_id, int gx, int gy,
    int* __restrict__ action_record)       // [3] = {action_id, gx, gy}
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    action_record[0] = action_id;
    action_record[1] = gx;
    action_record[2] = gy;
}

// 0x2A7 ARC3_DIFF_HIGHLIGHT — produce an RGBA overlay marking cells that
// differ between two frames. Alpha=0 where identical, 0xC0 highlight otherwise.
// Highlight color is passed via `hi_rgba` (launcher picks it, e.g. yellow).
__global__ void arc3_diff_highlight_kernel(
    const uint8_t* __restrict__ frame_a,    // [W*H] indices
    const uint8_t* __restrict__ frame_b,    // [W*H] indices
    uint8_t*       __restrict__ overlay_rgba, // [W*H*4]
    int W, int H, uint32_t hi_rgba)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int pidx = y * W + x;
    int dst = pidx * 4;
    uint8_t a = (frame_a[pidx] & 0x0Fu);
    uint8_t b = (frame_b[pidx] & 0x0Fu);
    if (a == b) {
        overlay_rgba[dst + 0] = 0;
        overlay_rgba[dst + 1] = 0;
        overlay_rgba[dst + 2] = 0;
        overlay_rgba[dst + 3] = 0;   // transparent — unchanged cell
    } else {
        overlay_rgba[dst + 0] = (uint8_t)((hi_rgba >>  0) & 0xFFu);
        overlay_rgba[dst + 1] = (uint8_t)((hi_rgba >>  8) & 0xFFu);
        overlay_rgba[dst + 2] = (uint8_t)((hi_rgba >> 16) & 0xFFu);
        overlay_rgba[dst + 3] = 0xC0;   // 75% highlight
    }
}

// 0x2A8 ARC3_LIVES_HUD — render a tiny HUD strip at (hud_x, hud_y) showing
// `lives_remaining` red squares and `moves_remaining / moves_total` blue bar.
// Output is an RGBA patch the launcher composes onto the screen via
// projection_screen.cu::screen_compose_kernel.
__global__ void arc3_lives_hud_kernel(
    uint8_t* __restrict__ hud_rgba,         // [Hw*Hh*4]
    int Hw, int Hh,
    int lives_remaining, int lives_total,
    int moves_remaining, int moves_total)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Hw || y >= Hh) return;
    int idx = (y * Hw + x) * 4;

    // Top half: red squares (lives). Bottom half: blue progress bar (moves).
    if (y < Hh / 2) {
        int per_life = Hw / (lives_total > 0 ? lives_total : 1);
        int slot = (per_life > 0) ? (x / per_life) : 0;
        int filled = slot < lives_remaining;
        hud_rgba[idx + 0] = filled ? 0xE0 : 0x20;
        hud_rgba[idx + 1] = 0x10;
        hud_rgba[idx + 2] = 0x10;
        hud_rgba[idx + 3] = 0xFF;
    } else {
        int fill_cols = (moves_total > 0) ? (Hw * moves_remaining) / moves_total : 0;
        int filled = x < fill_cols;
        hud_rgba[idx + 0] = 0x10;
        hud_rgba[idx + 1] = 0x40;
        hud_rgba[idx + 2] = filled ? 0xE0 : 0x20;
        hud_rgba[idx + 3] = 0xFF;
    }
}

}  // extern "C"
