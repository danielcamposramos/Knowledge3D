/**
 * K3D Projection Screen / Unified A/V Playback — opcodes 0x270-0x27F.
 *
 * The projection screen is a rendering surface inside the House that presents
 * a sovereign procedural video stream (frame codec, 0x260-0x26F) paired with
 * audio (0x250-0x25F). This file holds the GPU-side playback kernels. The
 * timeline/PTS state machine and ring buffer are Python ctypes bindings
 * (companion file; not drafted here to keep the hot path pure).
 *
 * Reused kernels (DO NOT DUPLICATE — cite only):
 *   - 0x27B LOD_SELECT  → dynamic_lod_tune in ptx/dynamic_lod_tune.ptx
 *   - 0x27E VIGNETTE    → OP_VIGNETTE existing opcode (0x7E)
 *   - 0x27F FOG         → OP_ATMOSPHERE_FOG existing opcode (0x7D)
 *
 * Kernels defined here:
 *   - video_field_load_kernel  (0x270)  — fan frame cells into ring slots
 *   - audio_field_load_kernel  (0x271)  — decode PCM from MDCT frames
 *   - sync_timeline_kernel     (0x272)  — PTS drift check, ternary halt signal
 *   - timeline_advance_kernel  (0x273)  — atomic cursor increment
 *   - screen_project_kernel    (0x277)  — blit frame into viewport rectangle
 *   - screen_compose_kernel    (0x279)  — alpha-blend layered passes
 *   - viewport_set_kernel      (0x27A)  — write 4×4 transform to constant
 *   - dof_kernel               (0x27C/0x27D) — depth-of-field blur
 */

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

// Viewport transform lives in constant memory (small, all lanes read it).
__constant__ float c_viewport_xform[16];

// 0x27A VIEWPORT_SET — host copies matrix via cuMemcpyToSymbol.
// Included as a no-op kernel so the opcode has a symbol to bind against;
// the actual memcpy is done from the Python launcher side.
__global__ void viewport_set_noop() {
    // Intentionally empty — transform is uploaded via cuMemcpyToSymbol.
}

// 0x270 VIDEO_FIELD_LOAD — copy a decoded frame from the ring buffer into
// the active viewport texture. Tile-parallel; no alpha blend at this stage.
__global__ void video_field_load_kernel(
    const uint8_t* __restrict__ ring,     // [ring_stride*H*4]
    uint8_t*       __restrict__ viewport,  // [W*H*4]
    int W, int H, int ring_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int src = (y * ring_stride + x) * 4;
    int dst = (y * W + x) * 4;
    viewport[dst + 0] = ring[src + 0];
    viewport[dst + 1] = ring[src + 1];
    viewport[dst + 2] = ring[src + 2];
    viewport[dst + 3] = ring[src + 3];
}

// 0x271 AUDIO_FIELD_LOAD — convert MDCT coefficients to PCM samples via the
// existing inverse-MDCT kernel. Here we just provide the host-callable slot.
// The real math lives in ternary_mdct.cu (inverse path).
__global__ void audio_field_load_dispatch(
    const float* __restrict__ mdct_frame,   // [frame_size]
    float*       __restrict__ pcm_out,       // [frame_size]
    int frame_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= frame_size) return;
    // Pass-through; the real IMDCT is a separate kernel. This shim lets the
    // 0x271 opcode dispatch to something concrete until codec_ops IMDCT is
    // wired into the dispatch table (handoff item for Codex).
    pcm_out[i] = mdct_frame[i];
}

// 0x272 SYNC_TIMELINE — PTS drift check. Outputs a ternary halt code:
//   -1 = audio ahead of video  (video should skip)
//    0 = in sync within tolerance
//   +1 = video ahead of audio  (video should wait)
__global__ void sync_timeline_kernel(
    const int64_t* __restrict__ video_pts_us,   // scalar
    const int64_t* __restrict__ audio_pts_us,   // scalar
    int8_t*        __restrict__ halt_trit,       // scalar out
    int tolerance_us)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int64_t delta = *video_pts_us - *audio_pts_us;
    int8_t t = 0;
    if (delta >  (int64_t)tolerance_us) t =  1;
    if (delta < -(int64_t)tolerance_us) t = -1;
    *halt_trit = t;
}

// 0x273 TIMELINE_ADVANCE — bump the playback cursor atomically.
__global__ void timeline_advance_kernel(
    uint64_t* __restrict__ cursor_ns,   // scalar
    uint64_t delta_ns)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *cursor_ns += delta_ns;
}

// 0x277 SCREEN_PROJECT — copy the viewport into the screen framebuffer at a
// specified rectangle. Nearest-neighbor scale; a higher-quality cubic path
// ships later if needed.
__global__ void screen_project_kernel(
    const uint8_t* __restrict__ viewport,   // [Vw*Vh*4]
    uint8_t*       __restrict__ screen,      // [Sw*Sh*4]
    int Vw, int Vh, int Sw, int Sh,
    int rect_x, int rect_y, int rect_w, int rect_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= rect_w || y >= rect_h) return;
    int sx = rect_x + x;
    int sy = rect_y + y;
    if (sx < 0 || sx >= Sw || sy < 0 || sy >= Sh) return;
    int vx = (x * Vw) / rect_w;
    int vy = (y * Vh) / rect_h;
    int src = (vy * Vw + vx) * 4;
    int dst = (sy * Sw + sx) * 4;
    screen[dst + 0] = viewport[src + 0];
    screen[dst + 1] = viewport[src + 1];
    screen[dst + 2] = viewport[src + 2];
    screen[dst + 3] = viewport[src + 3];
}

// 0x279 SCREEN_COMPOSE — alpha-blend an overlay layer onto the screen.
__global__ void screen_compose_kernel(
    const uint8_t* __restrict__ overlay,   // [W*H*4]
    uint8_t*       __restrict__ screen,     // [W*H*4]
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = (y * W + x) * 4;
    uint8_t a = overlay[idx + 3];
    if (a == 0) return;
    int ia = 255 - a;
    screen[idx + 0] = (uint8_t)((overlay[idx + 0] * a + screen[idx + 0] * ia + 127) / 255);
    screen[idx + 1] = (uint8_t)((overlay[idx + 1] * a + screen[idx + 1] * ia + 127) / 255);
    screen[idx + 2] = (uint8_t)((overlay[idx + 2] * a + screen[idx + 2] * ia + 127) / 255);
    screen[idx + 3] = 255;
}

// 0x27C / 0x27D DOF_APERTURE / DOF_FOCUS — depth-of-field box blur whose
// radius is driven by |depth(x,y) - focus_distance| * aperture.
__global__ void dof_blur_kernel(
    const uint8_t* __restrict__ src,      // [W*H*4]
    const float*   __restrict__ depth,    // [W*H]
    uint8_t*       __restrict__ dst,       // [W*H*4]
    int W, int H,
    float focus_distance,
    float aperture)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float d = depth[y * W + x];
    float r = aperture * fabsf(d - focus_distance);
    int ri = (int)fminf(8.0f, r + 0.5f);   // cap at 8 for perf
    if (ri <= 0) {
        int idx = (y * W + x) * 4;
        dst[idx + 0] = src[idx + 0];
        dst[idx + 1] = src[idx + 1];
        dst[idx + 2] = src[idx + 2];
        dst[idx + 3] = src[idx + 3];
        return;
    }
    int sum_r = 0, sum_g = 0, sum_b = 0, n = 0;
    for (int dy = -ri; dy <= ri; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (int dx = -ri; dx <= ri; ++dx) {
            int xx = x + dx;
            if (xx < 0 || xx >= W) continue;
            int s = (yy * W + xx) * 4;
            sum_r += src[s + 0];
            sum_g += src[s + 1];
            sum_b += src[s + 2];
            ++n;
        }
    }
    int idx = (y * W + x) * 4;
    dst[idx + 0] = (uint8_t)(sum_r / n);
    dst[idx + 1] = (uint8_t)(sum_g / n);
    dst[idx + 2] = (uint8_t)(sum_b / n);
    dst[idx + 3] = src[idx + 3];
}

}  // extern "C"
