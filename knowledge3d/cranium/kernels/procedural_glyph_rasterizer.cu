#include <math.h>

extern "C" __global__ void procedural_glyph_rasterizer(
    const float* __restrict__ segments,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ segment_lengths,
    const float* __restrict__ transforms,
    float* __restrict__ output,
    int glyph_count,
    int height,
    int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int glyph_idx = blockIdx.z;

    if (glyph_idx >= glyph_count || x >= width || y >= height) {
        return;
    }

    int pixel_index = glyph_idx * height * width + y * width + x;

    float norm_x = ((float)x + 0.5f) / (float)width * 2.0f - 1.0f;
    float norm_y = ((float)y + 0.5f) / (float)height * 2.0f - 1.0f;

    const float scale = transforms[glyph_idx * 4 + 0];
    const float rotation = transforms[glyph_idx * 4 + 1];
    const float tx = transforms[glyph_idx * 4 + 2];
    const float ty = transforms[glyph_idx * 4 + 3];

    float cos_r = cosf(rotation);
    float sin_r = sinf(rotation);
    float inv_scale = scale > 1e-6f ? 1.0f / scale : 1.0f;

    float local_x = (norm_x - tx) * inv_scale;
    float local_y = (norm_y - ty) * inv_scale;

    float rot_x =  local_x * cos_r + local_y * sin_r;
    float rot_y = -local_x * sin_r + local_y * cos_r;

    int seg_offset = segment_offsets[glyph_idx];
    int seg_count = segment_lengths[glyph_idx];

    float winding = 0.0f;
    float min_distance = 1e9f;

    for (int i = 0; i < seg_count; ++i) {
        const float x0 = segments[(seg_offset + i) * 4 + 0];
        const float y0 = segments[(seg_offset + i) * 4 + 1];
        const float x1 = segments[(seg_offset + i) * 4 + 2];
        const float y1 = segments[(seg_offset + i) * 4 + 3];

        // winding number contribution
        bool cond1 = (y0 <= rot_y) && (y1 > rot_y);
        bool cond2 = (y0 > rot_y) && (y1 <= rot_y);
        if (cond1 || cond2) {
            float vt = (rot_y - y0) / ((y1 - y0) + 1e-8f);
            float intersect = x0 + vt * (x1 - x0);
            if (intersect > rot_x) {
                winding += (y1 > y0) ? 1.0f : -1.0f;
            }
        }

        // distance for smooth edges
        float vx = x1 - x0;
        float vy = y1 - y0;
        float seg_len_sq = vx * vx + vy * vy + 1e-8f;
        float proj = ((rot_x - x0) * vx + (rot_y - y0) * vy) / seg_len_sq;
        proj = fminf(fmaxf(proj, 0.0f), 1.0f);
        float closest_x = x0 + proj * vx;
        float closest_y = y0 + proj * vy;
        float dx = rot_x - closest_x;
        float dy = rot_y - closest_y;
        float dist = sqrtf(dx * dx + dy * dy);
        if (dist < min_distance) {
            min_distance = dist;
        }
    }

    float inside = (winding != 0.0f) ? 1.0f : 0.0f;
    float aa = expf(-min_distance * 32.0f);  // simple exponential falloff
    float value = inside * 0.9f + aa * 0.1f;
    output[pixel_index] = value;
}
