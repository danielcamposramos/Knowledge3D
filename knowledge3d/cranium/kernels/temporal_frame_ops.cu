extern "C" __global__ void generate_temporal_frames_kernel(
    const float* seed,
    const float* palette,
    const float* time_points,
    unsigned char* out_frames,
    int seed_len,
    int frame_count,
    int width,
    int height,
    int pattern_selector,
    float freq,
    float scale,
    float shift_x,
    float shift_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = frame_count * width * height;
    if (idx >= total) {
        return;
    }

    int frame_idx = idx / (width * height);
    int pixel_idx = idx - frame_idx * width * height;
    int py = pixel_idx / width;
    int px = pixel_idx - py * width;

    float u = width > 0 ? ((float)px / (float)width) : 0.0f;
    float v = height > 0 ? ((float)py / (float)height) : 0.0f;
    float t = time_points[frame_idx];
    float offset = fmodf(t, 1.0f) * 2.0f;
    u = fmodf(u + offset, 1.0f);
    v = fmodf(v + offset, 1.0f);

    float value = 0.0f;
    if (pattern_selector == 0) {
        float phase_x = seed_len > 1 ? seed[1] : 0.0f;
        float phase_y = seed_len > 2 ? seed[2] : 0.0f;
        value = 0.5f
            + 0.25f * sinf((u * freq + phase_x) * 6.283185307179586f)
            + 0.25f * cosf((v * freq + phase_y) * 6.283185307179586f);
    } else if (pattern_selector == 1) {
        float min_dist_sq = 1e9f;
        for (int i = 0; i < 4; ++i) {
            int base = (i * 2) % max(seed_len, 1);
            float sx = 0.5f + 0.5f * sinf(seed[base] * 12.9898f + (float)i * 0.73f);
            float sy = 0.5f + 0.5f * cosf(seed[(base + 1) % max(seed_len, 1)] * 78.233f + (float)i * 0.37f);
            float dx = u - sx;
            float dy = v - sy;
            float dist_sq = dx * dx + dy * dy;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
        value = sqrtf(fminf(fmaxf(min_dist_sq, 0.0f), 2.0f)) * 0.70710678118f;
    } else {
        float cx = (u - 0.5f) * scale + shift_x;
        float cy = (v - 0.5f) * scale + shift_y;
        float x = 0.0f;
        float y = 0.0f;
        int iter = 0;
        const int max_iter = 24;
        while (iter < max_iter) {
            float x_new = x * x - y * y + cx;
            float y_new = 2.0f * x * y + cy;
            x = fminf(fmaxf(x_new, -1e3f), 1e3f);
            y = fminf(fmaxf(y_new, -1e3f), 1e3f);
            float mag_sq = fminf(fmaxf(x * x + y * y, 0.0f), 1e6f);
            if (mag_sq > 4.0f) {
                break;
            }
            ++iter;
        }
        value = ((float)iter) / (float)max_iter;
    }

    value = fminf(fmaxf(value, 0.0f), 1.0f);

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    if (value <= 0.5f) {
        float local_t = value * 2.0f;
        r = palette[0] + local_t * (palette[3] - palette[0]);
        g = palette[1] + local_t * (palette[4] - palette[1]);
        b = palette[2] + local_t * (palette[5] - palette[2]);
    } else {
        float local_t = (value - 0.5f) * 2.0f;
        r = palette[3] + local_t * (palette[6] - palette[3]);
        g = palette[4] + local_t * (palette[7] - palette[4]);
        b = palette[5] + local_t * (palette[8] - palette[5]);
    }

    int out_idx = idx * 3;
    out_frames[out_idx + 0] = (unsigned char)(fminf(fmaxf(r + 0.5f, 0.0f), 255.0f));
    out_frames[out_idx + 1] = (unsigned char)(fminf(fmaxf(g + 0.5f, 0.0f), 255.0f));
    out_frames[out_idx + 2] = (unsigned char)(fminf(fmaxf(b + 0.5f, 0.0f), 255.0f));
}
