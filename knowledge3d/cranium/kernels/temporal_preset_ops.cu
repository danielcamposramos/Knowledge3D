extern "C" __global__ void apply_temporal_preset_kernel(
    const unsigned char* base_frames,
    const float* overlay_rgb,
    const float* overlay_alpha,
    const float* edge,
    const float* warmth,
    const float* time_points,
    const int* shifts,
    unsigned char* out_frames,
    int frame_count,
    int width,
    int height,
    int mode,
    float alpha_scale_r,
    float alpha_scale_g,
    float alpha_scale_b,
    float bias
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

    int base_idx = idx * 3;
    int overlay_idx = pixel_idx * 3;

    float base_r = ((float)base_frames[base_idx + 0]) / 255.0f;
    float base_g = ((float)base_frames[base_idx + 1]) / 255.0f;
    float base_b = ((float)base_frames[base_idx + 2]) / 255.0f;

    float out_r = base_r;
    float out_g = base_g;
    float out_b = base_b;

    float t = time_points[frame_idx];
    float phase = 0.5f + 0.5f * sinf(t * 6.283185307179586f);

    if (mode == 0) {
        float blend = 0.06f + 0.08f * phase;
        float alpha = overlay_alpha[pixel_idx];
        float wr = blend * alpha * alpha_scale_r;
        float wg = blend * alpha * alpha_scale_g;
        float wb = blend * alpha * alpha_scale_b;
        out_r = base_r * (1.0f - wr) + overlay_rgb[overlay_idx + 0] * wr;
        out_g = base_g * (1.0f - wg) + overlay_rgb[overlay_idx + 1] * wg;
        out_b = base_b * (1.0f - wb) + overlay_rgb[overlay_idx + 2] * wb;
    } else if (mode == 1) {
        float pulse = 0.55f + 0.45f * sinf(t * 6.283185307179586f);
        float e = edge[pixel_idx];
        out_r = fminf(fmaxf(base_r * (1.0f + 0.1f * pulse) + e * pulse * 0.35f, 0.0f), 1.0f);
        out_g = fminf(fmaxf(base_g * (1.0f + 0.1f * pulse) + e * pulse * 0.35f, 0.0f), 1.0f);
        out_b = fminf(fmaxf(base_b * (1.0f + 0.1f * pulse) + e * pulse * 0.35f, 0.0f), 1.0f);
    } else if (mode == 2) {
        float pulse = phase;
        float w = warmth[pixel_idx];
        float add = w * (0.05f + 0.08f * bias * pulse);
        float mult = 0.92f + 0.12f * pulse;
        out_r = fminf(fmaxf(base_r * mult + add, 0.0f), 1.0f);
        out_g = fminf(fmaxf(base_g * mult + add, 0.0f), 1.0f);
        out_b = fminf(fmaxf(base_b * mult + add, 0.0f), 1.0f);
    } else if (mode == 3) {
        float cos_phase = 0.5f + 0.5f * cosf(t * 6.283185307179586f);
        float mix = 0.12f + 0.08f * cos_phase;
        int shift = shifts[frame_idx];
        int src_x = px - shift;
        while (src_x >= width) src_x -= width;
        while (src_x < 0) src_x += width;
        int rolled_idx = (py * width + src_x) * 3;
        out_r = fminf(fmaxf(base_r * (1.0f - mix) + overlay_rgb[rolled_idx + 0] * mix, 0.0f), 1.0f);
        out_g = fminf(fmaxf(base_g * (1.0f - mix) + overlay_rgb[rolled_idx + 1] * mix, 0.0f), 1.0f);
        out_b = fminf(fmaxf(base_b * (1.0f - mix) + overlay_rgb[rolled_idx + 2] * mix, 0.0f), 1.0f);
    }

    out_frames[base_idx + 0] = (unsigned char)(fminf(fmaxf(out_r * 255.0f + 0.5f, 0.0f), 255.0f));
    out_frames[base_idx + 1] = (unsigned char)(fminf(fmaxf(out_g * 255.0f + 0.5f, 0.0f), 255.0f));
    out_frames[base_idx + 2] = (unsigned char)(fminf(fmaxf(out_b * 255.0f + 0.5f, 0.0f), 255.0f));
}
