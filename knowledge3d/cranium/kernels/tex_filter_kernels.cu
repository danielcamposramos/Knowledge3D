__device__ inline int texture_clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline float texture_sample_nearest(TextureHandlePool* pool, uint32_t slot, int x, int y) {
    if (!texture_slot_valid(pool, slot)) {
        return 0.0f;
    }
    const int width = static_cast<int>(pool->width[slot]);
    const int height = static_cast<int>(pool->height[slot]);
    const int sx = texture_clamp_int(x, 0, width - 1);
    const int sy = texture_clamp_int(y, 0, height - 1);
    float* ptr = texture_slot_ptr(pool, slot);
    return ptr[sy * width + sx];
}

__device__ inline float texture_sample_bilinear(TextureHandlePool* pool, uint32_t slot, float x, float y) {
    if (!texture_slot_valid(pool, slot)) {
        return 0.0f;
    }
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float p00 = texture_sample_nearest(pool, slot, x0, y0);
    const float p10 = texture_sample_nearest(pool, slot, x1, y0);
    const float p01 = texture_sample_nearest(pool, slot, x0, y1);
    const float p11 = texture_sample_nearest(pool, slot, x1, y1);
    return texture_lerp(ty, texture_lerp(tx, p00, p10), texture_lerp(tx, p01, p11));
}

__device__ inline void texture_blur_into(TextureHandlePool* pool, uint32_t src_slot, uint32_t dst_slot, float sigma) {
    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
        return;
    }
    const int width = static_cast<int>(pool->width[src_slot]);
    const int height = static_cast<int>(pool->height[src_slot]);
    float* out = texture_slot_ptr(pool, dst_slot);
    const float w_center = fmaxf(0.2f, sigma);
    const float w_axis = fmaxf(0.1f, 0.5f * sigma);
    const float norm = w_center + 4.0f * w_axis;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float sum =
                w_center * texture_sample_nearest(pool, src_slot, x, y) +
                w_axis * texture_sample_nearest(pool, src_slot, x - 1, y) +
                w_axis * texture_sample_nearest(pool, src_slot, x + 1, y) +
                w_axis * texture_sample_nearest(pool, src_slot, x, y - 1) +
                w_axis * texture_sample_nearest(pool, src_slot, x, y + 1);
            out[y * width + x] = sum / norm;
        }
    }
}

__device__ inline void texture_warp_into(TextureHandlePool* pool, uint32_t base_slot, uint32_t warp_slot, uint32_t dst_slot, float intensity) {
    if (!texture_slot_valid(pool, base_slot) || !texture_slot_valid(pool, warp_slot) || !texture_slot_valid(pool, dst_slot)) {
        return;
    }
    const int width = static_cast<int>(pool->width[base_slot]);
    const int height = static_cast<int>(pool->height[base_slot]);
    float* out = texture_slot_ptr(pool, dst_slot);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float warp = texture_sample_nearest(pool, warp_slot, x, y);
            const float wx = static_cast<float>(x) + warp * intensity * 0.5f * static_cast<float>(width);
            const float wy = static_cast<float>(y) + warp * intensity * 0.5f * static_cast<float>(height);
            out[y * width + x] = texture_sample_bilinear(pool, base_slot, wx, wy);
        }
    }
}

__device__ inline float texture_blend_mode(float a, float b, float alpha, uint32_t mode) {
    switch (mode) {
        case 1u:  // multiply
            return texture_lerp(alpha, a, a * b);
        case 2u:  // add
            return a + alpha * b;
        case 3u:  // screen
            return texture_lerp(alpha, a, 1.0f - (1.0f - a) * (1.0f - b));
        default:  // over
            return texture_lerp(alpha, a, b);
    }
}

__device__ inline void texture_blend_into(TextureHandlePool* pool, uint32_t slot_a, uint32_t slot_b, uint32_t dst_slot, float alpha, uint32_t mode) {
    if (!texture_slot_valid(pool, slot_a) || !texture_slot_valid(pool, slot_b) || !texture_slot_valid(pool, dst_slot)) {
        return;
    }
    const int width = static_cast<int>(pool->width[slot_a]);
    const int height = static_cast<int>(pool->height[slot_a]);
    float* out = texture_slot_ptr(pool, dst_slot);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float a = texture_sample_nearest(pool, slot_a, x, y);
            const float b = texture_sample_nearest(pool, slot_b, x, y);
            out[y * width + x] = texture_blend_mode(a, b, alpha, mode);
        }
    }
}

__device__ inline void texture_normal_map_into(TextureHandlePool* pool, uint32_t src_slot, uint32_t dst_slot, float strength) {
    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
        return;
    }
    const int width = static_cast<int>(pool->width[src_slot]);
    const int height = static_cast<int>(pool->height[src_slot]);
    float* out = texture_slot_ptr(pool, dst_slot);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float dx = texture_sample_nearest(pool, src_slot, x + 1, y) - texture_sample_nearest(pool, src_slot, x - 1, y);
            const float dy = texture_sample_nearest(pool, src_slot, x, y + 1) - texture_sample_nearest(pool, src_slot, x, y - 1);
            out[y * width + x] = sqrtf(dx * dx + dy * dy) * strength;
        }
    }
}

__device__ inline void texture_turbulence_into(TextureHandlePool* pool, uint32_t src_slot, uint32_t dst_slot, float octaves) {
    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
        return;
    }
    const int width = static_cast<int>(pool->width[src_slot]);
    const int height = static_cast<int>(pool->height[src_slot]);
    const int octave_count = texture_clamp_int(static_cast<int>(floorf(octaves + 0.5f)), 1, 8);
    float* out = texture_slot_ptr(pool, dst_slot);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            float amp = 1.0f;
            float norm = 0.0f;
            for (int octave = 0; octave < octave_count; ++octave) {
                const float scale = static_cast<float>(1 << octave);
                const float sample = texture_sample_bilinear(pool, src_slot, static_cast<float>(x) / scale, static_cast<float>(y) / scale);
                sum += fabsf(sample) * amp;
                norm += amp;
                amp *= 0.5f;
            }
            out[y * width + x] = norm > 1.0e-6f ? sum / norm : 0.0f;
        }
    }
}

__device__ inline void texture_marble_into(TextureHandlePool* pool, uint32_t dst_slot, float vein_scale, float turbulence) {
    if (!texture_slot_valid(pool, dst_slot)) {
        return;
    }
    const int width = static_cast<int>(pool->width[dst_slot]);
    const int height = static_cast<int>(pool->height[dst_slot]);
    float* out = texture_slot_ptr(pool, dst_slot);
    const int maskx = (width > 0 ? width - 1 : 0) & 4095;
    const int masky = (height > 0 ? height - 1 : 0) & 4095;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float nx = static_cast<float>(x) / fmaxf(1.0f, static_cast<float>(width));
            const float ny = static_cast<float>(y) / fmaxf(1.0f, static_cast<float>(height));
            const int fx = static_cast<int>(nx * vein_scale * 65536.0f);
            const int fy = static_cast<int>(ny * vein_scale * 65536.0f);
            const float noise = texture_perlin_noise2(fx, fy, maskx, masky, 17);
            out[y * width + x] = 0.5f + 0.5f * sinf(nx * vein_scale + turbulence * noise);
        }
    }
}

__device__ inline void texture_transform_into(TextureHandlePool* pool, uint32_t src_slot, uint32_t dst_slot, float sx, float sy, float rot) {
    if (!texture_slot_valid(pool, src_slot) || !texture_slot_valid(pool, dst_slot)) {
        return;
    }
    const int width = static_cast<int>(pool->width[src_slot]);
    const int height = static_cast<int>(pool->height[src_slot]);
    float* out = texture_slot_ptr(pool, dst_slot);
    const float cx = 0.5f * static_cast<float>(width);
    const float cy = 0.5f * static_cast<float>(height);
    const float c = cosf(rot);
    const float s = sinf(rot);
    const float safe_sx = fabsf(sx) < 1.0e-6f ? 1.0f : sx;
    const float safe_sy = fabsf(sy) < 1.0e-6f ? 1.0f : sy;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float px = static_cast<float>(x) - cx;
            const float py = static_cast<float>(y) - cy;
            const float rx = ( c * px + s * py) / safe_sx + cx;
            const float ry = (-s * px + c * py) / safe_sy + cy;
            out[y * width + x] = texture_sample_bilinear(pool, src_slot, rx, ry);
        }
    }
}

