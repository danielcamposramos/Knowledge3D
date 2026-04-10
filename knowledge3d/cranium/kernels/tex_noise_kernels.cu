__device__ inline const uint16_t* texture_permutation_table() {
    return reinterpret_cast<const uint16_t*>(g_texture_permutation_table_ptr);
}

__device__ inline uint16_t texture_perm_lookup(int idx) {
    const uint16_t* table = texture_permutation_table();
    if (table != nullptr) {
        return table[idx & 4095];
    }
    return static_cast<uint16_t>(mix32(0x93638245u ^ static_cast<uint32_t>(idx)) & 4095u);
}

__device__ inline int texture_perm(int idx) {
    return static_cast<int>(texture_perm_lookup(idx));
}

__device__ inline float texture_lerp(float t, float a, float b) {
    return a + t * (b - a);
}

__device__ inline float texture_smoothstep_quintic(float x) {
    // kkrieger gentexture.cpp: x^3 * (10 + x * (6x - 15))
    return x * x * x * (10.0f + x * (6.0f * x - 15.0f));
}

__device__ inline float texture_pgradient2(int hash, float x, float y) {
    hash &= 7;
    const float u = (hash < 4) ? x : y;
    const float v = (hash < 4) ? y : x;
    return ((hash & 1) ? -u : u) + ((hash & 2) ? -2.0f * v : 2.0f * v);
}

__device__ inline float texture_value_noise2(int x, int y, int maskx, int masky, int seed) {
    static constexpr int kFixedOne = 0x10000;
    const int X = x >> 16;
    const int Y = y >> 16;
    const float fx = static_cast<float>(x & (kFixedOne - 1)) / 65536.0f;
    const float fy = static_cast<float>(y & (kFixedOne - 1)) / 65536.0f;
    const float u = texture_smoothstep_quintic(fx);
    const float v = texture_smoothstep_quintic(fy);
    maskx &= 4095;
    masky &= 4095;

    const float n00 = static_cast<float>(texture_perm(((X + 0) & maskx) + texture_perm(((Y + 0) & masky)) + seed)) / 2047.5f - 1.0f;
    const float n10 = static_cast<float>(texture_perm(((X + 1) & maskx) + texture_perm(((Y + 0) & masky)) + seed)) / 2047.5f - 1.0f;
    const float n01 = static_cast<float>(texture_perm(((X + 0) & maskx) + texture_perm(((Y + 1) & masky)) + seed)) / 2047.5f - 1.0f;
    const float n11 = static_cast<float>(texture_perm(((X + 1) & maskx) + texture_perm(((Y + 1) & masky)) + seed)) / 2047.5f - 1.0f;

    return texture_lerp(
        v,
        texture_lerp(u, n00, n10),
        texture_lerp(u, n01, n11));
}

__device__ inline float texture_perlin_noise2(int x, int y, int maskx, int masky, int seed) {
    static constexpr int kFixedOne = 0x10000;
    static constexpr float kPerlinScale = 0.4472135954999579f;  // 1/sqrt(5)
    const int X = x >> 16;
    const int Y = y >> 16;
    const float fx = static_cast<float>(x & (kFixedOne - 1)) / 65536.0f;
    const float fy = static_cast<float>(y & (kFixedOne - 1)) / 65536.0f;
    const float u = texture_smoothstep_quintic(fx);
    const float v = texture_smoothstep_quintic(fy);
    maskx &= 4095;
    masky &= 4095;

    return kPerlinScale *
        texture_lerp(
            v,
            texture_lerp(
                u,
                texture_pgradient2(texture_perm(((X + 0) & maskx) + texture_perm(((Y + 0) & masky)) + seed), fx, fy),
                texture_pgradient2(texture_perm(((X + 1) & maskx) + texture_perm(((Y + 0) & masky)) + seed), fx - 1.0f, fy)),
            texture_lerp(
                u,
                texture_pgradient2(texture_perm(((X + 0) & maskx) + texture_perm(((Y + 1) & masky)) + seed), fx, fy - 1.0f),
                texture_pgradient2(texture_perm(((X + 1) & maskx) + texture_perm(((Y + 1) & masky)) + seed), fx - 1.0f, fy - 1.0f)));
}

__device__ inline int texture_shuffle3(int x, int y, int z) {
    return texture_perm(texture_perm(texture_perm(x) + y) + z);
}

__device__ inline float texture_grid_noise2(int x, int y, int maskx, int masky, int seed, float falloff_scale) {
    const int i = x >> 16;
    const int j = y >> 16;
    const float xp = static_cast<float>(x & 0xffff) / 65536.0f;
    const float yp = static_cast<float>(y & 0xffff) / 65536.0f;
    float sum = 0.0f;

    for (int oy = 0; oy <= 1; ++oy) {
        for (int ox = 0; ox <= 1; ++ox) {
            const float xr = xp - static_cast<float>(ox);
            const float yr = yp - static_cast<float>(oy);
            float t = xr * xr + yr * yr;
            if (t < 1.0f) {
                t = 1.0f - t;
                t *= t;
                t *= t;
                sum += (t * falloff_scale) * texture_pgradient2(texture_shuffle3((i + ox) & maskx, (j + oy) & masky, seed), xr, yr);
            }
        }
    }
    return sum;
}

__device__ inline float texture_hash_to_unit(uint32_t value) {
    return static_cast<float>(mix32(value)) / 4294967295.0f;
}

__device__ inline float texture_voronoi_f1(float u, float v, float cell_count, float jitter) {
    const float cells = fmaxf(1.0f, cell_count);
    const float scaled_x = u * cells;
    const float scaled_y = v * cells;
    const int cell_x = static_cast<int>(floorf(scaled_x));
    const int cell_y = static_cast<int>(floorf(scaled_y));
    float best = 1.0e30f;

    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const int nx = cell_x + ox;
            const int ny = cell_y + oy;
            const uint32_t base = static_cast<uint32_t>((nx & 0xffff) | ((ny & 0xffff) << 16));
            const float jitter_x = texture_hash_to_unit(0x93638245u ^ base) - 0.5f;
            const float jitter_y = texture_hash_to_unit(0x93638245u ^ mix32(base + 0x9e3779b9u)) - 0.5f;
            const float fx = static_cast<float>(nx) + 0.5f + jitter * jitter_x;
            const float fy = static_cast<float>(ny) + 0.5f + jitter * jitter_y;
            const float dx = scaled_x - fx;
            const float dy = scaled_y - fy;
            best = fminf(best, dx * dx + dy * dy);
        }
    }
    return sqrtf(best);
}

