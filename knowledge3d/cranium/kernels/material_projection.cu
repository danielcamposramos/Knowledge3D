/**
 * GPU material projection kernels for planar/triplanar surface sampling.
 */

extern "C" {

__device__ __forceinline__ float wrap_unit(float value) {
    return value - floorf(value);
}

// Sample RGBA preview texture at planar coordinates mapped from mesh space.
__global__ void sample_planar_rgba_kernel(
    const float* preview,
    const float* coords,
    float* output,
    int vertex_count,
    int preview_width,
    int preview_height,
    float min_u,
    float min_v,
    float extent_u,
    float extent_v,
    float tiling
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) return;

    float u = (coords[idx * 2 + 0] - min_u) / extent_u;
    float v = (coords[idx * 2 + 1] - min_v) / extent_v;

    u = wrap_unit(u * tiling);
    v = wrap_unit(v * tiling);

    int x = (int)roundf(u * (float)(preview_width - 1));
    int y = (int)roundf((1.0f - v) * (float)(preview_height - 1));

    x = max(0, min(preview_width - 1, x));
    y = max(0, min(preview_height - 1, y));

    int src = (y * preview_width + x) * 4;
    int dst = idx * 4;
    output[dst + 0] = preview[src + 0];
    output[dst + 1] = preview[src + 1];
    output[dst + 2] = preview[src + 2];
    output[dst + 3] = preview[src + 3];
}

// Blend three sampled RGBA planes with per-vertex triplanar weights.
__global__ void blend_triplanar_rgba_kernel(
    const float* yz_rgba,
    const float* xz_rgba,
    const float* xy_rgba,
    const float* weights,
    float* output,
    int vertex_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) return;

    float wy = weights[idx * 3 + 0];
    float wx = weights[idx * 3 + 1];
    float wz = weights[idx * 3 + 2];

    int rgba = idx * 4;
    output[rgba + 0] = yz_rgba[rgba + 0] * wy + xz_rgba[rgba + 0] * wx + xy_rgba[rgba + 0] * wz;
    output[rgba + 1] = yz_rgba[rgba + 1] * wy + xz_rgba[rgba + 1] * wx + xy_rgba[rgba + 1] * wz;
    output[rgba + 2] = yz_rgba[rgba + 2] * wy + xz_rgba[rgba + 2] * wx + xy_rgba[rgba + 2] * wz;
    output[rgba + 3] = yz_rgba[rgba + 3] * wy + xz_rgba[rgba + 3] * wx + xy_rgba[rgba + 3] * wz;
}

}  // extern "C"
