/**
 * GPU material projection kernels for planar/triplanar surface sampling.
 */

extern "C" {

__device__ __forceinline__ float wrap_unit(float value) {
    return value - floorf(value);
}

__device__ __forceinline__ void sample_planar_rgba_device(
    const float* preview,
    int preview_width,
    int preview_height,
    float coord_u,
    float coord_v,
    float min_u,
    float min_v,
    float extent_u,
    float extent_v,
    float tiling,
    float* rgba_out
) {
    float u = (coord_u - min_u) / extent_u;
    float v = (coord_v - min_v) / extent_v;

    u = wrap_unit(u * tiling);
    v = wrap_unit(v * tiling);

    int x = (int)roundf(u * (float)(preview_width - 1));
    int y = (int)roundf((1.0f - v) * (float)(preview_height - 1));

    x = max(0, min(preview_width - 1, x));
    y = max(0, min(preview_height - 1, y));

    int src = (y * preview_width + x) * 4;
    rgba_out[0] = preview[src + 0];
    rgba_out[1] = preview[src + 1];
    rgba_out[2] = preview[src + 2];
    rgba_out[3] = preview[src + 3];
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

    int dst = idx * 4;
    sample_planar_rgba_device(
        preview,
        preview_width,
        preview_height,
        coords[idx * 2 + 0],
        coords[idx * 2 + 1],
        min_u,
        min_v,
        extent_u,
        extent_v,
        tiling,
        output + dst
    );
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

// Sample all three triplanar planes and blend in one pass.
__global__ void project_triplanar_rgba_kernel(
    const float* preview,
    const float* vertices,
    const float* weights,
    float* output,
    int vertex_count,
    int preview_width,
    int preview_height,
    float min_x,
    float min_y,
    float min_z,
    float extent_x,
    float extent_y,
    float extent_z,
    float tiling
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) return;

    int vertex = idx * 3;
    float yz_rgba[4];
    float xz_rgba[4];
    float xy_rgba[4];
    sample_planar_rgba_device(
        preview, preview_width, preview_height,
        vertices[vertex + 1], vertices[vertex + 2],
        min_y, min_z, extent_y, extent_z, tiling,
        yz_rgba
    );
    sample_planar_rgba_device(
        preview, preview_width, preview_height,
        vertices[vertex + 0], vertices[vertex + 2],
        min_x, min_z, extent_x, extent_z, tiling,
        xz_rgba
    );
    sample_planar_rgba_device(
        preview, preview_width, preview_height,
        vertices[vertex + 0], vertices[vertex + 1],
        min_x, min_y, extent_x, extent_y, tiling,
        xy_rgba
    );

    int rgba = idx * 4;
    float wy = weights[vertex + 0];
    float wx = weights[vertex + 1];
    float wz = weights[vertex + 2];
    output[rgba + 0] = yz_rgba[0] * wy + xz_rgba[0] * wx + xy_rgba[0] * wz;
    output[rgba + 1] = yz_rgba[1] * wy + xz_rgba[1] * wx + xy_rgba[1] * wz;
    output[rgba + 2] = yz_rgba[2] * wy + xz_rgba[2] * wx + xy_rgba[2] * wz;
    output[rgba + 3] = yz_rgba[3] * wy + xz_rgba[3] * wx + xy_rgba[3] * wz;
}

}  // extern "C"
