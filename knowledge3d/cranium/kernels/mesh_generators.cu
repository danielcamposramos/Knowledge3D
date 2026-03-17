// Phase H2 mesh generator kernels.
//
// These kernels back the sovereign primitive subset used by the mesh bridge.
// Complex constructive programs still compose through the host interpreter.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = 6.28318530717958647692f;

__device__ inline int clamp_min_i(int value, int minimum) {
    return value < minimum ? minimum : value;
}

__device__ inline float3 make_v3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ inline float3 sub3(float3 a, float3 b) {
    return make_v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 cross3(float3 a, float3 b) {
    return make_v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float3 normalize_safe(float3 v, float3 fallback = make_float3(0.0f, 0.0f, 1.0f)) {
    float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (norm <= 1e-8f) {
        return fallback;
    }
    float inv = 1.0f / norm;
    return make_v3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ inline void write_v3(float* out, int index, float3 value) {
    int base = index * 3;
    out[base + 0] = value.x;
    out[base + 1] = value.y;
    out[base + 2] = value.z;
}

__device__ inline void write_v2(float* out, int index, float u, float v) {
    int base = index * 2;
    out[base + 0] = u;
    out[base + 1] = v;
}

__device__ __constant__ float kCubePositions[72] = {
    -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f, -0.5f,
     0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f, -0.5f,
    -0.5f,  0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,
    -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f
};

__device__ __constant__ float kCubeNormals[72] = {
     0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,
     0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
    -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
     1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
     0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
     0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f
};

__device__ __constant__ float kCubeUVs[48] = {
    0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f,
    0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f,
    0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f,
    0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f,
    0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f,
    0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f
};

__device__ __constant__ unsigned int kCubeIndices[36] = {
     0,  1,  2,   0,  2,  3,
     4,  5,  6,   4,  6,  7,
     8,  9, 10,   8, 10, 11,
    12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19,
    20, 21, 22,  20, 22, 23
};

__device__ __constant__ float kIcosahedronPositions[36] = {
    -0.5257311f,  0.8506508f,  0.0f,
     0.5257311f,  0.8506508f,  0.0f,
    -0.5257311f, -0.8506508f,  0.0f,
     0.5257311f, -0.8506508f,  0.0f,
     0.0f,       -0.5257311f,  0.8506508f,
     0.0f,        0.5257311f,  0.8506508f,
     0.0f,       -0.5257311f, -0.8506508f,
     0.0f,        0.5257311f, -0.8506508f,
     0.8506508f,  0.0f,       -0.5257311f,
     0.8506508f,  0.0f,        0.5257311f,
    -0.8506508f,  0.0f,       -0.5257311f,
    -0.8506508f,  0.0f,        0.5257311f
};

}  // namespace

extern "C" {

__global__ void generate_plane_vertices(
    float* __restrict__ vertices,
    float* __restrict__ normals,
    float* __restrict__ uvs,
    float width,
    float depth,
    int segments_w,
    int segments_d
) {
    int seg_w = clamp_min_i(segments_w, 1);
    int seg_d = clamp_min_i(segments_d, 1);
    int cols = seg_w + 1;
    int total_vertices = cols * (seg_d + 1);
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= total_vertices) {
        return;
    }

    int row = vertex_index / cols;
    int col = vertex_index % cols;
    float u = (float)col / (float)seg_w;
    float v = (float)row / (float)seg_d;
    float x = (u - 0.5f) * width;
    float z = (v - 0.5f) * depth;

    write_v3(vertices, vertex_index, make_v3(x, 0.0f, z));
    write_v3(normals, vertex_index, make_v3(0.0f, 1.0f, 0.0f));
    write_v2(uvs, vertex_index, u, v);
}

__global__ void generate_cube_vertices(
    float* __restrict__ vertices,
    float* __restrict__ normals,
    float* __restrict__ uvs,
    unsigned int* __restrict__ indices,
    float size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float half = size * 0.5f;
    if (index < 24) {
        int base3 = index * 3;
        int base2 = index * 2;
        write_v3(
            vertices,
            index,
            make_v3(
                kCubePositions[base3 + 0] * half * 2.0f,
                kCubePositions[base3 + 1] * half * 2.0f,
                kCubePositions[base3 + 2] * half * 2.0f
            )
        );
        write_v3(
            normals,
            index,
            make_v3(
                kCubeNormals[base3 + 0],
                kCubeNormals[base3 + 1],
                kCubeNormals[base3 + 2]
            )
        );
        write_v2(uvs, index, kCubeUVs[base2 + 0], kCubeUVs[base2 + 1]);
    }
    if (index < 36) {
        indices[index] = kCubeIndices[index];
    }
}

__global__ void generate_uv_sphere_vertices(
    float* __restrict__ vertices,
    float* __restrict__ normals,
    float* __restrict__ uvs,
    float radius,
    int stacks,
    int slices
) {
    int stack_count = clamp_min_i(stacks, 3);
    int slice_count = clamp_min_i(slices, 3);
    int cols = slice_count + 1;
    int total_vertices = (stack_count + 1) * cols;
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= total_vertices) {
        return;
    }

    int stack = vertex_index / cols;
    int slice = vertex_index % cols;
    float phi = kPi * (float)stack / (float)stack_count;
    float theta = kTau * (float)slice / (float)slice_count;
    float y = cosf(phi);
    float ring = sinf(phi);
    float x = cosf(theta) * ring;
    float z = sinf(theta) * ring;
    float3 normal = normalize_safe(make_v3(x, y, z));

    write_v3(vertices, vertex_index, make_v3(normal.x * radius, normal.y * radius, normal.z * radius));
    write_v3(normals, vertex_index, normal);
    write_v2(uvs, vertex_index, (float)slice / (float)slice_count, (float)stack / (float)stack_count);
}

__global__ void generate_cylinder_vertices(
    float* __restrict__ vertices,
    float* __restrict__ normals,
    float* __restrict__ uvs,
    float radius,
    float height,
    int segments
) {
    int seg_count = clamp_min_i(segments, 3);
    int side_cols = seg_count + 1;
    int side_vertices = side_cols * 2;
    int total_vertices = side_vertices + (seg_count * 2) + 2;
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= total_vertices) {
        return;
    }

    float half_h = height * 0.5f;
    if (vertex_index < side_vertices) {
        int ring = vertex_index / side_cols;
        int seg = vertex_index % side_cols;
        int wrapped = seg == seg_count ? 0 : seg;
        float angle = kTau * (float)wrapped / (float)seg_count;
        float x = cosf(angle) * radius;
        float z = sinf(angle) * radius;
        float y = ring == 0 ? -half_h : half_h;
        float3 normal = normalize_safe(make_v3(x, 0.0f, z), make_v3(1.0f, 0.0f, 0.0f));
        write_v3(vertices, vertex_index, make_v3(x, y, z));
        write_v3(normals, vertex_index, normal);
        write_v2(uvs, vertex_index, (float)seg / (float)seg_count, (float)ring);
        return;
    }

    int local = vertex_index - side_vertices;
    if (local < seg_count) {
        float angle = kTau * (float)local / (float)seg_count;
        float x = cosf(angle) * radius;
        float z = sinf(angle) * radius;
        write_v3(vertices, vertex_index, make_v3(x, -half_h, z));
        write_v3(normals, vertex_index, make_v3(0.0f, -1.0f, 0.0f));
        write_v2(uvs, vertex_index, (x / (radius * 2.0f)) + 0.5f, (z / (radius * 2.0f)) + 0.5f);
        return;
    }
    if (local == seg_count) {
        write_v3(vertices, vertex_index, make_v3(0.0f, -half_h, 0.0f));
        write_v3(normals, vertex_index, make_v3(0.0f, -1.0f, 0.0f));
        write_v2(uvs, vertex_index, 0.5f, 0.5f);
        return;
    }
    local -= (seg_count + 1);
    if (local < seg_count) {
        float angle = kTau * (float)local / (float)seg_count;
        float x = cosf(angle) * radius;
        float z = sinf(angle) * radius;
        write_v3(vertices, vertex_index, make_v3(x, half_h, z));
        write_v3(normals, vertex_index, make_v3(0.0f, 1.0f, 0.0f));
        write_v2(uvs, vertex_index, (x / (radius * 2.0f)) + 0.5f, (z / (radius * 2.0f)) + 0.5f);
        return;
    }
    write_v3(vertices, vertex_index, make_v3(0.0f, half_h, 0.0f));
    write_v3(normals, vertex_index, make_v3(0.0f, 1.0f, 0.0f));
    write_v2(uvs, vertex_index, 0.5f, 0.5f);
}

__global__ void generate_cone_vertices(
    float* __restrict__ vertices,
    float* __restrict__ normals,
    float* __restrict__ uvs,
    float radius,
    float height,
    int segments
) {
    int seg_count = clamp_min_i(segments, 3);
    int side_cols = seg_count + 1;
    int total_vertices = side_cols + 1 + seg_count + 1;
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= total_vertices) {
        return;
    }

    float half_h = height * 0.5f;
    if (vertex_index < side_cols) {
        int seg = vertex_index;
        int wrapped = seg == seg_count ? 0 : seg;
        float angle = kTau * (float)wrapped / (float)seg_count;
        float x = cosf(angle) * radius;
        float z = sinf(angle) * radius;
        float3 normal = normalize_safe(make_v3(x, radius / fmaxf(height, 1e-6f), z), make_v3(1.0f, 0.0f, 0.0f));
        write_v3(vertices, vertex_index, make_v3(x, -half_h, z));
        write_v3(normals, vertex_index, normal);
        write_v2(uvs, vertex_index, (float)seg / (float)seg_count, 0.0f);
        return;
    }
    if (vertex_index == side_cols) {
        write_v3(vertices, vertex_index, make_v3(0.0f, half_h, 0.0f));
        write_v3(normals, vertex_index, make_v3(0.0f, 1.0f, 0.0f));
        write_v2(uvs, vertex_index, 0.5f, 1.0f);
        return;
    }

    int local = vertex_index - side_cols - 1;
    if (local < seg_count) {
        float angle = kTau * (float)local / (float)seg_count;
        float x = cosf(angle) * radius;
        float z = sinf(angle) * radius;
        write_v3(vertices, vertex_index, make_v3(x, -half_h, z));
        write_v3(normals, vertex_index, make_v3(0.0f, -1.0f, 0.0f));
        write_v2(uvs, vertex_index, (x / (radius * 2.0f)) + 0.5f, (z / (radius * 2.0f)) + 0.5f);
        return;
    }

    write_v3(vertices, vertex_index, make_v3(0.0f, -half_h, 0.0f));
    write_v3(normals, vertex_index, make_v3(0.0f, -1.0f, 0.0f));
    write_v2(uvs, vertex_index, 0.5f, 0.5f);
}

__global__ void generate_torus_vertices(
    float* __restrict__ vertices,
    float* __restrict__ normals,
    float* __restrict__ uvs,
    float major_r,
    float minor_r,
    int major_seg,
    int minor_seg
) {
    int major_count = clamp_min_i(major_seg, 3);
    int minor_count = clamp_min_i(minor_seg, 3);
    int cols = minor_count + 1;
    int total_vertices = (major_count + 1) * cols;
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= total_vertices) {
        return;
    }

    int major = vertex_index / cols;
    int minor = vertex_index % cols;
    int wrapped_major = major == major_count ? 0 : major;
    int wrapped_minor = minor == minor_count ? 0 : minor;
    float theta = kTau * (float)wrapped_major / (float)major_count;
    float phi = kTau * (float)wrapped_minor / (float)minor_count;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float cos_p = cosf(phi);
    float sin_p = sinf(phi);

    float x = (major_r + minor_r * cos_p) * cos_t;
    float y = minor_r * sin_p;
    float z = (major_r + minor_r * cos_p) * sin_t;
    float3 normal = normalize_safe(make_v3(cos_p * cos_t, sin_p, cos_p * sin_t));

    write_v3(vertices, vertex_index, make_v3(x, y, z));
    write_v3(normals, vertex_index, normal);
    write_v2(uvs, vertex_index, (float)major / (float)major_count, (float)minor / (float)minor_count);
}

__global__ void generate_icosphere_vertices(
    float* __restrict__ vertices,
    float* __restrict__ normals,
    float* __restrict__ uvs,
    float radius,
    int subdivisions
) {
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= 12) {
        return;
    }

    int base = vertex_index * 3;
    float3 normal = normalize_safe(
        make_v3(
            kIcosahedronPositions[base + 0],
            kIcosahedronPositions[base + 1],
            kIcosahedronPositions[base + 2]
        )
    );
    if (subdivisions > 0) {
        // Higher-order subdivision is still precomputed on the host bridge.
        normal = normalize_safe(normal);
    }
    float u = 0.5f + atan2f(normal.z, normal.x) / kTau;
    float clamped_y = fmaxf(-1.0f, fminf(1.0f, normal.y));
    float v = 0.5f - asinf(clamped_y) / kPi;
    write_v3(vertices, vertex_index, make_v3(normal.x * radius, normal.y * radius, normal.z * radius));
    write_v3(normals, vertex_index, normal);
    write_v2(uvs, vertex_index, u, v);
}

__global__ void mat4_transform_vertices(
    float* __restrict__ vertices,
    float* __restrict__ normals,
    const float* __restrict__ matrix,
    int vertex_count
) {
    int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= vertex_count) {
        return;
    }

    int base = vertex_index * 3;
    float x = vertices[base + 0];
    float y = vertices[base + 1];
    float z = vertices[base + 2];

    float tx =
        matrix[0] * x +
        matrix[1] * y +
        matrix[2] * z +
        matrix[3];
    float ty =
        matrix[4] * x +
        matrix[5] * y +
        matrix[6] * z +
        matrix[7];
    float tz =
        matrix[8] * x +
        matrix[9] * y +
        matrix[10] * z +
        matrix[11];
    vertices[base + 0] = tx;
    vertices[base + 1] = ty;
    vertices[base + 2] = tz;

    if (normals != nullptr) {
        float nx = normals[base + 0];
        float ny = normals[base + 1];
        float nz = normals[base + 2];
        float3 transformed = normalize_safe(
            make_v3(
                matrix[0] * nx + matrix[1] * ny + matrix[2] * nz,
                matrix[4] * nx + matrix[5] * ny + matrix[6] * nz,
                matrix[8] * nx + matrix[9] * ny + matrix[10] * nz
            ),
            make_v3(0.0f, 0.0f, 1.0f)
        );
        normals[base + 0] = transformed.x;
        normals[base + 1] = transformed.y;
        normals[base + 2] = transformed.z;
    }
}

__global__ void generate_index_buffer_grid(
    unsigned int* __restrict__ indices,
    int rows,
    int cols
) {
    int row_count = clamp_min_i(rows, 2);
    int col_count = clamp_min_i(cols, 2);
    int quads_per_row = col_count - 1;
    int total_quads = (row_count - 1) * quads_per_row;
    int quad_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad_index >= total_quads) {
        return;
    }

    int row = quad_index / quads_per_row;
    int col = quad_index % quads_per_row;
    unsigned int i0 = (unsigned int)(row * col_count + col);
    unsigned int i1 = i0 + 1u;
    unsigned int i2 = i0 + (unsigned int)col_count + 1u;
    unsigned int i3 = i0 + (unsigned int)col_count;
    int base = quad_index * 6;
    indices[base + 0] = i0;
    indices[base + 1] = i1;
    indices[base + 2] = i2;
    indices[base + 3] = i0;
    indices[base + 4] = i2;
    indices[base + 5] = i3;
}

__global__ void compute_face_normals(
    const float* __restrict__ vertices,
    const unsigned int* __restrict__ indices,
    float* __restrict__ face_normals,
    int triangle_count
) {
    int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle_index >= triangle_count) {
        return;
    }

    int index_offset = triangle_index * 3;
    unsigned int i0 = indices[index_offset + 0];
    unsigned int i1 = indices[index_offset + 1];
    unsigned int i2 = indices[index_offset + 2];
    float3 v0 = make_v3(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    float3 v1 = make_v3(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    float3 v2 = make_v3(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);
    float3 edge_1 = sub3(v1, v0);
    float3 edge_2 = sub3(v2, v0);
    float3 normal = normalize_safe(cross3(edge_1, edge_2));
    write_v3(face_normals, triangle_index, normal);
}

}  // extern "C"
