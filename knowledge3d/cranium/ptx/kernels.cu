extern "C" __global__ void apply_transform(
    const float* __restrict__ matrix, // 4x4 column-major
    float3* vertices,
    const unsigned int* __restrict__ mesh_offsets,
    const unsigned int* __restrict__ mesh_counts,
    unsigned int mesh_index
) {
    unsigned int start = mesh_offsets[mesh_index];
    unsigned int count = mesh_counts[mesh_index];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    unsigned int idx = start + tid;
    float3 v = vertices[idx];
    float x = v.x, y = v.y, z = v.z;

    float tx = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    float ty = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    float tz = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];

    vertices[idx] = make_float3(tx, ty, tz);
}

extern "C" __global__ void recalc_normals(
    const float3* __restrict__ vertices,
    const unsigned int* __restrict__ indices,
    float3* normals,
    const unsigned int* __restrict__ mesh_offsets,
    const unsigned int* __restrict__ mesh_counts,
    unsigned int mesh_index
) {
    unsigned int start = mesh_offsets[mesh_index];
    unsigned int count = mesh_counts[mesh_index] / 3; // triangle count
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    unsigned int base = start + tid * 3;
    unsigned int i0 = indices[base + 0];
    unsigned int i1 = indices[base + 1];
    unsigned int i2 = indices[base + 2];

    float3 v0 = vertices[i0];
    float3 v1 = vertices[i1];
    float3 v2 = vertices[i2];

    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 n = make_float3(
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    );

    atomicAdd(&normals[i0].x, n.x);
    atomicAdd(&normals[i0].y, n.y);
    atomicAdd(&normals[i0].z, n.z);
    atomicAdd(&normals[i1].x, n.x);
    atomicAdd(&normals[i1].y, n.y);
    atomicAdd(&normals[i1].z, n.z);
    atomicAdd(&normals[i2].x, n.x);
    atomicAdd(&normals[i2].y, n.y);
    atomicAdd(&normals[i2].z, n.z);
}

extern "C" __global__ void blend_embeddings(
    const float* __restrict__ source,
    float* target,
    float alpha,
    unsigned int embedding_dim,
    unsigned int node_offset
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= embedding_dim) return;

    unsigned int idx = node_offset * embedding_dim + tid;
    float src = source[tid];
    float dst = target[idx];
    target[idx] = dst + alpha * (src - dst);
}

extern "C" __global__ void scale_vertices(
    const float3 scale,
    float3* vertices,
    const unsigned int* __restrict__ mesh_offsets,
    const unsigned int* __restrict__ mesh_counts,
    unsigned int mesh_index
) {
    unsigned int start = mesh_offsets[mesh_index];
    unsigned int count = mesh_counts[mesh_index];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    unsigned int idx = start + tid;
    float3 v = vertices[idx];
    vertices[idx] = make_float3(v.x * scale.x, v.y * scale.y, v.z * scale.z);
}

extern "C" __global__ void offset_vertices(
    const float3 offset,
    float3* vertices,
    const unsigned int* __restrict__ mesh_offsets,
    const unsigned int* __restrict__ mesh_counts,
    unsigned int mesh_index
) {
    unsigned int start = mesh_offsets[mesh_index];
    unsigned int count = mesh_counts[mesh_index];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    unsigned int idx = start + tid;
    float3 v = vertices[idx];
    vertices[idx] = make_float3(v.x + offset.x, v.y + offset.y, v.z + offset.z);
}

extern "C" __global__ void normalize_embedding(
    float* embeddings,
    unsigned int embedding_dim,
    unsigned int node_index
) {
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int base = node_index * embedding_dim;

    float accum = 0.0f;
    for (unsigned int idx = tid; idx < embedding_dim; idx += blockDim.x) {
        float val = embeddings[base + idx];
        accum += val * val;
    }
    shared[tid] = accum;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float norm = sqrtf(shared[0]);
    if (norm < 1e-8f) {
        norm = 1e-8f;
    }

    for (unsigned int idx = tid; idx < embedding_dim; idx += blockDim.x) {
        embeddings[base + idx] /= norm;
    }
}
