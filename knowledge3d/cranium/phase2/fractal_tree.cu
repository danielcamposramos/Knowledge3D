#include <cuda_runtime.h>

extern "C" __global__ void generate_fractal_tree(
    const float* __restrict__ embedding_ptr,   // Root embedding
    float* __restrict__ vertices_out_ptr,      // xyz float3 per vertex
    unsigned int* __restrict__ indices_out_ptr,// u32 indices (LINES or TRIANGLES)
    float* __restrict__ embedding_out_ptr,     // per-vertex embeddings (row-major)
    unsigned int embedding_dim,                // dynamic dim
    unsigned int max_vertices,
    unsigned int max_indices
) {
    // Single-thread prototype kernel; expands a small branching structure
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Parameters (mapped from root embedding)
    const float angle = (embedding_ptr && embedding_dim > 0) ? embedding_ptr[0] * 3.1415926f : 0.5f;
    const float length = (embedding_ptr && embedding_dim > 1) ? (0.5f + embedding_ptr[1]) : 1.0f;
    const int depth = (embedding_ptr && embedding_dim > 2) ? max(1, min(5, (int)(embedding_ptr[2] * 4.0f + 1.0f))) : 3;

    // Root vertex at origin
    unsigned int vcount = 0;
    if (max_vertices >= 1) {
        vertices_out_ptr[0] = 0.0f; vertices_out_ptr[1] = 0.0f; vertices_out_ptr[2] = 0.0f;
        // copy root embedding row
        if (embedding_out_ptr && embedding_ptr) {
            for (unsigned int d = 0; d < embedding_dim; ++d) {
                embedding_out_ptr[d] = embedding_ptr[d];
            }
        }
        vcount = 1;
    }
    unsigned int icount = 0;

    // Simple breadth-like growth: for each depth level add up to 3 children per parent
    unsigned int parent_start = 0;
    for (int lvl = 1; lvl <= depth && vcount < max_vertices; ++lvl) {
        unsigned int parent_end = vcount; // iterate parents from [parent_start, parent_end)
        for (unsigned int p = parent_start; p < parent_end && vcount < max_vertices; ++p) {
            const float px = vertices_out_ptr[p*3 + 0];
            const float py = vertices_out_ptr[p*3 + 1];
            const float pz = vertices_out_ptr[p*3 + 2];
            for (int c = 0; c < 3 && vcount < max_vertices; ++c) {
                float a = (c == 0 ? angle : (c == 1 ? 0.0f : -angle));
                float dx = __sinf(a) * length;
                float dy = __cosf(a) * length;
                float dz = 0.4f; // upward bias
                // write child vertex
                vertices_out_ptr[vcount*3 + 0] = px + dx;
                vertices_out_ptr[vcount*3 + 1] = py + dy;
                vertices_out_ptr[vcount*3 + 2] = pz + dz;
                // mutate embedding: simple scaled copy
                if (embedding_out_ptr && embedding_ptr) {
                    unsigned int base_in = 0;
                    unsigned int base_out = vcount * embedding_dim;
                    for (unsigned int d = 0; d < embedding_dim; ++d) {
                        float val = embedding_ptr[base_in + d];
                        float scale = 1.0f + 0.05f * lvl;
                        embedding_out_ptr[base_out + d] = val * scale;
                    }
                }
                // add line indices parent->child (2 indices)
                if (indices_out_ptr && icount + 2 <= max_indices) {
                    indices_out_ptr[icount + 0] = p;
                    indices_out_ptr[icount + 1] = vcount;
                    icount += 2;
                }
                ++vcount;
            }
        }
        parent_start = parent_end;
    }
}

