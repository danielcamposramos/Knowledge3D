#include <cuda_runtime.h>

extern "C" __global__ void embedding_to_vertex_displacement(
    const float* __restrict__ embedding,    // Input: embedding vector (dynamic dim)
    float* __restrict__ vertex_out,         // Output: displaced vertices (float[12] for tetra)
    unsigned int embedding_dim,             // Actual dimension (e.g., 128)
    float disp_scale                        // Displacement scale (e.g., 0.2f)
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return; // single thread does the small job

    // Regular tetrahedron template (approx)
    // v0: ( 0.0,   1.0,   0.0)
    // v1: (-0.866, -0.5,  0.0)
    // v2: ( 0.866, -0.5,  0.0)
    // v3: ( 0.0,   0.0,   1.633)
    const float base[12] = {
        0.0f,   1.0f,  0.0f,
       -0.866f,-0.5f,  0.0f,
        0.866f,-0.5f,  0.0f,
        0.0f,   0.0f,  1.633f
    };

    // Displace each vertex by consecutive embedding triples
    // Clamp displacement to [-disp_scale, disp_scale]
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            int k = 3 * i + j;
            float e = (k < (int)embedding_dim) ? embedding[k] : 0.0f;
            float d = e * disp_scale;
            if (d >  disp_scale) d =  disp_scale;
            if (d < -disp_scale) d = -disp_scale;
            vertex_out[k] = base[k] + d;
        }
    }
}

