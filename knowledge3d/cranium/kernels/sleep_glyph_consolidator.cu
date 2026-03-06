#include <math.h>

extern "C" __global__ void cluster_glyphs_by_similarity(
    int* representative_indices,  // [N]
    const float* embeddings,      // [N x dim]
    int N,
    int dim,
    float similarity_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* emb_i = embeddings + (idx * dim);
    int best_rep = idx;

    for (int j = 0; j < idx; ++j) {
        const float* emb_j = embeddings + (j * dim);
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += emb_i[d] * emb_j[d];
        }
        if (dot >= similarity_threshold) {
            best_rep = j;
            break;
        }
    }

    representative_indices[idx] = best_rep;
}
