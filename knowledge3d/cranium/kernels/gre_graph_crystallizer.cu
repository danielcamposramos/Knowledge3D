// Graph Crystallizer - Sovereign multi-hop message passing primitive
//
// This kernel performs one graph propagation round:
//   refined[i] = tanh(self_weight * self[i] + neighbor_weight * mean(neighbors(i)))
//
// Multi-hop reasoning (K rounds) is orchestrated by the Python bridge, which
// launches this kernel repeatedly and swaps the input/output buffers between
// rounds. This keeps the kernel correct without relying on unsupported
// grid-wide synchronization.

#include <math.h>

extern "C" __global__ void gre_graph_crystallizer(
    const float* __restrict__ node_features,
    const int* __restrict__ adjacency,
    const int* __restrict__ neighbor_counts,
    float* __restrict__ refined_features,
    int node_count,
    int feature_dim,
    int max_neighbors,
    float self_weight,
    float neighbor_weight
)
{
    int node_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = static_cast<int>(blockDim.x * gridDim.x);

    for (int i = node_idx; i < node_count; i += stride) {
        const float* self_row = node_features + static_cast<size_t>(i) * feature_dim;
        float* out_row = refined_features + static_cast<size_t>(i) * feature_dim;
        int count = neighbor_counts[i];

        if (count <= 0) {
            for (int d = 0; d < feature_dim; ++d) {
                out_row[d] = self_row[d];
            }
            continue;
        }

        if (count > max_neighbors) {
            count = max_neighbors;
        }

        int valid_neighbors = 0;
        for (int n = 0; n < count; ++n) {
            int neighbor_idx = adjacency[i * max_neighbors + n];
            if (neighbor_idx >= 0 && neighbor_idx < node_count) {
                valid_neighbors += 1;
            }
        }

        if (valid_neighbors == 0) {
            for (int d = 0; d < feature_dim; ++d) {
                out_row[d] = self_row[d];
            }
            continue;
        }

        float inv_count = 1.0f / static_cast<float>(valid_neighbors);
        for (int d = 0; d < feature_dim; ++d) {
            float aggregate = 0.0f;
            for (int n = 0; n < count; ++n) {
                int neighbor_idx = adjacency[i * max_neighbors + n];
                if (neighbor_idx < 0 || neighbor_idx >= node_count) {
                    continue;
                }
                aggregate += node_features[static_cast<size_t>(neighbor_idx) * feature_dim + d];
            }
            aggregate *= inv_count;
            float mixed = self_weight * self_row[d] + neighbor_weight * aggregate;
            out_row[d] = tanhf(mixed);
        }
    }
}
