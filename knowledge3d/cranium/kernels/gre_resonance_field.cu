// Resonance Field - cross-galaxy interference scoring
//
// Candidates supported by semantically aligned entries from other galaxies are
// boosted; candidates contradicted by other galaxies are attenuated.

#include <math.h>

extern "C" __global__ void gre_resonance_field(
    const float* __restrict__ candidates,    // [N x D] candidate embeddings
    const int* __restrict__ galaxy_ids,      // [N] galaxy index per candidate
    const float* __restrict__ base_scores,   // [N] pre-existing scores
    float* __restrict__ resonance_scores,    // [N] output: interference-adjusted
    int N,
    int D
)
{
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = static_cast<int>(blockDim.x * gridDim.x);

    for (; i < N; i += stride) {
        float base = base_scores[i];
        int my_galaxy = galaxy_ids[i];

        float constructive = 0.0f;
        float destructive = 0.0f;
        int cross_count = 0;

        float self_norm_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            float value = candidates[i * D + d];
            self_norm_sq += value * value;
        }
        float self_norm_inv = (self_norm_sq > 1e-12f) ? rsqrtf(self_norm_sq) : 0.0f;

        for (int j = 0; j < N; ++j) {
            if (j == i || galaxy_ids[j] == my_galaxy) {
                continue;
            }

            float dot = 0.0f;
            float other_norm_sq = 0.0f;
            for (int d = 0; d < D; ++d) {
                float a = candidates[i * D + d];
                float b = candidates[j * D + d];
                dot += a * b;
                other_norm_sq += b * b;
            }
            float other_norm_inv = (other_norm_sq > 1e-12f) ? rsqrtf(other_norm_sq) : 0.0f;
            float sim = dot * self_norm_inv * other_norm_inv;

            if (sim > 0.3f) {
                constructive += sim * base_scores[j];
            } else if (sim < -0.2f) {
                destructive += fabsf(sim) * base_scores[j];
            }
            cross_count += 1;
        }

        if (cross_count > 0) {
            float inv_cross = 1.0f / static_cast<float>(cross_count);
            constructive *= inv_cross;
            destructive *= inv_cross;
        }

        float adjusted = base * (1.0f + 0.3f * constructive - 0.15f * destructive);
        resonance_scores[i] = fmaxf(0.0f, adjusted);
    }
}
