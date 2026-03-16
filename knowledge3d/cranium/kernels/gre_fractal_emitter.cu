// Fractal Emitter - multi-scale self-similarity scoring.

#include <math.h>

extern "C" __global__ void gre_fractal_emitter(
    const float* __restrict__ features,
    float* __restrict__ self_similarity,
    int N,
    int D,
    int num_scales
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float eps = 1e-12f;

    for (int i = idx; i < N; i += stride) {
        const float* row = features + (i * D);
        float total_similarity = 0.0f;
        int scales_used = 0;

        for (int scale = 1; scale <= num_scales; ++scale) {
            int sample_stride = 1 << scale;
            int sub_len = D / sample_stride;
            if (sub_len < 2) {
                break;
            }

            float dot = 0.0f;
            float norm_full = 0.0f;
            float norm_sub = 0.0f;
            for (int d = 0; d < sub_len; ++d) {
                float full_value = row[d];
                float sampled_value = row[d * sample_stride];
                dot += full_value * sampled_value;
                norm_full += full_value * full_value;
                norm_sub += sampled_value * sampled_value;
            }
            total_similarity += dot * rsqrtf((norm_full + eps) * (norm_sub + eps));
            scales_used += 1;
        }

        self_similarity[i] = scales_used > 0 ? (total_similarity / (float)scales_used) : 0.0f;
    }
}
