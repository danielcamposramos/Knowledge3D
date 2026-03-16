// Cognitive Executive - swarm chain trust evaluation from resonance diagnostics.

#include <math.h>

extern "C" __global__ void gre_cognitive_executive(
    const float* __restrict__ resonance_matrix,
    const float* __restrict__ chain_norms,
    float* __restrict__ trust_weights,
    float* __restrict__ coherence_score
)
{
    __shared__ float logits[8];
    __shared__ float offdiag_sum;
    __shared__ int offdiag_count;

    const int idx = threadIdx.x;
    if (idx >= 8) {
        return;
    }

    if (idx == 0) {
        offdiag_sum = 0.0f;
        offdiag_count = 0;
    }
    __syncthreads();

    float resonance_sum = 0.0f;
    int resonance_count = 0;
    for (int j = 0; j < 8; ++j) {
        if (j == idx) {
            continue;
        }
        float value = resonance_matrix[idx * 8 + j];
        resonance_sum += value;
        resonance_count += 1;
        atomicAdd(&offdiag_sum, value);
        atomicAdd(&offdiag_count, 1);
    }

    float mean_resonance = resonance_count > 0 ? (resonance_sum / (float)resonance_count) : 0.0f;
    float norm = fmaxf(chain_norms[idx], 0.0f);
    logits[idx] = mean_resonance * (1.0f + logf(norm + 1.0f));
    __syncthreads();

    if (idx == 0) {
        const float eps = 1e-12f;
        float max_logit = logits[0];
        for (int i = 1; i < 8; ++i) {
            max_logit = fmaxf(max_logit, logits[i]);
        }
        float denom = 0.0f;
        for (int i = 0; i < 8; ++i) {
            float exp_value = expf(logits[i] - max_logit);
            trust_weights[i] = exp_value;
            denom += exp_value;
        }
        for (int i = 0; i < 8; ++i) {
            trust_weights[i] = trust_weights[i] / (denom + eps);
        }
        coherence_score[0] = offdiag_count > 0 ? (offdiag_sum / (float)offdiag_count) : 0.0f;
    }
}
