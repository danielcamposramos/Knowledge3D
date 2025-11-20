// Simple additive synthesis kernel: sums sinusoids for each sample.
#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" __global__ void additive_synthesis(
    const float* freqs, const float* amps, const float* phases, int n_harmonics,
    float sample_rate, float duration, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = (int)(duration * sample_rate);
    if (idx >= total_samples) return;
    float t = (float)idx / sample_rate;
    float acc = 0.0f;
    for (int h = 0; h < n_harmonics; ++h) {
        float omega = 2.0f * CUDART_PI_F * freqs[h];
        acc += amps[h] * __sinf(omega * t + phases[h]);
    }
    output[idx] = acc;
}
