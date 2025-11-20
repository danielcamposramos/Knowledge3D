// Minimal ternary MDCT-style quantisation kernel.
#include <cuda_runtime.h>

extern "C" __global__ void ternary_mdct_quantize(const float* input, int n, float threshold, signed char* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = input[idx];
    if (v > threshold) {
        output[idx] = 1;
    } else if (v < -threshold) {
        output[idx] = -1;
    } else {
        output[idx] = 0;
    }
}
