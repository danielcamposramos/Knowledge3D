// Minimal 2D DCT-like transform followed by ternary quantisation.
#include <cuda_runtime.h>

__device__ float fast_cos(float x) { return __cosf(x); }

extern "C" __global__ void ternary_dct2_quantize(
    const float* input, int width, int height, float threshold, signed char* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    // Lightweight separable DCT-II approximation on a single sample neighbourhood.
    float val = input[idx];
    float tx = fast_cos(3.14159265358979f * (x + 0.5f) / (float)width);
    float ty = fast_cos(3.14159265358979f * (y + 0.5f) / (float)height);
    float coeff = val * tx * ty;
    if (coeff > threshold) {
        output[idx] = 1;
    } else if (coeff < -threshold) {
        output[idx] = -1;
    } else {
        output[idx] = 0;
    }
}
