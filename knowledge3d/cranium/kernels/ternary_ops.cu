// Ternary arithmetic kernels ({-1, 0, +1} encoded as int8).

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ int8_t ternary_add(int8_t a, int8_t b) {
    int sum = static_cast<int>(a) + static_cast<int>(b);
    if (sum > 1) return 1;
    if (sum < -1) return -1;
    return static_cast<int8_t>(sum);
}

__device__ __forceinline__ int8_t ternary_mul(int8_t a, int8_t b) {
    return static_cast<int8_t>(a * b);
}

extern "C" __global__
void ternary_add_kernel(const int8_t* a, const int8_t* b, int8_t* out, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    out[idx] = ternary_add(a[idx], b[idx]);
}

extern "C" __global__
void ternary_mul_kernel(const int8_t* a, const int8_t* b, int8_t* out, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    out[idx] = ternary_mul(a[idx], b[idx]);
}
