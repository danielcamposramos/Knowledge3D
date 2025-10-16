#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float swiglu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ void vec_add3_512(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int i = tid; i < 512; i += stride) {
        out[i] = a[i] + b[i] + c[i];
    }
}

__device__ __forceinline__ void vec_add_512(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int i = tid; i < 512; i += stride) {
        out[i] = a[i] + b[i];
    }
}

__device__ __forceinline__ void matvec_512x1024(
    const float* __restrict__ W,
    const float* __restrict__ v,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int row = tid; row < 1024; row += stride) {
        float sum = 0.0f;
#pragma unroll 8
        for (int col = 0; col < 512; ++col) {
            sum += W[row * 512 + col] * v[col];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void matvec_1024x512(
    const float* __restrict__ W,
    const float* __restrict__ v,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int row = tid; row < 512; row += stride) {
        float sum = 0.0f;
#pragma unroll 8
        for (int col = 0; col < 1024; ++col) {
            sum += W[row * 1024 + col] * v[col];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void swiglu_1024(
    const float* __restrict__ in,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int i = tid; i < 1024; i += stride) {
        out[i] = swiglu_scalar(in[i]);
    }
}

extern "C" __global__ void trm_step_fused(
    const float* __restrict__ q,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ z_new,
    float* __restrict__ y_new,
    float* __restrict__ workspace
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float* temp = workspace;                          // 512
    float* hidden = workspace + 512;                  // 1024
    float* temp2 = workspace + 512 + 1024;            // 512
    float* hidden2 = workspace + 512 + 1024 + 512;    // 1024

    // temp = q + y + z
    vec_add3_512(q, y, z, temp, tid, stride);
    __syncthreads();

    // hidden = W1 @ temp
    matvec_512x1024(W1, temp, hidden, tid, stride);
    __syncthreads();

    // hidden = swiglu(hidden)
    swiglu_1024(hidden, hidden, tid, stride);
    __syncthreads();

    // z_new = W2 @ hidden
    matvec_1024x512(W2, hidden, z_new, tid, stride);
    __syncthreads();

    // temp2 = y + z_new
    vec_add_512(y, z_new, temp2, tid, stride);
    __syncthreads();

    // hidden2 = W3 @ temp2
    matvec_512x1024(W3, temp2, hidden2, tid, stride);
    __syncthreads();

    // hidden2 = swiglu(hidden2)
    swiglu_1024(hidden2, hidden2, tid, stride);
    __syncthreads();

    // y_new = W4 @ hidden2
    matvec_1024x512(W4, hidden2, y_new, tid, stride);
}
