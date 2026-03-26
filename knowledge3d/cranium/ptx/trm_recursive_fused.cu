#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float swiglu_scalar_recursive(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ void vec_add3_512_recursive(
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

__device__ __forceinline__ void vec_add_512_recursive(
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

__device__ __forceinline__ void matvec_512x1024_recursive(
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

__device__ __forceinline__ void matvec_1024x512_recursive(
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

__device__ __forceinline__ void swiglu_1024_recursive(
    const float* __restrict__ in,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int i = tid; i < 1024; i += stride) {
        out[i] = swiglu_scalar_recursive(in[i]);
    }
}

extern "C" __global__ void trm_recursive_fused(
    const float* __restrict__ q,
    float* __restrict__ y,
    float* __restrict__ z,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ workspace,
    int* __restrict__ steps_out,
    float* __restrict__ drift_out,
    int max_steps,
    float epsilon
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float* temp = workspace;                            // 512
    float* hidden = workspace + 512;                    // 1024
    float* temp2 = workspace + 1536;                    // 512
    float* hidden2 = workspace + 2048;                  // 1024
    float* z_new = workspace + 3072;                    // 512
    float* y_new = workspace + 3584;                    // 512

    __shared__ float drift_reduce[256];
    __shared__ float max_drift;

    int steps_taken = 0;
    float drift = 0.0f;

    for (int step = 0; step < max_steps; ++step) {
        vec_add3_512_recursive(q, y, z, temp, tid, stride);
        __syncthreads();

        matvec_512x1024_recursive(W1, temp, hidden, tid, stride);
        __syncthreads();

        swiglu_1024_recursive(hidden, hidden, tid, stride);
        __syncthreads();

        matvec_1024x512_recursive(W2, hidden, z_new, tid, stride);
        __syncthreads();

        vec_add_512_recursive(y, z_new, temp2, tid, stride);
        __syncthreads();

        matvec_512x1024_recursive(W3, temp2, hidden2, tid, stride);
        __syncthreads();

        swiglu_1024_recursive(hidden2, hidden2, tid, stride);
        __syncthreads();

        matvec_1024x512_recursive(W4, hidden2, y_new, tid, stride);
        __syncthreads();

        float local_max = 0.0f;
        for (int i = tid; i < 512; i += stride) {
            float diff = fabsf(z_new[i] - z[i]);
            if (diff > local_max) {
                local_max = diff;
            }
        }

        drift_reduce[tid] = local_max;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                drift_reduce[tid] = fmaxf(drift_reduce[tid], drift_reduce[tid + offset]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            max_drift = drift_reduce[0];
        }
        __syncthreads();

        drift = max_drift;

        for (int i = tid; i < 512; i += stride) {
            z[i] = z_new[i];
            y[i] = y_new[i];
        }
        __syncthreads();

        steps_taken = step + 1;

        if (epsilon > 0.0f && drift < epsilon) {
            break;
        }
    }

    if (tid == 0) {
        steps_out[0] = steps_taken;
        drift_out[0] = drift;
    }
}
