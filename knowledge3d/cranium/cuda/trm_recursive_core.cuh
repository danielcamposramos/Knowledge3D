#pragma once

#include <math.h>

#ifndef GPU_TASK_EMBED_DIMS
#define GPU_TASK_EMBED_DIMS 32
#endif

#ifndef GPU_TASK_TRM_DIMS
#define GPU_TASK_TRM_DIMS 512
#endif

#ifndef GPU_TASK_TRM_HIDDEN_DIMS
#define GPU_TASK_TRM_HIDDEN_DIMS 1024
#endif

__device__ __forceinline__ float trm_swiglu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ void trm_vec_add3_512_device(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int i = tid; i < GPU_TASK_TRM_DIMS; i += stride) {
        out[i] = a[i] + b[i] + c[i];
    }
}

__device__ __forceinline__ void trm_vec_add_512_device(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int i = tid; i < GPU_TASK_TRM_DIMS; i += stride) {
        out[i] = a[i] + b[i];
    }
}

__device__ __forceinline__ void trm_matvec_512x1024_device(
    const float* __restrict__ W,
    const float* __restrict__ v,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int row = tid; row < GPU_TASK_TRM_HIDDEN_DIMS; row += stride) {
        float sum = 0.0f;
#pragma unroll 8
        for (int col = 0; col < GPU_TASK_TRM_DIMS; ++col) {
            sum += W[row * GPU_TASK_TRM_DIMS + col] * v[col];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void trm_matvec_1024x512_device(
    const float* __restrict__ W,
    const float* __restrict__ v,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int row = tid; row < GPU_TASK_TRM_DIMS; row += stride) {
        float sum = 0.0f;
#pragma unroll 8
        for (int col = 0; col < GPU_TASK_TRM_HIDDEN_DIMS; ++col) {
            sum += W[row * GPU_TASK_TRM_HIDDEN_DIMS + col] * v[col];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void trm_swiglu_1024_device(
    const float* __restrict__ in,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int i = tid; i < GPU_TASK_TRM_HIDDEN_DIMS; i += stride) {
        out[i] = trm_swiglu_scalar(in[i]);
    }
}

__device__ __forceinline__ void trm_recursive_core_device(
    const float* __restrict__ q,
    float* __restrict__ y,
    float* __restrict__ z,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ workspace,
    int tid,
    int stride,
    int max_steps,
    float epsilon,
    int* __restrict__ steps_shared,
    float* __restrict__ drift_shared,
    int* __restrict__ steps_out,
    float* __restrict__ drift_out
) {
    float* temp = workspace;
    float* hidden = workspace + GPU_TASK_TRM_DIMS;
    float* temp2 = workspace + GPU_TASK_TRM_DIMS + GPU_TASK_TRM_HIDDEN_DIMS;
    float* hidden2 = workspace + GPU_TASK_TRM_DIMS + GPU_TASK_TRM_HIDDEN_DIMS + GPU_TASK_TRM_DIMS;
    float* z_new = workspace + 3072;
    float* y_new = workspace + 3584;

    if (tid == 0) {
        *drift_shared = 0.0f;
        *steps_shared = 0;
    }
    __syncthreads();

    for (int step = 0; step < max_steps; ++step) {
        trm_vec_add3_512_device(q, y, z, temp, tid, stride);
        __syncthreads();

        trm_matvec_512x1024_device(W1, temp, hidden, tid, stride);
        __syncthreads();

        trm_swiglu_1024_device(hidden, hidden, tid, stride);
        __syncthreads();

        trm_matvec_1024x512_device(W2, hidden, z_new, tid, stride);
        __syncthreads();

        trm_vec_add_512_device(y, z_new, temp2, tid, stride);
        __syncthreads();

        trm_matvec_512x1024_device(W3, temp2, hidden2, tid, stride);
        __syncthreads();

        trm_swiglu_1024_device(hidden2, hidden2, tid, stride);
        __syncthreads();

        trm_matvec_1024x512_device(W4, hidden2, y_new, tid, stride);
        __syncthreads();

        if (tid == 0) {
            float local_max = 0.0f;
            for (int i = 0; i < GPU_TASK_TRM_DIMS; ++i) {
                const float diff = fabsf(z_new[i] - z[i]);
                if (diff > local_max) {
                    local_max = diff;
                }
            }
            *drift_shared = local_max;
            *steps_shared = step + 1;
        }
        __syncthreads();

        for (int i = tid; i < GPU_TASK_TRM_DIMS; i += stride) {
            z[i] = z_new[i];
            y[i] = y_new[i];
        }
        __syncthreads();

        if (epsilon > 0.0f && *drift_shared < epsilon) {
            break;
        }
    }

    if (tid == 0) {
        if (steps_out != nullptr) {
            *steps_out = *steps_shared;
        }
        if (drift_out != nullptr) {
            *drift_out = *drift_shared;
        }
    }
}

__device__ __forceinline__ void trm_project_query_device(
    const float* __restrict__ query_embedding,
    const float* __restrict__ route_embedding,
    float* __restrict__ q,
    int tid,
    int stride
) {
    for (int i = tid; i < GPU_TASK_TRM_DIMS; i += stride) {
        const int reduced = i % GPU_TASK_EMBED_DIMS;
        const float query_value = query_embedding[reduced];
        const float route_value = route_embedding[reduced];
        q[i] = (0.68f * query_value) + (0.32f * route_value);
    }
}

__device__ __forceinline__ void trm_project_latent_to_reasoning_device(
    const float* __restrict__ y,
    const float* __restrict__ z,
    float* __restrict__ out32,
    int tid,
    int stride
) {
    const int bucket = GPU_TASK_TRM_DIMS / GPU_TASK_EMBED_DIMS;
    for (int dim = tid; dim < GPU_TASK_EMBED_DIMS; dim += stride) {
        float accum = 0.0f;
        const int start = dim * bucket;
        for (int offset = 0; offset < bucket; ++offset) {
            const int index = start + offset;
            accum += 0.5f * y[index] + 0.5f * z[index];
        }
        out32[dim] = tanhf(accum / static_cast<float>(bucket));
    }
}
